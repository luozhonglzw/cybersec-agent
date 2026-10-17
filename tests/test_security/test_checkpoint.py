"""Phase v0.4.0-I1 TASK 3 —— checkpoint 严格反序列化白名单测试。

被锁定的三条安全性质(**不是**功能测试):

1. checkpoint 载荷里的 `(module, name)` 只能用于**查表**,不得驱动导入。
   否则"能写 checkpoint 库的一方"就能借 checkpoint 在本进程里执行任意
   可调用对象 —— 实测上游默认 hook 会执行 `Ext(0, ("builtins","eval",...))`。
2. 解不出来**绝不**静默变成 `None`。静默 `None` 会把"载荷损坏"伪装成
   "状态本来就是空的",上层随即把它误判成"checkpoint 丢失"→ 409,
   那是对世界的**错误陈述**。
3. 拒绝一律抛 `CheckpointPayloadRejectedError`(⊂ `CheckpointStorageError`),
   与"后端不可用"分开但同属一个错误族:两者都映射 503,都**绝不**退化成
   404 / 409。

反同义反复(本文件最重要的部分)
--------------------------------
`test_upstream_default_serde_executes_the_payload` 拿**同一个载荷**打上游
默认 serde,断言它**真的执行了** `builtins.eval`。没有这条负对照,下面所有
负向用例都可能是恒真的 —— 即护栏可能根本没有牙。

白名单是实测出来的,不是猜的
----------------------------
用真实 `TriageService`(真实节点 / 真实工具 / 真实 `AsyncSqliteSaver`)跑遍
全部可达场景(暂停 / 批准恢复 / 拒绝恢复 / 裸图 / 放宽 serde 恢复 / 惰性超时),
记录交给编码器的每个类型与解码器看到的每个 ext code:共 7 个 pydantic 模型,
只用 3 个 code(2 / 3 / 5)。`test_whitelist_registries_are_
exactly_the_measured_set` 把这份实测结果**冻成常量** —— 任何静默放宽
(多一个类型、多一个 code、多一个标签)都会让它变红。

⚠ 口径更正:最初从"裸 graph.ainvoke"推导,漏掉了 `SystemMessage`(该形状
不写入 system prompt),导致服务路径上 6/19 个 blob 解不开。现已改为服务路径
口径;裸图**不是**有效证据来源。

隔离要求(与 tests/test_security/test_store.py 同一条硬线)
--------------------------------------------------------
全部 hermetic:日志 / 情报 / audit.db / checkpoints.db 一律由 tmp_path 现场
生成;不读仓库 `data/`,也不在仓库里创建任何 `.db`。文件末尾的
`test_no_repo_checkpoint_db_created` 守着这条线。
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import aiosqlite
import ormsgpack
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_KW_ARGS,
    EXT_CONSTRUCTOR_POS_ARGS,
    EXT_CONSTRUCTOR_SINGLE_ARG,
    EXT_METHOD_SINGLE_ARG,
    EXT_NUMPY_ARRAY,
    EXT_PYDANTIC_V1,
    EXT_PYDANTIC_V2,
    JsonPlusSerializer,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, Interrupt

import app.security.checkpoint as checkpoint_module
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import TriageService
from app.schemas.approval import ApprovalDecision, ApprovalRequest
from app.schemas.policy import PolicyDecision
from app.schemas.response import ResponseAction, ResponsePlan
from app.schemas.risk import RiskAssessment, RiskEvidence
from app.security.checkpoint import (
    CheckpointBackendUnavailableError,
    CheckpointPayloadRejectedError,
    CheckpointStorageError,
    SafeAsyncSqliteSaver,
    StrictCheckpointSerde,
    build_checkpoint_serde,
)
from app.security.store import SqliteAuditStore

# 30 次失败登录 + 恶意情报 → critical → 含 block_ip / reset_credentials → 需审批
BRUTE_FORCE_IP = "203.0.113.66"
BRUTE_FORCE_LOGINS = 30
THREAD_ID = "thread-i1-serde"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_OPT = ormsgpack.OPT_NON_STR_KEYS


# ---------- 载荷构造工具(全部在内存里,不落盘)----------

def _blob(obj: object) -> bytes:
    """把任意 Python 对象编成 msgpack 字节 —— 只用于**伪造**载荷。"""
    return ormsgpack.packb(obj, option=_OPT)


def _ext(code: int, obj: object) -> ormsgpack.Ext:
    return ormsgpack.Ext(code, _blob(obj))


def _record(ext: ormsgpack.Ext) -> tuple[str, bytes]:
    """把伪造的扩展包成 `loads_typed` 接受的 `(tag, bytes)` 记录。

    嵌进一层 map 是刻意的:真实 checkpoint 载荷从不把 ext 放在顶层,
    而是嵌在 state dict 里 —— 这样测的才是真实解码路径。
    """
    return ("msgpack", _blob({"k": ext}))


# ---------- 正向样本:实测出来的 9 个类型 ----------

def _action() -> ResponseAction:
    return ResponseAction(
        action_type="block_ip",
        priority="high",
        target=BRUTE_FORCE_IP,
        rationale="疑似 SSH 爆破,建议封禁源 IP",
        requires_approval=True,
        reversible=True,
    )


def _plan() -> ResponsePlan:
    evidence = RiskEvidence(
        indicator=BRUTE_FORCE_IP, log_event_count=30, failed_login_count=30
    )
    return ResponsePlan(
        indicator=BRUTE_FORCE_IP,
        risk_level="critical",
        summary="疑似 SSH 爆破",
        actions=[_action()],
        assessment=RiskAssessment(
            indicator=BRUTE_FORCE_IP,
            risk_level="critical",
            score=95,
            confidence=90,
            reasons=["30 次失败登录"],
            evidence=evidence,
        ),
    )


def _approval_request() -> ApprovalRequest:
    return ApprovalRequest(
        thread_id=THREAD_ID,
        indicator=BRUTE_FORCE_IP,
        risk_level="critical",
        score=95,
        summary="疑似 SSH 爆破",
        actions=[_action()],
        policy_reasons=["block_ip 需人工审批"],
    )


def _approval_decision() -> ApprovalDecision:
    return ApprovalDecision(
        status="approved",
        operator="alice",
        interrupt_id="iid-1",
        reason="确认封禁",
    )


def _policy_decision() -> PolicyDecision:
    return PolicyDecision(
        outcome="require_approval",
        requires_approval=True,
        gated_actions=["block_ip"],
        reasons=["block_ip 需人工审批"],
        policy_version="v1",
    )


def _legitimate_samples() -> list[tuple[str, object]]:
    """实测集合里的 7 个 pydantic 模型 + datetime + Interrupt。

    刻意**不**包含 `ApprovalDecision` 之外的项目模型(如 `TriageOutcome`):
    它不在白名单里,因为实测中它从不出现在 checkpoint 载荷里。
    `SystemMessage` 是服务路径实测出来的(裸 graph 形状不写入 system prompt,
    故早期推导漏掉它)—— 见本文件顶部"口径更正"。
    """
    return [
        ("ApprovalDecision", _approval_decision()),
        ("ApprovalRequest", _approval_request()),
        ("PolicyDecision", _policy_decision()),
        ("ResponsePlan", _plan()),
        ("AIMessage", AIMessage(content="分析完成")),
        ("HumanMessage", HumanMessage(content="分析该 IP")),
        ("SystemMessage", SystemMessage(content="你是一名网络安全运营分析助手。")),
        ("datetime", datetime(2026, 10, 12, 5, 33, 16, 752686)),
        ("Interrupt", Interrupt(value={"thread_id": THREAD_ID}, id="iid-1")),
    ]


@pytest.fixture
def serde() -> StrictCheckpointSerde:
    return build_checkpoint_serde()


# =====================================================================
# A. 结构护栏(静态,比行为更根本)
# =====================================================================

def _module_ast() -> ast.Module:
    src = Path(checkpoint_module.__file__).read_text(encoding="utf-8")
    return ast.parse(src)


def test_module_has_no_dynamic_import():
    """载荷不得驱动导入:源码里不能出现 importlib / __import__ / 动态 import。"""
    tree = _module_ast()
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [m for m in imported if m == "importlib" or m.startswith("importlib.")]
    assert offenders == [], f"checkpoint 模块引入了动态导入面: {offenders}"


def test_module_has_no_dynamic_callable_dispatch():
    """白名单必须是**静态查表**:不得出现 eval/exec/getattr 这类动态分发。

    这条比"某个载荷被拒"更强:它保证**不存在**能把载荷内容变成可调用对象的
    代码路径,而不是"已知的几条路径被堵住了"。
    """
    banned = {"eval", "exec", "__import__", "compile", "getattr", "setattr"}
    hits: list[str] = []
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Name) and node.id in banned:
            hits.append(f"Name:{node.id}@{node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in banned:
            hits.append(f"Attr:{node.attr}@{node.lineno}")
    assert hits == [], f"checkpoint 模块存在动态分发面: {hits}"


def test_module_does_not_import_pickle():
    """pickle 反序列化就是任意代码执行 —— 这一层不得出现。"""
    src = Path(checkpoint_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert "pickle" not in imported


def test_whitelist_registries_are_exactly_the_measured_set():
    """把**实测**结果冻成常量:多一个类型 / code / 标签都必须是有意为之。

    实测来源:真实 `TriageService` 跑遍全部可达场景(暂停 / 批准恢复 / 拒绝
    恢复 / 裸图 / 放宽 serde 恢复 / 惰性超时),记录编码器看到的类型与解码器
    看到的 ext code。7 个 pydantic 模型、code ∈ {2,3,5}、标签 ∈ {msgpack,null}。
    这里用**集合相等**,不用"包含" —— 静默放宽是本阶段最需要防的偏差。

    ⚠ 口径更正记录:最初从"裸 graph.ainvoke"推导,漏掉了 `SystemMessage`
    (该形状不写入 system prompt),导致服务路径上 6/19 个 blob 解不开。
    现已改为服务路径口径 —— 见 app/security/checkpoint.py 模块 docstring。
    """
    assert set(checkpoint_module._ALLOWED_PYDANTIC) == {
        ("app.schemas.approval", "ApprovalDecision"),
        ("app.schemas.approval", "ApprovalRequest"),
        ("app.schemas.policy", "PolicyDecision"),
        ("app.schemas.response", "ResponsePlan"),
        ("langchain_core.messages.ai", "AIMessage"),
        ("langchain_core.messages.human", "HumanMessage"),
        ("langchain_core.messages.system", "SystemMessage"),
    }
    assert set(checkpoint_module._ALLOWED_CONSTRUCTOR_KWARGS) == {
        ("langgraph.types", "Interrupt"),
    }
    assert checkpoint_module._ALLOWED_METHODS == frozenset(
        {("datetime", "datetime", "fromisoformat")}
    )
    assert checkpoint_module._ALLOWED_TAGS == frozenset(
        {"msgpack", "null", "bytes", "bytearray"}
    )


def test_whitelist_values_are_the_real_imported_classes():
    """表里存的是**已经导入好的类对象**,不是字符串路径 —— 这是"查表"的前提。"""
    assert (
        checkpoint_module._ALLOWED_PYDANTIC[("app.schemas.response", "ResponsePlan")]
        is ResponsePlan
    )
    assert (
        checkpoint_module._ALLOWED_CONSTRUCTOR_KWARGS[("langgraph.types", "Interrupt")]
        is Interrupt
    )


def test_serde_disables_pickle_fallback(serde: StrictCheckpointSerde):
    """pickle 回退必须显式关闭,而不是依赖上游默认值。"""
    assert serde.pickle_fallback is False


def test_saver_and_serde_are_subclasses_not_rewrites():
    """继承而非重写:子类自动继承锁、事件循环绑定与同步接口。

    这里断言的是"关系",不是实现细节 —— 若哪天改成代理包装,
    这条会变红,提醒复核是否漏掉了某个入口。
    """
    assert issubclass(SafeAsyncSqliteSaver, AsyncSqliteSaver)
    assert issubclass(StrictCheckpointSerde, JsonPlusSerializer)


def test_rejection_error_is_not_a_value_error_subclass():
    """`CheckpointPayloadRejectedError` **不得**是 `ValueError` 的子类。

    `loads_typed` 会把 `ormsgpack` 吞掉 hook 异常后重抛的裸 `ValueError`
    翻译成拒绝错误。若拒绝错误本身是 `ValueError` 的子类,这层翻译就会把
    它自己再包一次,错误语义随之漂移。
    """
    assert not issubclass(CheckpointPayloadRejectedError, ValueError)
    assert issubclass(CheckpointPayloadRejectedError, CheckpointStorageError)


# =====================================================================
# B. 正向:白名单覆盖真实需求(否则 TASK 3 的 STOP 条件成立)
# =====================================================================

@pytest.mark.parametrize(
    "name,obj", _legitimate_samples(), ids=[n for n, _ in _legitimate_samples()]
)
def test_legitimate_payloads_round_trip(serde, name, obj):
    """实测集合里的每个类型都必须能**往返** —— 这是白名单正确性的正向控制。"""
    got = serde.loads_typed(serde.dumps_typed(obj))
    assert type(got) is type(obj)
    assert got == obj


def test_null_tag_round_trips(serde):
    """`null` 标签是真实存在的(未写过的状态),必须接受。"""
    assert serde.loads_typed(("null", b"")) is None


def test_bytes_tags_are_passed_through(serde):
    """`bytes` / `bytearray` 是惰性直通,不涉及类型标识符,故一并接受。"""
    payload = _blob("hello")
    assert serde.loads_typed(("bytes", payload)) == _blob("hello")
    assert serde.loads_typed(("bytearray", payload)) == _blob("hello")


def test_msgpack_nil_decodes_to_none_faithfully(serde):
    """msgpack nil → `None` 是**忠实解码**,不是"静默替换"失败模式。

    记下来是为了报告能说准确:`None` 的合法来源只有 `null` 标签与 nil,
    二者都是编码器**显式写出去**的;不存在"解不出来所以给个 None"。
    """
    assert serde.loads_typed(("msgpack", b"\xc0")) is None


# =====================================================================
# C. 负向:构造载荷必须被显式拒绝
# =====================================================================

_ADVERSARIAL: list[tuple[str, tuple[str, bytes]]] = [
    (
        "code0 builtins.eval(上游 RCE 面)",
        _record(_ext(EXT_CONSTRUCTOR_SINGLE_ARG, ("builtins", "eval", "6*7"))),
    ),
    (
        "code1 位置参数构造",
        _record(_ext(EXT_CONSTRUCTOR_POS_ARGS, ("builtins", "eval", ("6*7",)))),
    ),
    (
        "code2 未授权构造器",
        _record(_ext(EXT_CONSTRUCTOR_KW_ARGS, ("builtins", "eval", {"source": "6*7"}))),
    ),
    (
        "code3 未授权类方法",
        _record(_ext(EXT_METHOD_SINGLE_ARG, ("builtins", "eval", "6*7", "__call__"))),
    ),
    (
        "code4 pydantic v1",
        _record(_ext(EXT_PYDANTIC_V1, ("builtins", "eval", {}))),
    ),
    (
        "code6 ndarray",
        _record(_ext(EXT_NUMPY_ARRAY, ("i8", (1,), "C", b"\x00"))),
    ),
    (
        "code5 未授权类(os.system)",
        _record(_ext(EXT_PYDANTIC_V2, ("os", "system", {}, "model_validate_json"))),
    ),
    (
        "code5 模块错但名字对",
        _record(
            _ext(
                EXT_PYDANTIC_V2,
                ("app.schemas.evil", "ResponsePlan", {}, "model_validate_json"),
            )
        ),
    ),
    (
        "code5 形状错误(3 元)",
        _record(_ext(EXT_PYDANTIC_V2, ("a", "b", {}))),
    ),
    (
        "code5 类型标识非字符串",
        _record(_ext(EXT_PYDANTIC_V2, (1, 2, {}, "model_validate_json"))),
    ),
    (
        "code2 形状错误(2 元)",
        _record(_ext(EXT_CONSTRUCTOR_KW_ARGS, ("langgraph.types", "Interrupt"))),
    ),
    (
        "code3 形状错误(3 元)",
        _record(_ext(EXT_METHOD_SINGLE_ARG, ("datetime", "datetime", "2026-01-01"))),
    ),
    (
        "code5 载荷不是序列",
        _record(_ext(EXT_PYDANTIC_V2, "not-a-sequence")),
    ),
]


@pytest.mark.parametrize("label,record", _ADVERSARIAL, ids=[c[0] for c in _ADVERSARIAL])
def test_adversarial_payloads_are_rejected(serde, label, record):
    """白名单之外的扩展载荷一律**响亮拒绝**,绝不返回 `None`。"""
    with pytest.raises(CheckpointPayloadRejectedError):
        serde.loads_typed(record)


def test_nested_escape_inside_allowed_payload_is_rejected(serde):
    """外层 code 正确、内层夹带逃逸载荷 —— 必须仍被拒绝。

    这是"白名单只查了外层"这类旁路的正面检验:嵌套的 msgpack 扩展
    走的是**同一个** hook,因此不存在"外层合法 ⇒ 内层放行"。
    """
    nested = _ext(EXT_CONSTRUCTOR_SINGLE_ARG, ("builtins", "eval", "6*7"))
    inner = _blob(
        (
            "app.schemas.response",
            "ResponsePlan",
            {"trap": nested},
            "model_validate_json",
        )
    )
    with pytest.raises(CheckpointPayloadRejectedError):
        serde.loads_typed(_record(ormsgpack.Ext(EXT_PYDANTIC_V2, inner)))


def test_allowed_class_with_invalid_body_is_rejected(serde):
    """名字在白名单里,但内容不合法 —— 必须拒绝,而不是 `model_construct` 放行。

    上游在 pydantic 分支失败时会退到 `cls.model_construct(**dump)`
    (**跳过校验**地造对象)。本模块刻意不退:载荷来自我们自己的编码器,
    校验失败即数据损坏或被篡改,"造个未校验的对象继续跑"比响亮失败更危险。
    """
    bad = _ext(
        EXT_PYDANTIC_V2,
        ("app.schemas.response", "ResponsePlan", {"indicator": "x"}, "model_validate_json"),
    )
    with pytest.raises(CheckpointPayloadRejectedError):
        serde.loads_typed(_record(bad))


@pytest.mark.parametrize("tag", ["json", "pickle", "bogus", "", "Msgpack"])
def test_disallowed_tags_are_rejected(serde, tag):
    """`json` 走另一条动态导入面;`pickle` 就是任意代码执行;其余一律不认。"""
    with pytest.raises(CheckpointPayloadRejectedError):
        serde.loads_typed((tag, b"{}"))


@pytest.mark.parametrize(
    "label,raw",
    [
        ("never-used 0xc1", b"\xc1"),
        ("truncated array", b"\x93\x01"),
        ("truncated str", b"\xa5ab"),
        ("truncated map", b"\x82\xa1k"),
        ("truncated ext", b"\xc7\x10\x00\x01"),
        ("empty", b""),
    ],
)
def test_malformed_msgpack_is_rejected_as_our_error(serde, label, raw):
    """损坏的字节必须落进**我们的**错误族,不能漏成裸 `ValueError`。

    实测坑:`ormsgpack.unpackb` 会吞掉 `ext_hook` 的异常,重抛一个
    **不带原因**的 `ValueError("ext_hook failed")`(`__cause__` 与
    `__context__` 双双为 None)。若不在这里翻译回来,"载荷被拒绝"就退化成
    裸 `ValueError` —— 既不是我们的错误族,也映射不到 503。
    """
    with pytest.raises(CheckpointPayloadRejectedError):
        serde.loads_typed(("msgpack", raw))


@pytest.mark.parametrize(
    "bad",
    [
        ("msgpack",),           # 少一元
        ("msgpack", b"", b""),  # 多一元
        "msgpack",              # 不是元组
        (b"msgpack", b""),      # 标签不是 str
        None,
    ],
)
def test_malformed_record_shape_is_rejected(serde, bad):
    """`(tag, bytes)` 记录本身的形状也必须校验。"""
    with pytest.raises(CheckpointPayloadRejectedError):
        serde.loads_typed(bad)  # type: ignore[arg-type]


def test_rejection_never_returns_none(serde):
    """把"绝不静默 None"写成一条可执行的断言。

    对每个构造载荷记录**要么抛错、要么返回非 None**;任何一条返回 None
    都说明"解不出来 → 给个 None"这条失败模式回来了。
    """
    records = [record for _, record in _ADVERSARIAL]
    for record in records:
        try:
            got = serde.loads_typed(record)
        except CheckpointPayloadRejectedError:
            continue
        pytest.fail(f"构造载荷未被拒绝且返回了 {got!r}")


# =====================================================================
# D. 反同义反复:负对照
# =====================================================================

def test_upstream_default_serde_executes_the_payload():
    """**负对照** —— 上游默认 serde 对同一载荷**真的执行** `builtins.eval`。

    没有这条,上面所有负向用例都可能恒真(护栏没牙也照样绿)。
    这条把"被测面真实存在"钉死:换回默认 serde,`6*7` 会被求值成 42。
    """
    upstream = JsonPlusSerializer()
    record = _record(_ext(EXT_CONSTRUCTOR_SINGLE_ARG, ("builtins", "eval", "6*7")))
    assert upstream.loads_typed(record) == {"k": 42}


def test_our_serde_rejects_exactly_what_the_negative_control_executes(serde):
    """把负对照与护栏放在一起:同一个载荷,一边执行、一边拒绝。"""
    record = _record(_ext(EXT_CONSTRUCTOR_SINGLE_ARG, ("builtins", "eval", "6*7")))
    assert JsonPlusSerializer().loads_typed(record) == {"k": 42}
    with pytest.raises(CheckpointPayloadRejectedError):
        serde.loads_typed(record)


# =====================================================================
# E. 端到端:真实图 + 真实 SQLite saver + 严格 serde
# =====================================================================

def _log_record(ip: str, index: int) -> dict:
    return {
        "timestamp": f"2026-09-10T08:{index:02d}:00Z",
        "event_type": "login_failed",
        "source": "sshd",
        "source_ip": ip,
        "username": "root",
        "status": "failed",
        "severity": "high",
        "message": "Failed password for root",
    }


@pytest.fixture
def hermetic_paths(tmp_path: Path) -> tuple[Path, Path]:
    """现场生成日志与情报 —— 不读仓库 data/。"""
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    with logs.open("w", encoding="utf-8") as fh:
        for i in range(BRUTE_FORCE_LOGINS):
            fh.write(json.dumps(_log_record(BRUTE_FORCE_IP, i)) + "\n")
    intel.write_text(
        json.dumps(
            {
                "indicator": BRUTE_FORCE_IP,
                "indicator_type": "ip",
                "malicious": True,
                "confidence": 90,
                "severity": "high",
                "tags": ["ssh-brute-force"],
                "source": "test-fixture",
                "first_seen": "2026-09-01T00:00:00Z",
                "last_seen": "2026-09-10T00:00:00Z",
                "description": "SSH brute force source",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return logs, intel


async def test_real_graph_pause_resume_round_trips_through_strict_serde(
    tmp_path: Path, hermetic_paths
):
    """**最强的一条正向证据**:真实图暂停,换一个 saver 实例,再恢复成功。

    这是 TASK 3 STOP 条件的直接检验 —— "若必需图对象无法安全往返,则 STOP"。
    暂停时图会把 ApprovalRequest / ResponsePlan / 消息等写进 checkpoint;
    恢复时全部要经过严格解码器。能恢复,就说明白名单覆盖了真实需求。
    """
    logs, intel = hermetic_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    ckpt_db = tmp_path / "checkpoints.db"
    cfg = {"configurable": {"thread_id": THREAD_ID}}
    state = {
        "messages": [HumanMessage(content="分析该 IP")],
        "iteration_count": 0,
        "indicator": BRUTE_FORCE_IP,
    }

    conn = await aiosqlite.connect(str(ckpt_db))
    saver = SafeAsyncSqliteSaver(conn, serde=build_checkpoint_serde())
    await saver.setup()
    graph = create_agent_graph(
        FakeLLMClient("分析完成"),
        hitl=HitlConfig(
            checkpointer=saver, audit_store=store, logs_path=logs, intel_path=intel
        ),
    )
    await graph.ainvoke(state, cfg)
    snapshot = await graph.aget_state(cfg)
    assert snapshot.next == ("human_approval",), "应停在人工审批节点"
    interrupt_id = snapshot.tasks[0].interrupts[0].id
    await conn.close()

    # 换一个**全新**的 saver 实例读同一个文件 —— 走的全是严格解码路径
    conn2 = await aiosqlite.connect(str(ckpt_db))
    saver2 = SafeAsyncSqliteSaver(conn2, serde=build_checkpoint_serde())
    graph2 = create_agent_graph(
        FakeLLMClient("分析完成"),
        hitl=HitlConfig(
            checkpointer=saver2, audit_store=store, logs_path=logs, intel_path=intel
        ),
    )
    try:
        await graph2.ainvoke(
            Command(
                resume={
                    "status": "approved",
                    "operator": "alice",
                    "reason": "测试",
                    "interrupt_id": interrupt_id,
                }
            ),
            cfg,
        )
    finally:
        await conn2.close()

    events = [r.event for r in store.list_audit(thread_id=THREAD_ID)]
    assert events.count("approval.decided") == 1


async def test_saver_serde_is_the_strict_one(tmp_path: Path):
    """注入必须**真的生效** —— 否则上面那条端到端证据可能走的是默认 serde。"""
    conn = await aiosqlite.connect(str(tmp_path / "ckpt.db"))
    try:
        saver = SafeAsyncSqliteSaver(conn, serde=build_checkpoint_serde())
        assert isinstance(saver.serde, StrictCheckpointSerde)
    finally:
        await conn.close()


async def test_real_service_path_checkpoint_carries_a_system_message(
    tmp_path: Path, hermetic_paths
):
    """**服务路径**(不是裸图)的载荷必须含 `SystemMessage`,且全部可严格解回。

    为什么单独要这一条 —— 同文件的
    `test_real_graph_pause_resume_round_trips_through_strict_serde` 走的是
    **裸图**,初始状态里只有 `HumanMessage`,而那个形状**不写 system prompt**。
    I1 的白名单口径正是从裸图推导的,于是漏掉了 `SystemMessage`,导致服务
    路径上 6/19 个 blob 解不开(暂停可以,恢复 503)。

    本用例把三件事钉进仓库:
      1. 真实 `TriageService.triage()` 落盘的载荷里**确实**出现
         `SystemMessage` 的模块标识(否则这条用例已失去意义,必须复核);
      2. 严格解码器能解**每一个** blob —— 有一个解不开就说明白名单漏项;
      3. 换一个全新 saver 实例仍能 `resume` 成功(重启恢复仍然可用)。
    """
    logs, intel = hermetic_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    ckpt_db = tmp_path / "checkpoints.db"
    serde = build_checkpoint_serde()

    # ---- 进程 A:真实 TriageService 发起 triage → 停在人工审批 ----
    conn = await aiosqlite.connect(str(ckpt_db))
    try:
        service = TriageService(
            create_agent_graph(
                FakeLLMClient("分析完成"),
                hitl=HitlConfig(
                    checkpointer=SafeAsyncSqliteSaver(conn, serde=serde),
                    audit_store=store,
                    logs_path=logs,
                    intel_path=intel,
                ),
            ),
            store,
        )
        result = await service.triage(BRUTE_FORCE_IP)
        assert result.outcome.status == "pending_approval", result.outcome.status
        thread_id = result.outcome.thread_id
    finally:
        await conn.close()

    # ---- 直接读落盘载荷(不走 serde,看的是原始字节) ----
    raw = sqlite3.connect(str(ckpt_db))
    try:
        rows = raw.execute("SELECT type, checkpoint FROM checkpoints").fetchall()
        writes = raw.execute("SELECT type, value FROM writes").fetchall()
    finally:
        raw.close()

    blobs = [(t, bytes(b)) for t, b in (rows + writes) if b is not None]
    assert blobs, "服务路径必须写出 checkpoint 载荷"
    assert any(
        b"langchain_core.messages.system" in payload for _, payload in blobs
    ), (
        "服务路径的载荷里**没有** SystemMessage —— 要么这条用例已失去意义,"
        "要么服务路径不再注入 system prompt;两种情况都必须人工复核"
    )

    undecodable: list[tuple[str, str]] = []
    for type_, payload in blobs:
        try:
            serde.loads_typed((type_, payload))
        except Exception as exc:  # noqa: BLE001 - 逐条归类,不掩盖
            undecodable.append((type_, type(exc).__name__))
    assert undecodable == [], f"严格解码器解不开服务路径的载荷:{undecodable}"

    # ---- 进程 B:全新 saver 实例,只从磁盘读 → 恢复成功 ----
    conn2 = await aiosqlite.connect(str(ckpt_db))
    try:
        service2 = TriageService(
            create_agent_graph(
                FakeLLMClient("分析完成"),
                hitl=HitlConfig(
                    checkpointer=SafeAsyncSqliteSaver(
                        conn2, serde=build_checkpoint_serde()
                    ),
                    audit_store=store,
                    logs_path=logs,
                    intel_path=intel,
                ),
            ),
            store,
        )
        resumed = await service2.resume(
            thread_id, status="approved", operator="alice"
        )
    finally:
        await conn2.close()

    assert resumed.outcome.status == "completed", resumed.outcome.status
    decided = [
        r for r in store.list_audit(thread_id=thread_id) if r.event == "approval.decided"
    ]
    assert len(decided) == 1


# =====================================================================
# F. 后端故障归因(错误族边界,不触网、不触盘)
# =====================================================================

def test_backend_error_family_is_separate_from_payload_rejection():
    """两个子类同属一个族,但可区分:一个说"存储坏了",一个说"数据坏了"。"""
    assert issubclass(CheckpointBackendUnavailableError, CheckpointStorageError)
    assert issubclass(CheckpointPayloadRejectedError, CheckpointStorageError)
    assert not issubclass(
        CheckpointBackendUnavailableError, CheckpointPayloadRejectedError
    )


def test_backend_errors_are_not_value_errors():
    """错误族必须是 `RuntimeError` 系 —— 与 pydantic / msgpack 的 `ValueError` 分开。"""
    assert issubclass(CheckpointStorageError, RuntimeError)
    assert not issubclass(CheckpointStorageError, ValueError)


# =====================================================================
# G. 隔离:仓库里不得出现 checkpoints.db
# =====================================================================

def test_no_repo_checkpoint_db_created():
    """与 audit.db 同一条硬线:测试不得在仓库里创建 checkpoints.db。

    本文件所有用例都走 tmp_path;若这条失败,说明某个用例漏传了路径。
    """
    assert not (_REPO_ROOT / "data" / "checkpoints.db").exists()


def test_module_source_mentions_no_absolute_data_path():
    """源码里不得出现硬编码的仓库绝对路径 —— 默认路径必须是相对的。"""
    src = Path(checkpoint_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:\\\\", src), "源码含硬编码盘符路径"
