"""Phase 8.4 TriageService 测试(service 层)。

覆盖三组契约:

1. **发起与沉淀**
   triage() 生成 thread_id → 执行图 → 聚合 TriageOutcome →
   落 incident(+ 审批路径落 action_requests)。
   含 D6 的**行为级**验证:event_type 必须真的到达 plan 节点。

2. **恢复与校验门**
   resume() 的判定树把框架的 4 个"静默行为"变成明确错误:
   未知 thread → 404 语义;已完成 → 409;checkpoint 丢失 → 409。
   校验门是必需的 —— 实测确认框架在未知 thread 上 Command(resume=)
   **不报错**,直接从 START 新起一轮(看起来像成功)。

3. **契约与护栏**
   - resume() 签名里没有 interrupt_id(D7:客户端不能指定"审批的是哪一次");
   - triage() 签名里没有 thread_id(D3:thread_id 只能服务端生成);
   - app.core.triage **不 import app.api**(core 不许反向依赖传输层);
   - 答案提取与 SecurityAgent 同契约(等价实现,防止悄悄漂移)。

全部 hermetic:数据文件与 audit.db 都由 tmp_path 现场生成,
不读仓库 data/,也不在仓库里创建 audit.db。
"""
import ast
import inspect
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from collections import Counter
from contextlib import closing
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import NamedTuple

import aiosqlite
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.api.main import create_app
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import (
    HUMAN_APPROVAL_NODE,
    CheckpointLostError,
    NotAwaitingApprovalError,
    TriageDataUnavailableError,
    TriageResult,
    TriageService,
    UnknownThreadError,
    _extract_answer,
)
from app.schemas.approval import TriageOutcome
from app.security.checkpoint import (
    SafeAsyncSqliteSaver,
    build_checkpoint_serde,
)
from app.security.store import SqliteAuditStore

# 与 test_hitl_graph.py 同一套夹具语义:
# 30 次失败登录 + 恶意情报 → critical → 含 block_ip / isolate_host /
# reset_credentials → 策略要求人工审批
BRUTE_FORCE_IP = "203.0.113.66"
BRUTE_FORCE_LOGINS = 30

# 10 次失败登录,无情报命中 → low → 仅 monitor → 策略放行
LOW_RISK_IP = "198.51.100.7"
LOW_RISK_LOGINS = 10

REPLY = "分析完成"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRIAGE_SOURCE = _REPO_ROOT / "app" / "core" / "triage.py"


# ---------- hermetic 数据夹具 ----------

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


def _write_logs(path: Path) -> None:
    """两个 indicator 写在同一个文件:collect_evidence 按 source_ip 过滤,互不干扰。"""
    with path.open("w", encoding="utf-8") as f:
        for i in range(BRUTE_FORCE_LOGINS):
            f.write(json.dumps(_log_record(BRUTE_FORCE_IP, i)) + "\n")
        for i in range(LOW_RISK_LOGINS):
            f.write(json.dumps(_log_record(LOW_RISK_IP, i)) + "\n")


def _write_intel(path: Path) -> None:
    """只有 BRUTE_FORCE_IP 有恶意情报;LOW_RISK_IP 不命中。"""
    path.write_text(json.dumps({
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
    }) + "\n", encoding="utf-8")


class _Env(NamedTuple):
    service: TriageService
    store: SqliteAuditStore
    logs: Path
    intel: Path
    db_path: Path


def _service(
    store: SqliteAuditStore,
    logs: Path,
    intel: Path,
    checkpointer,
    reply: str = REPLY,
) -> TriageService:
    hitl = HitlConfig(
        checkpointer=checkpointer,
        audit_store=store,
        logs_path=logs,
        intel_path=intel,
    )
    return TriageService(create_agent_graph(FakeLLMClient(reply), hitl=hitl), store)


@pytest.fixture
def data_paths(tmp_path: Path) -> tuple[Path, Path]:
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)
    return logs, intel


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.db"


@pytest.fixture
def store(db_path: Path) -> SqliteAuditStore:
    return SqliteAuditStore(db_path)


@pytest.fixture
def env(data_paths, store, db_path) -> _Env:
    logs, intel = data_paths
    return _Env(
        service=_service(store, logs, intel, InMemorySaver()),
        store=store,
        logs=logs,
        intel=intel,
        db_path=db_path,
    )


def _incident_rows(db_path: Path) -> list[dict]:
    """直接读 incidents 表。

    store 目前**没有** list_incidents 读接口(Phase 8.4 未新增读 API),
    而 TriageOutcome / TriageResult 也刻意不暴露 incident_id ——
    所以这里直连 sqlite 验证"确实落了库"。schema 由 test_store.py 守着。
    """
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM incidents")]


def _action_rows(db_path: Path) -> list[dict]:
    """直接读 action_requests 表(同样没有读接口)。"""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM action_requests")]


# =====================================================================
# A. triage() 发起与沉淀
# =====================================================================

async def test_triage_pauses_when_approval_required(env):
    """高风险 → 图停在 human_approval → pending_approval + interrupt_id。"""
    result = await env.service.triage(BRUTE_FORCE_IP)

    assert result.outcome.status == "pending_approval"
    assert result.interrupt_id
    assert result.outcome.thread_id
    assert result.outcome.plan.risk_level == "critical"
    assert result.outcome.approval_request is not None
    # 还没人审批
    assert result.outcome.approval is None


async def test_triage_completes_when_policy_allows(env):
    """低风险 → 策略放行 → 图直接跑完 → completed,没有 interrupt。"""
    result = await env.service.triage(LOW_RISK_IP)

    assert result.outcome.status == "completed"
    assert result.interrupt_id is None
    assert result.outcome.plan.risk_level == "low"
    assert result.outcome.approval_request is None
    assert result.outcome.approval is None


async def test_triage_answer_comes_from_llm(env):
    result = await env.service.triage(LOW_RISK_IP)
    assert result.outcome.answer == REPLY


async def test_triage_thread_id_is_server_generated_hex32(env):
    """D3:thread_id 是服务端 uuid4().hex —— 32 位小写十六进制。"""
    result = await env.service.triage(LOW_RISK_IP)
    thread_id = result.outcome.thread_id

    assert len(thread_id) == 32
    assert all(c in "0123456789abcdef" for c in thread_id)


async def test_triage_thread_ids_are_unique_per_call(env):
    """同一 indicator 两次 triage 必须落在两个独立 thread 上。"""
    first = await env.service.triage(LOW_RISK_IP)
    second = await env.service.triage(LOW_RISK_IP)

    assert first.outcome.thread_id != second.outcome.thread_id


async def test_triage_pending_records_incident(env):
    """审批路径:incident 落库,字段来自规则引擎产出的 plan。"""
    result = await env.service.triage(BRUTE_FORCE_IP)
    rows = _incident_rows(env.db_path)

    assert len(rows) == 1
    assert rows[0]["indicator"] == BRUTE_FORCE_IP
    assert rows[0]["risk_level"] == "critical"
    assert rows[0]["score"] == result.outcome.plan.assessment.score
    assert rows[0]["summary"] == result.outcome.plan.summary


async def test_triage_pending_records_action_request_with_incident_id(env):
    """审批单按 action 粒度落库,且全部指向同一个 incident。"""
    result = await env.service.triage(BRUTE_FORCE_IP)
    rows = env.store.list_action_rows(thread_id=result.outcome.thread_id)

    assert {r["action"]["action_type"] for r in rows} == {
        "block_ip", "isolate_host", "reset_credentials",
    }
    incident_ids = {r["incident_id"] for r in rows}
    assert len(incident_ids) == 1
    assert None not in incident_ids  # 服务层补上了 incident_id


async def test_triage_pending_leaves_rows_pending(env):
    """派生状态:还没 approval.decided → 这些动作行仍是待审批。"""
    result = await env.service.triage(BRUTE_FORCE_IP)
    pending = env.store.pending_action_rows(thread_id=result.outcome.thread_id)

    assert len(pending) == 3


async def test_triage_allow_records_no_action_request(env):
    """allow 路径不得写审批单。

    为放行路径伪造一张审批单,会让 pending_action_rows 永远派生出一批
    "待审批",把派生状态的语义弄坏(它们永远等不到 approval.decided)。
    """
    result = await env.service.triage(LOW_RISK_IP)

    assert env.store.list_action_rows(thread_id=result.outcome.thread_id) == []
    assert env.store.pending_action_rows(thread_id=result.outcome.thread_id) == []


async def test_triage_allow_still_records_incident(env):
    """放行也是一次判定的沉淀 → 同样落 incident(只是没有审批单)。"""
    await env.service.triage(LOW_RISK_IP)
    assert len(_incident_rows(env.db_path)) == 1


async def test_event_type_reaches_plan_node(env):
    """D6 的行为级验证:event_type 必须真的到达 collect_evidence。

    AgentState 若没声明 event_type,LangGraph 会**静默丢弃**这个初始键
    (实测:ainvoke 传了不报错,节点里读不到),过滤就形同虚设。

    证据取自 plan.assessment.evidence.log_event_count:
        不带 event_type          → 30 条 login_failed 全部计入 → 30
        event_type=login_success → 过滤后 0 条               → 0
        event_type=login_failed  → 与不带一致                → 30
    注意 failed_login_count 恒为 30:collect_evidence 内部**固定**用
    login_failed 查爆破特征,不受调用方过滤影响 —— 所以这里断言的是
    log_event_count,而不是风险等级(两者在本夹具下都是 critical)。
    """
    plain = await env.service.triage(BRUTE_FORCE_IP)
    filtered_out = await env.service.triage(BRUTE_FORCE_IP, event_type="login_success")
    filtered_in = await env.service.triage(BRUTE_FORCE_IP, event_type="login_failed")

    assert plain.outcome.plan.assessment.evidence.log_event_count == BRUTE_FORCE_LOGINS
    assert filtered_out.outcome.plan.assessment.evidence.log_event_count == 0
    assert filtered_in.outcome.plan.assessment.evidence.log_event_count == BRUTE_FORCE_LOGINS


async def test_triage_missing_logs_raises_data_unavailable(tmp_path, store):
    """数据文件缺失 → 领域错误(API 层映射 503),而不是裸 FileNotFoundError。"""
    intel = tmp_path / "threat_intel.jsonl"
    _write_intel(intel)
    service = _service(store, tmp_path / "missing.jsonl", intel, InMemorySaver())

    with pytest.raises(TriageDataUnavailableError):
        await service.triage(BRUTE_FORCE_IP)


async def test_plan_failure_writes_exactly_one_plan_failed(tmp_path, store, db_path):
    """失败路径必须留痕:恰好一条 plan.failed。

    只记成功的审计是幸存者偏差 —— "判定失败了多少次、为什么失败"必须可查。
    同时验证失败**不留业务痕迹**:没有 incident、没有 action_requests、
    没有 plan.created(否则会把"没判定成功"记成"判定成功")。

    Phase 9.1-A 追加:失败会在 checkpoint 里留下 next=('plan',) 的残留快照,
    服务层现在清掉它(该 thread 是终态,不可能再被恢复)。
    """
    intel = tmp_path / "threat_intel.jsonl"
    _write_intel(intel)
    saver = InMemorySaver()
    service = _service(store, tmp_path / "missing.jsonl", intel, saver)

    with pytest.raises(TriageDataUnavailableError):
        await service.triage(BRUTE_FORCE_IP)

    records = store.list_audit()
    assert [r.event for r in records] == ["plan.failed"]

    failed = records[0]
    assert failed.actor == "system"
    assert failed.outcome == "failed"
    assert failed.detail == {
        "indicator": BRUTE_FORCE_IP,
        "error_type": "FileNotFoundError",
    }
    assert _incident_rows(db_path) == []
    assert _action_rows(db_path) == []

    # 残留快照已清理:失败 thread 不在 checkpoint 存储里
    assert failed.thread_id not in saver.storage


async def test_triage_data_error_message_is_sanitized(tmp_path, store):
    """错误消息与审计 detail 都不得泄露绝对路径。

    collect_evidence 抛出的 FileNotFoundError 消息里含部署路径,
    直接回传等于把内部目录结构告诉客户端;审计 detail 若照抄异常文本,
    等于把路径永久写进库 —— 两处都要挡。
    """
    intel = tmp_path / "threat_intel.jsonl"
    _write_intel(intel)
    missing = tmp_path / "missing.jsonl"
    service = _service(store, missing, intel, InMemorySaver())

    with pytest.raises(TriageDataUnavailableError) as excinfo:
        await service.triage(BRUTE_FORCE_IP)

    message = str(excinfo.value)
    assert str(missing) not in message
    assert str(tmp_path) not in message
    assert "/" not in message and "\\" not in message

    # 审计 detail 是自包含取证负载 —— 同样不能夹带路径或异常原文
    detail = store.list_audit(event="plan.failed")[0].detail
    assert set(detail) == {"indicator", "error_type"}
    assert not any(
        isinstance(v, str) and ("\\" in v or "/" in v) for v in detail.values()
    )


# =====================================================================
# B. resume() 与校验门
# =====================================================================

async def test_resume_approved_completes(env):
    paused = await env.service.triage(BRUTE_FORCE_IP)
    result = await env.service.resume(
        paused.outcome.thread_id, status="approved", operator="analyst-1"
    )

    assert result.outcome.status == "completed"
    assert result.outcome.approval.status == "approved"
    assert result.outcome.approval.operator == "analyst-1"
    assert result.interrupt_id == paused.interrupt_id


async def test_resume_denied_completes(env):
    paused = await env.service.triage(BRUTE_FORCE_IP)
    result = await env.service.resume(
        paused.outcome.thread_id,
        status="denied",
        operator="analyst-2",
        reason="误报,该 IP 是内部扫描器",
    )

    assert result.outcome.status == "completed"
    assert result.outcome.approval.status == "denied"
    assert result.outcome.approval.reason == "误报,该 IP 是内部扫描器"


async def test_resume_keeps_approval_request_for_audit_trail(env):
    """D2:终态仍带上 approval_request —— "批的是什么"必须留在响应里。

    这正是把 TriageOutcome 的校验从"当且仅当"放宽为单向蕴含的原因:
    旧的双向校验会把这个唯一合法的终态判为非法。
    """
    paused = await env.service.triage(BRUTE_FORCE_IP)
    result = await env.service.resume(
        paused.outcome.thread_id, status="approved", operator="analyst-1"
    )

    assert result.outcome.approval_request is not None
    assert result.outcome.approval_request.indicator == BRUTE_FORCE_IP
    assert result.outcome.plan is not None


async def test_resume_uses_server_side_interrupt_id(env):
    """D7:决定里记下的 interrupt_id 来自 checkpoint,不是客户端传的。"""
    paused = await env.service.triage(BRUTE_FORCE_IP)
    result = await env.service.resume(
        paused.outcome.thread_id, status="approved", operator="analyst-1"
    )

    assert paused.interrupt_id
    assert result.outcome.approval.interrupt_id == paused.interrupt_id


async def test_resume_writes_approval_decided_exactly_once(env):
    """暂停 + 恢复全过程,四个审计事件各恰好一条。

    这是"interrupt() 前无副作用"的**行为级**验证:
    若 approval.decided 被写到 interrupt() 之前,resume 重放节点会写两次
    (第二次撞主键 IntegrityError)。
    """
    paused = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = paused.outcome.thread_id
    await env.service.resume(thread_id, status="approved", operator="analyst-1")

    counts = Counter(r.event for r in env.store.list_audit(thread_id=thread_id))
    assert counts == {
        "plan.created": 1,
        "policy.evaluated": 1,
        "approval.requested": 1,
        "approval.decided": 1,
    }


async def test_resume_does_not_replay_plan_node(env):
    """resume 不重放已完成节点 —— plan 节点只跑一次(不重新采集证据)。"""
    paused = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = paused.outcome.thread_id
    await env.service.resume(thread_id, status="approved", operator="analyst-1")

    events = [r.event for r in env.store.list_audit(thread_id=thread_id)]
    assert events.count("plan.created") == 1
    assert events.count("policy.evaluated") == 1
    # 顺序也固定:plan → policy → requested → decided
    assert events == [
        "plan.created", "policy.evaluated", "approval.requested", "approval.decided",
    ]


async def test_resume_clears_pending_rows(env):
    """审批完成后,派生状态不再认为有待审批动作。"""
    paused = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = paused.outcome.thread_id
    assert env.store.pending_action_rows(thread_id=thread_id)

    await env.service.resume(thread_id, status="approved", operator="analyst-1")

    assert env.store.pending_action_rows(thread_id=thread_id) == []


async def test_resume_unknown_thread_raises(env):
    """未知 thread → UnknownThreadError(API 层映射 404)。

    框架本身在未知 thread 上 Command(resume=) **不报错**,会从 START
    新起一轮 —— 校验门必须把它变成明确错误。
    """
    with pytest.raises(UnknownThreadError):
        await env.service.resume(
            "thread-never-existed", status="approved", operator="analyst-1"
        )


async def test_resume_completed_thread_raises(env):
    """已完成 → NotAwaitingApprovalError(409),且不是 checkpoint 丢失。"""
    paused = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = paused.outcome.thread_id
    await env.service.resume(thread_id, status="approved", operator="analyst-1")

    with pytest.raises(NotAwaitingApprovalError) as excinfo:
        await env.service.resume(thread_id, status="denied", operator="analyst-2")

    # 精确区分:是"你来晚了",不是"服务重启过"
    assert not isinstance(excinfo.value, CheckpointLostError)


async def test_resume_after_checkpoint_loss_raises(env, data_paths, store):
    """服务重启场景:审计还在,checkpoint 没了 → 明确报错,不假装成功。

    这是 InMemorySaver 的直接后果(见 triage.py 已知限制 2)。
    用**同一个 store** + 一个全新的 checkpointer 模拟进程重启:
    审计证明审批曾被请求(有未决 action_requests),但新进程里没有 state。
    """
    paused = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = paused.outcome.thread_id
    logs, intel = data_paths

    restarted = _service(store, logs, intel, InMemorySaver())

    with pytest.raises(CheckpointLostError):
        await restarted.resume(thread_id, status="approved", operator="analyst-1")


def test_checkpoint_lost_is_not_awaiting_subclass():
    """两个错误对客户端同为 409,但根因不同 → 继承关系保证状态码一致。"""
    assert issubclass(CheckpointLostError, NotAwaitingApprovalError)


# =====================================================================
# C. 契约与护栏
# =====================================================================

def test_resume_does_not_accept_interrupt_id():
    """D7:客户端不能指定 interrupt_id(否则可伪造"审批的是哪一次暂停")。

    request_id(9.3-D)是**运维关联**参数,与 interrupt_id 无关 ——
    它由 API 边界生成,不携带任何恢复句柄语义,故不违反本护栏。
    """
    params = set(inspect.signature(TriageService.resume).parameters)
    assert "interrupt_id" not in params
    assert params == {"self", "thread_id", "status", "operator", "reason", "request_id"}


def test_triage_thread_id_is_never_client_supplied():
    """D3(A3-3 修订):thread_id 只能由**服务端**决定,客户端无法指定。

    A3-3 起 `TriageService.triage` 多了一个**可选**的 `thread_id` 参数 ——
    存在它的唯一理由是:HTTP 边界必须**在跑图之前**写好线程归属,而归属
    记录里要带 thread_id。这不改变 D3 的性质:

    - 参数默认 `None` ⇒ 不传时本方法照旧自己生成 uuid4;
    - 请求 DTO(`TriageRequest`)里**没有** thread_id 字段 ⇒ 客户端仍然
      没有任何输入面,无法复用别人暂停中的 state。

    这条护栏同时钉住"签名被改宽了"与"DTO 被加回了 thread_id"两件事。
    """
    params = inspect.signature(TriageService.triage).parameters
    assert params["thread_id"].default is None
    assert set(params) == {
        "self", "indicator", "event_type", "request_id", "thread_id",
    }

    from app.api.schemas import TriageRequest

    assert "thread_id" not in TriageRequest.model_fields


def test_triage_module_does_not_import_api_layer():
    """core 不许反向依赖传输层:app.core.triage 不得 import app.api。

    用 AST 看**真实 import**,不用正则 —— 模块 docstring 里提到
    "app.api" 字样是正常的说明文字,正则会把文档当成违规。
    """
    tree = ast.parse(_TRIAGE_SOURCE.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert not [m for m in modules if m.startswith("app.api")], sorted(modules)


def test_human_approval_node_name_matches_graph(env):
    """HUMAN_APPROVAL_NODE 常量必须与图里真实的节点名一致。

    它是校验门的判定依据(snapshot.next == (HUMAN_APPROVAL_NODE,));
    节点一旦改名而常量没跟着改,校验门会把所有合法恢复都判成 409。
    """
    nodes = set(env.service._graph.get_graph().nodes) - {"__start__", "__end__"}
    assert HUMAN_APPROVAL_NODE in nodes


def test_extract_answer_matches_security_agent():
    """triage 的答案提取与 SecurityAgent 同契约(等价实现,不许漂移)。

    刻意复制而不是调用私有方法(跨模块访问私有成员会把两处焊死),
    代价是可能分叉 —— 这条测试就是那个代价的对冲。
    """
    from app.core.agent import SecurityAgent

    cases = [
        [AIMessage(content="done")],
        [HumanMessage(content="hi"), AIMessage(content="done")],
        [AIMessage(content="first"), AIMessage(content="second")],
        [HumanMessage(content="hi")],
        [AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])],
        [AIMessage(content="done"), AIMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "2"}]
        )],
    ]
    for messages in cases:
        assert _extract_answer(messages) == SecurityAgent._extract_final_answer(messages)


def test_triage_result_is_frozen():
    """TriageResult 不可变:结果是事实,不允许被就地改写。"""
    result = TriageResult(
        outcome=TriageOutcome(thread_id="thread-x", status="completed")
    )
    with pytest.raises(FrozenInstanceError):
        result.interrupt_id = "tampered"


def test_no_repo_audit_db_created():
    """所有用例都走 tmp_path;仓库里不得出现 audit.db。"""
    assert not (_REPO_ROOT / "data" / "audit.db").exists()


# =====================================================================
# F. v0.4.0 S1:**真正的跨进程**持久化证据(§7.1)
# =====================================================================
#
# 为什么不能用"同一个进程里换一个 saver 实例"充数:那只能证明**对象**被
# 重建,证明不了**进程内内存**不是唯一的真相源。真正的持久化证据必须让
# 第一个进程**彻底退出**(它持有的内存随进程消失),再由**第二个操作系统
# 进程**从磁盘上的同一个文件恢复。
#
# 因此本用例启动一个真实子进程做暂停,子进程退出后再在父进程里恢复。
# 断言里显式比较两个 PID,把"确实是两个进程"钉死 —— 否则哪天有人把子进程
# 换成同进程调用,这条用例会静默退化成它本想避免的那种伪证据。

_CHILD_SCRIPT = textwrap.dedent(
    """
    import asyncio, json, os, sys
    from datetime import datetime, timezone
    from pathlib import Path

    import aiosqlite

    from app.core.graph import HitlConfig, create_agent_graph
    from app.core.llm import FakeLLMClient
    from app.core.triage import TriageService
    from app.schemas.ownership import ThreadOwnership
    from app.security.checkpoint import SafeAsyncSqliteSaver, build_checkpoint_serde
    from app.security.store import SqliteAuditStore


    async def main() -> None:
        logs, intel = Path(sys.argv[1]), Path(sys.argv[2])
        audit_db, ckpt_db = Path(sys.argv[3]), Path(sys.argv[4])
        indicator, owner, approver = sys.argv[5], sys.argv[6], sys.argv[7]

        store = SqliteAuditStore(audit_db)
        conn = await aiosqlite.connect(str(ckpt_db))
        saver = SafeAsyncSqliteSaver(conn, serde=build_checkpoint_serde())
        await saver.setup()
        service = TriageService(
            create_agent_graph(
                FakeLLMClient("分析完成"),
                hitl=HitlConfig(
                    checkpointer=saver,
                    audit_store=store,
                    logs_path=logs,
                    intel_path=intel,
                ),
            ),
            store,
        )
        result = await service.triage(indicator=indicator)
        thread_id = result.outcome.thread_id
        # 复刻真实 /triage 的另一半:登记逐线程审批指派(授权事实落在
        # **审计库**里,与 checkpoint 是两套独立存储)。
        store.record_thread_ownership(ThreadOwnership(
            thread_id=thread_id,
            owner=owner,
            approvers=(approver,),
            created_at=datetime.now(timezone.utc),
        ))
        # 关连接 = 刷 WAL。之后本进程退出,内存里的一切随之消失。
        await conn.close()
        print(json.dumps({
            "pid": os.getpid(),
            "thread_id": thread_id,
            "interrupt_id": result.interrupt_id,
        }))


    asyncio.run(main())
    """
)


def _run_child_pause(tmp_path: Path, name: str, logs, intel, db_path, ckpt_db,
                     owner: str, approver: str) -> dict:
    """在**真实子进程**里暂停一次,返回子进程自报的 pid / thread_id。"""
    script = tmp_path / name
    script.write_text(_CHILD_SCRIPT, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # 不在仓库里留 __pycache__

    child = subprocess.run(
        [sys.executable, str(script), str(logs), str(intel),
         str(db_path), str(ckpt_db), BRUTE_FORCE_IP, owner, approver],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout.strip().splitlines()[-1])


async def test_pause_survives_a_real_process_restart_and_can_be_resumed(
    tmp_path: Path, data_paths, db_path
):
    """**两个真实 OS 进程**:进程 A 暂停并退出,进程 B 从磁盘恢复并跑完。

    这是 v0.4.0 S1 的核心承诺:"HITL 状态在进程重启后仍然可恢复"。
    在 S1 之前 checkpoint 是进程内内存,重启即丢 —— 恢复只能 409。
    """
    from tests.conftest import TEST_APPROVER_SUBJECT, TEST_OWNER_SUBJECT

    logs, intel = data_paths
    ckpt_db = tmp_path / "checkpoints.db"
    payload = _run_child_pause(
        tmp_path, "child_pause.py", logs, intel, db_path, ckpt_db,
        TEST_OWNER_SUBJECT, TEST_APPROVER_SUBJECT,
    )

    # 断言"确实是另一个进程" —— 这条是整条用例的意义所在。
    assert payload["pid"] != os.getpid(), "子进程必须是真的另一个 OS 进程"
    thread_id = payload["thread_id"]
    assert payload["interrupt_id"]

    # ---- 父进程:全新的连接、全新的 saver,只从磁盘读 ------------------
    store = SqliteAuditStore(db_path)
    conn = await aiosqlite.connect(str(ckpt_db))
    try:
        service = _service(
            store, logs, intel,
            SafeAsyncSqliteSaver(conn, serde=build_checkpoint_serde()),
        )
        result = await service.resume(
            thread_id, status="approved", operator="approver-1"
        )
    finally:
        await conn.close()

    assert result.outcome.status == "completed"
    events = [r.event for r in store.list_audit(thread_id=thread_id)]
    assert events.count("approval.decided") == 1
    # 暂停与恢复各自恰好一次,没有因为重启而重放。
    assert events.count("approval.requested") == 1
    assert events.count("plan.created") == 1


def test_restart_then_authorization_still_holds(tmp_path, data_paths, db_path,
                                                monkeypatch):
    """重启之后**对象级授权与禁止自审批仍然生效**(端到端,真 lifespan)。

    走的是**真实** `create_app()` 的 lifespan(不注入 triage_service),
    因此 app 在自己的事件循环里打开子进程留下的那个 checkpoint 文件 ——
    这正是"重启后能不能恢复"的真实形态。

    断言两件事:
      1. 被指派的 approver 能恢复 → 200(持久化 checkpoint 确实可用);
      2. 属主自己仍然被拒 → 403(禁止自审批没有被持久化绕过)。
    """
    from app.api import main as api_main
    from app.core.config import get_settings
    from tests.conftest import (
        APPROVER_HEADERS,
        OWNER_HEADERS,
        TEST_APPROVER_SUBJECT,
        TEST_OWNER_SUBJECT,
    )

    logs, intel = data_paths
    ckpt_db = tmp_path / "checkpoints_restart.db"
    payload = _run_child_pause(
        tmp_path, "child_pause_auth.py", logs, intel, db_path, ckpt_db,
        TEST_OWNER_SUBJECT, TEST_APPROVER_SUBJECT,
    )
    assert payload["pid"] != os.getpid()
    thread_id = payload["thread_id"]

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(api_main, "LLMClient", lambda *a, **k: FakeLLMClient("ok"))
    monkeypatch.setenv("AUDIT_DB_PATH", str(db_path))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(ckpt_db))
    monkeypatch.delenv("AUDIT_BACKEND", raising=False)
    get_settings.cache_clear()

    try:
        # 1. 属主自审批 → 403(禁止自审批跨重启依然成立)
        with TestClient(create_app(), headers=OWNER_HEADERS) as client:
            denied = client.post("/resume", json={
                "thread_id": thread_id, "status": "approved", "operator": "x",
            })
        assert denied.status_code == 403, denied.text

        # 2. 被指派的 approver → 200(重启后 checkpoint 真的可用)
        with TestClient(create_app(), headers=APPROVER_HEADERS) as client:
            ok = client.post("/resume", json={
                "thread_id": thread_id, "status": "approved", "operator": "x",
            })
        assert ok.status_code == 200, ok.text
    finally:
        get_settings.cache_clear()

    events = [r.event for r in SqliteAuditStore(db_path).list_audit(thread_id=thread_id)]
    assert events.count("approval.decided") == 1


