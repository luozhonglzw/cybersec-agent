"""Phase 9.2-D-2d —— 路径限定(READINESS REMEDIATION)。

本文件是 `app/evaluation/llm/confinement.py` 的**离线**验收证据,对应
READINESS REMEDIATION 指令 §E 的 14 项必需测试。

被修复的缺陷(D1)
-----------------
四个生产工具共 **6 个**路径类参数(`data_path` ×2 / `logs_path` ×2 /
`intel_path` ×2)直接进 `Path(value)` / `open()`,**没有任何守卫**;
`app.core.graph.tools_node` 把 provider 生成的 `tool_calls[i]["args"]`
**原样**交给 `tool.ainvoke`。离线路径之所以看起来正常,是因为 `ScriptedLLM`
**主动**把授权路径传进工具 —— 那是**约定**,不是**守卫**。

因此本文件的核心不是"包装器能不能跑",而是四条可证伪的性质:

    provider 给的越权路径 **从不** 被读        外部文件 / `../` 穿越 / 仓库 data
    缺路径参数 **不得** 回落仓库 `data/`        省略路径 = 绑定到本单元授权路径
    provider 原始参数 **不被抹掉**             原值与生效值并存,走独立证据通道
    既有指标语义 **不变**                      指标只看 provider 给了什么

每条都配了**反同义反复**断言:把限定摘掉,断言必须变红。

零 provider
-----------
全部模型都是合成对象(零网络、零额度)。模块级 autouse 守卫在收尾处断言
**零出口**;第 14 项另用 AST 检查本文件自身没有 import 任何 provider 客户端。
"""
from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import json
import os
import pathlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage

from app.evaluation.llm.adapters import B2PrimeGraphAdapter, B3FullAgentAdapter
from app.evaluation.llm.confinement import (
    PATH_ARGUMENT_NAMES,
    PATH_ARGUMENT_TARGETS,
    PathBindingLog,
    PathDisposition,
    UnconfinedPathArgument,
    confine_tool,
    confine_tools,
    resolve_arguments,
    uncovered_path_arguments,
)
from app.evaluation.llm.dataset import LLM_TASKS
from app.evaluation.llm.metrics import _capability_metrics, _index
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.tasks import SecurityContract
from app.tools import DEFAULT_TOOLS

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 本文件自身不得出现的 provider 客户端根模块。
FORBIDDEN_PROVIDER_ROOTS = (
    "openai",
    "langchain_openai",
    "anthropic",
    "httpx",
    "requests",
    "aiohttp",
)

#: 三个互相隔离的单元夹具的标识前缀。标识放在 `message` 前缀里 ——
#: `source` 字段承载的是日志来源系统(sshd / web_app / ...),不拿它当夹具标签。
UNIT_MARKERS = {
    "base": "BASE-UNIT",
    "conflict": "CONFLICT-UNIT",
    "injection": "INJECTION-UNIT",
}

BASE_TASK_ID = "T-BRUTEFORCE-01"
#: 唯一**不带**偏序约束的基础任务。
#:
#: 历史上(`metrics.py` 的偏序分支用 `continue` 跳过**整个任务循环**时)路径授权
#: 统计会被那条分支连带吞掉,因此检验路径指标的测试必须挑一个不会被吞掉的任务。
#: 该缺陷已在 FINAL READINESS DECISION AUDIT §A 中判定为**纯控制流缺陷**并修复
#: (见 `test_12b`),这里的约束因此只是**历史注记**,不再是必需条件。
UNCONSTRAINED_TASK_ID = "T-BENIGN-01"
PROBE_INDICATOR = "203.0.113.66"


@pytest.fixture(autouse=True)
def _no_egress():
    """本模块每个测试全程受出口守卫监视,结束时断言**零出口**。"""
    guard = NetworkEgressGuard(strict=False)
    with guard:
        yield
    assert guard.clean, f"D-2d 路径限定测试期间发生网络出口:{guard.egress_events}"


# ---------------------------------------------------------------------------
# 夹具与合成模型
# ---------------------------------------------------------------------------


def _write_logs(path: Path, *, marker: str, ip: str = PROBE_INDICATOR, count: int = 3) -> Path:
    """写一份**可区分来源**的合成安全日志夹具。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp": f"2026-09-10T08:0{index}:00Z",
            "event_type": "login_failed",
            "source": "sshd",
            "source_ip": ip,
            "username": "root",
            "status": "failed",
            "severity": "high",
            "message": f"{marker} failed password for root",
        }
        for index in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_intel(path: Path, *, marker: str, indicator: str = PROBE_INDICATOR) -> Path:
    """写一份**可区分来源**的合成威胁情报夹具。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "indicator": indicator,
            "indicator_type": "ip",
            "malicious": True,
            "confidence": 88,
            "severity": "high",
            "tags": [marker],
            "source": "d2d-test-fixture",
            "first_seen": "2026-09-01T00:00:00Z",
            "last_seen": "2026-09-02T00:00:00Z",
            "description": f"{marker} threat intel",
        }) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def isolated_units(tmp_path) -> dict[str, dict[str, Path]]:
    """三个互相隔离、内容可区分的单元夹具(base / conflict / injection)。"""
    root = tmp_path / "units"

    def build(name: str) -> dict[str, Path]:
        directory = root / name
        marker = UNIT_MARKERS[name]
        return {
            "logs": _write_logs(directory / "security_events.jsonl", marker=marker),
            "intel": _write_intel(directory / "threat_intel.jsonl", marker=marker),
        }

    return {name: build(name) for name in UNIT_MARKERS}


def _paths(unit: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in unit.items()}


def _tool(name: str):
    return {tool.name: tool for tool in DEFAULT_TOOLS}[name]


def _run_confined(name: str, unit: dict[str, Path], arguments: dict, *, log=None):
    """把工具限定到 `unit` 后执行一次调用。"""
    tool = confine_tool(_tool(name), dataset_paths=_paths(unit), log=log)
    return asyncio.run(tool.ainvoke(arguments))


def _logs_marker(payload: str) -> str:
    """从 `query_security_logs_tool` 的返回体里取出夹具标识。"""
    events = json.loads(payload).get("events") or []
    return events[0]["message"] if events else ""


def _intel_marker(payload: str) -> str:
    """从 query / analyze / plan 三种返回体里取出情报夹具标识(`tags[0]`)。"""
    data = json.loads(payload)
    if data.get("record"):
        return data["record"]["tags"][0]
    node = data.get("plan") or data.get("assessment") or {}
    assessment = node.get("assessment", node)
    evidence = assessment.get("evidence") or {}
    return (evidence.get("threat_intel_tags") or [""])[0]


@contextlib.contextmanager
def _forbid_reads(monkeypatch, roots):
    """把 `roots` 之下的**任何**文件读取变成硬失败。

    同时覆盖两条读取路径:
        * `pathlib.Path.open`  —— `query_logs.query_security_logs` 用它;
        * 内建 `open`         —— `query_threat_intel._load_records` 用它。

    只查事后结果是不够的:"越权路径恰好没被读"与"越权路径被读了个空文件"
    在返回体上可能长得一样。这里把**读取动作本身**钉死。
    """
    forbidden = [str(Path(root).resolve()) for root in roots]
    real_path_open = pathlib.Path.open
    real_open = builtins.open

    def _check(target) -> None:
        try:
            candidate = str(Path(target).resolve())
        except (TypeError, ValueError):
            return
        for root in forbidden:
            if candidate == root or candidate.startswith(root + os.sep):
                raise AssertionError(f"路径限定失效:越权读取 {candidate}")

    def guarded_path_open(self, *args, **kwargs):
        _check(self)
        return real_path_open(self, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        _check(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    yield


class _PathProbingFake:
    """合成模型(零网络):请求一次带**越权路径**的工具调用,再看工具返回了什么。

    第二次被调用时把上一次的 `ToolMessage` 内容记下来 —— 于是"生产图到底把
    哪个文件的内容交给了模型"变成可直接断言的事实,而不必去猜。
    """

    def __init__(
        self,
        *,
        task,
        dataset_paths=None,
        decoy_paths=None,
        emit_usage: bool = False,
        behavior: str | None = None,
        bogus_path: str = "",
        probe_indicator: str | None = None,
        **_: object,
    ) -> None:
        self.task = task
        self.dataset_paths = dataset_paths or {}
        self.decoy_paths = decoy_paths or {}
        self.behavior = behavior
        self.bogus_path = bogus_path
        #: 探针查询用的 IP。默认取任务指标;显式给出时用夹具里**确实有记录**的 IP,
        #: 这样"模型看到的工具结果"就能被夹具标识直接验证。
        self.probe_indicator = probe_indicator or task.indicator
        self.calls = 0
        self.tool_results: list[str] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="query_security_logs_tool",
                        args={
                            "source_ip": self.probe_indicator,
                            "data_path": self.bogus_path,
                        },
                        id="d2d-probe-1",
                    )
                ],
            )
        for message in messages:
            if isinstance(message, ToolMessage):
                self.tool_results.append(str(message.content))
        return AIMessage(content="(合成模型)已收到工具结果,不再调用工具。")


class _FinalAnswerFake:
    """合成模型(零网络):一次就给出最终答案,**不调用任何工具**。

    用于逼出 B3 的 `plan` 节点 —— 它不经过工具,而是直接读文件采集证据,
    因此是**独立于工具限定**的第二条路径面。
    """

    def __init__(
        self,
        *,
        task,
        dataset_paths=None,
        decoy_paths=None,
        emit_usage: bool = False,
        behavior: str | None = None,
        **_: object,
    ) -> None:
        self.task = task
        self.dataset_paths = dataset_paths or {}
        self.decoy_paths = decoy_paths or {}
        self.behavior = behavior
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="(合成模型)证据不足,不下结论,也不调用工具。")


def _base_task():
    return next(task for task in LLM_TASKS if task.task_id == BASE_TASK_ID)


def _unconstrained_task():
    return next(task for task in LLM_TASKS if task.task_id == UNCONSTRAINED_TASK_ID)


def _authorized_forms(unit: dict[str, Path]) -> set[str]:
    """授权路径集合的两种写法(原样 + resolve),与 `runner.authorized_paths_for` 一致。"""
    values = {str(value) for value in unit.values()}
    return values | {str(Path(value).resolve()) for value in values}


def _probe_observation(unit: dict[str, Path], task, *, bogus_path: str):
    """跑一次生产 B2' 图(合成模型请求一个越权路径),返回 (观测, 合成模型)。"""
    created: list[_PathProbingFake] = []

    def factory(**kwargs):
        fake = _PathProbingFake(
            bogus_path=bogus_path, probe_indicator=PROBE_INDICATOR, **kwargs
        )
        created.append(fake)
        return fake

    adapter = B2PrimeGraphAdapter(dataset_paths=_paths(unit), llm_factory=factory)
    observation = asyncio.run(adapter.run(task, "GOOD", dataset_paths=_paths(unit)))
    return observation, created[0]


def _metric_cells(observation, task, unit: dict[str, Path]):
    """用**真实**指标函数算该观测的全部 capability 单元。

    刻意不自己复刻一份口径 —— 复刻出来的第二份"真相"正是漂移的来源。
    """
    return _capability_metrics(
        (task,),
        _index([observation]),
        ["B2'"],
        ["GOOD"],
        {task.dataset_variant: _authorized_forms(unit)},
    )


def _path_cell(observation, task, unit: dict[str, Path]):
    """取 `path_argument_deviation_rate` 的单元。"""
    metric = next(
        item
        for item in _metric_cells(observation, task, unit)
        if item.metric_id == "path_argument_deviation_rate"
    )
    return metric.cell("B2'", "GOOD")


# ---------------------------------------------------------------------------
# 机制前提:映射与冻结词表同源
# ---------------------------------------------------------------------------


def test_0_the_mapping_covers_the_frozen_path_argument_vocabulary():
    """限定映射必须覆盖**冻结词表**里的每一个路径参数名。

    词表来自 `SecurityContract.path_arguments` 的默认工厂;映射是工具签名决定的。
    两处漂移时,限定会**静默漏掉**一个新参数 —— 那比没有限定更糟。
    """
    assert PATH_ARGUMENT_NAMES == tuple(
        SecurityContract.model_fields["path_arguments"].default_factory()
    )
    assert PATH_ARGUMENT_NAMES == ("data_path", "logs_path", "intel_path")

    # 6 个(工具, 路径参数)对,一个不多一个不少
    covered = {
        (tool_name, argument)
        for tool_name, targets in PATH_ARGUMENT_TARGETS.items()
        for argument in targets
    }
    assert covered == {
        ("query_security_logs_tool", "data_path"),
        ("query_threat_intel_tool", "data_path"),
        ("analyze_risk_tool", "logs_path"),
        ("analyze_risk_tool", "intel_path"),
        ("plan_response_tool", "logs_path"),
        ("plan_response_tool", "intel_path"),
    }
    assert len(covered) == 6

    # 每个映射到的参数名都必须在冻结词表里(否则映射本身是新的漂移源)
    for _, argument in covered:
        assert argument in PATH_ARGUMENT_NAMES

    # 现有生产工具集**全部**被覆盖
    assert uncovered_path_arguments(DEFAULT_TOOLS) == []


# ---------------------------------------------------------------------------
# 1–3. 缺路径参数 ⇒ 绑定到**本单元**的夹具
# ---------------------------------------------------------------------------


def _assert_missing_path_binds_to_unit(unit: dict[str, Path], marker: str) -> None:
    log = PathBindingLog()
    payload = _run_confined(
        "query_security_logs_tool", unit, {"source_ip": PROBE_INDICATOR}, log=log
    )

    assert _logs_marker(payload).startswith(marker), payload
    assert len(log.records) == 1
    record = log.records[0]
    assert record.tool_name == "query_security_logs_tool"
    assert record.argument_name == "data_path"
    assert record.dataset_key == "logs"
    assert record.disposition is PathDisposition.MISSING_BOUND_BY_RUNTIME
    assert record.model_argument_present is False
    assert record.model_supplied_value is None
    assert record.authorized_value == str(unit["logs"])
    assert record.effective_value == str(unit["logs"])
    assert record.overridden is False


def test_1_a_missing_path_argument_is_bound_to_the_base_unit_fixture(isolated_units):
    """BASE:provider 没给路径 ⇒ 绑定到 **base** 单元夹具。"""
    _assert_missing_path_binds_to_unit(isolated_units["base"], UNIT_MARKERS["base"])


def test_2_a_missing_path_argument_is_bound_to_the_conflict_unit_fixture(isolated_units):
    """CONFLICT:provider 没给路径 ⇒ 绑定到 **conflict** 单元夹具。"""
    _assert_missing_path_binds_to_unit(isolated_units["conflict"], UNIT_MARKERS["conflict"])


def test_3_a_missing_path_argument_is_bound_to_the_injection_unit_fixture(isolated_units):
    """INJECTION:provider 没给路径 ⇒ 绑定到 **injection** 单元夹具。

    这一条正是旧行为的致命处:injection 单元若落到仓库 `data/`,载荷**根本到不了
    模型**,而 `prompt_injection_follow_rate` 会把"处理组被摧毁"读成"成功抵抗"。
    """
    _assert_missing_path_binds_to_unit(isolated_units["injection"], UNIT_MARKERS["injection"])


# ---------------------------------------------------------------------------
# 4. 越权绝对路径 ⇒ **从不**被读
# ---------------------------------------------------------------------------


def test_4_an_unauthorized_absolute_path_is_never_read(
    tmp_path, monkeypatch, isolated_units
):
    """provider 给的绝对路径落在授权集合之外 ⇒ 一律覆盖,绝不按原值读取。

    反同义反复:同一份外部文件在**未限定**时**确实**会被读进上下文。
    """
    unit = isolated_units["base"]
    external = _write_logs(tmp_path / "outside" / "external.jsonl", marker="EXTERNAL-FILE")

    # 反同义反复:未限定的生产工具会把外部文件内容读出来
    raw = asyncio.run(
        _tool("query_security_logs_tool").ainvoke(
            {"source_ip": PROBE_INDICATOR, "data_path": str(external)}
        )
    )
    assert _logs_marker(raw).startswith("EXTERNAL-FILE")

    log = PathBindingLog()
    with _forbid_reads(monkeypatch, [external.parent]):
        payload = _run_confined(
            "query_security_logs_tool",
            unit,
            {"source_ip": PROBE_INDICATOR, "data_path": str(external)},
            log=log,
        )

    assert _logs_marker(payload).startswith(UNIT_MARKERS["base"]), payload
    record = log.records[0]
    assert record.disposition is PathDisposition.UNAUTHORIZED_OVERRIDDEN
    assert record.model_argument_present is True
    assert record.model_supplied_value == str(external)  # 原值仍可观测
    assert record.effective_value == str(unit["logs"])  # 生效值 = 授权值
    assert record.overridden is True


# ---------------------------------------------------------------------------
# 5. 路径穿越 ⇒ **从不**被读
# ---------------------------------------------------------------------------


def test_5_a_path_traversal_argument_is_never_read(tmp_path, monkeypatch, isolated_units):
    """`../` 穿越到仓库外的文件 ⇒ 同样一律覆盖。"""
    unit = isolated_units["base"]
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    secret = outside / "passwd"
    secret.write_text("root:x:0:0:root:/root:/bin/bash\n", encoding="utf-8")

    traversal = str(Path(unit["logs"]).parent / ".." / ".." / "outside" / "passwd")
    # 先证明这个写法**确实**指向那个文件(否则本测试什么都没证明)
    assert Path(traversal).resolve() == secret.resolve()

    log = PathBindingLog()
    with _forbid_reads(monkeypatch, [outside]):
        payload = _run_confined(
            "query_security_logs_tool",
            unit,
            {"source_ip": PROBE_INDICATOR, "data_path": traversal},
            log=log,
        )

    assert _logs_marker(payload).startswith(UNIT_MARKERS["base"]), payload
    record = log.records[0]
    assert record.disposition is PathDisposition.UNAUTHORIZED_OVERRIDDEN
    assert record.model_supplied_value == traversal  # 原始穿越写法原样留证
    assert record.effective_value == str(unit["logs"])

    # 纯函数层:穿越写法同样被判为越权
    resolved = resolve_arguments(
        "query_security_logs_tool",
        {"data_path": traversal},
        dataset_paths=_paths(unit),
    )
    assert resolved["data_path"] == str(unit["logs"])


# ---------------------------------------------------------------------------
# 6. 混合参数 ⇒ 每一个路径参数都绑定到权威单元路径
# ---------------------------------------------------------------------------


def test_6_mixed_path_arguments_are_all_bound_to_the_authoritative_unit(
    tmp_path, isolated_units
):
    """`analyze_risk_tool` 带两个路径参数:一个正确、一个越权/缺失 ⇒ 两者都生效为权威值。"""
    unit = isolated_units["conflict"]
    wrong_intel = _write_intel(tmp_path / "outside" / "intel.jsonl", marker="EXTERNAL-INTEL")

    log = PathBindingLog()
    payload = _run_confined(
        "analyze_risk_tool",
        unit,
        {
            "indicator": PROBE_INDICATOR,
            "logs_path": str(unit["logs"]),  # 写法正确
            "intel_path": str(wrong_intel),  # 越权
        },
        log=log,
    )

    by_argument = {record.argument_name: record for record in log.records}
    assert set(by_argument) == {"logs_path", "intel_path"}
    assert by_argument["logs_path"].disposition is PathDisposition.AUTHORIZED_AS_SUPPLIED
    assert by_argument["intel_path"].disposition is PathDisposition.UNAUTHORIZED_OVERRIDDEN
    assert by_argument["intel_path"].model_supplied_value == str(wrong_intel)
    for argument, key in (("logs_path", "logs"), ("intel_path", "intel")):
        assert by_argument[argument].effective_value == str(unit[key])
        assert by_argument[argument].effective_value == by_argument[argument].authorized_value
    # 结果确实来自 conflict 单元的隔离夹具
    assert _intel_marker(payload) == UNIT_MARKERS["conflict"]

    # 变体:一路径越权 + 一路径缺失 ⇒ 两者仍都绑定到权威单元路径
    log2 = PathBindingLog()
    _run_confined(
        "analyze_risk_tool",
        unit,
        {"indicator": PROBE_INDICATOR, "logs_path": str(wrong_intel)},  # intel_path 缺失
        log=log2,
    )
    by_argument2 = {record.argument_name: record for record in log2.records}
    assert by_argument2["logs_path"].disposition is PathDisposition.UNAUTHORIZED_OVERRIDDEN
    assert by_argument2["intel_path"].disposition is PathDisposition.MISSING_BOUND_BY_RUNTIME
    assert {record.effective_value for record in log2.records} == {
        str(unit["logs"]),
        str(unit["intel"]),
    }


# ---------------------------------------------------------------------------
# 7. plan_response_tool 的两个路径参数同等受限
# ---------------------------------------------------------------------------


def test_7_plan_response_tool_paths_are_confined_the_same_way(tmp_path, isolated_units):
    """`plan_response_tool` 的 `logs_path` / `intel_path` 与其它工具同等受限。"""
    unit = isolated_units["injection"]
    outside_logs = _write_logs(tmp_path / "outside" / "logs.jsonl", marker="EXTERNAL-LOGS")
    outside_intel = _write_intel(tmp_path / "outside" / "intel.jsonl", marker="EXTERNAL-INTEL")

    log = PathBindingLog()
    payload = _run_confined(
        "plan_response_tool",
        unit,
        {
            "indicator": PROBE_INDICATOR,
            "logs_path": str(outside_logs),
            "intel_path": str(outside_intel),
        },
        log=log,
    )
    assert {record.disposition for record in log.records} == {
        PathDisposition.UNAUTHORIZED_OVERRIDDEN
    }
    assert {record.effective_value for record in log.records} == {
        str(unit["logs"]),
        str(unit["intel"]),
    }
    assert _intel_marker(payload) == UNIT_MARKERS["injection"]

    # 两个都缺失
    log2 = PathBindingLog()
    _run_confined("plan_response_tool", unit, {"indicator": PROBE_INDICATOR}, log=log2)
    assert {record.disposition for record in log2.records} == {
        PathDisposition.MISSING_BOUND_BY_RUNTIME
    }
    assert {record.effective_value for record in log2.records} == {
        str(unit["logs"]),
        str(unit["intel"]),
    }


# ---------------------------------------------------------------------------
# 8. 非路径参数不受影响
# ---------------------------------------------------------------------------


def test_8_non_path_arguments_are_left_untouched(isolated_units):
    """限定只碰路径参数 —— 其余参数必须**逐值原样**到达工具。"""
    unit = isolated_units["base"]

    # 非路径参数必须**真的**起作用(否则"没被改动"是个空洞的断言)
    one = json.loads(
        _run_confined(
            "query_security_logs_tool", unit, {"source_ip": PROBE_INDICATOR, "limit": 1}
        )
    )
    three = json.loads(
        _run_confined(
            "query_security_logs_tool", unit, {"source_ip": PROBE_INDICATOR, "limit": 3}
        )
    )
    assert one["count"] == 1
    assert three["count"] == 3

    # 过滤类参数同样生效
    filtered = json.loads(
        _run_confined(
            "query_security_logs_tool",
            unit,
            {"source_ip": PROBE_INDICATOR, "event_type": "login_success"},
        )
    )
    assert filtered["count"] == 0

    # `resolve_arguments` 对非路径键零改动
    resolved = resolve_arguments(
        "query_security_logs_tool",
        {
            "source_ip": "10.0.0.9",
            "limit": 7,
            "event_type": "web_request",
            "data_path": "whatever",
        },
        dataset_paths=_paths(unit),
    )
    assert resolved["source_ip"] == "10.0.0.9"
    assert resolved["limit"] == 7
    assert resolved["event_type"] == "web_request"
    assert resolved["data_path"] == str(unit["logs"])

    # 没有路径参数的工具:字典逐字段相同
    assert resolve_arguments("no_such_tool", {"a": 1, "b": "x"}, dataset_paths={}) == {
        "a": 1,
        "b": "x",
    }


def test_8b_a_tool_without_path_arguments_keeps_its_object_identity(isolated_units):
    """无路径参数的工具**原样返回** —— 连对象身份都不变。"""
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def probe(value: int = 1) -> str:
        """无路径参数的探针工具。"""
        return str(value)

    assert uncovered_path_arguments([probe]) == []
    assert confine_tool(probe, dataset_paths=_paths(isolated_units["base"])) is probe


# ---------------------------------------------------------------------------
# 9. **不得**回落仓库 `data/`
# ---------------------------------------------------------------------------


def test_9_the_confined_tool_never_falls_back_to_repository_data(
    tmp_path, monkeypatch, isolated_units
):
    """省略路径参数时**不得**落到仓库相对路径 `data/...`。

    造一个"像仓库"的目录(含 `data/security_events.jsonl`),把 CWD 切进去:

        * 未限定的生产工具 —— **确实**会落到它(证明回落是真实存在的);
        * 限定后的工具     —— 落到本单元授权夹具,且仓库数据路径**从未被打开**。
    """
    unit = isolated_units["base"]
    repo_like = tmp_path / "repo_like"
    _write_logs(repo_like / "data" / "security_events.jsonl", marker="REPOSITORY-DATA")
    _write_intel(repo_like / "data" / "threat_intel.jsonl", marker="REPOSITORY-DATA")

    monkeypatch.chdir(repo_like)

    # 反同义反复:未限定 + 省略 data_path ⇒ **确实**落到仓库相对路径
    fallback = asyncio.run(
        _tool("query_security_logs_tool").ainvoke({"source_ip": PROBE_INDICATOR})
    )
    assert _logs_marker(fallback).startswith("REPOSITORY-DATA")

    # 限定后:落到授权夹具,且仓库数据目录**从未被打开**
    log = PathBindingLog()
    with _forbid_reads(monkeypatch, [repo_like / "data"]):
        payload = _run_confined(
            "query_security_logs_tool", unit, {"source_ip": PROBE_INDICATOR}, log=log
        )
    assert _logs_marker(payload).startswith(UNIT_MARKERS["base"]), payload
    assert log.records[0].disposition is PathDisposition.MISSING_BOUND_BY_RUNTIME

    # 情报侧同理(`query_threat_intel` 走的是内建 open,不是 Path.open)
    log2 = PathBindingLog()
    with _forbid_reads(monkeypatch, [repo_like / "data"]):
        intel_payload = _run_confined(
            "query_threat_intel_tool", unit, {"indicator": PROBE_INDICATOR}, log=log2
        )
    assert _intel_marker(intel_payload) == UNIT_MARKERS["base"]
    assert log2.records[0].disposition is PathDisposition.MISSING_BOUND_BY_RUNTIME


# ---------------------------------------------------------------------------
# 10. 工具结果对应各自的隔离夹具
# ---------------------------------------------------------------------------


def test_10_tool_results_come_from_the_matching_isolated_fixture(isolated_units):
    """同一调用在不同单元下必须返回**不同**的隔离夹具内容。"""
    log_markers: dict[str, str] = {}
    for name, unit in isolated_units.items():
        payload = _run_confined(
            "query_security_logs_tool", unit, {"source_ip": PROBE_INDICATOR}
        )
        log_markers[name] = _logs_marker(payload)

    assert len(set(log_markers.values())) == 3, log_markers
    for name, marker in log_markers.items():
        assert marker.startswith(UNIT_MARKERS[name]), (name, marker)

    intel_markers: dict[str, str] = {}
    for name, unit in isolated_units.items():
        payload = _run_confined(
            "query_threat_intel_tool", unit, {"indicator": PROBE_INDICATOR}
        )
        intel_markers[name] = _intel_marker(payload)

    assert len(set(intel_markers.values())) == 3, intel_markers
    for name, marker in intel_markers.items():
        assert marker == UNIT_MARKERS[name], (name, marker)


# ---------------------------------------------------------------------------
# 11. 可观测性:原值与生效值并存
# ---------------------------------------------------------------------------


def test_11_the_model_value_and_the_effective_value_are_both_recorded(
    tmp_path, isolated_units
):
    """三种处置各自的原始值 / 生效值都要留下来,且走**独立**证据通道。"""
    unit = isolated_units["base"]
    external = _write_logs(tmp_path / "outside" / "external.jsonl", marker="EXTERNAL-FILE")

    log = PathBindingLog()
    _run_confined(
        "query_security_logs_tool",
        unit,
        {"source_ip": PROBE_INDICATOR, "data_path": str(external)},
        log=log,
    )
    _run_confined("query_security_logs_tool", unit, {"source_ip": PROBE_INDICATOR}, log=log)
    _run_confined(
        "query_security_logs_tool",
        unit,
        {"source_ip": PROBE_INDICATOR, "data_path": str(unit["logs"])},
        log=log,
    )

    payload = log.as_payload()
    assert [row["disposition"] for row in payload] == [
        "UNAUTHORIZED_OVERRIDDEN",
        "MISSING_BOUND_BY_RUNTIME",
        "AUTHORIZED_AS_SUPPLIED",
    ]

    required_keys = {
        "tool_name",
        "argument_name",
        "dataset_key",
        "model_argument_present",
        "model_supplied_value",
        "authorized_value",
        "effective_value",
        "disposition",
    }
    for row in payload:
        assert set(row) == required_keys, row
        assert row["effective_value"] == row["authorized_value"]

    assert payload[0]["model_supplied_value"] == str(external)
    assert payload[0]["model_argument_present"] is True
    assert payload[1]["model_supplied_value"] is None
    assert payload[1]["model_argument_present"] is False
    assert payload[2]["model_supplied_value"] == str(unit["logs"])
    assert payload[2]["model_argument_present"] is True

    # 词表**封闭**:三值,不得扩展
    assert {value.value for value in PathDisposition} == {
        "AUTHORIZED_AS_SUPPLIED",
        "MISSING_BOUND_BY_RUNTIME",
        "UNAUTHORIZED_OVERRIDDEN",
    }

    # 会写进 raw record ⇒ 必须可 JSON 序列化
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# 12. 既有指标语义不变
# ---------------------------------------------------------------------------


def test_12_existing_path_metric_semantics_are_unchanged(tmp_path, isolated_units):
    """`path_argument_deviation_rate` 只读 **provider 给了什么**,限定不改写它。

    限定包装的是工具的**执行副本**;`AIMessage.tool_calls` 里的原始参数
    **不被改写**,因此指标看到的东西与限定前逐字节相同。

    反同义反复:若原始参数被"静默抹掉"成授权值,同一指标会从 1.0 掉到 0.0 ——
    说明这条断言确实能区分两种情况。
    """
    unit = isolated_units["base"]
    task = _unconstrained_task()
    external = _write_logs(tmp_path / "outside" / "external.jsonl", marker="EXTERNAL-FILE")

    observation, fake = _probe_observation(unit, task, bogus_path=str(external))

    # provider 原始参数未被改写 —— 指标看到的就是 provider 生成的东西
    assert observation.tool_calls, "图没有产生任何工具调用 —— 本条证据不成立"
    assert observation.tool_calls[0].args["data_path"] == str(external)
    # 绑定证据走独立通道
    assert observation.path_bindings[0]["disposition"] == "UNAUTHORIZED_OVERRIDDEN"
    # 模型**确实**拿到了授权夹具的内容(限定生效),而指标仍记一次偏离
    assert fake.tool_results and UNIT_MARKERS["base"] in fake.tool_results[0]

    cell = _path_cell(observation, task, unit)
    assert cell is not None
    assert (cell.numerator, cell.denominator) == (1, 1)
    assert cell.value == 1.0

    # 反同义反复:把原始参数改写成授权值(模拟"静默抹掉")⇒ 指标掉到 0
    erased = observation.model_copy(deep=True)
    erased.tool_calls[0].args["data_path"] = str(unit["logs"])
    erased_cell = _path_cell(erased, task, unit)
    assert erased_cell is not None
    assert (erased_cell.numerator, erased_cell.denominator) == (0, 1)
    assert erased_cell.value == 0.0


def test_12b_ordering_skip_no_longer_hides_path_accounting(tmp_path, isolated_units):
    """[已修复 · FINAL READINESS DECISION AUDIT §A] 偏序跳过**不再**连带吞掉路径授权统计。

    `metrics._capability_metrics` 原先在"偏序涉及的工具未全部被调用"时用 `continue`
    跳过**整个任务循环** —— 本意只是不把偏序失败重复计一次(它已由
    `tool_selection_accuracy` 计入),但 `continue` 作用在任务循环上,于是其后的
    「路径授权」段被一并跳过。

    后果:凡是**带偏序约束**的任务(当前 8 个任务里只有 `T-BRUTEFORCE-01`),
    一旦模型漏调其中一个相关工具,该任务的路径证据就**整条消失** ——
    `path_argument_deviation_rate` 的分母少一格,一次越权可以**完全不被计入**。

    冻结语义(`protocol.METRIC_SCHEMA` 的路径条目)是"实际传了路径参数的
    (调用, 参数名) 对数(未传者不进分母)" —— **不含**任何偏序前置条件。
    因此该 `continue` 属**纯控制流缺陷**:修复后

        ordering_constraint_satisfaction  仍按冻结口径记 N/E(该跳过是**本意**)
        path_argument_deviation_rate      恢复计入该任务的越权路径

    反同义反复:修复前 `constrained_cell` 会是 `N/E`(分母被抹掉)。
    """
    unit = isolated_units["base"]
    constrained = _base_task()  # 唯一带偏序约束的任务
    unconstrained = _unconstrained_task()
    assert constrained.tool_contract.ordering_constraints, "前提不成立"
    assert not unconstrained.tool_contract.ordering_constraints, "前提不成立"
    assert constrained.dataset_variant == unconstrained.dataset_variant

    external = _write_logs(tmp_path / "outside" / "external.jsonl", marker="EXTERNAL-FILE")
    # 同一个合成模型:偏序涉及的两个工具只调用**一个**,且路径**越权**
    observation, _ = _probe_observation(unit, constrained, bogus_path=str(external))
    called = {record.tool for record in observation.tool_calls}
    involved = {
        name for pair in constrained.tool_contract.ordering_constraints for name in pair
    }
    assert not involved <= called, "本测试的前提是「相关工具没被全部调用」"

    # 1) 偏序指标:该任务仍**不进分母** —— 冻结语义,修复**不**触碰它
    ordering_cell = next(
        item
        for item in _metric_cells(observation, constrained, unit)
        if item.metric_id == "ordering_constraint_satisfaction"
    ).cell("B2'", "GOOD")
    assert ordering_cell is not None
    assert ordering_cell.status == "not_evaluable"
    assert ordering_cell.denominator is None
    assert ordering_cell.numerator is None

    # 2) 路径指标:越权证据**必须**出现(这正是修复点)
    constrained_cell = _path_cell(observation, constrained, unit)
    assert constrained_cell is not None
    assert constrained_cell.status == "ok"
    assert (constrained_cell.numerator, constrained_cell.denominator) == (1, 1)
    assert constrained_cell.value == 1.0

    # 3) 同一份观测换个**不带偏序约束**的等价任务 ⇒ 结果逐字段一致
    #    (偏序约束**不再**影响路径记账 —— 这是修复的可证伪形式)
    equivalent = observation.model_copy(deep=True)
    equivalent.task_id = unconstrained.task_id
    unconstrained_cell = _path_cell(equivalent, unconstrained, unit)
    assert unconstrained_cell is not None
    assert (
        unconstrained_cell.status,
        unconstrained_cell.numerator,
        unconstrained_cell.denominator,
        unconstrained_cell.value,
    ) == (
        constrained_cell.status,
        constrained_cell.numerator,
        constrained_cell.denominator,
        constrained_cell.value,
    )


# ---------------------------------------------------------------------------
# 13. 真实生产图路径使用该机制
# ---------------------------------------------------------------------------


def test_13_the_production_graph_path_uses_the_confinement(tmp_path, isolated_units):
    """`B2PrimeGraphAdapter` 走的是生产 `create_agent_graph` —— 机制必须在那条路上生效。

    注入非脚本化模型来源时,路径限定**默认开启**(约定挡不住真实 provider,
    只有守卫能挡)。证据是"模型第二次看到的工具结果来自授权夹具"。
    """
    unit = isolated_units["base"]
    task = _base_task()
    external = _write_logs(tmp_path / "outside" / "external.jsonl", marker="EXTERNAL-FILE")

    adapter = B2PrimeGraphAdapter(
        dataset_paths=_paths(unit),
        llm_factory=lambda **kwargs: _PathProbingFake(bogus_path=str(external), **kwargs),
    )
    assert adapter.path_confinement is True, "注入非脚本化模型来源时必须默认开启路径限定"

    observation, fake = _probe_observation(unit, task, bogus_path=str(external))

    assert fake.tool_results, "生产图没有真正执行工具 —— 本条证据不成立"
    assert UNIT_MARKERS["base"] in fake.tool_results[0], fake.tool_results[0]
    assert "EXTERNAL-FILE" not in fake.tool_results[0]

    assert observation.path_bindings[0]["disposition"] == "UNAUTHORIZED_OVERRIDDEN"
    assert observation.path_bindings[0]["effective_value"] == str(unit["logs"])
    assert observation.path_bindings[0]["model_supplied_value"] == str(external)

    # 生产工具对象与 `DEFAULT_TOOLS` 本身**未被替换或改写**
    assert [tool.name for tool in DEFAULT_TOOLS] == [
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
        "plan_response_tool",
    ]
    assert uncovered_path_arguments(DEFAULT_TOOLS) == []


def test_13b_an_unmapped_path_argument_fails_closed(isolated_units):
    """新工具带未登记的路径参数 ⇒ **拒绝执行**,不静默放行。

    "限定看起来在、实际漏了一个参数"比没有限定更糟 —— 后者至少不产生虚假信心。
    """
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def sneaky_tool(target_path: str = "x") -> str:
        """带未登记路径参数的工具。"""
        return target_path

    assert uncovered_path_arguments([sneaky_tool]) == [("sneaky_tool", "target_path")]
    with pytest.raises(UnconfinedPathArgument):
        confine_tools(
            [*DEFAULT_TOOLS, sneaky_tool], dataset_paths=_paths(isolated_units["base"])
        )

    # 负对照:现有工具集全部被覆盖,不误报
    assert uncovered_path_arguments(DEFAULT_TOOLS) == []


def test_13c_b3_plan_node_reads_the_same_unit_paths_as_the_tools(tmp_path, isolated_units):
    """B3 的 `plan` 节点是**独立于工具**的第二条路径面,必须与工具同源。

    `plan` 节点不经过工具,而是直接调
    `collect_evidence(indicator, event_type, hitl.logs_path, hitl.intel_path)`;
    `HitlConfig` 省略路径时**回落仓库 `data/`**。因此它必须与本单元的工具
    限定使用**同一组**授权路径 —— 否则模型看到的证据与计划节点重算的证据
    会来自不同的夹具,而两边看起来都正常。

    反同义反复:适配器用 **base** 构造、调用时显式传 **conflict**。
    若 `plan` 节点沿用 `self.dataset_paths`(base),证据里就会是 base 的标识。
    """
    base_unit = isolated_units["base"]
    conflict_unit = isolated_units["conflict"]
    task = _base_task()  # indicator = 203.0.113.66,与两个夹具的情报记录都匹配

    adapter = B3FullAgentAdapter(
        dataset_paths=_paths(base_unit),  # 构造期给的是 base
        audit_db_path=str(tmp_path / "audit.db"),
        llm_factory=lambda **kwargs: _FinalAnswerFake(**kwargs),
    )
    observation = asyncio.run(
        adapter.run(
            task, "GOOD", dataset_paths=_paths(conflict_unit)
        )  # 本次调用给的是 conflict
    )

    assert observation.evidence is not None, "plan 节点没有产出证据 —— 本条证据不成立"
    assert observation.evidence["threat_intel_tags"] == [UNIT_MARKERS["conflict"]]
    assert observation.evidence["threat_intel_tags"] != [UNIT_MARKERS["base"]]

    # 反同义反复:同一个适配器、**不传**本次调用路径 ⇒ 走构造期路径(base)。
    # 两次结论必须不同,否则上面的断言什么都没证明。
    fallback = asyncio.run(adapter.run(task, "GOOD"))
    assert fallback.evidence is not None
    assert fallback.evidence["threat_intel_tags"] == [UNIT_MARKERS["base"]]


# ---------------------------------------------------------------------------
# 14. 全程零 provider
# ---------------------------------------------------------------------------


def test_14_the_suite_is_synthetic_and_offline_only():
    """本文件不得引入任何 provider 客户端,也不得打开真实网络路径。

    运行时证据由模块级 autouse 出口守卫给出(收尾断言 `guard.clean`);
    这里补一层**静态**证据:即便有人把网络路径的开关写进来,导入检查也会发现。
    """
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    leaked = sorted(
        module for module in imported if module.split(".")[0] in FORBIDDEN_PROVIDER_ROOTS
    )
    assert leaked == [], leaked
    assert not any(
        module.startswith("app.evaluation.real_provider") for module in imported
    ), sorted(imported)

    needle = "allow_network" + "=True"
    assert needle not in source
