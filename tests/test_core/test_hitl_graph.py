"""Phase 8.3 HITL 安全层测试(graph 层)。

覆盖三组关键契约:

1. **结构性安全**(比功能更重要)
   - HITL 路径不注册 plan_response_tool:计划只能由规则引擎产出,
     LLM 无法通过"不调用规划工具"让策略门失效;
   - policy_gate 永不解析 messages:它只消费 state["plan"];
   - human_approval 节点内 interrupt() 之前**没有任何副作用** ——
     LangGraph 在 resume 时会重放该节点,前置副作用会执行两次,
     这是框架语义陷阱,用 AST 护栏锁死。

2. **interrupt / resume 生命周期**
   暂停 → aget_state().next == ("human_approval",) → Command(resume=) 恢复;
   以及 allow 分支完全不进入暂停点。

3. **审计恰好一次**
   暂停+恢复全过程,plan.created / policy.evaluated / approval.requested /
   approval.decided **各恰好一条**。这是上面第 1 组规则的**行为级验证**:
   护栏是静态的,这里是动态的,两者互为保险。

全部 hermetic:数据文件与 audit.db 都由 tmp_path 现场生成,
不读仓库 data/,也不在仓库里创建 audit.db。
"""
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

import app.core.graph as graph_module
from app.core.graph import (
    HITL_TOOLS,
    PLANNER_TOOL_NAME,
    AgentState,
    HitlConfig,
    create_agent_graph,
)
from app.core.llm import FakeLLMClient
from app.security.audit import compute_plan_digest
from app.security.store import SqliteAuditStore
from app.tools import DEFAULT_TOOLS

# 30 次失败登录 + 恶意情报 → critical → 含 block_ip / reset_credentials → 需审批
BRUTE_FORCE_IP = "203.0.113.66"
BRUTE_FORCE_LOGINS = 30

# 10 次失败登录,无情报命中 → low → 仅 monitor → 放行
LOW_RISK_IP = "198.51.100.7"
LOW_RISK_LOGINS = 10

THREAD_ID = "thread-hitl-1"
_REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.fixture
def data_paths(tmp_path: Path) -> tuple[Path, Path]:
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)
    return logs, intel


@pytest.fixture
def audit_store(tmp_path: Path) -> SqliteAuditStore:
    return SqliteAuditStore(tmp_path / "audit.db")


@pytest.fixture
def hitl(data_paths, audit_store) -> HitlConfig:
    logs, intel = data_paths
    return HitlConfig(
        checkpointer=InMemorySaver(),
        audit_store=audit_store,
        logs_path=logs,
        intel_path=intel,
    )


def _build(hitl: HitlConfig | None, reply: str = "分析完成"):
    """返回 (graph, llm)。llm 用于断言实际绑定了哪些工具。"""
    llm = FakeLLMClient(reply)
    return create_agent_graph(llm, hitl=hitl), llm


def _initial(indicator: str | None = None) -> dict:
    state: dict = {"messages": [HumanMessage(content="分析该 IP")], "iteration_count": 0}
    if indicator is not None:
        state["indicator"] = indicator
    return state


def _config(thread_id: str = THREAD_ID) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _event_counts(store: SqliteAuditStore, thread_id: str) -> Counter:
    return Counter(r.event for r in store.list_audit(thread_id=thread_id))


# =====================================================================
# A. 结构与向后兼容
# =====================================================================

def test_hitl_none_keeps_original_graph_shape():
    """hitl=None → 图结构与 Phase 8.3 之前**精确一致**,零漂移。

    这里用精确相等而不是 `<=`:多出任何节点都必须是有意的改动。
    """
    graph, _ = _build(None)
    nodes = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert nodes == {"agent", "tools"}


def test_hitl_enabled_adds_three_nodes(hitl):
    graph, _ = _build(hitl)
    nodes = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert nodes == {"agent", "tools", "plan", "policy_gate", "human_approval"}


def test_hitl_tools_is_default_minus_planner():
    """D2:计划由节点确定性产出,LLM 不再持有规划工具。

    绊线:DEFAULT_TOOLS 一旦改名/新增,这条会失败 —— 届时请有意更新,
    而不是让 HITL 悄悄多出一个计划来源。
    """
    assert [t.name for t in HITL_TOOLS] == [
        t.name for t in DEFAULT_TOOLS if t.name != PLANNER_TOOL_NAME
    ]
    assert PLANNER_TOOL_NAME not in {t.name for t in HITL_TOOLS}


def test_hitl_graph_does_not_bind_planner_tool(hitl):
    """行为级验证:实际绑给 LLM 的工具里没有规划工具。"""
    _, llm = _build(hitl)
    assert [t.name for t in llm.bound_tools] == [t.name for t in HITL_TOOLS]
    assert PLANNER_TOOL_NAME not in {t.name for t in llm.bound_tools}


def test_plain_graph_still_binds_all_default_tools():
    """hitl=None 时工具集不变(向后兼容)。"""
    _, llm = _build(None)
    assert [t.name for t in llm.bound_tools] == [t.name for t in DEFAULT_TOOLS]


def test_explicit_tools_always_win(hitl):
    """显式传入 tools 时优先于 HITL 默认集(不静默改写调用方的意图)。"""
    llm = FakeLLMClient("分析完成")
    explicit = [DEFAULT_TOOLS[0]]
    create_agent_graph(llm, tools=explicit, hitl=hitl)
    assert [t.name for t in llm.bound_tools] == [t.name for t in explicit]


def test_hitl_config_requires_both_checkpointer_and_audit_store(audit_store):
    """D3:两个字段都必填 —— "只给一个"在构造时就不成立。

    没有 checkpointer,interrupt() 是单向门(能暂停不能恢复);
    没有 audit_store,审批就没有留痕。两者缺一不可,所以放在同一个必填数据类里。
    """
    import inspect

    params = inspect.signature(HitlConfig).parameters
    assert params["checkpointer"].default is inspect.Parameter.empty
    assert params["audit_store"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        HitlConfig(checkpointer=InMemorySaver())
    with pytest.raises(TypeError):
        HitlConfig(audit_store=audit_store)


async def test_no_indicator_completes_without_pausing(hitl):
    """自由对话路径:没有 indicator → plan/policy_gate 短路 → 不暂停。"""
    graph, _ = _build(hitl)
    result = await graph.ainvoke(_initial(None), _config())
    assert "__interrupt__" not in result
    assert result.get("plan") is None
    assert result.get("approval_request") is None


async def test_no_indicator_writes_no_audit(hitl, audit_store):
    """短路路径不该产生任何审计噪声。"""
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(None), _config())
    assert audit_store.list_audit() == []


# =====================================================================
# B. interrupt / resume 生命周期
# =====================================================================

async def test_requires_approval_pauses_at_human_approval(hitl):
    graph, _ = _build(hitl)
    result = await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())

    assert "__interrupt__" in result
    assert len(result["__interrupt__"]) == 1

    state = await graph.aget_state(_config())
    assert state.next == ("human_approval",)


async def test_interrupt_payload_is_the_approval_request(hitl):
    """interrupt 载荷 == ApprovalRequest 的 JSON 形式(审批人看到的依据)。"""
    graph, _ = _build(hitl)
    result = await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    payload = result["__interrupt__"][0].value

    assert payload["thread_id"] == THREAD_ID
    assert payload["indicator"] == BRUTE_FORCE_IP
    assert payload["risk_level"] == "critical"
    assert set(payload["actions"][0]) == {
        "action_type", "priority", "target", "rationale",
        "requires_approval", "reversible",
    }
    assert {a["action_type"] for a in payload["actions"]} >= {
        "block_ip", "reset_credentials",
    }
    assert payload["policy_reasons"]


async def test_paused_state_exposes_plan_and_decision(hitl):
    """暂停态可完整内省:plan / policy_decision / approval_request 都在 state 里。"""
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    values = (await graph.aget_state(_config())).values

    assert values["plan"].indicator == BRUTE_FORCE_IP
    assert values["policy_decision"].outcome == "require_approval"
    assert values["approval_request"].thread_id == THREAD_ID
    assert values.get("approval_decision") is None


async def test_resume_approved_completes(hitl):
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    result = await graph.ainvoke(
        Command(resume={"status": "approved", "operator": "analyst-1"}),
        _config(),
    )

    assert "__interrupt__" not in result
    assert result["approval_decision"].status == "approved"
    assert result["approval_decision"].operator == "analyst-1"
    assert (await graph.aget_state(_config())).next == ()


async def test_resume_denied_completes(hitl):
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    result = await graph.ainvoke(
        Command(resume={"status": "denied", "operator": "analyst-2",
                        "reason": "证据不足"}),
        _config(),
    )

    assert result["approval_decision"].status == "denied"
    assert result["approval_decision"].reason == "证据不足"


async def test_resume_payload_is_validated(hitl):
    """resume 载荷仍受 Pydantic 约束:非法 status 必须被拒。"""
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    with pytest.raises(ValidationError):
        await graph.ainvoke(
            Command(resume={"status": "maybe", "operator": "x"}), _config()
        )


async def test_allow_path_never_pauses(hitl):
    """低风险 → allow → 不进入暂停点,也不产生审批单。"""
    graph, _ = _build(hitl)
    result = await graph.ainvoke(_initial(LOW_RISK_IP), _config())

    assert "__interrupt__" not in result
    assert result["policy_decision"].outcome == "allow"
    assert result.get("approval_request") is None
    assert result.get("approval_decision") is None
    assert (await graph.aget_state(_config())).next == ()


async def test_thread_id_isolation(hitl, audit_store):
    """两个线程交错:决定不串线,审计各归各的。"""
    graph, _ = _build(hitl)
    cfg_a, cfg_b = _config("thread-a"), _config("thread-b")

    await graph.ainvoke(_initial(BRUTE_FORCE_IP), cfg_a)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), cfg_b)
    await graph.ainvoke(
        Command(resume={"status": "approved", "operator": "a1"}), cfg_a
    )

    # b 仍待审批,不受 a 的决定影响
    assert (await graph.aget_state(cfg_b)).next == ("human_approval",)
    assert _event_counts(audit_store, "thread-b")["approval.decided"] == 0
    assert _event_counts(audit_store, "thread-a")["approval.decided"] == 1


# =====================================================================
# C. 审计:恰好一次 + 内容正确
# =====================================================================

async def test_each_audit_event_written_exactly_once(hitl, audit_store):
    """⚠️ 本文件最重要的一条。

    暂停 + 恢复全过程走完后,四个事件各恰好一条。
    如果哪天有人把审计写入挪到 interrupt() 之前,human_approval 会被 resume
    重放 → approval.decided 写两次 → 这条立刻失败(且 append-only 主键会先抛错)。
    """
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    await graph.ainvoke(
        Command(resume={"status": "approved", "operator": "a1"}), _config()
    )

    assert _event_counts(audit_store, THREAD_ID) == Counter({
        "plan.created": 1,
        "policy.evaluated": 1,
        "approval.requested": 1,
        "approval.decided": 1,
    })


async def test_allow_path_writes_only_two_events(hitl, audit_store):
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(LOW_RISK_IP), _config())

    assert _event_counts(audit_store, THREAD_ID) == Counter({
        "plan.created": 1,
        "policy.evaluated": 1,
    })


async def test_at_pause_requested_exists_but_not_decided(hitl, audit_store):
    """暂停时刻的库状态:待审已落库(否则"谁在等审批"查不到),决定尚未产生。"""
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())

    counts = _event_counts(audit_store, THREAD_ID)
    assert counts["approval.requested"] == 1
    assert counts["approval.decided"] == 0

    requested = audit_store.list_audit(thread_id=THREAD_ID, event="approval.requested")[0]
    # interrupt_id 由框架在 interrupt() 内部分配 → 写审批单时必然未知
    assert requested.interrupt_id is None
    assert requested.outcome is None


async def test_decided_event_carries_interrupt_id(hitl, audit_store):
    """恢复后 approval.decided 必须带上服务端恢复出来的 interrupt_id。"""
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())

    observed = (await graph.aget_state(_config())).tasks[0].interrupts[0].id
    await graph.ainvoke(
        Command(resume={"status": "approved", "operator": "analyst-1",
                        "interrupt_id": observed}),
        _config(),
    )

    decided = audit_store.list_audit(thread_id=THREAD_ID, event="approval.decided")[0]
    assert decided.interrupt_id == observed
    assert decided.actor == "analyst-1"
    assert decided.outcome == "approved"

    # 请求与决定指向同一个计划
    requested = audit_store.list_audit(thread_id=THREAD_ID, event="approval.requested")[0]
    assert decided.plan_digest == requested.plan_digest


async def test_plan_digest_matches_final_state_plan(hitl, audit_store):
    """审计里的摘要必须等于最终 state 里那个计划对象的摘要(同一性)。"""
    graph, _ = _build(hitl)
    result = await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())

    created = audit_store.list_audit(thread_id=THREAD_ID, event="plan.created")[0]
    assert created.plan_digest == compute_plan_digest(result["plan"])


async def test_policy_audit_records_outcome_and_gated_actions(hitl, audit_store):
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())

    evaluated = audit_store.list_audit(thread_id=THREAD_ID, event="policy.evaluated")[0]
    assert evaluated.outcome == "require_approval"
    assert set(evaluated.detail["gated_actions"]) >= {"block_ip", "reset_credentials"}
    assert evaluated.detail["policy_version"]


async def test_pending_derivation_tracks_real_writes(hitl, audit_store):
    """把 8.2 的派生查询接到 8.3 的真实写入上。

    状态不落库:暂停时 pending_action_rows 非空(尚无 approval.decided),
    恢复后自动为空(事件已追加)。
    """
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())

    # 审批单行由 8.4 的 triage 写入(8.3 不写 incident / action_requests),
    # 这里手工补一条,验证派生逻辑确实读的是真实审计流
    request = (await graph.aget_state(_config())).values["approval_request"]
    audit_store.record_action_request(request)

    assert len(audit_store.pending_action_rows(thread_id=THREAD_ID)) == len(request.actions)

    await graph.ainvoke(
        Command(resume={"status": "approved", "operator": "a1"}), _config()
    )
    assert audit_store.pending_action_rows(thread_id=THREAD_ID) == []


async def test_audit_records_are_append_only_under_hitl(hitl, audit_store):
    """HITL 全流程结束后,审计流仍满足 append-only(行数 == 事件数,无改写)。"""
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    await graph.ainvoke(
        Command(resume={"status": "approved", "operator": "a1"}), _config()
    )

    records = audit_store.list_audit()
    assert len(records) == 4
    assert len({r.id for r in records}) == 4


async def test_incident_is_not_written_in_phase_8_3(hitl, audit_store):
    """8.3 范围:不创建 incident —— 生命周期留给 8.4 的 triage/API 决定。"""
    graph, _ = _build(hitl)
    await graph.ainvoke(_initial(BRUTE_FORCE_IP), _config())
    await graph.ainvoke(
        Command(resume={"status": "approved", "operator": "a1"}), _config()
    )
    assert audit_store.get_incident("anything") is None
    assert audit_store.list_action_rows() == []


# =====================================================================
# D. 结构性护栏(AST)
# =====================================================================

def _graph_tree() -> ast.Module:
    return ast.parse(Path(graph_module.__file__).read_text(encoding="utf-8"))


def _find_func(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"未找到函数 {name}")


def _contains_call(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        for n in ast.walk(node)
    )


def _interrupt_index(func) -> int:
    index = next(
        (i for i, stmt in enumerate(func.body) if _contains_call(stmt, "interrupt")),
        None,
    )
    assert index is not None, f"{func.name} 里必须有 interrupt() 调用"
    return index


def test_human_approval_has_no_side_effect_before_interrupt():
    """⚠️ 框架语义护栏:interrupt() 之前不得有任何副作用。

    LangGraph 在 resume 时从被中断节点的函数体开头**重放**,
    interrupt() 之前的代码会执行两次。放在那里的 append_audit 会在恢复时
    写重复记录,并因 append-only 主键冲突抛 IntegrityError 打挂恢复流程。
    """
    func = _find_func(_graph_tree(), "human_approval_node")
    for stmt in func.body[:_interrupt_index(func)]:
        dumped = ast.dump(stmt)
        for forbidden in ("audit_store", "append_audit", "record_incident",
                          "record_action_request"):
            assert forbidden not in dumped, (
                f"interrupt() 之前出现了副作用 {forbidden}:会被 resume 重放"
            )


def test_human_approval_writes_decided_after_interrupt():
    """反向断言:approval.decided 的写入必须落在 interrupt() 之后。"""
    func = _find_func(_graph_tree(), "human_approval_node")
    after = func.body[_interrupt_index(func):]
    assert any("approval.decided" in ast.dump(stmt) for stmt in after), (
        "approval.decided 必须在 interrupt() 之后写入"
    )


def test_policy_gate_never_reads_messages():
    """策略门只消费 state["plan"],永不解析 messages 里的 ToolMessage。

    否则计划又变成"LLM 决定要不要给"的东西,门会退化为建议性。
    """
    assert "messages" not in ast.dump(_find_func(_graph_tree(), "policy_gate_node"))


def test_plan_node_uses_rules_not_tool_output():
    """plan 节点直接调规则引擎,不解析工具输出。"""
    dumped = ast.dump(_find_func(_graph_tree(), "plan_node"))
    assert "collect_evidence" in dumped
    assert "analyze_risk" in dumped
    assert "plan_response" in dumped
    assert "messages" not in dumped
    assert "ToolMessage" not in dumped


def test_policy_gate_writes_requested_before_pause():
    """approval.requested 必须在暂停前落库(否则待审查询无据可依)。"""
    assert "approval.requested" in ast.dump(
        _find_func(_graph_tree(), "policy_gate_node")
    )


def test_graph_module_has_no_new_dependency():
    """锁死零新依赖:不得引入 langgraph-checkpoint-sqlite / aiosqlite。"""
    imports: set[str] = set()
    for node in ast.walk(_graph_tree()):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    for forbidden in ("langgraph.checkpoint.sqlite", "aiosqlite", "sqlalchemy"):
        assert forbidden not in imports


def test_graph_module_does_not_touch_private_llm_model():
    """护栏:API Key 注入边界仍只在 LLMClient 一处,图模块不得访问 _model。"""
    source = Path(graph_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\._model\b", source)


# =====================================================================
# E. 隔离与契约
# =====================================================================

def test_no_repo_audit_db_created():
    """所有用例都走 tmp_path;仓库里不得出现 audit.db。"""
    assert not (_REPO_ROOT / "data" / "audit.db").exists()


def test_state_annotation_set_is_exact():
    """与 test_graph.py 的契约断言互为呼应(此处再锁一次,防止只改一处)。"""
    assert set(AgentState.__annotations__.keys()) == {
        "messages", "iteration_count",
        "indicator", "event_type", "plan", "policy_decision",
        "approval_request", "approval_decision",
    }
