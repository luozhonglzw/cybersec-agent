"""手写 LangGraph 版 ReAct 控制流(Phase 4 Step 4.2)+ HITL 安全层(Phase 8.3)。

Phase 3 的 SecurityAgent.chat() 用 while 循环表达了同样的控制流:
    LLM → 有 tool_calls? → 执行工具 → 回到 LLM / 结束

本模块用 LangGraph 的 State / Node / Edge 把它声明式地表达出来:

    START → agent → should_continue ─┬─ 无 tool_calls / 达到上限 → END
                                     └─ 有 tool_calls → tools → agent

Phase 8.3 在 ReAct 循环**之后**接上 HITL 安全层(仅在传入 HitlConfig 时):

    agent ─(无 tool_calls)─→ plan → policy_gate ─┬─ allow ──────────→ END
                                                 └─ require_approval
                                                       ↓
                                                 human_approval(interrupt 暂停)
                                                       ↓ 人审批后 resume
                                                      END

设计要点(勿改):

1. **开关是 HitlConfig,不是布尔值**。checkpointer 与 audit_store 都是必填
   字段 —— 把"审批必须有审计"从运行时检查提升为类型层保证。hitl=None 时
   编译出的图与 Phase 8.3 之前**逐字节一致**,既有测试零影响。

2. **plan 节点必须确定性产出**。它直接调用规则引擎
   (collect_evidence → analyze_risk → plan_response),**不解析**
   plan_response_tool 的 ToolMessage。原因与 policy.py 的论证同源:
   工具是"LLM 可选调用"的 —— 若计划来自 LLM 的工具选择,LLM 只要不调用
   规划工具,策略门就无计划可判,门形同虚设。安全控制必须结构性。

   因此 HITL 路径的工具集刻意去掉 plan_response_tool(HITL_TOOLS),
   保证全链路只有一个计划来源:state["plan"]。

3. **interrupt() 之前禁止任何副作用**(框架语义,实测确认):
   resume 时 LangGraph 会**从被中断节点的函数体开头重放**,
   `interrupt()` 之前的代码执行两次、之后的代码只执行一次。
   所以 human_approval 节点里 `interrupt()` 之前一行副作用都没有;
   approval.requested 写在**上游的 policy_gate**(已完成节点不重放),
   approval.decided 写在 `interrupt()` **之后**。
   这条由 tests/test_core/test_hitl_graph.py 的 AST 护栏锁定。

4. **审批状态不落库,只追加事件**。policy_gate 写 policy.evaluated +
   approval.requested;human_approval 写 approval.decided。
   "某 thread 是否仍待审批"由 store.pending_action_rows 派生,不存在可被
   改写的状态列(见 app/security/store.py 的 append-only 前提)。

5. **Phase 8.3 不写 incident**。incident 生命周期(创建时机、与审批单的
   先后)留给 8.4 的 triage/API 决定,避免提前绑定。

6. **interrupt_id 在 approval.requested 时为 NULL**。它由框架在
   interrupt() 内部分配,节点内拿不到;approval.decided 才填得上真实值
   (由 8.4 的 resume() 从 aget_state().tasks[*].interrupts[*].id 恢复后
   注入 resume 载荷)。这正是 audit_logs.interrupt_id 可空的原因。

checkpoint 的现实约束(必须文档化):
    本阶段只能用 InMemorySaver —— langgraph-checkpoint-sqlite **未安装**,
    引入即新增依赖。因此 checkpoint 是**进程内运行时状态,重启丢失**;
    持久事实在 audit_logs / incidents / action_requests 三张表里。
    两者不互相替代:checkpoint 决定"能否继续跑",审计决定"发生过什么"。
    解析路径:Phase 10 换 saver 实现,图与节点零改动。
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from app.core.llm import LLMClient
from app.schemas.approval import ApprovalDecision, ApprovalRequest
from app.schemas.policy import PolicyDecision
from app.schemas.response import ResponsePlan
from app.security.audit import build_audit_record
from app.security.policy import evaluate_policy
from app.security.store import SqliteAuditStore
from app.tools import DEFAULT_TOOLS
from app.tools.query_logs import DEFAULT_DATA_PATH as LOGS_DATA_PATH
from app.tools.query_threat_intel import DEFAULT_DATA_PATH as INTEL_DATA_PATH
from app.tools.response_planner import plan_response
from app.tools.risk_analyzer import analyze_risk, collect_evidence

logger = structlog.get_logger(__name__)

# HITL 路径不注册规划工具:计划由 plan 节点确定性产出(见模块 docstring 第 2 条)。
# 唯一的真相源仍是 app.tools.DEFAULT_TOOLS —— 这里是它的**子集**,
# 由 tests/test_core/test_hitl_graph.py 的 test_hitl_tools_is_default_minus_planner 守着。
PLANNER_TOOL_NAME = "plan_response_tool"
HITL_TOOLS: list[BaseTool] = [
    tool for tool in DEFAULT_TOOLS if tool.name != PLANNER_TOOL_NAME
]


@dataclass(frozen=True)
class HitlConfig:
    """HITL 安全层的依赖包。

    两个必填字段都是"没有它就不能开 HITL"的东西:
        checkpointer: 没有它 interrupt() 是单向门 —— 能暂停但无法恢复
                      (Command(resume=) 直接 RuntimeError,实测确认);
        audit_store:  没有它审批就没有留痕,而审批的全部意义就是留痕。

    把它们放在同一个必填数据类里,是为了让"只给了一个"这个中间态
    在**构造时**就不成立,而不是等到运行到 require_approval 分支才报错。

    logs_path / intel_path 省略时用工具层默认路径(data/*.jsonl);
    测试通过它们注入 tmp_path,保证 hermetic。
    """

    checkpointer: BaseCheckpointSaver
    audit_store: SqliteAuditStore
    logs_path: Path | None = None
    intel_path: Path | None = None


class AgentState(TypedDict, total=False):
    """Graph 的全部状态,不引入业务之外的额外字段。

    messages: 消息历史,add_messages reducer 负责按 id 去重追加
    iteration_count: agent 节点的调用次数,用于业务层 max_iterations 判断

    ---- Phase 8.3(HITL 安全层)----
    indicator: 判定对象(IP/域名/Hash),由调用方给出。缺失时 plan 节点跳过,
              整条 HITL 链路自动短路 —— 自由对话(chat)因此不需要另建一张图。
    plan: 规则引擎产出的**权威**处置计划。刻意与 messages 里可能存在的
          ToolMessage 计划分开:那个是给 LLM 叙事用的上下文,这个才是
          policy_gate 消费的对象。策略门永不解析 messages。
    policy_decision: 策略判定结果(allow / require_approval)
    approval_request: 需审批时的待审请求,也是 interrupt() 的载荷
    approval_decision: 恢复后人工给出的决定

    thread_id 刻意**不在这里** —— 它是运行时配置(configurable),
    放进 state 会制造第二个真相源,节点内用 _thread_id() 读取。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    indicator: str
    plan: ResponsePlan | None
    policy_decision: PolicyDecision | None
    approval_request: ApprovalRequest | None
    approval_decision: ApprovalDecision | None


def _thread_id() -> str:
    """从 LangGraph 运行时配置读取 thread_id。

    HITL 的审计与审批记录都必须带 thread_id,但它属于运行时配置而非业务
    状态(见 AgentState docstring)。有 checkpointer 时框架保证 thread_id
    存在,否则 ainvoke 会先抛 ValueError —— 这里只是把缺失情况说清楚。
    """
    configurable = get_config().get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if not thread_id:
        raise ValueError("HITL 需要 configurable.thread_id(用于审计与恢复)")
    return thread_id


def _select_tools(
    tools: list[BaseTool] | None, hitl: HitlConfig | None
) -> list[BaseTool]:
    """决定本次图构建使用哪套工具。

    显式传入的 tools 永远优先;省略时按是否启用 HITL 选择默认集。
    """
    if tools is not None:
        return list(tools)
    if hitl is not None:
        return list(HITL_TOOLS)
    return list(DEFAULT_TOOLS)


def _make_plan_node(hitl: HitlConfig):
    """产出权威处置计划的节点(确定性,不经过 LLM)。"""

    async def plan_node(state: AgentState) -> dict:
        indicator = state.get("indicator")
        if not indicator:
            # 自由对话路径:没有判定对象 → 不产出计划。
            # 下游 policy_gate 会因 plan is None 一并短路,整条 HITL 链路跳过。
            return {}

        evidence = collect_evidence(
            indicator,
            None,
            hitl.logs_path if hitl.logs_path is not None else LOGS_DATA_PATH,
            hitl.intel_path if hitl.intel_path is not None else INTEL_DATA_PATH,
        )
        plan = plan_response(analyze_risk(evidence))

        # 本节点是已完成节点,resume 时不会被重放 → 恰好写一次
        hitl.audit_store.append_audit(build_audit_record(
            "plan.created",
            thread_id=_thread_id(),
            plan=plan,
            detail={
                "indicator": plan.indicator,
                "risk_level": plan.risk_level,
                "score": plan.assessment.score,
                "confidence": plan.assessment.confidence,
                "actions": [a.action_type for a in plan.actions],
            },
        ))
        logger.info(
            "graph_plan_created",
            risk_level=plan.risk_level,
            action_count=len(plan.actions),
        )
        return {"plan": plan}

    return plan_node


def _make_policy_gate_node(hitl: HitlConfig):
    """策略门:强制调用 evaluate_policy,并把审批单物化进 state。

    approval.requested 写在这里而不是 human_approval,有两个原因:
    1. 必须**在暂停前**落库,否则"谁在等审批"无从查询;
    2. human_approval 里 interrupt() 之前的副作用会被 resume 重放(写两次),
       而本节点是已完成节点,不会重放。
    """

    async def policy_gate_node(state: AgentState) -> dict:
        plan = state.get("plan")
        if plan is None:
            return {}

        # 单输入契约:只消费 ResponsePlan,不接受配置 / LLM 输出
        decision = evaluate_policy(plan)
        thread_id = _thread_id()

        hitl.audit_store.append_audit(build_audit_record(
            "policy.evaluated",
            thread_id=thread_id,
            outcome=decision.outcome,
            reason="; ".join(decision.reasons),
            plan=plan,
            detail={
                "outcome": decision.outcome,
                "gated_actions": decision.gated_actions,
                "policy_version": decision.policy_version,
                "policy_reasons": decision.reasons,
            },
        ))

        if not decision.requires_approval:
            logger.info("graph_policy_allowed", policy_version=decision.policy_version)
            return {"policy_decision": decision}

        gated = set(decision.gated_actions)
        request = ApprovalRequest(
            thread_id=thread_id,
            indicator=plan.indicator,
            risk_level=plan.risk_level,
            score=plan.assessment.score,
            summary=plan.summary,
            actions=[a for a in plan.actions if a.action_type in gated],
            policy_reasons=decision.reasons,
        )

        # interrupt_id 此刻**尚不存在**(框架在 interrupt() 内部分配),
        # 因此这里不传 —— 该行必然为 NULL,由 approval.decided 补上真实值。
        hitl.audit_store.append_audit(build_audit_record(
            "approval.requested",
            thread_id=thread_id,
            plan=plan,
            detail={
                "indicator": plan.indicator,
                "risk_level": plan.risk_level,
                "gated_actions": decision.gated_actions,
                "policy_version": decision.policy_version,
            },
        ))
        logger.info(
            "graph_approval_requested",
            gated_actions=decision.gated_actions,
            indicator=plan.indicator,
        )
        return {"policy_decision": decision, "approval_request": request}

    return policy_gate_node


def _make_human_approval_node(hitl: HitlConfig):
    """人工审批暂停点。

    ⚠️ 本节点 `interrupt()` **之前不得有任何副作用**。
    LangGraph 在 resume 时会从函数体开头重放本节点,`interrupt()` 之前的
    代码会执行两次(实测确认)—— 一个放在那里的 append_audit 会在恢复时
    写入重复记录,并因 append-only 主键冲突抛 IntegrityError 把恢复流程打挂。
    所有写入必须放在 interrupt() **之后**(只执行一次)。
    """

    async def human_approval_node(state: AgentState) -> dict:
        request = state["approval_request"]

        # 载荷必须 JSON 可序列化(ApprovalRequest 含 datetime),故走 model_dump(mode="json")
        raw = interrupt(request.model_dump(mode="json"))

        # ---- 以下代码只在 resume 那一遍执行,恰好一次 ----
        decision = ApprovalDecision.model_validate(raw)
        hitl.audit_store.append_audit(build_audit_record(
            "approval.decided",
            actor=decision.operator,
            thread_id=_thread_id(),
            interrupt_id=decision.interrupt_id,
            outcome=decision.status,
            reason=decision.reason,
            plan=state.get("plan"),
            detail={
                "status": decision.status,
                "operator": decision.operator,
                "indicator": request.indicator,
                "gated_actions": [a.action_type for a in request.actions],
            },
        ))
        logger.info(
            "graph_approval_decided",
            status=decision.status,
            operator=decision.operator,
        )
        return {"approval_decision": decision}

    return human_approval_node


def _route_after_gate(state: AgentState) -> str:
    """条件边:有待审请求 → 进入暂停点;否则直接结束。"""
    return "human_approval" if state.get("approval_request") is not None else END


def create_agent_graph(
    llm_client: LLMClient,
    tools: list[BaseTool] | None = None,
    max_iterations: int = 5,
    hitl: HitlConfig | None = None,
):
    """构建并编译 ReAct graph(可选叠加 HITL 安全层)。

    参数与 SecurityAgent 保持一致风格:LLM 客户端、可选工具列表、迭代上限。
    省略 tools 时使用 app.tools.DEFAULT_TOOLS —— 与 SecurityAgent 同一个真相源,
    不再各自维护一份(Phase 7 前这里只兜底 1 个工具,与 agent 的 3 个不一致)。

    hitl:
        省略(None)→ 编译 Phase 8.3 之前那张两节点图,行为逐字节不变;
        传入 HitlConfig → 在 ReAct 循环之后追加 plan / policy_gate /
        human_approval 三个节点,并用 hitl.checkpointer 编译。
    """
    tools = _select_tools(tools, hitl)
    tool_map = {tool.name: tool for tool in tools}
    bound_model = llm_client.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        """调用 LLM,返回新的 AIMessage 与递增后的迭代计数。"""
        response = await bound_model.ainvoke(state["messages"])
        count = state.get("iteration_count", 0) + 1
        logger.info(
            "graph_agent_node",
            iteration_count=count,
            tool_call_count=len(response.tool_calls) if response.tool_calls else 0,
        )
        return {"messages": [response], "iteration_count": count}

    async def tools_node(state: AgentState) -> dict:
        """执行最后一条 AIMessage 的所有 tool_calls,返回 ToolMessage 列表。

        安全契约:任何异常只记录 error_type 与 tool_call_id,
        ToolMessage 内容是可重试的通用 JSON,不含 traceback / 路径 / 参数值。
        """
        last_ai: AIMessage = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        for tc in last_ai.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool is None:
                tool_messages.append(ToolMessage(
                    content=json.dumps({
                        "error": "未知工具",
                        "tool_name": tc["name"],
                        "suggest_retry": True,
                    }),
                    tool_call_id=tc["id"],
                ))
                continue

            try:
                result = await tool.ainvoke(tc["args"])
                tool_messages.append(ToolMessage(
                    content=result if isinstance(result, str) else str(result),
                    tool_call_id=tc["id"],
                ))
            except Exception as exc:
                logger.error(
                    "graph_tool_failed",
                    tool_name=tc["name"],
                    error_type=type(exc).__name__,
                    tool_call_id=tc["id"],
                )
                tool_messages.append(ToolMessage(
                    content=json.dumps({
                        "error": "工具执行失败",
                        "type": type(exc).__name__,
                        "details": "请调整查询条件后重试",
                    }),
                    tool_call_id=tc["id"],
                ))

        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> str:
        """条件边:按 tool_calls 与业务层迭代上限决定去向。

        返回值保持 END 常量不变 —— HITL 模式下由 path_map 把 END 重映射到
        plan 节点(实测可行),因此本函数在两种模式下**完全一致**。
        """
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        if state.get("iteration_count", 0) >= max_iterations:
            logger.info("graph_max_iterations_reached", max_iterations=max_iterations)
            return END
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")

    if hitl is None:
        # Phase 8.3 之前的图:END 就是 END
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        return graph.compile()

    # ---- HITL 安全层 ----
    graph.add_node("plan", _make_plan_node(hitl))
    graph.add_node("policy_gate", _make_policy_gate_node(hitl))
    graph.add_node("human_approval", _make_human_approval_node(hitl))

    # 同一个 should_continue,只是把 END 改接到 plan
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: "plan"})
    graph.add_edge("plan", "policy_gate")
    graph.add_conditional_edges(
        "policy_gate",
        _route_after_gate,
        {"human_approval": "human_approval", END: END},
    )
    graph.add_edge("human_approval", END)
    return graph.compile(checkpointer=hitl.checkpointer)
