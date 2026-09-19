"""Phase 9.2-A 观测适配器 —— 把三个被测对象跑成可比较的 `Observation`。

本模块是**唯一**允许 import 生产代码的地方(被测系统本来就要被 import)。
`oracles.py` 反过来只依赖标准库 —— 方向是单向的:观测依赖实现,判定不依赖。

三个基线(9.2-A 的诚实命名)
---------------------------
    B1 RuleOnlyAdapter      确定性内核直接串联:采集 → 分析 → 规划 → 策略判定。
                            **没有策略门**:策略结论只是被算出来,没有任何东西
                            消费它(没有中断、没有审计、没有暂停)。
    B2 GraphNoGateAdapter   编译 `hitl=None` 的 ReAct 图:LLM → (无 tool_calls) → END。
                            **没有计划、没有策略、没有审计** —— 只有一段叙事文本。
    B3 FullAgentAdapter     编译带 HITL 的完整图:plan → policy_gate →
                            (human_approval 中断 | END),审计写入 SQLite。

关于比较的诚实声明(必须写进报告,不能只写在代码里)
--------------------------------------------------
1. **B1 与 B3 共享同一个确定性内核**(同一套 collect_evidence / analyze_risk /
   plan_response / evaluate_policy)。因此 B1 与 B3 的差异**不是**"Agent 优于
   非 Agent",而是"有没有策略门"。任何把它解读为"Agent 更聪明"的说法都是
   过度解读,9.2-A 不提供这种证据。
2. **B2 与 B3 的差异**是"有门 / 无门",可以作为**安全性消融实验**:
   无门基线在"必须人工审批"的用例上必然不满足要求(它压根没有审批概念)。
3. **没有任何适配器执行真实处置动作。** Phase 8 只做审批,不接防火墙/EDR。
   因此消融实验度量的是**门控与留痕**,不是"实际危害"。这一点必须明说。
4. **B3 的 LLM 是 FakeLLMClient**,不发起任何真实网络调用;它的回复是一条常量
   文本。9.2-A 不评测叙事质量。
"""
from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.security.audit import compute_plan_digest
from app.security.policy import evaluate_policy
from app.security.store import SqliteAuditStore
from app.tools.response_planner import plan_response
from app.tools.risk_analyzer import analyze_risk, collect_evidence

logger = structlog.get_logger(__name__)

RunStatus = Literal["completed", "pending_approval", "not_gated"]

#: 9.2-A 使用的固定叙事回复。刻意是常量:叙事质量在本阶段不参与评测,
#: 但必须**确定** —— 否则确定性重复断言会被无关文本抖动打挂。
FAKE_NARRATIVE = "收到,已按结构化结果给出说明。"


class Observation(BaseModel):
    """一个用例在一个基线下的全部可观测量。

    字段为 `None` 表示**该基线不产生这项观测**(而不是"值为空")。这个区分是
    评测诚实性的关键:性质判定遇到 `None` 会记 not_evaluable,而不是记失败。
    """

    case_id: str
    adapter_id: str
    case_indicator: str

    # ---- 证据层 ----
    evidence: dict | None = Field(default=None, description="采集到的 RiskEvidence")

    # ---- 评估层 ----
    risk_level: str | None = None
    score: int | None = None
    confidence: int | None = None

    # ---- 计划层 ----
    plan_actions: list[str] | None = None
    plan_risk_level: str | None = None
    plan_indicator: str | None = None
    plan_digest: str | None = None

    # ---- 策略层 ----
    policy_outcome: str | None = None
    policy_requires_approval: bool | None = None
    gated_actions: list[str] | None = None

    # ---- 生命周期层 ----
    run_status: RunStatus = "not_gated"
    audit_events: list[str] | None = None
    audit_plan_digests: dict[str, str] | None = None

    # ---- 叙事层(9.2-A 不评测) ----
    answer: str | None = None

    error: str | None = None


class BaseAdapter:
    """适配器公共契约。"""

    adapter_id: str = "?"
    produces_plan: bool = False
    produces_gate: bool = False
    produces_audit: bool = False

    def __init__(
        self,
        *,
        logs_path: str,
        intel_path: str,
        audit_db_path: str | None = None,
        llm: Any | None = None,
    ) -> None:
        self.logs_path = logs_path
        self.intel_path = intel_path
        self.audit_db_path = audit_db_path
        self._llm = llm if llm is not None else FakeLLMClient(reply=FAKE_NARRATIVE)

    async def run(self, case: Any) -> Observation:  # pragma: no cover - 抽象
        raise NotImplementedError

    # ---- 共享的观测抽取 ----

    def _observation(self, case: Any, **kwargs: Any) -> Observation:
        return Observation(
            case_id=case.case_id,
            adapter_id=self.adapter_id,
            case_indicator=case.indicator,
            **kwargs,
        )

    @staticmethod
    def _from_assessment(assessment: Any) -> dict:
        return {
            "risk_level": assessment.risk_level,
            "score": assessment.score,
            "confidence": assessment.confidence,
        }

    @staticmethod
    def _from_plan(plan: Any) -> dict:
        return {
            "plan_actions": [action.action_type for action in plan.actions],
            "plan_risk_level": plan.risk_level,
            "plan_indicator": plan.indicator,
            "plan_digest": compute_plan_digest(plan),
        }

    @staticmethod
    def _from_policy(decision: Any) -> dict:
        return {
            "policy_outcome": decision.outcome,
            "policy_requires_approval": decision.requires_approval,
            "gated_actions": list(decision.gated_actions),
        }


class RuleOnlyAdapter(BaseAdapter):
    """B1:确定性内核直连,无策略门。

    这是"规则流水线"基线。它算得出策略结论,但**没有任何东西消费它** ——
    没有中断、没有审计、没有暂停。把它与 B3 对比,度量的是"门控与留痕"
    的有无,不是"智能程度"。
    """

    adapter_id = "B1"
    produces_plan = True
    produces_gate = False
    produces_audit = False

    async def run(self, case: Any) -> Observation:
        try:
            evidence = collect_evidence(
                case.indicator,
                logs_path=self.logs_path,
                intel_path=self.intel_path,
            )
            assessment = analyze_risk(evidence)
            plan = plan_response(assessment)
            decision = evaluate_policy(plan)
        except Exception as exc:  # pragma: no cover - 失败路径在 runner 里被记为错误
            logger.error("evaluation_adapter_failed", adapter=self.adapter_id,
                         case_id=case.case_id, error_type=type(exc).__name__)
            return self._observation(case, run_status="not_gated", error=type(exc).__name__)

        return self._observation(
            case,
            evidence=evidence.model_dump(mode="json"),
            **self._from_assessment(assessment),
            **self._from_plan(plan),
            **self._from_policy(decision),
            run_status="not_gated",
            audit_events=None,
            audit_plan_digests=None,
        )


class GraphNoGateAdapter(BaseAdapter):
    """B2:ReAct 图,无 HITL 安全层。

    这张图里根本没有 plan / policy_gate / human_approval 三个节点,所以它
    既产不出计划,也产不出策略结论,更产不出审计。它与 B3 的对比是**安全性
    消融**:无门基线在"必须人工审批"的用例上必然不满足。
    """

    adapter_id = "B2"
    produces_plan = False
    produces_gate = False
    produces_audit = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph = create_agent_graph(self._llm)

    async def run(self, case: Any) -> Observation:
        try:
            final_state = await self._graph.ainvoke({
                "messages": [
                    SystemMessage(content=SECURITY_ANALYST_SYSTEM_PROMPT),
                    HumanMessage(content=f"请分析指标 {case.indicator}"),
                ],
                "iteration_count": 0,
            })
        except Exception as exc:  # pragma: no cover
            logger.error("evaluation_adapter_failed", adapter=self.adapter_id,
                         case_id=case.case_id, error_type=type(exc).__name__)
            return self._observation(case, run_status="not_gated", error=type(exc).__name__)

        return self._observation(
            case,
            answer=_extract_answer(final_state.get("messages", [])),
            run_status="not_gated",
        )


class FullAgentAdapter(BaseAdapter):
    """B3:完整 HITL 图(plan → policy_gate → human_approval?)。

    本适配器**不代替人做决定**:它在图暂停后即停止,不去 resume。
    因此 `pending_approval` 是它的正常终态之一,而不是需要"跑完"的状态。
    这样做的原因:resume 需要一个人工决定,而 9.2-A 评测的是"系统是否要求
    了人工审批",不是"审批之后会怎样"。
    """

    adapter_id = "B3"
    produces_plan = True
    produces_gate = True
    produces_audit = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.audit_db_path is None:
            raise ValueError(
                "B3 需要 audit_db_path —— 审计数据库必须由调用方显式给出路径,"
                "绝不能落到仓库 data/ 下"
            )
        self._store = SqliteAuditStore(self.audit_db_path)
        # 每个适配器实例持有自己的 checkpointer:跨评测运行复用会让
        # thread_id 冲突(已暂停的线程被覆盖),这是 LangGraph 的已知语义。
        self._graph = create_agent_graph(
            self._llm,
            hitl=HitlConfig(
                checkpointer=InMemorySaver(),
                audit_store=self._store,
                logs_path=self.logs_path,
                intel_path=self.intel_path,
            ),
        )

    async def run(self, case: Any) -> Observation:
        thread_id = f"{self.adapter_id}:{case.case_id}"
        config = {"configurable": {"thread_id": thread_id}}
        try:
            await self._graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SECURITY_ANALYST_SYSTEM_PROMPT),
                        HumanMessage(content=f"请分析指标 {case.indicator}"),
                    ],
                    "iteration_count": 0,
                    "indicator": case.indicator,
                    "event_type": None,
                },
                config,
            )
            snapshot = await self._graph.aget_state(config)
        except Exception as exc:  # pragma: no cover
            logger.error("evaluation_adapter_failed", adapter=self.adapter_id,
                         case_id=case.case_id, error_type=type(exc).__name__)
            return self._observation(case, run_status="completed", error=type(exc).__name__)

        values = snapshot.values or {}
        run_status: RunStatus = "pending_approval" if snapshot.next else "completed"

        plan = values.get("plan")
        decision = values.get("policy_decision")

        audit_records = self._store.list_audit(thread_id=thread_id)
        audit_events = [record.event for record in audit_records]
        audit_plan_digests = {
            record.event: record.plan_digest
            for record in audit_records
            if record.plan_digest
        }

        payload: dict[str, Any] = {
            "answer": _extract_answer(values.get("messages", [])),
            "run_status": run_status,
            "audit_events": audit_events,
            "audit_plan_digests": audit_plan_digests,
        }
        if plan is not None:
            payload["evidence"] = plan.assessment.evidence.model_dump(mode="json")
            payload.update(self._from_assessment(plan.assessment))
            payload.update(self._from_plan(plan))
        if decision is not None:
            payload.update(self._from_policy(decision))

        return self._observation(case, **payload)


def _extract_answer(messages: list) -> str:
    """取最后一条 AIMessage 的文本(与 SecurityAgent 的外部契约一致)。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


ADAPTERS: dict[str, type[BaseAdapter]] = {
    RuleOnlyAdapter.adapter_id: RuleOnlyAdapter,
    GraphNoGateAdapter.adapter_id: GraphNoGateAdapter,
    FullAgentAdapter.adapter_id: FullAgentAdapter,
}
