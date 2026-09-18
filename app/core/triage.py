"""Triage Service —— Phase 8.4。

把"一次安全判定"编排成:执行 HITL graph → 聚合结果 → 沉淀审计。

依赖方向(勿改):
    本模块在 **core 层**,只向上给 API 层提供**原始类型**
    (indicator / status / operator / reason)。刻意**不 import app.api** ——
    否则 core 会反向依赖传输层 DTO,协议一改就穿透进业务层。
    把 TriageResult 转成 HTTP 响应模型是 API 层的职责。

为什么 triage() / resume() 不直接调用 graph 的内部节点:
    节点编排(plan → policy_gate → human_approval)是 create_agent_graph
    的契约。绕过节点直接调 evaluate_policy 会造出**第二条执行路径** ——
    plan.created / policy.evaluated / approval.requested 三个审计事件
    就会漏写,而"审批留痕"正是 HITL 的全部意义。
    本模块只通过图的公开入口(ainvoke / aget_state)与它交互。

thread_id 由服务端生成(D3),因为框架**完全不校验** thread_id ——
四个后果全部实测复现过:
    1. aget_state(未知 thread) → 空快照(values={}, next=(), created_at=None),
       **不报错**;
    2. Command(resume=) 在未知 thread 上 → **不报错**,直接从 START 新起
       一轮(响应看起来正常,实际什么都没恢复);
    3. 在已完成的 thread 上重复 resume → **静默返回陈旧决定**(HTTP 200),
       且不写任何新审计;
    4. 复用已有 thread_id → **覆盖暂停中的 state**(indicator 被换掉、
       messages 被追加、agent 重跑)= 劫持向量。
因此:thread_id 一律 uuid4 服务端生成,绝不接受客户端输入;
resume 之前必须先过 _validate_resumable() 校验门(第 2、3 条正靠它拦截)。

interrupt_id 由服务端从 checkpoint 恢复(D7),绝不采信客户端传的值:
    实测确认 result["__interrupt__"][0].id 与
    aget_state().tasks[*].interrupts[*].id 是同一个值。

审计写入顺序(必须保持):
    graph 节点写 plan.created / policy.evaluated / approval.requested
    (此时 incident_id 为 NULL —— incident 还不存在)
      → 服务写 record_incident(incident_id)
      → 服务写 record_action_request(request, incident_id=incident_id)
    于是 incident → action_requests 有关联。

已知限制(必须文档化,不得掩盖):
    1. **audit_logs.incident_id 为 NULL**。incident 在图跑完之后才创建
       (D4 决定 incident_id 不进 AgentState),所以图节点写审计时无从得知它。
       后果:list_audit(incident_id=...) 查不到本次判定的审计流,
       必须改用 list_audit(thread_id=...)。Phase 8.5 候选(需新增审计事件
       或在 State 里放 incident_id,两者都超出 8.4 范围)。
    2. **checkpoint 是进程内 InMemorySaver**(langgraph-checkpoint-sqlite
       未安装,引入即新增依赖)。服务重启后,暂停中的 thread 无法恢复 ——
       resume() 会命中 CheckpointLostError(409),而不是假装成功。
       持久事实在 incidents / action_requests / audit_logs 三张表里。
"""
import uuid
from dataclasses import dataclass

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.types import Command

from app.core.agent import MAX_ITERATIONS_REPLY, SECURITY_ANALYST_SYSTEM_PROMPT
from app.schemas.approval import ApprovalStatus, TriageOutcome, utc_now
from app.schemas.incident import Incident
from app.security.store import SqliteAuditStore

logger = structlog.get_logger(__name__)

# 暂停点节点的名字。与 graph.py 里 add_node("human_approval", ...) 必须一致;
# tests/test_core/test_triage_service.py 有一条断言把它和真实图结构对起来。
HUMAN_APPROVAL_NODE = "human_approval"


class TriageError(Exception):
    """triage / resume 的领域错误基类。API 层按子类映射 HTTP 状态码。"""


class UnknownThreadError(TriageError):
    """thread_id 从未存在过(既无 checkpoint 也无任何审计)→ 404。"""


class NotAwaitingApprovalError(TriageError):
    """thread 存在但不处于待审批状态(已完成 / 停在非预期节点)→ 409。"""


class CheckpointLostError(NotAwaitingApprovalError):
    """审计证明审批曾被请求,但 checkpoint 已丢失(进程重启)→ 409。

    继承 NotAwaitingApprovalError:对客户端而言都是"现在没法恢复",
    状态码相同;分成两个类是为了让日志和测试能区分根因 ——
    一个是"你来晚了",一个是"服务重启过"。
    """


class TriageDataUnavailableError(TriageError):
    """数据源不可用(日志 / 威胁情报文件缺失)→ 503。

    刻意**不**携带原始异常信息:collect_evidence 抛出的 FileNotFoundError
    消息里含**绝对路径**,属于内部部署信息,不能回给客户端。
    """


@dataclass(frozen=True)
class TriageResult:
    """triage() / resume() 的返回值。

    interrupt_id 刻意**不放进 TriageOutcome**(D7):
    TriageOutcome 是"判定结果"的领域模型,会随 checkpoint 持久化、也会被
    审计引用;而 interrupt_id 是**框架运行时**的恢复句柄 —— 由框架在
    interrupt() 内部分配,只在当前暂停期内有效。把运行时句柄混进领域模型,
    会让同一个"结果"在不同时刻具有不同字段含义。
    所以放在这一层单独传递,API 层再决定要不要暴露给客户端。
    """

    outcome: TriageOutcome
    interrupt_id: str | None = None


def _initial_messages(indicator: str, event_type: str | None) -> list[BaseMessage]:
    """构造 triage 的初始消息。

    system prompt 复用 SecurityAgent 的常量(单一真相源,agent.py 不修改),
    human 消息把 indicator / event_type 显式写进去。
    LLM 在这里只负责**叙事**:结构化判定(等级/动作/是否需审批)全部由
    plan 节点的规则引擎产出(见 app/core/graph.py docstring 第 2 条)。
    """
    target = indicator if not event_type else f"{indicator}(事件类型:{event_type})"
    return [
        SystemMessage(content=SECURITY_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=f"请分析该安全对象并给出处置建议:{target}"),
    ]


def _extract_answer(messages: list) -> str:
    """从最终消息历史中提取对外回答。

    与 SecurityAgent._extract_final_answer **同契约**,这里是等价实现 ——
    刻意不调用那个私有方法(跨模块访问私有成员会把两处实现焊死),
    也不修改 agent.py(Phase 8.4 约束 1)。
    两者的一致性由 tests/test_core/test_triage_service.py 的
    test_extract_answer_matches_security_agent 守着,防止悄悄漂移。

    末尾是带 tool_calls 的 AIMessage = 被迭代上限终止,
    此时返回受限说明而不是空串(与 Phase 3 起的对外行为一致)。
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if not msg.tool_calls and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        return MAX_ITERATIONS_REPLY
    return MAX_ITERATIONS_REPLY


def _interrupt_id_of(state: dict) -> str | None:
    """从 ainvoke 的返回体里取出本次暂停的 interrupt_id。

    实测确认它与 aget_state().tasks[*].interrupts[*].id 是同一个值;
    这里用返回体,省掉一次 checkpoint 读。
    """
    interrupts = state.get("__interrupt__") or ()
    return interrupts[0].id if interrupts else None


class TriageService:
    """HITL 判定的应用服务:triage() 发起,resume() 恢复。

    参数:
        graph: create_agent_graph(..., hitl=...) 编译出的图。刻意不写死
               CompiledStateGraph 的内部导入路径(langgraph 版本间会变),
               本模块只用到它的两个公开入口 ainvoke / aget_state。
        store: SqliteAuditStore,用于沉淀 incident / action_requests
               以及 resume 前的校验门查询。

    本类**不持有** checkpointer —— 它属于图的装配(组合根创建一次),
    在这里再拿一份就会出现两个 saver 实例、两套 checkpoint。
    """

    def __init__(self, graph, store: SqliteAuditStore) -> None:
        self._graph = graph
        self._store = store

    # ---------- 发起 ----------

    async def triage(
        self, indicator: str, *, event_type: str | None = None
    ) -> TriageResult:
        """对 indicator 发起一次 HITL 判定。

        返回:
            图在 human_approval 暂停 → status=pending_approval,
            interrupt_id 非空;
            图直接跑完(策略 allow)→ status=completed,interrupt_id=None。

        thread_id 由服务端生成(D3):见模块 docstring 的 4 个框架行为。
        """
        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = await self._graph.ainvoke(
                {
                    "messages": _initial_messages(indicator, event_type),
                    "iteration_count": 0,
                    "indicator": indicator,
                    "event_type": event_type,
                },
                config,
            )
        except FileNotFoundError as exc:
            # 数据文件缺失是**部署问题**,不是客户端错误(4xx)。
            # 原始消息含绝对路径 → 只记日志,不回传。
            logger.error(
                "triage_data_unavailable",
                thread_id=thread_id,
                indicator=indicator,
                error_type=type(exc).__name__,
            )
            raise TriageDataUnavailableError("安全数据源不可用") from exc

        plan = state.get("plan")
        request = state.get("approval_request")
        answer = _extract_answer(state.get("messages") or [])

        # 沉淀:无论是否需要审批,只要产出了计划就落一条 incident
        self._persist_incident(plan, request)

        if request is not None:
            logger.info(
                "triage_pending_approval",
                thread_id=thread_id,
                indicator=indicator,
                risk_level=plan.risk_level if plan is not None else None,
            )
            return TriageResult(
                outcome=TriageOutcome(
                    thread_id=thread_id,
                    status="pending_approval",
                    answer=answer,
                    plan=plan,
                    approval_request=request,
                ),
                interrupt_id=_interrupt_id_of(state),
            )

        logger.info(
            "triage_completed",
            thread_id=thread_id,
            indicator=indicator,
            has_plan=plan is not None,
        )
        return TriageResult(
            outcome=TriageOutcome(
                thread_id=thread_id,
                status="completed",
                answer=answer,
                plan=plan,
            )
        )

    # ---------- 恢复 ----------

    async def resume(
        self,
        thread_id: str,
        *,
        status: ApprovalStatus,
        operator: str,
        reason: str | None = None,
    ) -> TriageResult:
        """对暂停中的判定给出人工决定,并把图跑完。

        interrupt_id **由服务端恢复**,不接受客户端传入(D7):
        客户端能指定 interrupt_id 就等于能伪造"审批的是哪一次暂停"。

        这里不捕获 FileNotFoundError:plan 节点是**已完成节点**,
        resume 不会重放它(实测确认:resume 后 plan.created 仍只有一条),
        因此采集证据的 I/O 不会再次发生。
        """
        config = {"configurable": {"thread_id": thread_id}}
        interrupt_id = await self._validate_resumable(thread_id, config)

        state = await self._graph.ainvoke(
            Command(resume={
                "status": status,
                "operator": operator,
                "reason": reason,
                "interrupt_id": interrupt_id,
            }),
            config,
        )

        decision = state.get("approval_decision")
        logger.info(
            "triage_resumed",
            thread_id=thread_id,
            status=status,
            operator=operator,
        )
        return TriageResult(
            outcome=TriageOutcome(
                thread_id=thread_id,
                status="completed",
                answer=_extract_answer(state.get("messages") or []),
                plan=state.get("plan"),
                # 审批依据留痕:终态仍带上"批的是什么"(D2 放宽校验的原因)
                approval_request=state.get("approval_request"),
                approval=decision,
            ),
            interrupt_id=interrupt_id,
        )

    # ---------- 内部 ----------

    async def _validate_resumable(self, thread_id: str, config: dict) -> str:
        """恢复前的校验门 —— 框架自己不做任何校验(实测确认)。

        返回本次暂停的 interrupt_id。

        判定树:
            next == ("human_approval",) → 正常,取出 interrupt_id
            next 为其它非空值           → 停在非预期节点 → 409
            next == ()                  → 三种情况必须区分开,否则会把
                                          "未知 thread" 当成恢复成功:
                有未决 action_requests  → checkpoint 丢了(服务重启)→ 409
                有审计但无未决          → 审批已完成 → 409
                什么都没有              → thread 从未存在 → 404
        """
        snapshot = await self._graph.aget_state(config)

        if snapshot.next == (HUMAN_APPROVAL_NODE,):
            tasks = snapshot.tasks
            interrupts = tasks[0].interrupts if tasks else ()
            if not interrupts:
                # 停在 human_approval 却拿不到 interrupt 句柄:
                # 与"checkpoint 丢了"同类 —— 现在恢复不了。
                raise CheckpointLostError(
                    "thread 停在审批节点,但取不到 interrupt 句柄,无法恢复"
                )
            return interrupts[0].id

        if snapshot.next:
            raise NotAwaitingApprovalError(
                f"thread 未处于待审批状态(停在 {snapshot.next[0]!r})"
            )

        # ---- next == ():图没在跑。区分"跑完了" / "丢了" / "根本不存在" ----
        if self._store.pending_action_rows(thread_id=thread_id):
            raise CheckpointLostError(
                "审计显示仍有未决审批,但 checkpoint 已丢失(服务重启);"
                "本阶段 checkpoint 为进程内存储,无法恢复"
            )
        if self._store.list_audit(thread_id=thread_id):
            raise NotAwaitingApprovalError("该 thread 的审批已完成,不能重复提交")
        raise UnknownThreadError("未知的 thread_id")

    def _persist_incident(self, plan, request) -> str | None:
        """把一次判定的结果沉淀为 incident(+ 审批单),返回 incident_id。

        incident 在**图跑完之后**才创建(D4:incident_id 不进 AgentState),
        所以图节点写下的审计行 incident_id 为 NULL —— 详见模块 docstring
        的"已知限制 1"。

        request 为 None(策略 allow)时只落 incident,不写 action_requests:
        后者是**审批单**的存储,为 allow 路径伪造一张审批单会让
        pending_action_rows 永远派生出一批"待审批",破坏派生状态的语义。
        """
        if plan is None:
            return None

        incident = Incident(
            id=uuid.uuid4().hex,
            created_at=utc_now(),
            indicator=plan.indicator,
            risk_level=plan.risk_level,
            score=plan.assessment.score,
            summary=plan.summary,
            plan=plan,
        )
        self._store.record_incident(incident)
        if request is not None:
            self._store.record_action_request(request, incident_id=incident.id)
        return incident.id
