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

    失败路径(Phase 8.5):plan 节点写一条 plan.failed 后抛 PlanFailedError,
    服务**不写** incident / action_requests —— 没有产出计划,就没有可沉淀的对象。
    所以一次失败的判定在审计里恰好只有 1 条 plan.failed。

审批超时(Phase 9.1-A):
    pending 的 thread 不会永远等下去。窗口来自 APPROVAL_TIMEOUT,
    判定**全部是惰性的**(没有后台调度器):
        - triage() 入口 → reap_expired() 全量扫一遍已过期的 pending;
        - resume(thread_id) → 只检查本 thread(点检查,在取 interrupt_id 之前)。
    超时锚点 = 该 thread 的 **min(action_requests.requested_at)**,
    判定式 = utc_now() - 锚点 >= approval_timeout(闭区间)。
    锚点刻意取自 action_requests 而不是 approval.requested 审计的 ts:
    pending 的**定义**已经是"action_requests 行 + audit 的 NOT EXISTS",
    锚点必须落在同一处,否则会出现两套时间真相。

    超时的两个副作用,顺序固定:先 append 一条 approval.timeout(终态事实),
    再删除该 thread 的 checkpoint。超时**绝不**写 approval.decided ——
    没有人工决定,就不能留下决定的痕迹。超时是**不可逆**的:
    approval.timeout 一旦落库,该 thread 永久 409,不允许事后补批。

    超时后 pending_action_rows 不再把它算作 pending(store 侧谓词同步改成
    "既无 approval.decided 也无 approval.timeout")。这一步是必需的 ——
    否则每轮 reap 都会给同一个 thread 重复写一条 approval.timeout,
    而审计是事实日志,重复计数就是失真。

清理范围(Phase 9.1-A,刻意收窄):
    删除 checkpoint 只发生在**终态**:failed(plan 节点抛错后残留的
    next=('plan',) 快照)与 timed_out。pending_approval **绝不删除** ——
    删掉就等于把待审批的请求凭空抹掉。completed / allowed 本轮**不清理**
    (不是"必须保留",只是 9.1-A 不做),因此它们仍会占着 checkpoint 存储。

已知限制(必须文档化,不得掩盖):
    1. **audit_logs.incident_id 为 NULL**。incident 在图跑完之后才创建
       (D4 决定 incident_id 不进 AgentState),所以图节点写审计时无从得知它。
       后果:list_audit(incident_id=...) 查不到本次判定的审计流,
       必须改用 list_audit(thread_id=...)。Phase 8.5 候选(需新增审计事件
       或在 State 里放 incident_id,两者都超出 8.4 范围)。
       9.1-A **刻意不扩大**这个限制:approval.timeout 也不传 incident_id,
       与图节点保持一致。
    2. **checkpoint 的存储由组合根决定**(Phase v0.4.0 S1 起是落盘的 SQLite
       saver;S1 之前是进程内内存)。本服务层**不感知**这一点 —— 它只通过
       `HitlConfig.checkpointer` 拿到一个 saver 契约。
       即便 checkpoint 落盘,**审计与 checkpoint 之间也没有原子性**:
       "决定已记录"与"运行已完成"**不是**联合原子的,任何措辞都不得暗示
       它们是(见 PHASE v0.4.0-D4 §5.6)。
       持久事实在 incidents / action_requests / audit_logs 三张表里。
    3. **超时只在有流量时被观测**。triage() 是全项目唯一的"全局"请求入口。
       若长时间没有新的判定请求,已过期的 pending thread 会继续显示为
       pending、checkpoint 继续占存储,直到下一个 /triage 到达。
       这是"无后台调度器"的必然结果,不是 bug。
    4. **超时状态只能从 audit_logs 观测**。TriageOutcome.status 刻意不新增
       timed_out(不改 schema),也没有"查询 thread 状态"的端点 ——
       所以超时在 API 响应里只以 409 的形式出现。
    5. **completed / allowed 的 checkpoint 本轮不清理**。9.1-A 的清理范围
       只有 failed 与 timed_out 两个终态,不是"必须保留"。
    6. **决策唯一性是"存储层 at-most-once",不是端到端 exactly-once**
       (v0.4.0 S1 / TASK 4)。存储层保证同一**决策身份**最多落库一次:
       approval.decided 的身份是 (thread_id, interrupt_id),approval.timeout
       是 thread_id,由两条部分唯一索引强制;并发恢复里因此恰好一个胜者
       (实测:并发两次 resume,approval.decided 恰好一条)。
       它**不**保证:
         - 审计写入与 checkpoint 推进的原子性(见限制 2);
         - 客户端一定收到响应 —— 胜者的响应可能在网络上丢失,客户端重试
           会拿到 409。所以"重试安全"成立,"恰好一次交互"不成立;
         - **interrupt_id 为 NULL 的身份**:SQLite 与 PostgreSQL 都认为
           NULL 互不相等,因此 NULL 身份不受唯一索引保护(实测)。经 API 的
           /resume 由服务端注入 interrupt_id(D7),所以经 API 的决策恒为非
           NULL;这条只对"直接调用图"的旁路成立,本阶段不承诺该旁路。
"""
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.types import Command

from app.core.agent import MAX_ITERATIONS_REPLY, SECURITY_ANALYST_SYSTEM_PROMPT
from app.core.graph import PlanFailedError
from app.schemas.approval import ApprovalStatus, TriageOutcome, utc_now
from app.schemas.incident import Incident
from app.security.audit import build_audit_record

# checkpoint 存储故障族(v0.4.0 S1 / TASK 5)。服务层必须把它翻译成**领域**
# 错误,否则会以未归类异常的身份冒到 HTTP 边界 → 500;而它是**部署问题**,
# 必须是 503(见下方 `_translate_checkpoint_outage`)。
from app.security.checkpoint import CheckpointStorageError

# 存储层的**决策重复**错误(v0.4.0 S1 / TASK 4)。这是本模块唯一一处引用
# 具体存储模块(而非 `AuditStore` 契约)的地方,刻意为之:该错误类由
# SQLite 与 PostgreSQL 两个后端**共同**抛出,定义在 `store.py` 里,而
# `store_protocol.py` 不在本阶段的授权改动清单内 —— 契约层无法承载它。
# 服务层必须把这个**存储层**条件翻译成**领域**错误(与 CheckpointLostError
# 同一职责),否则它会被 HTTP 边界当成未归类异常 → 500,而不是 fail-closed 的 409。
from app.security.store import DecisionAlreadyRecordedError
from app.security.store_protocol import AuditStore

logger = structlog.get_logger(__name__)

# 暂停点节点的名字。与 graph.py 里 add_node("human_approval", ...) 必须一致;
# tests/test_core/test_triage_service.py 有一条断言把它和真实图结构对起来。
HUMAN_APPROVAL_NODE = "human_approval"

# 审批窗口。这是 **core 的生命周期策略默认值**,不是部署配置 ——
# Phase 9.1-A 刻意不为一个参数扩大部署配置面(Settings 目前只有
# audit_db_path 一项,它必须存在是因为组合根要知道库在哪)。
# 日后确需部署期可配,由组合根通过 TriageService(approval_timeout=...)
# 注入即可,生命周期语义不变。
APPROVAL_TIMEOUT: timedelta = timedelta(hours=24)

# 超时终态事件。SQL 侧(store.pending_action_rows)不得不写字面量,
# Python 侧集中在这里,避免两处漂移。
TIMEOUT_EVENT: str = "approval.timeout"

# 超时拒绝的对外消息。两处会抛同一个终态(恢复时判定 / 已落库后再次恢复),
# 消息必须一致,否则同一事实会出现两种说法。
_EXPIRED_MESSAGE = "该 thread 的审批已超时,不能再提交决定"


class TriageError(Exception):
    """triage / resume 的领域错误基类。API 层按子类映射 HTTP 状态码。"""


class UnknownThreadError(TriageError):
    """thread_id 从未存在过(既无 checkpoint 也无任何审计)→ 404。"""


class NotAwaitingApprovalError(TriageError):
    """thread 存在但不处于待审批状态(已完成 / 停在非预期节点)→ 409。"""


class CheckpointLostError(NotAwaitingApprovalError):
    """审计证明审批曾被请求,但 checkpoint 已丢失 → 409。

    继承 NotAwaitingApprovalError:对客户端而言都是"现在没法恢复",
    状态码相同;分成两个类是为了让日志和测试能区分根因 ——
    一个是"你来晚了",一个是"存储里已经没有这条运行状态"。
    """


class ApprovalExpiredError(NotAwaitingApprovalError):
    """审批窗口已过,该 thread 不能再被恢复 → 409。

    继承理由与 CheckpointLostError 完全同构:对客户端都是"现在没法恢复",
    状态码相同;单列一类是为了让日志和测试能区分根因 —— 这个的根因是
    "窗口过期了",与前两者都不同。

    超时是**不可逆**的:approval.timeout 一旦落库,该 thread 永久 409。
    不允许事后补批 —— 这是超时的语义本身,不是实现疏漏。
    """


class TriageDataUnavailableError(TriageError):
    """判定所需的数据源不可用 → 503。

    **所有** plan 节点失败统一映射到这里(Phase 8.5 决策):plan 节点的输入
    只有数据文件 + 纯规则函数,失败压倒性是数据/部署问题。刻意不细分
    "数据问题 503 / 代码 bug 500" —— 那需要给 PlanFailedError 加 cause_type
    判别字段;而真实 error_type 已写进 plan.failed 审计与 error 日志,
    排查信息不丢,不值得为此加一个分支。

    消息恒定且通用,不回传任何原始异常信息(路径属于内部部署信息)。
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


@contextmanager
def _translate_checkpoint_outage(
    thread_id: str | None, request_id: str | None
) -> Iterator[None]:
    """checkpoint 存储故障 → `TriageDataUnavailableError`(→ 503)。

    **为什么必须翻译,而不是让它自己冒出去**:HTTP 边界只把 `TriageError`
    映射成状态码,其它异常一律 500。而 checkpoint 故障是**部署问题**,
    不是客户端错误 —— 它必须是 503,且**绝不**能退化成:

      - 404 "未知 thread" —— 数据还在,只是读不到,说"不存在"是错误陈述;
      - 409 "checkpoint 丢失 / 审批已完成" —— 同样是对世界的错误陈述
        (PHASE v0.4.0-D4 §5.2 / §5.4)。

    覆盖两个子类,日志里用 `error_type` 区分,但状态码同为 503:

      - `CheckpointBackendUnavailableError` —— 文件 / 连接 / 磁盘故障;
      - `CheckpointPayloadRejectedError` —— 载荷损坏、被篡改,或不在白名单内。

    只翻译 checkpoint 自己的错误族。审计存储的失败**不**经过这里,因此
    v0.3.0 已文档化的"审计读 503 / 写 500"不对称保持不变(D4 §5.4 要求它不变)。
    """
    try:
        yield
    except CheckpointStorageError as exc:
        logger.error(
            "triage_checkpoint_unavailable",
            request_id=request_id,
            thread_id=thread_id,
            error_type=type(exc).__name__,
        )
        raise TriageDataUnavailableError("checkpoint 存储不可用") from exc


class TriageService:
    """HITL 判定的应用服务:triage() 发起,resume() 恢复。

    参数:
        graph: create_agent_graph(..., hitl=...) 编译出的图。刻意不写死
               CompiledStateGraph 的内部导入路径(langgraph 版本间会变),
               本模块只用到它的两个公开入口 ainvoke / aget_state。
        store: AuditStore(契约),用于沉淀 incident / action_requests
               以及 resume 前的校验门查询。
        approval_timeout: 审批窗口(keyword-only,默认 APPROVAL_TIMEOUT)。
               只有测试需要覆盖它(用 timedelta(0) 让 pending 立即过期);
               生产走默认值,组合根日后想部署期可配时注入即可。

    本类**不持有** checkpointer —— 它属于图的装配(组合根创建一次),
    在这里再拿一份就会出现两个 saver 实例、两套 checkpoint。

    那超时清理怎么删 checkpoint?从**图上取**:self._graph.checkpointer。
    它是 CompiledStateGraph 的**公开属性**,且实测 `is` 组合根传进
    create_agent_graph 的同一个实例 —— 因此不存在"第二份 saver"的问题,
    这正是本类不持有它的意义所在。这条依赖由
    tests/test_core/test_approval_lifecycle.py 的身份断言守着。
    """

    def __init__(
        self,
        graph,
        store: AuditStore,
        *,
        approval_timeout: timedelta = APPROVAL_TIMEOUT,
    ) -> None:
        self._graph = graph
        self._store = store
        self._approval_timeout = approval_timeout

    # ---------- 发起 ----------

    async def triage(
        self,
        indicator: str,
        *,
        event_type: str | None = None,
        request_id: str | None = None,
        thread_id: str | None = None,
    ) -> TriageResult:
        """对 indicator 发起一次 HITL 判定。

        返回:
            图在 human_approval 暂停 → status=pending_approval,
            interrupt_id 非空;
            图直接跑完(策略 allow)→ status=completed,interrupt_id=None。

        thread_id 由服务端生成(D3):见模块 docstring 的 4 个框架行为。

        thread_id(v0.3.0-A3-3 新增,可选):**HTTP 边界可以预先分配**一个
        thread_id 传进来。存在这个参数的**唯一**理由:线程归属必须在图开始
        执行**之前**写好,而归属记录里必须带 thread_id —— 于是调用方需要
        先知道它。默认 `None` ⇒ 本方法照旧自己生成一个 uuid4。

        ⚠️ 它**绝不是**客户端输入面:调用方(API 边界)只传自己生成的
        uuid4,永不回填请求体里的任何值。D3 的性质(客户端无法指定
        thread_id)因此没有改变 —— 改变的只是"由服务端的哪一层生成"。

        request_id(Phase 9.3-D):本**HTTP 请求**的运维身份,由 API 边界生成,
        与 thread_id 语义不同、绝不互相派生。它经 graph 的 `configurable`
        下发,不放进 AgentState(运维元数据不是 agent 推理状态)。

        入口先做惰性超时清理(见 reap_expired)。清理失败在这里**不吞** ——
        它是真实故障,应当响亮失败;唯一放宽的路径是下面那个"已在处理
        主失败"的 except 分支。
        """
        await self.reap_expired(request_id=request_id)

        if thread_id is None:
            thread_id = uuid.uuid4().hex
        elif not isinstance(thread_id, str) or not thread_id.strip():
            # 预分配的 id 必须是一个非空字符串 —— 空值会让 checkpoint 与
            # 归属记录落到一个"无名线程"上,后续任何 resume 都无处可依。
            raise ValueError("thread_id must be a non-empty string when provided")
        config = {"configurable": {"thread_id": thread_id, "request_id": request_id}}

        try:
            with _translate_checkpoint_outage(thread_id, request_id):
                state = await self._graph.ainvoke(
                    {
                        "messages": _initial_messages(indicator, event_type),
                        "iteration_count": 0,
                        "indicator": indicator,
                        "event_type": event_type,
                    },
                    config,
                )
        except PlanFailedError as exc:
            # plan 节点的失败契约:消息通用、不含路径(见 graph.PlanFailedError)。
            # 细节在 __cause__ 里,且已由 graph 层写进 plan.failed 审计与 error 日志。
            logger.error(
                "triage_plan_failed",
                request_id=request_id,
                thread_id=thread_id,
                indicator=indicator,
            )
            # 失败会在 checkpoint 里留下 next=('plan',) + tasks[0].error 的残留
            # 快照,清掉它 —— 该 thread 是终态(failed),不可能再被恢复。
            #
            # 错误优先级规则(9.1-A):主失败**已经确立**(→ 503),
            # 附带清理失败不得顶替它。所以这里降级为 warning 并保留原始
            # 领域异常。刻意只在这一个"已在处理主失败"的路径上放宽 ——
            # 其它路径的清理失败一律抛出。这不是通用错误框架,只是一条
            # 局部规则:谁先失败,谁就代表这次调用的结果。
            try:
                await self._drop_checkpoint(thread_id)
            except Exception as cleanup_exc:
                logger.warning(
                    "triage_cleanup_failed",
                    request_id=request_id,
                    thread_id=thread_id,
                    error_type=type(cleanup_exc).__name__,
                )
            raise TriageDataUnavailableError("安全数据源不可用") from exc

        plan = state.get("plan")
        request = state.get("approval_request")
        answer = _extract_answer(state.get("messages") or [])

        # 沉淀:无论是否需要审批,只要产出了计划就落一条 incident。
        # incident_id 只用于**日志关联** —— 绝不回填进 AgentState / 审计行
        # (D4:incident 在图跑完之后才创建),沉淀语义完全不变。
        incident_id = self._persist_incident(plan, request)

        if request is not None:
            # interrupt_id 此刻已可从返回体取到(与 aget_state().tasks[*].interrupts[*].id
            # 同值),它是恢复句柄、只在本次暂停期有效 —— 只用于日志关联。
            interrupt_id = _interrupt_id_of(state)
            logger.info(
                "triage_pending_approval",
                request_id=request_id,
                thread_id=thread_id,
                interrupt_id=interrupt_id,
                incident_id=incident_id,
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
                interrupt_id=interrupt_id,
            )

        logger.info(
            "triage_completed",
            request_id=request_id,
            thread_id=thread_id,
            incident_id=incident_id,
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
        request_id: str | None = None,
    ) -> TriageResult:
        """对暂停中的判定给出人工决定,并把图跑完。

        interrupt_id **由服务端恢复**,不接受客户端传入(D7):
        客户端能指定 interrupt_id 就等于能伪造"审批的是哪一次暂停"。

        operator(v0.3.0-A3-3 起语义收紧):它是**已认证主体**的 subject,
        由 API 边界从 `Principal` 取出后传入 —— **不是**调用方自述值。
        它会被写进 `Command(resume=...)`,图节点据此写审计 `actor`,
        因此审计归属从这一阶段起可信(客户端无法伪造)。图本身仍只读
        `decision.operator`(单一真相源未变);改变的是"谁往里填值"。

        **对象级授权不在这里**:本方法只做领域操作,不做"你有没有被指派
        审批这条线程"的判定 —— 那是 HTTP 边界的职责(`app/api/main.py`
        的 `_authorize_thread_read`),因为状态码(404/403)是 HTTP 概念。
        直接调用本方法的内部代码不经过那道闸门,这是**已知边界**。

        request_id(Phase 9.3-D):**本次** /resume HTTP 请求的运维身份。
        它与发起暂停那次 /triage 的 request_id **必然不同**(两次独立请求),
        而 thread_id 保持不变 —— 这正是把两者分开的理由(见 §8 冻结决定)。

        这里不捕获 PlanFailedError:plan 节点是**已完成节点**,
        resume 不会重放它(实测确认:resume 后 plan.created 仍只有一条),
        因此采集证据的 I/O 不会再次发生。
        """
        config = {"configurable": {"thread_id": thread_id, "request_id": request_id}}
        try:
            # `_validate_resumable` 内部会读 checkpoint(aget_state),
            # 因此它也必须在同一条 503 翻译之下 —— checkpoint 后端不可用时
            # 它不能表现为 404 / 409。
            with _translate_checkpoint_outage(thread_id, request_id):
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
        except DecisionAlreadyRecordedError as exc:
            # **并发恢复的败者**走到这里。校验门对它不可见:两个请求都在对方
            # 落库之前读到了 "停在 human_approval",于是都通过了
            # `_validate_resumable`。真正拦住它的是存储层的唯一索引。
            #
            # 实测确认两件事:(1) 该异常**原样穿出** `ainvoke`,没有被框架
            # 包装;(2) 并发场景下 approval.decided 恰好一条。
            #
            # 刻意转成与"顺序重复提交"**同一个**领域错误,而不是新开一个
            # 分支:两者是同一个领域事实(这条 thread 已经有决定了),因此
            # 应当收敛到同一个 fail-closed 的 409。
            raise NotAwaitingApprovalError(
                "该 thread 的审批已完成,不能重复提交"
            ) from exc

        decision = state.get("approval_decision")
        logger.info(
            "triage_resumed",
            request_id=request_id,
            thread_id=thread_id,
            interrupt_id=interrupt_id,
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

        判定树(9.1-A 起含超时):
            next == ("human_approval",) → 已过期则收成终态 → 409;
                                          否则正常,取出 interrupt_id
            next 为其它非空值           → 停在非预期节点 → 409
            next == ()                  → 五种情况必须区分开,否则会把
                                          "未知 thread" 当成恢复成功:
                有未决 action_requests:
                    已过期              → 收成终态 → 409(超时)
                    未过期              → checkpoint 丢了(存储不再持有该运行状态)→ 409
                有 approval.timeout     → 已超时 → 409
                有 approval.decided     → 审批已完成 → 409
                有其它审计(如 plan.failed)→ 未产生待审批项 → 409
                什么都没有              → thread 从未存在 → 404

        "有其它审计" 这一支是 9.1-A 补的:失败路径的 checkpoint 现在会被清掉,
        于是 resume 一个失败的 thread 会落到这里 —— 原来的消息
        "该 thread 的审批已完成,不能重复提交" 对它是**事实错误**
        (它从没完成,是失败了),必须分开说。
        """
        snapshot = await self._graph.aget_state(config)
        request_id = (config.get("configurable") or {}).get("request_id")

        if snapshot.next == (HUMAN_APPROVAL_NODE,):
            # 超时检查必须在取 interrupt_id **之前**:否则已过期的 thread
            # 会先拿到恢复句柄,之后才被判超时。
            await self._reject_if_expired(thread_id, request_id)
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
        # 先无条件判超时:已落过 approval.timeout 的 thread 永久作废,
        # 与 checkpoint 是否还在无关(清理可能失败,但结论不能因此松动)。
        await self._reject_if_expired(thread_id, request_id)
        if self._store.pending_action_rows(thread_id=thread_id):
            raise CheckpointLostError(
                "审计显示仍有未决审批,但对应的 checkpoint 已丢失,无法恢复;"
                "checkpoint 由持久化存储承载,该运行状态可能因存储被清空、"
                "重置或指向了另一份存储而丢失"
            )

        audit = self._store.list_audit(thread_id=thread_id)
        if audit:
            if any(record.event == "approval.decided" for record in audit):
                raise NotAwaitingApprovalError("该 thread 的审批已完成,不能重复提交")
            # 有审计但没有终态审批事件:规划失败(只有 plan.failed)或策略放行
            # (plan.created / policy.evaluated)。两者都不存在"待审批的决定"。
            raise NotAwaitingApprovalError(
                "该 thread 的判定未产生待审批项(规划失败或策略放行),不能提交决定"
            )
        raise UnknownThreadError("未知的 thread_id")

    # ---------- 内部:超时(惰性) ----------

    async def reap_expired(self, *, request_id: str | None = None) -> list[str]:
        """把已过期的待审批 thread 收成终态,返回本次收掉的 thread_id。

        惰性 = **只在请求到达时执行**,没有后台调度器(9.1-A 约束)。
        当前唯一的全局请求入口是 triage(),所以挂在它开头。代价见模块
        docstring 已知限制 3:长时间无判定请求时,过期的 pending 会继续
        显示为 pending。

        只处理**有 pending 行**的 thread:没有审批单的 thread 不参与超时
        判定(它可能已经 completed / allowed / failed)。

        顺带收益:pending 是纯 DB 派生,所以进程重启留下的僵尸 pending 行
        (没有 checkpoint、resume 永远 409)也会在窗口过期后被收成
        timed_out,不再无限期污染 pending 视图。

        清理失败在这里**不吞**:reap 是本方法自己的主任务,失败就是失败。
        """
        rows = self._store.pending_action_rows()
        if not rows:
            return []

        # 锚点 = 该 thread 的 min(requested_at)。min() 而非取第一行:
        # 同一 thread 的行由同一次 executemany 写入、requested_at 恒等,
        # 但 min() 让"窗口何时打开"的定义对未来多批次写入仍然成立。
        anchors: dict[str, datetime] = {}
        for row in rows:
            thread_id = row["thread_id"]
            requested_at = row["requested_at"]
            current = anchors.get(thread_id)
            if current is None or requested_at < current:
                anchors[thread_id] = requested_at

        now = utc_now()
        reaped: list[str] = []
        for thread_id, anchor in anchors.items():
            elapsed = now - anchor
            if elapsed < self._approval_timeout:
                continue
            await self._reap_one(thread_id, elapsed)
            reaped.append(thread_id)

        if reaped:
            logger.info(
                "triage_expired_reaped", request_id=request_id, count=len(reaped)
            )
        return reaped

    async def _reject_if_expired(
        self, thread_id: str, request_id: str | None = None
    ) -> None:
        """该 thread 已过审批窗口 → 收成 timed_out 终态并拒绝恢复。

        未过期(或根本没有 pending 行)时**什么都不做** —— 这个方法是
        校验门里的一个前置判定,不是清理器。

        顺序很关键:先看 approval.timeout **是否已落库**,再看窗口。
        只看窗口的话,一旦 adelete_thread 没生效(后端不支持删除),
        第二次 resume 会重新看到 pending 行 —— 但 pending 行的谓词已经
        把 timeout 算作终态,于是"已作废的审批"又能被恢复。
        先判已落库,超时就与清理是否成功彻底解耦。
        """
        if self._has_timed_out(thread_id):
            raise ApprovalExpiredError(_EXPIRED_MESSAGE)

        anchor = self._expired_anchor(thread_id)
        if anchor is None:
            return
        await self._reap_one(thread_id, utc_now() - anchor)
        logger.warning(
            "triage_resume_expired", request_id=request_id, thread_id=thread_id
        )
        raise ApprovalExpiredError(_EXPIRED_MESSAGE)

    def _expired_anchor(self, thread_id: str) -> datetime | None:
        """已过期则返回该 thread 的超时锚点;未过期或没有 pending 行则 None。

        判定式是**闭区间**:elapsed >= approval_timeout 即过期。用 >= 而非 >
        是为了让"窗口长度为 0"(测试用 timedelta(0))立刻生效。
        代价是"恰好等于 deadline"那一瞬无法在不注入时钟的前提下测到 ——
        刻意接受的测试缺口,不为此引入时钟注入。
        """
        anchor = self._thread_anchor(thread_id)
        if anchor is None:
            return None
        if (utc_now() - anchor) >= self._approval_timeout:
            return anchor
        return None

    def _thread_anchor(self, thread_id: str) -> datetime | None:
        """该 thread 的 pending 行里最早的 requested_at;无 pending 行则 None。

        requested_at 由 store 的 _from_iso 读回,是 tz-aware UTC
        (写侧拒绝 naive),所以这里的减法不会踩到时区混用。
        """
        rows = self._store.pending_action_rows(thread_id=thread_id)
        if not rows:
            return None
        return min(row["requested_at"] for row in rows)

    def _has_timed_out(self, thread_id: str) -> bool:
        """该 thread 是否已经落过 approval.timeout。

        _reap_one 的幂等闸门:即使 adelete_thread 因故没生效(thread 仍停在
        human_approval),第二次也不会写出第二条审计 —— 审计是事实日志,
        重复计数就是失真。
        """
        return bool(self._store.list_audit(thread_id=thread_id, event=TIMEOUT_EVENT))

    async def _reap_one(self, thread_id: str, elapsed: timedelta) -> None:
        """把一个已过期的 pending thread 收成 timed_out 终态。

        两个副作用,顺序固定:先 append 审计(终态事实),再删 checkpoint。

        超时**绝不**写 approval.decided —— 没有人工决定,就不能留下决定的
        痕迹。detail 里放 elapsed / timeout 秒数,让"为什么算过期"可复算。

        incident_id 刻意不传:图节点写审计时同样拿不到它(incident 在图跑完
        之后才创建,D4),9.1-A 不扩大这个已知限制。
        """
        if not self._has_timed_out(thread_id):
            self._store.append_audit(
                build_audit_record(
                    TIMEOUT_EVENT,
                    thread_id=thread_id,
                    detail={
                        "elapsed_seconds": int(elapsed.total_seconds()),
                        "timeout_seconds": int(self._approval_timeout.total_seconds()),
                    },
                )
            )
        await self._drop_checkpoint(thread_id)

    async def _drop_checkpoint(self, thread_id: str) -> None:
        """删除该 thread 的 checkpoint(幂等)。

        这是本模块**唯一**的 adelete_thread 调用点 —— 单一 choke point,
        由 tests/test_core/test_approval_lifecycle.py 的 AST 护栏钉住。
        "绝不删 pending" 这条不变量因此可以被结构性检查,而不只是靠约定。

        checkpointer 从图上取(self._graph.checkpointer):它是公开属性,
        且实测 `is` 组合根传进 create_agent_graph 的同一个实例。本类因此
        仍然**不持有** checkpointer,不违背类 docstring 的约束。

        `adelete_thread` 对未知 / 已删除的 thread 无异常(实测:v0.4.0 S1 之前
        是 `InMemorySaver`,之后是落盘的 `AsyncSqliteSaver` —— 后者内部就是一条
        plain DELETE,匹配不到行时同样静默),所以调用前不需要存在性预检。
        """
        await self._graph.checkpointer.adelete_thread(thread_id)

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
