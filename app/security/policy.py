"""Policy Engine —— 规则式策略判定(Phase 8.1)。

**纯函数 + 单输入契约**:
    ResponsePlan → evaluate_policy() → PolicyDecision
不读文件、不查库、不调用 LLM、不读环境变量、不依赖当前时间、不接受配置输入。

安全约束(勿改,Phase 8 的核心):
    LLM 的输出**永远不能**成为策略判定依据。本模块只消费 ResponsePlan ——
    它由 plan_response() 规则引擎生成,其中的等级、分数、requires_approval
    都不经过 LLM。日志与情报内容会进入 LLM 上下文(存在 prompt injection 面),
    但它们到不了这里:本函数的入参里**没有任何承载自然语言的字段**。
    这一条由 tests/test_security/test_policy.py::test_signature_is_single_input
    锁定。

为什么是纯函数 + 节点,而不是 Tool:
    工具是"LLM 可选调用"的。策略检查若做成工具,LLM 不调用即等于策略不存在
    —— 控制退化为建议性。安全控制必须结构性:由 policy_gate 节点强制调用
    (Phase 8.3),而不是指望 LLM 自觉。

规则集(Phase 8.1):
    R2 require_approval  任一动作 requires_approval=True → 需审批
    R3 require_approval  置信度 < CONFIDENCE_FLOOR 且含破坏性动作 → 需审批
    R4 allow             未触发上述任一规则 → 放行

    编号沿用 Phase 8 设计 Review 的 R1-R4,刻意不重排,便于与设计文档逐条对照。

    R1(受保护资产 → deny)在 8.1 **刻意未实现**:它的输入是"资产白名单",
    属于配置而非 ResponsePlan 的字段。把它做成函数入参会破坏"Policy Engine
    只消费 ResponsePlan"的单输入契约。待后续通过独立 PolicyConfig / Settings
    注入时再启用;PolicyOutcome 中的 "deny" 已保留,届时无需改动 schema。
    这样也避免留下"PROTECTED_TARGETS 永远为空、deny 分支不可达"的死代码。

规则设计要点:
    R3 在当前发布的动作属性表下会被 R2 覆盖(破坏性动作的 requires_approval
    都是 True),它是**纵深防御**:一旦未来有人把某个破坏性动作标记为免审批
    (例如引入"可信自动化"档位),低置信度下仍会被强制升级为人工审批。
    这不是冗余代码,请勿删除。
"""
from collections.abc import Iterable

from app.schemas.policy import PolicyDecision
from app.schemas.response import ActionType, ResponseAction, ResponsePlan

POLICY_VERSION = "phase8.1"

# 置信度地板:低于此值且含破坏性动作 → 强制人工审批。
# 取值参考 risk_analyzer 的 confidence 构成(20 基线 +30 情报 +20 爆破):
# 50 意味着"至少有两类证据相互印证"才允许免审批地执行破坏性动作。
CONFIDENCE_FLOOR = 50

# 破坏性动作:一旦执行就影响业务或不可逆。
DESTRUCTIVE_ACTIONS: frozenset[ActionType] = frozenset({
    "block_ip",
    "isolate_host",
    "reset_credentials",
})


def _unique_types(actions: Iterable[ResponseAction]) -> list[ActionType]:
    """按出现顺序去重动作类型(保持确定性,不用 set 以免顺序不稳)。"""
    out: list[ActionType] = []
    for action in actions:
        if action.action_type not in out:
            out.append(action.action_type)
    return out


def evaluate_policy(plan: ResponsePlan) -> PolicyDecision:
    """对处置计划做规则判定,返回 allow / require_approval。

    单输入契约:唯一入参是 ResponsePlan,判定完全确定 —— 同样的计划永远得出
    同样的结论,可穷举测试、可做 Phase 9 golden set 比对。

    返回的 gated_actions 是"触发该结论的动作类型":require_approval 时是需
    审批的动作,allow 时为空。

    R1(受保护资产 → deny)未实现,原因见模块 docstring。
    """
    reasons: list[str] = []

    # ---- R2:动作自带审批标记(Phase 7 规则侧已判定,此处只做汇总)----
    gated = _unique_types(a for a in plan.actions if a.requires_approval)
    if gated:
        reasons.append(
            f"动作 {gated} 按动作属性需要人工审批(requires_approval=True)"
        )

    # ---- R3:低置信度 + 破坏性动作 → 兜底升级为需审批 ----
    if plan.assessment.confidence < CONFIDENCE_FLOOR:
        destructive = _unique_types(
            a for a in plan.actions if a.action_type in DESTRUCTIVE_ACTIONS
        )
        extra = [t for t in destructive if t not in gated]
        if extra:
            gated.extend(extra)
            reasons.append(
                f"置信度 {plan.assessment.confidence} 低于 {CONFIDENCE_FLOOR},"
                f"且含破坏性动作 {extra},强制人工审批"
            )

    if gated:
        return PolicyDecision(
            outcome="require_approval",
            requires_approval=True,
            gated_actions=gated,
            reasons=reasons,
            policy_version=POLICY_VERSION,
        )

    # ---- R4:放行 ----
    return PolicyDecision(
        outcome="allow",
        requires_approval=False,
        gated_actions=[],
        reasons=[
            f"风险等级 {plan.risk_level}(分数 {plan.assessment.score}),"
            "无动作需要人工审批,策略放行"
        ],
        policy_version=POLICY_VERSION,
    )
