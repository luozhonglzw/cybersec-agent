"""审批(HITL)数据模型 —— Phase 8.1。

三个对象对应审批生命周期的三个阶段:
- ApprovalRequest : 图暂停时交给人的**待审请求**(interrupt 载荷,必须 JSON 可序列化);
- ApprovalDecision: 人给出的**决定**(随 checkpoint 持久化,并写入审计);
- TriageOutcome   : triage() / resume() 对外返回的**聚合结果**。

设计原则:
- status 用 Literal 受限枚举:approved / denied 是封闭集合,"maybe" 这类
  中间态不存在;审批要么通过要么拒绝,没有第三种;
- TriageOutcome 的 approval_request 与 status 是**单向蕴含**(Phase 8.4 D2):
  pending_approval ⇒ 必须有 approval_request,否则客户端拿到一个无法操作的
  响应;反向**不禁止** —— resume 之后的终态必须是
  completed + approval_request(审批依据留痕)+ approval(人工决定),
  用"当且仅当"会把这个唯一合法的终态判为非法,导致 /resume 无法表达结果;
- 所有时间字段用 tz-aware UTC,与 LogEvent / ThreatIntelRecord 一致。

安全约束(勿改):
    ApprovalRequest / ApprovalDecision **都不包含任何"动作是否已执行"的字段**。
    Phase 8 只做审批,**不执行**任何真实处置动作(没有防火墙 / EDR 适配器)。
    引入 execution_status 会制造"已执行"的假象,是安全风险 —— 真正的执行器
    必须挂在 policy_gate 之后,而不是塞进工具或本模型里。
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.response import ResponseAction, ResponsePlan
from app.schemas.risk import RiskLevel

ApprovalStatus = Literal["approved", "denied"]
TriageStatus = Literal["completed", "pending_approval"]


def utc_now() -> datetime:
    """当前 UTC 时间。集中一处,便于测试注入固定时间。"""
    return datetime.now(timezone.utc)


class ApprovalRequest(BaseModel):
    """交给人工审批的请求(interrupt 的载荷)。

    只包含**规则引擎产出**的字段:风险等级、分数、计划摘要、需审批的动作。
    不含任何 LLM 生成的自然语言判定 —— 审批人看到的依据必须是可复现的。
    """

    thread_id: str = Field(min_length=1, description="图执行线程 id,恢复时按它 resume")
    indicator: str = Field(min_length=1, description="审批对象(IP / 域名 / Hash)")
    risk_level: RiskLevel = Field(description="风险等级(规则引擎产出)")
    score: int = Field(ge=0, le=100, description="风险分数(规则引擎产出)")
    summary: str = Field(min_length=1, description="规则生成的一句话结论")
    actions: list[ResponseAction] = Field(
        min_length=1, description="需要人工审批的动作子集(非空)"
    )
    policy_reasons: list[str] = Field(
        default_factory=list, description="策略判定依据(为什么需要审批)"
    )
    requested_at: datetime = Field(default_factory=utc_now, description="请求时间(UTC)")


class ApprovalDecision(BaseModel):
    """人工审批的决定。写入审计,并随 checkpoint 持久化。"""

    status: ApprovalStatus = Field(description="决定(受限枚举)")
    operator: str = Field(min_length=1, description="审批人标识")
    interrupt_id: str | None = Field(
        default=None, description="对应的 interrupt id(框架分配,用于审计关联)"
    )
    reason: str | None = Field(default=None, description="审批意见(可选)")
    decided_at: datetime = Field(default_factory=utc_now, description="决定时间(UTC)")


class TriageOutcome(BaseModel):
    """triage() / resume() 的对外聚合结果。"""

    thread_id: str = Field(min_length=1, description="图执行线程 id")
    status: TriageStatus = Field(
        description="completed=已走完;pending_approval=等待人工审批"
    )
    answer: str = Field(default="", description="LLM 的叙事性回答(等待审批时可能为空)")
    plan: ResponsePlan | None = Field(
        default=None, description="结构化处置计划(规则引擎产出)"
    )
    approval_request: ApprovalRequest | None = Field(
        default=None, description="status=pending_approval 时的待审请求"
    )
    approval: ApprovalDecision | None = Field(
        default=None, description="已产生的人工决定(未审批时为 None)"
    )

    @model_validator(mode="after")
    def _validate_pending(self) -> "TriageOutcome":
        """单向蕴含:pending_approval ⇒ approval_request 存在(Phase 8.4 D2)。

        刻意**不**禁止 completed + approval_request:审批走完之后,
        approval_request 是"批的是什么"的留痕依据,approval 是"谁批的",
        两者都要留在响应里。旧的双向校验会把 resume 的终态判为非法。
        """
        if self.status == "pending_approval" and self.approval_request is None:
            raise ValueError("status=pending_approval 时必须给出 approval_request")
        return self
