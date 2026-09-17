"""策略判定数据模型 —— Policy Engine 的结构化输出契约(Phase 8.1)。

设计原则(与 LogEvent / RiskAssessment / ResponsePlan 一致):
- outcome 用 Literal 受限枚举:策略结论是封闭的三值(allow / deny /
  require_approval),不存在"部分允许"这类模糊态;
- requires_approval 与 outcome 的**一致性由模型自身保证**:二者若不一致,
  审批门会读到自相矛盾的结论 —— 在安全场景下这是不可接受的输入,宁可
  在校验边界直接拒绝,也不要让矛盾状态流到下游;
- gated_actions 记录**触发该结论的动作类型**,使审计能回答"为什么需要审批 /
  为什么被拒绝",而不是只留一个结论字符串;
- policy_version 必填:策略规则会演进,审计记录必须能定位当时生效的规则版本。

安全约束(勿改):
    PolicyDecision 是**规则引擎的输出**,不是 LLM 的输出。判定依据只能是
    ResponsePlan 这类结构化、由规则生成的对象;LLM 的自然语言永远不得成为
    策略判定输入(见 app/security/policy.py 的入参约束)。
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.response import ActionType

PolicyOutcome = Literal["allow", "deny", "require_approval"]


class PolicyDecision(BaseModel):
    """一次策略判定的结果。规则引擎生成,审批门与审计消费。"""

    outcome: PolicyOutcome = Field(description="策略结论(受限枚举)")
    requires_approval: bool = Field(
        description="是否需要人工审批;必须与 outcome 一致(模型自校验)"
    )
    gated_actions: list[ActionType] = Field(
        default_factory=list,
        description="触发该结论的动作类型;outcome=allow 时必为空",
    )
    reasons: list[str] = Field(
        default_factory=list, description="判定依据,逐条可读(审计用)"
    )
    policy_version: str = Field(
        min_length=1, description="策略规则版本,便于审计定位当时生效的规则"
    )

    @model_validator(mode="after")
    def _validate_consistency(self) -> "PolicyDecision":
        expected = self.outcome == "require_approval"
        if self.requires_approval != expected:
            raise ValueError(
                "requires_approval 必须与 outcome 一致:"
                f"outcome={self.outcome} 时 requires_approval 应为 {expected}"
            )
        if self.outcome != "allow" and not self.reasons:
            raise ValueError(f"outcome={self.outcome} 必须给出 reasons(审计要求)")
        if self.outcome == "allow" and self.gated_actions:
            raise ValueError("outcome=allow 时 gated_actions 必须为空")
        return self
