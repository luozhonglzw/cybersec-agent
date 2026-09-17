"""审计数据模型 —— Phase 8.1。

安全 Agent 必须能回答三个问题,字段直接对应:
    为什么封禁?  → plan_digest → incidents.plan → assessment.evidence
    谁批准?      → actor + interrupt_id
    依据是什么?  → event + reason + detail(自包含取证负载)

设计原则:
- event 用 Literal 受限枚举:审计词汇表封闭,不允许自由发挥;
- append-only:一条记录写入后**不再修改**(状态变化 = 追加新记录)。
  可变的"历史"不叫审计;
- plan_digest 用 sha256 十六进制(64 字符),用 pattern 在校验层拒绝格式错误
  —— 摘要写错会让"计划是否被改动"的比对失效;
- detail 是**自包含**的取证负载:审计记录必须能独立回答"依据是什么",
  不依赖其他表是否还存在。

已知局限(Phase 8,必须文档化,不得掩盖):
    本阶段没有身份认证 —— actor 只是调用方自称的字符串,**不具备不可否认性**。
    认证 / 签名留到 Phase 10(或后续引入最小 API key)。
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AuditEvent = Literal[
    "plan.created",
    "policy.evaluated",
    "approval.requested",
    "approval.decided",
    "approval.timeout",
]

# sha256 十六进制摘要:64 位小写十六进制
_SHA256_HEX = r"^[0-9a-f]{64}$"

# 非人工主体(规则引擎 / 图节点自己产生的记录)
SYSTEM_ACTOR = "system"


class AuditRecord(BaseModel):
    """一条审计流水。append-only,写入后不再修改。"""

    id: str = Field(min_length=1, description="记录 id")
    ts: datetime = Field(description="发生时间(UTC)")
    actor: str = Field(min_length=1, description="主体:system 或审批人标识")
    event: AuditEvent = Field(description="事件类型(受限枚举)")
    incident_id: str | None = Field(default=None, description="关联的 incident")
    thread_id: str | None = Field(default=None, description="图执行线程 id")
    interrupt_id: str | None = Field(default=None, description="关联的 interrupt id")
    outcome: str | None = Field(
        default=None, description="结论:allow / deny / approved / denied"
    )
    reason: str | None = Field(default=None, description="判定依据摘要")
    plan_digest: str | None = Field(
        default=None,
        pattern=_SHA256_HEX,
        description="plan 的规范化 sha256(小写十六进制),用于比对计划是否被改动",
    )
    detail: dict = Field(
        default_factory=dict, description="自包含取证负载(审计需能独立回答依据)"
    )
