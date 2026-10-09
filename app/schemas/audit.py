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

已知局限(必须文档化,不得掩盖):
    `actor` 自 Phase **v0.3.0-A3-3** 起取自**已认证主体**:`/resume` 的
    HTTP 边界把 `principal.subject` 交给服务层,调用方自述的 `operator`
    被忽略 —— 因此**客户端无法伪造 actor**。
    但它**仍不具备不可否认性**:没有签名、没有同时留存"声明值"与
    "已验证值"的对照,数据库持有者依然可以改写历史。可信归属的完整形态
    (同时记录 `verified_subject` 与 `claimed_operator`)与审计签名
    在后续的 A6 与更后面的阶段。本阶段**不得**声称已实现不可否认性。
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# 审计词汇表。刻意包含**失败事件**(plan.failed):失败若不留痕,
# "有多少次判定失败、为什么失败"就无从回答(F6「每次工具调用、每个审批
# 决策可查」)。失败事件与成功事件同等重要 —— 只记成功的审计是幸存者偏差。
AuditEvent = Literal[
    "plan.created",
    "plan.failed",
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
