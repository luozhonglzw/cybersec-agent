"""处置规划数据模型 —— Incident Response Planning 的结构化输出契约(Phase 7)。

设计原则(与 LogEvent / ThreatIntelRecord / RiskAssessment 一致):
- action_type / priority 用 Literal 受限枚举:动作词汇表是封闭的,LLM 不能
  自由发挥出 "run_shell" 这类计划外动作;
- requires_approval / reversible 由规则引擎判定,是 Phase 8 策略引擎的输入
  —— 是否需要人工审批不经过 LLM;
- assessment 内嵌:处置计划必须可审计、可复现,plan → assessment → evidence
  构成完整证据链;Phase 9 evaluation 直接用它做 golden set 比对;
- plan_response(assessment) 是纯函数:同样的评估永远得出同样的计划。

分工(Hybrid):规则引擎输出结构化动作与审批标记,LLM 负责基于它向用户解释
与汇报 —— 动作、等级与审批标记都不经过 LLM。
"""
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.risk import RiskAssessment, RiskLevel

# 动作词汇表。刻意保持封闭:新增动作 = 显式修改本枚举 + 属性表,
# 不允许 LLM 生成枚举外的动作。
ActionType = Literal[
    "no_action",          # 无需处置
    "monitor",            # 持续观察
    "collect_evidence",   # 补充取证 / 留证
    "block_ip",           # 封禁源 IP
    "reset_credentials",  # 强制该账号改密
    "isolate_host",       # 隔离主机
    "escalate",           # 升级人工介入
]

ActionPriority = Literal["low", "medium", "high", "critical"]


class ResponseAction(BaseModel):
    """一条处置建议。规则引擎生成,LLM 只负责解释。"""

    action_type: ActionType = Field(description="动作类型(受限枚举)")
    priority: ActionPriority = Field(description="动作固有紧急度,与 risk_level 解耦")
    target: str = Field(min_length=1, description="作用对象:IP / 主机 / 账号")
    rationale: str = Field(min_length=1, description="判定依据,必须引用证据")
    requires_approval: bool = Field(description="是否需人工审批(Phase 8 HITL 输入)")
    reversible: bool = Field(description="是否可回滚(审批决策依据)")


class ResponsePlan(BaseModel):
    """一次结构化处置规划。"""

    indicator: str = Field(min_length=1, description="规划对象(IP / 域名 / Hash)")
    risk_level: RiskLevel = Field(
        description="冗余自 assessment,便于读取;必须与 assessment.risk_level 一致"
    )
    summary: str = Field(min_length=1, description="规则生成的一句话结论(不经过 LLM)")
    actions: list[ResponseAction] = Field(
        min_length=1, description="至少一条;无风险时为单条 no_action"
    )
    assessment: RiskAssessment = Field(
        description="内嵌风险评估,保证计划与风险判定不漂移"
    )
