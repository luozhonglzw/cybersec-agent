"""Incident 数据模型 —— Phase 8.1。

Phase 7 曾把 incident 持久化延后(见 docs/architecture.md §11 的实现偏离说明),
Phase 8 与 HITL / audit lifecycle 一起落地。这里只定义最小契约:

    一次"分析 + 规划"的沉淀:对象是谁、风险多高、结论是什么、计划是什么。

risk_level / score 与 plan.assessment 有冗余 —— 这是**有意为之**,与
ResponsePlan.risk_level 同款处理:冗余字段便于列表查询与快速读取,并约定
必须与 assessment 一致(由 plan_response 单点生成,不存在第二个写入方)。

刻意不含的字段:
- status:审批状态在 action_requests 上(一次 incident 可能有多轮审批),
  在 incident 上再放一个 status 会产生两个真相源;
- execution_status:Phase 8 不执行任何真实动作(见 approval.py 的安全约束)。
"""
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.response import ResponsePlan
from app.schemas.risk import RiskLevel


class Incident(BaseModel):
    """一次安全事件的沉淀记录。"""

    id: str = Field(min_length=1, description="incident id")
    created_at: datetime = Field(description="创建时间(UTC)")
    indicator: str = Field(min_length=1, description="事件对象(IP / 域名 / Hash)")
    risk_level: RiskLevel = Field(description="风险等级(规则引擎产出)")
    score: int = Field(ge=0, le=100, description="风险分数(规则引擎产出)")
    summary: str = Field(min_length=1, description="一句话结论(规则生成)")
    plan: ResponsePlan = Field(
        description="结构化处置计划(内嵌 assessment 与证据链,可复现)"
    )
