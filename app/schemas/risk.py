"""风险评估数据模型 —— Rule-based Risk Analyzer 的结构化输出契约。

设计原则(与 LogEvent / ThreatIntelRecord 一致):
- risk_level 用 Literal 受限枚举,含 "none"(评估对象无风险/证据不足),
  与既有 severity 词汇表对齐;
- evidence 内嵌:风险判定必须可审计、可复现 —— 判定依据是结构化证据
  快照,不是一段散文;Phase 9 evaluation 用它做 golden set 比对;
- analyze_risk(evidence) 是纯函数:同样的证据永远得出同样的评估。

分工(Hybrid):Rule-based Tool 输出本模型的结构化风险,
LLM 负责基于它做解释与汇报 —— 数字与等级不经过 LLM。
"""
from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["none", "low", "medium", "high", "critical"]


class RiskEvidence(BaseModel):
    """风险判定的最小证据集:由查询工具采集,喂给纯规则函数。

    字段是"原始事实"而非"结论":失败登录数、情报标签等,
    不含任何打分结果。
    """
    indicator: str = Field(min_length=1, description="评估对象(IP/域名/Hash)")
    log_event_count: int = Field(ge=0, default=0, description="日志中相关事件总数")
    failed_login_count: int = Field(ge=0, default=0, description="其中失败登录数(爆破特征)")
    threat_intel_found: bool = Field(default=False, description="情报库是否命中该 IOC")
    threat_intel_malicious: bool | None = Field(
        default=None, description="情报是否标记恶意(未命中为 None)"
    )
    threat_intel_tags: list[str] = Field(default_factory=list, description="情报标签")
    threat_intel_severity: str | None = Field(default=None, description="情报严重级别")


class RiskAssessment(BaseModel):
    """一次结构化风险评估的结果。"""
    indicator: str = Field(min_length=1, description="评估对象")
    risk_level: RiskLevel = Field(description="风险等级(含 none)")
    score: int = Field(ge=0, le=100, description="规则引擎累计分数(0-100)")
    confidence: int = Field(ge=0, le=100, description="评估置信度(证据充分度)")
    reasons: list[str] = Field(default_factory=list, description="判定依据,逐条引用证据")
    evidence: RiskEvidence = Field(description="判定所依据的证据快照(可复现)")
