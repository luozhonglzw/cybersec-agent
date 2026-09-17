"""威胁情报 IOC 数据模型 —— 与 LogEvent 同风格的结构化数据契约。

设计原则:
- indicator_type / severity 用 Literal 受限枚举:脏数据在 Pydantic 校验时
  直接报错,而不是带着错误数据流进 Agent / LLM;
- indicator 保持 str:IOC(IP/域名/Hash)在 JSONL、工具参数、LLM 上下文中
  统一是字符串,格式合法性由查询时的 Exact Match 保证;
- confidence 用 0-100 整数:情报源通常给出百分制置信度,
  Field(ge=0, le=100) 在校验层拒绝越界值;
- 不引入 Embedding / 相似度:IOC 是唯一标识符,查询语义是精确匹配。

该模型被两处消费:scripts/seed_threat_intel.py(生成)、
Phase 5 的 query_threat_intel 工具(查询)。
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

IndicatorType = Literal["ip", "domain", "hash"]

Severity = Literal["info", "low", "medium", "high", "critical"]


class ThreatIntelRecord(BaseModel):
    """一条威胁情报 IOC 记录。"""

    indicator: str = Field(min_length=1, description="IOC 值:IP / 域名 / Hash 原文")
    indicator_type: IndicatorType = Field(description="IOC 类别(受限枚举)")
    malicious: bool = Field(description="是否恶意(false 表示已知可信/白名单)")
    confidence: int = Field(ge=0, le=100, description="情报置信度(0-100)")
    severity: Severity = Field(description="严重级别(与 LogEvent 同一套词汇表)")
    tags: list[str] = Field(default_factory=list, description="攻击模式标签,如 ssh-brute-force")
    source: str = Field(min_length=1, description="情报来源,如 internal-analysis / seed")
    first_seen: datetime = Field(description="首次发现时间(UTC)")
    last_seen: datetime = Field(description="最近活动时间(UTC)")
    description: str = Field(min_length=1, description="人可读的情报描述")
