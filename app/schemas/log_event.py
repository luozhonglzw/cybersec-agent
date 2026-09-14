"""结构化安全日志数据模型 —— Agent 要处理的"安全世界"的最小数据契约。

设计原则:
- event_type / severity / status 用 Literal 受限枚举:脏数据在 Pydantic 校验时
  直接报错,而不是带着错误数据流进 Agent / LLM;
- IP 保持 str(而非 IPv4Address 类型):JSONL 序列化、Phase 3 工具参数、
  未来 API/LLM 数据交换统一是字符串,用 validator 保证格式合法即可;
- 可选字段用 None 而不是空串/省略,语义清晰("没有"不等于"空")。

该模型被三处消费:scripts/seed_logs.py(生成)、Phase 3 的
query_security_logs 工具(查询/过滤)、未来 API 层(响应体)。
"""
from datetime import datetime
from ipaddress import ip_address
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

# 事件类型:登录(成功/失败)、权限提升、Web 请求、防火墙放行。
# 注意:"SSH 暴力破解"不是 event_type,而是大量 login_failed 事件构成的**模式**——
# 模式识别是 Agent 的工作,数据层只记录原子事实。
EventType = Literal[
    "login_success",
    "login_failed",
    "privilege_escalation",
    "web_request",
    "firewall_allow",
]

Severity = Literal["info", "low", "medium", "high", "critical"]

Status = Literal["success", "failed", "denied"]

Port = Annotated[int, Field(ge=0, le=65535)]


class LogEvent(BaseModel):
    """一条结构化安全日志事件。

    必填:timestamp / event_type / source / severity / message
    可选:其余字段按事件类型取值,不适用的为 None。
    """

    timestamp: datetime = Field(description="事件发生时间(UTC)")
    event_type: EventType = Field(description="事件类别(受限枚举)")
    source: str = Field(min_length=1, description="日志来源系统,如 sshd / web_app / sudo / firewall")
    source_ip: str | None = Field(default=None, description="源 IP(IPv4 字符串)")
    destination_ip: str | None = Field(default=None, description="目标 IP(IPv4 字符串)")
    source_port: Port | None = Field(default=None, description="源端口")
    destination_port: Port | None = Field(default=None, description="目标端口")
    username: str | None = Field(default=None, description="涉及的账号")
    action: str | None = Field(default=None, description="具体动作,如 sudo / GET /admin")
    status: Status | None = Field(default=None, description="结果:success / failed / denied")
    severity: Severity = Field(description="严重级别(与 Phase 7 风险分级同一套词汇表)")
    message: str = Field(min_length=1, description="类 syslog 原文,保留人可读语义")

    @field_validator("source_ip", "destination_ip")
    @classmethod
    def _validate_ipv4(cls, value: str | None) -> str | None:
        """可选字段为 None 直接放行;非 None 必须是合法 IPv4(拒绝 IPv6/主机名)。"""
        if value is None:
            return value
        try:
            parsed = ip_address(value)
        except ValueError:
            raise ValueError(f"非法 IPv4 地址: {value!r}") from None
        if parsed.version != 4:
            raise ValueError(f"只接受 IPv4 地址: {value!r}")
        return value
