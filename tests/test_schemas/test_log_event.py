"""LogEvent 模型校验测试:保证脏数据在边界处被拒绝。"""
import pytest
from pydantic import ValidationError

from app.schemas.log_event import LogEvent

VALID = {
    "timestamp": "2026-09-10T08:00:00Z",
    "event_type": "login_failed",
    "source": "sshd",
    "source_ip": "203.0.113.66",
    "destination_ip": "10.0.1.20",
    "source_port": 54321,
    "destination_port": 22,
    "username": "admin",
    "status": "failed",
    "severity": "medium",
    "message": "Failed password for admin from 203.0.113.66 port 22 ssh2",
}


def test_valid_log_event():
    """全部字段合法:构造成功,字段值保留。"""
    event = LogEvent(**VALID)
    assert event.event_type == "login_failed"
    assert event.source_ip == "203.0.113.66"
    assert event.severity == "medium"


def test_optional_fields_default_to_none():
    """可选字段缺省时是 None,不是空串。"""
    event = LogEvent(
        timestamp=VALID["timestamp"], event_type="login_success",
        source="sshd", severity="info", message="ok",
    )
    assert event.source_ip is None
    assert event.username is None
    assert event.status is None


@pytest.mark.parametrize("bad_ip", ["999.1.1.1", "not-an-ip", "example.com", "::1"])
def test_invalid_ipv4_rejected(bad_ip):
    """非法 IP(超界 / 乱字符串 / 主机名 / IPv6)→ ValidationError。"""
    with pytest.raises(ValidationError):
        LogEvent(**{**VALID, "source_ip": bad_ip})


def test_invalid_event_type_rejected():
    """event_type 是受限枚举,自由文本(如 'brute_force')被拒。"""
    with pytest.raises(ValidationError):
        LogEvent(**{**VALID, "event_type": "brute_force"})


def test_invalid_severity_rejected():
    with pytest.raises(ValidationError):
        LogEvent(**{**VALID, "severity": "extreme"})


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        LogEvent(**{**VALID, "status": "maybe"})


@pytest.mark.parametrize("bad_port", [-1, 65536])
def test_invalid_port_rejected(bad_port):
    """端口越界(0~65535 之外)→ ValidationError。"""
    with pytest.raises(ValidationError):
        LogEvent(**{**VALID, "source_port": bad_port})
