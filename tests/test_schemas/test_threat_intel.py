"""ThreatIntelRecord 的 Pydantic 校验测试。"""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.threat_intel import ThreatIntelRecord


def _valid(**overrides) -> dict:
    """一条合法记录的基准字段,按需覆盖。"""
    base = {
        "indicator": "203.0.113.66",
        "indicator_type": "ip",
        "malicious": True,
        "confidence": 95,
        "severity": "critical",
        "tags": ["ssh-brute-force", "scanner"],
        "source": "internal-analysis",
        "first_seen": datetime(2026, 9, 1, tzinfo=timezone.utc),
        "last_seen": datetime(2026, 9, 2, tzinfo=timezone.utc),
        "description": "SSH 暴力破解攻击源",
    }
    base.update(overrides)
    return base


def test_valid_record_constructs():
    """正常构造:全部字段通过校验,字段值原样保留。"""
    record = ThreatIntelRecord(**_valid())
    assert record.indicator == "203.0.113.66"
    assert record.indicator_type == "ip"
    assert record.malicious is True
    assert record.confidence == 95
    assert record.severity == "critical"
    assert record.tags == ["ssh-brute-force", "scanner"]


def test_tags_default_empty():
    """tags 缺省时为空列表而不是必填报错。"""
    record = ThreatIntelRecord(**_valid(tags=[]))
    assert record.tags == []


def test_invalid_severity_rejected():
    """severity 不在枚举内 → 校验失败。"""
    with pytest.raises(ValidationError, match="severity"):
        ThreatIntelRecord(**_valid(severity="catastrophic"))


@pytest.mark.parametrize("bad_confidence", [-1, 101, 999])
def test_invalid_confidence_rejected(bad_confidence):
    """confidence 超出 0-100 → 校验失败。"""
    with pytest.raises(ValidationError, match="confidence"):
        ThreatIntelRecord(**_valid(confidence=bad_confidence))


def test_confidence_boundary_values_accepted():
    """confidence 边界值 0 和 100 合法。"""
    assert ThreatIntelRecord(**_valid(confidence=0)).confidence == 0
    assert ThreatIntelRecord(**_valid(confidence=100)).confidence == 100


def test_invalid_indicator_type_rejected():
    """indicator_type 不在 ip/domain/hash 内 → 校验失败。"""
    with pytest.raises(ValidationError, match="indicator_type"):
        ThreatIntelRecord(**_valid(indicator_type="url"))
    with pytest.raises(ValidationError, match="indicator_type"):
        ThreatIntelRecord(**_valid(indicator_type="email"))


def test_empty_indicator_rejected():
    """indicator 为空串 → 校验失败。"""
    with pytest.raises(ValidationError, match="indicator"):
        ThreatIntelRecord(**_valid(indicator=""))


def test_missing_required_fields_rejected():
    """缺少必填字段(first_seen)→ 校验失败。"""
    data = _valid()
    del data["first_seen"]
    with pytest.raises(ValidationError, match="first_seen"):
        ThreatIntelRecord(**data)


def test_json_roundtrip():
    """model_dump_json → model_validate_json 往返无损。"""
    record = ThreatIntelRecord(**_valid())
    restored = ThreatIntelRecord.model_validate_json(record.model_dump_json())
    assert restored == record
