"""RiskAssessment / RiskEvidence 的 Pydantic 校验测试。"""
import pytest
from pydantic import ValidationError

from app.schemas.risk import RiskAssessment, RiskEvidence


def _evidence(**overrides) -> dict:
    base = {
        "indicator": "203.0.113.66",
        "log_event_count": 30,
        "failed_login_count": 30,
        "threat_intel_found": True,
        "threat_intel_malicious": True,
        "threat_intel_tags": ["ssh-brute-force"],
        "threat_intel_severity": "critical",
    }
    base.update(overrides)
    return base


def test_valid_assessment_constructs():
    """正常构造:全部字段通过校验。"""
    assessment = RiskAssessment(
        indicator="203.0.113.66",
        risk_level="critical",
        score=85,
        confidence=70,
        reasons=["威胁情报标记恶意"],
        evidence=RiskEvidence(**_evidence()),
    )
    assert assessment.risk_level == "critical"
    assert assessment.evidence.failed_login_count == 30


def test_none_risk_level_is_valid():
    """risk_level 枚举包含 "none"(无风险/证据不足)。"""
    assessment = RiskAssessment(
        indicator="10.9.9.9", risk_level="none", score=0, confidence=20,
        reasons=[], evidence=RiskEvidence(indicator="10.9.9.9"),
    )
    assert assessment.risk_level == "none"


def test_invalid_risk_level_rejected():
    """非法 risk_level → 校验失败。"""
    with pytest.raises(ValidationError, match="risk_level"):
        RiskAssessment(
            indicator="x", risk_level="extreme", score=0, confidence=0,
            reasons=[], evidence=RiskEvidence(indicator="x"),
        )


@pytest.mark.parametrize("field", ["score", "confidence"])
@pytest.mark.parametrize("bad", [-1, 101])
def test_out_of_range_scores_rejected(field, bad):
    """score / confidence 超出 0-100 → 校验失败。"""
    kwargs = dict(indicator="x", risk_level="none", confidence=50,
                  reasons=[], evidence=RiskEvidence(indicator="x"))
    if field == "confidence":
        kwargs["confidence"] = bad
        kwargs["score"] = 50
    else:
        kwargs["score"] = bad
    with pytest.raises(ValidationError, match=field):
        RiskAssessment(**kwargs)


def test_score_boundary_values_accepted():
    """边界值 0 与 100 合法。"""
    base = dict(indicator="x", risk_level="none", reasons=[],
                evidence=RiskEvidence(indicator="x"))
    assert RiskAssessment(**base, score=0, confidence=0).score == 0
    assert RiskAssessment(**base, score=100, confidence=100).score == 100


def test_evidence_negative_counts_rejected():
    """证据中的事件计数为负 → 校验失败。"""
    with pytest.raises(ValidationError, match="failed_login_count"):
        RiskEvidence(**_evidence(failed_login_count=-1))


def test_evidence_defaults():
    """RiskEvidence 缺省字段:计数 0、未命中、malicious 为 None。"""
    evidence = RiskEvidence(indicator="10.9.9.9")
    assert evidence.log_event_count == 0
    assert evidence.threat_intel_found is False
    assert evidence.threat_intel_malicious is None
    assert evidence.threat_intel_tags == []


def test_empty_indicator_rejected():
    """空 indicator → 校验失败。"""
    with pytest.raises(ValidationError, match="indicator"):
        RiskEvidence(indicator="")
    with pytest.raises(ValidationError, match="indicator"):
        RiskAssessment(
            indicator="", risk_level="none", score=0, confidence=0,
            reasons=[], evidence=RiskEvidence(indicator="x"),
        )


def test_json_roundtrip():
    """model_dump_json → model_validate_json 往返无损(含内嵌 evidence)。"""
    assessment = RiskAssessment(
        indicator="203.0.113.66", risk_level="critical", score=85,
        confidence=70, reasons=["a"],
        evidence=RiskEvidence(**_evidence()),
    )
    restored = RiskAssessment.model_validate_json(assessment.model_dump_json())
    assert restored == assessment
