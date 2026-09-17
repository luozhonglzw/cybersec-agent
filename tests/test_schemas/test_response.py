"""Phase 7:ResponsePlan / ResponseAction schema 校验测试(完全离线)。

覆盖 Literal 受限枚举、必填字段、min_length 约束、JSON roundtrip,
以及嵌套 assessment 证据链的完整性。
"""
import json

import pytest
from pydantic import ValidationError

from app.schemas.response import ResponseAction, ResponsePlan
from app.schemas.risk import RiskAssessment, RiskEvidence


def _evidence(**overrides) -> RiskEvidence:
    base = {
        "indicator": "203.0.113.66",
        "log_event_count": 30,
        "failed_login_count": 25,
    }
    base.update(overrides)
    return RiskEvidence(**base)


def _assessment(**overrides) -> RiskAssessment:
    base = {
        "indicator": "203.0.113.66",
        "risk_level": "high",
        "score": 70,
        "confidence": 70,
        "reasons": ["失败登录 25 次(≥20,疑似暴力破解)"],
        "evidence": _evidence(),
    }
    base.update(overrides)
    return RiskAssessment(**base)


def _action(**overrides) -> ResponseAction:
    base = {
        "action_type": "block_ip",
        "priority": "high",
        "target": "203.0.113.66",
        "rationale": "该指标具备恶意特征,建议在网络边界封禁",
        "requires_approval": True,
        "reversible": True,
    }
    base.update(overrides)
    return ResponseAction(**base)


def _plan(**overrides) -> ResponsePlan:
    base = {
        "indicator": "203.0.113.66",
        "risk_level": "high",
        "summary": "风险 high,建议 1 项动作",
        "actions": [_action()],
        "assessment": _assessment(),
    }
    base.update(overrides)
    return ResponsePlan(**base)


# ---------- 正常构造 ----------

def test_valid_plan_constructs():
    """合法 ResponsePlan 可构造,字段可读。"""
    plan = _plan()
    assert plan.indicator == "203.0.113.66"
    assert plan.risk_level == "high"
    assert plan.actions[0].action_type == "block_ip"
    assert plan.actions[0].requires_approval is True


# ---------- Literal 受限枚举 ----------

def test_invalid_action_type_rejected():
    """枚举外的动作类型被拒绝 —— LLM 不能发明计划外动作。"""
    with pytest.raises(ValidationError):
        _action(action_type="run_shell")


def test_invalid_priority_rejected():
    """非法 priority 被拒绝。"""
    with pytest.raises(ValidationError):
        _action(priority="urgent")


def test_invalid_risk_level_rejected():
    """非法 risk_level 被拒绝(与 RiskLevel 同一套词汇表)。"""
    with pytest.raises(ValidationError):
        _plan(risk_level="severe")


# ---------- min_length 约束 ----------

def test_empty_actions_rejected():
    """actions 至少一条 —— 无风险时是单条 no_action,而不是空列表。"""
    with pytest.raises(ValidationError):
        _plan(actions=[])


def test_empty_rationale_rejected():
    """rationale 不能为空 —— 每条动作必须可解释。"""
    with pytest.raises(ValidationError):
        _action(rationale="")


def test_empty_indicator_rejected():
    """indicator 不能为空。"""
    with pytest.raises(ValidationError):
        _plan(indicator="")


# ---------- 必填字段 ----------

def test_approval_flags_are_required():
    """requires_approval / reversible 是必填 —— 不能靠默认值蒙混。"""
    payload = _action().model_dump()
    del payload["requires_approval"]
    with pytest.raises(ValidationError):
        ResponseAction(**payload)

    payload = _action().model_dump()
    del payload["reversible"]
    with pytest.raises(ValidationError):
        ResponseAction(**payload)


# ---------- 序列化与嵌套证据链 ----------

def test_json_roundtrip():
    """model_dump(mode='json') 可再次校验 —— 工具 JSON 契约可用。"""
    plan = _plan()
    dumped = json.loads(json.dumps(plan.model_dump(mode="json")))
    restored = ResponsePlan.model_validate(dumped)
    assert restored.model_dump(mode="json") == plan.model_dump(mode="json")


def test_nested_assessment_evidence_preserved():
    """嵌套 assessment 及其 evidence 完整保留 —— plan → assessment → evidence 全链可追溯。"""
    plan = _plan(assessment=_assessment(evidence=_evidence(
        threat_intel_found=True,
        threat_intel_malicious=True,
    )))
    payload = plan.model_dump(mode="json")
    evidence = payload["assessment"]["evidence"]

    assert payload["assessment"]["risk_level"] == "high"
    assert evidence["indicator"] == "203.0.113.66"
    assert evidence["failed_login_count"] == 25
    assert evidence["threat_intel_malicious"] is True
