"""PolicyDecision schema contract 测试(Phase 8.1)。

只验证契约:字段必填性、枚举封闭性、模型自校验的一致性不变量。
策略规则本身(R1-R4)的验证在 tests/test_security/test_policy.py。

全部 hermetic:只构造内存对象,不读写任何数据文件。
"""
import json

import pytest
from pydantic import ValidationError

from app.schemas.policy import PolicyDecision


def _payload(**overrides) -> dict:
    """合法 allow 判定的最小载荷,按需覆盖字段。"""
    base = dict(
        outcome="allow",
        requires_approval=False,
        gated_actions=[],
        reasons=["无动作需要人工审批,策略放行"],
        policy_version="test-v1",
    )
    base.update(overrides)
    return base


# ---------- 合法构造 ----------

def test_allow_decision_valid():
    d = PolicyDecision(**_payload())
    assert d.outcome == "allow"
    assert d.requires_approval is False
    assert d.gated_actions == []


def test_require_approval_decision_valid():
    d = PolicyDecision(**_payload(
        outcome="require_approval",
        requires_approval=True,
        gated_actions=["block_ip", "reset_credentials"],
        reasons=["动作按属性需要人工审批"],
    ))
    assert d.requires_approval is True
    assert d.gated_actions == ["block_ip", "reset_credentials"]


def test_deny_decision_valid():
    d = PolicyDecision(**_payload(
        outcome="deny",
        requires_approval=False,
        gated_actions=["isolate_host"],
        reasons=["目标属于受保护资产"],
    ))
    assert d.outcome == "deny"
    assert d.gated_actions == ["isolate_host"]


def test_defaults_are_empty_collections():
    """gated_actions / reasons 未提供时是空列表,不是 None。"""
    d = PolicyDecision(
        outcome="allow", requires_approval=False,
        reasons=["r"], policy_version="v1",
    )
    assert d.gated_actions == []


# ---------- 枚举封闭性 ----------

def test_invalid_outcome_rejected():
    with pytest.raises(ValidationError):
        PolicyDecision(**_payload(outcome="maybe"))


def test_invalid_action_type_rejected():
    """gated_actions 受 ActionType 枚举约束,LLM 无法塞入计划外动作。"""
    with pytest.raises(ValidationError):
        PolicyDecision(**_payload(
            outcome="require_approval", requires_approval=True,
            gated_actions=["run_shell"], reasons=["x"],
        ))


# ---------- 一致性不变量(模型自校验)----------

def test_requires_approval_true_but_outcome_not_require_rejected():
    with pytest.raises(ValidationError, match="requires_approval"):
        PolicyDecision(**_payload(
            outcome="require_approval", requires_approval=False, reasons=["x"],
        ))


def test_requires_approval_false_but_outcome_require_rejected():
    with pytest.raises(ValidationError, match="requires_approval"):
        PolicyDecision(**_payload(outcome="allow", requires_approval=True))


def test_non_allow_outcome_requires_reasons():
    """deny / require_approval 必须给出理由(审计要求)。"""
    with pytest.raises(ValidationError, match="reasons"):
        PolicyDecision(**_payload(
            outcome="deny", requires_approval=False, reasons=[],
        ))


def test_allow_outcome_must_not_gate_actions():
    with pytest.raises(ValidationError, match="gated_actions"):
        PolicyDecision(**_payload(gated_actions=["block_ip"]))


# ---------- policy_version ----------

def test_policy_version_required():
    payload = _payload()
    payload.pop("policy_version")
    with pytest.raises(ValidationError):
        PolicyDecision(**payload)


def test_policy_version_empty_rejected():
    with pytest.raises(ValidationError):
        PolicyDecision(**_payload(policy_version=""))


# ---------- 序列化(审计与 interrupt 载荷需要)----------

def test_json_round_trip():
    d = PolicyDecision(**_payload(
        outcome="require_approval", requires_approval=True,
        gated_actions=["block_ip"], reasons=["r1", "r2"],
    ))
    again = PolicyDecision.model_validate_json(d.model_dump_json())
    assert again == d


def test_model_dump_json_is_plain_json():
    d = PolicyDecision(**_payload())
    assert json.loads(d.model_dump_json())["outcome"] == "allow"
