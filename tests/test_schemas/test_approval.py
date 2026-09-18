"""审批(HITL)schema contract 测试(Phase 8.1)。

覆盖 ApprovalRequest / ApprovalDecision / TriageOutcome:
必填性、枚举封闭性、状态一致性不变量、时间字段 tz-aware、JSON 往返
(interrupt 载荷必须可序列化,否则图无法暂停)。

全部 hermetic:只构造内存对象,不读写任何数据文件。
"""
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.approval import (
    ApprovalDecision,
    ApprovalRequest,
    TriageOutcome,
)
from app.schemas.response import ResponseAction, ResponsePlan
from app.schemas.risk import RiskAssessment, RiskEvidence

INDICATOR = "203.0.113.66"


def _action(**overrides) -> ResponseAction:
    base = dict(
        action_type="block_ip",
        priority="high",
        target=INDICATOR,
        rationale="该指标具备恶意特征,建议在网络边界封禁",
        requires_approval=True,
        reversible=True,
    )
    base.update(overrides)
    return ResponseAction(**base)


def _plan() -> ResponsePlan:
    evidence = RiskEvidence(
        indicator=INDICATOR,
        log_event_count=30,
        failed_login_count=30,
        threat_intel_found=True,
        threat_intel_malicious=True,
        threat_intel_tags=["ssh-brute-force"],
        threat_intel_severity="critical",
    )
    assessment = RiskAssessment(
        indicator=INDICATOR,
        risk_level="critical",
        score=85,
        confidence=70,
        reasons=["威胁情报标记该指标为恶意(tags=ssh-brute-force)"],
        evidence=evidence,
    )
    return ResponsePlan(
        indicator=INDICATOR,
        risk_level="critical",
        summary=f"指标 {INDICATOR} 风险等级 critical(分数 85),建议 2 项动作,其中 2 项需人工审批。",
        actions=[
            _action(),
            _action(action_type="reset_credentials", reversible=False),
        ],
        assessment=assessment,
    )


def _request(**overrides) -> ApprovalRequest:
    base = dict(
        thread_id="thread-abc",
        indicator=INDICATOR,
        risk_level="critical",
        score=85,
        summary="指标 203.0.113.66 风险等级 critical(分数 85)",
        actions=[_action()],
        policy_reasons=["动作按属性需要人工审批"],
    )
    base.update(overrides)
    return ApprovalRequest(**base)


# ---------- ApprovalRequest ----------

def test_approval_request_valid():
    r = _request()
    assert r.thread_id == "thread-abc"
    assert r.actions[0].action_type == "block_ip"
    assert r.policy_reasons


def test_approval_request_defaults_requested_at_to_utc():
    r = _request()
    assert r.requested_at.tzinfo is not None
    assert r.requested_at.utcoffset() == timezone.utc.utcoffset(None)


def test_approval_request_requires_actions_non_empty():
    """没有待审动作的审批请求是无意义的,必须在契约层拒绝。"""
    with pytest.raises(ValidationError):
        _request(actions=[])


def test_approval_request_rejects_empty_indicator():
    with pytest.raises(ValidationError):
        _request(indicator="")


def test_approval_request_rejects_empty_thread_id():
    with pytest.raises(ValidationError):
        _request(thread_id="")


def test_approval_request_rejects_score_out_of_range():
    with pytest.raises(ValidationError):
        _request(score=101)


def test_approval_request_rejects_invalid_risk_level():
    with pytest.raises(ValidationError):
        _request(risk_level="severe")


def test_approval_request_json_round_trip():
    """interrupt 载荷必须可 JSON 序列化,否则图无法暂停。"""
    r = _request()
    again = ApprovalRequest.model_validate_json(r.model_dump_json())
    assert again == r
    assert json.loads(r.model_dump_json())["thread_id"] == "thread-abc"


# ---------- ApprovalDecision ----------

def test_approval_decision_valid():
    d = ApprovalDecision(status="approved", operator="analyst-1")
    assert d.status == "approved"
    assert d.interrupt_id is None
    assert d.reason is None


def test_approval_decision_defaults_decided_at_to_utc():
    d = ApprovalDecision(status="denied", operator="analyst-1")
    assert d.decided_at.tzinfo is not None


def test_approval_decision_rejects_invalid_status():
    """只有 approved / denied 两种决定,没有中间态。"""
    with pytest.raises(ValidationError):
        ApprovalDecision(status="maybe", operator="analyst-1")


def test_approval_decision_rejects_empty_operator():
    with pytest.raises(ValidationError):
        ApprovalDecision(status="approved", operator="")


def test_approval_decision_json_round_trip():
    d = ApprovalDecision(
        status="denied", operator="analyst-1",
        interrupt_id="int-001", reason="资产重要性待确认",
    )
    again = ApprovalDecision.model_validate_json(d.model_dump_json())
    assert again == d


def test_approval_decision_has_no_execution_field():
    """安全约束:审批决定不得携带"是否已执行"的字段(Phase 8 不执行动作)。"""
    assert "execution_status" not in ApprovalDecision.model_fields
    assert "executed" not in ApprovalDecision.model_fields


# ---------- TriageOutcome ----------

def test_triage_outcome_completed_without_request():
    o = TriageOutcome(
        thread_id="thread-abc", status="completed",
        answer="该 IP 风险为 low,建议持续观察。",
    )
    assert o.approval_request is None
    assert o.answer


def test_triage_outcome_answer_defaults_to_empty():
    o = TriageOutcome(thread_id="thread-abc", status="completed")
    assert o.answer == ""


def test_triage_outcome_pending_requires_request():
    o = TriageOutcome(
        thread_id="thread-abc", status="pending_approval",
        plan=_plan(), approval_request=_request(),
    )
    assert o.status == "pending_approval"
    assert o.approval_request is not None


def test_triage_outcome_pending_without_request_rejected():
    with pytest.raises(ValidationError, match="approval_request"):
        TriageOutcome(thread_id="thread-abc", status="pending_approval")


def test_triage_outcome_completed_with_request_allowed():
    """Phase 8.4 D2:校验从"当且仅当"放宽为单向蕴含。

    completed + approval_request 是 /resume 的**唯一合法终态**:
    审批已走完,但"批的是什么"必须留在响应里(留痕依据)。
    旧的双向校验会把它判为非法,导致 resume 无法表达结果。
    """
    o = TriageOutcome(
        thread_id="thread-abc", status="completed",
        approval_request=_request(),
        approval=ApprovalDecision(
            status="approved", operator="analyst-1", interrupt_id="int-001",
        ),
    )
    assert o.status == "completed"
    assert o.approval_request is not None
    assert o.approval is not None


def test_triage_outcome_completed_without_request_still_allowed():
    """反向的"什么都没有"仍然是合法终态(未触发审批的 allow 路径)。"""
    o = TriageOutcome(thread_id="thread-abc", status="completed")
    assert o.approval_request is None
    assert o.approval is None


def test_triage_outcome_rejects_invalid_status():
    with pytest.raises(ValidationError):
        TriageOutcome(thread_id="thread-abc", status="running")


def test_triage_outcome_with_decision_json_round_trip():
    o = TriageOutcome(
        thread_id="thread-abc", status="completed", answer="已批准并记录。",
        plan=_plan(),
        approval=ApprovalDecision(
            status="approved", operator="analyst-1", interrupt_id="int-001",
        ),
    )
    again = TriageOutcome.model_validate_json(o.model_dump_json())
    assert again == o
    assert again.approval.status == "approved"
    assert again.plan.indicator == INDICATOR


def test_triage_outcome_has_no_execution_field():
    """安全约束:TriageOutcome 不得暴露"已执行"语义。"""
    assert "execution_status" not in TriageOutcome.model_fields
    assert "executed_actions" not in TriageOutcome.model_fields


def test_approval_request_uses_rule_engine_fields_only():
    """审批人看到的依据必须来自规则引擎:等级/分数/动作,无 LLM 文本字段。"""
    fields = set(ApprovalRequest.model_fields)
    assert {"risk_level", "score", "summary", "actions"} <= fields
    assert not {"llm_reasoning", "llm_analysis", "model_output"} & fields
