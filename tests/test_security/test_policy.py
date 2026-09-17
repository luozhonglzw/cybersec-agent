"""Policy Engine 规则测试(Phase 8.1)。

覆盖:
- R2/R3/R4 三条规则的分支穷举与优先级;
- 边界值(置信度恰好等于地板);
- **确定性**(同输入两次调用结果完全一致);
- **单输入契约与纯度约束**(签名只有 plan,不接受配置或 LLM 输出);
- 与真实规则引擎(analyze_risk → plan_response)联动,验证 R2 在真实计划上生效。

R1(受保护资产 → deny)在 8.1 未实现(输入是配置,不属于单输入契约),
因此 deny 当前不可达 —— 由 test_deny_is_not_emitted_in_phase_8_1 作为绊线锁定。

全部 hermetic:只构造内存对象与调用纯函数,不读写任何数据文件。
"""
import inspect
from typing import get_args

import pytest

from app.schemas.response import ActionType, ResponseAction, ResponsePlan
from app.schemas.risk import RiskAssessment, RiskEvidence
from app.security.policy import (
    CONFIDENCE_FLOOR,
    DESTRUCTIVE_ACTIONS,
    POLICY_VERSION,
    evaluate_policy,
)
from app.tools.response_planner import plan_response
from app.tools.risk_analyzer import analyze_risk

INDICATOR = "203.0.113.66"


def _action(
    action_type: ActionType = "monitor",
    *,
    requires_approval: bool | None = None,
    target: str = INDICATOR,
) -> ResponseAction:
    """构造动作;requires_approval 未显式给出时按动作类型推断(贴近真实属性表)。"""
    if requires_approval is None:
        requires_approval = action_type in DESTRUCTIVE_ACTIONS
    return ResponseAction(
        action_type=action_type,
        priority="low",
        target=target,
        rationale="测试依据",
        requires_approval=requires_approval,
        reversible=action_type != "reset_credentials",
    )


def _plan(
    actions: list[ResponseAction],
    *,
    confidence: int = 70,
    risk_level: str = "high",
    score: int = 70,
) -> ResponsePlan:
    evidence = RiskEvidence(indicator=INDICATOR)
    assessment = RiskAssessment(
        indicator=INDICATOR,
        risk_level=risk_level,
        score=score,
        confidence=confidence,
        reasons=["测试依据"],
        evidence=evidence,
    )
    return ResponsePlan(
        indicator=INDICATOR,
        risk_level=risk_level,
        summary="测试计划",
        actions=list(actions),
        assessment=assessment,
    )


# ---------- R2:动作自带审批标记 → require_approval ----------

def test_r2_action_with_requires_approval_is_gated():
    plan = _plan([_action("block_ip", requires_approval=True)])
    d = evaluate_policy(plan)
    assert d.outcome == "require_approval"
    assert d.requires_approval is True
    assert d.gated_actions == ["block_ip"]
    assert d.reasons


def test_r2_only_gated_actions_are_listed():
    plan = _plan([
        _action("monitor", requires_approval=False),
        _action("block_ip", requires_approval=True),
        _action("collect_evidence", requires_approval=False),
    ])
    assert evaluate_policy(plan).gated_actions == ["block_ip"]


def test_r2_deduplicates_action_types():
    """同一动作类型作用于不同目标时只记一次(确定性顺序,非 set 乱序)。"""
    plan = _plan([
        _action("block_ip", requires_approval=True, target="1.1.1.1"),
        _action("block_ip", requires_approval=True, target="2.2.2.2"),
    ])
    assert evaluate_policy(plan).gated_actions == ["block_ip"]


def test_r2_preserves_action_order():
    plan = _plan([
        _action("reset_credentials", requires_approval=True),
        _action("block_ip", requires_approval=True),
    ])
    assert evaluate_policy(plan).gated_actions == ["reset_credentials", "block_ip"]


# ---------- R3:低置信度 + 破坏性动作 → 兜底升级 ----------

def test_r3_low_confidence_upgrades_destructive_action():
    """置信度低于地板 + 破坏性动作未被 R2 覆盖 → 强制审批。"""
    plan = _plan(
        [_action("block_ip", requires_approval=False)],
        confidence=CONFIDENCE_FLOOR - 1,
    )
    d = evaluate_policy(plan)
    assert d.outcome == "require_approval"
    assert d.gated_actions == ["block_ip"]
    assert any("置信度" in r for r in d.reasons)


def test_r3_boundary_at_exactly_floor_is_not_low():
    """边界:置信度恰好等于地板 → 不算低置信度 → 放行。"""
    plan = _plan(
        [_action("block_ip", requires_approval=False)],
        confidence=CONFIDENCE_FLOOR,
    )
    assert evaluate_policy(plan).outcome == "allow"


def test_r3_does_not_gate_non_destructive_action():
    """低置信度 + 仅非破坏性动作(monitor)→ 仍放行,避免过度审批。"""
    plan = _plan([_action("monitor", requires_approval=False)], confidence=0)
    assert evaluate_policy(plan).outcome == "allow"


def test_r3_does_not_duplicate_r2_gated_action():
    """已被 R2 覆盖的动作不再重复计入,也不重复追加理由。"""
    plan = _plan([_action("block_ip", requires_approval=True)], confidence=0)
    d = evaluate_policy(plan)
    assert d.gated_actions == ["block_ip"]
    assert not any("置信度" in r for r in d.reasons)


def test_r3_covers_every_destructive_action_type():
    for action_type in sorted(DESTRUCTIVE_ACTIONS):
        plan = _plan([_action(action_type, requires_approval=False)], confidence=0)
        d = evaluate_policy(plan)
        assert d.outcome == "require_approval", action_type
        assert d.gated_actions == [action_type]


def test_destructive_actions_are_subset_of_action_type_enum():
    """防拼写错误:DESTRUCTIVE_ACTIONS 必须都在封闭词汇表内。"""
    assert DESTRUCTIVE_ACTIONS <= set(get_args(ActionType))


# ---------- R4:放行 ----------

def test_r4_plain_plan_is_allowed():
    plan = _plan([_action("monitor", requires_approval=False)])
    d = evaluate_policy(plan)
    assert d.outcome == "allow"
    assert d.requires_approval is False
    assert d.gated_actions == []
    assert d.reasons


def test_r4_no_action_plan_is_allowed():
    plan = _plan([_action("no_action", requires_approval=False)], risk_level="none", score=0)
    assert evaluate_policy(plan).outcome == "allow"


# ---------- 确定性与版本 ----------

def test_deterministic_same_input_same_output():
    plan = _plan([_action("block_ip"), _action("reset_credentials")])
    assert evaluate_policy(plan) == evaluate_policy(plan)


def test_deterministic_across_equivalent_instances():
    """两个内容相同的 plan(不同实例)→ 判定结果相等。"""
    first = _plan([_action("block_ip", requires_approval=True)])
    second = _plan([_action("block_ip", requires_approval=True)])
    assert evaluate_policy(first) == evaluate_policy(second)


def test_policy_version_always_set():
    plan = _plan([_action("monitor", requires_approval=False)])
    assert evaluate_policy(plan).policy_version == POLICY_VERSION


def test_outcome_always_one_of_three():
    for actions, confidence in (
        ([_action("monitor", requires_approval=False)], 70),
        ([_action("block_ip", requires_approval=True)], 70),
        ([_action("block_ip", requires_approval=False)], 0),
    ):
        d = evaluate_policy(_plan(actions, confidence=confidence))
        assert d.outcome in ("allow", "deny", "require_approval")
        assert d.reasons


# ---------- 单输入契约与纯度约束(C5)----------

def test_signature_is_single_input():
    """安全约束:唯一入参是 ResponsePlan —— 不接受配置输入,更不接受 LLM 输出。

    资产白名单之类的配置必须走独立 PolicyConfig / Settings 注入,
    不得混进本函数的签名(否则"Policy Engine 只消费 ResponsePlan"的
    单输入契约就被破坏了)。
    """
    params = inspect.signature(evaluate_policy).parameters
    assert list(params) == ["plan"]
    plan_param = params["plan"]
    assert plan_param.kind in (
        plan_param.POSITIONAL_ONLY, plan_param.POSITIONAL_OR_KEYWORD,
    )
    assert plan_param.annotation is ResponsePlan
    assert plan_param.default is plan_param.empty


def test_no_module_level_mutable_state():
    """纯函数:不得有模块级可变状态(否则两次调用可能不等)。"""
    import app.security.policy as policy_module
    for name in ("POLICY_VERSION", "CONFIDENCE_FLOOR", "DESTRUCTIVE_ACTIONS"):
        assert hasattr(policy_module, name)
    assert isinstance(policy_module.DESTRUCTIVE_ACTIONS, frozenset)
    assert isinstance(policy_module.CONFIDENCE_FLOOR, int)


def test_no_config_input_constants_exposed():
    """8.1 不引入配置常量:PROTECTED_TARGETS 之类的死分支不应存在。"""
    import app.security.policy as policy_module
    assert not hasattr(policy_module, "PROTECTED_TARGETS")


def test_deny_is_not_emitted_in_phase_8_1():
    """绊线测试:deny 当前不可达(R1 依赖配置,尚未实现)。

    一旦 R1 通过 PolicyConfig / Settings 落地,本测试会失败 —— 届时请连同
    该测试一起有意更新,而不是默默放宽断言。
    """
    outcomes = set()
    for actions, confidence in (
        ([_action("no_action", requires_approval=False)], 0),
        ([_action("monitor", requires_approval=False)], 0),
        ([_action("block_ip", requires_approval=False)], 0),
        ([_action("isolate_host", requires_approval=True)], 70),
        ([_action("reset_credentials", requires_approval=True)], 70),
    ):
        outcomes.add(evaluate_policy(_plan(actions, confidence=confidence)).outcome)
    assert outcomes == {"allow", "require_approval"}
    assert "deny" not in outcomes


# ---------- 与真实规则引擎联动(仍为纯计算,hermetic)----------

def test_real_critical_plan_requires_approval():
    """恶意情报 + 爆破日志 → critical 计划含需审批动作 → require_approval。"""
    evidence = RiskEvidence(
        indicator=INDICATOR,
        log_event_count=30,
        failed_login_count=30,
        threat_intel_found=True,
        threat_intel_malicious=True,
        threat_intel_tags=["ssh-brute-force"],
        threat_intel_severity="critical",
    )
    plan = plan_response(analyze_risk(evidence))
    assert plan.risk_level == "critical"

    d = evaluate_policy(plan)
    assert d.outcome == "require_approval"
    assert {"block_ip", "reset_credentials"} <= set(d.gated_actions)


def test_real_medium_malicious_intel_requires_approval():
    """仅情报恶意(40 分 → medium)→ 规划器修正项补上 block_ip → 仍需审批。

    这条同时证明 plan_response 的修正项是策略门的有效输入,不是冗余代码。
    """
    evidence = RiskEvidence(
        indicator=INDICATOR,
        threat_intel_found=True,
        threat_intel_malicious=True,
        threat_intel_tags=["c2"],
        threat_intel_severity="medium",
    )
    plan = plan_response(analyze_risk(evidence))
    assert plan.risk_level == "medium"

    d = evaluate_policy(plan)
    assert d.outcome == "require_approval"
    assert "block_ip" in d.gated_actions


def test_real_brute_force_only_requires_approval():
    """仅爆破日志(30 分 → medium)→ 修正项补上 reset_credentials → 需审批。"""
    evidence = RiskEvidence(
        indicator=INDICATOR, log_event_count=30, failed_login_count=30,
    )
    plan = plan_response(analyze_risk(evidence))
    assert plan.risk_level == "medium"

    d = evaluate_policy(plan)
    assert d.outcome == "require_approval"
    assert "reset_credentials" in d.gated_actions


def test_real_low_risk_plan_is_allowed():
    """失败登录 5-19(low)→ monitor → 放行。"""
    evidence = RiskEvidence(
        indicator=INDICATOR, log_event_count=10, failed_login_count=10,
    )
    plan = plan_response(analyze_risk(evidence))
    assert plan.risk_level == "low"
    assert evaluate_policy(plan).outcome == "allow"


def test_real_no_evidence_plan_is_allowed():
    """无任何证据 → no_action → 放行。"""
    plan = plan_response(analyze_risk(RiskEvidence(indicator=INDICATOR)))
    assert plan.actions[0].action_type == "no_action"
    assert evaluate_policy(plan).outcome == "allow"


def test_real_trusted_intel_downgrade_still_gates_credential_reset():
    """情报标记可信 → 等级被强制降为 low,但爆破修正项仍要求改密 → 仍需审批。

    这条锁定一个**有意的**设计取舍:可信标记只降等级,抹不掉"该指标上确实
    发生了 20+ 次失败登录"这一事实。误报抑制 ≠ 放行凭据风险,因此策略门
    仍然要求人工审批 —— 而不是静默放行。
    """
    evidence = RiskEvidence(
        indicator=INDICATOR,
        log_event_count=30,
        failed_login_count=30,
        threat_intel_found=True,
        threat_intel_malicious=False,
        threat_intel_tags=["trusted-scan-engine"],
        threat_intel_severity="info",
    )
    plan = plan_response(analyze_risk(evidence))
    assert plan.risk_level in ("none", "low")

    d = evaluate_policy(plan)
    assert d.outcome == "require_approval"
    assert "reset_credentials" in d.gated_actions
