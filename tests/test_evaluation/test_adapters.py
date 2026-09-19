"""Phase 9.2-A 适配器测试。

三个基线各自的**结构性**差异必须被锁死,否则消融实验就失去意义:

    B1 有计划 / 有策略结论 / **无**门 / **无**审计
    B2 **无**计划 / **无**策略结论 / **无**门 / **无**审计
    B3 有计划 / 有策略结论 / **有**门 / **有**审计

以及两条安全底线:
    * B3 必须显式接收 audit_db_path —— 审计库绝不允许落到仓库 data/ 下;
    * 任何适配器都不得执行真实处置动作(Phase 8 只做审批,不接执行器)。
"""
import asyncio

import pytest

from app.evaluation.adapters import (
    FAKE_NARRATIVE,
    FullAgentAdapter,
    GraphNoGateAdapter,
    RuleOnlyAdapter,
)
from app.evaluation.golden import GOLDEN_SET


def _case(case_id: str):
    return GOLDEN_SET.by_id(case_id)


def _adapter(cls, seed_dataset, tmp_path, *, with_audit: bool = False):
    kwargs = {
        "logs_path": str(seed_dataset["logs"]),
        "intel_path": str(seed_dataset["intel"]),
    }
    if with_audit:
        kwargs["audit_db_path"] = str(tmp_path / "audit.db")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# B1:确定性内核直连
# ---------------------------------------------------------------------------


async def test_b1_produces_plan_and_policy_but_no_gate_and_no_audit(seed_dataset, tmp_path):
    adapter = _adapter(RuleOnlyAdapter, seed_dataset, tmp_path)
    observation = await adapter.run(_case("BF-01"))

    assert observation.evidence is not None
    assert observation.plan_actions is not None
    assert observation.policy_outcome == "require_approval"
    assert observation.run_status == "not_gated"
    assert observation.audit_events is None
    assert observation.audit_plan_digests is None


async def test_b1_plan_matches_its_own_policy_decision(seed_dataset, tmp_path):
    """B1 的策略结论是**被算出来的**,但没有任何东西消费它。"""
    adapter = _adapter(RuleOnlyAdapter, seed_dataset, tmp_path)
    observation = await adapter.run(_case("TRUST-01"))
    assert observation.policy_outcome == "allow"
    assert observation.gated_actions == []
    assert observation.plan_actions == ["no_action"]


# ---------------------------------------------------------------------------
# B2:无门图
# ---------------------------------------------------------------------------


async def test_b2_produces_nothing_but_a_narrative(seed_dataset, tmp_path):
    adapter = _adapter(GraphNoGateAdapter, seed_dataset, tmp_path)
    observation = await adapter.run(_case("BF-01"))

    assert observation.evidence is None
    assert observation.plan_actions is None
    assert observation.policy_outcome is None
    assert observation.audit_events is None
    assert observation.run_status == "not_gated"
    assert observation.answer == FAKE_NARRATIVE


# ---------------------------------------------------------------------------
# B3:完整 HITL 图
# ---------------------------------------------------------------------------


async def test_b3_engages_the_gate_on_a_case_requiring_approval(seed_dataset, tmp_path):
    adapter = _adapter(FullAgentAdapter, seed_dataset, tmp_path, with_audit=True)
    observation = await adapter.run(_case("BF-01"))

    assert observation.run_status == "pending_approval"
    assert observation.policy_outcome == "require_approval"
    assert observation.plan_actions is not None
    assert observation.audit_events is not None
    assert "plan.created" in observation.audit_events
    assert "policy.evaluated" in observation.audit_events
    assert "approval.requested" in observation.audit_events
    # 9.2-A 刻意不 resume:人工决定不由评测代劳
    assert "approval.decided" not in observation.audit_events


async def test_b3_completes_without_pausing_on_a_benign_case(seed_dataset, tmp_path):
    adapter = _adapter(FullAgentAdapter, seed_dataset, tmp_path, with_audit=True)
    observation = await adapter.run(_case("NOISE-01"))

    assert observation.run_status == "completed"
    assert observation.policy_outcome == "allow"
    assert observation.audit_events is not None
    assert "approval.requested" not in observation.audit_events


async def test_b3_audit_chain_carries_a_single_plan_digest(seed_dataset, tmp_path):
    adapter = _adapter(FullAgentAdapter, seed_dataset, tmp_path, with_audit=True)
    observation = await adapter.run(_case("PG-01"))
    assert observation.audit_plan_digests is not None
    assert set(observation.audit_plan_digests.values()) == {observation.plan_digest}


def test_b3_refuses_to_run_without_an_explicit_audit_db_path(seed_dataset):
    with pytest.raises(ValueError, match="audit_db_path"):
        FullAgentAdapter(
            logs_path=str(seed_dataset["logs"]),
            intel_path=str(seed_dataset["intel"]),
        )


async def test_b3_audit_database_is_created_under_the_given_path(seed_dataset, tmp_path):
    db_path = tmp_path / "nested" / "audit.db"
    adapter = FullAgentAdapter(
        logs_path=str(seed_dataset["logs"]),
        intel_path=str(seed_dataset["intel"]),
        audit_db_path=str(db_path),
    )
    await adapter.run(_case("BF-01"))
    assert db_path.exists()


# ---------------------------------------------------------------------------
# 安全底线
# ---------------------------------------------------------------------------


def test_no_adapter_exposes_an_execution_surface():
    """Phase 8 只做审批,不接执行器。适配器不得暴露任何"已执行"字段。"""
    from app.evaluation.adapters import Observation

    banned = {"executed", "execution_status", "executed_actions", "applied"}
    assert banned & set(Observation.model_fields) == set()


async def test_b1_and_b3_are_deterministic_across_repeated_runs(seed_dataset, tmp_path):
    """同一输入重复执行,观测必须逐字段一致(叙事文本除外)。"""
    for cls, with_audit, tag in (
        (RuleOnlyAdapter, False, "b1"),
        (FullAgentAdapter, True, "b3"),
    ):
        first = _adapter(cls, seed_dataset, tmp_path / f"{tag}-1", with_audit=with_audit)
        second = _adapter(cls, seed_dataset, tmp_path / f"{tag}-2", with_audit=with_audit)
        left = await first.run(_case("BF-01"))
        right = await second.run(_case("BF-01"))
        left_payload = left.model_dump(mode="json")
        right_payload = right.model_dump(mode="json")
        left_payload.pop("answer")
        right_payload.pop("answer")
        assert left_payload == right_payload


async def test_b2_is_deterministic(seed_dataset, tmp_path):
    adapter = _adapter(GraphNoGateAdapter, seed_dataset, tmp_path)
    first = await adapter.run(_case("BF-01"))
    second = await adapter.run(_case("BF-01"))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
