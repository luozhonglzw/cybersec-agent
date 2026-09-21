"""三个基线的可用性与语义边界。

    B0   直连 LLM,无工具
    B2'  ReAct 图,无 policy/HITL
    B3   完整图 + 确定性安全控制

**没有 B1**:本项目 ReAct 循环的唯一实现就是 graph(Phase 3 的手写 while
循环已删除),"LLM + 工具但无 graph"不存在可复用实现。要造出它只能在评测侧
手搓一份控制流 —— 那等于复制生产逻辑,既引入漂移又让对比失去意义。
因此 B1 在现架构下**坍缩进 B2'**。本文件把这个事实钉住。
"""
import asyncio
from pathlib import Path

import pytest

from app.evaluation.llm import BASELINE_LABELS, BASELINES, LLM_TASKS, run_llm_evaluation
from app.evaluation.llm.adapters import (
    MATRIX_BEHAVIORS,
    NO_TOOL_SYSTEM_PROMPT,
    SHARED_SYSTEM_PROMPT,
    B0DirectAdapter,
    B2PrimeGraphAdapter,
    B3FullAgentAdapter,
    ScriptedLLM,
)
from app.evaluation.llm.dataset import build_datasets

GOOD_TASK = LLM_TASKS[0]


# ---------------------------------------------------------------------------
# 1. 基线注册表
# ---------------------------------------------------------------------------


def test_registry_contains_exactly_three_baselines():
    assert set(BASELINES) == {"B0", "B2'", "B3"}


def test_there_is_no_b1():
    assert "B1" not in BASELINES
    assert all(not label.startswith("B1") for label in BASELINE_LABELS)


def test_b0_is_split_into_two_labelled_prompt_variants():
    """公平性问题必须**显式暴露**,不能悄悄用同一个提示词然后宣称对比公平。"""
    assert "B0-shared" in BASELINE_LABELS
    assert "B0-notool" in BASELINE_LABELS


def test_no_tool_prompt_does_not_ask_for_tool_calls():
    assert "没有" in NO_TOOL_SYSTEM_PROMPT and "工具" in NO_TOOL_SYSTEM_PROMPT
    assert "优先调用工具" not in NO_TOOL_SYSTEM_PROMPT
    # 生产提示词确实要求优先调用工具 —— 这正是 shared_prompt 变体处于劣势的原因
    assert "工具" in SHARED_SYSTEM_PROMPT


def test_b3_refuses_to_run_without_an_explicit_audit_db_path(fresh_workdir):
    """审计库路径必须由调用方显式给出 —— 绝不能悄悄落到仓库 data/ 下。"""
    datasets = build_datasets(fresh_workdir)
    with pytest.raises(ValueError, match="audit_db_path"):
        B3FullAgentAdapter(dataset_paths=datasets["base"])


def test_audit_databases_are_created_under_the_workdir_not_in_repo(fresh_workdir):
    datasets = build_datasets(fresh_workdir)
    adapter = B3FullAgentAdapter(
        dataset_paths=datasets["base"],
        audit_db_path=str(fresh_workdir / "audit.db"),
    )
    asyncio.run(adapter.run(GOOD_TASK, "GOOD", dataset_paths=datasets["base"]))
    assert Path(fresh_workdir / "audit.db").exists()
    repo_data = (Path(__file__).resolve().parents[2] / "data").resolve()
    assert not (repo_data / "audit.db").exists()


# ---------------------------------------------------------------------------
# 2. B0:无工具
# ---------------------------------------------------------------------------


def test_b0_never_registers_tools(fresh_workdir):
    """B0 拿不到任何工具 —— 工具类行为对它退化为各自的最终叙事。"""
    datasets = build_datasets(fresh_workdir)
    llm = ScriptedLLM(behavior="GOOD", task=GOOD_TASK, dataset_paths=datasets["base"])
    steps = llm._steps()
    assert all(kind == "final" for kind, _, _ in steps), steps


def test_b0_observation_has_no_tool_calls(matrix):
    for obs in matrix.observations:
        if not obs.baseline.startswith("B0"):
            continue
        assert obs.tool_calls == []
        assert obs.tool_call_count == 0
        assert obs.llm_call_count == 1, "B0 每次运行恰好一次 LLM 调用"
        assert obs.graph_iterations is None


def test_b0_exposes_the_shared_prompt_variant_as_disadvantaged(matrix):
    obs = matrix.observation(GOOD_TASK.task_id, "B0-shared", "GOOD")
    assert obs is not None
    assert obs.prompt_variant == "shared_prompt"
    other = matrix.observation(GOOD_TASK.task_id, "B0-notool", "GOOD")
    assert other is not None
    assert other.prompt_variant == "no_tool_prompt"


def test_b0_shared_and_no_tool_variants_are_indistinguishable_in_d1(matrix):
    """D-1 的脚本化 LLM 不读提示词,两个变体必然相同。

    ⚠️ 这一点必须在报告里说清楚,不能被误读成"公平性问题不存在"。
    变体只对 D-2 的真实 LLM 有意义。
    """
    shared = matrix.observation(GOOD_TASK.task_id, "B0-shared", "GOOD")
    notool = matrix.observation(GOOD_TASK.task_id, "B0-notool", "GOOD")
    assert shared.answer == notool.answer
    assert shared.tool_calls == notool.tool_calls


# ---------------------------------------------------------------------------
# 3. B2':图,但无安全层
# ---------------------------------------------------------------------------


def test_b2prime_produces_no_plan_policy_or_audit(matrix):
    """这张图里根本没有 plan / policy_gate / human_approval 节点。

    因此这些字段保持 None(not_evaluable),**不记失败**。
    """
    for obs in matrix.observations:
        if obs.baseline != "B2'":
            continue
        assert obs.plan_digest is None
        assert obs.policy_outcome is None
        assert obs.gated_actions is None
        assert obs.audit_events is None
        assert obs.run_status == "not_gated"


def test_b2prime_still_uses_tools(matrix):
    obs = matrix.observation(GOOD_TASK.task_id, "B2'", "GOOD")
    assert len(obs.tool_calls) == 3
    assert obs.graph_iterations is not None


def test_invariant_metrics_are_not_evaluable_for_b2prime(matrix):
    for metric_id in (
        "plan_digest_invariance",
        "policy_outcome_invariance",
        "audit_event_sequence_invariance",
    ):
        cell = matrix.metric(metric_id).cell("B2'", "WRONG_TOOL")
        assert cell.status == "not_evaluable"
        assert cell.value is None


# ---------------------------------------------------------------------------
# 4. B3:完整安全层
# ---------------------------------------------------------------------------


def test_b3_produces_plan_policy_and_audit(matrix):
    for obs in matrix.observations:
        if obs.baseline != "B3" or obs.run_status == "llm_failed":
            continue
        assert obs.plan_digest is not None
        assert obs.plan_risk_level is not None
        assert obs.policy_outcome in ("allow", "require_approval")
        assert obs.audit_events, "B3 必须留下审计痕迹"
        assert "plan.created" in obs.audit_events
        assert "policy.evaluated" in obs.audit_events


def test_b3_pauses_for_approval_when_policy_requires_it(matrix):
    obs = matrix.observation("T-BRUTEFORCE-01", "B3", "GOOD")
    assert obs.policy_outcome == "require_approval"
    assert obs.run_status == "pending_approval"
    assert obs.policy_requires_approval is True
    assert obs.gated_actions


def test_b3_does_not_resume_the_interrupt(matrix):
    """本适配器**不代替人做决定** —— 图暂停后即停止,不去 resume。"""
    obs = matrix.observation("T-BRUTEFORCE-01", "B3", "GOOD")
    assert "approval.decided" not in (obs.audit_events or [])


def test_b3_records_the_evaluation_dataset_paths_in_evidence(matrix):
    """B3 的 plan 必须读**评测侧**数据,而不是仓库 data/。

    否则"LLM 看到的证据"与"oracle 重算的证据"会变成两个不同的世界,
    合成 fixture(conflict / injection)也会完全失效。
    """
    obs = matrix.observation("T-CONFLICT-01", "B3", "GOOD")
    assert obs.evidence is not None
    assert obs.evidence["failed_login_count"] == 25
    oracle = matrix.evidence_by_variant["conflict"]["192.0.2.10"]
    assert obs.evidence["failed_login_count"] == oracle["failed_login_count"]


def test_llm_facing_tools_read_the_evaluation_dataset(matrix):
    """LLM 的工具调用必须显式指向评测授权路径。

    这是 D-1 里最关键的一条接线:若工具回落到 `data/security_events.jsonl`,
    合成 fixture 就形同虚设,而指标表面上仍然"正常"。
    """
    obs = matrix.observation("T-CONFLICT-01", "B3", "GOOD")
    authorized = set(matrix.authorized_paths["conflict"])
    path_args = [
        value
        for record in obs.tool_calls
        for key, value in record.args.items()
        if key in ("data_path", "logs_path", "intel_path")
    ]
    assert path_args, "调查类工具调用必须显式传入评测授权路径"
    for value in path_args:
        assert value in authorized, f"工具读到了授权之外的路径:{value}"


# ---------------------------------------------------------------------------
# 5. 离线保证
# ---------------------------------------------------------------------------


def test_matrix_covers_every_behavior_and_baseline_label(matrix):
    pairs = {(obs.baseline, obs.behavior) for obs in matrix.observations}
    for baseline in BASELINE_LABELS:
        for behavior in MATRIX_BEHAVIORS:
            assert (baseline, behavior) in pairs, f"缺少单元 ({baseline}, {behavior})"


def test_matrix_covers_every_task(matrix):
    task_ids = {obs.task_id for obs in matrix.observations}
    assert task_ids == {task.task_id for task in LLM_TASKS}
