"""Phase 9.2-A 指标语义测试。

本文件守护的是**报告纪律**,不是数字大小:

    * 分母为 0 必须记 NOT_EVALUABLE,不能记 0;
    * 五类指标必须齐全,且不得出现任何"总分";
    * D 类必须自带 oracle_class == "D"(让人一眼看出它不是正确性证据);
    * E 类必须带原因;
    * 消融实验的结论必须能从 per_item 里被独立算出来。
"""
from app.evaluation.metrics import NOT_EVALUABLE_REASONS, MetricResult

CATEGORIES = {"A", "B", "C", "D", "E"}


def _per_adapter(metric: MetricResult, adapter_id: str, key: str) -> list[dict]:
    return [
        item for item in metric.per_item
        if item.get("adapter_id") == adapter_id
    ]


# ---------------------------------------------------------------------------
# 报告纪律
# ---------------------------------------------------------------------------


def test_every_metric_declares_a_category_in_abcde(evaluation):
    for metric in evaluation.metrics:
        assert metric.category in CATEGORIES, metric.metric_id


def test_all_five_categories_are_present(evaluation):
    present = {metric.category for metric in evaluation.metrics}
    assert present == CATEGORIES


def test_no_combined_agent_score_exists(evaluation):
    """不得出现单一"Agent 总分" —— 加权平均会把 D 类自洽性算进正确性。"""
    banned = ("overall", "total", "aggregate", "combined", "agent_score", "grade")
    for metric in evaluation.metrics:
        assert not any(word in metric.metric_id for word in banned), metric.metric_id
    assert len({metric.metric_id for metric in evaluation.metrics}) == len(evaluation.metrics)


def test_not_evaluable_metrics_never_carry_a_numeric_zero(evaluation):
    """NOT_EVALUABLE 不是 0 —— value 必须是 None,且必须给出原因。"""
    for metric in evaluation.metrics:
        if metric.status == "not_evaluable":
            assert metric.value is None, metric.metric_id
            assert metric.numerator is None, metric.metric_id
            assert metric.not_evaluable_reason, metric.metric_id


def test_e_metrics_are_all_not_evaluable_with_reasons(evaluation):
    e_metrics = evaluation.metrics_by_category("E")
    assert {m.metric_id for m in e_metrics} == set(NOT_EVALUABLE_REASONS)
    for metric in e_metrics:
        assert metric.status == "not_evaluable"
        assert metric.value is None
        assert metric.oracle_class == "E"
        assert metric.not_evaluable_reason


def test_task_success_is_declared_not_evaluable(evaluation):
    """Review 修正 1:确定性升级不等于任务成功。"""
    metric = evaluation.metric("task_success")
    assert metric.status == "not_evaluable"
    assert metric.category == "E"


def test_escalation_metric_is_not_named_task_success(evaluation):
    """旧名 task_success_rate 被禁止 —— 它会诱导把安全升级读成任务成功。"""
    ids = {metric.metric_id for metric in evaluation.metrics}
    assert "task_success_rate" not in ids
    assert "safety_escalation_success_rate" in ids


# ---------------------------------------------------------------------------
# D 类必须自我标识
# ---------------------------------------------------------------------------


def test_consistency_metrics_are_labelled_as_d_class(evaluation):
    expected = {
        "plan_internal_consistency",
        "policy_outcome_consistency",
        "gate_decision_agreement",
        "plan_digest_chain_integrity",
    }
    for metric in evaluation.metrics_by_category("D"):
        assert metric.metric_id in expected
        assert metric.oracle_class == "D"
        assert "D 类" in metric.definition


def test_consistency_metrics_are_green_on_the_unmutated_codebase(evaluation):
    """基线(未变异)下 D 类全绿 —— 这是后面反同义反复实验的对照前提。"""
    for metric in evaluation.metrics_by_category("D"):
        assert metric.status == "ok", metric.metric_id
        assert metric.value == 1.0, metric.metric_id


# ---------------------------------------------------------------------------
# 消融实验:结论必须能从 per_item 独立复算
# ---------------------------------------------------------------------------


def test_baselines_without_a_gate_never_satisfy_the_escalation_property(evaluation):
    """B1 / B2 在"必须人工审批"的用例上必然为 0 —— 结构性事实,不是缺陷。"""
    metric = evaluation.metric("safety_escalation_success_rate")
    assert metric.status == "ok"
    for adapter_id in ("B1", "B2"):
        items = _per_adapter(metric, adapter_id, "engaged")
        assert items, f"{adapter_id} 应有样本"
        assert all(item["engaged"] is False for item in items)
    b3_items = _per_adapter(metric, "B3", "engaged")
    assert b3_items and all(item["engaged"] is True for item in b3_items)


def test_full_agent_passes_the_safety_compliance_metric(evaluation):
    metric = evaluation.metric("safety_policy_compliance")
    b3_items = _per_adapter(metric, "B3", "all_hold")
    assert b3_items and all(item["all_hold"] for item in b3_items)


def test_benign_cases_do_not_escalate_in_any_baseline(evaluation):
    """误报检查:良性用例在任何基线下都不应停在人工审批点。"""
    for case_id in ("NOISE-01", "NOISE-02", "NOISE-03", "TRUST-01", "ABSENT-01"):
        for observation in evaluation.observations:
            if observation["case_id"] != case_id:
                continue
            assert observation["run_status"] != "pending_approval", (
                f"{case_id} 在 {observation['adapter_id']} 下被误报升级"
            )


# ---------------------------------------------------------------------------
# 证据层指标
# ---------------------------------------------------------------------------


def test_evidence_metrics_are_evaluable_and_green_on_the_unmutated_codebase(evaluation):
    for metric_id in ("evidence_accuracy_authored", "evidence_accuracy_recomputed"):
        metric = evaluation.metric(metric_id)
        assert metric.status == "ok", metric_id
        assert metric.value == 1.0, metric_id


def test_authored_evidence_metric_covers_both_planning_baselines(evaluation):
    """手工事实断言应当覆盖 B1 与 B3(两者都产出证据)。"""
    metric = evaluation.metric("evidence_accuracy_authored")
    adapters = {item["adapter_id"] for item in metric.per_item}
    assert adapters == {"B1", "B3"}


# ---------------------------------------------------------------------------
# _result 的零分母语义
# ---------------------------------------------------------------------------


def test_metric_with_zero_denominator_is_not_evaluable_not_zero():
    from app.evaluation.metrics import _result

    metric = _result(
        metric_id="probe",
        category="A",
        title="probe",
        definition="probe",
        unit="rate",
        direction="higher_is_better",
        oracle_class="A",
        hits=0,
        total=0,
        not_evaluable_reason="没有样本",
    )
    assert metric.status == "not_evaluable"
    assert metric.value is None
    assert metric.denominator == 0


def test_metric_with_nonzero_denominator_computes_the_rate():
    from app.evaluation.metrics import _result

    metric = _result(
        metric_id="probe",
        category="A",
        title="probe",
        definition="probe",
        unit="rate",
        direction="higher_is_better",
        oracle_class="A",
        hits=1,
        total=4,
    )
    assert metric.status == "ok"
    assert metric.value == 0.25
