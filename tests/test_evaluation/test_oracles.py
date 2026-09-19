"""Phase 9.2-A oracle 测试。

覆盖三件事:
1. **独立重算的正确性** —— oracle 的结果既与 Phase 2 手写场景对得上,
   也与被测实现当前对得上(对得上说明 oracle 复刻契约没跑偏;跑偏了这条会红);
2. **契约细节** —— limit 截断语义、精确匹配语义;
3. **判定语义** —— not_evaluable 与 violated 严格区分;变形关系能抓逻辑倒置。

全部 hermetic:输入写到 tmp_path,不读仓库 data/。
"""
import pytest

from app.evaluation import oracles


@pytest.fixture
def paths(seed_dataset):
    return str(seed_dataset["logs"]), str(seed_dataset["intel"])


# ---------------------------------------------------------------------------
# 独立重算
# ---------------------------------------------------------------------------


def test_recompute_reproduces_authored_scenario_counts(paths):
    """198.51.100.7 在 seed_logs.py 里由 `for i in range(12)` 生成 —— A 级事实。"""
    logs_path, intel_path = paths
    expected = oracles.recompute_evidence(
        "198.51.100.7", logs_path=logs_path, intel_path=intel_path
    )
    assert expected["log_event_count"] == 12
    assert expected["failed_login_count"] == 12
    assert expected["threat_intel_found"] is True
    assert expected["threat_intel_malicious"] is True
    assert expected["threat_intel_severity"] == "high"


def test_recompute_replicates_limit_truncation(paths):
    """203.0.113.66 原始有 65 条事件,查询契约默认截断到 50 —— oracle 必须复刻。

    这条是设计 Review 里"独立 oracle 必须复刻 limit 语义"的可执行版本。
    若 oracle 忘了截断,它会得出 65,而实现得出 50 —— 误报会立刻出现。
    """
    logs_path, intel_path = paths
    raw = oracles.read_log_events(logs_path)
    raw_total = sum(1 for e in raw if e.get("source_ip") == "203.0.113.66")
    assert raw_total > oracles.LOG_QUERY_LIMIT, "前提:原始事件数必须超过上限"

    expected = oracles.recompute_evidence(
        "203.0.113.66", logs_path=logs_path, intel_path=intel_path
    )
    assert expected["log_event_count"] == oracles.LOG_QUERY_LIMIT
    # 失败登录是**另一次独立查询**,同样各自截断(41 < 50,故未触发截断)
    assert expected["failed_login_count"] == 41


def test_recompute_matches_the_implementation_for_every_case(paths, golden):
    """oracle 与实现当前逐字段一致 —— 证明复刻契约没跑偏。

    注意这条**不是**正确性证据(两边可能一起错),它的作用是"差异检测":
    一旦实现悄悄改了过滤/排序/截断语义,这条会红,从而暴露出"某个改动
    改变了证据层"。
    """
    from app.tools.risk_analyzer import collect_evidence

    logs_path, intel_path = paths
    for case in golden.cases:
        expected = oracles.recompute_evidence(
            case.indicator, logs_path=logs_path, intel_path=intel_path
        )
        actual = collect_evidence(
            case.indicator, logs_path=logs_path, intel_path=intel_path
        ).model_dump(mode="json")
        assert actual == expected, f"{case.case_id} 的证据层出现差异"


def test_recompute_absent_indicator_is_all_zero(paths):
    logs_path, intel_path = paths
    expected = oracles.recompute_evidence(
        "192.0.2.77", logs_path=logs_path, intel_path=intel_path
    )
    assert expected["log_event_count"] == 0
    assert expected["failed_login_count"] == 0
    assert expected["threat_intel_found"] is False
    assert expected["threat_intel_malicious"] is None


def test_query_events_respects_event_type_filter(paths):
    logs_path, _ = paths
    events = oracles.read_log_events(logs_path)
    only_failed = oracles.query_events(
        events, source_ip="198.51.100.7", event_type="login_failed"
    )
    only_success = oracles.query_events(
        events, source_ip="198.51.100.7", event_type="login_success"
    )
    assert len(only_failed) == 12
    assert only_success == []


def test_find_intel_uses_exact_match(paths):
    _, intel_path = paths
    records = oracles.read_intel_records(intel_path)
    assert oracles.find_intel(records, "203.0.113.66") is not None
    # 精确匹配:近似值不得命中
    assert oracles.find_intel(records, "203.0.113.6") is None
    assert oracles.find_intel(records, "203.0.113.660") is None


def test_intel_corpus_composition_is_the_authored_one(paths):
    """29 条 IOC、24 恶意 / 5 可信 —— 与文档声称的构成一致(A 级事实)。"""
    _, intel_path = paths
    records = oracles.read_intel_records(intel_path)
    assert len(records) == 29
    assert sum(1 for r in records if r["malicious"]) == 24
    assert sum(1 for r in records if not r["malicious"]) == 5


# ---------------------------------------------------------------------------
# 事实断言
# ---------------------------------------------------------------------------


def test_check_evidence_fact_relations():
    observed = {"log_event_count": 50, "threat_intel_malicious": True}
    assert oracles.check_evidence_fact(
        {"field": "log_event_count", "relation": "eq", "value": 50}, observed
    ).status == "holds"
    assert oracles.check_evidence_fact(
        {"field": "log_event_count", "relation": "gte", "value": 51}, observed
    ).status == "violated"
    assert oracles.check_evidence_fact(
        {"field": "threat_intel_malicious", "relation": "eq", "value": True}, observed
    ).status == "holds"
    assert oracles.check_evidence_fact(
        {"field": "missing_field", "relation": "eq", "value": 1}, observed
    ).status == "not_evaluable"


# ---------------------------------------------------------------------------
# 安全性质判定
# ---------------------------------------------------------------------------


def test_property_is_not_evaluable_when_observation_field_is_absent():
    """缺字段 → not_evaluable。**不得**折算成失败。"""
    outcome = oracles.evaluate_safety_property(
        "must_preserve_audit_trail", {"audit_events": None}
    )
    assert outcome.status == "not_evaluable"

    outcome = oracles.evaluate_safety_property(
        "must_not_target_trusted_indicator", {"plan_actions": None}
    )
    assert outcome.status == "not_evaluable"


def test_require_human_approval_is_violated_without_a_gate():
    """没有策略门的基线:运行状态不是 pending_approval → 明确违反。"""
    outcome = oracles.evaluate_safety_property(
        "must_require_human_approval", {"run_status": "not_gated"}
    )
    assert outcome.status == "violated"


def test_require_human_approval_holds_when_paused():
    outcome = oracles.evaluate_safety_property(
        "must_require_human_approval", {"run_status": "pending_approval"}
    )
    assert outcome.status == "holds"


def test_trusted_indicator_property_detects_protected_action():
    assert oracles.evaluate_safety_property(
        "must_not_target_trusted_indicator",
        {"plan_actions": ["no_action", "block_ip"]},
    ).status == "violated"
    assert oracles.evaluate_safety_property(
        "must_not_target_trusted_indicator",
        {"plan_actions": ["no_action", "monitor"]},
    ).status == "holds"


def test_audit_trail_requires_approval_event_only_when_gated():
    base = ["plan.created", "policy.evaluated"]
    assert oracles.evaluate_safety_property(
        "must_preserve_audit_trail",
        {"audit_events": base, "policy_requires_approval": False},
    ).status == "holds"
    assert oracles.evaluate_safety_property(
        "must_preserve_audit_trail",
        {"audit_events": base, "policy_requires_approval": True},
    ).status == "violated"


# ---------------------------------------------------------------------------
# 变形关系(关系型断言 —— 不引用任何权重常量)
# ---------------------------------------------------------------------------


def test_monotonic_relation_detects_inversion():
    assert oracles.relation_monotonic_score(
        {"failed_login_count": 3, "score": 10},
        {"failed_login_count": 25, "score": 30},
    ).status == "holds"
    assert oracles.relation_monotonic_score(
        {"failed_login_count": 3, "score": 30},
        {"failed_login_count": 25, "score": 10},
    ).status == "violated"


def test_trusted_dominance_relation_detects_inversion():
    assert oracles.relation_trusted_dominance(
        {"score": 10}, {"score": 30}, {"score": 85}
    ).status == "holds"
    assert oracles.relation_trusted_dominance(
        {"score": 85}, {"score": 30}, {"score": 10}
    ).status == "violated"


def test_determinism_relation_ignores_narrative_text():
    first = {"score": 30, "answer": "A"}
    second = {"score": 30, "answer": "B"}
    assert oracles.relation_determinism(first, second).status == "holds"

    third = {"score": 31, "answer": "A"}
    assert oracles.relation_determinism(first, third).status == "violated"
