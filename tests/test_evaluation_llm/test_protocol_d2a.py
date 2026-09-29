"""Phase 9.2-D-2a —— 官方摘要与清单闸门。

三件事必须在**启动真实调用之前**就变成机械事实:

    任务清单摘要     事后改 benchmark 会被检出
    指标 schema 摘要 事后改判定口径会被检出
    清单闸门         候选清单不得被当成冻结清单使用

本文件的重点不是"摘要函数能跑",而是**摘要对改动敏感**。
一个对改动不敏感的摘要比没有摘要更糟:它提供虚假的安全感。
"""
import pytest

from app.evaluation.llm import protocol as P
from app.evaluation.llm.dataset import (
    LLM_TASKS,
    VARIANT_BASE,
    VARIANT_INJECTION_INERT,
    build_datasets,
)
from app.evaluation.llm.pilot import (
    PILOT_BASELINES,
    assert_candidate_only,
    build_candidate_manifest,
    manifest_freeze_blockers,
    pilot_plan,
    verify_candidate_manifest,
)

OLD_SUPERSEDED_DIGEST = (
    "7f3dc96732b1fbda6fee335961a5f4554bb42a195fe4d3dfd970711b336363d4"
)


# ---------------------------------------------------------------------------
# 1. 规范化
# ---------------------------------------------------------------------------


def test_canonical_json_is_key_order_and_whitespace_insensitive():
    assert P.canonical_json({"b": 1, "a": 2}) == P.canonical_json({"a": 2, "b": 1})
    assert " " not in P.canonical_json({"a": 1, "b": [1, 2]})


def test_canonical_json_keeps_non_ascii_unescaped():
    """中文载荷不因转义方式不同而算出不同摘要。"""
    assert "维护" in P.canonical_json({"k": "【系统维护通知】"})
    assert "\\u" not in P.canonical_json({"k": "【系统维护通知】"})


def test_sha256_hex_is_utf8_based():
    assert P.sha256_hex("a") == P.sha256_bytes(b"a")


# ---------------------------------------------------------------------------
# 2. 任务清单记录的形状
# ---------------------------------------------------------------------------


REQUIRED_TASK_RECORD_FIELDS = {
    "task_id", "family", "dataset_variant", "indicator", "prompt_sha256",
    "fixture_sha256", "tool_contract_sha256", "grounding_contract_sha256",
    "security_contract_sha256", "control_variant", "control_fixture_sha256",
}


def test_taskset_record_has_exactly_the_required_fields(d2a_datasets):
    record = P.taskset_records(d2a_datasets)[0]
    assert set(record) == REQUIRED_TASK_RECORD_FIELDS


def test_taskset_records_are_sorted_by_task_id(d2a_datasets):
    ids = [record["task_id"] for record in P.taskset_records(d2a_datasets)]
    assert ids == sorted(ids)


def test_non_injection_tasks_have_null_control(d2a_datasets):
    records = P.taskset_records(d2a_datasets)
    without = [r for r in records if r["control_variant"] is None]
    assert without, "至少应有一个不带注入契约的任务"
    for record in without:
        assert record["control_fixture_sha256"] is None


def test_injection_tasks_point_at_the_inert_control(d2a_datasets):
    records = [r for r in P.taskset_records(d2a_datasets) if r["control_variant"]]
    assert records, "至少应有一个带注入契约的任务"
    for record in records:
        assert record["control_variant"] == VARIANT_INJECTION_INERT
        assert record["control_fixture_sha256"] == P.fixture_sha256(
            d2a_datasets[VARIANT_INJECTION_INERT]
        )


def test_taskset_digest_is_deterministic(d2a_datasets):
    assert P.real_llm_taskset_digest(d2a_datasets) == P.real_llm_taskset_digest(d2a_datasets)


def test_candidate_taskset_digest_supersedes_the_old_proposed_digest(d2a_datasets):
    """旧的 7f3dc967… 是**在惰性对照之前**提出的,已被取代。"""
    assert P.real_llm_taskset_digest(d2a_datasets) != OLD_SUPERSEDED_DIGEST


def test_taskset_digest_depends_on_the_version(d2a_datasets, monkeypatch):
    baseline = P.real_llm_taskset_digest(d2a_datasets)
    monkeypatch.setattr(P, "REAL_LLM_TASKSET_VERSION", "9.2-D-9.9")
    assert P.real_llm_taskset_digest(d2a_datasets) != baseline


# ---------------------------------------------------------------------------
# 3. 任务清单摘要的**变异敏感性**
# ---------------------------------------------------------------------------


def _mutate_task(**updates):
    """替换掉第一个任务(保持其余不变)。"""
    head, *tail = LLM_TASKS
    return (head.model_copy(update=updates), *tail)


def test_prompt_mutation_changes_taskset_digest(d2a_datasets):
    baseline = P.real_llm_taskset_digest(d2a_datasets)
    mutated = _mutate_task(user_prompt=LLM_TASKS[0].user_prompt + "(改)")
    assert P.real_llm_taskset_digest(d2a_datasets, mutated) != baseline


def test_task_contract_mutation_changes_taskset_digest(d2a_datasets):
    contract = LLM_TASKS[0].tool_contract
    mutated = _mutate_task(
        tool_contract=contract.model_copy(
            update={"max_total_calls": contract.max_total_calls + 1}
        )
    )
    assert P.real_llm_taskset_digest(d2a_datasets, mutated) != P.real_llm_taskset_digest(
        d2a_datasets
    )


def test_grounding_contract_mutation_changes_taskset_digest(d2a_datasets):
    contract = LLM_TASKS[0].grounding_contract
    facts = list(contract.verifiable_facts)
    facts[0] = facts[0].model_copy(update={"field": facts[0].field + "_mutated"})
    mutated = _mutate_task(
        grounding_contract=contract.model_copy(update={"verifiable_facts": facts})
    )
    assert P.real_llm_taskset_digest(d2a_datasets, mutated) != P.real_llm_taskset_digest(
        d2a_datasets
    )


def test_security_contract_mutation_changes_taskset_digest(d2a_datasets):
    contract = LLM_TASKS[0].security_contract
    mutated = _mutate_task(
        security_contract=contract.model_copy(
            update={"path_arguments": [*contract.path_arguments, "extra_argument"]}
        )
    )
    assert P.real_llm_taskset_digest(d2a_datasets, mutated) != P.real_llm_taskset_digest(
        d2a_datasets
    )


def test_fixture_mutation_changes_taskset_digest(tmp_path):
    """改 fixture **内容**必须改变摘要(路径变了不算)。"""
    datasets = build_datasets(tmp_path / "a")
    baseline = P.real_llm_taskset_digest(datasets)
    with open(datasets[VARIANT_BASE]["logs"], "a", encoding="utf-8") as handle:
        handle.write('{"mutated": true}\n')
    assert P.real_llm_taskset_digest(datasets) != baseline


def test_control_fixture_mutation_changes_taskset_digest(tmp_path):
    """**只改对照、不改任务**也必须被检出 —— 否则归因会被静默改变。"""
    datasets = build_datasets(tmp_path / "a")
    baseline = P.real_llm_taskset_digest(datasets)

    control = datasets[VARIANT_INJECTION_INERT]["logs"]
    with open(control, "a", encoding="utf-8") as handle:
        handle.write('{"extra_control_only": true}\n')

    assert P.real_llm_taskset_digest(datasets) != baseline


def test_taskset_digest_is_independent_of_the_workdir(tmp_path):
    """换工作目录不改变摘要 —— 否则跨运行可比性就不成立。"""
    first = P.real_llm_taskset_digest(build_datasets(tmp_path / "a"))
    second = P.real_llm_taskset_digest(build_datasets(tmp_path / "b"))
    assert first == second


# ---------------------------------------------------------------------------
# 4. 指标 schema 摘要
# ---------------------------------------------------------------------------


def test_metric_schema_has_22_unique_entries():
    assert len(P.METRIC_SCHEMA) == 22
    assert len({entry.metric_id for entry in P.METRIC_SCHEMA}) == 22


def test_metric_schema_covers_all_categories():
    categories = {entry.category for entry in P.METRIC_SCHEMA}
    assert categories == {
        "capability", "grounding", "injection", "invariant", "efficiency",
    }


def test_metric_schema_injection_entries_state_the_exposure_gate():
    follow = next(e for e in P.METRIC_SCHEMA if e.metric_id == "prompt_injection_follow_rate")
    match = next(e for e in P.METRIC_SCHEMA if e.metric_id == "injection_target_match_rate")
    assert "已暴露" in follow.denominator_definition
    assert "不设暴露门槛" in match.denominator_definition
    assert match.classification == "descriptive"


def test_invariant_entries_are_classified_as_regression_invariants():
    invariants = [e for e in P.METRIC_SCHEMA if e.category == "invariant"]
    assert invariants
    assert all(e.classification == "regression_invariant" for e in invariants)


def test_metric_schema_digest_is_deterministic():
    assert P.metric_schema_digest() == P.metric_schema_digest()


def test_metric_semantic_mutation_changes_the_digest(monkeypatch):
    """改**语义**而**不改名字**必须被检出 —— 这是最危险的一类改动。"""
    baseline = P.metric_schema_digest()
    entries = list(P.METRIC_SCHEMA)
    index = next(
        i for i, e in enumerate(entries)
        if e.metric_id == "injection_target_match_rate"
    )
    entries[index] = entries[index].model_copy(
        update={"numerator_definition": "叙事命中注入目标的配对数(**已改为归因成立**)"}
    )
    monkeypatch.setattr(P, "METRIC_SCHEMA", tuple(entries))
    assert P.metric_schema_digest() != baseline


def test_metric_name_only_mutation_also_changes_the_digest(monkeypatch):
    baseline = P.metric_schema_digest()
    entries = list(P.METRIC_SCHEMA)
    entries[0] = entries[0].model_copy(update={"metric_id": "renamed_metric"})
    monkeypatch.setattr(P, "METRIC_SCHEMA", tuple(entries))
    assert P.metric_schema_digest() != baseline


def test_not_evaluable_semantics_are_inside_the_digest(monkeypatch):
    """「测不了」的理由同样是判定口径,必须进摘要域。"""
    baseline = P.metric_schema_digest()
    monkeypatch.setattr(
        P, "NOT_EVALUABLE_REASONS",
        {**dict(P.NOT_EVALUABLE_REASONS), "safety_containment_rate": "改过的理由"},
    )
    assert P.metric_schema_digest() != baseline


def test_invariant_semantics_declaration_is_inside_the_digest(monkeypatch):
    baseline = P.metric_schema_digest()
    monkeypatch.setattr(P, "INVARIANT_SEMANTICS", "ARCHITECTURE_REGRESSION_INVARIANTS:safety score")
    assert P.metric_schema_digest() != baseline


# ---------------------------------------------------------------------------
# 5. 清单闸门
# ---------------------------------------------------------------------------


def test_candidate_manifest_is_candidate(d2a_manifest):
    assert d2a_manifest.manifest_status == "candidate"
    assert_candidate_only(d2a_manifest)


def test_candidate_manifest_digest_recomputes(d2a_manifest):
    assert d2a_manifest.manifest_digest == P.compute_manifest_digest(d2a_manifest)


def test_candidate_manifest_records_the_frozen_pilot_structure(d2a_manifest):
    assert d2a_manifest.treatment_runs == 96
    assert d2a_manifest.control_runs == 12
    assert d2a_manifest.total_runs == 108
    assert d2a_manifest.logical_invocation_hard_ceiling == 324
    assert d2a_manifest.provider_http_attempt_ceiling == 972
    assert d2a_manifest.harness_level_retry == 0
    assert d2a_manifest.conditions == list(PILOT_BASELINES)
    assert d2a_manifest.repetition_count == 3


def test_candidate_manifest_carries_the_offline_task_digest(d2a_manifest, d2a_datasets):
    assert d2a_manifest.taskset_digest == P.real_llm_taskset_digest(d2a_datasets)
    assert d2a_manifest.metric_schema_digest == P.metric_schema_digest()
    assert d2a_manifest.failure_taxonomy_version == P.FAILURE_TAXONOMY_VERSION


def test_candidate_verification_passes(d2a_manifest, d2a_datasets):
    verify_candidate_manifest(
        d2a_manifest, datasets=d2a_datasets, expected_git_commit=d2a_manifest.git_commit
    )


def test_frozen_verification_rejects_a_candidate_manifest(d2a_manifest):
    with pytest.raises(P.ManifestError, match="candidate"):
        P.verify_manifest(d2a_manifest, require_frozen=True)


def test_frozen_verification_rejects_unresolved_placeholders(d2a_manifest):
    promoted = P.seal_manifest(d2a_manifest.model_copy(update={"manifest_status": "frozen"}))
    with pytest.raises(P.ManifestError, match="占位符"):
        P.verify_manifest(promoted, require_frozen=True)


def test_freeze_blockers_enumerate_what_d2b_must_resolve(d2a_manifest):
    blockers = manifest_freeze_blockers(d2a_manifest)
    assert "provider" in blockers
    assert "model" in blockers
    assert "endpoint_category" in blockers
    # 资源预算不再是"占位符字符串",而是**未决定的资源维度**:
    # 路径形如 `token_budget.boundedness(当前为 UNRESOLVED)`。
    assert any(item.startswith("token_budget.") for item in blockers)
    assert any(item.startswith("cost_budget.") for item in blockers)
    assert any("manifest_status" in item for item in blockers)


def test_manifest_mutation_changes_the_digest(d2a_manifest):
    baseline = P.compute_manifest_digest(d2a_manifest)
    mutated = d2a_manifest.model_copy(update={"repetition_count": 4})
    assert P.compute_manifest_digest(mutated) != baseline


def test_verify_rejects_a_tampered_digest(d2a_manifest):
    tampered = d2a_manifest.model_copy(update={"manifest_digest": "0" * 64})
    with pytest.raises(P.ManifestError, match="摘要不匹配"):
        P.verify_manifest(tampered)


def test_verify_rejects_a_git_commit_mismatch(d2a_manifest):
    with pytest.raises(P.ManifestError, match="git commit"):
        P.verify_manifest(d2a_manifest, expected_git_commit="0000000")


def test_verify_rejects_a_taskset_digest_mismatch(d2a_manifest):
    with pytest.raises(P.ManifestError, match="任务清单摘要"):
        P.verify_manifest(d2a_manifest, expected_taskset_digest="0" * 64)


def test_verify_rejects_a_metric_schema_mismatch(d2a_manifest):
    with pytest.raises(P.ManifestError, match="指标 schema 摘要"):
        P.verify_manifest(d2a_manifest, expected_metric_schema_digest="0" * 64)


def test_verify_rejects_an_incompatible_protocol_version(d2a_manifest):
    with pytest.raises(P.ManifestError, match="协议版本"):
        P.verify_manifest(d2a_manifest, expected_protocol_version="0.0-incompatible")


def test_seal_manifest_does_not_mutate_the_input(d2a_manifest):
    before = d2a_manifest.manifest_digest
    P.seal_manifest(d2a_manifest.model_copy(update={"repetition_count": 5}))
    assert d2a_manifest.manifest_digest == before


def test_manifest_digest_domain_excludes_itself(d2a_manifest):
    domain = P.manifest_digest_domain(d2a_manifest)
    assert "manifest_digest" not in domain


def test_placeholder_paths_finds_nested_placeholders():
    payload = {"a": P.TO_BE_FROZEN, "b": {"c": [P.TO_BE_FROZEN_AT_IMPLEMENTATION]}}
    found = P.placeholder_paths(payload)
    assert "a" in found
    assert "b.c[0]" in found


def test_pilot_plan_is_derived_not_hand_written():
    """规模必须由结构推导:任务集变一个任务,计划必须跟着变。"""
    plan = pilot_plan()
    head, *tail = LLM_TASKS
    smaller = pilot_plan(tasks=tuple(tail))
    assert smaller.task_count == plan.task_count - 1
    assert smaller.treatment_runs == plan.treatment_runs - 3 * len(PILOT_BASELINES)
