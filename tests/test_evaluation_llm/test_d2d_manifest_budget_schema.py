"""Phase 9.2-D-2d —— **类型化资源预算**与冻结语义的聚焦测试。

本文件守住四件事:

    类型层      "有没有数值上界"由**封闭词表**回答,不由字符串前缀回答
    冻结语义    未决定的资源维度**必须**阻止冻结,描述性字符串**不能**绕过
    HTTP 记账   配置的理论包络 / 权威逻辑调用数 / 实际尝试 / 传输层观测 **四者分开**
    报告        七个概念**分列**,不得合并成一个 budget 数字

**全程离线**:零 provider 调用、零网络出口、零凭据、零试点执行。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation import pilot_config as PC
from app.evaluation.d2d_pilot import build_d2d_manifest
from app.evaluation.llm.budget import (
    UNKNOWN,
    BudgetCounters,
    BudgetGovernor,
    budget_from_plan,
    pilot_budget,
)
from app.evaluation.llm.pilot import (
    assert_frozen_pilot_manifest,
    budget_report,
    build_candidate_manifest,
    load_datasets,
    manifest_freeze_blockers,
    manifest_report,
    pilot_plan,
)
from app.evaluation.llm.protocol import (
    BUDGET_SCHEMA_VERSION,
    DECIDED_BOUNDEDNESS,
    PROTOCOL_VERSION,
    BudgetBoundedness,
    ExperimentManifest,
    ManifestError,
    ResourceBudget,
    ScopedEnvelope,
    budget_freeze_blockers,
    metric_schema_digest,
    placeholder_paths,
    real_llm_taskset_digest,
    verify_manifest,
)
from app.evaluation.llm.raw import RawRecord
from app.evaluation.llm.runner import golden_digest, system_prompt_sha256, tool_schema_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 基线 commit —— 冻结在 D-2d 起点。
GIT_COMMIT = "dfe7ea4b677c74b6e60ed40f20fdc42b4988735b"

#: 固定时刻,使清单摘要可跨运行比对。
CREATED_AT = "2026-09-29T00:00:00+00:00"

#: 冻结摘要字面量(**实测值**,不是推导值)。
FROZEN_METRIC_SCHEMA_DIGEST = "4ee64596fcb94167cd6d1b66fc2e4d24c485cf04773e7c4c184e66776d2ba044"
FROZEN_TASKSET_DIGEST = "b3eaac74010241c21dd6faab4c57f64fc70a84a1463e6e4a453b1faced13d526"
FROZEN_GOLDEN_DIGEST = "5f157ed92df12cb1f1d3f175327c39c01e49062e6084036fb9de641dc50743ca"
FROZEN_SYSTEM_PROMPT_SHA256 = (
    "9643ebaea5914eac9a6d9364c0f19c73e7a2f7cdf4d7be8e9112c2a84e393150"
)
FROZEN_TOOL_SCHEMA_SHA256 = (
    "44d77a0ce8b1dc7471cca840d13f62b72bc06ef16131476a16fca363440194f5"
)

#: 历史推导值:SDK `max_retries = 2` 假设下的包络。
HISTORICAL_HTTP_CEILING = 972


# ---------------------------------------------------------------------------
# 夹具(hermetic:全部在 pytest 临时目录里生成)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    return load_datasets(tmp_path_factory.mktemp("d2d-budget-fixtures"))


@pytest.fixture(scope="module")
def candidate_manifest(datasets):
    return build_candidate_manifest(
        git_commit=GIT_COMMIT, datasets=datasets, created_at_utc=CREATED_AT
    )


@pytest.fixture(scope="module")
def d2d_manifest(datasets):
    return build_d2d_manifest(
        git_commit=GIT_COMMIT, datasets=datasets, created_at_utc=CREATED_AT
    )


# ---------------------------------------------------------------------------
# 1. 封闭词表
# ---------------------------------------------------------------------------


def test_01_the_boundedness_vocabulary_is_closed_to_exactly_four_values():
    """四值封闭词表。**加第五值即协议变更** —— 这条断言就是那道闸门。"""
    assert {member.value for member in BudgetBoundedness} == {
        "NUMERICALLY_BOUNDED",
        "NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL",
        "NO_NUMERIC_BOUND_NO_PROVENANCE",
        "UNRESOLVED",
    }


def test_02_the_forbidden_not_frozen_name_does_not_exist():
    """`NOT_FROZEN_NO_PROVENANCE` **不得**出现在词表里。

    冻结清单不得含名字带 `NOT_FROZEN` 的状态 —— 那会让"已冻结"与
    "未冻结"在名字层面就无法区分。
    """
    assert not hasattr(BudgetBoundedness, "NOT_FROZEN_NO_PROVENANCE")
    assert "NOT_FROZEN_NO_PROVENANCE" not in {
        member.value for member in BudgetBoundedness
    }


def test_03_only_unresolved_blocks_freezing():
    assert BudgetBoundedness.UNRESOLVED not in DECIDED_BOUNDEDNESS
    assert DECIDED_BOUNDEDNESS == {
        BudgetBoundedness.NUMERICALLY_BOUNDED,
        BudgetBoundedness.NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL,
        BudgetBoundedness.NO_NUMERIC_BOUND_NO_PROVENANCE,
    }


# ---------------------------------------------------------------------------
# 2. 构造期不变式
# ---------------------------------------------------------------------------


def test_04_a_scoped_envelope_cannot_claim_to_be_a_total_bound():
    with pytest.raises(ValidationError):
        ScopedEnvelope(
            scope="output",
            value=10,
            unit="tokens",
            provenance="x",
            is_total_bound=True,
        )


def test_05_a_resource_budget_cannot_participate_in_execution_admission():
    with pytest.raises(ValidationError):
        ResourceBudget(
            boundedness=BudgetBoundedness.NUMERICALLY_BOUNDED,
            value=1,
            unit="tokens",
            provenance="x",
            reason="x",
            execution_admission=True,
        )


def test_06_a_non_numeric_state_may_not_carry_a_number():
    """**没有上界**不得携带一个数值 —— 否则会被读成「上界是那个数」。"""
    for state in (
        BudgetBoundedness.NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL,
        BudgetBoundedness.NO_NUMERIC_BOUND_NO_PROVENANCE,
        BudgetBoundedness.UNRESOLVED,
    ):
        with pytest.raises(ValidationError):
            ResourceBudget(boundedness=state, value=1, reason="x")
        with pytest.raises(ValidationError):
            ResourceBudget(boundedness=state, unit="tokens", reason="x")
        with pytest.raises(ValidationError):
            ResourceBudget(boundedness=state, provenance="x", reason="x")


def test_07_a_numeric_bound_requires_value_unit_and_provenance():
    with pytest.raises(ValidationError):
        ResourceBudget(boundedness=BudgetBoundedness.NUMERICALLY_BOUNDED, reason="x")
    with pytest.raises(ValidationError):
        ResourceBudget(
            boundedness=BudgetBoundedness.NUMERICALLY_BOUNDED,
            value=5,
            reason="x",
        )
    ok = ResourceBudget(
        boundedness=BudgetBoundedness.NUMERICALLY_BOUNDED,
        value=5,
        unit="tokens",
        provenance="synthetic",
        reason="synthetic",
    )
    assert ok.decided is True


def test_08_a_decided_state_requires_a_reason_but_unresolved_does_not():
    for state in sorted(DECIDED_BOUNDEDNESS, key=lambda s: s.value):
        with pytest.raises(ValidationError):
            ResourceBudget(boundedness=state)
    # UNRESOLVED 可以没有理由 —— 它的理由就是"还没决定"。
    assert ResourceBudget(boundedness=BudgetBoundedness.UNRESOLVED).decided is False


# ---------------------------------------------------------------------------
# 3. 冻结语义(旧漏洞必须已被堵死)
# ---------------------------------------------------------------------------


def test_09_a_descriptive_string_can_no_longer_be_assigned_to_a_budget_field(
    candidate_manifest,
):
    """**旧漏洞的形状**:裸 `str` 字段下,填一句描述就能静默清空 freeze blocker。

    类型化之后,连"填一句描述"这个动作本身都不再合法。
    """
    payload = candidate_manifest.model_dump(mode="json")
    for field in ("token_budget", "cost_budget"):
        with pytest.raises(ValidationError):
            ExperimentManifest.model_validate({**payload, field: "cost is bounded"})
        with pytest.raises(ValidationError):
            ExperimentManifest.model_validate({**payload, field: "NOT_A_PLACEHOLDER"})


def test_10_undecided_resource_dimensions_block_freezing(candidate_manifest):
    blockers = budget_freeze_blockers(candidate_manifest)
    assert blockers == [
        "cost_budget.boundedness(当前为 UNRESOLVED)",
        "token_budget.boundedness(当前为 UNRESOLVED)",
    ]
    assert all("UNRESOLVED" in item for item in blockers)


def test_11_the_freeze_error_names_both_placeholders_and_undecided_dimensions(
    candidate_manifest,
):
    """占位符与"未决定的资源维度"必须**同时**出现在同一条报错里。

    分成两条会让人只修看得见的那一条,然后以为冻结只剩最后一步。
    """
    payload = candidate_manifest.model_dump(mode="json")
    payload["manifest_status"] = "frozen"
    payload["endpoint_category"] = "OPENAI_COMPATIBLE"
    promoted = ExperimentManifest.model_validate(payload)
    from app.evaluation.llm.protocol import seal_manifest

    with pytest.raises(ManifestError) as excinfo:
        verify_manifest(seal_manifest(promoted), require_frozen=True)
    message = str(excinfo.value)
    assert "占位符" in message
    assert "provider" in message
    assert "model" in message
    assert "token_budget" in message
    assert "cost_budget" in message
    assert "UNRESOLVED" in message


def test_12_asking_for_a_frozen_manifest_with_undecided_budgets_fails_closed(datasets):
    """`manifest_status="frozen"` 必须**当场**跑冻结闸门 —— 不得产出
    「标着 frozen 但其实还没准备好」的产物。"""
    with pytest.raises(ManifestError):
        build_candidate_manifest(
            git_commit=GIT_COMMIT,
            datasets=datasets,
            created_at_utc=CREATED_AT,
            manifest_status="frozen",
        )


def test_13_the_decided_unbounded_states_may_freeze(d2d_manifest):
    """`NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL` 与 `NO_NUMERIC_BOUND_NO_PROVENANCE`
    **允许**出现在冻结清单里 —— 冻结的是一份诚实的"无上界"声明。"""
    assert d2d_manifest.manifest_status == "frozen"
    assert d2d_manifest.token_budget.boundedness is (
        BudgetBoundedness.NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL
    )
    assert d2d_manifest.cost_budget.boundedness is (
        BudgetBoundedness.NO_NUMERIC_BOUND_NO_PROVENANCE
    )
    assert manifest_freeze_blockers(d2d_manifest) == []
    assert budget_freeze_blockers(d2d_manifest) == []
    verify_manifest(d2d_manifest, require_frozen=True)


def test_14_placeholder_paths_is_not_weakened():
    """`placeholder_paths()` 的语义**不得**与预算完备性检查合并。"""
    payload = {
        "a": "TO_BE_FROZEN",
        "b": {"c": ["TO_BE_FROZEN_AT_IMPLEMENTATION"]},
        "d": "cost is bounded",
    }
    assert sorted(placeholder_paths(payload)) == ["a", "b.c[0]"]


# ---------------------------------------------------------------------------
# 4. D-2d 资源声明
# ---------------------------------------------------------------------------


def test_15_the_token_declaration_is_protocol_unbounded_with_no_number():
    declaration = PC.token_budget()
    assert declaration.boundedness is (
        BudgetBoundedness.NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL
    )
    assert declaration.value is None
    assert declaration.unit is None
    assert declaration.provenance is None
    assert declaration.execution_admission is False
    assert "NOT numerically bounded by protocol" in declaration.reason


def test_16_the_only_strict_numeric_envelope_is_the_output_component():
    declaration = PC.token_budget()
    assert len(declaration.envelopes) == 1
    envelope = declaration.envelopes[0]
    assert envelope.scope == "output"
    assert envelope.unit == "tokens"
    assert envelope.value == 324 * PC.PILOT_OUTPUT_TOKEN_CAP == 331776
    assert envelope.is_total_bound is False


def test_17_the_envelope_provenance_states_its_derivation_and_its_limits():
    """溯源必须写明**怎么算出来的**与**它不是什么**。"""
    envelope = PC.token_budget().envelopes[0]
    assert "logical_invocation_hard_ceiling(324)" in envelope.provenance
    assert f"PILOT_OUTPUT_TOKEN_CAP({PC.PILOT_OUTPUT_TOKEN_CAP})" in (
        envelope.provenance
    )
    assert "OUTPUT COMPONENT ENVELOPE" in envelope.provenance
    assert "NOT a total-token budget" in envelope.provenance


def test_18_the_output_envelope_is_not_rendered_as_a_total_token_budget():
    """把 331776 说成 total token budget 必须**说不通** —— 结构上就不成立。"""
    declaration = PC.token_budget()
    assert declaration.value is None, "总量字段必须是 None(无总量上界)"
    assert declaration.boundedness is not BudgetBoundedness.NUMERICALLY_BOUNDED
    # 包络里也不得出现"总量"声明。
    for envelope in declaration.envelopes:
        assert envelope.scope != "total"
        assert envelope.is_total_bound is False


def test_19_the_cost_declaration_is_decided_and_carries_no_number():
    declaration = PC.cost_budget()
    assert declaration.boundedness is BudgetBoundedness.NO_NUMERIC_BOUND_NO_PROVENANCE
    assert declaration.value is None
    assert declaration.unit is None
    assert declaration.provenance is None
    assert declaration.execution_admission is False
    assert declaration.decided is True
    assert "NO_AUTHORITATIVE_PRICING_PROVENANCE" in declaration.reason


def test_20_the_cost_declaration_never_reads_as_satisfied_or_zero():
    """**禁止**的读法必须一个都不在文本里。"""
    text = PC.cost_budget().reason
    for forbidden in (
        "cost budget satisfied",
        "cost controlled",
        "cost <= ",
        "zero cost",
        "unknown == zero",
        "成本已满足",
        "成本已受控",
        "零成本",
    ):
        assert forbidden not in text


def test_21_neither_resource_declaration_participates_in_admission(d2d_manifest):
    assert d2d_manifest.token_budget.execution_admission is False
    assert d2d_manifest.cost_budget.execution_admission is False
    # 准入仍然只由两个权威计数器决定。
    assert d2d_manifest.total_runs == 108
    assert d2d_manifest.logical_invocation_hard_ceiling == 324


# ---------------------------------------------------------------------------
# 5. HTTP 尝试的四个量(§F)
# ---------------------------------------------------------------------------


def test_22_the_current_d2d_manifest_exposes_324_not_972(d2d_manifest):
    """**当前 D-2d 清单**不得再暴露 972 作为执行包络。"""
    assert d2d_manifest.provider_http_attempt_ceiling == 324
    assert d2d_manifest.provider_http_attempt_ceiling != HISTORICAL_HTTP_CEILING
    basis = d2d_manifest.provider_http_attempt_ceiling_basis
    assert "max_retries 0" in basis
    assert "假设" in basis
    assert "不是实测" in basis


def test_23_the_historical_972_is_preserved_for_the_legacy_default_plan():
    """历史推导值**逐字段不变** —— 既有产物因此仍然可复现。"""
    legacy = pilot_budget(
        baseline_labels=("B0-shared", "B0-notool", "B2'", "B3"), repetition_count=3
    )
    assert legacy.provider_http_attempt_ceiling == HISTORICAL_HTTP_CEILING
    assert legacy.sdk_max_retries_assumption == 2
    assert pilot_plan().provider_http_attempt_ceiling == HISTORICAL_HTTP_CEILING
    assert pilot_plan(sdk_max_retries=0).provider_http_attempt_ceiling == 324


def test_24_the_configured_envelope_is_not_an_admission_input():
    assert PC.HTTP_ATTEMPT_CEILING_IS_ADMISSION_INPUT is False
    assert PC.HISTORICAL_HTTP_ATTEMPT_CEILING_UNDER_SDK_RETRIES_2 == 972
    assert PC.pilot_http_attempt_ceiling() == 324
    assert PC.pilot_logical_invocation_hard_ceiling() == 324


def test_25_actual_http_attempts_stay_unknown_not_324_and_not_zero():
    """**实际**尝试不可观测时记 UNKNOWN —— 不得被理论包络顶替,也不得变成 0。"""
    governor = BudgetGovernor(
        budget=budget_from_plan(pilot_plan(sdk_max_retries=0))
    )
    governor.record_unit(provider_http_attempts=None)
    assert governor.counters.provider_http_attempts == UNKNOWN
    assert governor.counters.provider_http_attempts != 324
    assert governor.counters.provider_http_attempts != 0
    assert governor.experiment_provider_http_attempts == UNKNOWN


def test_26_no_module_writes_the_theoretical_envelope_into_actual_attempts():
    """AST 级检查:没有任何模块把 `provider_http_attempt_ceiling` 写进
    `provider_http_attempts` 关键字。理论包络**不得**覆盖实际记账。
    """

    def offenders_in(source: str) -> list[str]:
        found: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.keyword) and node.arg == "provider_http_attempts":
                segment = ast.unparse(node.value)
                if "ceiling" in segment:
                    found.append(segment)
        return found

    # 正对照:该形状**必须**被检出。
    assert offenders_in("f(provider_http_attempts=budget.provider_http_attempt_ceiling)")
    # 负对照:合法记账**不得**被检出。
    assert not offenders_in("f(provider_http_attempts=None)")
    assert not offenders_in("f(provider_http_attempts=observed)")

    actual: list[tuple[str, str]] = []
    for path in sorted((REPO_ROOT / "app").rglob("*.py")):
        for segment in offenders_in(path.read_text(encoding="utf-8")):
            actual.append((str(path.relative_to(REPO_ROOT)), segment))
    assert actual == []


def test_27_unknown_is_not_silently_converted_to_zero_in_the_record():
    """记录的默认值必须是 UNKNOWN,不是 0。"""
    assert RawRecord.model_fields["provider_http_attempts"].default == UNKNOWN
    counters = BudgetCounters(provider_http_attempts_unobservable=1)
    assert counters.provider_http_attempts == UNKNOWN
    assert counters.provider_http_attempts_observed == 0


# ---------------------------------------------------------------------------
# 6. 报告:七个概念分列
# ---------------------------------------------------------------------------


def test_28_the_manifest_report_separates_the_seven_concepts(d2d_manifest):
    report = budget_report(d2d_manifest)
    assert set(report) >= {
        "1_execution_admission",
        "2_token_total_boundedness",
        "3_output_token_scoped_envelope",
        "4_monetary_cost_boundedness",
        "5_configured_theoretical_http_attempt_envelope",
        "6_actual_provider_http_attempts",
        "7_independently_observed_transport_http_attempts",
    }
    assert report["1_execution_admission"]["logical_llm_invocations_ceiling"] == 324
    assert report["2_token_total_boundedness"]["value"] is None
    assert report["3_output_token_scoped_envelope"]["envelopes"][0]["value"] == 331776
    assert report["4_monetary_cost_boundedness"]["value"] is None
    assert report["5_configured_theoretical_http_attempt_envelope"]["value"] == 324
    assert report["6_actual_provider_http_attempts"]["value"] == UNKNOWN
    assert report["7_independently_observed_transport_http_attempts"]["value"] is None


def test_29_the_manifest_report_does_not_collapse_the_budget_into_one_number(
    d2d_manifest,
):
    report = manifest_report(d2d_manifest)
    assert "budget" in report
    # 顶层不得出现一个"合计预算"式的标量。
    assert not isinstance(report["budget"], (int, float, str))
    # 两个 324 是**不同的量**,必须落在不同的键下。
    assert report["budget"]["1_execution_admission"][
        "logical_llm_invocations_ceiling"
    ] == report["budget"]["5_configured_theoretical_http_attempt_envelope"]["value"]
    assert report["budget"]["6_actual_provider_http_attempts"]["value"] == UNKNOWN


def test_30_the_manifest_report_is_json_serialisable(d2d_manifest):
    """报告要能被原样写盘 —— 中文不得被转义成 `\\uXXXX`。"""
    text = json.dumps(manifest_report(d2d_manifest), ensure_ascii=False)
    assert "\\u" not in text
    json.loads(text)


def test_31_the_pilot_report_marks_missing_manifest_instead_of_omitting(d2a_outcome):
    """没有清单时,token / cost 两行必须记 `N/A` 而**不是**消失。"""
    from app.evaluation.llm.pilot_report import render_pilot_report

    text = render_pilot_report(d2a_outcome)
    assert "预算的七个概念" in text
    assert text.count("N/A(清单未随 outcome 传入)") == 3
    assert "**实际** provider HTTP 尝试" in text


def test_32_the_pilot_report_fills_token_and_cost_when_the_manifest_is_passed(
    d2a_outcome, d2d_manifest
):
    from app.evaluation.llm.pilot_report import render_pilot_report

    text = render_pilot_report(d2a_outcome, manifest=d2d_manifest)
    assert "NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL" in text
    assert "NO_NUMERIC_BOUND_NO_PROVENANCE" in text
    assert "output=331776 tokens" in text
    assert "N/A(清单未随 outcome 传入)" not in text


# ---------------------------------------------------------------------------
# 7. 版本与冻结摘要(回归)
# ---------------------------------------------------------------------------


def test_33_the_protocol_and_budget_schema_versions_are_bumped_together():
    assert PROTOCOL_VERSION == "9.2-D-2.2"
    assert BUDGET_SCHEMA_VERSION == "9.2-D-2.2"
    assert ExperimentManifest.model_fields["budget_schema_version"].default == (
        BUDGET_SCHEMA_VERSION
    )


def test_34_the_unrelated_frozen_digests_are_unchanged(datasets):
    """预算 schema 变更**不得**迫使其它摘要改变。"""
    assert metric_schema_digest() == FROZEN_METRIC_SCHEMA_DIGEST
    assert real_llm_taskset_digest(datasets) == FROZEN_TASKSET_DIGEST
    assert golden_digest.__module__  # 官方函数存在
    assert system_prompt_sha256() == FROZEN_SYSTEM_PROMPT_SHA256
    assert tool_schema_sha256() == FROZEN_TOOL_SCHEMA_SHA256


def test_35_the_manifest_digest_recomputes_for_both_candidate_and_frozen(
    candidate_manifest, d2d_manifest
):
    from app.evaluation.llm.protocol import compute_manifest_digest

    assert candidate_manifest.manifest_digest == compute_manifest_digest(
        candidate_manifest
    )
    assert d2d_manifest.manifest_digest == compute_manifest_digest(d2d_manifest)
    assert candidate_manifest.manifest_digest != d2d_manifest.manifest_digest


# ---------------------------------------------------------------------------
# 8. 冻结闸门(库级)
# ---------------------------------------------------------------------------


def test_36_the_gate_accepts_the_d2d_frozen_manifest(datasets, d2d_manifest):
    assert_frozen_pilot_manifest(
        d2d_manifest,
        datasets=datasets,
        expected_git_commit=GIT_COMMIT,
        plan=PC.d2d_plan(),
        expected_identity=PC.pilot_identity_fields(),
        expected_budgets=PC.d2d_resource_budgets(),
    )


def test_37_the_gate_rejects_a_candidate_manifest(datasets, candidate_manifest):
    with pytest.raises(ManifestError):
        assert_frozen_pilot_manifest(
            candidate_manifest,
            datasets=datasets,
            expected_git_commit=GIT_COMMIT,
            plan=PC.d2d_plan(),
        )


def test_38_the_gate_rejects_a_plan_mismatch(datasets, d2d_manifest):
    """清单说 324、计划说 972 ⇒ 必须拒绝。"""
    with pytest.raises(ManifestError, match="provider_http_attempt_ceiling"):
        assert_frozen_pilot_manifest(
            d2d_manifest, datasets=datasets, expected_git_commit=GIT_COMMIT
        )


def test_39_the_gate_rejects_an_identity_mismatch(datasets, d2d_manifest):
    with pytest.raises(ManifestError, match="provider"):
        assert_frozen_pilot_manifest(
            d2d_manifest,
            datasets=datasets,
            expected_git_commit=GIT_COMMIT,
            plan=PC.d2d_plan(),
            expected_identity={"provider": "someone-else"},
        )


def test_40_the_gate_rejects_a_budget_mismatch(datasets, d2d_manifest):
    wrong = ResourceBudget(
        boundedness=BudgetBoundedness.NUMERICALLY_BOUNDED,
        value=1,
        unit="tokens",
        provenance="synthetic",
        reason="synthetic",
    )
    with pytest.raises(ManifestError, match="token_budget"):
        assert_frozen_pilot_manifest(
            d2d_manifest,
            datasets=datasets,
            expected_git_commit=GIT_COMMIT,
            plan=PC.d2d_plan(),
            expected_budgets={"token_budget": wrong},
        )


def test_41_the_gate_rejects_a_digest_tamper(datasets, d2d_manifest):
    tampered = d2d_manifest.model_copy(update={"manifest_digest": "0" * 64})
    with pytest.raises(ManifestError, match="摘要不匹配"):
        assert_frozen_pilot_manifest(
            tampered, datasets=datasets, expected_git_commit=GIT_COMMIT
        )


def test_42_the_gate_rejects_a_git_commit_mismatch(datasets, d2d_manifest):
    with pytest.raises(ManifestError, match="git commit"):
        assert_frozen_pilot_manifest(
            d2d_manifest, datasets=datasets, expected_git_commit="0000000"
        )
