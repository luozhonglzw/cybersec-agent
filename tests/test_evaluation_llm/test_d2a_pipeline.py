"""Phase 9.2-D-2a —— 离线试点流水线端到端。

一条流水线要证明的不是"它能跑",而是**它在被观测的关键点上说了真话**:

    计数器互不混淆            run / attempt / invocation / HTTP attempt
    匹配对照真的只差一个变量   配对不变量逐条通过
    未暴露不算抵抗            暴露门槛的分母与 N/E
    失败不触发重跑            冻结分类表下 retry_allowed 恒为 0
    结果可复现且可核验         跨工作目录逐字节一致 + sidecar
    零网络出口                审计钩子实测
"""
import asyncio
import json
from pathlib import Path

import pytest

from app.evaluation.llm.derived import (
    N3_LIMITATION,
    NO_GENERAL_SAFETY_CLAIM,
    NO_LEADERBOARD_STATEMENT,
    NO_P_VALUE_STATEMENT,
    WILSON_CAVEAT,
)
from app.evaluation.llm.executor import (
    OfflineExecutor,
    RawArtifactExistsError,
    assert_no_silent_append,
    build_matched_pairs,
    plan_resume,
)
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.pilot_report import CAVEAT_KEYS, PILOT_REPORT_CAVEATS
from app.evaluation.llm.raw import (
    RawWriter,
    RecordStatus,
    ResumeEligibility,
    index_by_unit,
    record_signature,
)

RECORD_IDENTITY_FIELDS = (
    "run_id", "experiment_id", "protocol_version", "manifest_digest", "task_id",
    "condition", "baseline_label", "dataset_variant", "repetition_id", "execution_index",
)
RECORD_COUNTER_FIELDS = (
    "experimental_run_attempts", "logical_llm_invocations", "provider_http_attempts",
)


@pytest.fixture(scope="session")
def d2a_records(d2a_outcome):
    writer = RawWriter(d2a_outcome.raw_path, experiment_id=d2a_outcome.experiment_id)
    return writer.read_all()


# ---------------------------------------------------------------------------
# 1. 规模与计数器
# ---------------------------------------------------------------------------


def test_full_pilot_runs_all_108_units(d2a_outcome, d2a_plan):
    assert d2a_plan.total_runs == 108
    assert d2a_plan.treatment_runs == 96
    assert d2a_plan.control_runs == 12
    assert d2a_outcome.record_count == 108
    assert len(d2a_outcome.executed_keys) == 108
    assert d2a_outcome.skipped_frozen_keys == []


def test_sidecar_verifies(d2a_outcome):
    assert d2a_outcome.sidecar_ok is True
    assert d2a_outcome.raw_sha256


def test_budget_counts_treatment_and_control(d2a_outcome):
    """预算必须把**对照**算进去 —— 它们同样是真实运行。"""
    budget = d2a_outcome.budget
    assert budget["treatment_runs"] == 96
    assert budget["control_runs"] == 12
    assert budget["total_runs"] == 108
    assert budget["experimental_runs"] == 108


def test_experimental_run_attempts_is_not_logical_invocations(d2a_outcome):
    """两个量**不是一回事**:单元执行了 1 次,而图条件可能调 1~5 次模型。

    若把它们设为相等,调用预算会少算最多 5 倍,而"图跑了几轮"这个事实
    会从记录里消失。
    """
    budget = d2a_outcome.budget
    assert budget["experimental_run_attempts"] == 108
    assert budget["logical_llm_invocations"] > budget["experimental_run_attempts"]
    assert budget["logical_llm_invocations"] >= budget["logical_invocation_floor"]
    assert budget["logical_llm_invocations"] <= budget["logical_invocation_hard_ceiling"]


def test_provider_http_attempts_is_unknown_not_zero(d2a_outcome):
    """不可观测记 UNKNOWN —— **不记 0**。0 会被读成"零消耗"。"""
    assert d2a_outcome.budget["provider_http_attempts"] == "UNKNOWN"
    assert d2a_outcome.budget["provider_http_attempts_observed"] == 0
    assert d2a_outcome.budget["provider_http_attempts_unobservable"] == 108


def test_http_ceiling_is_declared_as_an_assumption(d2a_outcome):
    basis = d2a_outcome.budget["provider_http_attempt_ceiling_basis"]
    assert "假设" in basis
    assert "不是实测" in basis


def test_harness_level_retry_is_zero(d2a_outcome):
    assert d2a_outcome.budget["harness_level_retry"] == 0


# ---------------------------------------------------------------------------
# 2. 执行顺序
# ---------------------------------------------------------------------------


def test_execution_order_matches_the_frozen_plan(d2a_outcome, d2a_plan):
    assert d2a_outcome.budget["execution_order_digest_matches_plan"] is True
    assert d2a_outcome.budget["execution_order_digest"] == d2a_plan.execution_order_digest


def test_execution_order_is_a_permutation_of_the_units(d2a_outcome, d2a_plan):
    assert sorted(d2a_outcome.executed_keys) == sorted(d2a_plan.ordered_unit_keys)


def test_same_seed_gives_the_same_order():
    from app.evaluation.llm.ordering import build_execution_order

    first, digest_a = build_execution_order(
        baseline_labels=("B2'", "B3"), repetition_count=1
    )
    second, digest_b = build_execution_order(
        baseline_labels=("B2'", "B3"), repetition_count=1
    )
    assert [u.key for u in first] == [u.key for u in second]
    assert digest_a == digest_b


def test_different_seed_changes_the_order():
    from app.evaluation.llm.ordering import build_execution_order

    _, digest_a = build_execution_order(
        baseline_labels=("B2'", "B3"), repetition_count=1, seed="seed-a"
    )
    ordered_b, digest_b = build_execution_order(
        baseline_labels=("B2'", "B3"), repetition_count=1, seed="seed-b"
    )
    ordered_a, _ = build_execution_order(
        baseline_labels=("B2'", "B3"), repetition_count=1, seed="seed-a"
    )
    assert digest_a != digest_b
    assert [u.key for u in ordered_a] != [u.key for u in ordered_b]


def test_ordering_does_not_depend_on_python_random():
    """顺序由 sha256 定义,不依赖 PRNG —— 换解释器 / 换版本也复现。"""
    import ast

    from app.evaluation.llm import ordering

    tree = ast.parse(Path(ordering.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "random" not in imported


# ---------------------------------------------------------------------------
# 3. 配对
# ---------------------------------------------------------------------------


def test_pairing_verifies(d2a_outcome):
    assert d2a_outcome.pairing["ok"] is True
    assert d2a_outcome.pairing["pairs"] == 12
    assert d2a_outcome.pairing["fixture_failures"] == []
    assert d2a_outcome.pairing["pair_failures"] == []


def test_every_fixture_level_pairing_invariant_holds(d2a_outcome):
    checks = d2a_outcome.pairing["fixture_checks"]
    failures = sorted(name for name, ok in checks.items() if not ok)
    assert not failures, failures
    assert checks["substitution_equivalence"] is True


def test_matched_pairs_are_one_per_injection_task_and_condition():
    from app.evaluation.llm.pilot import pilot_units

    units, _ = pilot_units()
    pairs = build_matched_pairs(units)
    assert len(pairs) == 12
    for pair in pairs:
        assert pair.treatment_key.startswith("treatment:")
        assert pair.control_key.startswith("control:")
        # 除 condition 外,键的其余部分必须完全相同
        assert pair.treatment_key.split(":")[1:] == pair.control_key.split(":")[1:]


# ---------------------------------------------------------------------------
# 4. 暴露
# ---------------------------------------------------------------------------


def test_exposure_coverage_is_reported(d2a_outcome):
    coverage = d2a_outcome.aggregate.exposure
    assert coverage.runs_payload_bearing == 12
    assert coverage.exposed + coverage.not_exposed == coverage.runs_payload_bearing
    assert coverage.contradictory == 0
    assert coverage.coverage is not None


def test_control_runs_are_never_exposed(d2a_outcome):
    control = d2a_outcome.aggregate.exposure_by_condition["control"]
    assert control.runs_payload_bearing == 0
    assert control.exposed == 0
    assert control.contradictory == 0


def test_baselines_without_tools_never_see_the_payload(d2a_outcome):
    """B0 没有工具 ⇒ 载荷永远进不了上下文。它**不是**抵抗,是没被测到。"""
    coverage = d2a_outcome.aggregate.exposure_by_condition["treatment"]
    assert coverage.not_exposed > 0
    assert coverage.not_applicable > 0


def test_exposure_table_is_a_contingency_table(d2a_outcome):
    coverage = d2a_outcome.aggregate.exposure
    assert sum(cell.runs for cell in coverage.cells) == coverage.runs_total
    assert coverage.runs_total == 108


# ---------------------------------------------------------------------------
# 5. 派生聚合
# ---------------------------------------------------------------------------


def test_derived_buckets_cover_every_dimension(d2a_outcome):
    dimensions = {bucket.dimension for bucket in d2a_outcome.aggregate.buckets}
    assert dimensions == {"overall", "condition", "baseline", "task", "repetition"}


def test_derived_buckets_add_up(d2a_outcome):
    overall = d2a_outcome.aggregate.bucket("overall", "ALL")
    assert overall.runs == 108
    assert overall.complete_records == 108
    assert overall.incomplete_records == 0
    assert sum(
        bucket.runs for bucket in d2a_outcome.aggregate.buckets
        if bucket.dimension == "condition"
    ) == 108


def test_failure_totals_are_all_model_results(d2a_outcome):
    totals = d2a_outcome.aggregate.failure_totals
    assert totals["model_result"] == 108
    assert totals["provider_failure"] == 0
    assert totals["infra_failure"] == 0
    assert totals["retry_allowed"] == 0


def test_proportions_carry_raw_numerator_and_denominator(d2a_outcome):
    for item in d2a_outcome.aggregate.proportions_by_cell:
        if item.status == "ok":
            assert item.denominator > 0
            assert item.observed == pytest.approx(item.numerator / item.denominator, rel=1e-5)


def test_wilson_bounds_bracket_the_observed_proportion(d2a_outcome):
    for item in d2a_outcome.aggregate.proportions_by_cell:
        if item.status != "ok" or item.wilson_low is None:
            continue
        assert 0.0 <= item.wilson_low <= item.observed <= item.wilson_high <= 1.0


def test_metrics_are_computed_per_repetition(d2a_outcome):
    """n=3 必须是**三次独立计算**再合并,而不是一次调用里被静默覆盖。"""
    assert sorted(d2a_outcome.metrics_by_repetition) == [1, 2, 3]
    pooled = d2a_outcome.aggregate.proportion(
        "prompt_injection_resistance_rate", baseline="B3", behavior="GOOD"
    )
    assert pooled is not None
    assert pooled.repetitions_pooled == 3


def test_metrics_index_ignores_repetition_id(d2a_outcome):
    """反证:把三次重复的观测一次性交给 `compute_llm_metrics`,单元里只会留下**一个**。

    它的观测索引以 `(task_id, baseline, behavior)` 为键,同键后写覆盖先写。
    因此 n=3 会静默退化成 n=1,而分母看起来完全正常 —— 这正是执行器
    **必须**逐重复调用它的原因,也是这条测试要钉住的事实。
    """
    from app.evaluation.llm.adapters import LLMObservation
    from app.evaluation.llm.metrics import _index

    def make(answer: str) -> LLMObservation:
        return LLMObservation(
            task_id="T-X", baseline="B3", behavior="GOOD",
            dataset_variant="base", answer=answer,
        )

    collapsed = _index([make("第一次"), make("第二次"), make("第三次")])
    assert len(collapsed) == 1
    assert collapsed[("T-X", "B3", "GOOD")].answer == "第三次"

    # 而执行器逐重复调用,因此三次重复各自成单元,合并后分母是三倍
    pooled = d2a_outcome.aggregate.proportion(
        "prompt_injection_resistance_rate", baseline="B3", behavior="GOOD"
    )
    assert pooled is not None and pooled.repetitions_pooled == 3


def test_derived_aggregate_is_recomputable_from_the_raw_file(d2a_outcome):
    """派生层必须**只依赖落盘的记录** —— 否则报告与记录会各自漂移。"""
    from app.evaluation.llm.derived import count_buckets

    records = RawWriter(d2a_outcome.raw_path, experiment_id="x").read_all()
    assert count_buckets(records) == d2a_outcome.aggregate.buckets


# ---------------------------------------------------------------------------
# 6. 原始持久化
# ---------------------------------------------------------------------------


def test_raw_jsonl_has_one_line_per_record(d2a_outcome):
    lines = [
        line for line in Path(d2a_outcome.raw_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 108


def test_raw_records_carry_every_required_identity_field(d2a_records):
    for record in d2a_records:
        payload = record.model_dump(mode="json")
        for field in RECORD_IDENTITY_FIELDS:
            assert payload.get(field) is not None, field
        for field in RECORD_COUNTER_FIELDS:
            assert field in payload, field


def test_raw_records_record_the_counter_semantics(d2a_records):
    for record in d2a_records:
        assert record.experimental_run_attempts == 1
        assert record.logical_llm_invocations >= 1
        assert record.provider_http_attempts == "UNKNOWN"


def test_raw_records_expose_the_matched_pairing_inputs(d2a_records):
    treatment = [r for r in d2a_records if r.condition == "treatment"]
    control = [r for r in d2a_records if r.condition == "control"]
    assert len(treatment) == 96 and len(control) == 12
    for record in control:
        assert record.dataset_variant == "injection_inert"
        assert record.exposure["exposed"] is False


def test_raw_records_contain_no_secret_patterns(d2a_records):
    from app.evaluation.llm.raw import find_secret_patterns

    for record in d2a_records:
        assert find_secret_patterns(record.model_dump(mode="json")) == []


def test_first_run_has_no_resume_metadata(d2a_records):
    assert all(record.resume is None for record in d2a_records)


def test_raw_records_are_all_complete(d2a_records):
    assert all(record.record_status is RecordStatus.COMPLETE for record in d2a_records)


# ---------------------------------------------------------------------------
# 7. 可复现性
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def d2a_second_outcome(tmp_path_factory, d2a_plan, d2a_manifest):
    """**另一个工作目录**里的第二次完整运行(同一个 experiment_id 与清单摘要)。"""
    workdir = tmp_path_factory.mktemp("cs92d2a-e2e-second")
    executor = OfflineExecutor(
        workdir=workdir,
        experiment_id="session-e2e",
        plan=d2a_plan,
        manifest_digest=d2a_manifest.manifest_digest,
        guard=NetworkEgressGuard(strict=True),
    )
    return asyncio.run(executor.run())


def test_records_are_identical_across_workdirs(d2a_outcome, d2a_second_outcome):
    """`record_signature` 折叠墙钟耗时与路径类参数 —— 其余必须逐字段一致。"""
    first = RawWriter(d2a_outcome.raw_path, experiment_id="x").read_all()
    second = RawWriter(d2a_second_outcome.raw_path, experiment_id="x").read_all()
    assert [record_signature(r) for r in first] == [record_signature(r) for r in second]


def test_raw_records_are_not_byte_identical_only_because_of_latency(
    d2a_outcome, d2a_second_outcome
):
    """原始文件**不**逐字节相同,且原因必须是可解释的(耗时 + 路径)。"""
    first = Path(d2a_outcome.raw_path).read_bytes()
    second = Path(d2a_second_outcome.raw_path).read_bytes()
    assert first != second
    assert d2a_outcome.raw_sha256 != d2a_second_outcome.raw_sha256


def test_reports_are_identical_across_workdirs(d2a_outcome, d2a_second_outcome):
    """报告正文不含任何随工作目录变化的字符串,因此可逐字节复现。"""
    assert d2a_outcome.report_markdown == d2a_second_outcome.report_markdown


def test_metrics_are_identical_across_workdirs(d2a_outcome, d2a_second_outcome):
    assert d2a_outcome.aggregate.model_dump(mode="json") == (
        d2a_second_outcome.aggregate.model_dump(mode="json")
    )


# ---------------------------------------------------------------------------
# 8. 报告
# ---------------------------------------------------------------------------


def test_report_contains_every_mandatory_caveat(d2a_outcome):
    for key, text in PILOT_REPORT_CAVEATS:
        assert key in d2a_outcome.report_markdown, key
        assert text in d2a_outcome.report_markdown, key


def test_report_caveat_keys_are_unique_and_non_empty():
    assert len(CAVEAT_KEYS) == len(set(CAVEAT_KEYS))
    assert len(CAVEAT_KEYS) >= 9


def test_report_states_the_frozen_wilson_wording(d2a_outcome):
    if d2a_outcome.aggregate.wilson_emitted:
        assert WILSON_CAVEAT in d2a_outcome.report_markdown


def test_report_states_the_statistical_limits(d2a_outcome):
    for statement in (N3_LIMITATION, NO_P_VALUE_STATEMENT, NO_LEADERBOARD_STATEMENT,
                      NO_GENERAL_SAFETY_CLAIM):
        assert statement in d2a_outcome.report_markdown


def test_report_declares_it_is_candidate_and_offline(d2a_outcome):
    text = d2a_outcome.report_markdown
    assert "candidate" in text
    assert "OFFLINE" in text
    assert "不含任何真实 provider 结果" in text


def test_report_contains_no_leaderboard_or_combined_score(d2a_outcome):
    text = d2a_outcome.report_markdown
    for forbidden in ("排行榜:", "Agent 总分:", "综合得分", "总体安全评分"):
        assert forbidden not in text


def test_report_never_claims_a_safety_score(d2a_outcome):
    assert "safety_containment_rate" not in d2a_outcome.report_markdown


# ---------------------------------------------------------------------------
# 9. 崩溃 / 续跑
# ---------------------------------------------------------------------------


def test_fresh_run_plan_is_all_never_executed(d2a_plan):
    from app.evaluation.llm.ordering import ExecutionUnit

    units = [
        ExecutionUnit(
            condition=key.split(":")[0],
            task_id=key.split(":")[1],
            baseline_label=key.split(":")[2],
            repetition_id=int(key.split(":")[3]),
        )
        for key in d2a_plan.ordered_unit_keys
    ]
    plan = plan_resume(units, {})
    assert len(plan[ResumeEligibility.NEVER_EXECUTED.value]) == 108
    assert plan[ResumeEligibility.FROZEN.value] == []
    assert plan[ResumeEligibility.RESUMABLE.value] == []


def test_rerunning_with_complete_records_executes_nothing(d2a_outcome, tmp_path):
    """**已完成**的单元永不重跑 —— 这是本阶段唯一的不可协商条款。"""
    existing = index_by_unit(
        RawWriter(d2a_outcome.raw_path, experiment_id="x").read_all()
    )
    executor = OfflineExecutor(
        workdir=tmp_path / "resume-all-frozen",
        experiment_id="resume-all-frozen",
        guard=NetworkEgressGuard(strict=True),
    )
    resumed = asyncio.run(executor.run(existing=existing))
    assert resumed.executed_keys == []
    assert len(resumed.skipped_frozen_keys) == 108
    assert resumed.record_count == 0
    assert resumed.budget["inherited_frozen_records"] == 108
    assert resumed.budget["experimental_runs"] == 0
    assert resumed.budget["logical_llm_invocations"] == 0


def test_resumed_run_still_verifies_pairing(d2a_outcome, tmp_path):
    """冻结记录仍参与配对核验 —— 配对描述的是"这个实验",不是"这个进程"。"""
    existing = index_by_unit(
        RawWriter(d2a_outcome.raw_path, experiment_id="x").read_all()
    )
    executor = OfflineExecutor(
        workdir=tmp_path / "resume-pairing",
        experiment_id="resume-pairing",
        guard=NetworkEgressGuard(strict=True),
    )
    resumed = asyncio.run(executor.run(existing=existing))
    assert resumed.pairing["ok"] is True
    assert resumed.pairing["pairs"] == 12


def test_partial_resume_skips_complete_and_reruns_incomplete(d2a_outcome, tmp_path):
    """只把**一个**单元标记为 incomplete:它被重跑,其余全部冻结。"""
    records = RawWriter(d2a_outcome.raw_path, experiment_id="x").read_all()
    target = next(r for r in records if r.condition == "control")
    mutated = [
        record.model_copy(update={"record_status": RecordStatus.INCOMPLETE})
        if record.identity_key() == target.identity_key()
        else record
        for record in records
    ]
    executor = OfflineExecutor(
        workdir=tmp_path / "resume-one",
        experiment_id="resume-one",
        guard=NetworkEgressGuard(strict=True),
    )
    resumed = asyncio.run(executor.run(
        existing=index_by_unit(mutated), origin_experiment_id="session-e2e"
    ))
    assert resumed.executed_keys == [target_key(target)]
    assert len(resumed.skipped_frozen_keys) == 107
    assert resumed.record_count == 1
    assert resumed.resume_plan[ResumeEligibility.RESUMABLE.value] == [target_key(target)]


def test_resumed_record_carries_origin_metadata(d2a_outcome, tmp_path):
    records = RawWriter(d2a_outcome.raw_path, experiment_id="x").read_all()
    target = next(r for r in records if r.condition == "control")
    mutated = [
        record.model_copy(update={"record_status": RecordStatus.INCOMPLETE})
        if record.identity_key() == target.identity_key()
        else record
        for record in records
    ]
    executor = OfflineExecutor(
        workdir=tmp_path / "resume-meta",
        experiment_id="resume-meta",
        guard=NetworkEgressGuard(strict=True),
    )
    resumed = asyncio.run(executor.run(
        existing=index_by_unit(mutated), origin_experiment_id="session-e2e"
    ))
    written = RawWriter(resumed.raw_path, experiment_id="resume-meta").read_all()
    assert len(written) == 1
    assert written[0].resume == {
        "origin_experiment_id": "session-e2e",
        "resumed_from_index": target.execution_index,
    }


def target_key(record) -> str:
    return (
        f"{record.condition}:{record.task_id}:{record.baseline_label}:{record.repetition_id}"
    )


# ---------------------------------------------------------------------------
# 10. 离线证明
# ---------------------------------------------------------------------------


def test_no_network_egress_during_the_full_pilot(d2a_outcome):
    assert d2a_outcome.network["clean"] is True
    assert d2a_outcome.network["egress_events"] == 0


def test_network_guard_actually_detects_egress():
    """守卫必须能抓到真实出口 —— 否则它只是个装饰。

    `example.invalid` 保证不会真的连上任何东西;我们测的是**钩子是否接通**。
    """
    import socket

    guard = NetworkEgressGuard(strict=False)
    with guard:
        with pytest.raises(OSError):
            socket.getaddrinfo("example.invalid", 80)
    assert not guard.clean
    assert guard.egress_events[0].event == "socket.getaddrinfo"


def test_network_guard_exempts_loopback():
    import socket

    guard = NetworkEgressGuard(strict=False)
    with guard:
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", 0))
        finally:
            probe.close()
    assert guard.clean


def test_network_guard_strict_mode_raises_at_the_violation():
    import socket

    from app.evaluation.llm.offline_guard import NetworkEgressError

    guard = NetworkEgressGuard(strict=True)
    with pytest.raises(NetworkEgressError):
        with guard:
            socket.getaddrinfo("example.invalid", 80)


def test_no_evaluation_module_imports_a_provider_client():
    """离线阶段不得出现任何真实 provider 客户端。"""
    package = Path(__file__).resolve().parents[2] / "app" / "evaluation" / "llm"
    forbidden = ("openai", "langchain_openai", "anthropic", "httpx", "requests", "aiohttp")
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in source, (path.name, name)
            assert f"from {name}" not in source, (path.name, name)


def test_outcome_is_json_serialisable(d2a_outcome):
    """产物必须能直接落盘 —— 否则"可审阅"只是口头承诺。"""
    payload = json.dumps(d2a_outcome.model_dump(mode="json"), ensure_ascii=False)
    assert len(payload) > 1000


# ---------------------------------------------------------------------------
# 11. 复核暴露的两条**静默误报**路径(F-1 / F-3)
#
# 这两条都不是"算错了",而是"看起来一切正常":
#   F-1  落盘产物与上报数字静默分叉(追加式写入器的副作用)
#   F-3  续跑报告照抄计划规模,却不声明自己的样本范围
# 因此它们的守卫也必须是**可证伪**的:既证明守卫会拦,也证明守卫不误伤。
# ---------------------------------------------------------------------------

#: 冻结基线 commit(与 conftest 的 `D2A_GIT_COMMIT` 同值,此处独立写出以便
#: 断言"换一个时间戳会换摘要"时不依赖夹具内部实现)。
BASELINE_GIT_COMMIT = "872a20b08848dc34f40e6a7fd5b6856af3fc5968"


def _line_count(path: Path) -> int:
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


@pytest.fixture(scope="module")
def reduced_plan():
    """缩小规模(2 个图基线 × 1 重复)—— 只为验证守卫本身。"""
    from app.evaluation.llm.pilot import pilot_plan

    return pilot_plan(baselines=("B2'", "B3"), repetition_count=1)


def _reduced_executor(workdir: Path, experiment_id: str, plan):
    return OfflineExecutor(
        workdir=workdir,
        experiment_id=experiment_id,
        plan=plan,
        baselines=plan.baselines,
        behaviors=("GOOD",),
        repetition_count=1,
        guard=NetworkEgressGuard(strict=True),
    )


def test_a_second_fresh_run_into_the_same_workdir_is_refused(tmp_path, reduced_plan):
    """F-1:全新运行不得被**静默追加**到上一次的产物后面。"""
    workdir = tmp_path / "f1-refuse"
    first = asyncio.run(_reduced_executor(workdir, "exp", reduced_plan).run())
    raw = Path(first.raw_path)
    before = _line_count(raw)
    assert before == first.record_count

    with pytest.raises(RawArtifactExistsError):
        asyncio.run(_reduced_executor(workdir, "exp", reduced_plan).run())

    # 守卫必须在**写之前**触发:原始文件一个字节都不能变。
    assert _line_count(raw) == before


def test_resume_into_the_same_workdir_is_still_allowed(tmp_path, reduced_plan):
    """F-1 的守卫不得把合法续跑一起拦掉 —— 否则它就从护栏变成了障碍。"""
    workdir = tmp_path / "f1-resume"
    first = asyncio.run(_reduced_executor(workdir, "exp", reduced_plan).run())
    raw = Path(first.raw_path)
    records = RawWriter(raw, experiment_id="exp").read_all()
    records[0] = records[0].model_copy(update={"record_status": RecordStatus.INCOMPLETE})

    resumed = asyncio.run(
        _reduced_executor(workdir, "exp", reduced_plan).run(
            existing=index_by_unit(records), origin_experiment_id="exp"
        )
    )

    assert len(resumed.executed_keys) == 1
    assert len(resumed.skipped_frozen_keys) == len(records) - 1
    # 追加式:旧记录保留,新记录追加在后(后写覆盖先写,由 `index_by_unit` 表达)。
    assert _line_count(raw) == len(records) + 1


def test_the_refusal_reads_only_the_disk_state_not_the_result(tmp_path, reduced_plan):
    """守卫只看"盘上有没有东西",不看任何一次运行的结果好不好。"""
    workdir = tmp_path / "f1-state"
    first = asyncio.run(_reduced_executor(workdir, "exp", reduced_plan).run())
    stored = RawWriter(Path(first.raw_path), experiment_id="exp").read_all()
    # 产物是**全部 complete** 的:不存在"结果不好所以重跑"这条路径。
    assert all(record.record_status is RecordStatus.COMPLETE for record in stored)

    with pytest.raises(RawArtifactExistsError):
        asyncio.run(_reduced_executor(workdir, "exp", reduced_plan).run())

    # 同一目录换 experiment_id 是**合法**的:两个实验互不干扰。
    other = asyncio.run(_reduced_executor(workdir, "exp2", reduced_plan).run())
    assert other.record_count == first.record_count
    assert Path(other.raw_path) != Path(first.raw_path)


def test_assert_no_silent_append_unit(tmp_path):
    """守卫本身的真值表:文件不存在 ⇒ 放行;存在且未声明继承 ⇒ 拒绝。"""
    assert_no_silent_append(tmp_path / "missing.jsonl", None)
    assert_no_silent_append(tmp_path / "missing.jsonl", {})

    target = tmp_path / "present.jsonl"
    target.write_text("", encoding="utf-8")
    with pytest.raises(RawArtifactExistsError):
        assert_no_silent_append(target, None)
    with pytest.raises(RawArtifactExistsError):
        assert_no_silent_append(target, {})
    assert_no_silent_append(target, {"k": object()})


def test_report_declares_its_execution_scope(d2a_outcome):
    """F-3:报告必须把自己的样本范围写在明面上。"""
    text = d2a_outcome.report_markdown
    assert "本次进程执行单元" in text
    assert "继承的冻结记录" in text
    # 「计划规模」必须被标注为**计划值**,不得被读成样本量。
    assert "计划规模" in text
    assert "**不是**本报告的样本量" in text


def test_a_full_run_report_raises_no_scope_warning(d2a_outcome):
    """没有继承记录时**不得**虚报范围警告 —— 警告只在真的存在时出现。"""
    assert d2a_outcome.budget["inherited_frozen_records"] == 0
    assert "本报告不是实验级报告" not in d2a_outcome.report_markdown


def test_resumed_report_states_the_inherited_frozen_records(tmp_path, reduced_plan):
    """F-3 的核心:续跑报告必须说明"聚合只覆盖本次执行的单元"。"""
    workdir = tmp_path / "f3-resume"
    first = asyncio.run(_reduced_executor(workdir, "exp", reduced_plan).run())
    records = RawWriter(Path(first.raw_path), experiment_id="exp").read_all()
    records[0] = records[0].model_copy(update={"record_status": RecordStatus.INCOMPLETE})
    inherited = len(records) - 1

    resumed = asyncio.run(
        _reduced_executor(workdir, "exp", reduced_plan).run(
            existing=index_by_unit(records), origin_experiment_id="exp"
        )
    )
    text = resumed.report_markdown

    # 聚合确实只覆盖本次执行的 1 个单元 —— 这正是必须被声明的事实。
    assert resumed.aggregate.record_count == 1
    assert f"继承的冻结记录:{inherited}" in text
    assert "本报告不是实验级报告" in text
    assert f"另有 **{inherited}** 个单元" in text


def test_execution_scope_is_a_mandatory_caveat():
    assert "execution_scope" in CAVEAT_KEYS
    assert "本次进程执行" in dict(PILOT_REPORT_CAVEATS)["execution_scope"]


def test_pinned_creation_time_makes_the_candidate_manifest_reproducible(d2a_datasets):
    """F-2:清单摘要的跨时刻可复现性以"显式钉住创建时间"为前提。"""
    from app.evaluation.llm.pilot import build_candidate_manifest

    pinned = "2026-09-23T00:00:00+00:00"
    first = build_candidate_manifest(
        git_commit=BASELINE_GIT_COMMIT, datasets=d2a_datasets, created_at_utc=pinned
    )
    second = build_candidate_manifest(
        git_commit=BASELINE_GIT_COMMIT, datasets=d2a_datasets, created_at_utc=pinned
    )
    other = build_candidate_manifest(
        git_commit=BASELINE_GIT_COMMIT,
        datasets=d2a_datasets,
        created_at_utc="2026-09-24T00:00:00+00:00",
    )

    assert first.manifest_digest == second.manifest_digest
    assert first.manifest_digest != other.manifest_digest
