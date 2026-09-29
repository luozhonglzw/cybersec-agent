"""Phase 9.2-D-2d 资源预算 schema 的**变异 / 阴性对照**测试。

这个文件回答一个问题:

    上面那套聚焦测试,到底有没有牙?

做法是给每一条**坏行为**构造一个**脚本化变异**(全部通过 `monkeypatch`
在内存里应用,**不改仓库里的任何文件**),然后跑同一条断言。若断言在变异下
依然通过,那条断言就是**恒真的** —— 它没有测到任何东西。

八条变异 + 一条阴性对照:

    01  冻结检查忽略"未决定的资源维度"
    02  `UNRESOLVED` 被算作"已决定"
    03  D-2d 计划忽略 `sdk_max_retries`(清单退回历史值 972)
    04  输出**分量**包络被当成 total-token budget
    05  `UNKNOWN` 被转成 0
    06  冻结闸门跳过"执行不变量"核对
    07  冻结闸门跳过 provider 身份 / 资源声明交叉核对
    08  请求 `frozen` 不再 fail closed(跳过冻结闸门)
    09  阴性对照:无操作变异**不得**被判为"检出"

**全程离线**:零 provider 调用、零网络出口、零凭据。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

from app.evaluation import pilot_config as PC
from app.evaluation.llm import budget as B
from app.evaluation.llm import pilot as Pilot
from app.evaluation.llm import protocol as P
from app.evaluation.llm.budget import BudgetCounters
from app.evaluation.llm.protocol import (
    BudgetBoundedness,
    ManifestError,
    ResourceBudget,
    verify_manifest,
)

GIT_COMMIT = "dfe7ea4b677c74b6e60ed40f20fdc42b4988735b"
CREATED_AT = "2026-09-29T00:00:00+00:00"


# ---------------------------------------------------------------------------
# 夹具构造(仓库外)
# ---------------------------------------------------------------------------


def _datasets(tmp_path_factory: Any) -> dict[str, dict[str, str]]:
    return Pilot.load_datasets(tmp_path_factory.mktemp("d2d-mut-fixtures"))


def _candidate(tmp_path_factory: Any):
    return Pilot.build_candidate_manifest(
        git_commit=GIT_COMMIT,
        datasets=_datasets(tmp_path_factory),
        created_at_utc=CREATED_AT,
    )


def _frozen(tmp_path_factory: Any):
    from app.evaluation.d2d_pilot import build_d2d_manifest

    return build_d2d_manifest(
        git_commit=GIT_COMMIT,
        datasets=_datasets(tmp_path_factory),
        created_at_utc=CREATED_AT,
    )


# ---------------------------------------------------------------------------
# 探针(与聚焦测试里的断言同源)
# ---------------------------------------------------------------------------


def probe_freeze_check_covers_budget_dimensions(tmp_path_factory) -> None:
    """冻结检查必须覆盖"未决定的资源维度",不能只看占位符。"""
    candidate = _candidate(tmp_path_factory)
    payload = candidate.model_dump(mode="json")
    payload["manifest_status"] = "frozen"
    payload["provider"] = "deepseek"
    payload["model"] = "deepseek-flash"
    payload["endpoint_category"] = "OPENAI_COMPATIBLE"
    promoted = P.seal_manifest(P.ExperimentManifest.model_validate(payload))
    with pytest.raises(ManifestError):
        verify_manifest(promoted, require_frozen=True)


def probe_unresolved_is_not_decided(tmp_path_factory) -> None:
    """`UNRESOLVED` **不得**被算作已决定。"""
    assert P.budget_freeze_blockers(_candidate(tmp_path_factory)) != []
    assert BudgetBoundedness.UNRESOLVED not in P.DECIDED_BOUNDEDNESS


def probe_d2d_plan_uses_sdk_max_retries_zero(tmp_path_factory) -> None:
    """D-2d 清单的配置包络必须是 324,不是历史值 972。"""
    assert PC.d2d_plan().provider_http_attempt_ceiling == 324
    assert PC.d2d_plan().sdk_max_retries_assumption == 0
    assert _frozen(tmp_path_factory).provider_http_attempt_ceiling == 324


def probe_output_envelope_is_not_a_total_token_budget(tmp_path_factory) -> None:
    """输出**分量**包络**不得**被当成 total-token budget。"""
    declaration = PC.token_budget()
    assert declaration.value is None
    assert declaration.boundedness is (
        BudgetBoundedness.NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL
    )
    assert declaration.envelopes[0].scope == "output"


def probe_unknown_is_not_zero(tmp_path_factory) -> None:
    """不可观测记 `UNKNOWN`,**不是** 0。"""
    counters = BudgetCounters(provider_http_attempts_unobservable=1)
    assert counters.provider_http_attempts == "UNKNOWN"
    assert counters.provider_http_attempts != 0


def probe_gate_checks_execution_invariants(tmp_path_factory) -> None:
    """闸门必须核对"清单 vs 冻结计划"的执行不变量。"""
    frozen = _frozen(tmp_path_factory)
    with pytest.raises(ManifestError, match="provider_http_attempt_ceiling"):
        Pilot.assert_frozen_pilot_manifest(
            frozen, datasets=_datasets(tmp_path_factory), expected_git_commit=GIT_COMMIT
        )


def probe_gate_checks_identity_and_budgets(tmp_path_factory) -> None:
    """闸门必须交叉核对 provider 身份与资源声明。"""
    frozen = _frozen(tmp_path_factory)
    with pytest.raises(ManifestError, match="provider"):
        Pilot.assert_frozen_pilot_manifest(
            frozen,
            datasets=_datasets(tmp_path_factory),
            expected_git_commit=GIT_COMMIT,
            plan=PC.d2d_plan(),
            expected_identity={"provider": "someone-else"},
        )


def probe_frozen_request_fails_closed(tmp_path_factory) -> None:
    """要 `frozen` 就必须当场通过冻结闸门 —— 未决定的预算必须当场炸掉。"""
    with pytest.raises(ManifestError):
        Pilot.build_candidate_manifest(
            git_commit=GIT_COMMIT,
            datasets=_datasets(tmp_path_factory),
            created_at_utc=CREATED_AT,
            manifest_status="frozen",
        )


# ---------------------------------------------------------------------------
# 变异
# ---------------------------------------------------------------------------


def _mutate_freeze_check_ignores_budgets(mp) -> None:
    mp.setattr(P, "budget_freeze_blockers", lambda manifest: [])


def _mutate_unresolved_counts_as_decided(mp) -> None:
    mp.setattr(
        P,
        "DECIDED_BOUNDEDNESS",
        frozenset(P.DECIDED_BOUNDEDNESS | {BudgetBoundedness.UNRESOLVED}),
    )


def _mutate_plan_ignores_retry_assumption(mp) -> None:
    original = Pilot.pilot_budget

    def ignoring(*, baseline_labels, repetition_count, tasks=None, sdk_max_retries=2):
        kwargs = {"baseline_labels": baseline_labels, "repetition_count": repetition_count}
        if tasks is not None:
            kwargs["tasks"] = tasks
        return original(**kwargs)

    mp.setattr(Pilot, "pilot_budget", ignoring)


def _mutate_envelope_becomes_total_bound(mp) -> None:
    original = PC.token_budget

    def inflated():
        declaration = original()
        return declaration.model_copy(
            update={
                "boundedness": BudgetBoundedness.NUMERICALLY_BOUNDED,
                "value": declaration.envelopes[0].value,
                "unit": "tokens",
                "provenance": "envelope promoted to a total bound",
            }
        )

    mp.setattr(PC, "token_budget", inflated)


def _mutate_unknown_becomes_zero(mp) -> None:
    mp.setattr(B, "UNKNOWN", 0)


def _mutate_gate_skips_execution_invariants(mp) -> None:
    def weakened(
        manifest,
        *,
        datasets,
        expected_git_commit,
        plan=None,
        tasks=None,
        expected_identity=None,
        expected_budgets=None,
    ) -> None:
        verify_manifest(
            manifest, require_frozen=True, expected_git_commit=expected_git_commit
        )

    mp.setattr(Pilot, "assert_frozen_pilot_manifest", weakened)


def _mutate_gate_skips_cross_checks(mp) -> None:
    def weakened(
        manifest,
        *,
        datasets,
        expected_git_commit,
        plan=None,
        tasks=None,
        expected_identity=None,
        expected_budgets=None,
    ) -> None:
        verify_manifest(
            manifest, require_frozen=True, expected_git_commit=expected_git_commit
        )
        resolved = plan or Pilot.pilot_plan()
        for field, expected in resolved.as_manifest_fields().items():
            if getattr(manifest, field) != expected:
                raise ManifestError(f"{field} 不一致")

    mp.setattr(Pilot, "assert_frozen_pilot_manifest", weakened)


def _mutate_frozen_request_skips_the_gate(mp) -> None:
    mp.setattr(Pilot, "verify_manifest", lambda *args, **kwargs: None)


def _mutate_nothing(mp) -> None:  # noqa: ARG001 —— 阴性对照:刻意什么都不做
    return None


@dataclass(frozen=True)
class Mutation:
    key: str
    name: str
    apply: Callable[[Any], None]
    probe: Callable[..., None]
    note: str
    needs_tmp: bool = True


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        key="01",
        name="冻结检查忽略未决定的资源维度",
        apply=_mutate_freeze_check_ignores_budgets,
        probe=probe_freeze_check_covers_budget_dimensions,
        note="'冻结'于是只表示'没有占位符',而资源边界从未被回答",
    ),
    Mutation(
        key="02",
        name="UNRESOLVED 被算作已决定",
        apply=_mutate_unresolved_counts_as_decided,
        probe=probe_unresolved_is_not_decided,
        note="'还没决定'被读成'已决定',未定的预算因此可以冻结",
    ),
    Mutation(
        key="03",
        name="D-2d 计划忽略 sdk_max_retries",
        apply=_mutate_plan_ignores_retry_assumption,
        probe=probe_d2d_plan_uses_sdk_max_retries_zero,
        note="清单退回历史值 972 —— 两个互相矛盾的假设同时存在",
    ),
    Mutation(
        key="04",
        name="输出分量包络被当成总量上界",
        apply=_mutate_envelope_becomes_total_bound,
        probe=probe_output_envelope_is_not_a_total_token_budget,
        note="331776 从'输出分量'变成'total token 上界',而数字一个没变",
    ),
    Mutation(
        key="05",
        name="UNKNOWN 被转成 0",
        apply=_mutate_unknown_becomes_zero,
        probe=probe_unknown_is_not_zero,
        note="'不可观测'被伪装成'观测到零消耗'",
    ),
    Mutation(
        key="06",
        name="冻结闸门跳过执行不变量",
        apply=_mutate_gate_skips_execution_invariants,
        probe=probe_gate_checks_execution_invariants,
        note="'冻结了清单'与'按计划执行'变成两件互不相干的事",
    ),
    Mutation(
        key="07",
        name="冻结闸门跳过身份与资源交叉核对",
        apply=_mutate_gate_skips_cross_checks,
        probe=probe_gate_checks_identity_and_budgets,
        note="清单里的 provider 可以不是被授权的那一个",
    ),
    Mutation(
        key="08",
        name="请求 frozen 不再 fail closed",
        apply=_mutate_frozen_request_skips_the_gate,
        probe=probe_frozen_request_fails_closed,
        note="能产出'标着 frozen 但其实还没准备好'的清单",
    ),
    Mutation(
        key="09",
        name="阴性对照:无操作变异",
        apply=_mutate_nothing,
        probe=probe_unresolved_is_not_decided,
        note="若这个被判为'检出',说明变异工装本身在误报",
    ),
)


# ---------------------------------------------------------------------------
# 变异工装
# ---------------------------------------------------------------------------


def _detect(mutation: Mutation, tmp_path_factory) -> bool:
    """应用变异 → 跑探针。探针失败 ⇒ 检出。

    捕获 `BaseException` 而不是 `Exception`:pytest 的断言失败异常
    (`Failed`)继承自 `BaseException`;只捕获 `Exception` 会让一次**成功的
    检出**从判定里逃出去,表现成测试错误 —— 检出率于是虚低。
    """
    with pytest.MonkeyPatch.context() as mp:
        mutation.apply(mp)
        try:
            if mutation.needs_tmp:
                mutation.probe(tmp_path_factory)
            else:
                mutation.probe()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # noqa: BLE001 —— 见 docstring
            return True
    return False


def test_00_the_probes_pass_without_mutation(tmp_path_factory):
    """先证明探针本身在**未变异**时全部通过 —— 否则"检出"毫无意义。"""
    for mutation in MUTATIONS:
        try:
            if mutation.needs_tmp:
                mutation.probe(tmp_path_factory)
            else:
                mutation.probe()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 —— 失败原因需要原样报告
            pytest.fail(
                f"探针 {mutation.key}({mutation.name})在未变异时就失败了:"
                f"{type(exc).__name__}: {exc}"
            )


def test_01_the_negative_control_is_not_reported_as_detected(tmp_path_factory):
    negative = next(m for m in MUTATIONS if m.key == "09")
    assert _detect(negative, tmp_path_factory) is False


@pytest.mark.parametrize(
    "mutation",
    [m for m in MUTATIONS if m.key != "09"],
    ids=lambda m: f"{m.key}-{m.name}",
)
def test_02_each_mutation_is_detected(mutation, tmp_path_factory):
    """每一条坏行为都必须被套件检出。"""
    assert _detect(mutation, tmp_path_factory), (
        f"变异 {mutation.key}({mutation.name})未被检出 —— "
        f"对应断言是恒真的。{mutation.note}"
    )


def test_03_report_every_mutation_and_detection_status(tmp_path_factory, capsys):
    """把每个变异与检出状态打出来 —— 结论必须可复核,不能只写在散文里。"""
    rows: list[tuple[str, str, str]] = []
    for mutation in MUTATIONS:
        detected = _detect(mutation, tmp_path_factory)
        rows.append(
            (mutation.key, mutation.name, "DETECTED" if detected else "not detected")
        )
    with capsys.disabled():
        print("\nD-2d manifest budget schema mutation report")
        print(f"{'#':<4}{'变异':<40}{'状态':<14}说明")
        for key, name, status in rows:
            note = next(m.note for m in MUTATIONS if m.key == key)
            print(f"{key:<4}{name:<40}{status:<14}{note}")
    undetected = [row for row in rows if row[2] == "not detected"]
    assert len(undetected) == 1
    assert undetected[0][0] == "09"
    assert len(rows) == len(MUTATIONS) == 9


def test_04_a_non_numeric_budget_may_not_carry_a_number_in_the_manifest():
    """补充不变式(不经变异):非数值状态携带数值在构造期就被拒绝。"""
    with pytest.raises(Exception):
        ResourceBudget(
            boundedness=BudgetBoundedness.NO_NUMERIC_BOUND_NO_PROVENANCE,
            value=1,
            reason="x",
        )
