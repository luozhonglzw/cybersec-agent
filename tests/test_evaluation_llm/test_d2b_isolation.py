"""Phase 9.2-D-2b —— 实验身份隔离:校准记录**不得**冻结试点单元。

问题(设计阶段发现的身份碰撞)
----------------------------
记录的单元身份键是

    (condition, task_id, baseline_label, repetition_id)

—— **不含实验**。于是 D-2c 校准实验里某个同名单元的完整记录,会被试点实验
当成"已完成"而冻结继承:那个单元**再也不会被执行**,而报告看起来完全正常。

这是"静默少测了一个单元",与 F-1 / F-3 同类:不报错,只是数字变了。

修法
----
把 `experiment_id` 变成 eligibility / loading **边界的一部分**,并且
**fail closed**:跨实验的记录直接拒绝,而不是静默忽略 ——
静默忽略会让"传错了文件"这种错误永远不被发现。

本文件同时给出**负对照**:同实验的合法续跑必须照常工作,否则这条守卫
就从护栏变成了障碍。
"""
import asyncio
from pathlib import Path

import pytest

from app.evaluation.llm.dataset import VARIANT_BASE
from app.evaluation.llm.executor import OfflineExecutor, plan_resume
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.ordering import ExecutionUnit
from app.evaluation.llm.pilot import pilot_plan
from app.evaluation.llm.protocol import PROTOCOL_VERSION
from app.evaluation.llm.raw import (
    ForeignExperimentRecord,
    RawRecord,
    RawWriter,
    RecordStatus,
    ResumeEligibility,
    index_by_unit,
    is_resumable,
    resume_eligibility,
)


@pytest.fixture(autouse=True)
def _no_egress():
    guard = NetworkEgressGuard(strict=False)
    with guard:
        yield
    assert guard.clean, f"D-2b 隔离测试期间发生网络出口:{guard.egress_events}"


PILOT = "pilot-2026-09"
CALIBRATION = "calibration-2026-09"
UNIT = ExecutionUnit(
    condition="treatment",
    task_id="T-BRUTEFORCE-01",
    baseline_label="B3",
    repetition_id=1,
)


def _record(experiment_id: str, *, status=RecordStatus.COMPLETE, **overrides) -> RawRecord:
    payload = {
        "run_id": f"{experiment_id}:0000",
        "experiment_id": experiment_id,
        "protocol_version": PROTOCOL_VERSION,
        "manifest_digest": "d",
        "task_id": "T-BRUTEFORCE-01",
        "condition": "treatment",
        "baseline_label": "B3",
        "dataset_variant": VARIANT_BASE,
        "repetition_id": 1,
        "execution_index": 0,
        "experimental_run_attempts": 1,
        "logical_llm_invocations": 2,
        "provider_http_attempts": "UNKNOWN",
        "record_status": status,
        "failure": {"failure_class": "MODEL_ANSWER", "count_as_model_result": True},
    }
    payload.update(overrides)
    return RawRecord(**payload)


def _reduced_executor(workdir, experiment_id: str):
    baselines = ("B0-shared", "B2'")
    return OfflineExecutor(
        workdir=workdir,
        experiment_id=experiment_id,
        plan=pilot_plan(baselines=baselines, repetition_count=1),
        baselines=baselines,
        behaviors=("GOOD",),
        repetition_count=1,
        guard=NetworkEgressGuard(strict=True),
    )


# ---------------------------------------------------------------------------
# 14. 校准记录不能满足试点的完成身份
# ---------------------------------------------------------------------------


def test_14_a_calibration_record_cannot_satisfy_pilot_identity():
    """同名的校准完整记录**不是**试点的完成状态。"""
    calibration = _record(CALIBRATION)

    with pytest.raises(ForeignExperimentRecord):
        resume_eligibility(calibration, experiment_id=PILOT)
    with pytest.raises(ForeignExperimentRecord):
        is_resumable(calibration, experiment_id=PILOT)
    with pytest.raises(ForeignExperimentRecord):
        index_by_unit([calibration], experiment_id=PILOT)
    with pytest.raises(ForeignExperimentRecord):
        plan_resume([UNIT], {calibration.identity_key(): calibration}, experiment_id=PILOT)


def test_14b_the_guard_does_not_rely_on_the_caller_prefiltering():
    """把校准记录**混进**同一批里,试点侧必须拒绝而不是静默丢弃。

    静默丢弃会让"传错了文件"永远不被发现 —— 而它的后果恰恰是
    "某个单元静默地没被测到"。
    """
    mixed = [_record(PILOT), _record(CALIBRATION, execution_index=1)]
    with pytest.raises(ForeignExperimentRecord):
        index_by_unit(mixed, experiment_id=PILOT)


def test_14c_a_matching_unit_key_alone_does_not_freeze_the_pilot_unit():
    """**只有** `condition` / `task_id` / `baseline` / `repetition` 匹配,不足以冻结。"""
    calibration = _record(CALIBRATION)
    assert calibration.identity_key() == (
        UNIT.condition,
        UNIT.task_id,
        UNIT.baseline_label,
        UNIT.repetition_id,
    ), "本用例的前提是单元键**完全相同** —— 否则它证明不了任何东西"

    with pytest.raises(ForeignExperimentRecord):
        plan_resume([UNIT], {calibration.identity_key(): calibration}, experiment_id=PILOT)


def test_14d_an_incomplete_calibration_record_is_also_foreign():
    """`INCOMPLETE` 的跨实验记录同样必须被拒绝 —— 不能靠"它不完整"混过去。"""
    crashed = _record(CALIBRATION, status=RecordStatus.INCOMPLETE)
    with pytest.raises(ForeignExperimentRecord):
        resume_eligibility(crashed, experiment_id=PILOT)


# ---------------------------------------------------------------------------
# 15. 跨实验不得污染续跑(端到端)
# ---------------------------------------------------------------------------


def test_15_a_different_experiment_cannot_contaminate_resume(tmp_path):
    """端到端:把校准实验的记录喂给试点执行器 ⇒ 在**加载边界**就被拒绝。"""
    calibration_dir = tmp_path / "calibration"
    pilot_dir = tmp_path / "pilot"

    calibration_run = asyncio.run(
        _reduced_executor(calibration_dir, CALIBRATION).run()
    )
    calibration_records = RawWriter(
        Path(calibration_run.raw_path), experiment_id=CALIBRATION
    ).read_all()
    assert len(calibration_records) == 18

    foreign = index_by_unit(calibration_records, experiment_id=CALIBRATION)
    assert len(foreign) == 18

    pilot = _reduced_executor(pilot_dir, PILOT)
    with pytest.raises(ForeignExperimentRecord):
        asyncio.run(pilot.run(existing=foreign))

    # 守卫在**写之前**触发:试点目录里一个字节都没有落盘。
    assert not (pilot_dir / "raw").exists(), (
        "跨实验拒绝必须发生在加载边界 —— 落盘之后再拒绝会留下半份产物"
    )


def test_15b_the_two_experiments_do_not_interfere(tmp_path):
    """负对照:两个实验各自跑,互不影响(守卫不能把正常情形一起拦掉)。"""
    first = asyncio.run(_reduced_executor(tmp_path / "a", CALIBRATION).run())
    second = asyncio.run(_reduced_executor(tmp_path / "b", PILOT).run())

    assert first.record_count == second.record_count == 18
    assert Path(first.raw_path) != Path(second.raw_path)
    assert first.budget["inherited_experimental_runs"] == 0
    assert second.budget["inherited_experimental_runs"] == 0


# ---------------------------------------------------------------------------
# 16. 同实验的合法续跑仍然工作
# ---------------------------------------------------------------------------


def test_16_a_same_experiment_resume_still_works(tmp_path):
    """负对照 —— 没有它,`test_14` 的守卫可能是"永远拒绝"。"""
    workdir = tmp_path / "resume"
    first = asyncio.run(_reduced_executor(workdir, PILOT).run())
    records = RawWriter(Path(first.raw_path), experiment_id=PILOT).read_all()

    target = records[0]
    records[0] = target.model_copy(update={"record_status": RecordStatus.INCOMPLETE})
    frozen = [r for r in records if r.record_status is RecordStatus.COMPLETE]

    resumed = asyncio.run(
        _reduced_executor(workdir, PILOT).run(
            existing=index_by_unit(records, experiment_id=PILOT),
            origin_experiment_id=PILOT,
        )
    )

    assert len(resumed.executed_keys) == 1
    assert len(resumed.skipped_frozen_keys) == len(frozen)
    assert resumed.record_count == 1
    assert resumed.resume_plan[ResumeEligibility.RESUMABLE.value] == [target_key(target)]


def test_16b_the_foreign_guard_is_not_always_raising():
    """同实验的三种状态都必须照常判定 —— 证明守卫有**判别力**,不是恒真。"""
    assert (
        resume_eligibility(_record(PILOT), experiment_id=PILOT)
        is ResumeEligibility.FROZEN
    )
    assert (
        resume_eligibility(
            _record(PILOT, status=RecordStatus.INCOMPLETE), experiment_id=PILOT
        )
        is ResumeEligibility.RESUMABLE
    )
    assert (
        resume_eligibility(None, experiment_id=PILOT)
        is ResumeEligibility.NEVER_EXECUTED
    )
    assert resume_eligibility(None, experiment_id=CALIBRATION) is (
        ResumeEligibility.NEVER_EXECUTED
    ), "无记录时不该抛错 —— 它属于恢复,不是挑样本"


def test_16c_same_experiment_records_index_normally():
    records = [_record(PILOT), _record(PILOT, execution_index=1, repetition_id=2)]
    indexed = index_by_unit(records, experiment_id=PILOT)
    assert len(indexed) == 2
    assert all(value.experiment_id == PILOT for value in indexed.values())


def test_16d_the_eligibility_signature_requires_experiment_id():
    """`experiment_id` 必须是**必填 keyword** —— 留默认值等于允许调用方忘记传。"""
    import inspect

    for function in (resume_eligibility, is_resumable, index_by_unit):
        parameter = inspect.signature(function).parameters["experiment_id"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function.__name__
        assert parameter.default is inspect.Parameter.empty, function.__name__

    assert (
        inspect.signature(plan_resume).parameters["experiment_id"].default
        is inspect.Parameter.empty
    )


def target_key(record: RawRecord) -> str:
    return (
        f"{record.condition}:{record.task_id}:{record.baseline_label}:{record.repetition_id}"
    )
