"""D-2c 标定的**顺序阶段门**执行器。

门是顺序的,且**只有 PASS 放行**
-------------------------------

    C0a PASS → C0b PASS → C1 PASS → C2 PASS → C3 PASS → C4 PASS

`FAIL` / `INCONCLUSIVE` / `ABORT` 中的任何一个都**阻止所有后续阶段开始**。
`INCONCLUSIVE` 与 `FAIL` 在这一条上完全等价 —— 因为"没能证明"与"证明了不成立"
都不构成继续往下走的前提。

这条门不是文档里的约定,而是 `run()` 里的一段控制流:一旦某阶段非 PASS,
后续阶段**连运行器都不会被调用**。测试用探针计数机械证明这一点。

三条隔离(全部 fail closed)
--------------------------
1. **命名空间**:`experiment_id` 必须是标定命名空间,且**在构造任何产物之前**
   就断言。给试点 id 加后缀派生出来的标定 id 会在这里被拒。
2. **文件系统**:每个阶段一个**独立**工作目录;标定根与试点根**互不包含**;
   任何路径都不得落在仓库 `data/` 下。
3. **试点资格**:标定产物**永不**成为试点的完成资格 —— 由既有守卫
   `resume_eligibility()` 以试点身份机械证明(它会 fail closed)。

本模块**不 import 任何 provider 客户端**,也不发起任何网络 I/O。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.evaluation.calibration import config as cal_config
from app.evaluation.calibration import namespace as cal_namespace
from app.evaluation.calibration import stages as cal_stages
from app.evaluation.calibration.stages import (
    DEFAULT_STAGE_RUNNERS,
    STAGE_ORDER,
    Stage,
    StageContext,
    StageDeps,
    StageOutcome,
    StageScope,
    stage_ceiling,
    stage_floor,
)
from app.evaluation.calibration.verdict import (
    StageVerdict,
    Verdict,
    aborted,
)
from app.evaluation.llm.budget import (
    UNKNOWN,
    BudgetExceeded,
    BudgetGovernor,
)
from app.evaluation.llm.offline_guard import NetworkEgressGuard

#: 试点命名空间里用于**证明隔离**的合成 id。它从不被真正使用。
PROBE_PILOT_EXPERIMENT_ID = "d2d-pilot-isolation-probe"


class PilotEligibilityViolation(RuntimeError):
    """标定产物被当成了试点的完成资格。**绝不允许发生。**"""


def assert_not_pilot_eligible(
    *,
    artifact_experiment_id: str,
    pilot_experiment_id: str = PROBE_PILOT_EXPERIMENT_ID,
) -> None:
    """机械证明:标定产物**不是**试点的完成资格。

    两步,缺一不可:

        1. 产物 id 必须被判为**标定**命名空间(不是"不像试点"就算过);
        2. 把一条带该 id 的最小记录交给既有守卫 `resume_eligibility()`,
           并以**试点身份**查询 —— 它必须 fail closed。

    第 2 步是真正有牙的那一步:它用的不是本模块的判断,而是 D-2b 已经
    加固过的跨实验守卫。若有一天那条守卫被放宽,这个证明会立刻失败。
    """
    from app.evaluation.llm.raw import (
        ForeignExperimentRecord,
        RawRecord,
        resume_eligibility,
    )

    cal_namespace.assert_calibration_experiment_id(artifact_experiment_id)
    cal_namespace.assert_pilot_experiment_id(pilot_experiment_id)

    record = RawRecord(
        run_id="isolation-probe",
        experiment_id=artifact_experiment_id,
        protocol_version="d2c",
        manifest_digest="0" * 64,
        task_id="isolation-probe",
        condition="calibration",
        baseline_label="C0a",
        dataset_variant="none",
        repetition_id=1,
        execution_index=0,
    )
    try:
        eligibility = resume_eligibility(record, experiment_id=pilot_experiment_id)
    except ForeignExperimentRecord:
        return
    raise PilotEligibilityViolation(
        f"标定记录({artifact_experiment_id!r})被试点({pilot_experiment_id!r})"
        f"判定为 {eligibility.value} —— 标定产物**不得**成为试点的完成资格"
    )


def assert_no_pilot_eligible_records(workdir_root: Path | str) -> list[Path]:
    """扫描标定根下的 JSONL,证明其中**没有**试点命名空间的记录。

    返回被扫描到的文件清单(供报告引用)。任何一行若带 `experiment_id`
    且该 id 属于试点命名空间,立即拒绝。
    """
    root = cal_namespace.calibration_root(workdir_root)
    scanned: list[Path] = []
    if not root.exists():
        return scanned
    for path in sorted(root.rglob("*.jsonl")):
        scanned.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            experiment_id = payload.get("experiment_id")
            if experiment_id is None:
                continue
            try:
                kind = cal_namespace.classify_experiment_id(experiment_id)
            except cal_namespace.CalibrationNamespaceError as exc:
                raise PilotEligibilityViolation(
                    f"{path} 里的 experiment_id {experiment_id!r} 无法归类:{exc}"
                ) from exc
            if kind is not cal_namespace.ExperimentNamespace.CALIBRATION:
                raise PilotEligibilityViolation(
                    f"{path} 里出现了 {kind.value} 命名空间的记录"
                    f"({experiment_id!r}) —— 标定目录不得产出试点可消费的记录"
                )
    return scanned


# ---------------------------------------------------------------------------
# 运行产物
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRecord:
    """一个阶段的门记录。"""

    stage: Stage
    verdict: StageVerdict
    reason: str
    logical_invocations: int
    provider_http_attempts: int | str
    ceiling: int
    workdir: str
    evidence: dict[str, Any]

    @property
    def is_pass(self) -> bool:
        return self.verdict is StageVerdict.PASS


@dataclass
class CalibrationRun:
    """一次标定运行的**全部**可审阅产出(凭据无关)。"""

    experiment_id: str
    config: cal_config.CalibrationConfig
    records: list[StageRecord] = field(default_factory=list)
    attempted_stages: tuple[Stage, ...] = ()
    skipped_stages: tuple[Stage, ...] = ()
    halted_at: Stage | None = None
    network: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    scanned_artifacts: tuple[str, ...] = ()
    provider_http_attempts: int | str = UNKNOWN

    @property
    def passed_all(self) -> bool:
        """六个阶段**全部** PASS。少一个都不算。"""
        return (
            self.halted_at is None
            and tuple(record.stage for record in self.records) == STAGE_ORDER
            and all(record.is_pass for record in self.records)
        )

    @property
    def last_completed_stage(self) -> Stage | None:
        return self.records[-1].stage if self.records else None

    def verdict_of(self, stage: Stage) -> StageVerdict | None:
        for record in self.records:
            if record.stage is stage:
                return record.verdict
        return None

    def manifest_fields(self) -> dict[str, Any]:
        """凭据无关的清单字段。**不含**任何凭据、完整 URL 或请求体。"""
        return {
            "experiment_id": self.experiment_id,
            "provider": self.config.provider,
            "requested_model": self.config.requested_model,
            "provider_family": self.config.provider_family,
            "endpoint_category": self.config.endpoint_category.value,
            "request_digest": self.config.request_shape_digest(),
            "declared_output_token_cap": self.config.output_token_cap,
            "declared_temperature": self.config.recorded_temperature(),
            "streaming": self.config.streaming,
            "max_retries": self.config.max_retries,
            "timeout_seconds": self.config.timeout_seconds,
            "stage_order": [stage.value for stage in STAGE_ORDER],
            "attempted_stages": [stage.value for stage in self.attempted_stages],
            "skipped_stages": [stage.value for stage in self.skipped_stages],
            "halted_at": self.halted_at.value if self.halted_at else None,
            "stage_verdicts": {
                record.stage.value: record.verdict.value for record in self.records
            },
            "logical_invocations": self.budget.get("experiment_logical_llm_invocations"),
            "provider_http_attempts": self.provider_http_attempts,
            "passed_all": self.passed_all,
            "network": self.network,
        }


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------


class CalibrationHarness:
    """顺序阶段门执行器。

    它自己**不做**任何阶段判定 —— 判定在 `stages.py` 里。它只负责:
    顺序、门、预算、隔离、以及把每一阶段的结果如实记下来。
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        workdir_root: Path | str,
        config: cal_config.CalibrationConfig | None = None,
        deps: StageDeps | None = None,
        governor: BudgetGovernor | None = None,
        guard: NetworkEgressGuard | None = None,
        runners: Mapping[Stage, Callable[..., Any]] | None = None,
    ) -> None:
        # 命名空间断言**排在构造任何产物之前**。
        self.experiment_id = cal_namespace.assert_calibration_experiment_id(
            experiment_id
        )
        self.workdir_root = cal_namespace.assert_outside_repo_data(workdir_root)
        self.config = config or cal_config.default_calibration_config()
        self.deps = deps or StageDeps()
        self.governor = governor or BudgetGovernor(budget=cal_config.calibration_budget())
        self.guard = guard or NetworkEgressGuard(strict=True)
        self.runners = dict(runners or DEFAULT_STAGE_RUNNERS)

        # 隔离:标定根与试点根必须互不包含。这里立刻证明一次 ——
        # 等到"要写文件了"才检查,那时错误已经发生。
        cal_namespace.assert_workdirs_disjoint(
            cal_namespace.calibration_root(self.workdir_root),
            cal_namespace.pilot_root(self.workdir_root),
        )

    # ---- 工作目录 ----

    def stage_workdir(self, stage: Stage) -> Path:
        """某阶段的**独立**工作目录。

        每个阶段一个目录,所以一个失败阶段的半成品**不可能**被下一个阶段
        续跑 —— 续跑需要的输入根本不在它的目录里。
        """
        base = cal_namespace.calibration_stage_workdir(
            self.workdir_root, self.experiment_id
        )
        return cal_namespace.assert_outside_repo_data(base / stage.value)

    # ---- 主流程 ----

    async def run(self) -> CalibrationRun:
        run = CalibrationRun(experiment_id=self.experiment_id, config=self.config)
        attempted: list[Stage] = []
        skipped: list[Stage] = []
        halted_at: Stage | None = None

        with self.guard:
            for stage in STAGE_ORDER:
                if halted_at is not None:
                    # 门已经关闭 —— 连运行器都不调用。
                    skipped.append(stage)
                    continue

                # Tier 1:单元准入。装不下连结构下界就立刻拒绝,不产生任何调用。
                # 下界为 0 的阶段(C2)跳过准入检查 —— `admit_unit` 的下界是 1,
                # 对"不发起任何调用"的阶段做准入检查没有意义。
                floor = stage_floor(stage)
                if floor >= 1:
                    self.governor.admit_unit(min_invocations=floor)
                attempted.append(stage)

                outcome = await self._run_stage(stage)
                # 物理尝试只在传输层**直接观测到**时才登记;否则记不可观测
                # (总数因此为 UNKNOWN,而**不是** 0)。
                observed_attempts = (
                    outcome.provider_http_attempts
                    if isinstance(outcome.provider_http_attempts, int)
                    else None
                )
                self.governor.record_unit(provider_http_attempts=observed_attempts)
                self.governor.check()

                run.records.append(
                    StageRecord(
                        stage=stage,
                        verdict=outcome.verdict.verdict,
                        reason=outcome.verdict.reason,
                        logical_invocations=outcome.logical_invocations,
                        provider_http_attempts=outcome.provider_http_attempts,
                        ceiling=stage_ceiling(stage),
                        workdir=str(self.stage_workdir(stage)),
                        evidence=outcome.evidence,
                    )
                )
                if not outcome.is_pass:
                    halted_at = stage

            run.attempted_stages = tuple(attempted)
            run.skipped_stages = tuple(skipped)
            run.halted_at = halted_at
            run.network = self.guard.summary()

        run.budget = self.governor.snapshot()
        run.provider_http_attempts = self.governor.experiment_provider_http_attempts
        run.scanned_artifacts = tuple(
            str(path) for path in assert_no_pilot_eligible_records(self.workdir_root)
        )
        assert_not_pilot_eligible(artifact_experiment_id=self.experiment_id)
        return run

    async def _run_stage(self, stage: Stage) -> StageOutcome:
        """执行单个阶段。**任何异常都变成判定,不是崩溃。**

        顺序是硬要求:`BudgetExceeded`(以及继承它的
        `StageInvocationCeilingExceeded`)是 `AssertionError` 子类,必须排在
        通用 `except Exception` **之前** —— 否则它们会被兜底分支吞成
        "基础设施错误",而预算纪律就此消失。
        """
        workdir = self.stage_workdir(stage)
        workdir.mkdir(parents=True, exist_ok=True)
        scope = StageScope(
            stage=stage, ceiling=stage_ceiling(stage), governor=self.governor
        )
        ctx = StageContext(
            stage=stage,
            config=self.config,
            experiment_id=self.experiment_id,
            workdir=workdir,
            scope=scope,
            governor=self.governor,
        )
        runner = self.runners.get(stage)
        if runner is None:
            # 缺运行器 ⇒ ABORT,不是 KeyError 崩溃,也**不是**静默跳过。
            return self._abort_outcome(
                ctx, scope, f"阶段 {stage.value} 没有注册运行器"
            )
        try:
            return await runner(ctx, self.deps)
        except BudgetExceeded as exc:
            # 覆盖 `StageInvocationCeilingExceeded`(它是本类的子类)。
            return self._abort_outcome(ctx, scope, f"预算中止:{exc}")
        except cal_stages.StageDependencyMissing as exc:
            return self._abort_outcome(ctx, scope, f"依赖缺失:{exc}")
        except Exception as exc:  # noqa: BLE001 —— 兜底必须存在,但排在预算之后
            return self._abort_outcome(
                ctx, scope, f"阶段抛出 {type(exc).__name__}:{exc}"
            )

    @staticmethod
    def _abort_outcome(
        ctx: StageContext, scope: StageScope, reason: str
    ) -> StageOutcome:
        return StageOutcome(
            stage=ctx.stage,
            verdict=aborted(reason),
            evidence={"abort_reason": reason},
            logical_invocations=scope.invocations,
            provider_http_attempts=UNKNOWN,
        )


def offline_evidence(run: CalibrationRun) -> dict[str, Any]:
    """离线证据。**出口计数为 0 时才敢声称零真实调用。**

    若守卫观测到任何出口,`real_provider_calls` 记 `UNKNOWN` 而不是 0 ——
    "我们知道发生过出口"与"我们知道总共几次"是两件事。
    """
    egress = int(run.network.get("egress_events", 0) or 0)
    clean = bool(run.network.get("clean", False))
    offline = clean and egress == 0
    return {
        "offline": offline,
        "egress_events": egress,
        "socket_creation_events": run.network.get("socket_creation_events"),
        "loopback_events": run.network.get("loopback_events"),
        "real_provider_calls": 0 if offline else UNKNOWN,
        "api_credits_consumed": 0 if offline else UNKNOWN,
    }


def format_stage_table(run: CalibrationRun) -> Sequence[str]:
    """人类可读的阶段表(用于报告)。"""
    rows = [f"{'阶段':<6}{'判定':<14}{'调用':>5}{'上限':>6}  理由"]
    for record in run.records:
        rows.append(
            f"{record.stage.value:<6}{record.verdict.value:<14}"
            f"{record.logical_invocations:>5}{record.ceiling:>6}  {record.reason}"
        )
    for stage in run.skipped_stages:
        rows.append(f"{stage.value:<6}{'SKIPPED':<14}{'-':>5}{stage_ceiling(stage):>6}  门已关闭")
    return rows
