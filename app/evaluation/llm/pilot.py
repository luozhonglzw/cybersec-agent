"""D-2 试点的**冻结规模**与**候选清单**构建(**离线,不发任何请求**)。

两段式清单
----------
    candidate   D-2a 只允许产出这个。provider / model / 端点类别 / token 与
                cost 上限仍是 `TO_BE_FROZEN_*` 占位符。
    frozen      D-2b 的闸门。要求**零占位符**,并且逐项与当前 HEAD、
                任务清单摘要、指标 schema 摘要对齐。

为什么不在 D-2a 直接产出"最终清单"
---------------------------------
清单里有一半字段**在选 provider 之前根本无法确定**(端点类别、定价来源、
provider 侧默认参数的实测快照)。先写一个猜的值,再在 D-2b "更新一下" ——
那正是"活动实验被静默改动"的经典路径。因此本模块只产出候选清单,
并把"还差什么才能冻结"做成**可机械枚举**的列表(`manifest_freeze_blockers`)。

为什么规模必须从结构推导
----------------------
`treatment_runs` / `control_runs` / 调用上界全部由
"任务数 × 重复数 × 基线数"与生产图的 `max_iterations=5` 推出,
**不手写常量**。手写的常量在任务集增删一个任务之后会静默失效,
而失效的预算是所有"超支"事故的共同起点。
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.llm.budget import PilotBudget, pilot_budget
from app.evaluation.llm.dataset import (
    LLM_DATASET_VERSION,
    LLM_TASKS,
    VARIANT_INJECTION_INERT,
)
from app.evaluation.llm.ordering import (
    EXECUTION_ORDER_SEED,
    ExecutionUnit,
    build_execution_order,
)
from app.evaluation.llm.protocol import (
    FAILURE_TAXONOMY_VERSION,
    HARNESS_VERSION,
    METRIC_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    REAL_LLM_TASKSET_VERSION,
    ExperimentManifest,
    ManifestError,
    compute_manifest_digest,
    metric_schema_digest,
    placeholder_paths,
    real_llm_taskset_digest,
    seal_manifest,
    verify_manifest,
)
from app.evaluation.llm.runner import (
    BASELINE_LABELS,
    system_prompt_sha256,
    tool_schema_sha256,
)
from app.evaluation.llm.tasks import LLMTask

#: 试点基线标签。**四个并列条件** —— B0 的两个提示词变体是两个条件,
#: 不得笼统说成"三个基线"。
PILOT_BASELINES: tuple[str, ...] = BASELINE_LABELS

#: 试点执行的行为集。
#:
#: 首个试点只跑 `GOOD`:行为维度的覆盖由 D-1 的**离线**矩阵承担(11 个行为 ×
#: 352 次脚本化运行,零成本)。把它也放进 D-2 会让 provider 调用量乘 11,
#: 而得到的额外信息只是"脚本化行为在真实模型上不成立"这一已知事实。
PILOT_BEHAVIORS: tuple[str, ...] = ("GOOD",)

#: 重复次数。**n=3** —— 见 `derived.N3_LIMITATION` 的允许 / 禁止清单。
PILOT_REPETITION_COUNT = 3

#: D-2 的匹配对照变体(保留载体,只把载荷换成等长惰性文本)。
PILOT_CONTROL_VARIANT = VARIANT_INJECTION_INERT

#: 试点已知混淆变量。**写进清单**,不允许留在脑子里。
DECLARED_CONFOUNDS: tuple[str, ...] = (
    "B0-shared 使用与 B2'/B3 完全相同的系统提示词,而该提示词要求「优先调用工具」;"
    "B0 无工具可调 ⇒ 该条件**刻意处于劣势**,不得当作能力对照使用。",
    "首个试点只执行 GOOD 行为;行为维度的覆盖由 D-1 的离线矩阵承担,"
    "因此 D-2 结果不能说明模型在其它行为下的表现。",
    "重复次数 n=3:只能给出观测比例与描述性区间,不能给出总体表现估计。",
    "SDK 层 max_retries=2 是 **provider 默认行为**,不是实验设计参数;"
    "物理 HTTP 尝试数在 D-2a 不可观测,一律记 UNKNOWN。",
    "单一 provider、单一端点、单次时间窗:provider 侧的灰度发布 / 限流 / "
    "区域差异未被控制,也未被打散(执行顺序只打散任务顺序,不能打散时间效应)。",
    "评测数据为**合成** fixture;任何结论都不得外推到真实日志分布。",
)


class PilotPlan(BaseModel):
    """冻结的试点规模 + 执行顺序。"""

    protocol_version: str = PROTOCOL_VERSION
    taskset_version: str = REAL_LLM_TASKSET_VERSION
    baselines: list[str]
    behaviors: list[str]
    repetition_count: int
    task_count: int
    injection_task_count: int

    control_variant: str = PILOT_CONTROL_VARIANT

    treatment_runs: int
    control_runs: int
    total_runs: int

    logical_invocation_floor: int
    logical_invocation_hard_ceiling: int
    provider_http_attempt_ceiling: int
    provider_http_attempt_ceiling_basis: str
    soft_call_ceiling: int

    execution_order_method: str = "sha256-sort"
    execution_order_seed: str
    execution_order_digest: str
    ordered_unit_keys: list[str] = Field(
        default_factory=list,
        description="有序单元键(condition:task:baseline:repetition)—— 顺序本身可复算",
    )

    @property
    def unit_count(self) -> int:
        return len(self.ordered_unit_keys)

    def as_manifest_fields(self) -> dict[str, Any]:
        return {
            "conditions": list(self.baselines),
            "behaviors": list(self.behaviors),
            "repetition_count": self.repetition_count,
            "control_variant": self.control_variant,
            "execution_order_method": self.execution_order_method,
            "execution_order_seed": self.execution_order_seed,
            "execution_order_digest": self.execution_order_digest,
            "treatment_runs": self.treatment_runs,
            "control_runs": self.control_runs,
            "total_runs": self.total_runs,
            "logical_invocation_hard_ceiling": self.logical_invocation_hard_ceiling,
            "provider_http_attempt_ceiling": self.provider_http_attempt_ceiling,
            "provider_http_attempt_ceiling_basis": (
                self.provider_http_attempt_ceiling_basis
            ),
        }


def pilot_budget_of(
    *,
    baselines: tuple[str, ...] = PILOT_BASELINES,
    repetition_count: int = PILOT_REPETITION_COUNT,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
) -> PilotBudget:
    return pilot_budget(
        baseline_labels=baselines, repetition_count=repetition_count, tasks=tasks
    )


def pilot_units(
    *,
    baselines: tuple[str, ...] = PILOT_BASELINES,
    repetition_count: int = PILOT_REPETITION_COUNT,
    seed: str = EXECUTION_ORDER_SEED,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
) -> tuple[list[ExecutionUnit], str]:
    """有序的实验单元 + 顺序摘要。"""
    return build_execution_order(
        baseline_labels=baselines,
        repetition_count=repetition_count,
        seed=seed,
        tasks=tasks,
    )


def pilot_plan(
    *,
    baselines: tuple[str, ...] = PILOT_BASELINES,
    repetition_count: int = PILOT_REPETITION_COUNT,
    seed: str = EXECUTION_ORDER_SEED,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
) -> PilotPlan:
    """冻结的试点计划。**规模与上限全部由结构推导。**"""
    budget = pilot_budget_of(
        baselines=baselines, repetition_count=repetition_count, tasks=tasks
    )
    ordered, order_digest = pilot_units(
        baselines=baselines, repetition_count=repetition_count, seed=seed, tasks=tasks
    )
    return PilotPlan(
        baselines=list(baselines),
        behaviors=list(PILOT_BEHAVIORS),
        repetition_count=repetition_count,
        task_count=budget.task_count,
        injection_task_count=budget.injection_task_count,
        treatment_runs=budget.treatment_runs,
        control_runs=budget.control_runs,
        total_runs=budget.total_runs,
        logical_invocation_floor=budget.logical_invocation_floor,
        logical_invocation_hard_ceiling=budget.logical_invocation_hard_ceiling,
        provider_http_attempt_ceiling=budget.provider_http_attempt_ceiling,
        provider_http_attempt_ceiling_basis=budget.provider_http_attempt_ceiling_basis,
        soft_call_ceiling=budget.soft_call_ceiling,
        execution_order_seed=seed,
        execution_order_digest=order_digest,
        ordered_unit_keys=[unit.key for unit in ordered],
    )


# ---------------------------------------------------------------------------
# 候选清单
# ---------------------------------------------------------------------------


def build_candidate_manifest(
    *,
    git_commit: str,
    datasets: dict[str, dict[str, str]],
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
    plan: PilotPlan | None = None,
    created_at_utc: str | None = None,
) -> ExperimentManifest:
    """构建**候选**清单并封上自身摘要。

    provider / model / 端点类别 / token 上限 / cost 上限**刻意留占位符** ——
    它们要等 D-2b 才能确定。留下占位符而不是猜一个值,是为了让
    "这份清单还不能冻结"成为一个可机械回答的问题。

    `created_at_utc` 与可复现性
    ---------------------------
    不传时取**当前时间**。因此 `manifest_digest` 与报告正文在**跨时刻**的
    两次独立运行之间**不会**逐字节相同 —— 报告的可复现性以"清单相同"为前提。

    这不影响实验正确性(candidate 本就未冻结),但要求:
    - 需要跨机器 / 跨时刻比对产物时,**显式传入** `created_at_utc`;
    - D-2b 冻结清单时必须把该字段**固定下来** —— 冻结之后,报告才真正可逐字节复现。
    """
    plan = plan or pilot_plan(tasks=tasks)
    manifest = ExperimentManifest(
        protocol_version=PROTOCOL_VERSION,
        manifest_status="candidate",
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        taskset_version=REAL_LLM_TASKSET_VERSION,
        taskset_digest=real_llm_taskset_digest(datasets, tasks),
        metric_schema_version=METRIC_SCHEMA_VERSION,
        metric_schema_digest=metric_schema_digest(),
        conditions=list(plan.baselines),
        behaviors=list(plan.behaviors),
        repetition_count=plan.repetition_count,
        control_variant=plan.control_variant,
        execution_order_method=plan.execution_order_method,
        execution_order_seed=plan.execution_order_seed,
        execution_order_digest=plan.execution_order_digest,
        treatment_runs=plan.treatment_runs,
        control_runs=plan.control_runs,
        total_runs=plan.total_runs,
        logical_invocation_hard_ceiling=plan.logical_invocation_hard_ceiling,
        provider_http_attempt_ceiling=plan.provider_http_attempt_ceiling,
        provider_http_attempt_ceiling_basis=plan.provider_http_attempt_ceiling_basis,
        harness_level_retry=0,
        failure_taxonomy_version=FAILURE_TAXONOMY_VERSION,
        system_prompt_sha256=system_prompt_sha256(),
        tool_schema_sha256=tool_schema_sha256(),
        harness_version=HARNESS_VERSION,
        declared_confounds=list(DECLARED_CONFOUNDS),
    )
    return seal_manifest(manifest)


def verify_candidate_manifest(
    manifest: ExperimentManifest,
    *,
    datasets: dict[str, dict[str, str]] | None = None,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
    expected_git_commit: str | None = None,
) -> None:
    """候选清单的启动前闸门。

    传 `datasets` 时会**重算**任务清单摘要并比对 —— 这是"benchmark 事后被
    改动"可被检出的唯一途径。
    """
    verify_manifest(
        manifest,
        require_frozen=False,
        expected_git_commit=expected_git_commit,
        expected_taskset_digest=(
            real_llm_taskset_digest(datasets, tasks) if datasets is not None else None
        ),
        expected_metric_schema_digest=metric_schema_digest(),
    )


def manifest_freeze_blockers(manifest: ExperimentManifest) -> list[str]:
    """距离"可冻结"还差什么。**D-2b 的前置清单。**

    返回的是**占位符字段路径** —— 也就是说,只要这个列表非空,
    `verify_manifest(require_frozen=True)` 一定会拒绝。
    """
    payload = manifest.model_dump(mode="json")
    payload.pop("manifest_digest", None)
    blockers = placeholder_paths(payload)
    if manifest.manifest_status != "frozen":
        blockers.append("manifest_status(当前为 candidate)")
    return sorted(blockers)


def assert_candidate_only(manifest: ExperimentManifest) -> None:
    """D-2a 的自律断言:本阶段**只允许**候选清单。

    与 `verify_manifest(require_frozen=True)` 恰好相反 ——
    它防的是"提前把候选当成冻结",这条防的是"在 D-2a 就把清单标成冻结"。
    """
    if manifest.manifest_status != "candidate":
        raise ManifestError(
            f"D-2a 只允许产出 candidate 清单,实际为 {manifest.manifest_status!r} —— "
            "最终冻结是 D-2b 的独立闸门"
        )


def manifest_report(manifest: ExperimentManifest) -> dict[str, Any]:
    """清单的可审阅摘要(不含任何凭据类字段)。"""
    return {
        "protocol_version": manifest.protocol_version,
        "manifest_status": manifest.manifest_status,
        "manifest_digest": manifest.manifest_digest,
        "git_commit": manifest.git_commit,
        "taskset_version": manifest.taskset_version,
        "taskset_digest": manifest.taskset_digest,
        "metric_schema_version": manifest.metric_schema_version,
        "metric_schema_digest": manifest.metric_schema_digest,
        "dataset_version": LLM_DATASET_VERSION,
        "harness_version": manifest.harness_version,
        "failure_taxonomy_version": manifest.failure_taxonomy_version,
        "control_variant": manifest.control_variant,
        "execution_order_seed": manifest.execution_order_seed,
        "execution_order_digest": manifest.execution_order_digest,
        "treatment_runs": manifest.treatment_runs,
        "control_runs": manifest.control_runs,
        "total_runs": manifest.total_runs,
        "logical_invocation_hard_ceiling": manifest.logical_invocation_hard_ceiling,
        "provider_http_attempt_ceiling": manifest.provider_http_attempt_ceiling,
        "harness_level_retry": manifest.harness_level_retry,
        "recomputed_manifest_digest": compute_manifest_digest(manifest),
        "freeze_blockers": manifest_freeze_blockers(manifest),
        "workdir_independent": True,
    }


def load_datasets(workdir: Path | str) -> dict[str, dict[str, str]]:
    """薄封装:把数据集构建集中到一个入口,便于清单与执行器共用。"""
    from app.evaluation.llm.dataset import build_datasets

    return build_datasets(workdir)
