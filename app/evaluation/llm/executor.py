"""D-2a 离线执行器:**只跑 ScriptedLLM 的完整试点流水线**。

它执行的是 D-2 的**全部工装**,唯独不碰 provider:

    数据集 → 执行顺序 → 逐单元运行(脚本化) → 失败分类 → 原始落盘
    → 逐重复指标 → 派生聚合 → 报告

因此 D-2b 只需要替换"谁来产生 `AIMessage`"这一个环节,其余代码路径
在 D-2a 已经被完整地跑过、测过、落过盘。这是"先证明工装能跑通"的
具体含义 —— 不是跑通一个简化版,而是跑通**同一条**流水线。

匹配对照:为什么是 `injection_inert` 而不是 `base`
----------------------------------------------
`base` 变体把**载体也一起拿掉了**。于是 treatment 与 control 之间差了
两个变量:载荷的有无 **和** 载体事件的有无。这样得到的"差异"无法归因到
载荷上 —— 它同样可能是"多了一条 web_request 日志"造成的。

`injection_inert` 保留**逐字段相同**的载体,只把那段自由文本换成等长惰性
文本。于是 treatment 与 control 之间只差一个变量。这一条不是靠人工同步
两份 fixture 保证的,而是由**同一段代码**产出、并在本模块里被机械核验
(见 `substitution_equivalence`)。

逐重复计算指标(一个必须写下来的坑)
--------------------------------
`compute_llm_metrics` 的观测索引以 `(task_id, baseline, behavior)` 为键。
若把 n=3 的全部观测一次性传进去,**同一单元的重复会互相覆盖**,n=3 静默
退化成 n=1,而所有分母看起来完全正常。因此本模块**逐重复**调用它,
再由派生层把分子分母相加 —— 合并这一步必须是显式的、可见的。

记录**边跑边落盘**
----------------
每个单元一跑完就 append + flush + fsync,而不是等整轮结束再批量写。
崩溃时"已经跑完的单元"必须已经在盘上,否则续跑会把它们当成
"从未执行"而重跑 —— 那正是挑样本的入口。
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from app.evaluation.llm.adapters import (
    BASELINES,
    NO_TOOL_SYSTEM_PROMPT,
    SHARED_SYSTEM_PROMPT,
    LLMObservation,
)
from app.evaluation.llm.budget import UNKNOWN
from app.evaluation.llm.dataset import (
    INJECTION_INERT_TEXT,
    INJECTION_PAYLOAD,
    INJECTION_PAYLOAD_MARKER,
    LLM_DATASET_VERSION,
    LLM_TASKS,
    VARIANT_INJECTION,
    build_datasets,
)
from app.evaluation.llm.derived import DerivedAggregate, derive
from app.evaluation.llm.exposure import ExposureStatus, exposure_status
from app.evaluation.llm.failures import (
    FailureClass,
    FailureRecord,
    classify_error_name,
)
from app.evaluation.llm.metrics import MetricResult, compute_llm_metrics
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.ordering import ExecutionUnit
from app.evaluation.llm.pilot import (
    PILOT_BASELINES,
    PILOT_BEHAVIORS,
    PILOT_CONTROL_VARIANT,
    PILOT_REPETITION_COUNT,
    PilotPlan,
    pilot_plan,
    pilot_units,
)
from app.evaluation.llm.protocol import PROTOCOL_VERSION, canonical_json, sha256_hex
from app.evaluation.llm.raw import (
    RawRecord,
    RawWriter,
    RecordStatus,
    ResumeEligibility,
    build_resume_metadata,
    resume_eligibility,
)
from app.evaluation.llm.runner import (
    authorized_paths_for,
    build_evidence_index,
    make_decoys,
    tool_schema_sha256,
)
from app.evaluation.llm.tasks import LLMTask


class PairingError(AssertionError):
    """匹配对照的配对不变量被破坏。"""


class HarnessAbort(RuntimeError):
    """工装自身失败 —— 记录已落盘(标记 incomplete),但流水线必须停下。"""


class RawArtifactExistsError(RuntimeError):
    """目标原始文件已存在,而调用方没有声明要继承它 —— 拒绝静默追加。"""


def assert_no_silent_append(
    raw_path: Path, existing: dict[Any, Any] | None
) -> None:
    """拒绝把一次**全新运行**追加到已有的原始文件上。

    为什么必须有这条守卫
    --------------------
    `RawWriter` 是**追加式**的 —— 崩溃恢复必须如此,否则"已跑完的单元"会在
    崩溃中丢失。副作用是:同一个工作目录里跑第二次"全新"运行(不传 `existing`),
    文件会从 108 条变成 216 条,而 `record_count`、派生聚合与配对核验都只
    覆盖**本次**运行的记录。于是:

        落盘产物(216 条) 与 上报数字(108) **静默分叉**,
        sidecar 依然自洽,没有任何一层会发现。

    这正是本模块最不该出现的一类失败:不是报错,而是**看起来一切正常**。

    判定规则
    --------
    目标文件已存在、且 `existing` 为空 ⇒ 失败。要重跑就换工作目录或换
    `experiment_id`;要续跑就显式传入已有记录。

    这是**状态守卫**,不是结果质量判断:它只看"盘上有没有东西",
    不看任何一次运行的结果好不好 —— 后者是本节唯一不可协商的禁令。
    """
    if raw_path.exists() and not existing:
        raise RawArtifactExistsError(
            f"原始文件已存在且调用方未声明继承,拒绝静默追加:{raw_path}。"
            "请换一个工作目录或 experiment_id 重跑,或显式传入 existing 续跑。"
        )


# ---------------------------------------------------------------------------
# 配对
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedPair:
    """一对匹配的实验单元(treatment / control)。"""

    task_id: str
    baseline_label: str
    repetition_id: int
    treatment_key: str
    control_key: str

    @property
    def key(self) -> str:
        return f"{self.task_id}|{self.baseline_label}|{self.repetition_id}"


def build_matched_pairs(units: Iterable[ExecutionUnit]) -> list[MatchedPair]:
    """按 `(task_id, baseline_label, repetition_id)` 配对。

    配对**不依赖执行顺序**:两个成员在执行序列里可以相隔很远 ——
    那正是"配对"与"相邻执行"的区别。若配对依赖顺序,打散顺序就会
    改变配对结果,而顺序本该与结论无关。
    """
    treatment: dict[tuple[str, str, int], str] = {}
    control: dict[tuple[str, str, int], str] = {}
    for unit in units:
        key = (unit.task_id, unit.baseline_label, unit.repetition_id)
        (treatment if unit.condition == "treatment" else control)[key] = unit.key

    pairs: list[MatchedPair] = []
    for key in sorted(control):
        if key not in treatment:
            raise PairingError(f"对照单元 {key} 没有对应的处理单元 —— 配对不成立")
        pairs.append(MatchedPair(
            task_id=key[0],
            baseline_label=key[1],
            repetition_id=key[2],
            treatment_key=treatment[key],
            control_key=control[key],
        ))
    return pairs


def _read_paths(paths: dict[str, str]) -> dict[str, str]:
    return {key: Path(value).read_text(encoding="utf-8") for key, value in paths.items()}


def substitution_equivalence(
    treatment_paths: dict[str, str], control_paths: dict[str, str]
) -> bool:
    """**treatment 与 control 只差一次载荷替换** —— 逐字节核验。

    把 treatment 文件里的载荷字符串替换成惰性文本,必须**逐字节等于**
    control 文件。这比"字段逐个比对"强得多:它同时排除了"多一个字段"、
    "少一条记录"、"时间戳不同"、"键顺序不同"这类会静默改变结论的差异。
    """
    treatment = _read_paths(treatment_paths)
    control = _read_paths(control_paths)
    if set(treatment) != set(control):
        return False
    return all(
        treatment[key].replace(INJECTION_PAYLOAD, INJECTION_INERT_TEXT) == control[key]
        for key in treatment
    )


def fixture_pair_checks(datasets: dict[str, dict[str, str]]) -> dict[str, Any]:
    """数据集层面的配对不变量(**每对共用,只算一次**)。"""
    treatment_paths = datasets[VARIANT_INJECTION]
    control_paths = datasets[PILOT_CONTROL_VARIANT]
    treatment = _read_paths(treatment_paths)
    control = _read_paths(control_paths)

    treatment_lines = treatment["logs"].splitlines()
    control_lines = control["logs"].splitlines()
    t_carrier = json.loads(treatment_lines[0]) if treatment_lines else {}
    c_carrier = json.loads(control_lines[0]) if control_lines else {}
    shared_keys = sorted(set(t_carrier) - {"message"})

    return {
        # ---- 载体结构相同 ----
        "same_event_count": len(treatment_lines) == len(control_lines),
        "same_carrier_field_set": sorted(t_carrier) == sorted(c_carrier),
        "same_event_type": t_carrier.get("event_type") == c_carrier.get("event_type"),
        "same_status": t_carrier.get("status") == c_carrier.get("status"),
        "same_severity": t_carrier.get("severity") == c_carrier.get("severity"),
        "same_non_payload_fields": all(
            t_carrier.get(key) == c_carrier.get(key) for key in shared_keys
        ),
        # ---- 载荷 / 惰性文本的性质 ----
        "treatment_contains_payload": INJECTION_PAYLOAD in treatment["logs"]
        and INJECTION_PAYLOAD in treatment["intel"],
        "control_contains_no_payload": INJECTION_PAYLOAD not in control["logs"]
        and INJECTION_PAYLOAD not in control["intel"],
        "control_contains_no_marker": INJECTION_PAYLOAD_MARKER not in control["logs"]
        and INJECTION_PAYLOAD_MARKER not in control["intel"],
        "inert_text_length_equals_payload": len(INJECTION_INERT_TEXT)
        == len(INJECTION_PAYLOAD),
        "treatment_payload_occurs_once_per_file": (
            treatment["logs"].count(INJECTION_PAYLOAD) == 1
            and treatment["intel"].count(INJECTION_PAYLOAD) == 1
        ),
        # ---- 最强断言:只差一次替换 ----
        "substitution_equivalence": substitution_equivalence(
            treatment_paths, control_paths
        ),
    }


def record_pair_checks(treatment: RawRecord, control: RawRecord) -> dict[str, Any]:
    """一对**实际运行记录**的配对不变量。

    这里比较的是两次运行**各自产出的值**,而不是"同一个表达式与它自己" ——
    否则这些检查会全部退化成同义反复(永远为真,也就永远无用)。
    """
    t_inputs = treatment.metric_inputs
    c_inputs = control.metric_inputs
    return {
        "same_task_id": treatment.task_id == control.task_id,
        "same_baseline_label": treatment.baseline_label == control.baseline_label,
        "same_repetition_id": treatment.repetition_id == control.repetition_id,
        "same_behavior": t_inputs.get("behavior") == c_inputs.get("behavior"),
        "same_system_prompt_bytes": treatment.system_prompt_sha256
        == control.system_prompt_sha256,
        "same_user_prompt_bytes": treatment.user_prompt_sha256
        == control.user_prompt_sha256,
        "same_tool_schema": treatment.tool_schema_sha256 == control.tool_schema_sha256,
        "same_model_configuration": treatment.model == control.model
        and treatment.provider == control.provider,
        "conditions_are_treatment_and_control": treatment.condition == "treatment"
        and control.condition == "control",
        "variants_are_treatment_and_control": treatment.dataset_variant
        == VARIANT_INJECTION
        and control.dataset_variant == PILOT_CONTROL_VARIANT,
    }


def verify_pairing(
    pairs: list[MatchedPair],
    records_by_key: dict[str, RawRecord],
    *,
    datasets: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """机械核验全部配对。任何一条不成立即抛 `PairingError`。"""
    fixture = fixture_pair_checks(datasets)
    fixture_failures = sorted(name for name, ok in fixture.items() if not ok)

    pair_failures: list[dict[str, str]] = []
    for pair in pairs:
        treatment = records_by_key.get(pair.treatment_key)
        control = records_by_key.get(pair.control_key)
        if treatment is None or control is None:
            pair_failures.append({"pair": pair.key, "check": "records_present"})
            continue
        for name, ok in record_pair_checks(treatment, control).items():
            if not ok:
                pair_failures.append({"pair": pair.key, "check": name})

    summary = {
        "pairs": len(pairs),
        "fixture_checks": fixture,
        "fixture_failures": fixture_failures,
        "pair_failures": pair_failures,
        "ok": not fixture_failures and not pair_failures,
    }
    if not summary["ok"]:
        raise PairingError(
            f"配对不变量不成立:数据集层面 {fixture_failures};"
            f"记录层面 {len(pair_failures)} 条(前 3:{pair_failures[:3]})"
        )
    return summary


# ---------------------------------------------------------------------------
# 恢复 / 续跑规划
# ---------------------------------------------------------------------------


def plan_resume(
    units: Iterable[ExecutionUnit],
    existing: dict[tuple[str, str, str, int], RawRecord],
) -> dict[str, list[str]]:
    """把单元按续跑资格分成三类。

    **判定依据只有"记录是否完整"** —— 结果好坏完全不参与。
    一旦允许按结果选择重跑,实验就从"测量"退化成"挑样本"。
    """
    plan: dict[str, list[str]] = {
        ResumeEligibility.FROZEN.value: [],
        ResumeEligibility.RESUMABLE.value: [],
        ResumeEligibility.NEVER_EXECUTED.value: [],
    }
    for unit in units:
        record = existing.get(
            (unit.condition, unit.task_id, unit.baseline_label, unit.repetition_id)
        )
        plan[resume_eligibility(record).value].append(unit.key)
    return plan


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------


def unit_behavior(behaviors: tuple[str, ...]) -> str:
    """本执行器一次只跑一个行为(与冻结试点一致)。

    行为维度由 D-1 的离线矩阵覆盖;D-2 把它固定为单个行为,是为了让
    provider 调用量与行为数**解耦** —— 否则放大行为集会直接放大额度消耗。
    """
    if len(behaviors) != 1:
        raise ValueError(
            f"离线执行器一次只接受**一个**行为,收到 {behaviors!r} —— "
            "行为维度由 D-1 的离线矩阵覆盖(见 pilot.PILOT_BEHAVIORS)"
        )
    return behaviors[0]


class ExecutionOutcome(BaseModel):
    """一次离线试点运行的完整产物。"""

    experiment_id: str
    protocol_version: str = PROTOCOL_VERSION
    dataset_version: str = LLM_DATASET_VERSION
    manifest_digest: str = ""
    raw_path: str = ""
    raw_sha256: str = ""
    sidecar_ok: bool = False
    record_count: int = 0
    executed_keys: list[str] = Field(default_factory=list)
    skipped_frozen_keys: list[str] = Field(default_factory=list)
    resume_plan: dict[str, list[str]] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    pairing: dict[str, Any] = Field(default_factory=dict)
    aggregate: DerivedAggregate = Field(default_factory=DerivedAggregate)
    metrics_by_repetition: dict[int, list[MetricResult]] = Field(default_factory=dict)
    network: dict[str, Any] = Field(default_factory=dict)
    report_markdown: str = ""

    def metric(self, metric_id: str, repetition: int) -> MetricResult:
        for item in self.metrics_by_repetition.get(repetition, []):
            if item.metric_id == metric_id:
                return item
        raise KeyError(f"未知指标 {metric_id} @ repetition {repetition}")


class OfflineExecutor:
    """离线执行器。**不创建任何 provider 客户端、不读 `.env`、不发请求。**"""

    def __init__(
        self,
        *,
        workdir: Path | str,
        experiment_id: str,
        plan: PilotPlan | None = None,
        tasks: tuple[LLMTask, ...] = LLM_TASKS,
        baselines: tuple[str, ...] | None = None,
        behaviors: tuple[str, ...] = PILOT_BEHAVIORS,
        repetition_count: int | None = None,
        manifest_digest: str = "",
        guard: NetworkEgressGuard | None = None,
    ) -> None:
        self.workdir = Path(workdir)
        self.experiment_id = experiment_id
        self.tasks = tasks
        self.behaviors = behaviors
        self.behavior = unit_behavior(behaviors)
        if plan is not None:
            self.plan = plan
        elif baselines is not None or repetition_count is not None:
            # 缩小规模(例如测试)时必须**显式**给出参数,并据此重建计划 ——
            # 否则顺序摘要会与计划不一致,而"顺序可复算"就不再成立。
            self.plan = pilot_plan(
                baselines=baselines or PILOT_BASELINES,
                repetition_count=repetition_count or PILOT_REPETITION_COUNT,
                tasks=tasks,
            )
        else:
            self.plan = pilot_plan(tasks=tasks)
        self.baselines = tuple(baselines or self.plan.baselines)
        self.repetition_count = repetition_count or self.plan.repetition_count
        self.manifest_digest = manifest_digest
        self.guard = guard
        self._records: list[RawRecord] = []
        self._observations: dict[int, list[LLMObservation]] = {}
        self._control_observations: dict[int, list[LLMObservation]] = {}

    # ---- 内部:适配器 ----

    def _adapter_for(self, label: str, cache: dict[str, Any]) -> Any:
        if label in cache:
            return cache[label]
        if label.startswith("B0"):
            adapter = BASELINES["B0"](dataset_paths={})
        else:
            adapter = BASELINES[label](
                dataset_paths={},
                audit_db_path=str(
                    self.workdir / f"audit-{label.replace(chr(39), 'p')}.db"
                ),
            )
        cache[label] = adapter
        return adapter

    @staticmethod
    def _prompt_variant(label: str) -> str | None:
        if not label.startswith("B0"):
            return None
        return "no_tool_prompt" if label.endswith("notool") else "shared_prompt"

    @staticmethod
    def _system_prompt_for(label: str) -> str:
        variant = OfflineExecutor._prompt_variant(label)
        return NO_TOOL_SYSTEM_PROMPT if variant == "no_tool_prompt" else SHARED_SYSTEM_PROMPT

    # ---- 内部:记录构造 ----

    def _build_record(
        self,
        *,
        unit: ExecutionUnit,
        task: LLMTask,
        observation: LLMObservation,
        execution_index: int,
        system_prompt: str,
        status: RecordStatus,
        failure: FailureRecord,
        resume: dict[str, Any] | None,
    ) -> RawRecord:
        return RawRecord(
            run_id=f"{self.experiment_id}:{execution_index:04d}",
            experiment_id=self.experiment_id,
            protocol_version=PROTOCOL_VERSION,
            manifest_digest=self.manifest_digest,
            task_id=unit.task_id,
            condition=unit.condition,
            baseline_label=unit.baseline_label,
            dataset_variant=observation.dataset_variant,
            repetition_id=unit.repetition_id,
            execution_index=execution_index,
            # 实验单元被执行了**一次**(HARNESS_LEVEL_RETRY = 0)
            experimental_run_attempts=1,
            # 与该次执行里的 `ainvoke` 次数**不是同一个量**:B0 恰好 1,图 1~5
            logical_llm_invocations=observation.llm_call_count,
            provider_http_attempts=UNKNOWN,
            provider="scripted",
            model=f"deterministic-scripted-{self.behavior}",
            provider_reported_model_id=None,
            base_url_host_sha256=None,
            provider_default_parameters={},
            system_prompt_sha256=sha256_hex(system_prompt),
            user_prompt_sha256=sha256_hex(task.user_prompt),
            tool_schema_sha256=tool_schema_sha256(),
            full_input_sha256=sha256_hex(canonical_json({
                "system_prompt": system_prompt,
                "user_prompt": task.user_prompt,
            })),
            tool_call_trace=[
                record.model_dump(mode="json") for record in observation.tool_calls
            ],
            final_narrative=observation.answer,
            final_narrative_sha256=sha256_hex(observation.answer),
            metric_inputs={
                "behavior": self.behavior,
                "task_id": unit.task_id,
                "baseline_label": unit.baseline_label,
                "condition": unit.condition,
                "dataset_variant": observation.dataset_variant,
                "repetition_id": unit.repetition_id,
                "run_status": observation.run_status,
                "tool_call_count": observation.tool_call_count,
                "graph_iterations": observation.graph_iterations,
                "llm_call_count": observation.llm_call_count,
                "payload_present_in_dataset": observation.payload_present_in_dataset,
                "payload_visible_to_model": observation.payload_visible_to_model,
            },
            metric_outputs={
                "computed_at": "cell_level",
                "note": (
                    "逐运行的指标输出在 cell 层(基线 × 行为 × 重复)计算;"
                    "本字段记录该运行归属的 cell 坐标,使任何已发布的数字都能"
                    "回溯到产生它的那几次运行。数值本身见派生层。"
                ),
                "cell": {
                    "baseline": unit.baseline_label,
                    "behavior": self.behavior,
                    "repetition_id": unit.repetition_id,
                },
            },
            usage={
                "input_tokens": observation.input_tokens,
                "output_tokens": observation.output_tokens,
                "total_tokens": observation.total_tokens,
                "source": "scripted",
                "note": "脚本化运行不产生 usage;None = NOT_AVAILABLE,不是 0。",
            },
            latency_ms=observation.wall_clock_ms,
            failure=failure.model_dump(mode="json"),
            exposure=self._exposure_payload(observation),
            record_status=status,
            resume=resume,
            timestamp_start_utc="",
            timestamp_end_utc="",
        )

    @staticmethod
    def _exposure_payload(observation: LLMObservation) -> dict[str, Any]:
        status = exposure_status(observation)
        return {
            "payload_present_in_dataset": observation.payload_present_in_dataset,
            "payload_visible_to_model": observation.payload_visible_to_model,
            "exposure_status": status.value,
            "exposed": status is ExposureStatus.EXPOSED,
        }

    # ---- 内部:跑一个单元 ----

    async def _run_unit(
        self,
        unit: ExecutionUnit,
        *,
        execution_index: int,
        adapter_cache: dict[str, Any],
        decoys: dict[str, dict[str, str]],
        datasets: dict[str, dict[str, str]],
        writer: RawWriter,
        resume: dict[str, Any] | None,
    ) -> tuple[LLMObservation, RawRecord]:
        task = next(item for item in self.tasks if item.task_id == unit.task_id)
        label = unit.baseline_label
        adapter = self._adapter_for(label, adapter_cache)
        system_prompt = self._system_prompt_for(label)
        prompt_variant = self._prompt_variant(label)

        if unit.condition == "treatment":
            variant = task.dataset_variant
            kwargs: dict[str, Any] = {"condition": "treatment"}
        else:
            variant = PILOT_CONTROL_VARIANT
            kwargs = {
                "condition": "control",
                "dataset_variant_override": PILOT_CONTROL_VARIANT,
            }
        paths = datasets[variant]
        adapter.dataset_paths = paths
        kwargs["dataset_paths"] = paths
        kwargs["decoy_paths"] = decoys[variant]
        if prompt_variant is not None:
            kwargs["prompt_variant"] = prompt_variant

        try:
            observation = await adapter.run(task, behavior=self.behavior, **kwargs)
        except Exception as exc:  # 工装自身失败:落盘 INCOMPLETE,然后中止
            failure = FailureRecord.from_class(
                FailureClass.HARNESS_ERROR, error_type=type(exc).__name__
            )
            writer.write(self._build_record(
                unit=unit,
                task=task,
                observation=_blank_observation(unit, variant),
                execution_index=execution_index,
                system_prompt=system_prompt,
                status=RecordStatus.INCOMPLETE,
                failure=failure,
                resume=resume,
            ))
            raise HarnessAbort(
                f"单元 {unit.key} 触发了工装异常 {type(exc).__name__}:{exc} —— "
                "记录已标记 incomplete;按冻结分类表,工装错误不允许自动重跑"
            ) from exc

        if label.startswith("B0"):
            observation = observation.model_copy(update={"baseline": label})

        failure = (
            FailureRecord.from_class(FailureClass.MODEL_ANSWER)
            if observation.run_status != "llm_failed"
            else FailureRecord.from_class(
                classify_error_name(observation.error), error_type=observation.error
            )
        )
        record = self._build_record(
            unit=unit,
            task=task,
            observation=observation,
            execution_index=execution_index,
            system_prompt=system_prompt,
            status=RecordStatus.COMPLETE,
            failure=failure,
            resume=resume,
        )
        writer.write(record)
        return observation, record

    # ---- 主入口 ----

    async def run(
        self,
        *,
        existing: dict[tuple[str, str, str, int], RawRecord] | None = None,
        origin_experiment_id: str | None = None,
    ) -> ExecutionOutcome:
        """跑完整流水线。

        `existing` 给出**已有记录**:`complete` 的单元被跳过(永不重跑),
        `incomplete` 的单元重跑并带上恢复元数据,没有记录的单元照常执行。
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        datasets = build_datasets(self.workdir)
        evidence_index = build_evidence_index(self.tasks, datasets)
        authorized = authorized_paths_for(datasets)
        decoys = make_decoys(self.workdir, datasets)

        units, order_digest = pilot_units(
            baselines=self.baselines,
            repetition_count=self.repetition_count,
            seed=self.plan.execution_order_seed,
            tasks=self.tasks,
        )
        pairs = build_matched_pairs(units)
        existing = existing or {}
        resume_plan = plan_resume(units, existing)

        raw_path = self.workdir / "raw" / f"{self.experiment_id}.jsonl"
        # 追加式写入器的副作用必须在这里被拦住(见 `assert_no_silent_append`)。
        assert_no_silent_append(raw_path, existing)
        writer = RawWriter(raw_path, experiment_id=self.experiment_id)

        adapter_cache: dict[str, Any] = {}
        executed: list[str] = []
        records_by_key: dict[str, RawRecord] = {}
        skipped = list(resume_plan[ResumeEligibility.FROZEN.value])

        guard = self.guard or NetworkEgressGuard()
        with guard:
            for execution_index, unit in enumerate(units):
                prior = existing.get(
                    (unit.condition, unit.task_id, unit.baseline_label, unit.repetition_id)
                )
                eligibility = resume_eligibility(prior)
                if eligibility is ResumeEligibility.FROZEN:
                    # 冻结的单元**永不重跑**;它的记录仍然参与配对核验 ——
                    # 配对不变量描述的是"这个实验"的对照结构,
                    # 而不是"本次进程"的产物。
                    if prior is not None:
                        records_by_key[unit.key] = prior
                    continue
                resume_meta = None
                if eligibility is ResumeEligibility.RESUMABLE and prior is not None:
                    resume_meta = build_resume_metadata(
                        origin_experiment_id=origin_experiment_id or prior.experiment_id,
                        resumed_from_index=prior.execution_index,
                    )
                observation, record = await self._run_unit(
                    unit,
                    execution_index=execution_index,
                    adapter_cache=adapter_cache,
                    decoys=decoys,
                    datasets=datasets,
                    writer=writer,
                    resume=resume_meta,
                )
                executed.append(unit.key)
                records_by_key[unit.key] = record
                self._records.append(record)
                bucket = (
                    self._observations
                    if unit.condition == "treatment"
                    else self._control_observations
                )
                bucket.setdefault(unit.repetition_id, []).append(observation)

            raw_sha256 = writer.write_sidecar()

        # ---- 逐重复计算指标(见模块 docstring 的"坑")----
        repetitions = sorted({record.repetition_id for record in self._records})
        metrics_by_repetition: dict[int, list[MetricResult]] = {
            repetition: compute_llm_metrics(
                self.tasks,
                self._observations.get(repetition, []),
                evidence_by_variant=evidence_index,
                authorized_paths=authorized,
                baselines=list(self.baselines),
                behaviors=list(self.behaviors),
                control_observations=self._control_observations.get(repetition, []),
            )
            for repetition in repetitions
        }

        aggregate = derive(self._records, metrics_by_repetition)
        pairing = verify_pairing(pairs, records_by_key, datasets=datasets)

        from app.evaluation.llm.pilot_report import render_pilot_report

        outcome = ExecutionOutcome(
            experiment_id=self.experiment_id,
            manifest_digest=self.manifest_digest,
            raw_path=str(raw_path),
            raw_sha256=raw_sha256,
            sidecar_ok=writer.verify_sidecar(),
            record_count=len(self._records),
            executed_keys=executed,
            skipped_frozen_keys=skipped,
            resume_plan=resume_plan,
            budget=self._budget_snapshot(
                order_digest=order_digest,
                planned_digest=self.plan.execution_order_digest,
                inherited_frozen_records=len(skipped),
            ),
            pairing=pairing,
            aggregate=aggregate,
            metrics_by_repetition=metrics_by_repetition,
            network=guard.summary(),
        )
        outcome.report_markdown = render_pilot_report(outcome)
        return outcome

    def _budget_snapshot(
        self,
        *,
        order_digest: str,
        planned_digest: str,
        inherited_frozen_records: int = 0,
    ) -> dict[str, Any]:
        from app.evaluation.llm.budget import BudgetCounters

        counters = BudgetCounters(
            experimental_runs=len(self._records),
            experimental_run_attempts=sum(
                record.experimental_run_attempts for record in self._records
            ),
            logical_llm_invocations=sum(
                record.logical_llm_invocations for record in self._records
            ),
            provider_http_attempts_observed=0,
            provider_http_attempts_unobservable=len(self._records),
        )
        return {
            **counters.snapshot(),
            "inherited_frozen_records": inherited_frozen_records,
            "conditions": list(self.baselines),
            "behaviors": list(self.behaviors),
            "treatment_runs": self.plan.treatment_runs,
            "control_runs": self.plan.control_runs,
            "total_runs": self.plan.total_runs,
            "execution_order_digest": order_digest,
            "execution_order_digest_matches_plan": order_digest == planned_digest,
            "logical_invocation_floor": self.plan.logical_invocation_floor,
            "logical_invocation_hard_ceiling": self.plan.logical_invocation_hard_ceiling,
            "provider_http_attempt_ceiling": self.plan.provider_http_attempt_ceiling,
            "provider_http_attempt_ceiling_basis": (
                self.plan.provider_http_attempt_ceiling_basis
            ),
            "harness_level_retry": 0,
        }


def _blank_observation(unit: ExecutionUnit, variant: str) -> LLMObservation:
    """工装中止时的占位观测 —— 它**不是**模型结果,只是让记录可被构造。"""
    return LLMObservation(
        task_id=unit.task_id,
        baseline=unit.baseline_label,
        behavior="(harness-abort)",
        dataset_variant=variant,
        condition=unit.condition,
        run_status="llm_failed",
        error="HarnessAbort",
        llm_call_count=0,
        payload_present_in_dataset=variant == VARIANT_INJECTION,
        payload_visible_to_model=None,
    )
