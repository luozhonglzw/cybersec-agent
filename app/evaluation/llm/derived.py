"""派生聚合(**离线**):从原始记录与逐重复指标集合出可审阅的计数与比例。

为什么派生层读**原始记录**而不是内存观测
------------------------------------
派生结果必须是**可独立重算**的。原始 JSONL 是唯一被持久化的事实来源;
如果派生层读的是内存里的观测对象,那么"报告里的数"与"落盘的记录"之间
就多了一条无人核验的暗管 —— 事后任何一方被改动都不会被发现。

因此:

    计数类(失败 / 暴露 / 覆盖率)  ←  RawRecord(**落盘的那份**)
    比例类(follow / match / …)     ←  逐重复的 MetricResult(**已定义的判定口径**)

比例不从记录重算,是因为**判定口径只能有一份**。若派生层自己再实现一遍
"什么算命中",两份实现迟早会漂移,而漂移后的报告无法自证哪一份是对的。

n=3 的措辞是**冻结的**
--------------------
允许:观测比例 / 原始分子分母 / 描述性 Wilson 区间。
禁止:把 n=3 当作可靠的总体表现估计、把区间外推到一般模型能力、
或做显著性检验与 p 值主张。相关措辞以常量形式冻结在本模块,
报告必须原样引用 —— 不允许每个渲染器自己"换个说法"。
"""
import math
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from app.evaluation.llm.exposure import (
    ExposureCoverage,
    coverage_by_condition,
    exposure_coverage,
)
from app.evaluation.llm.failures import FailureClass, failure_summary
from app.evaluation.llm.metrics import MetricResult
from app.evaluation.llm.protocol import METRIC_SCHEMA
from app.evaluation.llm.raw import RawRecord, RecordStatus

#: **冻结措辞。** 只要报告里出现 Wilson 区间,就必须原样带上这一句。
WILSON_CAVEAT = (
    "Wilson intervals are descriptive uncertainty summaries over the observed "
    "pilot repetitions, not inferential confidence intervals for general model "
    "performance."
)

#: n=3 的允许 / 禁止清单。
N3_LIMITATION = (
    "首个试点的重复次数为 n=3。允许:观测比例、原始分子/分母、描述性 Wilson 区间。"
    "禁止:把 n=3 当作可靠的总体表现估计、把区间外推到一般模型能力、"
    "做显著性检验或给出 p 值主张。"
)

#: 不做显著性检验 —— 独立陈述,便于机械检查报告是否遗漏。
NO_P_VALUE_STATEMENT = (
    "本报告不报告 p 值,不做任何显著性检验。原因不是「懒得算」,"
    "而是 n=3 的重复次数不足以支撑任何显著性主张:给出 p 值会让读者"
    "误以为存在统计功效,而那正是过度解读的入口。"
)

#: 不产出排行榜 / 不产出 Agent 总分。
NO_LEADERBOARD_STATEMENT = (
    "本报告不产出排行榜,也不产出跨条件汇总的「Agent 总分」。"
    "B0-shared / B0-notool / B2' / B3 是四个**并列的实验条件**,不是四个选手;"
    "把它们排成一列会掩盖 B0-shared 刻意处于劣势这一设计事实。"
)

#: 不构成一般安全性结论。
NO_GENERAL_SAFETY_CLAIM = (
    "本报告不构成任何一般安全性结论。架构不变量是**回归护栏**,"
    "不是 safety score / containment rate / Agent quality。"
)

#: Wilson 区间的 z 值(95%,正态近似)。固定为常量以便复算。
WILSON_Z = 1.96

#: 参与"比例"输出的指标 —— **从 `METRIC_SCHEMA` 的 `unit` 推导**,
#: 不手写清单。单位里含"比例"的即比率型指标;改单位会自动反映到这里。
PROPORTION_METRIC_IDS: tuple[str, ...] = tuple(
    entry.metric_id for entry in METRIC_SCHEMA if "比例" in entry.unit
)

BucketDimension = Literal["overall", "condition", "baseline", "task", "repetition"]


def wilson_interval(
    numerator: int, denominator: int, *, z: float = WILSON_Z
) -> tuple[float, float] | None:
    """Wilson score 区间。`denominator <= 0` 时返回 `None`。

    **描述性**用途:它刻画的是"这几次重复里观测到的比例有多不稳",
    不是"真实总体比例的置信区间"。两者的区别见 `WILSON_CAVEAT`。
    """
    if denominator <= 0:
        return None
    n = denominator
    p = numerator / n
    z_squared = z * z
    scale = 1.0 + z_squared / n
    center = (p + z_squared / (2 * n)) / scale
    spread = (z * math.sqrt(p * (1 - p) / n + z_squared / (4 * n * n))) / scale
    return (max(0.0, round(center - spread, 6)), min(1.0, round(center + spread, 6)))


class Proportion(BaseModel):
    """一个比例:原始分子 / 原始分母 / 观测值 / 描述性 Wilson 区间。

    刻意**同时**保留原始分子与分母:只给比例的话,读者无法判断
    "0.0" 是"100 次里 0 次"还是"1 次里 0 次" —— 而这两者的信息量
    差了两个数量级。
    """

    label: str = Field(description="可读标签:metric|baseline|behavior|…")
    metric_id: str
    scope: dict[str, str] = Field(default_factory=dict, description="该比例覆盖的维度取值")
    numerator: int = 0
    denominator: int = 0
    observed: float | None = None
    wilson_low: float | None = None
    wilson_high: float | None = None
    status: Literal["ok", "not_evaluable"] = "ok"
    reason: str | None = None
    repetitions_pooled: int = Field(
        default=0, description="该比例合并了几个重复 —— n=3 时必须如实写出",
    )

    @classmethod
    def from_counts(
        cls,
        numerator: int,
        denominator: int,
        *,
        metric_id: str,
        label: str,
        scope: dict[str, str] | None = None,
        repetitions_pooled: int = 0,
        reason: str | None = None,
    ) -> "Proportion":
        """分母为 0 记 `not_evaluable` —— **不记 0,也不记 1**。"""
        if denominator <= 0:
            return cls(
                label=label,
                metric_id=metric_id,
                scope=scope or {},
                numerator=0,
                denominator=0,
                status="not_evaluable",
                reason=reason or "本比例没有可评测的样本",
                repetitions_pooled=repetitions_pooled,
            )
        low, high = wilson_interval(numerator, denominator) or (None, None)
        return cls(
            label=label,
            metric_id=metric_id,
            scope=scope or {},
            numerator=numerator,
            denominator=denominator,
            observed=round(numerator / denominator, 6),
            wilson_low=low,
            wilson_high=high,
            status="ok",
            repetitions_pooled=repetitions_pooled,
        )


class DerivedBucket(BaseModel):
    """一个维度取值上的计数汇总。**只放计数,不放比例。**"""

    dimension: BucketDimension
    key: str
    runs: int = 0
    model_results: int = 0
    provider_failures: int = 0
    infra_failures: int = 0
    not_evaluable: int = Field(
        default=0,
        description=(
            "没有可评测模型输出的运行数(失败分类的 `count_as_model_result == False`)。"
            "它是**覆盖率损失**,不是模型失败。"
        ),
    )
    payload_bearing: int = 0
    exposed: int = 0
    not_exposed: int = 0
    undetermined_exposure: int = 0
    complete_records: int = 0
    incomplete_records: int = 0


class DerivedAggregate(BaseModel):
    """完整的派生聚合产物。"""

    record_count: int = 0
    buckets: list[DerivedBucket] = Field(default_factory=list)
    failure_totals: dict[str, Any] = Field(default_factory=dict)
    exposure: ExposureCoverage = Field(default_factory=ExposureCoverage)
    exposure_by_condition: dict[str, ExposureCoverage] = Field(default_factory=dict)
    proportions_by_cell: list[Proportion] = Field(default_factory=list)
    proportions_by_repetition: list[Proportion] = Field(default_factory=list)

    wilson_emitted: bool = False
    wilson_caveat: str = WILSON_CAVEAT
    n3_limitation: str = N3_LIMITATION
    no_p_value_statement: str = NO_P_VALUE_STATEMENT
    no_leaderboard_statement: str = NO_LEADERBOARD_STATEMENT
    no_general_safety_claim: str = NO_GENERAL_SAFETY_CLAIM

    def bucket(self, dimension: BucketDimension, key: str) -> DerivedBucket | None:
        for item in self.buckets:
            if item.dimension == dimension and item.key == key:
                return item
        return None

    def proportion(self, metric_id: str, **scope: str) -> Proportion | None:
        for item in self.proportions_by_cell:
            if item.metric_id != metric_id:
                continue
            if all(item.scope.get(k) == v for k, v in scope.items()):
                return item
        return None


# ---------------------------------------------------------------------------
# 计数(来自原始记录)
# ---------------------------------------------------------------------------


def _bucket_for(dimension: BucketDimension, key: str) -> DerivedBucket:
    return DerivedBucket(dimension=dimension, key=key)


def _accumulate(bucket: DerivedBucket, record: RawRecord) -> None:
    bucket.runs += 1
    failure = record.failure or {}
    if failure.get("count_as_model_result"):
        bucket.model_results += 1
    else:
        bucket.not_evaluable += 1
    if failure.get("count_as_provider_failure"):
        bucket.provider_failures += 1
    if failure.get("count_as_infra_failure"):
        bucket.infra_failures += 1
    if record.record_status == RecordStatus.COMPLETE:
        bucket.complete_records += 1
    else:
        bucket.incomplete_records += 1
    if record.payload_present_in_dataset:
        bucket.payload_bearing += 1
        if record.exposed:
            bucket.exposed += 1
        elif record.payload_visible_to_model is None:
            bucket.undetermined_exposure += 1
        else:
            bucket.not_exposed += 1


def count_buckets(records: Iterable[RawRecord]) -> list[DerivedBucket]:
    """逐维度汇总计数:`overall` / `condition` / `baseline` / `task` / `repetition`。"""
    records = list(records)
    buckets: dict[tuple[str, str], DerivedBucket] = {}

    def slot(dimension: BucketDimension, key: str) -> DerivedBucket:
        return buckets.setdefault(
            (dimension, key), _bucket_for(dimension, key)
        )

    for record in records:
        for dimension, key in (
            ("overall", "ALL"),
            ("condition", record.condition),
            ("baseline", record.baseline_label),
            ("task", record.task_id),
            ("repetition", str(record.repetition_id)),
        ):
            _accumulate(slot(dimension, key), record)

    order = {"overall": 0, "condition": 1, "baseline": 2, "task": 3, "repetition": 4}
    return sorted(buckets.values(), key=lambda item: (order[item.dimension], item.key))


def failure_classes_of(records: Iterable[RawRecord]) -> list[FailureClass]:
    classes: list[FailureClass] = []
    for record in records:
        value = (record.failure or {}).get("failure_class")
        if value is None:
            continue
        try:
            classes.append(FailureClass(value))
        except ValueError:  # pragma: no cover - 词表外取值应已被分类器拦住
            continue
    return classes


# ---------------------------------------------------------------------------
# 比例(来自逐重复的指标结果)
# ---------------------------------------------------------------------------


def _cell_lookup(metrics: list[MetricResult]) -> dict[tuple[str, str, str], Any]:
    table: dict[tuple[str, str, str], Any] = {}
    for metric in metrics:
        for cell in metric.cells:
            table[(metric.metric_id, cell.baseline, cell.behavior)] = cell
    return table


def _scope_label(metric_id: str, **scope: str) -> str:
    parts = [metric_id] + [f"{key}={value}" for key, value in sorted(scope.items())]
    return "|".join(parts)


def proportions_by_cell(
    metrics_by_repetition: dict[int, list[MetricResult]],
) -> list[Proportion]:
    """把**逐重复**的单元合并成一个比例(原始分子 / 分母直接相加)。

    为什么必须逐重复分别算、再合并 —— 而不是把所有重复的观测一次丢给
    `compute_llm_metrics`:后者的观测索引以 `(task_id, baseline, behavior)`
    为键,**同一单元的多次重复会互相覆盖**,于是 n=3 会静默退化成 n=1,
    而分母看起来完全正常。这是 D-2a 明确防住的一类静默失真。
    """
    tables = {rep: _cell_lookup(items) for rep, items in metrics_by_repetition.items()}
    keys = sorted({key for table in tables.values() for key in table})

    out: list[Proportion] = []
    for metric_id, baseline, behavior in keys:
        if metric_id not in PROPORTION_METRIC_IDS:
            continue
        numerator = denominator = 0
        reason: str | None = None
        pooled = 0
        for rep in sorted(tables):
            cell = tables[rep].get((metric_id, baseline, behavior))
            if cell is None:
                continue
            if cell.status == "ok":
                numerator += int(cell.numerator or 0)
                denominator += int(cell.denominator or 0)
                pooled += 1
            else:
                reason = reason or cell.reason
        out.append(Proportion.from_counts(
            numerator,
            denominator,
            metric_id=metric_id,
            label=_scope_label(metric_id, baseline=baseline, behavior=behavior),
            scope={"baseline": baseline, "behavior": behavior},
            repetitions_pooled=pooled,
            reason=reason,
        ))
    return out


def proportions_by_repetition(
    metrics_by_repetition: dict[int, list[MetricResult]],
) -> list[Proportion]:
    """逐重复的比例 —— n=3 的**原始视图**,不做任何合并。"""
    out: list[Proportion] = []
    for rep in sorted(metrics_by_repetition):
        table = _cell_lookup(metrics_by_repetition[rep])
        for metric_id, baseline, behavior in sorted(table):
            if metric_id not in PROPORTION_METRIC_IDS:
                continue
            cell = table[(metric_id, baseline, behavior)]
            if cell.status != "ok":
                continue
            out.append(Proportion.from_counts(
                int(cell.numerator or 0),
                int(cell.denominator or 0),
                metric_id=metric_id,
                label=_scope_label(
                    metric_id, baseline=baseline, behavior=behavior, repetition=str(rep)
                ),
                scope={
                    "baseline": baseline,
                    "behavior": behavior,
                    "repetition": str(rep),
                },
                repetitions_pooled=1,
            ))
    return out


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def derive(
    records: Iterable[RawRecord],
    metrics_by_repetition: dict[int, list[MetricResult]],
) -> DerivedAggregate:
    """派生聚合的唯一入口。"""
    records = list(records)
    by_cell = proportions_by_cell(metrics_by_repetition)
    by_rep = proportions_by_repetition(metrics_by_repetition)
    coverage = coverage_by_condition(records)
    return DerivedAggregate(
        record_count=len(records),
        buckets=count_buckets(records),
        failure_totals=failure_summary(failure_classes_of(records)),
        exposure=exposure_coverage(records),
        exposure_by_condition={key: value for key, value in coverage.items()},
        proportions_by_cell=by_cell,
        proportions_by_repetition=by_rep,
        wilson_emitted=any(
            item.wilson_low is not None for item in (*by_cell, *by_rep)
        ),
    )
