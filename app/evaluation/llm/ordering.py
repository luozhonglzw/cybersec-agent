"""确定性执行顺序(**sha256 排序,不依赖 PRNG**)。

为什么随机化
------------
执行顺序若与任务身份固定绑定,provider 侧的瞬时效应(限流、灰度发布、
区域性延迟)就会**系统性地**落到同一批任务上 —— 那是一个与任务混淆的
混淆变量。打散顺序可以把这类效应摊平。

为什么用 sha256 排序而不是 `random`
-----------------------------------
`random.Random(seed).shuffle()` 的顺序在**同一次运行的同一个解释器**里
当然可复现,但它把可复现性押在 CPython 的 PRNG 实现细节上:换一个
Python 版本、换一种 `shuffle` 实现,同一个 seed 会给出**不同的顺序** ——
而实验的可复现性不该依赖这种偶然。

`sha256(seed + key)` 排序是完全**可移植**的:任何语言、任何版本、
任何平台,只要 sha256 不变,顺序就不变。而且它不需要"相信"某个 PRNG:
顺序由摘要直接定义,任何人可以独立复算。

同 seed + 同清单 ⇒ 同顺序;换 seed ⇒ 换顺序。两条都由
`execution_order_digest` 机械可证。
"""
import hashlib
from dataclasses import dataclass
from typing import Literal

from app.evaluation.llm.dataset import LLM_TASKS
from app.evaluation.llm.tasks import LLMTask

Condition = Literal["treatment", "control"]

#: **冻结的执行顺序种子。** 首次真实调用前必须固定;此后不得更改 ——
#: 改它等于换一套执行顺序,而活动实验不允许被静默改动。
EXECUTION_ORDER_SEED = "9.2d2-2026-09-23-sha256-order-v1"


@dataclass(frozen=True, order=True)
class ExecutionUnit:
    """一个实验单元 = 一次运行。

    `condition` 进 key 是必要的:treatment 与 control 是**两个不同的运行**,
    它们必须被独立排序,否则"配对"会退化成"相邻执行"。
    """

    condition: Condition
    task_id: str
    baseline_label: str
    repetition_id: int

    @property
    def key(self) -> str:
        return f"{self.condition}:{self.task_id}:{self.baseline_label}:{self.repetition_id}"


def order_hash(unit: ExecutionUnit, seed: str) -> str:
    """单元的排序摘要。刻意把 seed 与 key 用 `:` 连接后整体取 sha256。"""
    return hashlib.sha256(f"{seed}:{unit.key}".encode("utf-8")).hexdigest()


def build_execution_units(
    *,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
    baseline_labels: tuple[str, ...],
    repetition_count: int,
) -> list[ExecutionUnit]:
    """枚举全部实验单元(**未排序**)。

    treatment = 每个任务 × 每个基线标签 × 每个重复;
    control   = **仅带注入契约的任务** × 每个基线标签 × 每个重复。
    """
    units: list[ExecutionUnit] = []
    for baseline_label in baseline_labels:
        for repetition_id in range(1, repetition_count + 1):
            for task in tasks:
                units.append(ExecutionUnit(
                    condition="treatment",
                    task_id=task.task_id,
                    baseline_label=baseline_label,
                    repetition_id=repetition_id,
                ))
                if task.security_contract.injection is not None:
                    units.append(ExecutionUnit(
                        condition="control",
                        task_id=task.task_id,
                        baseline_label=baseline_label,
                        repetition_id=repetition_id,
                    ))
    return units


def order_units(units: list[ExecutionUnit], seed: str) -> list[ExecutionUnit]:
    """按 `(sha256(seed:key), key)` 升序排列。

    摘要相同时用 `key` 兜底 —— 摘要碰撞在实践中不会发生,但"顺序唯一"
    必须是**构造性**保证,不能靠概率。
    """
    return sorted(units, key=lambda unit: (order_hash(unit, seed), unit.key))


def execution_order_digest(ordered: list[ExecutionUnit], seed: str) -> str:
    """顺序的摘要 —— 让"两次生成了同一个顺序"可被机械核验。"""
    payload = "\n".join(f"{order_hash(unit, seed)}|{unit.key}" for unit in ordered)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_execution_order(
    *,
    baseline_labels: tuple[str, ...],
    repetition_count: int,
    seed: str = EXECUTION_ORDER_SEED,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
) -> tuple[list[ExecutionUnit], str]:
    """返回 `(有序单元, 顺序摘要)`。"""
    ordered = order_units(
        build_execution_units(
            tasks=tasks, baseline_labels=baseline_labels, repetition_count=repetition_count
        ),
        seed,
    )
    return ordered, execution_order_digest(ordered, seed)
