"""离线预算治理 —— 四个**互不混淆**的计数器。

为什么必须分开
--------------
"跑了多少次"有三种完全不同的含义,混用它们会让预算、覆盖率与
可复现性同时失真:

    experimental_runs            有几个实验单元被跑了
    experimental_run_attempts    实验单元被**执行**了几次(首试点恒为 1)
    logical_llm_invocations      这些运行里发生了几次 `ainvoke`
    provider_http_attempts       实际发出去了几个 HTTP 请求(含 SDK 内部重试)

`experimental_run_attempts` 与 `logical_llm_invocations` **不是一回事**:
B0 恰好 1 次,而图条件可能 1~5 次。把它们设成相等会掩盖"图跑了几轮"
这个事实,并让调用预算少算最多 5 倍。

`provider_http_attempts` 与 `logical_llm_invocations` 也**不是一回事**:
SDK 的 `max_retries=2` 会让一次 `ainvoke` 最多发 3 个 HTTP 请求。
`972 = 324 × 3` 是**在这个 provider 默认值成立的前提下的理论上界**,
不是实测值 —— 清单里必须写明它是假设,而不是把它当成观测。
"""
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.llm.dataset import LLM_TASKS
from app.evaluation.llm.tasks import LLMTask

#: 无法观测时的显式标记。**不伪造 0** —— 0 会被读成"零消耗"。
UNKNOWN = "UNKNOWN"

#: 生产图的迭代上限(`SecurityAgent` / `create_agent_graph` 默认值)。
MAX_GRAPH_ITERATIONS = 5

#: B0 直连恰好 1 次逻辑调用。
B0_LOGICAL_INVOCATIONS = 1

#: 实测到的 SDK 层重试默认值(`root_client.max_retries`)。**provider 默认行为**,
#: 不是我们设定的参数。
SDK_MAX_RETRIES_OBSERVED = 2

#: 每个逻辑调用在 SDK 默认重试下的最大物理尝试数。
MAX_HTTP_ATTEMPTS_PER_INVOCATION = SDK_MAX_RETRIES_OBSERVED + 1


class BudgetExceeded(AssertionError):
    """预算耗尽。**异常而不是返回值** —— 预算超限必须中断执行。"""


@dataclass
class BudgetCounters:
    """四个计数器。`provider_http_attempts` 用"可观测 + 不可观测"两段表示。"""

    experimental_runs: int = 0
    experimental_run_attempts: int = 0
    logical_llm_invocations: int = 0
    provider_http_attempts_observed: int = 0
    provider_http_attempts_unobservable: int = 0

    @property
    def provider_http_attempts(self) -> int | str:
        """任何一个调用不可观测,总数就是 `UNKNOWN` —— 不部分求和。"""
        if self.provider_http_attempts_unobservable:
            return UNKNOWN
        return self.provider_http_attempts_observed

    def snapshot(self) -> dict[str, Any]:
        return {
            "experimental_runs": self.experimental_runs,
            "experimental_run_attempts": self.experimental_run_attempts,
            "logical_llm_invocations": self.logical_llm_invocations,
            "provider_http_attempts": self.provider_http_attempts,
            "provider_http_attempts_observed": self.provider_http_attempts_observed,
            "provider_http_attempts_unobservable": self.provider_http_attempts_unobservable,
        }


@dataclass(frozen=True)
class PilotBudget:
    """冻结的试点规模与上限(**由结构推导,不手写常量**)。"""

    baseline_labels: tuple[str, ...]
    repetition_count: int
    task_count: int
    injection_task_count: int
    treatment_runs: int
    control_runs: int
    total_runs: int
    logical_invocation_floor: int
    logical_invocation_hard_ceiling: int
    provider_http_attempt_ceiling: int
    provider_http_attempt_ceiling_basis: str
    soft_call_ceiling: int

    def as_manifest_fields(self) -> dict[str, Any]:
        return {
            "treatment_runs": self.treatment_runs,
            "control_runs": self.control_runs,
            "total_runs": self.total_runs,
            "logical_invocation_hard_ceiling": self.logical_invocation_hard_ceiling,
            "provider_http_attempt_ceiling": self.provider_http_attempt_ceiling,
            "provider_http_attempt_ceiling_basis": self.provider_http_attempt_ceiling_basis,
        }


def _is_direct(label: str) -> bool:
    """B0 家族 = 直连、无工具、恰好 1 次逻辑调用。"""
    return label.startswith("B0")


def pilot_budget(
    *,
    baseline_labels: tuple[str, ...],
    repetition_count: int,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
) -> PilotBudget:
    """从**实际图行为**推导规模与上限。

    硬上界用 `max_iterations=5`(每次图运行 ≤5 次 `ainvoke`);
    下界用"每次图运行恰好 1 次"(模型不调工具直接作答)。
    两者都是**结构性**边界,不是经验估计。
    """
    task_count = len(tasks)
    injection_task_count = sum(
        1 for task in tasks if task.security_contract.injection is not None
    )
    direct = [label for label in baseline_labels if _is_direct(label)]
    graph = [label for label in baseline_labels if not _is_direct(label)]

    treatment_runs = task_count * repetition_count * len(baseline_labels)
    control_runs = injection_task_count * repetition_count * len(baseline_labels)
    total_runs = treatment_runs + control_runs

    direct_runs = (task_count + injection_task_count) * repetition_count * len(direct)
    graph_runs = (task_count + injection_task_count) * repetition_count * len(graph)

    floor = direct_runs * B0_LOGICAL_INVOCATIONS + graph_runs * 1
    ceiling = (
        direct_runs * B0_LOGICAL_INVOCATIONS
        + graph_runs * MAX_GRAPH_ITERATIONS
    )
    http_ceiling = ceiling * MAX_HTTP_ATTEMPTS_PER_INVOCATION

    return PilotBudget(
        baseline_labels=baseline_labels,
        repetition_count=repetition_count,
        task_count=task_count,
        injection_task_count=injection_task_count,
        treatment_runs=treatment_runs,
        control_runs=control_runs,
        total_runs=total_runs,
        logical_invocation_floor=floor,
        logical_invocation_hard_ceiling=ceiling,
        provider_http_attempt_ceiling=http_ceiling,
        provider_http_attempt_ceiling_basis=(
            f"理论上界 = 逻辑调用上界 {ceiling} × 每次调用的最大 HTTP 尝试数 "
            f"{MAX_HTTP_ATTEMPTS_PER_INVOCATION}"
            f"(= 1 + SDK 观测到的 provider 默认 max_retries {SDK_MAX_RETRIES_OBSERVED})。"
            "⚠️ 这是**假设下的上界**,不是实测值;物理尝试不可观测时记 UNKNOWN。"
        ),
        soft_call_ceiling=(ceiling * 2) // 3,
    )


@dataclass
class BudgetGovernor:
    """离线预算治理器。

    **不重跑任何东西** —— 它只记账、告警、必要时中止。
    """

    budget: PilotBudget
    counters: BudgetCounters = field(default_factory=BudgetCounters)
    warnings: list[str] = field(default_factory=list)

    def record_run(
        self,
        *,
        logical_invocations: int,
        provider_http_attempts: int | None = None,
    ) -> None:
        """登记**一个已执行的实验单元**。

        `experimental_run_attempts` 每次 +1 且只 +1 —— 因为
        `HARNESS_LEVEL_RETRY = 0`,单元不会被执行第二次。
        它与 `logical_invocations` **是两个量**,刻意不互相赋值。
        """
        self.counters.experimental_runs += 1
        self.counters.experimental_run_attempts += 1
        self.counters.logical_llm_invocations += logical_invocations
        if provider_http_attempts is None:
            self.counters.provider_http_attempts_unobservable += 1
        else:
            self.counters.provider_http_attempts_observed += provider_http_attempts

    def check(self) -> None:
        """超限即抛。**每次登记后都应调用**。"""
        if self.counters.logical_llm_invocations > self.budget.logical_invocation_hard_ceiling:
            raise BudgetExceeded(
                f"逻辑 LLM 调用 {self.counters.logical_llm_invocations} 超过硬上界 "
                f"{self.budget.logical_invocation_hard_ceiling} —— 立即 ABORT"
            )
        observed = self.counters.provider_http_attempts_observed
        if observed > self.budget.provider_http_attempt_ceiling:
            raise BudgetExceeded(
                f"观测到的物理 HTTP 尝试 {observed} 超过上界 "
                f"{self.budget.provider_http_attempt_ceiling} —— 立即 ABORT"
            )
        if (
            self.counters.logical_llm_invocations > self.budget.soft_call_ceiling
            and not self.warnings
        ):
            self.warnings.append(
                f"逻辑 LLM 调用 {self.counters.logical_llm_invocations} 已超过软上限 "
                f"{self.budget.soft_call_ceiling}(继续,但需在报告中说明)"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.counters.snapshot(),
            "ceilings": self.budget.as_manifest_fields(),
            "soft_call_ceiling": self.budget.soft_call_ceiling,
            "harness_level_retry": 0,
            "warnings": list(self.warnings),
        }
