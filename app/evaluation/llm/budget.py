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
SDK 的 `max_retries` 会让一次 `ainvoke` 最多发 `1 + max_retries` 个 HTTP 请求。
`972 = 324 × 3` 是**在 SDK 默认 `max_retries=2` 成立的前提下的理论包络**,
不是实测值 —— 清单里必须写明它是假设,而不是把它当成观测。
D-2d 把 provider 配置显式冻结成 `max_retries = 0`,因此它的配置包络是
`324 × 1 = 324`;972 只作为**历史推导值**保留。

两个作用域(**D-2b**)
--------------------
    本次进程(process)   本次进程实际执行了什么 —— 报告的样本量口径
    实验级(experiment)   继承的已消费量 + 本次进程 —— **硬上界只对实验级生效**

预算是实验级的:续跑时若重新发一份完整额度,上界就不再是上界。
两个作用域**不得合并**:合并会让"本次进程执行了几个单元"凭空变大。
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
#:
#: ⚠️ 它是**观测到的默认值**,不是"当前配置"。D-2d 的 provider 配置把它显式
#: 冻结成 0(`PILOT_MAX_RETRIES`),因此 D-2d 的 configured/theoretical
#: HTTP 尝试包络是 `324 × 1 = 324`,而**不是** `324 × 3 = 972`。
#: 972 作为历史推导值保留(见 `app/evaluation/pilot_config.py` 的分类),
#: 但它**不得**被当作当前 D-2d 清单的执行包络。
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
    #: 该包络建立在哪个 SDK 重试假设上。默认是**观测到的** SDK 默认值,
    #: D-2d 传 0(其冻结 provider 配置)。
    sdk_max_retries_assumption: int = SDK_MAX_RETRIES_OBSERVED

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
    sdk_max_retries: int = SDK_MAX_RETRIES_OBSERVED,
) -> PilotBudget:
    """从**实际图行为**推导规模与上限。

    硬上界用 `max_iterations=5`(每次图运行 ≤5 次 `ainvoke`);
    下界用"每次图运行恰好 1 次"(模型不调工具直接作答)。
    两者都是**结构性**边界,不是经验估计。

    `sdk_max_retries` —— 物理 HTTP 尝试包络的**唯一**假设来源
    ---------------------------------------------------------
    默认值是**观测到的 SDK 默认** `2` ⇒ 既有清单仍是 `972`(**逐字段不变**,
    历史推导值不得被静默改写)。D-2d 显式传 `0`(其冻结 provider 配置),
    于是包络变成 `324 × 1 = 324`。

    把假设做成**参数**而不是常量,是为了让"这个数字建立在什么之上"
    在调用点就可见 —— 两个互相矛盾的假设不能同时藏在两个模块里。
    """
    if sdk_max_retries < 0:
        raise ValueError(f"sdk_max_retries 不得为负:{sdk_max_retries!r}")

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
    attempts_per_invocation = sdk_max_retries + 1
    http_ceiling = ceiling * attempts_per_invocation

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
            f"**配置的理论包络**(不是观测值):逻辑调用上界 {ceiling} × "
            f"每次调用的最大 HTTP 尝试数 {attempts_per_invocation}"
            f"(= 1 + SDK max_retries {sdk_max_retries}) = {http_ceiling}。"
            "⚠️ 这是**假设下的上界**,不是实测值;物理尝试不可观测时记 UNKNOWN。"
        ),
        soft_call_ceiling=(ceiling * 2) // 3,
        sdk_max_retries_assumption=sdk_max_retries,
    )


@dataclass
class InheritedConsumption:
    """**继承的**已消费量(来自上一次运行留下的完整记录)。

    为什么不直接加进 `counters`:`counters` 描述"**本次进程**执行了什么",
    而预算是**实验级**的。把继承量并进 `counters` 会让报告里的
    "本次进程执行单元"凭空变大 —— 那是 F-3 的同一类失真。
    """

    experimental_runs: int = 0
    experimental_run_attempts: int = 0
    logical_llm_invocations: int = 0
    provider_http_attempts_unobservable: int = 0

    def __post_init__(self) -> None:
        for name in (
            "experimental_runs",
            "experimental_run_attempts",
            "logical_llm_invocations",
            "provider_http_attempts_unobservable",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"继承量 {name} 不得为负:{value!r}")


def budget_from_plan(plan: Any) -> PilotBudget:
    """从计划的**已声明**上限构建预算对象(**不重算**)。

    刻意不调用 `pilot_budget()` 重算:重算会**忽略调用方显式声明的上限**,
    于是"把上界写小一点"这种自检会被静默抵消 —— 预算就退化成装饰品。
    """
    return PilotBudget(
        baseline_labels=tuple(plan.baselines),
        repetition_count=plan.repetition_count,
        task_count=plan.task_count,
        injection_task_count=plan.injection_task_count,
        treatment_runs=plan.treatment_runs,
        control_runs=plan.control_runs,
        total_runs=plan.total_runs,
        logical_invocation_floor=plan.logical_invocation_floor,
        logical_invocation_hard_ceiling=plan.logical_invocation_hard_ceiling,
        provider_http_attempt_ceiling=plan.provider_http_attempt_ceiling,
        provider_http_attempt_ceiling_basis=plan.provider_http_attempt_ceiling_basis,
        soft_call_ceiling=plan.soft_call_ceiling,
        sdk_max_retries_assumption=getattr(
            plan, "sdk_max_retries_assumption", SDK_MAX_RETRIES_OBSERVED
        ),
    )


@dataclass
class BudgetGovernor:
    """预算治理器。**它不重跑任何东西** —— 只记账、告警、必要时中止。

    两个作用域,刻意分开
    -------------------
        本次进程(process)    `counters`         报告里的"本次进程执行单元"
        实验级(experiment)   继承 + 本次进程    **硬上界只对实验级生效**

    预算必须是实验级的:续跑时若重新发一份完整额度,上界就不再是上界。

    谁写哪个计数器(**硬分工**)
    --------------------------
        `reserve()`          唯一写 `logical_llm_invocations` 的地方,每次调用 +1
        `record_unit()`      只写单元级与物理尝试级,`logical_*` 一个字都不碰
        `seed_inherited()`   只写继承量

    这条分工不是风格问题:`logical_llm_invocations` 若同时被"每次调用"和
    "每个单元汇总"各加一次,账会翻倍,而两份数字看起来都很合理。
    """

    budget: PilotBudget
    counters: BudgetCounters = field(default_factory=BudgetCounters)
    inherited: InheritedConsumption = field(default_factory=InheritedConsumption)
    warnings: list[str] = field(default_factory=list)

    # ---- 实验级视图 ----

    @property
    def experiment_experimental_runs(self) -> int:
        return self.inherited.experimental_runs + self.counters.experimental_runs

    @property
    def experiment_logical_llm_invocations(self) -> int:
        return self.inherited.logical_llm_invocations + self.counters.logical_llm_invocations

    @property
    def experiment_provider_http_attempts(self) -> int | str:
        """任一单元不可观测,总数就是 `UNKNOWN` —— **不部分求和**。"""
        if (
            self.counters.provider_http_attempts_unobservable
            or self.inherited.provider_http_attempts_unobservable
        ):
            return UNKNOWN
        return self.counters.provider_http_attempts_observed

    # ---- 播种 / 准入 / 预留 / 登记 ----

    def seed_inherited(self, consumption: InheritedConsumption) -> None:
        """播种**已消费**量。刻意不经过 `record_unit()`。

        走 `record_unit()` 会把继承的单元记成本次进程新执行的单元,
        报告里的样本量于是凭空变大 —— 正是 F-3 要防的那类失真。
        """
        if self.inherited != InheritedConsumption():
            raise AssertionError(
                "继承量只能播种一次 —— 重复播种会让上界凭空变紧,"
                "而报告看起来完全正常"
            )
        self.inherited = consumption

    def admit_unit(self, *, min_invocations: int = 1) -> None:
        """**Tier 1:实验单元准入闸门。**

        在一个单元开始做任何工作**之前**调用。它**不消费**额度 ——
        只回答"现在开始这个单元,是否连它的结构性下界都装不下"。
        装不下就立刻拒绝,不产生任何调用。
        """
        if min_invocations < 1:
            raise ValueError(f"min_invocations 至少为 1,收到 {min_invocations!r}")
        projected = self.experiment_logical_llm_invocations + min_invocations
        if projected > self.budget.logical_invocation_hard_ceiling:
            raise BudgetExceeded(
                f"单元准入被拒:已消费 {self.experiment_logical_llm_invocations}"
                f"(继承 {self.inherited.logical_llm_invocations} + 本次进程 "
                f"{self.counters.logical_llm_invocations}),再执行一个单元至少需要 "
                f"{min_invocations} 次逻辑调用,将超过硬上界 "
                f"{self.budget.logical_invocation_hard_ceiling} —— 立即 ABORT"
            )

    def reserve(self, *, invocations: int = 1) -> None:
        """**Tier 2:每次逻辑调用之前的硬检查。**

        这是**唯一**写 `logical_llm_invocations` 的地方。
        越界即抛,且**在抛出前不改变任何计数器** ——
        "先记账再拒绝"会让上界自己把自己撑破。
        """
        if invocations < 1:
            raise ValueError(f"invocations 至少为 1,收到 {invocations!r}")
        projected = self.experiment_logical_llm_invocations + invocations
        if projected > self.budget.logical_invocation_hard_ceiling:
            raise BudgetExceeded(
                f"逻辑 LLM 调用被拒:已消费 {self.experiment_logical_llm_invocations}"
                f"(继承 {self.inherited.logical_llm_invocations} + 本次进程 "
                f"{self.counters.logical_llm_invocations}),再调用 {invocations} 次将超过"
                f"硬上界 {self.budget.logical_invocation_hard_ceiling} —— "
                "调用**未发生**,立即 ABORT"
            )
        self.counters.logical_llm_invocations += invocations

    def record_unit(self, *, provider_http_attempts: int | None = None) -> None:
        """登记**一个已结束的实验单元**。

        `experimental_run_attempts` 每次 +1 且只 +1 —— 因为
        `HARNESS_LEVEL_RETRY = 0`,单元不会被执行第二次。

        **刻意不接受 `logical_invocations` 入参**:那个量由 `reserve()`
        逐次累计。留一个"顺便把逻辑调用也加一遍"的入口,迟早会有人用它,
        然后账目翻倍而两份数字都自洽。
        """
        self.counters.experimental_runs += 1
        self.counters.experimental_run_attempts += 1
        if provider_http_attempts is None:
            self.counters.provider_http_attempts_unobservable += 1
        else:
            self.counters.provider_http_attempts_observed += provider_http_attempts

    # ---- 事后断言 ----

    def check(self) -> None:
        """超限即抛。**每次登记后都应调用** —— 它是兜底,不是主防线。"""
        if (
            self.experiment_logical_llm_invocations
            > self.budget.logical_invocation_hard_ceiling
        ):
            raise BudgetExceeded(
                f"逻辑 LLM 调用 {self.experiment_logical_llm_invocations} 超过硬上界 "
                f"{self.budget.logical_invocation_hard_ceiling} —— 立即 ABORT"
            )
        observed = self.counters.provider_http_attempts_observed
        if observed > self.budget.provider_http_attempt_ceiling:
            raise BudgetExceeded(
                f"观测到的物理 HTTP 尝试 {observed} 超过上界 "
                f"{self.budget.provider_http_attempt_ceiling} —— 立即 ABORT"
            )
        if (
            self.experiment_logical_llm_invocations > self.budget.soft_call_ceiling
            and not self.warnings
        ):
            self.warnings.append(
                f"逻辑 LLM 调用 {self.experiment_logical_llm_invocations} 已超过软上限 "
                f"{self.budget.soft_call_ceiling}(继续,但需在报告中说明)"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.counters.snapshot(),
            "inherited_experimental_runs": self.inherited.experimental_runs,
            "inherited_experimental_run_attempts": (
                self.inherited.experimental_run_attempts
            ),
            "inherited_logical_llm_invocations": self.inherited.logical_llm_invocations,
            "experiment_experimental_runs": self.experiment_experimental_runs,
            "experiment_logical_llm_invocations": self.experiment_logical_llm_invocations,
            "experiment_provider_http_attempts": self.experiment_provider_http_attempts,
            "ceilings": self.budget.as_manifest_fields(),
            "soft_call_ceiling": self.budget.soft_call_ceiling,
            "harness_level_retry": 0,
            "warnings": list(self.warnings),
        }
