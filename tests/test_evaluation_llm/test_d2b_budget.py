"""Phase 9.2-D-2b —— 预算执行边界(Tier 1 单元准入 + Tier 2 每次调用前硬检查)。

本文件对应 F-4 的修复验证。F-4 的形态不是"算错了",而是
**"治理器存在但没有任何人调用它"** —— 篡改硬上界为 0 之后,完整离线运行
照常跑完 18 个单元,而报告里的调用数看起来完全正常。

因此本文件的核心不是"预算数字对不对",而是三条**可证伪**的性质:

    Tier 2 在**调用之前**拒绝    被拒的那次调用**没有发生**(内层计数不变)
    上界是**实验级**的            继承的已消费量参与判定,且不算作本次进程的调用
    拒绝有**唯一分类**            `BUDGET_EXHAUSTED` / ABORT,绝不落进 `HARNESS_ERROR`

每条都配了反同义反复断言:如果守卫被摘掉,断言必须变红。
"""
import ast
import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolCall

from app.evaluation.llm.budget import (
    UNKNOWN,
    BudgetCounters,
    BudgetExceeded,
    BudgetGovernor,
    InheritedConsumption,
    budget_from_plan,
    pilot_budget,
)
from app.evaluation.llm.budgeted_llm import BudgetedLLM, budgeted
from app.evaluation.llm.dataset import LLM_TASKS
from app.evaluation.llm.executor import (
    BudgetAbort,
    HarnessAbort,
    OfflineExecutor,
    budget_abort_failure,
)
from app.evaluation.llm.failures import (
    FAILURE_TAXONOMY,
    HARNESS_LEVEL_RETRY,
    FailureClass,
)
from app.evaluation.llm.identity import EndpointCategory, provider_identity
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.pilot import pilot_plan
from app.evaluation.llm.raw import RawWriter, RecordStatus, index_by_unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_BASE_URL = "https://fake-provider.example.invalid/v1"

#: 本模块每个测试都在一个非严格出口守卫下运行,结束时断言**零出口**。
#: 与执行器内部的 `strict=True` 守卫叠加:内层在越界处炸掉,外层在收尾处兜底。
@pytest.fixture(autouse=True)
def _no_egress():
    guard = NetworkEgressGuard(strict=False)
    with guard:
        yield
    assert guard.clean, f"D-2b 预算测试期间发生网络出口:{guard.egress_events}"


# ---------------------------------------------------------------------------
# 夹具与假模型
# ---------------------------------------------------------------------------


def fake_identity():
    """注入模型用的**真实形态**身份(非 scripted)。"""
    return provider_identity(
        provider="fake-openai",
        model="fake-model-1",
        endpoint_category=EndpointCategory.OPENAI_COMPATIBLE,
        base_url=FAKE_BASE_URL,
    )


class CountingInner:
    """只记录调用次数的假模型 —— 用来证明"拒绝发生在调用之前"。

    它**没有** `call_count` 这个名字,刻意的:计数的事实来源必须是代理本身
    (`BudgetedLLM.invocations`),不能依赖某个假实现的私有属性。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.tools_bound = 0

    def bind_tools(self, tools):
        self.tools_bound += 1
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="ok")


class ToolLoopingFake:
    """每次都请求工具调用 —— 用来逼出图的**多次迭代**。

    没有预算时这张图会一直转到 `max_iterations`;有预算时它必须在
    第 (上界 + 1) 次调用**之前**被拦住。
    """

    def __init__(self, *, task, dataset_paths=None, **_):
        self.task = task
        self.dataset_paths = dataset_paths or {}
        self.calls = 0
        self._bound = False

    def bind_tools(self, tools):
        self._bound = True
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        args = {"source_ip": self.task.indicator}
        path = self.dataset_paths.get("logs")
        if path:
            args["data_path"] = path
        return AIMessage(
            content="",
            tool_calls=[ToolCall(
                name="query_security_logs_tool", args=args, id=f"loop-{self.calls}"
            )],
        )


class PlainFake:
    """一次就给出最终答案的假模型(供 B0 与图基线终止)。"""

    def __init__(self, *, task, dataset_paths=None, **_):
        self.task = task
        self.dataset_paths = dataset_paths or {}
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="(注入假模型)无法核实,不下结论。")


def _plan_with_ceiling(ceiling: int, *, baselines=("B0-shared",), repetition_count: int = 1):
    """把一个**已声明**的硬上界写进计划。

    `model_copy` 而不是重新推导 —— 这正是 `budget_from_plan` 存在的理由:
    治理器必须服从调用方声明的上界,而不是自己重算一个"看起来对"的值。
    """
    return pilot_plan(baselines=baselines, repetition_count=repetition_count).model_copy(
        update={"logical_invocation_hard_ceiling": ceiling}
    )


def _governor(ceiling: int) -> BudgetGovernor:
    return BudgetGovernor(budget=budget_from_plan(_plan_with_ceiling(ceiling)))


def _reduced_executor(workdir, experiment_id: str, **overrides):
    """缩小规模的执行器:两个基线 × 1 重复 × 单个行为 = 18 个单元。"""
    baselines = ("B0-shared", "B2'")
    plan = pilot_plan(baselines=baselines, repetition_count=1)
    kwargs: dict = {
        "workdir": workdir,
        "experiment_id": experiment_id,
        "plan": plan,
        "baselines": baselines,
        "behaviors": ("GOOD",),
        "repetition_count": 1,
        "guard": NetworkEgressGuard(strict=True),
    }
    kwargs.update(overrides)
    return OfflineExecutor(**kwargs)


def _raw_path(executor: OfflineExecutor) -> Path:
    return Path(executor.workdir) / "raw" / f"{executor.experiment_id}.jsonl"


# ---------------------------------------------------------------------------
# 1. logical ceiling = 0 ⇒ 零次模型调用
# ---------------------------------------------------------------------------


def test_1_a_zero_logical_ceiling_admits_no_model_invocation(tmp_path):
    """上界为 0 时,**一次模型调用都不允许发生**。

    这正是 F-4 的原始形态:篡改上界为 0 之后运行照常跑完。修复后它必须
    在第一个单元**准入**阶段就被拒绝,而不是"跑完了再事后警告"。
    """
    fake = CountingInner()
    executor = OfflineExecutor(
        workdir=tmp_path / "zero-ceiling",
        experiment_id="zero-ceiling",
        plan=_plan_with_ceiling(0),
        baselines=("B0-shared",),
        behaviors=("GOOD",),
        repetition_count=1,
        llm_factory=lambda **_: fake,
        identity=fake_identity(),
        guard=NetworkEgressGuard(strict=True),
    )

    with pytest.raises(BudgetAbort):
        asyncio.run(executor.run())

    assert fake.calls == 0, "上界为 0 时仍然发生了模型调用"
    assert executor.governor.counters.logical_llm_invocations == 0
    assert executor.governor.counters.experimental_runs == 0


def test_1b_tier1_rejection_writes_no_record(tmp_path):
    """准入被拒的单元**一次调用都没有发生**,因此不该被写成 INCOMPLETE。

    给它写一条"跑了但坏了"的记录会把"我们主动没开始"伪装成"跑了但失败",
    进而污染覆盖率损失的口径。
    """
    executor = OfflineExecutor(
        workdir=tmp_path / "tier1",
        experiment_id="tier1",
        plan=_plan_with_ceiling(0),
        baselines=("B0-shared",),
        behaviors=("GOOD",),
        repetition_count=1,
        llm_factory=lambda **_: CountingInner(),
        identity=fake_identity(),
        guard=NetworkEgressGuard(strict=True),
    )
    with pytest.raises(BudgetAbort):
        asyncio.run(executor.run())
    assert not _raw_path(executor).exists()


# ---------------------------------------------------------------------------
# 2–3. 精确额度与 off-by-one
# ---------------------------------------------------------------------------


def test_2_exact_remaining_budget_allows_exactly_that_many_invocations():
    """剩余额度恰好为 N ⇒ 恰好允许 N 次,第 N+1 次被拒。**没有 off-by-one。**"""
    governor = _governor(3)
    for _ in range(3):
        governor.reserve()
    assert governor.experiment_logical_llm_invocations == 3

    with pytest.raises(BudgetExceeded):
        governor.reserve()
    # 拒绝时**不记账** —— "先记账再拒绝"会让上界自己把自己撑破。
    assert governor.experiment_logical_llm_invocations == 3


def test_3_ceiling_plus_one_is_rejected_before_the_invocation():
    """**顺序**断言:检查必须发生在真正调用之前。

    若顺序反过来(先调用、后记账),被拒的那次调用**已经发出去了** ——
    真实 provider 下这意味着真实额度已经被消耗,而报告说"没有超支"。
    """
    inner = CountingInner()
    llm = budgeted(inner, _governor(1))

    asyncio.run(llm.ainvoke([]))
    assert inner.calls == 1

    with pytest.raises(BudgetExceeded):
        asyncio.run(llm.ainvoke([]))

    assert inner.calls == 1, "被拒的调用仍然穿透到了内层模型"
    assert llm.invocations == 1, "被拒的调用被计入了调用数"


def test_3b_tool_bound_calls_still_pass_through_the_budget():
    """`bind_tools()` 之后**仍然**受预算约束。

    这是本模块最容易写错的地方:图走的是
    `llm.bind_tools(tools)` → `bound.ainvoke(...)`。若 `bind_tools` 原样透传,
    图上每一次调用都会**绕过计数** —— 而报告里的调用数看起来完全正常。
    """
    inner = CountingInner()
    llm = budgeted(inner, _governor(1))
    bound = llm.bind_tools([])

    assert bound is not llm, "bind_tools 必须返回一个独立包装的代理"
    asyncio.run(bound.ainvoke([]))
    assert llm.invocations == 1
    assert inner.calls == 1

    with pytest.raises(BudgetExceeded):
        asyncio.run(bound.ainvoke([]))
    assert inner.calls == 1
    assert llm.invocations == 1


def test_3c_budgeted_does_not_double_wrap():
    """重复包裹会让每次调用被计两次 —— 而两份数字都自洽。"""
    inner = CountingInner()
    llm = budgeted(inner, _governor(10))
    assert budgeted(llm, _governor(10)) is llm


# ---------------------------------------------------------------------------
# 4. 多轮图运行不得越过上界
# ---------------------------------------------------------------------------


def test_4_a_multi_turn_run_cannot_cross_the_logical_ceiling(tmp_path):
    """图会迭代多次 —— 上界必须在**迭代途中**生效,而不是只在单元边界生效。

    只做 Tier 1 的实现在这里会失败:准入时还剩额度,于是整个图跑满
    `max_iterations`,上界被越过 5 倍。
    """
    plan = _plan_with_ceiling(3, baselines=("B2'",))
    created: list[ToolLoopingFake] = []

    def factory(**kwargs):
        fake = ToolLoopingFake(**kwargs)
        created.append(fake)
        return fake

    executor = OfflineExecutor(
        workdir=tmp_path / "graph-ceiling",
        experiment_id="graph-ceiling",
        plan=plan,
        baselines=("B2'",),
        behaviors=("GOOD",),
        repetition_count=1,
        llm_factory=factory,
        identity=fake_identity(),
        guard=NetworkEgressGuard(strict=True),
    )

    with pytest.raises(BudgetAbort):
        asyncio.run(executor.run())

    assert executor.governor.counters.logical_llm_invocations == 3
    assert sum(fake.calls for fake in created) == 3, "越过上界的调用被放行了"

    records = RawWriter(_raw_path(executor), experiment_id="graph-ceiling").read_all()
    assert len(records) == 1
    assert records[0].record_status is RecordStatus.INCOMPLETE
    assert records[0].failure["failure_class"] == "BUDGET_EXHAUSTED"
    assert records[0].logical_llm_invocations == 3, (
        "中止单元必须如实记下**实际已发生**的调用数,而不是 0"
    )


# ---------------------------------------------------------------------------
# 5–6. Resume 预算语义(实验级 vs 本次进程)
# ---------------------------------------------------------------------------


def test_5_resume_inherits_logical_consumption(tmp_path):
    """续跑必须继承**已消费**的逻辑调用数。

    不继承等于重新发一份完整额度 —— 上界就不再是上界。
    """
    workdir = tmp_path / "inherit"
    first = asyncio.run(_reduced_executor(workdir, "inherit").run())
    records = RawWriter(Path(first.raw_path), experiment_id="inherit").read_all()
    records[0] = records[0].model_copy(update={"record_status": RecordStatus.INCOMPLETE})
    frozen = [r for r in records if r.record_status is RecordStatus.COMPLETE]
    expected_logical = sum(r.logical_llm_invocations for r in frozen)

    resumed = asyncio.run(
        _reduced_executor(workdir, "inherit").run(
            existing=index_by_unit(records, experiment_id="inherit"),
            origin_experiment_id="inherit",
        )
    )

    budget = resumed.budget
    assert budget["inherited_experimental_runs"] == len(frozen)
    assert budget["inherited_logical_llm_invocations"] == expected_logical
    assert budget["experiment_logical_llm_invocations"] == (
        expected_logical + budget["logical_llm_invocations"]
    )
    # 反同义反复:继承量必须**真的**非零,否则这条断言什么都证明不了。
    assert expected_logical > 0


def test_6_inherited_consumption_is_not_counted_as_a_new_process_invocation(tmp_path):
    """两个作用域**不得合并**。

    继承量并进"本次进程"会让报告里的样本量凭空变大 —— 那是 F-3 的同类失真。
    """
    workdir = tmp_path / "scope"
    first = asyncio.run(_reduced_executor(workdir, "scope").run())
    records = RawWriter(Path(first.raw_path), experiment_id="scope").read_all()
    inherited_logical = sum(r.logical_llm_invocations for r in records)

    resumed = asyncio.run(
        _reduced_executor(workdir, "scope").run(
            existing=index_by_unit(records, experiment_id="scope"),
            origin_experiment_id="scope",
        )
    )

    budget = resumed.budget
    assert resumed.executed_keys == []
    # 本次进程:什么都没跑
    assert budget["experimental_runs"] == 0
    assert budget["logical_llm_invocations"] == 0
    # 实验级:全部继承
    assert budget["inherited_experimental_runs"] == len(records)
    assert budget["inherited_logical_llm_invocations"] == inherited_logical
    assert budget["experiment_experimental_runs"] == len(records)
    assert budget["experiment_logical_llm_invocations"] == inherited_logical
    assert inherited_logical > 0


# ---------------------------------------------------------------------------
# 7–8. 不重跑
# ---------------------------------------------------------------------------


def test_7_a_bad_model_result_is_not_automatically_rerun(tmp_path):
    """**坏结果是结果。** 记录完整 ⇒ 永不重跑,无论它有多差。

    这里用 D-2a 已有的脚本化失败注入(`LLM_FATAL_FAILURE`):图基线在第 2 次
    调用抛错 ⇒ 该单元以 `llm_failed` 结束,但记录是 **complete**。
    """
    workdir = tmp_path / "bad-result"
    first = asyncio.run(
        _reduced_executor(workdir, "bad-result", behaviors=("LLM_FATAL_FAILURE",)).run()
    )
    records = RawWriter(Path(first.raw_path), experiment_id="bad-result").read_all()
    assert records
    assert all(r.record_status is RecordStatus.COMPLETE for r in records)
    bad = [r for r in records if r.failure["failure_class"] != "MODEL_ANSWER"]
    assert bad, "本用例必须真的产出一批**非正常终态**的观测,否则断言是空的"

    resumed = asyncio.run(
        _reduced_executor(workdir, "bad-result").run(
            existing=index_by_unit(records, experiment_id="bad-result"),
            origin_experiment_id="bad-result",
        )
    )
    assert resumed.executed_keys == []
    assert resumed.budget["inherited_experimental_runs"] == len(records)


def test_8_harness_retry_remains_zero(d2a_outcome):
    """harness 层重试恒为 **0** —— 这是冻结的设计决定,不是遗漏。"""
    assert HARNESS_LEVEL_RETRY == 0
    assert all(
        rule.experimental_retry_allowed is False
        for rule in FAILURE_TAXONOMY.values()
    )
    assert d2a_outcome.budget["harness_level_retry"] == 0
    records = RawWriter(d2a_outcome.raw_path, experiment_id="session-e2e").read_all()
    assert all(r.experimental_run_attempts == 1 for r in records)


# ---------------------------------------------------------------------------
# 9–10. 分类与中止
# ---------------------------------------------------------------------------


def test_9_budget_exceeded_classifies_as_budget_exhausted():
    """`BudgetExceeded` → `BUDGET_EXHAUSTED` / ABORT,**绝不** `HARNESS_ERROR`。"""
    rule = FAILURE_TAXONOMY[FailureClass.BUDGET_EXHAUSTED]
    assert rule.classification == "ABORT"
    assert rule.aborts_pilot is True
    assert rule.experimental_retry_allowed is False
    assert rule.count_as_model_result is False

    record = budget_abort_failure("BudgetExceeded")
    assert record.failure_class is FailureClass.BUDGET_EXHAUSTED
    assert record.classification == "ABORT"
    assert record.aborts_pilot is True
    assert record.experimental_retry_allowed is False
    assert record.failure_class is not FailureClass.HARNESS_ERROR


def test_9b_budget_exceeded_is_an_assertion_error_subclass():
    """分类纪律的**根因**:它是 `AssertionError` 子类。

    通用 `except Exception` 会先接住它 —— 因此 `except BudgetExceeded`
    必须排在前面。这条断言把那个"为什么必须小心"的事实固定下来。
    """
    assert issubclass(BudgetExceeded, AssertionError)
    assert issubclass(BudgetAbort, RuntimeError)
    assert not issubclass(BudgetAbort, HarnessAbort)


def test_9c_the_adapter_fallback_does_not_swallow_budget_rejection():
    """适配器里的兜底 `except Exception` **不得**吞掉预算拒绝。

    若它吞掉,该单元会变成 `llm_failed` + `HARNESS_ERROR` ——
    报告把"我们主动停下来了"说成"工装坏了",而下游处置完全不同。
    这条是静态检查(顺序必须正确);行为层面的检查见 `test_4` 与 `test_10`。
    """
    import inspect

    from app.evaluation.llm import adapters as adapters_module

    for name in ("B0DirectAdapter", "B2PrimeGraphAdapter", "B3FullAgentAdapter"):
        source = inspect.getsource(getattr(adapters_module, name).run)
        assert "except BudgetExceeded:" in source, f"{name} 缺少预算专用的 except 分支"
        assert source.index("except BudgetExceeded:") < source.index(
            "except Exception as exc:"
        ), f"{name} 的 except 顺序错误 —— 通用分支会先接住它"


def test_10_budget_exhaustion_aborts_the_pilot(tmp_path):
    """中止的是**实验**,不是"重试一次"。"""
    plan = _plan_with_ceiling(2, baselines=("B2'",))
    executor = OfflineExecutor(
        workdir=tmp_path / "abort",
        experiment_id="abort",
        plan=plan,
        baselines=("B2'",),
        behaviors=("GOOD",),
        repetition_count=1,
        llm_factory=lambda **kwargs: ToolLoopingFake(**kwargs),
        identity=fake_identity(),
        guard=NetworkEgressGuard(strict=True),
    )

    with pytest.raises(BudgetAbort) as excinfo:
        asyncio.run(executor.run())
    assert not isinstance(excinfo.value, HarnessAbort)

    records = RawWriter(_raw_path(executor), experiment_id="abort").read_all()
    assert records[0].failure["failure_class"] == "BUDGET_EXHAUSTED"
    assert records[0].failure["classification"] == "ABORT"
    assert records[0].failure["aborts_pilot"] is True
    assert records[0].failure["experimental_retry_allowed"] is False


# ---------------------------------------------------------------------------
# 反同义反复:把"记账分工"固定下来
# ---------------------------------------------------------------------------


def test_record_unit_never_touches_the_logical_counter():
    """`record_unit()` 只写单元级与物理尝试级。

    留一个"顺便把逻辑调用也加一遍"的入口,迟早会有人用它,然后账目翻倍
    而两份数字都自洽 —— 因此该入口**在签名上就不存在**。
    """
    governor = _governor(10)
    governor.reserve()
    governor.record_unit(provider_http_attempts=None)

    assert governor.counters.logical_llm_invocations == 1
    assert governor.counters.experimental_runs == 1
    assert governor.counters.experimental_run_attempts == 1
    assert governor.experiment_logical_llm_invocations == 1

    signature = BudgetGovernor.record_unit.__code__.co_varnames[
        : BudgetGovernor.record_unit.__code__.co_argcount
    ]
    assert "logical_invocations" not in signature


def test_budget_from_plan_does_not_recompute_the_ceiling():
    """`budget_from_plan` 必须**服从**已声明的上界。

    若它调用 `pilot_budget()` 重算,调用方显式写小的上界会被静默抵消 ——
    预算就退化成装饰品,而"上界被遵守"这件事永远无法被自检。
    """
    plan = _plan_with_ceiling(7)
    budget = budget_from_plan(plan)
    assert budget.logical_invocation_hard_ceiling == 7

    recomputed = pilot_budget(baseline_labels=("B0-shared",), repetition_count=1)
    assert recomputed.logical_invocation_hard_ceiling != 7, (
        "重算值与声明值恰好相同 —— 这条断言无法区分'服从'与'重算'"
    )


def test_check_is_a_backstop_that_still_fires_out_of_band():
    """`check()` 是兜底:即使有人绕过 `reserve()` 直接改计数器,它也必须炸。"""
    governor = _governor(1)
    governor.counters.logical_llm_invocations = 5
    with pytest.raises(BudgetExceeded):
        governor.check()


def test_seed_inherited_can_only_be_applied_once():
    """重复播种会让上界凭空变紧,而报告看起来完全正常。"""
    governor = _governor(10)
    governor.seed_inherited(InheritedConsumption(logical_llm_invocations=1))
    with pytest.raises(AssertionError):
        governor.seed_inherited(InheritedConsumption(logical_llm_invocations=2))


def test_seeding_inherited_does_not_touch_process_counters():
    """播种**不经过** `record_unit()` —— 否则继承的单元会被记成本次进程新跑的。"""
    governor = _governor(10)
    governor.seed_inherited(
        InheritedConsumption(experimental_runs=5, logical_llm_invocations=9)
    )
    assert governor.counters.experimental_runs == 0
    assert governor.counters.logical_llm_invocations == 0
    assert governor.experiment_experimental_runs == 5
    assert governor.experiment_logical_llm_invocations == 9


def test_inherited_consumption_rejects_negative_values():
    with pytest.raises(ValueError):
        InheritedConsumption(logical_llm_invocations=-1)


def test_admit_unit_rejects_a_zero_minimum():
    with pytest.raises(ValueError):
        _governor(10).admit_unit(min_invocations=0)


def test_the_adapter_seam_accepts_a_factory_and_a_governor():
    """注入缝的**形状**:适配器不再把模型写死,且始终返回预算代理。"""
    from app.evaluation.llm.adapters import BASELINES, ScriptedLLM

    inner = CountingInner()
    adapter = BASELINES["B0"](
        dataset_paths={}, llm_factory=lambda **_: inner, governor=_governor(5)
    )
    llm = adapter._budgeted_llm(
        behavior="GOOD",
        task=LLM_TASKS[0],
        dataset_paths=None,
        decoy_paths=None,
        emit_usage=False,
    )
    assert isinstance(llm, BudgetedLLM)
    assert llm.inner is inner

    offline = BASELINES["B0"](dataset_paths={})
    default_llm = offline._budgeted_llm(
        behavior="GOOD",
        task=LLM_TASKS[0],
        dataset_paths=None,
        decoy_paths=None,
        emit_usage=False,
    )
    assert isinstance(default_llm.inner, ScriptedLLM), "离线默认实现必须仍是 ScriptedLLM"


def test_the_budget_module_does_not_derive_counters_from_each_other():
    """四个计数器**不得互相推导** —— AST 级检查。

    一条"逻辑调用 = 单元数 × 5"之类的推导会让账目自洽但失真,
    而且它看起来比"如实累加"更"聪明"。

    两条性质:
        计数器只被 `+=` 累加(没有任何一处把它**赋值**成一个推导值)
        模块里不存在以计数器为操作数的乘法
    """
    source = (REPO_ROOT / "app" / "evaluation" / "llm" / "budget.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    augmented = {
        node.target.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute)
    }
    assert "logical_llm_invocations" in augmented, (
        "逻辑调用数必须只通过 `+=` 累加"
    )

    counter_names = {
        "counters",
        "inherited",
        "experimental_runs",
        "experimental_run_attempts",
        "logical_llm_invocations",
    }
    multiplied = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and any(
            isinstance(side, ast.Attribute) and side.attr in counter_names
            for side in (node.left, node.right)
        )
    ]
    assert multiplied == [], (
        f"预算模块第 {multiplied} 行出现了以计数器为操作数的乘法推导"
    )


def test_budget_counters_snapshot_keeps_the_four_scopes_separate():
    counters = BudgetCounters(
        experimental_runs=2,
        experimental_run_attempts=2,
        logical_llm_invocations=6,
        provider_http_attempts_observed=0,
        provider_http_attempts_unobservable=2,
    )
    snapshot = counters.snapshot()
    assert snapshot["experimental_runs"] == 2
    assert snapshot["experimental_run_attempts"] == 2
    assert snapshot["logical_llm_invocations"] == 6
    assert snapshot["provider_http_attempts"] == UNKNOWN
