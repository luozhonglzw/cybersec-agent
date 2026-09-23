"""逐逻辑调用的预算执行边界(**provider 无关,零 provider 依赖**)。

为什么是**代理**而不是在图上挂钩子
--------------------------------
`create_agent_graph(llm_client)` 只要求对象提供 `bind_tools` 与 `ainvoke`
两个方法(见 `app/core/graph.py`)。因此包一层代理就能在
**不碰生产图、不碰适配器内部、不碰任何 provider 客户端**的前提下,
对每一次逻辑调用做"先检查、后放行"。

为什么 `bind_tools` 之后还要再包一层
----------------------------------
图走的是 `llm_client.bind_tools(tools)` → `bound_model.ainvoke(...)`。
若只包最外层对象、却让 `bind_tools` 原样透传,那么**图上的每一次调用都会绕过计数**
—— 而报告里的调用数看起来完全正常。这是本模块最容易写错的地方,
由 `test_d2b_budget.py` 的"绑定后仍计数"测试守住。

计数的事实来源
-------------
`invocations` 由**代理自己**数,不依赖某个假 LLM 的 `call_count`。
真实 provider 没有 `call_count`,而"调用了几次"必须对两类实现都成立。
"""
from typing import Any

from app.evaluation.llm.budget import BudgetGovernor


class _BudgetedBound:
    """`bind_tools` 之后的代理。

    必须独立包一层 —— 否则工具绑定后的 `ainvoke` 会绕过计数。
    """

    __slots__ = ("_inner", "_owner")

    def __init__(self, inner: Any, owner: "BudgetedLLM") -> None:
        self._inner = inner
        self._owner = owner

    async def ainvoke(self, messages: Any) -> Any:
        # 顺序是硬要求:先检查/记账(可能抛),**再**真正调用。
        self._owner._reserve_and_count()
        return await self._inner.ainvoke(messages)


class BudgetedLLM:
    """给任意 `bind_tools` / `ainvoke` 实现套上预算边界。

    `governor=None` 时**只计数、不拒绝** —— 用于离线单测里单独验证
    "调用次数"这一事实,而无需构造完整预算。
    """

    def __init__(self, inner: Any, governor: BudgetGovernor | None = None) -> None:
        self._inner = inner
        self._governor = governor
        self._invocations = 0

    @property
    def invocations(self) -> int:
        """本对象上实际发生的**逻辑调用**次数(provider 无关的事实来源)。"""
        return self._invocations

    @property
    def inner(self) -> Any:
        """被包裹的原始对象(只读访问;不提供 setter)。"""
        return self._inner

    def _reserve_and_count(self) -> None:
        """先检查(可能抛 `BudgetExceeded`),再计数。

        被拒绝时**不计数** —— 那次调用没有发生。
        """
        if self._governor is not None:
            self._governor.reserve()
        self._invocations += 1

    def bind_tools(self, tools: Any) -> "_BudgetedBound":
        return _BudgetedBound(self._inner.bind_tools(tools), self)

    async def ainvoke(self, messages: Any) -> Any:
        self._reserve_and_count()
        return await self._inner.ainvoke(messages)


def budgeted(llm: Any, governor: BudgetGovernor | None) -> BudgetedLLM:
    """把已包裹的对象**原样返回**,避免重复包裹导致的重复计数。"""
    if isinstance(llm, BudgetedLLM):
        return llm
    return BudgetedLLM(llm, governor)
