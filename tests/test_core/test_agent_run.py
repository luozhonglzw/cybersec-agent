"""`SecurityAgent.run()` 接缝的行为,以及 `chat()` 的兼容性。

S2-C 把"跑图 + 提取回答"抽成 `run()`,让 `/ask` 能拿到本次运行的
`ToolMessage`(引用校验的唯一可信证据来源)。这次重构的风险全在**兼容性**上:
`chat()` 是对外契约,签名、返回值、异常、日志事件都不能变。

因此这里同时钉住两件事:

1. `run()` 确实暴露了消息流(否则 `/ask` 无法校验引用);
2. `chat()` 与 `run().answer` **逐字节相同**,且异常语义不变。

最有力的兼容性证据其实是"既有测试一行未改仍然全绿"(`tests/test_core/test_agent.py`、
`tests/test_api/test_chat.py`、`tests/test_core/test_react_agent.py`),本模块
只补那些既有测试覆盖不到的接缝细节。
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.agent import (
    AgentRun,
    SECURITY_ANALYST_SYSTEM_PROMPT,
    SecurityAgent,
)
from app.core.llm import FakeLLMClient, LLMInvocationError
from app.core.rag_ask import RAG_ASK_SYSTEM_PROMPT

ECHO_TOOL_NAME = "echo_tool"
ECHO_RESULT = "echo:ok"


class _EchoTool:
    """一个只读、无副作用的假工具:只用来产生一条 `ToolMessage`。"""

    name = ECHO_TOOL_NAME

    async def ainvoke(self, args=None) -> str:  # noqa: D401 - 测试替身
        return ECHO_RESULT


class _OneToolLLM:
    """第一轮发起一次工具调用,之后给出最终回答。"""

    def __init__(self, answer: str = "最终回答") -> None:
        self.answer = answer
        self.bound_tools: list = []
        self._called = False

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        if not self._called:
            self._called = True
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": ECHO_TOOL_NAME, "args": {"text": "x"}, "id": "call-1"}
                ],
            )
        return AIMessage(content=self.answer)


@pytest.mark.asyncio
async def test_chat_returns_exactly_the_run_answer():
    """`chat()` 的返回值就是 `run().answer` —— 没有二次加工。"""
    fake = FakeLLMClient(reply="这是一条模拟回复")
    agent = SecurityAgent(fake)

    run = await agent.run("帮我看看日志")
    chat_answer = await agent.chat("帮我看看日志")

    assert run.answer == "这是一条模拟回复"
    assert chat_answer == run.answer


@pytest.mark.asyncio
async def test_chat_still_sends_the_default_system_prompt():
    """`chat()` 不传 `system_prompt`,因此仍用既有提示词(行为不变)。"""
    fake = FakeLLMClient()
    agent = SecurityAgent(fake)
    await agent.chat("用户问题")

    assert isinstance(fake.last_messages[0], SystemMessage)
    assert fake.last_messages[0].content == SECURITY_ANALYST_SYSTEM_PROMPT
    assert isinstance(fake.last_messages[1], HumanMessage)
    assert fake.last_messages[1].content == "用户问题"


@pytest.mark.asyncio
async def test_run_can_use_a_different_system_prompt_without_touching_the_graph():
    """`/ask` 换提示词只是换消息数组第一条,图与工具集不变。"""
    fake = FakeLLMClient()
    agent = SecurityAgent(fake)
    await agent.run("问题", system_prompt=RAG_ASK_SYSTEM_PROMPT)

    assert fake.last_messages[0].content == RAG_ASK_SYSTEM_PROMPT
    # 换提示词不影响工具绑定:图在构造期就已经绑好了
    assert [tool.name for tool in fake.bound_tools] == [
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
        "plan_response_tool",
    ]


@pytest.mark.asyncio
async def test_run_exposes_tool_messages_while_chat_does_not():
    """这正是加 `run()` 的理由:`/ask` 需要 `ToolMessage`,而 `chat()` 丢掉它。"""
    llm = _OneToolLLM(answer="依据知识库……")
    agent = SecurityAgent(llm, tools=[_EchoTool()])

    run = await agent.run("问题")

    tool_messages = [m for m in run.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == ECHO_RESULT
    assert tool_messages[0].type == "tool"

    chat_answer = await agent.chat("问题")
    assert chat_answer == run.answer == "依据知识库……"


@pytest.mark.asyncio
async def test_run_reports_the_iteration_count():
    """`iterations` 只为保持既有 `agent_chat_completed` 日志字段而暴露。"""
    fake = FakeLLMClient()
    run = await SecurityAgent(fake).run("问题")
    assert run.iterations == 1


@pytest.mark.asyncio
async def test_run_does_not_expose_graph_state():
    """只暴露 `answer` / `messages` / `iterations` —— 不是整个 AgentState。

    `policy_decision` / `approval_request` 属于 HITL 路径,`/ask` 的图根本不
    产生它们;把 state 整体交出去会让"某天顺手读一个安全字段"变成可预见的
    演化方向。
    """
    fake = FakeLLMClient()
    run = await SecurityAgent(fake).run("问题")

    assert {field for field in AgentRun.__dataclass_fields__} == {
        "answer",
        "messages",
        "iterations",
    }
    assert not hasattr(run, "policy_decision")
    assert not hasattr(run, "approval_request")


@pytest.mark.asyncio
async def test_run_propagates_llm_errors_unchanged():
    """异常语义不变:`LLMClientError` 原样上抛,不被包装。"""
    agent = SecurityAgent(FakeLLMClient(raise_error=True))

    with pytest.raises(LLMInvocationError):
        await agent.run("问题")
    with pytest.raises(LLMInvocationError):
        await agent.chat("问题")


@pytest.mark.asyncio
async def test_chat_keeps_the_max_iterations_reply():
    """达到迭代上限时的既有对外行为不变(返回固定说明,不是空串)。"""

    class _AlwaysCallsTool:
        def __init__(self) -> None:
            self._n = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self._n += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": ECHO_TOOL_NAME, "args": {}, "id": f"call-{self._n}"}
                ],
            )

    agent = SecurityAgent(_AlwaysCallsTool(), tools=[_EchoTool()], max_iterations=2)
    answer = await agent.chat("问题")

    assert answer == "当前分析达到最大工具调用次数,无法在限定步骤内完成分析。"


def test_agent_run_is_immutable():
    """结果是不可变值对象:没有"改一个字段就换个结论"的入口。"""
    run = AgentRun(answer="a", messages=(), iterations=0)
    with pytest.raises(Exception):
        run.answer = "b"  # type: ignore[misc]
