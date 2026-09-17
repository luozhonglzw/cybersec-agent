"""Phase 4 Step 4.2:手写 LangGraph ReAct graph 的离线测试。

复用 app.core.llm 的 FakeChatModel / FakeLLMClient,不调用真实 LLM API。
覆盖:graph 构建、节点存在、END 路由、单次/多次工具循环、
tool_call_id 传递、tool args 正确传参、异常安全契约、
max_iterations 业务层终止、消息顺序。
"""
import json
from typing import get_args, get_type_hints

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.core.graph import AgentState, create_agent_graph
from app.core.llm import FakeChatModel, FakeLLMClient


def _fake_llm(responses: list[str], tool_results: list[dict] | None = None) -> FakeLLMClient:
    """组装 graph 测试用的 FakeLLMClient(内部挂 FakeChatModel)。"""
    llm = FakeLLMClient("unused")
    llm._model = FakeChatModel(responses, tool_results or [])
    return llm


def _input(message: str = "查询日志") -> dict:
    return {"messages": [HumanMessage(content=message)], "iteration_count": 0}


# ---------- 构建与结构 ----------

def test_graph_builds_and_contains_nodes():
    """graph 可以构建并包含 agent / tools 节点。"""
    graph = create_agent_graph(_fake_llm(["done"]))
    node_names = set(graph.get_graph().nodes.keys())
    assert {"agent", "tools"} <= node_names


def test_agent_state_schema():
    """AgentState 的字段集是**精确**约定的:多一个少一个都算契约变更。

    Phase 8.3 有意扩展了 5 个 HITL 字段(见 app/core/graph.py 的 AgentState
    docstring)。这里保留精确断言而不是放宽成 `<=`,是为了让任何未来的
    字段增删都必须**有意**改这一行,而不是悄悄通过。
    """
    assert set(AgentState.__annotations__.keys()) == {
        # 基础字段(Phase 4,勿删)
        "messages",
        "iteration_count",
        # HITL 安全层(Phase 8.3)
        "indicator",
        "plan",
        "policy_decision",
        "approval_request",
        "approval_decision",
    }


def test_agent_state_keeps_base_fields():
    """messages 仍是 add_messages reducer,iteration_count 仍是普通字段。

    扩展 State 时最容易踩的坑就是把 messages 的 reducer 丢掉 ——
    那样消息会退化成 last-write-wins,整个 ReAct 循环失效。
    """
    from langgraph.graph.message import add_messages

    hints = get_type_hints(AgentState, include_extras=True)
    assert get_args(hints["messages"])[1] is add_messages
    assert hints["iteration_count"] is int


# ---------- END 路由 ----------

@pytest.mark.asyncio
async def test_no_tool_calls_reaches_end():
    """无 tool call → agent → END,消息顺序 System/Human 不受影响。"""
    graph = create_agent_graph(_fake_llm(["分析完成,无需查询"]))
    result = await graph.ainvoke(_input())

    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "分析完成,无需查询"
    assert result["iteration_count"] == 1


@pytest.mark.asyncio
async def test_one_tool_call_full_cycle():
    """一次工具调用:agent → tools → agent → END。"""
    tool_result = {
        "args": {"limit": 5},
        "result": json.dumps({"count": 5, "events": [{"event_type": "login_success"}] * 5}),
    }
    graph = create_agent_graph(_fake_llm(
        ["我需要查询日志", "查询到 5 条登录成功事件"], [tool_result]
    ))
    result = await graph.ainvoke(_input())

    msgs = result["messages"]
    # 顺序:Human → AI(tool_calls) → Tool → AI(final)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls
    assert isinstance(msgs[2], ToolMessage)
    assert isinstance(msgs[3], AIMessage) and not msgs[3].tool_calls
    assert result["iteration_count"] == 2


@pytest.mark.asyncio
async def test_multiple_tool_calls_loop():
    """多次工具调用可以在 agent ⇄ tools 间循环。"""
    tool_results = [
        {"args": {"event_type": "login_failed"}, "result": json.dumps({"count": 3, "events": []})},
        {"args": {"source_ip": "203.0.113.66"}, "result": json.dumps({"count": 2, "events": []})},
    ]
    graph = create_agent_graph(_fake_llm(
        ["查询失败登录", "查询特定 IP", "分析完成:3 条失败,2 条来自可疑 IP"],
        tool_results,
    ))
    result = await graph.ainvoke(_input())

    msgs = result["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert result["iteration_count"] == 3
    assert msgs[-1].content.startswith("分析完成")


# ---------- tool_call_id 与 args ----------

def _make_recording_tool(received: list):
    """创建记录收到的 args 的测试工具(BaseTool 是 pydantic 模型,不能随意挂属性)。"""
    from langchain_core.tools import tool

    @tool
    async def query_security_logs_tool(limit: int = 1) -> str:
        """查询安全日志(测试用)。"""
        received.append({"limit": limit})
        return json.dumps({"echo_limit": limit})

    return query_security_logs_tool


@pytest.mark.asyncio
async def test_tool_call_id_propagated_to_tool_message():
    """tool_call_id 从 AIMessage 严格传递到 ToolMessage。"""
    received: list = []
    llm = _fake_llm(["step1", "final"], [{"args": {"limit": 3}, "result": "{}"}])
    graph = create_agent_graph(llm, tools=[_make_recording_tool(received)])
    result = await graph.ainvoke(_input())

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].tool_call_id == ai_msgs[0].tool_calls[0]["id"]


@pytest.mark.asyncio
async def test_tool_receives_args_not_tool_call():
    """工具收到的是 tc["args"],不是整个 tool_call dict。"""
    received: list = []
    llm = _fake_llm(["step1", "final"], [{"args": {"limit": 7}, "result": "{}"}])
    graph = create_agent_graph(llm, tools=[_make_recording_tool(received)])
    await graph.ainvoke(_input())

    assert received == [{"limit": 7}]


# ---------- 异常安全契约 ----------

def _make_boom_tool(name: str = "query_security_logs_tool"):
    """创建总是抛错(含敏感字样)的测试工具。"""
    from langchain_core.tools import StructuredTool

    async def boom_tool(limit: int = 1) -> str:
        raise RuntimeError("D:/secret/path api_key=sk-xxx traceback")

    return StructuredTool.from_function(
        coroutine=boom_tool,
        name=name,
        description="总是抛错的测试工具",
    )


@pytest.mark.asyncio
async def test_tool_exception_returns_safe_tool_message():
    """工具异常 → ToolMessage 只含安全契约字段,无 traceback/路径/敏感参数。"""
    llm = _fake_llm(["step1", "final"], [{"args": {"limit": 1}, "result": "{}"}])
    graph = create_agent_graph(llm, tools=[_make_boom_tool()])
    result = await graph.ainvoke(_input())

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    parsed = json.loads(tool_msg.content)
    assert parsed["error"] == "工具执行失败"
    assert "secret" not in tool_msg.content
    assert "api_key" not in tool_msg.content
    # 循环继续:拿到错误结果后 LLM 仍能给出最终回答
    assert isinstance(result["messages"][-1], AIMessage)


@pytest.mark.asyncio
async def test_unknown_tool_returns_suggest_retry():
    """未知工具 → suggest_retry 的 JSON ToolMessage,不抛异常。"""
    llm = _fake_llm(["step1", "final"], [{"args": {"limit": 1}, "result": "{}"}])
    graph = create_agent_graph(llm, tools=[_make_boom_tool(name="some_other_tool")])
    result = await graph.ainvoke(_input())

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    parsed = json.loads(tool_msg.content)
    assert parsed["error"] == "未知工具"
    assert parsed["suggest_retry"] is True


# ---------- max_iterations ----------

@pytest.mark.asyncio
async def test_max_iterations_terminates_loop():
    """达到业务层 max_iterations 时终止,即使 LLM 仍想调用工具。"""
    tool_results = [{"args": {"limit": 1}, "result": json.dumps({"count": 1, "events": []})}] * 5
    llm = _fake_llm(["查询1", "查询2", "查询3", "查询4"], tool_results)
    graph = create_agent_graph(llm, max_iterations=2)
    result = await graph.ainvoke(_input())

    assert result["iteration_count"] == 2
    # 2 轮循环:2 条 AI(tool_calls) + 2 条 ToolMessage
    ai_with_calls = [m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert len(ai_with_calls) == 2


# ---------- 消息顺序 ----------

@pytest.mark.asyncio
async def test_message_order_preserved():
    """完整一轮的消息顺序:Human → AI(tool_calls) → Tool → AI(final)。"""
    tool_result = {"args": {"limit": 2}, "result": json.dumps({"count": 2, "events": []})}
    graph = create_agent_graph(_fake_llm(["查询", "完成"], [tool_result]))
    result = await graph.ainvoke(_input())

    msgs = result["messages"]
    types = [type(m) for m in msgs]
    assert types == [HumanMessage, AIMessage, ToolMessage, AIMessage]


# ---------- 执行轨迹(astream updates) ----------

class ScriptedTraceModel:
    """轨迹测试专用的脚本化模型:精确控制 tool_call_id 与响应内容。

    与 FakeChatModel 的区别:tool_call id 固定为 "call_trace_001",
    便于断言 id 从 AIMessage → tools node → ToolMessage 的传递。
    """

    def __init__(self):
        self.call_count = 0

    async def ainvoke(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "query_security_logs_tool",
                             "args": {"limit": 7}, "id": "call_trace_001"}],
            )
        return AIMessage(content="发现高危登录事件。")

    def bind_tools(self, tools):
        return self


def _make_trace_setup():
    """组装 trace 测试:脚本化模型 + 记录型工具。"""
    from langchain_core.tools import tool

    received: list = []

    @tool
    async def query_security_logs_tool(limit: int = 1) -> str:
        """记录收到的 args 的测试工具。"""
        received.append({"limit": limit})
        return json.dumps({"count": 0, "events": []})

    llm = FakeLLMClient("unused")
    llm._model = ScriptedTraceModel()
    graph = create_agent_graph(llm, tools=[query_security_logs_tool])
    return graph, received


@pytest.mark.asyncio
async def test_execution_trace_agent_tools_agent_end():
    """通过 astream(stream_mode='updates') 观察完整执行轨迹。

    每个 update 是节点增量:{'agent': {...}} / {'tools': {...}},
    不是完整 state —— 按 State Update 语义逐条断言。
    """
    graph, received = _make_trace_setup()
    initial = {
        "messages": [HumanMessage(content="查询最近的高危登录日志")],
        "iteration_count": 0,
    }

    updates = [u async for u in graph.astream(initial, stream_mode="updates")]
    node_order = [next(iter(u)) for u in updates]

    # 1. node 执行顺序:agent → tools → agent
    assert node_order == ["agent", "tools", "agent"]

    # 2. 第一次 agent update:AIMessage(tool_calls),iteration_count = 1
    first = updates[0]["agent"]
    first_ai = first["messages"][0]
    assert isinstance(first_ai, AIMessage) and first_ai.tool_calls
    assert first["iteration_count"] == 1

    # 3. tools update:ToolMessage,tool_call_id 传递,iteration_count 不自增
    second = updates[1]["tools"]
    tool_msg = second["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_trace_001"
    assert "iteration_count" not in second  # tools node 不修改迭代计数
    assert received == [{"limit": 7}]       # 工具实际收到 args

    # 4. 第二次 agent update:AIMessage(final text),iteration_count = 2
    third = updates[2]["agent"]
    final_ai = third["messages"][0]
    assert isinstance(final_ai, AIMessage) and not final_ai.tool_calls
    assert third["iteration_count"] == 2

    # 6. 最终 AIMessage 内容与无 tool_calls
    assert final_ai.content == "发现高危登录事件。"


@pytest.mark.asyncio
async def test_execution_trace_final_state():
    """最终完整 state(经 ainvoke 单独获取):消息顺序 + iteration_count。"""
    graph, _ = _make_trace_setup()
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="查询最近的高危登录日志")],
        "iteration_count": 0,
    })

    msgs = result["messages"]
    # 5. 顺序:graph 视角为 Human → AI(tool_calls) → Tool → AI(final)
    #    (SystemMessage 由 SecurityAgent.chat() 在入口注入,graph 内不含)
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert msgs[1].tool_calls[0]["id"] == "call_trace_001"
    assert msgs[2].tool_call_id == "call_trace_001"
    assert result["iteration_count"] == 2
    # 6. final AIMessage
    assert msgs[-1].content == "发现高危登录事件。"
    assert not msgs[-1].tool_calls
