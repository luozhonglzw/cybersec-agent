"""手写 LangGraph 版 ReAct 控制流(Phase 4 Step 4.2)。

Phase 3 的 SecurityAgent.chat() 用 while 循环表达了同样的控制流:
    LLM → 有 tool_calls? → 执行工具 → 回到 LLM / 结束

本模块用 LangGraph 的 State / Node / Edge 把它声明式地表达出来:

    START → agent → should_continue ─┬─ 无 tool_calls / 达到上限 → END
                                     └─ 有 tool_calls → tools → agent

与 Phase 3 的差异:
- 状态由 add_messages reducer 追加,节点返回增量而不是就地修改;
- 工具执行改为 tool.ainvoke(tc["args"]),修复了旧循环传整个
  tool_call dict 导致参数全部落到默认值的 bug;
- 业务层 max_iterations 与 LangGraph 的 recursion_limit 是两个概念:
  前者由 iteration_count + should_continue 控制,后者只是框架兜底。

本 Step 先独立存在并可测试,不接入 SecurityAgent.chat()。
"""
import json
from typing import Annotated, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.core.llm import LLMClient
from app.tools.query_logs import query_security_logs_tool

logger = structlog.get_logger(__name__)


class AgentState(TypedDict, total=False):
    """Graph 的全部状态,不引入业务之外的额外字段。

    messages: 消息历史,add_messages reducer 负责按 id 去重追加
    iteration_count: agent 节点的调用次数,用于业务层 max_iterations 判断
    """
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int


def create_agent_graph(
    llm_client: LLMClient,
    tools: list[BaseTool] | None = None,
    max_iterations: int = 5,
):
    """构建并编译 ReAct graph。

    参数与 SecurityAgent 保持一致风格:LLM 客户端、可选工具列表、迭代上限。
    """
    tools = tools or [query_security_logs_tool]
    tool_map = {tool.name: tool for tool in tools}
    bound_model = llm_client.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        """调用 LLM,返回新的 AIMessage 与递增后的迭代计数。"""
        response = await bound_model.ainvoke(state["messages"])
        count = state.get("iteration_count", 0) + 1
        logger.info(
            "graph_agent_node",
            iteration_count=count,
            tool_call_count=len(response.tool_calls) if response.tool_calls else 0,
        )
        return {"messages": [response], "iteration_count": count}

    async def tools_node(state: AgentState) -> dict:
        """执行最后一条 AIMessage 的所有 tool_calls,返回 ToolMessage 列表。

        安全契约:任何异常只记录 error_type 与 tool_call_id,
        ToolMessage 内容是可重试的通用 JSON,不含 traceback / 路径 / 参数值。
        """
        last_ai: AIMessage = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        for tc in last_ai.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool is None:
                tool_messages.append(ToolMessage(
                    content=json.dumps({
                        "error": "未知工具",
                        "tool_name": tc["name"],
                        "suggest_retry": True,
                    }),
                    tool_call_id=tc["id"],
                ))
                continue

            try:
                result = await tool.ainvoke(tc["args"])
                tool_messages.append(ToolMessage(
                    content=result if isinstance(result, str) else str(result),
                    tool_call_id=tc["id"],
                ))
            except Exception as exc:
                logger.error(
                    "graph_tool_failed",
                    tool_name=tc["name"],
                    error_type=type(exc).__name__,
                    tool_call_id=tc["id"],
                )
                tool_messages.append(ToolMessage(
                    content=json.dumps({
                        "error": "工具执行失败",
                        "type": type(exc).__name__,
                        "details": "请调整查询条件后重试",
                    }),
                    tool_call_id=tc["id"],
                ))

        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> str:
        """条件边:按 tool_calls 与业务层迭代上限决定去向。"""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        if state.get("iteration_count", 0) >= max_iterations:
            logger.info("graph_max_iterations_reached", max_iterations=max_iterations)
            return END
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
