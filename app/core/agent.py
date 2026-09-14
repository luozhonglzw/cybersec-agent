"""最小 SecurityAgent:User -> LLM -> Response。

本阶段(Phase 1)故意不使用 LangGraph,先实现最直白的数据流:
    user message -> [SystemMessage + HumanMessage] -> LLM -> assistant text

演进路线(记录在 docs/architecture.md §9.2):
- Phase 3:加入 Tool Calling + 手写 ReAct 循环;
- Phase 4:迁移到 LangGraph。

先亲手写一遍循环,才能理解 LangGraph 到底解决了什么问题,
而不是只会调用框架 API。
"""
import structlog
from datetime import datetime
from typing import List, Optional
import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.core.llm import LLMClient
from app.tools.query_logs import query_security_logs_tool

logger = structlog.get_logger(__name__)

SECURITY_ANALYST_SYSTEM_PROMPT = (
    "你是一名网络安全运营(SOC)分析助手。"
    "你的职责是帮助分析师理解安全事件、分析日志与威胁情报,"
    "并给出清晰、有依据的判断。"
    "工作原则:"
    "1. 先查询事实,再下结论"
    "2. 工具返回的是事实,不是最终判断"
    "3. 如果证据不足,要明确说明"
    "4. 不要把工具调用过程泄露给用户"
    "5. 对攻击判断给出证据"
    "6. 可以多次调用工具查询不同信息"
    "7. 达到最大工具调用次数时,要说明分析受限"
)


class SecurityAgent:
    """当前阶段:支持 Tool Calling 和手写 ReAct 循环的 Agent。

    参数:
        llm_client: LLMClient(或任何实现了 async chat(messages) -> str 的对象)
        tools: 可选工具列表,默认包含 query_security_logs_tool
        max_iterations: 最大工具调用次数,默认 5
    调用:
        await agent.chat(message) -> str

    谁调用:未来的 API 层(FastAPI)与 CLI demo。
    依赖注入:Agent 只依赖"能聊天的东西",不依赖具体实现 —— 测试时换 Fake。
    """
    def __init__(
        self,
        llm_client: LLMClient,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 5,
    ) -> None:
        self._llm = llm_client
        self._tools = tools or [query_security_logs_tool]
        self._max_iterations = max_iterations
        self._tool_map = {tool.name: tool for tool in self._tools}

    async def chat(self, message: str) -> str:
        """执行对话:支持 Tool Calling 和 ReAct 循环。

        参数:
            message: 用户输入的自然语言
        返回:
            LLM 的文本回复,可能包含多次工具调用后的最终分析
        流程:
            1. 组装 [SystemMessage, HumanMessage]
            2. 执行 ReAct 循环(max_iterations 次)
            3. 返回最终文本
        """
        logger.info("agent_chat_started", user_message_length=len(message))
        
        # 初始化消息列表
        messages = [
            SystemMessage(content=SECURITY_ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
        
        # 使用传入的 LLM 实例并绑定工具
        llm_with_tools = self._llm._model.bind_tools(self._tools)
        
        # 执行 ReAct 循环
        for iteration in range(self._max_iterations):
            logger.info("react_iteration", iteration=iteration + 1, max_iterations=self._max_iterations)
            
            # 调用 LLM
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # 检查是否需要工具调用
            if not response.tool_calls:
                logger.info("react_completed_no_tool_calls", iterations=iteration + 1)
                return response.content
            
            # 处理工具调用
            for tool_call in response.tool_calls:
                try:
                    # 执行工具
                    tool_message = await self._execute_tool(tool_call, messages)
                    messages.append(tool_message)
                except Exception as exc:
                    # 工具执行错误处理
                    error_message = ToolMessage(
                        content=json.dumps({
                            "error": "工具执行失败",
                            "type": type(exc).__name__,
                            "details": str(exc)
                        }),
                        tool_call_id=tool_call["id"]
                    )
                    messages.append(error_message)
        
        # 达到最大迭代次数
        logger.info("react_completed_max_iterations", max_iterations=self._max_iterations)
        return "当前分析达到最大工具调用次数,无法在限定步骤内完成分析。"

    async def _execute_tool(self, tool_call: ToolCall, messages: List) -> ToolMessage:
        """执行单个工具调用。

        参数:
            tool_call: 工具调用信息
            messages: 当前消息历史

        返回:
            ToolMessage: 工具执行结果
        """
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 查找工具
        if tool_name not in self._tool_map:
            return ToolMessage(
                content=json.dumps({
                    "error": "未知工具",
                    "tool_name": tool_name,
                    "suggest_retry": True
                }),
                tool_call_id=tool_call["id"]
            )
        
        tool = self._tool_map[tool_name]
        
        try:
            # 执行工具
            result = await tool.ainvoke(tool_call)
            
            # 验证结果是否为 JSON 字符串
            if not isinstance(result, str):
                result = str(result)
            
            return ToolMessage(content=result, tool_call_id=tool_call["id"])
            
        except Exception as exc:
            # 参数错误让 LLM 修正参数
            if "参数" in str(exc) or "argument" in str(exc).lower():
                error_info = {
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "suggest_retry": True
                }
            else:
                # 其他错误返回通用信息
                error_info = {
                    "error": "工具执行失败",
                    "type": "ToolExecutionError",
                    "details": "请稍后重试或简化查询条件"
                }
            
            return ToolMessage(
                content=json.dumps(error_info),
                tool_call_id=tool_call["id"]
            )