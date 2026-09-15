"""LLM Client:OpenAI-compatible Chat Model 的统一封装。

设计原因:
- 全项目只有这一个地方创建 ChatOpenAI —— 换 provider 只改 .env,不碰业务代码;
- 只有这一个地方取出 API Key 并注入 —— 日志安全容易保证;
- 错误在这里归类(初始化 vs 调用),调用方可以区分处理。

日志安全:只记录 provider / model / 消息数 / 延迟 / 错误类型,
绝不记录 API Key、Authorization 头或消息内容。
"""
import time
from typing import List, Optional

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class FakeChatModel:
    """测试专用的 LLM 行为模拟器。
    
    重要说明：此类仅用于测试 SecurityAgent 的 ReAct 控制流，
    不模拟真实的 ChatOpenAI.bind_tools() 工具选择机制。
    
    职责：
    1. 生成预设的 AIMessage 响应
    2. 根据预设响应生成 tool_calls 
    3. 返回预设的工具执行结果
    
    不负责：
    - 真实的工具绑定和选择
    - 参数类型验证
    - 工具路由逻辑
    """
    
    def __init__(self, responses: list[str], tool_results: list[dict] = None):
        self.responses = responses
        self.tool_results = tool_results or []
        self.call_count = 0
        self.messages_history: list = []
    
    async def ainvoke(self, messages: list):
        """模拟 LLM 调用，返回 AIMessage。
        
        Args:
            messages: 消息历史（仅用于调试，不参与工具选择）
            
        Returns:
            AIMessage: 可能包含 tool_calls 的响应消息
        """
        self.call_count += 1
        self.messages_history = messages
        
        if not self.responses:
            return AIMessage(content="没有预设响应")
        
        response = self.responses.pop(0)

        # 模拟工具调用:只要还有预设的工具结果,当前轮次就触发一次工具调用。
        # 这是简化的测试模拟:按顺序消耗预设的响应与工具结果,
        # 不反映真实的 bind_tools 工具选择逻辑。
        if self.tool_results:
            tool_result = self.tool_results.pop(0)
            tool_call = ToolCall(
                name="query_security_logs_tool",
                args=tool_result.get("args", {}),
                id=f"tool_call_{self.call_count}"  # 测试用的自生成 ID
            )
            return AIMessage(content="", tool_calls=[tool_call])

        return AIMessage(content=response)
    
    def bind_tools(self, tools):
        """测试用的 bind_tools 模拟。
        
        重要：此方法仅用于测试兼容性，
        不实现真实的工具绑定逻辑。
        """
        # 直接返回 self，因为测试中不需要真实的工具绑定
        return self


class LLMClientError(Exception):
    """LLM Client 所有错误的基类。"""


class LLMInitError(LLMClientError):
    """创建 Chat Model 失败(通常是配置问题)。"""


class LLMInvocationError(LLMClientError):
    """调用 LLM 失败(网络/限流/服务端错误等)。"""


class LLMClient:
    """OpenAI-compatible Chat Model 的最小封装。

    参数:
        settings: 全局配置;默认取 get_settings() 单例
    调用:
        await client.chat(messages) -> str  (assistant 的文本回复)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        try:
            self._model = ChatOpenAI(
                model=self._settings.llm_model,
                base_url=self._settings.llm_base_url,
                api_key=self._settings.llm_api_key.get_secret_value(),
            )
        except Exception as exc:
            # raise ... from exc 保留原始异常链,方便定位真正根因
            raise LLMInitError(
                f"创建 Chat Model 失败:model={self._settings.llm_model!r}, "
                f"base_url={self._settings.llm_base_url!r}"
            ) from exc
        # 注意:log context 里绝不能放 api_key
        self._log = logger.bind(
            provider=self._settings.llm_provider,
            model=self._settings.llm_model,
        )

    def bind_tools(self, tools):
        """把 LangChain 工具绑定到内部 Chat Model,返回绑定了工具的 runnable。

        调用方(SecurityAgent / 未来的 LangGraph agent 节点)只应通过
        本方法获取带工具的模型,不应直接访问 _model ——
        这样 API Key 的注入边界仍然只有 LLMClient 一处。
        """
        return self._model.bind_tools(tools)

    async def chat(self, messages: list[BaseMessage]) -> str:
        """把一组消息发给 LLM,返回 assistant 的文本回复。"""
        started = time.perf_counter()
        self._log.info("llm_request_started", message_count=len(messages))
        try:
            response = await self._model.ainvoke(messages)
        except Exception as exc:
            self._log.error(
                "llm_request_failed",
                error_type=type(exc).__name__,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise LLMInvocationError(
                f"LLM 调用失败({type(exc).__name__})"
            ) from exc

        self._log.info(
            "llm_request_completed",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        text = response.content
        # 当前阶段只处理纯文本;未来出现工具调用等内容块时兜底成字符串
        return text if isinstance(text, str) else str(text)


class FakeLLMClient:
    """内存中的假 LLM Client:记录收到的 messages,返回预设回复。

    为什么不用 MagicMock:显式的 Fake 可读性更好,
    断言"收到了什么"一眼就能看懂,不需要 mock 框架知识。
    """
    def __init__(
        self,
        reply: str = "收到,正在分析。",
        raise_error: bool = False,
        model=None,
    ) -> None:
        self.reply = reply
        self.raise_error = raise_error
        self.last_message: str | None = None
        self.last_messages: list = []
        # 可选注入底层模型(如 FakeChatModel);未注入时由本 Fake 自行响应
        self._model = model

    async def ainvoke(self, messages):
        """兼容 SecurityAgent 的 ainvoke 调用。"""
        self.last_messages = messages
        if self.raise_error:
            raise LLMInvocationError("LLM 调用失败(模拟异常)")
        # 记录最后一条用户消息,供 API 层测试断言透传
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                self.last_message = msg.content
                break
        return AIMessage(content=self.reply)

    def bind_tools(self, tools):
        """与 LLMClient.bind_tools 同契约:返回绑定了工具的可 ainvoke 对象。

        未注入 model 时返回自身(自身实现 ainvoke);
        注入后委托给 model.bind_tools,与生产路径一致。
        """
        self.bound_tools = tools
        if self._model is not None:
            return self._model.bind_tools(tools)
        return self
    
    async def chat(self, messages):
        self.last_messages = messages
        return self.reply
