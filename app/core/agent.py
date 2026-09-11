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
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import LLMClient

logger = structlog.get_logger(__name__)

SECURITY_ANALYST_SYSTEM_PROMPT = (
    "你是一名网络安全运营(SOC)分析助手。"
    "你的职责是帮助分析师理解安全事件、分析日志与威胁情报,"
    "并给出清晰、有依据的判断。"
    "你只能提供分析与建议,不能执行任何实际操作。"
)


class SecurityAgent:
    """当前阶段:一个只做"一次 LLM 往返"的最小 Agent。

    参数:
        llm_client: LLMClient(或任何实现了 async chat(messages) -> str 的对象)
    调用:
        await agent.chat(message) -> str

    谁调用:未来的 API 层(FastAPI)与 CLI demo。
    依赖注入:Agent 只依赖"能聊天的东西",不依赖具体实现 —— 测试时换 Fake。
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def chat(self, message: str) -> str:
        """执行一轮对话:system prompt + 用户消息 → assistant 回复。

        参数:
            message: 用户输入的自然语言
        返回:
            LLM 的文本回复
        流程:
            1. 组装 [SystemMessage, HumanMessage]
            2. 交给 LLMClient
            3. 返回文本
        """
        logger.info("agent_chat_started", user_message_length=len(message))
        messages = [
            SystemMessage(content=SECURITY_ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
        reply = await self._llm.chat(messages)
        logger.info("agent_chat_completed", reply_length=len(reply))
        return reply
