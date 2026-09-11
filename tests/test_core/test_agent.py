"""SecurityAgent 的最小行为测试:完全 mock LLM,不产生任何真实 API 请求。"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT, SecurityAgent


class FakeLLMClient:
    """内存中的假 LLM Client:记录收到的 messages,返回预设回复。

    为什么不用 MagicMock:显式的 Fake 可读性更好,
    断言"收到了什么"一眼就能看懂,不需要 mock 框架知识。
    """

    def __init__(self, reply: str = "收到,正在分析。") -> None:
        self.reply = reply
        self.last_messages: list = []

    async def chat(self, messages):
        self.last_messages = messages
        return self.reply


@pytest.mark.asyncio
async def test_chat_returns_llm_reply():
    """Agent 把 LLM 的回复原样返回给用户。"""
    agent = SecurityAgent(FakeLLMClient(reply="这是一条模拟回复"))
    result = await agent.chat("帮我看看日志")
    assert result == "这是一条模拟回复"


@pytest.mark.asyncio
async def test_chat_sends_system_prompt_and_user_message():
    """Agent 组装的消息序列:第一条是 system prompt,第二条是用户消息。"""
    fake = FakeLLMClient()
    agent = SecurityAgent(fake)
    await agent.chat("用户问题")

    assert len(fake.last_messages) == 2
    assert isinstance(fake.last_messages[0], SystemMessage)
    assert fake.last_messages[0].content == SECURITY_ANALYST_SYSTEM_PROMPT
    assert isinstance(fake.last_messages[1], HumanMessage)
    assert fake.last_messages[1].content == "用户问题"
