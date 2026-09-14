"""SecurityAgent 的最小行为测试:完全 mock LLM,不产生任何真实 API 请求。"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT, SecurityAgent
from app.core.llm import FakeLLMClient


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

    assert len(fake.last_messages) >= 2  # 现在包含响应
    assert isinstance(fake.last_messages[0], SystemMessage)
    assert fake.last_messages[0].content == SECURITY_ANALYST_SYSTEM_PROMPT
    assert isinstance(fake.last_messages[1], HumanMessage)
    assert fake.last_messages[1].content == "用户问题"
