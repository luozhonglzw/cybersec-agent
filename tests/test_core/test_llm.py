"""LLMClient 的最小测试:不产生任何真实 API 请求。

只验证"配置 → SDK 参数"的映射和错误归类,不验证 SDK 内部行为。
"""
import pytest
from langchain_core.messages import HumanMessage

from app.core.config import Settings
from app.core.llm import LLMClient, LLMInvocationError


def _settings() -> Settings:
    """构造仅用于测试的配置(假 key,不会发出任何请求)。"""
    return Settings(
        llm_model="test-model",
        llm_base_url="http://127.0.0.1:9",  # 指向不存在端口的假地址
        llm_api_key="sk-test-only",
    )


def test_client_builds_model_from_settings():
    """Settings 的 model 正确传给 ChatOpenAI。

    ChatOpenAI 构造是惰性的(不发网络请求),所以这个测试可以离线运行。
    """
    client = LLMClient(_settings())
    assert client._model.model_name == "test-model"  # noqa: SLF001


@pytest.mark.asyncio
async def test_invocation_error_is_wrapped(monkeypatch):
    """调用失败时抛出 LLMInvocationError,而不是裸的底层异常。

    这样调用方可以统一按"LLM 调用失败"处理,同时通过 __cause__ 看到原始异常。
    """

    class _Boom:
        """替代真实模型:ainvoke 永远抛错,模拟网络故障。"""

        async def ainvoke(self, messages):
            raise RuntimeError("network down")

    client = LLMClient(_settings())
    monkeypatch.setattr(client, "_model", _Boom())
    with pytest.raises(LLMInvocationError) as exc_info:
        await client.chat([HumanMessage(content="hi")])
    assert isinstance(exc_info.value.__cause__, RuntimeError)
