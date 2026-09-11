"""LLM Client:OpenAI-compatible Chat Model 的统一封装。

设计原因:
- 全项目只有这一个地方创建 ChatOpenAI —— 换 provider 只改 .env,不碰业务代码;
- 只有这一个地方取出 API Key 并注入 —— 日志安全容易保证;
- 错误在这里归类(初始化 vs 调用),调用方可以区分处理。

日志安全:只记录 provider / model / 消息数 / 延迟 / 错误类型,
绝不记录 API Key、Authorization 头或消息内容。
"""
import time

import structlog
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


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
