"""合成 **provider 形状** 的响应与模型(**零网络、零 SDK、零凭据**)。

为什么必须"形状正确"
--------------------
标定阶段要检验的事情之一是:"我们能不能从一次真实响应里读出该读的东西"。
如果合成响应只是一个带 `content` 的裸 `AIMessage`,那么 C0a 的十个捕获项
里会有八个恒为 `NOT_AVAILABLE` —— 测试全绿,而**真实的响应形状从未被检验**。
这类测试给出的信心是假的。

所以合成响应必须带上真实 provider 会带的字段:

    response_metadata["model_name"]           provider **自报**的模型标识
    response_metadata["token_usage"]          provider **原始**用量字典
    response_metadata["finish_reason"]        停止原因
    response_metadata["id"]                   响应 id
    response_metadata["system_fingerprint"]   系统指纹
    usage_metadata                            归一化用量(可能为 None)

两种 usage 形状(**刻意都提供**)
--------------------------------
    OpenAI 形状   缓存命中计数**嵌套**在 `prompt_tokens_details.cached_tokens`
    DeepSeek 形状 缓存计数是 **顶层** 字段 `prompt_cache_hit_tokens` /
                  `prompt_cache_miss_tokens`

两者都给出来,是为了让"LangChain 的归一化映射只认 OpenAI 形状"这件事
**可被机械证明**:同样的缓存命中数,OpenAI 形状会出现在 `usage_metadata` 里,
DeepSeek 形状**不会** —— 它只可能存活在原始字典里。于是"原始 usage 必须被
保留"这条要求就有了一个有牙的测试,而不是一句声明。

本模块**不 import 任何 provider 客户端**(`openai` / `langchain_openai` /
`anthropic` / `httpx` / `requests` / `aiohttp`),只 import `langchain_core`
的消息类型。它也不读环境、不读凭据、不发起任何 I/O。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import AIMessage

# ---------------------------------------------------------------------------
# 合成常量(**明显是合成物**,不与任何真实标识混淆)
# ---------------------------------------------------------------------------

#: 合成响应里 provider "自报"的模型标识。刻意与请求的模型**不同** ——
#: 这样"requested != provider-reported"这条区分才有内容可测。
SYNTHETIC_PROVIDER_MODEL = "DeepSeek-V4.1-Flash"

#: 合成响应 id。
SYNTHETIC_RESPONSE_ID = "chatcmpl-d2c-synthetic-0001"

#: 合成系统指纹。
SYNTHETIC_SYSTEM_FINGERPRINT = "fp_d2c_synthetic"

#: 合成 provider 的顶层缓存计数器名(DeepSeek 形状)。
DEEPSEEK_CACHE_HIT_KEY = "prompt_cache_hit_tokens"
DEEPSEEK_CACHE_MISS_KEY = "prompt_cache_miss_tokens"

#: 合成 provider 的嵌套缓存计数器路径(OpenAI 形状)。
OPENAI_CACHED_TOKENS_KEY = "prompt_tokens_details"


# ---------------------------------------------------------------------------
# usage 构造
# ---------------------------------------------------------------------------


def deepseek_shaped_raw_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int | None = None,
) -> dict[str, Any]:
    """DeepSeek 形状的**原始** usage。

    缓存计数在**顶层**。归一化映射不读它,因此它只可能存活在原始字典里 ——
    这正是"原始 usage 不得被丢弃"要检验的东西。
    """
    miss = (
        prompt_tokens - prompt_cache_hit_tokens
        if prompt_cache_miss_tokens is None
        else prompt_cache_miss_tokens
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        DEEPSEEK_CACHE_HIT_KEY: prompt_cache_hit_tokens,
        DEEPSEEK_CACHE_MISS_KEY: miss,
    }


def openai_shaped_raw_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    """OpenAI 形状的**原始** usage。缓存计数**嵌套**在 `prompt_tokens_details`。

    提供它是为了做**对照**:同样的缓存命中数,这个形状会被归一化映射读到,
    DeepSeek 形状不会。没有这个对照,"归一化只认 OpenAI 形状"就只是一句话。
    """
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        OPENAI_CACHED_TOKENS_KEY: {"cached_tokens": cached_tokens},
    }


def normalize_openai_shaped(raw: dict[str, Any]) -> dict[str, Any]:
    """复刻 LangChain 的归一化映射(**只读 OpenAI 形状的嵌套字段**)。

    刻意照抄映射行为而不是"聪明地"兼容两种形状:本函数的价值就在于
    它**不**认 DeepSeek 的顶层缓存计数器 —— 一旦它"顺手兼容"了,
    "原始字典必须保留"这条要求就失去了它的对照。
    """
    details = raw.get(OPENAI_CACHED_TOKENS_KEY)
    input_details: dict[str, Any] = {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            input_details["cache_read"] = cached
    return {
        "input_tokens": raw.get("prompt_tokens", 0),
        "output_tokens": raw.get("completion_tokens", 0),
        "total_tokens": raw.get("total_tokens", 0),
        "input_token_details": input_details,
    }


# ---------------------------------------------------------------------------
# 合成 AIMessage
# ---------------------------------------------------------------------------


def synthetic_ai_message(
    *,
    content: str = "",
    tool_calls: Sequence[dict[str, Any]] = (),
    finish_reason: str | None = "stop",
    raw_usage: dict[str, Any] | None = None,
    normalized_usage: dict[str, Any] | None = None,
    provider_model: str | None = SYNTHETIC_PROVIDER_MODEL,
    response_id: str | None = SYNTHETIC_RESPONSE_ID,
    system_fingerprint: str | None = SYNTHETIC_SYSTEM_FINGERPRINT,
    invalid_tool_calls: Sequence[dict[str, Any]] | None = None,
    with_metadata: bool = True,
) -> AIMessage:
    """构造一条 **provider 形状** 的 `AIMessage`。

    `with_metadata=False` 用来构造"provider 什么都没报"的对照响应 ——
    用于验证捕获层在字段缺失时如实记 `NOT_AVAILABLE`,而不是臆造。
    """
    metadata: dict[str, Any] = {}
    if with_metadata:
        if provider_model is not None:
            metadata["model_name"] = provider_model
        if raw_usage is not None:
            metadata["token_usage"] = dict(raw_usage)
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        if response_id is not None:
            metadata["id"] = response_id
        if system_fingerprint is not None:
            metadata["system_fingerprint"] = system_fingerprint

    kwargs: dict[str, Any] = {
        "content": content,
        "tool_calls": [dict(call) for call in tool_calls],
        "response_metadata": metadata,
    }
    if normalized_usage is not None:
        kwargs["usage_metadata"] = dict(normalized_usage)
    if invalid_tool_calls:
        # LangChain 把解析失败的 tool call 放在 additional_kwargs 里,
        # 并通过 `invalid_tool_calls` 属性暴露。
        kwargs["additional_kwargs"] = {"invalid_tool_calls": list(invalid_tool_calls)}
    return AIMessage(**kwargs)


def synthetic_tool_call(
    *, name: str, args: Any, call_id: str | None
) -> dict[str, Any]:
    """构造一个 tool call 字典(与 LangChain 线上形状一致)。

    `call_id=None` 用来构造"缺少 tool_call_id"的**畸形**输入 —— 它不是
    一个合法的 tool call,标定必须能把它判成缺陷而不是默默补一个 id。
    """
    call: dict[str, Any] = {"name": name, "args": args, "type": "tool_call"}
    if call_id is not None:
        call["id"] = call_id
    return call


# ---------------------------------------------------------------------------
# 合成模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticTurn:
    """合成模型的一轮输出。"""

    content: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    finish_reason: str | None = "stop"
    raw_usage: dict[str, Any] | None = None
    normalized_usage: dict[str, Any] | None = None
    provider_model: str | None = SYNTHETIC_PROVIDER_MODEL
    invalid_tool_calls: tuple[dict[str, Any], ...] = ()


class SyntheticProviderModel:
    """**确定性、离线**的 provider 形状假模型。

    它不试图"模拟得像真 LLM" —— 它按预设轮次逐字返回,没有任何随机性。
    它的用途是让**响应形状**可被精确控制,因为标定要检验的正是形状解析。

    只实现生产路径真正使用的两个方法(`bind_tools` / `ainvoke`),
    与 `ScriptedLLM` 保持同一份接口契约。**不发起任何网络调用。**
    """

    def __init__(
        self,
        *,
        turns: Sequence[SyntheticTurn],
        provider_model: str | None = SYNTHETIC_PROVIDER_MODEL,
        exhaust_note: str = "(合成脚本已耗尽)",
    ) -> None:
        self._turns = tuple(turns)
        self._provider_model = provider_model
        self._exhaust_note = exhaust_note
        self._invocations = 0
        self._bound_tool_names: tuple[str, ...] = ()

    # ---- 生产路径使用的接口 ----

    def bind_tools(self, tools: Any) -> "SyntheticProviderModel":
        self._bound_tool_names = tuple(
            getattr(tool, "name", str(tool)) for tool in tools
        )
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self._invocations += 1
        index = self._invocations - 1
        if index >= len(self._turns):
            return synthetic_ai_message(
                content=self._exhaust_note,
                finish_reason="stop",
                provider_model=self._provider_model,
                response_id=None,
                system_fingerprint=None,
            )
        turn = self._turns[index]
        return synthetic_ai_message(
            content=turn.content,
            tool_calls=turn.tool_calls,
            finish_reason=turn.finish_reason,
            raw_usage=turn.raw_usage,
            normalized_usage=turn.normalized_usage,
            provider_model=turn.provider_model,
        )

    # ---- 只读观测 ----

    @property
    def invocations(self) -> int:
        return self._invocations

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        return self._bound_tool_names
