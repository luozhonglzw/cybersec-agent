"""真实 provider 的**构造边界**(刻意放在 `app/evaluation/llm/` **之外**)。

为什么不在评测包里直接写 provider 构造
--------------------------------------
`app/evaluation/llm/` 有一条**冻结护栏**:包内任何模块都不得 import 任何
provider 客户端(`openai` / `langchain_openai` / `anthropic` / `httpx` /
`requests` / `aiohttp`)。这条护栏由
`tests/test_evaluation_llm/test_d2a_pipeline.py::test_no_evaluation_module_imports_a_provider_client`
机械守住,并在 D-2b 被 `test_d2b_boundary.py` 再次确认。

护栏的价值在于:它保证**离线评测路径永远不可能**因为一次重构而意外开始
打真实端点。为了让真实 provider 仍然可用,构造被放到这个包外模块里,
由调用方**显式注入**(见 `app/evaluation/llm/adapters.py` 的 `LLMFactory`)。

本模块的三条硬约束
------------------
1. **导入期零副作用。** 模块顶层只 import 标准库与 `app.evaluation.llm.identity`
   (它本身零 provider 依赖)。`langchain_openai` 的 import **在函数体内**,
   客户端构造也**在函数体内**。因此 `import app.evaluation.real_provider`
   不会加载任何 provider SDK,更不会建立任何连接。

2. **本阶段不发起任何调用。** 本模块只提供**构造**能力。D-2b 阶段
   **不调用** `make_llm_factory` / `build_chat_model`,也不构造任何客户端 ——
   它只被静态检查与单元测试以"构造器形态"使用。

3. **凭据绝不进入产物。** 凭据由调用方以参数传入,**不被存储**在本模块的任何
   对象里;身份对象只记录 base_url 的**主机名摘要**(见
   `identity.base_url_host_sha256`)。

候选(**不是**胜者)
-------------------
`CANDIDATES` 里的两条是 controller 正在收窄的**候选路径**,两者都是
OpenAI-compatible 形态,因此走**同一段**构造代码 —— 本模块**不实现**
任何"某模型专属"的行为分支。

    Candidate A   OpenAI-compatible **OpenAI** 官方模型路径
    Candidate B   OpenAI-compatible **DeepSeek** 模型路径

本阶段**不冻结**其中任何一个为胜者,也**不实现** Qwen 专属行为。
"""
from dataclasses import dataclass
from typing import Any, Callable

from app.evaluation.llm.identity import (
    REAL_PROVIDER_TEMPERATURE,
    EndpointCategory,
    ModelIdentity,
    provider_identity,
)

# ---------------------------------------------------------------------------
# 候选描述(**声明式**,零模型专属分支)
# ---------------------------------------------------------------------------

#: 候选 A:OpenAI 官方端点。
CANDIDATE_A = "candidate-a-openai"
#: 候选 B:DeepSeek 的 OpenAI-compatible 端点。
CANDIDATE_B = "candidate-b-deepseek"


@dataclass(frozen=True)
class ProviderCandidate:
    """一条**候选** provider 路径的声明式描述。

    它刻意**不含凭据**:只声明"需要读哪个环境变量",不持有其值。
    """

    candidate_id: str
    provider: str
    model: str
    endpoint_category: EndpointCategory
    api_key_env: str
    base_url_env: str | None
    default_base_url: str | None
    notes: str

    def credential_env_names(self) -> tuple[str, ...]:
        """调用方需要自己解析的环境变量名(**本模块不读环境**)。"""
        names = [self.api_key_env]
        if self.base_url_env:
            names.append(self.base_url_env)
        return tuple(names)


#: **候选**清单。顺序不代表偏好,更不代表胜者。
CANDIDATES: tuple[ProviderCandidate, ...] = (
    ProviderCandidate(
        candidate_id=CANDIDATE_A,
        provider="openai",
        model="gpt-4o-mini",
        endpoint_category=EndpointCategory.OPENAI_OFFICIAL,
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url=None,
        notes=(
            "OpenAI 官方端点。base_url 留空即用 SDK 默认值,"
            "因此不需要额外的兼容性假设。"
        ),
    ),
    ProviderCandidate(
        candidate_id=CANDIDATE_B,
        provider="deepseek",
        model="deepseek-chat",
        endpoint_category=EndpointCategory.OPENAI_COMPATIBLE,
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com/v1",
        notes=(
            "DeepSeek 的 OpenAI-compatible 端点。端点类别是 "
            "OPENAI_COMPATIBLE 而**不是** OPENAI_OFFICIAL —— "
            "两者在清单里必须可区分。"
        ),
    ),
)


def candidate_by_id(candidate_id: str) -> ProviderCandidate:
    for candidate in CANDIDATES:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(
        f"未知候选 {candidate_id!r};已知候选为 "
        f"{[item.candidate_id for item in CANDIDATES]}"
    )


def describe_candidates() -> list[dict[str, Any]]:
    """候选的可审阅摘要。**不含任何凭据,也不含完整 base_url。**"""
    return [
        {
            "candidate_id": candidate.candidate_id,
            "provider": candidate.provider,
            "model": candidate.model,
            "endpoint_category": candidate.endpoint_category.value,
            "credential_env_names": list(candidate.credential_env_names()),
            "notes": candidate.notes,
        }
        for candidate in CANDIDATES
    ]


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------


class MissingCredentialError(RuntimeError):
    """缺少凭据。

    **绝不回退到 `ScriptedLLM`。** 静默回退是这类代码最危险的失效模式:
    一次"以为在打真实模型"的运行会产出一份脚本化结果,而所有产物看起来
    完全正常 —— 身份字段甚至会因为默认值而自洽。
    """


class RealProviderDisabled(RuntimeError):
    """真实 provider 未被显式启用。

    调用方必须**显式**声明"我知道这会打网络"。默认拒绝。
    """


# ---------------------------------------------------------------------------
# 构造(**全部惰性**)
# ---------------------------------------------------------------------------


def build_chat_model(
    *,
    candidate: ProviderCandidate,
    api_key: str,
    base_url: str | None = None,
    temperature: float | None = REAL_PROVIDER_TEMPERATURE,
    timeout: float | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    streaming: bool | None = None,
    extra_body: dict[str, Any] | None = None,
    http_async_client: Any | None = None,
    allow_network: bool = False,
) -> Any:
    """构造一个 OpenAI-compatible 聊天模型。**唯一的 provider 客户端落点。**

    `allow_network=False`(默认)时**直接拒绝** —— 一个"不小心就发请求"的
    构造器迟早会在某个测试里被真的调用一次。

    **凭据检查排在网络开关之前**:缺凭据时先报"缺凭据",而不是先报"没开网络"。
    两者都是拒绝,但前者把注意力引向真正的配置问题。

    `langchain_openai` 的 import 在函数体内:导入本模块**不会**加载 provider SDK。

    D-2c 新增的四个可选旋钮(**全部默认 `None` ⇒ 与 D-2b 逐字段一致**)
    ------------------------------------------------------------------
    `temperature`  传 `None` 表示 **NOT_SET**:该字段**完全不进入**请求体。
                   这与"传 0.0"是两件不同的事 —— 前者表示"请求不依赖温度",
                   后者是一个被显式设定的采样参数。调用方**不得**用 0.0
                   冒充 NOT_SET(见 `app/evaluation/calibration/config.py`)。
    `max_retries`  显式设定 SDK 层重试次数。不传 ⇒ 沿用 SDK 默认值。
                   SDK 重试对 `BudgetGovernor` **不可见**:一次逻辑调用可能
                   产生多次物理 HTTP 尝试,因此标定阶段必须显式声明。
    `streaming`    显式设定,不依赖 SDK 默认值 —— 便于溯源。
    `extra_body`   原样透传给 provider 请求体的**通用**逃生口。
                   本模块**不理解**其中任何键的语义:provider 专属参数
                   (如关闭思考模式)由调用方以数据形式提供,因此这里
                   **不引入任何模型专属分支**。
    `http_async_client`
                   在**构造期**注入的异步 HTTP 客户端(**F8 修复**,
                   见 `app/evaluation/calibration/` 的标定证据缺口)。
                   用途:调用方传入一个自带记录钩子的真实
                   `httpx.AsyncClient`,即可观测**最终序列化的请求体**;
                   传入 `httpx.MockTransport` 客户端则可做完全离线的
                   形状验证。默认 `None` ⇒ 该 kwarg 完全不传 ⇒
                   与 D-2b / D-2c 的既有行为逐字段一致。

    **为什么必须是"构造期",而不是"构造后赋值"**
    --------------------------------------------
    已实测(见 `tests/test_evaluation_llm/test_d2c_f8_payload_seam.py`):
    `langchain_openai` 的 `BaseChatOpenAI` 在**构造期**就把
    `root_client` / `client` / `root_async_client` / `async_client` 全部物化
    (`chat_models/base.py:827-846`),并在这一刻读取 `self.http_async_client`。
    构造之后再赋值虽然会被存进实例,却**永远不会被读取** —— 于是
    "我挂上了拦截器"与"请求真的走了拦截器"分叉,而两边看起来都正常。
    因此本参数**只在构造期生效**;本模块**不提供**任何构造后替换入口。
    """
    if not api_key or not api_key.strip():
        raise MissingCredentialError(
            f"候选 {candidate.candidate_id!r} 缺少凭据 —— "
            "**不**回退到 ScriptedLLM:静默回退会让一次真实运行产出脚本化结果。"
        )
    if not allow_network:
        raise RealProviderDisabled(
            "真实 provider 构造需要显式 allow_network=True —— "
            "默认拒绝,避免任何一次意外网络出口。"
        )

    from langchain_openai import ChatOpenAI  # noqa: PLC0415 —— 刻意的惰性 import

    kwargs: dict[str, Any] = {
        "model": candidate.model,
        "api_key": api_key,
    }
    # NOT_SET 就是"不发这个字段"。用 `is not None` 而不是真值判断:
    # `temperature=0.0` 是合法且**必须**被送出的显式取值。
    if temperature is not None:
        kwargs["temperature"] = temperature
    resolved_base_url = base_url or candidate.default_base_url
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if streaming is not None:
        kwargs["streaming"] = streaming
    if extra_body is not None:
        # 拷一份:调用方的 dict 之后被改动不应影响已构造的客户端。
        kwargs["extra_body"] = dict(extra_body)
    if http_async_client is not None:
        # **必须**在这里传入:见 docstring —— SDK 客户端在构造期即被物化。
        kwargs["http_async_client"] = http_async_client
    return ChatOpenAI(**kwargs)


def make_llm_factory(
    *,
    candidate: ProviderCandidate,
    api_key: str,
    base_url: str | None = None,
    temperature: float | None = REAL_PROVIDER_TEMPERATURE,
    timeout: float | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    streaming: bool | None = None,
    extra_body: dict[str, Any] | None = None,
    http_async_client: Any | None = None,
    allow_network: bool = False,
) -> Callable[..., Any]:
    """返回一个符合 `adapters.LLMFactory` 协议的工厂。

    适配器会用 `behavior` / `task` / `dataset_paths` / `decoy_paths` /
    `emit_usage` 这几个 keyword 调用它 —— 它们描述的是**评测上下文**,
    与"用哪个模型"无关,因此这里原样忽略。

    ⚠️ 返回的工厂**每次调用都会构造一个新客户端**。D-2b 阶段不调用它。
    """
    def factory(**_: Any) -> Any:
        return build_chat_model(
            candidate=candidate,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            streaming=streaming,
            extra_body=extra_body,
            http_async_client=http_async_client,
            allow_network=allow_network,
        )

    return factory


def identity_for_candidate(
    candidate: ProviderCandidate,
    *,
    base_url: str | None = None,
    provider_reported_model_id: str | None = None,
    provider_default_parameters: dict[str, Any] | None = None,
    temperature: float | None = REAL_PROVIDER_TEMPERATURE,
) -> ModelIdentity:
    """构造与候选一致的 `ModelIdentity`。

    `base_url` 只被用来算**主机名摘要**;完整 URL(可能内嵌凭据)**不进入**身份对象。

    `temperature` 传 `None` 表示 **NOT_SET**(请求体里根本没有这个字段)。
    它必须与**实际送出的请求**取自同一个来源 —— 见
    `app/evaluation/calibration/config.py` 的单一事实来源约束。
    """
    return provider_identity(
        provider=candidate.provider,
        model=candidate.model,
        endpoint_category=candidate.endpoint_category,
        base_url=base_url or candidate.default_base_url,
        provider_reported_model_id=provider_reported_model_id,
        provider_default_parameters=provider_default_parameters,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 响应侧读取(**纯 getattr,不 import 任何 SDK**)
# ---------------------------------------------------------------------------


def provider_reported_model_id_of(message: Any) -> str | None:
    """从 `AIMessage.response_metadata["model_name"]` 取 provider **自报**的模型标识。

    为什么必须与"请求的模型"分开记录:provider 完全可能回一个别名、一个
    版本化标识,或一个家族名。把两者合并会让"我们请求了什么"与
    "provider 实际用了什么"无法区分 —— 而后者正是标定要确认的东西之一。

    缺失 / 空串一律返回 `None`(= NOT_AVAILABLE),**不猜**。
    """
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("model_name")
    if isinstance(value, str) and value.strip():
        return value
    return None


def raw_token_usage_of(message: Any) -> dict[str, Any] | None:
    """从 `AIMessage.response_metadata["token_usage"]` 取 provider **原始**用量字典。

    为什么需要原始字典:LangChain 的归一化 `usage_metadata` 只映射
    OpenAI 形状的嵌套字段。provider 的**顶层**专属计数器(例如缓存命中/未命中)
    不会出现在归一化结果里,却会原样保留在这个原始字典中。
    把它丢掉等于把"provider 报了什么"降级成"我们认得什么"。

    缺失 / 非字典一律返回 `None`,**不伪造 `{}` 或 0**。
    """
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    usage = metadata.get("token_usage")
    if isinstance(usage, dict):
        return dict(usage)
    return None
