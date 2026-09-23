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
    temperature: float = REAL_PROVIDER_TEMPERATURE,
    timeout: float | None = None,
    max_tokens: int | None = None,
    allow_network: bool = False,
) -> Any:
    """构造一个 OpenAI-compatible 聊天模型。**唯一的 provider 客户端落点。**

    `allow_network=False`(默认)时**直接拒绝** —— 一个"不小心就发请求"的
    构造器迟早会在某个测试里被真的调用一次。

    **凭据检查排在网络开关之前**:缺凭据时先报"缺凭据",而不是先报"没开网络"。
    两者都是拒绝,但前者把注意力引向真正的配置问题。

    `langchain_openai` 的 import 在函数体内:导入本模块**不会**加载 provider SDK。
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
        "temperature": temperature,
    }
    resolved_base_url = base_url or candidate.default_base_url
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def make_llm_factory(
    *,
    candidate: ProviderCandidate,
    api_key: str,
    base_url: str | None = None,
    temperature: float = REAL_PROVIDER_TEMPERATURE,
    timeout: float | None = None,
    max_tokens: int | None = None,
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
            allow_network=allow_network,
        )

    return factory


def identity_for_candidate(
    candidate: ProviderCandidate,
    *,
    base_url: str | None = None,
    provider_reported_model_id: str | None = None,
    provider_default_parameters: dict[str, Any] | None = None,
    temperature: float = REAL_PROVIDER_TEMPERATURE,
) -> ModelIdentity:
    """构造与候选一致的 `ModelIdentity`。

    `base_url` 只被用来算**主机名摘要**;完整 URL(可能内嵌凭据)**不进入**身份对象。
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
