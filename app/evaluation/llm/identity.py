"""模型身份与端点类别(**离线,零 provider 依赖**)。

为什么需要这个模块
------------------
D-2a 的记录构造把身份**写死**成 scripted 三连:

    provider="scripted"
    model=f"deterministic-scripted-{behavior}"
    usage.source="scripted"

对离线脚本化运行这是**正确**的。但真实 provider 运行复用同一条记录构造路径时,
它会**静默地**给每一条真实记录贴上"脚本化"标签 —— 不是算错,而是
"看起来一切正常"。这与 F-1 / F-3 是同一类失真。

因此身份必须**由执行上下文传入**,而不是由记录构造处自行决定。

本模块的依赖纪律
----------------
只依赖标准库与 pydantic。**刻意不 import 包内任何其它模块** ——
这样 `protocol.py` / `adapters.py` / `executor.py` / `pilot.py` 都可以引用它,
不会产生循环 import。
"""
from enum import Enum
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 端点类别(冻结词表)
# ---------------------------------------------------------------------------


class EndpointCategory(str, Enum):
    """端点类别。**冻结的三值词表** —— 不是自由文本。

    为什么要有这个词表:清单里"我们打的是哪个端点"必须是一个**可机械核验**的
    分类,而不是一段 URL。URL 会随 region / 灰度 / 代理变化,而且可能内嵌凭据;
    类别不会。
    """

    SCRIPTED_OFFLINE = "SCRIPTED_OFFLINE"      # 离线脚本化,零出口
    OPENAI_OFFICIAL = "OPENAI_OFFICIAL"        # 官方 OpenAI 端点
    OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"    # 第三方 OpenAI-compatible 端点


#: 合法取值的**有序**列表(供清单校验与错误信息使用)。
ENDPOINT_CATEGORIES: tuple[str, ...] = tuple(item.value for item in EndpointCategory)


class UnknownEndpointCategory(ValueError):
    """端点类别不在冻结词表内。**拒绝,不猜。**"""


def parse_endpoint_category(value: str) -> EndpointCategory:
    """解析端点类别。未知取值一律拒绝 —— 不做"挑一个最接近的"。"""
    try:
        return EndpointCategory(value)
    except ValueError as exc:
        raise UnknownEndpointCategory(
            f"未知 endpoint_category {value!r};合法取值为 {list(ENDPOINT_CATEGORIES)}。"
            "该词表是冻结的 —— 新增取值必须走协议版本变更,不得就地扩写。"
        ) from exc


# ---------------------------------------------------------------------------
# 凭据守卫
# ---------------------------------------------------------------------------

#: 身份里**绝不允许**出现的字段名(凭据 / 完整 URL)。
#:
#: 与 `runner.FORBIDDEN_METADATA_FIELDS` 同源同义。此处**刻意重写一份**而不是
#: import:`runner` → `adapters` → 本模块,若本模块反向 import `runner` 会形成
#: 循环。两份清单由 `test_d2b_provider_boundary.py` 的等价性测试守住。
CREDENTIAL_FIELD_NAMES: frozenset[str] = frozenset({
    "api_key",
    "llm_api_key",
    "authorization",
    "auth_header",
    "token",
    "secret",
    "password",
    "base_url",  # 只允许 base_url_host / base_url_host_sha256
})


def assert_no_credential_fields(payload: Any) -> None:
    """递归检查任意 payload 的**键名**是否命中凭据清单。"""
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() in CREDENTIAL_FIELD_NAMES:
                found.add(key.lower())
            assert_no_credential_fields(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_no_credential_fields(value)
    if found:
        raise AssertionError(
            f"模型身份出现被禁止的字段 {sorted(found)} —— "
            "凭据 / 完整 URL 一律不得进入评测产物"
        )


# ---------------------------------------------------------------------------
# 主机名处理
# ---------------------------------------------------------------------------


def base_url_host(base_url: str | None) -> str | None:
    """从 base_url 取出**主机名**。取不到则返回 `None`。

    只取主机名是刻意的:完整 URL 可能内嵌凭据(例如
    `https://user:pass@host/v1`),而主机名不含凭据。
    """
    if not base_url:
        return None
    host = urlsplit(base_url).hostname
    return host or None


def base_url_host_sha256(base_url: str | None) -> str | None:
    """主机名的摘要。清单与原始记录里只允许出现这个值。"""
    host = base_url_host(base_url)
    if host is None:
        return None
    return sha256(host.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 模型身份
# ---------------------------------------------------------------------------

#: 离线脚本化 provider 的标签(与 D-1 / D-2a 保持一致,不得改写)。
SCRIPTED_PROVIDER = "scripted"

#: 真实 provider 身份构造的**历史默认值**:`0`。
#:
#: 作用域(D-2c 更正)
#: ------------------
#: 这是 D-2a / D-2b 时代"真实 provider 身份"的默认温度,也是
#: `provider_identity()` / `identity_for_candidate()` 的默认入参。保留它是为了
#: 让**已冻结的 D-2b 记录与测试逐字段不变**,不是因为它仍是当前策略。
#:
#: 它**不是** D-2c 标定的温度策略。D-2c 冻结的是 **NOT_SET**:请求体里根本
#: 不带 `temperature` 字段,记录里写 `None`。两者不可互相冒充 ——
#: "不发送该参数"与"显式设为 0"在溯源上是两件事,把前者记成后者是伪造。
#: 见 `app/evaluation/calibration/config.py`。
#:
#: 原始理由(D-2a/D-2b 仍然成立):这不是"provider 默认值",而是**实验设计
#: 参数** —— n=3 的重复运行若各自采样温度不同,"同一个条件的两次运行差异"
#: 就同时包含采样噪声与真实差异。重复性优先于"发挥模型能力"。
#:
#: 注意:它**不是**清单里的冻结值。
REAL_PROVIDER_TEMPERATURE: float = 0.0


class ModelIdentity(BaseModel):
    """一次执行里"谁在产生 `AIMessage`"的完整身份。**不含任何凭据。**

    它是记录构造的**入参**,不是记录构造的常量 —— 这正是 D-2b 要修的失真点。
    """

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1, description="provider 标签(scripted / openai / deepseek …)")
    model: str = Field(min_length=1, description="模型标识")
    endpoint_category: EndpointCategory = Field(description="冻结词表内的端点类别")
    usage_source: str = Field(
        min_length=1,
        description=(
            "token 用量的来源:scripted = 假 LLM 自造;provider_reported = "
            "provider 返回的 usage;not_available = provider 未返回"
        ),
    )
    provider_reported_model_id: str | None = Field(
        default=None,
        description="provider 在响应里回传的模型标识(可能与请求值不同);不可得时 None",
    )
    temperature: float | None = Field(
        default=None,
        description=(
            "显式温度;`None` = **NOT_SET** —— 请求体里根本没有该字段"
            "(离线脚本化如此,D-2c 标定亦如此)。"
            f"`{REAL_PROVIDER_TEMPERATURE}` 是 D-2a/D-2b 真实 provider 身份的"
            "历史默认值,**不是** D-2c 的策略。"
        ),
    )
    base_url_host_sha256: str | None = Field(
        default=None, description="**只记主机名摘要**;完整 URL 可能内嵌凭据,一律不记"
    )
    provider_default_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="provider 侧默认参数的**观测快照**(不是我们设定的值)",
    )

    @property
    def is_scripted(self) -> bool:
        """是否离线脚本化身份。报告与测试据此区分两类记录。"""
        return self.endpoint_category is EndpointCategory.SCRIPTED_OFFLINE

    def assert_consistent(self) -> None:
        """身份自洽性:类别与 provider 标签不得互相矛盾。"""
        if self.is_scripted and self.provider != SCRIPTED_PROVIDER:
            raise AssertionError(
                f"端点类别为 SCRIPTED_OFFLINE,provider 却是 {self.provider!r} —— "
                f"离线身份必须是 {SCRIPTED_PROVIDER!r}"
            )
        if not self.is_scripted and self.provider == SCRIPTED_PROVIDER:
            raise AssertionError(
                f"provider 为 {SCRIPTED_PROVIDER!r},端点类别却是 "
                f"{self.endpoint_category.value} —— 不得把真实端点标成脚本化"
            )
        assert_no_credential_fields(self.provider_default_parameters)


def scripted_identity(behavior: str) -> ModelIdentity:
    """离线脚本化身份。**与 D-1 / D-2a 的记录逐字段一致** —— 不得改动。"""
    return ModelIdentity(
        provider=SCRIPTED_PROVIDER,
        model=f"deterministic-scripted-{behavior}",
        endpoint_category=EndpointCategory.SCRIPTED_OFFLINE,
        usage_source="scripted",
    )


def provider_identity(
    *,
    provider: str,
    model: str,
    endpoint_category: EndpointCategory | str,
    base_url: str | None = None,
    usage_source: str = "provider_reported",
    provider_reported_model_id: str | None = None,
    provider_default_parameters: dict[str, Any] | None = None,
    temperature: float | None = REAL_PROVIDER_TEMPERATURE,
) -> ModelIdentity:
    """构造真实 provider 身份。

    `base_url` **只**用来算主机名摘要,原值不进入身份对象。
    `temperature` 默认取冻结策略值(0);显式传 `None` 表示"不设定温度"。
    """
    category = (
        endpoint_category
        if isinstance(endpoint_category, EndpointCategory)
        else parse_endpoint_category(endpoint_category)
    )
    identity = ModelIdentity(
        provider=provider,
        model=model,
        endpoint_category=category,
        usage_source=usage_source,
        provider_reported_model_id=provider_reported_model_id,
        temperature=temperature,
        base_url_host_sha256=base_url_host_sha256(base_url),
        provider_default_parameters=dict(provider_default_parameters or {}),
    )
    identity.assert_consistent()
    return identity
