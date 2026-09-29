"""D-2d 正式试点的 provider 配置:**请求溯源与记录溯源的单一事实来源**。

为什么必须是"单一事实来源"
--------------------------
与 `app/evaluation/calibration/config.py` 完全同构的理由:**发出去的**参数与
**记录下来的**参数若来自两处代码,一次试点最可能的产出不是结果,而是一个
**自洽的假象** —— 记录说 `temperature = NOT_SET`,实际请求体里却带着
`temperature: 0.0`,而两边看起来都对。

所以这里只留一个对象:`PilotProviderConfig`。请求 kwargs 与记录值都从它派生,
两者在结构上无法分叉。

与 D-2b 候选表的关系(**必须显式说明**)
--------------------------------------
`app/evaluation/real_provider.py` 的 `CANDIDATES` 是 D-2b 的**候选**路径,
其中 `CANDIDATE_B` 的 `model` 是 `deepseek-chat`。**控制器已明确排除该模型**
(见 `EXCLUDED_MODELS`),因此 D-2d **不得**复用 `CANDIDATES[CANDIDATE_B]`。
本模块自建候选(`provider_candidate()`),`build_chat_model` 本来就接受调用方
传入的候选,因此不需要改动 `CANDIDATES`。

与 D-2c 标定配置的关系
----------------------
D-2c 的 `CalibrationConfig` 是**标定阶段专属**的模块(其 docstring 明确
"刻意不复用 `CANDIDATE_A` / `CANDIDATE_B`",`candidate_id` 也带 `d2c-cal-` 前缀)。
D-2d 是**正式试点**,需要自己的、以 `d2d-pilot-` 为前缀的冻结配置。
两者当前在**共享字段上取值相同**(provider / requested model / base_url /
streaming / temperature / thinking / timeout / max_retries / output cap),
但那是**巧合于同一份控制器决定**,不是"一个是另一个的实例"。

⚠️ 已知的**重复定义**残留:上述 9 个取值目前在两个模块里各写了一份。
按"最小改动"原则本阶段**没有**把 `calibration/config.py` 改成引用本模块
(那会动到已冻结的 D-2c 配置与其测试)。若控制器希望消除重复,应另开一次
"抽取共享 provider profile"的授权 —— 见最终报告 §13。

`temperature = NOT_SET` 的含义
-----------------------------
`NOT_SET` 用 `None` 表示,含义是**请求体里没有 `temperature` 字段**。
它与 `temperature=0.0` 是两件不同的事:后者是一个被显式设定的采样参数。
把前者记成后者是**伪造**。因此 `recorded_temperature()` 恒为 `None`,
且本模块**不提供**任何把它变成 `0.0` 的入口。

`http_attempt_ceiling` 的**分类**(§G)
------------------------------------
`pilot.pilot_plan()` 的**默认** `provider_http_attempt_ceiling = 972` 是
**历史推导值**,其推导假设是 **SDK `max_retries = 2`**(`1 + 2 = 3` 次尝试)。
该默认值**逐字段不变**,历史产物因此仍然可复现。

D-2d 的 provider 配置把 `sdk_max_retries` 显式冻结为 **0**,因此 D-2d 的
配置包络是 `pilot_plan(sdk_max_retries=0)` ⇒ **324**:

    * 972 **不是** D-2d 的执行包络;它是**历史**推导值;
    * D-2d 的清单必须携带 **324**(见 `d2d_plan()`);
    * 物理 HTTP 尝试**从不**用于准入(`HTTP_ATTEMPT_CEILING_IS_ADMISSION_INPUT = False`);
      准入只看逻辑调用数与实验单元数。

四个量**不得混用**
------------------
    ① 配置的理论包络          `d2d_plan().provider_http_attempt_ceiling` = 324
    ② 权威逻辑调用数          `d2d_plan().logical_invocation_hard_ceiling` = 324(精确)
    ③ **实际** provider 物理尝试 运行时观测;不可观测记 `UNKNOWN`,**不是** 324
    ④ 独立传输层观测           独立记录通道;不可得为 `None`

①**不得**覆盖③。把理论包络写进"实际尝试"是本项目明确禁止的记账方式。

本模块**不读取任何凭据**,也不持有凭据值。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.llm.identity import EndpointCategory
from app.evaluation.llm.protocol import (
    BudgetBoundedness,
    ResourceBudget,
    ScopedEnvelope,
)
from app.evaluation.real_provider import ProviderCandidate

# ---------------------------------------------------------------------------
# 冻结的 provider 身份(控制器决定)
# ---------------------------------------------------------------------------

PILOT_PROVIDER = "deepseek"

#: **请求的**模型标识。报告里凡称 "model" 一律指它。
PILOT_REQUESTED_MODEL = "deepseek-flash"

#: provider 自报的**家族**标识 —— **仅用于溯源对照**,绝不替代请求值。
PILOT_PROVIDER_FAMILY = "DeepSeek-V4.1-Flash"

PILOT_ENDPOINT_CATEGORY = EndpointCategory.OPENAI_COMPATIBLE
PILOT_BASE_URL = "https://api.deepseek.com"

#: 试点候选 id。带 `d2d-pilot-` 语义前缀,与标定的 `d2c-calibration-` 可区分。
PILOT_CANDIDATE_ID = "d2d-pilot-deepseek-flash"

#: 调用方需要自己解析的环境变量名。**本模块不读环境。**
PILOT_API_KEY_ENV = "DEEPSEEK_API_KEY"
PILOT_BASE_URL_ENV = "DEEPSEEK_BASE_URL"

#: 控制器明确排除、试点**不得使用**的模型标识。
EXCLUDED_MODELS: tuple[str, ...] = (
    "deepseek-chat",
    "deepseek-v4-flash",
    "deepseek-reasoner",
)

# ---------------------------------------------------------------------------
# 冻结的请求形状
# ---------------------------------------------------------------------------

#: 显式设定,不依赖 SDK 默认值 —— 便于溯源。
PILOT_STREAMING = False

#: **NOT_SET**。类型标注刻意写成 `None` 而不是 `float | None`:
#: 这个常量只有一个合法取值。
PILOT_TEMPERATURE: None = None

PILOT_TIMEOUT_SECONDS = 60.0

#: SDK 层重试。0 = 每次逻辑调用恰好 1 次物理 HTTP 尝试。
#:
#: 为什么必须是显式的:SDK 重试对 `BudgetGovernor` **不可见** ——
#: 默认值 2 会让一次"逻辑调用"最多产生 3 次物理请求,而计数器只记 1。
PILOT_MAX_RETRIES = 0

#: 工装层重试。恒为 0 —— 已完成的坏结果绝不重跑。
PILOT_HARNESS_RETRY = 0

#: **DECLARED** 输出上限。它表达的是**实验意图**,不是 provider 的实际执行。
PILOT_OUTPUT_TOKEN_CAP = 1024

#: 关闭思考模式的 provider 请求体片段。以**数据**形式提供,通用路径不解释它。
PILOT_THINKING_DISABLED_BODY: dict[str, Any] = {
    "thinking": {"type": "disabled"},
}

# ---------------------------------------------------------------------------
# 物理 HTTP 上界的**分类**(不用于准入)
# ---------------------------------------------------------------------------

#: **历史溯源**,不删除:在 `SDK max_retries = 2` 假设下的理论物理 HTTP 上界。
#: 它就是 `pilot.pilot_plan().provider_http_attempt_ceiling`(见该模块的推导),
#: 在 D-2d 的 provider 配置冻结为 `sdk_max_retries = 0` 之后**不再是执行硬预算**。
HISTORICAL_HTTP_ATTEMPT_CEILING_UNDER_SDK_RETRIES_2 = 972

#: 物理 HTTP 尝试**从不**进入准入判定。准入只看逻辑调用数与实验单元数。
#: 把它写成常量而不是只写在文档里,是为了让它可被机械断言。
HTTP_ATTEMPT_CEILING_IS_ADMISSION_INPUT = False


def pilot_logical_invocation_hard_ceiling() -> int:
    """逻辑调用硬上界。**从 `pilot_plan()` 取**,不另写一份常量。

    惰性 import:`pilot_config` 被导入时不拉入整条图/评测依赖链。
    """
    from app.evaluation.llm.pilot import pilot_plan  # noqa: PLC0415 - 刻意的惰性 import

    return pilot_plan().logical_invocation_hard_ceiling


def pilot_http_attempt_ceiling() -> int:
    """D-2d 自己的**配置包络** = 逻辑调用硬上界 × (1 + `sdk_max_retries`)。

    与 `pilot_plan()` 的**默认**历史值 972 **不同** —— 后者假设 `max_retries = 2`。
    这是**配置的声明**,不是观测值;真实观测永远记 `provider_http_attempts`
    (不可观测时为 `UNKNOWN`),两者**不得混用**,且**都不进入准入**。
    """
    return pilot_logical_invocation_hard_ceiling() * (PILOT_MAX_RETRIES + 1)


def d2d_plan():
    """D-2d 的冻结计划 —— 与默认计划**只差一个假设**(`sdk_max_retries`)。

    默认计划编码的是**观测到的** SDK 默认重试(⇒ 972);D-2d 冻结的是
    `PILOT_MAX_RETRIES = 0`(⇒ 324)。把它做成一个具名入口,是为了让
    "D-2d 用的是哪一个包络"在调用点就可见,而不是靠读者去追默认值。
    """
    from app.evaluation.llm.pilot import pilot_plan  # noqa: PLC0415 - 刻意的惰性 import

    return pilot_plan(sdk_max_retries=PILOT_MAX_RETRIES)


# ---------------------------------------------------------------------------
# 资源预算声明(**类型化**,D-2d 单一事实来源)
# ---------------------------------------------------------------------------

#: token 分量包络的单位。
TOKEN_ENVELOPE_UNIT = "tokens"


def token_budget() -> ResourceBudget:
    """D-2d 的 **token** 资源声明。

    总量**没有**协议级数值上界:输入侧 token 数由系统提示词 + 工具返回 +
    图迭代次数共同决定,三者都不是协议级有界的。

    唯一严格的数值包络是**输出分量**:
    `logical_invocation_hard_ceiling × PILOT_OUTPUT_TOKEN_CAP`。
    它是 **OUTPUT COMPONENT ENVELOPE, NOT a total-token budget** ——
    本模块**不**把它写成 total-token budget,也**不**把它写进执行准入。
    """
    ceiling = pilot_logical_invocation_hard_ceiling()
    envelope_value = ceiling * PILOT_OUTPUT_TOKEN_CAP
    return ResourceBudget(
        boundedness=BudgetBoundedness.NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL,
        reason=(
            "total provider tokens 未被数值上界约束:输入侧 token 数没有严格的"
            "协议级数值上界(系统提示词 + 工具返回 + 图迭代次数共同决定,"
            "三者都不是协议级有界的)⇒ total provider tokens "
            "NOT numerically bounded by protocol。"
        ),
        envelopes=(
            ScopedEnvelope(
                scope="output",
                value=envelope_value,
                unit=TOKEN_ENVELOPE_UNIT,
                provenance=(
                    "OUTPUT COMPONENT ENVELOPE, NOT a total-token budget —— "
                    f"logical_invocation_hard_ceiling({ceiling}) × "
                    f"PILOT_OUTPUT_TOKEN_CAP({PILOT_OUTPUT_TOKEN_CAP}) = "
                    f"{envelope_value} tokens。它**只**覆盖输出分量。"
                ),
            ),
        ),
    )


def cost_budget() -> ResourceBudget:
    """D-2d 的 **cost** 资源声明。

    **已决定**(不是 `UNRESOLVED`):monetary cost **没有**数值上界,因为本仓库
    没有任何权威定价溯源。已确认推不出上界 ⇒ 该状态**允许**出现在冻结清单里。

    ⚠️ 它的含义是「cost 未被数值约束」。**禁止**渲染成
    「cost budget satisfied」「cost controlled」「cost ≤ X」「零成本」
    或「unknown == zero」。
    """
    return ResourceBudget(
        boundedness=BudgetBoundedness.NO_NUMERIC_BOUND_NO_PROVENANCE,
        reason=(
            "NO_AUTHORITATIVE_PRICING_PROVENANCE:本仓库没有价目表、没有账单、"
            "没有 provider 定价快照 ⇒ monetary cost is NOT numerically bounded。"
            "该状态是**已决定**的(不是 UNRESOLVED):我们**不发明**一个"
            "无法证明的上界,也**不**把「推不出」记成「等于 0」。"
        ),
    )


def d2d_resource_budgets() -> dict[str, ResourceBudget]:
    """交给 `assert_frozen_pilot_manifest(..., expected_budgets=...)` 的映射。"""
    return {"token_budget": token_budget(), "cost_budget": cost_budget()}


@dataclass(frozen=True)
class PilotProviderConfig:
    """D-2d 试点的**唯一** provider 配置对象。

    请求 kwargs(`model_kwargs()`)与记录值(`recorded_temperature()`)都从
    同一个实例派生 —— 这是"请求溯源与记录溯源不得分叉"的机械保证。
    """

    provider: str = PILOT_PROVIDER
    requested_model: str = PILOT_REQUESTED_MODEL
    provider_family: str = PILOT_PROVIDER_FAMILY
    endpoint_category: EndpointCategory = PILOT_ENDPOINT_CATEGORY
    base_url: str = PILOT_BASE_URL
    candidate_id: str = PILOT_CANDIDATE_ID

    streaming: bool = PILOT_STREAMING
    #: `None` = NOT_SET。刻意不用 `float | None`,因为本类只有一种合法取值。
    temperature: None = PILOT_TEMPERATURE
    timeout_seconds: float = PILOT_TIMEOUT_SECONDS
    max_retries: int = PILOT_MAX_RETRIES
    output_token_cap: int = PILOT_OUTPUT_TOKEN_CAP

    _thinking_body: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(PILOT_THINKING_DISABLED_BODY),
        repr=False,
        compare=True,
    )

    def __post_init__(self) -> None:
        if not self.requested_model.strip():
            raise ValueError("requested_model 不得为空")
        if self.requested_model in EXCLUDED_MODELS:
            raise ValueError(
                f"模型 {self.requested_model!r} 已被控制器排除,试点不得使用;"
                f"排除清单:{list(EXCLUDED_MODELS)}"
            )
        if self.provider_family == self.requested_model:
            raise ValueError(
                "provider_family 与 requested_model 相同 —— 家族名只用于溯源对照,"
                "不得当作请求值"
            )
        if self.temperature is not None:
            raise ValueError(
                "D-2d 冻结的是 temperature = NOT_SET(None)。"
                f"收到 {self.temperature!r} —— 把 NOT_SET 记成具体数值是伪造。"
            )
        if self.max_retries != PILOT_MAX_RETRIES:
            raise ValueError(f"max_retries 冻结为 {PILOT_MAX_RETRIES}")
        if self.streaming is not PILOT_STREAMING:
            raise ValueError(f"streaming 冻结为 {PILOT_STREAMING}")
        if not self._thinking_body:
            raise ValueError("思考模式必须被显式关闭")

    # ---- 派生:请求侧 ----

    def thinking_body(self) -> dict[str, Any]:
        """关闭思考模式的请求体片段(**深拷贝** —— 调用方改动不影响本对象)。"""
        return copy.deepcopy(self._thinking_body)

    def provider_candidate(self) -> ProviderCandidate:
        """本试点对应的 `ProviderCandidate`。

        **不修改** `real_provider.CANDIDATES` —— 那是 D-2b 的候选表,其中
        `CANDIDATE_B` 的模型 `deepseek-chat` 已被控制器排除。
        """
        return ProviderCandidate(
            candidate_id=self.candidate_id,
            provider=self.provider,
            model=self.requested_model,
            endpoint_category=self.endpoint_category,
            api_key_env=PILOT_API_KEY_ENV,
            base_url_env=PILOT_BASE_URL_ENV,
            default_base_url=self.base_url,
            notes=(
                "D-2d 试点候选。请求模型固定为 "
                f"{self.requested_model!r};家族标识 {self.provider_family!r} "
                "只用于溯源对照。"
            ),
        )

    def model_kwargs(self, *, output_token_cap: int | None = None) -> dict[str, Any]:
        """交给 `build_chat_model` 的 kwargs(**不含凭据**)。

        上限**不通过** `max_tokens` 字段传递:本地 LangChain 会把该字段重写成
        `max_completion_tokens`,而我们要发的是 provider 文档里的那个字段名。
        因此它走 `extra_body`。
        """
        cap = self.output_token_cap if output_token_cap is None else output_token_cap
        body = self.thinking_body()
        body["max_tokens"] = cap
        return {
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "max_retries": self.max_retries,
            "streaming": self.streaming,
            "extra_body": body,
        }

    # ---- 派生:记录侧 ----

    def recorded_temperature(self) -> None:
        """记录进 `ModelIdentity.temperature` 的值。

        恒为 `None`(= NOT_SET),与 `model_kwargs()["temperature"]` **同源**。
        没有任何分支会把它变成 `0.0`。
        """
        return self.temperature

    # ---- 溯源 ----

    def request_shape(self) -> dict[str, Any]:
        """请求的**凭据无关**形状描述。可用于摘要与人工核对。"""
        kwargs = self.model_kwargs()
        return {
            "provider": self.provider,
            "requested_model": self.requested_model,
            "base_url": self.base_url,
            "endpoint_category": self.endpoint_category.value,
            "streaming": kwargs["streaming"],
            "temperature_present": kwargs["temperature"] is not None,
            "temperature": kwargs["temperature"],
            "timeout_seconds": kwargs["timeout"],
            "max_retries": kwargs["max_retries"],
            "extra_body": kwargs["extra_body"],
            "api_key_present": False,
        }

    def request_shape_digest(self) -> str:
        """请求形状的稳定摘要。

        **刻意只覆盖凭据无关的字段** —— 摘要不得依赖于凭据材料,
        否则摘要本身就成了凭据的旁路。
        """
        canonical = json.dumps(
            self.request_shape(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_pilot_config() -> PilotProviderConfig:
    """冻结的 D-2d provider 配置。"""
    return PilotProviderConfig()


# ---------------------------------------------------------------------------
# 清单冻结:控制器已授权的三个身份字段
# ---------------------------------------------------------------------------


def pilot_identity_fields() -> dict[str, str]:
    """交给 `build_candidate_manifest` 的三个**已授权**身份字段。

    `token_budget` / `cost_budget` **不在**这里 —— 它们是**类型化**声明,
    由 `d2d_resource_budgets()` 单独提供(`ResourceBudget` 不是 `str`)。
    """
    config = default_pilot_config()
    return {
        "provider": config.provider,
        "model": config.requested_model,
        "endpoint_category": config.endpoint_category.value,
    }
