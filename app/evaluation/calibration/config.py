"""D-2c 标定配置:**请求溯源与记录溯源的单一事实来源**。

为什么必须是"单一事实来源"
--------------------------
标定要回答的一个问题是"provider 到底接受不接受我们发的这个参数"。如果
**发出去的**参数与**记录下来的**参数来自两处代码,那么一次标定最可能的
产出不是答案,而是一个**自洽的假象**:记录说发了 `temperature=0`,
实际请求里根本没有这个字段,而两边看起来都对。

所以这里只留一个对象:`CalibrationConfig`。请求 kwargs 与记录值都从它派生,
两者在结构上无法分叉。

`temperature = NOT_SET` 的含义
-----------------------------
`NOT_SET` 用 `None` 表示,含义是**请求体里没有 `temperature` 字段**。
它与 `temperature=0.0` 是两件不同的事:

    NOT_SET    "这次请求不依赖温度"        → 记录 `None`
    0.0        "显式要求温度为 0"          → 记录 `0.0`

把前者记成后者是**伪造**:它会让读者以为"我们控制了采样",而实际上没有。
因此本模块的 `recorded_temperature()` 恒为 `None`,且没有提供任何把它
变成 `0.0` 的入口。

思考模式
--------
`thinking.type = "disabled"` 由 `extra_body` 以**数据**形式提供。
通用评测包(`app/evaluation/llm/`)**不认识**这个键 —— 它只是把 dict 透传。
这样"关闭思考模式"就不需要往冻结的通用路径里塞模型专属分支。

本模块**不读取任何凭据**,也不持有凭据值。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.llm.budget import PilotBudget
from app.evaluation.llm.identity import EndpointCategory
from app.evaluation.real_provider import ProviderCandidate

# ---------------------------------------------------------------------------
# 冻结的 provider 身份
# ---------------------------------------------------------------------------

CALIBRATION_PROVIDER = "deepseek"

#: **请求的**模型标识。报告里凡称 "model" 一律指它。
CALIBRATION_REQUESTED_MODEL = "deepseek-flash"

#: provider 自报的**家族**标识 —— **仅用于溯源对照**。
#:
#: 硬约束:它**绝不**可以替代 `CALIBRATION_REQUESTED_MODEL` 出现在请求里。
#: 它的存在恰恰是为了让"请求了什么"与"provider 说它是什么"可被区分。
CALIBRATION_PROVIDER_FAMILY = "DeepSeek-V4.1-Flash"

CALIBRATION_ENDPOINT_CATEGORY = EndpointCategory.OPENAI_COMPATIBLE
CALIBRATION_BASE_URL = "https://api.deepseek.com"

#: 标定用的候选 id。**刻意不复用** `CANDIDATE_A` / `CANDIDATE_B` ——
#: 那两个是 D-2b 的候选路径,标定不是它们的实例化。
CALIBRATION_CANDIDATE_ID = "d2c-calibration-deepseek-flash"

#: 调用方需要自己解析的环境变量名。**本模块不读环境。**
CALIBRATION_API_KEY_ENV = "DEEPSEEK_API_KEY"
CALIBRATION_BASE_URL_ENV = "DEEPSEEK_BASE_URL"

#: 控制器明确排除、本标定**不得使用**的模型标识。
EXCLUDED_MODELS: tuple[str, ...] = (
    "deepseek-chat",
    "deepseek-v4-flash",
    "deepseek-reasoner",
)

# ---------------------------------------------------------------------------
# 冻结的请求形状
# ---------------------------------------------------------------------------

#: 显式设定,不依赖 SDK 默认值 —— 便于溯源。
CALIBRATION_STREAMING = False

#: **NOT_SET**。见模块 docstring。类型标注刻意写成 `None` 而不是 `float | None`:
#: 这个常量只有一个合法取值。
CALIBRATION_TEMPERATURE: None = None

CALIBRATION_TIMEOUT_SECONDS = 60.0

#: SDK 层重试。0 = 每次逻辑调用恰好 1 次物理 HTTP 尝试。
#:
#: 为什么必须是显式的:SDK 重试对 `BudgetGovernor` **不可见** ——
#: 默认值 2 会让一次"逻辑调用"最多产生 3 次物理请求,而计数器只记 1。
CALIBRATION_MAX_RETRIES = 0

#: 工装层重试。恒为 0 —— 已完成的坏结果绝不重跑。
CALIBRATION_HARNESS_RETRY = 0

#: **DECLARED** 输出上限。它表达的是**实验意图**,不是 provider 的实际执行。
#: 在 C0b 之前,任何把它称为"已生效"的措辞都是伪造。
CALIBRATION_OUTPUT_TOKEN_CAP = 1024

#: 关闭思考模式的 provider 请求体片段。以**数据**形式提供,通用路径不解释它。
CALIBRATION_THINKING_DISABLED_BODY: dict[str, Any] = {
    "thinking": {"type": "disabled"},
}

# ---------------------------------------------------------------------------
# C0b 探针参数
# ---------------------------------------------------------------------------

#: C0b **专用**的临时小上限。**不进入** C1–C4。
CALIBRATION_C0B_TEMPORARY_CAP = 16

#: C0b 要求模型产出的长度。必须**远大于** `CALIBRATION_C0B_TEMPORARY_CAP`,
#: 否则"模型自然停下"与"上限生效"无法区分。
CALIBRATION_C0B_DEMANDED_OUTPUT_TOKENS = 2000

#: 把"要求产出的 token 数"折成"要求产出的字符数"的**近似**系数。
#:
#: 为什么需要它:token 用量是**首选**证据,但它可能缺失。缺失时唯一还能
#: 拿到的证据是响应字符长度 —— 而字符长度只有在有一个"要求长度"作对照时
#: 才有意义。这个系数是**判定参数**,不是实测比率,因此它只被用来构造
#: 对照量,绝不参与任何"provider 报了多少 token"的记录。
CALIBRATION_C0B_CHARS_PER_TOKEN_APPROX = 4

# ---------------------------------------------------------------------------
# 预算(全部 DECLARED)
# ---------------------------------------------------------------------------

NOMINAL_STAGE_COUNT = 6
NOMINAL_EXPERIMENTAL_RUNS = 6
NOMINAL_LOGICAL_INVOCATIONS = 13

HARD_EXPERIMENTAL_RUN_CEILING = 8
HARD_LOGICAL_INVOCATION_CEILING = 20

#: 硬上界与名义值之差。**它不是重跑额度。**
BUDGET_HEADROOM_RUNS = HARD_EXPERIMENTAL_RUN_CEILING - NOMINAL_EXPERIMENTAL_RUNS
BUDGET_HEADROOM_INVOCATIONS = (
    HARD_LOGICAL_INVOCATION_CEILING - NOMINAL_LOGICAL_INVOCATIONS
)

#: 六个阶段的名字。它们**就是**本次运行的"基线标签" —— 不是三个生产基线。
CALIBRATION_STAGE_LABELS: tuple[str, ...] = ("C0a", "C0b", "C1", "C2", "C3", "C4")

#: 单个阶段的**硬**逻辑调用上限。C3/C4 由控制器单独指定(3 / 5)。
#:
#: 之和恰好等于 `NOMINAL_LOGICAL_INVOCATIONS` —— 若两者不等,说明"每阶段上限"
#: 与"名义总量"互相矛盾,而两边看起来都对。
STAGE_LOGICAL_INVOCATION_CEILINGS: dict[str, int] = {
    "C0a": 1,
    "C0b": 1,
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 5,
}

#: 单个阶段的**结构性下界**。它不是估计值:每个阶段至少要发生的调用数
#: 由阶段契约决定 —— 握手 1 次、上限探针 1 次、工具形状 1 次、
#: 图路径 2 次(工具轮 + 终答)、评测单元 2 次(工具轮 + 终答)。
#:
#: **C2 的下界是 0,这是刻意的。** 本闸门的 C2 是对一段**已发生**的消息序列
#: 做机械校验(见 `stages.run_c2`),它自己不发起任何调用 —— 与 C2 契约里
#: "无真实 provider 流量"一致。它的上限仍是 2:若将来把 C2 改成实盘往返,
#: 最多允许 2 次。下界为 0 的阶段**不做单元准入检查**(准入检查的下界是 1)。
STAGE_LOGICAL_INVOCATION_FLOORS: dict[str, int] = {
    "C0a": 1,
    "C0b": 1,
    "C1": 1,
    "C2": 0,
    "C3": 2,
    "C4": 2,
}

#: 各阶段硬上限之和。
STAGE_CEILING_SUM = sum(STAGE_LOGICAL_INVOCATION_CEILINGS.values())

#: 各阶段下界之和 —— 本次运行**至少**要发生的逻辑调用数。
STAGE_FLOOR_SUM = sum(STAGE_LOGICAL_INVOCATION_FLOORS.values())

if STAGE_CEILING_SUM > HARD_LOGICAL_INVOCATION_CEILING:  # pragma: no cover - 配置自检
    raise ValueError(
        f"各阶段逻辑调用上限之和 {STAGE_CEILING_SUM} 超过实验级硬上界 "
        f"{HARD_LOGICAL_INVOCATION_CEILING} —— 配置自相矛盾"
    )
if STAGE_CEILING_SUM != NOMINAL_LOGICAL_INVOCATIONS:  # pragma: no cover - 配置自检
    raise ValueError(
        f"各阶段上限之和 {STAGE_CEILING_SUM} 与名义逻辑调用数 "
        f"{NOMINAL_LOGICAL_INVOCATIONS} 不一致 —— 两者必须同源"
    )
if set(STAGE_LOGICAL_INVOCATION_CEILINGS) != set(CALIBRATION_STAGE_LABELS):  # pragma: no cover
    raise ValueError("阶段上限表与阶段标签表不同源")
if set(STAGE_LOGICAL_INVOCATION_FLOORS) != set(CALIBRATION_STAGE_LABELS):  # pragma: no cover
    raise ValueError("阶段下界表与阶段标签表不同源")
for _stage in CALIBRATION_STAGE_LABELS:  # pragma: no cover - 配置自检
    if STAGE_LOGICAL_INVOCATION_FLOORS[_stage] > STAGE_LOGICAL_INVOCATION_CEILINGS[_stage]:
        raise ValueError(f"阶段 {_stage} 的下界超过其上限 —— 配置自相矛盾")

#: 标定的**理论**物理 HTTP 尝试上界。
#:
#: 它由 `max_retries = 0` 推出:一次逻辑调用恰好一次物理尝试。这个数字是
#: **推导出来的声明**,不是观测值 —— 真实观测永远记 `provider_http_attempts`
#: (不可观测时为 `UNKNOWN`),两者**不得混用**。
CALIBRATION_HTTP_ATTEMPT_CEILING = (
    HARD_LOGICAL_INVOCATION_CEILING * (CALIBRATION_MAX_RETRIES + 1)
)


@dataclass(frozen=True)
class CalibrationConfig:
    """标定的**唯一**配置对象。

    请求 kwargs(`model_kwargs()`)与记录值(`recorded_temperature()`)都从
    同一个实例派生 —— 这是"请求溯源与记录溯源不得分叉"的机械保证。
    """

    provider: str = CALIBRATION_PROVIDER
    requested_model: str = CALIBRATION_REQUESTED_MODEL
    provider_family: str = CALIBRATION_PROVIDER_FAMILY
    endpoint_category: EndpointCategory = CALIBRATION_ENDPOINT_CATEGORY
    base_url: str = CALIBRATION_BASE_URL
    candidate_id: str = CALIBRATION_CANDIDATE_ID

    streaming: bool = CALIBRATION_STREAMING
    #: `None` = NOT_SET。刻意不用 `float | None`,因为本类只有一种合法取值。
    temperature: None = CALIBRATION_TEMPERATURE
    timeout_seconds: float = CALIBRATION_TIMEOUT_SECONDS
    max_retries: int = CALIBRATION_MAX_RETRIES
    output_token_cap: int = CALIBRATION_OUTPUT_TOKEN_CAP

    _thinking_body: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(CALIBRATION_THINKING_DISABLED_BODY),
        repr=False,
        compare=True,
    )

    def __post_init__(self) -> None:
        if not self.requested_model.strip():
            raise ValueError("requested_model 不得为空")
        if self.requested_model in EXCLUDED_MODELS:
            raise ValueError(
                f"模型 {self.requested_model!r} 已被控制器排除,标定不得使用;"
                f"排除清单:{list(EXCLUDED_MODELS)}"
            )
        if self.provider_family == self.requested_model:
            raise ValueError(
                "provider_family 与 requested_model 相同 —— 家族名只用于溯源对照,"
                "不得当作请求值"
            )
        if self.temperature is not None:
            raise ValueError(
                "D-2c 冻结的是 temperature = NOT_SET(None)。"
                f"收到 {self.temperature!r} —— 把 NOT_SET 记成具体数值是伪造。"
            )
        if self.max_retries != CALIBRATION_MAX_RETRIES:
            raise ValueError(f"max_retries 冻结为 {CALIBRATION_MAX_RETRIES}")
        if self.streaming is not CALIBRATION_STREAMING:
            raise ValueError(f"streaming 冻结为 {CALIBRATION_STREAMING}")
        if not self._thinking_body:
            raise ValueError("思考模式必须被显式关闭")

    # ---- 派生:请求侧 ----

    def thinking_body(self) -> dict[str, Any]:
        """关闭思考模式的请求体片段(**深拷贝** —— 调用方改动不影响本对象)。"""
        return copy.deepcopy(self._thinking_body)

    def provider_candidate(self) -> ProviderCandidate:
        """本标定对应的 `ProviderCandidate`。

        **不修改** `real_provider.CANDIDATES` —— 那是 D-2b 的冻结候选表。
        标定自建候选,`build_chat_model` 本来就接受调用方传入的候选。
        """
        return ProviderCandidate(
            candidate_id=self.candidate_id,
            provider=self.provider,
            model=self.requested_model,
            endpoint_category=self.endpoint_category,
            api_key_env=CALIBRATION_API_KEY_ENV,
            base_url_env=CALIBRATION_BASE_URL_ENV,
            default_base_url=self.base_url,
            notes=(
                "D-2c 标定候选。请求模型固定为 "
                f"{self.requested_model!r};家族标识 {self.provider_family!r} "
                "只用于溯源对照。"
            ),
        )

    def model_kwargs(self, *, output_token_cap: int | None = None) -> dict[str, Any]:
        """交给 `build_chat_model` 的 kwargs(**不含凭据**)。

        `output_token_cap` 覆写仅由 C0b 使用 —— 它传一个**临时小上限**。
        上限**不通过** `max_tokens` 字段传递:本地 LangChain 会把该字段
        重写成 `max_completion_tokens`(见 D-2c 设计 §2a),而我们要发的是
        provider 文档里的那个字段名。因此它走 `extra_body`。
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


def default_calibration_config() -> CalibrationConfig:
    """冻结的标定配置。"""
    return CalibrationConfig()


def calibration_budget() -> PilotBudget:
    """标定的预算对象。

    为什么复用 `PilotBudget` 而**不**新写一个类
    ------------------------------------------
    `BudgetGovernor` 的记账纪律(唯一写 `logical_llm_invocations` 的
    `reserve()`、`record_unit()` 刻意不接受逻辑调用数、继承量与本次进程量
    分离)是经过 D-2b 加固的。为标定另写一个治理器等于把那套纪律复制一份,
    然后让两份各自漂移。

    因此这里复用同一个类,只是把字段**如实**填成标定的语义:

        baseline_labels     六个**阶段**名(不是三个生产基线)
        task_count          阶段数(每个阶段是一个实验单元)
        treatment_runs      名义实验运行数
        control_runs        0 —— 标定没有对照组,它不测量因果
        logical_invocation_hard_ceiling  实验级硬上界

    `provider_http_attempt_ceiling` 由 `max_retries = 0` 推出,并在
    `provider_http_attempt_ceiling_basis` 里**写明它是推导值而非观测值**。
    """
    return PilotBudget(
        baseline_labels=CALIBRATION_STAGE_LABELS,
        repetition_count=1,
        task_count=NOMINAL_STAGE_COUNT,
        injection_task_count=0,
        treatment_runs=NOMINAL_EXPERIMENTAL_RUNS,
        control_runs=0,
        total_runs=NOMINAL_EXPERIMENTAL_RUNS,
        logical_invocation_floor=STAGE_FLOOR_SUM,
        logical_invocation_hard_ceiling=HARD_LOGICAL_INVOCATION_CEILING,
        provider_http_attempt_ceiling=CALIBRATION_HTTP_ATTEMPT_CEILING,
        provider_http_attempt_ceiling_basis=(
            f"推导上界 = 逻辑调用硬上界 {HARD_LOGICAL_INVOCATION_CEILING} × "
            f"(1 + 标定设定的 max_retries {CALIBRATION_MAX_RETRIES}) = "
            f"{CALIBRATION_HTTP_ATTEMPT_CEILING}。"
            "⚠️ 这是**由重试配置推出的声明**,不是观测值;物理尝试不可观测时"
            "记 UNKNOWN,不得用本上界顶替观测。"
        ),
        # 标定没有"软上限"这一档:六个阶段各自有硬上限,越过即 ABORT。
        soft_call_ceiling=HARD_LOGICAL_INVOCATION_CEILING,
    )
