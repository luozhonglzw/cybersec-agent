"""冻结的失败 / 重试分类(**`HARNESS_LEVEL_RETRY = 0`**)。

本模块只做一件事:**把"出了什么错"映射到一个封闭类别**,并声明这个类别
对统计口径意味着什么。它**不重跑任何东西**。

为什么首个试点不允许 harness 层重试
-----------------------------------
重试看起来无害("网络抖一下就重试嘛"),但它会**系统性地**改变分母:
失败与重试的耦合会让"最终进入统计的运行"偏向那些**provider 恰好不抖**的
时段;更糟的是,一旦重试逻辑里掺入任何与结果相关的判断(哪怕只是
"这个答案看起来不对,再试一次"),实验就退化成了"挑好看的样本"。

因此首个试点的规则是:

    一个 provider / infra 失败 = **一个被记录的观测结果 + 一份覆盖率损失**,
    而不是一个重跑候选。

失败**降低可评测样本量** —— 报告必须如实呈现,不得静默补跑。

SDK 自身的重试(`root_client.max_retries = 2`)仍然存在,但它是
**provider 默认行为**,不是我们的实验设计:我们只能**记录**它
(可观测时),不能假装它不存在,也不能假装它是我们设定的。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FailureClass(str, Enum):
    """封闭的失败类别词表。"""

    MODEL_ANSWER = "MODEL_ANSWER"
    MODEL_REFUSAL = "MODEL_REFUSAL"
    MODEL_TOOL_ARG_INVALID = "MODEL_TOOL_ARG_INVALID"
    MODEL_NO_TOOL_CALL_WHEN_REQUIRED = "MODEL_NO_TOOL_CALL_WHEN_REQUIRED"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    PROVIDER_SERVER_ERROR = "PROVIDER_SERVER_ERROR"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_CONNECTION_ERROR = "PROVIDER_CONNECTION_ERROR"
    PROVIDER_AUTH_ERROR = "PROVIDER_AUTH_ERROR"
    PROVIDER_INVALID_REQUEST = "PROVIDER_INVALID_REQUEST"
    PROVIDER_CONTENT_FILTER = "PROVIDER_CONTENT_FILTER"
    HARNESS_ERROR = "HARNESS_ERROR"
    PROCESS_CRASH = "PROCESS_CRASH"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class FailureRule:
    """一个类别的统计语义。"""

    classification: str
    count_as_model_result: bool
    count_as_provider_failure: bool
    count_as_infra_failure: bool
    experimental_retry_allowed: bool
    record_as_finding: bool = False
    aborts_pilot: bool = False
    note: str = ""


#: **冻结的失败分类表。** 每一项的 `experimental_retry_allowed` 都是 False ——
#: 这不是遗漏,是首个试点的设计决定。
FAILURE_TAXONOMY: dict[FailureClass, FailureRule] = {
    FailureClass.MODEL_ANSWER: FailureRule(
        classification="COUNT_AS_MODEL_RESULT",
        count_as_model_result=True, count_as_provider_failure=False,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        note="正常终态。低分也是模型结果 —— 绝不因为答案差而重跑。",
    ),
    FailureClass.MODEL_REFUSAL: FailureRule(
        classification="COUNT_AS_MODEL_RESULT",
        count_as_model_result=True, count_as_provider_failure=False,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        note="模型主动拒答是**能力观测**,不是基础设施故障。",
    ),
    FailureClass.MODEL_TOOL_ARG_INVALID: FailureRule(
        classification="COUNT_AS_MODEL_RESULT",
        count_as_model_result=True, count_as_provider_failure=False,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        note="工具参数校验失败由工具返回错误,图继续 —— 这是模型能力结果。",
    ),
    FailureClass.MODEL_NO_TOOL_CALL_WHEN_REQUIRED: FailureRule(
        classification="COUNT_AS_MODEL_RESULT",
        count_as_model_result=True, count_as_provider_failure=False,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        note="该调工具却没调 —— 能力结果。",
    ),
    FailureClass.PROVIDER_RATE_LIMIT: FailureRule(
        classification="COUNT_AS_PROVIDER_FAILURE",
        count_as_model_result=False, count_as_provider_failure=True,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        note="SDK 可能内部重试(provider_default);harness **不**重跑。",
    ),
    FailureClass.PROVIDER_SERVER_ERROR: FailureRule(
        classification="COUNT_AS_PROVIDER_FAILURE",
        count_as_model_result=False, count_as_provider_failure=True,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        note="5xx。SDK 可能内部重试;harness **不**重跑。",
    ),
    FailureClass.PROVIDER_TIMEOUT: FailureRule(
        classification="COUNT_AS_PROVIDER_FAILURE",
        count_as_model_result=False, count_as_provider_failure=True,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        note="读超时。SDK 可能内部重试;harness **不**重跑。",
    ),
    FailureClass.PROVIDER_CONNECTION_ERROR: FailureRule(
        classification="COUNT_AS_INFRA_FAILURE",
        count_as_model_result=False, count_as_provider_failure=False,
        count_as_infra_failure=True, experimental_retry_allowed=False,
        note="DNS / TCP / TLS。SDK 可能内部重试;harness **不**重跑。",
    ),
    FailureClass.PROVIDER_AUTH_ERROR: FailureRule(
        classification="COUNT_AS_INFRA_FAILURE",
        count_as_model_result=False, count_as_provider_failure=False,
        count_as_infra_failure=True, experimental_retry_allowed=False,
        aborts_pilot=True,
        note="401/403 是**配置问题**,不是瞬时故障 —— 继续跑只会浪费额度。",
    ),
    FailureClass.PROVIDER_INVALID_REQUEST: FailureRule(
        classification="COUNT_AS_INFRA_FAILURE",
        count_as_model_result=False, count_as_provider_failure=False,
        count_as_infra_failure=True, experimental_retry_allowed=False,
        aborts_pilot=True,
        note="400 是**工装 bug**,继续跑会批量产生同样的错误。",
    ),
    FailureClass.PROVIDER_CONTENT_FILTER: FailureRule(
        classification="COUNT_AS_PROVIDER_FAILURE",
        count_as_model_result=False, count_as_provider_failure=True,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        record_as_finding=True,
        note="合成安全内容触发厂商策略 —— 这是**发现**,必须报告,不能重试掉。",
    ),
    FailureClass.HARNESS_ERROR: FailureRule(
        classification="COUNT_AS_INFRA_FAILURE",
        count_as_model_result=False, count_as_provider_failure=False,
        count_as_infra_failure=True, experimental_retry_allowed=False,
        note="评测代码自身抛错。系统性出现即 ABORT。",
    ),
    FailureClass.PROCESS_CRASH: FailureRule(
        classification="COUNT_AS_INFRA_FAILURE",
        count_as_model_result=False, count_as_provider_failure=False,
        count_as_infra_failure=True, experimental_retry_allowed=False,
        note="进程崩溃。单元**不**自动重跑;只允许「无完整记录才续跑」(见 raw.py)。",
    ),
    FailureClass.BUDGET_EXHAUSTED: FailureRule(
        classification="ABORT",
        count_as_model_result=False, count_as_provider_failure=False,
        count_as_infra_failure=False, experimental_retry_allowed=False,
        aborts_pilot=True,
        note="预算耗尽不是模型结果,也不是故障 —— 是实验停止条件。",
    ),
}

#: **首个试点固定为 0。** 不自动重跑任何实验单元。
HARNESS_LEVEL_RETRY = 0

#: 连续多少个同类 provider 失败触发暂停。
CONSECUTIVE_PROVIDER_FAILURE_LIMIT = 3


def rule_for(failure_class: FailureClass) -> FailureRule:
    return FAILURE_TAXONOMY[failure_class]


def is_model_result(failure_class: FailureClass) -> bool:
    return FAILURE_TAXONOMY[failure_class].count_as_model_result


def is_provider_failure(failure_class: FailureClass) -> bool:
    return FAILURE_TAXONOMY[failure_class].count_as_provider_failure


def is_infra_failure(failure_class: FailureClass) -> bool:
    return FAILURE_TAXONOMY[failure_class].count_as_infra_failure


def _http_status(exc: BaseException) -> int | None:
    """从异常里掏 HTTP 状态码(不依赖任何 SDK 的类层次)。

    刻意用**鸭子类型**而不是 `isinstance(exc, openai.APIStatusError)`:
    SDK 的异常层次会随版本变化,把分类逻辑绑死在某一版上是脆弱的设计。
    状态码是协议层的事实,比类名稳定得多。
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def classify_http_status(status: int) -> FailureClass:
    if status == 429:
        return FailureClass.PROVIDER_RATE_LIMIT
    if status in (401, 403):
        return FailureClass.PROVIDER_AUTH_ERROR
    if status == 400 or status == 422:
        return FailureClass.PROVIDER_INVALID_REQUEST
    if 500 <= status < 600:
        return FailureClass.PROVIDER_SERVER_ERROR
    return FailureClass.HARNESS_ERROR


def classify_error_name(name: str | None) -> FailureClass:
    """异常**类名** → 失败类别。

    为什么需要它:适配器在 `llm_failed` 分支里只留下 `type(exc).__name__`
    (异常对象本身没有被保留),离线阶段只能按名字分类。

    **这是离线阶段的限制,不是通用做法。** 类名会随 SDK 版本漂移;
    D-2b 拿到真实 provider 异常后应当走 `classify_exception` ——
    它优先读协议层的 `status_code`,比类名稳定得多。
    """
    if not name:
        return FailureClass.HARNESS_ERROR
    haystack = name.lower()
    if "timeout" in haystack or "timedout" in haystack:
        return FailureClass.PROVIDER_TIMEOUT
    if "connection" in haystack or "connecterror" in haystack:
        return FailureClass.PROVIDER_CONNECTION_ERROR
    if "contentfilter" in haystack or "content_filter" in haystack:
        return FailureClass.PROVIDER_CONTENT_FILTER
    if "ratelimit" in haystack or "rate_limit" in haystack:
        return FailureClass.PROVIDER_RATE_LIMIT
    if "authentication" in haystack or "permission" in haystack:
        return FailureClass.PROVIDER_AUTH_ERROR
    if "badrequest" in haystack or "unprocessable" in haystack:
        return FailureClass.PROVIDER_INVALID_REQUEST
    return FailureClass.HARNESS_ERROR


def classify_exception(exc: BaseException) -> FailureClass:
    """异常 → 失败类别。

    顺序很重要:**先看状态码**(协议层事实),再看类名(实现细节)。
    两者都没有的,归 `HARNESS_ERROR` —— 归类为"工装问题"而不是
    "模型问题"是保守的方向:它会让调查先看工装,而不是先怀疑模型。
    """
    status = _http_status(exc)
    if status is not None:
        return classify_http_status(status)
    return classify_error_name(type(exc).__name__)


class FailureRecord(BaseModel):
    """落进原始记录的一次失败分类。"""

    failure_class: FailureClass
    classification: str
    count_as_model_result: bool
    count_as_provider_failure: bool
    count_as_infra_failure: bool
    experimental_retry_allowed: bool = Field(
        default=False, description="首个试点恒为 False —— 见 FAILURE_TAXONOMY",
    )
    record_as_finding: bool = False
    aborts_pilot: bool = False
    error_type: str | None = None
    note: str = ""

    @classmethod
    def from_class(
        cls, failure_class: FailureClass, *, error_type: str | None = None
    ) -> "FailureRecord":
        rule = FAILURE_TAXONOMY[failure_class]
        return cls(
            failure_class=failure_class,
            classification=rule.classification,
            count_as_model_result=rule.count_as_model_result,
            count_as_provider_failure=rule.count_as_provider_failure,
            count_as_infra_failure=rule.count_as_infra_failure,
            experimental_retry_allowed=rule.experimental_retry_allowed,
            record_as_finding=rule.record_as_finding,
            aborts_pilot=rule.aborts_pilot,
            error_type=error_type,
            note=rule.note,
        )


def failure_summary(failure_classes: list[FailureClass]) -> dict[str, Any]:
    """把一串类别汇成计数。`model_result` 单独列出 —— 它不是"失败"。"""
    counts: dict[str, int] = {member.value: 0 for member in FailureClass}
    for item in failure_classes:
        counts[item.value] += 1
    return {
        "counts": counts,
        "model_result": sum(counts[c.value] for c in FailureClass if is_model_result(c)),
        "provider_failure": sum(counts[c.value] for c in FailureClass if is_provider_failure(c)),
        "infra_failure": sum(counts[c.value] for c in FailureClass if is_infra_failure(c)),
        "retry_allowed": sum(
            counts[c.value] for c in FailureClass
            if FAILURE_TAXONOMY[c].experimental_retry_allowed
        ),
    }
