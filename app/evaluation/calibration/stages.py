"""D-2c 标定的**六个阶段** —— 捕获什么、以及如何判定。

阶段契约(**顺序由 `harness.py` 强制执行**)
------------------------------------------

    C0a  provider 握手        —— 只回答"能不能连上、能不能读懂响应形状"
    C0b  长输出上限探针       —— 只回答"我们请求的输出上限有没有被执行"
    C1   工具调用形状         —— 只回答"模型给的 tool call 形状合不合法"
    C2   工具往返             —— 只回答"tool_call_id 能不能原样往返"
    C3   真实图代码路径       —— 在**真实** LangGraph 路径上跑合成模型
    C4   一个代表性评测单元   —— 走一次完整适配器,合成模型 + 真实工具

一条贯穿全模块的纪律:**不把"声明"说成"观测"**
--------------------------------------------

    请求的模型            ≠  provider 自报的模型
    声明的输出上限        ≠  观测到的输出用量
    阶段调用上限          ≠  实际发生的调用数
    理论 HTTP 尝试上界    ≠  观测到的物理尝试数

因此每个证据字典都**成对**记录这两侧,并且**永不互相顶替**。缺失的一侧
记 `None` / `NOT_AVAILABLE`,不臆造。

另一条纪律:**`finish_reason` 不参与判定**
-----------------------------------------
它只被记录。理由见 `verdict.py`:`stop` 推不出"上限被忽略",`length`
也推不出"上限生效"。判定只依据**长度证据**。

本模块**不 import 任何 provider 客户端**,也不发起任何网络 I/O ——
真实构造在 `app/evaluation/real_provider.py`,由调用方以**注入的调用器**
(`InvokeFn`)形式提供。因此本模块可以被完全离线地执行与测试。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from app.evaluation.calibration import config as cal_config
from app.evaluation.calibration.synthetic import SyntheticProviderModel
from app.evaluation.calibration.verdict import (
    StageVerdict,
    Verdict,
    c0b_verdict,
    failed,
    inconclusive,
    passed,
)
from app.evaluation.llm.budget import UNKNOWN, BudgetExceeded, BudgetGovernor
from app.evaluation.real_provider import (
    provider_reported_model_id_of,
    raw_token_usage_of,
)

# ---------------------------------------------------------------------------
# 阶段
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    """六个标定阶段。`value` 与冻结的阶段标签逐字一致。"""

    C0A = "C0a"
    C0B = "C0b"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"


#: 冻结的执行顺序。**顺序本身是契约的一部分** —— 打乱它等于换了实验。
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.C0A,
    Stage.C0B,
    Stage.C1,
    Stage.C2,
    Stage.C3,
    Stage.C4,
)


def stage_ceiling(stage: Stage) -> int:
    """某阶段的**硬**逻辑调用上限。"""
    return cal_config.STAGE_LOGICAL_INVOCATION_CEILINGS[stage.value]


def stage_floor(stage: Stage) -> int:
    """某阶段的**结构性**下界。"""
    return cal_config.STAGE_LOGICAL_INVOCATION_FLOORS[stage.value]


# ---------------------------------------------------------------------------
# 请求 / 结果 / 依赖
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRequest:
    """一次标定请求(**凭据无关**)。

    `model_kwargs` 一律由 `CalibrationConfig` 派生 —— 这是"请求溯源与记录
    溯源不得分叉"的落点。调用器不得往里面加任何东西。
    """

    stage: Stage
    messages: tuple[Any, ...]
    model_kwargs: Mapping[str, Any]
    output_token_cap: int
    demanded_output_tokens: int | None = None

    def extra_body(self) -> dict[str, Any]:
        body = self.model_kwargs.get("extra_body")
        return dict(body) if isinstance(body, Mapping) else {}


@dataclass(frozen=True)
class InvocationResult:
    """一次逻辑调用的结果 + **传输层证据**。

    `request_payload` 是传输边界上观测到的请求体(已剔除凭据)。
    它为 `None` 表示"传输层没有暴露请求体" —— 此时**上限放置位置无法核验**,
    阶段判定必须如实降级,而不是假定"我们发了"。
    """

    message: Any
    request_payload: Mapping[str, Any] | None = None
    http_attempts: int | None = None


#: 调用器:执行一次逻辑调用。**唯一**与 provider 接触的地方,由调用方注入。
InvokeFn = Callable[[CalibrationRequest], Awaitable[InvocationResult]]


class CapParameterRejected(RuntimeError):
    """provider **显式拒绝**了输出上限参数。"""


@dataclass(frozen=True)
class ToolContract:
    """工具契约:允许的工具名 + 各工具的参数 schema。"""

    allowed_tool_names: tuple[str, ...]
    arg_schemas: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationUnit:
    """C4 的"一个代表性评测单元"。"""

    task: Any
    baseline_label: str
    behavior: str
    dataset_paths: Mapping[str, str]


@dataclass
class StageDeps:
    """注入给阶段运行器的**传输层**。

    全部为 `None` 时,只有不依赖传输的阶段能跑;其余阶段会以
    `StageDependencyMissing` 拒绝 —— 拒绝而不是静默跳过,因为"静默跳过"
    会让一次配置错误的运行看起来像一次全部通过的运行。

    `graph_model` 与 `unit_model` **刻意分开**:它们是两份独立的合成脚本。
    共用一个实例会让先跑的 C3 把脚本耗尽,于是 C4 拿到的是"(脚本已耗尽)"
    —— 一个看起来跑通了、实际上什么都没测的评测单元。
    """

    invoke: InvokeFn | None = None
    tool_contract: ToolContract | None = None
    roundtrip: Sequence[Any] | None = None
    graph_model: SyntheticProviderModel | None = None
    unit_model: SyntheticProviderModel | None = None
    evaluation_unit: EvaluationUnit | None = None


class StageDependencyMissing(RuntimeError):
    """阶段所需的注入依赖缺失。**拒绝执行**,不静默跳过。"""


# ---------------------------------------------------------------------------
# 预算作用域
# ---------------------------------------------------------------------------


class StageInvocationCeilingExceeded(BudgetExceeded):
    """某阶段的逻辑调用超过**该阶段**的硬上限。

    刻意继承 `BudgetExceeded`:适配器里已经有一条
    `except BudgetExceeded: raise`(排在通用 `except Exception` 之前)。
    继承它,阶段上限越界就不会被适配器的兜底分支吞成"llm_failed" ——
    那会把一次**预算违规**伪装成一次**模型失败**。
    """


@dataclass
class StageScope:
    """阶段作用域:在实验级治理器之上再叠一层**每阶段**硬上限。

    谁写哪个计数器(**与 `budget.py` 同一条分工**)
    ---------------------------------------------
        `StageScope.reserve()`      阶段计数 + 委托给 `BudgetGovernor.reserve()`
        `BudgetGovernor.reserve()` **唯一**写 `logical_llm_invocations` 的地方

    本类**不**直接碰 `counters.logical_llm_invocations` —— 一旦有两处能写它,
    账就会翻倍而两份数字看起来都合理。
    """

    stage: Stage
    ceiling: int
    governor: BudgetGovernor
    _invocations: int = 0

    @property
    def invocations(self) -> int:
        return self._invocations

    def reserve(self, invocations: int = 1) -> None:
        if invocations < 1:
            raise ValueError(f"invocations 至少为 1,收到 {invocations!r}")
        if self._invocations + invocations > self.ceiling:
            raise StageInvocationCeilingExceeded(
                f"阶段 {self.stage.value} 的逻辑调用已达 {self._invocations},"
                f"再调用 {invocations} 次将超过该阶段硬上限 {self.ceiling} —— "
                "调用**未发生**,立即 ABORT"
            )
        # 先过实验级硬检查(可能抛),再计阶段数 —— 顺序与 budget.py 一致。
        self.governor.reserve(invocations=invocations)
        self._invocations += invocations


@dataclass
class StageContext:
    """阶段运行器能看到的一切。"""

    stage: Stage
    config: cal_config.CalibrationConfig
    experiment_id: str
    workdir: Path
    scope: StageScope
    governor: BudgetGovernor

    def request(
        self,
        *,
        messages: Sequence[Any],
        output_token_cap: int | None = None,
        demanded_output_tokens: int | None = None,
    ) -> CalibrationRequest:
        """构造请求。`model_kwargs` **只**来自冻结配置。"""
        kwargs = self.config.model_kwargs(output_token_cap=output_token_cap)
        cap = (
            self.config.output_token_cap
            if output_token_cap is None
            else output_token_cap
        )
        return CalibrationRequest(
            stage=self.stage,
            messages=tuple(messages),
            model_kwargs=kwargs,
            output_token_cap=cap,
            demanded_output_tokens=demanded_output_tokens,
        )

    async def invoke_once(self, invoke: InvokeFn, request: CalibrationRequest) -> InvocationResult:
        """执行**恰好一次**逻辑调用。预算是硬前置条件。"""
        self.scope.reserve()
        return await invoke(request)


@dataclass(frozen=True)
class StageOutcome:
    """一个阶段的完整产出。"""

    stage: Stage
    verdict: Verdict
    evidence: dict[str, Any]
    logical_invocations: int
    provider_http_attempts: int | str

    @property
    def is_pass(self) -> bool:
        return self.verdict.is_pass


# ---------------------------------------------------------------------------
# 捕获辅助(全部是纯读取,**不猜**)
# ---------------------------------------------------------------------------


def content_text(message: Any) -> str:
    """消息文本。内容为分块列表时把文本块拼起来。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def metadata_field(message: Any, key: str) -> Any:
    """从 `response_metadata` 取一个字段。缺失返回 `None`。"""
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def normalized_usage_of(message: Any) -> dict[str, Any] | None:
    """归一化用量(`usage_metadata`)。缺失返回 `None`。"""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return dict(usage)
    return None


def observed_output_tokens(message: Any) -> int | None:
    """**观测到的**输出 token 数。

    优先级:归一化用量 → provider 原始用量。都缺失时返回 `None` ——
    **不伪造 0**:0 会被读成"provider 报了零输出"。
    """
    usage = normalized_usage_of(message)
    if usage is not None:
        value = usage.get("output_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raw = raw_token_usage_of(message)
    if raw is not None:
        for key in ("completion_tokens", "output_tokens"):
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def invalid_tool_calls_of(message: Any) -> list[Any]:
    """`invalid_tool_calls`。属性缺失时返回空列表。"""
    value = getattr(message, "invalid_tool_calls", None)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _declared_thinking_disabled(request: CalibrationRequest) -> bool:
    """**声明**的思考模式状态 —— 取自请求 kwargs,不是取自文档。"""
    thinking = request.extra_body().get("thinking")
    return isinstance(thinking, Mapping) and thinking.get("type") == "disabled"


def _payload_thinking_disabled(payload: Mapping[str, Any] | None) -> bool | None:
    """**观测到的**思考模式状态。无法观测时返回 `None`。"""
    if payload is None:
        return None
    thinking = payload.get("thinking")
    return isinstance(thinking, Mapping) and thinking.get("type") == "disabled"


def _payload_cap(payload: Mapping[str, Any] | None) -> int | None:
    """请求体里观测到的输出上限。缺失 / 无法观测返回 `None`。"""
    if payload is None:
        return None
    value = payload.get("max_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


# ---------------------------------------------------------------------------
# C0a —— provider 握手
# ---------------------------------------------------------------------------


def c0a_evidence(
    *,
    request: CalibrationRequest,
    result: InvocationResult,
    config: cal_config.CalibrationConfig,
) -> dict[str, Any]:
    """C0a 的十个捕获项。**每一项都区分"声明"与"观测"。**"""
    message = result.message
    payload = result.request_payload
    text = content_text(message)
    return {
        # 1. 请求的模型(**声明**)
        "requested_model": config.requested_model,
        # 2. provider 自报的模型(**观测**;缺失 = NOT_AVAILABLE,不猜)
        "provider_reported_model": provider_reported_model_id_of(message),
        # 3. 非空响应
        "response_non_empty": bool(text.strip()),
        "response_chars": len(text),
        # 4. 归一化用量(可能为 None)
        "normalized_usage": normalized_usage_of(message),
        # 5. provider **原始**用量(可能为 None;**不替换**归一化字段)
        "raw_token_usage": raw_token_usage_of(message),
        # 6. finish_reason(仅记录,不参与判定)
        "finish_reason": metadata_field(message, "finish_reason"),
        # 7. 响应 id
        "response_id": metadata_field(message, "id"),
        # 8. 系统指纹
        "system_fingerprint": metadata_field(message, "system_fingerprint"),
        # 9. 关闭思考模式的证据:声明侧 + 观测侧
        "thinking_disabled_declared": _declared_thinking_disabled(request),
        "thinking_disabled_observed": _payload_thinking_disabled(payload),
        # 10. 凭据无关的请求溯源
        "request_digest": config.request_shape_digest(),
        "request_payload_available": payload is not None,
        "declared_output_token_cap": request.output_token_cap,
        "observed_output_tokens": observed_output_tokens(message),
        "provider_http_attempts": (
            UNKNOWN if result.http_attempts is None else result.http_attempts
        ),
    }


def c0a_verdict(evidence: Mapping[str, Any]) -> Verdict:
    """C0a 判定。

    **它不是** benchmark、延迟对比、性能得分或成本估算 —— 因此本函数
    不看任何耗时/成本字段,证据里也刻意不含它们。
    """
    if not evidence.get("requested_model"):
        return failed("请求的模型标识为空 —— 无从确认这次握手请求了什么")
    if not evidence.get("response_non_empty"):
        return failed("响应为空 —— provider 没有返回可用内容,握手不成立")
    if not evidence.get("thinking_disabled_declared"):
        return failed(
            "请求配置里没有显式关闭思考模式 —— 本阶段要求的是 **non-thinking** "
            "路径,配置缺失说明这次请求不是标定所声明的那次"
        )
    observed = evidence.get("thinking_disabled_observed")
    if observed is False:
        return failed(
            "观测到的请求体里没有关闭思考模式的字段 —— 声明与观测不一致,"
            "说明请求在到达 provider 之前被改写过"
        )
    if not evidence.get("request_digest"):
        return failed("缺少凭据无关的请求摘要 —— 无法事后核对这次请求的形状")
    return passed(
        "握手成立:请求模型 "
        f"{evidence.get('requested_model')!r} 得到非空响应,"
        "思考模式已按声明关闭"
        + (
            ";provider 自报模型 "
            f"{evidence.get('provider_reported_model')!r}"
            if evidence.get("provider_reported_model")
            else ";provider 未自报模型标识(记 NOT_AVAILABLE,不作为失败)"
        )
    )


# ---------------------------------------------------------------------------
# C0b —— 长输出上限探针
# ---------------------------------------------------------------------------


def c0b_evidence(
    *,
    request: CalibrationRequest,
    result: InvocationResult,
    provider_rejected: bool,
    provider_rejected_reason: str | None,
    config: cal_config.CalibrationConfig,
) -> dict[str, Any]:
    """C0b 的捕获项。

    包含**上限放置位置**的核验:`requested_cap` 是声明值,
    `observed_cap_in_payload` 是观测值。两者不一致(或观测不到)时,
    判定必须降级 —— 因为"上限没生效"与"上限根本没发出去"是两件事。
    """
    message = result.message
    payload = result.request_payload
    text = content_text(message)
    demanded_tokens = request.demanded_output_tokens
    demanded_chars = (
        None
        if demanded_tokens is None
        else demanded_tokens * cal_config.CALIBRATION_C0B_CHARS_PER_TOKEN_APPROX
    )
    return {
        "requested_cap": request.output_token_cap,
        "cap_placement_declared": "extra_body.max_tokens",
        "observed_cap_in_payload": _payload_cap(payload),
        "request_payload_available": payload is not None,
        "request_digest": config.request_shape_digest(),
        "finish_reason": metadata_field(message, "finish_reason"),
        "observed_output_tokens": observed_output_tokens(message),
        "raw_token_usage": raw_token_usage_of(message),
        "normalized_usage": normalized_usage_of(message),
        "observed_response_chars": len(text),
        "demanded_output_tokens": demanded_tokens,
        "demanded_response_chars": demanded_chars,
        "provider_rejected": provider_rejected,
        "provider_rejected_reason": provider_rejected_reason,
        "provider_http_attempts": (
            UNKNOWN if result.http_attempts is None else result.http_attempts
        ),
    }


def c0b_evidence_verdict(evidence: Mapping[str, Any]) -> Verdict:
    """由 C0b 证据得出判定。

    先做**前置核验**(上限到底有没有发出去),再做长度判定:

        观测到请求体里没有上限字段        → FAIL(参数从未到达 provider)
        请求体不可观测                    → INCONCLUSIVE(探针前提无法核验)

    这两条排在长度判定**之前** —— 否则一个"上限根本没发出去、模型自然写短"
    的运行会被判成 PASS,而那是完全错误的结论。
    """
    if not evidence.get("request_payload_available"):
        return inconclusive(
            "传输层没有暴露请求体 —— 无法核验上限是否真的发给了 provider;"
            "在前提无法核验的情况下,长度证据不能单独支撑任何结论"
        )
    if evidence.get("observed_cap_in_payload") is None:
        return failed(
            "观测到的请求体里**没有**输出上限字段 —— 上限从未到达 provider,"
            "因此'上限是否被执行'这个问题根本没有被问到"
        )
    if evidence.get("observed_cap_in_payload") != evidence.get("requested_cap"):
        return failed(
            f"请求体里的上限 {evidence.get('observed_cap_in_payload')!r} 与声明的 "
            f"{evidence.get('requested_cap')!r} 不一致 —— 请求在到达 provider 之前"
            "被改写过"
        )
    return c0b_verdict(
        requested_cap=evidence["requested_cap"],
        demanded_output_tokens=evidence["demanded_output_tokens"],
        observed_output_tokens=evidence.get("observed_output_tokens"),
        observed_chars=evidence.get("observed_response_chars"),
        demanded_chars=evidence.get("demanded_response_chars"),
        finish_reason=evidence.get("finish_reason"),
        provider_rejected=bool(evidence.get("provider_rejected")),
    )


# ---------------------------------------------------------------------------
# C1 —— 工具调用形状
# ---------------------------------------------------------------------------


class ToolCallDefect(str, Enum):
    """tool call 形状缺陷的封闭词表。"""

    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    ARGS_NOT_JSON = "ARGS_NOT_JSON"
    ARGS_SCHEMA_INVALID = "ARGS_SCHEMA_INVALID"
    MISSING_TOOL_CALL_ID = "MISSING_TOOL_CALL_ID"
    INVALID_TOOL_CALLS_PRESENT = "INVALID_TOOL_CALLS_PRESENT"


_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def _value_matches_type(value: Any, expected: str) -> bool:
    types = _JSON_TYPE_MAP.get(expected)
    if types is None:
        return True  # 未知类型不判 —— 不猜
    if expected in ("integer", "number") and isinstance(value, bool):
        # `bool` 是 `int` 的子类。不显式排除的话 `True` 会被判成合法整数。
        return False
    return isinstance(value, types)


def _schema_problems(args: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    """一个**极小**的 JSON-schema 子集校验:必填项 + 顶层类型。

    刻意不引入 `jsonschema` 依赖(D-2c 不做依赖变更),也刻意不做完整
    实现 —— 本阶段要判定的是"参数是否符合工具契约的基本形状",
    不是做一个通用校验器。
    """
    problems: list[str] = []
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    for key in required:
        if key not in args:
            problems.append(f"缺少必填参数 {key!r}")
    for key, value in args.items():
        spec = properties.get(key)
        if not isinstance(spec, Mapping):
            continue
        expected_types: list[str] = []
        declared = spec.get("type")
        if isinstance(declared, str):
            expected_types.append(declared)
        for sub in spec.get("anyOf") or []:
            if isinstance(sub, Mapping) and isinstance(sub.get("type"), str):
                expected_types.append(sub["type"])
        if not expected_types:
            continue
        if not any(_value_matches_type(value, item) for item in expected_types):
            problems.append(
                f"参数 {key!r} 的类型不符合契约(期望 {'/'.join(expected_types)})"
            )
    return problems


def tool_call_findings(
    *,
    tool_calls: Sequence[Mapping[str, Any]],
    invalid_tool_calls: Sequence[Any],
    contract: ToolContract,
) -> list[dict[str, Any]]:
    """逐条列出 tool call 形状缺陷。空列表 = 形状合法。"""
    findings: list[dict[str, Any]] = []
    if invalid_tool_calls:
        findings.append({
            "defect": ToolCallDefect.INVALID_TOOL_CALLS_PRESENT.value,
            "detail": f"响应带 {len(invalid_tool_calls)} 条解析失败的 tool call",
        })
    for index, call in enumerate(tool_calls):
        name = call.get("name")
        args = call.get("args")
        call_id = call.get("id")
        if not isinstance(name, str) or name not in contract.allowed_tool_names:
            findings.append({
                "defect": ToolCallDefect.UNKNOWN_TOOL.value,
                "index": index,
                "detail": f"工具名 {name!r} 不在允许清单里",
            })
            continue
        if not isinstance(call_id, str) or not call_id.strip():
            findings.append({
                "defect": ToolCallDefect.MISSING_TOOL_CALL_ID.value,
                "index": index,
                "detail": "tool_call_id 缺失或为空 —— 往返时无法配对",
            })
        if isinstance(args, str):
            try:
                json.loads(args)
            except json.JSONDecodeError:
                findings.append({
                    "defect": ToolCallDefect.ARGS_NOT_JSON.value,
                    "index": index,
                    "detail": "args 是字符串但不是合法 JSON",
                })
            continue
        if not isinstance(args, Mapping):
            findings.append({
                "defect": ToolCallDefect.ARGS_NOT_JSON.value,
                "index": index,
                "detail": f"args 类型为 {type(args).__name__},既不是字典也不是 JSON 字符串",
            })
            continue
        schema = contract.arg_schemas.get(name)
        if schema:
            for problem in _schema_problems(args, schema):
                findings.append({
                    "defect": ToolCallDefect.ARGS_SCHEMA_INVALID.value,
                    "index": index,
                    "detail": problem,
                })
    return findings


def c1_verdict(findings: Sequence[Mapping[str, Any]], *, call_count: int) -> Verdict:
    """C1 判定。有任何缺陷即 FAIL。"""
    if call_count == 0:
        return failed("模型没有给出任何 tool call —— 工具契约没有被检验到")
    if findings:
        return failed(
            f"发现 {len(findings)} 处 tool call 形状缺陷:"
            f"{sorted({item['defect'] for item in findings})}"
        )
    return passed(
        f"{call_count} 个 tool call 全部合法:工具名在允许清单内、"
        "参数为合法 JSON 且符合契约、tool_call_id 非空"
    )


# ---------------------------------------------------------------------------
# C2 —— 工具往返
# ---------------------------------------------------------------------------


class RoundTripDefect(str, Enum):
    """工具往返缺陷的封闭词表。"""

    TOOL_CALL_WITHOUT_RESULT = "TOOL_CALL_WITHOUT_RESULT"
    RESULT_WITHOUT_TOOL_CALL = "RESULT_WITHOUT_TOOL_CALL"
    DUPLICATE_RESULT = "DUPLICATE_RESULT"
    MISSING_FINAL_ASSISTANT = "MISSING_FINAL_ASSISTANT"
    EMPTY_FINAL_ASSISTANT = "EMPTY_FINAL_ASSISTANT"
    REASONING_CONTENT_REQUIRED = "REASONING_CONTENT_REQUIRED"


def tool_roundtrip_findings(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """逐条列出工具往返缺陷。空列表 = 往返成立。

    `REASONING_CONTENT_REQUIRED` 是**反面**缺陷:它只在"往返成立需要
    重放 `reasoning_content`"时才会出现。D-2c 走的是 non-thinking 路径,
    因此正常结果里**不该**出现它 —— 它存在的意义是让"我们并没有悄悄依赖
    思考内容"这件事可被断言。
    """
    findings: list[dict[str, Any]] = []

    call_ids: list[str] = []
    result_ids: list[str] = []
    final_text = ""
    saw_final_after_tool = False

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            call_id = call.get("id") if isinstance(call, Mapping) else None
            if isinstance(call_id, str) and call_id:
                call_ids.append(call_id)
        # `ToolMessage` 的判定用类名而不是 import —— 避免与 langchain_core
        # 的版本细节耦合,同时保持只读。
        if type(message).__name__ == "ToolMessage":
            tool_call_id = getattr(message, "tool_call_id", None)
            if isinstance(tool_call_id, str):
                result_ids.append(tool_call_id)
            continue
        if call_ids and not tool_calls:
            text = content_text(message)
            saw_final_after_tool = True
            if text.strip():
                final_text = text

    for call_id in call_ids:
        if call_id not in result_ids:
            findings.append({
                "defect": RoundTripDefect.TOOL_CALL_WITHOUT_RESULT.value,
                "detail": f"tool_call_id {call_id!r} 没有对应的 ToolMessage",
            })
    seen: set[str] = set()
    for result_id in result_ids:
        if result_id not in call_ids:
            findings.append({
                "defect": RoundTripDefect.RESULT_WITHOUT_TOOL_CALL.value,
                "detail": f"ToolMessage 的 tool_call_id {result_id!r} 没有对应的 tool call",
            })
        if result_id in seen:
            findings.append({
                "defect": RoundTripDefect.DUPLICATE_RESULT.value,
                "detail": f"tool_call_id {result_id!r} 有多个 ToolMessage",
            })
        seen.add(result_id)

    if not saw_final_after_tool:
        findings.append({
            "defect": RoundTripDefect.MISSING_FINAL_ASSISTANT.value,
            "detail": "工具结果之后没有出现最终助手回复 —— 往返没有闭合",
        })
    elif not final_text.strip():
        findings.append({
            "defect": RoundTripDefect.EMPTY_FINAL_ASSISTANT.value,
            "detail": "最终助手回复为空",
        })
    return findings


def c2_verdict(findings: Sequence[Mapping[str, Any]], *, round_tripped: int) -> Verdict:
    """C2 判定。有任何缺陷即 FAIL。"""
    if round_tripped == 0:
        return failed("没有任何 tool_call_id 完成往返 —— 往返契约没有被检验到")
    if findings:
        return failed(
            f"发现 {len(findings)} 处工具往返缺陷:"
            f"{sorted({item['defect'] for item in findings})}"
        )
    return passed(
        f"{round_tripped} 个 tool_call_id 全部原样往返,且工具结果之后"
        "出现了非空的最终助手回复(未依赖 reasoning_content 重放)"
    )


# ---------------------------------------------------------------------------
# 阶段运行器
# ---------------------------------------------------------------------------


def _outcome(
    ctx: StageContext,
    verdict: Verdict,
    evidence: dict[str, Any],
    *,
    http_attempts: int | None = None,
) -> StageOutcome:
    """组装阶段产出。

    `provider_http_attempts` 只在传输层**直接观测到**时才是数字;
    否则记 `UNKNOWN` —— **不记 0**。0 会被读成"一次物理请求都没发",
    而真相是"我们不知道"。
    """
    return StageOutcome(
        stage=ctx.stage,
        verdict=verdict,
        evidence=evidence,
        logical_invocations=ctx.scope.invocations,
        provider_http_attempts=UNKNOWN if http_attempts is None else http_attempts,
    )


def _require_invoke(deps: StageDeps) -> InvokeFn:
    if deps.invoke is None:
        raise StageDependencyMissing(
            "本阶段需要注入的调用器 —— 缺失时**拒绝执行**,不静默跳过"
        )
    return deps.invoke


async def run_c0a(ctx: StageContext, deps: StageDeps) -> StageOutcome:
    """C0a:一次握手。捕获十项,判定是否"连上且读懂"。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    invoke = _require_invoke(deps)
    request = ctx.request(
        messages=(
            SystemMessage(content="D-2c 标定:握手探针。请只回复一个词。"),
            HumanMessage(content="回复 OK。"),
        )
    )
    result = await ctx.invoke_once(invoke, request)
    evidence = c0a_evidence(request=request, result=result, config=ctx.config)
    return _outcome(
        ctx, c0a_verdict(evidence), evidence, http_attempts=result.http_attempts
    )


async def run_c0b(ctx: StageContext, deps: StageDeps) -> StageOutcome:
    """C0b:长输出上限探针。请求一个小上限,同时要求远长于它的输出。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    invoke = _require_invoke(deps)
    cap = cal_config.CALIBRATION_C0B_TEMPORARY_CAP
    demanded = cal_config.CALIBRATION_C0B_DEMANDED_OUTPUT_TOKENS
    request = ctx.request(
        messages=(
            SystemMessage(
                content=(
                    "D-2c 标定:上限探针。请**完整**列出一个长清单,"
                    "不要提前收尾。"
                )
            ),
            HumanMessage(
                content=(
                    f"请输出至少 {demanded} 个 token 的内容:"
                    "逐条列出你能想到的安全事件类型,每条一行。"
                )
            ),
        ),
        output_token_cap=cap,
        demanded_output_tokens=demanded,
    )
    rejected = False
    rejected_reason: str | None = None
    try:
        result = await ctx.invoke_once(invoke, request)
    except CapParameterRejected as exc:
        # provider 显式拒绝上限参数 —— 判定为 FAIL,但仍然产出一份**可复核**的证据。
        rejected = True
        rejected_reason = str(exc)
        result = InvocationResult(
            message=type("_Rejected", (), {"content": "", "response_metadata": {}})(),
            request_payload=None,
            http_attempts=None,
        )
    evidence = c0b_evidence(
        request=request,
        result=result,
        provider_rejected=rejected,
        provider_rejected_reason=rejected_reason,
        config=ctx.config,
    )
    return _outcome(
        ctx,
        c0b_evidence_verdict(evidence),
        evidence,
        http_attempts=result.http_attempts,
    )


async def run_c1(ctx: StageContext, deps: StageDeps) -> StageOutcome:
    """C1:工具调用形状。让模型给出一个 tool call,校验其形状。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    invoke = _require_invoke(deps)
    contract = deps.tool_contract or ToolContract(
        allowed_tool_names=(), arg_schemas={}
    )
    request = ctx.request(
        messages=(
            SystemMessage(content="D-2c 标定:工具形状探针。请调用一个工具。"),
            HumanMessage(content="请调用一个可用工具来完成一次查询。"),
        )
    )
    result = await ctx.invoke_once(invoke, request)
    message = result.message
    calls = [dict(call) for call in (getattr(message, "tool_calls", None) or [])]
    findings = tool_call_findings(
        tool_calls=calls,
        invalid_tool_calls=invalid_tool_calls_of(message),
        contract=contract,
    )
    evidence = {
        "allowed_tool_names": list(contract.allowed_tool_names),
        "tool_call_count": len(calls),
        "tool_calls": calls,
        "invalid_tool_calls": invalid_tool_calls_of(message),
        "findings": findings,
        "request_digest": ctx.config.request_shape_digest(),
    }
    return _outcome(ctx, c1_verdict(findings, call_count=len(calls)), evidence)


async def run_c2(ctx: StageContext, deps: StageDeps) -> StageOutcome:
    """C2:工具往返。

    注入的 `roundtrip` 是**已经发生过的**一段消息序列(由调用方提供)。
    本阶段对这段序列做机械校验 —— 它**不重新发起调用**,因此不会为了
    "验证往返"而额外消耗额度。这与 C2 契约里"无真实 provider 流量"一致。

    因此它的物理尝试数是**已知的 0**(没有调用就没有请求),而不是
    `UNKNOWN` —— 这一点与"调用发生了但没观测到"必须区分开。
    """
    messages = deps.roundtrip
    if messages is None:
        raise StageDependencyMissing(
            "本阶段需要注入的往返消息序列 —— 缺失时拒绝执行,不静默跳过"
        )
    findings = tool_roundtrip_findings(messages)
    call_ids = {
        call.get("id")
        for message in messages
        for call in (getattr(message, "tool_calls", None) or [])
        if isinstance(call.get("id"), str) and call.get("id")
    }
    evidence = {
        "message_count": len(messages),
        "tool_call_ids": sorted(call_ids),
        "findings": findings,
        "reasoning_content_replay_required": any(
            item["defect"] == RoundTripDefect.REASONING_CONTENT_REQUIRED.value
            for item in findings
        ),
        "provider_http_attempts": 0,
    }
    return _outcome(
        ctx, c2_verdict(findings, round_tripped=len(call_ids)), evidence, http_attempts=0
    )


async def run_c3(ctx: StageContext, deps: StageDeps) -> StageOutcome:
    """C3:**真实** LangGraph 代码路径 + 合成模型。

    为什么 `tools=[]`:本阶段要检验的是**图的控制流与消息管道**
    (agent_node → tools_node → agent_node 的往返),不是工具 I/O。
    给空工具集让本阶段完全 hermetic —— 不读任何数据文件。
    未知工具走的是图里既有的"未知工具"分支,同样经过真实的
    `ToolMessage(tool_call_id=...)` 构造路径。

    **阶段上限必须真的生效。** 图里的调用经 `BudgetedLLM` 走
    `StageScope.reserve()`,而不是直接走实验级治理器 —— 否则一次
    "图跑了 5 轮"的运行会绕过 C3 的 3 次上限,而阶段表里仍写着"上限 3"。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.core.graph import create_agent_graph
    from app.evaluation.llm.budgeted_llm import budgeted

    model = deps.graph_model
    if model is None:
        raise StageDependencyMissing(
            "C3 需要注入合成模型 —— 缺失时拒绝执行,不静默跳过"
        )
    llm = budgeted(model, ctx.scope)
    graph = create_agent_graph(llm, tools=[])
    try:
        final_state = await graph.ainvoke({
            "messages": [
                SystemMessage(content="D-2c 标定:图路径探针。"),
                HumanMessage(content="请调用工具,然后给出结论。"),
            ],
            "iteration_count": 0,
        })
    except BudgetExceeded as exc:
        # 预算 / 阶段上限越界必须**原样穿透**成为 ABORT,不得降级成"图执行失败"。
        evidence = {
            "budget_abort": str(exc),
            "logical_invocations": llm.invocations,
            "stage_scope_invocations": ctx.scope.invocations,
            "ceiling": stage_ceiling(ctx.stage),
            "graph_error": None,
        }
        raise
    except Exception as exc:  # noqa: BLE001 —— 任何失败都必须成为**判定**,不是崩溃
        evidence = {
            "graph_error": type(exc).__name__,
            "logical_invocations": llm.invocations,
            "stage_scope_invocations": ctx.scope.invocations,
            "ceiling": stage_ceiling(ctx.stage),
        }
        return _outcome(
            ctx, failed(f"真实图路径执行失败:{type(exc).__name__}"), evidence
        )

    messages = final_state.get("messages", [])
    findings = tool_roundtrip_findings(messages)
    evidence = {
        "message_kinds": [type(item).__name__ for item in messages],
        "iteration_count": final_state.get("iteration_count"),
        "logical_invocations": llm.invocations,
        "stage_scope_invocations": ctx.scope.invocations,
        "ceiling": stage_ceiling(ctx.stage),
        "findings": findings,
        "graph_error": None,
    }
    if ctx.scope.invocations > stage_ceiling(ctx.stage):
        return _outcome(
            ctx,
            failed(
                f"图路径发生了 {ctx.scope.invocations} 次逻辑调用,超过该阶段上限 "
                f"{stage_ceiling(ctx.stage)}"
            ),
            evidence,
        )
    verdict = c2_verdict(
        findings,
        round_tripped=len({
            call.get("id")
            for message in messages
            for call in (getattr(message, "tool_calls", None) or [])
            if isinstance(call.get("id"), str) and call.get("id")
        }),
    )
    if verdict.is_pass:
        verdict = passed(
            f"真实 LangGraph 路径跑通:{ctx.scope.invocations} 次逻辑调用"
            f"(上限 {stage_ceiling(ctx.stage)}),tool_call_id 原样往返,"
            "终答非空"
        )
    return _outcome(ctx, verdict, evidence)


async def run_c4(ctx: StageContext, deps: StageDeps) -> StageOutcome:
    """C4:**一个代表性评测单元** —— 合成模型 + 真实工具 + 真实适配器。

    走 `B2PrimeGraphAdapter`,即与正式评测**同一段**执行代码;模型由
    合成实现提供。因此它检验的是"评测单元在真实 provider 形状的响应下
    能不能跑通",而不是"模型的回答好不好"。

    与 C3 一样,预算边界传给**阶段作用域** —— 适配器里的每次调用因此
    同时受该阶段上限与实验级硬上界约束。
    """
    unit = deps.evaluation_unit
    if unit is None:
        raise StageDependencyMissing(
            "C4 需要注入一个代表性评测单元 —— 缺失时拒绝执行,不静默跳过"
        )
    from app.evaluation.llm.adapters import B2PrimeGraphAdapter

    factory = _synthetic_factory(deps.unit_model)
    adapter = B2PrimeGraphAdapter(
        dataset_paths=dict(unit.dataset_paths),
        llm_factory=factory,
        governor=ctx.scope,
    )
    observation = await adapter.run(
        unit.task, unit.behavior, dataset_paths=dict(unit.dataset_paths)
    )
    evidence = {
        "task_id": observation.task_id,
        "baseline_label": observation.baseline,
        "run_status": observation.run_status,
        "llm_call_count": observation.llm_call_count,
        "tool_call_count": observation.tool_call_count,
        "stage_scope_invocations": ctx.scope.invocations,
        "ceiling": stage_ceiling(ctx.stage),
        "error": observation.error,
        "answer_non_empty": bool((observation.answer or "").strip()),
    }
    if observation.run_status == "llm_failed":
        return _outcome(
            ctx,
            failed(f"评测单元执行失败:{observation.error}"),
            evidence,
        )
    if ctx.scope.invocations > stage_ceiling(ctx.stage):
        return _outcome(
            ctx,
            failed(
                f"评测单元发生了 {ctx.scope.invocations} 次逻辑调用,"
                f"超过该阶段上限 {stage_ceiling(ctx.stage)}"
            ),
            evidence,
        )
    if not evidence["answer_non_empty"]:
        return _outcome(ctx, failed("评测单元没有产出非空终答"), evidence)
    return _outcome(
        ctx,
        passed(
            f"评测单元跑通:{observation.baseline} × {observation.task_id},"
            f"{ctx.scope.invocations} 次逻辑调用"
            f"(上限 {stage_ceiling(ctx.stage)}),终答非空"
        ),
        evidence,
    )


def _synthetic_factory(model: SyntheticProviderModel | None) -> Any:
    """把**同一个**合成模型实例交给适配器的工厂协议。

    适配器会用 `behavior` / `task` / `dataset_paths` / `decoy_paths` /
    `emit_usage` 调用它 —— 这些描述评测上下文,与"用哪个模型"无关。
    这里**刻意复用同一个实例**:一个评测单元只应消耗一条脚本。
    该实例由 `StageDeps.unit_model` **单独**提供,不与 C3 共用。
    """
    if model is None:
        raise StageDependencyMissing("C4 需要注入合成模型(unit_model)")

    def factory(**_: Any) -> SyntheticProviderModel:
        return model

    return factory


#: 默认的阶段运行器表。**顺序不在此处** —— 顺序由 `harness.STAGE_ORDER` 强制。
DEFAULT_STAGE_RUNNERS: dict[Stage, Callable[..., Awaitable[StageOutcome]]] = {
    Stage.C0A: run_c0a,
    Stage.C0B: run_c0b,
    Stage.C1: run_c1,
    Stage.C2: run_c2,
    Stage.C3: run_c3,
    Stage.C4: run_c4,
}


def production_tool_contract() -> ToolContract:
    """从**生产**工具集派生工具契约(只读 schema,不执行任何工具)。

    刻意取 `tool_call_schema.model_json_schema()` 而**不是** `tool.args`:
    后者只返回 `properties`,**不含 `required`** —— 用它做校验会让
    "必填参数缺失"这类最典型的形状缺陷恒测不出来(一个恒真的护栏)。

    `app.tools` 的 import 放在函数体内,保持本模块导入期零副作用。
    """
    from app.tools import DEFAULT_TOOLS

    schemas: dict[str, Mapping[str, Any]] = {}
    for tool in DEFAULT_TOOLS:
        call_schema = getattr(tool, "tool_call_schema", None)
        if call_schema is not None and hasattr(call_schema, "model_json_schema"):
            schemas[tool.name] = call_schema.model_json_schema()
        else:  # pragma: no cover - 回退:至少保留 properties
            schemas[tool.name] = {"properties": dict(getattr(tool, "args", {}) or {})}
    return ToolContract(
        allowed_tool_names=tuple(tool.name for tool in DEFAULT_TOOLS),
        arg_schemas=schemas,
    )
