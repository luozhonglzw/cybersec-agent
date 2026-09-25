"""Phase 9.2-D-2c 标定基础设施的**聚焦测试**。

全程离线:零真实 provider 调用、零 API 额度消耗、零凭据。
所有 provider 行为都由 `app/evaluation/calibration/synthetic.py` 的合成形状
或测试内联的桩提供 —— 本文件**不构造任何真实 provider 客户端**。

测试分节(与 D-2c 实现契约逐节对应):

    1  冻结配置与"请求溯源 / 记录溯源同源"
    2  D1–D5 请求构造(extra_body / max_retries / streaming / temperature NOT_SET)
    3  C0a 捕获与判定
    4  C0b 七例判定(A–G)+ 前置核验
    5  C1 工具调用形状(八个用例)
    6  C2 工具往返
    7  C3/C4 真实代码路径与阶段上限
    8  顺序阶段门(非 PASS 阻断后续阶段)
    9  命名空间与文件系统隔离(D7/D8)
    10 可观察性四区分(§10)
    11 原始 usage 保留(§11)
    12 凭据安全(§12)
    13 离线出口守卫(§13)
    14 反同义反复:判定语义本身
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.evaluation import calibration as C
from app.evaluation.calibration import stages as S
from app.evaluation.calibration.synthetic import (
    DEEPSEEK_CACHE_HIT_KEY,
    OPENAI_CACHED_TOKENS_KEY,
    SYNTHETIC_PROVIDER_MODEL,
    SyntheticProviderModel,
    SyntheticTurn,
    deepseek_shaped_raw_usage,
    normalize_openai_shaped,
    openai_shaped_raw_usage,
    synthetic_ai_message,
    synthetic_tool_call,
)
from app.evaluation.llm.budget import UNKNOWN, BudgetExceeded
from app.evaluation.llm.dataset import LLM_TASKS, build_datasets
from app.evaluation.llm.offline_guard import (
    NetworkEgressError,
    NetworkEgressGuard,
)
from app.evaluation.real_provider import (
    build_chat_model,
    provider_reported_model_id_of,
    raw_token_usage_of,
)

#: 明显是合成物的凭据占位符。**不是**任何真实密钥。
SYNTHETIC_API_KEY = "sk-d2c-synthetic-placeholder-000000000000"

EXPERIMENT_ID = "d2c-cal-unit-20260923"

#: 两个"守卫有牙"探针**故意**制造的出口目标。
#:
#: 用一个**不可能被真实代码用到**的保留域名(`.invalid` 由 RFC 2606 保留),
#: 模块级守卫才能把"我们故意制造的出口"与"任何真实出口"区分开:
#: 前者被允许,后者一旦出现就是失败。
DELIBERATE_EGRESS_HOST = "d2c-deliberate-egress-probe.invalid"


# ---------------------------------------------------------------------------
# 模块级离线证明
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _module_runs_fully_offline():
    """整个 D-2c 聚焦套件必须在**出口守卫**下运行。

    为什么放在模块级而不是逐个测试里:逐个测试只能证明"那一个测试没联网"。
    一次误引入的真实 provider 调用完全可能出现在夹具、导入期或某个被复用的
    辅助函数里 —— 而它在任何单个测试的视角下都"没发生"。

    为什么是**记录型**而不是严格型:本模块里有两个探针**故意**制造出口
    (证明守卫有牙、证明意外出口不会被报成零调用)。严格型会在发生处抛错,
    那两个探针就无法完成它们自己的断言。因此这里改为在**模块结束时**核验:

        所有被记录到的出口,目标都必须是那个保留的哨兵域名。

    于是"故意出口"被允许,而**任何**指向真实主机(包括标定端点)的出口
    都会让模块失败。
    """
    guard = NetworkEgressGuard(strict=False)
    with guard:
        yield
    hosts = {event.detail for event in guard.egress_events}
    assert hosts <= {DELIBERATE_EGRESS_HOST}, (
        f"检测到非预期的对外网络访问目标:{sorted(hosts - {DELIBERATE_EGRESS_HOST})} —— "
        "本闸门必须完全离线"
    )


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def datasets(tmp_path_factory):
    """四个数据集变体 —— 落在临时目录,绝不写仓库 `data/`。"""
    return build_datasets(tmp_path_factory.mktemp("d2c-fixtures"))


@pytest.fixture
def workdir(tmp_path):
    """本次标定的工作目录根(仓库之外)。"""
    path = tmp_path / "d2c-workdir"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def representative_unit(datasets):
    """一个代表性评测单元:真实任务 + 真实工具契约 + 合成数据路径。"""
    task = LLM_TASKS[0]
    paths = datasets[task.dataset_variant]
    return S.EvaluationUnit(
        task=task,
        baseline_label="B2'",
        behavior="GOOD",
        dataset_paths=paths,
    )


@pytest.fixture(scope="session")
def tool_contract():
    """生产工具契约(**只读 schema**,不执行任何工具)。

    走 `stages.production_tool_contract()` —— 它取的是**完整** JSON schema
    (含 `required`),而不是只有 `properties` 的 `tool.args`。
    """
    return S.production_tool_contract()


def request_payload(request: S.CalibrationRequest, config: C.CalibrationConfig) -> dict:
    """把一次请求的 kwargs 折成"传输层会看到的请求体"(**凭据无关**)。

    刻意把 `temperature=None` 当作"字段不存在"处理 —— 与真实传输一致。
    """
    payload: dict = {"model": config.requested_model}
    for key, value in request.model_kwargs.items():
        if key == "extra_body":
            payload.update(value)
        elif key in ("temperature", "streaming"):
            if value is not None:
                payload[key] = value
    return payload


def make_invoker(
    *,
    config: C.CalibrationConfig | None = None,
    c0a_message=None,
    c0b_message=None,
    c1_message=None,
    on_invoke=None,
):
    """构造一个**按阶段分派**的合成调用器。零网络、零 SDK。"""
    cfg = config or C.default_calibration_config()
    seen: list[S.CalibrationRequest] = []

    async def invoke(request: S.CalibrationRequest) -> S.InvocationResult:
        seen.append(request)
        if on_invoke is not None:
            on_invoke(request)
        payload = request_payload(request, cfg)
        if request.stage is S.Stage.C0A:
            message = c0a_message or synthetic_ai_message(content="OK")
        elif request.stage is S.Stage.C0B:
            message = c0b_message or synthetic_ai_message(
                content="x" * 64,
                finish_reason="length",
                raw_usage=deepseek_shaped_raw_usage(
                    prompt_tokens=20, completion_tokens=request.output_token_cap
                ),
                normalized_usage={
                    "input_tokens": 20,
                    "output_tokens": request.output_token_cap,
                    "total_tokens": 20 + request.output_token_cap,
                },
            )
        elif request.stage is S.Stage.C1:
            message = c1_message or synthetic_ai_message(
                content="",
                tool_calls=[
                    synthetic_tool_call(
                        name="analyze_risk_tool",
                        args={"indicator": "203.0.113.66"},
                        call_id="c1-call-1",
                    )
                ],
                finish_reason="tool_calls",
            )
        else:  # pragma: no cover - C2–C4 不经过调用器
            raise AssertionError(f"调用器不处理阶段 {request.stage}")
        return S.InvocationResult(
            message=message, request_payload=payload, http_attempts=1
        )

    invoke.seen = seen  # type: ignore[attr-defined]
    return invoke


def roundtrip_messages(*, tool_call_id: str = "rt-1", final: str = "结论:高风险") -> list:
    """一段**已发生**的工具往返消息序列(合成)。"""
    return [
        SystemMessage(content="sys"),
        HumanMessage(content="q"),
        synthetic_ai_message(
            content="",
            tool_calls=[
                synthetic_tool_call(
                    name="analyze_risk_tool",
                    args={"indicator": "203.0.113.66"},
                    call_id=tool_call_id,
                )
            ],
            finish_reason="tool_calls",
        ),
        ToolMessage(content='{"ok":true}', tool_call_id=tool_call_id),
        synthetic_ai_message(content=final),
    ]


def build_deps(
    *,
    representative_unit,
    tool_contract,
    invoker=None,
    roundtrip=None,
    graph_turns=None,
    unit_turns=None,
) -> S.StageDeps:
    """装配六个阶段的全部合成依赖。"""
    graph_turns = graph_turns or (
        SyntheticTurn(
            content="",
            tool_calls=(
                synthetic_tool_call(name="no_such_tool", args={"x": 1}, call_id="c3-1"),
            ),
            finish_reason="tool_calls",
        ),
        SyntheticTurn(content="结论:已处理。"),
    )
    unit_turns = unit_turns or (
        SyntheticTurn(
            content="",
            tool_calls=(
                synthetic_tool_call(
                    name="analyze_risk_tool",
                    args={
                        "indicator": representative_unit.task.indicator,
                        "logs_path": representative_unit.dataset_paths["logs"],
                        "intel_path": representative_unit.dataset_paths["intel"],
                    },
                    call_id="c4-1",
                ),
            ),
            finish_reason="tool_calls",
        ),
        SyntheticTurn(content="结论:该指标风险等级已评估。"),
    )
    return S.StageDeps(
        invoke=invoker or make_invoker(),
        tool_contract=tool_contract,
        roundtrip=roundtrip or roundtrip_messages(),
        graph_model=SyntheticProviderModel(turns=graph_turns),
        unit_model=SyntheticProviderModel(turns=unit_turns),
        evaluation_unit=representative_unit,
    )


def run_full(
    *,
    workdir,
    deps,
    experiment_id: str = EXPERIMENT_ID,
    runners=None,
    guard=None,
):
    """跑一次完整标定。"""
    harness = C.CalibrationHarness(
        experiment_id=experiment_id,
        workdir_root=workdir,
        deps=deps,
        runners=runners,
        guard=guard,
    )
    import asyncio

    return asyncio.run(harness.run())


# ===========================================================================
# 1. 冻结配置与"请求溯源 / 记录溯源同源"
# ===========================================================================


def test_01_frozen_identity_is_the_controller_frozen_one():
    cfg = C.default_calibration_config()
    assert cfg.provider == "deepseek"
    assert cfg.requested_model == "deepseek-flash"
    assert cfg.provider_family == "DeepSeek-V4.1-Flash"
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.endpoint_category.value == "OPENAI_COMPATIBLE"
    assert cfg.streaming is False
    assert cfg.temperature is None
    assert cfg.max_retries == 0
    assert cfg.timeout_seconds == 60.0
    assert cfg.output_token_cap == 1024


def test_02_provider_family_is_provenance_only_never_a_substitute():
    """家族名只用于溯源对照 —— 它**绝不**能替代请求的模型标识。"""
    cfg = C.default_calibration_config()
    assert cfg.provider_family != cfg.requested_model
    assert cfg.requested_model not in cfg.provider_family
    # 请求 kwargs 里出现的必须是 requested_model 的候选,不是家族名。
    assert cfg.provider_candidate().model == cfg.requested_model


def test_03_excluded_models_are_rejected():
    for model in C.EXCLUDED_MODELS:
        with pytest.raises(ValueError):
            C.CalibrationConfig(requested_model=model)


def test_04_not_set_temperature_cannot_be_turned_into_a_number():
    """`NOT_SET` 是唯一合法取值 —— 把它记成 0.0 必须被拒绝。"""
    with pytest.raises(ValueError):
        C.CalibrationConfig(temperature=0.0)
    assert C.default_calibration_config().recorded_temperature() is None


def test_05_other_frozen_knobs_reject_drift():
    with pytest.raises(ValueError):
        C.CalibrationConfig(max_retries=1)
    with pytest.raises(ValueError):
        C.CalibrationConfig(streaming=True)
    with pytest.raises(ValueError):
        C.CalibrationConfig(provider_family=C.CALIBRATION_REQUESTED_MODEL)
    with pytest.raises(ValueError):
        C.CalibrationConfig(requested_model="   ")


def test_06_request_side_and_record_side_come_from_one_object():
    """请求溯源与记录溯源**必须同源** —— 否则一次标定只会产出"自洽的假象"。"""
    cfg = C.default_calibration_config()
    kwargs = cfg.model_kwargs()
    assert kwargs["temperature"] is cfg.recorded_temperature()
    shape = cfg.request_shape()
    assert shape["temperature"] is kwargs["temperature"]
    assert shape["temperature_present"] is False
    assert shape["max_retries"] == kwargs["max_retries"] == 0
    assert shape["streaming"] == kwargs["streaming"] is False


def test_07_stage_ceiling_sum_equals_nominal_invocations():
    """每阶段上限之和必须与名义调用数同源 —— 否则两者互相矛盾。"""
    assert C.STAGE_LOGICAL_INVOCATION_CEILINGS == {
        "C0a": 1,
        "C0b": 1,
        "C1": 1,
        "C2": 2,
        "C3": 3,
        "C4": 5,
    }
    assert sum(C.STAGE_LOGICAL_INVOCATION_CEILINGS.values()) == C.NOMINAL_LOGICAL_INVOCATIONS
    assert C.calibration_budget().logical_invocation_hard_ceiling == (
        C.HARD_LOGICAL_INVOCATION_CEILING
    )
    assert C.calibration_budget().total_runs == C.NOMINAL_EXPERIMENTAL_RUNS
    assert C.HARD_EXPERIMENTAL_RUN_CEILING == 8


def test_08_headroom_is_not_a_rerun_licence():
    """硬上界与名义值之差**不是**重跑许可 —— 配置里必须能读出这句话。"""
    budget = C.calibration_budget()
    basis = budget.provider_http_attempt_ceiling_basis
    assert "不是观测值" in basis
    assert budget.provider_http_attempt_ceiling == (
        C.HARD_LOGICAL_INVOCATION_CEILING * (C.CALIBRATION_MAX_RETRIES + 1)
    )


# ===========================================================================
# 2. D1–D5 请求构造
# ===========================================================================


def _recording_chat_model(monkeypatch):
    """把 `langchain_openai.ChatOpenAI` 换成**记录器**。

    这样可以直接断言"交给 SDK 的 kwargs",而不构造任何真实客户端、
    不 import 任何 provider SDK 的网络栈。
    """
    import langchain_openai

    captured: dict = {}

    class _Recorder:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Recorder)
    return captured


def test_09_temperature_none_omits_the_field_entirely(monkeypatch):
    """D4:`None` 必须意味着**不发送**,而不是发 0.0。"""
    captured = _recording_chat_model(monkeypatch)
    cfg = C.default_calibration_config()
    build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        allow_network=True,
        **cfg.model_kwargs(),
    )
    assert "temperature" not in captured, "NOT_SET 被静默替换成了一个具体数值"
    assert captured["model"] == cfg.requested_model
    assert captured["max_retries"] == 0
    assert captured["streaming"] is False
    assert captured["timeout"] == 60.0
    assert captured["base_url"] == cfg.base_url


def test_10_explicit_zero_temperature_is_still_sent(monkeypatch):
    """`0.0` 是合法且**必须**被送出的显式取值 —— 不能被当成"未设定"。"""
    captured = _recording_chat_model(monkeypatch)
    cfg = C.default_calibration_config()
    build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        temperature=0.0,
        allow_network=True,
    )
    assert captured["temperature"] == 0.0


def test_11_thinking_disable_travels_as_data_not_as_a_code_branch(monkeypatch):
    """D1:关闭思考模式由 `extra_body` 以**数据**形式透传。"""
    captured = _recording_chat_model(monkeypatch)
    cfg = C.default_calibration_config()
    build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        allow_network=True,
        **cfg.model_kwargs(),
    )
    assert captured["extra_body"]["thinking"] == {"type": "disabled"}
    assert captured["extra_body"]["max_tokens"] == cfg.output_token_cap


def test_12_cap_travels_via_extra_body_not_via_max_tokens():
    """上限**不**走 `max_tokens` 字段。

    本地 LangChain 会把 `max_tokens` 改写成 `max_completion_tokens`,
    因此标定显式走 `extra_body` —— 这是实测结论,不是风格偏好。
    """
    cfg = C.default_calibration_config()
    kwargs = cfg.model_kwargs()
    assert "max_tokens" not in kwargs
    assert kwargs["extra_body"]["max_tokens"] == cfg.output_token_cap


def test_13_c0b_override_cap_is_per_call_and_does_not_leak():
    """C0b 的临时上限只影响那一次请求,**不进入** C1–C4。"""
    cfg = C.default_calibration_config()
    assert cfg.model_kwargs(output_token_cap=16)["extra_body"]["max_tokens"] == 16
    assert cfg.model_kwargs()["extra_body"]["max_tokens"] == cfg.output_token_cap
    # 派生是纯函数:连续调用不改变冻结值。
    assert cfg.model_kwargs(output_token_cap=16)["extra_body"]["max_tokens"] == 16


def test_14_defaults_are_backwards_compatible_with_d2b(monkeypatch):
    """D2/D3:不传新旋钮时,构造出的 kwargs 与 D-2b 逐字段一致。"""
    captured = _recording_chat_model(monkeypatch)
    from app.evaluation.real_provider import candidate_by_id, CANDIDATE_B

    build_chat_model(
        candidate=candidate_by_id(CANDIDATE_B), api_key=SYNTHETIC_API_KEY, allow_network=True
    )
    assert captured["temperature"] == 0.0  # D-2a/D-2b 的历史默认值,未被改动
    for absent in ("max_retries", "streaming", "extra_body", "timeout"):
        assert absent not in captured


def test_15_provider_reported_model_and_raw_usage_readers(monkeypatch):
    """D5:两个响应侧读取函数 —— 缺失一律 `None`,**不猜**。"""
    message = synthetic_ai_message(
        content="OK",
        raw_usage=deepseek_shaped_raw_usage(
            prompt_tokens=10, completion_tokens=2, prompt_cache_hit_tokens=4
        ),
    )
    assert provider_reported_model_id_of(message) == SYNTHETIC_PROVIDER_MODEL
    assert raw_token_usage_of(message)[DEEPSEEK_CACHE_HIT_KEY] == 4

    bare = synthetic_ai_message(content="OK", with_metadata=False)
    assert provider_reported_model_id_of(bare) is None
    assert raw_token_usage_of(bare) is None
    assert provider_reported_model_id_of(object()) is None
    assert raw_token_usage_of(object()) is None


# ===========================================================================
# 3. C0a —— provider 握手
# ===========================================================================


def _c0a_outcome(
    *, workdir, deps, c0a_message=None, config=None, invoker=None
) -> C.StageRecord:
    """跑一次完整标定并返回 **C0a** 的那条记录。

    刻意跑完整门而不是只跑 C0a:这样"同一份依赖在完整链路里"才是被检验的
    事实,而不是"某个被裁剪过的子集里"。
    """
    import dataclasses

    if invoker is None and c0a_message is not None:
        invoker = make_invoker(config=config, c0a_message=c0a_message)
    if invoker is not None:
        deps = dataclasses.replace(deps, invoke=invoker)
    return run_full(workdir=workdir, deps=deps).records[0]


def test_16_c0a_captures_all_ten_items(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    outcome = _c0a_outcome(workdir=workdir, deps=deps)
    evidence = outcome.evidence
    for key in (
        "requested_model",
        "provider_reported_model",
        "response_non_empty",
        "normalized_usage",
        "raw_token_usage",
        "finish_reason",
        "response_id",
        "system_fingerprint",
        "thinking_disabled_declared",
        "thinking_disabled_observed",
        "request_digest",
    ):
        assert key in evidence, key
    assert outcome.verdict is C.StageVerdict.PASS
    assert evidence["requested_model"] == "deepseek-flash"
    assert evidence["provider_reported_model"] == SYNTHETIC_PROVIDER_MODEL
    assert evidence["response_non_empty"] is True


def test_17_c0a_is_not_a_benchmark_latency_or_cost_report(workdir, representative_unit, tool_contract):
    """C0a **不是** benchmark / 延迟对比 / 性能得分 / 成本估算。"""
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    evidence = _c0a_outcome(workdir=workdir, deps=deps).evidence
    for forbidden in ("latency", "duration", "cost", "price", "score", "throughput"):
        assert not any(forbidden in key for key in evidence), forbidden


def test_18_c0a_fails_on_empty_response(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    outcome = _c0a_outcome(
        workdir=workdir, deps=deps, c0a_message=synthetic_ai_message(content="   ")
    )
    assert outcome.verdict is C.StageVerdict.FAIL
    assert "空" in outcome.reason


def test_19_c0a_fails_when_thinking_disable_was_not_configured(workdir, representative_unit, tool_contract):
    """配置里没有关闭思考模式 ⇒ 这不是标定所声明的那次请求。"""
    cfg = C.CalibrationConfig()
    evidence = S.c0a_evidence(
        request=S.CalibrationRequest(
            stage=S.Stage.C0A,
            messages=(),
            model_kwargs={"extra_body": {"max_tokens": 1024}},  # 无 thinking
            output_token_cap=1024,
        ),
        result=S.InvocationResult(
            message=synthetic_ai_message(content="OK"), request_payload={"thinking": None}
        ),
        config=cfg,
    )
    assert evidence["thinking_disabled_declared"] is False
    assert S.c0a_verdict(evidence).verdict is C.StageVerdict.FAIL


def test_20_c0a_fails_when_observed_payload_contradicts_the_declaration(workdir):
    """声明关了思考模式,但观测到的请求体里没有该字段 ⇒ FAIL。"""
    cfg = C.CalibrationConfig()
    evidence = S.c0a_evidence(
        request=S.CalibrationRequest(
            stage=S.Stage.C0A,
            messages=(),
            model_kwargs={"extra_body": cfg.thinking_body()},
            output_token_cap=cfg.output_token_cap,
        ),
        result=S.InvocationResult(
            message=synthetic_ai_message(content="OK"),
            request_payload={"model": cfg.requested_model},
        ),
        config=cfg,
    )
    assert evidence["thinking_disabled_declared"] is True
    assert evidence["thinking_disabled_observed"] is False
    assert S.c0a_verdict(evidence).verdict is C.StageVerdict.FAIL


def test_21_c0a_missing_provider_metadata_is_not_a_failure(workdir):
    """provider 什么都没报 ⇒ 记 NOT_AVAILABLE,**不作为失败**。"""
    cfg = C.CalibrationConfig()
    evidence = S.c0a_evidence(
        request=S.CalibrationRequest(
            stage=S.Stage.C0A,
            messages=(),
            model_kwargs={"extra_body": cfg.thinking_body()},
            output_token_cap=cfg.output_token_cap,
        ),
        result=S.InvocationResult(
            message=synthetic_ai_message(content="OK", with_metadata=False),
            request_payload={"thinking": {"type": "disabled"}},
        ),
        config=cfg,
    )
    assert evidence["provider_reported_model"] is None
    assert evidence["raw_token_usage"] is None
    assert evidence["finish_reason"] is None
    assert S.c0a_verdict(evidence).verdict is C.StageVerdict.PASS


# ===========================================================================
# 4. C0b —— 七例(A–G)
# ===========================================================================


def _c0b_verdict_from_response(
    *,
    cap: int,
    demanded: int,
    content: str,
    finish_reason: str | None,
    output_tokens: int | None,
    payload_has_cap: bool = True,
    provider_rejected: bool = False,
):
    cfg = C.CalibrationConfig()
    request = S.CalibrationRequest(
        stage=S.Stage.C0B,
        messages=(),
        model_kwargs=cfg.model_kwargs(output_token_cap=cap),
        output_token_cap=cap,
        demanded_output_tokens=demanded,
    )
    payload: dict | None = {"thinking": {"type": "disabled"}}
    if payload_has_cap:
        payload["max_tokens"] = cap
    usage = None
    if output_tokens is not None:
        usage = {"input_tokens": 5, "output_tokens": output_tokens, "total_tokens": 5 + output_tokens}
    result = S.InvocationResult(
        message=synthetic_ai_message(
            content=content, finish_reason=finish_reason, normalized_usage=usage
        ),
        request_payload=payload,
    )
    evidence = S.c0b_evidence(
        request=request,
        result=result,
        provider_rejected=provider_rejected,
        provider_rejected_reason="synthetic rejection" if provider_rejected else None,
        config=cfg,
    )
    return S.c0b_evidence_verdict(evidence), evidence


CAP = 16
DEMANDED = 2000


def test_22_c0b_case_a_enforced_cap_passes():
    """A:观测输出与请求上限一致 ⇒ PASS。"""
    verdict, _ = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="x" * 60,
        finish_reason="length",
        output_tokens=CAP,
    )
    assert verdict.verdict is C.StageVerdict.PASS


def test_23_c0b_case_b_natural_early_termination_is_inconclusive():
    """B:模型远低于上限就自然结束 ⇒ INCONCLUSIVE(上限**未被触及**)。"""
    verdict, _ = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="short",
        finish_reason="stop",
        output_tokens=4,
    )
    assert verdict.verdict is C.StageVerdict.INCONCLUSIVE


def test_24_c0b_case_c_missing_usage_evidence_is_inconclusive():
    """C:用量证据缺失且长度也不足 ⇒ INCONCLUSIVE。"""
    verdict, evidence = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="short",
        finish_reason="stop",
        output_tokens=None,
    )
    assert evidence["observed_output_tokens"] is None
    assert verdict.verdict is C.StageVerdict.INCONCLUSIVE


def test_25_c0b_case_d_output_exceeds_cap_materially():
    """D:观测输出显著超过上限 ⇒ FAIL。"""
    verdict, _ = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="x" * 4000,
        finish_reason="stop",
        output_tokens=900,
    )
    assert verdict.verdict is C.StageVerdict.FAIL


def test_26_c0b_case_e_provider_rejects_the_cap_parameter():
    """E:provider 拒绝上限参数 ⇒ FAIL。"""
    verdict, evidence = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="",
        finish_reason=None,
        output_tokens=None,
        provider_rejected=True,
    )
    assert evidence["provider_rejected"] is True
    assert verdict.verdict is C.StageVerdict.FAIL


def test_27_c0b_case_f_finish_reason_stop_alone_must_not_imply_fail():
    """F:`finish_reason == "stop"` **单独**不得推出 FAIL。

    构造:输出恰好等于上限、finish_reason 为 stop。若判定把 stop 当作
    "上限被忽略"的证据,这里就会 FAIL —— 那是无效蕴含。
    """
    verdict, _ = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="x" * 60,
        finish_reason="stop",
        output_tokens=CAP,
    )
    assert verdict.verdict is not C.StageVerdict.FAIL
    assert verdict.verdict is C.StageVerdict.PASS


def test_28_c0b_case_g_finish_reason_length_alone_must_not_imply_pass():
    """G:`finish_reason == "length"` **单独**不得推出 PASS。

    构造:输出远低于上限、finish_reason 为 length。若判定把 length 当作
    "上限生效"的证据,这里就会 PASS —— 同样是无效蕴含。
    """
    verdict, _ = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="tiny",
        finish_reason="length",
        output_tokens=3,
    )
    assert verdict.verdict is not C.StageVerdict.PASS
    assert verdict.verdict is C.StageVerdict.INCONCLUSIVE


def test_29_c0b_verdict_function_never_branches_on_finish_reason():
    """机械证明:`finish_reason` 换任何取值都不改变判定。

    这是"判定只依据长度证据"的最强形式 —— 穷举同一组长度证据下的
    全部停止原因,判定必须完全相同。
    """
    reasons = ["stop", "length", "tool_calls", "content_filter", None, "unknown"]
    verdicts = {
        _c0b_verdict_from_response(
            cap=CAP,
            demanded=DEMANDED,
            content="x" * 60,
            finish_reason=reason,
            output_tokens=CAP,
        )[0].verdict
        for reason in reasons
    }
    assert verdicts == {C.StageVerdict.PASS}

    verdicts = {
        _c0b_verdict_from_response(
            cap=CAP,
            demanded=DEMANDED,
            content="tiny",
            finish_reason=reason,
            output_tokens=3,
        )[0].verdict
        for reason in reasons
    }
    assert verdicts == {C.StageVerdict.INCONCLUSIVE}


def test_30_c0b_fails_when_cap_never_reached_the_wire():
    """前置核验:请求体里没有上限字段 ⇒ FAIL(这个问题根本没被问到)。"""
    verdict, evidence = _c0b_verdict_from_response(
        cap=CAP,
        demanded=DEMANDED,
        content="x" * 60,
        finish_reason="length",
        output_tokens=CAP,
        payload_has_cap=False,
    )
    assert evidence["observed_cap_in_payload"] is None
    assert verdict.verdict is C.StageVerdict.FAIL


def test_31_c0b_is_inconclusive_when_the_wire_payload_is_unobservable():
    """请求体不可观测 ⇒ 探针前提无法核验 ⇒ INCONCLUSIVE(不是 PASS)。"""
    cfg = C.CalibrationConfig()
    request = S.CalibrationRequest(
        stage=S.Stage.C0B,
        messages=(),
        model_kwargs=cfg.model_kwargs(output_token_cap=CAP),
        output_token_cap=CAP,
        demanded_output_tokens=DEMANDED,
    )
    evidence = S.c0b_evidence(
        request=request,
        result=S.InvocationResult(
            message=synthetic_ai_message(
                content="x" * 60,
                finish_reason="length",
                normalized_usage={"input_tokens": 5, "output_tokens": CAP, "total_tokens": 21},
            ),
            request_payload=None,
        ),
        provider_rejected=False,
        provider_rejected_reason=None,
        config=cfg,
    )
    assert evidence["request_payload_available"] is False
    assert S.c0b_evidence_verdict(evidence).verdict is C.StageVerdict.INCONCLUSIVE


def test_32_c0b_demanded_length_must_far_exceed_the_cap():
    """要求长度必须**远大于**上限,否则"自然停下"与"上限生效"无法区分。"""
    with pytest.raises(ValueError):
        C.c0b_verdict(
            requested_cap=16,
            demanded_output_tokens=16,
            observed_output_tokens=16,
            observed_chars=None,
            demanded_chars=None,
            finish_reason="length",
            provider_rejected=False,
        )


def test_33_c0b_char_fallback_never_fabricates_a_pass():
    """无 token usage 时,字符证据最多只能给出 INCONCLUSIVE 或 FAIL。"""
    cfg = C.CalibrationConfig()
    request = S.CalibrationRequest(
        stage=S.Stage.C0B,
        messages=(),
        model_kwargs=cfg.model_kwargs(output_token_cap=CAP),
        output_token_cap=CAP,
        demanded_output_tokens=DEMANDED,
    )
    evidence = S.c0b_evidence(
        request=request,
        result=S.InvocationResult(
            message=synthetic_ai_message(content="x" * 5000, finish_reason="length"),
            request_payload={"max_tokens": CAP},
        ),
        provider_rejected=False,
        provider_rejected_reason=None,
        config=cfg,
    )
    verdict = S.c0b_evidence_verdict(evidence)
    assert verdict.verdict is C.StageVerdict.FAIL


# ===========================================================================
# 5. C1 —— 工具调用形状(八个用例)
# ===========================================================================


def _c1_findings(calls, *, invalid=(), contract):
    return S.tool_call_findings(
        tool_calls=calls, invalid_tool_calls=invalid, contract=contract
    )


def test_34_c1_valid_allowed_tool_name(tool_contract):
    findings = _c1_findings(
        [synthetic_tool_call(name="analyze_risk_tool", args={"indicator": "1.2.3.4"}, call_id="a")],
        contract=tool_contract,
    )
    assert findings == []


def test_35_c1_valid_json_args_as_a_string(tool_contract):
    """args 以 JSON 字符串给出(未解析)时,只要能解析就合法。"""
    findings = _c1_findings(
        [synthetic_tool_call(name="analyze_risk_tool", args='{"indicator": "1.2.3.4"}', call_id="a")],
        contract=tool_contract,
    )
    assert findings == []


def test_36_c1_schema_valid_args(tool_contract):
    findings = _c1_findings(
        [
            synthetic_tool_call(
                name="analyze_risk_tool",
                args={"indicator": "1.2.3.4", "logs_path": "/tmp/x.jsonl"},
                call_id="a",
            )
        ],
        contract=tool_contract,
    )
    assert findings == []


def test_37_c1_non_empty_tool_call_id_required(tool_contract):
    findings = _c1_findings(
        [synthetic_tool_call(name="analyze_risk_tool", args={"indicator": "1.2.3.4"}, call_id="   ")],
        contract=tool_contract,
    )
    assert [item["defect"] for item in findings] == [S.ToolCallDefect.MISSING_TOOL_CALL_ID.value]


def test_38_c1_invalid_tool_args_are_detected(tool_contract):
    """schema 违规:必填项缺失 + 类型不符。"""
    findings = _c1_findings(
        [synthetic_tool_call(name="analyze_risk_tool", args={"event_type": "x"}, call_id="a")],
        contract=tool_contract,
    )
    defects = {item["defect"] for item in findings}
    assert S.ToolCallDefect.ARGS_SCHEMA_INVALID.value in defects
    assert S.c1_verdict(findings, call_count=1).verdict is C.StageVerdict.FAIL


def test_39_c1_unknown_tool_is_detected(tool_contract):
    findings = _c1_findings(
        [synthetic_tool_call(name="delete_everything_tool", args={}, call_id="a")],
        contract=tool_contract,
    )
    assert [item["defect"] for item in findings] == [S.ToolCallDefect.UNKNOWN_TOOL.value]


def test_40_c1_missing_tool_call_id_is_detected(tool_contract):
    findings = _c1_findings(
        [synthetic_tool_call(name="analyze_risk_tool", args={"indicator": "1.2.3.4"}, call_id=None)],
        contract=tool_contract,
    )
    assert [item["defect"] for item in findings] == [S.ToolCallDefect.MISSING_TOOL_CALL_ID.value]


def test_41_c1_invalid_tool_calls_are_detected(tool_contract):
    findings = _c1_findings(
        [synthetic_tool_call(name="analyze_risk_tool", args={"indicator": "1.2.3.4"}, call_id="a")],
        invalid=[{"name": "analyze_risk_tool", "error": "parse error"}],
        contract=tool_contract,
    )
    assert [item["defect"] for item in findings] == [
        S.ToolCallDefect.INVALID_TOOL_CALLS_PRESENT.value
    ]


def test_42_c1_non_json_args_are_detected(tool_contract):
    findings = _c1_findings(
        [synthetic_tool_call(name="analyze_risk_tool", args="{not json", call_id="a")],
        contract=tool_contract,
    )
    assert [item["defect"] for item in findings] == [S.ToolCallDefect.ARGS_NOT_JSON.value]


def test_43_c1_zero_tool_calls_is_a_failure_not_a_vacuous_pass(tool_contract):
    """没有 tool call ⇒ 契约没有被检验到 ⇒ FAIL,而不是"无事发生"。"""
    assert S.c1_verdict([], call_count=0).verdict is C.StageVerdict.FAIL


def test_44_c1_boolean_is_not_an_integer_argument(tool_contract):
    """`bool` 是 `int` 的子类 —— 不显式排除的话 `True` 会被判成合法整数。"""
    findings = _c1_findings(
        [
            synthetic_tool_call(
                name="query_security_logs_tool",
                args={"limit": True},
                call_id="a",
            )
        ],
        contract=tool_contract,
    )
    assert [item["defect"] for item in findings] == [
        S.ToolCallDefect.ARGS_SCHEMA_INVALID.value
    ]


# ===========================================================================
# 6. C2 —— 工具往返
# ===========================================================================


def test_45_c2_exact_id_round_trip_passes():
    findings = S.tool_roundtrip_findings(roundtrip_messages(tool_call_id="exact-42"))
    assert findings == []
    assert S.c2_verdict(findings, round_tripped=1).verdict is C.StageVerdict.PASS


def test_46_c2_mismatched_tool_call_id_is_detected():
    messages = roundtrip_messages(tool_call_id="wanted")
    messages[3] = ToolMessage(content="{}", tool_call_id="other")
    findings = S.tool_roundtrip_findings(messages)
    defects = {item["defect"] for item in findings}
    assert S.RoundTripDefect.TOOL_CALL_WITHOUT_RESULT.value in defects
    assert S.RoundTripDefect.RESULT_WITHOUT_TOOL_CALL.value in defects


def test_47_c2_missing_result_is_detected():
    messages = roundtrip_messages()
    del messages[3]
    findings = S.tool_roundtrip_findings(messages)
    assert [item["defect"] for item in findings] == [
        S.RoundTripDefect.TOOL_CALL_WITHOUT_RESULT.value
    ]


def test_48_c2_missing_final_assistant_is_detected():
    messages = roundtrip_messages()
    messages[4] = synthetic_ai_message(
        content="",
        tool_calls=[
            synthetic_tool_call(
                name="analyze_risk_tool", args={"indicator": "1.2.3.4"}, call_id="rt-2"
            )
        ],
        finish_reason="tool_calls",
    )
    findings = S.tool_roundtrip_findings(messages)
    defects = {item["defect"] for item in findings}
    assert S.RoundTripDefect.MISSING_FINAL_ASSISTANT.value in defects


def test_49_c2_empty_final_assistant_is_detected():
    messages = roundtrip_messages(final="   ")
    findings = S.tool_roundtrip_findings(messages)
    assert [item["defect"] for item in findings] == [
        S.RoundTripDefect.EMPTY_FINAL_ASSISTANT.value
    ]


def test_50_c2_non_thinking_path_does_not_require_reasoning_content():
    """D-2c 走 non-thinking 路径 —— 往返成立**不得**依赖 `reasoning_content` 重放。"""
    messages = roundtrip_messages()
    for message in messages:
        assert "reasoning_content" not in getattr(message, "additional_kwargs", {})
    findings = S.tool_roundtrip_findings(messages)
    assert not any(
        item["defect"] == S.RoundTripDefect.REASONING_CONTENT_REQUIRED.value
        for item in findings
    )
    assert S.c2_verdict(findings, round_tripped=1).verdict is C.StageVerdict.PASS


def test_51_c2_zero_round_trips_is_a_failure_not_a_vacuous_pass():
    assert S.c2_verdict([], round_tripped=0).verdict is C.StageVerdict.FAIL


# ===========================================================================
# 7. C3/C4 —— 真实代码路径与阶段上限
# ===========================================================================


def test_52_full_calibration_passes_with_synthetic_transport(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    assert run.passed_all, [r.verdict.value for r in run.records]
    assert run.attempted_stages == S.STAGE_ORDER
    assert run.skipped_stages == ()


def test_53_c3_uses_the_real_langgraph_path(workdir, representative_unit, tool_contract):
    """C3 必须经过**真实** LangGraph 代码路径 —— 证据里能看到 ToolMessage。"""
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    evidence = run.records[S.STAGE_ORDER.index(S.Stage.C3)].evidence
    assert "ToolMessage" in evidence["message_kinds"]
    assert evidence["iteration_count"] >= 2
    assert evidence["graph_error"] is None


def test_54_c4_runs_a_representative_unit_with_a_real_tool(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    evidence = run.records[S.STAGE_ORDER.index(S.Stage.C4)].evidence
    assert evidence["baseline_label"] == "B2'"
    assert evidence["run_status"] == "not_gated"
    assert evidence["tool_call_count"] == 1
    assert evidence["answer_non_empty"] is True


def test_55_c3_and_c4_do_not_share_a_synthetic_script(workdir, representative_unit, tool_contract):
    """C3 与 C4 必须各有一条独立脚本 —— 共用会让 C4 拿到耗尽后的空脚本。"""
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    assert deps.graph_model is not deps.unit_model
    run = run_full(workdir=workdir, deps=deps)
    c4 = run.records[S.STAGE_ORDER.index(S.Stage.C4)].evidence
    assert c4["llm_call_count"] == 2


def test_56_stage_ceiling_actually_binds_inside_the_graph_path(workdir, representative_unit, tool_contract):
    """阶段上限必须真的生效 —— 否则阶段表写着"上限 3"而图跑了 5 轮。"""
    turns = tuple(
        SyntheticTurn(
            content="",
            tool_calls=(
                synthetic_tool_call(name="no_such_tool", args={"x": i}, call_id=f"c3-{i}"),
            ),
            finish_reason="tool_calls",
        )
        for i in range(10)
    )
    deps = build_deps(
        representative_unit=representative_unit, tool_contract=tool_contract, graph_turns=turns
    )
    run = run_full(workdir=workdir, deps=deps)
    c3 = run.records[S.STAGE_ORDER.index(S.Stage.C3)]
    assert c3.verdict is C.StageVerdict.ABORT
    assert "预算中止" in c3.reason
    assert c3.logical_invocations <= S.stage_ceiling(S.Stage.C3)
    assert run.halted_at is S.Stage.C3
    assert run.skipped_stages == (S.Stage.C4,)


def test_57_experiment_level_hard_ceiling_is_never_exceeded(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    assert run.budget["experiment_logical_llm_invocations"] <= (
        C.HARD_LOGICAL_INVOCATION_CEILING
    )
    assert run.budget["experiment_experimental_runs"] == C.NOMINAL_EXPERIMENTAL_RUNS


def test_58_c3_c4_never_produce_pilot_eligible_records(workdir, representative_unit, tool_contract):
    """C3/C4 都**不得**产出 D-2d 可用记录。"""
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    assert run.scanned_artifacts == ()
    C.assert_no_pilot_eligible_records(workdir)


def test_59_stage_workdirs_are_independent(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    harness = C.CalibrationHarness(
        experiment_id=EXPERIMENT_ID, workdir_root=workdir, deps=deps
    )
    dirs = {stage: harness.stage_workdir(stage) for stage in S.STAGE_ORDER}
    assert len(set(dirs.values())) == len(S.STAGE_ORDER)
    for path in dirs.values():
        assert C.REPO_DATA_DIR not in path.parents
        assert path != C.REPO_DATA_DIR


# ===========================================================================
# 8. 顺序阶段门
# ===========================================================================


def _spy_runners(hits: list[str]):
    def make(stage):
        async def runner(ctx, deps):
            hits.append(stage.value)
            return await S.DEFAULT_STAGE_RUNNERS[stage](ctx, deps)
        return runner

    return {stage: make(stage) for stage in S.STAGE_ORDER}


def _run_with_c0b_message(*, workdir, representative_unit, tool_contract, c0b_message, hits):
    invoker = make_invoker(c0b_message=c0b_message)
    deps = build_deps(
        representative_unit=representative_unit, tool_contract=tool_contract, invoker=invoker
    )
    return run_full(workdir=workdir, deps=deps, runners=_spy_runners(hits))


def test_60_inconclusive_c0b_blocks_every_later_stage(workdir, representative_unit, tool_contract):
    """INCONCLUSIVE **必须**阻止 C1 开始 —— 而且连运行器都不许被调用。"""
    hits: list[str] = []
    run = _run_with_c0b_message(
        workdir=workdir,
        representative_unit=representative_unit,
        tool_contract=tool_contract,
        c0b_message=synthetic_ai_message(
            content="short",
            finish_reason="stop",
            normalized_usage={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        ),
        hits=hits,
    )
    assert run.records[S.STAGE_ORDER.index(S.Stage.C0B)].verdict is C.StageVerdict.INCONCLUSIVE
    assert hits == ["C0a", "C0b"]
    assert run.skipped_stages == (S.Stage.C1, S.Stage.C2, S.Stage.C3, S.Stage.C4)
    assert run.halted_at is S.Stage.C0B
    assert run.passed_all is False


def test_61_failed_c0a_blocks_everything_immediately(workdir, representative_unit, tool_contract):
    hits: list[str] = []
    invoker = make_invoker(c0a_message=synthetic_ai_message(content=""))
    deps = build_deps(
        representative_unit=representative_unit, tool_contract=tool_contract, invoker=invoker
    )
    run = run_full(workdir=workdir, deps=deps, runners=_spy_runners(hits))
    assert hits == ["C0a"]
    assert run.halted_at is S.Stage.C0A
    assert run.skipped_stages == (S.Stage.C0B, S.Stage.C1, S.Stage.C2, S.Stage.C3, S.Stage.C4)


def test_62_stage_order_is_the_frozen_order():
    assert S.STAGE_ORDER == (
        S.Stage.C0A,
        S.Stage.C0B,
        S.Stage.C1,
        S.Stage.C2,
        S.Stage.C3,
        S.Stage.C4,
    )


def test_63_missing_dependency_aborts_rather_than_silently_skipping(workdir):
    """依赖缺失 ⇒ ABORT(门关闭),**不是**静默跳过。"""
    harness = C.CalibrationHarness(
        experiment_id=EXPERIMENT_ID, workdir_root=workdir, deps=S.StageDeps()
    )
    import asyncio

    run = asyncio.run(harness.run())
    assert run.halted_at is S.Stage.C0A
    assert run.records[0].verdict is C.StageVerdict.ABORT
    assert "依赖缺失" in run.records[0].reason
    assert run.passed_all is False


# ===========================================================================
# 9. 命名空间与文件系统隔离(D7 / D8)
# ===========================================================================


def test_64_calibration_and_pilot_ids_are_classified():
    assert C.classify_experiment_id("d2c-cal-c0a-20260923") is C.ExperimentNamespace.CALIBRATION
    assert C.classify_experiment_id("d2d-pilot-run1") is C.ExperimentNamespace.PILOT


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "d2c-cal-",          # 只有前缀
        "d2d-pilot-",        # 只有前缀
        "random-id",         # 未知前缀
        " d2c-cal-x",        # 首部空白
        "d2c-cal-x ",        # 尾部空白
        "d2c-cal-x-d2d-pilot-y",  # 同时含两个前缀(歧义)
    ],
)
def test_65_malformed_or_ambiguous_ids_are_rejected(bad):
    with pytest.raises(C.CalibrationNamespaceError):
        C.classify_experiment_id(bad)


def test_66_non_string_ids_are_rejected():
    for value in (None, 42, object(), ["d2c-cal-x"]):
        with pytest.raises(C.CalibrationNamespaceError):
            C.classify_experiment_id(value)


def test_67_calibration_id_must_not_be_derived_from_a_pilot_id():
    """给试点 id 加后缀**不是**造标定 id 的方式 —— 它仍以试点前缀开头。"""
    derived = "d2d-pilot-run1" + "-cal"
    with pytest.raises(C.CalibrationNamespaceError):
        C.assert_calibration_experiment_id(derived)


def test_68_pilot_assertions_reject_calibration_ids():
    with pytest.raises(C.CalibrationNamespaceError):
        C.assert_pilot_experiment_id("d2c-cal-c0a-20260923")


def test_69_calibration_experiment_id_is_independently_constructed():
    value = C.calibration_experiment_id(stage="c0a", stamp="20260923T120000Z")
    assert value.startswith(C.CALIBRATION_ID_PREFIX)
    assert C.classify_experiment_id(value) is C.ExperimentNamespace.CALIBRATION
    with pytest.raises(C.CalibrationNamespaceError):
        C.calibration_experiment_id(stage="  ", stamp="20260923")
    with pytest.raises(C.CalibrationNamespaceError):
        C.calibration_experiment_id(stage="c0a", stamp="")


def test_70_assert_returns_a_plain_str_not_the_original_object():
    """返回值必须是 `str` —— 非字符串对象的相等语义可能恒真。"""
    class _Weird(str):
        def __eq__(self, other):  # pragma: no cover - 只用于证明返回值被规整
            return True

    value = C.assert_calibration_experiment_id(_Weird("d2c-cal-x"))
    assert type(value) is str
    assert value == "d2c-cal-x"


def test_71_repo_data_is_never_a_valid_workdir():
    with pytest.raises(C.CalibrationNamespaceError):
        C.assert_outside_repo_data(C.REPO_DATA_DIR)
    with pytest.raises(C.CalibrationNamespaceError):
        C.assert_outside_repo_data(C.REPO_DATA_DIR / "nested")


def test_72_calibration_and_pilot_roots_are_disjoint(tmp_path):
    cal = C.calibration_root(tmp_path)
    pilot = C.pilot_root(tmp_path)
    C.assert_workdirs_disjoint(cal, pilot)
    with pytest.raises(C.CalibrationNamespaceError):
        C.assert_workdirs_disjoint(cal, cal)
    with pytest.raises(C.CalibrationNamespaceError):
        C.assert_workdirs_disjoint(cal, cal / "nested")


def test_73_harness_rejects_a_pilot_id_before_constructing_anything(workdir):
    with pytest.raises(C.CalibrationNamespaceError):
        C.CalibrationHarness(experiment_id="d2d-pilot-x", workdir_root=workdir)


def test_74_harness_rejects_a_workdir_inside_repo_data():
    with pytest.raises(C.CalibrationNamespaceError):
        C.CalibrationHarness(
            experiment_id=EXPERIMENT_ID, workdir_root=C.REPO_DATA_DIR
        )


def test_75_calibration_records_are_not_pilot_completion_evidence():
    """标定产物**永不**成为试点的完成资格 —— 由既有守卫机械证明。"""
    C.assert_not_pilot_eligible(artifact_experiment_id="d2c-cal-c0a-20260923")

    from app.evaluation.llm.raw import (
        ForeignExperimentRecord,
        RawRecord,
        resume_eligibility,
    )

    record = RawRecord(
        run_id="x",
        experiment_id="d2c-cal-c0a-20260923",
        protocol_version="d2c",
        manifest_digest="0" * 64,
        task_id="t",
        condition="calibration",
        baseline_label="C0a",
        dataset_variant="none",
        repetition_id=1,
        execution_index=0,
    )
    with pytest.raises(ForeignExperimentRecord):
        resume_eligibility(record, experiment_id="d2d-pilot-run1")


def test_76_pilot_eligible_records_in_the_calibration_tree_are_refused(workdir):
    """标定目录里出现试点命名空间的记录 ⇒ 拒绝(而不只是"没扫到")。"""
    root = C.calibration_root(workdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "sneaky.jsonl").write_text(
        json.dumps({"experiment_id": "d2d-pilot-run1"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(C.PilotEligibilityViolation):
        C.assert_no_pilot_eligible_records(workdir)


def test_77_repo_data_untouched_by_a_full_run(workdir, representative_unit, tool_contract):
    """一次完整运行不得在仓库 `data/` 下留下任何新文件。"""
    before = sorted(p.name for p in C.REPO_DATA_DIR.rglob("*")) if C.REPO_DATA_DIR.exists() else []
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run_full(workdir=workdir, deps=deps)
    after = sorted(p.name for p in C.REPO_DATA_DIR.rglob("*")) if C.REPO_DATA_DIR.exists() else []
    assert before == after


# ===========================================================================
# 10. 可观察性四区分(§10)
# ===========================================================================


def test_78_requested_model_is_kept_separate_from_provider_reported_model(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    evidence = _c0a_outcome(workdir=workdir, deps=deps).evidence
    assert evidence["requested_model"] == "deepseek-flash"
    assert evidence["provider_reported_model"] == SYNTHETIC_PROVIDER_MODEL
    assert evidence["requested_model"] != evidence["provider_reported_model"]


def test_79_declared_cap_is_kept_separate_from_observed_usage(workdir, representative_unit, tool_contract):
    invoker = make_invoker(
        c0b_message=synthetic_ai_message(
            content="x" * 60,
            finish_reason="length",
            normalized_usage={"input_tokens": 3, "output_tokens": 16, "total_tokens": 19},
        )
    )
    deps = build_deps(
        representative_unit=representative_unit, tool_contract=tool_contract, invoker=invoker
    )
    run = run_full(workdir=workdir, deps=deps)
    evidence = run.records[S.STAGE_ORDER.index(S.Stage.C0B)].evidence
    assert evidence["requested_cap"] == C.CALIBRATION_C0B_TEMPORARY_CAP
    assert evidence["observed_output_tokens"] == 16
    assert run.config.output_token_cap == C.CALIBRATION_OUTPUT_TOKEN_CAP
    # 声明的全局上限与本次探针的临时上限是**两个不同的量**,不得互相顶替。
    assert evidence["requested_cap"] != run.config.output_token_cap


def test_80_stage_ceiling_is_kept_separate_from_actual_invocations(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    for record in run.records:
        assert record.ceiling == S.stage_ceiling(record.stage)
        assert record.logical_invocations <= record.ceiling


def test_81_theoretical_http_ceiling_is_kept_separate_from_observed_attempts(workdir, representative_unit, tool_contract):
    """理论 HTTP 尝试上界 ≠ 观测到的物理尝试数。

    观测不可得时记 `UNKNOWN`,**不记 0**,也不用上界顶替。
    """
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    assert run.provider_http_attempts is UNKNOWN
    assert run.manifest_fields()["provider_http_attempts"] is UNKNOWN
    assert C.CALIBRATION_HTTP_ATTEMPT_CEILING == 20
    assert run.provider_http_attempts != C.CALIBRATION_HTTP_ATTEMPT_CEILING


def test_82_unknown_is_not_zero():
    assert UNKNOWN != 0
    assert isinstance(UNKNOWN, str)


def test_83_temperature_none_means_not_sent_not_zero(monkeypatch):
    """`temperature=None` 必须意味着 **NOT SENT**,不是 `0.0`。"""
    captured = _recording_chat_model(monkeypatch)
    cfg = C.default_calibration_config()
    build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        allow_network=True,
        **cfg.model_kwargs(),
    )
    assert captured.get("temperature") is None
    assert "temperature" not in captured
    assert cfg.recorded_temperature() is None
    assert cfg.request_shape()["temperature_present"] is False


# ===========================================================================
# 11. 原始 usage 保留(§11)
# ===========================================================================


def test_84_deepseek_cache_counters_survive_in_the_raw_usage():
    """DeepSeek 的**顶层**缓存计数器必须被保留,且**不被静默转成 0**。"""
    raw = deepseek_shaped_raw_usage(
        prompt_tokens=100, completion_tokens=7, prompt_cache_hit_tokens=64
    )
    message = synthetic_ai_message(content="OK", raw_usage=raw)
    captured = raw_token_usage_of(message)
    assert captured[DEEPSEEK_CACHE_HIT_KEY] == 64
    assert captured["prompt_cache_miss_tokens"] == 36
    assert captured[DEEPSEEK_CACHE_HIT_KEY] != 0


def test_85_normalized_usage_does_not_carry_deepseek_cache_counters():
    """归一化映射只认 OpenAI 形状 —— DeepSeek 的顶层计数**不会**出现在里面。

    这不是缺陷,而是事实。测试把它写下来,是为了让"原始字典必须保留"
    这条要求有一个明确的对照。
    """
    raw = deepseek_shaped_raw_usage(
        prompt_tokens=100, completion_tokens=7, prompt_cache_hit_tokens=64
    )
    normalized = normalize_openai_shaped(raw)
    assert "cache_read" not in normalized["input_token_details"]
    assert normalized["input_token_details"] == {}


def test_86_openai_shaped_usage_control_group():
    """对照组:同样的缓存命中数,OpenAI 形状**会**被归一化读到。"""
    raw = openai_shaped_raw_usage(prompt_tokens=100, completion_tokens=7, cached_tokens=64)
    normalized = normalize_openai_shaped(raw)
    assert normalized["input_token_details"]["cache_read"] == 64
    assert raw[OPENAI_CACHED_TOKENS_KEY]["cached_tokens"] == 64


def test_87_calibration_captures_raw_usage_without_replacing_normalized_usage(workdir, representative_unit, tool_contract):
    """捕获原始用量**不得**顶替既有归一化字段 —— 两者并存。"""
    raw = deepseek_shaped_raw_usage(
        prompt_tokens=100, completion_tokens=5, prompt_cache_hit_tokens=64
    )
    normalized = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}
    invoker = make_invoker(
        c0a_message=synthetic_ai_message(
            content="OK", raw_usage=raw, normalized_usage=normalized
        )
    )
    deps = build_deps(
        representative_unit=representative_unit, tool_contract=tool_contract, invoker=invoker
    )
    evidence = _c0a_outcome(workdir=workdir, deps=deps).evidence
    assert evidence["raw_token_usage"][DEEPSEEK_CACHE_HIT_KEY] == 64
    assert evidence["normalized_usage"] == normalized


def test_88_no_cost_computation_is_performed(workdir, representative_unit, tool_contract):
    """暂不实现 cost 计算 —— 证据里不得出现成本字段。"""
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    payload = json.dumps(run.manifest_fields(), ensure_ascii=False)
    for forbidden in ("cost", "price", "usd", "billing"):
        assert forbidden not in payload.lower()


# ===========================================================================
# 12. 凭据安全(§12)
# ===========================================================================


def test_89_credentials_never_appear_in_evidence_or_manifest(workdir, representative_unit, tool_contract):
    """合成凭据占位符**绝不**出现在任何产物里。"""
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    blob = json.dumps(
        {
            "manifest": run.manifest_fields(),
            "records": [
                {
                    "stage": record.stage.value,
                    "reason": record.reason,
                    "evidence": record.evidence,
                }
                for record in run.records
            ],
        },
        ensure_ascii=False,
        default=str,
    )
    assert SYNTHETIC_API_KEY not in blob
    assert "api_key" not in blob
    assert "Authorization" not in blob
    assert "DEEPSEEK_API_KEY" not in blob


def test_90_request_digest_does_not_depend_on_credentials():
    """摘要**只**覆盖凭据无关字段 —— 否则摘要本身就成了凭据的旁路。"""
    cfg = C.default_calibration_config()
    shape = cfg.request_shape()
    assert shape["api_key_present"] is False
    digest = cfg.request_shape_digest()
    assert isinstance(digest, str) and len(digest) == 64
    # 摘要稳定:同一配置多次派生得到同一结果。
    assert digest == C.default_calibration_config().request_shape_digest()


def test_91_run_manifest_passes_the_frozen_secret_guard(workdir, representative_unit, tool_contract):
    from app.evaluation.llm.raw import find_secret_patterns

    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    assert find_secret_patterns(run.manifest_fields()) == []
    for record in run.records:
        assert find_secret_patterns(record.evidence) == []


def test_92_full_credentialed_url_is_never_persisted(workdir, representative_unit, tool_contract):
    """完整(可能内嵌凭据的)URL 不得被持久化 —— 只留主机名摘要。"""
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    blob = json.dumps(run.manifest_fields(), ensure_ascii=False, default=str)
    assert "https://" not in blob
    assert C.CALIBRATION_BASE_URL not in blob
    # 身份侧的 host 摘要存在,而完整 URL 不存在 —— 两者不是同一个东西。
    assert run.config.request_shape_digest()


# ===========================================================================
# 13. 离线出口守卫(§13)
# ===========================================================================


def test_93_calibration_package_imports_no_provider_client():
    """实现侧的护栏:`app/evaluation/calibration/*.py` 不得 import 任何 provider 客户端。"""
    package = Path(__file__).resolve().parents[2] / "app" / "evaluation" / "calibration"
    forbidden = ("openai", "langchain_openai", "anthropic", "httpx", "requests", "aiohttp")
    checked = sorted(package.glob("*.py"))
    assert checked, "护栏没有覆盖到任何文件 —— 那是恒真的护栏"
    for path in checked:
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in source, (path.name, name)
            assert f"from {name}" not in source, (path.name, name)


def test_94_guardrail_collector_has_positive_and_negative_controls():
    """护栏本身要有牙:正对照必须被检出,负对照不得误报。"""
    package = Path(__file__).resolve().parents[2] / "app" / "evaluation" / "calibration"
    forbidden = ("openai", "langchain_openai", "anthropic", "httpx", "requests", "aiohttp")

    def scan(source: str) -> list[str]:
        return [
            name
            for name in forbidden
            if f"import {name}" in source or f"from {name}" in source
        ]

    assert scan("from langchain_openai import ChatOpenAI\n") == ["langchain_openai"]
    assert scan("import httpx\n") == ["httpx"]
    assert scan("from app.evaluation.llm.identity import EndpointCategory\n") == []
    assert package.exists()


def test_95_the_egress_guard_has_teeth():
    """守卫必须在**发生处**抛错,而不是只在事后检查。"""
    guard = NetworkEgressGuard(strict=True)
    with guard:
        with pytest.raises(NetworkEgressError):
            # `sys.audit` 会把事件交给审计钩子,**但不执行**该操作 ——
            # 因此这个探针零真实网络流量。
            sys.audit("socket.connect", None, (DELIBERATE_EGRESS_HOST, 443))
    assert guard.clean is False
    assert guard.egress_events
    assert {event.detail for event in guard.egress_events} == {DELIBERATE_EGRESS_HOST}


def test_96_an_accidental_egress_never_reports_zero_real_calls(workdir, representative_unit, tool_contract):
    """若实现意外尝试出口,运行**不得**声称 `real_provider_calls == 0`。"""

    def egress_on_invoke(request):
        sys.audit("socket.connect", None, (DELIBERATE_EGRESS_HOST, 443))

    invoker = make_invoker(on_invoke=egress_on_invoke)
    deps = build_deps(
        representative_unit=representative_unit, tool_contract=tool_contract, invoker=invoker
    )
    run = run_full(workdir=workdir, deps=deps)
    evidence = C.offline_evidence(run)
    assert evidence["offline"] is False
    assert evidence["real_provider_calls"] is UNKNOWN
    assert evidence["api_credits_consumed"] is UNKNOWN
    assert run.halted_at is S.Stage.C0A
    assert run.passed_all is False


def test_97_clean_run_reports_zero_real_calls(workdir, representative_unit, tool_contract):
    deps = build_deps(representative_unit=representative_unit, tool_contract=tool_contract)
    run = run_full(workdir=workdir, deps=deps)
    evidence = C.offline_evidence(run)
    assert evidence["offline"] is True
    assert evidence["egress_events"] == 0
    assert evidence["real_provider_calls"] == 0
    assert evidence["api_credits_consumed"] == 0


def test_98_harness_runs_inside_a_strict_guard_by_default(workdir):
    harness = C.CalibrationHarness(experiment_id=EXPERIMENT_ID, workdir_root=workdir)
    assert harness.guard.strict is True


# ===========================================================================
# 14. 反同义反复:判定语义本身
# ===========================================================================


@pytest.mark.parametrize(
    "verdict",
    [
        C.StageVerdict.INCONCLUSIVE,
        C.StageVerdict.FAIL,
        C.StageVerdict.ABORT,
    ],
)
def test_99_every_non_pass_verdict_blocks_progression(verdict):
    assert verdict.blocks_progression is True
    assert verdict.is_pass is False
    with pytest.raises(C.StageBlocked):
        C.assert_stage_passed(
            C.Verdict(verdict, "synthetic"), stage="C0b", next_stage="C1"
        )


def test_100_pass_alone_advances():
    assert C.StageVerdict.PASS.blocks_progression is False
    C.assert_stage_passed(C.Verdict(C.StageVerdict.PASS, "ok"), stage="C0b", next_stage="C1")


def test_101_a_verdict_without_a_reason_cannot_exist():
    """没有理由的判定无法复核 —— 必须构造不出来。"""
    with pytest.raises(ValueError):
        C.Verdict(C.StageVerdict.PASS, "   ")


def test_102_stage_invocation_ceiling_exceeded_is_a_budget_error():
    """阶段上限越界必须继承 `BudgetExceeded`,否则会被适配器兜底吞掉。"""
    assert issubclass(S.StageInvocationCeilingExceeded, BudgetExceeded)
    assert issubclass(S.StageInvocationCeilingExceeded, AssertionError)
