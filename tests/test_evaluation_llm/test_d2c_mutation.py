"""Phase 9.2-D-2c **变异 / 反同义反复**测试。

这个文件回答一个问题:

    上面那套测试,到底有没有牙?

做法是给每一条**坏行为**构造一个**脚本化变异**,然后跑同一批断言。若断言
在变异下依然通过,那条断言就是**恒真的** —— 它没有测到任何东西,而它会
让报告看起来完全正常。

十一个变异(前十个来自 D-2c 实现契约,第十一个是阴性对照):

    01  temperature=None 被静默替换成 0.0
    02  max_retries=0 被丢弃
    03  streaming=False 被丢弃
    04  thinking-disable 的 extra_body 被丢弃
    05  provider 自报模型被请求模型顶替
    06  原始 token_usage 被丢弃
    07  标定 id 被允许进入试点命名空间
    08  C0b 的 INCONCLUSIVE 被错误地推进到 C1
    09  C0b 判定只用 finish_reason
    10  标定记录成为试点完成资格
    11  阴性对照:无操作变异**不得**被判为"检出"

F10 追加的五个变异(12–16)
--------------------------

    12  请求体里没有 `tools` 字段也照样判通过(F10 前置核验被拆掉)
    13  生产绑定的工具名与生产工具集漂移(两个绑定点各自为政)
    14  观测侧从声明回填(声明-观测一致因此变成恒真)
    15  验证契约不由绑定派生(同一件事有了两个事实来源)
    16  digest 覆盖缺口标志被移除(缺口变成"不存在")

全程离线:零真实 provider 调用、零凭据、零网络出口。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
from langchain_core.messages import ToolMessage

from app.evaluation import calibration as C
from app.evaluation.calibration import config as cal_config
from app.evaluation.calibration import namespace as cal_namespace
from app.evaluation.calibration import stages as S
from app.evaluation.calibration.synthetic import (
    DEEPSEEK_CACHE_HIT_KEY,
    SYNTHETIC_PROVIDER_MODEL,
    SyntheticProviderModel,
    SyntheticTurn,
    deepseek_shaped_raw_usage,
    synthetic_ai_message,
    synthetic_tool_call,
)
from app.evaluation.llm import raw as llm_raw
from app.evaluation.llm.budget import BudgetGovernor
from app.evaluation.llm.dataset import LLM_TASKS, build_datasets
from app.evaluation.llm.raw import (
    ForeignExperimentRecord,
    RawRecord,
    resume_eligibility,
)
from app.evaluation.real_provider import build_chat_model

#: 明显是合成物的凭据占位符。**不是**任何真实密钥。
SYNTHETIC_API_KEY = "sk-d2c-mutation-placeholder-000000000000"

CAP = 16
DEMANDED = 2000
EXPERIMENT_ID = "d2c-cal-mutation-20260923"


# ---------------------------------------------------------------------------
# 探针(与聚焦测试里的断言同源)
# ---------------------------------------------------------------------------


def _captured_kwargs(cfg: C.CalibrationConfig) -> dict:
    """把交给 SDK 的 kwargs 录下来(**不构造真实客户端**)。"""
    import langchain_openai

    captured: dict = {}

    class _Recorder:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(langchain_openai, "ChatOpenAI", _Recorder)
        build_chat_model(
            candidate=cfg.provider_candidate(),
            api_key=SYNTHETIC_API_KEY,
            allow_network=True,
            **cfg.model_kwargs(),
        )
    return captured


def probe_temperature_not_set() -> None:
    """`NOT_SET` 必须意味着**不发该字段**,且记录侧同为 `None`。"""
    cfg = C.default_calibration_config()
    kwargs = _captured_kwargs(cfg)
    assert "temperature" not in kwargs, "NOT_SET 被静默替换成了一个具体数值"
    assert cfg.recorded_temperature() is None
    assert cfg.request_shape()["temperature_present"] is False


def probe_max_retries_explicit() -> None:
    cfg = C.default_calibration_config()
    assert _captured_kwargs(cfg)["max_retries"] == 0


def probe_streaming_explicit() -> None:
    cfg = C.default_calibration_config()
    assert _captured_kwargs(cfg)["streaming"] is False


def probe_thinking_disable_body() -> None:
    cfg = C.default_calibration_config()
    captured = _captured_kwargs(cfg)
    assert captured["extra_body"]["thinking"] == {"type": "disabled"}


def probe_provider_model_separation() -> None:
    cfg = C.CalibrationConfig()
    evidence = S.c0a_evidence(
        request=S.CalibrationRequest(
            stage=S.Stage.C0A,
            messages=(),
            model_kwargs={"extra_body": cfg.thinking_body()},
            output_token_cap=cfg.output_token_cap,
        ),
        result=S.InvocationResult(
            message=synthetic_ai_message(
                content="OK",
                raw_usage=deepseek_shaped_raw_usage(
                    prompt_tokens=10, completion_tokens=2, prompt_cache_hit_tokens=4
                ),
            ),
            request_payload={"thinking": {"type": "disabled"}},
        ),
        config=cfg,
    )
    assert evidence["requested_model"] == cfg.requested_model
    assert evidence["provider_reported_model"] == SYNTHETIC_PROVIDER_MODEL
    assert evidence["requested_model"] != evidence["provider_reported_model"]


def probe_raw_usage_preserved() -> None:
    cfg = C.CalibrationConfig()
    evidence = S.c0a_evidence(
        request=S.CalibrationRequest(
            stage=S.Stage.C0A,
            messages=(),
            model_kwargs={"extra_body": cfg.thinking_body()},
            output_token_cap=cfg.output_token_cap,
        ),
        result=S.InvocationResult(
            message=synthetic_ai_message(
                content="OK",
                raw_usage=deepseek_shaped_raw_usage(
                    prompt_tokens=100, completion_tokens=2, prompt_cache_hit_tokens=64
                ),
            ),
            request_payload={"thinking": {"type": "disabled"}},
        ),
        config=cfg,
    )
    raw = evidence["raw_token_usage"]
    assert raw is not None, "原始 token_usage 被丢弃"
    assert raw[DEEPSEEK_CACHE_HIT_KEY] == 64
    assert raw[DEEPSEEK_CACHE_HIT_KEY] != 0


def _raises(exc_type, fn) -> bool:
    """`fn()` 是否抛出了 `exc_type`。

    刻意**不用** `pytest.raises`:它在断言失败时抛出的 `Failed` 继承自
    `BaseException` 而非 `Exception`,会从"检出"判定里逃出去,把一次
    **成功的检出**表现成一个**测试错误**。
    """
    try:
        fn()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001 —— 抛了别的异常也算"没有抛期望的那个"
        return False
    return False


def probe_namespace_disjoint() -> None:
    """命名空间判据必须是"属于哪一类",不是"看起来像哪个"。

    刻意走 `cal_namespace.` 而不是包级别名:真实代码里
    `harness.py` 也是通过模块属性查找调用它的,因此这里才能检验到
    真正的守卫实现(包级别名是一个**绑定了函数对象**的副本,替换
    模块属性不会影响它)。
    """
    assert _raises(
        C.CalibrationNamespaceError,
        lambda: cal_namespace.assert_pilot_experiment_id("d2c-cal-c0a-20260923"),
    ), "标定 id 被试点命名空间接受了"
    assert _raises(
        C.CalibrationNamespaceError,
        lambda: cal_namespace.assert_calibration_experiment_id("d2d-pilot-run1"),
    ), "试点 id 被标定命名空间接受了"
    C.assert_not_pilot_eligible(artifact_experiment_id="d2c-cal-c0a-20260923")


def probe_pilot_eligibility() -> None:
    C.assert_not_pilot_eligible(artifact_experiment_id="d2c-cal-c0a-20260923")
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
    assert _raises(
        ForeignExperimentRecord,
        lambda: resume_eligibility(record, experiment_id="d2d-pilot-run1"),
    ), "跨实验的记录被当成了本实验的完成状态"


# ---- 门与判定探针(需要真实跑一次标定) ----


def _representative_deps(tmp_path, *, c0b_message=None, invoker=None):
    """装配一份完整的合成依赖(与聚焦测试同构)。"""
    datasets = build_datasets(tmp_path / "fixtures")
    task = LLM_TASKS[0]
    paths = datasets[task.dataset_variant]
    unit = S.EvaluationUnit(
        task=task, baseline_label="B2'", behavior="GOOD", dataset_paths=paths
    )
    contract = S.production_tool_contract()
    cfg = C.default_calibration_config()

    async def default_invoke(request):
        payload = {"model": cfg.requested_model}
        payload.update(request.model_kwargs.get("extra_body") or {})
        # 合成传输按**声明**折算 `tools` 字段 —— 与 `bind_declared_tools()`
        # 走同一个接缝(`S.bindable_tools`),因此"声明 → 绑定 → 线上"这条链
        # 在合成传输里同样成立。
        binding = request.tool_binding
        if binding is not None and S.bindable_tools(binding):
            payload["tools"] = [
                {"type": "function", "function": {"name": getattr(tool, "name", "")}}
                for tool in S.bindable_tools(binding)
            ]
        if request.stage is S.Stage.C0A:
            message = synthetic_ai_message(content="OK")
        elif request.stage is S.Stage.C0B:
            message = c0b_message or synthetic_ai_message(
                content="x" * 64,
                finish_reason="length",
                normalized_usage={
                    "input_tokens": 20,
                    "output_tokens": request.output_token_cap,
                    "total_tokens": 20 + request.output_token_cap,
                },
            )
        elif request.stage is S.Stage.C1:
            message = synthetic_ai_message(
                content="",
                tool_calls=[
                    synthetic_tool_call(
                        name="analyze_risk_tool",
                        args={"indicator": "203.0.113.66"},
                        call_id="c1-1",
                    )
                ],
                finish_reason="tool_calls",
            )
        else:  # pragma: no cover
            raise AssertionError(request.stage)
        return S.InvocationResult(
            message=message, request_payload=payload, http_attempts=1
        )

    deps = S.StageDeps(
        invoke=invoker or default_invoke,
        tool_contract=contract,
        tool_binding=S.production_tool_binding(),
        roundtrip=[
            synthetic_ai_message(
                content="",
                tool_calls=[
                    synthetic_tool_call(
                        name="analyze_risk_tool",
                        args={"indicator": "203.0.113.66"},
                        call_id="rt-1",
                    )
                ],
                finish_reason="tool_calls",
            ),
            ToolMessage(content="{}", tool_call_id="rt-1"),
            synthetic_ai_message(content="结论:高风险"),
        ],
        graph_model=SyntheticProviderModel(
            turns=(
                SyntheticTurn(
                    content="",
                    tool_calls=(
                        synthetic_tool_call(
                            name="no_such_tool", args={"x": 1}, call_id="c3-1"
                        ),
                    ),
                    finish_reason="tool_calls",
                ),
                SyntheticTurn(content="结论:已处理。"),
            )
        ),
        unit_model=SyntheticProviderModel(
            turns=(
                SyntheticTurn(
                    content="",
                    tool_calls=(
                        synthetic_tool_call(
                            name="analyze_risk_tool",
                            args={
                                "indicator": task.indicator,
                                "logs_path": paths["logs"],
                                "intel_path": paths["intel"],
                            },
                            call_id="c4-1",
                        ),
                    ),
                    finish_reason="tool_calls",
                ),
                SyntheticTurn(content="结论:已评估。"),
            )
        ),
        evaluation_unit=unit,
    )
    return deps


def _run(tmp_path, *, c0b_message=None, runners=None, experiment_id=EXPERIMENT_ID):
    import asyncio

    deps = _representative_deps(tmp_path, c0b_message=c0b_message)
    harness = C.CalibrationHarness(
        experiment_id=experiment_id,
        workdir_root=tmp_path / "wd",
        deps=deps,
        runners=runners,
    )
    return asyncio.run(harness.run())


def probe_inconclusive_blocks(tmp_path_factory) -> None:
    tmp_path = tmp_path_factory.mktemp("mut-gate")
    hits: list[str] = []

    def make(stage):
        async def runner(ctx, deps):
            hits.append(stage.value)
            return await S.DEFAULT_STAGE_RUNNERS[stage](ctx, deps)
        return runner

    runners = {stage: make(stage) for stage in S.STAGE_ORDER}
    run = _run(
        tmp_path,
        c0b_message=synthetic_ai_message(
            content="short",
            finish_reason="stop",
            normalized_usage={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        ),
        runners=runners,
    )
    assert run.records[1].verdict is C.StageVerdict.INCONCLUSIVE
    assert hits == ["C0a", "C0b"], "INCONCLUSIVE 之后仍有阶段被运行"
    assert run.skipped_stages == (S.Stage.C1, S.Stage.C2, S.Stage.C3, S.Stage.C4)


def probe_finish_reason_is_not_the_criterion() -> None:
    """`finish_reason` 单独不得决定任何判定。"""

    def verdict_for(finish_reason: str, output_tokens: int):
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
                    finish_reason=finish_reason,
                    normalized_usage={
                        "input_tokens": 5,
                        "output_tokens": output_tokens,
                        "total_tokens": 5 + output_tokens,
                    },
                ),
                request_payload={"max_tokens": CAP},
            ),
            provider_rejected=False,
            provider_rejected_reason=None,
            config=cfg,
        )
        return S.c0b_evidence_verdict(evidence).verdict

    # `stop` 单独不得推出 FAIL。
    assert verdict_for("stop", CAP) is not C.StageVerdict.FAIL
    # `length` 单独不得推出 PASS。
    assert verdict_for("length", 3) is not C.StageVerdict.PASS


# ---- F10 探针:C1 工具绑定路径(纯离线,不需要跑门) ----

#: 一个**合法**的 tool call(参数符合 `analyze_risk_tool` 的契约)。
_F10_LEGAL_CALL = synthetic_tool_call(
    name="analyze_risk_tool", args={"indicator": "1.2.3.4"}, call_id="f10-c1-1"
)


def _f10_request(binding: S.ToolBinding) -> S.CalibrationRequest:
    cfg = C.default_calibration_config()
    return S.CalibrationRequest(
        stage=S.Stage.C1,
        messages=(),
        model_kwargs=cfg.model_kwargs(),
        output_token_cap=cfg.output_token_cap,
        tool_binding=binding,
    )


def _f10_wire_payload(binding: S.ToolBinding) -> dict:
    """声明 `binding` 时合成传输上"应当"出现的请求体(**按声明折算**)。"""
    cfg = C.default_calibration_config()
    payload: dict = {"model": cfg.requested_model}
    payload.update(cfg.model_kwargs()["extra_body"])
    if binding.tools:
        payload["tools"] = [
            {"type": "function", "function": {"name": getattr(tool, "name", "")}}
            for tool in S.bindable_tools(binding)
        ]
    return payload


def _f10_evidence(
    *,
    binding: S.ToolBinding,
    payload,
    contract=None,
    supplied=None,
) -> dict:
    message = synthetic_ai_message(
        content="", tool_calls=[dict(_F10_LEGAL_CALL)], finish_reason="tool_calls"
    )
    return S.c1_evidence(
        request=_f10_request(binding),
        result=S.InvocationResult(message=message, request_payload=payload),
        contract=contract or binding.as_contract(),
        binding=binding,
        config=C.default_calibration_config(),
        supplied_contract=supplied,
    )


def probe_c1_missing_tools_field_is_a_failure() -> None:
    """声明了工具而请求体里没有 `tools` 字段 ⇒ **FAIL**。

    工具契约从未到达 provider,所以"模型给的 tool call 形状合不合法"
    这个问题根本没有被问到 —— 它绝不能因此得到一个"通过"。
    """
    binding = S.production_tool_binding()
    payload = _f10_wire_payload(binding)
    del payload["tools"]
    evidence = _f10_evidence(binding=binding, payload=payload)
    assert evidence["tools_field_present_in_payload"] is False
    assert evidence["binding_agreement"] is False
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is C.StageVerdict.FAIL, (
        f"工具契约没发出去,判定却是 {verdict.verdict.value}:{verdict.reason}"
    )


def probe_c1_production_binding_names_are_not_drifted() -> None:
    """生产绑定的工具名必须与生产工具集逐个一致。

    C1 与 C4 是**两个**绑定点;今天同源是巧合而非强制。这条断言把"同源"
    从约定变成可核验的事实。
    """
    from app.tools import DEFAULT_TOOLS

    expected = tuple(tool.name for tool in DEFAULT_TOOLS)
    binding = S.production_tool_binding()
    assert binding.tool_names == expected, (
        f"绑定声明的工具 {binding.tool_names!r} 与生产工具集 {expected!r} 漂移了"
    )
    assert S.production_tool_contract().allowed_tool_names == expected


def probe_c1_observation_is_never_filled_from_the_declaration() -> None:
    """观测侧只从请求体读。回填会让"声明-观测一致"变成一个恒真的判断。"""
    binding = S.production_tool_binding()
    payload = _f10_wire_payload(binding)
    del payload["tools"]
    evidence = _f10_evidence(binding=binding, payload=payload)
    assert evidence["observed_tools_in_payload"] == [], (
        "观测侧被声明回填了 —— 一致性核对因此失去意义"
    )
    assert evidence["observed_tools_in_payload"] != evidence["tool_binding_declared"]
    assert evidence["binding_agreement"] is False
    assert S.c1_evidence_verdict(evidence).verdict is C.StageVerdict.FAIL


def probe_c1_verifier_contract_is_derived_from_the_binding() -> None:
    """验证契约与绑定必须同源 —— 否则被校验的不是发出去的那一份。"""
    binding = S.production_tool_binding()
    supplied = S.production_tool_contract()
    assert supplied.allowed_tool_names == binding.tool_names
    evidence = _f10_evidence(
        binding=binding, payload=_f10_wire_payload(binding), supplied=supplied
    )
    assert evidence["contract_agreement"] is True
    assert evidence["supplied_contract_agreement"] is True
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is C.StageVerdict.PASS, verdict.reason


def probe_c1_digest_coverage_gap_is_recorded() -> None:
    """`request_digest` **不**覆盖工具绑定 —— 缺口必须被记账,不能消失。"""
    binding = S.production_tool_binding()
    evidence = _f10_evidence(binding=binding, payload=_f10_wire_payload(binding))
    assert "request_digest_covers_tool_binding" in evidence, (
        "覆盖缺口标志消失了 —— 缺口会被读成'没有缺口'"
    )
    assert evidence["request_digest_covers_tool_binding"] is False
    assert S.REQUEST_DIGEST_COVERS_TOOL_BINDING is False


# ---------------------------------------------------------------------------
# 变异
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mutation:
    """一条**坏行为**的脚本化版本。"""

    key: str
    name: str
    apply: Callable[[pytest.MonkeyPatch], None]
    probe: Callable[..., None]
    note: str
    needs_tmp: bool = False


def _drop_from_model_kwargs(mp: pytest.MonkeyPatch, key: str) -> None:
    original = cal_config.CalibrationConfig.model_kwargs

    def patched(self, *, output_token_cap=None):
        kwargs = original(self, output_token_cap=output_token_cap)
        kwargs.pop(key, None)
        return kwargs

    mp.setattr(cal_config.CalibrationConfig, "model_kwargs", patched)


def _mutate_temperature(mp: pytest.MonkeyPatch) -> None:
    def model_kwargs(self, *, output_token_cap=None):
        kwargs = cal_config.CalibrationConfig.__dict__["model_kwargs"](
            self, output_token_cap=output_token_cap
        )
        kwargs["temperature"] = 0.0
        return kwargs

    mp.setattr(cal_config.CalibrationConfig, "model_kwargs", model_kwargs)
    mp.setattr(
        cal_config.CalibrationConfig, "recorded_temperature", lambda self: 0.0
    )


def _mutate_missing_tools_field_accepted(mp: pytest.MonkeyPatch) -> None:
    """把 F10 的前置核验整段拆掉:直接落到形状判定。"""
    mp.setattr(
        S,
        "c1_evidence_verdict",
        lambda evidence: S.c1_verdict(
            evidence["findings"], call_count=int(evidence.get("tool_call_count") or 0)
        ),
    )


def _mutate_production_binding_drift(mp: pytest.MonkeyPatch) -> None:
    """让生产绑定少一个工具 —— 两个绑定点各自为政。"""
    original = S.production_tool_binding
    mp.setattr(
        S,
        "production_tool_binding",
        lambda: S.ToolBinding(tools=original().tools[:-1], source="mutated"),
    )


def _mutate_declaration_observation_fallback(mp: pytest.MonkeyPatch) -> None:
    """观测不到 `tools` 字段时,拿**声明**回填。"""
    original = S.c1_evidence

    def patched(**kwargs):
        evidence = original(**kwargs)
        if evidence["tools_field_present_in_payload"] is False:
            evidence["tools_field_present_in_payload"] = True
            evidence["observed_tools_in_payload"] = list(
                evidence["tool_binding_declared"]
            )
            evidence["binding_agreement"] = True
        return evidence

    mp.setattr(S, "c1_evidence", patched)


def _mutate_contract_divergence(mp: pytest.MonkeyPatch) -> None:
    """生产契约从**别处**派生,与绑定不同源。"""
    original = S.production_tool_binding
    mp.setattr(
        S,
        "production_tool_contract",
        lambda: S.ToolContract(
            allowed_tool_names=original().tool_names[:-1], arg_schemas={}
        ),
    )


def _mutate_digest_coverage_flag_removal(mp: pytest.MonkeyPatch) -> None:
    """把 digest 覆盖缺口标志从证据里抹掉。"""
    original = S.c1_evidence

    def patched(**kwargs):
        evidence = original(**kwargs)
        evidence.pop("request_digest_covers_tool_binding", None)
        return evidence

    mp.setattr(S, "c1_evidence", patched)


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        key="01",
        name="temperature=None 被静默替换成 0.0",
        apply=_mutate_temperature,
        probe=probe_temperature_not_set,
        note="NOT_SET 被记成一个具体数值 = 伪造'我们控制了采样'",
    ),
    Mutation(
        key="02",
        name="max_retries=0 被丢弃",
        apply=lambda mp: _drop_from_model_kwargs(mp, "max_retries"),
        probe=probe_max_retries_explicit,
        note="SDK 重试对预算不可见,丢弃它会让一次调用最多发 3 个物理请求",
    ),
    Mutation(
        key="03",
        name="streaming=False 被丢弃",
        apply=lambda mp: _drop_from_model_kwargs(mp, "streaming"),
        probe=probe_streaming_explicit,
        note="依赖 SDK 默认值会让请求形状无法溯源",
    ),
    Mutation(
        key="04",
        name="thinking-disable 的 extra_body 被丢弃",
        apply=lambda mp: _drop_from_model_kwargs(mp, "extra_body"),
        probe=probe_thinking_disable_body,
        note="思考模式未被关闭,标定就不是 non-thinking 路径上的标定",
    ),
    Mutation(
        key="05",
        name="provider 自报模型被请求模型顶替",
        apply=lambda mp: mp.setattr(
            S,
            "provider_reported_model_id_of",
            lambda message: cal_config.CALIBRATION_REQUESTED_MODEL,
        ),
        probe=probe_provider_model_separation,
        note="两者合并后,'我们请求了什么'与'provider 用了什么'无法区分",
    ),
    Mutation(
        key="06",
        name="原始 token_usage 被丢弃",
        apply=lambda mp: mp.setattr(S, "raw_token_usage_of", lambda message: None),
        probe=probe_raw_usage_preserved,
        note="DeepSeek 的顶层缓存计数器只存活在原始字典里,丢弃即永久丢失",
    ),
    Mutation(
        key="07",
        name="标定 id 被允许进入试点命名空间",
        apply=lambda mp: mp.setattr(
            cal_namespace, "assert_pilot_experiment_id", lambda value: str(value)
        ),
        probe=probe_namespace_disjoint,
        note="命名空间判据从'属于哪一类'退化成'看起来像哪个'",
    ),
    Mutation(
        key="08",
        name="C0b 的 INCONCLUSIVE 被错误推进到 C1",
        apply=lambda mp: mp.setattr(S.StageOutcome, "is_pass", property(lambda self: True)),
        probe=probe_inconclusive_blocks,
        note="'没能证明'被当成'通过',后续阶段的证据因此建立在未成立的结论上",
        needs_tmp=True,
    ),
    Mutation(
        key="09",
        name="C0b 判定只用 finish_reason",
        apply=lambda mp: mp.setattr(
            S,
            "c0b_evidence_verdict",
            lambda evidence: (
                C.Verdict(
                    C.StageVerdict.FAIL, "finish_reason 是 stop"
                )
                if evidence.get("finish_reason") == "stop"
                else C.Verdict(C.StageVerdict.PASS, "finish_reason 不是 stop")
            ),
        ),
        probe=probe_finish_reason_is_not_the_criterion,
        note="把无效蕴含当成判据 —— stop 推不出'上限被忽略'",
    ),
    Mutation(
        key="10",
        name="标定记录成为试点完成资格",
        apply=lambda mp: mp.setattr(
            llm_raw, "resume_eligibility", lambda record, *, experiment_id: llm_raw.ResumeEligibility.FROZEN
        ),
        probe=probe_pilot_eligibility,
        note="跨实验的完成状态被当成完成状态,某个单元会静默地永不被执行",
    ),
    Mutation(
        key="11",
        name="阴性对照:无操作变异",
        apply=lambda mp: None,
        probe=probe_temperature_not_set,
        note="若这个被判为'检出',说明变异工装本身在误报",
    ),
    Mutation(
        key="12",
        name="请求体里没有 tools 字段也照样判通过",
        apply=_mutate_missing_tools_field_accepted,
        probe=probe_c1_missing_tools_field_is_a_failure,
        note="工具契约从未到达 provider,而'形状合不合法'却得到了一个通过",
    ),
    Mutation(
        key="13",
        name="生产绑定与生产工具集漂移",
        apply=_mutate_production_binding_drift,
        probe=probe_c1_production_binding_names_are_not_drifted,
        note="C1 与 C4 两个绑定点各自为政,而两边看起来都正常",
    ),
    Mutation(
        key="14",
        name="观测侧从声明回填",
        apply=_mutate_declaration_observation_fallback,
        probe=probe_c1_observation_is_never_filled_from_the_declaration,
        note="'声明-观测一致'退化成恒真 —— 它正是'工具发出去没有'的唯一答案来源",
    ),
    Mutation(
        key="15",
        name="验证契约不由绑定派生",
        apply=_mutate_contract_divergence,
        probe=probe_c1_verifier_contract_is_derived_from_the_binding,
        note="被校验的契约不是被发出去的那一个,而校验结果照常给出",
    ),
    Mutation(
        key="16",
        name="digest 覆盖缺口标志被移除",
        apply=_mutate_digest_coverage_flag_removal,
        probe=probe_c1_digest_coverage_gap_is_recorded,
        note="缺口一旦不被记账,就会被读成'request_digest 覆盖了整条请求形状'",
    ),
)


# ---------------------------------------------------------------------------
# 变异工装
# ---------------------------------------------------------------------------


def _detect(mutation: Mutation, tmp_path_factory) -> bool:
    """应用变异 → 跑探针。探针失败 ⇒ 检出。

    捕获 `BaseException` 而不是 `Exception`:pytest 的断言失败异常
    (`Failed`)继承自 `BaseException`,若只捕获 `Exception`,一次**成功的
    检出**会从判定里逃出去,表现成测试错误 —— 那会让检出率虚低。
    `KeyboardInterrupt` / `SystemExit` 原样放行。
    """
    with pytest.MonkeyPatch.context() as mp:
        mutation.apply(mp)
        try:
            if mutation.needs_tmp:
                mutation.probe(tmp_path_factory)
            else:
                mutation.probe()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # noqa: BLE001 —— 见 docstring
            return True
    return False


def test_d2c_mutation_00_the_probes_pass_without_mutation(tmp_path_factory):
    """先证明探针本身在**未变异**时全部通过 —— 否则"检出"毫无意义。

    注意这里**不**应用变异:它检验的是探针本身。若某条探针在未变异时
    就失败,那它不是一条有效断言,而它的"检出"也只是别的原因造成的失败。
    """
    for mutation in MUTATIONS:
        try:
            if mutation.needs_tmp:
                mutation.probe(tmp_path_factory)
            else:
                mutation.probe()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 —— 失败原因需要原样报告
            pytest.fail(
                f"探针 {mutation.key}({mutation.name})在未变异时就失败了:"
                f"{type(exc).__name__}: {exc}"
            )


def test_d2c_mutation_01_the_negative_control_is_not_reported_as_detected(tmp_path_factory):
    """阴性对照必须**不**被判为检出,否则工装在误报。"""
    negative = next(m for m in MUTATIONS if m.key == "11")
    assert _detect(negative, tmp_path_factory) is False


@pytest.mark.parametrize(
    "mutation", [m for m in MUTATIONS if m.key != "11"], ids=lambda m: f"{m.key}-{m.name}"
)
def test_d2c_mutation_02_each_mutation_is_detected(mutation, tmp_path_factory):
    """每一条坏行为都必须被套件检出。"""
    assert _detect(mutation, tmp_path_factory), (
        f"变异 {mutation.key}({mutation.name})未被检出 —— "
        f"对应断言是恒真的。{mutation.note}"
    )


def test_d2c_mutation_03_report_every_mutation_and_detection_status(
    tmp_path_factory, capsys
):
    """把每个变异与检出状态打出来 —— 结论必须可复核,不能只写在散文里。"""
    rows: list[tuple[str, str, str]] = []
    for mutation in MUTATIONS:
        detected = _detect(mutation, tmp_path_factory)
        rows.append((mutation.key, mutation.name, "DETECTED" if detected else "not detected"))
    with capsys.disabled():
        print("\nD-2c mutation report")
        print(f"{'#':<4}{'变异':<42}{'状态':<14}说明")
        for key, name, status in rows:
            note = next(m.note for m in MUTATIONS if m.key == key)
            print(f"{key:<4}{name:<42}{status:<14}{note}")
    # 阴性对照是唯一"未被检出"的那一条。
    undetected = [row for row in rows if row[2] == "not detected"]
    assert len(undetected) == 1
    assert undetected[0][0] == "11"
    assert len(rows) == len(MUTATIONS) == 16
