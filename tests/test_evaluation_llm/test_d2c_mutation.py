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
    assert len(rows) == len(MUTATIONS) == 11
