"""Phase 9.2-D-2c 真实 provider 标定基础设施(**实现与测试闸门,零真实调用**)。

这个包回答两个问题
------------------
    1. 我们请求的配置,与 provider 实际接受/执行的配置,是不是同一件事?
    2. 标定这件事本身,能不能与正式试点**完全隔离**?

两条贯穿全包的设计纪律
--------------------

**请求溯源与记录溯源必须同源。** 两者若来自不同代码,一次标定最可能的产出
不是答案,而是一个**自洽的假象**:记录说发了 `temperature=0`,请求里其实
根本没有该字段,而两边看起来都对。因此所有请求 kwargs 与所有记录值都从
`CalibrationConfig` 这一个对象派生。

**声明与观测必须分开记。** 请求的模型 ≠ provider 自报的模型;声明的输出上限
≠ 观测到的输出用量;阶段调用上限 ≠ 实际调用数;理论 HTTP 尝试上界 ≠ 观测到的
物理尝试数。缺失的一侧记 `None` / `NOT_AVAILABLE` / `UNKNOWN`,**绝不臆造**,
也绝不互相顶替。

模块边界
--------

    config.py      冻结配置(**唯一**事实来源)
    namespace.py   标定 / 试点的命名空间与文件系统隔离(fail closed)
    verdict.py     PASS / INCONCLUSIVE / FAIL / ABORT 判定语义
    stages.py      六个阶段的捕获与判定 + 阶段运行器
    harness.py     顺序阶段门执行器 + 隔离证明
    synthetic.py   合成 provider 形状的响应与模型(零网络)

本包**不 import 任何 provider 客户端**,不读环境变量,不持有凭据,
不发起任何网络 I/O。真实 provider 构造在包外的
`app/evaluation/real_provider.py`,由调用方以**注入的调用器**形式提供。
"""

from app.evaluation.calibration.config import (
    CALIBRATION_BASE_URL,
    CALIBRATION_C0B_TEMPORARY_CAP,
    CALIBRATION_CANDIDATE_ID,
    CALIBRATION_HTTP_ATTEMPT_CEILING,
    CALIBRATION_MAX_RETRIES,
    CALIBRATION_OUTPUT_TOKEN_CAP,
    CALIBRATION_PROVIDER,
    CALIBRATION_PROVIDER_FAMILY,
    CALIBRATION_REQUESTED_MODEL,
    CALIBRATION_STAGE_LABELS,
    CALIBRATION_TEMPERATURE,
    EXCLUDED_MODELS,
    HARD_EXPERIMENTAL_RUN_CEILING,
    HARD_LOGICAL_INVOCATION_CEILING,
    NOMINAL_EXPERIMENTAL_RUNS,
    NOMINAL_LOGICAL_INVOCATIONS,
    NOMINAL_STAGE_COUNT,
    STAGE_LOGICAL_INVOCATION_CEILINGS,
    STAGE_LOGICAL_INVOCATION_FLOORS,
    CalibrationConfig,
    calibration_budget,
    default_calibration_config,
)
from app.evaluation.calibration.harness import (
    CalibrationHarness,
    CalibrationRun,
    PilotEligibilityViolation,
    StageRecord,
    assert_no_pilot_eligible_records,
    assert_not_pilot_eligible,
    format_stage_table,
    offline_evidence,
)
from app.evaluation.calibration.namespace import (
    CALIBRATION_ID_PREFIX,
    PILOT_ID_PREFIX,
    REPO_DATA_DIR,
    CalibrationNamespaceError,
    ExperimentNamespace,
    assert_calibration_experiment_id,
    assert_outside_repo_data,
    assert_pilot_experiment_id,
    assert_workdirs_disjoint,
    calibration_experiment_id,
    calibration_root,
    calibration_stage_workdir,
    classify_experiment_id,
    pilot_root,
)
from app.evaluation.calibration.stages import (
    DEFAULT_STAGE_RUNNERS,
    STAGE_ORDER,
    CalibrationRequest,
    CapParameterRejected,
    EvaluationUnit,
    InvocationResult,
    InvokeFn,
    RoundTripDefect,
    Stage,
    StageContext,
    StageDeps,
    StageDependencyMissing,
    StageInvocationCeilingExceeded,
    StageOutcome,
    StageScope,
    ToolCallDefect,
    ToolContract,
    c0a_evidence,
    c0a_verdict,
    c0b_evidence,
    c0b_evidence_verdict,
    c1_verdict,
    c2_verdict,
    observed_output_tokens,
    tool_call_findings,
    tool_roundtrip_findings,
)
from app.evaluation.calibration.synthetic import (
    SyntheticProviderModel,
    SyntheticTurn,
    deepseek_shaped_raw_usage,
    normalize_openai_shaped,
    openai_shaped_raw_usage,
    synthetic_ai_message,
    synthetic_tool_call,
)
from app.evaluation.calibration.verdict import (
    StageBlocked,
    StageVerdict,
    Verdict,
    assert_stage_passed,
    c0b_verdict,
)

__all__ = [
    "CALIBRATION_BASE_URL",
    "CALIBRATION_C0B_TEMPORARY_CAP",
    "CALIBRATION_CANDIDATE_ID",
    "CALIBRATION_HTTP_ATTEMPT_CEILING",
    "CALIBRATION_ID_PREFIX",
    "CALIBRATION_MAX_RETRIES",
    "CALIBRATION_OUTPUT_TOKEN_CAP",
    "CALIBRATION_PROVIDER",
    "CALIBRATION_PROVIDER_FAMILY",
    "CALIBRATION_REQUESTED_MODEL",
    "CALIBRATION_STAGE_LABELS",
    "CALIBRATION_TEMPERATURE",
    "DEFAULT_STAGE_RUNNERS",
    "EXCLUDED_MODELS",
    "HARD_EXPERIMENTAL_RUN_CEILING",
    "HARD_LOGICAL_INVOCATION_CEILING",
    "NOMINAL_EXPERIMENTAL_RUNS",
    "NOMINAL_LOGICAL_INVOCATIONS",
    "NOMINAL_STAGE_COUNT",
    "PILOT_ID_PREFIX",
    "REPO_DATA_DIR",
    "STAGE_LOGICAL_INVOCATION_CEILINGS",
    "STAGE_LOGICAL_INVOCATION_FLOORS",
    "STAGE_ORDER",
    "CalibrationConfig",
    "CalibrationHarness",
    "CalibrationNamespaceError",
    "CalibrationRequest",
    "CalibrationRun",
    "CapParameterRejected",
    "EvaluationUnit",
    "ExperimentNamespace",
    "InvocationResult",
    "InvokeFn",
    "PilotEligibilityViolation",
    "RoundTripDefect",
    "Stage",
    "StageBlocked",
    "StageContext",
    "StageDeps",
    "StageDependencyMissing",
    "StageInvocationCeilingExceeded",
    "StageOutcome",
    "StageRecord",
    "StageScope",
    "StageVerdict",
    "SyntheticProviderModel",
    "SyntheticTurn",
    "ToolCallDefect",
    "ToolContract",
    "Verdict",
    "assert_calibration_experiment_id",
    "assert_no_pilot_eligible_records",
    "assert_not_pilot_eligible",
    "assert_outside_repo_data",
    "assert_pilot_experiment_id",
    "assert_stage_passed",
    "assert_workdirs_disjoint",
    "c0a_evidence",
    "c0a_verdict",
    "c0b_evidence",
    "c0b_evidence_verdict",
    "c0b_verdict",
    "c1_verdict",
    "c2_verdict",
    "calibration_budget",
    "calibration_experiment_id",
    "calibration_root",
    "calibration_stage_workdir",
    "classify_experiment_id",
    "deepseek_shaped_raw_usage",
    "default_calibration_config",
    "format_stage_table",
    "normalize_openai_shaped",
    "observed_output_tokens",
    "offline_evidence",
    "openai_shaped_raw_usage",
    "pilot_root",
    "synthetic_ai_message",
    "synthetic_tool_call",
    "tool_call_findings",
    "tool_roundtrip_findings",
]
