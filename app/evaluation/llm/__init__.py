"""Phase 9.2-D-1 真实 LLM 评测工装(Real-LLM Evaluation Harness)。

本包回答的问题与 9.2-A **并列而不重叠**:

    9.2-A  确定性内核在多大程度上能被独立证据检验?
    9.2-D  LLM 在**工具行为、参数、叙事接地、注入抵抗**上表现如何?

为什么必须分成两套基准
----------------------
`/triage` 的权威安全路径(`plan` → `policy_gate` → `human_approval`)只消费
**调用方传入的 indicator** 与 **ResponsePlan**,**从不读 `messages`**。
因此 LLM 的推理 / 工具选择 / 叙事文本与安全产物之间**不存在耦合**。
把两者合成一个"Agent 总分"会把这种结构性分离掩盖掉,并诱使后来者为了
"让分数好看"去削弱这个分离 —— 而那个分离本身是当前架构的**安全性质**。

所以本包把指标分成 5 类,分别呈现,绝不合成总分:

    capability    LLM 能力:工具选择 / 参数合法性 / 参数语义 / 预算 / 顺序 / 路径授权
    grounding     叙事接地:与独立重算证据一致 / 与权威计划不矛盾
    injection     合成 prompt-injection 的**匹配对照归因** follow / resistance
                  (+ 一条 outcome-only 的 target-match,无因果含义)
    invariant     **架构回归不变量**(不是"安全得分")
    efficiency    调用数 / 迭代数 / 延迟 / token(全部 D 类描述性)

D-1 全程离线
------------
`provider = "scripted"`,模型是 `deterministic-scripted-<behavior>`。
**不创建真实 provider 客户端、不读 `.env`、不发起网络请求、不消耗 API 额度。**
真实 provider 的评测(D-2)需要**单独的用户批准**。

模块边界(勿混)
--------------
    tasks.py         任务契约 schema(形状)
    dataset.py       8 个任务 + 评测侧 fixture 生成(内容)
    adapters.py      ScriptedLLM + B0 / B2' / B3 → LLMObservation
    extract.py       确定性声明抽取(无 judge)
    metrics.py       指标定义与聚合(5 类分开)
    runner.py        编排 + 可复现性元数据 + D-1 清单
    report.py        D-1 渲染(必须分节,不得合成单一分数)

Phase 9.2-D-2a(离线试点基础设施,**全部离线**)
--------------------------------------------
    protocol.py       任务清单摘要 / 指标 schema 摘要 / 清单与闸门
    ordering.py       确定性执行顺序(sha256 排序,不用 PRNG)
    budget.py         四个互不混淆的计数器 + 预算治理器
    failures.py       冻结的失败分类表(全部 experimental_retry_allowed=False)
    raw.py            原始结果持久化(append+fsync+sidecar)+ 崩溃/续跑语义
    exposure.py       载荷暴露实测 + 覆盖率列联表
    derived.py        派生聚合 + 描述性 Wilson(冻结措辞)
    pilot.py          冻结的试点规模 + **候选**清单构建
    executor.py       离线执行器(只跑 ScriptedLLM)+ 匹配对照配对核验
    offline_guard.py  网络出口守卫(CPython 审计钩子)
    pilot_report.py   D-2a 报告渲染(九条强制声明)

D-2a 的三条硬约束(已内化为结构)
--------------------------------
1. **只产出候选清单。** provider / model / 端点类别 / token 与 cost 上限
   仍是占位符;`verify_manifest(require_frozen=True)` 会机械拒绝。
2. **不重跑任何实验单元。** `HARNESS_LEVEL_RETRY = 0`;失败按冻结分类表
   记录成"一个观测结果 + 一份覆盖率损失",不是重跑候选。
3. **未暴露不是抵抗。** 载荷从未进入模型可见上下文的运行既不进 follow
   的分子也不进分母,记 `N/E`。

三条语义约束(Phase 9.2-D Review 的修正,已内化为结构)
----------------------------------------------------
1. **不变量不是安全得分。** `plan_digest_invariance` / `policy_outcome_invariance`
   / `audit_event_sequence_invariance` 在现架构下**按构造**为 1.0。
   它们是回归护栏:一旦未来有人把安全路径耦合到 LLM 行为,它们会立刻变红。
   **禁止**把 1.0 表述为"100% 安全容纳",本包也**不定义** `safety_containment_rate`。
2. **"遵循注入"必须有可观测的禁止性后果。** 仅复述 / 提及注入文本**不算** follow
   —— 那恰恰是正确行为。判据是"注入载荷指定的那条虚假声明是否被采纳"。
3. **NOT_EVALUABLE 不是 0。** 分母为 0 或缺独立参照物时如实记"不可评测"。

Phase 9.2-D-2b(真实 provider 就绪,**仍然全程离线**)
--------------------------------------------------
    identity.py       模型身份 + 冻结的 `endpoint_category` 词表(**零 provider 依赖**)
    budgeted_llm.py   逐逻辑调用的预算边界代理(provider 无关)

本阶段**没有**新增任何真实 provider 调用。它修的是三条"真实 provider 一接上
就会静默失真"的路径:

    模型身份        由执行上下文传入,不再写死 scripted 三连(RK-2)
    预算执行        Tier 1 单元准入 + Tier 2 每次调用前硬检查(F-4)
    实验身份隔离    跨实验记录 fail closed,校准不得冻结试点单元

真实 provider 的**构造**刻意留在本包之外(`app/evaluation/real_provider.py`):
本包有一条冻结护栏 —— `app/evaluation/llm/*.py` 不得 import 任何 provider 客户端。
模型通过 `LLMFactory` 注入缝进入,`ScriptedLLM` 仍是**离线默认实现**。

本包不修改任何生产代码、不引入新依赖、不改动 9.2-A 的冻结用例。
"""

from app.evaluation.llm.adapters import (
    B0DirectAdapter,
    B2PrimeGraphAdapter,
    B3FullAgentAdapter,
    BASELINES,
    LLMFactory,
    LLMObservation,
    MATRIX_BEHAVIORS,
    ScriptedLLM,
    ToolCallRecord,
)
from app.evaluation.llm.budget import (
    MAX_GRAPH_ITERATIONS,
    UNKNOWN,
    BudgetCounters,
    BudgetExceeded,
    BudgetGovernor,
    InheritedConsumption,
    PilotBudget,
    budget_from_plan,
    pilot_budget,
)
from app.evaluation.llm.budgeted_llm import BudgetedLLM, budgeted
from app.evaluation.llm.dataset import (
    INJECTION_PAYLOAD,
    LLM_DATASET_VERSION,
    LLM_TASKS,
    build_datasets,
    task_families,
    tasks_by_id,
)
from app.evaluation.llm.derived import (
    N3_LIMITATION,
    NO_GENERAL_SAFETY_CLAIM,
    NO_LEADERBOARD_STATEMENT,
    NO_P_VALUE_STATEMENT,
    WILSON_CAVEAT,
    DerivedAggregate,
    Proportion,
    derive,
    wilson_interval,
)
from app.evaluation.llm.executor import (
    BudgetAbort,
    ExecutionOutcome,
    MatchedPair,
    OfflineExecutor,
    PairingError,
    RawArtifactExistsError,
    assert_no_silent_append,
    budget_abort_failure,
    build_matched_pairs,
    plan_resume,
    verify_pairing,
)
from app.evaluation.llm.exposure import (
    EXPOSURE_COVERAGE_DEFINITION,
    EXPOSURE_COVERAGE_OUTPUT_ID,
    ExposureCoverage,
    ExposureStatus,
    coverage_by_condition,
    exposure_coverage,
    exposure_status,
)
from app.evaluation.llm.extract import (
    Claim,
    NarrativeClaims,
    extract_claims,
)
from app.evaluation.llm.failures import (
    FAILURE_TAXONOMY,
    HARNESS_LEVEL_RETRY,
    FailureClass,
    FailureRecord,
    classify_exception,
    classify_http_status,
)
from app.evaluation.llm.identity import (
    CREDENTIAL_FIELD_NAMES,
    ENDPOINT_CATEGORIES,
    REAL_PROVIDER_TEMPERATURE,
    SCRIPTED_PROVIDER,
    EndpointCategory,
    ModelIdentity,
    UnknownEndpointCategory,
    assert_no_credential_fields,
    base_url_host,
    base_url_host_sha256,
    parse_endpoint_category,
    provider_identity,
    scripted_identity,
)
from app.evaluation.llm.metrics import (
    NOT_EVALUABLE_REASONS,
    MetricCell,
    MetricResult,
    compute_llm_metrics,
)
from app.evaluation.llm.offline_guard import (
    NetworkEgressGuard,
    NetworkEgressError,
)
from app.evaluation.llm.ordering import (
    EXECUTION_ORDER_SEED,
    ExecutionUnit,
    build_execution_order,
    execution_order_digest,
)
from app.evaluation.llm.pilot import (
    DECLARED_CONFOUNDS,
    PILOT_BASELINES,
    PILOT_BEHAVIORS,
    PILOT_REPETITION_COUNT,
    PilotPlan,
    assert_candidate_only,
    build_candidate_manifest,
    manifest_freeze_blockers,
    pilot_plan,
    verify_candidate_manifest,
)
from app.evaluation.llm.pilot_report import (
    CAVEAT_KEYS,
    PILOT_REPORT_CAVEATS,
    render_pilot_report,
)
from app.evaluation.llm.protocol import (
    ExperimentManifest,
    ManifestError,
    PROTOCOL_VERSION,
    canonical_json,
    compute_manifest_digest,
    metric_schema_digest,
    real_llm_taskset_digest,
    verify_manifest,
)
from app.evaluation.llm.raw import (
    ForeignExperimentRecord,
    RawRecord,
    RawWriter,
    RecordStatus,
    ResumeEligibility,
    SecretLeakError,
    is_resumable,
    resume_eligibility,
)
from app.evaluation.llm.report import render_markdown, write_report
from app.evaluation.llm.runner import (
    BASELINE_LABELS,
    FORBIDDEN_METADATA_FIELDS,
    LLMEvaluationResult,
    PilotManifest,
    RunMetadata,
    assert_no_secrets,
    build_metadata,
    metric_signature,
    observation_signature,
    pilot_manifest,
    result_signature,
    run_llm_evaluation,
)
from app.evaluation.llm.tasks import (
    ArgumentConstraint,
    GroundingContract,
    InjectionContract,
    LLMTask,
    ProhibitedClaim,
    SecurityContract,
    ToolContract,
    VerifiableFact,
)

__all__ = [
    "ArgumentConstraint",
    "BASELINE_LABELS",
    "BASELINES",
    "B0DirectAdapter",
    "B2PrimeGraphAdapter",
    "B3FullAgentAdapter",
    "BudgetAbort",
    "BudgetCounters",
    "BudgetExceeded",
    "BudgetGovernor",
    "BudgetedLLM",
    "CAVEAT_KEYS",
    "CREDENTIAL_FIELD_NAMES",
    "Claim",
    "DECLARED_CONFOUNDS",
    "DerivedAggregate",
    "ENDPOINT_CATEGORIES",
    "EXECUTION_ORDER_SEED",
    "EXPOSURE_COVERAGE_DEFINITION",
    "EXPOSURE_COVERAGE_OUTPUT_ID",
    "EndpointCategory",
    "ExecutionOutcome",
    "ExecutionUnit",
    "ExperimentManifest",
    "ExposureCoverage",
    "ExposureStatus",
    "FAILURE_TAXONOMY",
    "FORBIDDEN_METADATA_FIELDS",
    "FailureClass",
    "FailureRecord",
    "ForeignExperimentRecord",
    "GroundingContract",
    "HARNESS_LEVEL_RETRY",
    "INJECTION_PAYLOAD",
    "InheritedConsumption",
    "InjectionContract",
    "LLMFactory",
    "LLM_DATASET_VERSION",
    "LLMEvaluationResult",
    "LLMObservation",
    "LLMTask",
    "LLM_TASKS",
    "MATRIX_BEHAVIORS",
    "MAX_GRAPH_ITERATIONS",
    "ManifestError",
    "MatchedPair",
    "MetricCell",
    "MetricResult",
    "ModelIdentity",
    "N3_LIMITATION",
    "NOT_EVALUABLE_REASONS",
    "NO_GENERAL_SAFETY_CLAIM",
    "NO_LEADERBOARD_STATEMENT",
    "NO_P_VALUE_STATEMENT",
    "NarrativeClaims",
    "NetworkEgressError",
    "NetworkEgressGuard",
    "OfflineExecutor",
    "PILOT_BASELINES",
    "PILOT_BEHAVIORS",
    "PILOT_REPETITION_COUNT",
    "PILOT_REPORT_CAVEATS",
    "PROTOCOL_VERSION",
    "PairingError",
    "PilotBudget",
    "PilotManifest",
    "PilotPlan",
    "ProhibitedClaim",
    "Proportion",
    "REAL_PROVIDER_TEMPERATURE",
    "SCRIPTED_PROVIDER",
    "RawArtifactExistsError",
    "RawRecord",
    "RawWriter",
    "RecordStatus",
    "ResumeEligibility",
    "RunMetadata",
    "ScriptedLLM",
    "SecretLeakError",
    "SecurityContract",
    "ToolCallRecord",
    "ToolContract",
    "UNKNOWN",
    "UnknownEndpointCategory",
    "VerifiableFact",
    "WILSON_CAVEAT",
    "assert_candidate_only",
    "assert_no_credential_fields",
    "assert_no_secrets",
    "assert_no_silent_append",
    "base_url_host",
    "base_url_host_sha256",
    "budget_from_plan",
    "budget_abort_failure",
    "budgeted",
    "build_candidate_manifest",
    "build_datasets",
    "build_execution_order",
    "build_matched_pairs",
    "build_metadata",
    "canonical_json",
    "classify_exception",
    "classify_http_status",
    "compute_llm_metrics",
    "compute_manifest_digest",
    "coverage_by_condition",
    "derive",
    "execution_order_digest",
    "exposure_coverage",
    "exposure_status",
    "extract_claims",
    "is_resumable",
    "manifest_freeze_blockers",
    "metric_schema_digest",
    "metric_signature",
    "observation_signature",
    "parse_endpoint_category",
    "pilot_budget",
    "pilot_manifest",
    "pilot_plan",
    "plan_resume",
    "provider_identity",
    "real_llm_taskset_digest",
    "render_markdown",
    "render_pilot_report",
    "result_signature",
    "resume_eligibility",
    "run_llm_evaluation",
    "scripted_identity",
    "task_families",
    "tasks_by_id",
    "verify_candidate_manifest",
    "verify_manifest",
    "verify_pairing",
    "wilson_interval",
    "write_report",
]
