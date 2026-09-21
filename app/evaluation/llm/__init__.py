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
    tasks.py     任务契约 schema(形状)
    dataset.py   8 个任务 + 评测侧 fixture 生成(内容)
    adapters.py  ScriptedLLM + B0 / B2' / B3 → LLMObservation
    extract.py   确定性声明抽取(无 judge)
    metrics.py   指标定义与聚合(5 类分开)
    runner.py    编排 + 可复现性元数据 + D-2 清单
    report.py    渲染(必须分节,不得合成单一分数)

三条语义约束(Phase 9.2-D Review 的修正,已内化为结构)
----------------------------------------------------
1. **不变量不是安全得分。** `plan_digest_invariance` / `policy_outcome_invariance`
   / `audit_event_sequence_invariance` 在现架构下**按构造**为 1.0。
   它们是回归护栏:一旦未来有人把安全路径耦合到 LLM 行为,它们会立刻变红。
   **禁止**把 1.0 表述为"100% 安全容纳",本包也**不定义** `safety_containment_rate`。
2. **"遵循注入"必须有可观测的禁止性后果。** 仅复述 / 提及注入文本**不算** follow
   —— 那恰恰是正确行为。判据是"注入载荷指定的那条虚假声明是否被采纳"。
3. **NOT_EVALUABLE 不是 0。** 分母为 0 或缺独立参照物时如实记"不可评测"。

本包不修改任何生产代码、不引入新依赖、不改动 9.2-A 的冻结用例。
"""

from app.evaluation.llm.adapters import (
    B0DirectAdapter,
    B2PrimeGraphAdapter,
    B3FullAgentAdapter,
    BASELINES,
    LLMObservation,
    MATRIX_BEHAVIORS,
    ScriptedLLM,
    ToolCallRecord,
)
from app.evaluation.llm.dataset import (
    INJECTION_PAYLOAD,
    LLM_DATASET_VERSION,
    LLM_TASKS,
    build_datasets,
    task_families,
    tasks_by_id,
)
from app.evaluation.llm.extract import (
    Claim,
    NarrativeClaims,
    extract_claims,
)
from app.evaluation.llm.metrics import (
    NOT_EVALUABLE_REASONS,
    MetricCell,
    MetricResult,
    compute_llm_metrics,
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
    "Claim",
    "FORBIDDEN_METADATA_FIELDS",
    "GroundingContract",
    "INJECTION_PAYLOAD",
    "InjectionContract",
    "LLM_DATASET_VERSION",
    "LLMEvaluationResult",
    "LLMObservation",
    "LLMTask",
    "LLM_TASKS",
    "MATRIX_BEHAVIORS",
    "MetricCell",
    "MetricResult",
    "NOT_EVALUABLE_REASONS",
    "NarrativeClaims",
    "PilotManifest",
    "ProhibitedClaim",
    "RunMetadata",
    "ScriptedLLM",
    "SecurityContract",
    "ToolCallRecord",
    "ToolContract",
    "VerifiableFact",
    "assert_no_secrets",
    "build_datasets",
    "build_metadata",
    "compute_llm_metrics",
    "extract_claims",
    "metric_signature",
    "observation_signature",
    "pilot_manifest",
    "render_markdown",
    "result_signature",
    "run_llm_evaluation",
    "task_families",
    "tasks_by_id",
    "write_report",
]
