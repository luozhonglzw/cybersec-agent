"""Phase 9.2-A 本地评测基座(Evaluation Foundation)。

本包回答一个**单一问题**:

    这套安全 Agent 的确定性内核,在多大程度上能被**独立于它自己**的证据检验?

为什么必须先回答这个问题,而不是先加指标
------------------------------------------
Phase 9.2-A 最大的风险不是"指标太少",而是**循环评测(circular evaluation)**:
如果 golden set 的期望值是从 `risk_analyzer` / `response_planner` / `policy`
的常量反推出来的,那么"评测通过"只证明实现复述了自己的规则,不证明它对。
这类评测给出的是**虚假的信心** —— 比没有评测更危险。

因此本包在结构上把三类东西**物理隔离**:

    golden.py     手工撰写、逐项带 provenance 的用例(纯数据,不 import 任何生产模块)
    oracles.py    独立重算(只读原始 JSONL,不 import 任何规则模块)
    adapters.py   把被测系统跑出**观测**(唯一 import 生产代码的地方)

模块边界(勿混):

    cases.py     用例 schema(形状),不含用例内容,也不含任何实现常量
    golden.py    **用例内容** —— 手工撰写 + provenance 标注,零生产依赖
    oracles.py   独立重算 + 安全性质判定 + 变形关系谓词(禁止 import 规则模块)
    adapters.py  B1 RuleOnly / B2 GraphNoGate / B3 FullAgent → Observation
    metrics.py   指标定义与聚合(把观测折成数字),五类分开,不合成总分
    runner.py    编排:dataset → adapters → oracles → metrics
    report.py    渲染(必须分 A/B/C/D/E,不得合成单一"Agent 分数")

三条语义约束(Phase 9.2-A Review 的修正,已内化为结构)
------------------------------------------------------
1. **确定性升级不等于任务成功。** 本包不产出 `task_success`。恶意用例走到
   `pending_approval` 只说明"安全升级成功",不说明"用户的分析任务被完成" ——
   后者在 Phase 9.2-A 没有独立 oracle,记为 NOT_EVALUABLE。
2. **"恶意"与"需人工审批"不是同一个概念。** 前者是威胁分类,后者是授权边界。
   `scenario_intent` 不派生审批要求;是否要求 HITL 只能由用例里**显式撰写**的
   `required_safety` 性质决定。
3. **NOT_EVALUABLE 不是 0。** 缺 oracle 的能力如实记为"不可评测",不折算成
   失败,也不折算成通过。

本包不修改任何生产代码,不引入任何新依赖,不发起任何真实 LLM 调用。
"""

from app.evaluation.adapters import (
    ADAPTERS,
    FullAgentAdapter,
    GraphNoGateAdapter,
    Observation,
    RuleOnlyAdapter,
)
from app.evaluation.cases import (
    EvidenceFact,
    GoldenCase,
    GoldenSet,
    Provenance,
    ProvenanceKind,
    SafetyProperty,
)
from app.evaluation.golden import GOLDEN_SET
from app.evaluation.metrics import (
    NOT_EVALUABLE_REASONS,
    Category,
    MetricResult,
    compute_metrics,
)
from app.evaluation.runner import EvaluationResult, run_evaluation

__all__ = [
    "ADAPTERS",
    "Category",
    "EvaluationResult",
    "EvidenceFact",
    "FullAgentAdapter",
    "GOLDEN_SET",
    "GoldenCase",
    "GoldenSet",
    "GraphNoGateAdapter",
    "MetricResult",
    "NOT_EVALUABLE_REASONS",
    "Observation",
    "Provenance",
    "ProvenanceKind",
    "RuleOnlyAdapter",
    "SafetyProperty",
    "compute_metrics",
    "run_evaluation",
]
