"""Phase 9.2-A 评测用例契约(golden case schema)。

本模块只定义**用例的形状**;用例内容在 `golden.py`,判定在 `oracles.py`。

本模块刻意**没有**的东西(这是设计决定,不是遗漏)
--------------------------------------------------
`GoldenCase` 不含 `expected_score` / `expected_risk_level` / `expected_actions`。

理由:这三个值在 Phase 9.2-A 只能从规则常量(权重表、分档阈值、动作属性表)
反推出来。一旦写进用例,"评测"就退化成"实现复述自己的规则" —— 这是循环评测
最典型的形式。风险等级与动作集是 **D. 实现内部观测**,不是 ground truth。

`tests/test_evaluation/test_independence.py` 用机械方式锁定这条:
`GoldenCase.model_fields` 里不得出现这三个名字,`golden.py` 也不得 import
任何生产模块。

`scenario_intent` 不派生审批要求
--------------------------------
"这个指标是恶意的"(威胁分类)与"这一步必须人工审批"(授权边界)是两个不同的
概念。用例是否要求 HITL,只能由 `required_safety` 里**显式撰写**的性质决定 ——
不允许从 `scenario_intent` 或 `scenario_family` 自动推导。这条同样由测试锁定
(`SafetyProperty` 必须显式声明 `property_id`)。

provenance 的证据等级
---------------------
沿用 Phase 9.2-A Review 的五分法:

    A_independent_ground_truth      原始数据/标注本身,先于被测规则被写下
    B_human_authored_expectation    人写的"应该发生什么"(需求 / 场景意图)
    C_derived_from_deterministic_spec  由**书面规范**机械推导(不是由实现代码推导)
    D_implementation_observation    实现自己跑出来的结果(不能当 ground truth)
    E_not_independently_evaluable   没有独立 oracle

`SafetyProperty.provenance.kind` 只允许 A / B / C。用 D 或 E 去支撑一条"安全
性质"等于自己证明自己,由 `_reject_weak_provenance` 在校验层直接拒绝。
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ProvenanceKind = Literal[
    "A_independent_ground_truth",
    "B_human_authored_expectation",
    "C_derived_from_deterministic_spec",
    "D_implementation_observation",
    "E_not_independently_evaluable",
]

#: 可以作为"安全性质"依据的证据等级 —— 刻意不含 D / E
STRONG_PROVENANCE_KINDS: frozenset[str] = frozenset({
    "A_independent_ground_truth",
    "B_human_authored_expectation",
    "C_derived_from_deterministic_spec",
})

#: 用例里被禁止出现的字段名(循环评测的直接入口)
FORBIDDEN_CASE_FIELDS: frozenset[str] = frozenset({
    "expected_score",
    "expected_risk_level",
    "expected_actions",
})


class Provenance(BaseModel):
    """一条断言的来源。**只记录来源,不记录结论。**

    statement 必须是来源处的**原文或可逐字复核的转述**,不允许写成"因此应该
    判为 high"这类结论 —— 结论一旦混进 provenance,来源就不可复核了。
    """

    kind: ProvenanceKind = Field(description="证据等级(A/B/C/D/E)")
    source: str = Field(
        min_length=1,
        description="可逐字复核的出处,如 scripts/seed_logs.py:87-100 或 docs/architecture.md:77",
    )
    statement: str = Field(
        min_length=1, description="来源处的原文/可复核转述(不得写成结论)"
    )
    authored_in: str = Field(
        min_length=1, description="该事实在哪个阶段被写下,如 'Phase 2 seed data'"
    )


class EvidenceFact(BaseModel):
    """对**原始数据**的一条手工断言(不是对实现输出的断言)。

    例:198.51.100.7 在 seed_logs.py 里由 `for i in range(12)` 生成 12 条
    login_failed —— 这是 A 级事实,先于风险规则存在,可用来检验采集层。

    只允许出现在"原始事实"上:log_event_count / failed_login_count /
    threat_intel_found / threat_intel_malicious / threat_intel_severity / tags。
    不允许对 score / risk_level / actions 写 EvidenceFact。
    """

    field: str = Field(min_length=1, description="RiskEvidence 的字段名")
    relation: Literal["eq", "gte", "lte"] = Field(description="断言关系")
    value: int | bool | str | None = Field(description="断言的右值")
    provenance: Provenance = Field(description="该断言的来源")

    @field_validator("field")
    @classmethod
    def _only_raw_evidence_fields(cls, value: str) -> str:
        allowed = {
            "log_event_count",
            "failed_login_count",
            "threat_intel_found",
            "threat_intel_malicious",
            "threat_intel_severity",
            "threat_intel_tags",
        }
        if value not in allowed:
            raise ValueError(
                f"EvidenceFact.field 只能断言原始证据字段 {sorted(allowed)},"
                f"收到 {value!r} —— 对 score / risk_level / actions 的断言"
                "属于实现观测,不是 ground truth"
            )
        return value


class SafetyProperty(BaseModel):
    """一条**手工撰写**的安全性质。是否要求 HITL 只能由它决定。

    property_id 取自固定的性质词汇表(见 oracles.py 的 PROPERTY_REQUIREMENTS)。
    显式枚举而不是自由文本,是为了让"用例要求了什么"可机械统计、可机械比对。
    """

    property_id: Literal[
        "must_require_human_approval",
        "must_not_require_human_approval",
        "must_not_target_trusted_indicator",
        "must_preserve_audit_trail",
    ] = Field(description="性质 id(封闭词汇表)")
    statement: str = Field(min_length=1, description="该性质的人可读陈述")
    provenance: Provenance = Field(description="该性质的来源(只允许 A/B/C)")

    @field_validator("provenance")
    @classmethod
    def _reject_weak_provenance(cls, value: Provenance) -> Provenance:
        """拒绝用实现观测 / 不可评测来支撑一条安全性质。

        这是**反循环评测的核心护栏**:一条安全性质若只能靠"实现就是这么做的"
        来支撑,它就不是独立性质,而是实现的复述。
        """
        if value.kind not in STRONG_PROVENANCE_KINDS:
            raise ValueError(
                f"安全性质的 provenance.kind 必须是 A/B/C 之一,收到 {value.kind!r} ——"
                "用实现观测(D)或不可独立评测(E)支撑的安全性质是循环论证"
            )
        return value


class GoldenCase(BaseModel):
    """一条手工撰写的评测用例。

    刻意**没有** expected_score / expected_risk_level / expected_actions
    (见模块 docstring)。用例能表达的全部内容就是:

        "这个指标是什么(原始事实)" + "在这种情况下什么必须成立(安全性质)"
    """

    case_id: str = Field(min_length=1, description="用例 id,如 BF-01")
    scenario_family: str = Field(
        min_length=1, description="场景族:brute_force_escalation / password_guessing / ..."
    )
    indicator: str = Field(min_length=1, description="评测对象(IP / 域名 / Hash)")
    scenario_intent: str = Field(
        min_length=1,
        description="这个指标在数据里**被撰写成**什么(Phase 2 场景意图原文);不派生审批要求",
    )
    intent_provenance: Provenance = Field(description="场景意图的来源")
    evidence_facts: list[EvidenceFact] = Field(
        default_factory=list, description="对原始数据的手工断言(A 级)"
    )
    required_safety: list[SafetyProperty] = Field(
        min_length=1, description="必须成立的安全性质(至少一条)"
    )


class GoldenSet(BaseModel):
    """冻结的用例集合。"""

    version: str = Field(min_length=1, description="golden set 版本")
    cases: list[GoldenCase] = Field(min_length=1, description="用例列表")

    @property
    def case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases]

    def by_id(self, case_id: str) -> GoldenCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"未知用例 id: {case_id}")

    def cases_requiring(self, property_id: str) -> list[GoldenCase]:
        """显式声明了某条安全性质的用例 —— 不允许由 scenario_intent 推导。"""
        return [
            case
            for case in self.cases
            if any(p.property_id == property_id for p in case.required_safety)
        ]


def assert_case_schema_is_clean() -> None:
    """机械断言:用例 schema 里不存在被禁止的期望值字段。

    供测试与 runner 自检调用。写成函数而不是靠人肉 review,是因为"某天有人
    觉得加个 expected_risk_level 会更方便"是完全可以预见的演化方向。
    """
    leaked = FORBIDDEN_CASE_FIELDS & set(GoldenCase.model_fields)
    if leaked:
        raise AssertionError(
            f"GoldenCase 出现了被禁止的期望值字段 {sorted(leaked)} ——"
            "这些值只能从实现常量反推,写进用例即构成循环评测"
        )
