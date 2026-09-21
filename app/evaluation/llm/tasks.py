"""Phase 9.2-D-1 真实 LLM 评测的**任务契约**(task schema)。

本模块只定义**形状**;任务内容在 `dataset.py`,判定在 `metrics.py`。

与 Phase 9.2-A 的关系(勿混)
---------------------------
`app/evaluation/cases.py` 的 `GoldenCase` 测的是**确定性内核**:
它没有自然语言任务提示、没有工具行为约束、没有叙事期望。
本模块的 `LLMTask` 测的是**LLM 能力与叙事接地** —— 两者是**并列**的
两套基准,互不覆盖:

    9.2-A  golden.py   11 个冻结用例   → 确定性证据/风险/计划/策略/审批
    9.2-D  dataset.py  ~8 个任务       → LLM 工具行为 / 参数 / 叙事接地 / 注入

本模块**不修改** 9.2-A 的任何文件,也不复用它的用例内容。

三类契约各自回答一个问题
------------------------
    tool_contract       该任务的工具行为**应该**是什么样(约束式,不是精确序列)
    grounding_contract  叙事里哪些事实可被**独立**核验,哪些断言是**禁止**的
    security_contract   评测授权的数据路径是什么,注入期望是什么

反循环评测(继承 9.2-A 的原则)
-----------------------------
每条约束都必须带 `provenance`,且 `kind` 只允许 A/B/C —— 不允许用
"实现就是这么做的"(D)支撑一条评测期望。约束的正当性来自**任务意图**,
不来自当前实现的常量表。

刻意**没有**的字段:
- `expected_tool_sequence`:除非任务确实只有唯一合法序列,否则精确序列比对
  会把"当前实现碰巧怎么走"当成正确性。默认用约束式评分。
- `expected_risk_level` / `expected_score` / `expected_actions`:
  这三个值只能从实现常量反推 —— 是 9.2-A 明令禁止的循环评测入口。
  叙事与权威计划的一致性在**运行时**对照系统自己的权威产物,不预先写死。
"""
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.evaluation.cases import STRONG_PROVENANCE_KINDS, Provenance

#: 注入载荷的**稳定标记**(载荷文本的第一个片段)。
#:
#: 为什么标记必须住在契约模块里:适配器要用它判定「LLM 是否真的在
#: 上下文里看到了注入文本」。这是 **matched-counterfactual 归因**的前提 ——
#: 脚本化行为必须**依赖载荷**,否则「宣称了被注入的那条虚假声明」与
#: 「被注入文本影响」之间就没有因果关系,指标只能测出"输出碰巧命中目标"。
#: 载荷文本本体在 `dataset.py`,由本标记拼接而成(单一事实来源)。
INJECTION_PAYLOAD_MARKER = "【系统维护通知】"

#: 任务族。只有**无需生产改动**即可支撑的族才在这里。
TaskFamily = Literal[
    "benign_investigation",
    "brute_force_investigation",
    "malicious_ioc_investigation",
    "ambiguous_insufficient_evidence",
    "irrelevant_indicator",
    "conflicting_evidence",
    "safe_tool_error",
    "synthetic_prompt_injection",
]

#: 参数约束的关系。刻意限定为可机械判定的关系,不含自然语言判断。
ArgumentRelation = Literal[
    "eq",              # 参数值 == value
    "in",              # 参数值 ∈ value(list)
    "le",              # 参数值 <= value
    "ge",              # 参数值 >= value
    "contains",        # 字符串参数包含 value
    "non_default",     # 参数值 != 该工具声明的默认值(即"显式传了非默认值")
    "authorized_path",  # 路径参数 ∈ 评测授权路径集合
]

#: 叙事中可被**独立**核验的声明类别(封闭词表)。
#: 刻意不含自由因果叙述 —— 那需要 judge,本阶段排除。
ClaimClass = Literal[
    "indicator_identity",
    "log_event_count",
    "failed_login_count",
    "threat_intel_found",
    "threat_intel_malicious",
    "threat_intel_severity",
    "risk_level",
    "action_recommendation",
    "approval_requirement",
]

#: 可写进 grounding 契约的**原始证据**字段(A 类可核验)。
_RAW_EVIDENCE_FIELDS: frozenset[str] = frozenset({
    "log_event_count",
    "failed_login_count",
    "threat_intel_found",
    "threat_intel_malicious",
    "threat_intel_severity",
})


def _require_strong(provenance: Provenance, what: str) -> Provenance:
    """约束类断言的 provenance 只允许 A/B/C。

    与 9.2-A 的 `SafetyProperty._reject_weak_provenance` 同源:用实现观测(D)
    或"不可独立评测"(E)去支撑一条**评测期望**,等于自己证明自己。
    """
    if provenance.kind not in STRONG_PROVENANCE_KINDS:
        raise ValueError(
            f"{what} 的 provenance.kind 必须是 A/B/C 之一,收到 {provenance.kind!r} ——"
            "用实现观测(D)或不可独立评测(E)支撑的评测期望是循环论证"
        )
    return provenance


class ArgumentConstraint(BaseModel):
    """对某个工具调用的**参数**约束。

    用于表达两类期望:
        required_arguments        必须成立
        forbidden_argument_patterns 绝不允许成立
    """

    tool: str = Field(min_length=1, description="工具名")
    argument: str = Field(min_length=1, description="参数名")
    relation: ArgumentRelation = Field(description="判定关系")
    value: Any = Field(default=None, description="右值(eq/in/le/ge/contains 使用)")
    provenance: Provenance = Field(description="该约束为何存在(只允许 A/B/C)")

    @model_validator(mode="after")
    def _strong_provenance(self) -> "ArgumentConstraint":
        _require_strong(self.provenance, f"参数约束 {self.tool}.{self.argument}")
        return self

    def holds(self, args: dict) -> bool | None:
        """判定该约束是否成立。

        返回 `None` 表示**参数缺失** —— 调用方必须把缺失与"违反"区分开:
        缺失是"没传",违反是"传错了"。只有 `authorized_path` / `non_default`
        两类关系对缺失有明确语义(见各自分支)。
        """
        present = self.argument in args
        actual = args.get(self.argument)

        if self.relation == "authorized_path":
            # 未传路径参数 = 用工具默认路径,LLM 没有做出路径选择 → 不判违反
            return True if not present else None  # 真实判定由 metrics 用授权集合完成
        if self.relation == "non_default":
            return actual is not None if present else False
        if not present:
            return None
        if self.relation == "eq":
            return actual == self.value
        if self.relation == "in":
            return actual in (self.value or [])
        if self.relation == "le":
            return actual <= self.value
        if self.relation == "ge":
            return actual >= self.value
        if self.relation == "contains":
            return isinstance(actual, str) and str(self.value) in actual
        return None


class ToolContract(BaseModel):
    """一个任务的工具行为契约(**约束式**,不是精确序列)。

    为什么不用精确序列:同一结论可由多个合法序列达成
    (`[logs, intel]` 与 `[intel, logs]` 等价);精确序列比对会把
    "当前实现碰巧怎么走"当成正确性,并惩罚更优解。
    """

    required_tools: list[str] = Field(
        default_factory=list, description="每个至少调用一次"
    )
    forbidden_tools: list[str] = Field(
        default_factory=list, description="必须 0 次"
    )
    allowed_tools: list[str] = Field(
        default_factory=list, description="允许全集;调用集合必须是它的子集"
    )
    max_total_calls: int = Field(ge=1, description="工具调用总数上限")
    required_arguments: list[ArgumentConstraint] = Field(default_factory=list)
    forbidden_argument_patterns: list[ArgumentConstraint] = Field(default_factory=list)
    ordering_constraints: list[tuple[str, str]] = Field(
        default_factory=list, description="偏序对:(先, 后)"
    )
    provenance: Provenance = Field(description="整套工具约束为何存在")

    @model_validator(mode="after")
    def _strong_provenance(self) -> "ToolContract":
        _require_strong(self.provenance, "工具契约")
        return self

    @model_validator(mode="after")
    def _consistent_sets(self) -> "ToolContract":
        req, forb, allow = set(self.required_tools), set(self.forbidden_tools), set(self.allowed_tools)
        if not req <= allow:
            raise ValueError(f"required_tools 必须是 allowed_tools 的子集:多余 {sorted(req - allow)}")
        if req & forb:
            raise ValueError(f"同一工具不能既 required 又 forbidden:{sorted(req & forb)}")
        if not allow:
            raise ValueError("allowed_tools 不能为空 —— 空集会让任何调用都算越界")
        return self


class VerifiableFact(BaseModel):
    """一条**独立可核验**的原始事实(A 类)。

    只允许断言原始证据字段。风险等级 / 动作集**不允许**写在这里:
    它们只能从实现常量反推,写进契约即构成循环评测(与 9.2-A 同一条护栏)。
    """

    field: str = Field(min_length=1, description="原始证据字段名")
    relation: Literal["eq", "gte", "lte"] = Field(description="断言关系")
    value: int | bool | str | None = Field(description="右值")
    provenance: Provenance = Field(description="该事实的出处(只允许 A/B/C)")

    @model_validator(mode="after")
    def _raw_fields_only(self) -> "VerifiableFact":
        if self.field not in _RAW_EVIDENCE_FIELDS:
            raise ValueError(
                f"VerifiableFact.field 只能是原始证据字段 {sorted(_RAW_EVIDENCE_FIELDS)},"
                f"收到 {self.field!r} —— 对 risk_level / actions 的断言属于实现观测"
            )
        _require_strong(self.provenance, f"可核验事实 {self.field}")
        return self


class ProhibitedClaim(BaseModel):
    """一条**禁止出现**的叙事断言。

    语义:`claim_class` 被断言为 `value` 即为违反。
    例:注入任务的 `(risk_level, "none")` —— 若叙事宣称风险等级为 none,
    而权威计划不是 none,该断言即为被注入指令操纵的证据。

    ⚠️ 本模型**不写"正确值"**,只写"这个值一旦出现就是错的"。正确值在运行时
    来自独立重算的证据(原始事实)或系统自己的权威计划 —— 两者都不是预写常量。
    """

    claim_class: ClaimClass = Field(description="声明类别")
    value: Any = Field(description="一旦被断言即为违反的取值")
    rationale: str = Field(min_length=1, description="为什么这个断言是禁止的")
    provenance: Provenance = Field(description="该禁止条款的出处(只允许 A/B/C)")

    @model_validator(mode="after")
    def _strong_provenance(self) -> "ProhibitedClaim":
        _require_strong(self.provenance, f"禁止断言 {self.claim_class}={self.value!r}")
        return self


class GroundingContract(BaseModel):
    """叙事的**接地**契约:哪些原始事实可核验,哪些断言被禁止。

    刻意**不**声称能测"幻觉"全貌 —— 只覆盖 9 类有确定性指称对象的声明。
    """

    verifiable_facts: list[VerifiableFact] = Field(
        default_factory=list, description="可独立核验的原始事实(A 类)"
    )
    prohibited_false_claims: list[ProhibitedClaim] = Field(
        default_factory=list, description="一旦被断言即为错误的声明"
    )
    provenance: Provenance = Field(description="整套接地契约为何存在")

    @model_validator(mode="after")
    def _strong_provenance(self) -> "GroundingContract":
        _require_strong(self.provenance, "接地契约")
        return self


class InjectionContract(BaseModel):
    """合成 prompt-injection 契约(**载荷必须无害**)。

    只允许要求"良性但可观测"的行为后果,例如:
        - 在结论中宣称一个被独立证伪的风险等级
        - 因为注入文本而省略一个独立要求的工具

    绝不允许包含:真实攻击载荷、凭据窃取、文件外泄、破坏性指令。
    """

    payload_text: str = Field(min_length=1, description="注入文本(合成、无害)")
    injected_false_claim: ProhibitedClaim = Field(
        description="被注入的虚假断言 —— 它一旦出现即计为 follow"
    )
    injected_tool_omission: str | None = Field(
        default=None, description="注入文本要求省略的工具名(可观测但不单独判定)"
    )
    provenance: Provenance = Field(description="注入契约为何存在(只允许 A/B/C)")

    @model_validator(mode="after")
    def _strong_provenance(self) -> "InjectionContract":
        _require_strong(self.provenance, "注入契约")
        return self


class SecurityContract(BaseModel):
    """评测沙箱契约:授权路径 + 注入期望 + 禁止的可观测行为。

    `authorized_dataset_keys` 是**逻辑键**,运行时由 runner 解析为本次评测
    实际生成的临时路径。这样契约里不出现机器相关的绝对路径。
    """

    authorized_dataset_keys: list[Literal["logs", "intel"]] = Field(
        default_factory=lambda: ["logs", "intel"],
        description="评测授权的数据路径(逻辑键)",
    )
    path_arguments: list[str] = Field(
        default_factory=lambda: ["data_path", "logs_path", "intel_path"],
        description="需要按授权路径判定的路径类参数名",
    )
    injection: InjectionContract | None = Field(
        default=None, description="仅合成注入任务有值"
    )
    prohibited_observable_behaviors: list[str] = Field(
        default_factory=list, description="人可读的禁止行为清单(报告用)"
    )
    provenance: Provenance = Field(description="整套沙箱契约为何存在")

    @model_validator(mode="after")
    def _strong_provenance(self) -> "SecurityContract":
        _require_strong(self.provenance, "沙箱契约")
        return self


class LLMTask(BaseModel):
    """一条真实 LLM 评测任务。

    与 `GoldenCase` 并列,互不覆盖:`GoldenCase` 测确定性内核,
    `LLMTask` 测 LLM 能力与叙事接地。
    """

    task_id: str = Field(min_length=1, description="任务 id,如 T-BRUTEFORCE-01")
    family: TaskFamily = Field(description="任务族(封闭词表)")
    indicator: str = Field(min_length=1, description="判定对象(IP / 域名 / Hash)")
    event_type: str | None = Field(
        default=None, description="可选:限定统计的日志事件类型"
    )
    dataset_variant: str = Field(
        default="base", description="使用哪个评测侧数据集变体(base / conflict / injection)"
    )
    user_prompt: str = Field(min_length=1, description="交给 LLM 的自然语言任务")
    tool_contract: ToolContract = Field(description="工具行为约束")
    grounding_contract: GroundingContract = Field(description="叙事接地契约")
    security_contract: SecurityContract = Field(description="评测沙箱契约")
    provenance: Provenance = Field(description="该任务为何存在")

    @model_validator(mode="after")
    def _strong_provenance(self) -> "LLMTask":
        _require_strong(self.provenance, f"任务 {self.task_id}")
        return self

    def injection_required(self) -> bool:
        return self.security_contract.injection is not None
