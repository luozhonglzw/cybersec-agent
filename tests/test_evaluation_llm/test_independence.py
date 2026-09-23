"""Phase 9.2-D-1 独立性护栏 —— 把"不要循环评测"从口头约定变成机械约束。

本文件是 D-1 的**绊线**。它检查的不是"评测结果对不对",而是
"评测有没有资格被相信":

    1. 抽取器 / 任务 schema 是否真的独立于被测实现(AST 级 import 检查)
    2. 任务契约是否真的没有循环评测入口(expected_* 字段)
    3. 每条约束是否都带 A/B/C 级 provenance
    4. 9.2-A 的冻结用例是否真的没被碰过
    5. 任务清单是否真的被冻结(显式清单 + 数据集版本)
    6. 是否真的没有合成单一"Agent 总分"

这些护栏的存在理由很实际:"某天有人觉得加个 expected_risk_level 会更方便"
"某天有人为了省事让抽取器直接 import 生产的动作词表" —— 都是完全可以预见的
演化方向。约定挡不住它们,AST 挡得住。
"""
import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.cases import Provenance
from app.evaluation.golden import GOLDEN_SET
from app.evaluation.llm.dataset import LLM_DATASET_VERSION, LLM_TASKS, task_families
from app.evaluation.llm.tasks import (
    GroundingContract,
    LLMTask,
    ProhibitedClaim,
    ToolContract,
    VerifiableFact,
)
from app.evaluation.runner import golden_digest

#: Phase 9.2-A 冻结的 golden 摘要 —— D-1 不得改变它。
FROZEN_GOLDEN_DIGEST = (
    "5f157ed92df12cb1f1d3f175327c39c01e49062e6084036fb9de641dc50743ca"
)

#: 冻结的 D-1 任务清单:task_id → family。
FROZEN_LLM_TASK_INVENTORY: dict[str, str] = {
    "T-BENIGN-01": "benign_investigation",
    "T-BRUTEFORCE-01": "brute_force_investigation",
    "T-MALICIOUS-IOC-01": "malicious_ioc_investigation",
    "T-AMBIGUOUS-01": "ambiguous_insufficient_evidence",
    "T-IRRELEVANT-01": "irrelevant_indicator",
    "T-CONFLICT-01": "conflicting_evidence",
    "T-TOOLERROR-01": "safe_tool_error",
    "T-INJECTION-01": "synthetic_prompt_injection",
}

#: 冻结的数据集版本。
#:
#: `9.2-D-1.1` → `9.2-D-1.2`(Phase 9.2-D-2a):数据集新增 `injection_inert`
#: 变体(D-2 的匹配对照)。任务内容、载荷字节、既有三个变体**全部未变** ——
#: 版本号跟着**数据集内容**走,而不是跟着"这算哪个阶段"走。
FROZEN_DATASET_VERSION = "9.2-D-1.2"

#: 承载被测规则的生产模块 —— 评测侧的判定模块不得直接 import 它们。
FORBIDDEN_RULE_MODULES = (
    "app.tools.risk_analyzer",
    "app.tools.response_planner",
    "app.security.policy",
)

#: 从实现常量反推的字段名 —— 出现即构成循环评测入口。
FORBIDDEN_TASK_FIELD_FRAGMENTS = ("expected", "oracle", "golden", "reference_value")

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_PKG = REPO_ROOT / "app" / "evaluation" / "llm"


# ---------------------------------------------------------------------------
# AST 工具
# ---------------------------------------------------------------------------


def _imports_from_source(source: str) -> set[str]:
    """从源码文本收集 import 目标(用于正负对照测试)。

    规则:
        `import a.b.c`            → 记 `a.b.c`
        `from a.b import c`       → 记 `a.b.c`(**不**记裸 `a.b`)

    为什么不记裸 `a.b`:那样 `from app.tools import DEFAULT_TOOLS` 会因为
    `app.tools` 是 `app.tools.risk_analyzer` 的前缀而被误判。而
    `from app.tools import risk_analyzer` 会记成 `app.tools.risk_analyzer`,
    仍然被抓住 —— 检查的是**实际被拉进来的名字**,不是包名。
    """
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _collect_imports(path: Path) -> set[str]:
    """收集模块里**所有** import 目标,包括函数体内的局部 import。

    用 `ast.walk` 而不是只看模块顶层:把 `from app.tools.risk_analyzer import
    analyze_risk` 藏进函数体是完全可行的规避手段,必须一并查出来。
    """
    return _imports_from_source(path.read_text(encoding="utf-8"))


def _matches(module: str, target: str) -> bool:
    return (
        module == target
        or module.startswith(target + ".")
        or target.startswith(module + ".")
    )


def _calls_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name:
                names.add(name)
    return names


# ---------------------------------------------------------------------------
# 1. 判定模块的独立性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["tasks.py", "extract.py"])
def test_declarative_modules_import_only_stdlib_and_pydantic(module_name):
    """任务 schema 与抽取器必须**只能**用标准库 / pydantic / 用例 schema。

    这两者承载"期望从哪来"的判断;一旦它们能读到实现,期望就变成了抄写。
    """
    imported = _collect_imports(LLM_PKG / module_name)
    allowed_prefixes = ("app.evaluation.cases",)
    offenders = sorted(
        module
        for module in imported
        if module.split(".")[0] not in {"typing", "re", "json", "pathlib", "dataclasses"}
        and module.split(".")[0] not in {"pydantic", "pydantic_core"}
        and not any(module.startswith(prefix) for prefix in allowed_prefixes)
        and not module.startswith("app.evaluation.cases.")
        and module
    )
    assert offenders == [], f"{module_name} 出现了越界 import:{offenders}"


@pytest.mark.parametrize(
    "module_name", ["tasks.py", "dataset.py", "extract.py", "metrics.py", "runner.py"]
)
def test_no_evaluation_module_imports_a_rule_module(module_name):
    """评测侧不得直接 import 承载被测规则的三个模块。"""
    imported = _collect_imports(LLM_PKG / module_name)
    leaked = sorted(
        module
        for module in imported
        if any(_matches(module, target) for target in FORBIDDEN_RULE_MODULES)
    )
    assert leaked == [], f"{module_name} 引入了被测规则模块:{leaked}"


def test_metrics_never_calls_a_rule_function():
    """指标层不得**调用**被测规则函数(AST 级检查,不误伤文档文字)。"""
    forbidden = {
        "collect_evidence", "analyze_risk", "plan_response", "evaluate_policy",
        "query_security_logs", "query_threat_intel",
    }
    offenders = sorted(_calls_in(LLM_PKG / "metrics.py") & forbidden)
    assert offenders == [], f"metrics.py 调用了被测函数:{offenders}"


def test_extract_never_calls_a_rule_function():
    forbidden = {
        "collect_evidence", "analyze_risk", "plan_response", "evaluate_policy",
        "query_security_logs", "query_threat_intel",
    }
    offenders = sorted(_calls_in(LLM_PKG / "extract.py") & forbidden)
    assert offenders == [], f"extract.py 调用了被测函数:{offenders}"


def test_oracle_module_still_does_not_import_rule_modules():
    """9.2-D 复用了 9.2-A 的 oracle —— 它的独立性必须继续成立。"""
    imported = _collect_imports(REPO_ROOT / "app" / "evaluation" / "oracles.py")
    leaked = sorted(
        module
        for module in imported
        if any(_matches(module, target) for target in FORBIDDEN_RULE_MODULES)
    )
    assert leaked == []


@pytest.mark.parametrize(
    "source",
    [
        "from app.tools.risk_analyzer import analyze_risk",
        "import app.tools.risk_analyzer",
        "from app.tools import risk_analyzer",
        "from app.tools.response_planner import plan_response",
        "from app.security.policy import evaluate_policy",
        "import app.security.policy",
        "def f():\n    from app.security.policy import evaluate_policy\n    return evaluate_policy",
    ],
)
def test_rule_module_detection_has_teeth(source):
    """**正对照**:各种规避写法都必须被检测出来。

    没有这组用例,`test_no_evaluation_module_imports_a_rule_module` 可能
    因为检测器本身失效而永远通过 —— 那是最糟的情况:护栏看起来在,实际不在。
    """
    imported = _imports_from_source(source)
    leaked = [
        module
        for module in imported
        if any(_matches(module, target) for target in FORBIDDEN_RULE_MODULES)
    ]
    assert leaked, f"检测器漏掉了:{source!r}(解析出 {sorted(imported)})"


@pytest.mark.parametrize(
    "source",
    [
        "from app.tools import DEFAULT_TOOLS",
        "from app.tools.query_logs import DEFAULT_LIMIT",
        "from app.evaluation.oracles import recompute_evidence",
    ],
)
def test_rule_module_detection_does_not_false_positive(source):
    """**负对照**:合法 import 不得被误判,否则护栏会被人为绕过(直接删掉)。"""
    imported = _imports_from_source(source)
    leaked = [
        module
        for module in imported
        if any(_matches(module, target) for target in FORBIDDEN_RULE_MODULES)
    ]
    assert leaked == [], f"误判:{source!r}(解析出 {sorted(imported)})"


# ---------------------------------------------------------------------------
# 2. 任务 schema 不得有循环评测入口
# ---------------------------------------------------------------------------


def test_llm_task_has_no_expected_value_fields():
    for field in LLMTask.model_fields:
        assert not any(
            fragment in field for fragment in FORBIDDEN_TASK_FIELD_FRAGMENTS
        ), f"LLMTask 出现了疑似期望值字段:{field}"


def test_tool_contract_has_no_exact_sequence_field():
    """精确序列比对会把"当前实现碰巧怎么走"当成正确性。"""
    assert "expected_tool_sequence" not in ToolContract.model_fields
    for field in ToolContract.model_fields:
        assert "sequence" not in field


def test_prohibited_claim_never_stores_the_correct_value():
    """禁止条款只写"这个值一旦出现就是错的",不写"正确值"。"""
    fields = set(ProhibitedClaim.model_fields)
    assert fields == {"claim_class", "value", "rationale", "provenance"}


def test_verifiable_fact_rejects_derived_fields():
    """可核验事实只能断言**原始证据**字段。"""
    provenance = Provenance(
        kind="A_independent_ground_truth",
        source="x",
        statement="y",
        authored_in="z",
    )
    for bad_field in ("risk_level", "score", "plan_actions", "actions", "confidence"):
        with pytest.raises(ValidationError):
            VerifiableFact(field=bad_field, relation="eq", value=1, provenance=provenance)


def test_tool_contract_rejects_weak_provenance():
    weak = Provenance(
        kind="D_implementation_observation",
        source="app/tools/query_logs.py",
        statement="实现就是这么做的",
        authored_in="nowhere",
    )
    with pytest.raises(ValidationError):
        ToolContract(
            required_tools=[], allowed_tools=["a"], max_total_calls=1, provenance=weak
        )


def test_grounding_contract_rejects_weak_provenance():
    weak = Provenance(
        kind="E_not_independently_evaluable",
        source="x",
        statement="y",
        authored_in="z",
    )
    with pytest.raises(ValidationError):
        GroundingContract(provenance=weak)


def test_task_rejects_weak_provenance():
    with pytest.raises(ValidationError):
        LLMTask(
            task_id="X", family="benign_investigation", indicator="1.2.3.4",
            user_prompt="p",
            tool_contract=ToolContract(
                required_tools=[], allowed_tools=["a"], max_total_calls=1,
                provenance=Provenance(kind="B_human_authored_expectation", source="s",
                                      statement="t", authored_in="i"),
            ),
            grounding_contract=GroundingContract(
                provenance=Provenance(kind="B_human_authored_expectation", source="s",
                                      statement="t", authored_in="i"),
            ),
            security_contract=_security_contract(),
            provenance=Provenance(kind="D_implementation_observation", source="s",
                                  statement="t", authored_in="i"),
        )


def _security_contract():
    from app.evaluation.llm.tasks import SecurityContract

    return SecurityContract(
        provenance=Provenance(
            kind="B_human_authored_expectation", source="s", statement="t", authored_in="i"
        )
    )


# ---------------------------------------------------------------------------
# 3. 每条约束都带强 provenance
# ---------------------------------------------------------------------------


def test_every_task_contract_carries_strong_provenance():
    strong = {"A_independent_ground_truth", "B_human_authored_expectation",
              "C_derived_from_deterministic_spec"}
    for task in LLM_TASKS:
        assert task.provenance.kind in strong, task.task_id
        assert task.tool_contract.provenance.kind in strong, task.task_id
        assert task.grounding_contract.provenance.kind in strong, task.task_id
        assert task.security_contract.provenance.kind in strong, task.task_id
        for constraint in task.tool_contract.required_arguments:
            assert constraint.provenance.kind in strong, (task.task_id, constraint.argument)
        for fact in task.grounding_contract.verifiable_facts:
            assert fact.provenance.kind in strong, (task.task_id, fact.field)
        for claim in task.grounding_contract.prohibited_false_claims:
            assert claim.provenance.kind in strong, (task.task_id, claim.claim_class)


def test_every_provenance_has_a_checkable_source():
    for task in LLM_TASKS:
        for provenance in (
            task.provenance,
            task.tool_contract.provenance,
            task.grounding_contract.provenance,
            task.security_contract.provenance,
        ):
            assert provenance.source.strip()
            assert provenance.statement.strip()
            assert provenance.authored_in.strip()


def test_argument_validity_and_semantic_accuracy_use_disjoint_provenance():
    """参数合法性只用 C 类边界,参数语义只用 B 类意图 —— 两条指标互不污染。"""
    for task in LLM_TASKS:
        by_kind: dict[str, set[str]] = {}
        for constraint in task.tool_contract.required_arguments:
            by_kind.setdefault(constraint.provenance.kind, set()).add(constraint.argument)
        c_args = by_kind.get("C_derived_from_deterministic_spec", set())
        b_args = by_kind.get("B_human_authored_expectation", set())
        assert c_args, f"{task.task_id} 没有任何 C 类(工具契约边界)约束"
        assert b_args, f"{task.task_id} 没有任何 B 类(任务意图)约束"
        assert c_args & b_args == set(), f"{task.task_id} 的 B/C 约束重叠:{c_args & b_args}"


# ---------------------------------------------------------------------------
# 4. 冻结凭据
# ---------------------------------------------------------------------------


def test_llm_task_inventory_is_frozen():
    actual = {task.task_id: task.family for task in LLM_TASKS}
    assert actual == FROZEN_LLM_TASK_INVENTORY


def test_llm_dataset_version_is_frozen():
    assert LLM_DATASET_VERSION == FROZEN_DATASET_VERSION


def test_task_families_are_unique_and_complete():
    families = task_families()
    assert len(families) == len(set(families)), "每个任务族应恰好一个任务"
    assert set(families) == set(FROZEN_LLM_TASK_INVENTORY.values())


def test_phase_92a_golden_set_is_untouched():
    """D-1 不得修改、覆盖或重新解释 9.2-A 的 11 个冻结用例。"""
    assert golden_digest(GOLDEN_SET) == FROZEN_GOLDEN_DIGEST
    assert len(GOLDEN_SET.cases) == 11


# ---------------------------------------------------------------------------
# 5. 报告不得合成总分
# ---------------------------------------------------------------------------


def test_no_combined_agent_score_metric_is_defined(matrix):
    banned = ("overall", "total", "aggregate", "combined", "agent_score", "grade")
    for metric in matrix.metrics:
        assert not any(word in metric.metric_id for word in banned), metric.metric_id


def test_report_never_presents_a_total_score(matrix):
    from app.evaluation.llm.report import render_markdown

    text = render_markdown(matrix)
    for line in text.splitlines():
        if "总分" in line:
            assert "绝不合成" in line or "不合成" in line, f"报告里出现了总分陈述:{line!r}"


# ---------------------------------------------------------------------------
# 生产代码导入边界(**实测断言**,不是文档声明)
# ---------------------------------------------------------------------------


#: 生产模块前缀。D-1 包中**允许**出现这些 import 的模块只有下面两个。
PRODUCTION_PREFIXES = ("app.core", "app.security", "app.tools", "app.api", "app.schemas")

#: 实际边界。若未来有模块新增生产 import,这条测试会变红 —— 这是刻意的:
#: 边界一旦漂移,"评测侧新增、生产零改动"的说法就不再成立。
PRODUCTION_IMPORTERS = ("adapters.py", "runner.py")


def _production_imports(module_name: str) -> set[str]:
    return {
        m
        for m in _collect_imports(LLM_PKG / module_name)
        if m.startswith(PRODUCTION_PREFIXES)
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "__init__.py",
        "tasks.py",
        "dataset.py",
        "extract.py",
        "metrics.py",
        "report.py",
    ],
)
def test_evaluation_only_modules_never_import_production(module_name):
    """声明 / 抽取 / 判定 / 渲染模块**不得**触碰生产代码。

    尤其是 `metrics.py` 与 `extract.py`:它们一旦 import 规则模块,
    判定就会从"对照独立契约"退化成"复述实现常量"。
    """
    leaked = _production_imports(module_name)
    assert not leaked, f"{module_name} 引入了生产模块:{sorted(leaked)}"


def test_production_import_boundary_is_exactly_adapters_and_runner():
    """边界是**两个**模块,不是一个 —— 报告里不得声称"仅 adapters.py"。"""
    importers = sorted(
        path.name
        for path in LLM_PKG.glob("*.py")
        if _production_imports(path.name)
    )
    assert importers == list(PRODUCTION_IMPORTERS), (
        f"实际的生产导入边界是 {importers},与记录的 {list(PRODUCTION_IMPORTERS)} 不符"
    )


def test_runner_production_imports_are_metadata_only():
    """`runner.py` 只允许为了**元数据摘要**引入生产符号。

    它引入的两个名字恰好就是:系统提示词(算 sha256)与工具集(算 schema 摘要)。
    多引入任何一个都意味着评测判定开始依赖生产实现。
    """
    allowed = {
        "app.core.agent.SECURITY_ANALYST_SYSTEM_PROMPT",
        "app.tools.DEFAULT_TOOLS",
    }
    actual = _production_imports("runner.py")
    assert actual == allowed, (
        f"runner.py 的生产 import 应为 {sorted(allowed)},实际 {sorted(actual)}"
    )


def test_adapters_is_the_only_module_that_executes_production_code():
    """只有 `adapters.py` 用生产代码**执行**评测(B2'/B3 的图与工具)。"""
    from app.evaluation.llm import adapters as adapters_module

    src = Path(adapters_module.__file__).read_text(encoding="utf-8")
    for symbol in ("create_agent_graph", "HitlConfig", "SqliteAuditStore"):
        assert symbol in src, f"adapters.py 缺少执行所需的生产符号 {symbol}"
