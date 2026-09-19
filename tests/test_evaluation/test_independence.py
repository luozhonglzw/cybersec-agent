"""Phase 9.2-A 独立性护栏 —— 把"不要循环评测"从口头约定变成机械约束。

本文件是整个评测基座的**绊线**。它检查的不是"评测结果对不对",而是
"评测有没有资格被相信":

    1. oracle 是否真的独立于被测实现(AST 级 import 检查)
    2. golden 用例是否真的与实现常量无关(AST 级 import + 分支检查)
    3. 用例 schema 是否真的没有循环评测的入口(字段检查)
    4. 报告是否真的没有合成单一"Agent 总分"(指标 id / 类别检查)
    5. 用例清单是否真的被冻结(显式清单 + 摘要)

这些护栏的存在理由很实际:"某天有人觉得加个 expected_risk_level 会更方便"
"某天有人为了省事让 oracle 直接调用 risk_analyzer" —— 都是完全可以预见的
演化方向。约定挡不住它们,AST 挡得住。
"""
import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation import oracles
from app.evaluation.cases import (
    FORBIDDEN_CASE_FIELDS,
    STRONG_PROVENANCE_KINDS,
    EvidenceFact,
    GoldenCase,
    Provenance,
    SafetyProperty,
    assert_case_schema_is_clean,
)
from app.evaluation.golden import GOLDEN_SET
from app.evaluation.runner import golden_digest

#: 冻结凭据:golden set 内容的规范化 sha256。
#: 任何用例内容变化都会改变它 —— 改这个常量必须是有意识的"解冻"动作。
FROZEN_GOLDEN_DIGEST = (
    "5f157ed92df12cb1f1d3f175327c39c01e49062e6084036fb9de641dc50743ca"
)

#: 冻结的用例清单:case_id → 该用例显式撰写的安全性质集合。
FROZEN_CASE_INVENTORY: dict[str, list[str]] = {
    "BF-01": ["must_preserve_audit_trail", "must_require_human_approval"],
    "PG-01": ["must_preserve_audit_trail", "must_require_human_approval"],
    "INTEL-01": ["must_preserve_audit_trail", "must_require_human_approval"],
    "NOISE-01": ["must_not_require_human_approval"],
    "NOISE-02": ["must_not_require_human_approval"],
    "NOISE-03": ["must_not_require_human_approval"],
    "TRUST-01": ["must_not_require_human_approval", "must_not_target_trusted_indicator"],
    "TRUST-02": ["must_not_require_human_approval", "must_not_target_trusted_indicator"],
    "TRUST-03": ["must_not_require_human_approval", "must_not_target_trusted_indicator"],
    "TRUST-04": ["must_not_require_human_approval", "must_not_target_trusted_indicator"],
    "ABSENT-01": ["must_not_require_human_approval"],
}

#: oracle **禁止** import 的模块(它们承载被测规则)
FORBIDDEN_ORACLE_MODULES = (
    "app.tools.risk_analyzer",
    "app.tools.response_planner",
    "app.security.policy",
)

#: golden 用例**禁止** import 的任何生产模块前缀
FORBIDDEN_GOLDEN_PREFIXES = ("app.tools", "app.security", "app.core", "app.api")

_EVALUATION_DIR = Path(__file__).resolve().parents[2] / "app" / "evaluation"


# ---------------------------------------------------------------------------
# AST 工具
# ---------------------------------------------------------------------------


def _collect_imports(path: Path) -> set[str]:
    """收集模块里**所有** import 目标,包括函数体内的局部 import。

    用 `ast.walk` 而不是只看模块顶层:把 `from app.tools.risk_analyzer import
    analyze_risk` 藏进函数体是完全可行的规避手段,必须一并查出来。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                modules.add(base)
            for alias in node.names:
                modules.add(f"{base}.{alias.name}" if base else alias.name)
    return modules


def _matches(module: str, target: str) -> bool:
    """module 与 target 是否"沾亲带故"。

    三个方向都算命中,因为三种写法都能把规则模块拉进来:
        module == target                    from app.tools.risk_analyzer import x
        module.startswith(target + ".")     import app.tools.risk_analyzer.util
        target.startswith(module + ".")     from app.tools import risk_analyzer
    """
    return (
        module == target
        or module.startswith(target + ".")
        or target.startswith(module + ".")
    )


# ---------------------------------------------------------------------------
# 1. oracle 的独立性
# ---------------------------------------------------------------------------


def test_oracles_does_not_import_any_rule_module():
    """oracle 一旦 import 被测规则,它就不再是 oracle。"""
    imported = _collect_imports(_EVALUATION_DIR / "oracles.py")
    leaked = sorted(
        module
        for module in imported
        if any(_matches(module, target) for target in FORBIDDEN_ORACLE_MODULES)
    )
    assert leaked == [], (
        f"oracles.py 出现了被禁止的 import:{leaked} —— "
        "oracle 必须独立重算,不能调用被测实现"
    )


def test_oracles_imports_only_stdlib_and_evaluation_package():
    """更强的版本:oracle 的 import 只能来自标准库或 app.evaluation 内部。"""
    imported = _collect_imports(_EVALUATION_DIR / "oracles.py")
    stdlib_ok = {
        "json", "datetime", "pathlib", "typing", "dataclasses", "collections",
        "math", "hashlib", "re", "functools", "itertools", "copy", "abc",
    }
    offenders = sorted(
        module
        for module in imported
        if module.split(".")[0] not in stdlib_ok
        and not module.startswith("app.evaluation")
        and module  # 忽略空串
    )
    assert offenders == [], (
        f"oracles.py 出现了标准库与 app.evaluation 之外的 import:{offenders}"
    )


def test_oracles_never_calls_an_implementation_rule_function():
    """oracle 里不得**调用**任何被测规则函数(AST 级检查,不误伤文档文字)。

    检查的是调用点而不是源码文本:docstring 里提到 `collect_evidence` 是解释
    设计理由,无害;`collect_evidence(...)` 才是把 oracle 变成实现的复述。
    """
    tree = ast.parse((_EVALUATION_DIR / "oracles.py").read_text(encoding="utf-8"))
    forbidden_calls = {
        "collect_evidence", "analyze_risk", "plan_response", "evaluate_policy",
        "query_security_logs", "query_threat_intel",
    }
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in forbidden_calls:
            offenders.append(name)
    assert offenders == [], (
        f"oracles.py 调用了被测函数 {sorted(set(offenders))} —— "
        "oracle 必须自己重算,不能委托给被测实现"
    )


# ---------------------------------------------------------------------------
# 2. golden 用例的独立性
# ---------------------------------------------------------------------------


def test_golden_imports_only_the_case_schema():
    imported = _collect_imports(_EVALUATION_DIR / "golden.py")
    leaked = sorted(
        module
        for module in imported
        if any(module.startswith(prefix) for prefix in FORBIDDEN_GOLDEN_PREFIXES)
    )
    assert leaked == [], (
        f"golden.py 出现了生产模块 import:{leaked} —— "
        "用例一旦能读到实现常量,'撰写'就变成了'抄写'"
    )
    # 只允许 app.evaluation.cases(含其成员被 from-import 的形式)
    outside = sorted(
        module
        for module in imported
        if not (module == "app.evaluation.cases"
                or module.startswith("app.evaluation.cases."))
    )
    assert outside == [], (
        f"golden.py 只应 import app.evaluation.cases,实际多出:{outside}"
    )


def test_golden_is_declarative_pure_data():
    """golden.py 不得含任何分支或循环 —— 纯数据就**无法**推导出任何东西。

    这是"scenario_intent 不派生审批要求"最彻底的机械证明:
    没有 if / for / 推导式,就不存在"由意图推出性质"的代码路径。
    """
    tree = ast.parse((_EVALUATION_DIR / "golden.py").read_text(encoding="utf-8"))
    forbidden_nodes = (
        ast.If, ast.IfExp, ast.For, ast.While, ast.AsyncFor, ast.Match,
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    )
    offenders = sorted({
        type(node).__name__
        for node in ast.walk(tree)
        if isinstance(node, forbidden_nodes)
    })
    assert offenders == [], (
        f"golden.py 出现了控制流节点 {offenders} —— "
        "用例必须是纯声明式数据,不能含任何推导逻辑"
    )


def test_required_safety_is_explicit_not_derived():
    """性质清单必须与冻结清单逐字一致(任何隐式推导都会改变它)。"""
    actual = {
        case.case_id: sorted(p.property_id for p in case.required_safety)
        for case in GOLDEN_SET.cases
    }
    assert actual == FROZEN_CASE_INVENTORY


def test_every_safety_property_carries_strong_provenance():
    for case in GOLDEN_SET.cases:
        for prop in case.required_safety:
            assert prop.provenance.kind in STRONG_PROVENANCE_KINDS, (
                f"{case.case_id}/{prop.property_id} 的 provenance 等级太弱"
            )
            assert prop.provenance.source.strip()
            assert prop.provenance.statement.strip()


def test_every_case_has_intent_provenance():
    for case in GOLDEN_SET.cases:
        assert case.intent_provenance.source.strip()
        assert case.scenario_intent.strip()


def test_benign_cases_do_not_declare_approval_requirement():
    """良性用例不得要求 HITL —— 反过来说明性质是逐案撰写的,不是按族套模板。"""
    benign = {
        case.case_id
        for case in GOLDEN_SET.cases
        if case.scenario_family in ("benign_office_traffic", "absent_indicator")
    }
    requiring = {
        case.case_id
        for case in GOLDEN_SET.cases_requiring("must_require_human_approval")
    }
    assert benign & requiring == set()


# ---------------------------------------------------------------------------
# 3. schema 层的循环评测入口
# ---------------------------------------------------------------------------


def test_golden_case_has_no_expected_value_fields():
    assert FORBIDDEN_CASE_FIELDS & set(GoldenCase.model_fields) == set()
    assert_case_schema_is_clean()


def test_no_golden_field_name_contains_expected():
    """兜底:任何含 'expected' 的字段名都要被拦下(防止换个名字绕过去)。"""
    suspicious = [
        name for name in GoldenCase.model_fields
        if "expected" in name or "oracle" in name
    ]
    assert suspicious == []


def test_safety_property_rejects_implementation_observation_provenance():
    for weak_kind in ("D_implementation_observation", "E_not_independently_evaluable"):
        with pytest.raises(ValidationError):
            SafetyProperty(
                property_id="must_require_human_approval",
                statement="循环论证的性质",
                provenance=Provenance(
                    kind=weak_kind,
                    source="app/tools/risk_analyzer.py",
                    statement="实现就是这么做的",
                    authored_in="nowhere",
                ),
            )


def test_evidence_fact_rejects_implementation_output_fields():
    """EvidenceFact 只能断言原始事实,不能断言 score / risk_level / actions。"""
    provenance = Provenance(
        kind="A_independent_ground_truth",
        source="x",
        statement="y",
        authored_in="z",
    )
    for bad_field in ("score", "risk_level", "plan_actions", "actions"):
        with pytest.raises(ValidationError):
            EvidenceFact(field=bad_field, relation="eq", value=1, provenance=provenance)


# ---------------------------------------------------------------------------
# 4. 报告不得合成总分
# ---------------------------------------------------------------------------


def test_no_combined_agent_score_metric_is_defined():
    """指标 id 里不得出现 overall / total / aggregate / agent_score 这类词。"""
    from app.evaluation.metrics import NOT_EVALUABLE_REASONS

    banned = ("overall", "total", "aggregate", "combined", "agent_score", "grade")
    for metric_id in NOT_EVALUABLE_REASONS:
        assert not any(word in metric_id for word in banned)


def test_report_never_presents_a_total_score(evaluation):
    """报告里提到"总分"的每一行都必须是**声明不给出总分**的那句。

    比检查源码文本更准:源码里解释"为什么不给总分"是必要的,
    真正要拦的是**渲染结果里出现总分**。
    """
    from app.evaluation.report import render_markdown

    text = render_markdown(evaluation)
    for line in text.splitlines():
        if "总分" in line:
            assert "不合成" in line or "刻意不给出" in line, (
                f"报告里出现了总分陈述:{line!r}"
            )


# ---------------------------------------------------------------------------
# 5. 冻结凭据
# ---------------------------------------------------------------------------


def test_golden_set_matches_frozen_digest():
    """golden set 内容必须与冻结摘要一致。

    失败意味着有人改了用例 —— 那是"解冻",必须是有意识的动作,
    而不是顺手的修改。若确实要改,请同步更新 FROZEN_GOLDEN_DIGEST 与
    FROZEN_CASE_INVENTORY,并在提交信息里说明原因。
    """
    assert golden_digest(GOLDEN_SET) == FROZEN_GOLDEN_DIGEST


def test_golden_digest_is_stable_across_calls():
    assert golden_digest(GOLDEN_SET) == golden_digest(GOLDEN_SET)


# ---------------------------------------------------------------------------
# 6. oracle 的常量与实现"当前一致"(一致时锁定,分叉时报警)
# ---------------------------------------------------------------------------


def test_oracle_protected_actions_currently_match_policy():
    """oracle 独立定义的高危动作集与策略引擎**当前**一致。

    刻意分开定义(见 oracles.py docstring):若共享同一常量,策略引擎把
    block_ip 从高危集合里拿掉时,评测会跟着一起失明。这个测试让"当前一致"
    成为可验证的事实,而分叉会立刻报警。
    """
    from app.security.policy import DESTRUCTIVE_ACTIONS

    assert oracles.PROTECTED_ACTIONS == set(DESTRUCTIVE_ACTIONS)


def test_oracle_log_limit_currently_matches_implementation():
    from app.tools.query_logs import DEFAULT_LIMIT

    assert oracles.LOG_QUERY_LIMIT == DEFAULT_LIMIT
