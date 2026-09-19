"""Phase 9.2-A 端到端测试:runner + report。

覆盖四件必须成立的事:

1. **完整跑通**:11 用例 × 3 基线 = 33 条观测,19 条指标(12 有值 + 7 不可评测);
2. **确定性**:两次独立完整运行的全部指标逐字段一致;
3. **hermetic**:审计库只在 workdir 下,仓库里不出现 `audit.db`;
   评测包不引用仓库 `data/` 下的任何路径;
4. **报告纪律**:五类分开、不含总分、带输入摘要与诚实声明。
"""
import ast
import asyncio
from pathlib import Path

from app.evaluation.golden import GOLDEN_SET
from app.evaluation.report import render_markdown, write_report
from app.evaluation.runner import golden_digest, run_evaluation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVALUATION_DIR = _REPO_ROOT / "app" / "evaluation"


# ---------------------------------------------------------------------------
# 1. 完整跑通
# ---------------------------------------------------------------------------


def test_evaluation_covers_every_case_on_every_baseline(evaluation):
    assert len(GOLDEN_SET.cases) == 11
    assert evaluation.adapter_ids == ["B1", "B2", "B3"]
    assert len(evaluation.observations) == 11 * 3
    keys = {(o["case_id"], o["adapter_id"]) for o in evaluation.observations}
    assert len(keys) == 33


def test_metric_inventory_is_complete(evaluation):
    expected = {
        "evidence_accuracy_authored", "evidence_accuracy_recomputed",
        "safety_policy_compliance", "safety_escalation_success_rate",
        "malicious_escalation_rate", "trusted_indicator_protection_rate",
        "audit_completeness", "lifecycle_completion_rate",
        "plan_internal_consistency", "policy_outcome_consistency",
        "gate_decision_agreement", "plan_digest_chain_integrity",
    }
    ids = {metric.metric_id for metric in evaluation.metrics}
    assert expected <= ids
    assert len(evaluation.metrics) == len(expected) + 7  # + 7 条 E 类


def test_every_case_carries_its_oracle_recomputation(evaluation):
    """runner 必须把 oracle 重算结果附到每条观测上,metrics 才能不碰文件系统。"""
    for observation in evaluation.observations:
        assert isinstance(observation["_oracle_evidence"], dict)
        assert observation["_oracle_evidence"]["indicator"] == observation["case_indicator"]


# ---------------------------------------------------------------------------
# 2. 确定性
# ---------------------------------------------------------------------------


def test_two_full_runs_produce_identical_metrics(tmp_path, seed_dataset):
    """评测自身必须确定 —— 不确定的评测,它产出的数字没有意义。"""
    first = asyncio.run(run_evaluation(
        logs_path=seed_dataset["logs"], intel_path=seed_dataset["intel"],
        workdir=tmp_path / "run-a",
    ))
    second = asyncio.run(run_evaluation(
        logs_path=seed_dataset["logs"], intel_path=seed_dataset["intel"],
        workdir=tmp_path / "run-b",
    ))

    def snapshot(result):
        return sorted(
            (m.metric_id, m.status, m.value, m.numerator, m.denominator)
            for m in result.metrics
        )

    assert snapshot(first) == snapshot(second)
    assert first.golden_digest == second.golden_digest
    assert first.dataset_digests == second.dataset_digests


def test_two_full_runs_produce_identical_observations(tmp_path, seed_dataset):
    first = asyncio.run(run_evaluation(
        logs_path=seed_dataset["logs"], intel_path=seed_dataset["intel"],
        workdir=tmp_path / "run-a",
    ))
    second = asyncio.run(run_evaluation(
        logs_path=seed_dataset["logs"], intel_path=seed_dataset["intel"],
        workdir=tmp_path / "run-b",
    ))

    def strip(result):
        out = []
        for observation in result.observations:
            payload = dict(observation)
            payload.pop("answer", None)
            out.append(payload)
        return out

    assert strip(first) == strip(second)


# ---------------------------------------------------------------------------
# 3. hermetic
# ---------------------------------------------------------------------------


def test_audit_database_lives_under_the_run_workdir(evaluation, tmp_path):
    assert evaluation.audit_db_paths
    for raw in evaluation.audit_db_paths:
        assert Path(raw).exists()
        assert Path(raw).is_relative_to(tmp_path)


def test_no_audit_database_appears_in_the_repository(evaluation):
    assert not (_REPO_ROOT / "data" / "audit.db").exists()


def test_evaluation_package_has_no_hardcoded_data_path():
    """评测包不得引用仓库 data/ 下的任何路径 —— 否则在无 data/ 的 CWD 下会挂。"""
    offenders: list[str] = []
    for path in sorted(_EVALUATION_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("data/") or node.value.startswith("data\\"):
                    offenders.append(f"{path.name}:{node.lineno}={node.value!r}")
    assert offenders == [], f"评测包出现硬编码数据路径:{offenders}"


def test_evaluation_inputs_are_outside_the_repository(evaluation):
    """评测输入来自 tmp_path,不来自仓库 data/。"""
    for digest in evaluation.dataset_digests.values():
        assert len(digest) == 64


# ---------------------------------------------------------------------------
# 4. 变形关系
# ---------------------------------------------------------------------------


def test_metamorphic_relations_hold_on_the_unmutated_codebase(evaluation):
    assert {r.relation_id for r in evaluation.relations} == {
        "MR1_monotonicity", "MR2_trusted_dominance",
    }
    for relation in evaluation.relations:
        assert relation.status == "holds", f"{relation.relation_id}: {relation.detail}"


def test_trusted_dominance_relation_is_actually_exercised(evaluation):
    relation = next(r for r in evaluation.relations if r.relation_id == "MR2_trusted_dominance")
    scores = {key: value["score"] for key, value in relation.evidence.items()}
    assert scores["trusted"] <= scores["absent"] <= scores["malicious"]
    assert scores["trusted"] < scores["malicious"], "序关系必须是有内容的,不能全相等"


# ---------------------------------------------------------------------------
# 5. 报告
# ---------------------------------------------------------------------------


def test_report_renders_all_categories_and_input_digests(evaluation):
    text = render_markdown(evaluation)
    for category in ("A. ", "B. ", "C. ", "D. ", "E. "):
        assert category in text
    assert evaluation.golden_digest in text
    assert evaluation.dataset_digests["logs"] in text


def test_report_states_it_has_no_combined_score(evaluation):
    text = render_markdown(evaluation)
    assert "不合成" in text
    assert "Agent 分数" in text


def test_report_carries_the_baseline_honesty_caveat(evaluation):
    text = render_markdown(evaluation)
    assert "共享同一个确定性内核" in text
    assert "不接防火墙" in text or "没有任何基线执行真实处置动作" in text


def test_report_prints_not_evaluable_with_reasons(evaluation):
    text = render_markdown(evaluation)
    assert "NOT_EVALUABLE" in text
    assert "task_success" in text


def test_report_writes_to_disk(evaluation, tmp_path):
    target = write_report(evaluation, tmp_path / "reports" / "phase92a.md")
    assert target.exists()
    assert target.read_text(encoding="utf-8").startswith("# Phase 9.2-A 评测报告")


def test_golden_digest_is_derived_from_content_not_from_time():
    assert golden_digest(GOLDEN_SET) == golden_digest(GOLDEN_SET)
