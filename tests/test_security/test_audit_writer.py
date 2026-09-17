"""审计记录构造测试(Phase 8.2)。

覆盖 app/security/audit.py:
- compute_plan_digest:确定性、敏感性(改一个字段摘要就变)、格式、
  与"手工规范化 JSON + sha256"等价;
- new_record_id:格式与唯一性;
- build_audit_record:默认主体、默认时间、ts 注入、detail 拷贝语义、
  摘要**由 plan 推导**(调用方无法注入)。

三条契约由**签名/源码护栏**锁死,而不是靠注释:
1. 摘要不可注入 —— build_audit_record 没有 plan_digest 参数;
2. 本模块是纯计算 —— 源码不含 sqlite3 / open( / Path( 等 I/O 字样;
3. compute_plan_digest 只接受 plan。

全部 hermetic:只构造内存对象,不读写任何数据文件。
"""
import ast
import hashlib
import inspect
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

import app.security.audit as audit_module
from app.schemas.audit import SYSTEM_ACTOR, AuditRecord
from app.schemas.response import ResponseAction, ResponsePlan
from app.schemas.risk import RiskAssessment, RiskEvidence
from app.security.audit import (
    build_audit_record,
    compute_plan_digest,
    new_record_id,
)
from app.tools.response_planner import plan_response
from app.tools.risk_analyzer import analyze_risk

INDICATOR = "203.0.113.66"
FIXED_TS = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _plan(
    *,
    indicator: str = INDICATOR,
    risk_level: str = "high",
    score: int = 70,
    confidence: int = 70,
    summary: str = "测试计划",
    action_type: str = "block_ip",
) -> ResponsePlan:
    evidence = RiskEvidence(indicator=indicator, failed_login_count=30)
    assessment = RiskAssessment(
        indicator=indicator,
        risk_level=risk_level,
        score=score,
        confidence=confidence,
        reasons=["测试依据"],
        evidence=evidence,
    )
    action = ResponseAction(
        action_type=action_type,
        priority="high",
        target=indicator,
        rationale="测试依据",
        requires_approval=True,
        reversible=True,
    )
    return ResponsePlan(
        indicator=indicator,
        risk_level=risk_level,
        summary=summary,
        actions=[action],
        assessment=assessment,
    )


# ---------- compute_plan_digest:确定性与格式 ----------

def test_digest_is_64_lowercase_hex():
    assert _HEX64.match(compute_plan_digest(_plan()))


def test_digest_deterministic_same_instance():
    plan = _plan()
    assert compute_plan_digest(plan) == compute_plan_digest(plan)


def test_digest_deterministic_across_equivalent_instances():
    """内容相同的两个 plan(不同实例)→ 摘要相同。"""
    assert compute_plan_digest(_plan()) == compute_plan_digest(_plan())


def test_digest_matches_manual_canonical_sha256():
    """与文档定义的规范化流程逐字等价,防止实现漂移。

    规范:model_dump(mode="json") → json.dumps(sort_keys=True,
    ensure_ascii=False, separators=(",", ":")) → sha256 hex。
    """
    plan = _plan()
    canonical = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert compute_plan_digest(plan) == expected


def test_digest_equals_dump_json_equivalence_for_ascii_plan():
    """纯 ASCII 计划下,摘要与 Pydantic 的 model_dump_json 口径一致。

    (Pydantic 的 model_dump_json 不接受 sort_keys,故实现走 json.dumps;
     本测试用等价的 ASCII 计划证明两者口径不冲突。)
    """
    plan = _plan(indicator="198.51.100.7", summary="ascii-only plan")
    canonical = json.dumps(
        json.loads(plan.model_dump_json()),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert compute_plan_digest(plan) == hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


# ---------- compute_plan_digest:敏感性 ----------

@pytest.mark.parametrize(
    "override",
    [
        {"indicator": "10.9.9.9"},
        {"risk_level": "medium"},
        {"score": 71},
        {"confidence": 71},
        {"summary": "另一个结论"},
        {"action_type": "isolate_host"},
    ],
    ids=["indicator", "risk_level", "score", "confidence", "summary", "action"],
)
def test_digest_changes_when_any_field_changes(override):
    """改任意一个字段(含嵌套 assessment / action)→ 摘要必须变。

    摘要存在的意义就是"比对计划是否被改动过";若某个字段改了摘要不变,
    篡改就能逃过比对。
    """
    baseline = compute_plan_digest(_plan())
    mutated = compute_plan_digest(_plan(**override))
    assert mutated != baseline


def test_digest_handles_non_ascii_without_escaping_drift():
    """中文 summary 必须走 ensure_ascii=False 口径(否则摘要口径会漂移)。"""
    plan = _plan(summary="疑似 SSH 爆破,建议封禁源 IP")
    canonical = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert compute_plan_digest(plan) == hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    assert compute_plan_digest(plan) == compute_plan_digest(
        _plan(summary="疑似 SSH 爆破,建议封禁源 IP")
    )


def test_digest_on_real_planner_output():
    """真实规则引擎产出也能算摘要(证明不是只对测试夹具有效)。"""
    plan = plan_response(analyze_risk(RiskEvidence(
        indicator=INDICATOR, log_event_count=30, failed_login_count=30,
    )))
    assert _HEX64.match(compute_plan_digest(plan))


# ---------- compute_plan_digest:单输入契约 ----------

def test_compute_digest_signature_is_single_input():
    params = inspect.signature(compute_plan_digest).parameters
    assert list(params) == ["plan"]
    assert params["plan"].annotation is ResponsePlan
    assert params["plan"].default is params["plan"].empty


# ---------- new_record_id ----------

def test_new_record_id_is_32_hex():
    assert _HEX32.match(new_record_id())


def test_new_record_id_unique():
    ids = {new_record_id() for _ in range(200)}
    assert len(ids) == 200


# ---------- build_audit_record:基本组装 ----------

def test_build_record_defaults_actor_to_system():
    rec = build_audit_record("plan.created", plan=_plan())
    assert rec.actor == SYSTEM_ACTOR


def test_build_record_uses_injected_ts():
    rec = build_audit_record("plan.created", plan=_plan(), ts=FIXED_TS)
    assert rec.ts == FIXED_TS


def test_build_record_default_ts_is_tz_aware_utc():
    """省略 ts 时取当前 UTC —— 必须是 tz-aware(naive 会破坏审计流排序)。"""
    rec = build_audit_record("policy.evaluated")
    assert rec.ts.tzinfo is not None
    assert rec.ts.utcoffset() == timedelta(0)


def test_build_record_generates_unique_ids():
    a = build_audit_record("policy.evaluated", ts=FIXED_TS)
    b = build_audit_record("policy.evaluated", ts=FIXED_TS)
    assert a.id != b.id
    assert _HEX32.match(a.id)


def test_build_record_passes_through_optional_fields():
    rec = build_audit_record(
        "approval.decided",
        actor="analyst-1",
        incident_id="inc-001",
        thread_id="thread-abc",
        interrupt_id="int-001",
        outcome="approved",
        reason="证据充分",
        plan=_plan(),
        detail={"gated_actions": ["block_ip"]},
        ts=FIXED_TS,
    )
    assert rec.event == "approval.decided"
    assert rec.actor == "analyst-1"
    assert rec.incident_id == "inc-001"
    assert rec.thread_id == "thread-abc"
    assert rec.interrupt_id == "int-001"
    assert rec.outcome == "approved"
    assert rec.reason == "证据充分"
    assert rec.detail["gated_actions"] == ["block_ip"]


def test_build_record_defaults_optional_fields_to_none():
    rec = build_audit_record("policy.evaluated", ts=FIXED_TS)
    assert rec.incident_id is None
    assert rec.thread_id is None
    assert rec.interrupt_id is None
    assert rec.outcome is None
    assert rec.reason is None
    assert rec.plan_digest is None
    assert rec.detail == {}


# ---------- build_audit_record:摘要由 plan 推导,不可注入 ----------

def test_digest_derived_from_plan():
    plan = _plan()
    rec = build_audit_record("plan.created", plan=plan, ts=FIXED_TS)
    assert rec.plan_digest == compute_plan_digest(plan)


def test_no_plan_means_no_digest():
    """没有关联计划的事件(如 approval.timeout)不写摘要。"""
    rec = build_audit_record("approval.timeout", ts=FIXED_TS)
    assert rec.plan_digest is None


def test_digest_cannot_be_injected():
    """护栏:签名里没有 plan_digest —— 摘要只能由 plan 推导。

    若允许调用方传摘要,就会出现两个真相源:传错值/传旧值都不会被发现,
    摘要也就失去"比对计划是否被改动"的作用。
    """
    params = inspect.signature(build_audit_record).parameters
    assert "plan_digest" not in params


def test_different_plan_yields_different_digest_in_record():
    a = build_audit_record("plan.created", plan=_plan(score=70), ts=FIXED_TS)
    b = build_audit_record("plan.created", plan=_plan(score=71), ts=FIXED_TS)
    assert a.plan_digest != b.plan_digest


# ---------- detail 拷贝语义 ----------

def test_detail_is_copied_not_aliased():
    """调用方后续修改原 dict 不得影响已构造的记录(审计记录必须冻结)。

    必须是**深**拷贝:浅拷贝只隔离最外层,嵌套 list 仍与调用方共享 ——
    事后一次 append 就能改写已落库的审计证据。
    """
    source = {"gated_actions": ["block_ip"]}
    rec = build_audit_record("policy.evaluated", detail=source, ts=FIXED_TS)
    source["gated_actions"].append("isolate_host")
    source["new_key"] = "污染"
    assert rec.detail["gated_actions"] == ["block_ip"]
    assert "new_key" not in rec.detail


def test_detail_nested_dict_is_deep_copied():
    """嵌套 dict 同样隔离 —— 浅拷贝在这里会漏。"""
    source = {"policy": {"gated": ["block_ip"], "version": "phase8.1"}}
    rec = build_audit_record("policy.evaluated", detail=source, ts=FIXED_TS)
    source["policy"]["gated"].append("isolate_host")
    source["policy"]["version"] = "tampered"
    assert rec.detail["policy"]["gated"] == ["block_ip"]
    assert rec.detail["policy"]["version"] == "phase8.1"


def test_detail_none_yields_independent_dict():
    a = build_audit_record("policy.evaluated", ts=FIXED_TS)
    b = build_audit_record("policy.evaluated", ts=FIXED_TS)
    a.detail["k"] = "v"
    assert b.detail == {}


# ---------- 构造仍受 schema 约束(不在本模块绕过校验)----------

def test_invalid_event_still_rejected():
    """审计词汇表封闭:构造层不得放行枚举外的事件。"""
    with pytest.raises(ValidationError):
        build_audit_record("user.did.something", ts=FIXED_TS)


def test_returned_object_is_audit_record():
    assert isinstance(build_audit_record("policy.evaluated", ts=FIXED_TS), AuditRecord)


# ---------- 纯度护栏:本模块不碰 I/O ----------

def _imported_modules(module) -> set[str]:
    """模块真实 import 的模块名(经 AST,不受注释/文档字符串里的同名文字干扰)。"""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _code_strings(module) -> list[str]:
    """模块里所有**非 docstring** 的字符串字面量。

    注释根本不在 AST 里,文档字符串被显式排除 —— 因此
    "源码里提到了 sqlite3" 不会误报,只有真正的代码字面量才会被检查。
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (
            ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
        )):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_module_does_no_io():
    """源码级护栏:纯计算模块不得引入 I/O 依赖。

    store.py 才是唯一落盘的地方;audit.py 一旦开始 I/O,
    "构造"与"持久化"的边界就没了。
    """
    assert "sqlite3" not in _imported_modules(audit_module)
    for text in _code_strings(audit_module):
        for forbidden in ("open(", "write_text", "sqlite3"):
            assert forbidden not in text, (forbidden, text)


def test_module_imports_no_policy_module():
    """分工护栏:audit.py 不做策略判定,不依赖 policy.py。"""
    assert "app.security.policy" not in _imported_modules(audit_module)
    assert not hasattr(audit_module, "evaluate_policy")
