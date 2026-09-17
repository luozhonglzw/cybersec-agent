"""AuditRecord schema contract 测试(Phase 8.1)。

覆盖:必填性、event 枚举封闭性、plan_digest 的 sha256 格式校验、
可选字段默认值、tz-aware 时间、JSON 往返。

全部 hermetic:只构造内存对象,不读写任何数据文件。
"""
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.audit import SYSTEM_ACTOR, AuditRecord

VALID_DIGEST = "a" * 64  # 64 位小写十六进制


def _payload(**overrides) -> dict:
    base = dict(
        id="audit-001",
        ts=datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc),
        actor=SYSTEM_ACTOR,
        event="policy.evaluated",
    )
    base.update(overrides)
    return base


# ---------- 合法构造与默认值 ----------

def test_minimal_record_valid():
    r = AuditRecord(**_payload())
    assert r.actor == "system"
    assert r.incident_id is None
    assert r.thread_id is None
    assert r.interrupt_id is None
    assert r.outcome is None
    assert r.reason is None
    assert r.plan_digest is None
    assert r.detail == {}


def test_full_record_valid():
    r = AuditRecord(**_payload(
        event="approval.decided",
        actor="analyst-1",
        incident_id="inc-001",
        thread_id="thread-abc",
        interrupt_id="int-001",
        outcome="approved",
        reason="证据充分",
        plan_digest=VALID_DIGEST,
        detail={"gated_actions": ["block_ip"]},
    ))
    assert r.event == "approval.decided"
    assert r.plan_digest == VALID_DIGEST
    assert r.detail["gated_actions"] == ["block_ip"]


# ---------- event 枚举封闭性 ----------

def test_all_declared_events_accepted():
    for event in (
        "plan.created", "policy.evaluated", "approval.requested",
        "approval.decided", "approval.timeout",
    ):
        assert AuditRecord(**_payload(event=event)).event == event


def test_unknown_event_rejected():
    """审计词汇表封闭:不允许自由发挥事件类型。"""
    with pytest.raises(ValidationError):
        AuditRecord(**_payload(event="user.did.something"))


def test_event_is_required():
    payload = _payload()
    payload.pop("event")
    with pytest.raises(ValidationError):
        AuditRecord(**payload)


# ---------- 必填字段 ----------

def test_id_required_non_empty():
    with pytest.raises(ValidationError):
        AuditRecord(**_payload(id=""))


def test_actor_required_non_empty():
    with pytest.raises(ValidationError):
        AuditRecord(**_payload(actor=""))


def test_ts_required():
    payload = _payload()
    payload.pop("ts")
    with pytest.raises(ValidationError):
        AuditRecord(**payload)


def test_ts_keeps_timezone():
    r = AuditRecord(**_payload())
    assert r.ts.tzinfo is not None
    assert r.ts.utcoffset() == timezone.utc.utcoffset(None)


# ---------- plan_digest 格式校验 ----------

def test_valid_sha256_digest_accepted():
    assert AuditRecord(**_payload(plan_digest=VALID_DIGEST)).plan_digest == VALID_DIGEST


def test_none_digest_accepted():
    """没有 plan 的事件(如 approval.timeout)不需要摘要。"""
    assert AuditRecord(**_payload(plan_digest=None)).plan_digest is None


def test_short_digest_rejected():
    with pytest.raises(ValidationError):
        AuditRecord(**_payload(plan_digest="abc123"))


def test_non_hex_digest_rejected():
    with pytest.raises(ValidationError):
        AuditRecord(**_payload(plan_digest="z" * 64))


def test_uppercase_digest_rejected():
    """摘要统一小写十六进制,大小写混用会让比对失效。"""
    with pytest.raises(ValidationError):
        AuditRecord(**_payload(plan_digest="A" * 64))


def test_overlong_digest_rejected():
    with pytest.raises(ValidationError):
        AuditRecord(**_payload(plan_digest="a" * 65))


# ---------- detail 自包含取证负载 ----------

def test_detail_defaults_to_independent_dict():
    """default_factory:两个实例不共享同一个 dict(否则会互相污染)。"""
    a = AuditRecord(**_payload())
    b = AuditRecord(**_payload())
    a.detail["k"] = "v"
    assert b.detail == {}


def test_detail_accepts_nested_payload():
    r = AuditRecord(**_payload(detail={
        "outcome": "require_approval",
        "gated_actions": ["block_ip"],
        "policy_reasons": ["动作按属性需要人工审批"],
    }))
    assert r.detail["gated_actions"] == ["block_ip"]
    assert r.detail["outcome"] == "require_approval"


# ---------- 序列化 ----------

def test_json_round_trip():
    r = AuditRecord(**_payload(
        event="approval.requested", thread_id="thread-abc",
        plan_digest=VALID_DIGEST, detail={"indicator": "203.0.113.66"},
    ))
    again = AuditRecord.model_validate_json(r.model_dump_json())
    assert again == r


def test_model_dump_json_is_plain_json():
    r = AuditRecord(**_payload())
    parsed = json.loads(r.model_dump_json())
    assert parsed["event"] == "policy.evaluated"
    assert parsed["detail"] == {}
