"""SQLite / PostgreSQL **共享**行为契约测试(Phase v0.2.0-M1b)。

**同一套断言**,参数化跑在两个后端上 —— 不是写两份等价测试然后声称它们一致。

```
PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe -m pytest \
  tests/test_postgres/test_store_contract.py -q -p no:randomly --basetemp=<唯一路径>
```

## 关于 PostgreSQL 侧是**共享库**

append-only 意味着用例写入的行**删不掉**,库会跨用例、跨运行累积。
所以本模块的**所有**断言都按 `thread_id` / `incident_id` **限定作用域**,
绝不写"全库只有 N 行"这种断言。唯一例外是 `pending_action_rows()` 的
**无过滤**分支 —— 那里只断言"我这几行在结果里",不断言总量。

**由此产生的一条硬约束**:不得向 `action_requests` / `audit_logs` 注入脏数据。
那两行会被无过滤读取(`pending_action_rows()` 是 `triage.py:545` 的真实路径;
`list_audit()` 亦有无过滤用法)扫到,一次注入就让**永久**抛错 ——
"验证脏数据被检出"会顺手把别的契约用例变成永久失败。因此脏数据端到端
只注入 `incidents`(只按主键读,影响完全限定在那一行),另外两条
(`policy_reasons` 非数组、`detail_json` 非法)走**映射层**断言 ——
它们验证的本来就是"行 → 对象"这一步。

## 关于后端差异

三处差异被**显式暴露**而不是被宽泛的 `except` 掩盖:

| 差异 | SQLite | PostgreSQL |
|---|---|---|
| 占位符 | `?` | `%s` |
| 直连目标 | 文件路径 | DSN |
| 重复主键异常 | `sqlite3.IntegrityError` | `psycopg.errors.IntegrityError` |
| append-only 拒绝 | `sqlite3.IntegrityError`(库层触发器) | `psycopg.errors.InsufficientPrivilege`(**权限层**) |

最后一行尤其重要:两者都 fail closed,但**拦住的层不同** ——
PostgreSQL 的运行时角色根本没有 UPDATE 权限,请求到不了触发器。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.audit import AuditRecord
from app.security.audit import build_audit_record, compute_plan_digest
from app.security.store_protocol import AuditStore
from tests.test_postgres._store_backends import (
    INDICATOR,
    TS,
    TS_LATER,
    Backend,
    make_action,
    make_incident,
    make_request,
    postgres_backend,
    sqlite_backend,
    uid,
)

pytestmark = pytest.mark.postgres


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path: Path, pg_app_dsn: str) -> Backend:
    """同一份契约,两个后端。

    SQLite 用 `tmp_path` 下的全新库(用例间零共享);
    PostgreSQL 用受限的 application-runtime 角色连真实实例。

    两个分支都用 `yield`(而不是一个 `return` 一个 `yield`)——
    函数里只要出现 `yield` 就整体是生成器,`return X` 会变成
    `StopIteration.value`,夹具拿不到值。
    """
    if request.param == "sqlite":
        yield sqlite_backend(str(tmp_path / "audit.db"))
        return
    pg = postgres_backend(pg_app_dsn)
    try:
        yield pg
    finally:
        pg.store.close()


# ===========================================================================
# 1. Protocol 兼容性
# ===========================================================================


def test_backend_satisfies_the_audit_store_protocol(backend: Backend) -> None:
    assert isinstance(backend.store, AuditStore)


def test_backend_exposes_exactly_the_eight_contract_methods(
    backend: Backend,
) -> None:
    """公开方法集合与契约**双向相等** —— 既不少,也不多出未声明的行为方法。

    `close` 是**唯一**被允许的例外:它是 `PostgresAuditStore` 的连接池生命周期
    方法,刻意**不属于** `AuditStore` 契约(契约只管 8 个行为方法;
    构造与生命周期是各后端自己的事 —— SQLite 每次操作新开连接,因此没有
    `close`)。这里把它显式减掉,而不是放宽成"包含关系",这样任何**其他**
    多出来的公开方法仍会被抓住。
    """
    expected = {
        "record_incident", "record_action_request", "append_audit",
        "get_incident", "get_approval_request", "list_action_rows",
        "pending_action_rows", "list_audit",
    }
    lifecycle = {"close"}
    actual = {
        name for name in dir(backend.store)
        if not name.startswith("_") and callable(getattr(backend.store, name))
    }
    assert actual - lifecycle == expected, f"多出或缺少方法: {actual - lifecycle ^ expected}"
    assert actual - expected <= lifecycle, (
        f"出现了契约与生命周期之外的公开方法: {actual - expected - lifecycle}"
    )


# ===========================================================================
# 2. incidents 往返
# ===========================================================================


def test_incident_round_trip(backend: Backend) -> None:
    iid = uid("inc")
    backend.store.record_incident(make_incident(incident_id=iid))
    got = backend.store.get_incident(iid)
    assert got is not None
    assert got.id == iid
    assert got.indicator == INDICATOR
    assert got.risk_level == "high"
    assert got.score == 70


def test_incident_plan_survives_json_round_trip(backend: Backend) -> None:
    iid = uid("inc")
    incident = make_incident(incident_id=iid)
    backend.store.record_incident(incident)
    got = backend.store.get_incident(iid)
    assert got is not None
    assert got.plan == incident.plan


def test_get_incident_missing_returns_none(backend: Backend) -> None:
    assert backend.store.get_incident(uid("absent")) is None


def test_incident_created_at_kept_tz_aware(backend: Backend) -> None:
    iid = uid("inc")
    backend.store.record_incident(make_incident(incident_id=iid))
    got = backend.store.get_incident(iid)
    assert got is not None
    assert got.created_at.tzinfo is not None
    assert got.created_at == TS


def test_incident_rejects_naive_datetime(backend: Backend) -> None:
    naive = datetime(2026, 9, 17, 10, 0, 0)
    with pytest.raises(ValueError):
        backend.store.record_incident(
            make_incident(incident_id=uid("inc"), created_at=naive)
        )


def test_incident_non_utc_offset_normalized_to_utc(backend: Backend) -> None:
    iid = uid("inc")
    plus8 = timezone(timedelta(hours=8))
    local = datetime(2026, 9, 17, 18, 0, 0, tzinfo=plus8)
    backend.store.record_incident(make_incident(incident_id=iid, created_at=local))
    got = backend.store.get_incident(iid)
    assert got is not None
    assert got.created_at == datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
    assert got.created_at.utcoffset() == timedelta(0)


def test_duplicate_incident_pk_rejected(backend: Backend) -> None:
    iid = uid("inc")
    backend.store.record_incident(make_incident(incident_id=iid))
    with pytest.raises(backend.integrity_exc):
        backend.store.record_incident(make_incident(incident_id=iid))


# ===========================================================================
# 3. action_requests 往返与多行原子性
# ===========================================================================


def test_one_row_per_action(backend: Backend) -> None:
    tid = uid("thread")
    actions = [make_action("block_ip"), make_action("monitor"), make_action("escalate")]
    backend.store.record_action_request(make_request(thread_id=tid, actions=actions))
    rows = backend.store.list_action_rows(thread_id=tid)
    assert len(rows) == 3
    assert [r["action"]["action_type"] for r in rows] == [
        "block_ip", "monitor", "escalate"
    ]


def test_record_action_request_returns_ids_in_action_order(backend: Backend) -> None:
    tid = uid("thread")
    actions = [make_action("block_ip"), make_action("monitor"), make_action("escalate")]
    ids = backend.store.record_action_request(
        make_request(thread_id=tid, actions=actions)
    )
    assert len(ids) == 3
    assert len(set(ids)) == 3
    rows = backend.store.list_action_rows(thread_id=tid)
    assert [r["row_id"] for r in rows] == ids


def test_approval_request_round_trip(backend: Backend) -> None:
    tid = uid("thread")
    actions = [make_action("block_ip"), make_action("isolate_host")]
    original = make_request(thread_id=tid, actions=actions)
    backend.store.record_action_request(original)
    got = backend.store.get_approval_request(tid)
    assert got is not None
    assert got.thread_id == tid
    assert got.indicator == INDICATOR
    assert got.risk_level == "high"
    assert got.score == 70
    assert got.actions == actions
    assert got.policy_reasons == ["动作按属性需要人工审批"]
    assert got.requested_at == TS


def test_get_approval_request_missing_returns_none(backend: Backend) -> None:
    assert backend.store.get_approval_request(uid("absent")) is None


def test_request_level_fields_repeated_per_row(backend: Backend) -> None:
    """request 级字段在每行重复存一份 —— 这是刻意的冗余。"""
    tid = uid("thread")
    actions = [make_action("block_ip"), make_action("monitor")]
    backend.store.record_action_request(make_request(thread_id=tid, actions=actions))
    rows = backend.store.list_action_rows(thread_id=tid)
    assert len(rows) == 2
    assert {r["thread_id"] for r in rows} == {tid}
    assert {r["indicator"] for r in rows} == {INDICATOR}
    assert {r["score"] for r in rows} == {70}
    assert {tuple(r["policy_reasons"]) for r in rows} == {
        ("动作按属性需要人工审批",)
    }


def test_policy_reasons_survive_round_trip(backend: Backend) -> None:
    tid = uid("thread")
    reasons = ["原因一", "reason-two", "原因 3"]
    backend.store.record_action_request(
        make_request(thread_id=tid, policy_reasons=reasons)
    )
    got = backend.store.get_approval_request(tid)
    assert got is not None
    assert got.policy_reasons == reasons


def test_action_flags_round_trip_as_bool(backend: Backend) -> None:
    tid = uid("thread")
    actions = [make_action("block_ip"), make_action("reset_credentials")]
    backend.store.record_action_request(make_request(thread_id=tid, actions=actions))
    got = backend.store.get_approval_request(tid)
    assert got is not None
    for action in got.actions:
        assert isinstance(action.requires_approval, bool)
        assert isinstance(action.reversible, bool)
    assert got.actions[0].reversible is True
    assert got.actions[1].reversible is False


def test_incident_id_is_nullable(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.record_action_request(make_request(thread_id=tid))
    rows = backend.store.list_action_rows(thread_id=tid)
    assert rows and all(r["incident_id"] is None for r in rows)


def test_incident_id_is_persisted_when_given(backend: Backend) -> None:
    tid, iid = uid("thread"), uid("inc")
    backend.store.record_action_request(
        make_request(thread_id=tid), incident_id=iid
    )
    rows = backend.store.list_action_rows(thread_id=tid)
    assert rows and all(r["incident_id"] == iid for r in rows)


def test_action_rows_filter_by_thread_id(backend: Backend) -> None:
    tid_a, tid_b = uid("thread-a"), uid("thread-b")
    backend.store.record_action_request(make_request(thread_id=tid_a))
    backend.store.record_action_request(make_request(thread_id=tid_b))
    assert len(backend.store.list_action_rows(thread_id=tid_a)) == 1
    assert len(backend.store.list_action_rows(thread_id=tid_b)) == 1


def test_action_rows_filter_by_incident_id(backend: Backend) -> None:
    tid_a, tid_b = uid("thread-a"), uid("thread-b")
    iid = uid("inc")
    backend.store.record_action_request(make_request(thread_id=tid_a), incident_id=iid)
    backend.store.record_action_request(make_request(thread_id=tid_b))
    rows = backend.store.list_action_rows(incident_id=iid)
    assert [r["thread_id"] for r in rows] == [tid_a]


def test_action_rows_projection_shape(backend: Backend) -> None:
    """投影的**键集合**逐字固定 —— `seq` 是实现细节,不得泄漏到投影里。"""
    tid = uid("thread")
    backend.store.record_action_request(make_request(thread_id=tid))
    row = backend.store.list_action_rows(thread_id=tid)[0]
    assert set(row) == {
        "row_id", "incident_id", "thread_id", "indicator", "risk_level",
        "score", "summary", "policy_reasons", "action", "requested_at",
    }
    assert set(row["action"]) == {
        "action_type", "priority", "target", "rationale",
        "requires_approval", "reversible",
    }
    assert isinstance(row["requested_at"], datetime)
    assert row["requested_at"].tzinfo is not None


def test_action_request_rejects_naive_datetime(backend: Backend) -> None:
    with pytest.raises(ValueError):
        backend.store.record_action_request(
            make_request(
                thread_id=uid("thread"),
                requested_at=datetime(2026, 9, 17, 10, 0, 0),
            )
        )


def test_duplicate_action_row_pk_rejected(backend: Backend, monkeypatch) -> None:
    tid = uid("thread")
    actions = [make_action("block_ip"), make_action("monitor")]
    frozen = uuid.uuid4()
    monkeypatch.setattr(uuid, "uuid4", lambda: frozen)
    with pytest.raises(backend.integrity_exc):
        backend.store.record_action_request(
            make_request(thread_id=tid, actions=actions)
        )
    monkeypatch.undo()
    assert backend.store.list_action_rows(thread_id=tid) == []


def test_multirow_write_is_atomic(backend: Backend, monkeypatch) -> None:
    """多行写入是**原子**的:任一行失败 ⇒ 一行都不落库。

    构造方式:让 `uuid4` 恒返回同一个值,于是第 2 行的主键必然与第 1 行冲突。
    若实现没有把多行包在同一个事务里,库里就会留下"写了一半的审批单" ——
    那正是一个无法审批、也无法撤销的脏状态。
    """
    tid = uid("thread")
    actions = [make_action("block_ip"), make_action("monitor")]
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=7))
    with pytest.raises(backend.integrity_exc):
        backend.store.record_action_request(
            make_request(thread_id=tid, actions=actions)
        )
    monkeypatch.undo()

    assert backend.store.list_action_rows(thread_id=tid) == [], (
        "多行写入失败后不得留下任何一行"
    )
    assert backend.store.get_approval_request(tid) is None


# ===========================================================================
# 4. pending 派生(状态不落库)
# ===========================================================================


def test_pending_rows_include_unanswered_request(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.record_action_request(make_request(thread_id=tid))
    assert len(backend.store.pending_action_rows(thread_id=tid)) == 1


def test_pending_rows_exclude_decided_thread(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.record_action_request(make_request(thread_id=tid))
    backend.store.append_audit(
        build_audit_record("approval.decided", thread_id=tid, outcome="approved")
    )
    assert backend.store.pending_action_rows(thread_id=tid) == []


def test_pending_rows_cleared_by_timeout(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.record_action_request(make_request(thread_id=tid))
    backend.store.append_audit(build_audit_record("approval.timeout", thread_id=tid))
    assert backend.store.pending_action_rows(thread_id=tid) == []


def test_pending_rows_scope_decision_to_its_own_thread(backend: Backend) -> None:
    tid_a, tid_b = uid("thread-a"), uid("thread-b")
    backend.store.record_action_request(make_request(thread_id=tid_a))
    backend.store.record_action_request(make_request(thread_id=tid_b))
    backend.store.append_audit(
        build_audit_record("approval.decided", thread_id=tid_a, outcome="denied")
    )
    assert backend.store.pending_action_rows(thread_id=tid_a) == []
    assert len(backend.store.pending_action_rows(thread_id=tid_b)) == 1


def test_pending_rows_ignore_unrelated_events(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.record_action_request(make_request(thread_id=tid))
    for event in ("plan.created", "policy.evaluated", "approval.requested"):
        backend.store.append_audit(build_audit_record(event, thread_id=tid))
    assert len(backend.store.pending_action_rows(thread_id=tid)) == 1


def test_pending_rows_empty_when_no_request(backend: Backend) -> None:
    assert backend.store.pending_action_rows(thread_id=uid("absent")) == []


def test_pending_rows_unfiltered_contains_my_pending_thread(backend: Backend) -> None:
    """无过滤分支:只断言"我这几行在里面"。

    PostgreSQL 侧是**共享且不可清理**的库,断言总量必然 flaky。
    """
    tid = uid("thread")
    backend.store.record_action_request(make_request(thread_id=tid))
    threads = {row["thread_id"] for row in backend.store.pending_action_rows()}
    assert tid in threads


# ===========================================================================
# 5. audit_logs 往返 / 过滤 / 排序 / limit
# ===========================================================================


def test_append_and_list_audit_round_trip(backend: Backend) -> None:
    tid = uid("thread")
    record = build_audit_record(
        "plan.created", thread_id=tid, incident_id=uid("inc"), detail={"k": "v"}
    )
    backend.store.append_audit(record)
    got = backend.store.list_audit(thread_id=tid)
    assert got == [record]


def test_audit_plan_digest_persisted(backend: Backend) -> None:
    tid = uid("thread")
    plan = make_incident(incident_id=uid("inc")).plan
    digest = compute_plan_digest(plan)
    backend.store.append_audit(
        build_audit_record("plan.created", thread_id=tid, plan=plan)
    )
    got = backend.store.list_audit(thread_id=tid)
    assert got[0].plan_digest == digest


def test_audit_nullable_fields_round_trip_as_none(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.append_audit(build_audit_record("plan.created", thread_id=tid))
    got = backend.store.list_audit(thread_id=tid)[0]
    assert got.incident_id is None
    assert got.interrupt_id is None
    assert got.outcome is None
    assert got.reason is None
    assert got.plan_digest is None


def test_audit_detail_round_trip_with_non_ascii(backend: Backend) -> None:
    tid = uid("thread")
    detail = {"结论": "疑似爆破", "nested": {"列表": [1, 2, 3]}}
    backend.store.append_audit(
        build_audit_record("plan.created", thread_id=tid, detail=detail)
    )
    got = backend.store.list_audit(thread_id=tid)[0]
    assert got.detail == detail


def test_audit_ts_round_trip_tz_aware(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.append_audit(build_audit_record("plan.created", thread_id=tid, ts=TS))
    got = backend.store.list_audit(thread_id=tid)[0]
    assert got.ts == TS
    assert got.ts.tzinfo is not None


def test_audit_rejects_naive_datetime(backend: Backend) -> None:
    with pytest.raises(ValueError):
        backend.store.append_audit(
            AuditRecord(
                id=uuid.uuid4().hex,
                ts=datetime(2026, 9, 17, 10, 0, 0),
                actor="system",
                event="plan.created",
            )
        )


def test_duplicate_audit_pk_rejected(backend: Backend) -> None:
    tid = uid("thread")
    record = build_audit_record("plan.created", thread_id=tid)
    backend.store.append_audit(record)
    with pytest.raises(backend.integrity_exc):
        backend.store.append_audit(record)


def test_list_audit_ordered_by_ts_not_insertion_order(backend: Backend) -> None:
    tid = uid("thread")
    backend.store.append_audit(
        build_audit_record("plan.created", thread_id=tid, ts=TS_LATER)
    )
    backend.store.append_audit(
        build_audit_record("plan.created", thread_id=tid, ts=TS)
    )
    got = backend.store.list_audit(thread_id=tid)
    assert [r.ts for r in got] == [TS, TS_LATER]


def test_list_audit_identical_timestamps_tie_break_is_deterministic(
    backend: Backend,
) -> None:
    """同一 `ts` 的多行:插入顺序 == 读取顺序,且**反复读结果一致**。

    SQLite 靠 `rowid`、PostgreSQL 靠 `seq` —— 两者的共同保证是
    "同一份数据反复读得到同一顺序",不是"全局因果序"。
    """
    tid = uid("thread")
    ids = []
    for _ in range(5):
        record = build_audit_record("plan.created", thread_id=tid, ts=TS)
        ids.append(record.id)
        backend.store.append_audit(record)

    first = [r.id for r in backend.store.list_audit(thread_id=tid)]
    second = [r.id for r in backend.store.list_audit(thread_id=tid)]
    assert first == ids
    assert first == second, "同一份数据反复读必须得到同一顺序"


def test_list_audit_filters(backend: Backend) -> None:
    tid_a, tid_b = uid("thread-a"), uid("thread-b")
    iid = uid("inc")
    backend.store.append_audit(
        build_audit_record("plan.created", thread_id=tid_a, incident_id=iid)
    )
    backend.store.append_audit(
        build_audit_record("policy.evaluated", thread_id=tid_a, incident_id=iid)
    )
    backend.store.append_audit(build_audit_record("plan.created", thread_id=tid_b))

    assert len(backend.store.list_audit(thread_id=tid_a)) == 2
    assert len(backend.store.list_audit(incident_id=iid)) == 2
    assert len(backend.store.list_audit(event="policy.evaluated", thread_id=tid_a)) == 1
    assert len(
        backend.store.list_audit(thread_id=tid_a, event="plan.created")
    ) == 1
    assert backend.store.list_audit(event="approval.timeout", thread_id=tid_a) == []


def test_list_audit_default_is_ascending_without_limit(backend: Backend) -> None:
    tid = uid("thread")
    for offset in range(4):
        backend.store.append_audit(
            build_audit_record(
                "plan.created", thread_id=tid, ts=TS + timedelta(minutes=offset)
            )
        )
    got = backend.store.list_audit(thread_id=tid)
    assert len(got) == 4
    assert got == sorted(got, key=lambda r: r.ts)


def test_list_audit_limit_bounds_result(backend: Backend) -> None:
    tid = uid("thread")
    for offset in range(4):
        backend.store.append_audit(
            build_audit_record(
                "plan.created", thread_id=tid, ts=TS + timedelta(minutes=offset)
            )
        )
    assert len(backend.store.list_audit(thread_id=tid, limit=2)) == 2
    assert len(backend.store.list_audit(thread_id=tid, limit=0)) == 0


def test_list_audit_descending_reverses_order(backend: Backend) -> None:
    tid = uid("thread")
    stamps = [TS + timedelta(minutes=o) for o in range(3)]
    for stamp in stamps:
        backend.store.append_audit(
            build_audit_record("plan.created", thread_id=tid, ts=stamp)
        )
    got = backend.store.list_audit(thread_id=tid, descending=True)
    assert [r.ts for r in got] == list(reversed(stamps))


def test_list_audit_descending_deterministic_on_tied_ts(backend: Backend) -> None:
    tid = uid("thread")
    ids = []
    for _ in range(4):
        record = build_audit_record("plan.created", thread_id=tid, ts=TS)
        ids.append(record.id)
        backend.store.append_audit(record)
    got = [r.id for r in backend.store.list_audit(thread_id=tid, descending=True)]
    assert got == list(reversed(ids))


def test_list_audit_descending_with_limit(backend: Backend) -> None:
    tid = uid("thread")
    stamps = [TS + timedelta(minutes=o) for o in range(4)]
    for stamp in stamps:
        backend.store.append_audit(
            build_audit_record("plan.created", thread_id=tid, ts=stamp)
        )
    got = backend.store.list_audit(thread_id=tid, descending=True, limit=2)
    assert [r.ts for r in got] == list(reversed(stamps))[:2]


# ===========================================================================
# 6. 未知 id / 空结果
# ===========================================================================


def test_unknown_ids_yield_empty_results(backend: Backend) -> None:
    absent = uid("absent")
    assert backend.store.get_incident(absent) is None
    assert backend.store.get_approval_request(absent) is None
    assert backend.store.list_action_rows(thread_id=absent) == []
    assert backend.store.list_action_rows(incident_id=absent) == []
    assert backend.store.pending_action_rows(thread_id=absent) == []
    assert backend.store.list_audit(thread_id=absent) == []
    assert backend.store.list_audit(incident_id=absent) == []


def _seed_row(backend: Backend, table: str) -> str:
    """往目标表写一行,返回它的主键。

    append-only 的行级触发器**只在有行被处理时**才触发(SQLite 侧),
    所以"拒绝 UPDATE/DELETE"这类验证必须先有**真实存在的行** ——
    `WHERE 1 = 0` 命中 0 行会让断言恒真。
    """
    tid = uid("seed")
    if table == "incidents":
        iid = uid("inc")
        backend.store.record_incident(make_incident(incident_id=iid))
        return iid
    if table == "action_requests":
        return backend.store.record_action_request(
            make_request(thread_id=tid)
        )[0]
    record = build_audit_record("plan.created", thread_id=tid)
    backend.store.append_audit(record)
    return record.id


# ===========================================================================
# 7. append-only(库层,绕过 store 直连)
# ===========================================================================


@pytest.mark.parametrize(
    "table", ["incidents", "action_requests", "audit_logs"]
)
def test_append_only_rejects_update(backend: Backend, table: str) -> None:
    """命中**真实行**的 UPDATE 必须被拒。

    两个后端拦在不同层,但都 fail closed:
        SQLite     → 库层行触发器 RAISE(ABORT)
        PostgreSQL → 运行时角色没有 UPDATE 权限(语句级就被拒,到不了触发器)
    """
    row_id = _seed_row(backend, table)
    with pytest.raises(backend.append_only_update_exc):
        with backend.raw() as conn:
            conn.execute(
                f"UPDATE {table} SET id = id WHERE id = {backend.placeholder}",
                (row_id,),
            )


@pytest.mark.parametrize(
    "table", ["incidents", "action_requests", "audit_logs"]
)
def test_append_only_rejects_delete(backend: Backend, table: str) -> None:
    row_id = _seed_row(backend, table)
    with pytest.raises(backend.append_only_delete_exc):
        with backend.raw() as conn:
            conn.execute(
                f"DELETE FROM {table} WHERE id = {backend.placeholder}",
                (row_id,),
            )


def test_append_only_update_did_not_change_the_row(backend: Backend) -> None:
    """被拒之后,行内容必须原样 —— 拒绝不能是"先改后报错"。"""
    iid = uid("inc")
    backend.store.record_incident(make_incident(incident_id=iid))
    with pytest.raises(backend.append_only_update_exc):
        with backend.raw() as conn:
            conn.execute(
                f"UPDATE incidents SET summary = 'tampered'"
                f" WHERE id = {backend.placeholder}",
                (iid,),
            )
    got = backend.store.get_incident(iid)
    assert got is not None and got.summary == "疑似 SSH 爆破"


# ===========================================================================
# 8. 脏数据在读取边界暴露(不静默跳过)
# ===========================================================================
#
# 为什么脏数据只注入 `incidents`,而 `action_requests` / `audit_logs` 走
# **映射层**断言:
#
# PostgreSQL 侧是**共享且不可清理**的库(append-only ⇒ 脏行删不掉)。
# 往 `action_requests` 注入一行非数组的 `policy_reasons`,会让**无过滤**的
# `pending_action_rows()` 从此永久抛错 —— 那是 `triage.py:545` 的真实调用路径,
# 于是"验证脏数据被检出"会顺手把另一个契约用例变成永久失败。
# 往 `audit_logs` 注入同理会影响无过滤的 `list_audit()`。
#
# `incidents` 只按主键读(`get_incident`),注入的影响被**完全限定在那一行**,
# 所以那里做端到端验证是安全的。另外两条走映射层 —— 它们验证的本来就是
# "行 → 对象"这一步的行为,不需要真的落库。


def test_malformed_plan_json_raises_on_read(backend: Backend) -> None:
    """端到端:`incidents.plan_json` 不是合法 JSON ⇒ `get_incident` 响亮失败。

    抛的是 `pydantic.ValidationError`(不是 `json.JSONDecodeError`)——
    因为 `ResponsePlan.model_validate_json` 把 JSON 解析错误包进了 Pydantic
    的校验错误里。这里断言**实际行为**,而不是我以为的行为。
    """
    iid = uid("inc")
    ph = backend.placeholder
    with backend.raw() as conn:
        conn.execute(
            f"INSERT INTO incidents"
            f" (id, created_at, indicator, risk_level, score, summary, plan_json)"
            f" VALUES ({', '.join([ph] * 7)})",
            (iid, TS.isoformat(), INDICATOR, "high", 70, "s", "not json"),
        )
    with pytest.raises(ValidationError):
        backend.store.get_incident(iid)


def test_row_to_reasons_rejects_non_array_json(backend: Backend) -> None:
    """`policy_reasons` 是合法 JSON 但**不是数组** ⇒ 投影响亮失败。"""
    row = {
        "policy_reasons": '{"a": 1}',
        "action_type": "block_ip",
        "priority": "high",
        "target": INDICATOR,
        "rationale": "r",
        "requires_approval": 1,
        "reversible": 0,
    }
    with pytest.raises(ValueError, match="policy_reasons 列不是 JSON 数组"):
        backend.store._row_to_reasons(row)


def test_row_to_audit_rejects_malformed_detail_json(backend: Backend) -> None:
    """`detail_json` 不是合法 JSON ⇒ 行映射响亮失败。"""
    row = {
        "id": uid("row"),
        "ts": TS.isoformat(),
        "actor": "system",
        "event": "plan.created",
        "incident_id": None,
        "thread_id": None,
        "interrupt_id": None,
        "outcome": None,
        "reason": None,
        "plan_digest": None,
        "detail_json": "not json",
    }
    with pytest.raises(json.JSONDecodeError):
        backend.store._row_to_audit(row)


def test_row_to_audit_rejects_out_of_vocabulary_event(backend: Backend) -> None:
    """越界事件名 ⇒ Pydantic 在读取边界拒绝(不静默放过)。"""
    row = {
        "id": uid("row"),
        "ts": TS.isoformat(),
        "actor": "system",
        "event": "not.an.event",
        "incident_id": None,
        "thread_id": None,
        "interrupt_id": None,
        "outcome": None,
        "reason": None,
        "plan_digest": None,
        "detail_json": "{}",
    }
    with pytest.raises(ValidationError):
        backend.store._row_to_audit(row)
