"""`seq` 的定序语义、并发写入与 READ COMMITTED 可见性(Phase v0.2.0-M1b)。

对应 Controller 的 TASK 4。**五个场景**逐一覆盖:

1. 相同时间戳
2. 并发插入
3. 回滚的事务与序号空洞
4. READ COMMITTED 下的读者可见性
5. 已提交且可见的行集合上的确定性排序

## 本模块**不**声称的东西(与 TASK 4 的约束一致)

- **不**声称全局因果序:两个事务的 `seq` 大小关系与它们的因果关系无关;
- **不**声称提交序:序列推进不参与事务回滚,先分配的事务可能后提交
  (`test_allocation_order_is_not_commit_order` 构造了可观测的反例);
- **不**声称 exactly-once:回滚会**消耗**序号并留下空洞;
- 唯一声称的是 —— **在"已提交且对读者可见"的行集合上,`ORDER BY (ts, seq)`
  是确定的**:同一份数据反复读得到同一顺序。
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import psycopg
import pytest

from app.security.audit import build_audit_record
from app.security.store_postgres import PostgresAuditStore
from tests.test_postgres._store_backends import TS, uid

pytestmark = pytest.mark.postgres


def _seq_of(dsn: str, record_id: str) -> int | None:
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT seq FROM audit_logs WHERE id = %s", (record_id,))
            row = cur.fetchone()
    return None if row is None else row[0]


def _seqs_for_thread(dsn: str, thread_id: str) -> list[int]:
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT seq FROM audit_logs WHERE thread_id = %s ORDER BY seq",
                (thread_id,),
            )
            return [row[0] for row in cur.fetchall()]


# ===========================================================================
# 1. 相同时间戳
# ===========================================================================


def test_identical_timestamps_order_by_allocation_and_stay_stable(
    pg_app_dsn: str,
) -> None:
    """同一 `ts` 的多行:读取顺序 == 追加顺序,且反复读结果一致。"""
    store = PostgresAuditStore(pg_app_dsn)
    try:
        tid = uid("thread")
        ids = []
        for _ in range(6):
            record = build_audit_record("plan.created", thread_id=tid, ts=TS)
            ids.append(record.id)
            store.append_audit(record)

        first = [r.id for r in store.list_audit(thread_id=tid)]
        second = [r.id for r in store.list_audit(thread_id=tid)]
        third = [r.id for r in store.list_audit(thread_id=tid, descending=True)]

        assert first == ids
        assert first == second, "同一份数据反复读必须得到同一顺序"
        assert third == list(reversed(ids))
    finally:
        store.close()


def test_identical_timestamps_do_not_collapse_into_one_row(pg_app_dsn: str) -> None:
    """同 `ts` 的 6 行必须是 **6 行** —— `seq` 不能把 tie 变成覆盖。"""
    store = PostgresAuditStore(pg_app_dsn)
    try:
        tid = uid("thread")
        for _ in range(6):
            store.append_audit(build_audit_record("plan.created", thread_id=tid, ts=TS))
        assert len(store.list_audit(thread_id=tid)) == 6
        assert len(_seqs_for_thread(pg_app_dsn, tid)) == 6
    finally:
        store.close()


# ===========================================================================
# 2. 并发插入
# ===========================================================================


def test_concurrent_inserts_are_all_recorded_without_loss_or_duplication(
    pg_app_dsn: str,
) -> None:
    """8 个线程 × 每个 5 条:全部落库、无丢失、无重复、`seq` 唯一。

    同一个 store 实例被多线程共用 —— 这同时验证连接池是线程安全的。
    """
    workers, per_worker = 8, 5
    store = PostgresAuditStore(pg_app_dsn, min_size=2, max_size=8)
    tid = uid("thread")
    barrier = threading.Barrier(workers)

    def _worker(index: int) -> list[str]:
        barrier.wait(timeout=30)
        written: list[str] = []
        for _ in range(per_worker):
            record = build_audit_record(
                "plan.created", thread_id=tid, ts=TS + timedelta(seconds=index)
            )
            store.append_audit(record)
            written.append(record.id)
        return written

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batches = [f.result(timeout=120) for f in [pool.submit(_worker, i)
                                                       for i in range(workers)]]
    finally:
        store.close()

    expected = {rid for batch in batches for rid in batch}
    assert len(expected) == workers * per_worker

    reader = PostgresAuditStore(pg_app_dsn)
    try:
        got = [r.id for r in reader.list_audit(thread_id=tid)]
        assert len(got) == workers * per_worker
        assert set(got) == expected, "不得丢失或重复任何一行"

        seqs = _seqs_for_thread(pg_app_dsn, tid)
        assert len(seqs) == len(set(seqs)), "seq 必须唯一"
        assert seqs == sorted(seqs)

        # 反复读结果一致(已提交可见行集合上的确定性)
        assert [r.id for r in reader.list_audit(thread_id=tid)] == got
    finally:
        reader.close()


def test_concurrent_inserts_share_no_seq_values_across_threads(
    pg_app_dsn: str,
) -> None:
    """不同 thread 的并发写入也不会撞 `seq`。"""
    store = PostgresAuditStore(pg_app_dsn, min_size=2, max_size=6)
    threads = [uid("thread") for _ in range(4)]
    barrier = threading.Barrier(len(threads))

    def _worker(tid: str) -> None:
        barrier.wait(timeout=30)
        for _ in range(4):
            store.append_audit(build_audit_record("plan.created", thread_id=tid, ts=TS))

    try:
        with ThreadPoolExecutor(max_workers=len(threads)) as pool:
            for f in [pool.submit(_worker, t) for t in threads]:
                f.result(timeout=120)
    finally:
        store.close()

    all_seqs: list[int] = []
    for tid in threads:
        all_seqs.extend(_seqs_for_thread(pg_app_dsn, tid))
    assert len(all_seqs) == len(set(all_seqs)), "跨 thread 的 seq 也必须唯一"


# ===========================================================================
# 3. 回滚与序号空洞
# ===========================================================================


def test_rolled_back_transaction_consumes_a_seq_and_leaves_a_gap(
    pg_app_dsn: str,
) -> None:
    """回滚 ⇒ 行不存在,但序号**不回收** —— 空洞是设计的一部分。

    这条性质本身是**中性**的:它保证已提交行的相对顺序不受回滚影响,
    也意味着 `seq` 有空洞、**不能**当连续计数使用。
    """
    record_id = uuid.uuid4().hex
    rolled_back_seq: int | None = None

    conn = psycopg.connect(pg_app_dsn, connect_timeout=10)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_logs (id, ts, actor, event, detail_json)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING seq",
                (record_id, TS.isoformat(), "system", "plan.created", "{}"),
            )
            rolled_back_seq = cur.fetchone()[0]
        conn.rollback()
    finally:
        conn.close()

    assert rolled_back_seq is not None
    assert _seq_of(pg_app_dsn, record_id) is None, "回滚后该行必须不存在"

    store = PostgresAuditStore(pg_app_dsn)
    try:
        tid = uid("thread")
        record = build_audit_record("plan.created", thread_id=tid, ts=TS)
        store.append_audit(record)
        committed_seq = _seq_of(pg_app_dsn, record.id)
        assert committed_seq is not None
        assert committed_seq > rolled_back_seq, "序号不因回滚而回收"
    finally:
        store.close()


def test_gap_is_observable_but_ordering_is_unaffected(pg_app_dsn: str) -> None:
    """空洞存在 ⇒ `seq` 不是连续计数;但**顺序**仍然正确。"""
    store = PostgresAuditStore(pg_app_dsn)
    tid = uid("thread")
    try:
        before = [
            build_audit_record("plan.created", thread_id=tid, ts=TS)
            for _ in range(3)
        ]
        for record in before:
            store.append_audit(record)

        conn = psycopg.connect(pg_app_dsn, connect_timeout=10)
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_logs (id, ts, actor, event, detail_json)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (uuid.uuid4().hex, TS.isoformat(), "system",
                     "plan.created", "{}"),
                )
            conn.rollback()
        finally:
            conn.close()

        after = build_audit_record("plan.created", thread_id=tid, ts=TS)
        store.append_audit(after)

        seqs = _seqs_for_thread(pg_app_dsn, tid)
        assert len(seqs) == 4
        assert seqs == sorted(seqs)
        assert seqs[-1] - seqs[-2] > 1, "中间应当留有一个空洞"
        assert [r.id for r in store.list_audit(thread_id=tid)] == [
            r.id for r in before
        ] + [after.id]
    finally:
        store.close()


# ===========================================================================
# 4. READ COMMITTED 可见性
# ===========================================================================


def test_reader_cannot_see_uncommitted_rows(pg_app_dsn: str) -> None:
    """未提交的写入对 store(独立连接)不可见;提交后才可见。"""
    tid = uid("thread")
    record_id = uuid.uuid4().hex

    writer = psycopg.connect(pg_app_dsn, connect_timeout=10)
    store = PostgresAuditStore(pg_app_dsn)
    try:
        writer.autocommit = False
        with writer.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_logs (id, ts, actor, event, thread_id, detail_json)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (record_id, TS.isoformat(), "system", "plan.created", tid, "{}"),
            )

        assert store.list_audit(thread_id=tid) == [], (
            "READ COMMITTED 下未提交的行不得可见"
        )

        writer.commit()

        visible = store.list_audit(thread_id=tid)
        assert [r.id for r in visible] == [record_id]
    finally:
        writer.close()
        store.close()


def test_reader_sees_newly_committed_rows_on_each_read(pg_app_dsn: str) -> None:
    """READ COMMITTED:每次语句取新快照 ⇒ 后续提交对**下一次读**可见。"""
    tid = uid("thread")
    store = PostgresAuditStore(pg_app_dsn)
    writer = psycopg.connect(pg_app_dsn, connect_timeout=10)
    try:
        writer.autocommit = False
        for expected_count in range(1, 4):
            with writer.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_logs"
                    " (id, ts, actor, event, thread_id, detail_json)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    (uuid.uuid4().hex, TS.isoformat(), "system",
                     "plan.created", tid, "{}"),
                )
            writer.commit()
            assert len(store.list_audit(thread_id=tid)) == expected_count
    finally:
        writer.close()
        store.close()


# ===========================================================================
# 5. 确定性排序 + 不得过度声称
# ===========================================================================


def test_ordering_is_deterministic_for_committed_visible_rows(
    pg_app_dsn: str,
) -> None:
    """混合并发写入后,已提交可见行集合上的排序仍然确定。"""
    store = PostgresAuditStore(pg_app_dsn, min_size=2, max_size=6)
    tid = uid("thread")
    barrier = threading.Barrier(4)

    def _worker(index: int) -> None:
        barrier.wait(timeout=30)
        for offset in range(3):
            store.append_audit(
                build_audit_record(
                    "plan.created",
                    thread_id=tid,
                    ts=TS + timedelta(seconds=index * 10 + offset),
                )
            )

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in [pool.submit(_worker, i) for i in range(4)]:
                f.result(timeout=120)

        reads = [
            [r.id for r in store.list_audit(thread_id=tid)] for _ in range(5)
        ]
        assert all(read == reads[0] for read in reads), "反复读必须一致"
        assert len(reads[0]) == 12

        # `ts` 严格递增,因此顺序由 ts 决定,与写入交错无关
        stamps = [r.ts for r in store.list_audit(thread_id=tid)]
        assert stamps == sorted(stamps)
    finally:
        store.close()


def test_allocation_order_is_not_commit_order(pg_app_dsn: str) -> None:
    """**负向对照**:构造出"`seq` 顺序与提交顺序相反"的可观测场景。

    事务 A 先分配 `seq`(较小)、**后**提交;事务 B 后分配、**先**提交。
    提交顺序用**观察者连接的可见性**判定(不用本机时钟 —— Windows 上
    `time.monotonic()` 分辨率约 15.6ms,两次提交会落在同一 tick)。

    这证明 `seq` 只是**分配顺序**,把它当提交序或因果序使用在并发下会读错。
    """
    tid = uid("thread")
    id_a, id_b = uuid.uuid4().hex, uuid.uuid4().hex
    insert = (
        "INSERT INTO audit_logs (id, ts, actor, event, thread_id, detail_json)"
        " VALUES (%s, %s, %s, %s, %s, %s)"
    )
    params_a = (id_a, TS.isoformat(), "system", "plan.created", tid, "{}")
    params_b = (id_b, TS.isoformat(), "system", "plan.created", tid, "{}")

    conn_a = psycopg.connect(pg_app_dsn, connect_timeout=10)
    conn_b = psycopg.connect(pg_app_dsn, connect_timeout=10)
    observer = psycopg.connect(pg_app_dsn, autocommit=True, connect_timeout=10)

    def visible() -> set[str]:
        with observer.cursor() as cur:
            cur.execute(
                "SELECT id FROM audit_logs WHERE id IN (%s, %s)", (id_a, id_b)
            )
            return {row[0] for row in cur.fetchall()}

    try:
        conn_a.autocommit = False
        conn_b.autocommit = False

        with conn_a.cursor() as cur:
            cur.execute(insert, params_a)
        with conn_b.cursor() as cur:
            cur.execute(insert, params_b)

        assert visible() == set()

        conn_b.commit()
        assert visible() == {id_b}, "B 先提交 ⇒ 只看到 B"

        conn_a.commit()
        assert visible() == {id_a, id_b}
    finally:
        observer.close()
        conn_a.close()
        conn_b.close()

    seq_a = _seq_of(pg_app_dsn, id_a)
    seq_b = _seq_of(pg_app_dsn, id_b)
    assert seq_a is not None and seq_b is not None
    assert seq_a < seq_b, "分配顺序:A 先于 B"

    store = PostgresAuditStore(pg_app_dsn)
    try:
        by_seq = [r.id for r in store.list_audit(thread_id=tid)]
        assert by_seq == [id_a, id_b], "ORDER BY seq 给出的是分配顺序"
        assert by_seq != [id_b, id_a], "若相等说明本用例没构造出对照"
    finally:
        store.close()
