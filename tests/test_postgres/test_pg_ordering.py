"""identity `seq` 的定序语义与它的**诚实边界**(Phase v0.2.0-M1a)。

## 为什么需要 `seq`

SQLite 侧的三条读路径依赖隐式 `rowid`:

- `app/security/store.py:364` —— `list_action_rows` 的 `ORDER BY rowid`
- `app/security/store.py:415` —— `pending_action_rows` 的 `ORDER BY rowid`
- `app/security/store.py:444` —— `list_audit` 的 `ORDER BY ts, rowid`

**PostgreSQL 没有 `rowid`**。因此 `audit_logs` 与 `action_requests` 各加一列
`BIGINT GENERATED ALWAYS AS IDENTITY` 的 `seq`,作为确定性的 tiebreaker。

## 本模块钉住的边界(不得过度声称)

`GENERATED ALWAYS AS IDENTITY` 的取值来自序列,而**序列的推进不参与事务回滚**。
由此有三条可测的推论:

1. **串行写入下**,`seq` 严格递增,等价于插入顺序;
2. **回滚会消耗序号**,留下空洞 —— 已回滚的行不存在,但它的序号不会给别人;
3. **并发写入下**,`seq` 是**分配顺序**,不是**提交顺序**:先分配的事务可能后提交。

第 3 条是本模块的**负向对照**:它主动构造出 "`ORDER BY seq` 的顺序与提交顺序
相反" 的可观测场景,用来证明我们**没有**声称 `seq` 给出全局提交序。
凡是把 `seq` 当提交序使用的读路径,都会在并发下读错 —— 这是已知的、
有界的、必须在 M1b 里显式处理的限制。
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.postgres

_INSERT_AUDIT_RETURNING_SEQ = (
    "INSERT INTO audit_logs (id, ts, actor, event, detail_json)"
    " VALUES (%s, %s, %s, %s, %s) RETURNING seq"
)

_INSERT_REQUEST_RETURNING_SEQ = (
    "INSERT INTO action_requests"
    " (id, incident_id, thread_id, indicator, risk_level, score, summary,"
    "  policy_reasons, action_type, priority, target, rationale,"
    "  requires_approval, reversible, requested_at)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    " RETURNING seq"
)


def _audit_params(
    *,
    row_id: str | None = None,
    ts: str = "2026-10-08T00:00:00+00:00",
    actor: str = "system",
):
    return (row_id or uuid.uuid4().hex, ts, actor, "plan.created", "{}")


def _request_params(
    *, row_id: str | None = None, thread_id: str = "thread-order"
):
    return (
        row_id or uuid.uuid4().hex,
        None,
        thread_id,
        "203.0.113.9",
        "high",
        80,
        "s",
        "[]",
        "block_ip",
        "high",
        "203.0.113.9",
        "r",
        1,
        0,
        "2026-10-08T00:00:00+00:00",
    )


def _manual_connection(dsn: str):
    """非 autocommit 连接 —— 需要显式控制 BEGIN / COMMIT / ROLLBACK。"""
    return psycopg.connect(dsn, connect_timeout=10)


# ---------------------------------------------------------------------------
# 1. 串行写入:seq 严格递增,等价于插入顺序
# ---------------------------------------------------------------------------


def test_serial_inserts_produce_strictly_increasing_seq(pg_app_connection) -> None:
    seqs: list[int] = []
    with pg_app_connection.cursor() as cur:
        for _ in range(5):
            cur.execute(_INSERT_AUDIT_RETURNING_SEQ, _audit_params())
            seqs.append(cur.fetchone()[0])

    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs), "seq 必须唯一(有 UNIQUE(seq) 约束)"
    assert all(b > a for a, b in zip(seqs, seqs[1:])), "串行写入下必须严格递增"


def test_serial_inserts_produce_strictly_increasing_seq_for_action_requests(
    pg_app_connection,
) -> None:
    seqs: list[int] = []
    with pg_app_connection.cursor() as cur:
        for _ in range(5):
            cur.execute(_INSERT_REQUEST_RETURNING_SEQ, _request_params())
            seqs.append(cur.fetchone()[0])

    assert all(b > a for a, b in zip(seqs, seqs[1:]))


def test_seq_is_the_tiebreaker_for_equal_timestamps(pg_app_connection) -> None:
    """同一 `ts` 的多行,`ORDER BY ts, seq` 给出确定的插入顺序。

    这正是 SQLite 侧 `list_audit` 的 `ORDER BY ts, rowid` 在 PG 上的等价物。
    """
    same_ts = "2026-10-08T12:00:00+00:00"
    ids: list[str] = []
    with pg_app_connection.cursor() as cur:
        for _ in range(4):
            params = _audit_params(ts=same_ts)
            ids.append(params[0])
            cur.execute(_INSERT_AUDIT_RETURNING_SEQ, params)

        cur.execute(
            "SELECT id FROM audit_logs WHERE ts = %s ORDER BY ts, seq",
            (same_ts,),
        )
        ordered = [row[0] for row in cur.fetchall()]

    assert ordered == ids, "同 ts 下必须按 seq(即插入顺序)确定排序"


# ---------------------------------------------------------------------------
# 2. seq 由数据库生成,调用方无法指定
# ---------------------------------------------------------------------------


def test_seq_cannot_be_set_explicitly(pg_app_connection) -> None:
    """`GENERATED ALWAYS` 拒绝显式赋值(SQLSTATE 428C9)。

    这保证调用方**无法**通过伪造 `seq` 来重排审计流 —— 与 SQLite 侧
    `rowid` 由库分配的语义一致。
    """
    with pytest.raises(psycopg.errors.Error) as excinfo:
        with pg_app_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_logs (id, seq, ts, actor, event, detail_json)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (uuid.uuid4().hex, 999_999_999, "2026-10-08T00:00:00+00:00",
                 "system", "plan.created", "{}"),
            )
    assert excinfo.value.sqlstate == "428C9"


# ---------------------------------------------------------------------------
# 3. 回滚消耗序号:留下空洞
# ---------------------------------------------------------------------------


def test_sequence_advances_even_when_the_transaction_rolls_back(pg_app_dsn) -> None:
    """已回滚的行不存在,但它的序号**不会**被回收 —— 空洞是设计的一部分。

    这条性质本身是**中性**的:它既保证"已提交行的相对顺序不受回滚影响",
    也意味着 `seq` 有空洞、不能当连续计数使用。
    """
    conn = _manual_connection(pg_app_dsn)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(_INSERT_AUDIT_RETURNING_SEQ, _audit_params())
            rolled_back_seq = cur.fetchone()[0]
        conn.rollback()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM audit_logs WHERE seq = %s", (rolled_back_seq,)
            )
            assert cur.fetchone()[0] == 0, "回滚后该行必须不存在"

        with conn.cursor() as cur:
            cur.execute(_INSERT_AUDIT_RETURNING_SEQ, _audit_params())
            committed_seq = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    assert committed_seq > rolled_back_seq, "序号不因回滚而回收"


# ---------------------------------------------------------------------------
# 4. 负向对照:分配顺序 != 提交顺序
# ---------------------------------------------------------------------------


def _visible_rows(observer, row_ids: tuple[str, str]) -> set[str]:
    """在一个**新事务**里读最新已提交状态。

    观察者连接是 autocommit 的,所以每次查询都是一个独立快照,能看到
    "此刻已经提交了什么" —— 这正是判定提交顺序的直接证据。
    """
    with observer.cursor() as cur:
        cur.execute(
            "SELECT id FROM audit_logs WHERE id IN (%s, %s)", row_ids
        )
        return {row[0] for row in cur.fetchall()}


def test_seq_is_allocation_order_not_commit_order(pg_app_dsn) -> None:
    """**负向对照**:构造出"seq 顺序与提交顺序相反"的可观测场景。

    两个并发事务:
      - 事务 A 先 INSERT(先拿到较小的 seq),但**后**提交;
      - 事务 B 后 INSERT(拿到较大的 seq),但**先**提交。

    提交顺序用**观察者连接的可见性**来判定,而不是用本机时钟:
    观察者是一个独立的 autocommit 连接,在 B 提交后、A 提交前查询,
    能看到 B 的行、看不到 A 的行 —— 这就是"B 先提交"的直接证据。
    (本机 `time.monotonic()` 在 Windows 上分辨率约 15.6ms,两次提交会落在
    同一 tick,拿它判序会得到 flaky 的假失败。)

    于是按 `seq` 排序得到的顺序,与按提交顺序得到的顺序**相反**。
    这正是"`seq` 是分配顺序,不是提交顺序"的实证,也是本模块存在的理由:
    它把这条限制钉成可执行的证据,防止后续实现把 `seq` 当成全局提交序来用。
    """
    conn_a = _manual_connection(pg_app_dsn)
    conn_b = _manual_connection(pg_app_dsn)
    observer = psycopg.connect(pg_app_dsn, connect_timeout=10, autocommit=True)
    try:
        conn_a.autocommit = False
        conn_b.autocommit = False

        id_a = uuid.uuid4().hex
        id_b = uuid.uuid4().hex

        # A 先分配序号(未提交)
        with conn_a.cursor() as cur:
            cur.execute(_INSERT_AUDIT_RETURNING_SEQ, _audit_params(row_id=id_a))
            seq_a = cur.fetchone()[0]

        # B 后分配序号(也未提交)
        with conn_b.cursor() as cur:
            cur.execute(_INSERT_AUDIT_RETURNING_SEQ, _audit_params(row_id=id_b))
            seq_b = cur.fetchone()[0]

        assert seq_a < seq_b, "分配顺序:A 先于 B"

        # 两个事务都未提交时,观察者谁也看不到
        assert _visible_rows(observer, (id_a, id_b)) == set()

        # B 先提交
        conn_b.commit()
        assert _visible_rows(observer, (id_a, id_b)) == {id_b}, (
            "B 提交后应当只看到 B —— 这就是 B 先提交的直接证据"
        )

        # A 后提交
        conn_a.commit()
        assert _visible_rows(observer, (id_a, id_b)) == {id_a, id_b}

        # 读回:ORDER BY seq 给出的顺序(A, B)与提交顺序(B, A)**相反**
        with observer.cursor() as cur:
            cur.execute(
                "SELECT id FROM audit_logs WHERE id IN (%s, %s) ORDER BY seq",
                (id_a, id_b),
            )
            by_seq = [row[0] for row in cur.fetchall()]

        assert by_seq == [id_a, id_b], "ORDER BY seq 给出的是分配顺序"
        assert by_seq != [id_b, id_a], "若相等则说明本用例没有构造出对照"
    finally:
        observer.close()
        conn_a.close()
        conn_b.close()
