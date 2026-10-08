"""PostgreSQL 基线 schema 的结构性断言(Phase v0.2.0-M1a)。

这里断言的是"PG schema 与既有 SQLite 契约**逐列兼容**",以及"CHECK 约束
既没有漏掉封闭词表、也没有越权拒绝契约允许的取值"。

期望的列顺序不是抄来的常量,而是**现场**用 `SqliteAuditStore` 建一个临时
SQLite 库、读 `PRAGMA table_info` 得到的 —— 任一侧改了列,这里就会失败。
"""
from __future__ import annotations

import sqlite3
import uuid

import pytest

from app.security.store import SqliteAuditStore

pytestmark = pytest.mark.postgres

_TABLES = ("incidents", "action_requests", "audit_logs")

#: 只有这两张表有 identity seq 列(SQLite 的隐式 rowid 的对应物)。
_SEQ_TABLES = ("action_requests", "audit_logs")

#: 时间列与 JSON 列在 PG 里必须仍是 TEXT。改成 timestamptz / jsonb 会让
#: `_from_iso()` / `json.loads(row[...])` 拿到非字符串而崩 —— 那是静默的
#: 语义变更,不是"优化"。
_MUST_STAY_TEXT = {
    "incidents": ("created_at", "plan_json"),
    "action_requests": ("policy_reasons", "requested_at"),
    "audit_logs": ("ts", "detail_json"),
}


def _insert_audit(cur, **overrides) -> str:
    """插入一条合法审计行,返回 id。`overrides` 可覆盖任意列。"""
    row = {
        "id": overrides.pop("id", uuid.uuid4().hex),
        "ts": overrides.pop("ts", "2026-10-08T00:00:00+00:00"),
        "actor": overrides.pop("actor", "system"),
        "event": overrides.pop("event", "plan.created"),
        "detail_json": overrides.pop("detail_json", "{}"),
    }
    optional = ("incident_id", "thread_id", "interrupt_id", "outcome", "reason", "plan_digest")
    for key in optional:
        row.setdefault(key, overrides.pop(key, None))
    assert not overrides, f"unexpected overrides: {sorted(overrides)}"

    cur.execute(
        "INSERT INTO audit_logs"
        " (id, ts, actor, event, incident_id, thread_id, interrupt_id,"
        "  outcome, reason, plan_digest, detail_json)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            row["id"], row["ts"], row["actor"], row["event"],
            row["incident_id"], row["thread_id"], row["interrupt_id"],
            row["outcome"], row["reason"], row["plan_digest"],
            row["detail_json"],
        ),
    )
    return row["id"]


@pytest.fixture(scope="module")
def sqlite_column_order(tmp_path_factory) -> dict[str, list[str]]:
    """用真实 `SqliteAuditStore` 建临时库,读回 SQLite 侧的列顺序。"""
    db_path = tmp_path_factory.mktemp("sqlite-parity") / "audit.db"
    SqliteAuditStore(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return {
            table: [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            for table in _TABLES
        }
    finally:
        conn.close()


def _pg_columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_schema = 'public' AND table_name = %s"
            " ORDER BY ordinal_position",
            (table,),
        )
        return [row[0] for row in cur.fetchall()]


def _pg_column_type(conn, table: str, column: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT data_type FROM information_schema.columns"
            " WHERE table_schema = 'public' AND table_name = %s"
            " AND column_name = %s",
            (table, column),
        )
        (data_type,) = cur.fetchone()
        return data_type


# ---------------------------------------------------------------------------
# 表与列
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", _TABLES)
def test_pg_columns_are_sqlite_columns_plus_seq(
    table: str, pg_migrator_connection, sqlite_column_order
) -> None:
    """PG 列 = SQLite 列,且 `seq` 只插在 id 之后(仅两张表)。"""
    sqlite_columns = sqlite_column_order[table]
    expected = list(sqlite_columns)
    if table in _SEQ_TABLES:
        expected = [sqlite_columns[0], "seq", *sqlite_columns[1:]]

    assert _pg_columns(pg_migrator_connection, table) == expected


def test_incidents_has_no_seq_column(pg_migrator_connection, sqlite_column_order) -> None:
    """`incidents` 只按主键读,没有排序需求,因此刻意不加 seq。"""
    assert "seq" not in _pg_columns(pg_migrator_connection, "incidents")
    assert "seq" not in sqlite_column_order["incidents"]


@pytest.mark.parametrize("table", _SEQ_TABLES)
def test_seq_is_a_generated_always_identity(table: str, pg_migrator_connection) -> None:
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT is_identity, identity_generation, is_nullable"
            " FROM information_schema.columns"
            " WHERE table_schema='public' AND table_name=%s AND column_name='seq'",
            (table,),
        )
        is_identity, generation, nullable = cur.fetchone()
    assert is_identity == "YES"
    assert generation == "ALWAYS"
    assert nullable == "NO"


@pytest.mark.parametrize("table,columns", sorted(_MUST_STAY_TEXT.items()))
def test_timestamp_and_json_columns_stay_text(
    table: str, columns: tuple[str, ...], pg_migrator_connection
) -> None:
    """兼容性护栏:这些列一旦变成 timestamptz / jsonb,既有读路径就会崩。"""
    for column in columns:
        assert _pg_column_type(pg_migrator_connection, table, column) == "text", (
            f"{table}.{column} 必须保持 text(见 store.py 的 _from_iso / json.loads)"
        )


def test_boolean_like_columns_are_integer_not_boolean(pg_migrator_connection) -> None:
    """既有写路径写的是 `int(action.requires_approval)` —— 列类型必须能收下它。"""
    for column in ("requires_approval", "reversible"):
        assert (
            _pg_column_type(pg_migrator_connection, "action_requests", column)
            == "integer"
        )


# ---------------------------------------------------------------------------
# 刻意不引入的东西
# ---------------------------------------------------------------------------


def test_no_foreign_keys_are_defined(pg_migrator_connection) -> None:
    """M1a 明确保留可空 `incident_id` 且**不加**外键。"""
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT conrelid::regclass::text, conname FROM pg_constraint"
            " WHERE contype = 'f' AND connamespace = 'public'::regnamespace"
        )
        found = cur.fetchall()
    assert found == [], f"unexpected foreign keys: {found}"


def test_no_unique_thread_action_target_constraint(pg_migrator_connection) -> None:
    """`UNIQUE(thread_id, action_type, target)` 不是已证的幂等键,M1a 明确不加。"""
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes"
            " WHERE schemaname='public' AND tablename='action_requests'"
        )
        definitions = [row[0] for row in cur.fetchall()]

    for definition in definitions:
        if "UNIQUE" not in definition.upper():
            continue
        lowered = definition.lower()
        assert not (
            "thread_id" in lowered and "action_type" in lowered and "target" in lowered
        ), f"发现未获授权的幂等键约束: {definition}"


def test_incident_id_remains_nullable(pg_migrator_connection) -> None:
    for table in ("action_requests", "audit_logs"):
        with pg_migrator_connection.cursor() as cur:
            cur.execute(
                "SELECT is_nullable FROM information_schema.columns"
                " WHERE table_schema='public' AND table_name=%s AND column_name='incident_id'",
                (table,),
            )
            (nullable,) = cur.fetchone()
        assert nullable == "YES"


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------


def test_expected_indexes_exist(pg_migrator_connection) -> None:
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname='public'"
        )
        present = {row[0] for row in cur.fetchall()}

    # 前四条与 SQLite 一一对应;后两条服务既有读路径的排序与派生查询。
    expected = {
        "idx_audit_thread",
        "idx_audit_incident",
        "idx_req_thread",
        "idx_req_incident",
        "idx_audit_ts_seq",
        "idx_audit_thread_event",
    }
    assert expected <= present, f"缺少索引: {sorted(expected - present)}"


# ---------------------------------------------------------------------------
# CHECK 约束:该拒的拒,该放的放
# ---------------------------------------------------------------------------


def test_plan_digest_check_accepts_null_and_valid_hex(pg_app_connection) -> None:
    with pg_app_connection.cursor() as cur:
        _insert_audit(cur, plan_digest=None)
        _insert_audit(cur, plan_digest="a" * 64)
        _insert_audit(cur, plan_digest="0123456789abcdef" * 4)


@pytest.mark.parametrize(
    "bad_digest",
    [
        "A" * 64,          # 大写
        "a" * 63,          # 太短
        "a" * 65,          # 太长
        "g" * 64,          # 非十六进制字符
    ],
)
def test_plan_digest_check_rejects_bad_format(pg_app_connection, bad_digest: str) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_app_connection.cursor() as cur:
            _insert_audit(cur, plan_digest=bad_digest)


def test_plan_digest_check_matches_pydantic_pattern() -> None:
    """PG 的正则必须与 `app/schemas/audit.py` 的 `_SHA256_HEX` 语义一致。"""
    from app.schemas import audit as audit_schema

    assert audit_schema._SHA256_HEX == r"^[0-9a-f]{64}$"


@pytest.mark.parametrize(
    "bad_event",
    ["plan.created ", "PLAN.CREATED", "plan.unknown", "", "approval"],
)
def test_audit_event_check_rejects_out_of_vocabulary(
    pg_app_connection, bad_event: str
) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_app_connection.cursor() as cur:
            _insert_audit(cur, event=bad_event)


def test_audit_event_check_accepts_the_closed_vocabulary(pg_app_connection) -> None:
    from app.schemas.audit import AuditEvent

    with pg_app_connection.cursor() as cur:
        for event in AuditEvent.__args__:  # type: ignore[attr-defined]
            _insert_audit(cur, event=event)


def test_outcome_is_deliberately_unconstrained(pg_app_connection) -> None:
    """契约里 `outcome` 是 `str | None`,不是 Literal —— 加 CHECK 会过度约束。"""
    with pg_app_connection.cursor() as cur:
        for outcome in ("allow", "deny", "approved", "denied", "something_else", ""):
            _insert_audit(cur, outcome=outcome)


def test_risk_level_check_rejects_out_of_vocabulary(pg_app_connection) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_app_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO incidents"
                " (id, created_at, indicator, risk_level, score, summary, plan_json)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (uuid.uuid4().hex, "2026-10-08T00:00:00+00:00", "203.0.113.9",
                 "catastrophic", 10, "s", "{}"),
            )


@pytest.mark.parametrize("score", [-1, 101])
def test_score_check_rejects_out_of_range(pg_app_connection, score: int) -> None:
    """pydantic 是 `ge=0, le=100`,PG 的 CHECK 必须与之对齐(不多不少)。"""
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_app_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO incidents"
                " (id, created_at, indicator, risk_level, score, summary, plan_json)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (uuid.uuid4().hex, "2026-10-08T00:00:00+00:00", "203.0.113.9",
                 "low", score, "s", "{}"),
            )


@pytest.mark.parametrize("score", [0, 100])
def test_score_check_accepts_boundaries(pg_app_connection, score: int) -> None:
    with pg_app_connection.cursor() as cur:
        cur.execute(
            "INSERT INTO incidents"
            " (id, created_at, indicator, risk_level, score, summary, plan_json)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (uuid.uuid4().hex, "2026-10-08T00:00:00+00:00", "203.0.113.9",
             "low", score, "s", "{}"),
        )
