"""角色隔离与 append-only 强制(Phase v0.2.0-M1a)。

分两层验证,因为它们是**两条独立的防线**:

1. **权限层**(application-runtime 角色):没有 UPDATE / DELETE / TRUNCATE,
   schema 上也没有 CREATE —— 这些语句会被直接拒绝,根本不触及触发器。
2. **触发器层**(migration-owner 角色,即表 owner):owner 天然拥有全部 DML
   权限,所以真正拦住它的是库层触发器。

任何一层单独存在都不够:只有权限层,一次误授权就失守;只有触发器层,
TRUNCATE 会绕过行触发器(所以另有语句级 TRUNCATE 触发器)。
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.postgres

APPEND_ONLY_MARKER = "is append-only"

#: 全部受 append-only 保护的表。
#:
#: v0.3.0-A3-2 从三张扩到四张 —— 归属表与审计三表受**同一套**保护,
#: 因此权限层与触发器层的用例一律按这个集合参数化,不留"新表没被测到"的缝。
#: (A3-2-FIX2 把归属收成单行,`thread_approvers` 已从物理 schema 移除。)
_ALL_TABLES = (
    "incidents",
    "action_requests",
    "audit_logs",
    "thread_owners",
)


def _insert_audit(cur, *, actor: str = "system") -> str:
    row_id = uuid.uuid4().hex
    cur.execute(
        "INSERT INTO audit_logs (id, ts, actor, event, detail_json)"
        " VALUES (%s, %s, %s, %s, %s)",
        (row_id, "2026-10-08T00:00:00+00:00", actor, "plan.created", "{}"),
    )
    return row_id


def _insert_incident(cur) -> str:
    row_id = uuid.uuid4().hex
    cur.execute(
        "INSERT INTO incidents"
        " (id, created_at, indicator, risk_level, score, summary, plan_json)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            row_id,
            "2026-10-08T00:00:00+00:00",
            "203.0.113.9",
            "high",
            80,
            "s",
            "{}",
        ),
    )
    return row_id


def _insert_action_request(cur) -> str:
    row_id = uuid.uuid4().hex
    cur.execute(
        "INSERT INTO action_requests"
        " (id, incident_id, thread_id, indicator, risk_level, score, summary,"
        "  policy_reasons, action_type, priority, target, rationale,"
        "  requires_approval, reversible, requested_at)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            row_id,
            None,
            "thread-x",
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
        ),
    )
    return row_id


def _insert_thread_owner(cur) -> str:
    """插一行 thread_owners(v0.3.0-A3-2;A3-2-FIX2 起含 approvers 列)。

    返回 thread_id。审批集合是**同一行**里的规范序 JSON 数组 —— 没有独立的
    审批行表,所以这里不需要(也不可能)配套插第二张表。
    """
    thread_id = f"th-{uuid.uuid4().hex}"
    cur.execute(
        "INSERT INTO thread_owners (thread_id, owner, approvers, created_at)"
        " VALUES (%s, %s, %s, %s)",
        (thread_id, "alice", '["bob"]', "2026-10-08T00:00:00+00:00"),
    )
    return thread_id


#: 每张表的"插入一行"辅助 —— 触发器负向用例必须先有真实的行。
_INSERT_ONE = {
    "incidents": _insert_incident,
    "action_requests": _insert_action_request,
    "audit_logs": _insert_audit,
    "thread_owners": _insert_thread_owner,
}

#: 每张表"按主键定位一行"用的列名。
#:
#: v0.3.0-A3-2 新增的归属表**没有 `id` 列**(主键是 `thread_id`),所以行触发器
#: 用例不能再硬写 `id`。
_ROW_KEY = {
    "incidents": "id",
    "action_requests": "id",
    "audit_logs": "id",
    "thread_owners": "thread_id",
}


# ---------------------------------------------------------------------------
# 所有权
# ---------------------------------------------------------------------------


def test_tables_are_owned_by_migrator_not_by_the_runtime_role(
    pg_migrator_connection,
) -> None:
    """表 owner 必须是 migration-owner —— 运行时角色不得拥有任何表。"""
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT tablename, tableowner FROM pg_tables"
            " WHERE schemaname = 'public' ORDER BY tablename"
        )
        owners = dict(cur.fetchall())

    for table in (*_ALL_TABLES, "alembic_version"):
        assert owners[table] == "cybersec_migrator", f"{table} owner 不是迁移角色"


def test_runtime_role_is_not_a_superuser_and_has_no_ddl_attributes(
    pg_migrator_connection,
) -> None:
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles"
            " WHERE rolname = 'cybersec_app'"
        )
        rolsuper, rolcreatedb, rolcreaterole = cur.fetchone()
    assert (rolsuper, rolcreatedb, rolcreaterole) == (False, False, False)


def test_migrator_role_is_not_a_superuser(pg_migrator_connection) -> None:
    """迁移角色也不是超级用户 —— 它只是对象的所有者。"""
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles"
            " WHERE rolname = 'cybersec_migrator'"
        )
        rolsuper, rolcreatedb, rolcreaterole = cur.fetchone()
    assert (rolsuper, rolcreatedb, rolcreaterole) == (False, False, False)


# ---------------------------------------------------------------------------
# 第一层:运行时角色的权限
# ---------------------------------------------------------------------------


def test_runtime_role_can_insert_and_select(pg_app_connection) -> None:
    with pg_app_connection.cursor() as cur:
        row_id = _insert_audit(cur)
        cur.execute("SELECT id, actor FROM audit_logs WHERE id = %s", (row_id,))
        assert cur.fetchone() == (row_id, "system")


def test_runtime_role_can_insert_action_request_rows(pg_app_connection) -> None:
    """`int(...)` 写布尔列这条既有写路径必须在 PG 上原样可用。"""
    row_id = uuid.uuid4().hex
    with pg_app_connection.cursor() as cur:
        cur.execute(
            "INSERT INTO action_requests"
            " (id, incident_id, thread_id, indicator, risk_level, score, summary,"
            "  policy_reasons, action_type, priority, target, rationale,"
            "  requires_approval, reversible, requested_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (row_id, None, "thread-x", "203.0.113.9", "high", 80, "s", "[]",
             "block_ip", "high", "203.0.113.9", "r", 1, 0,
             "2026-10-08T00:00:00+00:00"),
        )
        cur.execute(
            "SELECT requires_approval, reversible, incident_id, seq"
            " FROM action_requests WHERE id = %s",
            (row_id,),
        )
        requires_approval, reversible, incident_id, seq = cur.fetchone()

    # 读回来仍是整数 0/1,`bool(...)` 强转与 SQLite 侧行为一致
    assert bool(requires_approval) is True
    assert bool(reversible) is False
    assert incident_id is None  # 可空,且不依赖任何外键
    assert isinstance(seq, int)


def test_runtime_role_cannot_update(pg_app_connection) -> None:
    with pg_app_connection.cursor() as cur:
        row_id = _insert_audit(cur)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute(
                "UPDATE audit_logs SET actor = 'attacker' WHERE id = %s", (row_id,)
            )


def test_runtime_role_cannot_delete(pg_app_connection) -> None:
    with pg_app_connection.cursor() as cur:
        row_id = _insert_audit(cur)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute("DELETE FROM audit_logs WHERE id = %s", (row_id,))


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_runtime_role_cannot_truncate(pg_app_connection, table: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table}")


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_runtime_role_cannot_alter_schema(pg_app_connection, table: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN smuggled TEXT")


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_runtime_role_cannot_drop_tables(pg_app_connection, table: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute(f"DROP TABLE {table}")


def test_runtime_role_cannot_create_objects_in_public(pg_app_connection) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute("CREATE TABLE app_smuggled (id int)")


def test_runtime_role_cannot_disable_the_append_only_trigger(pg_app_connection) -> None:
    """禁用触发器需要表所有权 —— 运行时角色没有,所以这条路也被堵死。"""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute(
                "ALTER TABLE audit_logs DISABLE TRIGGER audit_logs_no_update"
            )


def test_runtime_role_cannot_read_alembic_version(pg_app_connection) -> None:
    """连迁移元数据表都不给读 —— 最小权限。"""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with pg_app_connection.cursor() as cur:
            cur.execute("SELECT version_num FROM alembic_version")


# ---------------------------------------------------------------------------
# 第二层:触发器(对表 owner 生效)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_migrator_update_is_blocked_by_trigger(pg_migrator_connection, table: str) -> None:
    """owner 有 DML 权限,所以拦住它的是触发器,而不是权限。

    **必须命中真实的行**:`FOR EACH ROW` 触发器只在有行被处理时触发,
    `UPDATE ... WHERE false` 命中 0 行 ⇒ 触发器根本不跑 ⇒ 写成那样就是
    恒真的假绿。所以这里先插入一行,再按主键 UPDATE 它。
    """
    with pg_migrator_connection.cursor() as cur:
        row_id = _INSERT_ONE[table](cur)

    with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
        with pg_migrator_connection.cursor() as cur:
            key = _ROW_KEY[table]
            cur.execute(
                f"UPDATE {table} SET {key} = {key} WHERE {key} = %s", (row_id,)
            )
    assert APPEND_ONLY_MARKER in str(excinfo.value)
    assert "UPDATE rejected" in str(excinfo.value)


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_migrator_delete_is_blocked_by_trigger(pg_migrator_connection, table: str) -> None:
    """同 UPDATE:必须命中真实的行,否则触发器不触发。"""
    with pg_migrator_connection.cursor() as cur:
        row_id = _INSERT_ONE[table](cur)

    with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
        with pg_migrator_connection.cursor() as cur:
            key = _ROW_KEY[table]
            cur.execute(f"DELETE FROM {table} WHERE {key} = %s", (row_id,))
    assert APPEND_ONLY_MARKER in str(excinfo.value)
    assert "DELETE rejected" in str(excinfo.value)


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_row_trigger_does_not_fire_when_no_row_is_targeted(
    pg_migrator_connection, table: str
) -> None:
    """**负向对照**:命中 0 行时行触发器不触发 —— 这是 PG 语义,不是缺陷。

    存在的意义是**防止将来有人把上面两条用例改回 `WHERE false`**:
    那种写法会静默退化成恒真断言(测试仍绿,但什么都没验证)。本用例把
    "0 行 ⇒ 不触发"这一事实钉住,于是"必须插入真实行"就成了不可回退的约束。

    它同时说明 append-only **不能只靠行触发器**:
    - 整表清空走 `TRUNCATE`,行触发器不管,靠语句级 TRUNCATE 触发器;
    - 批量改写虽然会命中行,但真正的第一道防线是运行时角色没有 UPDATE/DELETE 权限。
    """
    key = _ROW_KEY[table]
    with pg_migrator_connection.cursor() as cur:
        cur.execute(f"UPDATE {table} SET {key} = {key} WHERE false")
        assert cur.rowcount == 0, "该 UPDATE 本应命中 0 行"
    with pg_migrator_connection.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE false")
        assert cur.rowcount == 0, "该 DELETE 本应命中 0 行"


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_migrator_truncate_is_blocked_by_trigger(pg_migrator_connection, table: str) -> None:
    """TRUNCATE 不触发行触发器 —— 靠语句级 TRUNCATE 触发器拦住。"""
    with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
        with pg_migrator_connection.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table}")
    assert APPEND_ONLY_MARKER in str(excinfo.value)
    assert "TRUNCATE rejected" in str(excinfo.value)


def test_trigger_message_matches_the_sqlite_wording(pg_migrator_connection) -> None:
    """PG 与 SQLite 的拒绝消息逐字一致:`<table> is append-only: <OP> rejected`。"""
    with pg_migrator_connection.cursor() as cur:
        row_id = _insert_audit(cur)

    with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
        with pg_migrator_connection.cursor() as cur:
            cur.execute("UPDATE audit_logs SET actor = actor WHERE id = %s", (row_id,))
    assert "audit_logs is append-only: UPDATE rejected" in str(excinfo.value)


def test_trigger_uses_integrity_violation_sqlstate(pg_migrator_connection) -> None:
    """错误码刻意选 23000,与 SQLite 侧 RAISE(ABORT) 的 IntegrityError 对齐。"""
    with pg_migrator_connection.cursor() as cur:
        row_id = _insert_incident(cur)

    with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
        with pg_migrator_connection.cursor() as cur:
            cur.execute("UPDATE incidents SET id = id WHERE id = %s", (row_id,))
    assert excinfo.value.sqlstate == "23000"


def test_all_twelve_triggers_exist(pg_migrator_connection) -> None:
    """触发器集合**双向相等**。

    v0.3.0-A3-2 从 9 条扩到 12 条:0001 的 3 表 × (UPDATE/DELETE/TRUNCATE)
    加上 0002 的 1 表 × (UPDATE/DELETE/TRUNCATE)。用 `==` 而不是 `>=` ——
    少一条会被抓,多出一条来路不明的触发器同样会被抓。
    """
    with pg_migrator_connection.cursor() as cur:
        cur.execute(
            "SELECT tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid"
            " WHERE NOT t.tgisinternal ORDER BY tgname"
        )
        names = {row[0] for row in cur.fetchall()}

    expected = {
        f"{table}_no_{op}"
        for table in _ALL_TABLES
        for op in ("update", "delete", "truncate")
    }
    assert len(expected) == 12
    assert names == expected


def test_the_trigger_is_the_component_that_blocks_the_owner(pg_migrator_dsn) -> None:
    """**护栏有牙**:证明"拦住表 owner 的确实是触发器",而不是别的东西。

    表 owner 天然拥有全部 DML 权限,所以"UPDATE 被拒"这个现象本身并不足以
    定位原因 —— 也可能是某条 CHECK、某个权限设置。本用例把变量隔离出来:

      1. 触发器启用 ⇒ UPDATE 被拒(基线);
      2. **事务内**临时 `DISABLE TRIGGER` ⇒ 同一 UPDATE 成功 ——
         说明拦截者就是触发器;
      3. `ROLLBACK` ⇒ 触发器恢复启用、拦截重现。

    `ALTER TABLE ... DISABLE TRIGGER` 在 PostgreSQL 里是**事务性**的,
    所以 ROLLBACK 会把它恢复;即使中途抛错,连接关闭也会触发回滚。
    本用例**不留下任何持久改动**,不构成对安全控制的绕过。
    """
    conn = psycopg.connect(pg_migrator_dsn, connect_timeout=10)
    try:
        conn.autocommit = False
        row_id = uuid.uuid4().hex
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_logs (id, ts, actor, event, detail_json)"
                " VALUES (%s, %s, %s, %s, %s)",
                (row_id, "2026-10-08T00:00:00+00:00", "system", "plan.created", "{}"),
            )
        conn.commit()

        # 1. 基线:触发器启用 ⇒ 被拒
        with pytest.raises(psycopg.errors.IntegrityError):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE audit_logs SET actor = 'x' WHERE id = %s", (row_id,)
                )
        conn.rollback()

        # 2. 事务内禁用触发器 ⇒ 成功(证明拦截来源)
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE audit_logs DISABLE TRIGGER audit_logs_no_update"
            )
            cur.execute(
                "UPDATE audit_logs SET actor = 'x' WHERE id = %s", (row_id,)
            )
            assert cur.rowcount == 1, "触发器被禁用后该 UPDATE 本应成功"
        conn.rollback()

        # 3. 回滚后:触发器恢复启用,拦截重现
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tgenabled FROM pg_trigger"
                " WHERE tgname = 'audit_logs_no_update'"
            )
            assert cur.fetchone()[0] == "O", "ROLLBACK 后触发器应恢复为启用"
        with pytest.raises(psycopg.errors.IntegrityError):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE audit_logs SET actor = 'x' WHERE id = %s", (row_id,)
                )
        conn.rollback()
    finally:
        conn.close()
