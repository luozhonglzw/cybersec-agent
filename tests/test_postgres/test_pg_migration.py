"""迁移机制本身的测试(Phase v0.2.0-M1a)。

分三类:

1. **只读的结构断言** —— 迁移脚本声明了唯一的基线 revision,且它的
   安全相关语句(守卫、append-only 触发器、所有权、最小授权)都在;
2. **守卫的精确测试** —— 直接执行守卫语句,断言运行时角色被拒绝。
   (不能只靠 `alembic upgrade` 去测:运行时角色连 `alembic_version` 都
   读不了,会在到达守卫之前就因权限失败 —— 那样测出来的是别的东西。)
3. **round-trip** —— `upgrade → downgrade base → upgrade`。
   这是**破坏性**用例(会 DROP 三张表),只允许跑在一次性测试库上;
   `finally` 强制恢复到 head,保证任何中途失败都不会让后续用例丢 schema。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import psycopg
import pytest

from tests.test_postgres.conftest import run_downgrade_as, run_migration_as

pytestmark = pytest.mark.postgres

REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_PATH = (
    REPO_ROOT / "migrations" / "versions" / "0001_baseline_audit_schema.py"
)

BASELINE_REVISION = "0001_baseline_audit_schema"
_EXPECTED_TABLES = {"incidents", "action_requests", "audit_logs"}


def _load_migration_module() -> Any:
    """按文件路径载入迁移模块。

    迁移文件名以数字开头(`0001_...`),不是合法的 Python 标识符,
    因此不能用 `import` 语句导入。
    """
    spec = importlib.util.spec_from_file_location(
        "m1a_baseline_audit_schema", _MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_tables(dsn: str) -> set[str]:
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# 1. 只读结构断言
# ---------------------------------------------------------------------------


def test_migration_declares_a_single_baseline_revision() -> None:
    module = _load_migration_module()
    assert module.revision == BASELINE_REVISION
    assert module.down_revision is None, "基线迁移不得有前驱"


def test_migration_statements_are_a_tuple_of_single_statement_strings() -> None:
    """每条目 = 恰好一条语句 —— 这是错误定位能力的前提。"""
    module = _load_migration_module()
    statements = module._STATEMENTS_UP
    assert isinstance(statements, tuple)
    assert all(isinstance(s, str) and s.strip() for s in statements)


def test_guard_is_the_first_statement() -> None:
    """守卫必须**最先**执行,否则后面的 DDL 会先落地。"""
    module = _load_migration_module()
    first = module._STATEMENTS_UP[0]
    assert "current_user" in first
    assert "refusing to run migrations as the application runtime role" in first


def test_migration_creates_exactly_the_three_contract_tables() -> None:
    module = _load_migration_module()
    joined = "\n".join(module._STATEMENTS_UP)
    for table in _EXPECTED_TABLES:
        assert f"CREATE TABLE {table}" in joined


def test_migration_has_no_foreign_keys_and_no_composite_unique() -> None:
    """两条**刻意不做**的事,在源码层也要钉住,防止后人"顺手补上"。"""
    module = _load_migration_module()
    joined = "\n".join(module._STATEMENTS_UP).upper()
    assert "FOREIGN KEY" not in joined
    assert "REFERENCES" not in joined
    assert "UNIQUE (THREAD_ID, ACTION_TYPE, TARGET)" not in joined


def test_migration_hands_ownership_to_the_migrator_role() -> None:
    module = _load_migration_module()
    joined = "\n".join(module._STATEMENTS_UP)
    for table in _EXPECTED_TABLES:
        assert f"ALTER TABLE {table} OWNER TO cybersec_migrator" in joined


def test_migration_grants_only_select_and_insert_to_the_runtime_role() -> None:
    """最小授权的源码层断言:GRANT 里不得出现 UPDATE / DELETE / TRUNCATE。"""
    module = _load_migration_module()
    grants = [
        s for s in module._STATEMENTS_UP if s.strip().upper().startswith("GRANT")
    ]
    assert grants, "迁移里应当有 GRANT 语句"
    for statement in grants:
        upper = statement.upper()
        assert "UPDATE" not in upper
        assert "DELETE" not in upper
        assert "TRUNCATE" not in upper
        assert "CREATE" not in upper


def test_migration_creates_nine_append_only_triggers() -> None:
    module = _load_migration_module()
    joined = "\n".join(module._STATEMENTS_UP)
    for table in _EXPECTED_TABLES:
        for op in ("UPDATE", "DELETE", "TRUNCATE"):
            assert f"BEFORE {op} ON {table}" in joined, f"缺少 {table} 的 {op} 触发器"


def test_migration_declares_a_downgrade_path() -> None:
    module = _load_migration_module()
    assert callable(module.downgrade)
    assert module._STATEMENTS_DOWN, "缺少 downgrade 语句"


# ---------------------------------------------------------------------------
# 2. 守卫的精确测试(直接执行守卫语句)
# ---------------------------------------------------------------------------


def test_guard_statement_refuses_the_runtime_role(pg_app_dsn: str) -> None:
    """运行时角色执行守卫 ⇒ 明确拒绝。

    刻意**绕过 alembic** 直接执行守卫:通过 `alembic upgrade` 测不出来,
    因为运行时角色连 `alembic_version` 都没有 SELECT 权限,会在到达守卫
    之前就失败 —— 那样断言到的会是 InsufficientPrivilege,而不是守卫本身。
    """
    module = _load_migration_module()
    guard = module._STATEMENTS_UP[0]

    conn = psycopg.connect(pg_app_dsn, connect_timeout=10, autocommit=True)
    try:
        with pytest.raises(psycopg.errors.Error) as excinfo:
            with conn.cursor() as cur:
                cur.execute(guard)
    finally:
        conn.close()

    assert "refusing to run migrations as the application runtime role" in str(
        excinfo.value
    )


def test_guard_statement_passes_for_the_migration_owner(pg_migrator_connection) -> None:
    """同一守卫在 migration-owner 下必须放行(否则迁移根本跑不起来)。"""
    module = _load_migration_module()
    with pg_migrator_connection.cursor() as cur:
        cur.execute(module._STATEMENTS_UP[0])


def test_running_alembic_as_the_runtime_role_fails(pg_app_dsn: str) -> None:
    """纵深防御:即便守卫被绕过,`alembic upgrade` 在运行时角色下也跑不动。"""
    with pytest.raises(Exception):
        run_migration_as(pg_app_dsn, "head")


# ---------------------------------------------------------------------------
# 3. 迁移元数据与 round-trip
# ---------------------------------------------------------------------------


def test_alembic_version_table_records_the_baseline_revision(
    pg_migrator_connection,
) -> None:
    with pg_migrator_connection.cursor() as cur:
        cur.execute("SELECT version_num FROM alembic_version")
        rows = cur.fetchall()
    assert rows == [(BASELINE_REVISION,)]


def test_migration_round_trips_and_leaves_the_schema_at_head(
    pg_migrator_dsn: str,
) -> None:
    """`upgrade → downgrade base → upgrade`。

    **破坏性**:`downgrade` 会 DROP 三张表并清空其内容。append-only 保护的是
    "运行期的改写",**不是**"故意的整体拆除" —— 所以本用例只允许跑在
    一次性本地测试库上。

    `finally` 强制恢复到 head:即使中途断言失败,后续用例也不会因为
    缺 schema 而连锁 ERROR。
    """
    try:
        assert _EXPECTED_TABLES <= _public_tables(pg_migrator_dsn), "前置:schema 应在 head"

        run_downgrade_as(pg_migrator_dsn, "base")
        remaining = _public_tables(pg_migrator_dsn)
        assert not (_EXPECTED_TABLES & remaining), "downgrade 后三张表应当已删除"

        run_migration_as(pg_migrator_dsn, "head")
        restored = _public_tables(pg_migrator_dsn)
        assert _EXPECTED_TABLES <= restored, "upgrade 后三张表应当回来"
    finally:
        run_migration_as(pg_migrator_dsn, "head")
