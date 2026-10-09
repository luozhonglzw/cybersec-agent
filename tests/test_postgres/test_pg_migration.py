"""迁移机制本身的测试(Phase v0.2.0-M1a;v0.3.0-A3-2 扩展,A3-2-FIX2 修订)。

分三类:

1. **只读的结构断言** —— 迁移脚本声明了唯一的基线 revision,且它的
   安全相关语句(守卫、append-only 触发器、所有权、最小授权)都在;
2. **守卫的精确测试** —— 直接执行守卫语句,断言运行时角色被拒绝。
   (不能只靠 `alembic upgrade` 去测:运行时角色连 `alembic_version` 都
   读不了,会在到达守卫之前就因权限失败 —— 那样测出来的是别的东西。)
3. **round-trip 与降级安全** —— `upgrade → downgrade base → upgrade`,
   以及 v0.3.0-A3-2 的 N-17(从 0001 前向升级不动既有行)、
   N-18(降级到 0001 只删 0002 自己的对象、**保留 0001 的共享函数**),
   和 A3-2-FIX2 新增的 N-19(**完整四步往返** `0001→0002→0001→0002`,
   并在 0001 检查点验证原表 / 原触发器 / 共享函数 / 运行时角色权限)。
   这些是**破坏性**用例(会 DROP 表),只允许跑在一次性测试库上;
   每个用例的 `finally` 强制恢复到 head,保证任何中途失败都不会让
   后续用例丢 schema。
"""
from __future__ import annotations

import importlib.util
import uuid
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
_MIGRATION_0002_PATH = (
    REPO_ROOT / "migrations" / "versions" / "0002_thread_ownership.py"
)

BASELINE_REVISION = "0001_baseline_audit_schema"
HEAD_REVISION = "0002_thread_ownership"
_EXPECTED_TABLES = {"incidents", "action_requests", "audit_logs"}
#: A3-2-FIX2 起归属是**单行**表(旧设计里的 `thread_approvers` 已移除)。
_OWNERSHIP_TABLES = {"thread_owners"}

#: 0001 创建并拥有的共享 append-only 函数。0002 **复用**它,**不**拥有它。
_APPEND_ONLY_FUNCTION = "cybersec_reject_append_only_mutation"


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


def _load_migration_0002_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "a32_thread_ownership", _MIGRATION_0002_PATH
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


def _append_only_function_exists(dsn: str) -> bool:
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_proc WHERE proname = %s",
                (_APPEND_ONLY_FUNCTION,),
            )
            return cur.fetchone() is not None


def _seed_incident(cur: Any) -> str:
    row_id = uuid.uuid4().hex
    cur.execute(
        "INSERT INTO incidents"
        " (id, created_at, indicator, risk_level, score, summary, plan_json)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (row_id, "2026-10-08T00:00:00+00:00", "203.0.113.9", "high", 80, "s", "{}"),
    )
    return row_id


#: 运行时角色在**三张原表**上必须做不了的破坏性操作(3 表 × 3 操作)。
#: `WHERE false` 刻意命中 0 行 —— 权限层应当在**语句级**就拒绝,
#: 根本到不了触发器(那正是"两条独立防线"里第一道的作用)。
_ORIGINAL_TABLE_OPS: tuple[str, ...] = tuple(
    statement
    for table, column in (
        ("incidents", "score"),
        ("action_requests", "score"),
        ("audit_logs", "actor"),
    )
    for statement in (
        f"UPDATE {table} SET {column} = {column} WHERE false",
        f"DELETE FROM {table} WHERE false",
        f"TRUNCATE TABLE {table}",
    )
)

#: 归属表上的破坏性操作(升级回 0002 之后必须同样被拒)。
_OWNERSHIP_TABLE_OPS: tuple[str, ...] = (
    "UPDATE thread_owners SET owner = owner WHERE false",
    "DELETE FROM thread_owners WHERE false",
    "TRUNCATE TABLE thread_owners",
)


def _runtime_role_errors(app_dsn: str, statements: tuple[str, ...]) -> dict[str, str]:
    """以运行时角色逐条尝试破坏性语句,返回 `{语句: 结果}`。

    用 **autocommit** 连接:一条语句失败会让非 autocommit 事务进入 aborted
    状态,后续语句全部报 "current transaction is aborted",掩盖真正被测的
    行为(与 conftest 里 `_connect` 的理由一致)。
    """
    results: dict[str, str] = {}
    conn = psycopg.connect(app_dsn, connect_timeout=10, autocommit=True)
    try:
        for statement in statements:
            try:
                with conn.cursor() as cur:
                    cur.execute(statement)
            except psycopg.errors.InsufficientPrivilege as exc:
                results[statement] = type(exc).__name__
            else:  # pragma: no cover - 安全护栏失守时才会走到
                results[statement] = "SUCCEEDED"
    finally:
        conn.close()
    return results


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


def test_migration_0002_declares_the_0001_baseline_as_its_predecessor() -> None:
    """0002 必须**前向**挂在 0001 上 —— 绝不重写 0001,也不分叉。"""
    module = _load_migration_0002_module()
    assert module.revision == HEAD_REVISION
    assert module.down_revision == BASELINE_REVISION


def test_migration_0002_downgrade_drops_exactly_one_table_and_no_function() -> None:
    """源码级护栏(结构性,不需要数据库):down-path 的**形状**必须正确。

    这是"照着 up-path 反向抄一遍"唯一会出错的地方 —— 所以除了端到端用例
    (N-18 / N-19),再加一条**不依赖数据库**的结构断言:任何人在 down-path 里
    加一句 `DROP FUNCTION`,这里立刻失败。
    """
    module = _load_migration_0002_module()
    down = module._STATEMENTS_DOWN
    assert len(down) == 1, "down-path 必须恰好一条语句"
    assert all(s.strip().upper().startswith("DROP TABLE IF EXISTS") for s in down)
    joined = " ".join(down).upper()
    assert "FUNCTION" not in joined
    assert "CYBERSEC_REJECT_APPEND_ONLY_MUTATION" not in joined
    assert "THREAD_OWNERS" in down[0].upper()
    # A3-2-FIX2:旧的逐审批人行表已从物理 schema 移除,down-path 不得再提它
    assert "THREAD_APPROVERS" not in joined


def test_migration_0002_reuses_the_shared_function_and_creates_no_new_one() -> None:
    """0002 **不**新建函数 —— 复用 0001 的共享函数,因此也不拥有它。"""
    module = _load_migration_0002_module()
    joined = "\n".join(module._STATEMENTS_UP).upper()
    assert "CREATE FUNCTION" not in joined
    assert "CREATE OR REPLACE FUNCTION" not in joined
    # 三条触发器都指向 0001 的函数(单表 × UPDATE/DELETE/TRUNCATE)
    assert joined.count("CYBERSEC_REJECT_APPEND_ONLY_MUTATION()") == 3


def test_migration_0002_creates_exactly_the_single_row_ownership_table() -> None:
    """0002 只建**一张**表,且形状是单行不可变审批集合。

    A3-2-FIX2 的设计修订在源码层也要钉住:任何重新引入可追加审批行表的
    改动都会让这条断言失败。
    """
    module = _load_migration_0002_module()
    joined = "\n".join(module._STATEMENTS_UP)
    assert joined.count("CREATE TABLE") == 1
    assert "CREATE TABLE thread_owners" in joined
    assert "thread_approvers" not in joined
    for column in ("thread_id", "owner", "approvers", "created_at"):
        assert column in joined
    assert "PRIMARY KEY" in joined
    # 三条 append-only 触发器
    for op in ("UPDATE", "DELETE", "TRUNCATE"):
        assert f"BEFORE {op} ON thread_owners" in joined


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


def test_alembic_version_table_records_the_head_revision(
    pg_migrator_connection,
) -> None:
    """`alembic_version` 记录的必须是 **head**,而不是基线。

    v0.3.0-A3-2 引入 0002 之后 head 从 `0001_...` 变成 `0002_...`。断言本身
    **没有变弱**(仍是"恰好一行、等于 head"),只是期望值跟着 head 走 ——
    把它写成常量而不是"取 max"是刻意的:head 移动必须是一次显式修改。
    """
    with pg_migrator_connection.cursor() as cur:
        cur.execute("SELECT version_num FROM alembic_version")
        rows = cur.fetchall()
    assert rows == [(HEAD_REVISION,)]


def test_migration_round_trips_and_leaves_the_schema_at_head(
    pg_migrator_dsn: str,
) -> None:
    """`upgrade → downgrade base → upgrade`。

    **破坏性**:`downgrade` 会 DROP 五张表并清空其内容。append-only 保护的是
    "运行期的改写",**不是**"故意的整体拆除" —— 所以本用例只允许跑在
    一次性本地测试库上。

    `finally` 强制恢复到 head:即使中途断言失败,后续用例也不会因为
    缺 schema 而连锁 ERROR。
    """
    try:
        assert _EXPECTED_TABLES <= _public_tables(pg_migrator_dsn), "前置:schema 应在 head"
        assert _OWNERSHIP_TABLES <= _public_tables(pg_migrator_dsn), "前置:0002 应已应用"

        run_downgrade_as(pg_migrator_dsn, "base")
        remaining = _public_tables(pg_migrator_dsn)
        assert not (_EXPECTED_TABLES & remaining), "downgrade 后三张表应当已删除"
        assert not (_OWNERSHIP_TABLES & remaining), "downgrade 后归属表应当已删除"

        run_migration_as(pg_migrator_dsn, "head")
        restored = _public_tables(pg_migrator_dsn)
        assert _EXPECTED_TABLES <= restored, "upgrade 后三张表应当回来"
        assert _OWNERSHIP_TABLES <= restored, "upgrade 后归属表应当回来"
    finally:
        run_migration_as(pg_migrator_dsn, "head")


# ---------------------------------------------------------------------------
# 4. v0.3.0-A3-2:N-17 前向升级 / N-18 降级安全 / N-19 完整四步往返
# ---------------------------------------------------------------------------


def test_upgrade_from_0001_preserves_preexisting_rows(
    pg_migrator_dsn: str, pg_migrator_connection
) -> None:
    """**N-17**:从 0001 前向升级到 0002,既有行必须**一行不动**。

    步骤:先降到 0001(此时库里是 0001 形态)→ 播一行"0001 时代"的审计数据
    → 升到 head → 断言那一行还在、且新的归属表已就位。

    这一条同时覆盖了"在持有 pre-0002 数据的库上升级"这个真实运维场景 ——
    0002 只加表加触发器,不碰任何既有列。
    """
    try:
        run_downgrade_as(pg_migrator_dsn, BASELINE_REVISION)
        assert not (_OWNERSHIP_TABLES & _public_tables(pg_migrator_dsn)), (
            "前置:降到 0001 后不应再有归属表"
        )

        with pg_migrator_connection.cursor() as cur:
            row_id = _seed_incident(cur)

        run_migration_as(pg_migrator_dsn, "head")
        assert _OWNERSHIP_TABLES <= _public_tables(pg_migrator_dsn), (
            "升级后归属表应当已建立"
        )

        with pg_migrator_connection.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM incidents WHERE id = %s", (row_id,))
            assert cur.fetchone()[0] == 1, "升级不得动既有行"
    finally:
        run_migration_as(pg_migrator_dsn, "head")


def test_downgrade_to_0001_keeps_the_shared_function_and_original_protection(
    pg_migrator_dsn: str, pg_migrator_connection
) -> None:
    """**N-18**:降级到 0001 只删 0002 自己的对象。

    这是整条迁移链上**唯一**"照着 up-path 反向抄一遍会出错"的地方:
    `cybersec_reject_append_only_mutation()` 由 0001 创建并拥有。如果 0002 的
    downgrade 也删它,那么"降级 0002、保留 0001"会留下三张原表的触发器引用
    一个不存在的函数 —— 一个坏掉的库。

    因此本用例断言三件事:
      1. 归属表确实被删掉;
      2. 三张原表还在,且**共享函数仍然存在**;
      3. 三张原表的 append-only 触发器**依然开火**(对表 owner 生效)。
    第 3 条是关键 —— 只查 `pg_proc` 里函数名存在还不够,必须证明保护真的还在。
    """
    try:
        assert _OWNERSHIP_TABLES <= _public_tables(pg_migrator_dsn), "前置:0002 应已应用"
        assert _append_only_function_exists(pg_migrator_dsn), "前置:共享函数应存在"

        run_downgrade_as(pg_migrator_dsn, BASELINE_REVISION)

        remaining = _public_tables(pg_migrator_dsn)
        assert not (_OWNERSHIP_TABLES & remaining), "归属表应当已删除"
        assert _EXPECTED_TABLES <= remaining, "三张原表**不得**被 0002 的 downgrade 删掉"
        assert _append_only_function_exists(pg_migrator_dsn), (
            "0002 的 downgrade 删掉了 0001 拥有的共享函数 —— 这是坏库"
        )

        with pg_migrator_connection.cursor() as cur:
            row_id = _seed_incident(cur)

        with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
            with pg_migrator_connection.cursor() as cur:
                cur.execute(
                    "UPDATE incidents SET score = score WHERE id = %s", (row_id,)
                )
        assert "append-only" in str(excinfo.value), (
            "降级到 0001 之后,三张原表的 append-only 触发器必须依然开火"
        )
    finally:
        run_migration_as(pg_migrator_dsn, "head")


def test_downgrade_then_upgrade_restores_the_ownership_tables(
    pg_migrator_dsn: str,
) -> None:
    """降级到 0001 再升回 head,归属表及其触发器必须原样回来。

    补上 round-trip 之外的这一格:上面的用例只证明"降级删对了东西",
    这里证明"降级之后还能升回来" —— 否则一次失败的降级会把库永久卡住。
    """
    try:
        run_downgrade_as(pg_migrator_dsn, BASELINE_REVISION)
        assert not (_OWNERSHIP_TABLES & _public_tables(pg_migrator_dsn))

        run_migration_as(pg_migrator_dsn, "head")
        restored = _public_tables(pg_migrator_dsn)
        assert _OWNERSHIP_TABLES <= restored
        assert _EXPECTED_TABLES <= restored
        assert _append_only_function_exists(pg_migrator_dsn)
    finally:
        run_migration_as(pg_migrator_dsn, "head")


def test_full_round_trip_0001_0002_0001_0002(
    pg_migrator_dsn: str, pg_app_dsn: str
) -> None:
    """**N-19**(A3-2-FIX2 新增):完整四步往返 `0001 → 0002 → 0001 → 0002`。

    单条边的用例(N-17 / N-18)证明不了"整条链能连续走通"。本用例从 **0001
    起点**出发走完四步,并在 **0001 检查点**上验证 Controller 点名的四件事:

      1. 三张原表仍然存在;
      2. 共享 append-only 函数仍然存在;
      3. 原表的 append-only 触发器**依然开火**(对表 owner 生效);
      4. 运行时角色**没有**获得 UPDATE / DELETE / TRUNCATE 权限(3 表 × 3 操作)。

    升级回 0002 之后,再验证归属表及其保护已就位(含运行时角色在归属表上
    同样做不了破坏性操作)。这是"降级不损坏、升级可恢复"的端到端证据。

    **破坏性**:会 DROP/CREATE 表。`finally` 强制恢复到 head。
    """
    try:
        # ---- 起点:回到 0001 ----
        run_downgrade_as(pg_migrator_dsn, BASELINE_REVISION)
        at_0001 = _public_tables(pg_migrator_dsn)
        assert not (_OWNERSHIP_TABLES & at_0001), "前置:0001 上不应有归属表"
        assert _EXPECTED_TABLES <= at_0001, "前置:0001 上三张原表应在"

        # ---- 0001 -> 0002 ----
        run_migration_as(pg_migrator_dsn, HEAD_REVISION)
        assert _OWNERSHIP_TABLES <= _public_tables(pg_migrator_dsn), (
            "0001 -> 0002 之后归属表应当已建立"
        )

        # ---- 0002 -> 0001(检查点)----
        run_downgrade_as(pg_migrator_dsn, BASELINE_REVISION)
        checkpoint = _public_tables(pg_migrator_dsn)
        assert not (_OWNERSHIP_TABLES & checkpoint), "0002 -> 0001 后归属表应当已删除"
        assert _EXPECTED_TABLES <= checkpoint, "检查点:三张原表必须仍在"
        assert _append_only_function_exists(pg_migrator_dsn), (
            "检查点:0002 的 downgrade 不得删掉 0001 拥有的共享函数"
        )

        # 3) 原表触发器仍开火(以表 owner 身份,命中真实行)
        conn = psycopg.connect(pg_migrator_dsn, connect_timeout=10, autocommit=True)
        try:
            with conn.cursor() as cur:
                row_id = _seed_incident(cur)
            with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE incidents SET score = score WHERE id = %s", (row_id,)
                    )
        finally:
            conn.close()
        assert "append-only" in str(excinfo.value), (
            "检查点:降级到 0001 之后原表的 append-only 触发器必须依然开火"
        )

        # 4) 运行时角色在 0001 检查点上没有任何破坏性权限
        original_errors = _runtime_role_errors(pg_app_dsn, _ORIGINAL_TABLE_OPS)
        assert set(original_errors.values()) == {"InsufficientPrivilege"}, (
            f"检查点:运行时角色获得了原表的破坏性权限: {original_errors}"
        )
        assert len(original_errors) == 9, "应当逐条探测 3 表 × 3 操作"

        # ---- 0001 -> 0002(再次)----
        run_migration_as(pg_migrator_dsn, HEAD_REVISION)
        restored = _public_tables(pg_migrator_dsn)
        assert _OWNERSHIP_TABLES <= restored, "再次升级后归属表应当回来"
        assert _EXPECTED_TABLES <= restored
        assert _append_only_function_exists(pg_migrator_dsn)

        # 归属表的保护也回来了
        ownership_errors = _runtime_role_errors(pg_app_dsn, _OWNERSHIP_TABLE_OPS)
        assert set(ownership_errors.values()) == {"InsufficientPrivilege"}, (
            f"再次升级后运行时角色获得了归属表的破坏性权限: {ownership_errors}"
        )
    finally:
        run_migration_as(pg_migrator_dsn, "head")
