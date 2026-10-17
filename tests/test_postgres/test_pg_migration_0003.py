"""迁移 0003 的**永久回归测试**(Phase v0.4.0-I3 / TASK 2)。

背景:PHASE v0.4.0-I2 发现 0003 的预检(`GROUP BY ... HAVING COUNT(*) > 1`)
与它守护的**部分唯一索引**口径不一致 —— `GROUP BY` 把 NULL 当作相等,而唯一
索引把 NULL 当作互不相等。I2-FIX 做了最小修正(为每个身份列加
`<col> IS NOT NULL`),但当时只在**仓库外**的一次性探针里验证过。本文件把那
次验证**固化进仓库**,使这条语义从此有永久护栏。

为什么**必须**每个用例一个独立数据库(而不是复用共享的 `cybersec_test`):

- 0003 的三张表在库层是 append-only 的,运行时角色没有 DELETE / TRUNCATE
  权限,表 owner 也被触发器拦住 —— 一个被**故意写入重复决策行**的库**永远
  清不干净**;
- 更致命的是:脏库的预检会**永远**拒绝升级到 head。若这类用例跑在共享库上,
  它会污染整个 `tests/test_postgres/**` 套件。

因此本文件自建自删,绝不触碰共享库、绝不改迁移历史、绝不碰生产数据。
用例全部带 `postgres` 标记,默认离线套件不收集(见 pyproject 的
`addopts = -m 'not postgres'`)。
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
_MIGRATION_0003_PATH = (
    REPO_ROOT / "migrations" / "versions" / "0003_decision_uniqueness.py"
)

#: 0002 自身的 revision —— 预检的**前置状态**。刻意与 head 分成两个常量:
#: 0003 引入后 head 已不是 0002,把两者混用会写出恒真的断言。
REVISION_0002 = "0002_thread_ownership"
HEAD_REVISION = "0003_decision_uniqueness"

INDEX_DECIDED = "uq_audit_decided_per_interrupt"
INDEX_TIMEOUT = "uq_audit_timeout_per_thread"
_APPEND_ONLY_FUNCTION = "cybersec_reject_append_only_mutation"
_MIGRATOR_ROLE = "cybersec_migrator"

#: 决策身份(与 `app/security/store.py` 的单一真相源一致;本文件**只读**
#: 用它来构造数据,不复制为"另一个真相源")。
_DECIDED_EVENT = "approval.decided"
_TIMEOUT_EVENT = "approval.timeout"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_migration_0003_module() -> Any:
    """按文件路径载入 0003(文件名以数字开头,不是合法标识符)。"""
    spec = importlib.util.spec_from_file_location(
        "i3_decision_uniqueness", _MIGRATION_0003_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_db(dsn: str, dbname: str) -> str:
    """把连接串的库名换成 `dbname`(其余部分原样保留)。

    **必须产出 URI 形式**:Alembic 走 SQLAlchemy,不认 libpq 的
    `host=... port=...` conninfo 串 —— 用 `psycopg.conninfo.make_conninfo`
    会得到 `ArgumentError: Could not parse SQLAlchemy URL`(实测)。因此这里
    做 URI 尾部替换,而不是重新序列化连接参数。
    """
    base = dsn.split("?", 1)[0]
    return base.rsplit("/", 1)[0] + "/" + dbname


def _connect(dsn: str):
    """autocommit 连接 —— 一条语句失败不会毒化后续语句。"""
    return psycopg.connect(dsn, connect_timeout=10, autocommit=True)


def _insert_audit(
    dsn: str,
    *,
    event: str,
    thread_id: str | None = None,
    interrupt_id: str | None = None,
) -> None:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO audit_logs (id, ts, actor, event, incident_id, thread_id,"
            " interrupt_id, outcome, reason, plan_digest, detail_json)"
            " VALUES (%s, %s, 'regression', %s, NULL, %s, %s, NULL, NULL, NULL, '{}')",
            (uuid.uuid4().hex, "2026-10-13T00:00:00+00:00", event, thread_id,
             interrupt_id),
        )


def _index_names(dsn: str) -> list[str]:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
            " AND indexname IN (%s, %s) ORDER BY indexname",
            (INDEX_DECIDED, INDEX_TIMEOUT),
        )
        return [row[0] for row in cur.fetchall()]


def _alembic_head(dsn: str) -> str | None:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT version_num FROM alembic_version")
        rows = cur.fetchall()
        return rows[0][0] if rows else None


def _audit_row_count(dsn: str) -> int:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM audit_logs")
        return cur.fetchone()[0]


def _count_rows(dsn: str, event: str, thread_id: str, interrupt_id: str | None) -> int:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM audit_logs WHERE event = %s AND thread_id = %s"
            " AND interrupt_id IS NOT DISTINCT FROM %s",
            (event, thread_id, interrupt_id),
        )
        return cur.fetchone()[0]


def _probe_groups(dsn: str, sql: str) -> int:
    """预检 SQL 报出的"重复身份组数"。"""
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return len(cur.fetchall())


def _public_tables(dsn: str) -> set[str]:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {row[0] for row in cur.fetchall()}


def _trigger_names(dsn: str) -> set[str]:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")
        return {row[0] for row in cur.fetchall()}


def _append_only_function_exists(dsn: str) -> bool:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_proc WHERE proname = %s", (_APPEND_ONLY_FUNCTION,))
        return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# fixture:一次性数据库
# ---------------------------------------------------------------------------


@pytest.fixture
def make_fresh_db(pg_connection, pg_migrator_dsn: str, pg_app_dsn: str):
    """**工厂**夹具:每次调用建一个全新数据库(owner = migration-owner)。

    刻意做成工厂而不是单个夹具:一个用例常常需要**多个**互不干扰的库
    (例如"脏库上预检拦下"与"干净库上索引拒绝"必须分开观察)。若用普通夹具,
    用例内重复写 `fresh_db` 只会拿到**同一个**已注入的值,两个"库"其实是
    一个 —— 那会让断言悄悄变成同义反复(实测踩过)。

    用例结束统一删除;`WITH (FORCE)` 保证即便有残留连接也能删干净。
    """
    created: list[str] = []

    def _make() -> tuple[str, str]:
        name = f"cybersec_i3_{uuid.uuid4().hex[:12]}"
        with pg_connection.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{name}" OWNER {_MIGRATOR_ROLE}')
        created.append(name)
        return _with_db(pg_migrator_dsn, name), _with_db(pg_app_dsn, name)

    try:
        yield _make
    finally:
        for name in created:
            with pg_connection.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


# ---------------------------------------------------------------------------
# 1. 合法历史 NULL 身份 —— 预检**不得**拒绝
# ---------------------------------------------------------------------------


def test_null_identity_history_upgrades_cleanly(make_fresh_db) -> None:
    """**核心回归**:NULL 身份的历史行是**合法**数据,迁移必须放行。

    这是 I2 发现的缺陷的直接反例:修复前,`GROUP BY` 会把多行 NULL 归成一组,
    预检据此拒绝升级 —— 而这个库其实是干净的(唯一索引根本不会判它们冲突)。
    """
    migrator_dsn, _ = make_fresh_db()
    run_migration_as(migrator_dsn, REVISION_0002)

    # 各类 NULL 身份:同一 thread 的 NULL interrupt、同一 interrupt 的 NULL
    # thread、双 NULL,以及 NULL thread 的超时。
    for _ in range(2):
        _insert_audit(migrator_dsn, event=_DECIDED_EVENT,
                      thread_id="t-null-iid", interrupt_id=None)
    for _ in range(2):
        _insert_audit(migrator_dsn, event=_DECIDED_EVENT,
                      thread_id=None, interrupt_id="i-null-tid")
    for _ in range(2):
        _insert_audit(migrator_dsn, event=_DECIDED_EVENT,
                      thread_id=None, interrupt_id=None)
    for _ in range(2):
        _insert_audit(migrator_dsn, event=_TIMEOUT_EVENT, thread_id=None)
    # 一条正常的非 NULL 身份,证明"放行 NULL"没有顺带放过真重复。
    _insert_audit(migrator_dsn, event=_DECIDED_EVENT,
                  thread_id="t-ok", interrupt_id="i-ok")

    before = _audit_row_count(migrator_dsn)
    assert before == 9, f"前置:应有 9 行,实际 {before}"

    # ---- 负对照:修复**前**的预检口径在同一份数据上会误报 ----
    # 这段 SQL 是 I2-FIX 之前 0003 的真实预检文本(少了 `IS NOT NULL` 谓词)。
    # 把它钉在这里,是为了让"这条测试有牙"成为**可执行的**事实:如果哪天有人
    # 把谓词删掉,本用例的上半部分(升级必须成功)会立刻变红,而这里会证明
    # 失败原因正是预检把 NULL 身份当成了重复。
    _PREFIX_PREFLIGHT = (
        "SELECT thread_id, interrupt_id, COUNT(*) AS n FROM audit_logs"
        " WHERE event = 'approval.decided'"
        " GROUP BY thread_id, interrupt_id HAVING COUNT(*) > 1"
    )
    assert _probe_groups(migrator_dsn, _PREFIX_PREFLIGHT) > 0, (
        "负对照失效:修复前的预检口径竟然没有误报 —— 本用例的升级断言将失去意义"
    )

    run_migration_as(migrator_dsn, HEAD_REVISION)

    assert _alembic_head(migrator_dsn) == HEAD_REVISION
    assert _index_names(migrator_dsn) == sorted([INDEX_DECIDED, INDEX_TIMEOUT]), (
        "NULL 身份的历史数据下,两条部分唯一索引都应当建立成功"
    )
    assert _audit_row_count(migrator_dsn) == before, "升级不得增删既有行"
    assert _count_rows(migrator_dsn, _DECIDED_EVENT, "t-null-iid", None) == 2
    assert _count_rows(migrator_dsn, _DECIDED_EVENT, "t-ok", "i-ok") == 1


# ---------------------------------------------------------------------------
# 2. 真重复的非 NULL 身份 —— 预检**必须**响亮失败
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event, thread_id, interrupt_id, expected_index",
    [
        (_DECIDED_EVENT, "t-dup", "i-dup", INDEX_DECIDED),
        (_TIMEOUT_EVENT, "t-dup", None, INDEX_TIMEOUT),
    ],
    ids=["decided", "timeout"],
)
def test_real_duplicate_identity_blocks_the_migration_loudly(
    make_fresh_db, event: str, thread_id: str, interrupt_id: str | None,
    expected_index: str,
) -> None:
    """真重复(非 NULL 身份完全相同)必须**响亮失败**,且不留半成品。

    断言的不只是"抛错",而是四条可行动的性质:点名索引、不建索引、head 停在
    0002、历史行一行不动。少任何一条,运维拿到的都只是一句无从下手的报错。
    """
    migrator_dsn, _ = make_fresh_db()
    run_migration_as(migrator_dsn, REVISION_0002)
    _insert_audit(migrator_dsn, event=event, thread_id=thread_id,
                  interrupt_id=interrupt_id)
    _insert_audit(migrator_dsn, event=event, thread_id=thread_id,
                  interrupt_id=interrupt_id)

    with pytest.raises(Exception) as excinfo:
        run_migration_as(migrator_dsn, HEAD_REVISION)

    assert expected_index in str(excinfo.value), (
        f"失败消息必须点名索引 {expected_index};实际:{str(excinfo.value)[:200]}"
    )
    assert _index_names(migrator_dsn) == [], "预检失败时不得留下任何索引"
    assert _alembic_head(migrator_dsn) == REVISION_0002, "head 必须停在 0002"
    assert _audit_row_count(migrator_dsn) == 2, "预检只读:两行重复行必须原封不动"
    assert _count_rows(migrator_dsn, event, thread_id, interrupt_id) == 2


# ---------------------------------------------------------------------------
# 3. 预检 / 索引 语义对齐
# ---------------------------------------------------------------------------


def test_preflight_semantics_match_the_index_predicate(make_fresh_db) -> None:
    """预检扫描的行集必须**等于**索引覆盖的行集。

    两个方向都要证:
      - NULL 身份数据上,预检报 0 组,索引也接受重复 → 一致;
      - 真重复数据上,预检报 1 组,索引也拒绝 → 一致。
    """
    module = _load_migration_0003_module()
    probe_decided = module._duplicate_probe_sql(
        _DECIDED_EVENT, ("thread_id", "interrupt_id")
    )
    probe_timeout = module._duplicate_probe_sql(_TIMEOUT_EVENT, ("thread_id",))

    # ---- 方向一:NULL 身份 → 预检 0 组,索引接受 ----
    migrator_dsn, _ = make_fresh_db()
    run_migration_as(migrator_dsn, REVISION_0002)
    for _ in range(3):
        _insert_audit(migrator_dsn, event=_DECIDED_EVENT,
                      thread_id="t", interrupt_id=None)
    for _ in range(3):
        _insert_audit(migrator_dsn, event=_TIMEOUT_EVENT, thread_id=None)
    assert _probe_groups(migrator_dsn, probe_decided) == 0
    assert _probe_groups(migrator_dsn, probe_timeout) == 0

    run_migration_as(migrator_dsn, HEAD_REVISION)
    # 索引也接受:再写一行同样 NULL 身份的行不得被拒。
    _insert_audit(migrator_dsn, event=_DECIDED_EVENT, thread_id="t", interrupt_id=None)
    _insert_audit(migrator_dsn, event=_TIMEOUT_EVENT, thread_id=None)
    assert _count_rows(migrator_dsn, _DECIDED_EVENT, "t", None) == 4

    # ---- 方向二:真重复 → 预检报出来,索引也拒绝 ----
    migrator_dsn2, _ = make_fresh_db()
    run_migration_as(migrator_dsn2, REVISION_0002)
    _insert_audit(migrator_dsn2, event=_DECIDED_EVENT,
                  thread_id="td", interrupt_id="id")
    _insert_audit(migrator_dsn2, event=_DECIDED_EVENT,
                  thread_id="td", interrupt_id="id")
    assert _probe_groups(migrator_dsn2, probe_decided) == 1, (
        "真重复必须被预检报出来"
    )

    # 索引侧的对照:在一个**已到 head** 的干净库上直接写重复 → 索引拒绝。
    # (在 0002 的脏库上跑 upgrade 会先被预检拦下,抛的是 RuntimeError 而不是
    #  UniqueViolation —— 所以必须换一个库来单独观察索引的行为。)
    migrator_dsn3, _ = make_fresh_db()
    run_migration_as(migrator_dsn3, HEAD_REVISION)
    _insert_audit(migrator_dsn3, event=_DECIDED_EVENT,
                  thread_id="td", interrupt_id="id")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_audit(migrator_dsn3, event=_DECIDED_EVENT,
                      thread_id="td", interrupt_id="id")


# ---------------------------------------------------------------------------
# 4. 既有行保留
# ---------------------------------------------------------------------------


def test_existing_rows_are_preserved_across_the_upgrade(make_fresh_db) -> None:
    """升级 0002 → 0003 只建索引,**不动任何既有行**(含非决策事件)。"""
    migrator_dsn, _ = make_fresh_db()
    run_migration_as(migrator_dsn, REVISION_0002)
    for _ in range(20):
        _insert_audit(migrator_dsn, event="plan.created", thread_id="t-p")
    for _ in range(5):
        _insert_audit(migrator_dsn, event="policy.evaluated", thread_id="t-p")
    _insert_audit(migrator_dsn, event=_DECIDED_EVENT, thread_id="t-a", interrupt_id="i-a")
    before = _audit_row_count(migrator_dsn)
    assert before == 26

    run_migration_as(migrator_dsn, HEAD_REVISION)

    assert _audit_row_count(migrator_dsn) == before, "既有行必须一行不动"
    # 非决策事件在索引建立之后**仍然可以重复写入**(部分索引的 WHERE 子句
    # 把它排除在外) —— 这是"没有过度约束"的直接证据。
    _insert_audit(migrator_dsn, event="plan.created", thread_id="t-p")
    assert _audit_row_count(migrator_dsn) == before + 1


# ---------------------------------------------------------------------------
# 5. 可逆降级
# ---------------------------------------------------------------------------


def test_downgrade_is_reversible_and_drops_only_its_own_indexes(make_fresh_db) -> None:
    """`head → 0002 → head`:只删自己那两条索引,别人的对象一个不碰。"""
    migrator_dsn, _ = make_fresh_db()
    run_migration_as(migrator_dsn, REVISION_0002)
    tables_at_0002 = _public_tables(migrator_dsn)
    triggers_at_0002 = _trigger_names(migrator_dsn)

    run_migration_as(migrator_dsn, HEAD_REVISION)
    assert _index_names(migrator_dsn) == sorted([INDEX_DECIDED, INDEX_TIMEOUT])

    run_downgrade_as(migrator_dsn, REVISION_0002)
    assert _index_names(migrator_dsn) == [], "降级必须删掉自己的两条索引"
    assert _alembic_head(migrator_dsn) == REVISION_0002
    assert _public_tables(migrator_dsn) == tables_at_0002, (
        "0003 的 downgrade 不得删表 —— audit_logs 属于 0001、thread_owners 属于 0002"
    )
    assert _append_only_function_exists(migrator_dsn), (
        "0003 的 downgrade 不得删 0001 拥有的共享函数"
    )
    assert _trigger_names(migrator_dsn) == triggers_at_0002, (
        "0003 的 downgrade 不得删任何触发器"
    )

    run_migration_as(migrator_dsn, HEAD_REVISION)
    assert _index_names(migrator_dsn) == sorted([INDEX_DECIDED, INDEX_TIMEOUT]), (
        "降级之后必须还能升回来"
    )


# ---------------------------------------------------------------------------
# 6. 既有 append-only 触发器在 0003 上依然有效
# ---------------------------------------------------------------------------

#: 运行时角色必须做不了的破坏性操作(3 表 × 3 操作)。
_DESTRUCTIVE_OPS: tuple[str, ...] = tuple(
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


def test_append_only_protection_is_intact_at_0003(make_fresh_db) -> None:
    """两条独立防线在 0003 上都必须还在。

    第一道(权限层):运行时角色在语句级就被拒,根本到不了触发器。
    第二道(触发器层):即便以表 owner 身份,改写也被触发器拦下。
    只查 `pg_proc` 里函数名存在**不够** —— 必须证明保护真的开火。
    """
    migrator_dsn, app_dsn = make_fresh_db()
    run_migration_as(migrator_dsn, HEAD_REVISION)

    # ---- 第一道:运行时角色 ----
    with _connect(app_dsn) as conn:
        for statement in _DESTRUCTIVE_OPS:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with conn.cursor() as cur:
                    cur.execute(statement)

    # ---- 第二道:表 owner 也被触发器拦下 ----
    _insert_audit(migrator_dsn, event="plan.created", thread_id="t-trig")
    with pytest.raises(psycopg.errors.IntegrityError) as excinfo:
        with _connect(migrator_dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE audit_logs SET actor = actor WHERE thread_id = 't-trig'")
    assert "append-only" in str(excinfo.value), (
        "触发器必须在 0003 上依然开火(即便索引已建立)"
    )

    with pytest.raises(psycopg.errors.IntegrityError):
        with _connect(migrator_dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM audit_logs WHERE thread_id = 't-trig'")
