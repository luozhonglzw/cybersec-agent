"""`SqliteAuditStore` 的线程归属存储测试(Phase v0.3.0-A3-2,A3-2-FIX2 修订)。

与 `tests/test_postgres/test_store_contract.py` 的关系:**不重复**。

那个文件跑的是**两个后端共享**的行为契约。本文件只放**SQLite 独有**或
**共享库不能做**的事:

1. **脏数据注入** —— PostgreSQL 侧是共享且 append-only 的库,注入一行脏数据
   **永久删不掉**;SQLite 每个用例都是 `tmp_path` 下的全新库,可以随便注入。
   因此"存储被破坏"的形态只能在这里验证。
2. **schema 自身的形状** —— 表/触发器集合、幂等重建、`_TABLES` 与
   `_OWNERSHIP_TABLES` 的分工、**不存在**可追加的审批行表。
3. **序列化边界** —— `approvers` 列的规范 JSON 形态,以及反序列化的严格性。

A3-2-FIX2 起归属是**单行**(`thread_id` 主键 + 属主 + 规范序审批集合 +
`created_at`)。因此"原子性"与"无孤儿"是**结构性**的,不再依赖事务边界;
本文件用"只发一条 INSERT"的白盒证据把这件事钉住。

所有用例一律用 `tmp_path`,不在仓库里创建 `audit.db`(用户硬性约束)。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.ownership import ThreadOwnership
from app.security.store import (
    _OWNERSHIP_TABLES,
    _TABLES,
    SqliteAuditStore,
)

TS = datetime(2026, 10, 8, 9, 0, 0, tzinfo=timezone.utc)
OWNER = "alice"
APPROVER_B = "bob"
APPROVER_C = "carol"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.db"


@pytest.fixture
def store(db_path: Path) -> SqliteAuditStore:
    return SqliteAuditStore(db_path)


def _ownership(
    thread_id: str = "th-1",
    *,
    owner: str = OWNER,
    approvers: tuple[str, ...] = (APPROVER_B,),
    created_at: datetime = TS,
) -> ThreadOwnership:
    return ThreadOwnership(
        thread_id=thread_id,
        owner=owner,
        approvers=approvers,
        created_at=created_at,
    )


def _raw(db_path: Path):
    """直连数据库(**绕过 store**),autocommit —— 用于注入脏数据。"""
    return sqlite3.connect(db_path, isolation_level=None)


def _table_names(db_path: Path) -> set[str]:
    conn = _raw(db_path)
    try:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()


def _trigger_names(db_path: Path) -> set[str]:
    conn = _raw(db_path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. schema 形状
# ---------------------------------------------------------------------------


def test_schema_creates_the_ownership_table(store: SqliteAuditStore, db_path: Path) -> None:
    assert set(_OWNERSHIP_TABLES) <= _table_names(db_path)


def test_there_is_no_independently_appendable_approver_table(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """**结构性的不可变成员集合** —— 审批集合没有自己的表。

    这是 A3-2-FIX2 的核心:只要存在一张可 INSERT 的 `thread_approvers`,
    "注册之后再补一个审批人"就总是可能的。表不存在 ⇒ 那条写入面不存在。
    """
    assert "thread_approvers" not in _table_names(db_path)
    # 负对照:归属表本身确实建出来了,所以上面的断言不是因为"什么都没建"
    assert "thread_owners" in _table_names(db_path)


def test_ownership_table_has_no_approver_lookup_index(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """刻意**不**建"按审批人反查"的索引。

    审批集合是单行里的一个 JSON 数组,普通索引服务不了该查询;而 v0.3.0
    没有任何调用方需要它。这里把"确实没有"钉住,免得有人以为它被漏掉了。
    """
    conn = _raw(db_path)
    try:
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
    finally:
        conn.close()
    assert "idx_approvers_subject" not in indexes


def test_ownership_tables_are_separate_from_the_audit_tables() -> None:
    """`_TABLES` 必须仍是**三张审计表** —— 新表不得混进去。

    混进去会让"审计表"这个既有概念的集合悄悄改变含义(文档、既有护栏测试
    都按它取值)。需要遍历全部表的地方显式写 `(*_TABLES, *_OWNERSHIP_TABLES)`。
    """
    assert _TABLES == ("incidents", "action_requests", "audit_logs")
    assert _OWNERSHIP_TABLES == ("thread_owners",)
    assert not set(_TABLES) & set(_OWNERSHIP_TABLES)


def test_ownership_table_has_append_only_triggers(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """触发器总数 6 → 8(1 张新表 × UPDATE/DELETE)。"""
    names = _trigger_names(db_path)
    assert len(names) == 8
    for table in _OWNERSHIP_TABLES:
        assert f"{table}_no_update" in names
        assert f"{table}_no_delete" in names


def test_schema_is_idempotent(db_path: Path) -> None:
    """重复构造(即重复执行 DDL)不得失败 —— 全部 `IF NOT EXISTS`。"""
    SqliteAuditStore(db_path)
    SqliteAuditStore(db_path)
    store = SqliteAuditStore(db_path)
    store.record_thread_ownership(_ownership())
    assert store.get_thread_ownership("th-1") == _ownership()


def test_no_repo_audit_db_created() -> None:
    """用户硬性约束:测试不得在仓库里创建 audit.db。"""
    assert not (_REPO_ROOT / "data" / "audit.db").exists()


# ---------------------------------------------------------------------------
# 2. 往返、规范序列化与拒绝
# ---------------------------------------------------------------------------


def test_round_trip(store: SqliteAuditStore) -> None:
    o = _ownership(approvers=(APPROVER_C, APPROVER_B))
    store.record_thread_ownership(o)
    got = store.get_thread_ownership("th-1")
    assert got == o
    assert got is not None and got.approvers == (APPROVER_B, APPROVER_C)


def test_approver_column_holds_a_canonical_json_array(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """`approvers` 列是**规范序**(字典序)JSON 数组的文本。

    "确定性规范序列化"是承重性质:同一指派必须只有一个字节表示。输入
    `(carol, bob)` 必须落成 `["bob","carol"]` —— 而不是原样保存输入顺序。
    """
    store.record_thread_ownership(_ownership(approvers=(APPROVER_C, APPROVER_B)))
    conn = _raw(db_path)
    try:
        raw = conn.execute(
            "SELECT approvers FROM thread_owners WHERE thread_id = 'th-1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert raw == json.dumps([APPROVER_B, APPROVER_C], ensure_ascii=False)
    assert json.loads(raw) == [APPROVER_B, APPROVER_C]


def test_serialization_keeps_non_ascii_subjects_verbatim(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """`ensure_ascii=False`:非 ASCII 主体不得被转义成 `\\uXXXX`。

    转义会让同一指派有两种文本表示,破坏"规范序列化"。
    """
    store.record_thread_ownership(_ownership(approvers=("用户乙",)))
    conn = _raw(db_path)
    try:
        raw = conn.execute(
            "SELECT approvers FROM thread_owners WHERE thread_id = 'th-1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert raw == '["用户乙"]'
    assert store.get_thread_ownership("th-1").approvers == ("用户乙",)


def test_unknown_thread_returns_none(store: SqliteAuditStore) -> None:
    assert store.get_thread_ownership("nope") is None


def test_duplicate_registration_is_rejected_and_leaves_the_original(
    store: SqliteAuditStore,
) -> None:
    store.record_thread_ownership(_ownership())
    with pytest.raises(sqlite3.IntegrityError):
        store.record_thread_ownership(_ownership(owner="mallory"))
    got = store.get_thread_ownership("th-1")
    assert got is not None and got.owner == OWNER


def test_single_row_carries_owner_and_created_at(
    store: SqliteAuditStore, db_path: Path
) -> None:
    store.record_thread_ownership(_ownership(approvers=(APPROVER_B, APPROVER_C)))
    conn = _raw(db_path)
    try:
        rows = conn.execute(
            "SELECT thread_id, owner, approvers, created_at FROM thread_owners"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, "归属必须恰好一行"
    thread_id, owner, approvers, created_at = rows[0]
    assert thread_id == "th-1"
    assert owner == OWNER
    assert json.loads(approvers) == [APPROVER_B, APPROVER_C]
    assert created_at == TS.isoformat()


# ---------------------------------------------------------------------------
# 3. 原子性:单条 INSERT(结构性,白盒证据)
# ---------------------------------------------------------------------------


class _RecordingConnection:
    """把真实连接包一层,记录执行过的 SQL。

    store 的写惯用法是 `with closing(self._connect()) as conn, conn:`
    —— 第二个 `conn` 是事务上下文管理器,所以这个代理必须实现
    `__enter__` / `__exit__`(委托给真实连接),否则 `with` 直接 TypeError。
    """

    def __init__(self, conn: sqlite3.Connection, statements: list[str]) -> None:
        self._conn = conn
        self._statements = statements

    def execute(self, sql: str, params=()):
        self._statements.append(sql)
        return self._conn.execute(sql, params)

    def __enter__(self) -> "_RecordingConnection":
        self._conn.__enter__()
        return self

    def __exit__(self, *exc_info) -> bool | None:
        return self._conn.__exit__(*exc_info)

    def close(self) -> None:
        self._conn.close()


def test_registration_issues_exactly_one_insert_statement(
    store: SqliteAuditStore, monkeypatch
) -> None:
    """**原子性的结构性证据**:整笔注册只发一条 INSERT。

    单条语句天然原子 —— 不存在"写了一半"的中间态,因此也不需要靠事务边界
    来保证。这条断言把"实现退化成多条语句"变成不可回退的约束。
    """
    statements: list[str] = []
    real_connect = store._connect

    def fake_connect() -> _RecordingConnection:
        return _RecordingConnection(real_connect(), statements)

    monkeypatch.setattr(store, "_connect", fake_connect)
    store.record_thread_ownership(_ownership(approvers=(APPROVER_B, APPROVER_C)))

    inserts = [s for s in statements if "INSERT INTO" in s.upper()]
    assert len(inserts) == 1, statements
    assert "INSERT INTO thread_owners" in inserts[0]
    assert len(statements) == 1, "归属注册不得附带任何其它语句"


def test_duplicate_registration_does_not_alter_the_stored_row(
    store: SqliteAuditStore,
) -> None:
    """被拒的重复注册必须让原行**逐字不动**(不是"先覆盖后报错")。"""
    original = _ownership(approvers=(APPROVER_B,))
    store.record_thread_ownership(original)
    with pytest.raises(sqlite3.IntegrityError):
        store.record_thread_ownership(
            _ownership(owner="mallory", approvers=(APPROVER_C,))
        )
    assert store.get_thread_ownership("th-1") == original


# ---------------------------------------------------------------------------
# 4. 脏数据在读取边界暴露(PostgreSQL 侧做不了 —— 注入的行删不掉)
# ---------------------------------------------------------------------------


def _inject(db_path: Path, thread_id: str, owner: str, approvers_raw: str,
            created_at: str) -> None:
    conn = _raw(db_path)
    try:
        conn.execute(
            "INSERT INTO thread_owners (thread_id, owner, approvers, created_at)"
            " VALUES (?, ?, ?, ?)",
            (thread_id, owner, approvers_raw, created_at),
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "raw",
    [
        "not json",          # 不是 JSON
        '{"a": 1}',          # JSON 但不是数组
        '"bob"',             # JSON 字符串(会被 tuple() 静默拆成字符)
        "[1, 2]",            # 数组但元素不是字符串
        "[null]",            # 数组但含 null
    ],
)
def test_malformed_approvers_json_is_rejected_on_read(
    store: SqliteAuditStore, db_path: Path, raw: str
) -> None:
    """**畸形序列化数据不得静默变成空/宽松指派**(Controller TASK 2)。

    这些是"形状"错误:不是 JSON / 不是数组 / 元素不是字符串。刻意显式拒绝
    `"bob"` —— `tuple("bob")` 会变成 `('b','o','b')`,一个"看起来像审批人、
    其实是被拆开的字符串"的静默误读。
    """
    _inject(db_path, "th-broken", OWNER, raw, TS.isoformat())
    with pytest.raises(ValueError):
        store.get_thread_ownership("th-broken")


@pytest.mark.parametrize("raw", ["[]", '["bob","bob"]', '["bob","bob","carol"]'])
def test_permissive_or_degenerate_approver_sets_are_rejected_on_read(
    store: SqliteAuditStore, db_path: Path, raw: str
) -> None:
    """空数组与重复项**必须响亮失败**,绝不能被读成"没人能审批"或"去重后合法"。

    空集合恰好是 fail-closed 终态 —— 静默接受会把"存储被破坏"伪装成
    "正常的拒绝";重复项被静默去重则是把一个畸形的存储值当成合法指派。
    """
    _inject(db_path, "th-broken", OWNER, raw, TS.isoformat())
    with pytest.raises(ValidationError):
        store.get_thread_ownership("th-broken")


def test_owner_appearing_as_an_approver_is_rejected_on_read(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """存储里若把属主写进审批集合(自审批),读取必须拒绝(D-7 第二道)。"""
    _inject(db_path, "th-self", OWNER, json.dumps([OWNER]), TS.isoformat())
    with pytest.raises(ValidationError):
        store.get_thread_ownership("th-self")


def test_non_canonical_owner_in_the_database_is_rejected_on_read(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """库里的属主带前后空白 ⇒ 读取时被模型拒绝(脏数据不静默通过)。"""
    _inject(db_path, "th-padded", " alice ", json.dumps([APPROVER_B]), TS.isoformat())
    with pytest.raises(ValidationError):
        store.get_thread_ownership("th-padded")


def test_naive_timestamp_in_the_database_is_rejected_on_read(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """库里的时间是 naive ⇒ 读取时被模型拒绝(与 `_to_iso` 同口径)。"""
    _inject(db_path, "th-naive", OWNER, json.dumps([APPROVER_B]), "2026-10-08T09:00:00")
    with pytest.raises(ValidationError):
        store.get_thread_ownership("th-naive")


def test_non_canonical_order_in_the_database_is_normalized_not_rejected(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """顺序不是"畸形" —— 同一集合的另一种书写被归一化,而不是被拒。

    顺序不承载语义,所以 `["carol","bob"]` 与 `["bob","carol"]` 是**同一**
    指派;读取按字典序归一化,与写入侧一致。
    """
    _inject(
        db_path, "th-order", OWNER,
        json.dumps([APPROVER_C, APPROVER_B]), TS.isoformat(),
    )
    got = store.get_thread_ownership("th-order")
    assert got is not None and got.approvers == (APPROVER_B, APPROVER_C)


def test_orphan_approver_rows_are_structurally_impossible(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """"有审批行、无属主行"这个形态**不存在** —— 没有独立的审批行表。

    这是 A3-2-FIX2 相对旧两表设计的一个具体收益:旧的孤儿形态(以及它带来的
    "孤儿不构成授权"这条需要额外论证的性质)现在连构造都构造不出来。
    """
    assert "thread_approvers" not in _table_names(db_path)
    assert store.get_thread_ownership("th-orphan") is None


# ---------------------------------------------------------------------------
# 5. append-only(库层,绕过 store 直连)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", _OWNERSHIP_TABLES)
def test_update_is_rejected_by_the_trigger(
    store: SqliteAuditStore, db_path: Path, table: str
) -> None:
    store.record_thread_ownership(_ownership())
    conn = _raw(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(f"UPDATE {table} SET thread_id = 'tampered'")
    finally:
        conn.close()


@pytest.mark.parametrize("table", _OWNERSHIP_TABLES)
def test_delete_is_rejected_by_the_trigger(
    store: SqliteAuditStore, db_path: Path, table: str
) -> None:
    store.record_thread_ownership(_ownership())
    conn = _raw(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(f"DELETE FROM {table}")
    finally:
        conn.close()


def test_approver_set_cannot_be_appended_after_registration(
    store: SqliteAuditStore, db_path: Path
) -> None:
    """**D-GATE-1 的回归**:注册之后无法通过直连 SQL 追加一个审批人。

    旧两表设计里这是真实可行的(向 `thread_approvers` 直连 INSERT 一行,
    且被 `get_thread_ownership` 原样读回)。现在没有那张表,任何"追加"
    尝试都只能失败;而改写既有行的 `approvers` 列则被 append-only 触发器
    拒绝。
    """
    store.record_thread_ownership(_ownership(approvers=(APPROVER_B,)))

    conn = _raw(db_path)
    try:
        # 1) 试图改既有行的审批集合 —— 被库层触发器拒绝
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE thread_owners SET approvers = ? WHERE thread_id = 'th-1'",
                (json.dumps([APPROVER_B, "mallory"]),),
            )
        # 2) 试图新增一行"同一线程的第二个审批人" —— 被主键拒绝
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO thread_owners (thread_id, owner, approvers, created_at)"
                " VALUES ('th-1', 'mallory', ?, ?)",
                (json.dumps(["mallory"]), TS.isoformat()),
            )
    finally:
        conn.close()

    assert store.get_thread_ownership("th-1").approvers == (APPROVER_B,)


def test_rejected_mutation_leaves_the_ownership_intact(
    store: SqliteAuditStore, db_path: Path
) -> None:
    store.record_thread_ownership(_ownership(approvers=(APPROVER_B, APPROVER_C)))
    conn = _raw(db_path)
    try:
        for statement in (
            "UPDATE thread_owners SET owner = 'tampered'",
            "DELETE FROM thread_owners",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)
        count = conn.execute("SELECT COUNT(*) FROM thread_owners").fetchone()[0]
    finally:
        conn.close()
    assert count == 1
    assert store.get_thread_ownership("th-1") == _ownership(
        approvers=(APPROVER_B, APPROVER_C)
    )


def test_store_exposes_no_update_or_delete_api(store: SqliteAuditStore) -> None:
    """store 的公开面上不存在"改归属"的方法 —— 改派在 v0.3.0 不存在(L-2)。"""
    public = {name for name in dir(store) if not name.startswith("_")}
    for forbidden in (
        "update_thread_ownership",
        "delete_thread_ownership",
        "reassign_thread_ownership",
        "remove_approver",
        "add_approver",
    ):
        assert forbidden not in public
