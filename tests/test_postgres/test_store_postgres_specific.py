"""`PostgresAuditStore` 的 PostgreSQL 专有性质(Phase v0.2.0-M1b)。

这些性质**没有** SQLite 对等物,因此不进共享契约:

- 连接池的生命周期与清理;
- 数据库不可用时的故障注入行为;
- DSN 脱敏;
- SQL 注入抵抗(参数绑定的实证);
- 连接身份是受限角色,且该角色**没有** UPDATE / DELETE / TRUNCATE / DDL;
- **并发**注册同一线程只有一个胜者(v0.3.0-A3-2-FIX2);
- **源码级**护栏:写路径只有 INSERT,没有 UPDATE / DELETE / UPSERT / ON CONFLICT,
  也没有任何 DDL。
"""
from __future__ import annotations

import ast
import re
import threading
from pathlib import Path

import psycopg
import pytest
from psycopg_pool import PoolTimeout

import app.security.store_postgres as pg_store_module
from app.security.audit import build_audit_record
from app.security.store_postgres import PostgresAuditStore, redact_dsn
from tests.test_postgres._store_backends import (
    APPROVER_B,
    APPROVER_C,
    OWNER,
    TS,
    make_incident,
    make_ownership,
    make_request,
    uid,
)

pytestmark = pytest.mark.postgres

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 指向一个**关闭**的端口。不是任何真实凭据。
UNREACHABLE_DSN = (
    "postgresql://cybersec_app:not_a_real_password@127.0.0.1:1/cybersec_test"
)


# ---------------------------------------------------------------------------
# 1. 连接池生命周期
# ---------------------------------------------------------------------------


def test_constructor_performs_no_io(pg_app_dsn: str) -> None:
    """构造器**不**连接数据库 —— 因此也做不了运行时 DDL。

    这也是"数据库不可用应当在真正需要写审计时暴露"的前提。
    """
    store = PostgresAuditStore(pg_app_dsn)
    try:
        assert store._pool is None, "构造器不得开启连接池"
    finally:
        store.close()


def test_constructor_does_not_raise_for_unreachable_database() -> None:
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    store.close()  # 关一个从未开启的池必须是安全的


def test_pool_opens_lazily_on_first_operation(pg_app_dsn: str) -> None:
    store = PostgresAuditStore(pg_app_dsn)
    try:
        assert store._pool is None
        store.list_audit(thread_id=uid("absent"))
        assert store._pool is not None, "首次操作后池应当已开启"
    finally:
        store.close()


def test_close_is_idempotent_and_releases_the_pool(pg_app_dsn: str) -> None:
    store = PostgresAuditStore(pg_app_dsn)
    store.list_audit(thread_id=uid("absent"))
    assert store._pool is not None
    store.close()
    assert store._pool is None
    store.close()  # 第二次必须是 no-op
    assert store._pool is None


def test_operations_after_close_raise_and_do_not_silently_reopen(
    pg_app_dsn: str,
) -> None:
    """关掉之后再调用 ⇒ 明确报错。

    刻意**不**做静默复活:一个被关掉的审计 store 若能悄悄重开连接,
    就把"生命周期管理错误"藏了起来。
    """
    store = PostgresAuditStore(pg_app_dsn)
    store.list_audit(thread_id=uid("absent"))
    store.close()

    with pytest.raises(RuntimeError, match="已关闭"):
        store.list_audit(thread_id=uid("absent"))
    assert store._pool is None, "失败后也不得偷偷重开池"


def test_no_idle_in_transaction_connections_left_behind(pg_app_dsn: str) -> None:
    """操作完成后不得残留 `idle in transaction` 连接。

    这是 autocommit + 显式事务这套设计的直接可观测后果:悬挂事务会长期占住
    行锁、阻止 vacuum,是连接池最常见的隐性故障。
    """
    store = PostgresAuditStore(pg_app_dsn, min_size=2, max_size=4)
    try:
        tid = uid("thread")
        store.record_action_request(make_request(thread_id=tid))
        store.append_audit(build_audit_record("plan.created", thread_id=tid, ts=TS))
        store.list_audit(thread_id=tid)

        with psycopg.connect(pg_app_dsn, autocommit=True, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT state, count(*) FROM pg_stat_activity"
                    " WHERE usename = 'cybersec_app' GROUP BY state"
                )
                states = dict(cur.fetchall())
    finally:
        store.close()

    # 先证明这次查询**确实看到了** store 的连接 —— 否则下面的断言可能
    # 因为"一条都没查到"而恒真。
    assert states.get("idle", 0) >= 1, f"没有观察到 store 的空闲连接: {states}"
    assert states.get("idle in transaction", 0) == 0, f"残留悬挂事务: {states}"
    assert states.get("idle in transaction (aborted)", 0) == 0


# ---------------------------------------------------------------------------
# 2. 故障注入:数据库不可用
# ---------------------------------------------------------------------------


def test_unreachable_database_fails_loudly_on_first_operation() -> None:
    """数据库不可用 ⇒ 操作**响亮失败**,绝不静默成功。"""
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        with pytest.raises(PoolTimeout):
            store.list_audit(thread_id=uid("absent"))
    finally:
        store.close()


def test_write_to_unreachable_database_raises_and_writes_nothing() -> None:
    """写路径同样必须失败 —— 这是"必需写入"契约的前提。"""
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        with pytest.raises(PoolTimeout):
            store.append_audit(
                build_audit_record("plan.created", thread_id=uid("thread"), ts=TS)
            )
    finally:
        store.close()


def test_failure_is_repeatable_and_never_degrades_into_success() -> None:
    """连续失败必须**一直**失败 —— 不允许第 N 次悄悄变成成功。"""
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        for _ in range(3):
            with pytest.raises(PoolTimeout):
                store.list_audit(thread_id=uid("absent"))
    finally:
        store.close()


def test_pool_is_not_leaked_when_open_fails() -> None:
    """开启失败时不得留下半开的池。"""
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        with pytest.raises(PoolTimeout):
            store.list_audit(thread_id=uid("absent"))
        assert store._pool is None, "开启失败后不得留下池对象"
    finally:
        store.close()


def test_empty_dsn_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        PostgresAuditStore("   ")
    with pytest.raises(ValueError):
        PostgresAuditStore("")


def test_append_audit_signals_success_only_by_not_raising(pg_app_dsn: str) -> None:
    """写方法的返回值不携带状态 —— 成功**只能**由"没抛异常"表达。

    这条直接决定了 `graph.py` 的"必需写入 vs best-effort"契约能否成立:

    - `_audit_plan_failed`(best-effort)用 `try/except` 包住调用 ——
      它**依赖异常**来知道写失败了;
    - `plan.created`(必需写入)不包 —— 它依赖异常向上传播来中止;
    - `triage.py` 的 `record_incident` / `record_action_request` 同样不包。

    若 store 把失败变成静默成功(或返回一个假的成功标记),上面几处会**同时**
    静默失真。因此:返回值不携带状态,状态只由异常表达。
    """
    store = PostgresAuditStore(pg_app_dsn)
    try:
        tid = uid("thread")
        assert store.append_audit(
            build_audit_record("plan.created", thread_id=tid, ts=TS)
        ) is None
        assert store.record_incident(
            make_incident(incident_id=uid("inc"))
        ) is None
    finally:
        store.close()


def test_invalid_pool_sizes_are_rejected_at_construction(pg_app_dsn: str) -> None:
    with pytest.raises(ValueError):
        PostgresAuditStore(pg_app_dsn, min_size=-1)
    with pytest.raises(ValueError):
        PostgresAuditStore(pg_app_dsn, min_size=3, max_size=2)


# ---------------------------------------------------------------------------
# 3. DSN 脱敏
# ---------------------------------------------------------------------------


def test_repr_does_not_leak_the_password(pg_app_dsn: str) -> None:
    store = PostgresAuditStore(pg_app_dsn)
    try:
        assert "local_only" not in repr(store)
        assert "***" in repr(store)
    finally:
        store.close()


def test_connection_error_message_does_not_leak_the_password() -> None:
    """psycopg 的连接失败消息里不得出现口令。

    这条是**实测**而不是假设 —— 依赖库的消息格式会变,所以钉住它。
    """
    dsn = "postgresql://cybersec_app:super_secret_value@127.0.0.1:1/cybersec_test"
    with pytest.raises(psycopg.OperationalError) as excinfo:
        psycopg.connect(dsn, connect_timeout=1)
    assert "super_secret_value" not in str(excinfo.value)


def test_redact_dsn_replaces_the_password() -> None:
    redacted = redact_dsn("postgresql://user:hunter2@db.example:5432/appdb")
    assert "hunter2" not in redacted
    assert "***" in redacted
    assert "appdb" in redacted


def test_redact_dsn_does_not_echo_unparseable_input() -> None:
    """解析失败时返回占位符,**不**回显原文 —— 否则脱敏反而成了泄漏点。"""
    assert redact_dsn("this is not a dsn") == "<unparseable dsn>"


def test_redact_dsn_accepts_an_empty_dsn_without_leaking() -> None:
    """空 DSN 是**合法**的 conninfo(取默认值),解析成空串,不泄漏任何东西。"""
    assert redact_dsn("") == ""
    assert redact_dsn("   ") == ""


def test_redact_dsn_handles_a_dsn_without_password() -> None:
    redacted = redact_dsn("postgresql://user@db.example:5432/appdb")
    assert "***" not in redacted
    assert "appdb" in redacted


# ---------------------------------------------------------------------------
# 4. SQL 注入抵抗(参数绑定实证)
# ---------------------------------------------------------------------------

_INJECTION_PAYLOADS = [
    "'; DROP TABLE audit_logs; --",
    "' OR '1'='1",
    "x'; UPDATE audit_logs SET actor='pwned'; --",
    'x" OR 1=1 --',
]


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
def test_injection_payload_in_thread_filter_is_treated_as_a_literal(
    pg_app_dsn: str, payload: str
) -> None:
    """过滤条件里的注入载荷被当成**普通字符串**:查不到东西,表也不受损。"""
    store = PostgresAuditStore(pg_app_dsn)
    try:
        assert store.list_audit(thread_id=payload) == []
        assert store.list_action_rows(thread_id=payload) == []
        assert store.pending_action_rows(thread_id=payload) == []
        assert store.get_approval_request(payload) is None
        assert store.get_incident(payload) is None

        # 三张表仍在,且仍可正常读写
        tid = uid("thread")
        store.append_audit(build_audit_record("plan.created", thread_id=tid, ts=TS))
        assert len(store.list_audit(thread_id=tid)) == 1
    finally:
        store.close()


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
def test_injection_payload_in_event_filter_is_treated_as_a_literal(
    pg_app_dsn: str, payload: str
) -> None:
    store = PostgresAuditStore(pg_app_dsn)
    try:
        assert store.list_audit(event=payload) == []
    finally:
        store.close()


def test_injection_payload_is_stored_verbatim_not_executed(pg_app_dsn: str) -> None:
    """载荷写进去、读出来必须**逐字相同** —— 证明它没被当 SQL 执行。"""
    store = PostgresAuditStore(pg_app_dsn)
    payload = "'; DROP TABLE incidents; --"
    try:
        record = build_audit_record(
            "plan.created", thread_id=uid("thread"), ts=TS, detail={"note": payload}
        )
        store.append_audit(record)
        got = store.list_audit(thread_id=record.thread_id)
        assert len(got) == 1
        assert got[0].detail["note"] == payload
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 5. 连接身份与权限
# ---------------------------------------------------------------------------


def test_store_connects_as_the_restricted_runtime_role(pg_app_dsn: str) -> None:
    """store 实际使用的连接身份就是受限角色。

    注意:池连接配置了 `row_factory=dict_row`,所以 `fetchone()` 返回的是
    **字典**而不是元组 —— 按列名取值,不能按位置解包。
    """
    store = PostgresAuditStore(pg_app_dsn)
    try:
        with store._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user AS cu, session_user AS su")
                row = cur.fetchone()
        assert row["cu"] == "cybersec_app"
        assert row["su"] == "cybersec_app"
    finally:
        store.close()


def test_runtime_role_has_no_update_delete_truncate_or_ddl(pg_app_dsn: str) -> None:
    """**通过 store 自己的池连接**验证受限角色做不了破坏性操作。

    与 M1a 的 `test_pg_roles_and_triggers.py` 互补:那边验证角色本身,
    这边验证 store 实际使用的连接确实就是那个受限角色。
    """
    store = PostgresAuditStore(pg_app_dsn)
    forbidden = [
        "UPDATE audit_logs SET actor = 'x' WHERE 1 = 0",
        "DELETE FROM audit_logs WHERE 1 = 0",
        "TRUNCATE TABLE audit_logs",
        "ALTER TABLE audit_logs ADD COLUMN smuggled TEXT",
        "DROP TABLE audit_logs",
        "CREATE TABLE smuggled (id int)",
    ]
    try:
        for statement in forbidden:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with store._connection() as conn:
                    conn.execute(statement)
    finally:
        store.close()


def test_runtime_role_cannot_see_migration_metadata(pg_app_dsn: str) -> None:
    store = PostgresAuditStore(pg_app_dsn)
    try:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with store._connection() as conn:
                conn.execute("SELECT version_num FROM alembic_version")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 6. 源码级护栏
# ---------------------------------------------------------------------------


def _code_strings(module) -> list[str]:
    """模块里所有**非 docstring** 的字符串字面量(SQL 就住在这里)。"""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_source_contains_no_mutating_or_ddl_statements() -> None:
    """源码级护栏:写路径**只有 INSERT**。

    匹配的是真正的数据修改语句(带 `SET` 的 UPDATE、带 `FROM` 的 DELETE),
    以及任何 DDL 与 upsert 形态 —— 注释和文档字符串不在 AST 字符串里,
    所以"文档里说本模块不含 UPDATE"不会误报。
    """
    patterns = (
        r"\bUPDATE\s+\w+\s+SET\b",
        r"\bDELETE\s+FROM\b",
        r"\bINSERT\s+OR\s+REPLACE\b",
        r"\bON\s+CONFLICT\b",
        r"\bUPSERT\b",
        r"\bMERGE\s+INTO\b",
        r"\bDROP\s+(TABLE|INDEX|TRIGGER|FUNCTION|SCHEMA)\b",
        r"\bALTER\s+(TABLE|INDEX|SEQUENCE|ROLE)\b",
        r"\bCREATE\s+(TABLE|INDEX|TRIGGER|FUNCTION|SCHEMA|ROLE|OR\s+REPLACE)\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bTRUNCATE\b",
    )
    for text in _code_strings(pg_store_module):
        for pattern in patterns:
            assert not re.search(pattern, text, flags=re.IGNORECASE), (pattern, text)


def test_source_has_exactly_the_expected_insert_targets() -> None:
    """写入面只有 INSERT INTO 这四张表,没有第五个写入点。

    v0.3.0-A3-2 把集合从 3 扩到 4(新增 `thread_owners`);A3-2-FIX2 把归属
    收成单行后集合就是这四张。断言仍是**集合相等**,不是"包含" —— 悄悄多出
    一个写入点(例如重新引入可追加的审批行表)照样会被抓到。
    """
    targets: set[str] = set()
    for text in _code_strings(pg_store_module):
        targets.update(re.findall(r"INSERT\s+INTO\s+(\w+)", text, flags=re.IGNORECASE))
    assert targets == {
        "incidents",
        "action_requests",
        "audit_logs",
        "thread_owners",
    }


def test_source_never_interpolates_caller_values_into_sql() -> None:
    """源码级护栏:**调用方传进来的值**从不被插进 SQL 文本。

    唯一允许的 f-string 插值是模块级**常量列清单**
    (`{_ACTION_REQUEST_COLUMNS}`)—— 它不来自调用方。过滤值、`limit`、
    `event`、`thread_id` 等一律走 `%s` 参数绑定。

    这条是注入抵抗的**静态**证据;动态证据在
    `test_injection_payload_in_thread_filter_is_treated_as_a_literal` 等用例。
    """
    caller_values = (
        "{thread_id}", "{incident_id}", "{event}", "{limit}",
        "{request.", "{record.", "{incident.", "{self._dsn}",
    )
    for text in _code_strings(pg_store_module):
        for needle in caller_values:
            assert needle not in text, f"SQL 文本里插入了调用方值 {needle}: {text}"


def test_source_sql_literals_are_constant_or_parameterized() -> None:
    """每条**完整的** SELECT/INSERT 语句,只要带 WHERE 就必须带 `%s` 占位符。

    只看完整的语句字面量 —— `" WHERE "` 这类拼接片段本身不含值,
    不在检查范围内。
    """
    for text in _code_strings(pg_store_module):
        upper = text.upper()
        if "WHERE" in upper and ("SELECT" in upper or "INSERT" in upper):
            assert "%s" in text, f"含 WHERE 却没有参数占位符: {text}"


def test_source_imports_timestamp_helpers_from_the_sqlite_module() -> None:
    """时间语义与审批集合的序列化必须**单一真相源** —— 不得在 PG 侧另写一份。

    复制一份"归一化到 UTC + 拒绝 naive"或"规范序 JSON"的逻辑迟早会漂移,
    而审计流依赖顺序、"同一指派只有一个字节表示"是承重性质。
    """
    source = Path(pg_store_module.__file__).read_text(encoding="utf-8")
    assert "from app.security.store import (" in source
    for helper in (
        "_deserialize_approvers",
        "_from_iso",
        "_serialize_approvers",
        "_to_iso",
    ):
        assert helper in source, f"PG store 没有复用 {helper}"
    # 负对照:不得在 PG 侧**定义**一份同名的本地实现(那会绕开单一真相源)
    assert "def _serialize_approvers" not in source
    assert "def _deserialize_approvers" not in source
    assert "def _to_iso" not in source
    assert "def _from_iso" not in source


def test_source_does_not_modify_the_sqlite_module() -> None:
    """本阶段不得改动 `store.py` —— 它是 v0.1.0 的既有行为。"""
    sqlite_store = _REPO_ROOT / "app" / "security" / "store.py"
    assert sqlite_store.exists()
    source = sqlite_store.read_text(encoding="utf-8")
    # 既有实现里不得出现任何针对 PG 的适配痕迹
    assert "psycopg" not in source
    assert "store_postgres" not in source


# ---------------------------------------------------------------------------
# 7. 线程归属的数据库级不可变性(v0.3.0-A3-2-FIX2)
# ---------------------------------------------------------------------------


def test_runtime_role_late_approver_insert_is_rejected(pg_app_dsn: str) -> None:
    """**D-GATE-1 的 PG 侧回归**:运行时角色无法在注册之后追加审批人。

    运行时角色**有** `INSERT ON thread_owners`(那是正常注册路径需要的),
    因此"追加"只能表现为**再插一行同一 `thread_id`** —— 被主键拒绝。
    而"改写既有行的 approvers 列"则被权限层拒绝(没有 UPDATE)。

    这两条合起来,把旧两表设计里"直连 INSERT 一行就多一个审批人"的缺口
    彻底关掉。
    """
    store = PostgresAuditStore(pg_app_dsn)
    try:
        tid = uid("th")
        store.record_thread_ownership(
            make_ownership(thread_id=tid, approvers=(APPROVER_B,))
        )

        # 1) 追加一行"同一线程的第二个审批人" —— 主键拒绝
        with pytest.raises(psycopg.errors.IntegrityError):
            with store._connection() as conn:
                conn.execute(
                    "INSERT INTO thread_owners"
                    " (thread_id, owner, approvers, created_at)"
                    " VALUES (%s, %s, %s, %s)",
                    (tid, "mallory", '["mallory"]', TS.isoformat()),
                )

        # 2) 改写既有行的审批集合 —— 权限层拒绝(无 UPDATE)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with store._connection() as conn:
                conn.execute(
                    "UPDATE thread_owners SET approvers = %s WHERE thread_id = %s",
                    ('["bob","mallory"]', tid),
                )

        got = store.get_thread_ownership(tid)
        assert got is not None and got.approvers == (APPROVER_B,)
    finally:
        store.close()


def test_runtime_role_cannot_delete_or_truncate_ownership(pg_app_dsn: str) -> None:
    """运行时角色在归属表上没有 DELETE / TRUNCATE。"""
    store = PostgresAuditStore(pg_app_dsn)
    try:
        tid = uid("th")
        store.record_thread_ownership(make_ownership(thread_id=tid))
        for statement in (
            "DELETE FROM thread_owners WHERE false",
            "TRUNCATE TABLE thread_owners",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with store._connection() as conn:
                    conn.execute(statement)
    finally:
        store.close()


def test_concurrent_same_thread_registration_has_one_winner(pg_app_dsn: str) -> None:
    """**并发**注册同一线程:恰好一个胜者,**审批集合不会被合并**。

    两个线程各自用池里的**不同连接**同时注册同一个 `thread_id`,但带上
    **不同**的审批集合。唯一约束保证只有一个 INSERT 成功,另一个得到
    `UniqueViolation`。读回的审批集合必须**恰好等于胜者的那一份** ——
    不是并集、不是交集、不是"谁最后写谁赢"的覆盖。

    这是旧两表设计无法提供的性质:那里"追加"是合法的,于是并发下集合可能
    被合并(见 A3-2-FIX 报告的 F-2)。单行结构把这件事变成不可能。

    刻意用 `threading.Barrier` 让两个注册尽量同时发生 —— 但**断言与交错无关**:
    无论是否真的重叠,主键唯一性都保证结果相同,所以本用例不 flaky。
    """
    tid = uid("th-race")
    store = PostgresAuditStore(pg_app_dsn, min_size=2, max_size=4)
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}
    lock = threading.Lock()

    def attempt(name: str, approvers: tuple[str, ...]) -> None:
        barrier.wait(timeout=10)
        try:
            store.record_thread_ownership(
                make_ownership(thread_id=tid, approvers=approvers)
            )
        except psycopg.errors.IntegrityError:
            outcome = "rejected"
        else:
            outcome = "ok"
        with lock:
            results[name] = outcome

    try:
        threads = [
            threading.Thread(target=attempt, args=("a", (APPROVER_B,))),
            threading.Thread(target=attempt, args=("b", (APPROVER_C,))),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert sorted(results.values()) == ["ok", "rejected"], results

        got = store.get_thread_ownership(tid)
        assert got is not None
        # 恰好一个胜者:集合大小为 1,且等于 B 或 C 之一 —— 绝不是 {B, C}
        assert len(got.approvers) == 1, f"审批集合被合并了: {got.approvers}"
        assert got.approvers in ((APPROVER_B,), (APPROVER_C,))
        assert got.owner == OWNER
    finally:
        store.close()
