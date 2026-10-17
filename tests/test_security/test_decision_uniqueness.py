"""Phase v0.4.0-I1 TASK 4 —— 决策身份的**存储层**唯一性测试。

被锁定的性质(**不是**功能测试)
--------------------------------
决策身份 = **事件 + 作用域**,且唯一性必须由**存储层**强制:

    approval.decided → (thread_id, interrupt_id)   一次暂停只允许一个决定
    approval.timeout → (thread_id)                 一条线程只允许一次超时

为什么必须是存储层:恢复路径可能被**并发**触发,"先查再写"之间存在竞态
窗口;只有数据库在提交点判定的唯一约束才能给出"恰好一个胜者"。本文件
`test_at_most_once_holds_under_real_concurrency` 就是这条的直接检验。

反同义反复(本文件最重要的两条)
--------------------------------
1. `test_non_decision_primary_key_collision_is_not_mislabelled` —— 把一个
   **非决策事件**的主键撞车喂进去,断言它**原样**抛 `sqlite3.IntegrityError`。
   没有这条,"按身份回查"的分类器可能把所有完整性错误都吞成"决策重复",
   而下面的负向用例照样全绿。
2. `test_driver_error_alone_does_not_name_the_index` —— 断言**裸驱动报错**
   里**没有**索引名,而预检消息里有。没有这条,预检就只是"多此一举"的
   说法,无法证明它真的补充了信息。

隔离:全部 hermetic,数据库一律由 tmp_path 现场生成,不读仓库 `data/`。
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import (
    NotAwaitingApprovalError,
    TriageService,
)
from app.security.audit import build_audit_record
from app.security.checkpoint import (
    SafeAsyncSqliteSaver,
    build_checkpoint_serde,
)
import app.security.store as store_module
from app.security.store import (
    AuditUniquenessPreflightError,
    DecisionAlreadyRecordedError,
    SqliteAuditStore,
)

import aiosqlite

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TS = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)

INDEX_DECIDED = "uq_audit_decided_per_interrupt"
INDEX_TIMEOUT = "uq_audit_timeout_per_thread"


def _decided(thread_id: str, interrupt_id: str, *, actor: str = "alice"):
    return build_audit_record(
        "approval.decided",
        actor=actor,
        thread_id=thread_id,
        interrupt_id=interrupt_id,
        outcome="approved",
        ts=_TS,
    )


def _timeout(thread_id: str):
    return build_audit_record("approval.timeout", thread_id=thread_id, ts=_TS)


@pytest.fixture
def store(tmp_path: Path) -> SqliteAuditStore:
    return SqliteAuditStore(tmp_path / "audit.db")


def _index_names(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


# =====================================================================
# A. 结构性:索引确实被建出来,且形状就是冻结的那两条
# =====================================================================

def test_store_declares_exactly_the_two_frozen_indexes():
    """索引 SQL 的**形状**冻结:部分唯一索引 + 正确的 WHERE 子句。

    用集合相等而不是"包含":少一条、多一条、或 WHERE 子句被改宽成普通
    唯一索引(那会把正常的多条审计行判成冲突),都必须是有意为之。
    """
    declared = dict(store_module._DECISION_UNIQUE_INDEXES)
    assert set(declared) == {INDEX_DECIDED, INDEX_TIMEOUT}
    assert declared[INDEX_DECIDED] == (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_audit_decided_per_interrupt"
        " ON audit_logs (thread_id, interrupt_id)"
        " WHERE event = 'approval.decided'"
    )
    assert declared[INDEX_TIMEOUT] == (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_audit_timeout_per_thread"
        " ON audit_logs (thread_id) WHERE event = 'approval.timeout'"
    )


def test_identity_spec_is_exactly_the_frozen_set():
    """身份定义是**单一真相源**,两个后端都从这里取 —— 冻成常量。"""
    assert store_module._DECISION_IDENTITIES == {
        "approval.decided": ("thread_id", "interrupt_id"),
        "approval.timeout": ("thread_id",),
    }


def test_indexes_are_actually_created_in_a_real_store(store, tmp_path: Path):
    """绊线:上面那条只是断言"源码里写了",这条断言"库里真的有"。

    没有这条,把 `_decision_unique_index_statements()` 改成返回空列表
    (或在 `__init__` 里忘了调用)也能让所有结构断言保持绿色。
    """
    assert {INDEX_DECIDED, INDEX_TIMEOUT} <= _index_names(tmp_path / "audit.db")


def test_append_only_triggers_still_present_alongside_the_indexes(store, tmp_path):
    """新增索引**不得**挤掉 append-only 触发器。"""
    names = _index_names(tmp_path / "audit.db")
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        triggers = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert "audit_logs_no_update" in triggers
    assert "audit_logs_no_delete" in triggers
    assert isinstance(names, set)


def test_repeated_construction_is_idempotent(tmp_path: Path):
    """`__init__` 每次构造都跑 DDL ⇒ `IF NOT EXISTS` 必须真的幂等。"""
    db = tmp_path / "audit.db"
    SqliteAuditStore(db)
    SqliteAuditStore(db)
    SqliteAuditStore(db)
    assert {INDEX_DECIDED, INDEX_TIMEOUT} <= _index_names(db)


# =====================================================================
# B. 行为:at-most-once,且不过度约束
# =====================================================================

def test_duplicate_decided_is_rejected(store):
    store.append_audit(_decided("t1", "i1"))
    with pytest.raises(DecisionAlreadyRecordedError):
        store.append_audit(_decided("t1", "i1", actor="bob"))


def test_duplicate_timeout_is_rejected(store):
    store.append_audit(_timeout("t2"))
    with pytest.raises(DecisionAlreadyRecordedError):
        store.append_audit(_timeout("t2"))


def test_rejection_leaves_the_first_row_intact(store):
    """拒绝必须是 fail closed:**不覆盖**、**不新增**,已落库的那条不动。

    append-only 的意义就在这里 —— 若实现改成 upsert,第一行的 actor 会
    从 alice 变成 bob,这条断言立刻失败。
    """
    store.append_audit(_decided("t3", "i3", actor="alice"))
    with pytest.raises(DecisionAlreadyRecordedError):
        store.append_audit(_decided("t3", "i3", actor="bob"))
    rows = store.list_audit(thread_id="t3")
    decided = [r for r in rows if r.event == "approval.decided"]
    assert len(decided) == 1
    assert decided[0].actor == "alice"


def test_different_interrupt_same_thread_is_allowed(store):
    """同一线程的**不同**暂停各自可以有一个决定 —— 不得过度约束。"""
    store.append_audit(_decided("t4", "i4a"))
    store.append_audit(_decided("t4", "i4b"))
    assert len(
        [r for r in store.list_audit(thread_id="t4") if r.event == "approval.decided"]
    ) == 2


def test_same_interrupt_different_thread_is_allowed(store):
    """不同线程恰好复用同一个 interrupt_id 也必须放行。"""
    store.append_audit(_decided("t5a", "shared"))
    store.append_audit(_decided("t5b", "shared"))
    assert len(store.list_audit(thread_id="t5a")) == 1
    assert len(store.list_audit(thread_id="t5b")) == 1


def test_non_decision_events_are_not_over_constrained(store):
    """非决策事件天然可重复 —— 唯一索引**不得**波及它们。

    用 `plan.created` 连写三条(内容等价、id 不同),必须全部落地。
    这条挡的是"把部分唯一索引误写成普通唯一索引"这类改宽。
    """
    for _ in range(3):
        store.append_audit(build_audit_record("plan.created", thread_id="t6", ts=_TS))
    assert len(store.list_audit(thread_id="t6")) == 3


def test_approval_requested_may_repeat_with_null_interrupt_id(store):
    """`approval.requested` 的 interrupt_id 恒为 NULL,且天然可重复。

    它**不**是决策事件,不受唯一约束;若哪天有人把它也加进 `_DECISION_
    IDENTITIES`,这条会失败(同时 NULL 身份其实也挡不住重复)。
    """
    for _ in range(2):
        store.append_audit(
            build_audit_record("approval.requested", thread_id="t7", ts=_TS)
        )
    assert len(store.list_audit(thread_id="t7")) == 2


def test_null_interrupt_id_is_not_covered_by_the_index(store):
    """**已知限制,显式钉住**:NULL 身份不受唯一索引保护。

    SQLite 与 PostgreSQL 都认为 NULL 互不相等(实测)。本项目经 API 的
    /resume 由服务端注入 interrupt_id(D7),所以经 API 的决策恒为非 NULL;
    这条断言记录的是"直接调用图的旁路不在承诺范围内",而不是一个待修的 bug。
    若将来改用 `COALESCE` 表达式索引把 NULL 也纳入,这条会变红并提醒更新文档。
    """
    store.append_audit(
        build_audit_record(
            "approval.decided", thread_id="t8", interrupt_id=None, ts=_TS
        )
    )
    store.append_audit(
        build_audit_record(
            "approval.decided", thread_id="t8", interrupt_id=None, ts=_TS
        )
    )
    assert len(
        [r for r in store.list_audit(thread_id="t8") if r.event == "approval.decided"]
    ) == 2, "NULL 身份不受唯一索引保护 —— 若这条失败,说明行为已改变,须同步文档"


def test_at_most_once_holds_under_real_concurrency(tmp_path: Path):
    """**真并发**:N 个线程同时写同一个决策身份 → 恰好一条落库。

    这条是"唯一性必须落在存储层"的直接证据 —— 任何"先查再写"的实现在这里
    都会漏。用独立 store 实例(独立连接)模拟并发请求。
    """
    db = tmp_path / "audit.db"
    SqliteAuditStore(db)  # 建表
    workers = 8

    def _attempt(_: int) -> str:
        try:
            SqliteAuditStore(db).append_audit(_decided("t9", "i9"))
            return "ok"
        except DecisionAlreadyRecordedError:
            return "conflict"

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(_attempt, range(workers)))

    assert outcomes.count("ok") == 1, outcomes
    assert outcomes.count("conflict") == workers - 1, outcomes
    assert len(SqliteAuditStore(db).list_audit(thread_id="t9")) == 1


def test_append_only_still_bites_with_the_indexes_present(tmp_path: Path):
    """索引不得削弱 append-only:UPDATE / DELETE 仍必须被触发器拒绝。"""
    db = tmp_path / "audit.db"
    store = SqliteAuditStore(db)
    store.append_audit(_decided("t10", "i10"))
    conn = sqlite3.connect(db)
    try:
        for sql in (
            "UPDATE audit_logs SET actor = 'mallory' WHERE thread_id = 't10'",
            "DELETE FROM audit_logs WHERE thread_id = 't10'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
    finally:
        conn.close()
    assert len(store.list_audit(thread_id="t10")) == 1


# =====================================================================
# C. 分类器:决策重复与别的完整性错误必须分得开(反同义反复)
# =====================================================================

def test_non_decision_primary_key_collision_is_not_mislabelled(store):
    """**反同义反复**:主键撞车**不得**被说成"决策重复"。

    实测背景:SQLite 里 `PRIMARY KEY` 冲突与部分唯一索引冲突的消息形状
    **完全一样**(都是 `UNIQUE constraint failed: <表>.<列>`),只点名列、
    不点索引。若分类器改成解析消息措辞,这条会立刻暴露它把两者混为一谈。
    """
    record = build_audit_record("plan.created", thread_id="t11", ts=_TS)
    store.append_audit(record)
    with pytest.raises(sqlite3.IntegrityError) as caught:
        store.append_audit(record)  # 同一个 id
    assert not isinstance(caught.value, DecisionAlreadyRecordedError)
    assert "audit_logs.id" in str(caught.value)


def test_decision_error_is_not_a_driver_error_subclass():
    """调用方必须能区分"业务冲突(409)"与"别的完整性错误"。

    若它继承了 `sqlite3.IntegrityError`,HTTP 边界就再也分不清这两者。
    """
    assert not issubclass(DecisionAlreadyRecordedError, sqlite3.IntegrityError)
    assert issubclass(DecisionAlreadyRecordedError, RuntimeError)


# =====================================================================
# D. 预检:历史重复必须响亮失败,且**绝不**自动修数据
# =====================================================================

def _seed_duplicate_decided(db: Path, thread_id: str, interrupt_id: str) -> None:
    """在**索引不存在**的库上塞入重复行(模拟 v0.3.0 的历史库)。

    绕过 store 的 append_audit(它现在会被索引拒绝),直接用裸连接写入 ——
    这正是"历史数据先于约束存在"的真实形态。
    """
    conn = sqlite3.connect(db)
    try:
        for i in range(2):
            conn.execute(
                "INSERT INTO audit_logs (id, ts, actor, event, thread_id,"
                " interrupt_id, detail_json) VALUES (?, ?, ?, 'approval.decided',"
                " ?, ?, '{}')",
                (f"legacy-{i}", _TS.isoformat(), "alice", thread_id, interrupt_id),
            )
        conn.commit()
    finally:
        conn.close()


def test_preflight_rejects_a_database_with_historical_duplicates(tmp_path: Path):
    """历史库里有重复决策行 ⇒ 构造时响亮失败,而不是抛一句驱动报错。"""
    db = tmp_path / "legacy.db"
    SqliteAuditStore(db)
    conn = sqlite3.connect(db)
    conn.execute("DROP INDEX IF EXISTS uq_audit_decided_per_interrupt")
    conn.commit()
    conn.close()
    _seed_duplicate_decided(db, "legacy-thread", "legacy-iid")

    with pytest.raises(AuditUniquenessPreflightError):
        SqliteAuditStore(db)


def test_preflight_message_names_the_offending_index(tmp_path: Path):
    """预检的**全部价值**在于它比驱动报错更能指导行动 —— 必须点名索引。"""
    db = tmp_path / "legacy2.db"
    SqliteAuditStore(db)
    conn = sqlite3.connect(db)
    conn.execute("DROP INDEX IF EXISTS uq_audit_decided_per_interrupt")
    conn.commit()
    conn.close()
    _seed_duplicate_decided(db, "legacy-thread", "legacy-iid")

    with pytest.raises(AuditUniquenessPreflightError) as caught:
        SqliteAuditStore(db)
    message = str(caught.value)
    assert INDEX_DECIDED in message
    assert "legacy" not in message, "错误消息不得回显数据内容"


def test_driver_error_alone_does_not_name_the_index(tmp_path: Path):
    """**反同义反复**:裸驱动报错里**没有**索引名。

    这条证明预检不是"多此一举的说法" —— 没有预检,运维只会看到
    `UNIQUE constraint failed: audit_logs.thread_id, audit_logs.interrupt_id`,
    既不知道是哪条不变量,也拿不到索引名。
    """
    db = tmp_path / "dirty.db"
    SqliteAuditStore(db)
    conn = sqlite3.connect(db)
    conn.execute("DROP INDEX IF EXISTS uq_audit_decided_per_interrupt")
    conn.commit()
    conn.close()
    _seed_duplicate_decided(db, "legacy-thread", "legacy-iid")

    conn = sqlite3.connect(db)
    try:
        with pytest.raises(sqlite3.IntegrityError) as caught:
            conn.execute(store_module._DECISION_UNIQUE_INDEXES[0][1])
    finally:
        conn.close()
    assert INDEX_DECIDED not in str(caught.value)


def test_preflight_never_modifies_historical_rows(tmp_path: Path):
    """预检**只读**:失败之后行数必须原封不动。

    审计表是 append-only 证据,"自动清理重复"就是篡改证据 —— 这条钉住
    "绝不自动修复"。
    """
    db = tmp_path / "legacy3.db"
    SqliteAuditStore(db)
    conn = sqlite3.connect(db)
    conn.execute("DROP INDEX IF EXISTS uq_audit_decided_per_interrupt")
    conn.commit()
    conn.close()
    _seed_duplicate_decided(db, "legacy-thread", "legacy-iid")

    with pytest.raises(AuditUniquenessPreflightError):
        SqliteAuditStore(db)

    conn = sqlite3.connect(db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
    finally:
        conn.close()
    assert count == 2, "预检不得删除或改写历史行"


def test_preflight_catches_duplicate_timeouts(tmp_path: Path):
    """超时身份也要预检 —— 两条索引都要有对应的探针。"""
    db = tmp_path / "legacy4.db"
    SqliteAuditStore(db)
    conn = sqlite3.connect(db)
    conn.execute("DROP INDEX IF EXISTS uq_audit_timeout_per_thread")
    for i in range(2):
        conn.execute(
            "INSERT INTO audit_logs (id, ts, actor, event, thread_id, detail_json)"
            " VALUES (?, ?, 'system', 'approval.timeout', 'legacy-t', '{}')",
            (f"to-{i}", _TS.isoformat()),
        )
    conn.commit()
    conn.close()

    with pytest.raises(AuditUniquenessPreflightError) as caught:
        SqliteAuditStore(db)
    assert INDEX_TIMEOUT in str(caught.value)


def test_preflight_allows_a_clean_populated_database(tmp_path: Path):
    """正常库(有决策行、无重复)必须照常构造 —— 不得误报。"""
    db = tmp_path / "clean.db"
    store = SqliteAuditStore(db)
    store.append_audit(_decided("c1", "ci1"))
    store.append_audit(_timeout("c2"))
    SqliteAuditStore(db)  # 再构造一次不得抛
    assert len(store.list_audit(thread_id="c1")) == 1


# =====================================================================
# E. 服务层:重复决策必须收敛成 fail-closed 的领域错误(→ 409)
# =====================================================================

def _log_record(index: int) -> dict:
    return {
        "timestamp": f"2026-09-10T08:{index:02d}:00Z",
        "event_type": "login_failed",
        "source": "sshd",
        "source_ip": "203.0.113.66",
        "username": "root",
        "status": "failed",
        "severity": "high",
        "message": "Failed password for root",
    }


@pytest.fixture
def graph_env(tmp_path: Path):
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    with logs.open("w", encoding="utf-8") as fh:
        for i in range(30):
            fh.write(json.dumps(_log_record(i)) + "\n")
    intel.write_text(
        json.dumps({
            "indicator": "203.0.113.66", "indicator_type": "ip",
            "malicious": True, "confidence": 90, "severity": "high",
            "tags": ["ssh-brute-force"], "source": "test-fixture",
            "first_seen": "2026-09-01T00:00:00Z",
            "last_seen": "2026-09-10T00:00:00Z",
            "description": "SSH brute force source",
        }) + "\n",
        encoding="utf-8",
    )
    return logs, intel


async def _paused_service(tmp_path: Path, graph_env, thread_id: str):
    logs, intel = graph_env
    store = SqliteAuditStore(tmp_path / "audit.db")
    conn = await aiosqlite.connect(str(tmp_path / f"{thread_id}.db"))
    saver = SafeAsyncSqliteSaver(conn, serde=build_checkpoint_serde())
    await saver.setup()
    graph = create_agent_graph(
        FakeLLMClient("分析完成"),
        hitl=HitlConfig(
            checkpointer=saver, audit_store=store, logs_path=logs, intel_path=intel
        ),
    )
    cfg = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="分析该 IP")],
            "iteration_count": 0,
            "indicator": "203.0.113.66",
        },
        cfg,
    )
    return conn, TriageService(graph, store), store


async def test_second_sequential_service_resume_is_a_domain_conflict(
    tmp_path: Path, graph_env
):
    """顺序重复提交 → `NotAwaitingApprovalError`(既有 409 映射),不是 500。"""
    conn, service, _ = await _paused_service(tmp_path, graph_env, "svc-seq")
    try:
        await service.resume("svc-seq", status="approved", operator="alice")
        with pytest.raises(NotAwaitingApprovalError):
            await service.resume("svc-seq", status="approved", operator="bob")
    finally:
        await conn.close()


async def test_concurrent_service_resume_yields_one_decision_and_one_conflict(
    tmp_path: Path, graph_env
):
    """**并发**恢复:恰好一个决定落库,败者拿到领域冲突错误。

    两个请求都在对方落库前通过校验门(它们都读到"停在 human_approval"),
    真正拦住败者的是存储层唯一索引。异常原样穿出 `ainvoke`(实测),由
    服务层翻译成与顺序重复**同一个**领域错误。
    """
    import asyncio

    conn, service, store = await _paused_service(tmp_path, graph_env, "svc-conc")

    async def _one(operator: str) -> str:
        try:
            await service.resume("svc-conc", status="approved", operator=operator)
            return "ok"
        except NotAwaitingApprovalError:
            return "conflict"

    try:
        outcomes = await asyncio.gather(_one("alice"), _one("bob"))
    finally:
        await conn.close()

    assert sorted(outcomes) == ["conflict", "ok"], outcomes
    decided = [r for r in store.list_audit(thread_id="svc-conc")
               if r.event == "approval.decided"]
    assert len(decided) == 1


# =====================================================================
# F. 迁移 0003(纯结构性,不需要数据库 ⇒ 离线可跑)
# =====================================================================

_MIGRATION_0003 = (
    _REPO_ROOT / "migrations" / "versions" / "0003_decision_uniqueness.py"
)


def _load_0003():
    spec = importlib.util.spec_from_file_location("m0003", _MIGRATION_0003)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0003_hangs_forward_on_0002():
    module = _load_0003()
    assert module.revision == "0003_decision_uniqueness"
    assert module.down_revision == "0002_thread_ownership"


def test_migration_0003_up_creates_exactly_the_two_indexes():
    module = _load_0003()
    joined = "\n".join(module._STATEMENTS_UP)
    assert joined.count("CREATE UNIQUE INDEX") == 2
    assert INDEX_DECIDED in joined
    assert INDEX_TIMEOUT in joined
    # 部分索引的 WHERE 子句必须在,否则会把非决策事件也纳入唯一约束。
    assert "WHERE event = 'approval.decided'" in joined
    assert "WHERE event = 'approval.timeout'" in joined


def test_migration_0003_down_drops_only_its_own_indexes():
    """down-path 的**形状**护栏:绝不碰 0001 / 0002 的对象。

    这是"照 up-path 反向抄一遍"唯一会出错的地方 —— 顺手删掉别人的表或
    函数,就会留下一个坏掉的库。
    """
    module = _load_0003()
    down = module._STATEMENTS_DOWN
    assert len(down) == 2, "down-path 必须恰好两条语句"
    assert all(s.strip().upper().startswith("DROP INDEX IF EXISTS") for s in down)
    joined = " ".join(down).upper()
    assert "DROP TABLE" not in joined
    assert "DROP FUNCTION" not in joined
    assert "AUDIT_LOGS" not in joined, "不得删表 —— audit_logs 属于 0001"
    assert "THREAD_OWNERS" not in joined, "不得删表 —— thread_owners 属于 0002"


def test_migration_0003_up_contains_no_data_mutation():
    """UP 只建索引:不改数据、不删数据(历史行一律交给人处置)。"""
    module = _load_0003()
    joined = " ".join(module._STATEMENTS_UP).upper()
    for forbidden in ("DELETE ", "UPDATE ", "DROP ", "TRUNCATE", "ALTER "):
        assert forbidden not in joined, f"UP 不得包含 {forbidden!r}"


def test_migration_0003_identity_spec_cannot_drift_from_the_store():
    """迁移里的身份定义必须与应用里的**一致**。

    迁移刻意**不** import 应用代码(它是历史快照),代价就是这份常量会被
    抄一遍。这条断言把"抄错"变成红灯 —— 否则迁移与应用会各说各话。
    """
    module = _load_0003()
    migration_spec = {
        event: columns for _index, event, columns in module._IDENTITIES
    }
    assert migration_spec == store_module._DECISION_IDENTITIES


def test_migration_0003_preflight_runs_before_the_indexes():
    """预检必须在第一条 CREATE UNIQUE INDEX **之前**执行。

    顺序反了就毫无意义:索引自己会先撞上历史重复行,抛出一句只点名列、
    不点索引名的驱动报错。
    """
    source = _MIGRATION_0003.read_text(encoding="utf-8")
    upgrade_body = source.split("def upgrade() -> None:", 1)[1]
    assert upgrade_body.index("_assert_no_duplicate_decisions()") < upgrade_body.index(
        "op.execute(statement)"
    )
