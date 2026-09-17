"""SqliteAuditStore 持久化层测试(Phase 8.2)。

覆盖:
- 三张表(incidents / action_requests / audit_logs)的写入与读回往返;
- D2 粒度:一行一个 action(action_requests 独立于 audit_logs 存在的意义);
- 派生查询 pending_action_rows(状态不落库,由"有无 approval.decided"推导);
- append-only 的三重保障:PRIMARY KEY 响亮失败、库层禁改触发器、源码无 UPDATE/DELETE;
- 时间列拒绝 naive datetime;
- 构造函数必须显式传 db_path(不存在"忘记传就落到仓库 data/"的可能)。

隔离要求(用户硬性约束):
    所有用例一律使用 pytest 的 tmp_path;禁止写真实 data/、禁止在仓库里
    创建 audit.db。文件末尾的 test_no_repo_audit_db_created 守着这条线。
"""
import ast
import inspect
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import app.security.store as store_module
from app.schemas.approval import ApprovalRequest
from app.schemas.audit import SYSTEM_ACTOR
from app.schemas.incident import Incident
from app.schemas.response import ResponseAction, ResponsePlan
from app.schemas.risk import RiskAssessment, RiskEvidence
from app.security.audit import build_audit_record, compute_plan_digest
from app.security.store import SqliteAuditStore

INDICATOR = "203.0.113.66"
THREAD_ID = "thread-abc"
INCIDENT_ID = "inc-001"
TS = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
TS_LATER = datetime(2026, 9, 17, 10, 5, 0, tzinfo=timezone.utc)

_TABLES = ("incidents", "action_requests", "audit_logs")
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------- 夹具构造(全部内存对象,不碰数据文件)----------

def _action(action_type: str = "block_ip", *, target: str = INDICATOR) -> ResponseAction:
    return ResponseAction(
        action_type=action_type,
        priority="high",
        target=target,
        rationale="疑似 SSH 爆破,建议封禁源 IP",
        requires_approval=action_type in ("block_ip", "isolate_host", "reset_credentials"),
        reversible=action_type != "reset_credentials",
    )


def _plan(*, summary: str = "疑似 SSH 爆破") -> ResponsePlan:
    evidence = RiskEvidence(
        indicator=INDICATOR, log_event_count=30, failed_login_count=30,
    )
    assessment = RiskAssessment(
        indicator=INDICATOR,
        risk_level="high",
        score=70,
        confidence=70,
        reasons=["30 次失败登录"],
        evidence=evidence,
    )
    return ResponsePlan(
        indicator=INDICATOR,
        risk_level="high",
        summary=summary,
        actions=[_action()],
        assessment=assessment,
    )


def _incident(*, incident_id: str = INCIDENT_ID, created_at: datetime = TS) -> Incident:
    return Incident(
        id=incident_id,
        created_at=created_at,
        indicator=INDICATOR,
        risk_level="high",
        score=70,
        summary="疑似 SSH 爆破",
        plan=_plan(),
    )


def _request(
    *,
    thread_id: str = THREAD_ID,
    actions: list[ResponseAction] | None = None,
    requested_at: datetime = TS,
) -> ApprovalRequest:
    return ApprovalRequest(
        thread_id=thread_id,
        indicator=INDICATOR,
        risk_level="high",
        score=70,
        summary="疑似 SSH 爆破",
        actions=actions if actions is not None else [_action()],
        policy_reasons=["动作按属性需要人工审批"],
        requested_at=requested_at,
    )


@pytest.fixture
def store(tmp_path: Path) -> SqliteAuditStore:
    """每个用例一个独立的临时库,用例之间零共享。"""
    return SqliteAuditStore(tmp_path / "audit.db")


# ---------- 构造函数契约 ----------

def test_db_path_is_required_without_default():
    """护栏:db_path 必填且无默认值。

    生产默认值必须由组合根从 Settings 注入;一旦这里出现默认值
    (比如 "data/audit.db"),测试忘记传路径就会静默写进仓库。
    """
    params = inspect.signature(SqliteAuditStore.__init__).parameters
    assert list(params) == ["self", "db_path"]
    assert params["db_path"].default is params["db_path"].empty


def test_parent_directory_is_created(tmp_path: Path):
    """换台机器 data/ 里只有 .gitignore,首次使用要能自建父目录。"""
    target = tmp_path / "nested" / "deeper" / "audit.db"
    assert not target.parent.exists()
    SqliteAuditStore(target)
    assert target.exists()


def test_schema_creation_is_idempotent(tmp_path: Path):
    """重复实例化同一个路径不得报错(IF NOT EXISTS 幂等)。"""
    path = tmp_path / "audit.db"
    SqliteAuditStore(path)
    SqliteAuditStore(path)


def test_all_three_tables_created(store: SqliteAuditStore, tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()
    assert set(_TABLES) <= names


def test_store_holds_no_long_lived_connection(store: SqliteAuditStore):
    """每次操作新开连接、用完即关 —— 因此本类没有 close()。"""
    assert not hasattr(store, "close")


def test_store_does_not_import_policy_module():
    """分工护栏:store.py 是纯 persistence,不做策略判定。"""
    assert "app.security.policy" not in _imported_modules(store_module)
    assert not hasattr(store_module, "evaluate_policy")


# ---------- incidents 往返 ----------

def test_record_and_get_incident_round_trip(store: SqliteAuditStore):
    incident = _incident()
    store.record_incident(incident)
    assert store.get_incident(INCIDENT_ID) == incident


def test_get_incident_missing_returns_none(store: SqliteAuditStore):
    assert store.get_incident("nope") is None


def test_incident_plan_survives_json_round_trip(store: SqliteAuditStore):
    """内嵌 assessment / evidence 必须原样还原(证据链不可丢)。"""
    store.record_incident(_incident())
    restored = store.get_incident(INCIDENT_ID)
    assert restored is not None
    assert restored.plan.assessment.evidence.failed_login_count == 30
    assert restored.plan.assessment.reasons == ["30 次失败登录"]
    assert compute_plan_digest(restored.plan) == compute_plan_digest(_incident().plan)


def test_incident_created_at_kept_tz_aware(store: SqliteAuditStore):
    store.record_incident(_incident())
    restored = store.get_incident(INCIDENT_ID)
    assert restored is not None
    assert restored.created_at.tzinfo is not None
    assert restored.created_at == TS


def test_incident_rejects_naive_datetime(store: SqliteAuditStore):
    """naive datetime 会让"字典序 == 时间序"失效 → 必须拒绝。"""
    naive = datetime(2026, 9, 17, 10, 0, 0)
    with pytest.raises(ValueError):
        store.record_incident(_incident(created_at=naive))


def test_incident_non_utc_offset_normalized_to_utc(store: SqliteAuditStore):
    """带偏移的时间写入时统一归一化到 UTC(同一时刻,统一表示)。"""
    from datetime import timedelta

    beijing = timezone(timedelta(hours=8))
    store.record_incident(_incident(created_at=TS.astimezone(beijing)))
    restored = store.get_incident(INCIDENT_ID)
    assert restored is not None
    assert restored.created_at == TS


# ---------- action_requests:一行一个 action(D2)----------

def test_one_row_per_action(store: SqliteAuditStore):
    """D2:粒度保持 action 级,不把整个 actions 数组塞进单行。"""
    request = _request(actions=[
        _action("block_ip"),
        _action("reset_credentials"),
        _action("collect_evidence", target="host-7"),
    ])
    ids = store.record_action_request(request)
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert len(store.list_action_rows()) == 3


def test_record_action_request_returns_ids_in_action_order(store: SqliteAuditStore):
    request = _request(actions=[_action("block_ip"), _action("reset_credentials")])
    ids = store.record_action_request(request)
    rows = store.list_action_rows()
    assert [row["row_id"] for row in rows] == ids
    assert [row["action"]["action_type"] for row in rows] == [
        "block_ip", "reset_credentials",
    ]


def test_approval_request_round_trip(store: SqliteAuditStore):
    request = _request(actions=[_action("block_ip"), _action("reset_credentials")])
    store.record_action_request(request)
    assert store.get_approval_request(THREAD_ID) == request


def test_get_approval_request_missing_returns_none(store: SqliteAuditStore):
    assert store.get_approval_request("nope") is None


def test_request_level_fields_repeated_per_row(store: SqliteAuditStore):
    """刻意的冗余:request 级字段在每行重复,换来"按动作查询"的能力。"""
    store.record_action_request(_request(actions=[_action("block_ip"), _action("escalate")]))
    rows = store.list_action_rows()
    assert {row["thread_id"] for row in rows} == {THREAD_ID}
    assert {row["indicator"] for row in rows} == {INDICATOR}
    assert {row["score"] for row in rows} == {70}


def test_policy_reasons_survive_round_trip(store: SqliteAuditStore):
    store.record_action_request(_request())
    restored = store.get_approval_request(THREAD_ID)
    assert restored is not None
    assert restored.policy_reasons == ["动作按属性需要人工审批"]


def test_action_flags_round_trip_as_bool(store: SqliteAuditStore):
    """INTEGER 0/1 读回必须是 bool(否则 requires_approval 会变成 1/0)。"""
    store.record_action_request(_request(actions=[_action("block_ip")]))
    action = store.get_approval_request(THREAD_ID).actions[0]
    assert action.requires_approval is True
    assert action.reversible is True


def test_incident_id_is_nullable(store: SqliteAuditStore):
    """D4:写入顺序由 Phase 8.3 决定,现在不提前绑定生命周期。"""
    store.record_action_request(_request())
    assert store.list_action_rows()[0]["incident_id"] is None


def test_incident_id_is_persisted_when_given(store: SqliteAuditStore):
    store.record_action_request(_request(), incident_id=INCIDENT_ID)
    assert store.list_action_rows()[0]["incident_id"] == INCIDENT_ID


def test_action_rows_filter_by_thread_id(store: SqliteAuditStore):
    store.record_action_request(_request(thread_id="t-1"))
    store.record_action_request(_request(thread_id="t-2"))
    assert len(store.list_action_rows(thread_id="t-1")) == 1
    assert len(store.list_action_rows()) == 2


def test_action_rows_filter_by_incident_id(store: SqliteAuditStore):
    store.record_action_request(_request(thread_id="t-1"), incident_id="inc-a")
    store.record_action_request(_request(thread_id="t-2"), incident_id="inc-b")
    assert len(store.list_action_rows(incident_id="inc-a")) == 1


def test_action_rows_projection_shape(store: SqliteAuditStore):
    store.record_action_request(_request(), incident_id=INCIDENT_ID)
    row = store.list_action_rows()[0]
    assert set(row) == {
        "row_id", "incident_id", "thread_id", "indicator", "risk_level",
        "score", "summary", "policy_reasons", "action", "requested_at",
    }
    assert row["action"]["action_type"] == "block_ip"
    assert row["requested_at"] == TS


def test_action_rows_requested_at_tz_aware(store: SqliteAuditStore):
    store.record_action_request(_request())
    assert store.list_action_rows()[0]["requested_at"].tzinfo is not None


def test_action_request_rejects_naive_datetime(store: SqliteAuditStore):
    with pytest.raises(ValueError):
        store.record_action_request(
            _request(requested_at=datetime(2026, 9, 17, 10, 0, 0))
        )


# ---------- pending_action_rows:派生状态(不落库)----------

def test_pending_rows_include_unanswered_request(store: SqliteAuditStore):
    store.record_action_request(_request())
    pending = store.pending_action_rows()
    assert len(pending) == 1
    assert pending[0]["thread_id"] == THREAD_ID


def test_pending_rows_exclude_decided_thread(store: SqliteAuditStore):
    """状态不落库:出现 approval.decided 后,该 thread 自动不再 pending。"""
    store.record_action_request(_request())
    assert len(store.pending_action_rows()) == 1

    store.append_audit(build_audit_record(
        "approval.decided",
        actor="analyst-1",
        thread_id=THREAD_ID,
        outcome="approved",
        ts=TS_LATER,
    ))
    assert store.pending_action_rows() == []


def test_pending_rows_scope_decision_to_its_own_thread(store: SqliteAuditStore):
    """一个 thread 的决定不得影响另一个 thread 的 pending 状态。"""
    store.record_action_request(_request(thread_id="t-1"))
    store.record_action_request(_request(thread_id="t-2"))
    store.append_audit(build_audit_record(
        "approval.decided", thread_id="t-1", outcome="denied", ts=TS_LATER,
    ))
    pending = store.pending_action_rows()
    assert [row["thread_id"] for row in pending] == ["t-2"]


def test_pending_rows_filter_by_thread_id(store: SqliteAuditStore):
    store.record_action_request(_request(thread_id="t-1"))
    store.record_action_request(_request(thread_id="t-2"))
    assert len(store.pending_action_rows(thread_id="t-1")) == 1


def test_pending_rows_ignores_unrelated_events(store: SqliteAuditStore):
    """只有 approval.decided 能解除 pending;其他事件不算数。"""
    store.record_action_request(_request())
    for event in ("plan.created", "policy.evaluated", "approval.requested"):
        store.append_audit(build_audit_record(event, thread_id=THREAD_ID, ts=TS))
    assert len(store.pending_action_rows()) == 1


def test_pending_rows_empty_when_no_request(store: SqliteAuditStore):
    assert store.pending_action_rows() == []


# ---------- audit_logs 往返 ----------

def test_append_and_list_audit_round_trip(store: SqliteAuditStore):
    plan = _plan()
    record = build_audit_record(
        "plan.created",
        thread_id=THREAD_ID,
        incident_id=INCIDENT_ID,
        plan=plan,
        detail={"action_count": 1},
        ts=TS,
    )
    store.append_audit(record)
    assert store.list_audit() == [record]


def test_audit_plan_digest_persisted(store: SqliteAuditStore):
    plan = _plan()
    store.append_audit(build_audit_record("plan.created", plan=plan, ts=TS))
    stored = store.list_audit()[0]
    assert stored.plan_digest == compute_plan_digest(plan)


def test_audit_nullable_fields_round_trip_as_none(store: SqliteAuditStore):
    store.append_audit(build_audit_record("approval.timeout", ts=TS))
    stored = store.list_audit()[0]
    assert stored.actor == SYSTEM_ACTOR
    assert stored.incident_id is None
    assert stored.thread_id is None
    assert stored.interrupt_id is None
    assert stored.outcome is None
    assert stored.reason is None
    assert stored.plan_digest is None
    assert stored.detail == {}


def test_audit_detail_round_trip_with_non_ascii(store: SqliteAuditStore):
    store.append_audit(build_audit_record(
        "policy.evaluated",
        reason="动作按属性需要人工审批",
        detail={"gated_actions": ["block_ip"], "note": "疑似 SSH 爆破"},
        ts=TS,
    ))
    stored = store.list_audit()[0]
    assert stored.detail["note"] == "疑似 SSH 爆破"
    assert stored.reason == "动作按属性需要人工审批"


def test_audit_ts_round_trip_tz_aware(store: SqliteAuditStore):
    store.append_audit(build_audit_record("policy.evaluated", ts=TS))
    stored = store.list_audit()[0]
    assert stored.ts == TS
    assert stored.ts.tzinfo is not None


def test_audit_rejects_naive_datetime(store: SqliteAuditStore):
    record = build_audit_record("policy.evaluated", ts=TS)
    record.ts = datetime(2026, 9, 17, 10, 0, 0)  # naive
    with pytest.raises(ValueError):
        store.append_audit(record)


def test_list_audit_ordered_by_ts_not_insertion_order(store: SqliteAuditStore):
    """审计流顺序由 ts 决定,与写入先后无关。"""
    store.append_audit(build_audit_record("approval.decided", ts=TS_LATER))
    store.append_audit(build_audit_record("plan.created", ts=TS))
    assert [r.event for r in store.list_audit()] == [
        "plan.created", "approval.decided",
    ]


def test_list_audit_filters(store: SqliteAuditStore):
    store.append_audit(build_audit_record(
        "plan.created", thread_id="t-1", incident_id="inc-a", ts=TS,
    ))
    store.append_audit(build_audit_record(
        "policy.evaluated", thread_id="t-2", incident_id="inc-b", ts=TS_LATER,
    ))
    assert len(store.list_audit(thread_id="t-1")) == 1
    assert len(store.list_audit(incident_id="inc-b")) == 1
    assert len(store.list_audit(event="plan.created")) == 1
    assert len(store.list_audit()) == 2
    assert store.list_audit(event="approval.timeout") == []


# ---------- append-only 保障 1:PRIMARY KEY 响亮失败 ----------

def test_duplicate_incident_pk_rejected(store: SqliteAuditStore):
    store.record_incident(_incident())
    with pytest.raises(sqlite3.IntegrityError):
        store.record_incident(_incident())


def test_duplicate_audit_pk_rejected(store: SqliteAuditStore):
    record = build_audit_record("plan.created", ts=TS)
    store.append_audit(record)
    with pytest.raises(sqlite3.IntegrityError):
        store.append_audit(record)


def test_duplicate_action_row_pk_rejected(store: SqliteAuditStore, tmp_path: Path):
    """行 id 是主键,重复插入必须报错而不是静默覆盖。"""
    store.record_action_request(_request())
    row_id = store.list_action_rows()[0]["row_id"]

    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO action_requests (id, thread_id, indicator, risk_level,"
                " score, summary, policy_reasons, action_type, priority, target,"
                " rationale, requires_approval, reversible, requested_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row_id, THREAD_ID, INDICATOR, "high", 70, "s", "[]",
                 "block_ip", "high", INDICATOR, "r", 1, 1, TS.isoformat()),
            )
    finally:
        conn.close()
    assert len(store.list_action_rows()) == 1


# ---------- append-only 保障 2:库层禁改触发器 ----------

def _seed(store: SqliteAuditStore, table: str) -> None:
    """往指定表塞一行真数据。

    SQLite 的 BEFORE UPDATE/DELETE 触发器是**行级**的:匹配不到行就不触发。
    空表上做 UPDATE/DELETE 会"看起来通过",那是假通过 —— 因此必须先播种。
    """
    if table == "incidents":
        store.record_incident(_incident())
    elif table == "action_requests":
        store.record_action_request(_request())
    elif table == "audit_logs":
        store.append_audit(build_audit_record("plan.created", ts=TS))
    else:  # pragma: no cover - 防御性分支
        raise AssertionError(f"未知表: {table}")


@pytest.mark.parametrize("table", _TABLES)
def test_update_trigger_rejects(
    store: SqliteAuditStore, tmp_path: Path, table: str,
):
    """把"我们承诺不 UPDATE"变成"数据库拒绝 UPDATE"。"""
    _seed(store, table)
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(f"UPDATE {table} SET id = 'tampered'")
    finally:
        conn.close()


@pytest.mark.parametrize("table", _TABLES)
def test_delete_trigger_rejects(
    store: SqliteAuditStore, tmp_path: Path, table: str,
):
    _seed(store, table)
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(f"DELETE FROM {table}")
    finally:
        conn.close()


@pytest.mark.parametrize("table", _TABLES)
def test_row_survives_rejected_mutation(
    store: SqliteAuditStore, tmp_path: Path, table: str,
):
    """被拒的修改不得留下痕迹:行还在,值没变。"""
    _seed(store, table)
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        for statement in (
            f"UPDATE {table} SET id = 'tampered'",
            f"DELETE FROM {table}",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_trigger_rejects_update_of_real_row(store: SqliteAuditStore, tmp_path: Path):
    """具体值也不得被改写(上面的 id 改动已证明被拒,这里锁定业务字段)。"""
    store.record_incident(_incident())
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE incidents SET score = 0 WHERE id = ?", (INCIDENT_ID,))
    finally:
        conn.close()
    restored = store.get_incident(INCIDENT_ID)
    assert restored is not None
    assert restored.score == 70


def test_triggers_exist_for_every_table(store: SqliteAuditStore, tmp_path: Path):
    """六条触发器(UPDATE/DELETE × 三张表)必须都在 —— 否则上面两条测试会假通过。"""
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    finally:
        conn.close()
    for table in _TABLES:
        assert f"{table}_no_update" in names
        assert f"{table}_no_delete" in names


# ---------- append-only 保障 3:源码级护栏 ----------

def _imported_modules(module) -> set[str]:
    """模块真实 import 的模块名(经 AST,不受注释/文档字符串里的同名文字干扰)。"""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _code_strings(module) -> list[str]:
    """模块里所有**非 docstring** 的字符串字面量(SQL 就住在这里)。

    注释不在 AST 里,文档字符串被显式排除 —— 所以"文档里写了
    '本模块不含 UPDATE / DELETE / INSERT OR REPLACE'"不会误报,
    只有真正的 SQL 字面量才会被检查。
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (
            ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
        )):
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


def test_source_contains_no_update_or_delete_statements():
    """源码级护栏:store.py 不得出现真正的 UPDATE / DELETE / REPLACE 语句。

    DDL 里的 "BEFORE UPDATE ON x" / "BEFORE DELETE ON x" 是触发器定义,
    不是数据修改语句 —— 因此匹配的是带 SET 的 UPDATE、带 FROM 的 DELETE。
    """
    for text in _code_strings(store_module):
        for pattern in (
            r"\bUPDATE\s+\w+\s+SET\b",
            r"\bDELETE\s+FROM\b",
            r"\bINSERT\s+OR\s+REPLACE\b",
            r"\bDROP\s+TABLE\b",
            r"\bALTER\s+TABLE\b",
        ):
            assert not re.search(pattern, text, flags=re.IGNORECASE), (pattern, text)


def test_source_has_exactly_three_insert_targets():
    """写入面只有 INSERT INTO 三张表,没有第四个写入点。"""
    targets: set[str] = set()
    for text in _code_strings(store_module):
        targets.update(re.findall(r"INSERT\s+INTO\s+(\w+)", text, flags=re.IGNORECASE))
    assert targets == {"incidents", "action_requests", "audit_logs"}


def test_trigger_ddl_is_present_in_source():
    """绊线:禁改触发器是 append-only 的库层保障,不得被悄悄删掉。"""
    sql = "\n".join(_code_strings(store_module))
    assert "RAISE(ABORT" in sql
    assert "BEFORE UPDATE ON" in sql
    assert "BEFORE DELETE ON" in sql


# ---------- 隔离:仓库里不得出现 audit.db ----------

def test_no_repo_audit_db_created():
    """用户硬性约束:测试不得在仓库里创建 audit.db。

    本文件所有用例都走 tmp_path;若这条断言失败,说明某个用例漏传了路径。
    """
    assert not (_REPO_ROOT / "data" / "audit.db").exists()


def test_store_writes_only_under_tmp_path(tmp_path: Path):
    """落盘位置就是传入路径本身,不会旁生别的文件。"""
    db = tmp_path / "audit.db"
    store = SqliteAuditStore(db)
    store.record_incident(_incident())
    store.record_action_request(_request())
    store.append_audit(build_audit_record("plan.created", ts=TS))
    assert db.exists()
    assert {p.name for p in tmp_path.iterdir()} == {"audit.db"}
