"""Phase v0.2.0-M1c:FastAPI + PostgreSQL 审计后端的**真实** API 集成测试。

本模块覆盖 TASK 6 的 A–N 十四项,并把**同一组断言同时跑在 SQLite 与
PostgreSQL 两个后端上**(`backend` / `client` 夹具是参数化的)——
后端差异只允许出现在夹具与后端专属用例里,不允许出现在行为断言里。

Gate-F 纪律:数据库不可达 / 角色缺失 / 迁移失败一律 **ERROR**,绝不 skip
(见 `conftest.py`)。默认 `pytest` 运行不收集本目录(`-m 'not postgres'`)。

hermetic:数据文件与 SQLite 库都由 `tmp_path` 现场生成,不读仓库 `data/`;
LLM 一律用 `FakeLLMClient` —— **没有任何真实 provider 调用**。

共享库的现实约束(与 M1b 相同):PostgreSQL 测试库是 append-only 的,
写入的行删不掉。因此所有断言都按 `uuid4` 生成的 thread_id 作用域,
不依赖"库是空的"。
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.conninfo import conninfo_to_dict

import app.api.main as api_main
from app.api.main import create_app
from app.core.agent import SecurityAgent
from app.core.config import get_settings
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import APPROVAL_TIMEOUT, TriageService
from app.schemas.audit import AuditRecord
from app.security.audit import build_audit_record
from app.security.store import SqliteAuditStore
from app.security.store_postgres import PostgresAuditStore
from tests.conftest import (
    ANALYST_HEADERS,
    APPROVER_HEADERS,
    OWNER_HEADERS,
    TEST_APPROVER_SUBJECT,
    TEST_OWNER_SUBJECT,
    VIEWER_HEADERS,
)

#: A3-3:发起判定必须显式指派审批人,且不能是发起人自己。
APPROVERS = [TEST_APPROVER_SUBJECT]

pytestmark = pytest.mark.postgres

#: 30 次失败登录 + 恶意情报 → critical → 策略要求人工审批
BRUTE_FORCE_IP = "203.0.113.66"
#: 10 次失败登录,无情报命中 → low → 仅 monitor → 策略放行
LOW_RISK_IP = "198.51.100.7"

#: 语法合法但**没有监听者**的地址(端口 1),用于"数据库不可用"故障注入。
UNREACHABLE_DSN = (
    "postgresql://cybersec_app:not_a_real_password@127.0.0.1:1/cybersec_test"
)

BACKENDS = ("sqlite", "postgres")


# ---------------------------------------------------------------------------
# hermetic 数据夹具(与 tests/test_api/test_triage.py 同源,不读仓库 data/)
# ---------------------------------------------------------------------------


def _log_record(ip: str, index: int) -> dict:
    return {
        "timestamp": f"2026-09-10T08:{index:02d}:00Z",
        "event_type": "login_failed",
        "source": "sshd",
        "source_ip": ip,
        "username": "root",
        "status": "failed",
        "severity": "high",
        "message": "Failed password for root",
    }


@pytest.fixture
def data_paths(tmp_path: Path) -> tuple[Path, Path]:
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    with logs.open("w", encoding="utf-8") as handle:
        for i in range(30):
            handle.write(_json(_log_record(BRUTE_FORCE_IP, i)))
        for i in range(10):
            handle.write(_json(_log_record(LOW_RISK_IP, i)))
    intel.write_text(
        _json(
            {
                "indicator": BRUTE_FORCE_IP,
                "indicator_type": "ip",
                "malicious": True,
                "confidence": 90,
                "severity": "high",
                "tags": ["ssh-brute-force"],
                "source": "test-fixture",
                "first_seen": "2026-09-01T00:00:00Z",
                "last_seen": "2026-09-10T00:00:00Z",
                "description": "SSH brute force source",
            }
        ),
        encoding="utf-8",
    )
    return logs, intel


def _json(payload: dict) -> str:
    return json.dumps(payload) + "\n"


@pytest.fixture
def missing_data_paths(tmp_path: Path) -> tuple[Path, Path]:
    """**不存在**的数据文件 —— 用来让 plan 节点确定性地失败。"""
    return tmp_path / "nope.jsonl", tmp_path / "nope-intel.jsonl"


# ---------------------------------------------------------------------------
# 后端夹具(参数化:同一组断言跑两个后端)
# ---------------------------------------------------------------------------


@pytest.fixture(params=BACKENDS)
def backend(request, tmp_path: Path, pg_app_dsn: str):
    """返回 `(backend_name, store)`;PostgreSQL 侧用受限的 cybersec_app 角色。"""
    if request.param == "postgres":
        store: Any = PostgresAuditStore(pg_app_dsn)
    else:
        store = SqliteAuditStore(tmp_path / "audit.db")
    try:
        yield request.param, store
    finally:
        if isinstance(store, PostgresAuditStore):
            store.close()


def build_app(
    store: Any,
    data_paths: tuple[Path, Path],
    *,
    approval_timeout: timedelta = APPROVAL_TIMEOUT,
    reply: str = "分析完成",
):
    """装配一个完整但**注入式**的 app(lifespan 不运行,不需要 .env)。

    注入式装配刻意不接管资源释放 —— PostgreSQL store 由 `backend` 夹具关闭。
    """
    logs, intel = data_paths
    hitl = HitlConfig(
        checkpointer=InMemorySaver(),
        audit_store=store,
        logs_path=logs,
        intel_path=intel,
    )
    service = TriageService(
        create_agent_graph(FakeLLMClient(reply), hitl=hitl),
        store,
        approval_timeout=approval_timeout,
    )
    return create_app(
        agent=SecurityAgent(FakeLLMClient(reply)),
        triage_service=service,
        audit_store=store,
    )


@pytest.fixture
def client(backend, data_paths) -> TestClient:
    _, store = backend
    return TestClient(build_app(store, data_paths))


def _events(store: Any, thread_id: str) -> list[str]:
    return [record.event for record in store.list_audit(thread_id=thread_id)]


def _start_triage(client: TestClient, indicator: str = BRUTE_FORCE_IP) -> str:
    resp = client.post("/triage", json={"indicator": indicator, "approvers": APPROVERS})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending_approval"
    return body["thread_id"]


# ---------------------------------------------------------------------------
# 故障注入用的转发 store(只覆盖一个方法,其余全部转发)
# ---------------------------------------------------------------------------


class _DelegatingStore:
    """把 10 个契约方法原样转发给真实 store,可按方法名注入失败。

    刻意**不**继承具体后端:它只满足 `AuditStore` 契约,这正是"图与
    triage 服务只依赖契约"的活证据。

    A3-2 把契约从 8 个方法扩到 10 个(新增线程归属的读写),这里必须跟着
    转发 —— 否则 `/triage` 在登记归属时拿到 `AttributeError` 而不是真实
    的后端行为,故障注入用例会测到一个**假的** 500。
    """

    def __init__(self, inner: Any, *, fail_on: str | None = None, exc=None) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self._exc = exc or psycopg.OperationalError
        self.calls: list[str] = []
        self.appended: list[AuditRecord] = []

    def _gate(self, name: str) -> None:
        self.calls.append(name)
        if self._fail_on == name:
            raise self._exc("simulated backend failure")

    # ---- 写 ----
    def record_incident(self, incident) -> None:
        self._gate("record_incident")
        return self._inner.record_incident(incident)

    def record_action_request(self, request, *, incident_id=None):
        self._gate("record_action_request")
        return self._inner.record_action_request(request, incident_id=incident_id)

    def append_audit(self, record: AuditRecord) -> None:
        self.appended.append(record)
        self._gate("append_audit")
        return self._inner.append_audit(record)

    def record_thread_ownership(self, ownership) -> None:
        self._gate("record_thread_ownership")
        return self._inner.record_thread_ownership(ownership)

    # ---- 读 ----
    def get_incident(self, incident_id):
        self._gate("get_incident")
        return self._inner.get_incident(incident_id)

    def get_thread_ownership(self, thread_id):
        self._gate("get_thread_ownership")
        return self._inner.get_thread_ownership(thread_id)

    def get_approval_request(self, thread_id):
        self._gate("get_approval_request")
        return self._inner.get_approval_request(thread_id)

    def list_action_rows(self, *, thread_id=None, incident_id=None):
        self._gate("list_action_rows")
        return self._inner.list_action_rows(thread_id=thread_id, incident_id=incident_id)

    def pending_action_rows(self, *, thread_id=None):
        self._gate("pending_action_rows")
        return self._inner.pending_action_rows(thread_id=thread_id)

    def list_audit(self, *, thread_id=None, incident_id=None, event=None,
                   limit=None, descending=False):
        self._gate("list_audit")
        return self._inner.list_audit(
            thread_id=thread_id,
            incident_id=incident_id,
            event=event,
            limit=limit,
            descending=descending,
        )


# =====================================================================
# A. 默认 SQLite 行为不变 + 后端等价
# =====================================================================


def test_default_backend_is_sqlite_and_unchanged(tmp_path, monkeypatch, data_paths):
    """不设 AUDIT_BACKEND 时,lifespan 装配的是 SQLite 后端(与 v0.1.0 一致)。"""
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    monkeypatch.delenv("AUDIT_BACKEND", raising=False)
    monkeypatch.delenv("AUDIT_POSTGRES_DSN", raising=False)
    monkeypatch.setattr(api_main, "LLMClient", lambda *a, **k: FakeLLMClient("ok"))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            store = client.app.state.audit_store
            assert isinstance(store, SqliteAuditStore)
            assert not isinstance(store, PostgresAuditStore)
            assert (tmp_path / "audit.db").exists()
            assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 200
    finally:
        get_settings.cache_clear()


def test_triage_persists_incident_action_requests_and_audit(client, backend):
    """C:一次 /triage 的三类持久化事实在两个后端上完全同形。"""
    name, store = backend
    thread_id = _start_triage(client)

    # 审计事件:顺序固定(plan.created → policy.evaluated → approval.requested)
    assert _events(store, thread_id) == [
        "plan.created",
        "policy.evaluated",
        "approval.requested",
    ]
    # 计划只产出一次 —— 不存在重复执行
    assert len(store.list_audit(thread_id=thread_id, event="plan.created")) == 1

    # 审批单:按动作粒度落库,incident_id 已回填(证明 incident 也落了库)
    rows = store.list_action_rows(thread_id=thread_id)
    assert rows, f"{name}: 审批单没有落库"
    assert {row["thread_id"] for row in rows} == {thread_id}
    incident_ids = {row["incident_id"] for row in rows}
    assert len(incident_ids) == 1 and None not in incident_ids
    incident = store.get_incident(incident_ids.pop())
    assert incident is not None and incident.indicator == BRUTE_FORCE_IP

    # pending 派生可见 + request 级模型可重建
    assert store.pending_action_rows(thread_id=thread_id)
    request = store.get_approval_request(thread_id=thread_id)
    assert request is not None and request.indicator == BRUTE_FORCE_IP


def test_allowed_triage_persists_only_incident(client, backend):
    """策略放行路径:落 incident,但**不**伪造审批单(否则 pending 会失真)。"""
    name, store = backend
    resp = client.post("/triage", json={"indicator": LOW_RISK_IP, "approvers": APPROVERS})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["approval_request"] is None
    thread_id = body["thread_id"]

    assert _events(store, thread_id) == ["plan.created", "policy.evaluated"]
    assert store.list_action_rows(thread_id=thread_id) == []
    assert store.pending_action_rows(thread_id=thread_id) == []


# =====================================================================
# D. /resume 保留审批 / 拒绝 / 超时 / checkpoint 语义
# =====================================================================


@pytest.mark.parametrize("decision", ["approved", "denied"])
def test_resume_records_the_decision_and_ends_the_thread(client, backend, decision):
    name, store = backend
    thread_id = _start_triage(client)

    resp = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_id, "status": decision, "operator": "alice"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["approval"]["status"] == decision
    assert _events(store, thread_id)[-1] == "approval.decided"

    # 终态:不再是 pending,重复 resume → 409(不写第二条决定)
    assert store.pending_action_rows(thread_id=thread_id) == []
    again = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_id, "status": decision, "operator": "alice"},
    )
    assert again.status_code == 409
    assert len(store.list_audit(thread_id=thread_id, event="approval.decided")) == 1


def test_resume_unknown_thread_is_404(client):
    resp = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": uuid.uuid4().hex, "status": "approved", "operator": "x"},
    )
    assert resp.status_code == 404


def test_expired_approval_cannot_be_resumed_and_writes_timeout(backend, data_paths):
    """超时:写 approval.timeout、**绝不**写 approval.decided、且不可逆。"""
    _, store = backend
    client = TestClient(build_app(store, data_paths, approval_timeout=timedelta(0)))
    thread_id = _start_triage(client)

    resp = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_id, "status": "approved", "operator": "alice"},
    )
    assert resp.status_code == 409
    events = _events(store, thread_id)
    assert "approval.timeout" in events
    assert "approval.decided" not in events
    assert store.pending_action_rows(thread_id=thread_id) == []

    # 超时不可逆:补批也 409,且不会多写一条 timeout
    again = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_id, "status": "approved", "operator": "alice"},
    )
    assert again.status_code == 409
    assert len(store.list_audit(thread_id=thread_id, event="approval.timeout")) == 1


def test_checkpoint_is_in_memory_so_a_restart_cannot_resume(backend, data_paths):
    """M:checkpoint 仍是进程内 InMemorySaver —— 不声称跨重启恢复。

    新 app 实例 = 新 saver。旧 thread 的 pending 行还在库里(审计事实),
    但 checkpoint 已丢 → 409,且**不写**任何决定。
    """
    _, store = backend
    first = TestClient(build_app(store, data_paths))
    thread_id = _start_triage(first)

    second = TestClient(build_app(store, data_paths))
    resp = second.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_id, "status": "approved", "operator": "alice"},
    )
    assert resp.status_code == 409
    events = _events(store, thread_id)
    assert "approval.decided" not in events
    assert "approval.timeout" not in events
    # 审计事实仍在库里 —— 说明"丢失"的是 checkpoint,不是持久化
    assert "approval.requested" in events


# =====================================================================
# E. /audit/events 正确读取后端记录
# =====================================================================


def _seed(store: Any, count: int, *, thread_id: str) -> list[str]:
    ids: list[str] = []
    for i in range(count):
        record = build_audit_record(
            "plan.created", thread_id=thread_id, reason=f"r{i:03d}"
        )
        store.append_audit(record)
        ids.append(record.id)
    return ids


def test_audit_events_returns_backend_records(client, backend):
    name, store = backend
    thread_id = uuid.uuid4().hex
    ids = _seed(store, 3, thread_id=thread_id)

    asc = client.get(
        "/audit/events",
        headers=VIEWER_HEADERS,
        params={"thread_id": thread_id, "order": "asc"},
    )
    assert asc.status_code == 200
    assert [row["id"] for row in asc.json()] == ids

    desc = client.get(
        "/audit/events", headers=VIEWER_HEADERS, params={"thread_id": thread_id}
    )
    assert [row["id"] for row in desc.json()] == ids[::-1]

    limited = client.get(
        "/audit/events",
        headers=VIEWER_HEADERS,
        params={"thread_id": thread_id, "limit": 2, "order": "asc"},
    )
    assert [row["id"] for row in limited.json()] == ids[:2]

    assert client.get(
        "/audit/events",
        headers=VIEWER_HEADERS,
        params={"thread_id": uuid.uuid4().hex},
    ).json() == []

    # 读路径不得产生新的写入
    before = len(store.list_audit(thread_id=thread_id))
    client.get(
        "/audit/events", headers=VIEWER_HEADERS, params={"thread_id": thread_id}
    )
    assert len(store.list_audit(thread_id=thread_id)) == before == 3


def test_audit_events_after_triage_shows_the_whole_flow(client):
    """端到端:一次 /triage + /resume 之后,审计流可经 HTTP 读回。

    A3-3:`approval.decided` 的 `actor` 取**已认证主体**(approver 的 subject),
    请求体里那个 `operator: "bob"` 是**客户端伪造**的,不改变权威归属 ——
    这里同时断言"真实 actor 落库"与"伪造值无效"。
    """
    thread_id = _start_triage(client)
    client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_id, "status": "denied", "operator": "bob"},
    )
    rows = client.get(
        "/audit/events",
        headers=VIEWER_HEADERS,
        params={"thread_id": thread_id, "order": "asc"},
    ).json()
    assert [row["event"] for row in rows] == [
        "plan.created",
        "policy.evaluated",
        "approval.requested",
        "approval.decided",
    ]
    assert rows[-1]["outcome"] == "denied"
    assert rows[-1]["actor"] == TEST_APPROVER_SUBJECT
    assert rows[-1]["actor"] != "bob"


# =====================================================================
# F/G/H. 数据库不可用:不回退、读 503、写 500
# =====================================================================


def test_unreachable_postgres_never_falls_back_to_sqlite(data_paths):
    """F:显式选 postgres 且不可用 → 响亮失败,绝不悄悄退回 SQLite。"""
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        app = build_app(store, data_paths)
        client = TestClient(app)
        assert isinstance(app.state.audit_store, PostgresAuditStore)
        assert not isinstance(app.state.audit_store, SqliteAuditStore)

        resp = client.get("/audit/events", headers=VIEWER_HEADERS)
        assert resp.status_code == 503
        assert resp.json() == {"detail": "audit store unavailable"}
    finally:
        store.close()


def test_unreachable_postgres_at_startup_still_does_not_fall_back(monkeypatch, tmp_path):
    """F(启动期):lifespan 不因数据库不可达而换后端。

    `PostgresAuditStore` 的构造器不做 I/O,所以启动会成功 —— 但装配出来的
    仍是 PostgreSQL 后端,第一次读就以 503 暴露,而不是从 SQLite 返回 200。
    `AUDIT_DB_PATH` 指向一个临时文件,断言它**没有**被创建 —— 这是
    "没有悄悄建 SQLite 库"的直接证据。
    """
    sqlite_path = tmp_path / "must_not_be_created.db"
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("AUDIT_DB_PATH", str(sqlite_path))
    monkeypatch.setenv("AUDIT_BACKEND", "postgres")
    monkeypatch.setenv("AUDIT_POSTGRES_DSN", UNREACHABLE_DSN)
    monkeypatch.setattr(api_main, "LLMClient", lambda *a, **k: FakeLLMClient("ok"))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            store = client.app.state.audit_store
            assert isinstance(store, PostgresAuditStore)
            assert not isinstance(store, SqliteAuditStore)
            assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 503
        assert not sqlite_path.exists(), "不得因 PostgreSQL 不可用而回退到 SQLite"
    finally:
        get_settings.cache_clear()


def test_read_path_database_failure_is_503_with_fixed_text(data_paths):
    """G:读路径失败 → 503 + 固定文案,不泄露内部信息。"""
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        client = TestClient(build_app(store, data_paths))
        resp = client.get("/audit/events", headers=VIEWER_HEADERS)
        assert resp.status_code == 503
        assert resp.json() == {"detail": "audit store unavailable"}
        body = resp.text
        for leaked in ("127.0.0.1", "not_a_real_password", "psycopg", "Traceback"):
            assert leaked not in body
    finally:
        store.close()


def test_read_path_lifecycle_misuse_is_not_masked_as_503(pg_app_dsn, data_paths):
    """边界:关闭后再用是**生命周期误用**,必须 500,不能被当成"数据源不可用"。

    这是把"读失败 → 503"限定在**持久化层失败**的负向对照:如果把
    `RuntimeError` 也收进 503,真正的编码错误就会被伪装成部署问题。
    """
    store = PostgresAuditStore(pg_app_dsn)
    store.close()
    client = TestClient(build_app(store, data_paths), raise_server_exceptions=False)
    assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 500


def test_read_path_503_requires_the_postgres_error_mapping(data_paths, monkeypatch):
    """反同义反复:把 psycopg 错误从读边界移除 → 同一个请求变成 500。

    没有这条对照,`... == 503` 可能只是"恰好如此"。它证明
    `POSTGRES_STORE_ERRORS` 真的在承担 503 的映射,而不是装饰。
    """
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        client = TestClient(build_app(store, data_paths), raise_server_exceptions=False)
        assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 503

        monkeypatch.setattr(api_main, "_STORE_READ_FAILURES", (ValueError,))
        assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 500, (
            "移除 psycopg 错误映射后仍返回 503 —— 说明 503 与映射无关"
        )
    finally:
        store.close()


def test_write_path_database_failure_is_500(data_paths):
    """H:写路径失败保持 500(不映射成 503),且不返回成功体。"""
    store = PostgresAuditStore(UNREACHABLE_DSN, open_timeout=0.5, connect_timeout=1)
    try:
        client = TestClient(build_app(store, data_paths), raise_server_exceptions=False)
        resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS})
        assert resp.status_code == 500
        assert "thread_id" not in resp.text
        assert "pending_approval" not in resp.text
    finally:
        store.close()


def test_write_path_failure_during_operation_is_500(backend, data_paths):
    """H(运行中故障):图写到一半数据库掉线 → 500,绝不假成功。"""
    _, real = backend
    store = _DelegatingStore(real, fail_on="append_audit", exc=psycopg.OperationalError)
    client = TestClient(build_app(store, data_paths), raise_server_exceptions=False)
    resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS})
    assert resp.status_code == 500
    assert store.calls.count("append_audit") == 1


# =====================================================================
# I/J. 持久化失败语义:必需写入不假成功 / best-effort 不顶替原失败
# =====================================================================


def test_mandatory_persistence_failure_never_returns_false_success(backend, data_paths):
    """I:图跑完了、审计也写了,但 incident 落库失败 → 500(不是 200)。

    反同义反复:审计确实写进去了(证明图真的跑完了),所以这个 500
    不是"什么都没发生",而是"必需写入失败,所以整体失败"。
    """
    name, real = backend
    store = _DelegatingStore(real, fail_on="record_incident")
    client = TestClient(build_app(store, data_paths), raise_server_exceptions=False)
    resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS})

    assert resp.status_code == 500
    assert "thread_id" not in resp.text
    assert "incident" not in resp.text
    # 图确实执行过:plan.created / policy.evaluated / approval.requested 都写了
    written = [record.event for record in store.appended]
    assert written == ["plan.created", "policy.evaluated", "approval.requested"]
    assert store.calls.count("record_incident") == 1


def test_best_effort_plan_failed_audit_failure_does_not_replace_the_failure(
    backend, missing_data_paths
):
    """J:plan 失败时,写 plan.failed 审计也失败 → 仍报**原始**领域失败(503)。

    控制流护栏:`_audit_plan_failed` 只记日志、绝不抛出,所以审计旁路的
    故障不能顶替主失败。若它泄漏出来,响应会是 500 而不是 503。
    """
    _, real = backend
    store = _DelegatingStore(real, fail_on="append_audit")
    client = TestClient(build_app(store, missing_data_paths), raise_server_exceptions=False)

    resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "安全数据源不可用"}
    # 旁路确实被尝试过 —— 否则这条用例什么都没证明
    assert store.calls.count("append_audit") == 1
    # 旁路自身的写入失败了:记录被捕获但没落库
    assert [record.event for record in store.appended] == ["plan.failed"]


def test_plan_failed_is_audited_when_the_audit_store_is_healthy(backend, missing_data_paths):
    """J 的正向对照:store 健康时,plan.failed 必须真的落库。"""
    _, real = backend
    store = _DelegatingStore(real)
    client = TestClient(build_app(store, missing_data_paths), raise_server_exceptions=False)

    resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS})
    assert resp.status_code == 503
    assert [record.event for record in store.appended] == ["plan.failed"]
    assert store.appended[0].outcome == "failed"
    # 失败路径不写 incident / 审批单
    assert store.calls.count("record_incident") == 0
    assert store.calls.count("record_action_request") == 0


# =====================================================================
# K. 无重复执行 / 无新增授权绕过
# =====================================================================


def test_no_new_endpoints_were_added(client):
    """K:路由集合不变 —— 本阶段不新增任何入口。"""
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {"/chat", "/triage", "/resume", "/audit/events"}


def test_high_risk_still_requires_approval_and_does_not_auto_execute(client):
    """K:高风险仍停在人工审批 —— 没有自动执行、没有绕过。"""
    thread_id = _start_triage(client)
    assert thread_id
    # 停在 human_approval:重复 /triage 不会"顺手"把上一个批了
    body = client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS}).json()
    assert body["thread_id"] != thread_id
    assert body["status"] == "pending_approval"


def test_denial_is_recorded_and_not_converted_to_approval(client, backend):
    _, store = backend
    thread_id = _start_triage(client)
    resp = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_id, "status": "denied", "operator": "mallory"},
    )
    assert resp.status_code == 200
    assert resp.json()["approval"]["status"] == "denied"
    decided = store.list_audit(thread_id=thread_id, event="approval.decided")
    assert len(decided) == 1 and decided[0].outcome == "denied"


def test_hitl_toolset_still_excludes_the_planner():
    """K:单计划源不变量 —— HITL 工具集里没有规划工具。"""
    from app.core.graph import HITL_TOOLS
    from app.tools import DEFAULT_TOOLS

    assert {tool.name for tool in DEFAULT_TOOLS} == {
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
        "plan_response_tool",
    }
    assert "plan_response_tool" not in {tool.name for tool in HITL_TOOLS}


# =====================================================================
# K2. 对象级授权:SQLite / PostgreSQL 对等(同一组断言跑两个后端)
# =====================================================================


def test_object_authorization_is_backend_independent(backend, data_paths):
    """A3-3:对象级授权读的是**归属表**,行为必须与后端无关。

    归属由 `/triage` 经**真实 store** 落库(SQLite 与 PostgreSQL 各一份),
    随后同一组负例/正例断言在两个后端上都必须成立 —— 授权判定不得因为
    "换了后端"而放宽或收紧。这是 TASK 6 的 SQLite/PostgreSQL 对等项:
    授权是**应用层**逻辑,但它读的事实来自存储层,两侧必须给出同一答案。
    """
    name, store = backend
    client = TestClient(build_app(store, data_paths))

    # ---- 归属经真实后端落库,再原样读回(授权判定的**事实来源**) ----
    thread_id = _start_triage(client)
    ownership = store.get_thread_ownership(thread_id)
    assert ownership is not None, name
    assert ownership.owner == TEST_OWNER_SUBJECT, name
    assert ownership.approvers == (TEST_APPROVER_SUBJECT,), name

    # ---- 属主自审批 → 403(两个后端同一判定) ----
    owner_self = client.post(
        "/resume",
        headers=OWNER_HEADERS,
        json={"thread_id": thread_id, "status": "approved", "operator": "owner"},
    )
    assert owner_self.status_code == 403, name
    assert owner_self.json() == {"detail": "self-approval is prohibited"}, name

    # ---- 未知线程 → 404(角色准入通过,但对象不存在) ----
    unknown = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": uuid.uuid4().hex, "status": "approved", "operator": "x"},
    )
    assert unknown.status_code == 404, name

    # ---- 被指派的 approver 可按线程范围读审计(对象授权**通过**) ----
    scoped = client.get(
        "/audit/events",
        headers=APPROVER_HEADERS,
        params={"thread_id": thread_id, "order": "asc"},
    )
    assert scoped.status_code == 200, name
    assert [row["event"] for row in scoped.json()] == [
        "plan.created",
        "policy.evaluated",
        "approval.requested",
    ]

    # ---- approver 不带范围 → 403(两个后端同一判定) ----
    unfiltered = client.get("/audit/events", headers=APPROVER_HEADERS)
    assert unfiltered.status_code == 403, name
    assert unfiltered.json() == {
        "detail": "an explicit thread scope is required for this role"
    }, name

    # ---- 无关 / 无归属线程 → 404,且与"根本不存在"不可区分 ----
    unrelated = client.get(
        "/audit/events", headers=APPROVER_HEADERS, params={"thread_id": uuid.uuid4().hex}
    )
    missing = client.get(
        "/audit/events", headers=APPROVER_HEADERS, params={"thread_id": uuid.uuid4().hex}
    )
    assert unrelated.status_code == missing.status_code == 404, name
    assert unrelated.json() == missing.json() == {"detail": "not found"}, name


# =====================================================================
# K3. 授权负例:跨线程隔离 / analyst 范围 / 历史无归属读取(两后端)
# =====================================================================


def test_cross_thread_isolation_and_legacy_reads_are_backend_independent(
    backend, data_paths
):
    """A4/TASK 6:补齐 PostgreSQL 侧的授权**负例**覆盖。

    A3-3 的对等用例只覆盖了"属主自审批 / 未知线程 / 未过滤读取"三个格子。
    这里补上同样容易在换后端时悄悄放宽的另外三格 —— 同一组断言跑
    SQLite 与 PostgreSQL:

      - **跨线程隔离**:被指派到线程 A 的 approver,在线程 C 上未被指派,
        C 对他与"不存在"不可区分(404);
      - **analyst 范围**:只能读**自己拥有**的线程,别人的 → 404;
      - **历史无归属**:viewer 仍能读到 A3-3 之前留下的无归属审计行。

    末尾放一条**正对照**(被指派者确实能审批 A)—— 否则上面那些 404 可能
    只是"所有 /resume 都失败"这种恒真的假证据。
    """
    name, store = backend
    client = TestClient(build_app(store, data_paths))

    # 线程 A:owner = test-owner,指派 test-approver(默认主体发起)
    thread_a = _start_triage(client)

    # 线程 C:owner = test-analyst,指派 test-owner
    #   ⇒ test-approver **没有**被指派到 C
    created = client.post(
        "/triage",
        json={"indicator": BRUTE_FORCE_IP, "approvers": [TEST_OWNER_SUBJECT]},
        headers=ANALYST_HEADERS,
    )
    assert created.status_code == 200, created.text
    thread_c = created.json()["thread_id"]

    # ---- 跨线程隔离:test-approver 在 A 上被指派,在 C 上没有 ----
    isolated = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_c, "status": "approved", "operator": "x"},
    )
    assert isolated.status_code == 404, name
    assert isolated.json() == {"detail": "未知的 thread_id"}, name

    # ---- analyst:自己拥有的 → 200;别人的 → 404 ----
    own = client.get(
        "/audit/events",
        headers=ANALYST_HEADERS,
        params={"thread_id": thread_c, "order": "asc"},
    )
    assert own.status_code == 200, name
    assert [row["event"] for row in own.json()] == [
        "plan.created",
        "policy.evaluated",
        "approval.requested",
    ]
    other = client.get(
        "/audit/events", headers=ANALYST_HEADERS, params={"thread_id": thread_a}
    )
    assert other.status_code == 404, name

    # ---- 历史无归属的审计行:viewer 仍可读 ----
    ownerless = uuid.uuid4().hex
    _seed(store, 2, thread_id=ownerless)
    legacy = client.get(
        "/audit/events",
        headers=VIEWER_HEADERS,
        params={"thread_id": ownerless, "order": "asc"},
    )
    assert legacy.status_code == 200, name
    assert len(legacy.json()) == 2, name

    # ---- 正对照:被指派到 A 的 approver 确实能审批 A ----
    ok = client.post(
        "/resume",
        headers=APPROVER_HEADERS,
        json={"thread_id": thread_a, "status": "denied", "operator": "x"},
    )
    assert ok.status_code == 200, name
    assert ok.json()["approval"]["status"] == "denied", name


# =====================================================================
# L. 重复 lifespan 启停与资源释放
# =====================================================================


def _app_connection_count(conn) -> int:
    return conn.execute(
        "SELECT count(*) FROM pg_stat_activity WHERE usename = 'cybersec_app'"
    ).fetchone()[0]


def _wait_for_connection_count(conn, limit: int, timeout: float = 5.0) -> int:
    """等到 cybersec_app 的活动连接数 <= limit(连接关闭是异步的)。

    退出 lifespan 后 `pool.close()` 会终止连接,但服务端回收 `pg_stat_activity`
    行有一个很短的滞后;固定 sleep 是脆的,轮询才是稳的。
    """
    deadline = time.monotonic() + timeout
    count = _app_connection_count(conn)
    while count > limit and time.monotonic() < deadline:
        time.sleep(0.05)
        count = _app_connection_count(conn)
    return count


def test_repeated_lifespan_startup_shutdown_closes_the_pool(
    monkeypatch, pg_app_dsn, pg_connection
):
    """L:反复启停不留连接;每次退出都关闭连接池。"""
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("AUDIT_BACKEND", "postgres")
    monkeypatch.setenv("AUDIT_POSTGRES_DSN", pg_app_dsn)
    monkeypatch.setattr(api_main, "LLMClient", lambda *a, **k: FakeLLMClient("ok"))
    get_settings.cache_clear()
    try:
        before = _app_connection_count(pg_connection)
        for _ in range(3):
            with TestClient(create_app()) as client:
                store = client.app.state.audit_store
                assert isinstance(store, PostgresAuditStore)
                assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 200
                assert store._pool is not None, "一次读之后池必须已开"
            assert store._closed is True, "退出 lifespan 必须关池"
        after = _wait_for_connection_count(pg_connection, before)
        assert after <= before, f"连接泄漏:{before} → {after}"
    finally:
        get_settings.cache_clear()


def test_pool_is_not_created_per_request(pg_app_dsn, data_paths):
    """L:池是进程级的 —— 多次请求复用同一个池对象。"""
    store = PostgresAuditStore(pg_app_dsn)
    try:
        client = TestClient(build_app(store, data_paths))
        assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 200
        pool = store._pool
        assert pool is not None
        for _ in range(3):
            assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 200
        assert store._pool is pool, "不得每请求建池"
    finally:
        store.close()


# =====================================================================
# N. 无真实 provider 调用 / 无外部动作
# =====================================================================


def test_no_real_llm_client_is_ever_constructed(backend, data_paths, monkeypatch):
    """N:整个 API 流程里不得构造真实 `LLMClient`(否则会读 .env 并联网)。"""
    _, store = backend

    def _tripwire(*args, **kwargs):
        raise AssertionError("测试中不得构造真实 LLMClient")

    monkeypatch.setattr("app.core.llm.LLMClient.__init__", _tripwire)
    client = TestClient(build_app(store, data_paths))
    assert client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS}).status_code == 200
    assert client.post("/chat", json={"message": "hi"}).status_code == 200


# =====================================================================
# 安全审计:最小权限 / 注入抵抗 / 凭据不泄漏
# =====================================================================


def test_app_connects_as_the_restricted_runtime_role(pg_app_dsn):
    """运行期身份必须是 cybersec_app,且不能 DDL / UPDATE / DELETE / TRUNCATE。"""
    store = PostgresAuditStore(pg_app_dsn)
    try:
        with store._connection() as conn:
            identity = conn.execute(
                "SELECT current_user AS cu, session_user AS su"
            ).fetchone()
            assert identity["cu"] == "cybersec_app"
            assert identity["su"] == "cybersec_app"

            for statement in (
                "CREATE TABLE m1c_should_not_exist (id int)",
                "UPDATE incidents SET summary = 'x'",
                "DELETE FROM incidents",
                "TRUNCATE incidents",
            ):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute(statement)
    finally:
        store.close()


@pytest.mark.parametrize(
    "payload",
    ["'; DROP TABLE audit_logs; --", "1 OR 1=1", "%", "_", "' OR '1'='1"],
)
def test_audit_filter_is_injection_resistant(client, backend, payload):
    """注入载荷一律走参数绑定 → 200 + 空结果,且表结构完好。"""
    _, store = backend
    resp = client.get(
        "/audit/events", headers=VIEWER_HEADERS, params={"thread_id": payload}
    )
    assert resp.status_code == 200
    assert resp.json() == []
    # 表还在,读仍然可用(证明没有语句逃逸)
    assert store.list_audit(thread_id=uuid.uuid4().hex) == []


def test_postgres_password_never_leaks_through_the_api(client, pg_app_dsn):
    """凭据不得出现在任何响应体里(含校验失败与错误路径)。"""
    password = conninfo_to_dict(pg_app_dsn)["password"]
    responses = [
        client.get("/audit/events", headers=VIEWER_HEADERS),
        client.get("/audit/events", headers=VIEWER_HEADERS, params={"limit": 0}),
        client.post("/triage", json={"indicator": LOW_RISK_IP, "approvers": APPROVERS}),
        client.post(
            "/resume",
            headers=APPROVER_HEADERS,
            json={"thread_id": "x", "status": "approved", "operator": "y"},
        ),
    ]
    for resp in responses:
        assert password not in resp.text
        assert "not_a_real_password" not in resp.text
