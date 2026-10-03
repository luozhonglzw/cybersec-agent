"""Phase 9.3-D:运维日志关联的确定性离线测试。

覆盖(Controller 冻结的 T1–T9):

    T1  HTTP 请求拿到 request_id,started/completed 共享它
    T2  两次 HTTP 请求拿到**不同**的 request_id
    T3  /triage 既有日志带 request_id + thread_id
    T4  HITL 暂停路径保留 request_id A + thread_id T + interrupt_id I
    T5  /resume 生成**新的** request_id B、同一 thread_id T、同一 interrupt_id I
        且 A != B
    T6  graph 既有事件拿到 request_id/thread_id,且图结果不变
    T7  异常路径保留既有 HTTP 语义,同时保留可用的请求关联
    T8  敏感请求材料不出现在新增/增补的日志字段里
    T9  代表性 triage/HITL 流程的**审计语义**不变

风格(冻结):
- 用 structlog.testing.capture_logs() + 精确字段断言,不做整段 JSON 快照;
- 全部 hermetic:数据文件与 audit.db 由 tmp_path 现场生成,不读仓库 data/;
- 无网络、无 provider、无 D-2。
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from structlog.testing import capture_logs

from app.api.main import create_app
from app.core.agent import SecurityAgent
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import TriageService
from app.security.store import SqliteAuditStore

# 30 次失败登录 + 恶意情报 → critical → 策略要求人工审批
BRUTE_FORCE_IP = "203.0.113.66"
BRUTE_FORCE_LOGINS = 30

# 10 次失败登录,无情报命中 → low → 仅 monitor → 策略放行
LOW_RISK_IP = "198.51.100.7"
LOW_RISK_LOGINS = 10

# 允许出现在日志里的键(白名单思路)。任何新增键都必须先在这里显式登记 ——
# 这条断言的意义是:不小心把请求体 / 头 / 消息内容塞进日志会立刻变红。
_ALLOWED_LOG_KEYS = frozenset({
    "event", "log_level",
    # 关联标识符
    "request_id", "thread_id", "interrupt_id", "incident_id", "tool_call_id",
    # HTTP 边界元数据
    "method", "path", "status_code", "duration_ms",
    # 业务/诊断元数据(均为计数、枚举或工具名,不含原始内容)
    "error_type", "tool_name", "indicator", "risk_level", "action_count",
    "gated_actions", "policy_version", "status", "operator", "has_plan",
    "user_message_length", "max_iterations", "iterations", "iteration_count",
    "tool_call_count", "count",
})

# 绝不允许出现的键名(即使值为空也算违规)
_FORBIDDEN_LOG_KEYS = frozenset({
    "message", "body", "headers", "authorization", "cookie", "cookies",
    "api_key", "apikey", "token", "secret", "prompt", "content", "args",
    "query", "params", "env", "environment",
})


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


def _write_logs(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(BRUTE_FORCE_LOGINS):
            f.write(json.dumps(_log_record(BRUTE_FORCE_IP, i)) + "\n")
        for i in range(LOW_RISK_LOGINS):
            f.write(json.dumps(_log_record(LOW_RISK_IP, i)) + "\n")


def _write_intel(path: Path) -> None:
    path.write_text(json.dumps({
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
    }) + "\n", encoding="utf-8")


@pytest.fixture
def data_paths(tmp_path: Path) -> tuple[Path, Path]:
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)
    return logs, intel


def _service(store, logs: Path, intel: Path, checkpointer) -> TriageService:
    hitl = HitlConfig(
        checkpointer=checkpointer,
        audit_store=store,
        logs_path=logs,
        intel_path=intel,
    )
    return TriageService(create_agent_graph(FakeLLMClient("分析完成"), hitl=hitl), store)


@pytest.fixture
def client(tmp_path: Path, data_paths) -> TestClient:
    """同时装配 /chat 与 /triage —— 两条链路都要能被关联断言覆盖。"""
    logs, intel = data_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    service = _service(store, logs, intel, InMemorySaver())
    return TestClient(create_app(agent=SecurityAgent(FakeLLMClient()), triage_service=service))


def _events(caps, name: str) -> list[dict]:
    return [rec for rec in caps if rec["event"] == name]


def _one(caps, name: str) -> dict:
    matched = _events(caps, name)
    assert len(matched) == 1, f"{name} 应恰好一条,实际 {len(matched)}"
    return matched[0]


def _request_id_of(caps) -> str:
    return _one(caps, "http.request.started")["request_id"]


# =====================================================================
# T1 / T2 — HTTP 边界
# =====================================================================

def test_t1_started_and_completed_share_one_request_id(client):
    with capture_logs() as caps:
        resp = client.post("/chat", json={"message": "分析一下 203.0.113.66"})

    assert resp.status_code == 200
    started = _one(caps, "http.request.started")
    completed = _one(caps, "http.request.completed")

    rid = started["request_id"]
    assert isinstance(rid, str) and len(rid) == 32
    assert completed["request_id"] == rid
    assert started["method"] == "POST" and started["path"] == "/chat"
    assert completed["status_code"] == 200
    assert isinstance(completed["duration_ms"], (int, float))
    assert completed["duration_ms"] >= 0


def test_t2_two_requests_get_distinct_request_ids(client):
    with capture_logs() as caps:
        client.post("/chat", json={"message": "a"})
        client.post("/chat", json={"message": "b"})

    ids = [rec["request_id"] for rec in _events(caps, "http.request.started")]
    assert len(ids) == 2
    assert ids[0] != ids[1]


# =====================================================================
# T3 / T4 — /triage 关联与 HITL 暂停
# =====================================================================

def test_t3_triage_service_and_graph_events_carry_request_and_thread_id(client):
    with capture_logs() as caps:
        resp = client.post("/triage", json={"indicator": LOW_RISK_IP})

    assert resp.status_code == 200
    thread_id = resp.json()["thread_id"]
    request_id = _request_id_of(caps)

    completed = _one(caps, "triage_completed")
    assert completed["request_id"] == request_id
    assert completed["thread_id"] == thread_id

    # graph 层既有事件同样带上这一对(同一 thread_id,同一 request_id)
    plan_ev = _one(caps, "graph_plan_created")
    assert plan_ev["request_id"] == request_id
    assert plan_ev["thread_id"] == thread_id

    agent_ev = _events(caps, "graph_agent_node")
    assert agent_ev, "agent 节点应产生日志"
    assert all(e["request_id"] == request_id for e in agent_ev)
    assert all(e["thread_id"] == thread_id for e in agent_ev)


def test_t4_hitl_interrupt_keeps_request_thread_and_interrupt_id(client):
    with capture_logs() as caps:
        resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP})

    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "pending_approval"
    assert body["interrupt_id"]

    request_id = _request_id_of(caps)
    pending = _one(caps, "triage_pending_approval")
    assert pending["request_id"] == request_id
    assert pending["thread_id"] == body["thread_id"]
    assert pending["interrupt_id"] == body["interrupt_id"]
    assert pending["incident_id"]

    # 图在暂停前落下的审批请求事件同样带关联
    requested = _one(caps, "graph_approval_requested")
    assert requested["request_id"] == request_id
    assert requested["thread_id"] == body["thread_id"]


# =====================================================================
# T5 — /resume 生成新 request_id、复用 thread_id / interrupt_id
# =====================================================================

def test_t5_resume_gets_new_request_id_same_thread_and_interrupt(client):
    with capture_logs() as caps_triage:
        paused = client.post("/triage", json={"indicator": BRUTE_FORCE_IP}).json()
    request_id_a = _request_id_of(caps_triage)

    with capture_logs() as caps_resume:
        resp = client.post("/resume", json={
            "thread_id": paused["thread_id"],
            "status": "approved",
            "operator": "analyst-1",
        })
    assert resp.status_code == 200

    request_id_b = _request_id_of(caps_resume)
    assert request_id_b != request_id_a, "两次独立 HTTP 请求必须得到不同的 request_id"

    resumed = _one(caps_resume, "triage_resumed")
    assert resumed["request_id"] == request_id_b
    assert resumed["thread_id"] == paused["thread_id"]
    assert resumed["interrupt_id"] == paused["interrupt_id"]

    decided = _one(caps_resume, "graph_approval_decided")
    assert decided["request_id"] == request_id_b
    assert decided["thread_id"] == paused["thread_id"]


# =====================================================================
# T6 — graph 事件增补字段不改变图结果
# =====================================================================

def test_t6_graph_events_enriched_without_changing_result(client):
    with capture_logs() as caps:
        resp = client.post("/triage", json={"indicator": LOW_RISK_IP})

    body = resp.json()
    # 图结果与既有契约完全一致
    assert body["status"] == "completed"
    assert body["interrupt_id"] is None
    assert body["approval_request"] is None
    assert body["plan"]["risk_level"] == "low"

    request_id = _request_id_of(caps)
    for name in ("graph_agent_node", "graph_plan_created", "graph_policy_allowed"):
        for rec in _events(caps, name):
            assert rec["request_id"] == request_id
            assert rec["thread_id"] == body["thread_id"]


# =====================================================================
# T7 — 异常路径保留 HTTP 语义与关联
# =====================================================================

def test_t7_exception_path_preserves_status_and_correlation(tmp_path, data_paths):
    logs, intel = data_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    missing = tmp_path / "missing.jsonl"
    app = create_app(triage_service=_service(store, missing, intel, InMemorySaver()))
    client = TestClient(app)

    with capture_logs() as caps:
        resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP})

    # 既有 HTTP 语义不变:数据源不可用 → 503,detail 不含路径
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert str(tmp_path) not in detail
    assert "/" not in detail and "\\" not in detail

    # 关联仍在
    request_id = _request_id_of(caps)
    rejected = _one(caps, "triage_rejected")
    assert rejected["request_id"] == request_id
    assert rejected["status_code"] == 503

    # 4xx 领域错误同样保留关联
    with capture_logs() as caps2:
        resp2 = client.post("/resume", json={
            "thread_id": "thread-never-existed", "status": "approved", "operator": "a",
        })
    assert resp2.status_code == 404
    assert _one(caps2, "triage_rejected")["request_id"] == _request_id_of(caps2)


def test_t7b_llm_client_error_handler_keeps_502_and_correlation():
    """既有 LLMClientError → 502 的映射不变,且关联被保留。"""
    app = create_app(agent=SecurityAgent(FakeLLMClient(raise_error=True)))

    with capture_logs() as caps:
        resp = TestClient(app).post("/chat", json={"message": "x"})

    assert resp.status_code == 502
    request_id = _request_id_of(caps)
    failed = _one(caps, "chat_failed_upstream")
    assert failed["request_id"] == request_id
    assert _one(caps, "http.request.completed")["status_code"] == 502


def test_t7c_unhandled_exception_is_reraised_with_correlation():
    """端点抛出**未处理**异常时,中间件记录终态后原样重抛 —— HTTP 语义不变。"""

    class _BoomAgent:
        async def chat(self, message, *, request_id=None):
            raise RuntimeError("boom")

    app = create_app(agent=_BoomAgent())

    with capture_logs() as caps:
        resp = TestClient(app, raise_server_exceptions=False).post(
            "/chat", json={"message": "x"}
        )

    assert resp.status_code == 500
    started = _one(caps, "http.request.started")
    completed = _one(caps, "http.request.completed")
    assert completed["request_id"] == started["request_id"]
    assert completed["status_code"] == 500
    assert completed["error_type"] == "RuntimeError"


# =====================================================================
# T8 — 敏感材料不入日志
# =====================================================================

def test_t8_sensitive_material_absent_from_logs(client):
    secret = "sk-LIVE-SECRET-0123456789abcdef"
    with capture_logs() as caps:
        client.post("/chat", json={"message": f"我的密钥是 {secret},请分析"})

    blob = repr(caps)
    assert secret not in blob, "用户消息内容绝不能被日志记录"

    for rec in caps:
        keys = set(rec) - {"event", "log_level"}
        assert keys <= _ALLOWED_LOG_KEYS, f"出现未登记字段:{sorted(keys - _ALLOWED_LOG_KEYS)}"
        assert not (keys & _FORBIDDEN_LOG_KEYS), f"出现禁止字段:{sorted(keys & _FORBIDDEN_LOG_KEYS)}"

    # /triage 路径同样检查
    with capture_logs() as caps2:
        client.post("/triage", json={"indicator": BRUTE_FORCE_IP})
    for rec in caps2:
        keys = set(rec) - {"event", "log_level"}
        assert keys <= _ALLOWED_LOG_KEYS
        assert not (keys & _FORBIDDEN_LOG_KEYS)


# =====================================================================
# T9 — 审计语义不变
# =====================================================================

def _audit_semantics(store: SqliteAuditStore, thread_id: str) -> dict:
    """审计的**语义不变式**(刻意不是整库/整行的字节比较)。

    interrupt_id 由框架在 interrupt() 内部分配,**每次运行本就不同** ——
    所以跨运行比较的是它的**存在性模式**(哪几条带、哪几条为 NULL),
    而不是字面值;字面值的一致性在单次流程内断言(见 t9b)。
    """
    rows = store.list_audit(thread_id=thread_id)
    return {
        "events": [r.event for r in rows],
        "plan_digests": [r.plan_digest for r in rows],
        "outcomes": [r.outcome for r in rows],
        "interrupt_id_presence": [r.interrupt_id is not None for r in rows],
        "actors": [r.actor for r in rows],
        "thread_ids": [r.thread_id for r in rows],
    }


def _run_hitl_flow(tmp_path: Path, data_paths, *, capture: bool) -> tuple[dict, str, dict]:
    logs, intel = data_paths
    store = SqliteAuditStore(tmp_path / f"audit-{capture}.db")
    service = _service(store, logs, intel, InMemorySaver())
    client = TestClient(create_app(triage_service=service))

    if capture:
        with capture_logs():
            paused = client.post("/triage", json={"indicator": BRUTE_FORCE_IP}).json()
            client.post("/resume", json={
                "thread_id": paused["thread_id"], "status": "approved", "operator": "analyst-1",
            })
    else:
        paused = client.post("/triage", json={"indicator": BRUTE_FORCE_IP}).json()
        client.post("/resume", json={
            "thread_id": paused["thread_id"], "status": "approved", "operator": "analyst-1",
        })
    return paused, paused["thread_id"], _audit_semantics(store, paused["thread_id"])


def test_t9_audit_semantics_unchanged_by_logging(tmp_path, data_paths):
    _, _, baseline = _run_hitl_flow(tmp_path, data_paths, capture=False)
    _, logged_thread_id, with_logging = _run_hitl_flow(tmp_path, data_paths, capture=True)

    # 事件类型与顺序
    assert with_logging["events"] == baseline["events"]
    assert with_logging["events"] == [
        "plan.created", "policy.evaluated", "approval.requested", "approval.decided",
    ]
    # plan_digest 行为不变(同一输入 → 同一摘要)
    assert with_logging["plan_digests"] == baseline["plan_digests"]
    assert all(d and len(d) == 64 for d in with_logging["plan_digests"])
    # 策略/审批结果与主体不变
    assert with_logging["outcomes"] == baseline["outcomes"]
    assert with_logging["actors"] == baseline["actors"]
    # 只有 approval.decided 带 interrupt_id(其余为 NULL)—— 模式跨运行一致
    assert with_logging["interrupt_id_presence"] == baseline["interrupt_id_presence"]
    assert with_logging["interrupt_id_presence"] == [False, False, False, True]
    # 审计仍全部挂在同一次运行自己的 thread_id 上
    assert with_logging["thread_ids"] == [logged_thread_id] * 4


def test_t9b_audit_thread_and_interrupt_linkage_preserved(tmp_path, data_paths):
    """审计仍以 thread_id 关联,approval.decided 仍带真实 interrupt_id。"""
    logs, intel = data_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    service = _service(store, logs, intel, InMemorySaver())
    client = TestClient(create_app(triage_service=service))

    paused = client.post("/triage", json={"indicator": BRUTE_FORCE_IP}).json()
    client.post("/resume", json={
        "thread_id": paused["thread_id"], "status": "approved", "operator": "analyst-1",
    })

    rows = store.list_audit(thread_id=paused["thread_id"])
    assert all(r.thread_id == paused["thread_id"] for r in rows)
    decided = [r for r in rows if r.event == "approval.decided"]
    assert len(decided) == 1
    assert decided[0].interrupt_id == paused["interrupt_id"]
    assert decided[0].outcome == "approved"
