"""Phase 8.4 API 层测试:POST /triage 与 POST /resume。

覆盖 HTTP 契约(状态码 / 响应形状)与两条**安全属性**:

1. thread_id 只能由服务端生成(D3)
   请求体里塞 thread_id 不会生效 —— 框架不校验 thread_id,若"客户端可指定"
   成立,就能复用别人暂停中的 state(覆盖 indicator、重跑 agent)= 劫持向量。

2. interrupt_id 只能由服务端恢复(D7)
   客户端能指定 interrupt_id 就等于能伪造"审批的是哪一次暂停"。

3. 框架的静默行为必须变成明确状态码
   未知 thread → 404;已完成 / checkpoint 丢失 → 409;
   数据源不可用 → 503(且 detail 不含路径)。

全部 hermetic:数据文件与 audit.db 都由 tmp_path 现场生成,
不读仓库 data/,也不需要 .env(注入 triage_service → lifespan 不运行)。
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api.main import create_app
from app.core.agent import SecurityAgent
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import TriageService
from app.security.store import SqliteAuditStore
from tests.conftest import APPROVER_HEADERS, TEST_APPROVER_SUBJECT

# 30 次失败登录 + 恶意情报 → critical → 策略要求人工审批
BRUTE_FORCE_IP = "203.0.113.66"
BRUTE_FORCE_LOGINS = 30

# 10 次失败登录,无情报命中 → low → 仅 monitor → 策略放行
LOW_RISK_IP = "198.51.100.7"
LOW_RISK_LOGINS = 10


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


@pytest.fixture
def store(tmp_path: Path) -> SqliteAuditStore:
    return SqliteAuditStore(tmp_path / "audit.db")


def _service(store, logs: Path, intel: Path, checkpointer) -> TriageService:
    hitl = HitlConfig(
        checkpointer=checkpointer,
        audit_store=store,
        logs_path=logs,
        intel_path=intel,
    )
    return TriageService(create_agent_graph(FakeLLMClient("分析完成"), hitl=hitl), store)


@pytest.fixture
def service(data_paths, store) -> TriageService:
    logs, intel = data_paths
    return _service(store, logs, intel, InMemorySaver())


@pytest.fixture
def client(service, store) -> TestClient:
    """**发起人**身份(conftest 的默认主体)的客户端。

    A3-3 起必须同时注入 `audit_store`:/triage 要在跑图之前写线程归属,
    /resume 要先读它做对象授权。用的是 service 自己持有的同一个 store。
    """
    return TestClient(create_app(triage_service=service, audit_store=store))


@pytest.fixture
def approver_client(service, store) -> TestClient:
    """**审批人**身份的客户端(与发起人不同,禁止自审批 D-7)。

    与 `client` 共享同一个 `service`(因此共享同一张图、同一个
    checkpointer、同一个 store)—— 只是认证身份不同。
    """
    return TestClient(
        create_app(triage_service=service, audit_store=store),
        headers=APPROVER_HEADERS,
    )


#: A3-3:发起一次判定必须显式指派审批人,且不能是自己。
APPROVERS = [TEST_APPROVER_SUBJECT]


# =====================================================================
# A. POST /triage
# =====================================================================

def test_triage_pending_approval_returns_200(client):
    resp = client.post("/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending_approval"
    assert body["thread_id"]
    assert body["interrupt_id"]
    assert body["plan"]["risk_level"] == "critical"
    assert body["approval_request"]["indicator"] == BRUTE_FORCE_IP
    assert body["approval"] is None


def test_triage_allowed_returns_completed(client):
    resp = client.post("/triage", json={"indicator": LOW_RISK_IP, "approvers": APPROVERS})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["interrupt_id"] is None
    assert body["approval_request"] is None
    assert body["plan"]["risk_level"] == "low"


def test_triage_response_shape_is_exact(client):
    """响应字段集是精确契约:多一个少一个都算接口变更。"""
    body = client.post("/triage", json={"indicator": LOW_RISK_IP, "approvers": APPROVERS}).json()

    assert set(body) == {
        "thread_id", "status", "answer", "plan",
        "approval_request", "approval", "interrupt_id",
    }


def test_triage_accepts_event_type(client):
    """event_type 是可选过滤条件,透传到 plan 节点(D6)。"""
    resp = client.post(
        "/triage",
        json={"indicator": BRUTE_FORCE_IP, "event_type": "login_success", "approvers": APPROVERS},
    )

    assert resp.status_code == 200
    assert resp.json()["plan"]["assessment"]["evidence"]["log_event_count"] == 0


def test_triage_ignores_client_supplied_thread_id(client, store):
    """D3:请求体里塞 thread_id 不生效 —— 服务端生成的值才是唯一真相。

    Pydantic 默认丢弃未声明字段,所以这里断言的是行为后果:
    返回的 thread_id 是新的 32 位 id,且没有任何审计落在攻击者指定的 id 上。
    """
    resp = client.post(
        "/triage",
        json={"indicator": LOW_RISK_IP, "thread_id": "hijack-attempt", "approvers": APPROVERS},
    )

    assert resp.status_code == 200
    thread_id = resp.json()["thread_id"]
    assert thread_id != "hijack-attempt"
    assert len(thread_id) == 32
    assert store.list_audit(thread_id="hijack-attempt") == []


@pytest.mark.parametrize("payload", [
    {},                                   # 缺 indicator
    {"indicator": ""},                    # 空串
    {"indicator": 123},                   # 类型错
    {"indicator": LOW_RISK_IP, "event_type": 5},  # event_type 类型错
])
def test_triage_rejects_invalid_payload(client, payload):
    assert client.post("/triage", json=payload).status_code == 422


def test_triage_data_unavailable_returns_503_without_path(tmp_path, store):
    """数据源缺失 → 503,且 detail 不含路径(部署信息不外泄)。"""
    intel = tmp_path / "threat_intel.jsonl"
    _write_intel(intel)
    missing = tmp_path / "missing.jsonl"
    app = create_app(
        triage_service=_service(store, missing, intel, InMemorySaver()),
        audit_store=store,
    )

    resp = TestClient(app).post(
        "/triage", json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS}
    )

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert str(tmp_path) not in detail
    assert "/" not in detail and "\\" not in detail


# =====================================================================
# B. POST /resume
# =====================================================================

def _pause(client, indicator: str = BRUTE_FORCE_IP) -> dict:
    resp = client.post("/triage", json={"indicator": indicator, "approvers": APPROVERS})
    assert resp.status_code == 200
    return resp.json()


def test_resume_approved_returns_200(client, approver_client):
    paused = _pause(client)
    resp = approver_client.post("/resume", json={
        "thread_id": paused["thread_id"],
        "status": "approved",
        "operator": "analyst-1",
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["approval"]["status"] == "approved"
    # A3-3:actor 是**已认证主体**,不是请求体里自述的 "analyst-1"
    assert body["approval"]["operator"] == TEST_APPROVER_SUBJECT
    assert body["interrupt_id"] == paused["interrupt_id"]


def test_resume_denied_returns_200(client, approver_client):
    paused = _pause(client)
    resp = approver_client.post("/resume", json={
        "thread_id": paused["thread_id"],
        "status": "denied",
        "operator": "analyst-2",
        "reason": "误报",
    })

    assert resp.status_code == 200
    assert resp.json()["approval"]["status"] == "denied"
    assert resp.json()["approval"]["reason"] == "误报"


def test_resume_keeps_approval_request(client, approver_client):
    """终态仍带 approval_request —— "批的是什么"要留在响应里(D2)。"""
    paused = _pause(client)
    body = approver_client.post("/resume", json={
        "thread_id": paused["thread_id"], "status": "approved", "operator": "a",
    }).json()

    assert body["approval_request"] is not None
    assert body["approval_request"]["indicator"] == BRUTE_FORCE_IP


def test_resume_ignores_client_supplied_interrupt_id(client, approver_client):
    """D7:客户端塞 interrupt_id 不生效,用的是服务端从 checkpoint 恢复的值。"""
    paused = _pause(client)
    body = approver_client.post("/resume", json={
        "thread_id": paused["thread_id"],
        "status": "approved",
        "operator": "analyst-1",
        "interrupt_id": "forged-interrupt-id",
    }).json()

    assert body["interrupt_id"] == paused["interrupt_id"]
    assert body["approval"]["interrupt_id"] == paused["interrupt_id"]
    assert body["approval"]["interrupt_id"] != "forged-interrupt-id"


def test_resume_unknown_thread_returns_404(client):
    resp = client.post("/resume", json={
        "thread_id": "thread-never-existed", "status": "approved", "operator": "a",
    })

    assert resp.status_code == 404


def test_resume_completed_thread_returns_409(client, approver_client):
    """重复 resume 已完成 thread:框架会**静默返回陈旧决定**,校验门必须拦住。"""
    paused = _pause(client)
    payload = {
        "thread_id": paused["thread_id"], "status": "approved", "operator": "analyst-1",
    }
    assert approver_client.post("/resume", json=payload).status_code == 200

    resp = approver_client.post("/resume", json={**payload, "status": "denied"})
    assert resp.status_code == 409


def test_resume_checkpoint_lost_returns_409(data_paths, store):
    """服务重启:审计还在,checkpoint 没了 → 409,而不是从 START 重跑一轮。"""
    logs, intel = data_paths
    paused = _pause(TestClient(create_app(
        triage_service=_service(store, logs, intel, InMemorySaver()),
        audit_store=store,
    )))

    restarted = TestClient(
        create_app(
            triage_service=_service(store, logs, intel, InMemorySaver()),
            audit_store=store,
        ),
        headers=APPROVER_HEADERS,
    )
    resp = restarted.post("/resume", json={
        "thread_id": paused["thread_id"], "status": "approved", "operator": "a",
    })

    assert resp.status_code == 409


@pytest.mark.parametrize("payload", [
    {"status": "approved", "operator": "a"},                      # 缺 thread_id
    {"thread_id": "t", "operator": "a"},                          # 缺 status
    {"thread_id": "t", "status": "maybe", "operator": "a"},       # status 非枚举
    {"thread_id": "t", "status": "approved"},                     # 缺 operator
    {"thread_id": "t", "status": "approved", "operator": ""},     # operator 空串
])
def test_resume_rejects_invalid_payload(client, payload):
    assert client.post("/resume", json=payload).status_code == 422


# =====================================================================
# C. 装配与回归
# =====================================================================

def test_triage_unavailable_without_service_returns_503():
    """只注入了 agent(测试场景)→ /triage 给明确 503,不是带 traceback 的 500。"""
    client = TestClient(create_app(agent=SecurityAgent(FakeLLMClient())))
    assert client.post("/triage", json={"indicator": LOW_RISK_IP, "approvers": APPROVERS}).status_code == 503


def test_openapi_documents_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) >= {"/chat", "/triage", "/resume"}


def test_every_triage_error_subclass_is_mapped_to_a_status():
    """新增 TriageError 子类必须显式归类。

    _triage_status_for 的兜底是 500。忘了归类的子类会**静默**退化成 500
    (带 traceback 的内部错误),而不是它应有的 4xx —— 这条把它变成红灯。
    """
    from app.api.main import _triage_status_for
    from app.core import triage as triage_module
    from app.core.triage import TriageError

    subclasses = [
        obj for name in dir(triage_module)
        if isinstance(obj := getattr(triage_module, name), type)
        and issubclass(obj, TriageError)
        and obj is not TriageError
    ]

    assert len(subclasses) >= 4  # Unknown / NotAwaiting / CheckpointLost / DataUnavailable
    for cls in subclasses:
        assert _triage_status_for(cls("x")) != 500, cls.__name__
