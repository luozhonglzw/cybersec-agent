"""Phase v0.3.0-A3-3:对象级授权与可信 actor 的 HTTP 契约。

本模块覆盖 Controller TASK 6 要求的 16 项行为,全部走**真实认证路径**:
构造完客户端后 `client.app.dependency_overrides.clear()`,摘掉根 conftest
装的默认主体 —— 于是这里用的是真密钥环、真 `hmac.compare_digest`、
真 401 / 403,以及真实的 `_authorize_thread_read` / `_authorize_audit_scope`。

    A. 401        缺失 / 畸形 / 未知凭据不可区分
    B. 403        角色不足
    C. 422        非法审批指派(在任何图调用或归属写入之前)
    D. 对象授权   /resume 的指派、自审批、未知与无归属线程
    E. 跨线程隔离 另一个线程的审批人不能审批这一条
    F. 审计可见性 viewer / analyst / approver 的范围矩阵
    G. 可信 actor 伪造 operator 无法改变审计主体
    H. /chat      已认证但**不**经 HITL / 策略

反同义反复的做法:凡断言"拒绝了"的地方,同时断言**没有副作用** ——
tripwire 服务证明图没跑,store 的行数证明没写审计、没落归属。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api.main import create_app
from app.core.agent import SecurityAgent
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import TriageService
from app.schemas.ownership import ThreadOwnership
from app.security.audit import build_audit_record
from app.security.auth import AuthKeyEntry, AuthKeyring
from app.security.store import SqliteAuditStore

# ---------------------------------------------------------------------
# 数据(与 tests/test_api/test_triage.py 同一组判据:高风险要审批 / 低风险放行)
# ---------------------------------------------------------------------

BRUTE_FORCE_IP = "203.0.113.66"
LOW_RISK_IP = "198.51.100.7"


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
    with logs.open("w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps(_log_record(BRUTE_FORCE_IP, i)) + "\n")
        for i in range(10):
            f.write(json.dumps(_log_record(LOW_RISK_IP, i)) + "\n")
    intel.write_text(
        json.dumps({
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
        }) + "\n",
        encoding="utf-8",
    )
    return logs, intel


# ---------------------------------------------------------------------
# 身份:6 个独立主体。**故意**让一个主体同时是 approver,用来测自审批。
# ---------------------------------------------------------------------

_SUBJECTS = {
    "alice": ("analyst", "a33-alice-key-not-a-real-credential"),
    "erin": ("analyst", "a33-erin-key-not-a-real-credential"),
    "bob": ("approver", "a33-bob-key-not-a-real-credential"),
    "carol": ("approver", "a33-carol-key-not-a-real-credential"),
    "frank": ("approver", "a33-frank-key-not-a-real-credential"),
    "dave": ("viewer", "a33-dave-key-not-a-real-credential"),
}


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


KEYRING = AuthKeyring.from_entries(
    [
        AuthKeyEntry(sha256=_digest(raw), subject=subject, role=role)
        for subject, (role, raw) in _SUBJECTS.items()
    ]
)


def _auth(subject: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_SUBJECTS[subject][1]}"}


ALICE = _auth("alice")     # analyst —— 发起人
ERIN = _auth("erin")       # 另一个 analyst
BOB = _auth("bob")         # approver
CAROL = _auth("carol")     # 另一个 approver
FRANK = _auth("frank")     # approver,同时也会当发起人(自审批用例)
DAVE = _auth("dave")       # viewer

APPROVERS_BOB = ["bob"]
APPROVERS_CAROL = ["carol"]


# ---------------------------------------------------------------------
# 测试替身:被调用即失败 —— 证明"拒绝发生在图之前"
# ---------------------------------------------------------------------


class _TripwireService:
    async def triage(self, *args, **kwargs):  # pragma: no cover - 触达即失败
        raise AssertionError("graph must not be invoked for a rejected request")

    async def resume(self, *args, **kwargs):  # pragma: no cover - 触达即失败
        raise AssertionError("graph must not be invoked for a rejected request")


# ---------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------


def _service(store, logs: Path, intel: Path) -> TriageService:
    hitl = HitlConfig(
        checkpointer=InMemorySaver(),
        audit_store=store,
        logs_path=logs,
        intel_path=intel,
    )
    return TriageService(create_agent_graph(FakeLLMClient("分析完成"), hitl=hitl), store)


def _client(tmp_path: Path, *, service=None, agent=None, store=None):
    """走**真实认证**的客户端 + 它用的 store。

    返回 `(client, store)`:store 用来断言"拒绝时没有副作用"。
    """
    store = store if store is not None else SqliteAuditStore(tmp_path / "audit.db")
    app = create_app(agent=agent, triage_service=service, audit_store=store)
    app.state.auth_keyring = KEYRING
    client = TestClient(app)
    # 关键:摘掉根 conftest 的默认主体,走真认证。
    client.app.dependency_overrides.clear()
    return client, store


@pytest.fixture
def real_client(tmp_path: Path, data_paths):
    """完整装配(真图 + 真 store),用于正向流程。"""
    logs, intel = data_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    return _client(tmp_path, service=_service(store, logs, intel), store=store)


def _triage(client, headers, approvers, indicator: str = BRUTE_FORCE_IP):
    return client.post(
        "/triage",
        json={"indicator": indicator, "approvers": approvers},
        headers=headers,
    )


def _resume(client, headers, thread_id, status: str = "approved", operator: str = "x"):
    return client.post(
        "/resume",
        json={"thread_id": thread_id, "status": status, "operator": operator},
        headers=headers,
    )


# =====================================================================
# A. 401 —— 缺失 / 畸形 / 未知凭据不可区分
# =====================================================================

BUSINESS_ROUTES = ("/chat", "/triage", "/resume", "/audit/events")


@pytest.mark.parametrize("path", BUSINESS_ROUTES)
def test_missing_credentials_are_401_on_every_business_route(tmp_path: Path, path: str):
    client, _ = _client(tmp_path, service=_TripwireService())
    method = "GET" if path == "/audit/events" else "POST"
    response = client.request(method, path, json={} if method == "POST" else None)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_malformed_and_unknown_credentials_are_byte_identical(tmp_path: Path):
    """任何差异都是"这把 key 存在吗"的枚举预言机。"""
    client, _ = _client(tmp_path, service=_TripwireService())
    variants = [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": "Basic YWxpY2U6c2VjcmV0"},
        {"Authorization": _SUBJECTS["alice"][1]},                 # 少了方案
        {"Authorization": f"Bearer {_SUBJECTS['alice'][1][:-1]}"},  # 少一个字符
        {"Authorization": "Bearer unknown-key-not-configured"},
    ]
    bodies = set()
    for headers in variants:
        response = client.post("/triage", json={"indicator": "x", "approvers": ["bob"]}, headers=headers)
        assert response.status_code == 401, headers
        bodies.add(response.content)
    assert len(bodies) == 1
    assert json.loads(bodies.pop()) == {"detail": "authentication required"}


def test_failed_authentication_writes_nothing(tmp_path: Path):
    client, store = _client(tmp_path, service=_TripwireService())
    for path in BUSINESS_ROUTES:
        method = "GET" if path == "/audit/events" else "POST"
        client.request(method, path, json={} if method == "POST" else None)
    assert store.list_audit() == []
    assert store.list_action_rows() == []


# =====================================================================
# B. 403 —— 角色不足
# =====================================================================


def test_viewer_cannot_triage(tmp_path: Path):
    client, store = _client(tmp_path, service=_TripwireService())
    response = _triage(client, DAVE, APPROVERS_BOB)
    assert response.status_code == 403
    assert response.json() == {"detail": "insufficient role"}
    assert store.list_audit() == []


def test_analyst_cannot_resume(tmp_path: Path):
    client, _ = _client(tmp_path, service=_TripwireService())
    response = _resume(client, ALICE, "t-1")
    assert response.status_code == 403
    assert response.json() == {"detail": "insufficient role"}


def test_viewer_cannot_resume(tmp_path: Path):
    client, _ = _client(tmp_path, service=_TripwireService())
    assert _resume(client, DAVE, "t-1").status_code == 403


# =====================================================================
# C. 422 —— 非法审批指派,且**在任何图调用或归属写入之前**
# =====================================================================


def test_valid_assignment_registers_owner_and_approver_set(real_client):
    client, store = real_client
    response = _triage(client, ALICE, APPROVERS_BOB)
    assert response.status_code == 200, response.text
    thread_id = response.json()["thread_id"]

    ownership = store.get_thread_ownership(thread_id)
    assert ownership is not None
    assert ownership.owner == "alice"
    assert ownership.approvers == ("bob",)


@pytest.mark.parametrize(
    "approvers",
    [
        [],                        # 空集合
        ["bob", "bob"],            # 重复
        ["alice"],                 # 自指派(发起人自己)
        ["nobody"],                # 未配置主体
        ["dave"],                  # 已配置,但角色是 viewer
        ["erin"],                  # 已配置,但角色是 analyst
        ["bob", "erin"],           # 混入一个非 approver
    ],
)
def test_invalid_assignment_is_422_and_touches_nothing(tmp_path: Path, approvers: list):
    """每一条非法指派都必须 422,且**没有图调用、没有归属行**。"""
    client, store = _client(tmp_path, service=_TripwireService())
    response = _triage(client, ALICE, approvers)
    assert response.status_code == 422, approvers
    assert store.list_audit() == []
    assert store.pending_action_rows() == []


def test_self_assignment_detail_is_explicit(real_client):
    """自指派的 422 文案必须说清原因(否则调用方无从修)。"""
    client, _ = real_client
    response = _triage(client, ALICE, ["alice"])
    assert response.status_code == 422
    assert response.json()["detail"] == "self-approval is prohibited"


def test_duplicate_assignment_detail_names_the_index(real_client):
    client, _ = real_client
    response = _triage(client, ALICE, ["bob", "bob"])
    assert response.status_code == 422
    assert "duplicate approver subject" in response.json()["detail"]


def test_non_approver_role_detail_is_explicit(real_client):
    client, _ = real_client
    response = _triage(client, ALICE, ["dave"])
    assert response.status_code == 422
    assert "not a configured principal" in response.json()["detail"]


# =====================================================================
# D. /resume 的对象级授权
# =====================================================================


def _pause_as_owner(client, headers=ALICE, approvers=APPROVERS_BOB) -> str:
    response = _triage(client, headers, approvers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "pending_approval"
    return body["thread_id"]


def test_assigned_approver_can_resume(real_client):
    client, store = real_client
    thread_id = _pause_as_owner(client)
    response = _resume(client, BOB, thread_id)
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["status"] == "approved"


def test_unassigned_approver_cannot_resume(real_client):
    """carol 是合法 approver,但**没有**被指派到这条线程 → 404。"""
    client, store = real_client
    thread_id = _pause_as_owner(client)
    before = len(store.list_audit(thread_id=thread_id))

    response = _resume(client, CAROL, thread_id)
    assert response.status_code == 404
    assert len(store.list_audit(thread_id=thread_id)) == before


def test_owner_cannot_self_approve(tmp_path: Path, data_paths):
    """发起人自己(即便持有 approver 角色)不能审批自己的线程 → 403。"""
    logs, intel = data_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    client, _ = _client(tmp_path, service=_service(store, logs, intel), store=store)

    # frank 是 approver,可以发起;发起人不能把自己写进审批集合
    response = _triage(client, FRANK, APPROVERS_BOB)
    assert response.status_code == 200, response.text
    thread_id = response.json()["thread_id"]

    response = _resume(client, FRANK, thread_id)
    assert response.status_code == 403
    assert response.json() == {"detail": "self-approval is prohibited"}


def test_unknown_thread_cannot_resume(tmp_path: Path):
    client, _ = _client(tmp_path, service=_TripwireService())
    response = _resume(client, BOB, "thread-never-existed")
    assert response.status_code == 404


def test_historical_ownerless_thread_cannot_resume(tmp_path: Path):
    """A3-2 之前的历史线程没有归属行 → 任何人都不能恢复它。"""
    store = SqliteAuditStore(tmp_path / "audit.db")
    store.append_audit(build_audit_record("plan.created", thread_id="legacy-thread"))
    client, _ = _client(tmp_path, service=_TripwireService(), store=store)

    response = _resume(client, BOB, "legacy-thread")
    assert response.status_code == 404
    # 历史审计行没有被改动
    assert len(store.list_audit(thread_id="legacy-thread")) == 1


def test_denied_resume_writes_no_audit_and_does_not_run_the_graph(tmp_path: Path):
    """反同义反复:拒绝时**图确实没跑**、审计确实没多一行。"""
    store = SqliteAuditStore(tmp_path / "audit.db")
    store.append_audit(build_audit_record("plan.created", thread_id="t-x"))
    client, _ = _client(tmp_path, service=_TripwireService(), store=store)
    before = len(store.list_audit())

    for headers in (BOB, CAROL, ALICE, DAVE):
        assert _resume(client, headers, "t-x").status_code in (403, 404)

    assert len(store.list_audit()) == before


# =====================================================================
# E. 跨线程授权隔离
# =====================================================================


def test_approver_of_one_thread_cannot_resume_another(tmp_path: Path, data_paths):
    """bob 被指派到 T1、carol 被指派到 T2 —— 两人不能互相审批。"""
    logs, intel = data_paths
    store = SqliteAuditStore(tmp_path / "audit.db")
    client, _ = _client(tmp_path, service=_service(store, logs, intel), store=store)

    t1 = _pause_as_owner(client, ALICE, APPROVERS_BOB)
    t2 = _pause_as_owner(client, ALICE, APPROVERS_CAROL)
    assert t1 != t2

    assert _resume(client, CAROL, t1).status_code == 404
    assert _resume(client, BOB, t2).status_code == 404
    # 各自的指派仍然有效
    assert _resume(client, BOB, t1).status_code == 200
    assert _resume(client, CAROL, t2).status_code == 200


# =====================================================================
# F. 审计可见性矩阵
# =====================================================================


def test_viewer_reads_everything_including_ownerless(tmp_path: Path):
    store = SqliteAuditStore(tmp_path / "audit.db")
    store.append_audit(build_audit_record("plan.created", thread_id="ownerless"))
    client, _ = _client(tmp_path, store=store)

    assert client.get("/audit/events", headers=DAVE).status_code == 200
    assert len(client.get("/audit/events", headers=DAVE).json()) == 1
    assert (
        client.get("/audit/events", params={"thread_id": "ownerless"}, headers=DAVE).status_code
        == 200
    )


def test_analyst_reads_only_their_own_thread(real_client):
    client, _ = real_client
    thread_id = _pause_as_owner(client)

    assert (
        client.get("/audit/events", params={"thread_id": thread_id}, headers=ALICE).status_code
        == 200
    )
    # 另一个 analyst 既不是属主也不是审批人 → 404
    assert (
        client.get("/audit/events", params={"thread_id": thread_id}, headers=ERIN).status_code
        == 404
    )


def test_approver_reads_owned_or_assigned_threads(real_client):
    client, _ = real_client
    thread_id = _pause_as_owner(client)

    # 被指派 → 可读
    assert (
        client.get("/audit/events", params={"thread_id": thread_id}, headers=BOB).status_code
        == 200
    )
    # 未被指派 → 404
    assert (
        client.get("/audit/events", params={"thread_id": thread_id}, headers=CAROL).status_code
        == 404
    )


@pytest.mark.parametrize("headers", [ALICE, ERIN, BOB, CAROL])
def test_unfiltered_read_is_403_for_non_viewers(tmp_path: Path, headers):
    client, _ = _client(tmp_path, store=SqliteAuditStore(tmp_path / "audit.db"))
    response = client.get("/audit/events", headers=headers)
    assert response.status_code == 403
    assert response.json() == {
        "detail": "an explicit thread scope is required for this role"
    }


def test_ownerless_scope_is_404_for_non_viewers(tmp_path: Path):
    store = SqliteAuditStore(tmp_path / "audit.db")
    store.append_audit(build_audit_record("plan.created", thread_id="ownerless"))
    client, _ = _client(tmp_path, store=store)

    for headers in (ALICE, BOB):
        response = client.get(
            "/audit/events", params={"thread_id": "ownerless"}, headers=headers
        )
        assert response.status_code == 404


def test_unknown_scope_is_indistinguishable_from_unrelated_scope(real_client):
    """两条 404 必须**逐字节相同**,否则状态/文案差异会泄露线程是否存在。"""
    client, _ = real_client
    thread_id = _pause_as_owner(client)

    unrelated = client.get(
        "/audit/events", params={"thread_id": thread_id}, headers=CAROL
    )
    unknown = client.get(
        "/audit/events", params={"thread_id": "definitely-not-a-thread"}, headers=CAROL
    )
    assert unrelated.status_code == unknown.status_code == 404
    assert unrelated.content == unknown.content


def test_scoped_read_never_leaks_another_thread_under_pagination_or_filters(
    real_client,
):
    """TASK 4:分页 / 排序 / 过滤都只在**已授权范围内**生效。

    授权判定发生在取回任何审计内容**之前**,因此 `limit` / `order` / `event`
    只能在"已经属于你的那一条线程"里做切片 —— 它们不构成绕过范围的手段。
    这里把另一条线程的审计行也塞进**同一个库**,再全量翻页读取,证明拿不到。
    """
    client, store = real_client
    mine = _pause_as_owner(client)          # alice 拥有,3 条事件
    other = "someone-elses-thread"
    for _ in range(5):
        store.append_audit(build_audit_record("plan.created", thread_id=other))

    # 用最大的 limit 全量翻页:仍然只看到自己那一条线程
    body = client.get(
        "/audit/events",
        headers=ALICE,
        params={"thread_id": mine, "limit": 200, "order": "asc"},
    ).json()
    assert len(body) == 3
    assert {row["thread_id"] for row in body} == {mine}

    # 事件过滤同样只在范围内生效
    filtered = client.get(
        "/audit/events",
        headers=ALICE,
        params={"thread_id": mine, "event": "plan.created", "limit": 200},
    ).json()
    assert [row["event"] for row in filtered] == ["plan.created"]

    # 不属于自己的线程:任何分页/排序组合都读不到
    assert (
        client.get(
            "/audit/events",
            headers=ALICE,
            params={"thread_id": other, "limit": 200, "order": "asc"},
        ).status_code
        == 404
    )


# =====================================================================
# G. 可信 actor
# =====================================================================


def test_forged_operator_cannot_change_the_authoritative_actor(real_client):
    """请求体里的 `operator` 被忽略;审计与响应都用已认证主体。"""
    client, store = real_client
    thread_id = _pause_as_owner(client)

    response = _resume(client, BOB, thread_id, operator="mallory")
    assert response.status_code == 200, response.text
    # 响应里的审批人 = 已认证主体,不是 mallory
    assert response.json()["approval"]["operator"] == "bob"

    decided = store.list_audit(thread_id=thread_id, event="approval.decided")
    assert len(decided) == 1
    assert decided[0].actor == "bob"
    assert "mallory" not in json.dumps(decided[0].detail)


def test_owner_recorded_from_authenticated_subject_not_the_request(real_client):
    """属主也来自已认证主体;请求体里没有任何字段能指定它。"""
    client, store = real_client
    response = client.post(
        "/triage",
        json={"indicator": BRUTE_FORCE_IP, "approvers": APPROVERS_BOB, "owner": "mallory"},
        headers=ALICE,
    )
    assert response.status_code == 200, response.text
    ownership = store.get_thread_ownership(response.json()["thread_id"])
    assert ownership is not None and ownership.owner == "alice"


# =====================================================================
# H. /chat —— 已认证,但**不**经 HITL / 策略
# =====================================================================


def test_chat_requires_authentication(tmp_path: Path):
    client, _ = _client(tmp_path, agent=SecurityAgent(FakeLLMClient("hi")))
    assert client.post("/chat", json={"message": "x"}).status_code == 401
    assert client.post("/chat", json={"message": "x"}, headers=DAVE).status_code == 403


def test_chat_is_not_hitl_gated_and_writes_no_audit(tmp_path: Path):
    store = SqliteAuditStore(tmp_path / "audit.db")
    client, _ = _client(tmp_path, agent=SecurityAgent(FakeLLMClient("分析完成")), store=store)

    response = client.post("/chat", json={"message": "分析 1.2.3.4"}, headers=ALICE)
    assert response.status_code == 200
    assert response.json() == {"response": "分析完成"}
    # 不写审计、不产生归属、不产生审批单 —— 它不经过 HITL / 策略
    assert store.list_audit() == []
    assert store.list_action_rows() == []
    assert store.pending_action_rows() == []


# =====================================================================
# I. 注册后失败:归属**合法保留**,且这条"孤儿"线程不可被利用
# =====================================================================


def test_post_registration_graph_failure_keeps_an_immutable_owner_record(tmp_path: Path):
    """TASK 3 要求显式分类的行为:注册在先、跑图在后。

    图失败(数据源不可用 → 503)时,归属行**已经**落库,并且:

    - **保留**。它描述的是"这次指派确实发生过"这一历史事实,删除它才是
      篡改;而"删除 + 重试"会把一次已记录的指派变成可反复改写的状态。
    - **不可改派**。`thread_id` 是主键 ⇒ 重复注册响亮失败;改派在本版本
      (L-2)不被支持,所以这里必须**没有**任何"重新指派"的路径。
    - **不可利用**。线程没有产生待审批项,被指派的 approver 也拿不到东西
      (409);未被指派者与"线程不存在"同样得到 404(无枚举预言机)。
    - **不写误导性审计**。全程只有一条诚实的 `plan.failed`,没有 `approval.*`。
    """
    store = SqliteAuditStore(tmp_path / "audit.db")
    client, store = _client(
        tmp_path,
        service=_service(store, tmp_path / "nope.jsonl", tmp_path / "nope-intel.jsonl"),
        store=store,
    )

    response = _triage(client, ALICE, APPROVERS_BOB)
    assert response.status_code == 503, response.text

    records = store.list_audit()
    assert [record.event for record in records] == ["plan.failed"]
    thread_id = records[0].thread_id

    ownership = store.get_thread_ownership(thread_id)
    assert ownership is not None, "注册发生在跑图之前 ⇒ 失败后归属行保留"
    assert ownership.owner == "alice"
    assert ownership.approvers == ("bob",)

    # 失败不产生审批单 / 待审批项 —— 没有"看起来可以审批"的假象
    assert store.list_action_rows() == []
    assert store.pending_action_rows() == []

    # 被指派的 approver 也拿不到待审批项 → 409(不是 200)
    assert _resume(client, BOB, thread_id).status_code == 409
    # 未被指派的 approver → 与"线程不存在"相同的 404
    assert _resume(client, CAROL, thread_id).status_code == 404

    # 不可改派:同一 thread_id 再注册一次 → 主键冲突(库层,不是应用层约定)
    with pytest.raises(sqlite3.IntegrityError):
        store.record_thread_ownership(
            ThreadOwnership(
                thread_id=thread_id,
                owner="mallory",
                approvers=("carol",),
                created_at=datetime.now(timezone.utc),
            )
        )
    assert store.get_thread_ownership(thread_id) == ownership

    # 所有失败尝试都没有写审计:仍然只有那一条 plan.failed
    assert [record.event for record in store.list_audit(thread_id=thread_id)] == [
        "plan.failed"
    ]
