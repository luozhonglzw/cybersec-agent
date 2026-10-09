"""Phase v0.3.0-A3-1:认证与端点准入的 HTTP 契约。

覆盖冻结验收契约里的 **N-1 / N-2 / N-9 / N-11 / N-13 / N-19**,外加
TASK 4 要求的错误与审计契约(401 不可区分、403 权限不足、认证失败**不写
审计行**)。

## 这些用例**故意**摘掉 conftest 的默认主体

根 `tests/conftest.py` 给既有套件装了一个默认主体的
`dependency_overrides`(让 100+ 处老用例不必逐个加 `Authorization` 头)。
本模块在构造完客户端后 `client.app.dependency_overrides.clear()`,把那个
缝摘掉 —— 于是这里走的是**真实认证路径**:真密钥环、真
`hmac.compare_digest`、真 401/403。

## 本模块与 A3-3 的分工

本模块只覆盖**认证与角色准入**(401 / 403 / 结构性路由覆盖)。A3-3 引入的
**对象级授权**(逐线程审批指派、禁止自审批、审计读取范围收敛)与**可信
actor** 由 `tests/test_api/test_object_authorization.py` 覆盖 —— 那里同样
摘掉 conftest 的默认主体,走真实认证路径。

因此本模块的最后两条用例只钉住"**DTO 层**的兼容性"(operator 字段仍在、
图仍读 `decision.operator`),不声称任何授权语义。

全部 hermetic:`audit.db` 由 `tmp_path` 现场生成;注入依赖使 lifespan 不
运行 —— 不需要 `.env`,也不需要真实 LLM。
"""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.llm import FakeLLMClient
from app.core.triage import UnknownThreadError
from app.security.auth import (
    AuthKeyEntry,
    AuthKeyring,
    require_principal,
)
from app.security.store import SqliteAuditStore

#: 测试专用原始密钥。**不是**真实凭据,不得复用。
RAW_ANALYST = "analyst-test-key-not-a-real-credential"
RAW_APPROVER = "approver-test-key-not-a-real-credential"
RAW_VIEWER = "viewer-test-key-not-a-real-credential"

SUBJECT_ANALYST = "alice"
SUBJECT_APPROVER = "bob"
SUBJECT_VIEWER = "carol"


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


#: 三个独立主体、三种角色 —— 也顺带证明"同一角色可由多个 subject 承担"
#: (冻结设计 §2.4:这是禁止自审批能落地的前提)。
KEYRING = AuthKeyring.from_entries(
    [
        AuthKeyEntry(sha256=_digest(RAW_ANALYST), subject=SUBJECT_ANALYST, role="analyst"),
        AuthKeyEntry(sha256=_digest(RAW_APPROVER), subject=SUBJECT_APPROVER, role="approver"),
        AuthKeyEntry(sha256=_digest(RAW_VIEWER), subject=SUBJECT_VIEWER, role="viewer"),
    ]
)


def _auth(raw: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {raw}"}


ANALYST = _auth(RAW_ANALYST)
APPROVER = _auth(RAW_APPROVER)
VIEWER = _auth(RAW_VIEWER)

#: 冻结设计 D-3:这四条文档路由保持公开。
PUBLIC_ROUTE_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}

#: 四条业务路由(全部要求认证)。
BUSINESS_ROUTES = ("/chat", "/triage", "/resume", "/audit/events")


# =====================================================================
# 测试替身:被调用即失败 —— 用来证明认证**短路在端点体之前**
# =====================================================================


class _TripwireAgent:
    """认证失败时端点体不得运行,因此本对象不得被触达。"""

    async def chat(self, *args, **kwargs):  # pragma: no cover - 触达即失败
        raise AssertionError("agent must not be reached when authentication fails")


class _TripwireService:
    async def triage(self, *args, **kwargs):  # pragma: no cover - 触达即失败
        raise AssertionError("triage must not be reached when authentication fails")

    async def resume(self, *args, **kwargs):  # pragma: no cover - 触达即失败
        raise AssertionError("resume must not be reached when authentication fails")


class _UnknownThreadService:
    """`/resume` 走到领域层后抛"未知线程" → 404(证明准入守卫确实放行了)。"""

    async def triage(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("triage not used here")

    async def resume(self, thread_id, *, status, operator, reason=None, request_id=None):
        raise UnknownThreadError("未知的 thread_id")


# =====================================================================
# 夹具
# =====================================================================


def _client(tmp_path: Path | None = None, **kwargs) -> TestClient:
    """构造走**真实认证**的客户端(摘掉 conftest 的默认主体)。"""
    if tmp_path is not None:
        kwargs.setdefault("audit_store", SqliteAuditStore(tmp_path / "audit.db"))
    app = create_app(**kwargs)
    # lifespan 不运行(注入了依赖),这里补上同一装配:密钥环挂到 app.state。
    app.state.auth_keyring = KEYRING
    client = TestClient(app)
    # 关键:摘掉根 conftest 装的默认主体,走真认证。
    client.app.dependency_overrides.clear()
    return client


def _guarded_client(tmp_path: Path) -> TestClient:
    """四个端点都能"成功"走完的客户端(除非认证先短路)。"""
    return _client(
        tmp_path,
        agent=_TripwireAgent(),
        triage_service=_TripwireService(),
    )


# =====================================================================
# A. 401:缺失 / 畸形 / 未知一律不可区分(N-1 / N-9 / N-13)
# =====================================================================


@pytest.mark.parametrize("path", BUSINESS_ROUTES)
def test_anonymous_request_is_rejected_on_every_business_route(tmp_path: Path, path: str):
    """N-1 / N-13:无凭据 → 四条业务路由全部 401。

    端点是"被认证短路"而不是"被端点体拒绝" —— tripwire 替身保证端点体
    一旦运行就会炸。
    """
    client = _guarded_client(tmp_path)
    method = "GET" if path == "/audit/events" else "POST"
    response = client.request(method, path, json={} if method == "POST" else None)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_missing_malformed_and_unknown_credentials_are_byte_identical(tmp_path: Path):
    """N-9:三种失败的**响应体逐字节相同**。

    任何差异都是"这把 key 存在吗"的枚举预言机。
    """
    client = _guarded_client(tmp_path)

    variants = [
        {},                                                     # 没给
        {"Authorization": "Bearer"},                            # 有方案没 token
        {"Authorization": "Bearer   "},                         # token 是空白
        {"Authorization": "Basic YWxpY2U6c2VjcmV0"},             # 方案不对
        {"Authorization": RAW_ANALYST},                          # 少了方案
        {"Authorization": f"Bearer {RAW_ANALYST[:-1]}"},         # 少了个字符
        {"Authorization": "Bearer unknown-key-not-configured"},  # 完全未知
    ]
    bodies = set()
    for headers in variants:
        response = client.post("/chat", json={"message": "x"}, headers=headers)
        assert response.status_code == 401, headers
        bodies.add(response.content)
    assert len(bodies) == 1, f"401 响应体必须逐字节相同,实测出现 {len(bodies)} 种"

    assert json.loads(bodies.pop()) == {"detail": "authentication required"}


def test_failed_authentication_writes_no_audit_row(tmp_path: Path):
    """认证失败**不写审计行**。

    否则一个未认证的调用方就能往 append-only 表里灌数据 —— 在不可变表上
    制造写放大。认证失败是**日志**事件,不是审计事件。
    """
    store = SqliteAuditStore(tmp_path / "audit.db")
    client = _client(
        tmp_path,
        agent=_TripwireAgent(),
        triage_service=_TripwireService(),
        audit_store=store,
    )

    for path in BUSINESS_ROUTES:
        method = "GET" if path == "/audit/events" else "POST"
        assert client.request(method, path, json={} if method == "POST" else None).status_code == 401
        assert client.request(
            method,
            path,
            json={} if method == "POST" else None,
            headers={"Authorization": "Bearer unknown-key-not-configured"},
        ).status_code == 401

    assert store.list_audit() == []
    assert store.pending_action_rows() == []
    assert store.list_action_rows() == []


# =====================================================================
# B. 403:已认证但角色不足(N-2 + 冻结矩阵)
# =====================================================================


@pytest.mark.parametrize("path", ["/chat", "/triage"])
def test_viewer_is_refused_on_the_operator_routes(tmp_path: Path, path: str):
    """N-2:`viewer` 只持只读审计权限,不是操作者 → 403。"""
    client = _guarded_client(tmp_path)
    response = client.post(path, json={"message": "x", "indicator": "1.2.3.4"}, headers=VIEWER)

    assert response.status_code == 403
    assert response.json() == {"detail": "insufficient role"}


@pytest.mark.parametrize("raw", [RAW_ANALYST, RAW_APPROVER])
def test_analyst_and_approver_may_chat(tmp_path: Path, raw: str):
    """正向:`analyst` 与 `approver` 都能进 `/chat`(approver 是 analyst 的超集)。"""
    client = _client(tmp_path, agent=_TripwireAgent())
    # tripwire 会炸 —— 但那是**端点体被触达**的证据,正是我们要的:
    # 说明认证与准入都放行了。
    with pytest.raises(AssertionError, match="agent must not be reached"):
        client.post("/chat", json={"message": "x"}, headers=_auth(raw))


def test_only_an_approver_may_reach_resume(tmp_path: Path):
    """`/resume` 的准入是 `approver` —— **角色**准入,不是对象授权。"""
    client = _client(tmp_path, triage_service=_UnknownThreadService())
    body = {"thread_id": "t-1", "status": "approved", "operator": "alice"}

    for headers in ({}, VIEWER, ANALYST):
        response = client.post("/resume", json=body, headers=headers)
        assert response.status_code in (401, 403)
    assert client.post("/resume", json=body, headers=VIEWER).json() == {
        "detail": "insufficient role"
    }
    assert client.post("/resume", json=body, headers=ANALYST).json() == {
        "detail": "insufficient role"
    }

    # approver 放行 → 到达领域层 → 未知线程 404(而不是 401/403)。
    response = client.post("/resume", json=body, headers=APPROVER)
    assert response.status_code == 404
    assert response.json() == {"detail": "未知的 thread_id"}


def test_only_a_viewer_may_read_the_audit_stream_unfiltered(tmp_path: Path):
    """A3-3:未给 `thread_id` 时只有 `viewer` 可以读整条审计流。

    `analyst` / `approver` 的读取范围必须**显式收敛到一条线程**(它们只能
    读自己拥有 / 被指派审批的线程),因此"不带范围的全量读取"对它们 403。
    按线程范围收敛的正例见 `tests/test_api/test_object_authorization.py`。
    """
    client = _client(tmp_path)
    assert client.get("/audit/events", headers=VIEWER).status_code == 200
    assert client.get("/audit/events", headers=VIEWER).json() == []

    for headers in (ANALYST, APPROVER):
        response = client.get("/audit/events", headers=headers)
        assert response.status_code == 403
        assert response.json() == {
            "detail": "an explicit thread scope is required for this role"
        }


# =====================================================================
# C. N-11:结构性路由覆盖(含正对照)
# =====================================================================


def _depends_on_authentication(dependant) -> bool:
    """依赖树里有没有认证缝 —— 递归,因为角色守卫依赖 `require_principal`。"""
    for sub in dependant.dependencies:
        if sub.call is require_principal:
            return True
        if _depends_on_authentication(sub):
            return True
    return False


def _unprotected_application_routes(app) -> set[str]:
    return {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute) and not _depends_on_authentication(route.dependant)
    }


def test_every_application_route_requires_authentication(tmp_path: Path):
    """N-11:枚举 `app.routes`,文档允许清单之外的**每一条**都挂认证。

    这是"以后有人加了新路由却忘了加认证"的护栏 —— 静态审阅发现不了,
    这条测试能。
    """
    app = create_app(
        agent=_TripwireAgent(),
        triage_service=_TripwireService(),
        audit_store=SqliteAuditStore(tmp_path / "audit.db"),
    )
    assert _unprotected_application_routes(app) == set()


def test_the_public_surface_is_exactly_the_documented_routes(tmp_path: Path):
    """D-3:未受保护的表面**恰好**是那四条文档路由,一条不多。"""
    app = create_app(
        agent=_TripwireAgent(),
        triage_service=_TripwireService(),
        audit_store=SqliteAuditStore(tmp_path / "audit.db"),
    )
    non_api = {route.path for route in app.routes if not isinstance(route, APIRoute)}
    assert non_api == PUBLIC_ROUTE_PATHS

    api_paths = {route.path for route in app.routes if isinstance(route, APIRoute)}
    assert api_paths == set(BUSINESS_ROUTES)


def test_the_route_coverage_check_has_teeth(tmp_path: Path):
    """正对照:故意加一条**没有认证**的路由,覆盖检查必须变红。

    没有这条,"全部受保护"可能只是因为检查器收不到任何路由(恒真)。
    """
    app = create_app(
        agent=_TripwireAgent(),
        triage_service=_TripwireService(),
        audit_store=SqliteAuditStore(tmp_path / "audit.db"),
    )
    assert _unprotected_application_routes(app) == set()

    @app.get("/__deliberately_unprotected_probe__")
    async def _probe():  # pragma: no cover - 只为让检查器变红
        return {}

    assert _unprotected_application_routes(app) == {"/__deliberately_unprotected_probe__"}


def test_documentation_routes_stay_public(tmp_path: Path):
    """D-3:文档路由不需要凭据(它们只描述契约,不暴露数据)。"""
    client = _client(tmp_path)
    for path in sorted(PUBLIC_ROUTE_PATHS):
        assert client.get(path).status_code == 200, path


# =====================================================================
# D. N-19:带上有效凭据后,领域语义与 v0.2.0 逐字相同
# =====================================================================


def test_authenticated_triage_without_a_service_returns_503(tmp_path: Path):
    client = _client(tmp_path, agent=_TripwireAgent())
    response = client.post(
        "/triage",
        json={"indicator": "1.2.3.4", "approvers": [SUBJECT_APPROVER]},
        headers=ANALYST,
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "triage service unavailable"}


def test_authenticated_chat_llm_failure_returns_502(tmp_path: Path):
    from app.core.agent import SecurityAgent

    client = _client(tmp_path, agent=SecurityAgent(FakeLLMClient(raise_error=True)))
    response = client.post("/chat", json={"message": "x"}, headers=ANALYST)
    assert response.status_code == 502
    assert response.json() == {"detail": "LLM service unavailable"}


def test_authenticated_audit_without_a_store_returns_503():
    """`tmp_path=None` ⇒ **不**注入 audit_store,才能验到 503 那条分支。"""
    client = _client(agent=_TripwireAgent())
    response = client.get("/audit/events", headers=VIEWER)
    assert response.status_code == 503
    assert response.json() == {"detail": "audit store unavailable"}


# =====================================================================
# E. DTO 层兼容性(A3-3 后 operator 字段仍在,但已非权威)
# =====================================================================


def test_resume_request_still_accepts_an_operator_field():
    """`operator` 字段**保留**(A3-3)。

    它已**不是**权威身份 —— `/resume` 的 actor 取自已认证主体的 subject。
    保留字段只是为了线上请求的**错误兼容性**(缺字段仍 422)。真正证明
    "伪造它无法改变 actor"的用例在 `test_object_authorization.py`。
    """
    from app.api.schemas import ResumeRequest

    assert "operator" in ResumeRequest.model_fields
    assert "verified_subject" not in ResumeRequest.model_fields


def test_graph_still_records_the_decision_operator():
    """结构性事实:审计 `actor` 仍取自 `decision.operator`(图未改动)。

    A3-3 改变的是**谁往 `decision.operator` 里填值**(HTTP 边界填已认证
    subject,而不是客户端自述值),图本身保持单一真相源不变。
    """
    source = (Path(__file__).resolve().parents[2] / "app" / "core" / "graph.py").read_text(
        encoding="utf-8"
    )
    assert "actor=decision.operator" in source
