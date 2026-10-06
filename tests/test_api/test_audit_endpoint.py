"""Phase 9.3-F:GET /audit/events —— 只读审计查询端点测试。

覆盖三层:

1. HTTP 契约
   参数集恰为 {thread_id, event, limit, order}(全是 query 参数,无 path 参数);
   响应是**裸列表** `list[AuditRecord]`(无信封);空结果 / 未知 thread_id
   一律 200 + `[]`;过滤 / 排序 / 分页语义;422 校验;503 错误契约。

2. 只读边界
   读审计不得产生新的审计写入(读不会再写一条审计),不得触达写路径 /
   策略 / 审批 / resume / 工具 / provider;未声明的 path-like 参数
   无法把读取重定向到别的库。

3. 结构性不变量(AST)
   HTTP 读 → `SqliteAuditStore.list_audit` → **参数绑定** SELECT;
   方向与 LIMIT 由固定程序逻辑决定,不拼接调用方传入的 ORDER BY / 列名。

全部 hermetic:audit.db 由 tmp_path 现场生成;只注入 audit_store,
lifespan 不运行 —— 不需要 .env,也不需要真实 LLM。
"""
import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.security.store as store_module
from app.api.main import create_app
from app.core.agent import SecurityAgent
from app.core.llm import FakeLLMClient
from app.schemas.audit import AuditRecord
from app.security.audit import build_audit_record
from app.security.store import SqliteAuditStore

BASE_TS = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

AUDIT_EVENT_VALUES = {
    "plan.created",
    "plan.failed",
    "policy.evaluated",
    "approval.requested",
    "approval.decided",
    "approval.timeout",
}


# ---------- 夹具 ----------

@pytest.fixture
def store(tmp_path: Path) -> SqliteAuditStore:
    """每个用例一个独立的临时库,用例之间零共享。"""
    return SqliteAuditStore(tmp_path / "audit.db")


@pytest.fixture
def client(store: SqliteAuditStore) -> TestClient:
    """只注入 audit_store → lifespan 不运行(无需 .env / 真实 LLM)。"""
    return TestClient(create_app(audit_store=store))


def _seed(
    store: SqliteAuditStore,
    count: int,
    *,
    thread_id: str = "t-1",
    event: str = "plan.created",
) -> list[str]:
    """按 ts 递增写入 count 条审计;reason 携带序号,便于断言顺序。

    返回写入记录的 id 列表(升序)。
    """
    ids: list[str] = []
    for i in range(count):
        record = build_audit_record(
            event,
            thread_id=thread_id,
            ts=BASE_TS + timedelta(seconds=i),
            reason=f"r{i:03d}",
        )
        store.append_audit(record)
        ids.append(record.id)
    return ids


def _reasons(resp) -> list[str]:
    return [row["reason"] for row in resp.json()]


def _audit_get_spec(client: TestClient) -> dict:
    spec = client.get("/openapi.json").json()
    return spec["paths"]["/audit/events"]["get"]


def _declared_enum(param: dict) -> set:
    """从(可能被 `anyOf` 包裹的)OpenAPI 参数 schema 里取出 enum 值集。"""
    schema = param["schema"]
    for branch in (schema, *schema.get("anyOf", [])):
        if "enum" in branch:
            return set(branch["enum"])
    raise AssertionError(f"参数未声明 enum:{schema}")


def _function_def(module, name: str):
    """从模块源码里取出指定函数(含嵌套函数)的 AST 节点。"""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"未找到函数定义:{name}")


# =====================================================================
# A. HTTP 契约:参数集 / 响应形状 / 无信封
# =====================================================================

def test_openapi_documents_audit_events_endpoint(client):
    spec = client.get("/openapi.json").json()["paths"]
    assert "/audit/events" in spec
    assert "get" in spec["/audit/events"]


def test_query_contract_exposes_exactly_four_declared_parameters(client):
    """查询契约恰为四个**声明式**参数 —— 多一个即接口变更。"""
    params = _audit_get_spec(client)["parameters"]
    names = {p["name"] for p in params}
    assert names == {"thread_id", "event", "limit", "order"}
    # 全部是 query 参数:契约里**没有** path 参数。
    assert {p["in"] for p in params} == {"query"}


def test_endpoint_declares_no_path_parameters(client):
    """路径边界:OpenAPI 里没有 `in: path` 的模板参数。"""
    spec = _audit_get_spec(client)
    assert not [p for p in spec["parameters"] if p["in"] == "path"]
    assert "{" not in "/audit/events"


def test_event_parameter_reuses_closed_audit_event_enum(client):
    """event 必须复用封闭 Literal(AuditEvent),不是自由字符串。"""
    params = {p["name"]: p for p in _audit_get_spec(client)["parameters"]}
    assert _declared_enum(params["event"]) == AUDIT_EVENT_VALUES
    assert params["event"]["required"] is False


def test_order_parameter_is_closed_enum_with_desc_default(client):
    params = {p["name"]: p for p in _audit_get_spec(client)["parameters"]}
    schema = params["order"]["schema"]
    assert set(schema["enum"]) == {"desc", "asc"}
    assert schema.get("default") == "desc"


def test_limit_parameter_bounds_and_default(client):
    params = {p["name"]: p for p in _audit_get_spec(client)["parameters"]}
    schema = params["limit"]["schema"]
    assert schema["minimum"] == 1
    assert schema["maximum"] == 200
    assert schema.get("default") == 50


def test_response_schema_is_bare_array_of_audit_record(client):
    """无信封:200 响应是数组,元素是 AuditRecord。"""
    schema = _audit_get_spec(client)["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert schema["type"] == "array"
    assert schema["items"]["$ref"].endswith("/AuditRecord")


def test_empty_store_returns_200_empty_list(client):
    resp = client.get("/audit/events")
    assert resp.status_code == 200
    assert resp.json() == []


def test_unknown_thread_returns_200_empty_list(client, store):
    """未知 thread_id 与"无数据"不区分 —— 避免用响应差异探测库里有什么。"""
    _seed(store, 2)
    resp = client.get("/audit/events", params={"thread_id": "never-existed"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_response_body_is_bare_array_not_object(store, client):
    _seed(store, 1)
    body = client.get("/audit/events").json()
    assert isinstance(body, list)
    assert not isinstance(body, dict)


def test_record_shape_is_exactly_audit_record_fields(store, client):
    """响应元素的字段集恰等于 AuditRecord 的字段集。"""
    _seed(store, 1)
    row = client.get("/audit/events").json()[0]
    assert set(row) == set(AuditRecord.model_fields)


# =====================================================================
# B. 排序 / 分页
# =====================================================================

def test_default_order_is_descending(store, client):
    _seed(store, 3)
    assert _reasons(client.get("/audit/events")) == ["r002", "r001", "r000"]


def test_explicit_desc_matches_default(store, client):
    _seed(store, 3)
    default = _reasons(client.get("/audit/events"))
    explicit = _reasons(client.get("/audit/events", params={"order": "desc"}))
    assert default == explicit == ["r002", "r001", "r000"]


def test_order_asc_returns_oldest_first(store, client):
    _seed(store, 3)
    assert _reasons(client.get("/audit/events", params={"order": "asc"})) == [
        "r000", "r001", "r002",
    ]


def test_limit_bounds_result_size_to_newest(store, client):
    _seed(store, 5)
    assert _reasons(client.get("/audit/events", params={"limit": 2})) == [
        "r004", "r003",
    ]


def test_limit_default_is_fifty(store, client):
    _seed(store, 60)
    assert len(client.get("/audit/events").json()) == 50


def test_limit_accepts_boundary_values(store, client):
    _seed(store, 3)
    assert len(client.get("/audit/events", params={"limit": 1}).json()) == 1
    assert len(client.get("/audit/events", params={"limit": 200}).json()) == 3


def test_asc_with_limit_returns_oldest(store, client):
    """方向与 LIMIT 组合:asc + limit 取最旧的若干条。"""
    _seed(store, 5)
    assert _reasons(
        client.get("/audit/events", params={"order": "asc", "limit": 2})
    ) == ["r000", "r001"]


# =====================================================================
# C. 过滤
# =====================================================================

def test_filter_by_thread_id(store, client):
    _seed(store, 2, thread_id="t-1")
    _seed(store, 3, thread_id="t-2")
    body = client.get("/audit/events", params={"thread_id": "t-1"}).json()
    assert len(body) == 2
    assert {row["thread_id"] for row in body} == {"t-1"}


def test_filter_by_event(store, client):
    _seed(store, 2, event="plan.created")
    _seed(store, 1, event="approval.decided")
    body = client.get("/audit/events", params={"event": "approval.decided"}).json()
    assert len(body) == 1
    assert body[0]["event"] == "approval.decided"


def test_filters_combine(store, client):
    _seed(store, 2, thread_id="t-1", event="plan.created")
    _seed(store, 1, thread_id="t-2", event="plan.created")
    _seed(store, 1, thread_id="t-1", event="policy.evaluated")
    body = client.get(
        "/audit/events", params={"thread_id": "t-1", "event": "plan.created"}
    ).json()
    assert len(body) == 2


def test_no_filter_returns_all(store, client):
    _seed(store, 4)
    assert len(client.get("/audit/events").json()) == 4


# =====================================================================
# D. 校验(422,保持 FastAPI 默认)
# =====================================================================

def test_invalid_event_returns_422(client):
    assert client.get("/audit/events", params={"event": "bogus"}).status_code == 422


def test_invalid_order_returns_422(client):
    assert client.get("/audit/events", params={"order": "sideways"}).status_code == 422


@pytest.mark.parametrize("limit", [0, -1, 201, 1000])
def test_limit_out_of_range_returns_422(client, limit):
    assert client.get(
        "/audit/events", params={"limit": limit}
    ).status_code == 422


def test_non_integer_limit_returns_422(client):
    assert client.get(
        "/audit/events", params={"limit": "abc"}
    ).status_code == 422


def test_valid_event_values_all_accepted(client):
    for value in sorted(AUDIT_EVENT_VALUES):
        assert client.get(
            "/audit/events", params={"event": value}
        ).status_code == 200


# =====================================================================
# E. 未暴露面 / 路径边界
# =====================================================================

_FORBIDDEN_QUERY_NAMES = (
    "incident_id",
    "approval_id",
    "request_id",
    "interrupt_id",
    "actor",
    "outcome",
    "plan_digest",
    "db_path",
    "database_path",
    "sqlite_path",
    "file_path",
    "offset",
    "cursor",
)


@pytest.mark.parametrize("name", _FORBIDDEN_QUERY_NAMES)
def test_forbidden_query_parameters_are_not_declared(client, name):
    """这些输入刻意不出现在契约里(未声明 = 不可作为过滤器)。"""
    params = {p["name"] for p in _audit_get_spec(client)["parameters"]}
    assert name not in params


def test_undeclared_filter_parameter_has_no_effect(store, client):
    """未声明参数被忽略:传 incident_id 不改变结果集(不是过滤器)。"""
    _seed(store, 3, thread_id="t-1")
    baseline = client.get("/audit/events").json()
    with_extra = client.get(
        "/audit/events", params={"incident_id": "whatever", "offset": 2}
    ).json()
    assert with_extra == baseline


def test_path_like_query_parameters_cannot_redirect_database(
    store, client, tmp_path: Path
):
    """路径边界:未声明的 path-like 参数无法把读取重定向到别的库。

    查询契约里没有路径参数,这些键只会被忽略 —— 读取仍来自注入的 store,
    且不会在参数指向的位置创建任何文件。
    """
    _seed(store, 2)
    evil = tmp_path / "evil.db"
    resp = client.get("/audit/events", params={
        "db_path": str(evil),
        "database_path": str(evil),
        "sqlite_path": str(evil),
        "file_path": str(evil),
    })
    assert resp.status_code == 200
    assert len(resp.json()) == 2
    assert not evil.exists()


def test_endpoint_declares_no_security_scheme(client):
    """**无认证**:端点不声明任何安全方案。

    它是只读的,但**不是** "authorized endpoint" —— 不得暗示调用方已通过
    身份校验或只有审计员可见。
    """
    spec = client.get("/openapi.json").json()
    assert _audit_get_spec(client).get("security", []) == []
    assert "securitySchemes" not in spec.get("components", {})


# =====================================================================
# F. 503 错误契约
# =====================================================================

def test_store_read_failure_returns_503_fixed_detail(store, client, monkeypatch):
    """sqlite3 读失败 → 503 + 固定文案,绝不回传异常文本 / 路径 / traceback。"""

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError(
            "unable to open database file: /srv/secret/audit.db"
        )

    monkeypatch.setattr(store, "list_audit", boom)
    resp = client.get("/audit/events")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "audit store unavailable"}
    assert "secret" not in resp.text
    assert "OperationalError" not in resp.text
    assert "Traceback" not in resp.text
    assert "/" not in resp.text and "\\" not in resp.text


def test_row_decode_failure_returns_503(store, client, monkeypatch):
    """持久化行解码/校验失败(ValueError 家族)→ 503,而不是 500。"""

    def boom(*args, **kwargs):
        raise ValueError("bad persisted row")

    monkeypatch.setattr(store, "list_audit", boom)
    resp = client.get("/audit/events")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "audit store unavailable"}


@pytest.mark.parametrize("bad_ts,bad_event", [
    ("not-a-timestamp", "plan.created"),  # ts 解码失败(datetime.fromisoformat)
    (BASE_TS.isoformat(), "bogus.event"),  # event 校验失败(pydantic ValidationError)
])
def test_corrupt_persisted_row_returns_503(
    store, client, tmp_path: Path, bad_ts: str, bad_event: str
):
    """端到端:库里存在解码不出来的行 → 503,而不是带 traceback 的 500。"""
    store.append_audit(build_audit_record("plan.created", ts=BASE_TS))

    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        conn.execute(
            "INSERT INTO audit_logs"
            " (id, ts, actor, event, incident_id, thread_id, interrupt_id,"
            "  outcome, reason, plan_digest, detail_json)"
            " VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?)",
            ("corrupt-row", bad_ts, "system", bad_event, "{}"),
        )
        conn.commit()
    finally:
        conn.close()

    resp = client.get("/audit/events")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "audit store unavailable"}


def test_audit_endpoint_unavailable_without_store_returns_503():
    """只注入了 agent(测试场景)→ /audit/events 给明确 503,不是 500。"""
    client = TestClient(create_app(agent=SecurityAgent(FakeLLMClient())))
    resp = client.get("/audit/events")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "audit store unavailable"}


# =====================================================================
# G. 只读边界:读不产生写
# =====================================================================

def test_audit_read_does_not_append_audit(store, client):
    """读审计**不得**再写一条审计。"""
    _seed(store, 3)
    before = len(store.list_audit())
    assert client.get("/audit/events").status_code == 200
    assert client.get("/audit/events", params={"order": "asc"}).status_code == 200
    assert len(store.list_audit()) == before == 3


def test_audit_read_is_idempotent(store, client):
    _seed(store, 3)
    first = client.get("/audit/events").json()
    second = client.get("/audit/events").json()
    assert first == second


# =====================================================================
# H. 结构性不变量(AST)
# =====================================================================

_FORBIDDEN_CALL_NAMES = frozenset({
    # store 写 / 派生方法 —— 读路径不得调用
    "append_audit",
    "record_incident",
    "record_action_request",
    "pending_action_rows",
    "list_action_rows",
    "get_incident",
    "get_approval_request",
    # 策略 / 图 / 服务 / 工具 / provider —— 读路径不得触达
    "evaluate_policy",
    "triage",
    "resume",
    "invoke",
    "ainvoke",
    "chat",
    "bind_tools",
    "create_agent_graph",
})


def test_endpoint_calls_list_audit_and_nothing_forbidden():
    """HTTP 读只调用 store.list_audit,不触达任何写 / 策略 / 工具 / provider。"""
    import app.api.main as api_main

    fn = _function_def(api_main, "list_audit_events")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}

    assert "list_audit" in attrs, "端点必须经 SqliteAuditStore.list_audit 读"
    assert not (attrs & _FORBIDDEN_CALL_NAMES), attrs & _FORBIDDEN_CALL_NAMES
    assert not (names & _FORBIDDEN_CALL_NAMES), names & _FORBIDDEN_CALL_NAMES


def test_endpoint_reads_store_via_require_store():
    """取 store 走 _require_store(缺失 → 503),不直接摸 app.state。"""
    import app.api.main as api_main

    fn = _function_def(api_main, "list_audit_events")
    called = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_require_store" in called


def test_store_list_audit_uses_parameter_binding_not_string_interpolation():
    """store.list_audit 必须走参数绑定,不得用 f-string / % / format 拼 SQL。"""
    fn = _function_def(store_module, "list_audit")
    assert not any(isinstance(n, ast.JoinedStr) for n in ast.walk(fn))
    assert not any(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
        for n in ast.walk(fn)
    )
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "format"
        for n in ast.walk(fn)
    )

    executes = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "execute"
    ]
    assert executes, "list_audit 必须通过 conn.execute 执行 SQL"
    for call in executes:
        assert len(call.args) == 2, "execute 必须以 (sql, params) 参数绑定方式调用"


def test_store_list_audit_sql_literals_are_fixed():
    """方向 / LIMIT 由固定字面量决定,占位符只出现 `?`。"""
    fn = _function_def(store_module, "list_audit")
    strings = [
        n.value
        for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    joined = "\n".join(strings)
    assert "ORDER BY ts, rowid" in joined          # 默认升序(逐字保留)
    assert "ORDER BY ts DESC, rowid DESC" in joined  # 降序分支
    assert "LIMIT ?" in joined                     # LIMIT 在 SQL 层,参数绑定


def test_response_reuses_existing_audit_record_schema(client):
    """复用既有 AuditRecord schema —— 本阶段不引入新的响应模型。"""
    spec = client.get("/openapi.json").json()
    ref = _audit_get_spec(client)["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["items"]["$ref"]
    assert ref == "#/components/schemas/AuditRecord"
    assert "AuditRecord" in spec["components"]["schemas"]
    # 响应模型里不得出现存储/传输层内部标识。
    assert set(AuditRecord.model_fields) == {
        "id", "ts", "actor", "event", "incident_id", "thread_id",
        "interrupt_id", "outcome", "reason", "plan_digest", "detail",
    }
