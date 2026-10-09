"""Phase v0.2.0-M1c:审计后端选择(配置 + 组合根)的离线测试。

覆盖三层,全部**离线**且 hermetic —— 不需要 PostgreSQL、不需要 .env、
不需要真实 LLM:

1. `Settings` 契约
   默认后端是 sqlite;`postgres` 缺 DSN 时在**构造期**失败;
   DSN 用 SecretStr,repr / str / 校验错误文本里都不得出现口令。

2. `build_audit_store` / `close_audit_store`
   按 `audit_backend` 选择后端,**不自动回退**;`PostgresAuditStore`
   的构造器不做 I/O(不连接、不建池);`close()` 只对持有连接池的后端
   生效,且**不是** `AuditStore` 契约的第 9 个方法。

3. 结构性不变量(AST / 源码扫描)
   全项目只有组合根读 `audit_backend`;图节点与 triage 服务里没有任何
   backend 分支;`lifespan` 拥有 store 的创建与关闭(含部分启动失败路径)。

真实 PostgreSQL 上的 API 行为由 `tests/test_postgres/test_app_integration.py`
覆盖(默认套件不收集)。
"""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

import app.api.main as api_main
from app.api.main import build_audit_store, close_audit_store, create_app
from app.core.config import Settings, get_settings
from app.core.graph import HitlConfig
from app.core.llm import FakeLLMClient
from app.core.triage import TriageService
from app.security.store import SqliteAuditStore
from app.security.store_postgres import PostgresAuditStore
from app.security.store_protocol import AuditStore
from tests.conftest import VIEWER_HEADERS

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"

#: 语法合法、但不会被连接的 DSN —— 这些用例刻意**不**碰真实数据库。
#: 口令是明显的假值,且必须在任何输出里都被脱敏。
FAKE_DSN = "postgresql://cybersec_app:fake_password_local_only@127.0.0.1:1/nope"
FAKE_PASSWORD = "fake_password_local_only"

#: 组合根允许读后端配置的两个文件 —— 多一个都算"分支渗透到业务层"。
BACKEND_CONFIG_FILES = frozenset({"api/main.py", "core/config.py"})


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """`get_settings` 是 lru_cache 的:用例改环境变量前后都必须清缓存。

    不清的话,前一个用例缓存的 Settings 会渗进后一个用例,
    让"环境变量决定后端"这类断言变成假的。
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(**kwargs) -> Settings:
    """构造 Settings,不读 .env(避免本机 .env 影响断言)。"""
    base = {"llm_model": "m", "llm_api_key": "k", "_env_file": None}
    base.update(kwargs)
    return Settings(**base)


# =====================================================================
# A. Settings 契约
# =====================================================================


def test_default_backend_is_sqlite():
    """不设任何环境变量时后端是 sqlite —— 与 v0.1.0 行为一致。"""
    s = _settings()
    assert s.audit_backend == "sqlite"
    assert s.audit_postgres_dsn is None


def test_unknown_backend_value_is_rejected():
    """取值集是封闭的:没有 auto / fallback / pg 之类的别名。"""
    with pytest.raises(ValidationError):
        _settings(audit_backend="mysql")
    with pytest.raises(ValidationError):
        _settings(audit_backend="auto")


def test_postgres_without_dsn_is_rejected_at_construction():
    """显式选 postgres 却不给 DSN → 构造期就失败,而不是运行到一半才炸。"""
    with pytest.raises(ValidationError) as info:
        _settings(audit_backend="postgres")
    assert "AUDIT_POSTGRES_DSN" in str(info.value)


def test_postgres_with_blank_dsn_is_rejected():
    """空串 / 纯空白与"没给"同义。"""
    for blank in ("", "   ", "\n"):
        with pytest.raises(ValidationError):
            _settings(audit_backend="postgres", audit_postgres_dsn=blank)


def test_dsn_is_a_secret_and_never_shown_in_repr_or_str():
    """DSN 里带口令 → repr / str 都必须是 **********。"""
    s = _settings(audit_backend="postgres", audit_postgres_dsn=FAKE_DSN)
    assert isinstance(s.audit_postgres_dsn, SecretStr)
    assert s.audit_postgres_dsn.get_secret_value() == FAKE_DSN
    for rendered in (repr(s), str(s)):
        assert FAKE_PASSWORD not in rendered
        assert "**********" in rendered


def test_validation_error_never_echoes_the_dsn():
    """校验失败的文本里不得出现口令(错误信息会进日志)。"""
    with pytest.raises(ValidationError) as info:
        Settings(
            llm_model="m",
            llm_api_key="k",
            _env_file=None,
            audit_backend="postgres",
            audit_postgres_dsn="",
        )
    assert FAKE_PASSWORD not in str(info.value)


def test_env_vars_select_the_backend(monkeypatch):
    """环境变量优先级最高(部署就是靠它切后端)。"""
    monkeypatch.setenv("AUDIT_BACKEND", "postgres")
    monkeypatch.setenv("AUDIT_POSTGRES_DSN", FAKE_DSN)
    s = Settings(llm_model="m", llm_api_key="k", _env_file=None)
    assert s.audit_backend == "postgres"
    assert s.audit_postgres_dsn.get_secret_value() == FAKE_DSN


def test_sqlite_configuration_is_untouched(monkeypatch):
    """既有 SQLite 配置项语义不变(默认路径仍是 data/audit.db)。"""
    monkeypatch.delenv("AUDIT_DB_PATH", raising=False)
    assert _settings().audit_db_path == Path("data/audit.db")
    monkeypatch.setenv("AUDIT_DB_PATH", "/tmp/x/audit.db")
    assert _settings().audit_db_path == Path("/tmp/x/audit.db")


# =====================================================================
# B. build_audit_store / close_audit_store
# =====================================================================


def test_sqlite_backend_builds_a_sqlite_store(tmp_path: Path):
    store = build_audit_store(_settings(audit_db_path=tmp_path / "audit.db"))
    assert isinstance(store, SqliteAuditStore)
    assert (tmp_path / "audit.db").exists()


def test_postgres_backend_builds_a_postgres_store_without_touching_the_network():
    """装配期**不建池、不连接** —— 数据库不可用的失败不提前到进程启动。"""
    store = build_audit_store(
        _settings(audit_backend="postgres", audit_postgres_dsn=FAKE_DSN)
    )
    assert isinstance(store, PostgresAuditStore)
    assert store._pool is None, "构造期不得开池"
    assert FAKE_PASSWORD not in repr(store)
    close_audit_store(store)


def test_postgres_without_dsn_raises_instead_of_falling_back_to_sqlite(tmp_path: Path):
    """**不回退**是硬约束:宁可不启动,也不把审计悄悄写进另一个库。

    这里刻意绕过 `Settings`(用 stub 对象)来验证组合根自身的第二道闸门。
    """
    stub = SimpleNamespace(
        audit_backend="postgres",
        audit_postgres_dsn=None,
        audit_db_path=tmp_path / "audit.db",
    )
    with pytest.raises(ValueError, match="AUDIT_POSTGRES_DSN"):
        build_audit_store(stub)
    assert not (tmp_path / "audit.db").exists(), "不得留下 SQLite 库"


def test_unparseable_dsn_fails_clearly_without_echoing_it(tmp_path: Path):
    """不可解析的连接串 → 启动期清晰失败,且错误链里没有原始文本。

    psycopg 的解析错误消息会内嵌连接串片段,所以组合根刻意 `from None`:
    `__cause__` 必须是 None,否则口令会随 traceback 进入日志。
    """
    stub = SimpleNamespace(
        audit_backend="postgres",
        audit_postgres_dsn=SecretStr(f"garbage-{FAKE_PASSWORD}"),
        audit_db_path=tmp_path / "audit.db",
    )
    with pytest.raises(ValueError) as info:
        build_audit_store(stub)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True
    assert FAKE_PASSWORD not in str(info.value)
    assert not (tmp_path / "audit.db").exists()


def test_close_is_a_noop_for_sqlite(tmp_path: Path):
    """SQLite 后端没有长期资源 → close 不抛错,库仍可用。"""
    store = build_audit_store(_settings(audit_db_path=tmp_path / "audit.db"))
    close_audit_store(store)
    close_audit_store(store)
    assert store.list_audit() == []


def test_close_closes_the_postgres_pool_and_is_idempotent():
    store = build_audit_store(
        _settings(audit_backend="postgres", audit_postgres_dsn=FAKE_DSN)
    )
    assert store._closed is False
    close_audit_store(store)
    assert store._closed is True
    close_audit_store(store)  # 幂等
    assert store._closed is True


def test_close_is_not_a_contract_method():
    """TASK 4 约束:`close()` 不得进入 `AuditStore` 契约。

    只有 PostgreSQL 后端持有可关闭的资源;把 close 塞进契约等于让契约
    描述实现细节,并强迫 SQLite 实现一个空方法。

    v0.3.0-A3-2 把方法集从 8 扩到 10(新增两个归属方法),这里同步跟上 ——
    断言仍是**集合相等**,`close` 依然必须不在其中。
    """
    protocol_methods = {
        name
        for name in vars(AuditStore)
        if not name.startswith("_") and callable(getattr(AuditStore, name))
    }
    assert protocol_methods == {
        "record_incident",
        "record_action_request",
        "append_audit",
        "record_thread_ownership",
        "get_incident",
        "get_approval_request",
        "get_thread_ownership",
        "list_action_rows",
        "pending_action_rows",
        "list_audit",
    }
    assert "close" not in protocol_methods
    assert not hasattr(SqliteAuditStore, "close")


def test_both_backends_satisfy_the_protocol(tmp_path: Path):
    sqlite_store = SqliteAuditStore(tmp_path / "audit.db")
    pg_store = build_audit_store(
        _settings(audit_backend="postgres", audit_postgres_dsn=FAKE_DSN)
    )
    try:
        assert isinstance(sqlite_store, AuditStore)
        assert isinstance(pg_store, AuditStore)
    finally:
        close_audit_store(pg_store)


def test_create_app_accepts_the_protocol_not_a_concrete_backend():
    """注入点是契约类型 —— 组合根不把具体后端写进 API 签名。"""
    params = inspect.signature(create_app).parameters
    assert params["audit_store"].annotation == AuditStore | None
    assert inspect.signature(api_main._require_store).return_annotation is AuditStore


# =====================================================================
# C. 结构性不变量:后端选择只发生在组合根
# =====================================================================


def _python_files() -> list[Path]:
    return sorted(APP_ROOT.rglob("*.py"))


def test_only_the_composition_root_reads_the_backend_setting():
    """全项目只有 config.py(定义)与 main.py(选择)提到 `audit_backend`。

    这条护栏防的是"后端分支渗透":一旦某个图节点或端点里出现
    `settings.audit_backend == ...`,后端差异就从组合根漏进了业务层。
    """
    hits = {
        path.relative_to(APP_ROOT).as_posix()
        for path in _python_files()
        if "audit_backend" in path.read_text(encoding="utf-8")
        or "audit_postgres_dsn" in path.read_text(encoding="utf-8")
    }
    assert hits == BACKEND_CONFIG_FILES, hits


def test_graph_and_triage_contain_no_backend_branch():
    """图与 triage 服务对后端无感知 —— 只依赖 `AuditStore` 契约。

    检查的是**分支符号**,不是随便出现的单词:graph.py 的文档里会提到
    "langgraph-checkpoint-sqlite"(那是 checkpoint 的依赖说明,与本后端
    选择无关),所以不能拿 `"sqlite" in source` 当判据 —— 那既会误报,
    也证明不了"没有分支"。
    """
    forbidden = (
        "audit_backend",
        "audit_postgres_dsn",
        "build_audit_store",
        "close_audit_store",
        "PostgresAuditStore",
        "SqliteAuditStore",
    )
    for rel in ("core/graph.py", "core/triage.py"):
        source = (APP_ROOT / rel).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{rel} 出现后端分支符号 {token}"


def test_hitl_config_and_triage_service_depend_on_the_protocol():
    """两个 HITL 依赖点的类型注解必须是契约,不是具体后端。"""
    from app.core.triage import TriageService

    assert HitlConfig.__dataclass_fields__["audit_store"].type is AuditStore
    hints = inspect.get_annotations(TriageService.__init__)
    assert hints["store"] is AuditStore


def _find_func(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"未找到函数定义:{name}")


def test_async_nodes_call_the_store_synchronously_by_design():
    """TASK 5 的结构性事实:async 节点里**直接同步调用** store 方法。

    这不是 M1c 引入的:SQLite 版从 Phase 8.3 起就是同一形态。实测两个后端
    都会阻塞事件循环,而 SQLite 阻塞得更严重(见报告 §3)。本用例把
    "当前形态是有意为之、不是遗漏"钉死:

    - 若有人顺手把 `append_audit(...)` 改成 `await ...`,这里立刻变红 ——
      因为 `AuditStore` 契约是**同步**的,`await` 一个非 awaitable 会直接
      TypeError,而契约改造属于另一个阶段(需要 async 后端实现)。
    - 若有人给节点加了任何 `await`,也立刻变红 —— 节点体内的 await 会改变
      interrupt 重放语义,那属于图执行改造,超出本阶段授权。
    """
    tree = ast.parse((APP_ROOT / "core" / "graph.py").read_text(encoding="utf-8"))
    for name in ("plan_node", "policy_gate_node", "human_approval_node"):
        fn = _find_func(tree, name)
        assert isinstance(fn, ast.AsyncFunctionDef), name
        assert not any(isinstance(node, ast.Await) for node in ast.walk(fn)), (
            f"{name} 里出现了 await —— 节点内的 await 会改变 interrupt 重放语义"
        )
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append_audit"
            for node in ast.walk(fn)
        ), f"{name} 应当同步调用 append_audit"


def test_lifespan_creates_and_closes_the_store():
    """资源归属护栏:`lifespan` 必须自己建 store,并在 finally 里关掉它。

    这样"每请求建池"与"退出不关池"两种误用都无法悄悄出现。
    """
    source = (APP_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    lifespan = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    called = {
        node.func.id
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_audit_store" in called
    assert "close_audit_store" in called

    finally_blocks = [node for node in ast.walk(lifespan) if isinstance(node, ast.Try)]
    assert finally_blocks, "lifespan 必须用 try/finally 覆盖部分启动失败"
    assert any(
        "close_audit_store" in ast.dump(handler)
        for block in finally_blocks
        for handler in block.finalbody
    ), "关闭动作必须在 finally 里"


def test_no_module_creates_a_postgres_store_at_import_time():
    """装配只发生在组合根的 lifespan 里,不在模块顶层。

    模块顶层装配会让"导入 app"产生副作用(建池),也会让测试无法替换后端。
    """
    source = (APP_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:  # 只看模块顶层语句
        assert not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"build_audit_store", "PostgresAuditStore"}
        ), "模块顶层不得装配审计后端"


# =====================================================================
# D. lifespan 装配(离线:不连接数据库)
# =====================================================================


@pytest.fixture
def env_llm(monkeypatch):
    """lifespan 会构造 LLMClient(需要 .env),测试里换成 Fake。"""
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(api_main, "LLMClient", lambda *a, **k: FakeLLMClient("ok"))
    yield monkeypatch


def test_lifespan_composes_sqlite_by_default(env_llm, tmp_path: Path):
    env_llm.setenv("AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    env_llm.delenv("AUDIT_BACKEND", raising=False)
    with TestClient(create_app()) as client:
        store = client.app.state.audit_store
        assert isinstance(store, SqliteAuditStore)
        # A3-3 起 analyst / approver 必须给显式 thread 范围才能读审计,
        # 未过滤读取只有 viewer 可以 —— 这里只验"端点可用",用 viewer 身份。
        assert client.get("/audit/events", headers=VIEWER_HEADERS).status_code == 200


def test_lifespan_composes_postgres_when_explicitly_selected(env_llm, tmp_path: Path):
    sqlite_path = tmp_path / "must_not_be_created.db"
    env_llm.setenv("AUDIT_DB_PATH", str(sqlite_path))
    env_llm.setenv("AUDIT_BACKEND", "postgres")
    env_llm.setenv("AUDIT_POSTGRES_DSN", FAKE_DSN)
    with TestClient(create_app()) as client:
        store = client.app.state.audit_store
        assert isinstance(store, PostgresAuditStore)
        assert store._pool is None, "装配期不得开池(也不得连库)"
        assert not sqlite_path.exists(), "选了 postgres 就不得顺手建 SQLite 库"


def test_lifespan_closes_the_postgres_store_on_shutdown(env_llm, tmp_path: Path):
    env_llm.setenv("AUDIT_DB_PATH", str(tmp_path / "must_not_be_created.db"))
    env_llm.setenv("AUDIT_BACKEND", "postgres")
    env_llm.setenv("AUDIT_POSTGRES_DSN", FAKE_DSN)
    with TestClient(create_app()) as client:
        store = client.app.state.audit_store
        assert store._closed is False
    assert store._closed is True, "退出 lifespan 必须关闭连接池"


def test_lifespan_closes_the_store_when_partial_startup_fails(env_llm, tmp_path: Path):
    """部分启动失败:store 已建好,但后续装配抛错 → 仍必须关池。"""
    captured: list = []
    real_build = api_main.build_audit_store

    def _record(settings):
        store = real_build(settings)
        captured.append(store)
        return store

    env_llm.setenv("AUDIT_DB_PATH", str(tmp_path / "must_not_be_created.db"))
    env_llm.setenv("AUDIT_BACKEND", "postgres")
    env_llm.setenv("AUDIT_POSTGRES_DSN", FAKE_DSN)
    env_llm.setattr(api_main, "build_audit_store", _record)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated partial-startup failure")

    env_llm.setattr(api_main, "LLMClient", _boom)

    with pytest.raises(RuntimeError, match="simulated partial-startup failure"):
        with TestClient(create_app()):
            pass  # pragma: no cover - 启动就会失败

    assert len(captured) == 1
    assert captured[0]._closed is True
