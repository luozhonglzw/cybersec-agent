"""AuditStore Protocol 与 SqliteAuditStore 的**结构性等价**护栏(Phase v0.2.0-M1a)。

这里断言的是"契约没有漂移",不是"功能正确":
- 方法集必须逐一对应(不多不少);
- 每个方法的**签名**(参数名、参数种类、默认值、注解、返回注解)必须逐字相等;
- 任一侧改了签名而另一侧没跟上,测试必须失败。

为了证明这些断言**有牙**,文件末尾给了两个变异负对照:
- 少一个方法的桩 → 必须被判为"不满足契约";
- 改一个参数默认值的桩 → 必须被签名比对捕获。

本文件**不依赖任何数据库**,因此留在默认离线套件里(不带 `postgres` 标记)。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.security.store import SqliteAuditStore
from app.security.store_protocol import AuditStore

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "app" / "security" / "store_protocol.py"
STORE_PATH = REPO_ROOT / "app" / "security" / "store.py"


def _public_callables(cls: type) -> set[str]:
    """类自身 `__dict__` 里定义的公开可调用成员(排除 dunder 与下划线私有)。"""
    return {
        name
        for name, value in vars(cls).items()
        if callable(value) and not name.startswith("_")
    }


# 契约方法集(显式写死,顺序即文档顺序)
EXPECTED_METHODS = {
    "record_incident",
    "record_action_request",
    "append_audit",
    "get_incident",
    "get_approval_request",
    "list_action_rows",
    "pending_action_rows",
    "list_audit",
}


def test_protocol_is_a_runtime_checkable_protocol() -> None:
    assert getattr(AuditStore, "_is_protocol", False) is True
    assert getattr(AuditStore, "_is_runtime_protocol", False) is True


def test_protocol_declares_exactly_the_expected_methods() -> None:
    assert _public_callables(AuditStore) == EXPECTED_METHODS


def test_sqlite_store_public_methods_are_exactly_the_protocol_methods() -> None:
    """不多不少:`SqliteAuditStore` 的公开方法与契约方法集必须相等。

    若将来给 `SqliteAuditStore` 加了公开方法而没加进契约,这里会失败 ——
    那正是"契约落后于实现"的信号。
    """
    assert _public_callables(SqliteAuditStore) == EXPECTED_METHODS


def test_constructor_is_deliberately_excluded_from_the_protocol() -> None:
    """构造器**刻意**不在契约里(文件路径 vs 连接串,见模块 docstring)。"""
    assert "__init__" not in _public_callables(AuditStore)
    assert "db_path" not in {
        p for m in EXPECTED_METHODS for p in inspect.signature(
            getattr(AuditStore, m)
        ).parameters
    }


@pytest.mark.parametrize("method_name", sorted(EXPECTED_METHODS))
def test_signatures_match_sqlite_store_exactly(method_name: str) -> None:
    """逐方法比对完整签名:参数名 / 种类 / 默认值 / 注解 / 返回注解。"""
    protocol_sig = inspect.signature(getattr(AuditStore, method_name))
    store_sig = inspect.signature(getattr(SqliteAuditStore, method_name))
    assert protocol_sig == store_sig, (
        f"{method_name} 的签名不一致:\n"
        f"  protocol: {protocol_sig}\n"
        f"  sqlite  : {store_sig}"
    )


@pytest.mark.parametrize("method_name", sorted(EXPECTED_METHODS))
def test_keyword_only_parameters_stay_keyword_only(method_name: str) -> None:
    """显式守住 KEYWORD_ONLY —— 这些方法历史上就是 keyword-only 的。"""
    sig = inspect.signature(getattr(AuditStore, method_name))
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            assert inspect.signature(
                getattr(SqliteAuditStore, method_name)
            ).parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_sqlite_store_structurally_satisfies_the_protocol(tmp_path: Path) -> None:
    """`runtime_checkable` 的 isinstance 只查属性存在性,但它是第一道闸。"""
    store = SqliteAuditStore(tmp_path / "audit.db")
    assert isinstance(store, AuditStore)


def test_protocol_declares_no_private_helpers() -> None:
    """私有辅助方法刻意不进契约(行对象类型在后端之间不同)。"""
    private = {
        name
        for name, value in vars(AuditStore).items()
        if callable(value) and name.startswith("_") and not name.startswith("__")
    }
    assert private == set()


def test_protocol_module_imports_no_database_driver() -> None:
    """结构性护栏:契约模块不得 import 任何数据库/迁移驱动。

    契约一旦绑定到某个驱动,就不再是契约。用 AST 收集导入名,
    对 `import X` 与 `from X import Y` 分别处理。
    """
    tree = ast.parse(PROTOCOL_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported.add(node.module)

    forbidden = {"sqlite3", "psycopg", "psycopg_pool", "sqlalchemy", "alembic"}
    leaked = {
        name
        for name in imported
        if name.split(".")[0] in forbidden
    }
    assert leaked == set(), f"契约模块泄漏了驱动依赖: {sorted(leaked)}"

    # 正对照:契约**必须** import 三个领域模型,否则上面的收集器是坏的
    assert "app.schemas.audit" in imported
    assert "app.schemas.approval" in imported
    assert "app.schemas.incident" in imported


def test_store_module_remains_the_only_sqlite_implementation() -> None:
    """契约不得把 SQLite 实现搬进来:`store_protocol.py` 里不能出现 SQL 关键字。"""
    source = PROTOCOL_PATH.read_text(encoding="utf-8")
    for keyword in ("INSERT INTO", "CREATE TABLE", "SELECT "):
        assert keyword not in source


# ---------------------------------------------------------------------------
# 变异负对照:证明上面的断言不是恒真
# ---------------------------------------------------------------------------


class _MissingOneMethod:
    """少一个方法 —— 必须被判为不满足契约。"""

    def record_incident(self, incident) -> None: ...
    def record_action_request(self, request, *, incident_id=None) -> list: ...
    def append_audit(self, record) -> None: ...
    def get_incident(self, incident_id): ...
    def get_approval_request(self, thread_id): ...
    def list_action_rows(self, *, thread_id=None, incident_id=None) -> list: ...
    def pending_action_rows(self, *, thread_id=None) -> list: ...
    # list_audit 刻意缺失


def test_negative_control_missing_method_is_rejected() -> None:
    assert _public_callables(_MissingOneMethod) != EXPECTED_METHODS
    assert not isinstance(_MissingOneMethod(), AuditStore)


def test_negative_control_wrong_default_is_caught_by_signature_compare() -> None:
    """把 `descending` 的默认值从 False 改成 True —— 签名比对必须发现。"""

    class _WrongDefault:
        def list_audit(
            self,
            *,
            thread_id: str | None = None,
            incident_id: str | None = None,
            event: str | None = None,
            limit: int | None = None,
            descending: bool = True,
        ) -> list: ...

    mutated = inspect.signature(_WrongDefault.list_audit)
    canonical = inspect.signature(getattr(AuditStore, "list_audit"))
    assert mutated != canonical


def test_negative_control_wrong_parameter_kind_is_caught() -> None:
    """把 `thread_id` 从 keyword-only 改成位置参数 —— 签名比对必须发现。"""

    class _WrongKind:
        def list_audit(
            self,
            thread_id: str | None = None,
            *,
            incident_id: str | None = None,
            event: str | None = None,
            limit: int | None = None,
            descending: bool = False,
        ) -> list: ...

    mutated = inspect.signature(_WrongKind.list_audit)
    canonical = inspect.signature(getattr(AuditStore, "list_audit"))
    assert mutated != canonical
