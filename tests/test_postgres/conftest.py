"""PostgreSQL 集成测试的公共夹具(Phase v0.2.0-M0R 建立,M1a 扩展)。

这些用例需要一个**真实的** PostgreSQL 实例,由仓库根目录的 opt-in
`compose.postgres.yaml` 提供:

    docker compose -f compose.postgres.yaml up -d --wait
    # 角色引导(仅空数据目录会自动执行;已有卷需手动补一次):
    #   docker exec -i cybersec-pg-test-postgres-1 psql -v ON_ERROR_STOP=1 \
    #     -U cybersec_test -d cybersec_test -f - < scripts/pg/bootstrap_roles.sql
    uv run --no-sync pytest -m postgres -q
    docker compose -f compose.postgres.yaml down

**Gate-F 纪律(硬要求)**:数据库不可达、角色缺失、迁移失败时夹具一律**抛错**,
绝不 `skip`。被跳过的 PG 用例不得被当作通过 —— 因此本文件刻意**不使用**
`pytest.importorskip` / `pytest.skip` / `skipif`。

默认的 `pytest` 运行**不收集**本目录的用例(见 pyproject 的
`addopts = -m 'not postgres'`),所以既有的离线套件在没有 Docker 的
机器(包括 CI 的 `test` job)上仍然全绿。

**三个角色,三种用途**(M1a 起):
- `CYBERSEC_TEST_PG_DSN`         → 引导用的 superuser 角色(`cybersec_test`),
                                   仅用于 M0R 的连通性 smoke test;
- `CYBERSEC_TEST_PG_APP_DSN`     → application-runtime(`cybersec_app`),
                                   权限/触发器/写入路径测试都用它;
- `CYBERSEC_TEST_PG_MIGRATOR_DSN`→ migration-owner(`cybersec_migrator`),
                                   只有迁移与所有权测试用它。

凭据一律是一次性本地开发凭据,不得承载任何真实数据。

**测试库不会自动重置(刻意的设计后果)**:三张表在库层是 append-only 的,
运行时角色没有 DELETE / TRUNCATE 权限,表 owner 也被触发器拦住。所以
用例写入的行**留得下来、删不掉**。这不成问题 —— 全部用例都用 `uuid4()`
生成主键,不依赖"库是空的"。需要真正的干净状态时,走这两条**受控**路径:

    # 1) 迁移 round-trip(DROP + 重建三张表)
    #    tests/test_postgres/test_pg_migration.py 已覆盖并自动化
    # 2) 丢弃数据卷
    docker compose -f compose.postgres.yaml down -v

**禁止**为了"清库"而临时禁用触发器或给运行时角色补授权 —— 那是绕过安全控制。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PG_DSN_ENV = "CYBERSEC_TEST_PG_DSN"
PG_APP_DSN_ENV = "CYBERSEC_TEST_PG_APP_DSN"
PG_MIGRATOR_DSN_ENV = "CYBERSEC_TEST_PG_MIGRATOR_DSN"

#: Alembic 读取迁移 DSN 的变量名(与 migrations/env.py 保持一致)。
MIGRATION_DSN_ENV = "CYBERSEC_PG_MIGRATION_DSN"

#: 与 compose.postgres.yaml 保持一致的一次性本地开发连接串。
#: 这些不是机密,不得复用,也不得指向任何真实数据。
DEFAULT_PG_DSN = (
    "postgresql://cybersec_test:cybersec_test_local_only"
    "@127.0.0.1:55432/cybersec_test"
)
DEFAULT_PG_APP_DSN = (
    "postgresql://cybersec_app:cybersec_app_local_only"
    "@127.0.0.1:55432/cybersec_test"
)
DEFAULT_PG_MIGRATOR_DSN = (
    "postgresql://cybersec_migrator:cybersec_migrator_local_only"
    "@127.0.0.1:55432/cybersec_test"
)


def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _connect(dsn: str, who: str):
    """建立 autocommit 连接。

    autocommit 是刻意的:权限测试会让语句失败(如 InsufficientPrivilege),
    在非 autocommit 下整个事务会进入 aborted 状态,后续语句全部报
    "current transaction is aborted",掩盖真正被测的行为。
    """
    import psycopg

    try:
        return psycopg.connect(dsn, connect_timeout=10, autocommit=True)
    except Exception as exc:  # noqa: BLE001 - 故意把任何失败都放大成 ERROR
        raise RuntimeError(
            f"PostgreSQL integration test database is unreachable as {who}. "
            "Start it with `docker compose -f compose.postgres.yaml up -d --wait`, "
            f"or point the corresponding CYBERSEC_TEST_PG_* env var at a running "
            f"instance. Underlying error: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# DSN
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """引导 superuser 角色的连接串(M0R 遗留,连通性 smoke test 用)。"""
    return _env_or_default(PG_DSN_ENV, DEFAULT_PG_DSN)


@pytest.fixture(scope="session")
def pg_app_dsn() -> str:
    """application-runtime 角色的连接串。"""
    return _env_or_default(PG_APP_DSN_ENV, DEFAULT_PG_APP_DSN)


@pytest.fixture(scope="session")
def pg_migrator_dsn() -> str:
    """migration-owner 角色的连接串。"""
    return _env_or_default(PG_MIGRATOR_DSN_ENV, DEFAULT_PG_MIGRATOR_DSN)


# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_connection(pg_dsn: str):
    """会话级真实 psycopg 连接(superuser 角色)。"""
    conn = _connect(pg_dsn, "the bootstrap superuser role")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session")
def pg_app_connection(pg_app_dsn: str):
    """会话级 application-runtime 连接(autocommit)。"""
    conn = _connect(pg_app_dsn, "the application runtime role")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session")
def pg_migrator_connection(pg_migrator_dsn: str):
    """会话级 migration-owner 连接(autocommit)。"""
    conn = _connect(pg_migrator_dsn, "the migration owner role")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# schema 就位
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def pg_schema_at_head(pg_migrator_dsn: str) -> None:
    """把被测库升到 Alembic head(幂等)。

    autouse:本目录下**每个** PG 用例都要求 schema 已就位。缺 schema 会
    直接 ERROR,而不是悄悄跑出一堆假绿。

    这里刻意**不**用 `pytest.skip` —— 迁移失败就是失败。
    """
    previous = os.environ.get(MIGRATION_DSN_ENV)
    os.environ[MIGRATION_DSN_ENV] = pg_migrator_dsn
    try:
        from alembic import command
        from alembic.config import Config

        config = Config(str(REPO_ROOT / "alembic.ini"))
        command.upgrade(config, "head")
    except Exception as exc:  # noqa: BLE001 - 故意把任何失败都放大成 ERROR
        raise RuntimeError(
            "Failed to bring the PostgreSQL test database to Alembic head. "
            "Check that the server is running, that scripts/pg/bootstrap_roles.sql "
            "has been applied, and that the migration-owner DSN is correct. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if previous is None:
            os.environ.pop(MIGRATION_DSN_ENV, None)
        else:
            os.environ[MIGRATION_DSN_ENV] = previous


def _run_alembic(dsn: str, action, revision: str) -> None:
    """以指定 DSN 跑一次 Alembic 命令;失败原样抛出,供负向测试断言。"""
    previous = os.environ.get(MIGRATION_DSN_ENV)
    os.environ[MIGRATION_DSN_ENV] = dsn
    try:
        from alembic import command
        from alembic.config import Config

        config = Config(str(REPO_ROOT / "alembic.ini"))
        action(config, revision)
    finally:
        if previous is None:
            os.environ.pop(MIGRATION_DSN_ENV, None)
        else:
            os.environ[MIGRATION_DSN_ENV] = previous


def run_migration_as(dsn: str, revision: str = "head") -> None:
    """以指定 DSN 运行一次 **upgrade**;失败原样抛出,供负向测试断言。

    注意方向:`upgrade` 只升不降。要回到旧版本必须用 `run_downgrade_as`
    —— 传 `"base"` 给本函数是**空操作**,不会删除任何表。
    """
    from alembic import command

    _run_alembic(dsn, command.upgrade, revision)


def run_downgrade_as(dsn: str, revision: str = "base") -> None:
    """以指定 DSN 运行一次 **downgrade**;失败原样抛出,供负向测试断言。"""
    from alembic import command

    _run_alembic(dsn, command.downgrade, revision)
