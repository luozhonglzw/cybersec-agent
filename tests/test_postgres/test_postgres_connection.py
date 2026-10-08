"""最小 PostgreSQL 连接 smoke test(Phase v0.2.0-M0R)。

范围刻意保持最小:只证明「项目 Python 环境能连上真实 PostgreSQL 并执行
`SELECT 1`」。这里**不**定义任何 schema、表、索引、触发器或迁移 —— 那些
属于 M1 之后,本次未获授权。

运行方式(需要真实实例):

    docker compose -f compose.postgres.yaml up -d --wait
    uv run --no-sync pytest -m postgres -q
    docker compose -f compose.postgres.yaml down

这些用例在默认 `pytest` 运行中被**取消收集**(不产生 skip 记录);
显式 `-m postgres` 选中时,库不可达会 ERROR/FAILED 而不是 skip。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.postgres


def test_select_one_against_real_postgres(pg_connection) -> None:
    """M0R 的核心断言:真实 PostgreSQL 上 `SELECT 1` 返回 1。"""
    with pg_connection.cursor() as cur:
        cur.execute("SELECT 1")
        row = cur.fetchone()

    assert row == (1,)


def test_server_identifies_as_postgresql(pg_connection) -> None:
    """证明连到的是 PostgreSQL 服务端,而不是别的什么东西。"""
    with pg_connection.cursor() as cur:
        cur.execute("SELECT version()")
        (version,) = cur.fetchone()

    assert isinstance(version, str)
    assert version.startswith("PostgreSQL ")


def test_connected_to_the_expected_database_and_user(
    pg_connection, pg_dsn: str
) -> None:
    """连接落在 DSN 指定的库与角色上(期望值从 DSN 推导,不写死)。"""
    from psycopg.conninfo import conninfo_to_dict

    params = conninfo_to_dict(pg_dsn)

    with pg_connection.cursor() as cur:
        cur.execute("SELECT current_database(), current_user")
        database, user = cur.fetchone()

    assert database == params.get("dbname")
    assert user == params.get("user")


def test_connection_pool_executes_select_one(pg_dsn: str) -> None:
    """`psycopg_pool` 是本次新增的第二个依赖,这里证明它对真实库可用。

    池化连接是 M1 之后 `PostgresAuditStore` 的既定形态,所以在 smoke 阶段
    就把它跑通一次,避免把"依赖能装"误当成"依赖能用"。
    """
    from psycopg_pool import ConnectionPool

    with ConnectionPool(pg_dsn, min_size=1, max_size=2, open=True, timeout=10) as pool:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                assert cur.fetchone() == (1,)
