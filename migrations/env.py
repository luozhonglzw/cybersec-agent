"""Alembic 运行环境(Phase v0.2.0-M1a)。

两点刻意的设计:

1. **连接串只来自环境变量** `CYBERSEC_PG_MIGRATION_DSN`。
   版本库里不存在任何凭据,`alembic.ini` 的 `sqlalchemy.url` 是空的。

2. **`target_metadata = None`** —— 本阶段不使用 autogenerate。
   基线 schema 全部是**手写 raw SQL**(见 `versions/`),因为:
   - 我们要的 CHECK 约束、identity 列、触发器、角色授权,autogenerate
     本来就表达不出来,反而会给出"看起来受管、其实漏掉"的假象;
   - 显式 SQL 让"数据库里到底有什么"可以逐行审阅。

DSN 归一化:SQLAlchemy 默认把 `postgresql://` 解析成 psycopg2 方言,
而本项目用的是 **psycopg v3**,所以这里把 scheme 改写为
`postgresql+psycopg://`。
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

#: 迁移连接串的唯一来源(必须使用 migration-owner 身份)。
MIGRATION_DSN_ENV = "CYBERSEC_PG_MIGRATION_DSN"

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False:迁移在测试进程内被调用时,
    # 不应把 pytest / 应用已经配置好的 logger 一并关掉。
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# 不用 autogenerate:基线是手写 raw SQL。
target_metadata = None


def migration_url() -> str:
    """从环境变量取迁移 DSN,并归一化到 psycopg v3 方言。"""
    url = os.environ.get(MIGRATION_DSN_ENV)
    if not url:
        raise RuntimeError(
            f"{MIGRATION_DSN_ENV} is not set. Migrations must run as the "
            "migration-owner role, e.g.\n"
            "  CYBERSEC_PG_MIGRATION_DSN=postgresql://cybersec_migrator:<pwd>"
            "@127.0.0.1:55432/cybersec_test alembic upgrade head"
        )
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def run_migrations_offline() -> None:
    """离线模式:只生成 SQL,不连库。"""
    context.configure(
        url=migration_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式:建立真实连接后执行迁移。"""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = migration_url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
