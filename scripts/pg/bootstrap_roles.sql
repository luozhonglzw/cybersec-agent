-- PostgreSQL 角色引导(Phase v0.2.0-M1a)
--
-- 作用:创建两个**职责分离**的登录角色。
--
--   cybersec_migrator —— migration-owner。拥有全部 schema 对象,
--                        负责运行 Alembic 迁移(需要 schema 上的 CREATE)。
--   cybersec_app      —— application-runtime。只有 SELECT / INSERT;
--                        没有 UPDATE / DELETE / TRUNCATE,也没有 schema 上的 CREATE。
--
-- 为什么角色**不**放在 Alembic 迁移里:
--   迁移本身要由 migration-owner 身份运行,而 migration-owner 不可能创建自己;
--   集群级对象(角色)属于环境引导,不是版本化 schema 变更。
--
-- 幂等:可重复执行。重复执行会重置两个角色的口令并重申权限,不会报错。
--
-- 用法(superuser 身份执行一次):
--   psql -v ON_ERROR_STOP=1 -U <superuser> -d <db> -f scripts/pg/bootstrap_roles.sql
--
-- 口令来源(按优先级):
--   1. 会话 GUC cybersec.migrator_password / cybersec.app_password
--      —— 例如:PGOPTIONS='-c cybersec.app_password=...' psql ... -f ...
--   2. 下面的**一次性本地开发默认值**
--
-- 默认值不是机密:它们只用于 127.0.0.1 上的一次性测试实例,不得复用、
-- 不得承载任何真实数据。需要别的口令时走上面的 GUC,不必改这个文件。

\set ON_ERROR_STOP on

DO $m1a_bootstrap$
DECLARE
    v_migrator_password text := coalesce(
        current_setting('cybersec.migrator_password', true),
        'cybersec_migrator_local_only'
    );
    v_app_password text := coalesce(
        current_setting('cybersec.app_password', true),
        'cybersec_app_local_only'
    );
    v_db text := current_database();
BEGIN
    -- ---------- migration-owner ----------
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cybersec_migrator') THEN
        EXECUTE 'CREATE ROLE cybersec_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE';
    END IF;
    EXECUTE format(
        'ALTER ROLE cybersec_migrator WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
        v_migrator_password
    );

    -- ---------- application-runtime ----------
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cybersec_app') THEN
        EXECUTE 'CREATE ROLE cybersec_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE';
    END IF;
    EXECUTE format(
        'ALTER ROLE cybersec_app WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
        v_app_password
    );

    -- ---------- 数据库级 ----------
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO cybersec_migrator', v_db);
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO cybersec_app', v_db);

    -- ---------- schema 级 ----------
    -- 迁移角色可以建对象;运行时角色只能"用"schema,不能建任何东西。
    EXECUTE 'GRANT USAGE, CREATE ON SCHEMA public TO cybersec_migrator';
    EXECUTE 'REVOKE ALL ON SCHEMA public FROM cybersec_app';
    EXECUTE 'GRANT USAGE ON SCHEMA public TO cybersec_app';
END
$m1a_bootstrap$;

-- 刻意**不**做:把 TEMP 权限从运行时角色收走。
-- 临时表是会话级的临时草稿区,既不落库也不触碰审计 schema,
-- 收走它只会增加破坏其它工具的风险,换不来实际的安全收益。
