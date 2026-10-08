"""baseline audit schema (incidents / action_requests / audit_logs)

Revision ID: 0001_baseline_audit_schema
Revises:
Create Date: 2026-10-08

Phase v0.2.0-M1a —— PostgreSQL 基线 schema。

**范围**:只建 schema、约束、索引、append-only 触发器与角色授权。
本迁移**不**实现任何 CRUD、**不**切换应用后端、**不**引入应用侧 ORM。

**逐条对齐 SQLite 契约**(`app/security/store.py`):
- 三张表名、列名、列顺序与 SQLite 完全一致,以便按列名取值的读路径原样复用;
- 时间列是 **TEXT**,存 tz-aware UTC 的 ISO8601 —— 刻意**不用** `timestamptz`,
  否则驱动会返回 `datetime` 对象,而 `_from_iso()` 期待字符串;
- `policy_reasons` / `plan_json` / `detail_json` 是 **TEXT** 而非 `jsonb`,
  否则 `json.loads(row[...])` 会拿到 dict 而抛错;
- `requires_approval` / `reversible` 是 **INTEGER**(不是 boolean),因为既有写路径
  写的是 `int(action.requires_approval)`;列类型必须让这条写路径原样可用;
- **不加外键**:`incident_id` 可空,且写入顺序由调用方决定;
- **不加** `UNIQUE(thread_id, action_type, target)`:它不是已证的幂等键。

**SQLite 的 `rowid` 在 PostgreSQL 不存在**。因此 `audit_logs` 与 `action_requests`
各加一列 identity `seq`,作为 `ORDER BY` 的确定性 tiebreaker。

**关于 seq 的诚实边界**(不得过度声称):
`GENERATED ALWAYS AS IDENTITY` 的取值来自序列,而序列的推进**不参与事务回滚**。
因此:
- `seq` 反映的是**分配顺序**,不是**提交顺序**。并发写入下,先提交的事务
  可能持有更大的 `seq`;
- 回滚会**消耗**序号并留下空洞;
- 因此 `ORDER BY seq` 只在**串行写入**(或单个写者)的前提下等价于"插入顺序",
  并发下它只是"分配顺序"。本迁移**不**声称 `seq` 给出全局提交序。
  详见 `tests/test_postgres/test_pg_ordering.py` 的负向对照。
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0001_baseline_audit_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


MIGRATOR_ROLE = "cybersec_migrator"
APP_ROLE = "cybersec_app"

#: 六条审计事件(封闭词表),与 `app/schemas/audit.py` 的 `AuditEvent` 一致。
_AUDIT_EVENTS = (
    "plan.created",
    "plan.failed",
    "policy.evaluated",
    "approval.requested",
    "approval.decided",
    "approval.timeout",
)

#: 风险等级(封闭词表),与 `app/schemas/risk.py` 的 `RiskLevel` 一致。
_RISK_LEVELS = ("none", "low", "medium", "high", "critical")

#: 处置动作(封闭词表),与 `app/schemas/response.py` 的 `ActionType` 一致。
_ACTION_TYPES = (
    "no_action",
    "monitor",
    "collect_evidence",
    "block_ip",
    "reset_credentials",
    "isolate_host",
    "escalate",
)

#: 动作优先级(封闭词表),与 `ActionPriority` 一致。
_PRIORITIES = ("low", "medium", "high", "critical")


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


# 每一条目 = **恰好一条** SQL 语句。逐条执行,而不是拼一个多语句字符串:
# 一来错误信息能精确指到出问题的那一条,二来避免依赖驱动的多语句行为。
_STATEMENTS_UP: tuple[str, ...] = (
    # ------------------------------------------------------------------
    # 0. 守卫:必须以 migration-owner 身份运行,且两个角色都已存在
    # ------------------------------------------------------------------
    f"""
    DO $m1a_guard$
    BEGIN
        IF current_user = '{APP_ROLE}' THEN
            RAISE EXCEPTION
                'refusing to run migrations as the application runtime role %',
                current_user
                USING HINT = 'run migrations as the migration-owner role '
                             '{MIGRATOR_ROLE}';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{MIGRATOR_ROLE}') THEN
            RAISE EXCEPTION
                'role {MIGRATOR_ROLE} is missing; run scripts/pg/bootstrap_roles.sql first';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
            RAISE EXCEPTION
                'role {APP_ROLE} is missing; run scripts/pg/bootstrap_roles.sql first';
        END IF;
    END
    $m1a_guard$
    """,
    # ------------------------------------------------------------------
    # 1. incidents
    # ------------------------------------------------------------------
    f"""
    CREATE TABLE incidents (
        id          TEXT    PRIMARY KEY,
        created_at  TEXT    NOT NULL,
        indicator   TEXT    NOT NULL,
        risk_level  TEXT    NOT NULL,
        score       INTEGER NOT NULL,
        summary     TEXT    NOT NULL,
        plan_json   TEXT    NOT NULL,
        CONSTRAINT incidents_risk_level_known
            CHECK (risk_level IN ({_quoted(_RISK_LEVELS)})),
        CONSTRAINT incidents_score_in_range
            CHECK (score >= 0 AND score <= 100)
    )
    """,
    # ------------------------------------------------------------------
    # 2. action_requests
    # ------------------------------------------------------------------
    f"""
    CREATE TABLE action_requests (
        id                TEXT    PRIMARY KEY,
        seq               BIGINT  GENERATED ALWAYS AS IDENTITY
                                  (SEQUENCE NAME action_requests_seq),
        incident_id       TEXT,
        thread_id         TEXT    NOT NULL,
        indicator         TEXT    NOT NULL,
        risk_level        TEXT    NOT NULL,
        score             INTEGER NOT NULL,
        summary           TEXT    NOT NULL,
        policy_reasons    TEXT    NOT NULL,
        action_type       TEXT    NOT NULL,
        priority          TEXT    NOT NULL,
        target            TEXT    NOT NULL,
        rationale         TEXT    NOT NULL,
        requires_approval INTEGER NOT NULL,
        reversible        INTEGER NOT NULL,
        requested_at      TEXT    NOT NULL,
        CONSTRAINT action_requests_seq_unique
            UNIQUE (seq),
        CONSTRAINT action_requests_risk_level_known
            CHECK (risk_level IN ({_quoted(_RISK_LEVELS)})),
        CONSTRAINT action_requests_score_in_range
            CHECK (score >= 0 AND score <= 100),
        CONSTRAINT action_requests_action_type_known
            CHECK (action_type IN ({_quoted(_ACTION_TYPES)})),
        CONSTRAINT action_requests_priority_known
            CHECK (priority IN ({_quoted(_PRIORITIES)})),
        CONSTRAINT action_requests_requires_approval_is_bool
            CHECK (requires_approval IN (0, 1)),
        CONSTRAINT action_requests_reversible_is_bool
            CHECK (reversible IN (0, 1))
    )
    """,
    # ------------------------------------------------------------------
    # 3. audit_logs
    # ------------------------------------------------------------------
    f"""
    CREATE TABLE audit_logs (
        id           TEXT   PRIMARY KEY,
        seq          BIGINT GENERATED ALWAYS AS IDENTITY
                            (SEQUENCE NAME audit_logs_seq),
        ts           TEXT   NOT NULL,
        actor        TEXT   NOT NULL,
        event        TEXT   NOT NULL,
        incident_id  TEXT,
        thread_id    TEXT,
        interrupt_id TEXT,
        outcome      TEXT,
        reason       TEXT,
        plan_digest  TEXT,
        detail_json  TEXT   NOT NULL,
        CONSTRAINT audit_logs_seq_unique
            UNIQUE (seq),
        CONSTRAINT audit_logs_event_known
            CHECK (event IN ({_quoted(_AUDIT_EVENTS)})),
        CONSTRAINT audit_logs_plan_digest_format
            CHECK (plan_digest IS NULL OR plan_digest ~ '^[0-9a-f]{{64}}$')
    )
    """,
    # `outcome` 刻意**不加** CHECK:契约里它是 `str | None`
    # (描述是 allow/deny/approved/denied,但不是 Literal),
    # 加封闭词表会拒绝契约允许的取值。
    # 同理 `actor` / `reason` / `interrupt_id` / `thread_id` 不受约束。
    # ------------------------------------------------------------------
    # 4. 索引
    # ------------------------------------------------------------------
    # 4.1-4.4 与 SQLite 的四条索引一一对应(兼容)
    "CREATE INDEX idx_audit_thread ON audit_logs (thread_id)",
    "CREATE INDEX idx_audit_incident ON audit_logs (incident_id)",
    "CREATE INDEX idx_req_thread ON action_requests (thread_id)",
    "CREATE INDEX idx_req_incident ON action_requests (incident_id)",
    # 4.5 `list_audit` 的默认排序是 (ts, seq);反向扫描同一索引即得 DESC 分支。
    "CREATE INDEX idx_audit_ts_seq ON audit_logs (ts, seq)",
    # 4.6 `pending_action_rows` 的派生查询按 (thread_id, event) 做 NOT EXISTS,
    #     这条复合索引直接服务该子查询。
    "CREATE INDEX idx_audit_thread_event ON audit_logs (thread_id, event)",
    # ------------------------------------------------------------------
    # 5. append-only:一个共享函数 + 每表 UPDATE / DELETE / TRUNCATE 触发器
    # ------------------------------------------------------------------
    # 错误码刻意用 '23000'(integrity_constraint_violation):
    # 驱动会把它映射成 IntegrityError,与 SQLite 侧 RAISE(ABORT) 的
    # sqlite3.IntegrityError 语义对齐;消息格式也与 SQLite 逐字一致。
    """
    CREATE OR REPLACE FUNCTION cybersec_reject_append_only_mutation()
    RETURNS trigger
    LANGUAGE plpgsql
    AS $m1a_append_only$
    BEGIN
        RAISE EXCEPTION '% is append-only: % rejected', TG_TABLE_NAME, TG_OP
            USING ERRCODE = '23000';
    END;
    $m1a_append_only$
    """,
    # 行级触发器覆盖 UPDATE / DELETE。
    # TRUNCATE **不触发行触发器**,所以另外给每表一条语句级 TRUNCATE 触发器:
    # 否则持有 DML 权限的角色(例如表 owner)可以静默清空审计流。
    # 运行时角色即便没有这条触发器也做不了 TRUNCATE(无权限),两层各自独立。
    """
    CREATE TRIGGER incidents_no_update BEFORE UPDATE ON incidents
        FOR EACH ROW EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER incidents_no_delete BEFORE DELETE ON incidents
        FOR EACH ROW EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER incidents_no_truncate BEFORE TRUNCATE ON incidents
        FOR EACH STATEMENT EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER action_requests_no_update BEFORE UPDATE ON action_requests
        FOR EACH ROW EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER action_requests_no_delete BEFORE DELETE ON action_requests
        FOR EACH ROW EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER action_requests_no_truncate BEFORE TRUNCATE ON action_requests
        FOR EACH STATEMENT EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER audit_logs_no_update BEFORE UPDATE ON audit_logs
        FOR EACH ROW EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER audit_logs_no_delete BEFORE DELETE ON audit_logs
        FOR EACH ROW EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    """
    CREATE TRIGGER audit_logs_no_truncate BEFORE TRUNCATE ON audit_logs
        FOR EACH STATEMENT EXECUTE FUNCTION cybersec_reject_append_only_mutation()
    """,
    # ------------------------------------------------------------------
    # 6. 所有权:对象必须属于 migration-owner,**绝不**属于运行时角色
    # ------------------------------------------------------------------
    f"ALTER TABLE incidents OWNER TO {MIGRATOR_ROLE}",
    f"ALTER TABLE action_requests OWNER TO {MIGRATOR_ROLE}",
    f"ALTER TABLE audit_logs OWNER TO {MIGRATOR_ROLE}",
    f"ALTER SEQUENCE action_requests_seq OWNER TO {MIGRATOR_ROLE}",
    f"ALTER SEQUENCE audit_logs_seq OWNER TO {MIGRATOR_ROLE}",
    f"ALTER FUNCTION cybersec_reject_append_only_mutation() OWNER TO {MIGRATOR_ROLE}",
    # ------------------------------------------------------------------
    # 7. 运行时角色授权:只有 SELECT / INSERT
    # ------------------------------------------------------------------
    # 先 REVOKE ALL 再 GRANT,保证不残留任何历史授权。
    # 刻意**不**给:UPDATE / DELETE / TRUNCATE / REFERENCES / TRIGGER /
    # CREATE(在 schema 上)。TRUNCATE 不在默认授权里,因此运行时角色
    # 无法清空审计表 —— 这一层不依赖触发器。
    "REVOKE ALL ON TABLE incidents, action_requests, audit_logs FROM PUBLIC",
    f"REVOKE ALL ON TABLE incidents, action_requests, audit_logs FROM {APP_ROLE}",
    f"GRANT SELECT, INSERT ON TABLE incidents, action_requests, audit_logs TO {APP_ROLE}",
    f"REVOKE ALL ON SEQUENCE action_requests_seq, audit_logs_seq FROM {APP_ROLE}",
    f"GRANT USAGE, SELECT ON SEQUENCE action_requests_seq, audit_logs_seq TO {APP_ROLE}",
    f"REVOKE ALL ON SCHEMA public FROM {APP_ROLE}",
    f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}",
)

_STATEMENTS_DOWN: tuple[str, ...] = (
    "DROP TABLE IF EXISTS audit_logs",
    "DROP TABLE IF EXISTS action_requests",
    "DROP TABLE IF EXISTS incidents",
    "DROP FUNCTION IF EXISTS cybersec_reject_append_only_mutation()",
)


def upgrade() -> None:
    """建立基线 schema、append-only 保护与角色授权。"""
    for statement in _STATEMENTS_UP:
        op.execute(statement)


def downgrade() -> None:
    """**破坏性**:删除三张表及其全部内容。

    append-only 约束保护的是"运行期的改写",不是"故意的整体拆除"。
    本函数只在一次性测试库上执行过,任何真实环境都不得运行它。
    """
    for statement in _STATEMENTS_DOWN:
        op.execute(statement)
