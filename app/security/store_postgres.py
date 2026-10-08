"""PostgreSQL 持久化层 —— Phase v0.2.0-M1b。

`SqliteAuditStore` 的 PostgreSQL 对等实现,满足 `app.security.store_protocol.AuditStore`
的**全部 8 个方法**。装配在**组合根**(`app/api/main.py` 的 `build_audit_store`)
按 `Settings` 的后端设置显式 opt-in(Phase v0.2.0-M1c);本模块自身不做后端
选择,默认后端仍是 SQLite。

## 与 SQLite 版的**逐条对齐**(这些是刻意的,不是巧合)

- **列名、列顺序、投影形状**逐字一致 —— 见 `migrations/versions/0001_baseline_audit_schema.py`;
- **时间列是 TEXT**,存 tz-aware UTC 的 ISO8601。因此 `ORDER BY ts` 的字典序
  **就是**时间序,与 SQLite 完全同构;也因此读回来是 `str`,`_from_iso()` 直接可用;
- **`policy_reasons` / `plan_json` / `detail_json` 是 TEXT**,读侧 `json.loads(row[...])` 原样可用;
- **`requires_approval` / `reversible` 是 INTEGER**,`bool(row[...])` 强转后与 SQLite 同结果;
- 时间语义(`_to_iso` / `_from_iso`)**直接复用 `store.py` 的实现**,不另写一份 ——
  复制一份"归一化到 UTC + 拒绝 naive"的逻辑,迟早会漂移,而审计流依赖顺序。
  这里 import 的是同包的私有辅助,**不修改 `store.py` 一个字节**。

## `rowid` → `seq`(唯一的结构性差异)

SQLite 的读路径依赖隐式 `rowid` 做 tiebreaker;PostgreSQL **没有 `rowid`**,
改用 M1a 引入的 identity 列 `seq`。

**关于 `seq` 的诚实边界(不得过度声称)**:`GENERATED ALWAYS AS IDENTITY` 的取值
来自序列,而序列推进**不参与事务回滚**。所以:

- `seq` 反映**分配顺序**,不是**提交顺序**,也不是因果序;
- 并发写入下,先分配的事务可能后提交 —— `ORDER BY seq` 给出的顺序与提交顺序
  可以**相反**(`tests/test_postgres/test_pg_ordering.py` 有可执行的负向对照);
- 回滚会**消耗**序号并留下空洞。

因此本模块**只**声称:"在已提交且可见的行集合上,`ORDER BY (ts, seq)` 是**确定**的
——同一份数据反复读得到同一顺序"。**不**声称全局因果序,**不**声称 exactly-once。

## append-only

与 SQLite 版同款的 5 个前提,但**保护机制更强**:SQLite 靠库层触发器,
PostgreSQL 是**权限层 + 触发器层两道独立防线** ——

1. 连接身份是受限的 `cybersec_app` 角色,它**没有** `UPDATE` / `DELETE` /
   `TRUNCATE` 权限,在 schema 上也没有 `CREATE`;
2. 即便用表 owner 身份,9 条库层触发器也会 `RAISE EXCEPTION`。

本模块**只发 INSERT**,不含 `UPDATE` / `DELETE` / `UPSERT` / `ON CONFLICT`
—— 由 `tests/test_postgres/test_store_postgres_specific.py` 的源码级护栏守着。

## 不做运行时 DDL

schema 全部来自 Alembic 迁移。本模块**不建表、不建索引、不建触发器**,
也不持有 migration-owner 凭据 —— 它连 DDL 权限都没有。

## 连接池生命周期(与 SQLite 的**明确差异**,不掩盖)

`SqliteAuditStore` 在构造器里建 schema,且"每次操作新开连接、用完即关、没有 close()"。
PostgreSQL 侧**不同**,且是刻意的:

- **构造器不做任何 I/O** —— 不建表、不连接。连接池在**首次操作**时惰性开启。
  理由:构造期不碰数据库,才谈得上"不做运行时 DDL";且数据库不可用应当在
  **真正需要写审计的那一刻**暴露给调用方,由调用方按既有的
  "必需写入 vs best-effort" 契约处理,而不是把失败挪到进程启动。
- **必须显式 `close()`** —— 池持有真实连接,不关就是泄漏。`close()` 之后
  本实例**终止**:再调用任何方法抛 `RuntimeError`(不做静默复活)。

## 凭据

DSN 只存在于实例内部,**不写日志、不进 `repr`、不进异常消息**(`redact_dsn` 供
调用方按需打印)。源码与配置里不存在任何真实凭据。
"""
from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.schemas.approval import ApprovalRequest
from app.schemas.audit import AuditRecord
from app.schemas.incident import Incident
from app.schemas.response import ResponseAction, ResponsePlan

# 时间语义**复用** SQLite 版的实现 —— 单一真相源,杜绝两侧漂移。
# 只 import 两个纯函数,不 import 那个类,也不触碰 store.py 的任何内容。
from app.security.store import _from_iso, _to_iso

#: 写入列清单 —— **不含 `seq`**(identity 列由数据库生成,显式赋值会被拒绝)。
_ACTION_REQUEST_COLUMNS = (
    "id, incident_id, thread_id, indicator, risk_level, score, summary,"
    " policy_reasons, action_type, priority, target, rationale,"
    " requires_approval, reversible, requested_at"
)

_AUDIT_LOG_COLUMNS = (
    "id, ts, actor, event, incident_id, thread_id, interrupt_id,"
    " outcome, reason, plan_digest, detail_json"
)

#: `pending_action_rows` 的两个终态事件 —— 与 `store.py` / `triage.py` 一致。
_TERMINAL_EVENTS = ("approval.decided", "approval.timeout")

#: 本后端的**持久化层**异常基类集合,供 API 读边界把它们收敛成同一个窄类型
#: (与 `sqlite3.Error` 同位)。只收 `psycopg.Error` 一个基类即可覆盖:
#: - 连接/协议/权限/完整性失败(`OperationalError` / `IntegrityError` / ...);
#: - 连接池借出超时 `psycopg_pool.PoolTimeout` 与池已关 `PoolClosed`
#:   —— 二者都继承 `psycopg.OperationalError`(实测 MRO:
#:   PoolTimeout → OperationalError → DatabaseError → Error),因此已被覆盖。
#: **不**包含 `RuntimeError`(`close()` 之后再使用本实例),那是生命周期
#: 误用而非"数据源不可用",必须继续表现为 500。
POSTGRES_STORE_ERRORS: tuple[type[BaseException], ...] = (psycopg.Error,)

#: 连接池默认参数。测试可覆盖(尤其是 `open_timeout`,用于故障注入)。
_DEFAULT_MIN_SIZE = 1
_DEFAULT_MAX_SIZE = 4
_DEFAULT_OPEN_TIMEOUT = 5.0
_DEFAULT_CONNECT_TIMEOUT = 10


def redact_dsn(dsn: str) -> str:
    """把 DSN 里的口令换成 `***`,供日志/诊断使用。

    解析失败时**不**回显原文 —— 宁可返回一个占位符,也不冒泄漏口令的风险。
    """
    try:
        info = conninfo_to_dict(dsn)
    except Exception:  # noqa: BLE001 - 任何解析失败都不回显原文
        return "<unparseable dsn>"
    if info.get("password"):
        info["password"] = "***"
    return make_conninfo(**info)


class PostgresAuditStore:
    """`AuditStore` 契约的 PostgreSQL 实现。

    连接身份**必须**是受限的 application-runtime 角色(`cybersec_app`):
    它没有 `UPDATE` / `DELETE` / `TRUNCATE`,也没有 DDL 权限。本类不做运行时
    schema 变更,因此不需要、也不应持有 migration-owner 凭据。
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = _DEFAULT_MIN_SIZE,
        max_size: int = _DEFAULT_MAX_SIZE,
        open_timeout: float = _DEFAULT_OPEN_TIMEOUT,
        connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
    ) -> None:
        if not dsn or not dsn.strip():
            raise ValueError("dsn 不能为空")
        if min_size < 0:
            raise ValueError("min_size 不能为负")
        if max_size < 1 or max_size < min_size:
            raise ValueError("max_size 必须 >= 1 且 >= min_size")
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._open_timeout = open_timeout
        self._connect_timeout = connect_timeout
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()
        self._closed = False

    def __repr__(self) -> str:
        return (
            f"PostgresAuditStore(dsn={redact_dsn(self._dsn)!r},"
            f" min_size={self._min_size}, max_size={self._max_size},"
            f" closed={self._closed})"
        )

    # ---------- 连接池生命周期 ----------

    def close(self) -> None:
        """关闭连接池并终止本实例(幂等)。

        调用后本实例不可再用 —— 任何操作抛 `RuntimeError`。
        刻意**不**做"静默复活":一个被关掉的审计 store 若还能悄悄重开连接,
        会把"生命周期管理错误"藏起来。
        """
        with self._lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    def _ensure_pool(self) -> ConnectionPool:
        """惰性开启连接池(首次操作时)。

        `min_size >= 1` + `open(wait=True)` ⇒ 数据库不可用时在这里**响亮失败**,
        而不是返回一个"看起来健康"的池。
        """
        if self._pool is not None:
            return self._pool
        with self._lock:
            if self._closed:
                raise RuntimeError("PostgresAuditStore 已关闭,不能再使用")
            if self._pool is None:
                pool = ConnectionPool(
                    conninfo=self._dsn,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    kwargs={
                        "row_factory": dict_row,
                        # autocommit:单语句操作天然无悬挂事务;需要原子性的多行写入
                        # 用显式 `conn.transaction()` 包住。这样池归还连接时
                        # 不会残留 "idle in transaction"。
                        "autocommit": True,
                        "connect_timeout": self._connect_timeout,
                    },
                    open=False,
                )
                try:
                    pool.open(wait=True, timeout=self._open_timeout)
                except Exception:
                    pool.close()
                    raise
                self._pool = pool
        return self._pool

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        """借出一个池化连接;退出时归还给池。"""
        with self._ensure_pool().connection() as conn:
            yield conn

    # ---------- 写:incidents ----------

    def record_incident(self, incident: Incident) -> None:
        """写入一条 incident 快照(append-only)。"""
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO incidents"
                " (id, created_at, indicator, risk_level, score, summary, plan_json)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    incident.id,
                    _to_iso(incident.created_at),
                    incident.indicator,
                    incident.risk_level,
                    incident.score,
                    incident.summary,
                    incident.plan.model_dump_json(),
                ),
            )

    # ---------- 写:action_requests ----------

    def record_action_request(
        self,
        request: ApprovalRequest,
        *,
        incident_id: str | None = None,
    ) -> list[str]:
        """写入一张审批单:每个 ResponseAction 一行。

        与 SQLite 版同语义:request 级字段在每行重复存一份(刻意的冗余),
        `incident_id` 由调用方给、可空。返回写入的行 id 列表,
        **顺序与 `request.actions` 一致**。

        **多行写入是原子的**:N 条 INSERT 包在同一个显式事务里,
        要么全成,要么全不成 —— 不存在"写了一半的审批单"。
        """
        policy_reasons = json.dumps(request.policy_reasons, ensure_ascii=False)
        requested_at = _to_iso(request.requested_at)

        rows = [
            (
                uuid.uuid4().hex,
                incident_id,
                request.thread_id,
                request.indicator,
                request.risk_level,
                request.score,
                request.summary,
                policy_reasons,
                action.action_type,
                action.priority,
                action.target,
                action.rationale,
                int(action.requires_approval),
                int(action.reversible),
                requested_at,
            )
            for action in request.actions
        ]

        with self._connection() as conn:
            # 显式事务:autocommit 连接下 `transaction()` 会真的 BEGIN/COMMIT。
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO action_requests ({_ACTION_REQUEST_COLUMNS})"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                        " %s, %s, %s, %s, %s)",
                        rows,
                    )
        return [row[0] for row in rows]

    # ---------- 写:audit_logs ----------

    def append_audit(self, record: AuditRecord) -> None:
        """追加一条审计记录(append-only)。"""
        with self._connection() as conn:
            conn.execute(
                f"INSERT INTO audit_logs ({_AUDIT_LOG_COLUMNS})"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    record.id,
                    _to_iso(record.ts),
                    record.actor,
                    record.event,
                    record.incident_id,
                    record.thread_id,
                    record.interrupt_id,
                    record.outcome,
                    record.reason,
                    record.plan_digest,
                    json.dumps(record.detail, ensure_ascii=False),
                ),
            )

    # ---------- 读:incidents ----------

    def get_incident(self, incident_id: str) -> Incident | None:
        """按 id 读回 incident;不存在返回 None。读取时重新过 Pydantic 校验。"""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE id = %s", (incident_id,)
            ).fetchone()
        if row is None:
            return None
        return Incident(
            id=row["id"],
            created_at=_from_iso(row["created_at"]),
            indicator=row["indicator"],
            risk_level=row["risk_level"],
            score=row["score"],
            summary=row["summary"],
            plan=ResponsePlan.model_validate_json(row["plan_json"]),
        )

    # ---------- 读:action_requests ----------

    def get_approval_request(self, thread_id: str) -> ApprovalRequest | None:
        """按 thread_id 把逐动作的行重新组装成 request 级模型。"""
        rows = self._select_action_rows(thread_id=thread_id)
        if not rows:
            return None
        first = rows[0]
        return ApprovalRequest(
            thread_id=first["thread_id"],
            indicator=first["indicator"],
            risk_level=first["risk_level"],
            score=first["score"],
            summary=first["summary"],
            actions=[self._row_to_action(row) for row in rows],
            policy_reasons=self._row_to_reasons(first),
            requested_at=_from_iso(first["requested_at"]),
        )

    def list_action_rows(
        self,
        *,
        thread_id: str | None = None,
        incident_id: str | None = None,
    ) -> list[dict]:
        """按动作粒度读出动作行的扁平投影(action 已过 Pydantic 校验)。

        投影形状与 SQLite 版**逐键一致**;`seq` 是实现细节,不进投影。
        """
        return [
            self._project(row)
            for row in self._select_action_rows(
                thread_id=thread_id, incident_id=incident_id
            )
        ]

    def pending_action_rows(self, *, thread_id: str | None = None) -> list[dict]:
        """派生查询:仍处于待审批状态的动作行(状态不落库)。

        判定完全来自已落库的事实 —— 某 thread_id 在 audit_logs 里**没有**
        对应的**终态事件**(`approval.decided` / `approval.timeout`),即为 pending。
        本方法不写入、不判定策略,只是读取侧的集合差投影。

        排序用 `r.seq`(SQLite 侧是 `r.rowid`)—— 见模块文档里关于 `seq`
        只是**分配顺序** tiebreaker 的说明。
        """
        sql = (
            "SELECT r.* FROM action_requests r"
            " WHERE NOT EXISTS ("
            "   SELECT 1 FROM audit_logs a"
            "   WHERE a.event IN (%s, %s)"
            "     AND a.thread_id = r.thread_id"
            " )"
        )
        params: list[Any] = list(_TERMINAL_EVENTS)
        if thread_id is not None:
            sql += " AND r.thread_id = %s"
            params.append(thread_id)
        sql += " ORDER BY r.seq"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._project(row) for row in rows]

    # ---------- 读:audit_logs ----------

    def list_audit(
        self,
        *,
        thread_id: str | None = None,
        incident_id: str | None = None,
        event: str | None = None,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[AuditRecord]:
        """按条件读审计流,按 `(ts, seq)` 升序。

        与 SQLite 版同语义:`ts` 是 ISO8601 UTC 文本,字典序 == 时间序;
        `seq` 只是同 `ts` 下的确定性 tiebreaker。

        方向由**固定程序逻辑**二选一 —— 不拼接调用方传入的 ORDER BY 片段、
        列名或原始 SQL;`limit` 非 None 时在 **SQL 层** 施加 `LIMIT %s`,
        值与过滤条件一律走参数绑定。
        """
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = %s")
            params.append(thread_id)
        if incident_id is not None:
            clauses.append("incident_id = %s")
            params.append(incident_id)
        if event is not None:
            clauses.append("event = %s")
            params.append(event)

        sql = "SELECT * FROM audit_logs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if descending:
            sql += " ORDER BY ts DESC, seq DESC"
        else:
            sql += " ORDER BY ts, seq"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_audit(row) for row in rows]

    # ---------- 内部:行 ↔ 对象 ----------

    def _select_action_rows(
        self,
        *,
        thread_id: str | None = None,
        incident_id: str | None = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = %s")
            params.append(thread_id)
        if incident_id is not None:
            clauses.append("incident_id = %s")
            params.append(incident_id)

        sql = "SELECT * FROM action_requests"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq"

        with self._connection() as conn:
            return conn.execute(sql, params).fetchall()

    @staticmethod
    def _row_to_action(row: dict) -> ResponseAction:
        """行 → ResponseAction(重新过 Pydantic 校验,脏数据在此暴露)。"""
        return ResponseAction(
            action_type=row["action_type"],
            priority=row["priority"],
            target=row["target"],
            rationale=row["rationale"],
            requires_approval=bool(row["requires_approval"]),
            reversible=bool(row["reversible"]),
        )

    @staticmethod
    def _row_to_reasons(row: dict) -> list[str]:
        reasons = json.loads(row["policy_reasons"])
        if not isinstance(reasons, list):
            raise ValueError("policy_reasons 列不是 JSON 数组")
        return reasons

    def _project(self, row: dict) -> dict:
        """行 → 扁平投影(动作部分已过 Pydantic 校验)。"""
        return {
            "row_id": row["id"],
            "incident_id": row["incident_id"],
            "thread_id": row["thread_id"],
            "indicator": row["indicator"],
            "risk_level": row["risk_level"],
            "score": row["score"],
            "summary": row["summary"],
            "policy_reasons": self._row_to_reasons(row),
            "action": self._row_to_action(row).model_dump(mode="json"),
            "requested_at": _from_iso(row["requested_at"]),
        }

    @staticmethod
    def _row_to_audit(row: dict) -> AuditRecord:
        """行 → AuditRecord(重新过 Pydantic 校验:event 枚举、sha256 格式等)。"""
        return AuditRecord(
            id=row["id"],
            ts=_from_iso(row["ts"]),
            actor=row["actor"],
            event=row["event"],
            incident_id=row["incident_id"],
            thread_id=row["thread_id"],
            interrupt_id=row["interrupt_id"],
            outcome=row["outcome"],
            reason=row["reason"],
            plan_digest=row["plan_digest"],
            detail=json.loads(row["detail_json"]),
        )
