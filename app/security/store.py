"""SQLite 持久化层 —— Phase 8.2。

**只负责 persistence**,不承担任何判定:
    做:  Pydantic 对象 ↔ SQLite 行 的序列化/反序列化、SQL、行代理主键
    不做: 策略判定(不 import app.security.policy)、时间生成、plan_digest 计算、
          AuditRecord 构造、决定"该不该写"

    plan_digest 由 app.security.audit.compute_plan_digest 算好,
    AuditRecord / Incident / ApprovalRequest 由调用方构造好再传进来。
    行代理主键(uuid4)是本模块自己生成的 —— 那是**存储关注点**;
    业务标识(incident_id / thread_id)一律由调用方给出,本模块不推断。

append-only(三张表,缺一不可的 5 个前提):
    1. 只用 INSERT —— 本模块不含 UPDATE / DELETE / INSERT OR REPLACE;
    2. PRIMARY KEY 存在 —— 重复写入抛 sqlite3.IntegrityError(响亮失败,不静默覆盖);
    3. 库层禁改触发器 —— BEFORE UPDATE/DELETE 直接 RAISE(ABORT);
    4. 表里不存在可变状态列 —— "当前状态"无处可存,只能派生;
    5. 待审批状态是派生值 —— 某 thread_id 在 audit_logs 里没有对应的
       approval.decided 行 → 仍 pending。

    第 3 条把"我们承诺不 UPDATE"变成"数据库拒绝 UPDATE";第 1 条由
    tests/test_security/test_store.py 的结构性护栏测试守着(读源码断言无
    UPDATE / DELETE / INSERT OR REPLACE 字样)。

同步 API 与连接策略(刻意的选择):
    - sqlite3 是 stdlib,零新依赖(aiosqlite 未安装,引入即违反 Phase 8 约束);
    - 写入频率极低(一次 triage 个位数行,且只在审批路径上,不在 ReAct 热循环),
      阻塞事件循环的时间可忽略。async 是**调用点**的问题:真变热了在调用点
      await asyncio.to_thread(store.append_audit, rec) 即可,本模块无需改;
    - 每次操作新开一个连接(用完即关),彻底回避 check_same_thread 与
      跨线程复用问题。本模块不持有长连接,因此没有 close()。

时间列一律存 tz-aware UTC 的 ISO8601 文本:
    写入时把时间归一化到 UTC 并拒绝 naive datetime —— 一旦混入本地时区,
    字典序排序就不再等于时间序,而审计流完全依赖顺序。

读取一律重新过 Pydantic 校验:
    脏数据在读取边界就报错,不静默跳过(与 query_logs / query_threat_intel 同惯例)。

刻意不做的事:
    - 不建外键约束:incident_id 可空且写入顺序由 Phase 8.3 决定,
      现在加 FK 会把生命周期提前绑定;
    - 不启用 WAL:当前单进程低频写入,不触发 database is locked。
"""
import json
import sqlite3
import uuid
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.approval import ApprovalRequest
from app.schemas.audit import AuditRecord
from app.schemas.incident import Incident
from app.schemas.response import ResponseAction, ResponsePlan

_TABLES: tuple[str, ...] = ("incidents", "action_requests", "audit_logs")

_ACTION_REQUEST_COLUMNS = (
    "id, incident_id, thread_id, indicator, risk_level, score, summary,"
    " policy_reasons, action_type, priority, target, rationale,"
    " requires_approval, reversible, requested_at"
)

_AUDIT_LOG_COLUMNS = (
    "id, ts, actor, event, incident_id, thread_id, interrupt_id,"
    " outcome, reason, plan_digest, detail_json"
)


def _append_only_triggers(table: str) -> tuple[str, str]:
    """生成一张表的禁改触发器(UPDATE / DELETE 各一条)。"""
    return (
        f"CREATE TRIGGER IF NOT EXISTS {table}_no_update"
        f" BEFORE UPDATE ON {table}"
        f" BEGIN SELECT RAISE(ABORT, '{table} is append-only: UPDATE rejected'); END",
        f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete"
        f" BEFORE DELETE ON {table}"
        f" BEGIN SELECT RAISE(ABORT, '{table} is append-only: DELETE rejected'); END",
    )


def _schema_statements() -> list[str]:
    """全部 DDL:三张表 + 索引 + 六条禁改触发器(幂等,可重复执行)。"""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id          TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            indicator   TEXT NOT NULL,
            risk_level  TEXT NOT NULL,
            score       INTEGER NOT NULL,
            summary     TEXT NOT NULL,
            plan_json   TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS action_requests (
            id                TEXT PRIMARY KEY,
            incident_id       TEXT,
            thread_id         TEXT NOT NULL,
            indicator         TEXT NOT NULL,
            risk_level        TEXT NOT NULL,
            score             INTEGER NOT NULL,
            summary           TEXT NOT NULL,
            policy_reasons    TEXT NOT NULL,
            action_type       TEXT NOT NULL,
            priority          TEXT NOT NULL,
            target            TEXT NOT NULL,
            rationale         TEXT NOT NULL,
            requires_approval INTEGER NOT NULL,
            reversible        INTEGER NOT NULL,
            requested_at      TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id           TEXT PRIMARY KEY,
            ts           TEXT NOT NULL,
            actor        TEXT NOT NULL,
            event        TEXT NOT NULL,
            incident_id  TEXT,
            thread_id    TEXT,
            interrupt_id TEXT,
            outcome      TEXT,
            reason       TEXT,
            plan_digest  TEXT,
            detail_json  TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_audit_thread ON audit_logs(thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_incident ON audit_logs(incident_id)",
        "CREATE INDEX IF NOT EXISTS idx_req_thread ON action_requests(thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_req_incident ON action_requests(incident_id)",
    ]
    for table in _TABLES:
        statements.extend(_append_only_triggers(table))
    return statements


def _to_iso(value: datetime) -> str:
    """datetime → tz-aware UTC 的 ISO8601 文本。

    拒绝 naive datetime:混入本地时区会破坏"字典序 == 时间序"的前提,
    而审计流完全依赖顺序。
    """
    if value.tzinfo is None:
        raise ValueError("时间必须带时区(naive datetime 会破坏审计流的排序)")
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str) -> datetime:
    """ISO8601 文本 → tz-aware datetime(写侧已归一化为 UTC)。"""
    return datetime.fromisoformat(value)


class SqliteAuditStore:
    """incidents / action_requests / audit_logs 三张 append-only 表的读写入口。

    db_path 是**必填位置参数,没有默认值** —— 生产默认值由组合根从 Settings
    注入,测试必须显式传 tmp_path,因此不存在"忘记传路径就落到仓库 data/"的
    可能(见 tests/test_security/test_store.py 的护栏测试)。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        # 换台机器 data/ 里只有 .gitignore,首次使用需要能自建父目录
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            for statement in _schema_statements():
                conn.execute(statement)

    # ---------- 内部 ----------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------- 写:incidents ----------

    def record_incident(self, incident: Incident) -> None:
        """写入一条 incident 快照(append-only)。"""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO incidents"
                " (id, created_at, indicator, risk_level, score, summary, plan_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
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
        """写入一张审批单:每个 ResponseAction 一行(D2:粒度保持 action 级)。

        request 级字段在每行重复存一份 —— 这是刻意的冗余,换来"按动作查询"
        的能力(action_requests 独立于 audit_logs 存在的意义就在这里)。
        incident_id 由调用方给,可空(写入顺序由 Phase 8.3 决定)。

        返回写入的行 id 列表,顺序与 request.actions 一致。
        """
        policy_reasons = json.dumps(request.policy_reasons, ensure_ascii=False)
        requested_at = _to_iso(request.requested_at)

        rows = []
        for action in request.actions:
            rows.append((
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
            ))

        with closing(self._connect()) as conn, conn:
            conn.executemany(
                f"INSERT INTO action_requests ({_ACTION_REQUEST_COLUMNS})"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return [row[0] for row in rows]

    # ---------- 写:audit_logs ----------

    def append_audit(self, record: AuditRecord) -> None:
        """追加一条审计记录(append-only)。"""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT INTO audit_logs ({_AUDIT_LOG_COLUMNS})"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)
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
        """按 thread_id 读回审批单(把逐动作的行重新组装成 request 级模型)。

        同一 thread_id 的所有行共享 request 级字段,动作从各行还原。
        """
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
        """按动作粒度读出待审批动作的扁平投影(action 已过 Pydantic 校验)。

        返回的每个 dict 形如:
            {row_id, incident_id, thread_id, indicator, risk_level, score,
             summary, policy_reasons, action, requested_at}
        其中 action 是 ResponseAction 的 JSON dump。
        """
        return [self._project(row) for row in self._select_action_rows(
            thread_id=thread_id, incident_id=incident_id,
        )]

    def pending_action_rows(self, *, thread_id: str | None = None) -> list[dict]:
        """派生查询:仍处于待审批状态的动作行。

        判定完全来自已落库的事实 —— 某 thread_id 在 audit_logs 里**没有**
        对应的 approval.decided 行,即为 pending。本方法不写入、不判定策略,
        只是读取侧的集合差投影(状态不落库,因此不存在被改写的可能)。
        """
        sql = (
            "SELECT r.* FROM action_requests r"
            " WHERE NOT EXISTS ("
            "   SELECT 1 FROM audit_logs a"
            "   WHERE a.event = 'approval.decided' AND a.thread_id = r.thread_id"
            " )"
        )
        params: list[object] = []
        if thread_id is not None:
            sql += " AND r.thread_id = ?"
            params.append(thread_id)
        sql += " ORDER BY r.rowid"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._project(row) for row in rows]

    # ---------- 读:audit_logs ----------

    def list_audit(
        self,
        *,
        thread_id: str | None = None,
        incident_id: str | None = None,
        event: str | None = None,
    ) -> list[AuditRecord]:
        """按条件读审计流,按 (ts, rowid) 升序 —— 顺序确定,不依赖存储顺序。"""
        clauses: list[str] = []
        params: list[object] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if incident_id is not None:
            clauses.append("incident_id = ?")
            params.append(incident_id)
        if event is not None:
            clauses.append("event = ?")
            params.append(event)

        sql = "SELECT * FROM audit_logs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts, rowid"

        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_audit(row) for row in rows]

    # ---------- 内部:行 ↔ 对象 ----------

    def _select_action_rows(
        self,
        *,
        thread_id: str | None = None,
        incident_id: str | None = None,
    ) -> Sequence[sqlite3.Row]:
        clauses: list[str] = []
        params: list[object] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if incident_id is not None:
            clauses.append("incident_id = ?")
            params.append(incident_id)

        sql = "SELECT * FROM action_requests"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"

        with closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchall()

    @staticmethod
    def _row_to_action(row: sqlite3.Row) -> ResponseAction:
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
    def _row_to_reasons(row: sqlite3.Row) -> list[str]:
        reasons = json.loads(row["policy_reasons"])
        if not isinstance(reasons, list):
            raise ValueError("policy_reasons 列不是 JSON 数组")
        return reasons

    def _project(self, row: sqlite3.Row) -> dict:
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
    def _row_to_audit(row: sqlite3.Row) -> AuditRecord:
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
