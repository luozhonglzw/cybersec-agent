"""两个后端的统一包装 + 共享测试数据工厂(Phase v0.2.0-M1b)。

目的是让**同一份**行为契约测试能逐字跑在 `SqliteAuditStore` 与
`PostgresAuditStore` 上 —— 不是写两套断言然后声称它们等价,而是**同一套断言**。

后端差异只有三处,**显式暴露、不藏**:

1. **占位符**:SQLite 用 `?`,PostgreSQL 用 `%s`;
2. **直连方式**:SQLite 是文件路径,PostgreSQL 是 DSN;
3. **append-only 拒绝的异常类**(见下)。

第 3 点值得单独说明:两个后端都在"运行期改写"这件事上 fail closed,
但**拦住的层不同** ——

    SQLite     → 库层触发器 RAISE(ABORT)  → sqlite3.IntegrityError
    PostgreSQL → 运行时角色没有 UPDATE 权限 → psycopg.errors.InsufficientPrivilege
                 (权限层先拦住,请求根本到不了触发器)

把这两个类归一成一个宽泛的 `except Exception` 会**掩盖**真实差异,所以这里
按后端显式列出期望的异常类,让差异成为断言的一部分。
"""
from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg

from app.schemas.approval import ApprovalRequest
from app.schemas.incident import Incident
from app.schemas.ownership import ThreadOwnership
from app.schemas.response import ResponseAction, ResponsePlan
from app.schemas.risk import RiskAssessment, RiskEvidence

INDICATOR = "203.0.113.66"
TS = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
TS_LATER = datetime(2026, 9, 17, 10, 5, 0, tzinfo=timezone.utc)

#: 用于隔离的终态事件(与 store.py / triage.py 一致)
TERMINAL_EVENTS = ("approval.decided", "approval.timeout")


def uid(prefix: str = "x") -> str:
    """每个用例一个唯一标识。

    PostgreSQL 侧是**共享库**:append-only 意味着用例写入的行删不掉,
    库会跨用例、跨运行累积。因此所有断言都必须按 thread_id / incident_id
    **限定作用域**,绝不能断言"全库只有 N 行"。
    """
    return f"{prefix}-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# 数据工厂 —— 两个后端共用同一份输入
# ---------------------------------------------------------------------------


def make_action(
    action_type: str = "block_ip", *, target: str = INDICATOR
) -> ResponseAction:
    return ResponseAction(
        action_type=action_type,
        priority="high",
        target=target,
        rationale="疑似 SSH 爆破,建议封禁源 IP",
        requires_approval=action_type in (
            "block_ip", "isolate_host", "reset_credentials"
        ),
        reversible=action_type != "reset_credentials",
    )


def make_plan(*, summary: str = "疑似 SSH 爆破") -> ResponsePlan:
    evidence = RiskEvidence(
        indicator=INDICATOR, log_event_count=30, failed_login_count=30
    )
    assessment = RiskAssessment(
        indicator=INDICATOR,
        risk_level="high",
        score=70,
        confidence=70,
        reasons=["30 次失败登录"],
        evidence=evidence,
    )
    return ResponsePlan(
        indicator=INDICATOR,
        risk_level="high",
        summary=summary,
        actions=[make_action()],
        assessment=assessment,
    )


def make_incident(
    *, incident_id: str, created_at: datetime = TS, score: int = 70
) -> Incident:
    return Incident(
        id=incident_id,
        created_at=created_at,
        indicator=INDICATOR,
        risk_level="high",
        score=score,
        summary="疑似 SSH 爆破",
        plan=make_plan(),
    )


def make_request(
    *,
    thread_id: str,
    actions: list[ResponseAction] | None = None,
    requested_at: datetime = TS,
    policy_reasons: list[str] | None = None,
    indicator: str = INDICATOR,
) -> ApprovalRequest:
    return ApprovalRequest(
        thread_id=thread_id,
        indicator=indicator,
        risk_level="high",
        score=70,
        summary="疑似 SSH 爆破",
        actions=actions if actions is not None else [make_action()],
        policy_reasons=(
            policy_reasons
            if policy_reasons is not None
            else ["动作按属性需要人工审批"]
        ),
        requested_at=requested_at,
    )


#: 归属测试用的主体名。**必须**是不同的字面量:`owner not in approvers` 是
#: 模型的硬校验,用同一个名字会让用例在构造期就炸,而不是在断言处失败。
OWNER = "alice"
APPROVER_B = "bob"
APPROVER_C = "carol"


def make_ownership(
    *,
    thread_id: str,
    owner: str = OWNER,
    approvers: tuple[str, ...] = (APPROVER_B,),
    created_at: datetime = TS,
) -> ThreadOwnership:
    """v0.3.0-A3-2 的归属聚合 —— 两个后端共用同一份输入。"""
    return ThreadOwnership(
        thread_id=thread_id,
        owner=owner,
        approvers=approvers,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# 后端包装
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """一个被测后端的统一视图。"""

    name: str
    store: object
    placeholder: str
    raw_target: str
    integrity_exc: tuple[type[BaseException], ...]
    append_only_update_exc: tuple[type[BaseException], ...]
    append_only_delete_exc: tuple[type[BaseException], ...]

    @contextmanager
    def raw(self) -> Iterator[object]:
        """直连数据库(**绕过 store**),用于 append-only 与脏数据注入。

        走 store 是发不出 UPDATE 的(源码里根本没有),所以这类验证必须
        直连数据库 —— 它验证的是**库层**的护栏,不是 store 的行为。
        """
        if self.name == "sqlite":
            # isolation_level=None ⇒ autocommit,与 PostgreSQL 侧的 autocommit=True
            # 对齐。否则 sqlite3 默认开事务,连接一关就**回滚**,
            # 注入的脏数据/种子行会静默消失,测试变成恒真。
            conn: object = sqlite3.connect(self.raw_target, isolation_level=None)
            conn.row_factory = sqlite3.Row  # type: ignore[attr-defined]
        else:
            conn = psycopg.connect(
                self.raw_target, autocommit=True, connect_timeout=10
            )
        try:
            yield conn
        finally:
            conn.close()  # type: ignore[attr-defined]


def sqlite_backend(db_path: str) -> Backend:
    from app.security.store import SqliteAuditStore

    return Backend(
        name="sqlite",
        store=SqliteAuditStore(db_path),
        placeholder="?",
        raw_target=db_path,
        integrity_exc=(sqlite3.IntegrityError,),
        append_only_update_exc=(sqlite3.IntegrityError,),
        append_only_delete_exc=(sqlite3.IntegrityError,),
    )


def postgres_backend(dsn: str) -> Backend:
    from app.security.store_postgres import PostgresAuditStore

    return Backend(
        name="postgres",
        store=PostgresAuditStore(dsn),
        placeholder="%s",
        raw_target=dsn,
        # psycopg 的 UniqueViolation / CheckViolation 都是 IntegrityError 的子类。
        integrity_exc=(psycopg.errors.IntegrityError,),
        # 权限层先拦住 —— 请求根本到不了触发器。
        append_only_update_exc=(psycopg.errors.InsufficientPrivilege,),
        append_only_delete_exc=(psycopg.errors.InsufficientPrivilege,),
    )
