"""审计记录构造(纯函数)—— Phase 8.2。

职责边界(与 store.py 的分工,勿混):
    本模块**只做纯计算与对象构造**,不碰任何 I/O:
        compute_plan_digest(plan)   规范化摘要
        new_record_id()             记录 id
        build_audit_record(...)     组装 AuditRecord(摘要由 plan 推导)
    store.py 只负责把已构造好的对象写进 SQLite、再读出来。

为什么 plan_digest 必须由 plan 推导,而不是由调用方传入:
    摘要的作用是"比对计划是否被改动过"。若允许调用方手写 digest,就出现了
    两个真相源 —— 传错值、传旧值都不会被发现,摘要也就失去意义。
    本模块只接受 plan 对象,digest 在内部计算,调用方无法注入。

规范化的定义:
    plan.model_dump(mode="json")
      → json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"))
      → sha256 十六进制
    键序、空白、非 ASCII 转义全部被消除,同一份计划永远得到同一个摘要。
    (注:Pydantic 的 model_dump_json() 不接受 sort_keys 参数,故走 json.dumps。)
"""
import copy
import hashlib
import json
import uuid
from datetime import datetime

from app.schemas.approval import utc_now
from app.schemas.audit import SYSTEM_ACTOR, AuditEvent, AuditRecord
from app.schemas.response import ResponsePlan


def compute_plan_digest(plan: ResponsePlan) -> str:
    """返回 plan 的规范化 sha256(64 位小写十六进制)。"""
    canonical = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_record_id() -> str:
    """返回新的审计记录 id(uuid4 十六进制)。"""
    return uuid.uuid4().hex


def build_audit_record(
    event: AuditEvent,
    *,
    actor: str = SYSTEM_ACTOR,
    incident_id: str | None = None,
    thread_id: str | None = None,
    interrupt_id: str | None = None,
    outcome: str | None = None,
    reason: str | None = None,
    plan: ResponsePlan | None = None,
    detail: dict | None = None,
    ts: datetime | None = None,
) -> AuditRecord:
    """组装一条 AuditRecord。

    参数:
        plan: 传入时自动计算 plan_digest —— 调用方无法手写摘要;
              没有关联计划的审计事件(如 approval.timeout)不传。
        ts:   省略时取当前 UTC。集中在此处生成,便于测试注入固定时间。
        detail: 自包含取证负载,**深拷贝**一份 —— 审计记录必须与调用方的
              后续修改彻底隔离。浅拷贝只复制最外层,嵌套的 list/dict 仍与
              调用方共享,事后一次 append 就能改写已落库的审计证据。
    """
    return AuditRecord(
        id=new_record_id(),
        ts=ts if ts is not None else utc_now(),
        actor=actor,
        event=event,
        incident_id=incident_id,
        thread_id=thread_id,
        interrupt_id=interrupt_id,
        outcome=outcome,
        reason=reason,
        plan_digest=compute_plan_digest(plan) if plan is not None else None,
        detail=copy.deepcopy(detail) if detail else {},
    )
