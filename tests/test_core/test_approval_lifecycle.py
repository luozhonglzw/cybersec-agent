"""Phase 9.1-A 审批生命周期测试:超时 / 终态清理 / 恢复守卫。

覆盖五组契约:

0. **装配契约(D1)**
   service 从图上取到的 checkpointer 必须**就是**组合根那一个 ——
   否则清理删的是"另一套 checkpoint",暂停态永远不会被释放(静默失效)。

A. **惰性超时 reap**(挂在 triage() 入口)
   pending 的 thread 过窗口后被收成 timed_out:落一条 approval.timeout +
   删除 checkpoint。没有后台调度器 —— 清理只在请求到达时发生。

B. **resume 侧的超时点检查**
   校验门在取 interrupt_id **之前**判超时;超时后该 thread 永久 409,
   且**绝不**写 approval.decided(没有人工决定,就不能留下决定的痕迹)。

C. **failed thread 清理**
   plan 节点失败会在 checkpoint 里留下 next=('plan',) 的残留快照,
   9.1-A 起清掉它;resume 该 thread 的消息也必须准确(不是"审批已完成")。

D. **不变量:绝不删 pending**(本阶段最重要的一条)
   只有 failed / timed_out 两个**终态**会删 checkpoint。pending_approval /
   completed / allowed 一律不动。变异测试 M4 专门打这条。

超时怎么在测试里触发 —— 不需要注入时钟:
    ApprovalRequest.requested_at 由调用方给定(store 只拒绝 naive),
    所以要么用 approval_timeout=timedelta(0)(一切 pending 立即过期),
    要么直接造一条 requested_at 在两天前的 pending 行。
    后者同时覆盖了"进程重启后 checkpoint 已丢、只剩 pending 行"的僵尸形态。

全部 hermetic:数据文件与 audit.db 都由 tmp_path 现场生成,
不读仓库 data/,也不在仓库里创建 audit.db。
"""
import ast
import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import FakeLLMClient
from app.core.triage import (
    APPROVAL_TIMEOUT,
    TIMEOUT_EVENT,
    ApprovalExpiredError,
    NotAwaitingApprovalError,
    TriageDataUnavailableError,
    TriageService,
    UnknownThreadError,
)
from app.schemas.approval import ApprovalRequest, utc_now
from app.schemas.response import ResponseAction
from app.security.store import SqliteAuditStore

# 与 test_triage_service.py 同一套夹具语义:
# 30 次失败登录 + 恶意情报 → critical → 含 block_ip / isolate_host /
# reset_credentials → 策略要求人工审批
BRUTE_FORCE_IP = "203.0.113.66"
BRUTE_FORCE_LOGINS = 30

# 10 次失败登录,无情报命中 → low → 仅 monitor → 策略放行
LOW_RISK_IP = "198.51.100.7"
LOW_RISK_LOGINS = 10

REPLY = "分析完成"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRIAGE_SOURCE = _REPO_ROOT / "app" / "core" / "triage.py"


# ---------- hermetic 数据夹具 ----------

def _log_record(ip: str, index: int) -> dict:
    return {
        "timestamp": f"2026-09-10T08:{index:02d}:00Z",
        "event_type": "login_failed",
        "source": "sshd",
        "source_ip": ip,
        "username": "root",
        "status": "failed",
        "severity": "high",
        "message": "Failed password for root",
    }


def _write_logs(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(BRUTE_FORCE_LOGINS):
            f.write(json.dumps(_log_record(BRUTE_FORCE_IP, i)) + "\n")
        for i in range(LOW_RISK_LOGINS):
            f.write(json.dumps(_log_record(LOW_RISK_IP, i)) + "\n")


def _write_intel(path: Path) -> None:
    """只有 BRUTE_FORCE_IP 有恶意情报;LOW_RISK_IP 不命中。"""
    path.write_text(json.dumps({
        "indicator": BRUTE_FORCE_IP,
        "indicator_type": "ip",
        "malicious": True,
        "confidence": 90,
        "severity": "high",
        "tags": ["ssh-brute-force"],
        "source": "test-fixture",
        "first_seen": "2026-09-01T00:00:00Z",
        "last_seen": "2026-09-10T00:00:00Z",
        "description": "SSH brute force source",
    }) + "\n", encoding="utf-8")


# ---------- 装配 ----------

class _Env(NamedTuple):
    service: TriageService
    store: SqliteAuditStore
    graph: object
    checkpointer: InMemorySaver
    logs: Path
    intel: Path

    def has_checkpoint(self, thread_id: str) -> bool:
        """该 thread 在 checkpoint 存储里是否还在(不经过任何包装)。"""
        return thread_id in self.checkpointer.storage


class _UndeletableSaver(InMemorySaver):
    """adelete_thread 静默不生效 —— 模拟"清理失败,其余一切正常"的后端。

    用来验证一条强不变量:超时的**不可逆性**不依赖清理是否成功。
    """

    async def adelete_thread(self, thread_id: str) -> None:
        return None


class _ExplodingSaver(InMemorySaver):
    """adelete_thread 直接抛错 —— 用来验证错误优先级规则(9.1-A D2)。"""

    async def adelete_thread(self, thread_id: str) -> None:
        raise RuntimeError("backend refuses to delete")


def _make_env(
    store: SqliteAuditStore,
    logs: Path,
    intel: Path,
    *,
    approval_timeout: timedelta = APPROVAL_TIMEOUT,
    checkpointer: InMemorySaver | None = None,
) -> _Env:
    saver = InMemorySaver() if checkpointer is None else checkpointer
    hitl = HitlConfig(
        checkpointer=saver,
        audit_store=store,
        logs_path=logs,
        intel_path=intel,
    )
    graph = create_agent_graph(FakeLLMClient(REPLY), hitl=hitl)
    return _Env(
        service=TriageService(graph, store, approval_timeout=approval_timeout),
        store=store,
        graph=graph,
        checkpointer=saver,
        logs=logs,
        intel=intel,
    )


@pytest.fixture
def data_paths(tmp_path: Path) -> tuple[Path, Path]:
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)
    return logs, intel


@pytest.fixture
def store(tmp_path: Path) -> SqliteAuditStore:
    return SqliteAuditStore(tmp_path / "audit.db")


@pytest.fixture
def env(data_paths, store) -> _Env:
    """生产默认窗口(24h):新产生的 pending 一律不过期。"""
    logs, intel = data_paths
    return _make_env(store, logs, intel)


@pytest.fixture
def zero_env(data_paths, store) -> _Env:
    """approval_timeout=timedelta(0):任何 pending 都立即算过期。

    这是合法的退化配置(窗口长度为零),不是测试后门 ——
    闭区间判定 elapsed >= timeout 让它必然成立。
    """
    logs, intel = data_paths
    return _make_env(store, logs, intel, approval_timeout=timedelta(0))


def _request(*, thread_id: str, age: timedelta = timedelta(0)) -> ApprovalRequest:
    """构造一张审批单;requested_at 可回溯,用来制造"很久以前请求的"行。"""
    return ApprovalRequest(
        thread_id=thread_id,
        indicator=BRUTE_FORCE_IP,
        risk_level="critical",
        score=90,
        summary="疑似 SSH 爆破",
        actions=[
            ResponseAction(
                action_type="block_ip",
                priority="high",
                target=BRUTE_FORCE_IP,
                rationale="疑似 SSH 爆破,建议封禁源 IP",
                requires_approval=True,
                reversible=True,
            )
        ],
        policy_reasons=["destructive action requires approval"],
        requested_at=utc_now() - age,
    )


def _seed_pending_row(store: SqliteAuditStore, thread_id: str, *, age: timedelta) -> None:
    """直接写一条待审批行,不经过图。

    超时判定只依赖 action_requests.requested_at + audit_logs,与 checkpoint
    是否存在无关 —— 进程重启后的僵尸 pending 行就是这个形态。
    """
    store.record_action_request(_request(thread_id=thread_id, age=age))


def _events(store: SqliteAuditStore, thread_id: str) -> list[str]:
    return [record.event for record in store.list_audit(thread_id=thread_id)]


# =====================================================================
# 0. 装配契约:checkpointer 从图上取(D1)
# =====================================================================

def test_graph_exposes_the_same_checkpointer_instance(store, data_paths):
    """service 从图上取到的 checkpointer 必须**就是**组合根那一个。

    超时清理靠 self._graph.checkpointer.adelete_thread 删快照。如果图上
    暴露的是另一个实例,清理会删到"另一套 checkpoint",而真正的暂停态
    永远不会被释放 —— 静默失效,没有任何报错。
    所以这条身份断言是清理功能的前提,不是锦上添花。
    """
    logs, intel = data_paths
    saver = InMemorySaver()
    env = _make_env(store, logs, intel, checkpointer=saver)

    assert env.graph.checkpointer is saver
    assert env.service._graph.checkpointer is saver


def test_checkpointer_exposes_async_delete_capability(store, data_paths):
    """清理依赖的是 adelete_thread 这个**能力**,不是具体类型。

    断言"有这个方法"而不是"是 InMemorySaver":将来换持久化后端时,
    只要还提供 adelete_thread,本模块的清理逻辑就不用改。
    """
    logs, intel = data_paths
    env = _make_env(store, logs, intel)

    assert callable(getattr(env.graph.checkpointer, "adelete_thread", None))


# =====================================================================
# A. 惰性超时 reap
# =====================================================================

async def test_reap_expired_reaps_thread_whose_window_passed(zero_env):
    """过窗口的 pending → 收成 timed_out,返回被收的 thread_id。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    reaped = await zero_env.service.reap_expired()

    assert reaped == [thread_id]
    assert _events(zero_env.store, thread_id).count(TIMEOUT_EVENT) == 1


async def test_reap_deletes_expired_checkpoint(zero_env):
    """清理范围含 timed_out:checkpoint 必须真的被删掉。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id
    assert zero_env.has_checkpoint(thread_id)

    await zero_env.service.reap_expired()

    assert not zero_env.has_checkpoint(thread_id)


async def test_reaped_thread_is_no_longer_pending(zero_env):
    """超时解除 pending —— 否则每轮 reap 都会重复写审计。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id
    assert len(zero_env.store.pending_action_rows(thread_id=thread_id)) == 3

    await zero_env.service.reap_expired()

    assert zero_env.store.pending_action_rows(thread_id=thread_id) == []


async def test_timeout_audit_written_exactly_once(zero_env):
    """反复 reap 不得重复计数 —— 审计是事实日志,重复就是失真。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    for _ in range(3):
        await zero_env.service.reap_expired()

    assert len(zero_env.store.list_audit(thread_id=thread_id, event=TIMEOUT_EVENT)) == 1


async def test_reap_one_is_idempotent_when_called_twice(zero_env):
    """_reap_one 自身幂等:同一 thread 收两次也只写一条审计。

    为什么不能只靠 reap_expired 的 pending 过滤:reap_expired 是
    "先查 pending、再写审计",两步之间没有事务。两个并发的 /triage 请求
    可能都读到"这个 thread 还 pending",于是都去收它。
    幂等闸门让后到的那次只做清理、不重复落审计 —— 这里直接把
    _reap_one 调两次来模拟那次交错。
    """
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    await zero_env.service._reap_one(thread_id, timedelta(seconds=1))
    await zero_env.service._reap_one(thread_id, timedelta(seconds=1))

    assert len(zero_env.store.list_audit(thread_id=thread_id, event=TIMEOUT_EVENT)) == 1


async def test_timeout_detail_is_self_contained(zero_env):
    """detail 要能独立回答"为什么算过期" —— 审计不依赖其他表。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    await zero_env.service.reap_expired()

    record = zero_env.store.list_audit(event=TIMEOUT_EVENT)[0]

    assert record.actor == "system"
    assert record.thread_id == result.outcome.thread_id
    assert set(record.detail) == {"elapsed_seconds", "timeout_seconds"}
    assert record.detail["timeout_seconds"] == 0
    assert record.detail["elapsed_seconds"] >= 0


async def test_timeout_audit_does_not_carry_incident_id(zero_env):
    """9.1-A 刻意不扩大 incident 关联限制:timeout 也不传 incident_id。

    图节点写审计时同样拿不到 incident(它在图跑完之后才创建,D4)。
    timeout 必须与之一致 —— 不要在这一阶段单独给一个事件开小灶,
    否则"哪些审计行能按 incident 查"会变成一张需要记忆的清单。
    """
    await zero_env.service.triage(BRUTE_FORCE_IP)
    await zero_env.service.reap_expired()

    assert zero_env.store.list_audit(event=TIMEOUT_EVENT)[0].incident_id is None


async def test_triage_entry_reaps_expired_pending(zero_env):
    """惰性触发点:reap 挂在 triage() 入口(没有后台调度器)。"""
    first = await zero_env.service.triage(BRUTE_FORCE_IP)
    expired_id = first.outcome.thread_id

    await zero_env.service.triage(LOW_RISK_IP)

    assert not zero_env.has_checkpoint(expired_id)
    assert len(zero_env.store.list_audit(thread_id=expired_id, event=TIMEOUT_EVENT)) == 1


async def test_reap_selects_only_expired_threads(env):
    """只有过窗口的那个被收;另一个原样留着(按 thread 分别判,不搞连坐)。"""
    _seed_pending_row(env.store, "thread-old", age=timedelta(days=2))
    _seed_pending_row(env.store, "thread-fresh", age=timedelta(0))

    reaped = await env.service.reap_expired()

    assert reaped == ["thread-old"]
    assert env.store.pending_action_rows(thread_id="thread-old") == []
    assert len(env.store.pending_action_rows(thread_id="thread-fresh")) == 1


async def test_reap_resolves_zombie_pending_without_checkpoint(env):
    """进程重启留下的僵尸 pending 行(无 checkpoint)也会被收成终态。

    这类行本来会让 resume() 永远返回 CheckpointLostError,并让 pending
    视图永远不干净。窗口过期后它们转成 timed_out,不再污染派生状态。
    """
    _seed_pending_row(env.store, "thread-zombie", age=timedelta(days=3))
    assert not env.has_checkpoint("thread-zombie")

    reaped = await env.service.reap_expired()

    assert reaped == ["thread-zombie"]
    assert len(env.store.list_audit(thread_id="thread-zombie", event=TIMEOUT_EVENT)) == 1


async def test_reap_is_noop_on_empty_store(env):
    assert await env.service.reap_expired() == []


# =====================================================================
# B. resume 侧的超时点检查
# =====================================================================

async def test_resume_expired_thread_raises_approval_expired(zero_env):
    result = await zero_env.service.triage(BRUTE_FORCE_IP)

    with pytest.raises(ApprovalExpiredError):
        await zero_env.service.resume(
            result.outcome.thread_id, status="approved", operator="alice"
        )


async def test_resume_expired_reaps_and_deletes_checkpoint(zero_env):
    """resume 也是惰性触发点:超时判定顺带完成终态清理。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    with pytest.raises(ApprovalExpiredError):
        await zero_env.service.resume(thread_id, status="approved", operator="alice")

    assert not zero_env.has_checkpoint(thread_id)
    assert len(zero_env.store.list_audit(thread_id=thread_id, event=TIMEOUT_EVENT)) == 1


async def test_resume_expired_never_writes_approval_decided(zero_env):
    """**核心**:超时绝不写 approval.decided。

    没有人工决定,就不能留下决定的痕迹。若这里写了一条 decided,
    审计会显示"有人批准了",而实际上没有任何人做过决定 ——
    这是最严重的一类审计失真。
    """
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    with pytest.raises(ApprovalExpiredError):
        await zero_env.service.resume(thread_id, status="approved", operator="alice")

    assert "approval.decided" not in _events(zero_env.store, thread_id)


async def test_resume_expired_message_mentions_timeout_not_completion(zero_env):
    """消息必须说"超时",不能说"已完成" —— 后者是事实错误。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)

    with pytest.raises(ApprovalExpiredError) as excinfo:
        await zero_env.service.resume(
            result.outcome.thread_id, status="denied", operator="alice"
        )

    message = str(excinfo.value)
    assert "超时" in message
    assert "已完成" not in message


async def test_resume_after_timeout_stays_expired(zero_env):
    """超时不可逆:第二次 resume 仍然 409,且不重复写审计。"""
    result = await zero_env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    for _ in range(2):
        with pytest.raises(ApprovalExpiredError):
            await zero_env.service.resume(thread_id, status="approved", operator="alice")

    assert len(zero_env.store.list_audit(thread_id=thread_id, event=TIMEOUT_EVENT)) == 1


async def test_timed_out_thread_is_irreversible_even_if_cleanup_failed(data_paths, store):
    """清理失败(后端不支持删除)时,超时**依然**不可逆。

    若判定只看 pending 行,删除失败会让第二次 resume 重新拿到 interrupt_id
    —— 一个已经作废的审批会被恢复。所以 _reject_if_expired 先看
    approval.timeout 是否已落库,再看窗口。
    """
    logs, intel = data_paths
    env = _make_env(
        store,
        logs,
        intel,
        approval_timeout=timedelta(0),
        checkpointer=_UndeletableSaver(),
    )
    result = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    with pytest.raises(ApprovalExpiredError):
        await env.service.resume(thread_id, status="approved", operator="alice")
    assert env.has_checkpoint(thread_id)  # 删除没生效

    with pytest.raises(ApprovalExpiredError):
        await env.service.resume(thread_id, status="approved", operator="alice")

    assert len(store.list_audit(thread_id=thread_id, event=TIMEOUT_EVENT)) == 1


async def test_resume_non_expired_thread_still_succeeds(env):
    """回归:未过期的 thread 恢复行为完全不变。"""
    result = await env.service.triage(BRUTE_FORCE_IP)

    resumed = await env.service.resume(
        result.outcome.thread_id, status="approved", operator="alice"
    )

    assert resumed.outcome.status == "completed"
    assert resumed.outcome.approval.status == "approved"


# =====================================================================
# C. failed thread 清理
# =====================================================================

async def test_plan_failure_deletes_residual_checkpoint(data_paths, store, tmp_path):
    """plan 失败会在 checkpoint 里留下 next=('plan',) 的残留快照,清掉它。

    该 thread 是终态(failed),不可能再被恢复 —— 留着它只是内存泄漏,
    而且会让 aget_state 返回一个带 error 的"半死"快照。
    """
    _, intel = data_paths
    env = _make_env(store, tmp_path / "missing.jsonl", intel)

    with pytest.raises(TriageDataUnavailableError):
        await env.service.triage(BRUTE_FORCE_IP)

    failed_thread = store.list_audit(event="plan.failed")[0].thread_id
    assert failed_thread is not None
    assert not env.has_checkpoint(failed_thread)


async def test_resume_after_plan_failure_reports_no_pending_item(data_paths, store, tmp_path):
    """resume 一个失败的 thread:消息必须准确,不能说"审批已完成"。

    失败路径的 checkpoint 现在会被清掉,于是 resume 会落到"有审计但无终态
    审批事件"这一支。原来的消息对它是**事实错误**(它从没完成,是失败了)。
    """
    _, intel = data_paths
    env = _make_env(store, tmp_path / "missing.jsonl", intel)

    with pytest.raises(TriageDataUnavailableError):
        await env.service.triage(BRUTE_FORCE_IP)
    failed_thread = store.list_audit(event="plan.failed")[0].thread_id

    with pytest.raises(NotAwaitingApprovalError) as excinfo:
        await env.service.resume(failed_thread, status="approved", operator="alice")

    message = str(excinfo.value)
    assert "未产生待审批项" in message
    assert "已完成" not in message


async def test_resume_allowed_thread_reports_no_pending_item(env):
    """策略放行的 thread 同样没有待审批项 —— 消息要覆盖这一支。"""
    result = await env.service.triage(LOW_RISK_IP)

    with pytest.raises(NotAwaitingApprovalError) as excinfo:
        await env.service.resume(
            result.outcome.thread_id, status="approved", operator="alice"
        )

    assert "未产生待审批项" in str(excinfo.value)


async def test_resume_completed_thread_message_is_unchanged(env):
    """回归:真正审批完成的 thread 仍然说"已完成"。"""
    result = await env.service.triage(BRUTE_FORCE_IP)
    await env.service.resume(
        result.outcome.thread_id, status="approved", operator="alice"
    )

    with pytest.raises(NotAwaitingApprovalError) as excinfo:
        await env.service.resume(
            result.outcome.thread_id, status="denied", operator="bob"
        )

    assert "已完成" in str(excinfo.value)


async def test_resume_unknown_thread_still_unknown(env):
    """回归:从未存在的 thread 仍是 404 语义(不能被新分支吞掉)。"""
    with pytest.raises(UnknownThreadError):
        await env.service.resume("nope", status="approved", operator="alice")


# ---------- 错误优先级(9.1-A D2)----------

async def test_cleanup_failure_does_not_mask_primary_failure(data_paths, store, tmp_path):
    """清理失败**不得**顶替已经确立的主失败。

    主失败:plan 节点抛错 → TriageDataUnavailableError(→ 503)。
    附带动作:清掉残留的 failed checkpoint。清理失败时,调用方必须仍然
    拿到**原来那个**领域错误 —— 否则一个 503 会变成带 traceback 的 500,
    把真正的原因(数据源不可用)埋掉。

    这条规则只在这个"已在处理主失败"的路径上成立;其它路径的清理失败
    一律响亮抛出(reap / resume 侧都有用例守着)。
    """
    _, intel = data_paths
    env = _make_env(
        store,
        tmp_path / "missing.jsonl",
        intel,
        checkpointer=_ExplodingSaver(),
    )

    with pytest.raises(TriageDataUnavailableError) as excinfo:
        await env.service.triage(BRUTE_FORCE_IP)

    assert str(excinfo.value) == "安全数据源不可用"
    # 主失败的留痕不受清理失败影响
    assert [r.event for r in store.list_audit()] == ["plan.failed"]


async def test_cleanup_failure_does_not_mask_primary_error_cause(data_paths, store, tmp_path):
    """保留因果链:清理失败不得截断 __cause__。

    triage() 的链路是三层:
        FileNotFoundError(数据文件缺失)
          → PlanFailedError(plan 节点的失败契约,消息通用、不含路径)
            → TriageDataUnavailableError(对外领域错误 → 503)
    清理失败若把中间层换掉,排查信息就断了 —— 所以这里逐层验证整条链。
    """
    _, intel = data_paths
    env = _make_env(
        store,
        tmp_path / "missing.jsonl",
        intel,
        checkpointer=_ExplodingSaver(),
    )

    with pytest.raises(TriageDataUnavailableError) as excinfo:
        await env.service.triage(BRUTE_FORCE_IP)

    chain: list[str] = []
    node: BaseException | None = excinfo.value
    while node is not None:
        chain.append(type(node).__name__)
        node = node.__cause__

    assert chain == [
        "TriageDataUnavailableError",
        "PlanFailedError",
        "FileNotFoundError",
    ]


# =====================================================================
# D. 不变量:绝不删 pending(本阶段最重要的一条)
# =====================================================================

async def test_non_expired_pending_thread_survives_reap(env):
    """**本阶段最重要的不变量**:未过期的待审批 thread 绝不能被删。

    删掉 pending 的 checkpoint = 把一个正在等人批的请求凭空抹掉:
    resume() 会变成"checkpoint 丢了",而实际上它只是还没到期。
    """
    result = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    await env.service.reap_expired()

    assert env.has_checkpoint(thread_id)
    assert len(env.store.pending_action_rows(thread_id=thread_id)) == 3
    assert env.store.list_audit(thread_id=thread_id, event=TIMEOUT_EVENT) == []


async def test_allowed_thread_is_not_touched_by_reap(env):
    """completed / allowed 没有 pending 行 → 不在 reap 范围内。

    9.1-A 的清理范围只有 failed 与 timed_out 两个终态;
    completed / allowed 本轮**不清理**(不是"必须保留",只是没做)。
    """
    result = await env.service.triage(LOW_RISK_IP)

    await env.service.reap_expired()

    assert env.has_checkpoint(result.outcome.thread_id)


async def test_resumed_thread_is_not_touched_by_reap(env):
    """人工审批走完的 thread 也不能被 reap 波及。"""
    result = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id
    await env.service.resume(thread_id, status="approved", operator="alice")

    await env.service.reap_expired()

    assert env.has_checkpoint(thread_id)


async def test_pending_thread_survives_full_request_cycle(env):
    """把 pending 的 thread 拖过一整轮其它请求,它必须一直活着。"""
    result = await env.service.triage(BRUTE_FORCE_IP)
    thread_id = result.outcome.thread_id

    await env.service.triage(LOW_RISK_IP)
    await env.service.reap_expired()
    await env.service.triage(BRUTE_FORCE_IP)

    assert env.has_checkpoint(thread_id)
    assert len(env.store.pending_action_rows(thread_id=thread_id)) == 3

    resumed = await env.service.resume(thread_id, status="approved", operator="alice")
    assert resumed.outcome.status == "completed"


# ---------- 结构性护栏:单一删除入口 ----------

def test_only_one_delete_thread_call_site():
    """adelete_thread 在 triage.py 里必须恰好出现 1 次,且在 _drop_checkpoint 内。

    清理散落多处时,"绝不删 pending" 就无法靠审查一处来保证。
    钉住调用点数量与归属,让新增一条删除路径变成红灯,而不是悄悄出现。
    """
    tree = ast.parse(_TRIAGE_SOURCE.read_text(encoding="utf-8"))

    def _delete_calls(node) -> list[ast.Call]:
        return [
            child for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "adelete_thread"
        ]

    assert len(_delete_calls(tree)) == 1

    drop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_drop_checkpoint"
    )
    assert len(_delete_calls(drop)) == 1


def test_timeout_event_name_is_single_sourced():
    """TIMEOUT_EVENT 与 store 的 SQL 字面量必须一致。

    SQL 侧不得不写字面量,Python 侧用常量 —— 两处一旦漂移,
    超时会解除 pending 但判定/写入用的是另一个名字(或反之),
    表现为"reap 每轮重复写审计"这种很难一眼看出的故障。
    """
    store_source = (_REPO_ROOT / "app" / "security" / "store.py").read_text(encoding="utf-8")
    assert f"'{TIMEOUT_EVENT}'" in store_source
