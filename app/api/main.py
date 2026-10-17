"""FastAPI 应用入口:HTTP 层只做协议转换,不碰 LLM。

调用链(职责单向依赖):
    FastAPI endpoint
        ↓ 依赖注入
    SecurityAgent(app/core/agent.py) / TriageService(app/core/triage.py)
        ↓
    LLMClient / AuditStore(SQLite 或 PostgreSQL) / HITL graph
        ↓
    ChatOpenAI / SQLite / PostgreSQL

测试时通过 create_app(agent=..., triage_service=...) 注入 Fake,
整个 API 测试不需要真实 LLM,也不需要 .env。

本模块是**组合根**:只有这里知道"生产环境怎么把这些零件装起来"。
core 层只声明依赖(构造函数参数),不自己去找依赖。

审计后端选择(Phase v0.2.0-M1c)也**只在这里**发生一次 —— 见
`build_audit_store`。图节点、TriageService 与端点里没有任何
"是 sqlite 还是 postgres" 的分支;它们只依赖 `AuditStore` 契约。

认证与端点准入(Phase v0.3.0-A3-1)同样在本模块装配:5 条业务路由各挂
一条**显式 FastAPI 依赖**(`require_analyst` / `require_approver` /
`require_principal`),认证密钥环在 `lifespan` 里构建一次并挂到
`app.state.auth_keyring`。文档路由(`/openapi.json` / `/docs` /
`/docs/oauth2-redirect` / `/redoc`)按冻结设计保持公开。

`/ask`(v0.5.0-S2-C)是第五条业务路由:准入与 `/chat` 相同(analyst /
approver),但走**独立的**图与工具集(只有一个只读检索工具)。它的装配
见 `build_rag_ask_service`,编排与 groundedness 见 `app/core/rag_ask.py`。

**对象级授权与可信 actor(Phase v0.3.0-A3-3)也在这里** —— 因为它们是
**HTTP 边界的判定**(状态码是 HTTP 概念),而且需要同时看到已认证主体与
线程归属:

    /triage  校验审批指派(422)→ 生成 thread_id → **先写归属** → 再跑图
    /resume  先读归属做对象授权(404 / 403)→ 再把**已认证 subject**
             当作 operator 交给服务层(权威 actor)
    /audit/events  先按角色收敛读取范围(403 / 404)→ 再读审计内容

三条判定都在**任何副作用之前**完成。`AuditStore` 的 10 方法契约**未改**。
"""
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
import sqlite3
import time
from typing import Literal
import uuid

import aiosqlite
import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from psycopg.conninfo import conninfo_to_dict

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    ResumeRequest,
    RetrievalInfo,
    TriageRequest,
    TriageResponse,
)
from app.core.agent import SecurityAgent
from app.core.config import Settings, get_settings
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import LLMClient, LLMClientError
from app.core.rag_ask import AskOutcome, GroundedAskService
from app.core.triage import (
    NotAwaitingApprovalError,
    TriageDataUnavailableError,
    TriageError,
    TriageResult,
    TriageService,
    UnknownThreadError,
)
from app.rag import (
    DEFAULT_KNOWLEDGE_DIR,
    HashingEmbedder,
    IndexUnavailable,
    RagError,
    build_in_memory_retriever,
    load_corpus,
)
from app.schemas.approval import utc_now
from app.schemas.audit import AuditEvent, AuditRecord
from app.schemas.ownership import ThreadOwnership
from app.security.auth import (
    AuthKeyring,
    Principal,
    require_analyst,
    require_approver,
    require_principal,
)
from app.security.checkpoint import SafeAsyncSqliteSaver, build_checkpoint_serde
from app.security.store import SqliteAuditStore
from app.security.store_postgres import POSTGRES_STORE_ERRORS, PostgresAuditStore
from app.security.store_protocol import AuditStore
from app.tools.search_knowledge import build_search_knowledge_tool, build_stub_knowledge_tool
from app.vectorstore import build_qdrant_retriever

logger = structlog.get_logger(__name__)

#: 审计**读**边界要收敛成 503 的持久化层异常。
#:
#: - `sqlite3.Error` —— 文件库损坏 / 表缺失 / 锁 / 无法打开;
#: - `ValueError`     —— 覆盖 json.JSONDecodeError 与 pydantic.ValidationError
#:   (二者都是 ValueError 子类),即"持久化行解码/校验失败";
#: - `POSTGRES_STORE_ERRORS` —— PostgreSQL 侧的连接/协议/权限失败与连接池
#:   借出超时(见 store_postgres 的说明)。
#:
#: 刻意**不**在这里塞 `Exception`:读路径的 bug 必须继续表现为 500,
#: 不能被伪装成"数据源不可用"。
_STORE_READ_FAILURES: tuple[type[BaseException], ...] = (
    sqlite3.Error,
    ValueError,
    *POSTGRES_STORE_ERRORS,
)


class AuditStoreUnavailableError(Exception):
    """审计读边界的数据源失败(Phase 9.3-F)。

    刻意是**本模块私有**的窄类型:只有 GET /audit/events 的读路径会抛出它,
    因此对应的异常处理器不会改变 /chat、/triage、/resume 的既有语义。
    这样做而不是注册一个全局 `sqlite3.Error` 处理器,是为了把异常处理
    严格限定在审计读边界内 —— 全局处理器会把**写路径**上的 sqlite 故障
    也一并改写成 503,那是本次未获授权的行为变更。
    """


#: Phase v0.3.0-A3-3:对象级授权失败的固定文案。
#:
#: - `_SELF_APPROVAL_DETAIL`(403)—— 属主试图审批自己发起的线程。
#:   用**专属**文案而不是通用的 "insufficient role":后者会让人以为
#:   是角色问题,而事实是"角色对、但对象关系不允许"。属主本来就知道
#:   这条线程存在(是他发起的),所以这里不存在存在性泄露。
#: - `_AUDIT_SCOPE_REQUIRED_DETAIL`(403)—— analyst / approver 未给
#:   `thread_id`。审计范围必须**显式**收敛到一条线程,不允许"不带范围
#:   地全量读取";viewer 不受此限(它的职责就是全量只读)。
#: - `_AUDIT_NOT_FOUND_DETAIL`(404)—— 非 viewer 请求了一条**与他无关**
#:   或**无归属**的线程。文案**不区分**这两种情形(也不区分"线程不存在"),
#:   否则响应差异就成了"这条线程存不存在"的探测预言机。
_SELF_APPROVAL_DETAIL = "self-approval is prohibited"
_AUDIT_SCOPE_REQUIRED_DETAIL = "an explicit thread scope is required for this role"
_AUDIT_NOT_FOUND_DETAIL = "not found"


def _approver_eligible_subjects(request: Request) -> frozenset[str]:
    """配置里持有 `approver` 角色的全部 subject。

    这是"某个 subject 有没有资格被指派为审批人"的**唯一**答案来源 ——
    调用方自述、请求体字段、请求头都不是。密钥环缺失(例如应用不是经
    `lifespan` 装配的)时 fail-closed:给 503 而不是"校验不了就放行"。
    生产路径上 `require_analyst` 已经先要求过密钥环,所以这个分支只在
    测试/装配异常时出现。
    """
    keyring = getattr(request.app.state, "auth_keyring", None)
    if not isinstance(keyring, AuthKeyring):
        raise HTTPException(
            status_code=503, detail="authentication configuration unavailable"
        )
    return keyring.subjects_with_role("approver")


def _validate_approver_assignment(
    request: Request, *, owner: str, approvers: Sequence[str]
) -> tuple[str, ...]:
    """校验 `/triage` 的审批指派,返回**规范序**(字典序)的审批人元组。

    在任何图调用或归属写入**之前**执行;任一条不满足 → 422:

    1. 非空(`TriageRequest.approvers` 的 `min_length=1` 已先挡一层,这里
       再挡一层是为了让"绕过 schema 直接构造"也无处可逃);
    2. 每个都是非空、无前后空白的字符串;
    3. **无重复** —— 重复的指派是调用方的笔误,不是"两个人";
    4. **不得包含发起人自己**(禁止自审批,D-7);
    5. 每个都必须是**已配置的 approver 角色主体**。

    第 4、5 条是这一层的核心:**"谁能审批"必须来自被记录下来的配置事实,
    而不是从角色推导的默认值**。一个未配置或没有 approver 角色的 subject
    被指派进来,意味着这条线程可能**永远没人能审批** —— 那正是冻结设计里
    的 fail-closed 终态,绝不能在**发起时**悄悄埋下。

    返回排序后的元组:与 `ThreadOwnership` 的归一化一致,保证同一指派
    只有一种表示。
    """
    eligible = _approver_eligible_subjects(request)
    seen: set[str] = set()
    for index, subject in enumerate(approvers):
        if not isinstance(subject, str) or not subject.strip():
            raise HTTPException(
                status_code=422,
                detail=f"approvers[{index}] must be a non-empty subject identifier",
            )
        if subject != subject.strip():
            raise HTTPException(
                status_code=422,
                detail=f"approvers[{index}] must not carry surrounding whitespace",
            )
        if subject in seen:
            raise HTTPException(
                status_code=422,
                detail=f"duplicate approver subject at index {index}: {subject!r}",
            )
        seen.add(subject)
        if subject == owner:
            raise HTTPException(status_code=422, detail=_SELF_APPROVAL_DETAIL)
        if subject not in eligible:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"approver {subject!r} is not a configured principal "
                    "holding the approver role"
                ),
            )
    return tuple(sorted(seen))


def _authorize_thread_read(
    store: AuditStore, principal: Principal, thread_id: str
) -> None:
    """`/resume` 的对象级授权 —— 在**任何** graph / checkpointer 访问之前。

    判定(顺序刻意固定):

        ownership 不存在            → 404(未知 / 历史无归属 / 与本主体无关)
        主体 == ownership.owner     → 403(属主自审批,D-7)
        主体 ∉ ownership.approvers  → 404(未被指派到这条线程)

    "未被指派" 与 "线程不存在" 必须**给同一个响应**:把 403 用在这里会
    让任何 approver 都能通过状态码差异枚举"哪些 thread_id 是存在的"。
    属主自审批之所以可以给 403,是因为属主**本来就知道**这条线程存在
    (是他自己发起的),这里没有新增泄露。

    本函数只读归属、不碰 checkpoint / 图 / 审计内容,因此"未授权即拒绝"
    不会产生任何副作用。
    """
    ownership = store.get_thread_ownership(thread_id)
    if ownership is None:
        raise UnknownThreadError("未知的 thread_id")
    if principal.subject == ownership.owner:
        raise HTTPException(status_code=403, detail=_SELF_APPROVAL_DETAIL)
    if principal.subject not in ownership.approvers:
        raise UnknownThreadError("未知的 thread_id")


def _authorize_audit_scope(
    store: AuditStore, principal: Principal, thread_id: str | None
) -> None:
    """`/audit/events` 的读取范围收敛 —— 在**取回审计内容之前**。

    角色矩阵(冻结):

        viewer   全量可读(含历史无归属的线程),`thread_id` 可省可给;
        analyst  **必须**给出 `thread_id`,且只能是**自己拥有**的线程;
        approver **必须**给出 `thread_id`,且只能是**自己拥有或显式被指派
                 审批**的线程。

    403 = 该角色**不允许**做这次读取(缺少显式 thread 范围);
    404 = 允许做,但**这条线程不在他的范围里**(含无归属/不存在)。

    两条刻意守住的边界:
    - 这是**对象授权**,不是"取全量再在 Python 里筛":未授权时根本
      不调用 `list_audit`,因此越权者拿不到任何审计内容;
    - 404 的文案对"无关线程"与"不存在的线程"**完全相同**,不给枚举预言机。
    """
    if principal.role == "viewer":
        return
    if thread_id is None:
        raise HTTPException(status_code=403, detail=_AUDIT_SCOPE_REQUIRED_DETAIL)
    ownership = store.get_thread_ownership(thread_id)
    if ownership is None:
        raise HTTPException(status_code=404, detail=_AUDIT_NOT_FOUND_DETAIL)
    if principal.role == "analyst":
        if ownership.owner != principal.subject:
            raise HTTPException(status_code=404, detail=_AUDIT_NOT_FOUND_DETAIL)
        return
    # approver:自己拥有的,或显式被指派审批的
    if ownership.owner != principal.subject and principal.subject not in (
        ownership.approvers
    ):
        raise HTTPException(status_code=404, detail=_AUDIT_NOT_FOUND_DETAIL)


def build_audit_store(settings: Settings) -> AuditStore:
    """按 `settings.audit_backend` 装配审计后端 —— **唯一的后端选择点**。

    这个函数是组合根的一部分,也是全项目**唯一**读 `audit_backend` 的地方。
    图节点 / TriageService / 端点只依赖 `AuditStore` 契约,因此后端切换
    不会渗透到业务层(不会出现"某个节点里偷偷判断后端"的分支)。

    两条硬约束:
    - **不自动回退**。显式选了 postgres 而 DSN 缺失时,`Settings` 的校验
      已经在启动期拒绝;这里不做 try/except 退回 SQLite —— 那会让审计
      悄悄落到另一个库。
    - **不做 DDL**。`PostgresAuditStore` 只发 SELECT / INSERT,且运行期
      角色(`cybersec_app`)没有任何 DDL 权限;建表与迁移只走 Alembic。

    `PostgresAuditStore` 的构造器**不做任何 I/O**(不连接、不建池),
    连接池在首次操作时惰性开启 —— 因此本函数不会把数据库不可用的失败
    提前到进程启动,失败仍发生在真正需要读/写审计的那一刻,由既有的
    "必需写入 vs best-effort" 契约处理。
    """
    if settings.audit_backend == "postgres":
        dsn = settings.audit_postgres_dsn
        # Settings 的 model_validator 已保证非空;这里是**第二道**闸门,
        # 防止有人绕过 Settings 直接构造一个假的配置对象。
        if dsn is None or not dsn.get_secret_value().strip():
            raise ValueError(
                "audit_backend='postgres' 需要 AUDIT_POSTGRES_DSN;"
                "未提供 DSN 时不会回退到 SQLite。"
            )
        secret = dsn.get_secret_value()
        # 语法校验:不可解析的连接串在**启动期**就响亮失败,而不是等第一次
        # 写审计才炸。刻意 `from None` —— 实测 psycopg 的解析错误文本会内嵌
        # 连接串片段("missing \"=\" after \"...\" in connection info string"),
        # 把 `__cause__` 链上去等于把口令写进 traceback。
        try:
            conninfo_to_dict(secret)
        except Exception:
            raise ValueError(
                "AUDIT_POSTGRES_DSN 不是合法的 libpq 连接串(内容不回显)"
            ) from None
        return PostgresAuditStore(secret)
    return SqliteAuditStore(settings.audit_db_path)


def close_audit_store(store: AuditStore) -> None:
    """释放审计后端的进程级资源(幂等)。

    只有持有连接池的后端需要清理。**刻意不把 `close()` 加进
    `AuditStore` Protocol**:SQLite 版每次操作新开连接、用完即关,
    根本没有可关闭的长期资源 —— 为一个后端的具体需求给所有实现强加
    一个方法,会让契约描述"实现细节"而不是"行为"。这里按具体类型判断。

    `SqliteAuditStore` 无 `close()`,因此走 isinstance 分支而不是
    `getattr` 探测:后者会静默接受任何恰好叫 close 的属性。
    """
    if isinstance(store, PostgresAuditStore):
        store.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时装配进程级依赖,进程内复用;退出时释放后端资源。

    为什么在 lifespan 而不是模块顶层创建:
    - 依赖只在服务真正启动时初始化,导入 app 不会产生副作用(也不会读 .env);
    - 测试可以用 create_app(...) 覆盖,不走这里。

    checkpointer 必须**只在这里创建一次**(Phase 8.4 约束 6/7):
    它是一个**有状态单例**,不是可以随手 new 的配置对象。创建两次 = 两套
    互不可见的 state,暂停在 A 上、去 B 里 resume 只会命中"未知 thread"。

    Phase v0.4.0 S1 把 checkpoint 从进程内内存(`InMemorySaver`)换成**落盘的
    SQLite saver** —— 那是跨进程恢复的**唯一**前提。单例约束因此更强:现在
    "两套 state"指的是**同一个文件被两个连接同时写**,而不是两份内存。
    连接由本函数持有,并在 `finally` 里关闭(不关会让进程退出时挂住)。

    资源归属(Phase v0.2.0-M1c):审计后端的连接池由**本函数**拥有,
    且**每个进程只建一个** —— 绝不在请求处理里建池。`finally` 覆盖
    "正常退出"与"部分启动失败"两条路径:例如 LLMClient() 抛错时,
    已经建好的连接池必须被关掉,否则进程退出前一直占着数据库连接。

    认证密钥环(Phase v0.3.0-A3-1)同样**只在这里**构建一次,并挂到
    `app.state.auth_keyring` 供 `require_principal` 取用。它的配置在
    `get_settings()` 里已经校验过:配置缺失/非法 → 这里直接抛错,
    进程**拒绝启动**(fail-closed)。密钥环本身不可变,构建后不再改动。
    """
    settings = get_settings()
    # 认证边界先于任何业务装配:配置不对就没必要再往下建 store / LLM。
    app.state.auth_keyring = AuthKeyring.from_entries(settings.auth_api_keys)
    # audit_db_path 来自 Settings(D5);logs_path / intel_path 不注入,
    # 由工具层默认值提供(HitlConfig 的 None 语义)。
    store = build_audit_store(settings)
    # 同一实例既供写入路径(HITL 图 + TriageService),也供只读审计端点
    # (GET /audit/events)。端点只调用 list_audit,不会产生新的写入点。
    app.state.audit_store = store
    checkpoint_conn: aiosqlite.Connection | None = None
    try:
        # 持久化 checkpoint(Phase v0.4.0 S1)。连接必须**在运行中的事件循环里**
        # 建立:`AsyncSqliteSaver.__init__` 会调用 `asyncio.get_running_loop()`,
        # 并把同步接口 `run_coroutine_threadsafe` 到那个循环上。
        #
        # 刻意**不**用 `from_conn_string(...)`:那个入口不接受 `serde`,而我们必须
        # 注入严格反序列化白名单(见 `app/security/checkpoint.py`)。
        checkpoint_path = settings.checkpoint_db_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_conn = await aiosqlite.connect(str(checkpoint_path))
        checkpointer = SafeAsyncSqliteSaver(
            checkpoint_conn, serde=build_checkpoint_serde()
        )
        # 启动期建表(journal_mode=WAL + checkpoints/writes)。这里失败 → 进程
        # **拒绝启动**:一个"看起来起来了、其实写不了 checkpoint"的进程会在
        # 第一个 /triage 上失败,而那比启动失败难诊断得多。
        await checkpointer.setup()

        # 一个 LLMClient 给三条链路共用:bind_tools 每次返回新的 runnable,
        # 不存在互相污染;少建一个 ChatOpenAI 实例。
        llm = LLMClient()
        app.state.agent = SecurityAgent(llm)
        # /ask 的装配是**独立的**:它自己建图、自己绑工具(只有一个只读检索
        # 工具)。知识库不可用时这里返回 None → /ask 503,其余端点照常。
        app.state.rag_ask = build_rag_ask_service(llm, settings)
        app.state.triage_service = TriageService(
            create_agent_graph(
                llm,
                hitl=HitlConfig(checkpointer=checkpointer, audit_store=store),
            ),
            store,
        )
        yield
    finally:
        # 两套存储互相独立,关闭顺序无所谓,但**两条**都必须关。
        if checkpoint_conn is not None:
            await checkpoint_conn.close()
        # 幂等:SQLite 审计后端没有可关闭的资源,这里是 no-op。
        close_audit_store(store)
        # 释放 Qdrant Local 存储目录锁(如果装配了该后端)。
        rag_ask_service = getattr(app.state, "rag_ask", None)
        if rag_ask_service is not None:
            rag_ask_service.close()


def _triage_status_for(exc: TriageError) -> int:
    """领域错误 → HTTP 状态码。

    404: thread 从未存在 —— 客户端指错了对象;
    409: 存在但不能恢复(审批已完成 / 审批已超时 / checkpoint 丢失)——
         是状态冲突,不是参数格式错误,所以不用 4xx 里的参数类状态码;
    503: 数据源不可用 —— 部署问题,不是客户端的错;
    500: 兜底。新增 TriageError 子类若忘了归类,会落到这里并被测试发现。

    ApprovalExpiredError 继承 NotAwaitingApprovalError → 409,无需单独分支
    (与 CheckpointLostError 同一处理方式)。
    """
    if isinstance(exc, UnknownThreadError):
        return 404
    if isinstance(exc, NotAwaitingApprovalError):  # 含 CheckpointLostError / ApprovalExpiredError
        return 409
    if isinstance(exc, TriageDataUnavailableError):
        return 503
    return 500


def _require_agent(request: Request) -> SecurityAgent:
    """取出进程级 SecurityAgent(与 _require_service 同构)。

    缺失时给明确的 503,而不是让 AttributeError 变成带 traceback 的 500:
    后者既难排查,又会把内部结构泄露给客户端。
    这个分支只在"注入了 triage_service 但没注入 agent"时出现(测试场景)。

    与 /triage 的护栏对齐(Phase 9.1-A):此前 /chat 直接取
    request.app.state.agent,同一个测试场景下 /triage 返回 503 而 /chat
    返回 500 —— 不对称。create_app 的 docstring 一直写着"反之 /chat 不可用
    (503)",这里让它成为真话。
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="chat service unavailable")
    return agent


def _require_service(request: Request) -> TriageService:
    """取出进程级 TriageService。

    缺失时给明确的 503,而不是让 AttributeError 变成带 traceback 的 500:
    后者既难排查,又会把内部结构泄露给客户端。
    这个分支只在"注入了 agent 但没注入 triage_service"时出现(测试场景)。
    """
    service = getattr(request.app.state, "triage_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="triage service unavailable")
    return service


def _request_id_of(request: Request) -> str | None:
    """取出本次请求的 request_id(中间件在入口写入 request.state)。

    中间件缺失时(例如直接调用端点函数而非走 HTTP)安静返回 None ——
    关联是观测性信息,绝不能成为请求成功与否的前提。
    """
    return getattr(request.state, "request_id", None)


def _require_store(request: Request) -> AuditStore:
    """取出进程级审计后端(与 _require_agent / _require_service 同构)。

    返回 `AuditStore` 契约而不是具体类:端点只用到 list_audit,
    与后端是 SQLite 还是 PostgreSQL 无关。

    缺失时给明确的 503,而不是让 AttributeError 变成带 traceback 的 500:
    后者既难排查,又会把内部结构泄露给客户端。这个分支只在"注入了 agent /
    triage_service 但没注入 audit_store"时出现(测试场景)。

    detail 与读失败共用同一固定文案:审计库不可用是一个部署事实,
    不应因"没装配"与"读不动"而给客户端两套不同的内部信号。
    """
    store = getattr(request.app.state, "audit_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="audit store unavailable")
    return store


def _require_rag_ask(request: Request) -> GroundedAskService:
    """取出进程级 `GroundedAskService`(与 _require_agent 同构)。

    缺失有**两种**来源,给同一个 503:

    1. 部署没有可用知识库(启动期装配失败 → `None`);
    2. 测试只注入了其它依赖,没注入本服务。

    两者对客户端是同一个事实:"这条能力当前不可用"。区分它们只会把
    部署内部状态暴露出去,对调用方没有任何用处。
    """
    service = getattr(request.app.state, "rag_ask", None)
    if service is None:
        raise HTTPException(status_code=503, detail="rag service unavailable")
    return service


def _to_response(result: TriageResult) -> TriageResponse:
    """TriageResult → HTTP 响应模型(纯协议转换,无业务判断)。"""
    return TriageResponse(
        **result.outcome.model_dump(),
        interrupt_id=result.interrupt_id,
    )


def _to_ask_response(outcome: AskOutcome) -> AskResponse:
    """AskOutcome → HTTP 响应模型(纯协议转换,无业务判断)。

    这里**只**搬运 S1 冻结的五个字段。服务端内部还有两样东西刻意不出现:

    - `unverified_citations` / `malformed_markers` —— 它们**参与** groundedness
      判定(N-4),但 N-5 要求保持 S1 冻结的响应体,所以只留在服务端日志里;
    - `messages` / 工具原始输出 —— 含 system prompt 与未截断的工具结果,
      交出去等于把"模型看到了什么"变成对外契约。
    """
    return AskResponse(
        answer=outcome.answer,
        grounded=outcome.grounded,
        citations=[
            Citation(
                citation_id=citation.citation_id,
                source_id=citation.source_id,
                source_type=citation.source_type,
                title=citation.title,
                url=citation.url,
                score=citation.score,
            )
            for citation in outcome.citations
        ],
        invalid_citations=list(outcome.invalid_citations),
        retrieval=RetrievalInfo(
            status=outcome.status,
            hit_count=outcome.hit_count,
            top_k=outcome.top_k,
        ),
    )


def build_rag_ask_service(
    llm_client: LLMClient,
    settings: Settings | None = None,
) -> GroundedAskService | None:
    """装配 `/ask` 专用链路:一个只读检索工具 + 一个独立对话图。

    **显式 opt-in(N-7)**:`SecurityAgent(llm, tools=[tool])` 传入的是**非空**
    列表,而 `SecurityAgent.__init__` 只在 `tools` 为假值时才回退到
    `DEFAULT_TOOLS` —— 因此这条链路里 `query_security_logs_tool` /
    `query_threat_intel_tool` / `analyze_risk_tool` / `plan_response_tool`
    一个都不在。它不是"默认工具集减去几个",而是"只有一个"。

    为什么不复用 `/chat` 的 agent:`/chat` 用的是四个默认工具(含
    `plan_response_tool` 这类会产出处置计划的工具)。`/ask` 的语义是
    "依据知识库回答并给出引用",把处置类工具挂进来既扩大能力面,也让
    "回答里的引用"可能来自与知识库无关的工具输出。

    装配失败(语料缺失 / 清单不符 / 索引不匹配)→ 返回 `None`,
    `/ask` 给出固定的 **503**;`/chat`、`/triage` 与 `/audit/events`
    完全不受影响。这是刻意的降级方向:知识库坏掉不该让整个进程起不来。

    qdrant-local 索引不可用时**不**返回 None:捕获 ``IndexUnavailable``,
    注入一个总是返回 ``rag_error`` 的 stub 工具,让查询时得到
    ``status="unavailable"`` 与 ``grounded=false``(S1 冻结语义)。
    """
    try:
        corpus = load_corpus(DEFAULT_KNOWLEDGE_DIR)
        if settings is not None and settings.rag_backend == "qdrant-local":
            retriever = build_qdrant_retriever(
                corpus, HashingEmbedder(), settings.rag_qdrant_path
            )
        else:
            retriever = build_in_memory_retriever(corpus, HashingEmbedder())
    except IndexUnavailable as exc:
        # qdrant-local 索引损坏/缺失:用 stub 工具让查询时返回 unavailable,
        # 而不是在装配期直接 503。
        logger.warning(
            "rag_ask_unavailable", reason=exc.reason, backend="qdrant-local"
        )
        stub_tool = build_stub_knowledge_tool(exc.reason)
        return GroundedAskService(
            SecurityAgent(llm_client, tools=[stub_tool]),
        )
    except RagError as exc:
        # 只记原因码:路径、集合名、语料内容都不进日志。
        logger.warning("rag_ask_unavailable", reason=exc.reason)
        return None
    tool = build_search_knowledge_tool(retriever)
    store = getattr(retriever, "store", None)
    return GroundedAskService(
        SecurityAgent(llm_client, tools=[tool]),
        store=store,
    )


def create_app(
    agent: SecurityAgent | None = None,
    triage_service: TriageService | None = None,
    audit_store: AuditStore | None = None,
    rag_ask: GroundedAskService | None = None,
) -> FastAPI:
    """构建 FastAPI 应用。

    参数:
        agent: 已组装好的 SecurityAgent(测试注入 Fake 用);
        triage_service: 已组装好的 TriageService(测试注入 Fake graph 用);
        audit_store: 已组装好的**任意** AuditStore 实现(测试注入只读审计
                     端点用)。类型是契约而不是 `SqliteAuditStore` ——
                     PostgreSQL 后端同样可以被注入;
        rag_ask: 已组装好的 `/ask` 服务(测试注入脚本化 LLM 与固定检索夹具用)。

    注入**任意一个**即视为"调用方接管了装配",lifespan 不再运行 ——
    否则它会去构造真实 LLMClient(需要 .env),测试根本跑不起来。
    代价是只注入 agent 时 /triage 与 /audit/events 不可用(503),以此类推;
    生产路径 create_app() 不带参数,四者都会装配好(后端由
    Settings.audit_backend 决定)。

    注意:注入式装配**不**接管资源释放 —— 若注入的是 PostgresAuditStore,
    由注入方自己负责 `close()`(lifespan 没运行,不会替你关池)。
    """
    use_lifespan = (
        agent is None
        and triage_service is None
        and audit_store is None
        and rag_ask is None
    )
    app = FastAPI(
        title="CyberSec Agent",
        version="0.4.0",
        lifespan=lifespan if use_lifespan else None,
    )
    if agent is not None:
        app.state.agent = agent
    if triage_service is not None:
        app.state.triage_service = triage_service
    if audit_store is not None:
        app.state.audit_store = audit_store
    if rag_ask is not None:
        app.state.rag_ask = rag_ask

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        """HTTP 关联边界(Phase 9.3-D)。

        为**每一个**请求生成一个不透明的 request_id(uuid4().hex),写入
        `request.state` 供端点显式取用,并发两条边界事件:

            http.request.started   /   http.request.completed

        安全约束(冻结):
        - 绝不记录请求体 / query 值 / 头 / Authorization / cookie / 提示词;
        - request_id 不从用户输入、thread_id、IP、头或时间戳派生;
        - 不采信调用方传入的 request id(本阶段无 X-Request-ID 协议);
        - 异常逃逸时**原样抛出**,HTTP 语义完全交给既有处理器 ——
          这里只为它记一条带关联的终态事件,不新增状态码映射。
        """
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        logger.info(
            "http.request.started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "http.request.completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                error_type=type(exc).__name__,
            )
            raise
        logger.info(
            "http.request.completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    @app.exception_handler(LLMClientError)
    async def llm_error_handler(request: Request, exc: LLMClientError) -> JSONResponse:
        # LLM 初始化/调用失败是上游依赖问题 → 502,不是客户端的错(4xx)
        logger.error(
            "chat_failed_upstream",
            request_id=_request_id_of(request),
            error_type=type(exc).__name__,
        )
        return JSONResponse(status_code=502, content={"detail": "LLM service unavailable"})

    @app.exception_handler(TriageError)
    async def triage_error_handler(request: Request, exc: TriageError) -> JSONResponse:
        """领域错误 → HTTP。detail 只回传领域消息(不含路径等内部信息)。"""
        status_code = _triage_status_for(exc)
        logger.warning(
            "triage_rejected",
            request_id=_request_id_of(request),
            status_code=status_code,
            error_type=type(exc).__name__,
        )
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.exception_handler(AuditStoreUnavailableError)
    async def audit_store_error_handler(
        request: Request, exc: AuditStoreUnavailableError
    ) -> JSONResponse:
        """审计读失败 → 503;detail 是固定文案。

        窄范围:只在 GET /audit/events 的读路径上被抛出,因此不会波及
        写路径或其它端点的既有错误语义。

        绝不回传:绝对路径、原始 SQLite 异常文本、repr(exc)、traceback、
        环境变量或凭据。只记录**异常类型名**(不是消息)用于排障关联。
        """
        logger.error(
            "audit_read_failed",
            request_id=_request_id_of(request),
            error_type=type(exc.__cause__).__name__,
        )
        return JSONResponse(
            status_code=503, content={"detail": "audit store unavailable"}
        )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        # 准入:analyst / approver。依赖而不是中间件 —— 这样既有的 API 测试
        # 可以用 app.dependency_overrides 换掉认证缝,而不必改动它们自己。
        dependencies=[Depends(require_analyst)],
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        """对话式安全分析入口。校验交给 Pydantic,业务交给 Agent。

        **准入是 analyst / approver,仅此而已**(Phase v0.3.0-A3-3 明确):
        本端点**不**经过 HITL 图、**不**做策略判定、**不**写审计、**不**要求
        审批 —— 它是一条纯 LLM 对话路径。不得把它描述成"受策略/审批保护",
        那会让调用方以为这里的输出已经过人工把关。
        """
        reply = await _require_agent(request).chat(
            payload.message, request_id=_request_id_of(request)
        )
        return ChatResponse(response=reply)

    @app.post(
        "/ask",
        response_model=AskResponse,
        # 与 /chat 同一准入:analyst / approver。viewer 只持只读审计权限。
        dependencies=[Depends(require_analyst)],
    )
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        """带引用的知识库问答:检索 → 回答 → 引用校验 → 结构化 groundedness。

        **这条端点回答什么、不回答什么**:

        - 回答"依据知识库检索到的内容,这个问题能得出什么",并在
          `citations` 里给出可点击的来源;
        - **不**回答"这个回答是否正确"。`grounded=true` 只是**结构性**判定:
          本次检索成功、至少一条引用、且没有任何编造 / 存疑 / 写错的标记。
          一条引用可以完全合法却与问题无关 —— 哈希检索对无关问题也会给出
          非零分数,因此"检索有命中"既不是"有答案"的证据,也不是 grounded 的
          理由。

        **准入**:analyst / approver(401 / 403)。本端点**不**经过 HITL 图、
        不做策略判定、不写审计、不要求审批 —— 它与 `/chat` 一样是只读对话
        路径,只是多了检索与引用校验。不得把它描述成"经过人工把关"。

        **失败语义(S1 冻结矩阵)**:

            RAG 未装配 / 知识库不可用   → 503 {"detail":"rag service unavailable"}
            索引陈旧 / 证据自相矛盾     → 200, grounded=false, status=unavailable
            检索无命中                  → 200, grounded=false, status=empty
            模型未调用检索工具          → 200, grounded=false, status=not_used
            回答里有编造 / 写错的引用   → 200, grounded=false
            模型 / provider 失败        → 502 {"detail":"LLM service unavailable"}

        外部知识源不可用时**不**返回 5xx:那是"服务坏了"的信号,而"这次没有
        依据"是一个正常的业务结论。只有 LLM 本身不可用才是 502。

        **响应里刻意没有的东西**:`messages`、工具原始输出、索引路径、
        `unverified_citations` / `malformed_markers`(它们参与 groundedness
        判定,但不进 S1 冻结的响应体)、任何 traceback。
        """
        service = _require_rag_ask(request)
        outcome = await service.ask(
            payload.question, request_id=_request_id_of(request)
        )
        return _to_ask_response(outcome)

    @app.post("/triage", response_model=TriageResponse)
    async def triage(
        payload: TriageRequest,
        request: Request,
        principal: Principal = Depends(require_analyst),
    ) -> TriageResponse:
        """发起一次 HITL 判定。

        thread_id 由服务端生成(D3):请求体里没有这个字段,客户端无法指定。
        A3-3 起它在这里生成 —— 因为**归属必须在跑图之前写好**,而归属要带
        thread_id。生成者是服务端(HTTP 边界),不是客户端,所以 D3 的性质
        (客户端无法指定 thread_id)没有改变。

        **执行顺序(冻结,不得重排)**:

            1. 认证 + 角色准入(依赖)—— 401 / 403;
            2. body 校验 —— 缺 `approvers` 等 → 422;
            3. **审批指派校验** —— 空 / 重复 / 自指派 / 非配置 approver → 422;
            4. 取 store / service —— 缺失 → 503;
            5. **写归属**(owner = 已认证 subject,审批集合 = 规范序);
            6. 调用服务层跑图。

        第 3 步必须在第 5、6 步之前:任何一次非法的指派都不得留下归属行,
        更不得触发图执行。"先校验、再落库、最后才跑图"是这里唯一正确的顺序。
        """
        approvers = _validate_approver_assignment(
            request, owner=principal.subject, approvers=payload.approvers
        )
        service = _require_service(request)
        store = _require_store(request)
        thread_id = uuid.uuid4().hex
        store.record_thread_ownership(
            ThreadOwnership(
                thread_id=thread_id,
                owner=principal.subject,
                approvers=approvers,
                created_at=utc_now(),
            )
        )
        result = await service.triage(
            payload.indicator,
            event_type=payload.event_type,
            request_id=_request_id_of(request),
            thread_id=thread_id,
        )
        return _to_response(result)

    @app.post("/resume", response_model=TriageResponse)
    async def resume(
        payload: ResumeRequest,
        request: Request,
        principal: Principal = Depends(require_approver),
    ) -> TriageResponse:
        """对暂停中的判定给出人工决定,并把图跑完。

        interrupt_id 不在请求体里(D7):服务端从 checkpoint 恢复,
        客户端无法指定"审批的是哪一次暂停"。

        **两道闸门(Phase v0.3.0-A3-3)**:

        1. **角色准入**(依赖 `require_approver`)—— 不是已认证 approver → 401/403;
        2. **对象级授权**(`_authorize_thread_read`)—— 必须在
           `_validate_resumable` 与任何 graph / checkpointer 访问**之前**:
           未指派 → 404,属主自审批 → 403。

        **权威 actor**:`payload.operator` 是调用方自述值,**被忽略**;
        交给服务层的 operator 是 `principal.subject`(已认证主体)。
        因此审计 `actor` 不再能被客户端伪造。字段本身保留只为线上请求的
        错误兼容性(缺字段仍 422),它不影响任何判定。
        """
        store = _require_store(request)
        service = _require_service(request)
        _authorize_thread_read(store, principal, payload.thread_id)
        result = await service.resume(
            payload.thread_id,
            status=payload.status,
            operator=principal.subject,
            reason=payload.reason,
            request_id=_request_id_of(request),
        )
        return _to_response(result)

    @app.get("/audit/events", response_model=list[AuditRecord])
    async def list_audit_events(
        request: Request,
        principal: Principal = Depends(require_principal),
        thread_id: str | None = None,
        event: AuditEvent | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        order: Literal["desc", "asc"] = Query(default="desc"),
    ) -> list[AuditRecord]:
        """只读查询审计流(Phase 9.3-F;读取范围收敛于 v0.3.0-A3-3)。

        这是**只读**端点:它只调用 AuditStore.list_audit,不追加任何
        审计(读审计不会再写一条审计)、不触发策略/审批/resume/checkpoint、
        不调用工具 / MCP / provider / 模型,也不发起外部网络请求。
        `AuditStore` 是契约:后端是 SQLite 还是 PostgreSQL 对本端点透明。

        **准入(Phase v0.3.0-A3-1)**:要求已认证主体
        (`require_principal`)—— viewer / analyst / approver 三者皆可。

        **读取范围(Phase v0.3.0-A3-3,冻结矩阵)**:

            viewer   全量可读,含**历史无归属**的线程;`thread_id` 可省可给;
            analyst  必须给出 `thread_id`,且只能是**自己拥有**的线程;
            approver 必须给出 `thread_id`,且只能是**自己拥有或显式被指派
                     审批**的线程。

        analyst / approver 不带 `thread_id` → **403**;带了但线程不在自己的
        范围里(含无归属 / 不存在)→ **404**。授权在**取回任何审计内容之前**
        完成 —— 不做"先全量取回再在 Python 里过滤",那样越权者虽然看不到
        结果,却已经让服务端把不属于他的数据读了出来。

        响应是**裸列表** `list[AuditRecord]`,无信封;对**已授权**的范围,
        空结果一律 200 + `[]`。

        参数刻意只有四个,且都是**声明式**的:
            thread_id / event  —— 等值过滤(全部参数绑定,无字符串拼接);
            limit              —— 1..200(边界由本层校验),SQL 层 LIMIT;
            order              —— desc(默认) / asc,方向由固定程序逻辑选择。

        刻意**不暴露**的输入:incident_id / approval_id / request_id /
        interrupt_id / actor / outcome / plan_digest / offset / cursor,
        以及任何 db_path / database_path / sqlite_path / file_path /
        raw SQL —— 查询契约里没有路径参数,因此未声明的 path-like 参数
        无法把读取重定向到别的库。

        校验失败一律 422(FastAPI 默认);审计库读失败 → 503 固定文案。
        """
        store = _require_store(request)
        _authorize_audit_scope(store, principal, thread_id)
        try:
            return store.list_audit(
                thread_id=thread_id,
                event=event,
                limit=limit,
                descending=order == "desc",
            )
        except _STORE_READ_FAILURES as exc:
            # 两类后端各自的**持久化层**失败都在读边界收敛成同一个窄类型,
            # 由上面的处理器映射成 503:
            #   sqlite3.Error          —— 库损坏 / 表缺失 / 锁 / 无法打开;
            #   ValueError             —— json.JSONDecodeError 与 pydantic
            #                             ValidationError(持久化行解码/校验失败);
            #   POSTGRES_STORE_ERRORS  —— psycopg 的连接/协议/权限失败与连接池
            #                             借出超时(PoolTimeout 是其子类)。
            # 注意这里**不**捕获 Exception:读路径自身的 bug 必须继续是 500。
            raise AuditStoreUnavailableError() from exc

    return app


app = create_app()
