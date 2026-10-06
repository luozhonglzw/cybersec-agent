"""FastAPI 应用入口:HTTP 层只做协议转换,不碰 LLM。

调用链(职责单向依赖):
    FastAPI endpoint
        ↓ 依赖注入
    SecurityAgent(app/core/agent.py) / TriageService(app/core/triage.py)
        ↓
    LLMClient / SqliteAuditStore / HITL graph
        ↓
    ChatOpenAI / SQLite

测试时通过 create_app(agent=..., triage_service=...) 注入 Fake,
整个 API 测试不需要真实 LLM,也不需要 .env。

本模块是**组合根**:只有这里知道"生产环境怎么把这些零件装起来"。
core 层只声明依赖(构造函数参数),不自己去找依赖。
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import sqlite3
import time
from typing import Literal
import uuid

import structlog
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import InMemorySaver

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ResumeRequest,
    TriageRequest,
    TriageResponse,
)
from app.core.agent import SecurityAgent
from app.core.config import get_settings
from app.core.graph import HitlConfig, create_agent_graph
from app.core.llm import LLMClient, LLMClientError
from app.core.triage import (
    NotAwaitingApprovalError,
    TriageDataUnavailableError,
    TriageError,
    TriageResult,
    TriageService,
    UnknownThreadError,
)
from app.schemas.audit import AuditEvent, AuditRecord
from app.security.store import SqliteAuditStore

logger = structlog.get_logger(__name__)


class AuditStoreUnavailableError(Exception):
    """审计读边界的数据源失败(Phase 9.3-F)。

    刻意是**本模块私有**的窄类型:只有 GET /audit/events 的读路径会抛出它,
    因此对应的异常处理器不会改变 /chat、/triage、/resume 的既有语义。
    这样做而不是注册一个全局 `sqlite3.Error` 处理器,是为了把异常处理
    严格限定在审计读边界内 —— 全局处理器会把**写路径**上的 sqlite 故障
    也一并改写成 503,那是本次未获授权的行为变更。
    """


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时装配进程级依赖,进程内复用。

    为什么在 lifespan 而不是模块顶层创建:
    - 依赖只在服务真正启动时初始化,导入 app 不会产生副作用(也不会读 .env);
    - 测试可以用 create_app(...) 覆盖,不走这里。

    checkpointer 必须**只在这里创建一次**(Phase 8.4 约束 6/7):
    InMemorySaver 是进程内的 checkpoint 存储。创建两次 = 两套互不可见的
    state,暂停在 A 上、去 B 里 resume 只会命中"未知 thread"。
    它是一个**有状态单例**,不是可以随手 new 的配置对象。
    """
    settings = get_settings()
    # audit_db_path 来自 Settings(D5);logs_path / intel_path 不注入,
    # 由工具层默认值提供(HitlConfig 的 None 语义)。
    store = SqliteAuditStore(settings.audit_db_path)
    # 同一实例既供写入路径(HITL 图 + TriageService),也供只读审计端点
    # (GET /audit/events)。端点只调用 list_audit,不会产生新的写入点。
    app.state.audit_store = store

    # 一个 LLMClient 给两条链路共用:bind_tools 每次返回新的 runnable,
    # 不存在互相污染;少建一个 ChatOpenAI 实例。
    llm = LLMClient()
    app.state.agent = SecurityAgent(llm)
    app.state.triage_service = TriageService(
        create_agent_graph(
            llm,
            hitl=HitlConfig(checkpointer=InMemorySaver(), audit_store=store),
        ),
        store,
    )
    yield


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


def _require_store(request: Request) -> SqliteAuditStore:
    """取出进程级 SqliteAuditStore(与 _require_agent / _require_service 同构)。

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


def _to_response(result: TriageResult) -> TriageResponse:
    """TriageResult → HTTP 响应模型(纯协议转换,无业务判断)。"""
    return TriageResponse(
        **result.outcome.model_dump(),
        interrupt_id=result.interrupt_id,
    )


def create_app(
    agent: SecurityAgent | None = None,
    triage_service: TriageService | None = None,
    audit_store: SqliteAuditStore | None = None,
) -> FastAPI:
    """构建 FastAPI 应用。

    参数:
        agent: 已组装好的 SecurityAgent(测试注入 Fake 用);
        triage_service: 已组装好的 TriageService(测试注入 Fake graph 用);
        audit_store: 已组装好的 SqliteAuditStore(测试注入只读审计端点用)。

    注入**任意一个**即视为"调用方接管了装配",lifespan 不再运行 ——
    否则它会去构造真实 LLMClient(需要 .env),测试根本跑不起来。
    代价是只注入 agent 时 /triage 与 /audit/events 不可用(503),以此类推;
    生产路径 create_app() 不带参数,三者都会装配好。
    """
    use_lifespan = agent is None and triage_service is None and audit_store is None
    app = FastAPI(
        title="CyberSec Agent",
        version="0.1.0",
        lifespan=lifespan if use_lifespan else None,
    )
    if agent is not None:
        app.state.agent = agent
    if triage_service is not None:
        app.state.triage_service = triage_service
    if audit_store is not None:
        app.state.audit_store = audit_store

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

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        """对话式安全分析入口。校验交给 Pydantic,业务交给 Agent。"""
        reply = await _require_agent(request).chat(
            payload.message, request_id=_request_id_of(request)
        )
        return ChatResponse(response=reply)

    @app.post("/triage", response_model=TriageResponse)
    async def triage(payload: TriageRequest, request: Request) -> TriageResponse:
        """发起一次 HITL 判定。

        thread_id 由服务端生成(D3):请求体里没有这个字段,客户端无法指定。
        """
        result = await _require_service(request).triage(
            payload.indicator,
            event_type=payload.event_type,
            request_id=_request_id_of(request),
        )
        return _to_response(result)

    @app.post("/resume", response_model=TriageResponse)
    async def resume(payload: ResumeRequest, request: Request) -> TriageResponse:
        """对暂停中的判定给出人工决定,并把图跑完。

        interrupt_id 不在请求体里(D7):服务端从 checkpoint 恢复,
        客户端无法指定"审批的是哪一次暂停"。
        """
        result = await _require_service(request).resume(
            payload.thread_id,
            status=payload.status,
            operator=payload.operator,
            reason=payload.reason,
            request_id=_request_id_of(request),
        )
        return _to_response(result)

    @app.get("/audit/events", response_model=list[AuditRecord])
    async def list_audit_events(
        request: Request,
        thread_id: str | None = None,
        event: AuditEvent | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        order: Literal["desc", "asc"] = Query(default="desc"),
    ) -> list[AuditRecord]:
        """只读查询审计流(Phase 9.3-F)。

        这是**只读**端点:它只调用 SqliteAuditStore.list_audit,不追加任何
        审计(读审计不会再写一条审计)、不触发策略/审批/resume/checkpoint、
        不调用工具 / MCP / provider / 模型,也不发起外部网络请求。

        **本端点没有任何认证** —— 它是只读的,但**不是** "authorized
        endpoint"。措辞上不得暗示调用方已通过身份校验、或只有审计员可见;
        认证留到后续阶段。

        响应是**裸列表** `list[AuditRecord]`,无信封;空结果(含未知
        thread_id / 无匹配过滤)一律 200 + `[]`,不区分"不存在"与"无数据"
        (避免用响应差异探测库里有什么)。

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
        try:
            return store.list_audit(
                thread_id=thread_id,
                event=event,
                limit=limit,
                descending=order == "desc",
            )
        except (sqlite3.Error, ValueError) as exc:
            # sqlite3.Error: 库损坏 / 表缺失 / 锁 / 无法打开等持久化层失败。
            # ValueError: 覆盖 json.JSONDecodeError 与 pydantic ValidationError
            #   (二者都是 ValueError 子类)—— 即"持久化行解码/校验失败"。
            # 两类都在**读边界**收敛成同一个窄类型,由上面的处理器映射成 503。
            raise AuditStoreUnavailableError() from exc

    return app


app = create_app()
