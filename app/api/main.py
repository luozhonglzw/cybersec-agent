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
import time
import uuid

import structlog
from fastapi import FastAPI, HTTPException, Request
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
from app.security.store import SqliteAuditStore

logger = structlog.get_logger(__name__)


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


def _to_response(result: TriageResult) -> TriageResponse:
    """TriageResult → HTTP 响应模型(纯协议转换,无业务判断)。"""
    return TriageResponse(
        **result.outcome.model_dump(),
        interrupt_id=result.interrupt_id,
    )


def create_app(
    agent: SecurityAgent | None = None,
    triage_service: TriageService | None = None,
) -> FastAPI:
    """构建 FastAPI 应用。

    参数:
        agent: 已组装好的 SecurityAgent(测试注入 Fake 用);
        triage_service: 已组装好的 TriageService(测试注入 Fake graph 用)。

    注入**任意一个**即视为"调用方接管了装配",lifespan 不再运行 ——
    否则它会去构造真实 LLMClient(需要 .env),测试根本跑不起来。
    代价是只注入 agent 时 /triage 不可用(503),反之 /chat 不可用;
    生产路径 create_app() 不带参数,两者都会装配好。
    """
    use_lifespan = agent is None and triage_service is None
    app = FastAPI(
        title="CyberSec Agent",
        version="0.1.0",
        lifespan=lifespan if use_lifespan else None,
    )
    if agent is not None:
        app.state.agent = agent
    if triage_service is not None:
        app.state.triage_service = triage_service

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

    return app


app = create_app()
