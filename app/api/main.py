"""FastAPI 应用入口:HTTP 层只做协议转换,不碰 LLM。

调用链(职责单向依赖):
    FastAPI endpoint
        ↓ 依赖注入
    SecurityAgent(app/core/agent.py)
        ↓
    LLMClient(app/core/llm.py)
        ↓
    ChatOpenAI

测试时通过 create_app(agent=fake_agent) 注入 Fake,
整个 API 测试不需要真实 LLM,也不需要 .env。
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.core.agent import SecurityAgent
from app.core.llm import LLMClient, LLMClientError

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时创建一次 SecurityAgent(含 LLMClient),进程内复用。

    为什么在 lifespan 而不是模块顶层创建:
    - 依赖只在服务真正启动时初始化,导入 app 不会产生副作用;
    - 测试可以用 create_app(agent=...) 覆盖,不走这里。
    """
    app.state.agent = SecurityAgent(LLMClient())
    yield


def create_app(agent: SecurityAgent | None = None) -> FastAPI:
    """构建 FastAPI 应用。

    参数:
        agent: 已组装好的 SecurityAgent(测试注入 Fake 用);
               生产/开发启动时不传,lifespan 中创建真实 Agent。
    """
    app = FastAPI(
        title="CyberSec Agent",
        version="0.1.0",
        lifespan=lifespan if agent is None else None,
    )
    if agent is not None:
        app.state.agent = agent

    @app.exception_handler(LLMClientError)
    async def llm_error_handler(request: Request, exc: LLMClientError) -> JSONResponse:
        # LLM 初始化/调用失败是上游依赖问题 → 502,不是客户端的错(4xx)
        logger.error("chat_failed_upstream", error_type=type(exc).__name__)
        return JSONResponse(status_code=502, content={"detail": "LLM service unavailable"})

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        """对话式安全分析入口。校验交给 Pydantic,业务交给 Agent。"""
        reply = await request.app.state.agent.chat(payload.message)
        return ChatResponse(response=reply)

    return app


app = create_app()
