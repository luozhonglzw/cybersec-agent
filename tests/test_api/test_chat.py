"""POST /chat 的 API 测试:注入 FakeLLMClient,完全离线。

调用链(FastAPI → SecurityAgent → FakeLLMClient)与生产一致,
只把最底层的 LLM 换成假的 —— 测试覆盖的是 HTTP 层与 Agent 层的契约。
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.agent import SecurityAgent
from app.core.llm import LLMInvocationError, FakeLLMClient


def make_client(fake) -> TestClient:
    return TestClient(create_app(agent=SecurityAgent(fake)))


def test_chat_normal_request():
    """正常请求:200,body 为 {"response": ...}。"""
    client = make_client(FakeLLMClient(reply="这是一条模拟回复"))
    resp = client.post("/chat", json={"message": "帮我分析一下最近服务器有没有受到攻击"})

    assert resp.status_code == 200
    assert resp.json() == {"response": "这是一条模拟回复"}


@pytest.mark.parametrize(
    "body",
    [
        {"message": ""},   # 空 message
        {},                # 缺字段
        {"message": 123},  # 类型错误
    ],
)
def test_chat_invalid_request(body):
    """非法请求:Pydantic 校验失败 → 422,不会到达 Agent。"""
    client = make_client(FakeLLMClient())
    resp = client.post("/chat", json=body)

    assert resp.status_code == 422


def test_chat_llm_failure_returns_502():
    """LLM 调用异常:LLMClientError → 502,不把内部堆栈泄露给客户端。"""
    client = make_client(FakeLLMClient(raise_error=True))
    resp = client.post("/chat", json={"message": "分析一下"})

    assert resp.status_code == 502
    assert "unavailable" in resp.json()["detail"]


def test_agent_receives_user_message():
    """API 层透传用户消息给 Agent(经 Fake 记录验证)。"""
    fake = FakeLLMClient()
    client = make_client(fake)
    client.post("/chat", json={"message": "用户问题"})

    assert fake.last_message == "用户问题"


def test_chat_without_agent_returns_503():
    """只注入了 triage_service → /chat 给明确 503,不是带 traceback 的 500。

    与 /triage 的护栏对称(Phase 9.1-A 对齐):此前 /chat 直接取
    request.app.state.agent,缺 agent 时 AttributeError → 500 + traceback,
    而 /triage 缺 service 时是 503。create_app 的 docstring 一直写着
    "反之 /chat 不可用(503)",这条让它成为真话。
    """
    client = TestClient(create_app(triage_service=object()))
    resp = client.post("/chat", json={"message": "分析一下"})

    assert resp.status_code == 503
    assert resp.json()["detail"] == "chat service unavailable"
