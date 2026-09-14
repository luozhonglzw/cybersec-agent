"""POST /chat 的 API 测试:注入 FakeLLMClient,完全离线。

调用链(FastAPI → SecurityAgent → FakeLLMClient)与生产一致,
只把最底层的 LLM 换成假的 —— 测试覆盖的是 HTTP 层与 Agent 层的契约。
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.agent import SecurityAgent
from app.core.llm import LLMInvocationError


class FakeLLMClient:
    """返回预设回复的假 LLM;raise_error 时抛 LLMInvocationError。"""

    def __init__(self, reply: str = "模拟分析结果", raise_error: bool = False) -> None:
        self.reply = reply
        self.raise_error = raise_error
        self.last_message: str | None = None

    async def chat(self, messages):
        self.last_message = messages[-1].content
        if self.raise_error:
            raise LLMInvocationError("LLM 调用失败(模拟)")
        return self.reply


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
