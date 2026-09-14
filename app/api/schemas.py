"""API 层的 Request / Response Schema。

为什么单独一个文件:API 的数据契约(contract)与业务层解耦,
未来 SecurityAgent 返回结构化结果(风险等级、处置建议)时,
只改这里和路由,不污染 core 层。
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 的请求体。"""

    message: str = Field(min_length=1, description="用户输入的自然语言消息")


class ChatResponse(BaseModel):
    """POST /chat 的响应体。"""

    response: str = Field(description="Agent 的回复文本")
