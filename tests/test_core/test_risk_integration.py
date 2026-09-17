"""Phase 6 集成测试:三工具注册 + 日志 → 情报 → 风险分析 → 解释 的完整链路。"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT, SecurityAgent
from app.core.llm import FakeLLMClient

BRUTE_FORCE_IP = "203.0.113.66"


class RiskScriptModel:
    """四步剧本:查日志 → 查情报 → 风险分析 → 解释性最终回答。"""
    def __init__(self):
        self.call_count = 0

    async def ainvoke(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "query_security_logs_tool",
                "args": {"event_type": "login_failed", "source_ip": BRUTE_FORCE_IP},
                "id": "call_r1",
            }])
        if self.call_count == 2:
            return AIMessage(content="", tool_calls=[{
                "name": "query_threat_intel_tool",
                "args": {"indicator": BRUTE_FORCE_IP, "indicator_type": "ip"},
                "id": "call_r2",
            }])
        if self.call_count == 3:
            return AIMessage(content="", tool_calls=[{
                "name": "analyze_risk_tool",
                "args": {"indicator": BRUTE_FORCE_IP},
                "id": "call_r3",
            }])
        return AIMessage(
            content="综合评估:该 IP 风险等级为 critical,"
            "依据是 30 次失败登录的日志证据与情报库的 ssh-brute-force 标记。"
        )

    def bind_tools(self, tools):
        return self


def _make_agent() -> SecurityAgent:
    llm = FakeLLMClient("unused")
    llm._model = RiskScriptModel()
    return SecurityAgent(llm)


def test_default_tools_include_all_four():
    """默认注册四个工具:日志 / 情报 / 风险分析 / 处置规划。"""
    agent = _make_agent()
    assert {t.name for t in agent._tools} == {
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
        "plan_response_tool",
    }


def test_system_prompt_guides_risk_tool_usage():
    """prompt 第 9 条:优先用风险工具的结构化结果,LLM 负责解释。"""
    assert "风险分析工具" in SECURITY_ANALYST_SYSTEM_PROMPT
    assert "解释" in SECURITY_ANALYST_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_full_risk_chain():
    """日志 → 情报 → 风险分析 → 解释:四步链消息顺序与证据链正确。"""
    result = await _make_agent().chat("分析最近 SSH 登录异常")
    state = await _make_agent()._graph.ainvoke({
        "messages": [HumanMessage(content="分析最近 SSH 登录异常")],
        "iteration_count": 0,
    })
    msgs = state["messages"]

    # 消息序列:Human → (AI→Tool) ×3 → AI(final)
    assert [type(m) for m in msgs] == [
        HumanMessage, AIMessage, ToolMessage,
        AIMessage, ToolMessage,
        AIMessage, ToolMessage,
        AIMessage,
    ]
    # 三个工具按顺序调用
    names = [m.tool_calls[0]["name"] for m in msgs if isinstance(m, AIMessage) and m.tool_calls]
    assert names == [
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
    ]
    # 风险工具的结构化结果进入了上下文且等级为 high/critical
    risk_result = json.loads(msgs[6].content)["assessment"]
    assert risk_result["risk_level"] in ("high", "critical")
    # 最终回答引用了等级(Hybrid:工具出数字,LLM 出解释)
    assert "critical" in result
    assert "失败登录" in result
