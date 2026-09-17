"""Phase 7 集成测试:四工具注册 + 日志 → 情报 → 风险 → 处置规划 → 解释 完整链路。

与 test_risk_integration.py 的差别:
本文件用 tmp_path 现场生成数据文件,完全不依赖仓库内 data/*.jsonl ——
该目录被 data/.gitignore 排除,新克隆的仓库里并不存在,
因此本文件是可重复、可离线、不依赖外部数据的测试。
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT, SecurityAgent
from app.core.graph import create_agent_graph
from app.core.llm import FakeLLMClient
from app.tools import DEFAULT_TOOLS

BRUTE_FORCE_IP = "203.0.113.66"
FAILED_LOGIN_COUNT = 25


def _write_logs(path) -> None:
    """生成 25 条来自 BRUTE_FORCE_IP 的 login_failed 事件。"""
    with path.open("w", encoding="utf-8") as f:
        for i in range(FAILED_LOGIN_COUNT):
            f.write(json.dumps({
                "timestamp": f"2026-09-10T08:{i:02d}:00Z",
                "event_type": "login_failed",
                "source": "sshd",
                "source_ip": BRUTE_FORCE_IP,
                "username": "root",
                "status": "failed",
                "severity": "high",
                "message": "Failed password for root",
            }) + "\n")


def _write_intel(path) -> None:
    """生成一条标记恶意的情报记录。"""
    path.write_text(json.dumps({
        "indicator": BRUTE_FORCE_IP,
        "indicator_type": "ip",
        "malicious": True,
        "confidence": 90,
        "severity": "high",
        "tags": ["ssh-brute-force"],
        "source": "test",
        "first_seen": "2026-09-01T00:00:00Z",
        "last_seen": "2026-09-10T00:00:00Z",
        "description": "SSH brute force source",
    }) + "\n", encoding="utf-8")


class ResponseScriptModel:
    """五步剧本:查日志 → 查情报 → 风险分析 → 处置规划 → 解释性最终回答。

    所有工具调用都显式传入 tmp 数据文件路径,使整条链路脱离仓库 data/。
    """
    def __init__(self, logs_path, intel_path):
        self.call_count = 0
        self._logs = str(logs_path)
        self._intel = str(intel_path)

    async def ainvoke(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "query_security_logs_tool",
                "args": {
                    "event_type": "login_failed",
                    "source_ip": BRUTE_FORCE_IP,
                    "data_path": self._logs,
                    "limit": 50,
                },
                "id": "call_p1",
            }])
        if self.call_count == 2:
            return AIMessage(content="", tool_calls=[{
                "name": "query_threat_intel_tool",
                "args": {
                    "indicator": BRUTE_FORCE_IP,
                    "indicator_type": "ip",
                    "data_path": self._intel,
                },
                "id": "call_p2",
            }])
        if self.call_count == 3:
            return AIMessage(content="", tool_calls=[{
                "name": "analyze_risk_tool",
                "args": {
                    "indicator": BRUTE_FORCE_IP,
                    "logs_path": self._logs,
                    "intel_path": self._intel,
                },
                "id": "call_p3",
            }])
        if self.call_count == 4:
            return AIMessage(content="", tool_calls=[{
                "name": "plan_response_tool",
                "args": {
                    "indicator": BRUTE_FORCE_IP,
                    "logs_path": self._logs,
                    "intel_path": self._intel,
                },
                "id": "call_p4",
            }])
        return AIMessage(
            content="综合评估:该 IP 风险等级为 critical,"
            "已生成 5 项处置动作,其中 3 项需人工审批。"
        )

    def bind_tools(self, tools):
        return self


def _make_agent(tmp_path) -> SecurityAgent:
    logs = tmp_path / "logs.jsonl"
    intel = tmp_path / "intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)

    llm = FakeLLMClient("unused")
    llm._model = ResponseScriptModel(logs, intel)
    return SecurityAgent(llm)


# ---------- 工具注册 ----------

def test_default_tools_share_single_source():
    """DEFAULT_TOOLS 是唯一真相源:graph 与 agent 的默认工具集完全一致。"""
    agent = SecurityAgent(FakeLLMClient())
    assert [t.name for t in agent._tools] == [t.name for t in DEFAULT_TOOLS]

    graph = create_agent_graph(FakeLLMClient())
    assert {"agent", "tools"} <= set(graph.get_graph().nodes.keys())


def test_system_prompt_guides_response_planner_usage():
    """prompt 第 10 条:优先用处置规划工具的结构化计划,审批标记不由 LLM 推断。"""
    assert "处置规划工具" in SECURITY_ANALYST_SYSTEM_PROMPT
    assert "人工审批" in SECURITY_ANALYST_SYSTEM_PROMPT


# ---------- 完整链路 ----------

@pytest.mark.asyncio
async def test_full_response_chain(tmp_path):
    """日志 → 情报 → 风险 → 处置规划 → 解释:五步链消息顺序与计划内容正确。"""
    state = await _make_agent(tmp_path)._graph.ainvoke({
        "messages": [HumanMessage(content="分析该 IP 并给出处置建议")],
        "iteration_count": 0,
    })
    msgs = state["messages"]

    # 消息序列:Human → (AI→Tool) ×4 → AI(final)
    assert [type(m) for m in msgs] == [
        HumanMessage,
        AIMessage, ToolMessage,
        AIMessage, ToolMessage,
        AIMessage, ToolMessage,
        AIMessage, ToolMessage,
        AIMessage,
    ]

    # 四个工具按顺序调用
    names = [
        m.tool_calls[0]["name"]
        for m in msgs if isinstance(m, AIMessage) and m.tool_calls
    ]
    assert names == [
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
        "plan_response_tool",
    ]

    # 处置规划工具的结构化结果进入上下文
    plan = json.loads(msgs[8].content)["plan"]
    assert plan["indicator"] == BRUTE_FORCE_IP
    assert plan["risk_level"] == "critical"
    # 计划与内嵌评估不漂移
    assert plan["risk_level"] == plan["assessment"]["risk_level"]
    # 证据链完整:plan → assessment → evidence
    evidence = plan["assessment"]["evidence"]
    assert evidence["failed_login_count"] == FAILED_LOGIN_COUNT
    assert evidence["threat_intel_malicious"] is True
    # 关键动作齐备,审批标记由规则引擎给出
    actions = {a["action_type"]: a for a in plan["actions"]}
    assert {"block_ip", "isolate_host", "reset_credentials"} <= set(actions)
    assert actions["block_ip"]["requires_approval"] is True
    assert actions["reset_credentials"]["reversible"] is False


@pytest.mark.asyncio
async def test_chat_returns_llm_explanation(tmp_path):
    """Hybrid 分工:规则引擎出计划,LLM 出解释性回答。"""
    result = await _make_agent(tmp_path).chat("分析该 IP 并给出处置建议")
    assert "critical" in result
    assert "人工审批" in result
