"""Phase 6 集成测试:四工具注册 + 日志 → 情报 → 风险分析 → 解释 的完整链路。

Hermetic:本文件的数据由 `test_data_dir` fixture 用 tmp_path 现场生成,
不依赖仓库内 `data/*.jsonl`(该目录被 data/.gitignore 排除,新克隆的
仓库里并不存在,依赖它会导致 fresh clone 上测试失败)。
工具调用通过参数显式注入临时路径,不修改任何全局默认值。
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT, SecurityAgent
from app.core.llm import FakeLLMClient

BRUTE_FORCE_IP = "203.0.113.66"
FAILED_LOGIN_COUNT = 30

# 规则引擎期望值:40(情报恶意) + 15(情报 severity=high) + 30(失败登录 ≥20) = 85
EXPECTED_SCORE = 85
EXPECTED_RISK_LEVEL = "critical"


def _write_logs(path) -> None:
    """生成 FAILED_LOGIN_COUNT 条来自 BRUTE_FORCE_IP 的 login_failed 事件。"""
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
    """生成 BRUTE_FORCE_IP 的恶意情报记录。"""
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


@pytest.fixture
def test_data_dir(tmp_path):
    """最小必要数据:30 条失败登录 + 1 条恶意情报。返回 (logs_path, intel_path)。"""
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)
    return logs, intel


class RiskScriptModel:
    """四步剧本:查日志 → 查情报 → 风险分析 → 解释性最终回答。

    所有工具调用都显式传入临时数据文件路径,使整条链路脱离仓库 data/。
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
                "id": "call_r1",
            }])
        if self.call_count == 2:
            return AIMessage(content="", tool_calls=[{
                "name": "query_threat_intel_tool",
                "args": {
                    "indicator": BRUTE_FORCE_IP,
                    "indicator_type": "ip",
                    "data_path": self._intel,
                },
                "id": "call_r2",
            }])
        if self.call_count == 3:
            return AIMessage(content="", tool_calls=[{
                "name": "analyze_risk_tool",
                "args": {
                    "indicator": BRUTE_FORCE_IP,
                    "logs_path": self._logs,
                    "intel_path": self._intel,
                },
                "id": "call_r3",
            }])
        return AIMessage(
            content="综合评估:该 IP 风险等级为 critical,"
            "依据是 30 次失败登录的日志证据与情报库的 ssh-brute-force 标记。"
        )

    def bind_tools(self, tools):
        return self


def _make_agent(logs_path, intel_path) -> SecurityAgent:
    """每次调用创建全新 agent(剧本模型有状态,不可复用)。"""
    llm = FakeLLMClient("unused")
    llm._model = RiskScriptModel(logs_path, intel_path)
    return SecurityAgent(llm)


def test_default_tools_include_all_four():
    """默认注册四个工具:日志 / 情报 / 风险分析 / 处置规划(无需数据文件)。"""
    agent = SecurityAgent(FakeLLMClient())
    assert {t.name for t in agent._tools} == {
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
        "plan_response_tool",
    }


def test_system_prompt_guides_risk_tool_usage():
    """prompt 第 9 条:优先用风险工具的结构化结果,LLM 负责解释(无需数据文件)。"""
    assert "风险分析工具" in SECURITY_ANALYST_SYSTEM_PROMPT
    assert "解释" in SECURITY_ANALYST_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_full_risk_chain(test_data_dir):
    """日志 → 情报 → 风险分析 → 解释:四步链消息顺序与证据链正确。"""
    logs, intel = test_data_dir
    result = await _make_agent(logs, intel).chat("分析最近 SSH 登录异常")
    state = await _make_agent(logs, intel)._graph.ainvoke({
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
    # 风险工具的结构化结果进入了上下文,且完全由现场数据决定
    assessment = json.loads(msgs[6].content)["assessment"]
    assert assessment["risk_level"] == EXPECTED_RISK_LEVEL
    assert assessment["score"] == EXPECTED_SCORE
    assert assessment["evidence"]["failed_login_count"] == FAILED_LOGIN_COUNT
    assert assessment["evidence"]["threat_intel_malicious"] is True
    # 最终回答引用了等级(Hybrid:工具出数字,LLM 出解释)
    assert "critical" in result
    assert "失败登录" in result
