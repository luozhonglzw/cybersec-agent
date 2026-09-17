"""Phase 5.2-B:多工具串联 + Evidence Fusion 测试。

验证 SecurityAgent(经 LangGraph)能完成:
    日志证据(query_security_logs)
    + 威胁情报证据(query_threat_intel)
    → 综合分析回答

ScriptedTraceModel 模式(与 test_graph.py 一致)驱动 LLM 剧本,
两个工具真实执行(本地 JSONL 数据,离线)。
"""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.agent import SecurityAgent
from app.core.llm import FakeLLMClient

BRUTE_FORCE_IP = "203.0.113.66"


class FusionScriptModel:
    """三步融合剧本:查日志 → 查情报 → 综合回答。

    与 test_graph.ScriptedTraceModel 同模式,但跨两个工具,
    并记录每轮调用供断言。
    """
    def __init__(self):
        self.call_count = 0
        self.seen_tool_calls: list[dict] = []

    async def ainvoke(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            tc = {"name": "query_security_logs_tool",
                  "args": {"event_type": "login_failed", "source_ip": BRUTE_FORCE_IP},
                  "id": "call_step1_logs"}
            self.seen_tool_calls.append(tc)
            return AIMessage(content="", tool_calls=[tc])
        if self.call_count == 2:
            tc = {"name": "query_threat_intel_tool",
                  "args": {"indicator": BRUTE_FORCE_IP, "indicator_type": "ip"},
                  "id": "call_step2_intel"}
            self.seen_tool_calls.append(tc)
            return AIMessage(content="", tool_calls=[tc])
        return AIMessage(
            content="根据日志证据和威胁情报,该 IP 存在高风险 SSH 暴力破解行为。"
        )

    def bind_tools(self, tools):
        return self


def _make_fusion_agent() -> SecurityAgent:
    """每次调用创建全新 agent(剧本模型有状态,不可复用)。"""
    llm = FakeLLMClient("unused")
    llm._model = FusionScriptModel()
    return SecurityAgent(llm)


# ---------- 注册验证 ----------

def test_default_tools_include_both():
    """默认 tools 至少包含日志与威胁情报两个工具(Phase 6 起还有风险分析)。"""
    agent = _make_fusion_agent()
    assert {"query_security_logs_tool", "query_threat_intel_tool"} <= {
        t.name for t in agent._tools
    }


def test_system_prompt_mentions_intel_cross_check():
    """system prompt 包含 IOC 融合指导(不写死必须调用某工具)。"""
    from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT
    assert "IOC" in SECURITY_ANALYST_SYSTEM_PROMPT
    assert "威胁情报" in SECURITY_ANALYST_SYSTEM_PROMPT


# ---------- 完整融合流程 ----------

@pytest.mark.asyncio
async def test_evidence_fusion_full_chain():
    """日志 → 情报 → 综合回答:两个工具按顺序调用,消息顺序正确。

    剧本模型有状态,chat() 与 ainvoke() 各用全新 agent。
    """
    result = await _make_fusion_agent().chat("分析最近 SSH 登录异常")
    full_state = await _make_fusion_agent()._graph.ainvoke({
        "messages": [HumanMessage(content="分析最近 SSH 登录异常")],
        "iteration_count": 0,
    })
    msgs = full_state["messages"]

    types = [type(m) for m in msgs]
    # 3. ToolMessage 顺序:Human → AI(logs) → Tool(log) → AI(intel) → Tool(intel) → AI(final)
    assert types == [HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage, AIMessage]

    # 1. 调用顺序:第一步日志、第二步情报
    assert msgs[1].tool_calls[0]["name"] == "query_security_logs_tool"
    assert msgs[3].tool_calls[0]["name"] == "query_threat_intel_tool"

    # 2. 第二个工具收到正确参数
    assert msgs[3].tool_calls[0]["args"] == {
        "indicator": BRUTE_FORCE_IP, "indicator_type": "ip",
    }

    # 4. 两份证据都进入了上下文
    log_evidence = json.loads(msgs[2].content)
    intel_evidence = json.loads(msgs[4].content)
    assert log_evidence["count"] > 0  # 203.0.113.66 存在失败登录(Phase 2 数据)
    assert intel_evidence["found"] is True
    assert "ssh-brute-force" in intel_evidence["record"]["tags"]

    # 6. 最终回答包含两类证据的融合
    final = msgs[-1]
    assert not final.tool_calls
    assert "日志证据" in result and "威胁情报" in result


@pytest.mark.asyncio
async def test_fusion_chat_returns_fused_answer():
    """chat() 对外契约:返回综合文本(非空、非工具内部信息)。"""
    result = await _make_fusion_agent().chat("分析最近 SSH 登录异常")
    assert isinstance(result, str)
    assert "SSH 暴力破解" in result
    assert "tool_call" not in result


@pytest.mark.asyncio
async def test_intel_not_found_path_still_terminates():
    """情报未命中(found=false)时循环仍正常终止于最终回答。"""
    class _NotFoundScriptModel:
        def __init__(self):
            self.call_count = 0

        async def ainvoke(self, messages):
            self.call_count += 1
            if self.call_count == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "query_threat_intel_tool",
                    "args": {"indicator": "10.9.9.9", "indicator_type": "ip"},
                    "id": "call_nf",
                }])
            return AIMessage(content="未发现该 IOC 的情报记录,无法给出攻击结论。")

        def bind_tools(self, tools):
            return self

    llm = FakeLLMClient("unused")
    llm._model = _NotFoundScriptModel()
    agent = SecurityAgent(llm)

    result = await agent.chat("查一下 10.9.9.9 的情报")
    assert "未发现" in result or "情报" in result
