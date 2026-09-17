"""Phase 5.2-B:多工具串联 + Evidence Fusion 测试。

验证 SecurityAgent(经 LangGraph)能完成:
    日志证据(query_security_logs)
    + 威胁情报证据(query_threat_intel)
    → 综合分析回答

ScriptedTraceModel 模式(与 test_graph.py 一致)驱动 LLM 剧本,
两个工具真实执行(临时 JSONL 数据,离线)。

Hermetic:数据由 `test_data_dir` fixture 用 tmp_path 现场生成,不依赖仓库内
`data/*.jsonl`(该目录被 data/.gitignore 排除,新克隆的仓库里并不存在)。
工具调用通过参数显式注入临时路径,不修改任何全局默认值。
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.agent import SecurityAgent
from app.core.llm import FakeLLMClient

BRUTE_FORCE_IP = "203.0.113.66"
UNKNOWN_IP = "10.9.9.9"
FAILED_LOGIN_COUNT = 12


def _write_logs(path) -> None:
    """生成 FAILED_LOGIN_COUNT 条来自 BRUTE_FORCE_IP 的 login_failed 事件。"""
    with path.open("w", encoding="utf-8") as f:
        for i in range(FAILED_LOGIN_COUNT):
            f.write(json.dumps({
                "timestamp": f"2026-09-10T09:{i:02d}:00Z",
                "event_type": "login_failed",
                "source": "sshd",
                "source_ip": BRUTE_FORCE_IP,
                "username": "admin",
                "status": "failed",
                "severity": "medium",
                "message": "Failed password for admin",
            }) + "\n")


def _write_intel(path) -> None:
    """生成 BRUTE_FORCE_IP 的恶意情报记录(UNKNOWN_IP 不在库中)。"""
    path.write_text(json.dumps({
        "indicator": BRUTE_FORCE_IP,
        "indicator_type": "ip",
        "malicious": True,
        "confidence": 80,
        "severity": "high",
        "tags": ["ssh-brute-force"],
        "source": "test-fixture",
        "first_seen": "2026-09-01T00:00:00Z",
        "last_seen": "2026-09-10T00:00:00Z",
        "description": "SSH brute force source",
    }) + "\n", encoding="utf-8")


@pytest.fixture
def test_data_dir(tmp_path):
    """最小必要数据:12 条失败登录 + 1 条恶意情报。返回 (logs_path, intel_path)。"""
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)
    return logs, intel


class FusionScriptModel:
    """三步融合剧本:查日志 → 查情报 → 综合回答。

    与 test_graph.ScriptedTraceModel 同模式,但跨两个工具,
    并记录每轮调用供断言。工具参数显式注入临时数据路径。
    """
    def __init__(self, logs_path, intel_path):
        self.call_count = 0
        self._logs = str(logs_path)
        self._intel = str(intel_path)
        self.seen_tool_calls: list[dict] = []

    async def ainvoke(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            tc = {
                "name": "query_security_logs_tool",
                "args": {
                    "event_type": "login_failed",
                    "source_ip": BRUTE_FORCE_IP,
                    "data_path": self._logs,
                },
                "id": "call_step1_logs",
            }
            self.seen_tool_calls.append(tc)
            return AIMessage(content="", tool_calls=[tc])
        if self.call_count == 2:
            tc = {
                "name": "query_threat_intel_tool",
                "args": {
                    "indicator": BRUTE_FORCE_IP,
                    "indicator_type": "ip",
                    "data_path": self._intel,
                },
                "id": "call_step2_intel",
            }
            self.seen_tool_calls.append(tc)
            return AIMessage(content="", tool_calls=[tc])
        return AIMessage(
            content="根据日志证据和威胁情报,该 IP 存在高风险 SSH 暴力破解行为。"
        )

    def bind_tools(self, tools):
        return self


def _make_fusion_agent(logs_path, intel_path) -> SecurityAgent:
    """每次调用创建全新 agent(剧本模型有状态,不可复用)。"""
    llm = FakeLLMClient("unused")
    llm._model = FusionScriptModel(logs_path, intel_path)
    return SecurityAgent(llm)


# ---------- 注册验证(无需数据文件) ----------

def test_default_tools_include_both():
    """默认 tools 至少包含日志与威胁情报两个工具(Phase 7 起共四个)。"""
    agent = SecurityAgent(FakeLLMClient())
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
async def test_evidence_fusion_full_chain(test_data_dir):
    """日志 → 情报 → 综合回答:两个工具按顺序调用,消息顺序正确。

    剧本模型有状态,chat() 与 ainvoke() 各用全新 agent。
    """
    logs, intel = test_data_dir
    result = await _make_fusion_agent(logs, intel).chat("分析最近 SSH 登录异常")
    full_state = await _make_fusion_agent(logs, intel)._graph.ainvoke({
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

    # 2. 第二个工具收到正确的查询意图,且指向临时数据文件(hermetic)
    intel_args = msgs[3].tool_calls[0]["args"]
    assert intel_args["indicator"] == BRUTE_FORCE_IP
    assert intel_args["indicator_type"] == "ip"
    assert intel_args["data_path"] == str(intel)

    # 4. 两份证据都进入了上下文,内容完全由现场数据决定
    log_evidence = json.loads(msgs[2].content)
    intel_evidence = json.loads(msgs[4].content)
    assert log_evidence["count"] == FAILED_LOGIN_COUNT
    assert intel_evidence["found"] is True
    assert "ssh-brute-force" in intel_evidence["record"]["tags"]

    # 6. 最终回答包含两类证据的融合
    final = msgs[-1]
    assert not final.tool_calls
    assert "日志证据" in result and "威胁情报" in result


@pytest.mark.asyncio
async def test_fusion_chat_returns_fused_answer(test_data_dir):
    """chat() 对外契约:返回综合文本(非空、非工具内部信息)。"""
    logs, intel = test_data_dir
    result = await _make_fusion_agent(logs, intel).chat("分析最近 SSH 登录异常")
    assert isinstance(result, str)
    assert "SSH 暴力破解" in result
    assert "tool_call" not in result


@pytest.mark.asyncio
async def test_intel_not_found_path_still_terminates(test_data_dir):
    """情报未命中(found=false)时循环仍正常终止于最终回答。"""
    logs, intel = test_data_dir

    class _NotFoundScriptModel:
        def __init__(self):
            self.call_count = 0

        async def ainvoke(self, messages):
            self.call_count += 1
            if self.call_count == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "query_threat_intel_tool",
                    "args": {
                        "indicator": UNKNOWN_IP,
                        "indicator_type": "ip",
                        "data_path": str(intel),
                    },
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
