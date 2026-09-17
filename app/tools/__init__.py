"""工具层统一入口:集中维护默认工具清单。

为什么需要这个模块:
    Phase 7 之前,默认工具列表在 app/core/agent.py 与 app/core/graph.py 各写了
    一份,且 graph 的兜底默认(1 个工具)与 agent 的默认(3 个)不一致 —— 调用方
    一旦省略 tools 参数,会静默拿到一个能力残缺的 graph,属于典型的漂移陷阱。
    现在只有一个真相源:加工具、换顺序,只改这里。

刻意不做的事:
    不引入 ToolRegistry / 插件式注册 / 依赖注入容器。工具数量是个位数,一张
    列表足够;等真实痛点出现再引入复杂度(与 Principle 2「简单优先」一致)。
"""
from langchain_core.tools import BaseTool

from app.tools.query_logs import query_security_logs_tool
from app.tools.query_threat_intel import query_threat_intel_tool
from app.tools.risk_analyzer import analyze_risk_tool
from app.tools.response_planner import plan_response_tool

# SecurityAgent 与 create_agent_graph 共用的默认工具清单(顺序即注册顺序)。
# 顺序有意义:与 prompt 中"先查事实 → 再查情报 → 再评级 → 再规划"的引导一致。
DEFAULT_TOOLS: list[BaseTool] = [
    query_security_logs_tool,
    query_threat_intel_tool,
    analyze_risk_tool,
    plan_response_tool,
]

__all__ = ["DEFAULT_TOOLS"]
