"""SecurityAgent:对外对话入口,控制流委托给 LangGraph。

调用链(Phase 4 起):
    SecurityAgent.chat(message)
        → 构造 [SystemMessage, HumanMessage]
        → graph.ainvoke(AgentState)     (app/core/graph.py 是唯一控制流实现)
        → 从最终 state 提取 AI 文本回答

Phase 3 的手写 ReAct for loop 已删除,由 graph 的
agent / tools / should_continue 三个节点替代。
"""
import structlog
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from app.core.graph import create_agent_graph
from app.core.llm import LLMClient
from app.tools.query_logs import query_security_logs_tool
from app.tools.query_threat_intel import query_threat_intel_tool
from app.tools.risk_analyzer import analyze_risk_tool

logger = structlog.get_logger(__name__)

SECURITY_ANALYST_SYSTEM_PROMPT = (
    "你是一名网络安全运营(SOC)分析助手。"
    "你的职责是帮助分析师理解安全事件、分析日志与威胁情报,"
    "并给出清晰、有依据的判断。"
    "工作原则:"
    "1. 先查询事实,再下结论"
    "2. 工具返回的是事实,不是最终判断"
    "3. 如果证据不足,要明确说明"
    "4. 不要把工具调用过程泄露给用户"
    "5. 对攻击判断给出证据"
    "6. 可以多次调用工具查询不同信息"
    "7. 达到最大工具调用次数时,要说明分析受限"
    "8. 当日志分析发现可疑 IOC(IP/域名/Hash)时,可以查询威胁情报库核实其信誉;"
    "给出结论时综合日志证据与威胁情报证据,并明确说明每项证据的来源"
    "9. 需要给出风险结论时,优先调用风险分析工具获取结构化评估(等级/分数/依据),"
    "再基于其证据与判定依据陈述结论;工具的结构化结果是事实,你的职责是解释它们"
)

MAX_ITERATIONS_REPLY = "当前分析达到最大工具调用次数,无法在限定步骤内完成分析。"


class SecurityAgent:
    """构造并持有 LangGraph,对外提供与 Phase 3 相同的 chat() 契约。

    参数:
        llm_client: LLMClient(或任何实现 bind_tools/ainvoke 的对象)
        tools: 可选工具列表,默认包含 query_security_logs_tool
        max_iterations: 业务层最大迭代次数,默认 5
    调用:
        await agent.chat(message) -> str
    """
    def __init__(
        self,
        llm_client: LLMClient,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 5,
    ) -> None:
        self._llm = llm_client
        self._tools = tools or [
            query_security_logs_tool,
            query_threat_intel_tool,
            analyze_risk_tool,
        ]
        self._max_iterations = max_iterations
        # 通过公开 bind_tools() 接口注入 LLM;graph 是唯一的控制流实现
        self._graph = create_agent_graph(
            llm_client, tools=self._tools, max_iterations=max_iterations
        )

    async def chat(self, message: str) -> str:
        """执行对话:控制流全部由 graph 完成,这里只做输入组装与输出提取。

        LLM 调用失败(LLMClientError)原样抛出,由 API 层统一转 502;
        工具异常已在 graph 的 tools 节点内按安全契约转为 ToolMessage。
        """
        logger.info(
            "agent_chat_started",
            user_message_length=len(message),
            max_iterations=self._max_iterations,
        )

        final_state = await self._graph.ainvoke({
            "messages": [
                SystemMessage(content=SECURITY_ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ],
            "iteration_count": 0,
        })

        answer = self._extract_final_answer(final_state["messages"])
        logger.info("agent_chat_completed", iterations=final_state.get("iteration_count", 0))
        return answer

    @staticmethod
    def _extract_final_answer(messages: list) -> str:
        """从最终 state 的消息历史中提取对外回答。

        正常流程末尾是不带 tool_calls 的 AIMessage,直接取其文本;
        达到 max_iterations 时末尾可能是带 tool_calls 的 AIMessage
        (LLM 还想调工具但被业务上限终止),此时保持 Phase 3 外部行为:
        返回分析受限说明,而不是空字符串。
        """
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            if not msg.tool_calls and msg.content:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
            # 最近一条 AIMessage 仍带 tool_calls → 迭代上限终止
            return MAX_ITERATIONS_REPLY
        return MAX_ITERATIONS_REPLY
