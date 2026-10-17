"""SecurityAgent:对外对话入口,控制流委托给 LangGraph。

调用链(Phase 4 起):
    SecurityAgent.chat(message)
        → 构造 [SystemMessage, HumanMessage]
        → graph.ainvoke(AgentState)     (app/core/graph.py 是唯一控制流实现)
        → 从最终 state 提取 AI 文本回答

Phase 3 的手写 ReAct for loop 已删除,由 graph 的
agent / tools / should_continue 三个节点替代。

v0.5.0-S2-C 增加 `run()`(内部接缝)
----------------------------------
`/ask` 需要把"回答里的引用标记"对齐到"本次真实工具结果",而 `chat()` 只
返回一段文本 —— 消息流被丢弃,`ToolMessage` 无从取得。于是把"跑图 + 提取
回答"抽成 `run()`,同时返回 `messages`。

两条纪律:

- `chat()` 的公开签名、返回值、异常与日志事件**完全不变**,它只是
  `run()` 的薄封装;`graph.py` 一行未改。
- `AgentRun.messages` 是**服务端内部**产物(含 system prompt、工具原始输出、
  中间推理),只供引用校验使用,**绝不得**出现在任何 HTTP 响应里。
"""
from dataclasses import dataclass

import structlog
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from app.core.graph import create_agent_graph
from app.core.llm import LLMClient
from app.tools import DEFAULT_TOOLS

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
    "10. 需要给出处置建议时,优先调用处置规划工具获取结构化计划"
    "(动作/优先级/是否需要人工审批),再基于它向用户说明;"
    "是否需要人工审批由工具判定,你不要自行推断"
)

MAX_ITERATIONS_REPLY = "当前分析达到最大工具调用次数,无法在限定步骤内完成分析。"


@dataclass(frozen=True)
class AgentRun:
    """一次图执行的结果。

    字段刻意只有三个,且**不含** `AgentState` 本身:

    - `answer` —— 与 `chat()` 返回的字符串逐字节相同;
    - `messages` —— 本次运行的完整消息流(含 system prompt、`ToolMessage`、
      中间 `AIMessage`)。引用校验需要它来区分"真实工具结果"与"模型自述"。
      **内部产物**:含工具原始输出,不得进入 HTTP 响应或日志;
    - `iterations` —— 业务层迭代次数,仅为保持既有 `agent_chat_completed`
      日志字段不变而暴露。

    不暴露 `AgentState` 是刻意的:`policy_decision` / `approval_request` 之类
    的字段属于 HITL 路径,`/ask` 的图根本不产生它们;把整个 state 交出去
    会让"某天顺手读一个安全字段"变成可预见的演化方向。
    """

    answer: str
    messages: tuple[BaseMessage, ...]
    iterations: int


class SecurityAgent:
    """构造并持有 LangGraph,对外提供与 Phase 3 相同的 chat() 契约。

    参数:
        llm_client: LLMClient(或任何实现 bind_tools/ainvoke 的对象)
        tools: 可选工具列表;省略时使用 app.tools.DEFAULT_TOOLS(单一真相源)
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
        self._tools = tools or list(DEFAULT_TOOLS)
        self._max_iterations = max_iterations
        # 通过公开 bind_tools() 接口注入 LLM;graph 是唯一的控制流实现
        self._graph = create_agent_graph(
            llm_client, tools=self._tools, max_iterations=max_iterations
        )

    async def chat(self, message: str, *, request_id: str | None = None) -> str:
        """执行对话:控制流全部由 graph 完成,这里只做输入组装与输出提取。

        LLM 调用失败(LLMClientError)原样抛出,由 API 层统一转 502;
        工具异常已在 graph 的 tools 节点内按安全契约转为 ToolMessage。

        request_id(Phase 9.3-D):一次 HTTP 请求的运维身份,由 API 边界生成。
        它**不是** agent 推理状态,因此经 graph 的 `configurable` 传入,
        不放进 AgentState。`/chat` 路径没有 thread_id(图未编译 checkpointer),
        所以这里只带 request_id。

        自 v0.5.0-S2-C 起本方法是 `run()` 的薄封装:**签名、返回值、异常与
        两条日志事件全部不变**,只是把消息流丢弃在外。既有调用方(API 端点、
        测试)无需任何改动。
        """
        logger.info(
            "agent_chat_started",
            request_id=request_id,
            user_message_length=len(message),
            max_iterations=self._max_iterations,
        )

        result = await self.run(message, request_id=request_id)

        logger.info(
            "agent_chat_completed",
            request_id=request_id,
            iterations=result.iterations,
        )
        return result.answer

    async def run(
        self,
        message: str,
        *,
        request_id: str | None = None,
        system_prompt: str = SECURITY_ANALYST_SYSTEM_PROMPT,
    ) -> AgentRun:
        """跑一次图,返回回答与**本次运行的消息流**。

        这是 `chat()` 与 `/ask` 共用的执行接缝,本身**不写日志**:两个入口的
        事件名与字段不同(`agent_chat_*` vs `rag_ask_*`),把日志留在各自入口
        比在这里加一个"事件名参数"更清楚。

        `system_prompt` 默认是既有的分析助手提示词,因此 `chat()` 的行为与
        改造前逐字节一致;`/ask` 传入自己的提示词(要求带 `[src:...]` 标记)。
        换提示词**不**改变图的拓扑、工具集或任何安全判定 —— 它只是消息数组
        的第一条。

        异常语义不变:`LLMClientError` 原样抛出(由 API 层转 502),工具异常
        已在 `tools_node` 内被转为 `ToolMessage`。
        """
        final_state = await self._graph.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=message),
                ],
                "iteration_count": 0,
            },
            {"configurable": {"request_id": request_id}},
        )

        messages = final_state["messages"]
        return AgentRun(
            answer=self._extract_final_answer(messages),
            messages=tuple(messages),
            iterations=final_state.get("iteration_count", 0),
        )

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
