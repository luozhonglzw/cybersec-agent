"""Incident Response Planner —— 规则引擎版处置规划(Phase 7)。

分层(与 risk_analyzer.py 完全同构):
- 核心函数 plan_response(assessment) 是**纯计算**:输入 RiskAssessment,
  输出 ResponsePlan,不读文件、不查库、不调用 LLM —— 同样的评估永远得出
  同样的计划,规则分支可穷举测试、可做 Phase 9 golden set;
- wrapper plan_response_tool(...) 是工具入口:复用 collect_evidence +
  analyze_risk 采集证据并评估,再委托纯函数。

安全约束(重要,勿改):
    工具签名只接受 indicator 等**查询意图**参数,绝不接受 risk_level / score。
    若 LLM 能把等级当参数传进来,它就能幻觉或篡改风险等级 —— 这违反 Phase 6
    确立的"数字与等级不经过 LLM"。等级只能由规则引擎产出。

设计要点:
- 只消费 assessment(含 evidence),不重新推导风险等级。误报抑制等判定已在
  risk_analyzer 完成,此处复用其结果而非重复实现 —— 避免两个真相源;
- risk_level 是粗粒度分档,下方两条修正项用于补足档位内部的关键差异
  (仅情报恶意时分数 40 只到 medium,基础动作集里没有 block_ip)。
  修正项不是冗余,请勿删除。

LLM 的角色:拿到结构化计划后负责解释、取舍与汇报(Hybrid 的叙事侧)。
"""
import json

from app.schemas.response import (
    ActionPriority,
    ActionType,
    ResponseAction,
    ResponsePlan,
)
from app.schemas.risk import RiskAssessment, RiskLevel
from app.tools.query_logs import DEFAULT_DATA_PATH as LOGS_DATA_PATH
from app.tools.query_threat_intel import DEFAULT_DATA_PATH as INTEL_DATA_PATH
from app.tools.risk_analyzer import (
    BRUTE_FORCE_THRESHOLD,
    analyze_risk,
    collect_evidence,
)

# ---- 风险等级 → 基础动作集(确定性映射,Phase 7 先硬编码,不做配置化)----
BASE_ACTIONS: dict[RiskLevel, list[ActionType]] = {
    "none": ["no_action"],
    "low": ["monitor"],
    "medium": ["monitor", "collect_evidence"],
    "high": ["block_ip", "reset_credentials", "escalate", "collect_evidence"],
    "critical": [
        "block_ip",
        "isolate_host",
        "reset_credentials",
        "escalate",
        "collect_evidence",
    ],
}

# ---- 动作固有属性:(是否需人工审批, 是否可回滚)----
# 破坏性或影响业务的动作必须人工审批;reset_credentials 已改密,不可逆。
ACTION_PROPERTIES: dict[ActionType, tuple[bool, bool]] = {
    "no_action": (False, True),
    "monitor": (False, True),
    "collect_evidence": (False, True),
    "escalate": (False, True),
    "block_ip": (True, True),
    "isolate_host": (True, True),
    "reset_credentials": (True, False),
}

# ---- 动作固有紧急度(描述动作本身的影响面,不由 risk_level 决定)----
ACTION_PRIORITY: dict[ActionType, ActionPriority] = {
    "no_action": "low",
    "monitor": "low",
    "collect_evidence": "medium",
    "escalate": "high",
    "block_ip": "high",
    "isolate_host": "critical",
    "reset_credentials": "high",
}

# ---- 动作依据(逐条引用证据,保证每条动作可解释、可审计)----
ACTION_RATIONALE: dict[ActionType, str] = {
    "no_action": "风险评估为无风险,无需处置",
    "monitor": "保持观察,暂不执行影响业务的处置动作",
    "collect_evidence": "保留原始日志与情报证据,供后续复盘与取证",
    "block_ip": "该指标具备恶意特征,建议在网络边界封禁",
    "reset_credentials": "存在爆破或凭据风险,建议强制该账号改密",
    "isolate_host": "风险等级为 critical,建议隔离相关主机以阻断扩散",
    "escalate": "需要人工介入确认与决策",
}


def plan_response(assessment: RiskAssessment) -> ResponsePlan:
    """纯规则引擎:根据风险评估生成处置计划。

    不读文件、不查库、不调用 LLM —— 完全确定性。
    """
    actions: list[ActionType] = list(BASE_ACTIONS[assessment.risk_level])
    evidence = assessment.evidence

    # 修正项 1:情报标记恶意 → 必须封禁。
    # 仅情报恶意时分数 40 只到 medium,而 medium 的基础动作集不含 block_ip。
    if evidence.threat_intel_found and evidence.threat_intel_malicious:
        if "block_ip" not in actions:
            actions.append("block_ip")

    # 修正项 2:失败登录达爆破规模 → 必须改密。
    # 20 次失败登录(+30 分)同样只到 medium,基础动作集不含 reset_credentials。
    if evidence.failed_login_count >= BRUTE_FORCE_THRESHOLD:
        if "reset_credentials" not in actions:
            actions.append("reset_credentials")

    return ResponsePlan(
        indicator=assessment.indicator,
        risk_level=assessment.risk_level,
        summary=_build_summary(assessment, actions),
        actions=[
            ResponseAction(
                action_type=action_type,
                priority=ACTION_PRIORITY[action_type],
                target=assessment.indicator,
                rationale=ACTION_RATIONALE[action_type],
                requires_approval=ACTION_PROPERTIES[action_type][0],
                reversible=ACTION_PROPERTIES[action_type][1],
            )
            for action_type in actions
        ],
        assessment=assessment,
    )


def _build_summary(assessment: RiskAssessment, actions: list[ActionType]) -> str:
    """规则生成的一句话结论(不经过 LLM)。"""
    approvals = sum(1 for a in actions if ACTION_PROPERTIES[a][0])
    return (
        f"指标 {assessment.indicator} 风险等级 {assessment.risk_level}"
        f"(分数 {assessment.score}),建议 {len(actions)} 项动作,"
        f"其中 {approvals} 项需人工审批。"
    )


def _create_tool_wrapper():
    """返回 LangChain 工具实例,职责与 risk_analyzer 的包装器一致。"""
    from langchain_core.tools import tool

    @tool
    def plan_response_tool(
        indicator: str,
        event_type: str | None = None,
        logs_path: str = str(LOGS_DATA_PATH),
        intel_path: str = str(INTEL_DATA_PATH),
    ) -> str:
        """对指定安全指标(IP/域名/Hash)生成结构化处置建议(Incident Response Plan)。

        内部自动采集本地日志证据与威胁情报证据,先做规则风险评估,再据此生成
        处置动作清单(动作类型 / 优先级 / 是否需要人工审批 / 是否可回滚)。

        参数说明:
        - indicator: 规划对象,如 "203.0.113.66"
        - event_type: 可选,限定统计的日志事件类型
        - logs_path / intel_path: 数据文件路径

        返回:JSON 格式的 ResponsePlan(含内嵌 assessment 证据链)。
        """
        try:
            evidence = collect_evidence(indicator, event_type, logs_path, intel_path)
            assessment = analyze_risk(evidence)
            plan = plan_response(assessment)
            return json.dumps({"plan": plan.model_dump(mode="json")})
        except (ValueError, FileNotFoundError) as exc:
            return json.dumps({
                "error": str(exc),
                "type": type(exc).__name__,
                "suggest_retry": True,
            })
        except Exception:
            # 其他错误不暴露内部细节(traceback / 路径)
            return json.dumps({
                "error": "处置规划失败",
                "type": "ResponsePlanningError",
                "suggest_retry": True,
            })

    return plan_response_tool


# 导出工具实例
plan_response_tool = _create_tool_wrapper()
