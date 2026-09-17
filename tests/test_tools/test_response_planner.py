"""Phase 7:Rule-based Response Planner 测试(完全离线)。

分两层验证:
- 纯函数 plan_response(assessment):规则分支穷举,零文件依赖;
- 工具包装器 plan_response_tool:JSON 契约、错误归类、签名安全约束。

其中若干用例专门"锁住"设计决策,防止后人误删:
- 情报恶意 / 爆破规模的修正项(仅靠 risk_level 会漏掉关键动作);
- 工具签名不得暴露 risk_level / score(否则 LLM 可篡改风险等级)。
"""
import json

from app.schemas.risk import RiskAssessment, RiskEvidence
from app.tools.response_planner import (
    ACTION_PROPERTIES,
    ACTION_PRIORITY,
    BASE_ACTIONS,
    plan_response,
    plan_response_tool,
)

IP = "203.0.113.66"


def _evidence(**overrides) -> RiskEvidence:
    base = {
        "indicator": IP,
        "log_event_count": 0,
        "failed_login_count": 0,
        "threat_intel_found": False,
        "threat_intel_malicious": None,
        "threat_intel_tags": [],
        "threat_intel_severity": None,
    }
    base.update(overrides)
    return RiskEvidence(**base)


def _assessment(risk_level: str = "none", score: int = 0, **evidence_overrides) -> RiskAssessment:
    return RiskAssessment(
        indicator=IP,
        risk_level=risk_level,
        score=score,
        confidence=50,
        reasons=["测试构造"],
        evidence=_evidence(**evidence_overrides),
    )


def _types(plan) -> list[str]:
    return [a.action_type for a in plan.actions]


def _by_type(plan) -> dict:
    return {a.action_type: a for a in plan.actions}


# ---------- 确定性与纯函数性 ----------

def test_plan_is_deterministic():
    """同样的评估永远得出同样的计划(纯函数,可做 Phase 9 golden set)。"""
    assessment = _assessment("critical", 85, failed_login_count=25)
    first = plan_response(assessment).model_dump(mode="json")
    second = plan_response(assessment).model_dump(mode="json")
    assert first == second


def test_pure_function_needs_no_files():
    """纯函数零 I/O:手工构造的 assessment 直接可用,不需要任何数据文件。"""
    plan = plan_response(_assessment("low", 15))
    assert _types(plan) == ["monitor"]


# ---------- risk_level → 基础动作集 ----------

def test_risk_none_yields_no_action():
    assert _types(plan_response(_assessment("none", 0))) == ["no_action"]


def test_risk_low_yields_monitor():
    assert _types(plan_response(_assessment("low", 15))) == ["monitor"]


def test_risk_medium_yields_monitor_and_collect():
    assert set(_types(plan_response(_assessment("medium", 40)))) == {
        "monitor", "collect_evidence",
    }


def test_risk_high_includes_block_ip_and_reset():
    types = set(_types(plan_response(_assessment("high", 70))))
    assert {"block_ip", "reset_credentials"} <= types


def test_risk_critical_includes_isolate_host():
    types = set(_types(plan_response(_assessment("critical", 85))))
    assert {"block_ip", "isolate_host", "reset_credentials"} <= types


def test_actions_never_empty_for_any_level():
    """任何等级都必须至少产出一条动作(无风险时是 no_action)。"""
    for level in ("none", "low", "medium", "high", "critical"):
        assert len(plan_response(_assessment(level, 0)).actions) >= 1


def test_no_duplicate_actions():
    """修正项不得产生重复动作。"""
    plan = plan_response(_assessment(
        "high", 70, threat_intel_found=True, threat_intel_malicious=True,
        failed_login_count=25,
    ))
    types = _types(plan)
    assert len(types) == len(set(types))


# ---------- 修正项(锁住设计决策,勿删)----------

def test_malicious_intel_forces_block_ip_at_medium():
    """仅情报恶意时分数 40 只到 medium,而 medium 基础动作集不含 block_ip —— 修正项必须补上。"""
    assert "block_ip" not in BASE_ACTIONS["medium"]

    plan = plan_response(_assessment(
        "medium", 40, threat_intel_found=True, threat_intel_malicious=True,
    ))
    assert "block_ip" in _types(plan)


def test_brute_force_forces_reset_credentials_at_medium():
    """20 次失败登录(+30)同样只到 medium,基础动作集不含 reset_credentials —— 修正项必须补上。"""
    assert "reset_credentials" not in BASE_ACTIONS["medium"]

    plan = plan_response(_assessment("medium", 30, failed_login_count=25))
    assert "reset_credentials" in _types(plan)


def test_trusted_intel_with_brute_force_is_not_no_action():
    """可信情报 + 爆破规模:不得因误报抑制直接判为无需处置(疑似被攻陷的可信主机)。"""
    plan = plan_response(_assessment(
        "low", 10,
        threat_intel_found=True, threat_intel_malicious=False,
        failed_login_count=25,
    ))
    types = _types(plan)
    assert "no_action" not in types
    assert "reset_credentials" in types


# ---------- 动作属性(Phase 8 审批输入)----------

def test_destructive_actions_require_approval():
    """破坏性 / 影响业务的动作必须人工审批。"""
    plan = plan_response(_assessment("critical", 85))
    actions = _by_type(plan)
    for action_type in ("block_ip", "isolate_host", "reset_credentials"):
        assert actions[action_type].requires_approval is True


def test_readonly_actions_do_not_require_approval():
    """只读 / 无副作用动作无需审批(属性表覆盖全部四种,并端到端确认)。"""
    for action_type in ("no_action", "monitor", "collect_evidence", "escalate"):
        assert ACTION_PROPERTIES[action_type][0] is False

    high_actions = _by_type(plan_response(_assessment("high", 70)))
    assert high_actions["collect_evidence"].requires_approval is False
    assert high_actions["escalate"].requires_approval is False

    none_actions = _by_type(plan_response(_assessment("none", 0)))
    assert none_actions["no_action"].requires_approval is False


def test_reset_credentials_is_not_reversible():
    """已改密不可逆 —— 审批决策依赖该标记。"""
    plan = plan_response(_assessment("high", 70))
    assert _by_type(plan)["reset_credentials"].reversible is False


def test_block_ip_is_reversible():
    """封禁可解封。"""
    plan = plan_response(_assessment("high", 70))
    assert _by_type(plan)["block_ip"].reversible is True


def test_every_action_has_rationale_and_target():
    """每条动作都可解释:rationale 非空,且 target 指向被评估指标。"""
    plan = plan_response(_assessment("critical", 85))
    for action in plan.actions:
        assert action.rationale
        assert action.target == IP


def test_action_priority_is_action_intrinsic():
    """priority 描述动作自身影响面,不随 risk_level 变(与 risk_analyzer 的档位解耦)。"""
    actions = _by_type(plan_response(_assessment("critical", 85)))
    assert actions["isolate_host"].priority == "critical"
    assert actions["block_ip"].priority == "high"
    assert actions["collect_evidence"].priority == "medium"
    assert ACTION_PRIORITY["escalate"] == "high"


# ---------- 计划与评估不漂移 ----------

def test_plan_risk_level_matches_assessment():
    """顶层 risk_level 必须与内嵌 assessment 一致(防漂移)。"""
    for level in ("none", "low", "medium", "high", "critical"):
        plan = plan_response(_assessment(level, 0))
        assert plan.risk_level == plan.assessment.risk_level == level


def test_summary_is_rule_generated_and_mentions_level():
    """summary 由规则生成,含等级与分数(不经过 LLM)。"""
    plan = plan_response(_assessment("critical", 85))
    assert "critical" in plan.summary
    assert "85" in plan.summary


# ---------- 工具包装器契约 ----------

def test_tool_name_registered():
    assert plan_response_tool.name == "plan_response_tool"


def test_tool_schema_matches_signature():
    """工具参数与函数签名一致(沿用 test_tool_schema.py 的做法)。"""
    assert set(plan_response_tool.args.keys()) == {
        "indicator", "event_type", "logs_path", "intel_path",
    }


def test_tool_schema_does_not_expose_risk_level():
    """安全约束:LLM 不得输入 risk_level / score,等级只能由规则引擎产出。"""
    assert "risk_level" not in plan_response_tool.args
    assert "score" not in plan_response_tool.args


def test_tool_invoke_returns_valid_plan_json(tmp_path):
    """真实 .invoke() 路径:空数据文件 → 合法 ResponsePlan JSON(no_action)。"""
    logs = tmp_path / "logs.jsonl"
    intel = tmp_path / "intel.jsonl"
    logs.write_text("", encoding="utf-8")
    intel.write_text("", encoding="utf-8")

    raw = plan_response_tool.invoke({
        "indicator": IP,
        "logs_path": str(logs),
        "intel_path": str(intel),
    })
    payload = json.loads(raw)

    assert "error" not in payload
    plan = payload["plan"]
    assert plan["indicator"] == IP
    assert plan["risk_level"] == "none"
    assert [a["action_type"] for a in plan["actions"]] == ["no_action"]
    # 内嵌证据链完整
    assert plan["assessment"]["evidence"]["log_event_count"] == 0


def test_tool_missing_data_file_returns_error_with_retry(tmp_path):
    """数据文件缺失 → 结构化错误 + suggest_retry,不抛异常、不带 traceback。"""
    raw = plan_response_tool.invoke({
        "indicator": IP,
        "logs_path": str(tmp_path / "missing_logs.jsonl"),
        "intel_path": str(tmp_path / "missing_intel.jsonl"),
    })
    payload = json.loads(raw)

    assert payload["suggest_retry"] is True
    assert payload["type"] == "FileNotFoundError"
    assert "error" in payload
    assert "Traceback" not in raw
