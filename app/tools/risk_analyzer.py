"""Rule-based Risk Analyzer 工具。

分层(Hybrid 设计的规则侧):
- 核心函数 analyze_risk(evidence) 是**纯计算**:输入 RiskEvidence,
  输出 RiskAssessment,不依赖任何数据源 —— 同样的证据永远得出同样的
  评估,规则分支可穷举测试、可做 Phase 9 golden set;
- wrapper collect_evidence_and_analyze(...) 是便利接口:调用查询工具
  采集证据 → 组装 RiskEvidence → 委托纯函数。采集与分析分离,
  未来可支持 LLM 直接传入 Evidence 调用分析。

LLM 的角色:拿到结构化评估后负责解释、取舍与汇报(Hybrid 的叙事侧)。
"""
import json
from pathlib import Path

from app.schemas.risk import RiskAssessment, RiskEvidence
from app.tools.query_logs import DEFAULT_DATA_PATH as LOGS_DATA_PATH
from app.tools.query_logs import query_security_logs
from app.tools.query_threat_intel import query_threat_intel

# ---- 规则阈值与权重(模块级常量,Phase 6 先硬编码,不做配置化)----
BRUTE_FORCE_THRESHOLD = 20      # 失败登录 ≥ 此值视为爆破特征
SUSPICIOUS_LOGIN_THRESHOLD = 5  # 5-19 视为可疑登录活动

WEIGHT_INTEL_MALICIOUS = 40
WEIGHT_INTEL_TAGS = 15
WEIGHT_BRUTE_FORCE = 30
WEIGHT_SUSPICIOUS_LOGIN = 15

# 分数 → risk_level 映射(区间左闭右开)
SCORE_THRESHOLDS = [
    (80, "critical"),
    (55, "high"),
    (30, "medium"),
    (10, "low"),
    (0, "none"),
]


def analyze_risk(evidence: RiskEvidence) -> RiskAssessment:
    """纯规则引擎:根据证据快照计算结构化风险。

    不读文件、不查库、不调用 LLM —— 完全确定性。
    """
    score = 0
    reasons: list[str] = []
    confidence = 20  # 基线:只有证据本身,无外部佐证

    # 规则 1:威胁情报命中且标记恶意
    if evidence.threat_intel_found and evidence.threat_intel_malicious:
        score += WEIGHT_INTEL_MALICIOUS
        confidence += 30
        reasons.append(
            f"威胁情报标记该指标为恶意(tags={evidence.threat_intel_tags or '无'})"
        )
        if evidence.threat_intel_severity in ("high", "critical"):
            score += WEIGHT_INTEL_TAGS
            reasons.append(f"情报严重级别为 {evidence.threat_intel_severity}")

    # 规则 2:失败登录规模(爆破特征)
    if evidence.failed_login_count >= BRUTE_FORCE_THRESHOLD:
        score += WEIGHT_BRUTE_FORCE
        confidence += 20
        reasons.append(
            f"失败登录 {evidence.failed_login_count} 次(≥{BRUTE_FORCE_THRESHOLD},疑似暴力破解)"
        )
    elif evidence.failed_login_count >= SUSPICIOUS_LOGIN_THRESHOLD:
        score += WEIGHT_SUSPICIOUS_LOGIN
        confidence += 10
        reasons.append(
            f"失败登录 {evidence.failed_login_count} 次(可疑登录活动)"
        )

    # 规则 3:情报明确标记可信 → 强制降级(误报抑制)
    if evidence.threat_intel_found and evidence.threat_intel_malicious is False:
        score = min(score, 10)
        reasons.append("威胁情报标记该指标为可信,风险降级")

    # 规则 4:完全无证据
    if (evidence.log_event_count == 0
            and not evidence.threat_intel_found):
        reasons.append("日志与威胁情报均无相关记录,证据不足")

    score = min(score, 100)
    risk_level = next(
        level for threshold, level in SCORE_THRESHOLDS if score >= threshold
    )
    confidence = min(confidence, 100)

    return RiskAssessment(
        indicator=evidence.indicator,
        risk_level=risk_level,
        score=score,
        confidence=confidence,
        reasons=reasons,
        evidence=evidence,
    )


def collect_evidence(
    indicator: str,
    event_type: str | None = None,
    logs_path: Path | str = LOGS_DATA_PATH,
    intel_path: Path | str = Path("data/threat_intel.jsonl"),
) -> RiskEvidence:
    """便利接口:从本地数据源采集证据,组装 RiskEvidence。"""
    events = query_security_logs(
        source_ip=indicator, event_type=event_type, data_path=logs_path
    ) if event_type else query_security_logs(
        source_ip=indicator, data_path=logs_path
    )
    failed_logins = query_security_logs(
        source_ip=indicator, event_type="login_failed", data_path=logs_path
    )

    intel = query_threat_intel(indicator, data_path=intel_path)

    return RiskEvidence(
        indicator=indicator,
        log_event_count=len(events),
        failed_login_count=len(failed_logins),
        threat_intel_found=intel is not None,
        threat_intel_malicious=intel.malicious if intel else None,
        threat_intel_tags=intel.tags if intel else [],
        threat_intel_severity=intel.severity if intel else None,
    )


def _create_tool_wrapper():
    from langchain_core.tools import tool

    @tool
    def analyze_risk_tool(
        indicator: str,
        event_type: str | None = None,
        logs_path: str = str(LOGS_DATA_PATH),
        intel_path: str = "data/threat_intel.jsonl",
    ) -> str:
        """对指定安全指标(IP/域名/Hash)做结构化风险评估。

        内部自动采集本地日志证据与威胁情报证据,输出风险等级
        (none/low/medium/high/critical)、分数、置信度与判定依据。

        参数说明:
        - indicator: 评估对象,如 "203.0.113.66"
        - event_type: 可选,限定统计的日志事件类型
        - logs_path / intel_path: 数据文件路径

        返回:JSON 格式的 RiskAssessment(含 evidence 证据快照)。
        """
        try:
            evidence = collect_evidence(indicator, event_type, logs_path, intel_path)
            assessment = analyze_risk(evidence)
            return json.dumps({"assessment": assessment.model_dump(mode="json")})
        except (ValueError, FileNotFoundError) as exc:
            return json.dumps({
                "error": str(exc),
                "type": type(exc).__name__,
                "suggest_retry": True,
            })
        except Exception:
            return json.dumps({
                "error": "风险分析失败",
                "type": "RiskAnalysisError",
                "suggest_retry": True,
            })

    return analyze_risk_tool


# 导出工具实例
analyze_risk_tool = _create_tool_wrapper()
