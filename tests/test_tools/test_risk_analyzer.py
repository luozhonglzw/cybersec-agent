"""Rule-based Risk Analyzer 测试。

两层验证:
- analyze_risk(evidence) 纯函数:手工构造证据,穷举规则分支,
  不触碰任何数据文件 —— 确定性;
- wrapper 便利接口:基于固定 seed 数据(203.0.113.66 等)端到端采集。
"""
import json
from pathlib import Path

import pytest

from app.schemas.risk import RiskEvidence
from app.tools.risk_analyzer import (
    analyze_risk,
    analyze_risk_tool,
    collect_evidence,
)

BRUTE_FORCE_IP = "203.0.113.66"
TRUSTED_IP = "192.0.2.10"
INTEL_PATH = Path("data/threat_intel.jsonl")


def _evidence(**overrides) -> RiskEvidence:
    base = dict(
        indicator="203.0.113.66",
        log_event_count=0,
        failed_login_count=0,
        threat_intel_found=False,
        threat_intel_malicious=None,
        threat_intel_tags=[],
        threat_intel_severity=None,
    )
    base.update(overrides)
    return RiskEvidence(**base)


# ---------- 纯函数:规则分支 ----------

def test_no_evidence_gives_none():
    """无任何证据 → none + 证据不足 reason。"""
    result = analyze_risk(_evidence())
    assert result.risk_level == "none"
    assert result.score == 0
    assert any("证据不足" in r for r in result.reasons)


def test_malicious_intel_only():
    """仅情报命中(无日志)→ 恶意 +40 起。"""
    result = analyze_risk(_evidence(
        threat_intel_found=True, threat_intel_malicious=True,
        threat_intel_tags=["c2"], threat_intel_severity="medium",
    ))
    assert result.score == 40
    assert result.risk_level == "medium"
    assert any("威胁情报" in r for r in result.reasons)


def test_brute_force_log_pattern():
    """仅日志爆破特征(无情报)→ +30。"""
    result = analyze_risk(_evidence(
        log_event_count=30, failed_login_count=30,
    ))
    assert result.score == 30
    assert result.risk_level == "medium"
    assert any("暴力破解" in r for r in result.reasons)


def test_suspicious_login_band():
    """失败登录 5-19 → +15(可疑,未到爆破阈值)。"""
    result = analyze_risk(_evidence(log_event_count=10, failed_login_count=10))
    assert result.score == 15
    assert result.risk_level == "low"


def test_below_suspicious_threshold():
    """失败登录 <5 → 不加分。"""
    result = analyze_risk(_evidence(log_event_count=3, failed_login_count=3))
    assert result.score == 0
    assert result.risk_level == "none"


def test_full_correlation_critical():
    """日志爆破 + 恶意情报(critical):40+15+30 = 85 → critical。"""
    result = analyze_risk(_evidence(
        log_event_count=30, failed_login_count=30,
        threat_intel_found=True, threat_intel_malicious=True,
        threat_intel_tags=["ssh-brute-force"], threat_intel_severity="critical",
    ))
    assert result.score == 85
    assert result.risk_level == "critical"
    # reasons 同时引用两类证据(证据来源可审计)
    assert any("威胁情报" in r for r in result.reasons)
    assert any("失败登录" in r for r in result.reasons)
    assert result.confidence == 70  # 20 基线 +30 情报 +20 爆破


def test_trusted_intel_forces_downgrade():
    """情报标记可信 → 即使有大量失败登录也强制降级(误报抑制)。"""
    result = analyze_risk(_evidence(
        log_event_count=30, failed_login_count=30,
        threat_intel_found=True, threat_intel_malicious=False,
        threat_intel_tags=["trusted-scan-engine"], threat_intel_severity="info",
    ))
    assert result.score <= 10
    assert result.risk_level in ("none", "low")
    assert any("可信" in r for r in result.reasons)


def test_malicious_intel_not_high_severity():
    """恶意情报但 severity=low → 不加严重度加权(仅 40)。"""
    result = analyze_risk(_evidence(
        threat_intel_found=True, threat_intel_malicious=True,
        threat_intel_severity="low",
    ))
    assert result.score == 40


def test_deterministic():
    """纯函数确定性:同证据两次调用结果完全一致。"""
    evidence = _evidence(
        log_event_count=30, failed_login_count=30,
        threat_intel_found=True, threat_intel_malicious=True,
        threat_intel_severity="high",
    )
    assert analyze_risk(evidence) == analyze_risk(evidence)


def test_evidence_snapshot_embedded():
    """评估结果内嵌证据快照(可复现判定)。"""
    evidence = _evidence(log_event_count=7, failed_login_count=7)
    result = analyze_risk(evidence)
    assert result.evidence == evidence
    assert result.evidence.failed_login_count == 7


def test_max_reachable_score():
    """全部信号命中(恶意+critical 情报+爆破日志)= 85,封顶逻辑不误伤。"""
    result = analyze_risk(_evidence(
        log_event_count=999, failed_login_count=999,
        threat_intel_found=True, threat_intel_malicious=True,
        threat_intel_severity="critical",
    ))
    assert result.score == 85  # 40(恶意)+15(严重度)+30(爆破),min(100) 封顶
    assert result.risk_level == "critical"


# ---------- wrapper:便利接口(固定 seed 数据) ----------

@pytest.fixture(scope="module")
def paths():
    """确保情报数据存在(与 Phase 5.1 一致的固定数据)。"""
    if not INTEL_PATH.exists():
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/seed_threat_intel.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    return Path("data/security_events.jsonl"), INTEL_PATH


def test_collect_evidence_brute_force_ip(paths):
    """203.0.113.66:30 条失败登录 + 情报 critical → high/critical。"""
    evidence = collect_evidence(BRUTE_FORCE_IP, logs_path=paths[0], intel_path=paths[1])
    assert evidence.failed_login_count >= 20
    assert evidence.threat_intel_found is True
    assert evidence.threat_intel_malicious is True
    assert "ssh-brute-force" in evidence.threat_intel_tags

    result = analyze_risk(evidence)
    assert result.risk_level in ("high", "critical")
    assert result.score >= 55


def test_wrapper_output_format(paths):
    """wrapper 返回 {"assessment": {...}},可反序列化,datetime 无残留。"""
    raw = analyze_risk_tool.invoke({
        "indicator": BRUTE_FORCE_IP,
        "logs_path": str(paths[0]),
        "intel_path": str(paths[1]),
    })
    parsed = json.loads(raw)
    assert set(parsed.keys()) == {"assessment"}
    a = parsed["assessment"]
    assert a["risk_level"] in ("high", "critical")
    assert isinstance(a["reasons"], list) and a["reasons"]
    assert a["evidence"]["threat_intel_malicious"] is True


def test_wrapper_trusted_ip(paths):
    """可信 IP → 误报抑制路径:等级 none/low。"""
    raw = analyze_risk_tool.invoke({
        "indicator": TRUSTED_IP,
        "logs_path": str(paths[0]),
        "intel_path": str(paths[1]),
    })
    parsed = json.loads(raw)
    assert parsed["assessment"]["risk_level"] in ("none", "low")


def test_wrapper_unknown_indicator(paths):
    """无任何证据的 IP → none + 证据不足。"""
    raw = analyze_risk_tool.invoke({
        "indicator": "10.9.9.9",
        "logs_path": str(paths[0]),
        "intel_path": str(paths[1]),
    })
    parsed = json.loads(raw)
    assert parsed["assessment"]["risk_level"] == "none"


def test_wrapper_error_format():
    """空 indicator → 错误契约 JSON。"""
    raw = analyze_risk_tool.invoke({"indicator": ""})
    parsed = json.loads(raw)
    assert "error" in parsed
    assert parsed["suggest_retry"] is True
