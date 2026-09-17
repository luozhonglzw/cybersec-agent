"""Rule-based Risk Analyzer 测试。

两层验证:
- analyze_risk(evidence) 纯函数:手工构造证据,穷举规则分支,
  不触碰任何数据文件 —— 确定性;
- wrapper 便利接口:用 tmp_path 现场生成最小数据(203.0.113.66 等)端到端采集。

Hermetic:本文件的数据由 `paths` fixture 用 tmp_path 现场生成,不依赖仓库内
`data/*.jsonl`(该目录被 data/.gitignore 排除,新克隆的仓库里并不存在,
依赖它会导致 fresh clone 上测试失败,且原实现会通过 subprocess 回写真实 data/)。
所有工具调用显式注入临时路径,不修改任何全局默认值。
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
UNKNOWN_IP = "10.9.9.9"
FAILED_LOGIN_COUNT = 30


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


# ---------- wrapper:便利接口(tmp_path 现场数据,hermetic) ----------

def _write_logs(path: Path) -> None:
    """生成 FAILED_LOGIN_COUNT 条来自 BRUTE_FORCE_IP 的 login_failed 事件。"""
    with path.open("w", encoding="utf-8") as f:
        for i in range(FAILED_LOGIN_COUNT):
            f.write(json.dumps({
                "timestamp": f"2026-09-10T07:{i:02d}:00Z",
                "event_type": "login_failed",
                "source": "sshd",
                "source_ip": BRUTE_FORCE_IP,
                "username": "root",
                "status": "failed",
                "severity": "high",
                "message": "Failed password for root",
            }) + "\n")


def _write_intel(path: Path) -> None:
    """最小情报库:BRUTE_FORCE_IP 恶意(critical) + TRUSTED_IP 可信。

    保留"已知可信"记录,使误报抑制路径(规则 3)仍被真实覆盖;
    UNKNOWN_IP 故意不在库中,用于无证据分支。
    """
    records = [
        {
            "indicator": BRUTE_FORCE_IP,
            "indicator_type": "ip",
            "malicious": True,
            "confidence": 95,
            "severity": "critical",
            "tags": ["ssh-brute-force", "scanner"],
            "source": "test-fixture",
            "first_seen": "2026-09-01T00:00:00Z",
            "last_seen": "2026-09-02T00:00:00Z",
            "description": "SSH 暴力破解攻击源",
        },
        {
            "indicator": TRUSTED_IP,
            "indicator_type": "ip",
            "malicious": False,
            "confidence": 99,
            "severity": "info",
            "tags": ["trusted-scan-engine"],
            "source": "test-fixture",
            "first_seen": "2026-09-01T11:00:00Z",
            "last_seen": "2026-09-02T11:00:00Z",
            "description": "内部扫描引擎,已知可信",
        },
    ]
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


@pytest.fixture
def paths(tmp_path):
    """最小必要数据:30 条失败登录 + 2 条情报(1 恶意 / 1 可信)。

    返回 (logs_path, intel_path),全部位于 pytest 临时目录 ——
    不读仓库 data/,也不通过 subprocess 回写真实数据目录。
    """
    logs = tmp_path / "security_events.jsonl"
    intel = tmp_path / "threat_intel.jsonl"
    _write_logs(logs)
    _write_intel(intel)
    return logs, intel


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
        "indicator": UNKNOWN_IP,
        "logs_path": str(paths[0]),
        "intel_path": str(paths[1]),
    })
    parsed = json.loads(raw)
    assert parsed["assessment"]["risk_level"] == "none"


def test_wrapper_error_format(paths):
    """空 indicator → 错误契约 JSON(显式注入临时路径,脱离仓库 data/)。"""
    raw = analyze_risk_tool.invoke({
        "indicator": "",
        "logs_path": str(paths[0]),
        "intel_path": str(paths[1]),
    })
    parsed = json.loads(raw)
    assert "error" in parsed
    assert parsed["suggest_retry"] is True
