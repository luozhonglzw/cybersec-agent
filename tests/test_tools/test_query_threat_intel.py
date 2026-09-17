"""query_threat_intel 工具测试:纯函数 + LangChain wrapper + Phase 2 关联。

全部离线:查询的是 scripts/seed_threat_intel.py 生成的固定数据。
"""
import json
from pathlib import Path

import pytest

from app.schemas.threat_intel import ThreatIntelRecord
from app.tools.query_threat_intel import (
    query_threat_intel,
    query_threat_intel_tool,
)

SEED_SCRIPT = Path("scripts/seed_threat_intel.py")
BRUTE_FORCE_IP = "203.0.113.66"       # Phase 2 SSH brute force 攻击者
EVIL_DOMAIN = "evil-example.com"
MALWARE_HASH = "a" * 64               # seed 中的"恶意样本 #1"
TRUSTED_DOMAIN = "trusted-example.com"


@pytest.fixture(scope="module")
def intel_data(tmp_path_factory) -> Path:
    """运行 seed 脚本生成模块级共享的临时数据文件(保证可重复、离线)。"""
    out = tmp_path_factory.mktemp("intel") / "threat_intel.jsonl"
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(SEED_SCRIPT), str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


# ---------- 纯函数:精确查询 ----------

def test_exact_ip_lookup(intel_data):
    """精确查询恶意 IP。"""
    record = query_threat_intel(BRUTE_FORCE_IP, data_path=intel_data)
    assert record is not None
    assert record.indicator_type == "ip"
    assert record.malicious is True
    assert "ssh-brute-force" in record.tags


def test_domain_lookup(intel_data):
    """精确查询恶意域名。"""
    record = query_threat_intel(EVIL_DOMAIN, data_path=intel_data)
    assert record is not None
    assert record.indicator_type == "domain"
    assert "phishing" in record.tags


def test_hash_lookup(intel_data):
    """精确查询恶意 Hash。"""
    record = query_threat_intel(MALWARE_HASH, data_path=intel_data)
    assert record is not None
    assert record.indicator_type == "hash"
    assert record.severity == "critical"


def test_unknown_ioc_returns_none(intel_data):
    """不存在的 IOC → None(不是异常,是正常业务结果)。"""
    assert query_threat_intel("10.9.9.9", data_path=intel_data) is None
    assert query_threat_intel("no-such-example.invalid", data_path=intel_data) is None


def test_indicator_type_filter_match(intel_data):
    """indicator_type 与记录一致 → 正常命中。"""
    record = query_threat_intel(BRUTE_FORCE_IP, indicator_type="ip", data_path=intel_data)
    assert record is not None


def test_indicator_type_filter_mismatch(intel_data):
    """indicator 命中但类型不符 → None(IP 不会当 domain 命中)。"""
    assert query_threat_intel(BRUTE_FORCE_IP, indicator_type="domain", data_path=intel_data) is None


def test_indicator_type_filter_invalid(intel_data):
    """非法 indicator_type → ValueError。"""
    with pytest.raises(ValueError, match="indicator_type 必须是"):
        query_threat_intel(BRUTE_FORCE_IP, indicator_type="url", data_path=intel_data)


def test_empty_indicator_rejected(intel_data):
    """空 indicator → ValueError。"""
    with pytest.raises(ValueError, match="indicator 不能为空"):
        query_threat_intel("", data_path=intel_data)


def test_missing_file_raises_file_not_found(tmp_path):
    """数据文件不存在 → FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        query_threat_intel(BRUTE_FORCE_IP, data_path=tmp_path / "nope.jsonl")


def test_case_sensitive_exact_match(intel_data):
    """精确匹配是大小写敏感的:变体不命中(IOC 语义就是 ==)。"""
    # IP 全由数字构成,大小写不变,用数值变体验证精确匹配
    assert query_threat_intel("203.0.113.67", data_path=intel_data) is None
    # 域名有字母:大写变体不命中
    assert query_threat_intel(EVIL_DOMAIN.upper(), data_path=intel_data) is None


# ---------- wrapper:返回格式 ----------

def test_wrapper_found_format(intel_data):
    """wrapper 命中:{"found": true, "record": {...}},record 可反序列化为模型。"""
    result = query_threat_intel_tool.invoke({
        "indicator": BRUTE_FORCE_IP,
        "data_path": str(intel_data),
    })
    parsed = json.loads(result)
    assert parsed["found"] is True
    restored = ThreatIntelRecord.model_validate(parsed["record"])
    assert restored.indicator == BRUTE_FORCE_IP


def test_wrapper_not_found_format(intel_data):
    """wrapper 未命中:{"found": false, ...} + 固定 message。"""
    result = query_threat_intel_tool.invoke({
        "indicator": "10.9.9.9",
        "data_path": str(intel_data),
    })
    parsed = json.loads(result)
    assert parsed == {
        "found": False,
        "indicator": "10.9.9.9",
        "message": "No threat intelligence found",
    }


def test_wrapper_error_format(tmp_path):
    """wrapper 错误:{"error": ..., "type": ..., "suggest_retry": true}。"""
    result = query_threat_intel_tool.invoke({
        "indicator": "",
        "data_path": str(tmp_path),
    })
    parsed = json.loads(result)
    assert parsed["error"] == "indicator 不能为空"
    assert parsed["type"] == "ValueError"
    assert parsed["suggest_retry"] is True


def test_wrapper_json_serializable_datetimes(intel_data):
    """wrapper 返回的 record 中 datetime 已序列化为 ISO 字符串。"""
    result = query_threat_intel_tool.invoke({
        "indicator": EVIL_DOMAIN,
        "data_path": str(intel_data),
    })
    parsed = json.loads(result)  # 能 json.loads 本身就说明无 datetime 残留
    assert isinstance(parsed["record"]["first_seen"], str)


def test_wrapper_schema_params():
    """wrapper 暴露给 LLM 的参数与核心函数一致。"""
    schema = query_threat_intel_tool.args_schema.model_json_schema()["properties"]
    assert set(schema.keys()) == {"indicator", "indicator_type", "data_path"}


# ---------- Phase 2 关联验证 ----------

def test_phase2_brute_force_ip_cross_reference(intel_data):
    """Phase 2 日志中的攻击 IP 203.0.113.66 能查到对应威胁情报。

    这是"log evidence + threat intel evidence"关联能力的地基:
    Phase 2 的 seed_logs.py 用同一 IP 作为 SSH brute force 攻击者,
    本测试证明同一标识符在两个数据集之间可以精确关联。
    """
    # 1. Phase 2 日志数据中确实存在该攻击 IP 的 login_failed 事件
    from app.tools.query_logs import query_security_logs
    log_events = query_security_logs(
        source_ip=BRUTE_FORCE_IP, event_type="login_failed"
    )
    assert len(log_events) > 0, "Phase 2 日志中应存在该 IP 的失败登录"

    # 2. 威胁情报库中该 IP 存在且标记为 SSH 暴力破解
    intel = query_threat_intel(BRUTE_FORCE_IP, indicator_type="ip", data_path=intel_data)
    assert intel is not None
    assert intel.malicious is True
    assert "ssh-brute-force" in intel.tags

    # 3. 关联成立:日志 IP == 情报 indicator(Exact Match 语义)
    assert log_events[0].source_ip == intel.indicator


def test_seed_data_reproducible(tmp_path):
    """seed 脚本两次运行输出逐字节一致(可复现性)。"""
    out1 = tmp_path / "a.jsonl"
    out2 = tmp_path / "b.jsonl"
    import subprocess, sys
    for out in (out1, out2):
        result = subprocess.run(
            [sys.executable, str(SEED_SCRIPT), str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    assert out1.read_bytes() == out2.read_bytes()


def test_seed_data_size_in_range(intel_data):
    """数据规模在 20-50 条之间。"""
    count = sum(1 for line in open(intel_data, encoding="utf-8") if line.strip())
    assert 20 <= count <= 50


def test_trusted_ioc_not_malicious(intel_data):
    """可信 IOC(误报对照组)存在且 malicious=False。"""
    record = query_threat_intel(TRUSTED_DOMAIN, data_path=intel_data)
    assert record is not None
    assert record.malicious is False
