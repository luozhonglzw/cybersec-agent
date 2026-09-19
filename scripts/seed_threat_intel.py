"""生成模拟威胁情报数据(seed data)。

特性(与 seed_logs.py 一致):
- 固定 random seed + 固定时间:任何时候运行,输出逐字节一致;
- 覆盖恶意 IP / 恶意域名 / 恶意 Hash 三类 IOC,外加少量"已知可信"记录
  用于测试误报路径;
- 203.0.113.66 与 Phase 2 的 SSH brute force 攻击者对应 ——
  未来的 Agent 可以把"日志证据"与"威胁情报证据"关联起来;
- 输出 JSONL(data/threat_intel.jsonl):每行一个 JSON 对象,可独立用
  ThreatIntelRecord.model_validate_json() 重新校验。

用法:uv run python scripts/seed_threat_intel.py [输出路径]
"""
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.schemas.threat_intel import ThreatIntelRecord

SEED = 42
# 固定时间(UTC):与 seed_logs.py 的时间线对齐(2026-09-10 前后)
FIRST_SEEN = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)

BRUTE_FORCE_IP = "203.0.113.66"      # 与 Phase 2 SSH brute force 场景对应
PASSWORD_GUESS_IP = "198.51.100.7"   # 与 Phase 2 密码猜测场景对应

DEFAULT_OUTPUT = Path("data/threat_intel.jsonl")


def _records() -> list[ThreatIntelRecord]:
    """手工构造 29 条固定 IOC 记录(不依赖 now(),时间从 FIRST_SEEN 偏移)。"""
    def rec(offset_hours: int, **kw) -> ThreatIntelRecord:
        seen = FIRST_SEEN + timedelta(hours=offset_hours)
        return ThreatIntelRecord(
            first_seen=seen,
            last_seen=seen + timedelta(hours=24),
            **kw,
        )

    r: list[ThreatIntelRecord] = []
    # ---- 恶意 IP(与 Phase 2 攻击场景对应的在前) ----
    r.append(rec(0, indicator=BRUTE_FORCE_IP, indicator_type="ip", malicious=True,
                 confidence=95, severity="critical",
                 tags=["ssh-brute-force", "scanner"], source="internal-analysis",
                 description="SSH 暴力破解攻击源,Phase 2 场景 3/4/5 的攻击者"))
    r.append(rec(1, indicator=PASSWORD_GUESS_IP, indicator_type="ip", malicious=True,
                 confidence=80, severity="high",
                 tags=["password-guessing"], source="internal-analysis",
                 description="低速密码猜测来源,Phase 2 场景 2 的攻击者"))
    for i, (ip, conf, sev, tags) in enumerate([
        ("203.0.113.100", 70, "high", ["botnet", "c2"]),
        ("203.0.113.101", 60, "medium", ["scanner"]),
        ("203.0.113.102", 90, "critical", ["ransomware", "c2"]),
        ("198.51.100.50", 55, "medium", ["spam-source"]),
        ("198.51.100.51", 75, "high", ["web-attack", "sql-injection"]),
        ("198.51.100.52", 40, "low", ["suspicious"]),
        ("192.0.2.200", 85, "high", ["malware-distribution"]),
        ("192.0.2.201", 65, "medium", ["phishing"]),
    ]):
        r.append(rec(2 + i, indicator=ip, indicator_type="ip", malicious=True,
                     confidence=conf, severity=sev, tags=tags,
                     source="seed-intel", description=f"模拟恶意 IP #{i + 1}"))
    # ---- 已知可信 IP(误报测试用) ----
    r.append(rec(11, indicator="192.0.2.10", indicator_type="ip", malicious=False,
                 confidence=99, severity="info",
                 tags=["trusted-scan-engine"], source="seed-intel",
                 description="内部扫描引擎,已知可信"))
    r.append(rec(12, indicator="192.0.2.11", indicator_type="ip", malicious=False,
                 confidence=90, severity="info",
                 tags=["cdn"], source="seed-intel", description="CDN 出口节点"))
    # ---- 恶意域名 ----
    domains = [
        ("evil-example.com", 95, "critical", ["phishing", "c2"]),
        ("malware-example.net", 85, "high", ["malware-distribution"]),
        ("c2-bad.example.org", 90, "critical", ["c2", "beacon"]),
        ("phish-login-example.com", 80, "high", ["phishing"]),
        ("exfil-bad-example.net", 70, "high", ["data-exfiltration"]),
        ("scan-evil-example.org", 60, "medium", ["scanner"]),
        ("spam-bad-example.com", 50, "low", ["spam-source"]),
        ("payload-evil-example.net", 75, "high", ["exploit-kit"]),
    ]
    for i, (d, conf, sev, tags) in enumerate(domains):
        r.append(rec(13 + i, indicator=d, indicator_type="domain", malicious=True,
                     confidence=conf, severity=sev, tags=tags,
                     source="seed-intel", description=f"模拟恶意域名 #{i + 1}"))
    # ---- 已知可信域名 ----
    r.append(rec(21, indicator="trusted-example.com", indicator_type="domain",
                 malicious=False, confidence=99, severity="info",
                 tags=["partner-site"], source="seed-intel",
                 description="业务合作方站点,已知可信"))
    r.append(rec(22, indicator="updates-trusted-example.org", indicator_type="domain",
                 malicious=False, confidence=95, severity="info",
                 tags=["software-update"], source="seed-intel",
                 description="内部软件更新源"))
    # ---- 恶意 Hash(SHA256) ----
    hashes = [
        ("a" * 64, 95, "critical", ["ransomware"]),
        ("b" * 64, 85, "high", ["trojan"]),
        ("c" * 64, 90, "critical", ["rat", "c2"]),
        ("d" * 64, 70, "high", ["dropper"]),
        ("e" * 64, 60, "medium", ["adware"]),
        ("f" * 64, 80, "high", ["backdoor"]),
    ]
    for i, (h, conf, sev, tags) in enumerate(hashes):
        r.append(rec(23 + i, indicator=h, indicator_type="hash", malicious=True,
                     confidence=conf, severity=sev, tags=tags,
                     source="seed-intel", description=f"模拟恶意样本 SHA256 #{i + 1}"))
    # ---- 已知可信 Hash(内部工具签名) ----
    r.append(rec(29, indicator="0" * 64, indicator_type="hash", malicious=False,
                 confidence=99, severity="info",
                 tags=["internal-tool"], source="seed-intel",
                 description="内部工具哈希,已知可信"))
    return r


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    records = _records()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    print(f"写入 {len(records)} 条威胁情报记录到 {output}")


if __name__ == "__main__":
    main()
