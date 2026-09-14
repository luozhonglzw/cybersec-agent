"""seed_logs 的可重复性与数据质量测试。

这些测试是 Phase 9 Evaluation 的地基:golden set 必须逐字节可复现。
"""
from pathlib import Path

from app.schemas.log_event import LogEvent
from scripts.seed_logs import BASE_TIME, BRUTE_FORCE_IP, generate_events


def test_generation_is_deterministic():
    """两次 generate_events() 生成完全相同的事件序列(固定 seed + 固定基准时间)。"""
    first, second = generate_events(), generate_events()
    assert [e.model_dump() for e in first] == [e.model_dump() for e in second]


def test_events_sorted_by_timestamp():
    timestamps = [e.timestamp for e in generate_events()]
    assert timestamps == sorted(timestamps)


def test_jsonl_roundtrip(tmp_path: Path):
    """每行 JSONL 都能重新解析并通过 LogEvent 校验(生成 ⇄ 校验闭环)。"""
    output = tmp_path / "events.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for event in generate_events():
            f.write(event.model_dump_json() + "\n")

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0
    for line in lines:
        LogEvent.model_validate_json(line)  # 任何一行非法直接失败


def test_all_eight_scenarios_present():
    """8 个场景在数据中真实存在(以可观察的数据特征断言)。"""
    events = generate_events()
    failed = [e for e in events if e.event_type == "login_failed"]
    successes = [e for e in events if e.event_type == "login_success"]
    priv = [e for e in events if e.event_type == "privilege_escalation"]
    web = [e for e in events if e.event_type == "web_request"]
    fw = [e for e in events if e.event_type == "firewall_allow"]

    # 场景 1:正常登录(内网 IP 的成功登录)
    assert any(e.source_ip and e.source_ip.startswith("10.0.2.") for e in successes)
    # 场景 2:单次失败噪声(内网 IP 的失败)
    assert any(e.source_ip and e.source_ip.startswith("10.0.2.") for e in failed)
    # 场景 3:同一用户(admin)多次失败 —— 密码猜测
    admin_failures = [e for e in failed if e.username == "admin"]
    assert len(admin_failures) >= 10
    # 场景 4:撒网式爆破 —— 同一 IP 对大量不同用户各失败 1~2 次
    spray_users = {e.username for e in failed if e.source_ip == BRUTE_FORCE_IP and e.username != "admin"}
    assert len(spray_users) >= 25
    # 场景 5:爆破 IP 后续成功登录(攻击升级)
    assert any(e.event_type == "login_success" and e.source_ip == BRUTE_FORCE_IP for e in events)
    # 场景 6:权限提升(sudo 失败 → 加入 sudo 组)
    assert any(e.status == "failed" for e in priv)
    assert any(e.severity == "critical" and e.status == "success" for e in priv)
    # 场景 7:Web 攻击迹象(路径扫描 / 注入特征)
    assert any(e.action and (".." in e.action or "'" in e.action or "/admin" in e.action) for e in web)
    # 场景 8:正常防火墙流量
    assert len(fw) > 0


def test_no_event_before_base_time():
    """所有事件都落在基准时间之后(时间戳由固定基准 + 相对偏移生成)。"""
    assert all(e.timestamp >= BASE_TIME for e in generate_events())
