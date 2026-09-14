"""生成结构化模拟安全日志(seed data)。

特性:
- 固定 random seed + 固定基准时间:任何时候运行,输出逐字节一致 —— 这批数据
  是 Phase 9 Evaluation golden set 的地基,必须可复现;
- 8 个安全场景(见 SCENARIOS),不是随机噪声:正常/失败/爆破/提权/Web 攻击
  混在同一时间线里,给 Phase 3 的 Agent 留下真实的多步推理问题;
- 输出 JSONL(data/security_events.jsonl):每行一个 JSON 对象,可独立用
  LogEvent.model_validate_json() 重新校验。

用法:uv run python scripts/seed_logs.py [输出路径]
"""
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.schemas.log_event import LogEvent

SEED = 42
# 固定基准时间(UTC):2026-09-10 08:00,一个"普通工作日的早晨"
BASE_TIME = datetime(2026, 9, 10, 8, 0, 0, tzinfo=timezone.utc)

SERVER_IP = "10.0.1.20"  # 被保护的服务器
WEB_IP = "10.0.1.30"     # Web 应用服务器

# 内网"正常用户"地址段(文档保留网段 RFC1918)
OFFICE_IPS = [f"10.0.2.{i}" for i in range(11, 26)]
OFFICE_USERS = ["alice", "bob", "carol", "dave", "eve"]
BRUTE_FORCE_IP = "203.0.113.66"        # 场景 3/4/5 的攻击者(文档保留网段)
PASSWORD_GUESS_IP = "198.51.100.7"     # 场景 2 的密码猜测者

DEFAULT_OUTPUT = Path("data/security_events.jsonl")


def _event(**kwargs) -> LogEvent:
    """按秒级偏移构造事件,默认补全 destination 与端口。"""
    defaults: dict = {
        "timestamp": BASE_TIME + timedelta(minutes=kwargs.pop("offset_minutes", 0)),
        "destination_ip": SERVER_IP,
    }
    defaults.update(kwargs)
    return LogEvent(**defaults)


def generate_events() -> list[LogEvent]:
    """生成全部模拟事件。入口先重置 seed —— 保证重复调用结果一致(测试依赖此性质)。"""
    rng = random.Random(SEED)
    events: list[LogEvent] = []

    # ---- 场景 1:正常登录(基线流量:工作时段、内网 IP、低频成功)----
    for i in range(25):
        events.append(_event(
            offset_minutes=rng.uniform(0, 480),
            event_type="login_success", source="sshd",
            source_ip=rng.choice(OFFICE_IPS), username=rng.choice(OFFICE_USERS),
            source_port=rng.randint(49152, 65535), destination_port=22,
            status="success", severity="info",
            message=f"Accepted publickey for user from {rng.choice(OFFICE_IPS)}",
        ))

    # ---- 场景 2:单次登录失败(噪声:输错密码,不应触发告警)----
    for _ in range(12):
        user = rng.choice(OFFICE_USERS)
        ip = rng.choice(OFFICE_IPS)
        events.append(_event(
            offset_minutes=rng.uniform(0, 480),
            event_type="login_failed", source="sshd",
            source_ip=ip, username=user,
            source_port=rng.randint(49152, 65535), destination_port=22,
            status="failed", severity="low",
            message=f"Failed password for {user} from {ip} port 22 ssh2",
        ))

    # ---- 场景 3:同一用户多次失败(密码猜测:同一 IP 对同一账号高频失败)----
    for i in range(12):
        events.append(_event(
            offset_minutes=200 + i * 0.5,
            event_type="login_failed", source="sshd",
            source_ip=PASSWORD_GUESS_IP, username="admin",
            source_port=rng.randint(49152, 65535), destination_port=22,
            status="failed", severity="medium",
            message=f"Failed password for admin from {PASSWORD_GUESS_IP} port 22 ssh2",
        ))

    # ---- 场景 4:SSH 撒网式爆破(同一 IP 对大量不同用户各失败 1~2 次)----
    spray_users = [f"user{i:02d}" for i in range(30)]
    minute = 300.0
    for user in spray_users:
        for _ in range(rng.randint(1, 2)):
            events.append(_event(
                offset_minutes=minute,
                event_type="login_failed", source="sshd",
                source_ip=BRUTE_FORCE_IP, username=user,
                source_port=rng.randint(49152, 65535), destination_port=22,
                status="failed", severity="medium",
                message=f"Failed password for invalid user {user} from {BRUTE_FORCE_IP} port 22 ssh2",
            ))
            minute += rng.uniform(0.3, 1.0)

    # ---- 场景 5:爆破 IP 后续成功登录(攻击升级,最有分析价值的故事线)----
    events.append(_event(
        offset_minutes=340,
        event_type="login_success", source="sshd",
        source_ip=BRUTE_FORCE_IP, username="svc_backup",
        source_port=rng.randint(49152, 65535), destination_port=22,
        status="success", severity="high",
        message=f"Accepted password for svc_backup from {BRUTE_FORCE_IP} port 22 ssh2",
    ))

    # ---- 场景 6:权限提升(sudo 失败两次后成功 + 新用户被加入 sudo 组)----
    for i, sev in enumerate(["low", "medium"], start=1):
        events.append(_event(
            offset_minutes=355 + i,
            event_type="privilege_escalation", source="sudo",
            source_ip=BRUTE_FORCE_IP, username="svc_backup",
            action="sudo cat /etc/shadow", status="failed", severity=sev,
            message=f"svc_backup : user NOT in sudoers ; TTY=pts/0 ; PWD=/home/svc_backup ; USER=root",
        ))
    events.append(_event(
        offset_minutes=360,
        event_type="privilege_escalation", source="usermod",
        source_ip=BRUTE_FORCE_IP, username="svc_backup",
        action="usermod -aG sudo svc_backup", status="success", severity="critical",
        message="svc_backup added to sudo group by uid=0 from session of svc_backup",
    ))

    # ---- 场景 7:Web 攻击迹象(路径扫描 / SQL 注入特征 / 异常 UA)----
    attack_requests = [
        ("/admin", "GET", "sqlmap/1.8"),
        ("/wp-login.php", "GET", "sqlmap/1.8"),
        ("/.env", "GET", "curl/8.4.0"),
        ("/api/users?id=1' OR '1'='1", "GET", "python-requests/2.32"),
        ("/../../etc/passwd", "GET", "curl/8.4.0"),
        ("/admin/config.php", "POST", "Mozilla/5.0"),
    ]
    for i in range(20):
        path, method, ua = attack_requests[i % len(attack_requests)]
        events.append(_event(
            offset_minutes=380 + i * 2,
            event_type="web_request", source="web_app",
            source_ip=BRUTE_FORCE_IP, destination_ip=WEB_IP,
            source_port=rng.randint(49152, 65535), destination_port=443,
            action=f"{method} {path}",
            status="denied" if i % 2 else "failed", severity="medium",
            message=f'{method} {path} 403 from {BRUTE_FORCE_IP} UA="{ua}"',
        ))

    # ---- 场景 8:正常业务流量(防火墙放行,背景数据)----
    for _ in range(30):
        src = rng.choice(OFFICE_IPS)
        events.append(_event(
            offset_minutes=rng.uniform(0, 480),
            event_type="firewall_allow", source="firewall",
            source_ip=src, destination_ip=WEB_IP,
            source_port=rng.randint(49152, 65535), destination_port=443,
            status="success", severity="info",
            message=f"ALLOW tcp {src} -> {WEB_IP}:443",
        ))

    # 按时间排序(stable sort,同一时刻保持生成顺序),形成统一时间线
    events.sort(key=lambda e: e.timestamp)
    return events


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    events = generate_events()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(event.model_dump_json() + "\n")

    print(f"写入 {len(events)} 条事件到 {output}")
    by_type: dict[str, int] = {}
    for e in events:
        by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
    for event_type, count in sorted(by_type.items()):
        print(f"  {event_type:24} {count}")


if __name__ == "__main__":
    main()
