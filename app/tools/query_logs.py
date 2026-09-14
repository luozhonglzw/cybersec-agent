"""query_security_logs:安全日志查询工具(纯 Python,Phase 3 Step 1)。

职责边界:
    工具提供**事实**(过滤后的 LogEvent 列表);
    攻击判断 / 风险评分 / 推理是 Agent 的工作(Phase 3 Step 2+ 才接入 LLM)。

数据流:JSONL → LogEvent(逐行过 Pydantic 校验)→ 过滤 → 排序 → limit → LogEvent[]
工具层绝不绕过 Phase 2 的数据验证,也不返回裸 dict。

本阶段刻意不做的事(Step 2 再做):@tool 装饰器、tool schema、bind_tools、ReAct。
先把函数本身做可靠。
"""
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.log_event import LogEvent, Severity

# severity 的等级顺序:显式建立,不用字符串比较("critical" < "high" 是字典序陷阱)
SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

DEFAULT_DATA_PATH = Path("data/security_events.jsonl")
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


def query_security_logs(
    *,
    event_type: str | None = None,
    source_ip: str | None = None,
    username: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    min_severity: Severity | None = None,
    limit: int = DEFAULT_LIMIT,
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> list[LogEvent]:
    """按条件查询安全日志,返回 timestamp 升序的 LogEvent 列表。

    参数(全部可选,None = 不过滤):
        event_type:    精确匹配事件类型(如 "login_failed")
        source_ip:     精确匹配源 IP(字符串,如 "203.0.113.66")
        username:      精确匹配账号
        start_time:    timestamp >= start_time(naive 视为 UTC)
        end_time:      timestamp <= end_time(naive 视为 UTC)
        min_severity:  最低严重级别(含本身),如 "high" → high + critical
        limit:         最多返回条数,1 <= limit <= MAX_LIMIT(200)
        data_path:     数据文件路径;默认项目内 JSONL,测试可注入 tmp_path

    返回:
        list[LogEvent],按 (timestamp, 原始行号) 升序 —— 排序确定性不依赖文件顺序。

    抛出:
        FileNotFoundError:数据文件不存在
        ValueError:JSONL 非法行 / 参数非法(limit、时间区间、severity)
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= MAX_LIMIT):
        raise ValueError(f"limit 必须是 1~{MAX_LIMIT} 的整数,收到: {limit!r}")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError(f"start_time 不能晚于 end_time: {start_time} > {end_time}")
    if min_severity is not None and min_severity not in SEVERITY_ORDER:
        raise ValueError(f"min_severity 必须是 {list(SEVERITY_ORDER)} 之一,收到: {min_severity!r}")

    # naive datetime(如 "2026-09-10T08:00")统一视为 UTC,避免与数据侧 tz-aware 比较崩溃
    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time is not None and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    min_severity_rank = None if min_severity is None else SEVERITY_ORDER[min_severity]

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"安全日志数据文件不存在: {data_path}(先运行 scripts/seed_logs.py 生成)"
        )

    results: list[LogEvent] = []
    with data_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = LogEvent.model_validate_json(line)
            except Exception as exc:
                # 带行号报错,不静默跳过 —— 脏数据必须暴露而不是被悄悄丢弃
                raise ValueError(f"{data_path} 第 {line_no} 行不是合法 LogEvent: {exc}") from exc

            if event_type is not None and event.event_type != event_type:
                continue
            if source_ip is not None and event.source_ip != source_ip:
                continue
            if username is not None and event.username != username:
                continue
            if start_time is not None and event.timestamp < start_time:
                continue
            if end_time is not None and event.timestamp > end_time:
                continue
            if min_severity_rank is not None and SEVERITY_ORDER[event.severity] < min_severity_rank:
                continue
            results.append(event)

    # 排序确定性:timestamp 相同时按文件行号,不依赖文件当前顺序的隐式行为
    results.sort(key=lambda e: e.timestamp)
    return results[:limit]
