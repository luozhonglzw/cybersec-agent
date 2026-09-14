"""query_security_logs 测试:全部使用 tmp_path 临时数据文件,不依赖固定数据。"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.log_event import LogEvent
from app.tools.query_logs import DEFAULT_DATA_PATH, MAX_LIMIT, query_security_logs

T0 = datetime(2026, 9, 10, 8, 0, 0, tzinfo=timezone.utc)


def make_event(offset_minutes: float = 0, **overrides) -> dict:
    """构造一条合法 LogEvent 的 dict,默认是 login_failed。"""
    base = {
        "timestamp": (T0.timestamp() + offset_minutes * 60) * 1000,  # 占位,下面替换
    }
    del base["timestamp"]
    base = {
        "timestamp": T0.isoformat(),
        "event_type": "login_failed",
        "source": "sshd",
        "source_ip": "10.0.2.11",
        "destination_ip": "10.0.1.20",
        "source_port": 54321,
        "destination_port": 22,
        "username": "admin",
        "status": "failed",
        "severity": "medium",
        "message": "test event",
    }
    from datetime import timedelta

    base["timestamp"] = (T0 + timedelta(minutes=offset_minutes)).isoformat()
    base.update(overrides)
    return base


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    """覆盖主要分支的 8 条测试数据:多种 event_type / severity / IP / 时间。"""
    rows = [
        make_event(0, severity="info", event_type="firewall_allow", source_ip="10.0.2.11"),
        make_event(10, severity="low", username="bob", source_ip="10.0.2.12"),
        make_event(20, severity="medium", username="admin", source_ip="198.51.100.7"),
        make_event(30, severity="high", event_type="login_success", status="success",
                   username="svc_backup", source_ip="203.0.113.66"),
        make_event(40, severity="critical", event_type="privilege_escalation", source="sudo",
                   username="svc_backup", source_ip="203.0.113.66", action="usermod"),
        make_event(50, severity="medium", event_type="web_request", source="web_app",
                   username=None, source_ip="203.0.113.66", action="GET /admin"),
        make_event(60, severity="info", event_type="firewall_allow", source_ip="10.0.2.13"),
        make_event(70, severity="low", username="bob", source_ip="10.0.2.12"),
    ]
    return write_jsonl(tmp_path / "events.jsonl", rows)


def test_no_filters_returns_all(data_file):
    results = query_security_logs(limit=100, data_path=data_file)
    assert len(results) == 8


def test_filter_by_event_type(data_file):
    results = query_security_logs(event_type="firewall_allow", data_path=data_file)
    assert len(results) == 2
    assert all(e.event_type == "firewall_allow" for e in results)


def test_filter_by_source_ip(data_file):
    results = query_security_logs(source_ip="203.0.113.66", data_path=data_file)
    assert len(results) == 3
    assert all(e.source_ip == "203.0.113.66" for e in results)


def test_filter_by_username(data_file):
    results = query_security_logs(username="bob", data_path=data_file)
    assert len(results) == 2
    assert all(e.username == "bob" for e in results)


def test_filter_by_time_range(data_file):
    from datetime import timedelta

    results = query_security_logs(
        start_time=T0 + timedelta(minutes=20),
        end_time=T0 + timedelta(minutes=50),
        limit=100, data_path=data_file,
    )
    assert [e.username or e.action for e in results] == [
        "admin", "svc_backup", "svc_backup", "GET /admin",
    ]


def test_filter_by_min_severity(data_file):
    """min_severity='high' → 只剩 high + critical(等级顺序,不是字典序)。"""
    results = query_security_logs(min_severity="high", limit=100, data_path=data_file)
    assert {e.severity for e in results} == {"high", "critical"}
    assert len(results) == 2


def test_combined_filters(data_file):
    """source_ip + event_type + 时间窗口 组合。"""
    from datetime import timedelta

    results = query_security_logs(
        source_ip="203.0.113.66",
        event_type="login_success",
        start_time=T0 + timedelta(minutes=25),
        end_time=T0 + timedelta(minutes=35),
        data_path=data_file,
    )
    assert len(results) == 1
    assert results[0].username == "svc_backup"


def test_limit_truncates_results(data_file):
    results = query_security_logs(limit=3, data_path=data_file)
    assert len(results) == 3
    # 截断的是时间最早的三条
    assert results[0].timestamp < results[-1].timestamp


@pytest.mark.parametrize("bad_limit", [0, -1, MAX_LIMIT + 1, "10"])
def test_invalid_limit_rejected(data_file, bad_limit):
    with pytest.raises(ValueError):
        query_security_logs(limit=bad_limit, data_path=data_file)


def test_start_after_end_rejected(data_file):
    from datetime import timedelta

    with pytest.raises(ValueError):
        query_security_logs(
            start_time=T0 + timedelta(hours=2),
            end_time=T0 + timedelta(hours=1),
            data_path=data_file,
        )


def test_invalid_min_severity_rejected(data_file):
    with pytest.raises(ValueError):
        query_security_logs(min_severity="extreme", data_path=data_file)


def test_results_sorted_by_timestamp_stably(data_file):
    """时间戳打乱写入,返回仍按 timestamp 升序;相同 timestamp 保持文件顺序。"""
    shuffled = tmp_shuffle = [
        make_event(30), make_event(10), make_event(10, username="bob"),
        make_event(20), make_event(0),
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = write_jsonl(Path(td) / "shuffled.jsonl", shuffled)
        results = query_security_logs(limit=100, data_path=path)
        timestamps = [e.timestamp for e in results]
        assert timestamps == sorted(timestamps)
        # 两条 +10min 的事件:admin 在文件中先出现,排序后仍在前
        assert results[1].username == "admin" and results[2].username == "bob"


def test_returns_log_events_not_dicts(data_file):
    results = query_security_logs(limit=1, data_path=data_file)
    assert results
    assert all(isinstance(e, LogEvent) for e in results)


def test_missing_data_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        query_security_logs(data_path=tmp_path / "nope.jsonl")


def test_corrupt_line_raises_with_line_number(tmp_path):
    path = write_jsonl(tmp_path / "bad.jsonl", [make_event(0), {"garbage": True}])
    with pytest.raises(ValueError, match="第 2 行"):
        query_security_logs(data_path=path)


def test_naive_datetime_treated_as_utc(data_file):
    from datetime import timedelta

    # naive datetime 也能查询(视为 UTC),不会抛 TypeError
    results = query_security_logs(
        start_time=(T0 + timedelta(minutes=15)).replace(tzinfo=None),
        end_time=(T0 + timedelta(minutes=25)).replace(tzinfo=None),
        limit=100, data_path=data_file,
    )
    assert len(results) == 1
    assert results[0].username == "admin"


def test_default_limit_is_bounded(data_file):
    """不传 limit 时使用默认值(50):数据不足 50 条返回全部。"""
    results = query_security_logs(data_path=data_file)
    assert len(results) == 8
