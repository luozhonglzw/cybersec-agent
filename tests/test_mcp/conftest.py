"""MCP 只读暴露层测试的公共夹具。

**Hermetic 是硬要求**:本目录下的用例一律不得读仓库 `data/`。
`pinned_paths` 是 autouse 夹具,它把 `app.mcp.tools` 的两个数据路径常量钉到
`tmp_path`,因此任何"忘了覆盖路径"的用例都会得到 `DATA_UNAVAILABLE`,
而不是悄悄读到仓库里的真实数据 —— 失败方向是安全的。

共享能力一律以 **fixture** 形式提供(与仓库既有测试目录一致),
测试模块之间不互相 import。
"""
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mcp.client import Client

import app.mcp.tools as mcp_tools
from app.mcp import build_mcp_server

T0 = datetime(2026, 9, 10, 8, 0, 0, tzinfo=timezone.utc)


def _make_event(offset_minutes: float = 0, **overrides) -> dict:
    """构造一条合法 LogEvent 的 dict(默认 login_failed / medium)。"""
    base = {
        "timestamp": (T0 + timedelta(minutes=offset_minutes)).isoformat(),
        "event_type": "login_failed",
        "source": "sshd",
        "source_ip": "203.0.113.66",
        "destination_ip": "10.0.1.20",
        "source_port": 54321,
        "destination_port": 22,
        "username": "admin",
        "status": "failed",
        "severity": "medium",
        "message": "test event",
    }
    base.update(overrides)
    return base


def _make_intel(**overrides) -> dict:
    """构造一条合法 ThreatIntelRecord 的 dict(默认恶意 IP)。"""
    base = {
        "indicator": "203.0.113.66",
        "indicator_type": "ip",
        "malicious": True,
        "confidence": 90,
        "severity": "high",
        "tags": ["ssh-brute-force"],
        "source": "seed",
        "first_seen": T0.isoformat(),
        "last_seen": T0.isoformat(),
        "description": "known scanner",
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@asynccontextmanager
async def _open_client():
    """进程内 MCP 会话(无网络、无子进程、无端口监听)。"""
    async with Client(build_mcp_server()) as client:
        yield client


def _text_of(result) -> str:
    """把一个 `CallToolResult` 的文本内容拼起来。"""
    return "\n".join(block.text for block in result.content)


@dataclass(frozen=True)
class PinnedPaths:
    """被钉到 `tmp_path` 的两个数据路径(默认**不存在**,除非用例自己写入)。"""

    logs: Path
    intel: Path


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def pinned_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PinnedPaths:
    """把适配层的数据路径钉到 `tmp_path`,杜绝读到仓库 `data/`。

    适配器在**调用时**从模块全局读取这两个名字,因此 `monkeypatch.setattr`
    对已经构造好的工具同样生效 —— 不需要重建 server。
    """
    paths = PinnedPaths(logs=tmp_path / "events.jsonl", intel=tmp_path / "intel.jsonl")
    monkeypatch.setattr(mcp_tools, "LOGS_DATA_PATH", paths.logs)
    monkeypatch.setattr(mcp_tools, "INTEL_DATA_PATH", paths.intel)
    return paths


@pytest.fixture
def seeded(pinned_paths: PinnedPaths) -> PinnedPaths:
    """写入一组覆盖主要分支的 hermetic 数据。"""
    _write_jsonl(
        pinned_paths.logs,
        [
            _make_event(0, severity="info", event_type="firewall_allow", source="firewall"),
            _make_event(10, severity="low", username="bob"),
            _make_event(20, severity="medium"),
            _make_event(30, severity="high", event_type="login_success", status="success"),
            _make_event(
                40,
                severity="critical",
                event_type="privilege_escalation",
                source="sudo",
                action="usermod",
            ),
        ],
    )
    _write_jsonl(pinned_paths.intel, [_make_intel()])
    return pinned_paths


@pytest.fixture
def open_client():
    """返回"进程内 MCP 会话"的异步上下文管理器工厂。"""
    return _open_client


@pytest.fixture
def event_factory():
    return _make_event


@pytest.fixture
def intel_factory():
    return _make_intel


@pytest.fixture
def jsonl_writer():
    return _write_jsonl


@pytest.fixture
def text_of():
    return _text_of
