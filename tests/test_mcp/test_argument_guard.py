"""协议层参数守卫(`ReadOnlyArgumentGuard`)的行为。

守卫存在的理由是一条**实测**的 SDK 事实:mcp 2.3.0 对未声明的多余参数是
**静默忽略**的 —— 参数模型没有 `extra="forbid"`,`input_schema` 也不带
`additionalProperties`。所以"路径字段不在 schema 里"只说明调用方**送不进来**
(签名层已经保证),并不构成**拒绝**:`{"data_path": "/etc/passwd"}` 会被无声丢弃,
调用方拿到一个看起来正常的成功结果。

本文件要证明的正是"从忽略升级为拒绝",并且**拒绝发生在 handler 之前**。
"""
from inspect import signature

import pytest
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS

import app.mcp.tools as mcp_tools
from app.mcp import READONLY_TOOL_SPECS
from app.mcp.server import ALLOWED_ARGUMENTS, _reject_undeclared_arguments

#: 必须被拒绝的路径键 —— 与冻结的 `_SCHEMA_PATH_FIELDS` 同集合。
PATH_KEYS = ("data_path", "logs_path", "intel_path")

#: 每个工具的**最小合法**参数 —— 让"被拒绝"只可能归因于多加的那个路径键。
MINIMAL_ARGUMENTS: dict[str, dict] = {
    "query_security_logs": {"limit": 1},
    "query_threat_intel": {"indicator": "203.0.113.66"},
    "analyze_risk": {"indicator": "203.0.113.66"},
}


@pytest.mark.parametrize("key", PATH_KEYS)
@pytest.mark.parametrize("tool_name", sorted(MINIMAL_ARGUMENTS))
async def test_path_arguments_are_rejected_not_ignored(tool_name, key, seeded, open_client):
    """任何路径键都必须被**显式拒绝**,而不是被静默丢弃。"""
    arguments = {**MINIMAL_ARGUMENTS[tool_name], key: "/etc/passwd"}
    async with open_client() as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool(tool_name, arguments)
    assert excinfo.value.code == INVALID_PARAMS
    assert key in excinfo.value.message
    assert excinfo.value.data["tool"] == tool_name
    assert excinfo.value.data["undeclared"] == [key]


@pytest.mark.parametrize("tool_name", sorted(MINIMAL_ARGUMENTS))
async def test_minimal_arguments_alone_are_accepted(tool_name, seeded, open_client):
    """负对照:去掉路径键之后,同一组参数必须成功 —— 证明拒绝归因于该键。"""
    async with open_client() as client:
        result = await client.call_tool(tool_name, MINIMAL_ARGUMENTS[tool_name])
    assert result.is_error is False


async def test_arbitrary_undeclared_argument_is_rejected(seeded, open_client):
    """不止路径键 —— 任何未声明的键都拒绝(默认拒绝而非默认放行)。"""
    async with open_client() as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("analyze_risk", {"indicator": "x", "totally_unknown": 1})
    assert excinfo.value.code == INVALID_PARAMS
    assert excinfo.value.data["undeclared"] == ["totally_unknown"]


async def test_declared_arguments_are_accepted(seeded, open_client):
    """正对照:声明过的键一个都不许被误伤。"""
    async with open_client() as client:
        result = await client.call_tool(
            "query_security_logs",
            {
                "event_type": "login_failed",
                "source_ip": "203.0.113.66",
                "username": "admin",
                "start_time": "2026-09-10T00:00:00Z",
                "end_time": "2026-09-11T00:00:00Z",
                "min_severity": "low",
                "limit": 10,
            },
        )
    assert result.is_error is False


async def test_guard_short_circuits_before_the_handler_runs(
    monkeypatch, pinned_paths, open_client
):
    """守卫必须**先于** handler 生效:被拦下的调用不得抵达核心实现。

    先直接跑一次合法调用证明探针是活的(否则"没被调用"可能只是探针坏了),
    再跑被拦下的调用。
    """
    calls: list[dict] = []

    def spy(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(mcp_tools, "query_security_logs", spy)

    async with open_client() as client:
        # 正对照:探针确实在链路上。
        allowed = await client.call_tool("query_security_logs", {"limit": 1})
        assert allowed.is_error is False
        assert len(calls) == 1, "探针没有被触达 —— 后续的'未被调用'不构成证据"

        calls.clear()
        # 负对照:被守卫拦下 ⇒ 核心实现一次都没跑。
        with pytest.raises(MCPError):
            await client.call_tool("query_security_logs", {"limit": 1, "data_path": "/etc/passwd"})
        assert calls == [], "守卫没有在 handler 之前短路"


def test_guard_rejects_non_object_arguments():
    """`arguments` 不是对象时也要拒绝 —— 直接对守卫函数取证。

    这条路径走不到客户端:SDK 的参数模型会在守卫**之后**先一步拒绝非对象。
    守卫仍然要覆盖它,因为它跑在模型校验之前,是唯一的兜底。
    """
    with pytest.raises(MCPError) as excinfo:
        _reject_undeclared_arguments(
            {"name": "analyze_risk", "arguments": ["not", "a", "mapping"]}
        )
    assert excinfo.value.code == INVALID_PARAMS
    assert excinfo.value.data["tool"] == "analyze_risk"


def test_guard_ignores_unknown_tool_names():
    """未知工具名**不**由守卫否决 —— 那是 SDK "Unknown tool" 分支的职责。

    重复实现只会让两处错误语义有机会分叉。
    """
    assert "definitely_not_a_tool" not in ALLOWED_ARGUMENTS
    # 不抛异常即通过
    _reject_undeclared_arguments({"name": "definitely_not_a_tool", "arguments": {"x": 1}})
    _reject_undeclared_arguments({"arguments": {"x": 1}})
    _reject_undeclared_arguments({"name": "analyze_risk"})
    _reject_undeclared_arguments("not-a-mapping")


def test_guard_allow_list_is_derived_from_the_adapter_signatures():
    """允许集合必须**派生自签名**,不能是手抄的第二份清单。"""
    for spec in READONLY_TOOL_SPECS:
        assert ALLOWED_ARGUMENTS[spec.name] == frozenset(signature(spec.fn).parameters)


async def test_listing_tools_is_unaffected_by_the_guard(open_client):
    async with open_client() as client:
        listed = await client.list_tools()
    assert len(listed.tools) == len(READONLY_TOOL_SPECS)
