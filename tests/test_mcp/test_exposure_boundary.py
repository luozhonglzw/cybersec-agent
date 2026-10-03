"""暴露边界的结构性证据。

本文件回答两个问题:

    1. 路径排除到底是**结构性**的,还是只是"没写进 schema"?
       —— 即使绕开协议层守卫、直接同进程调用 `server.call_tool(...)` 注入
          `data_path`,也必须无法改变真正被读的文件。
    2. 这次暴露有没有引入**新的 I/O 面**?
       —— 传输只有 stdio;不注册 resources / prompts;不 import 任何网络客户端。
"""
import ast
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from app.mcp import build_mcp_server
from app.mcp.server import main

MCP_PKG = Path(__file__).resolve().parents[2] / "app" / "mcp"


# ---------------------------------------------------------------------------
# 1. 路径排除是结构性的
# ---------------------------------------------------------------------------


async def test_injected_path_cannot_redirect_the_file_read(
    pinned_paths, tmp_path, jsonl_writer, event_factory
):
    """绕开守卫、直接进程内调用:注入的路径必须**无效**。

    签名里没有路径形参 ⇒ SDK 的参数模型把多余键丢掉 ⇒ 适配器仍然用钉死的路径。
    这是"结构性排除"的实质证据:不是"我们检查了并拒绝",而是"根本送不进去"。
    """
    decoy = tmp_path / "decoy.jsonl"
    # 钉死路径里 3 条,诱饵里 1 条 —— 数量即可区分读到的是哪一个。
    jsonl_writer(pinned_paths.logs, [event_factory(0), event_factory(10), event_factory(20)])
    jsonl_writer(decoy, [event_factory(0, username="decoy")])

    server = build_mcp_server()
    result = await server.call_tool("query_security_logs", {"limit": 50, "data_path": str(decoy)})
    assert result.is_error is False
    assert result.structured_content["count"] == 3, "注入的路径改变了实际读取的文件"
    assert all(event["username"] != "decoy" for event in result.structured_content["events"])


async def test_injected_path_is_silently_dropped_which_is_why_the_guard_exists(
    pinned_paths, jsonl_writer
):
    """把 SDK 的行为事实钉住:进程内直连**不经过**中间件,多余键被静默丢弃。

    这条断言不是"我们接受静默丢弃",而是把两层防线的分工写清楚:
        - 协议层(`ReadOnlyArgumentGuard`)负责**拒绝**外部调用方的多余键;
        - 签名层负责让"送不进来"成为**结构事实**,即使守卫被绕过。
    两者任一单独存在都不够 —— 所以两条都测。
    """
    jsonl_writer(pinned_paths.logs, [])
    server = build_mcp_server()
    result = await server.call_tool("query_security_logs", {"limit": 1, "data_path": "/etc/passwd"})
    assert result.is_error is False  # 直连路径下没有被拒绝 —— 守卫只在协议层生效
    assert result.structured_content["count"] == 0  # 读的是钉死路径(空文件),不是 /etc/passwd


async def test_protocol_layer_does_reject_the_same_call(pinned_paths, jsonl_writer, open_client):
    """同一组参数走协议层 ⇒ 被拒绝。与上一条构成对照。"""
    jsonl_writer(pinned_paths.logs, [])
    async with open_client() as client:
        with pytest.raises(MCPError):
            await client.call_tool("query_security_logs", {"limit": 1, "data_path": "/etc/passwd"})


# ---------------------------------------------------------------------------
# 2. 没有引入新的 I/O 面
# ---------------------------------------------------------------------------


def test_transport_is_stdio_only():
    """入口只允许 stdio —— 不引入网络监听面。"""
    tree = ast.parse((MCP_PKG / "server.py").read_text(encoding="utf-8"))
    transports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "run":
            for keyword in node.keywords:
                if keyword.arg == "transport":
                    transports.append(ast.literal_eval(keyword.value))
    assert transports == ["stdio"], f"入口传输不是 stdio:{transports}"

    # 任何 sse / streamable-http 的字面量都不该出现在本包里。
    for path in sorted(MCP_PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert not (literals & {"sse", "streamable-http", "streamable_http"}), (
            f"{path.name}: 出现网络传输字面量"
        )


def test_mcp_package_imports_no_network_client():
    """本包不得 import 任何网络客户端 —— 只读暴露不该自带出网能力。"""
    forbidden = {"socket", "httpx", "requests", "urllib", "urllib3", "http.client", "subprocess"}
    offenders: list[str] = []
    for path in sorted(MCP_PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [
                    f"{path.name}: import {alias.name}"
                    for alias in node.names
                    if alias.name in forbidden or alias.name.split(".")[0] in forbidden
                ]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module in forbidden or node.module.split(".")[0] in forbidden:
                    offenders.append(f"{path.name}: from {node.module}")
    assert not offenders, f"只读暴露层出现了网络/子进程依赖:{offenders}"


async def test_no_resources_or_prompts_are_registered():
    """暴露面只有工具。"""
    server = build_mcp_server()
    assert await server.list_resources() == []
    assert await server.list_resource_templates() == []
    assert await server.list_prompts() == []


async def test_server_identity_is_configured():
    server = build_mcp_server()
    assert server.name == "cybersec-agent"
    assert server.instructions


async def test_read_only_tools_do_not_need_a_network_transport(
    pinned_paths, jsonl_writer, open_client, text_of
):
    """端到端确认:整个调用链在**进程内**完成,不需要任何端口。"""
    jsonl_writer(pinned_paths.logs, [])
    async with open_client() as client:
        result = await client.call_tool("query_security_logs", {"limit": 1})
    assert result.is_error is False
    assert text_of(result)


def test_entry_point_is_callable():
    """`main` 是入口(不在这里真正启动 stdio,否则会阻塞测试)。"""
    assert callable(main)
    assert main.__module__ == "app.mcp.server"


def test_package_init_does_not_eagerly_import_the_entry_module():
    """`app/mcp/__init__.py` 不得在导入期拉入 `app.mcp.server`。

    入口是 `python -m app.mcp.server`。若包导入期就把该模块放进 `sys.modules`,
    runpy 会把它**执行两次**(一次作为 `app.mcp.server`、一次作为 `__main__`)
    并打出 RuntimeWarning —— 对 stdio 服务端来说这是会被客户端看见的噪声。
    因此再导出必须是惰性的(PEP 562),本断言把这条约束钉住。
    """
    tree = ast.parse((MCP_PKG / "__init__.py").read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in tree.body:  # 只看模块顶层,函数体内的惰性导入是允许的
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.mcp.server"):
            offenders.append(f"from {node.module} import ...")
        if isinstance(node, ast.Import):
            offenders += [
                f"import {alias.name}" for alias in node.names if alias.name.startswith("app.mcp.server")
            ]
    assert not offenders, f"包导入期不得拉入入口模块:{offenders}"


def test_lazy_reexports_resolve():
    """惰性再导出的名字必须真的取得到(否则 `from app.mcp import x` 会炸)。"""
    import app.mcp as package

    expected = {
        "ALLOWED_ARGUMENTS",
        "SERVER_NAME",
        "ReadOnlyArgumentGuard",
        "build_mcp_server",
        "main",
        "public_argument_names",
    }
    exported = set(package.__all__)
    assert expected <= exported
    for name in sorted(expected):
        assert getattr(package, name) is not None
        assert name in dir(package)
    assert package.READONLY_TOOL_SPECS
