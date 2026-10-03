"""MCP 服务器装配与入口(Phase 9.3-E)。

本模块只做三件事:
    1. 把 `app.mcp.tools` 里的只读适配器注册进一个 `MCPServer`;
    2. 装一道**协议层**参数守卫(`ReadOnlyArgumentGuard`),把 SDK 的
       「静默忽略多余参数」升级为「显式拒绝」;
    3. 提供 stdio 入口(本地子进程 stdin/stdout)。

刻意不做的事:
    - **不**启用 SSE / streamable-HTTP:那会引入一个网络监听面。只读暴露的
      信任模型是"把既有能力换一个接口表达",不是新开一个远程服务;
    - **不**注册 resources / prompts:本阶段的暴露面只有工具;
    - **不**做鉴权 / 限流 / 多租户:stdio 由宿主进程拉起,身份由进程边界承担。
"""
from collections.abc import Mapping
from inspect import signature
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS

from app.mcp.tools import READONLY_TOOL_SPECS, ReadOnlyToolSpec

SERVER_NAME = "cybersec-agent"
SERVER_TITLE = "CyberSec-Agent (read-only tools)"
SERVER_INSTRUCTIONS = (
    "Read-only access to the CyberSec-Agent tool layer: query the local "
    "security event log, look up an indicator in the local threat-intelligence "
    "store, and assess one indicator's risk. No write or remediation action is "
    "exposed; response planning and approval stay outside this interface."
)


def public_argument_names(spec: ReadOnlyToolSpec) -> frozenset[str]:
    """一个工具**对外**声明的参数名集合(直接取自适配器签名)。

    签名是唯一事实来源:守卫的允许集合与下发的 `input_schema` 都从这里派生,
    因此两者结构上不可能分叉。
    """
    return frozenset(signature(spec.fn).parameters)


#: 工具名 → 允许的参数名。模块导入时一次性算好。
ALLOWED_ARGUMENTS: dict[str, frozenset[str]] = {
    spec.name: public_argument_names(spec) for spec in READONLY_TOOL_SPECS
}


class ReadOnlyArgumentGuard(ServerMiddleware[Any]):
    """在参数校验**之前**拒绝任何未声明的调用参数。

    为什么需要它:mcp 2.3.0 对未声明的多余参数是**静默忽略**的 —— `Tool.fn_metadata`
    生成的参数模型没有 `extra="forbid"`,下发的 `input_schema` 也不带
    `additionalProperties`。于是"路径字段不在 schema 里"只意味着调用方**送不进来**,
    并不构成**拒绝**:`{"data_path": "/etc/passwd"}` 会被无声丢弃,调用方拿到一个
    看起来正常的成功结果。

    `ServerMiddleware` 跑在参数校验之前,因此在这里否决可以保证:
        - 未声明的键**从未**抵达任何 handler(包括未来的 handler);
        - 失败以 `INVALID_PARAMS` 协议错误表达,而不是被当成工具输出。
    """

    async def __call__(
        self,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        if ctx.method == "tools/call":
            _reject_undeclared_arguments(ctx.params)
        return await call_next(ctx)


def _reject_undeclared_arguments(params: Any) -> None:
    """核对 `tools/call` 的原始参数;出现未声明键即抛 `INVALID_PARAMS`。

    未知工具名**不**在这里否决 —— 那是 SDK 自己的 "Unknown tool" 分支的职责,
    重复实现只会让两处错误语义有机会分叉。
    """
    if not isinstance(params, Mapping):
        return
    name = params.get("name")
    allowed = ALLOWED_ARGUMENTS.get(name) if isinstance(name, str) else None
    if allowed is None:
        return
    arguments = params.get("arguments")
    if arguments is None:
        return
    if not isinstance(arguments, Mapping):
        raise MCPError(
            code=INVALID_PARAMS,
            message=f"Invalid arguments for {name}: expected an object",
            data={"tool": name},
        )
    undeclared = sorted(str(key) for key in arguments if key not in allowed)
    if undeclared:
        raise MCPError(
            code=INVALID_PARAMS,
            message=(
                f"Undeclared argument(s) for {name}: {', '.join(undeclared)}. "
                f"Accepted: {', '.join(sorted(allowed))}."
            ),
            data={"tool": name, "undeclared": undeclared},
        )


def build_mcp_server() -> MCPServer[Any]:
    """装配只读 MCP 服务器(不启动、不监听)。"""
    server: MCPServer[Any] = MCPServer(
        name=SERVER_NAME,
        title=SERVER_TITLE,
        instructions=SERVER_INSTRUCTIONS,
        middleware=[ReadOnlyArgumentGuard()],
    )
    for spec in READONLY_TOOL_SPECS:
        server.add_tool(
            spec.fn,
            name=spec.name,
            title=spec.title,
            description=spec.description,
        )
    return server


def main() -> None:
    """stdio 入口:本地子进程 stdin/stdout,**不监听任何网络端口**。"""
    build_mcp_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover - 进程入口
    main()
