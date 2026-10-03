"""MCP 只读暴露层(Phase 9.3-E)。

对外只暴露三个**只读**工具:查询安全日志、查询威胁情报、结构化风险评估。
写入型 / 需人工审批的处置工具(`plan_response`)刻意不暴露。

设计要点(详见 `app.mcp.tools` 与 `app.mcp.server` 的模块 docstring):
    - 公开签名里**没有**任何文件路径形参 —— 路径由服务端钉死;
    - SDK 对多余参数是静默忽略的,因此另有一道**协议层**守卫显式拒绝未声明键;
    - 失败收敛为固定分类 + 固定文案,不回显路径、不回显回溯;
    - 传输只支持 stdio,不引入网络监听面。

为什么 `server` 子模块是**惰性**再导出的:
    入口是 `python -m app.mcp.server`。若本文件在导入期就 `import app.mcp.server`,
    那么 runpy 在把该模块当 `__main__` 执行之前,它**已经**在 `sys.modules` 里了 ——
    模块会被执行两次并触发 RuntimeWarning。惰性再导出(PEP 562)让
    `from app.mcp import build_mcp_server` 照常可用,同时保证 `-m` 只执行一次。
"""
from app.mcp.tools import (
    READONLY_TOOL_SPECS,
    ReadOnlyToolErrorCode,
    ReadOnlyToolSpec,
)

#: 需要从 `app.mcp.server` 惰性取出的名字。
_LAZY_FROM_SERVER = frozenset(
    {
        "ALLOWED_ARGUMENTS",
        "SERVER_INSTRUCTIONS",
        "SERVER_NAME",
        "SERVER_TITLE",
        "ReadOnlyArgumentGuard",
        "build_mcp_server",
        "main",
        "public_argument_names",
    }
)

__all__ = sorted(
    {
        "READONLY_TOOL_SPECS",
        "ReadOnlyToolErrorCode",
        "ReadOnlyToolSpec",
    }
    | _LAZY_FROM_SERVER
)


def __getattr__(name: str):
    """PEP 562:按需从 `app.mcp.server` 取名字,避免 `-m` 下的重复执行。"""
    if name in _LAZY_FROM_SERVER:
        from app.mcp import server as _server

        return getattr(_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_FROM_SERVER)
