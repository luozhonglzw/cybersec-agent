"""MCP 只读工具适配层(Phase 9.3-E)。

职责边界 —— 本模块**只做适配**,不做任何安全判断:
    - 复用 `app/tools/` 的核心查询 / 分析函数(与 LangChain 工具同一实现);
    - 把「客户端可控参数面」收缩到**纯业务参数**;
    - 把内部数据文件路径**钉死在服务端**,不出现在任何公开签名里;
    - 把失败收敛成**固定分类 + 固定文案**(不含路径、不含回溯)。

为什么路径必须**结构性**消失,而不是「忽略」:
    `query_security_logs(..., data_path=...)` / `query_threat_intel(..., data_path=...)` /
    `collect_evidence(..., logs_path=..., intel_path=...)` 里的路径形参是**任意文件读原语** ——
    谁控制它,谁就能让工具去读任意可读文件;并且底层异常文本会把路径**回显**出来
    (例如 `FileNotFoundError(f"... {data_path}(先运行 scripts/seed_logs.py 生成)")`)。
    MCP 客户端属于外部输入源,因此这些形参**不得**出现在公开契约中。

    而 mcp 2.3.0 对**未声明**的多余参数是**静默忽略**的:`Tool.fn_metadata` 生成的参数
    模型未设 `extra="forbid"`,下发的 `input_schema` 也不带 `additionalProperties`。
    所以「路径字段不在 schema 里」**不足以**构成拒绝。本层因此做两件事:

      1. **签名层**:适配器函数**没有任何路径形参** —— 即使同进程直接调用
         `server.call_tool(...)` 注入 `data_path`,也只会被 SDK 丢弃,无法抵达文件 I/O;
      2. **协议层**:`app/mcp/server.py` 的 `ReadOnlyArgumentGuard` 中间件在参数校验
         **之前**核对键集合,出现任何未声明键即以 `INVALID_PARAMS` 显式否决调用。

    两层互为独立机制:任一层被未来的重构移除,另一层仍然成立。

公开工具名与来源工具(`app.tools.DEFAULT_TOOLS`):
    ``query_security_logs`` ← ``query_security_logs_tool``
    ``query_threat_intel``  ← ``query_threat_intel_tool``
    ``analyze_risk``        ← ``analyze_risk_tool``

刻意不暴露 ``plan_response_tool``:它产出需要人工审批的处置动作(写路径),
不属于「只读暴露」的范围。

失败分类(封闭词表,见 `ReadOnlyToolErrorCode`):

    ============================================ ==========================
    触发条件                                      分类
    ============================================ ==========================
    参数不合法(schema 层或适配层自检)             ``INVALID_ARGUMENT``
    数据源缺失 / 不可读 / 损坏                     ``DATA_UNAVAILABLE``
    工具本身抛出未分类异常                         ``TOOL_FAILURE``
    结果无法转成 JSON 兼容形式                     ``INTERNAL_ERROR``
    ============================================ ==========================
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Callable, Literal

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app.tools.query_logs import DEFAULT_DATA_PATH as LOGS_DATA_PATH
from app.tools.query_logs import DEFAULT_LIMIT, MAX_LIMIT, query_security_logs
from app.tools.query_threat_intel import DEFAULT_DATA_PATH as INTEL_DATA_PATH
from app.tools.query_threat_intel import query_threat_intel
from app.tools.risk_analyzer import analyze_risk, collect_evidence


class ReadOnlyToolErrorCode(str, Enum):
    """只读工具失败的**封闭**分类词表。自由文本一律不在此列。"""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    TOOL_FAILURE = "TOOL_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


#: 每个分类对外的**固定**文案。
#:
#: 固定文案是刻意的:底层异常文本可能内嵌数据文件路径,把 ``str(exc)`` 透出去
#: 等于把服务端的目录结构回显给外部调用方。分类码 + 固定文案之外不携带任何
#: 来自底层异常的字节。
_SAFE_MESSAGES: dict[ReadOnlyToolErrorCode, str] = {
    ReadOnlyToolErrorCode.INVALID_ARGUMENT: (
        "Invalid argument: the request did not satisfy this tool's validation rules."
    ),
    ReadOnlyToolErrorCode.DATA_UNAVAILABLE: (
        "Data unavailable: the backing data source could not be read."
    ),
    ReadOnlyToolErrorCode.TOOL_FAILURE: (
        "Tool failure: the request could not be completed."
    ),
    ReadOnlyToolErrorCode.INTERNAL_ERROR: (
        "Internal error: the result could not be produced."
    ),
}


def _tool_error(code: ReadOnlyToolErrorCode) -> ToolError:
    """构造对外错误。分类码进消息前缀,便于调用方机械判别与测试断言。

    返回 SDK 的 ``ToolError`` 而不是自定义异常:它会被 SDK 转成
    ``is_error=True`` 的工具结果(而不是崩溃),消息按原样保留;
    原始异常只作为 ``__cause__`` 留在服务端。
    """
    return ToolError(f"{code.value}: {_SAFE_MESSAGES[code]}")


#: 严重级别的封闭词表。与 ``app.tools.query_logs.SEVERITY_ORDER`` 的键集
#: **必须一致** —— 由 ``tests/test_mcp/test_public_contract.py`` 机械核对,
#: 防止两处词表悄悄分叉。
SeverityLiteral = Literal["info", "low", "medium", "high", "critical"]

#: IOC 类别的封闭词表。与 ``app.tools.query_threat_intel.VALID_INDICATOR_TYPES``
#: 必须一致(同样由测试核对)。
IndicatorTypeLiteral = Literal["ip", "domain", "hash"]


# ---------------------------------------------------------------------------
# 公开适配器 —— 形参即公开契约,内部路径一律不出现在这里
# ---------------------------------------------------------------------------


def query_security_logs_readonly(
    event_type: str | None = None,
    source_ip: str | None = None,
    username: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    min_severity: SeverityLiteral | None = None,
    limit: Annotated[int, Field(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Query the local security event log (read-only).

    Filters are conjunctive; omitting one means "do not filter on it".
    ``start_time`` / ``end_time`` are ISO 8601; a value without a timezone is
    read as UTC. Events come back sorted by timestamp, capped by ``limit``.
    """
    start_dt = _parse_iso8601(start_time)
    end_dt = _parse_iso8601(end_time)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise _tool_error(ReadOnlyToolErrorCode.INVALID_ARGUMENT)

    events = _call_core(
        query_security_logs,
        event_type=event_type,
        source_ip=source_ip,
        username=username,
        start_time=start_dt,
        end_time=end_dt,
        min_severity=min_severity,
        limit=limit,
        data_path=LOGS_DATA_PATH,
    )
    return {"count": len(events), "events": [_dump(event) for event in events]}


def query_threat_intel_readonly(
    indicator: str,
    indicator_type: IndicatorTypeLiteral | None = None,
) -> dict[str, Any]:
    """Look up one indicator of compromise in the local threat-intel store (read-only).

    The lookup is an **exact** match on the indicator string, not a fuzzy or
    similarity search. ``indicator_type`` optionally narrows the match.
    """
    if not indicator or not indicator.strip():
        raise _tool_error(ReadOnlyToolErrorCode.INVALID_ARGUMENT)

    record = _call_core(
        query_threat_intel,
        indicator=indicator,
        indicator_type=indicator_type,
        data_path=INTEL_DATA_PATH,
    )
    if record is None:
        return {
            "found": False,
            "indicator": indicator,
            "message": "No threat intelligence found",
        }
    return {"found": True, "record": _dump(record)}


def analyze_risk_readonly(
    indicator: str,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Assess one indicator's risk from the local logs and threat-intel store (read-only).

    Evidence is collected from the server-side data sources, then scored by the
    deterministic rule engine. Returns the risk level, score, confidence, the
    reasons behind the score, and the evidence snapshot.
    """
    if not indicator or not indicator.strip():
        raise _tool_error(ReadOnlyToolErrorCode.INVALID_ARGUMENT)

    evidence = _call_core(
        collect_evidence,
        indicator=indicator,
        event_type=event_type,
        logs_path=LOGS_DATA_PATH,
        intel_path=INTEL_DATA_PATH,
    )
    assessment = _call_core(analyze_risk, evidence=evidence)
    return {"assessment": _dump(assessment)}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _parse_iso8601(value: str | None) -> datetime | None:
    """解析 ISO 8601 时间;失败时抛**固定文案**的 INVALID_ARGUMENT。

    底层 ``ValueError`` 的文本可能回显调用方输入,一律不透出。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise _tool_error(ReadOnlyToolErrorCode.INVALID_ARGUMENT)
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise _tool_error(ReadOnlyToolErrorCode.INVALID_ARGUMENT) from exc


def _call_core(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """调用核心实现,并把底层异常收敛成**安全**的分类错误。

    调用方参数已由 schema 层(``Literal`` / ``Field(ge=, le=)``)与适配层自检拦下,
    因此核心层残留的 ``ValueError`` 归因于**数据完整性**(例如 JSONL 某行损坏),
    与 ``OSError`` 一并归入 ``DATA_UNAVAILABLE``;其余异常归 ``TOOL_FAILURE``。
    """
    try:
        return fn(**kwargs)
    except (FileNotFoundError, OSError) as exc:
        raise _tool_error(ReadOnlyToolErrorCode.DATA_UNAVAILABLE) from exc
    except ValueError as exc:
        raise _tool_error(ReadOnlyToolErrorCode.DATA_UNAVAILABLE) from exc
    except Exception as exc:  # noqa: BLE001 - 兜底:不让任何底层异常文本外泄
        raise _tool_error(ReadOnlyToolErrorCode.TOOL_FAILURE) from exc


def _dump(model: Any) -> dict[str, Any]:
    """把领域模型转成 JSON 兼容 dict;失败归 ``INTERNAL_ERROR``。"""
    try:
        return model.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - 兜底
        raise _tool_error(ReadOnlyToolErrorCode.INTERNAL_ERROR) from exc


# ---------------------------------------------------------------------------
# 注册清单
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReadOnlyToolSpec:
    """一个只读工具的公开契约。

    ``source_tool`` 是它在 ``app.tools.DEFAULT_TOOLS`` 中的对应工具名 ——
    公开参数集**必须**等于来源工具参数集减去内部路径参数,该不变式由
    ``tests/test_mcp/test_public_contract.py`` 机械核对。
    """

    name: str
    source_tool: str
    fn: Callable[..., Any]
    title: str
    description: str


READONLY_TOOL_SPECS: tuple[ReadOnlyToolSpec, ...] = (
    ReadOnlyToolSpec(
        name="query_security_logs",
        source_tool="query_security_logs_tool",
        fn=query_security_logs_readonly,
        title="Query security logs",
        description=(
            "Query the local security event log and return matching events, "
            "sorted by timestamp. Read-only."
        ),
    ),
    ReadOnlyToolSpec(
        name="query_threat_intel",
        source_tool="query_threat_intel_tool",
        fn=query_threat_intel_readonly,
        title="Query threat intelligence",
        description=(
            "Look up one indicator of compromise (IP / domain / hash) in the "
            "local threat-intelligence store by exact match. Read-only."
        ),
    ),
    ReadOnlyToolSpec(
        name="analyze_risk",
        source_tool="analyze_risk_tool",
        fn=analyze_risk_readonly,
        title="Analyze risk",
        description=(
            "Assess one indicator's risk level from the local logs and "
            "threat-intelligence store using the deterministic rule engine. Read-only."
        ),
    ),
)

__all__ = [
    "IndicatorTypeLiteral",
    "READONLY_TOOL_SPECS",
    "ReadOnlyToolErrorCode",
    "ReadOnlyToolSpec",
    "SeverityLiteral",
    "analyze_risk_readonly",
    "query_security_logs_readonly",
    "query_threat_intel_readonly",
]
