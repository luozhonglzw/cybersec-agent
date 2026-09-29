"""D-2d 路径限定:把路径类工具参数**在工具执行之前**绑定到本单元授权的数据路径。

根因(为什么必须有这一层)
------------------------
四个生产工具共接受 **6 个**路径类参数(`data_path` / `logs_path` / `intel_path`),
它们的值**直接**进入 `Path(value)` / `open()`,生产侧**没有任何**路径限定。
`app.core.graph.tools_node` 把 provider 生成的 `tool_calls[i]["args"]`
**原样**交给 `tool.ainvoke`,只按**工具名**查表,**从不校验参数值**。

离线路径之所以看起来正常,是因为 `ScriptedLLM` **主动**把评测授权路径传进
工具 —— 那是一条**约定**,不是**守卫**(见 `adapters.ScriptedLLM` 的 docstring,
它自己就写明了"不传的话工具会落到 `data/security_events.jsonl`(仓库数据),
于是合成 fixture(conflict / injection)也会完全失效")。真实 provider 不受该
约定约束:它既不知道隔离夹具的路径,也没有义务使用它们。

失效是**静默且选择性**的:`write_seed_dataset` 与仓库 `data/` 同源生成,因此
`base` 变体看起来完全正常;而 `conflict` 变体会读到错误证据、`injection` 变体
的载荷**根本到不了模型** —— 处理组被摧毁,`prompt_injection_follow_rate` 却会
读成"抵抗"。既有指标 `path_argument_deviation_rate` **结构上测不出**这种失效:
它的分母明确定义为"实际传了路径参数的 (调用, 参数名) 对数(未传者不进分母)",
省略路径反而得满分。

本模块提供的机制
----------------
`confine_tools(tools, dataset_paths=..., log=...)` 返回一组**评测作用域**的工具
包装器:名字 / 描述 / `args_schema` 与生产工具**逐字段一致**(因此 `bind_tools`
生成的线上 schema 与 `tool_schema_sha256` 不变),`ainvoke` 时先把路径类参数
**确定性地**解析为本单元授权的 `DatasetPaths` 值,再委托给**同一个**生产工具
对象执行。因此:

    * 生产工具实现**零改动**;
    * `DEFAULT_TOOLS` **零改动**(不注册、不替换、不改全局语义);
    * 提示词(系统 / 任务)**零改动**;
    * 数据集路径是**实验环境状态**,不是 LLM 的实验决策变量。

作用域与不可绕过性
------------------
本模块**不 import 任何生产模块** —— 工具对象由调用方传入,因此
`tests/test_evaluation_llm/test_independence.py` 记录的
生产导入边界(`adapters.py` / `runner.py`)**保持不变**。

限定在 `adapters.py` 的**图构建边界**上生效:包装器是 `create_agent_graph`
唯一能看到的工具对象,因此 provider 生成的任何参数都必然经过本层,
**无法绕过**。未启用时(离线 D-1 矩阵)工具对象原样传入,`PATH_DEVIATION`
等既有探针的行为逐字节不变。

可观测性(不得静默抹掉 provider 生成的东西)
------------------------------------------
每次绑定都记一条 `PathBindingRecord`,**同时**保留 provider 原始生成值
(`model_supplied_value`)与实际生效值(`effective_value`),并用 `disposition`
说明处置类别。`AIMessage.tool_calls` 里的原始参数**不被改写** —— 因此
`path_argument_deviation_rate` 等既有指标的语义**不变**,新的绑定证据单独暴露。
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from app.evaluation.llm.tasks import SecurityContract

#: 路径类参数名的**冻结词表**。刻意从 `SecurityContract.path_arguments` 的
#: 默认工厂取,而不是另写一份 —— 两处若各写一份,加参数时必然漂移。
PATH_ARGUMENT_NAMES: tuple[str, ...] = tuple(
    SecurityContract.model_fields["path_arguments"].default_factory()
)

#: 每个工具**哪些**参数承载路径,以及它指向**哪个**数据集键。
#:
#: 参数名取自上面的冻结词表;映射关系由工具签名决定,并由
#: `tests/test_evaluation_llm/test_d2d_path_confinement.py` 与
#: `SecurityContract.path_arguments` 交叉核对。
PATH_ARGUMENT_TARGETS: dict[str, dict[str, str]] = {
    "query_security_logs_tool": {"data_path": "logs"},
    "query_threat_intel_tool": {"data_path": "intel"},
    "analyze_risk_tool": {"logs_path": "logs", "intel_path": "intel"},
    "plan_response_tool": {"logs_path": "logs", "intel_path": "intel"},
}


class PathDisposition(str, Enum):
    """provider 生成的路径参数被如何处置。**三值封闭词表,不得扩展取值。**"""

    #: provider 给的正是本单元授权的路径(原始写法或 resolve 后等价)。
    AUTHORIZED_AS_SUPPLIED = "AUTHORIZED_AS_SUPPLIED"
    #: provider **没有**给这个参数 —— 由运行时按单元授权路径绑定。
    #: 这一支正是旧行为下会静默回落到仓库 `data/` 的那一支。
    MISSING_BOUND_BY_RUNTIME = "MISSING_BOUND_BY_RUNTIME"
    #: provider 给了,但不是本单元授权的路径(绝对路径越权 / `../` 穿越 /
    #: 别的变体的夹具 / 仓库 data 路径)—— 一律**覆盖**,绝不按原值读取。
    UNAUTHORIZED_OVERRIDDEN = "UNAUTHORIZED_OVERRIDDEN"


class PathBindingRecord(BaseModel):
    """一次(工具调用 × 路径参数)的绑定事实。**原始值与生效值并存。**"""

    tool_name: str = Field(min_length=1)
    argument_name: str = Field(min_length=1)
    dataset_key: str = Field(min_length=1, description="logs / intel")
    model_argument_present: bool = Field(
        description="provider 是否给了这个参数(空串 / None 视为没给)"
    )
    model_supplied_value: str | None = Field(
        default=None, description="provider 原始生成值;没给则为 None"
    )
    authorized_value: str = Field(description="本单元授权的数据路径")
    effective_value: str = Field(description="实际交给工具的值(**恒等于授权值**)")
    disposition: PathDisposition

    @property
    def overridden(self) -> bool:
        return self.disposition is PathDisposition.UNAUTHORIZED_OVERRIDDEN


class PathBindingLog:
    """一次单元运行内的绑定记录收集器(可变;逐次 `ainvoke` 追加)。"""

    def __init__(self) -> None:
        self.records: list[PathBindingRecord] = []

    def extend(self, records: Iterable[PathBindingRecord]) -> None:
        self.records.extend(records)

    def as_payload(self) -> list[dict[str, Any]]:
        return [record.model_dump(mode="json") for record in self.records]

    def __len__(self) -> int:  # pragma: no cover - 便利方法
        return len(self.records)


def _equivalent_forms(value: str) -> set[str]:
    """一个路径的**两种合法写法**:原样 与 `resolve()` 后的绝对路径。

    与 `runner.authorized_paths_for` 的口径一致 —— 判定必须对两种写法都成立,
    否则会把"写法不同"误判成"越权"。
    """
    forms = {value}
    try:
        forms.add(str(Path(value).resolve()))
    except OSError:  # pragma: no cover - resolve 在正常路径上不会抛
        pass
    return forms


def resolve_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    dataset_paths: dict[str, str],
    log: PathBindingLog | None = None,
) -> dict[str, Any]:
    """把 `arguments` 里的路径类参数解析为本单元授权路径。

    * 无路径参数的工具 → 原样返回(**不复制语义,不做任何改动**)。
    * 有路径参数的工具 → 每个路径参数**一律**被替换为授权值;同时产出一条
      `PathBindingRecord` 说明 provider 原本给了什么。

    `dataset_paths` 缺键时**抛 `KeyError`** —— 配置错误必须立刻暴露,
    绝不退回到任何默认路径。
    """
    targets = PATH_ARGUMENT_TARGETS.get(tool_name)
    if not targets:
        return dict(arguments)

    resolved = dict(arguments)
    records: list[PathBindingRecord] = []
    for argument_name, dataset_key in targets.items():
        authorized = dataset_paths[dataset_key]
        raw = arguments.get(argument_name)
        present = raw is not None and str(raw).strip() != ""

        if not present:
            disposition = PathDisposition.MISSING_BOUND_BY_RUNTIME
            supplied: str | None = None
        elif str(raw) in _equivalent_forms(authorized):
            disposition = PathDisposition.AUTHORIZED_AS_SUPPLIED
            supplied = str(raw)
        else:
            disposition = PathDisposition.UNAUTHORIZED_OVERRIDDEN
            supplied = str(raw)

        resolved[argument_name] = authorized
        records.append(
            PathBindingRecord(
                tool_name=tool_name,
                argument_name=argument_name,
                dataset_key=dataset_key,
                model_argument_present=present,
                model_supplied_value=supplied,
                authorized_value=authorized,
                effective_value=authorized,
                disposition=disposition,
            )
        )

    if log is not None:
        log.extend(records)
    return resolved


def uncovered_path_arguments(tools: Iterable[Any]) -> list[tuple[str, str]]:
    """列出**未被映射覆盖**的路径类参数。

    判据:参数名落在冻结词表 `PATH_ARGUMENT_NAMES` 内,或以后缀 `_path` 结尾
    (新工具最可能用的命名)。返回非空即表示"新加了路径参数却没人管它" ——
    此时必须**拒绝执行**,而不是让它悄悄绕过限定。

    没有这条检查,限定会随生产工具演化**静默失效**:新工具的路径参数不在映射里,
    `resolve_arguments` 原样放行,而一切看起来都正常。
    """
    uncovered: list[tuple[str, str]] = []
    for tool in tools:
        covered = PATH_ARGUMENT_TARGETS.get(getattr(tool, "name", ""), {})
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            continue
        properties = (schema.model_json_schema() or {}).get("properties") or {}
        for argument_name in properties:
            looks_like_path = (
                argument_name in PATH_ARGUMENT_NAMES or argument_name.endswith("_path")
            )
            if looks_like_path and argument_name not in covered:
                uncovered.append((tool.name, argument_name))
    return uncovered


class UnconfinedPathArgument(RuntimeError):
    """存在**未被限定覆盖**的路径类参数。**拒绝执行**,不静默放行。"""


def confine_tool(
    tool: Any,
    *,
    dataset_paths: dict[str, str],
    log: PathBindingLog | None = None,
) -> Any:
    """把单个工具包成**路径受限**的等价工具。

    名字 / 描述 / `args_schema` 与原件**逐字段一致**(因此 provider 看到的
    schema 不变);`ainvoke` 先解析路径参数,再委托**同一个**工具对象执行。
    """
    name = getattr(tool, "name", "")
    if name not in PATH_ARGUMENT_TARGETS:
        # 无路径参数的工具**原样返回** —— 连对象身份都不变。
        return tool

    from langchain_core.tools import StructuredTool  # noqa: PLC0415 - 惰性,避免顶层依赖

    async def _arun(**kwargs: Any) -> Any:
        resolved = resolve_arguments(
            name, kwargs, dataset_paths=dataset_paths, log=log
        )
        return await tool.ainvoke(resolved)

    return StructuredTool(
        name=name,
        description=getattr(tool, "description", "") or "",
        args_schema=tool.args_schema,
        coroutine=_arun,
    )


def confine_tools(
    tools: Iterable[Any],
    *,
    dataset_paths: dict[str, str],
    log: PathBindingLog | None = None,
) -> list[Any]:
    """把一组工具整体包成路径受限版本(fail closed)。

    未覆盖的路径类参数存在时**抛 `UnconfinedPathArgument`**,而不是放行 ——
    "限定看起来在、实际漏了一个参数"比没有限定更糟。
    """
    materialized = list(tools)
    uncovered = uncovered_path_arguments(materialized)
    if uncovered:
        raise UnconfinedPathArgument(
            f"存在未被路径限定覆盖的路径类参数:{sorted(uncovered)} —— "
            "请把它们加进 PATH_ARGUMENT_TARGETS,或显式确认它们不是路径。"
        )
    return [
        confine_tool(tool, dataset_paths=dataset_paths, log=log)
        for tool in materialized
    ]
