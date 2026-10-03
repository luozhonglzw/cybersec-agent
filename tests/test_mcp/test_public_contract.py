"""公开契约不变式(Phase 9.3-E §9 / §11)。

这里检查的不是"工具返回得对不对",而是**对外契约有没有越界**:

    1. 暴露面只有三个只读工具,处置类工具不在其中;
    2. 公开参数集 == 来源应用工具参数集 **减去**内部路径参数;
    3. 下发的 wire schema 里没有任何路径字段;
    4. 生产 MCP 代码**没有**引入 `_SCHEMA_PATH_FIELDS`(不靠评测侧常量兜底);
    5. 封闭词表(严重级别 / IOC 类别)与 `app/tools` 的词表不漂移;
    6. 冻结的工具 schema 身份摘要未被改动。

为什么这些要写成机械断言:"哪天顺手把 `data_path` 加回去好排查"、
"哪天把路径字段表 import 进来做过滤"都是完全可预见的演化方向。
"""
import ast
import json
from inspect import signature
from pathlib import Path
from typing import Literal, get_args, get_type_hints

from app.evaluation.llm.runner import _SCHEMA_PATH_FIELDS, tool_schema_sha256
from app.mcp import READONLY_TOOL_SPECS, build_mcp_server
from app.mcp.tools import (
    IndicatorTypeLiteral,
    SeverityLiteral,
    analyze_risk_readonly,
    query_security_logs_readonly,
    query_threat_intel_readonly,
)
from app.tools import DEFAULT_TOOLS
from app.tools.query_logs import SEVERITY_ORDER
from app.tools.query_threat_intel import VALID_INDICATOR_TYPES

#: 9.3-C1 起冻结的工具 schema 身份摘要。MCP 只读暴露**不得**改变它。
FROZEN_TOOL_SCHEMA_SHA256 = (
    "44d77a0ce8b1dc7471cca840d13f62b72bc06ef16131476a16fca363440194f5"
)

MCP_PKG = Path(__file__).resolve().parents[2] / "app" / "mcp"


def _literal_members(annotation) -> set[str]:
    """从 `Literal[...] | None` 这样的注解里取出 Literal 的成员。"""
    for arg in get_args(annotation):
        if get_args(arg) and getattr(arg, "__origin__", None) is Literal:
            return set(get_args(arg))
    raise AssertionError(f"注解里没有 Literal:{annotation!r}")


async def _wire_schemas(open_client) -> dict[str, dict]:
    """取**客户端实际看到**的 input_schema(而不是我们内部的签名)。"""
    async with open_client() as client:
        listed = await client.list_tools()
    return {tool.name: tool.input_schema for tool in listed.tools}


# ---------------------------------------------------------------------------
# 1. 暴露面
# ---------------------------------------------------------------------------


async def test_only_the_three_readonly_tools_are_exposed(open_client):
    schemas = await _wire_schemas(open_client)
    assert set(schemas) == {spec.name for spec in READONLY_TOOL_SPECS}
    assert set(schemas) == {"query_security_logs", "query_threat_intel", "analyze_risk"}


async def test_response_planning_is_not_exposed(open_client):
    """`plan_response` 会产出需人工审批的处置动作 ⇒ 不得进入只读接口。"""
    schemas = await _wire_schemas(open_client)
    assert "plan_response" not in schemas
    assert "plan_response_tool" not in schemas
    assert all("plan" not in name for name in schemas)


def test_specs_point_at_real_application_tools():
    """每个 spec 声明的来源工具都必须在 `DEFAULT_TOOLS` 里真实存在。"""
    by_name = {tool.name: tool for tool in DEFAULT_TOOLS}
    assert {spec.source_tool for spec in READONLY_TOOL_SPECS} <= set(by_name)


# ---------------------------------------------------------------------------
# 2 / 3. 参数集与 wire schema
# ---------------------------------------------------------------------------


async def test_public_parameter_set_is_source_tool_parameters_minus_paths(open_client):
    """§11 的核心不变式:公开参数集 = 来源工具参数集 − 内部路径参数。"""
    schemas = await _wire_schemas(open_client)
    by_name = {tool.name: tool for tool in DEFAULT_TOOLS}
    for spec in READONLY_TOOL_SPECS:
        source_params = set(by_name[spec.source_tool].args)
        expected = source_params - set(_SCHEMA_PATH_FIELDS)
        actual = set(schemas[spec.name]["properties"])
        assert actual == expected, (
            f"{spec.name}: 公开参数集与来源工具不一致;"
            f"多出 {sorted(actual - expected)},缺少 {sorted(expected - actual)}"
        )


async def test_public_parameter_set_matches_the_adapter_signature(open_client):
    """wire schema 必须与适配器签名同源 —— 两者不得分叉。"""
    schemas = await _wire_schemas(open_client)
    for spec in READONLY_TOOL_SPECS:
        declared = set(signature(spec.fn).parameters)
        assert set(schemas[spec.name]["properties"]) == declared, (
            f"{spec.name}: wire schema 与适配器签名不一致"
        )


async def test_no_path_field_reaches_the_wire_schema(open_client):
    """任何路径字段都不得出现在下发的 schema 里。"""
    schemas = await _wire_schemas(open_client)
    for name, schema in schemas.items():
        properties = set(schema.get("properties", {}))
        assert not (properties & set(_SCHEMA_PATH_FIELDS)), (
            f"{name}: 公开 schema 出现路径字段 {sorted(properties & set(_SCHEMA_PATH_FIELDS))}"
        )
        # 路径字段的默认值会内嵌平台相关分隔符;整个 schema 文本里也不该出现。
        rendered = json.dumps(schema)
        assert "data_path" not in rendered
        assert "logs_path" not in rendered
        assert "intel_path" not in rendered


async def test_wire_schema_does_not_promise_extra_property_rejection(open_client):
    """记录 SDK 的行为事实:`input_schema` **不带** `additionalProperties`。

    这不是缺陷声明,而是本层为什么需要 `ReadOnlyArgumentGuard` 的**前提**:
    既然 schema 自己不拒绝多余键,拒绝就必须由我们显式提供。
    若将来 SDK 补上了 `additionalProperties: false`,本断言会失败,
    届时守卫可以简化 —— 这是一个"事实变了就响"的哨兵。
    """
    schemas = await _wire_schemas(open_client)
    for name, schema in schemas.items():
        assert "additionalProperties" not in schema, (
            f"{name}: SDK 已开始下发 additionalProperties,请重新评估守卫的必要性"
        )


# ---------------------------------------------------------------------------
# 4. 生产代码不得依赖评测侧常量
# ---------------------------------------------------------------------------


def test_production_mcp_code_does_not_import_the_path_field_registry():
    """§11:生产 MCP 代码不得 import `_SCHEMA_PATH_FIELDS` 之类的东西。

    路径排除必须是**签名事实**,不是"拿一张表过滤"。表在评测侧、会变;
    签名不会。
    """
    files = sorted(MCP_PKG.glob("*.py"))
    assert files, "没有找到 app/mcp 下的模块"
    offenders: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("app.evaluation"):
                    offenders.append(f"{path.name}: from {node.module}")
                if any(alias.name == "_SCHEMA_PATH_FIELDS" for alias in node.names):
                    offenders.append(f"{path.name}: imports _SCHEMA_PATH_FIELDS")
            elif isinstance(node, ast.Import):
                if any(alias.name.startswith("app.evaluation") for alias in node.names):
                    offenders.append(f"{path.name}: import app.evaluation")
    assert not offenders, f"生产 MCP 代码不得依赖评测侧常量:{offenders}"


def test_adapter_signatures_contain_no_path_parameter():
    """签名层排除:适配器形参名里不得出现任何已登记的路径字段。"""
    for fn in (
        query_security_logs_readonly,
        query_threat_intel_readonly,
        analyze_risk_readonly,
    ):
        params = set(signature(fn).parameters)
        assert not (params & set(_SCHEMA_PATH_FIELDS)), (
            f"{fn.__name__}: 公开签名出现路径形参 {sorted(params & set(_SCHEMA_PATH_FIELDS))}"
        )


# ---------------------------------------------------------------------------
# 5. 词表不漂移
# ---------------------------------------------------------------------------


def test_severity_literal_matches_the_application_vocabulary():
    assert _literal_members(get_type_hints(query_security_logs_readonly)["min_severity"]) == set(
        SEVERITY_ORDER
    )
    assert set(get_args(SeverityLiteral)) == set(SEVERITY_ORDER)


def test_indicator_type_literal_matches_the_application_vocabulary():
    assert _literal_members(
        get_type_hints(query_threat_intel_readonly)["indicator_type"]
    ) == set(VALID_INDICATOR_TYPES)
    assert set(get_args(IndicatorTypeLiteral)) == set(VALID_INDICATOR_TYPES)


# ---------------------------------------------------------------------------
# 6. 冻结面
# ---------------------------------------------------------------------------


def test_frozen_tool_schema_digest_is_unchanged():
    """MCP 暴露层不得改变既有工具 schema 的身份摘要。"""
    assert tool_schema_sha256() == FROZEN_TOOL_SCHEMA_SHA256


def test_building_the_server_does_not_mutate_the_application_tool_list():
    """装配 MCP server 不得触碰 `DEFAULT_TOOLS`(顺序与身份都不变)。"""
    before = [(tool.name, id(tool)) for tool in DEFAULT_TOOLS]
    build_mcp_server()
    after = [(tool.name, id(tool)) for tool in DEFAULT_TOOLS]
    assert before == after
