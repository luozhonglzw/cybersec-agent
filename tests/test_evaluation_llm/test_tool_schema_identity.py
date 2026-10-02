"""工具 schema 身份摘要的**跨平台同一性**。

冻结摘要 `FROZEN_TOOL_SCHEMA_SHA256` 是在某一个平台上标定的;而工具 schema 里路径类
字段的默认值来自 `str(Path(...))`,其分隔符随宿主变化(Windows `\\` / Linux `/`)。
身份摘要吃的是序列化后的字节,所以只要身份层直接吃这个字符串,**同一份逻辑 schema**
就会在两个平台上算出两个摘要 —— 冻结常量便只能在单个平台上成立。

这里用**字面量**构造两个平台表示(而不是引用实现里的规范分隔符常量:常量一改测试
就跟着改,等于没有护栏)。
"""
from app.evaluation.llm.runner import (
    _SCHEMA_PATH_FIELDS,
    _tool_schema_digest_for,
    tool_schema_sha256,
)
from app.tools import DEFAULT_TOOLS

FROZEN_TOOL_SCHEMA_SHA256 = (
    "44d77a0ce8b1dc7471cca840d13f62b72bc06ef16131476a16fca363440194f5"
)


def test_tool_schema_identity_is_host_independent():
    """同一份逻辑 schema 的两个平台表示 → 同一个身份摘要;且规范化**有界**。"""
    # 1) 跨平台同一性:路径默认值的分隔符差异必须被身份层吸收。
    posix = {
        "t": {"data_path": {"type": "string", "default": "data/security_events.jsonl"}}
    }
    windows = {
        "t": {
            "data_path": {
                "type": "string",
                "default": "data\\security_events.jsonl",
            }
        }
    }
    assert _tool_schema_digest_for(posix) == _tool_schema_digest_for(windows)

    # 2) 有界性(护栏有牙):**非路径字段**的同类差异不得被抹平 ——
    #    否则"规范化"就退化成"对整段 JSON 做全局 replace"。
    note_posix = {"t": {"note": {"type": "string", "default": "a/b"}}}
    note_windows = {"t": {"note": {"type": "string", "default": "a\\b"}}}
    assert _tool_schema_digest_for(note_posix) != _tool_schema_digest_for(note_windows)

    # 3) 声明的路径字段集合必须**完备**:凡是默认值含路径分隔符的字段都要登记在案。
    detected = {
        field
        for tool in DEFAULT_TOOLS
        for field, spec in tool.args.items()
        if isinstance(spec.get("default"), str)
        and ("/" in spec["default"] or "\\" in spec["default"])
    }
    assert detected == set(_SCHEMA_PATH_FIELDS), (
        f"schema 里出现未声明的路径字段:{sorted(detected - set(_SCHEMA_PATH_FIELDS))}"
    )

    # 4) 真实 schema 的身份摘要仍等于冻结值(期望值**未改**)。
    assert tool_schema_sha256() == FROZEN_TOOL_SCHEMA_SHA256
