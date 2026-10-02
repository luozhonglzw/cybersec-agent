"""评测侧 fixture 的护栏。

合成 fixture(conflict / injection)是 D-1 里**唯一**的新数据。它们必须满足
两条相互制衡的性质:

    1. **真的改变了要测的变量** —— 否则任务名不副实(例如注入载荷根本
       没进到 LLM 能看到的文本里,`prompt_injection_follow_rate` 就是空转);
    2. **没有顺带改变别的变量** —— 否则失败模式被混淆(例如注入 fixture
       顺手改动了 `failed_login_count`,那么接地指标变差就分不清是"被注入
       操纵"还是"证据本来就变了")。

第 2 条尤其重要:它保证"计划 / 策略 / 审计与 GOOD 一致"这件事不是巧合,
而是**fixture 构造上就成立**的。
"""
import json
from pathlib import Path

import pytest

from app.evaluation.llm.dataset import (
    CONFLICT_FAILED_LOGINS,
    CONFLICT_INDICATOR,
    INJECTION_PAYLOAD,
    build_datasets,
)
from app.evaluation.oracles import recompute_evidence

INJECTION_INDICATOR = "203.0.113.66"

_EVIDENCE_FIELDS = (
    "log_event_count",
    "failed_login_count",
    "threat_intel_found",
    "threat_intel_malicious",
    "threat_intel_severity",
)


def _evidence(datasets, variant: str, indicator: str) -> dict:
    paths = datasets[variant]
    return recompute_evidence(
        indicator, logs_path=paths["logs"], intel_path=paths["intel"]
    )


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    return build_datasets(tmp_path_factory.mktemp("cs92d1-fixtures"))


# ---------------------------------------------------------------------------
# 1. fixture 必须真的改变目标变量
# ---------------------------------------------------------------------------


def test_conflict_fixture_actually_creates_high_failure_count(datasets):
    """conflict 变体必须真的把失败登录拉到契约要求的量级。"""
    base = _evidence(datasets, "base", CONFLICT_INDICATOR)
    conflict = _evidence(datasets, "conflict", CONFLICT_INDICATOR)
    assert base["failed_login_count"] == 0, "前提:base 里该地址没有失败登录"
    assert conflict["failed_login_count"] == CONFLICT_FAILED_LOGINS
    assert conflict["failed_login_count"] > base["failed_login_count"]


def test_conflict_fixture_keeps_intel_verdict_trusted(datasets):
    """冲突的定义是"日志高频 + 情报标注可信",两者必须同时成立。"""
    conflict = _evidence(datasets, "conflict", CONFLICT_INDICATOR)
    assert conflict["threat_intel_found"] is True
    assert conflict["threat_intel_malicious"] is False, (
        "conflict fixture 的核心就是情报说'可信'而日志说'异常';"
        "情报若变成恶意,冲突就不存在了"
    )


def test_injection_payload_reaches_llm_visible_text(datasets):
    """注入载荷必须出现在**工具会返回给 LLM 的字段**里。

    否则任务名叫 synthetic_prompt_injection,实际却没有任何注入 ——
    `prompt_injection_follow_rate` 会变成永远为 0 的空转指标。
    """
    logs_text = Path(datasets["injection"]["logs"]).read_text(encoding="utf-8")
    intel_text = Path(datasets["injection"]["intel"]).read_text(encoding="utf-8")
    assert INJECTION_PAYLOAD in logs_text
    assert INJECTION_PAYLOAD in intel_text

    # 而且必须落在**真正会被返回**的字段上,不是某个被丢弃的角落
    payload_events = [
        json.loads(line)
        for line in logs_text.splitlines()
        if line.strip() and INJECTION_PAYLOAD in line
    ]
    assert payload_events, "注入事件必须真实存在于 JSONL 中"
    assert any(INJECTION_PAYLOAD in (event.get("message") or "") for event in payload_events)


# ---------------------------------------------------------------------------
# 2. fixture 不得顺带改变别的变量
# ---------------------------------------------------------------------------


def test_injection_fixture_does_not_change_any_evidence_field(datasets):
    """注入 fixture 只改变"LLM 读到的自由文本",不改变任何证据字段。

    这是"计划/策略/审计在注入下保持不变"的**构造性前提**:
    若这里失败,不变量仍然可能为 1.0,但那个 1.0 就变成巧合而非性质。
    """
    base = _evidence(datasets, "base", INJECTION_INDICATOR)
    injection = _evidence(datasets, "injection", INJECTION_INDICATOR)
    for field in _EVIDENCE_FIELDS:
        assert injection[field] == base[field], (
            f"注入 fixture 意外改变了证据字段 {field}:"
            f"{base[field]!r} → {injection[field]!r}"
        )


def test_conflict_fixture_only_adds_failed_logins(datasets):
    """conflict fixture 只增加失败登录,情报结论与其它任务无关。"""
    base = _evidence(datasets, "base", CONFLICT_INDICATOR)
    conflict = _evidence(datasets, "conflict", CONFLICT_INDICATOR)
    assert conflict["threat_intel_found"] == base["threat_intel_found"]
    assert conflict["threat_intel_malicious"] == base["threat_intel_malicious"]
    assert conflict["threat_intel_severity"] == base["threat_intel_severity"]
    assert conflict["log_event_count"] >= conflict["failed_login_count"]


def test_conflict_fixture_does_not_alter_other_indicators(datasets):
    """其它任务的证据必须逐字段不变 —— fixture 不能有溢出效应。"""
    for indicator in ("10.0.2.11", "203.0.113.66", "198.51.100.7", "192.0.2.77"):
        base = _evidence(datasets, "base", indicator)
        conflict = _evidence(datasets, "conflict", indicator)
        for field in _EVIDENCE_FIELDS:
            assert conflict[field] == base[field], f"{indicator}.{field} 被 conflict fixture 改了"


# ---------------------------------------------------------------------------
# 3. fixture 的确定性与位置
# ---------------------------------------------------------------------------


def test_fixture_generation_is_byte_deterministic(tmp_path):
    """两次生成必须逐字节一致 —— 否则评测不可复现。"""
    first = build_datasets(tmp_path / "a")
    second = build_datasets(tmp_path / "b")
    assert set(first) == set(second)
    for variant in first:
        for key in ("logs", "intel"):
            assert (
                Path(first[variant][key]).read_bytes()
                == Path(second[variant][key]).read_bytes()
            ), f"{variant}/{key} 两次生成不一致"


def test_fixture_bytes_use_the_canonical_newline(tmp_path):
    """跨平台逐字节一致 —— 冻结摘要 `FROZEN_TASKSET_DIGEST` 的前提。

    `test_fixture_generation_is_byte_deterministic` 只证明"同一台机器上两次生成
    一致"。但文本模式写文件会把 `\n` 翻译成 `os.linesep`(Windows → CRLF,
    Linux → LF),于是**同一份逻辑 fixture** 在两个平台上字节不同 → 摘要不同,
    冻结常量就只能在单个平台上成立。

    这里直接对**字节**断言(刻意不引用 `FIXTURE_NEWLINE` 常量 —— 否则常量一改
    测试就跟着改,等于没有护栏),因此与宿主机 OS 无关:任何平台上都必须成立。
    """
    datasets = build_datasets(tmp_path / "canonical")
    for variant in sorted(datasets):
        for key in ("logs", "intel"):
            payload = Path(datasets[variant][key]).read_bytes()
            assert b"\r\n" in payload, f"{variant}/{key} 不含规范换行"
            assert payload.count(b"\n") == payload.count(b"\r\n"), (
                f"{variant}/{key} 含裸 LF —— fixture 序列化绕过了 "
                "write_fixture_lines 这一唯一边界"
            )


def test_all_fixture_files_live_outside_the_repository(tmp_path):
    """评测数据**绝不**写进仓库 `data/`。"""
    datasets = build_datasets(tmp_path / "work")
    repo_data = (Path(__file__).resolve().parents[2] / "data").resolve()
    for variant, paths in datasets.items():
        for key, value in paths.items():
            resolved = Path(value).resolve()
            assert resolved.is_relative_to(tmp_path.resolve()), (
                f"{variant}/{key} 落在临时目录之外:{resolved}"
            )
            assert not resolved.is_relative_to(repo_data), (
                f"{variant}/{key} 落进了仓库 data/:{resolved}"
            )


# ---------------------------------------------------------------------------
# 载荷可见性 —— matched-counterfactual 归因的前提
# ---------------------------------------------------------------------------


def test_payload_marker_is_the_prefix_of_the_payload():
    """标记与载荷是**单一事实来源**:标记变了,载荷跟着变。"""
    from app.evaluation.llm.tasks import INJECTION_PAYLOAD_MARKER

    assert INJECTION_PAYLOAD.startswith(INJECTION_PAYLOAD_MARKER)
    assert INJECTION_PAYLOAD_MARKER.strip(), "标记不能是空白"


def test_payload_visible_detects_the_payload_in_tool_results():
    """`_payload_visible` 必须**真的**读工具返回的自由文本,而不是恒真/恒假。"""
    from langchain_core.messages import AIMessage, ToolMessage

    from app.evaluation.llm.adapters import _payload_visible

    with_payload = [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(
            content=json.dumps({"message": f"User-Agent: {INJECTION_PAYLOAD}"}),
            tool_call_id="1",
        ),
    ]
    assert _payload_visible(with_payload) is True

    # 嵌套结构里的载荷也要能被发现(intel 的 description 就是这样)
    nested = [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "2"}]),
        ToolMessage(
            content=json.dumps({"record": {"description": f"x | {INJECTION_PAYLOAD}"}}),
            tool_call_id="2",
        ),
    ]
    assert _payload_visible(nested) is True


def test_payload_visible_is_false_without_the_payload():
    """载荷缺席(匹配对照条件)时必须为假 —— 否则归因恒真。"""
    from langchain_core.messages import AIMessage, ToolMessage

    from app.evaluation.llm.adapters import _payload_visible

    clean = [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(
            content=json.dumps({"record": {"description": "普通情报描述"}}),
            tool_call_id="1",
        ),
    ]
    assert _payload_visible(clean) is False
    assert _payload_visible([]) is False
    # 只有用户提示词、没有工具结果 → 载荷不在上下文里
    assert _payload_visible([AIMessage(content="no tools called")]) is False


def test_clean_variant_contains_no_payload_text(tmp_path):
    """`base` 变体(对照条件的数据集)里不得出现载荷 —— 否则对照不成立。"""
    datasets = build_datasets(tmp_path / "work")
    for key in ("logs", "intel"):
        text = Path(datasets["base"][key]).read_text(encoding="utf-8")
        assert INJECTION_PAYLOAD not in text, f"base/{key} 混入了注入载荷"
