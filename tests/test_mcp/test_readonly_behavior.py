"""只读适配器的行为与失败分类。

失败分类是**封闭**的(见 `ReadOnlyToolErrorCode`),而且对外只给固定文案:
底层异常文本可能内嵌数据文件路径,透出去等于把服务端目录结构回显给外部调用方。
本文件同时验证"分类正确"和"文案没有泄漏"。
"""
import app.mcp.tools as mcp_tools
from app.mcp.tools import ReadOnlyToolErrorCode


async def _error_text(tool_name: str, arguments: dict, open_client) -> str:
    async with open_client() as client:
        result = await client.call_tool(tool_name, arguments)
    assert result.is_error is True
    return "\n".join(block.text for block in result.content)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


async def test_query_security_logs_returns_matching_events(seeded, open_client):
    async with open_client() as client:
        result = await client.call_tool("query_security_logs", {"limit": 50})
    assert result.is_error is False
    payload = result.structured_content
    assert payload["count"] == 5
    assert [event["event_type"] for event in payload["events"]][0] == "firewall_allow"


async def test_query_security_logs_filters_are_conjunctive(seeded, open_client):
    async with open_client() as client:
        result = await client.call_tool("query_security_logs", {"min_severity": "high", "limit": 50})
    severities = {event["severity"] for event in result.structured_content["events"]}
    assert severities == {"high", "critical"}


async def test_query_security_logs_respects_limit(seeded, open_client):
    async with open_client() as client:
        result = await client.call_tool("query_security_logs", {"limit": 2})
    assert result.structured_content["count"] == 2


async def test_query_threat_intel_reports_a_hit(seeded, open_client):
    async with open_client() as client:
        result = await client.call_tool("query_threat_intel", {"indicator": "203.0.113.66"})
    assert result.structured_content["found"] is True
    assert result.structured_content["record"]["indicator_type"] == "ip"


async def test_query_threat_intel_reports_a_miss(seeded, open_client):
    async with open_client() as client:
        result = await client.call_tool("query_threat_intel", {"indicator": "198.51.100.9"})
    assert result.is_error is False
    assert result.structured_content["found"] is False


async def test_analyze_risk_returns_a_structured_assessment(seeded, open_client):
    async with open_client() as client:
        result = await client.call_tool("analyze_risk", {"indicator": "203.0.113.66"})
    assessment = result.structured_content["assessment"]
    assert assessment["risk_level"] in {"none", "low", "medium", "high", "critical"}
    assert assessment["evidence"]["threat_intel_found"] is True


# ---------------------------------------------------------------------------
# 失败分类
# ---------------------------------------------------------------------------


async def test_missing_data_source_is_data_unavailable(pinned_paths, open_client):
    """`pinned_paths` 默认不存在 ⇒ 数据源缺失。"""
    text = await _error_text("query_security_logs", {"limit": 1}, open_client)
    assert ReadOnlyToolErrorCode.DATA_UNAVAILABLE.value in text


async def test_corrupt_data_line_is_data_unavailable(pinned_paths, open_client):
    pinned_paths.logs.write_text("{not json}\n", encoding="utf-8")
    text = await _error_text("query_security_logs", {"limit": 1}, open_client)
    assert ReadOnlyToolErrorCode.DATA_UNAVAILABLE.value in text


async def test_empty_indicator_is_invalid_argument(seeded, open_client):
    text = await _error_text("query_threat_intel", {"indicator": "   "}, open_client)
    assert ReadOnlyToolErrorCode.INVALID_ARGUMENT.value in text


async def test_unparseable_time_is_invalid_argument(seeded, open_client):
    text = await _error_text("query_security_logs", {"start_time": "not-a-time"}, open_client)
    assert ReadOnlyToolErrorCode.INVALID_ARGUMENT.value in text


async def test_reversed_time_window_is_invalid_argument(seeded, open_client):
    text = await _error_text(
        "query_security_logs",
        {"start_time": "2026-09-11T00:00:00Z", "end_time": "2026-09-10T00:00:00Z"},
        open_client,
    )
    assert ReadOnlyToolErrorCode.INVALID_ARGUMENT.value in text


async def test_unclassified_core_failure_is_tool_failure(monkeypatch, seeded, open_client):
    def boom(**kwargs):
        raise RuntimeError("internal detail at C:/secret/location")

    monkeypatch.setattr(mcp_tools, "query_security_logs", boom)
    text = await _error_text("query_security_logs", {"limit": 1}, open_client)
    assert ReadOnlyToolErrorCode.TOOL_FAILURE.value in text
    assert "secret" not in text
    assert "RuntimeError" not in text


async def test_undumpable_result_is_internal_error(monkeypatch, seeded, open_client):
    class Undumpable:
        def model_dump(self, **kwargs):
            raise TypeError("cannot dump at C:/secret/location")

    monkeypatch.setattr(mcp_tools, "query_security_logs", lambda **kwargs: [Undumpable()])
    text = await _error_text("query_security_logs", {"limit": 1}, open_client)
    assert ReadOnlyToolErrorCode.INTERNAL_ERROR.value in text
    assert "secret" not in text


# ---------------------------------------------------------------------------
# 文案不泄漏
# ---------------------------------------------------------------------------


async def test_error_messages_never_echo_paths_or_tracebacks(pinned_paths, open_client):
    """固定文案之外不得携带任何来自底层异常的字节。"""
    pinned_paths.logs.write_text("{not json}\n", encoding="utf-8")
    text = await _error_text("query_security_logs", {"limit": 1}, open_client)
    assert str(pinned_paths.logs) not in text
    assert "events.jsonl" not in text
    assert "Traceback" not in text
    assert "scripts/seed_logs.py" not in text


async def test_threat_intel_error_does_not_echo_the_data_path(pinned_paths, open_client):
    text = await _error_text("query_threat_intel", {"indicator": "203.0.113.66"}, open_client)
    assert str(pinned_paths.intel) not in text
    assert "intel.jsonl" not in text


def test_every_error_code_has_a_fixed_message():
    """分类词表与文案表必须一一对应(新增分类却忘了写文案 ⇒ 这里红)。"""
    assert set(mcp_tools._SAFE_MESSAGES) == set(ReadOnlyToolErrorCode)
    for message in mcp_tools._SAFE_MESSAGES.values():
        assert message


async def test_hermetic_pinning_prevents_reading_the_repository_data(pinned_paths, open_client):
    """钉死路径后,调用**不可能**读到仓库 `data/`。

    这是 hermetic 判据的机械版本:数据源不存在 ⇒ DATA_UNAVAILABLE,
    而不是悄悄读到仓库里的真实数据后返回一个"成功"。
    """
    assert not pinned_paths.logs.exists()
    text = await _error_text("query_security_logs", {"limit": 1}, open_client)
    assert ReadOnlyToolErrorCode.DATA_UNAVAILABLE.value in text
    assert "count" not in text


async def test_intel_and_risk_also_fail_closed_without_pinned_data(pinned_paths, open_client):
    for tool_name, arguments in (
        ("query_threat_intel", {"indicator": "203.0.113.66"}),
        ("analyze_risk", {"indicator": "203.0.113.66"}),
    ):
        text = await _error_text(tool_name, arguments, open_client)
        assert ReadOnlyToolErrorCode.DATA_UNAVAILABLE.value in text


async def test_intel_hit_and_miss_share_one_code_path(seeded, intel_factory, jsonl_writer, open_client):
    """对照:同样的数据源,命中与未命中都是**成功**,不是错误。"""
    jsonl_writer(seeded.intel, [intel_factory(indicator="198.51.100.9", indicator_type="ip")])
    async with open_client() as client:
        hit = await client.call_tool("query_threat_intel", {"indicator": "198.51.100.9"})
        miss = await client.call_tool("query_threat_intel", {"indicator": "203.0.113.66"})
    assert hit.is_error is False and miss.is_error is False
    assert hit.structured_content["found"] is True
    assert miss.structured_content["found"] is False
