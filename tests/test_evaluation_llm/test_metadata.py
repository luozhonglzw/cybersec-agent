"""可复现性元数据与凭据泄露护栏。

D-2 会接真实 provider。届时最容易发生的事是"为了方便排查,顺手把 base_url
和请求头记下来" —— 而 base_url 完全可能内嵌凭据。本文件把这条边界做成
**机械守卫**,而不是靠 code review 的自觉。
"""
import asyncio
import hashlib
from datetime import datetime, timezone

import pytest

from app.evaluation.llm import (
    FORBIDDEN_METADATA_FIELDS,
    LLM_TASKS,
    assert_no_secrets,
    build_metadata,
    run_llm_evaluation,
)
from app.evaluation.llm.adapters import SHARED_SYSTEM_PROMPT
from app.evaluation.runner import golden_digest
from app.evaluation.golden import GOLDEN_SET

#: Phase 9.2-A 冻结的 golden 摘要 —— 跨阶段回归锚点,本阶段不得改变它。
FROZEN_GOLDEN_DIGEST = (
    "5f157ed92df12cb1f1d3f175327c39c01e49062e6084036fb9de641dc50743ca"
)

REQUIRED_METADATA_FIELDS = (
    "provider",
    "model",
    "base_url_host",
    "temperature",
    "system_prompt_sha256",
    "tool_schema_sha256",
    "task_prompt_sha256",
    "run_timestamp_utc",
    "repetition_id",
)


# ---------------------------------------------------------------------------
# 1. 凭据绝不入档
# ---------------------------------------------------------------------------


def test_forbidden_field_set_covers_the_obvious_leaks():
    for field in ("api_key", "authorization", "token", "secret", "password"):
        assert field in FORBIDDEN_METADATA_FIELDS
    assert "base_url" in FORBIDDEN_METADATA_FIELDS, (
        "完整 base_url 可能内嵌凭据 —— 只允许记录主机名"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"provider": "x", "api_key": "sk-xxx"},
        {"provider": "x", "authorization": "Bearer xxx"},
        {"provider": "x", "base_url": "https://u:p@host/v1"},
        {"provider": "x", "LLM_API_KEY": "sk-xxx"},
        {"provider": "x", "Token": "abc"},
    ],
)
def test_assert_no_secrets_rejects_credential_fields(payload):
    with pytest.raises(AssertionError):
        assert_no_secrets(payload)


def test_assert_no_secrets_accepts_a_clean_payload():
    assert_no_secrets({"provider": "scripted", "model": "m", "base_url_host": "host"})


def test_metadata_contains_no_credential_fields(matrix):
    dumped = matrix.metadata.model_dump()
    assert_no_secrets(dumped)
    assert FORBIDDEN_METADATA_FIELDS & {key.lower() for key in dumped} == set()


def test_metadata_never_records_a_full_url(matrix):
    host = matrix.metadata.base_url_host
    assert host is None or "://" not in host


# ---------------------------------------------------------------------------
# 2. schema 完整性
# ---------------------------------------------------------------------------


def test_metadata_has_every_required_field(matrix):
    dumped = matrix.metadata.model_dump()
    for field in REQUIRED_METADATA_FIELDS:
        assert field in dumped, f"元数据缺字段 {field}"


def test_metadata_records_scripted_provider_and_scripted_model(matrix):
    assert matrix.metadata.provider == "scripted"
    assert matrix.metadata.model.startswith("deterministic-scripted-")


def test_single_behavior_run_encodes_the_behavior_in_the_model_name(fresh_workdir):
    result = asyncio.run(run_llm_evaluation(
        workdir=fresh_workdir,
        baselines=("B3",),
        behaviors=("GOOD",),
    ))
    assert result.metadata.model == "deterministic-scripted-GOOD"


def test_metadata_records_provider_defaults_instead_of_inventing_values(matrix):
    """当前 LLMClient 未设置 temperature 等参数 —— 记 `provider_default`,不猜。"""
    for field in ("temperature", "max_tokens", "timeout", "max_retries"):
        assert getattr(matrix.metadata, field) == "provider_default"


def test_metadata_timestamps_are_utc_iso8601(matrix):
    parsed = datetime.fromisoformat(matrix.metadata.run_timestamp_utc)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_metadata_repetition_id_is_zero_for_d1(matrix):
    assert matrix.metadata.repetition_id == 0


# ---------------------------------------------------------------------------
# 3. 摘要与锚点
# ---------------------------------------------------------------------------


def test_system_prompt_digest_matches_the_actual_prompt(matrix):
    expected = hashlib.sha256(SHARED_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert matrix.metadata.system_prompt_sha256 == expected


def test_task_prompt_digests_cover_every_task(matrix):
    assert set(matrix.metadata.task_prompt_sha256) == {task.task_id for task in LLM_TASKS}
    for task in LLM_TASKS:
        expected = hashlib.sha256(task.user_prompt.encode("utf-8")).hexdigest()
        assert matrix.metadata.task_prompt_sha256[task.task_id] == expected


def test_phase_92a_golden_digest_is_unchanged(matrix):
    """D-1 不得触碰 9.2-A 的冻结用例。"""
    assert matrix.metadata.golden_digest == FROZEN_GOLDEN_DIGEST
    assert golden_digest(GOLDEN_SET) == FROZEN_GOLDEN_DIGEST


def test_tool_schema_digest_is_stable_and_recorded(matrix):
    first = build_metadata(LLM_TASKS, behavior="GOOD")
    second = build_metadata(LLM_TASKS, behavior="GOOD")
    assert first.tool_schema_sha256 == second.tool_schema_sha256
    assert matrix.metadata.tool_schema_sha256 == first.tool_schema_sha256


# ---------------------------------------------------------------------------
# 4. token 字段:NOT_AVAILABLE 而不是 0
# ---------------------------------------------------------------------------


def test_token_fields_are_none_never_zero(matrix):
    """脚本化运行不产生 usage —— 记 None。0 会被读成"零消耗"。"""
    for obs in matrix.observations:
        assert obs.input_tokens is None
        assert obs.output_tokens is None
        assert obs.total_tokens is None


def test_no_cost_metric_is_fabricated(matrix):
    ids = {metric.metric_id for metric in matrix.metrics}
    assert not any("cost" in metric_id or "price" in metric_id for metric_id in ids)
