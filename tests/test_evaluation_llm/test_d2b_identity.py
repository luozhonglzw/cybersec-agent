"""Phase 9.2-D-2b —— 模型身份、端点类别词表、凭据守卫。

D-2a 的记录构造把身份**写死**成 scripted 三连:

    provider="scripted" / model="deterministic-scripted-<behavior>" / usage.source="scripted"

对离线脚本化运行这是正确的;但真实 provider 复用同一条构造路径时,它会
**静默地**给每一条真实记录贴上"脚本化"标签 —— 不是算错,而是"看起来一切正常"。

本文件同时守住三件事:

    身份由执行上下文传入        且离线身份与 D-2a **逐字段一致**(不回归)
    端点类别是**冻结词表**      不得用端点 URL 代替,不得就地扩写
    provider_http_attempts     不可观测时记 UNKNOWN,**不伪造 0**
"""
import ast
import asyncio
from hashlib import sha256
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.evaluation.llm.budget import (
    MAX_HTTP_ATTEMPTS_PER_INVOCATION,
    SDK_MAX_RETRIES_OBSERVED,
    UNKNOWN,
    BudgetGovernor,
    budget_from_plan,
    pilot_budget,
)
from app.evaluation.llm.executor import OfflineExecutor
from app.evaluation.llm.identity import (
    CREDENTIAL_FIELD_NAMES,
    ENDPOINT_CATEGORIES,
    REAL_PROVIDER_TEMPERATURE,
    SCRIPTED_PROVIDER,
    EndpointCategory,
    ModelIdentity,
    UnknownEndpointCategory,
    assert_no_credential_fields,
    base_url_host,
    base_url_host_sha256,
    parse_endpoint_category,
    provider_identity,
    scripted_identity,
)
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.pilot import pilot_plan
from app.evaluation.llm.protocol import (
    TO_BE_FROZEN_AFTER_PROVIDER_SELECTION,
    ExperimentManifest,
    ManifestError,
    seal_manifest,
    verify_manifest,
)
from app.evaluation.llm.raw import RawWriter
from app.evaluation.llm.runner import FORBIDDEN_METADATA_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_BASE_URL = "https://user:secret-token@fake-provider.example.invalid/v1"


@pytest.fixture(autouse=True)
def _no_egress():
    guard = NetworkEgressGuard(strict=False)
    with guard:
        yield
    assert guard.clean, f"D-2b 身份测试期间发生网络出口:{guard.egress_events}"


@pytest.fixture(scope="session")
def d2a_records(d2a_outcome):
    return RawWriter(
        d2a_outcome.raw_path, experiment_id=d2a_outcome.experiment_id
    ).read_all()


class _PlainFake:
    def __init__(self, *, task, dataset_paths=None, **_):
        self.task = task
        self.dataset_paths = dataset_paths or {}
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="(注入假模型)不下结论。")


def _fake_identity():
    return provider_identity(
        provider="fake-openai",
        model="fake-model-1",
        endpoint_category=EndpointCategory.OPENAI_COMPATIBLE,
        base_url=FAKE_BASE_URL,
    )


# ---------------------------------------------------------------------------
# 12. 离线脚本化身份**逐字段不变**
# ---------------------------------------------------------------------------


def test_12_scripted_offline_identity_is_unchanged():
    identity = scripted_identity("GOOD")
    assert identity.provider == SCRIPTED_PROVIDER == "scripted"
    assert identity.model == "deterministic-scripted-GOOD"
    assert identity.endpoint_category is EndpointCategory.SCRIPTED_OFFLINE
    assert identity.usage_source == "scripted"
    assert identity.is_scripted is True
    assert identity.temperature is None
    assert identity.base_url_host_sha256 is None
    assert identity.provider_reported_model_id is None
    assert identity.provider_default_parameters == {}
    identity.assert_consistent()


def test_12b_committed_d2a_records_still_carry_the_scripted_identity(d2a_records):
    """D-2a 的产物语义必须**逐字段**保持 —— 否则这是一次静默的行为回归。"""
    assert len(d2a_records) == 108
    for record in d2a_records:
        assert record.provider == "scripted"
        assert record.model == "deterministic-scripted-GOOD"
        assert record.endpoint_category == "SCRIPTED_OFFLINE"
        assert record.usage["source"] == "scripted"
        assert record.temperature is None
        assert record.provider_reported_model_id is None
        assert record.base_url_host_sha256 is None


def test_12c_a_default_executor_still_produces_scripted_records(tmp_path):
    """省略注入缝 ⇒ 适配器仍然构造 `ScriptedLLM`,离线行为不变。"""
    executor = OfflineExecutor(
        workdir=tmp_path / "default",
        experiment_id="default",
        plan=pilot_plan(baselines=("B0-shared",), repetition_count=1),
        baselines=("B0-shared",),
        behaviors=("GOOD",),
        repetition_count=1,
        guard=NetworkEgressGuard(strict=True),
    )
    outcome = asyncio.run(executor.run())
    records = RawWriter(Path(outcome.raw_path), experiment_id="default").read_all()
    assert records
    assert {r.provider for r in records} == {"scripted"}
    assert {r.endpoint_category for r in records} == {"SCRIPTED_OFFLINE"}


# ---------------------------------------------------------------------------
# 13. 注入的真实形态身份**不得**被标成 scripted
# ---------------------------------------------------------------------------


def test_13_an_injected_real_like_identity_is_not_labelled_scripted(tmp_path):
    executor = OfflineExecutor(
        workdir=tmp_path / "injected",
        experiment_id="injected",
        plan=pilot_plan(baselines=("B0-shared",), repetition_count=1),
        baselines=("B0-shared",),
        behaviors=("GOOD",),
        repetition_count=1,
        llm_factory=lambda **kwargs: _PlainFake(**kwargs),
        identity=_fake_identity(),
        guard=NetworkEgressGuard(strict=True),
    )
    outcome = asyncio.run(executor.run())
    records = RawWriter(Path(outcome.raw_path), experiment_id="injected").read_all()
    assert records

    for record in records:
        assert record.provider == "fake-openai"
        assert record.model == "fake-model-1"
        assert record.endpoint_category == "OPENAI_COMPATIBLE"
        assert record.usage["source"] == "provider_reported"
        assert record.temperature == REAL_PROVIDER_TEMPERATURE == 0.0
        assert record.base_url_host_sha256 == sha256(
            b"fake-provider.example.invalid"
        ).hexdigest()

    # 反同义反复:注入身份必须与离线身份**可区分**,否则断言是空的。
    assert records[0].provider != scripted_identity("GOOD").provider
    assert records[0].endpoint_category != "SCRIPTED_OFFLINE"


def test_13b_injecting_a_factory_with_a_scripted_identity_is_refused(tmp_path):
    """注入自定义工厂却仍声称 scripted ⇒ 直接拒绝。

    这正是 D-2b 要修的那条失真路径:真实模型跑出来的记录**静默地**
    声称自己是脚本化运行。
    """
    with pytest.raises(ValueError, match="SCRIPTED_OFFLINE"):
        OfflineExecutor(
            workdir=tmp_path / "refused",
            experiment_id="refused",
            plan=pilot_plan(baselines=("B0-shared",), repetition_count=1),
            baselines=("B0-shared",),
            behaviors=("GOOD",),
            repetition_count=1,
            llm_factory=lambda **kwargs: _PlainFake(**kwargs),
            identity=scripted_identity("GOOD"),
            guard=NetworkEgressGuard(strict=True),
        )


def test_13c_identity_rejects_contradictory_provider_and_category():
    with pytest.raises(AssertionError):
        provider_identity(
            provider=SCRIPTED_PROVIDER,
            model="m",
            endpoint_category=EndpointCategory.OPENAI_OFFICIAL,
        )
    with pytest.raises(AssertionError):
        provider_identity(
            provider="openai",
            model="m",
            endpoint_category=EndpointCategory.SCRIPTED_OFFLINE,
        )


# ---------------------------------------------------------------------------
# 17. endpoint_category 词表校验
# ---------------------------------------------------------------------------


def test_17_the_endpoint_category_vocabulary_is_frozen():
    assert tuple(item.value for item in EndpointCategory) == (
        "SCRIPTED_OFFLINE",
        "OPENAI_OFFICIAL",
        "OPENAI_COMPATIBLE",
    )
    assert ENDPOINT_CATEGORIES == (
        "SCRIPTED_OFFLINE",
        "OPENAI_OFFICIAL",
        "OPENAI_COMPATIBLE",
    )


def test_17b_parse_endpoint_category_rejects_unknown_values():
    assert parse_endpoint_category("OPENAI_COMPATIBLE") is EndpointCategory.OPENAI_COMPATIBLE
    for bad in ("openai", "OPENAI", "ANTHROPIC", "", "https://api.deepseek.com/v1"):
        with pytest.raises(UnknownEndpointCategory):
            parse_endpoint_category(bad)


def test_17c_the_manifest_validates_endpoint_category_but_still_allows_the_placeholder(
    d2a_manifest,
):
    """候选清单仍是候选 —— 占位符必须合法;一旦填了值就必须在词表内。"""
    payload = d2a_manifest.model_dump(mode="json")
    assert payload["endpoint_category"] == TO_BE_FROZEN_AFTER_PROVIDER_SELECTION

    # 占位符:合法(candidate 清单仍是 candidate)
    ExperimentManifest.model_validate(payload)

    for good in ENDPOINT_CATEGORIES:
        validated = ExperimentManifest.model_validate(
            {**payload, "endpoint_category": good}
        )
        assert validated.endpoint_category == good

    # 端点 URL 不得代替类别 —— URL 会随 region / 灰度 / 代理漂移,还可能内嵌凭据
    for bad in ("ANTHROPIC", "https://api.deepseek.com/v1", "openai", "  "):
        with pytest.raises(ValidationError):
            ExperimentManifest.model_validate({**payload, "endpoint_category": bad})


def test_17d_provider_and_model_reject_blank_but_allow_placeholders(d2a_manifest):
    payload = d2a_manifest.model_dump(mode="json")
    ExperimentManifest.model_validate(payload)
    for field in ("provider", "model"):
        with pytest.raises(ValidationError):
            ExperimentManifest.model_validate({**payload, field: "   "})
        with pytest.raises(ValidationError):
            ExperimentManifest.model_validate({**payload, field: ""})
        ExperimentManifest.model_validate({**payload, field: "openai"})


def test_17e_no_final_manifest_value_was_fabricated(d2a_manifest):
    """§7:本阶段**不得**冻结真实 provider 清单。

    即使把端点类别填成一个合法值,清单仍然因为 provider / model / token / cost
    的占位符而**无法冻结** —— 这就是"没有伪造最终值"的机械证据。
    """
    payload = d2a_manifest.model_dump(mode="json")
    payload["endpoint_category"] = "OPENAI_COMPATIBLE"
    payload["manifest_status"] = "frozen"
    candidate = seal_manifest(ExperimentManifest.model_validate(payload))

    with pytest.raises(ManifestError) as excinfo:
        verify_manifest(candidate, require_frozen=True)
    message = str(excinfo.value)
    assert "provider" in message
    assert "model" in message
    assert "token_budget" in message
    assert "cost_budget" in message


# ---------------------------------------------------------------------------
# 18. provider_http_attempts 不可观测时保持 UNKNOWN
# ---------------------------------------------------------------------------


def test_18_provider_http_attempts_remains_unknown_when_unobservable(d2a_records):
    """不可观测记 UNKNOWN —— **不记 0**,也不做推测性计数。"""
    assert d2a_records
    for record in d2a_records:
        assert record.provider_http_attempts == UNKNOWN
        assert record.provider_http_attempts != 0

    governor = BudgetGovernor(
        budget=budget_from_plan(pilot_plan(baselines=("B0-shared",), repetition_count=1))
    )
    governor.record_unit(provider_http_attempts=None)
    assert governor.counters.provider_http_attempts == UNKNOWN
    assert governor.experiment_provider_http_attempts == UNKNOWN

    # 即使**另一个**单元被直接观测到,总量仍是 UNKNOWN —— 不部分求和。
    governor.record_unit(provider_http_attempts=3)
    assert governor.counters.provider_http_attempts_observed == 3
    assert governor.counters.provider_http_attempts == UNKNOWN
    assert governor.experiment_provider_http_attempts == UNKNOWN


def test_18b_the_http_ceiling_is_declared_as_a_theoretical_bound():
    """`972 = 324 × 3` 只在"SDK 默认重试成立"的假设下成立,不得当成实测值。"""
    budget = pilot_budget(
        baseline_labels=("B0-shared", "B0-notool", "B2'", "B3"), repetition_count=3
    )
    assert budget.logical_invocation_hard_ceiling == 324
    assert budget.provider_http_attempt_ceiling == 972
    assert budget.provider_http_attempt_ceiling == (
        budget.logical_invocation_hard_ceiling * MAX_HTTP_ATTEMPTS_PER_INVOCATION
    )
    assert MAX_HTTP_ATTEMPTS_PER_INVOCATION == SDK_MAX_RETRIES_OBSERVED + 1
    assert "假设" in budget.provider_http_attempt_ceiling_basis
    assert "不是实测" in budget.provider_http_attempt_ceiling_basis


def test_18c_no_module_fabricates_an_http_attempt_count():
    """AST 级检查:没有任何评测模块把**推导出的**物理尝试数写进记录。

    只有 `identity` / `raw` 允许出现 `UNKNOWN`,而且必须来自观测缺失。
    """
    package = REPO_ROOT / "app" / "evaluation" / "llm"
    offenders: list[tuple[str, int]] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg != "provider_http_attempts":
                continue
            if node.value is None:
                continue
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                continue
            if isinstance(node.value, ast.Name) and node.value.id == "UNKNOWN":
                continue
            offenders.append((path.name, node.lineno))
    assert offenders == [], (
        "评测模块向记录写入了**非观测**的物理尝试数:"
        f"{offenders} —— 推测性计数会让'不可观测'变成'看起来观测到了'"
    )


# ---------------------------------------------------------------------------
# 凭据守卫
# ---------------------------------------------------------------------------


def test_credential_field_names_match_the_existing_guard():
    """与 `runner.FORBIDDEN_METADATA_FIELDS` **同源同义**(刻意重写而非 import)。"""
    assert CREDENTIAL_FIELD_NAMES == set(FORBIDDEN_METADATA_FIELDS)


def test_assert_no_credential_fields_has_teeth_and_does_not_false_positive():
    with pytest.raises(AssertionError):
        assert_no_credential_fields({"api_key": "x"})
    with pytest.raises(AssertionError):
        assert_no_credential_fields({"a": [{"b": {"token": "x"}}]})
    with pytest.raises(AssertionError):
        assert_no_credential_fields({"base_url": "https://x/y"})

    assert_no_credential_fields({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url_host_sha256": "abc",
        "provider_default_parameters": {"temperature": 0.0},
    })


def test_identity_rejects_credential_bearing_default_parameters():
    with pytest.raises(AssertionError):
        provider_identity(
            provider="openai",
            model="gpt-4o-mini",
            endpoint_category="OPENAI_OFFICIAL",
            provider_default_parameters={"api_key": "sk-should-never-be-here"},
        )


def test_base_url_host_sha256_never_leaks_the_full_url():
    identity = provider_identity(
        provider="deepseek",
        model="deepseek-chat",
        endpoint_category="OPENAI_COMPATIBLE",
        base_url="https://user:secret-token@api.deepseek.com/v1",
    )
    assert identity.base_url_host_sha256 == sha256(b"api.deepseek.com").hexdigest()
    dumped = identity.model_dump_json()
    assert "secret-token" not in dumped
    assert "user:" not in dumped
    assert "api.deepseek.com" not in dumped
    assert "/v1" not in dumped


def test_base_url_helpers_handle_missing_and_malformed_values():
    assert base_url_host(None) is None
    assert base_url_host("") is None
    assert base_url_host_sha256(None) is None
    assert base_url_host("https://api.deepseek.com/v1") == "api.deepseek.com"


def test_model_identity_is_frozen():
    identity = scripted_identity("GOOD")
    with pytest.raises(ValidationError):
        identity.provider = "openai"


def test_model_identity_is_a_pydantic_model_without_credential_fields():
    fields = set(ModelIdentity.model_fields)
    assert fields & CREDENTIAL_FIELD_NAMES == set()
    assert "provider_http_attempts" not in fields, (
        "物理尝试数不可观测,不得出现在身份里 —— 那会诱使有人填一个猜测值"
    )
