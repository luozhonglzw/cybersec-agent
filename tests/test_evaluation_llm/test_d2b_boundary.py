"""Phase 9.2-D-2b —— 注入缝、离线 import 护栏、零网络出口。

三条边界
--------
11. **注入缝可用**:假模型能通过 `LLMFactory` 进入适配器并跑完整条流水线。
19. **离线 import 护栏完好**:`app/evaluation/llm/*.py` 里**没有**任何
    provider 客户端 import;真实 provider 的构造留在包外,且包内不引用它。
20. **零网络出口**:整条 D-2b 流程在严格出口守卫下运行;真实 provider 路径
    默认**关闭**;测试套件自身不得打开它。
"""
import ast
import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.evaluation.llm.adapters import BASELINES, ScriptedLLM
from app.evaluation.llm.budgeted_llm import BudgetedLLM
from app.evaluation.llm.dataset import LLM_TASKS
from app.evaluation.llm.executor import OfflineExecutor
from app.evaluation.llm.identity import EndpointCategory, provider_identity
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.pilot import pilot_plan
from app.evaluation.llm.raw import RawWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_PKG = REPO_ROOT / "app" / "evaluation" / "llm"
REAL_PROVIDER_MODULE = REPO_ROOT / "app" / "evaluation" / "real_provider.py"

FORBIDDEN_CLIENT_MODULES = (
    "openai",
    "langchain_openai",
    "anthropic",
    "httpx",
    "requests",
    "aiohttp",
)

FAKE_BASE_URL = "https://fake-provider.example.invalid/v1"


@pytest.fixture(autouse=True)
def _no_egress():
    """本模块每个测试全程受出口守卫监视,结束时断言**零出口**。"""
    guard = NetworkEgressGuard(strict=False)
    with guard:
        yield
    assert guard.clean, f"D-2b 边界测试期间发生网络出口:{guard.egress_events}"


class FakeInjectedLLM:
    """注入缝用的假模型:确定性、零网络、可观测调用次数。"""

    def __init__(self, *, behavior, task, dataset_paths=None, decoy_paths=None, **_):
        self.behavior = behavior
        self.task = task
        self.dataset_paths = dataset_paths or {}
        self.decoy_paths = decoy_paths or {}
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=f"(注入假模型)行为 {self.behavior};不下任何结论。")


def _fake_identity():
    return provider_identity(
        provider="fake-openai",
        model="fake-model-1",
        endpoint_category=EndpointCategory.OPENAI_COMPATIBLE,
        base_url=FAKE_BASE_URL,
    )


# ---------------------------------------------------------------------------
# 11. 假模型能通过注入缝执行
# ---------------------------------------------------------------------------


def test_11_a_fake_injected_llm_can_execute_through_the_adapter(tmp_path):
    """注入缝必须**真的**接通:从工厂构造 → 适配器 → 图 → 观测 → 落盘。

    这条同时是 D-2b 的"真实 provider 就绪"证据:真实构造只是这条缝上的
    另一个实现,评测流水线本身不需要任何改动。
    """
    created: list[FakeInjectedLLM] = []

    def factory(**kwargs):
        fake = FakeInjectedLLM(**kwargs)
        created.append(fake)
        return fake

    baselines = ("B0-shared", "B2'")
    executor = OfflineExecutor(
        workdir=tmp_path / "inject",
        experiment_id="inject",
        plan=pilot_plan(baselines=baselines, repetition_count=1),
        baselines=baselines,
        behaviors=("GOOD",),
        repetition_count=1,
        llm_factory=factory,
        identity=_fake_identity(),
        guard=NetworkEgressGuard(strict=True),
    )
    outcome = asyncio.run(executor.run())

    assert created, "注入的工厂从未被调用 —— 注入缝没有接通"
    assert sum(fake.calls for fake in created) > 0
    assert outcome.record_count == 18

    records = RawWriter(Path(outcome.raw_path), experiment_id="inject").read_all()
    assert len(records) == 18
    assert all(record.provider == "fake-openai" for record in records)
    assert all(record.logical_llm_invocations >= 1 for record in records)
    assert all(record.final_narrative.startswith("(注入假模型)") for record in records)
    assert outcome.network["clean"] is True


def test_11b_the_adapter_always_returns_a_budgeted_proxy():
    """无论是否注入,适配器返回的**始终**是预算代理。"""
    inner_fake = FakeInjectedLLM(behavior="GOOD", task=LLM_TASKS[0])
    injected = BASELINES["B0"](dataset_paths={}, llm_factory=lambda **_: inner_fake)
    proxy = injected._budgeted_llm(
        behavior="GOOD", task=LLM_TASKS[0],
        dataset_paths=None, decoy_paths=None, emit_usage=False,
    )
    assert isinstance(proxy, BudgetedLLM)
    assert proxy.inner is inner_fake

    offline = BASELINES["B0"](dataset_paths={})
    default_proxy = offline._budgeted_llm(
        behavior="GOOD", task=LLM_TASKS[0],
        dataset_paths=None, decoy_paths=None, emit_usage=False,
    )
    assert isinstance(default_proxy, BudgetedLLM)
    assert isinstance(default_proxy.inner, ScriptedLLM)


def test_11c_the_factory_receives_the_evaluation_context():
    """工厂必须收到 `behavior` / `task` / `dataset_paths` 等评测上下文。

    少了它们,注入的模型无法把工具参数指向**评测授权**的数据路径 ——
    于是"LLM 看到的证据"与"oracle 重算的证据"变成两个世界。
    """
    captured: dict = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeInjectedLLM(**kwargs)

    adapter = BASELINES["B0"](dataset_paths={}, llm_factory=factory)
    adapter._budgeted_llm(
        behavior="GOOD", task=LLM_TASKS[0],
        dataset_paths={"logs": "/tmp/logs.jsonl", "intel": "/tmp/intel.jsonl"},
        decoy_paths={"logs": "/tmp/decoy.jsonl"},
        emit_usage=True,
    )
    assert captured["behavior"] == "GOOD"
    assert captured["task"] is LLM_TASKS[0]
    assert captured["dataset_paths"]["logs"].endswith("logs.jsonl")
    assert captured["decoy_paths"]["logs"].endswith("decoy.jsonl")
    assert captured["emit_usage"] is True


# ---------------------------------------------------------------------------
# 19. 离线 import 护栏完好
# ---------------------------------------------------------------------------


def _module_level_imports(path: Path) -> set[str]:
    """只收集**模块顶层**的 import(函数体内的惰性 import 不算)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _all_imports(path: Path) -> set[str]:
    """收集**全部** import,包括函数体内的(藏起来也要查出来)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_19_the_offline_provider_import_guard_remains_intact():
    """`app/evaluation/llm/*.py` 里**不得**出现任何 provider 客户端。"""
    package_files = sorted(LLM_PKG.glob("*.py"))
    assert package_files, "评测包为空 —— 护栏什么都没检查"
    for path in package_files:
        source = path.read_text(encoding="utf-8")
        for name in FORBIDDEN_CLIENT_MODULES:
            assert f"import {name}" not in source, (path.name, name)
            assert f"from {name}" not in source, (path.name, name)


def test_19b_the_guard_covers_the_d2b_modules_too():
    """新模块(`identity.py` / `budgeted_llm.py`)同样在护栏覆盖范围内。"""
    for name in ("identity.py", "budgeted_llm.py"):
        path = LLM_PKG / name
        assert path.exists(), name
        imported = _all_imports(path)
        leaked = sorted(
            module for module in imported
            if module.split(".")[0] in FORBIDDEN_CLIENT_MODULES
        )
        assert leaked == [], (name, leaked)


def test_19c_the_detector_has_teeth():
    """正/负对照 —— 没有这组,上面的断言可能因为检测器失效而永远通过。"""
    positive = "from langchain_openai import ChatOpenAI\n"
    assert any(f"from {name}" in positive for name in FORBIDDEN_CLIENT_MODULES)

    negative = "from app.evaluation.llm.identity import ModelIdentity\n"
    assert not any(f"from {name}" in negative for name in FORBIDDEN_CLIENT_MODULES)


def test_19d_the_provider_boundary_lives_outside_the_guarded_package():
    assert REAL_PROVIDER_MODULE.exists()
    assert not (LLM_PKG / "real_provider.py").exists(), (
        "真实 provider 的构造**不得**落进被护栏覆盖的包内"
    )


def test_19e_the_guarded_package_does_not_import_the_provider_boundary():
    """包内模块不得引用包外的 provider 构造 —— 那等于绕过护栏。"""
    offenders: list[str] = []
    for path in sorted(LLM_PKG.glob("*.py")):
        for module in _all_imports(path):
            if module.startswith("app.evaluation.real_provider"):
                offenders.append(path.name)
    assert offenders == [], offenders


def test_19f_the_provider_boundary_imports_no_client_at_module_level():
    """`real_provider.py` 的 provider SDK import 必须**在函数体内**。

    导入期加载 SDK 会让"只想读一下候选清单"这件事也拉起一整套 provider 栈;
    更糟的是,它会让"离线"与"在线"在**导入期**就不再可区分。
    """
    module_level = _module_level_imports(REAL_PROVIDER_MODULE)
    leaked = sorted(
        module for module in module_level
        if module.split(".")[0] in FORBIDDEN_CLIENT_MODULES
    )
    assert leaked == [], leaked
    # 但它**确实**在函数体内 import 了客户端 —— 否则它什么也构造不出来。
    assert "langchain_openai" in _all_imports(REAL_PROVIDER_MODULE)


def test_19g_importing_the_provider_boundary_constructs_nothing():
    """导入 `real_provider` 不得构造任何客户端、不得读取任何环境变量。"""
    from app.evaluation import real_provider

    assert real_provider.CANDIDATES, "候选清单不得为空"
    for candidate in real_provider.CANDIDATES:
        assert candidate.api_key_env
        assert candidate.endpoint_category in tuple(EndpointCategory)
        # 声明里只记"读哪个环境变量",**不持有**其值
        fields = set(candidate.__dataclass_fields__)
        assert "api_key" not in fields
        assert "base_url" not in fields
        assert {"api_key_env", "base_url_env", "default_base_url"} <= fields


# ---------------------------------------------------------------------------
# 20. 零网络出口
# ---------------------------------------------------------------------------


def test_20_the_d2b_flow_produces_zero_network_egress(tmp_path):
    """注入缝下的完整流程:**零出口**。

    守卫是 `strict=True`:一旦出现对外访问,在**发生处**就炸掉。
    """
    baselines = ("B0-shared", "B2'")
    guard = NetworkEgressGuard(strict=True)
    executor = OfflineExecutor(
        workdir=tmp_path / "egress",
        experiment_id="egress",
        plan=pilot_plan(baselines=baselines, repetition_count=1),
        baselines=baselines,
        behaviors=("GOOD",),
        repetition_count=1,
        llm_factory=lambda **kwargs: FakeInjectedLLM(**kwargs),
        identity=_fake_identity(),
        guard=guard,
    )
    outcome = asyncio.run(executor.run())

    assert outcome.network["clean"] is True
    assert outcome.network["egress_events"] == 0
    assert guard.egress_events == []


def test_20b_the_real_provider_path_is_closed_by_default():
    """真实构造默认**拒绝** —— 一个"不小心就发请求"的构造器迟早会被真的调用。"""
    from app.evaluation.real_provider import (
        CANDIDATES,
        MissingCredentialError,
        RealProviderDisabled,
        build_chat_model,
        make_llm_factory,
    )

    candidate = CANDIDATES[0]

    # 有凭据但没显式开网络 ⇒ 拒绝
    with pytest.raises(RealProviderDisabled):
        build_chat_model(candidate=candidate, api_key="not-a-real-key")
    with pytest.raises(RealProviderDisabled):
        make_llm_factory(candidate=candidate, api_key="not-a-real-key")(
            behavior="GOOD"
        )

    # 没凭据 ⇒ 拒绝,**且不回退到 ScriptedLLM**
    with pytest.raises(MissingCredentialError):
        build_chat_model(candidate=candidate, api_key="")


def test_20c_the_d2b_test_suite_never_enables_the_network_path():
    """测试套件自身**不得**打开真实 provider 路径 —— 否则"零出口"只是运气。

    关键字**拼接构造**,避免本文件自己命中这条检查。
    """
    needle = "allow_network" + "=True"
    offenders: list[str] = []
    for path in sorted(Path(__file__).parent.glob("test_d2b_*.py")):
        source = path.read_text(encoding="utf-8")
        if needle in source:
            offenders.append(path.name)
    assert offenders == [], offenders


def test_20d_the_candidate_list_is_declarative_and_credential_free():
    """候选是**声明**,不是胜者;而且不含任何凭据。"""
    from app.evaluation.real_provider import CANDIDATE_A, CANDIDATE_B, describe_candidates

    described = describe_candidates()
    assert [item["candidate_id"] for item in described] == [CANDIDATE_A, CANDIDATE_B]
    assert {item["endpoint_category"] for item in described} == {
        "OPENAI_OFFICIAL",
        "OPENAI_COMPATIBLE",
    }
    for item in described:
        assert set(item) == {
            "candidate_id",
            "provider",
            "model",
            "endpoint_category",
            "credential_env_names",
            "notes",
        }
        assert "api_key" not in item
        assert "base_url" not in item
