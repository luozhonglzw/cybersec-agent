"""Phase 9.2-D-2d / F12 —— 正式驱动的 **fail-closed 冻结闸门** 与**调用顺序**。

顺序本身就是安全性质
--------------------
`OfflineExecutor` 只收 `manifest_digest: str`,**看不到清单对象** ——
它**不可能**成为"先于 provider 构造"的那道闸门。因此闸门只能放在**驱动入口**,
而且必须是**第一件事**。

F12:为什么"先于 provider 构造"不够
-----------------------------------
旧断言只证明"闸门早于 `make_llm_factory` / `OfflineExecutor`"(见 `test_02`
里的旧口径对照)。但驱动当时还会自己 `load_datasets()` / `build_d2d_manifest()`
/ `d2d_plan()` 取默认值 —— 于是闸门之前仍有实质动作,而且**闸门要核对的那三样
东西正是驱动自己挑的**。"闸门先于执行"这句话因此在语义上是空的。

本文件现在断言的是**更强**的不变量:

    ① `run_d2d_pilot()` 的**第一条实质语句**就是 `assert_frozen_pilot_manifest(...)`
       —— 它之前**没有任何调用**(`test_01`);
    ② 驱动里**不存在**任何准备动作(`load_datasets` / `build_d2d_manifest` /
       `d2d_plan`)—— 不是"排在后面",而是**根本不出现**(`test_02b`);
    ③ 清单上任何一项都不得先于闸门出现(`test_02c`);
    ④ `datasets` / `manifest` / `plan` 是**必填**的仅关键字参数,无隐式默认
       (`test_02d`);
    ⑤ 闸门拿到的就是**调用方交来的**那三个对象(`test_01b`);
    ⑥ 所有构造动作(`default_pilot_config` / `make_llm_factory` / `build_chat_model`
       / `OfflineExecutor` / `asyncio.run`)都在闸门**之后**(`test_04b`)。

全部用 **AST** 证明(不靠阅读代码),并配正/负对照证明检测器有牙。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.evaluation import d2d_pilot
from app.evaluation.d2d_pilot import (
    D2DPilotDisabled,
    build_d2d_manifest,
    run_d2d_pilot,
)
from app.evaluation.llm.pilot import load_datasets
from app.evaluation.pilot_config import d2d_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_PKG = REPO_ROOT / "app" / "evaluation" / "llm"
DRIVER = REPO_ROOT / "app" / "evaluation" / "d2d_pilot.py"

GIT_COMMIT = "dfe7ea4b677c74b6e60ed40f20fdc42b4988735b"
CREATED_AT = "2026-09-29T00:00:00+00:00"

GATE = "assert_frozen_pilot_manifest"

#: **准备**动作 —— 属于 pre-execution preparation,**不得**出现在正式驱动里。
PREPARATION_CALLS = (
    "load_datasets",
    "build_d2d_manifest",
    "d2d_plan",
)

#: 闸门必须先于这些调用。
MUST_FOLLOW_THE_GATE = (
    "default_pilot_config",
    "make_llm_factory",
    "build_chat_model",
    "OfflineExecutor",
    "asyncio.run",
)

#: 控制器要求 2 的**完整**清单 —— 这些调用**都不得**先于闸门。
FORBIDDEN_BEFORE_THE_GATE = PREPARATION_CALLS + MUST_FOLLOW_THE_GATE

#: 驱动**必须**真的构造这两样 —— 否则"顺序"就无从谈起。
REQUIRED_IN_DRIVER = ("make_llm_factory", "OfflineExecutor")

#: 正式驱动必须接收的**已准备好**的输入。
FORMAL_DRIVER_REQUIRED_INPUTS = ("datasets", "manifest", "plan")


# ---------------------------------------------------------------------------
# AST 工装
# ---------------------------------------------------------------------------


def _function_node(source: str, function_name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", None)


def _call_positions(source: str, function_name: str) -> dict[str, int]:
    """函数体内每个被调用名字的**最早**行号(含所有名字,不只是白名单)。"""
    found: dict[str, int] = {}
    for node in ast.walk(_function_node(source, function_name)):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name is None:
            continue
        found[name] = min(node.lineno, found.get(name, node.lineno))
    return found


def _referenced_names(source: str, function_name: str) -> set[str]:
    """函数体内出现的所有裸名字 / 属性名(**含**非调用位置)。"""
    names: set[str] = set()
    for node in ast.walk(_function_node(source, function_name)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _first_substantive_statement(source: str, function_name: str) -> ast.stmt:
    """跳过 docstring 后的**第一条语句**。"""
    body = list(_function_node(source, function_name).body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    assert body, f"{function_name}() 的函数体是空的"
    return body[0]


def _is_call_to(statement: ast.stmt, name: str) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _called_name(statement.value) == name
    )


# ---------------------------------------------------------------------------
# ① 闸门是第一条实质语句
# ---------------------------------------------------------------------------


def test_01_the_gate_is_the_first_statement_of_the_formal_driver():
    """`run_d2d_pilot()` 的**第一条实质语句**必须是冻结闸门。"""
    source = DRIVER.read_text(encoding="utf-8")
    first = _first_substantive_statement(source, "run_d2d_pilot")
    assert _is_call_to(first, GATE), (
        f"run_d2d_pilot() 的第一条实质语句必须是 {GATE}(...),"
        f"实际是 line {first.lineno}: {ast.unparse(first)!r}"
    )

    positions = _call_positions(source, "run_d2d_pilot")
    gate = positions[GATE]
    earlier = {name: line for name, line in positions.items() if line < gate}
    assert earlier == {}, f"闸门之前不得存在任何调用,实际有:{earlier}"


def test_01b_the_gate_receives_exactly_the_supplied_inputs():
    """闸门必须拿**调用方交来的** manifest / datasets / plan,不得自行挑选。"""
    source = DRIVER.read_text(encoding="utf-8")
    statement = _first_substantive_statement(source, "run_d2d_pilot")
    assert _is_call_to(statement, GATE)
    call: ast.Call = statement.value  # type: ignore[assignment]

    positional = [ast.unparse(arg) for arg in call.args]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}

    assert positional == ["manifest"], positional
    assert keywords["datasets"] == "datasets", keywords
    assert keywords["plan"] == "plan", keywords
    assert keywords["expected_git_commit"] == "git_commit", keywords


# ---------------------------------------------------------------------------
# ② 驱动不做准备 / 清单上任何一项都不得先于闸门 / 三个输入必填
# ---------------------------------------------------------------------------


def test_02b_the_formal_driver_performs_no_preparation():
    """准备动作**根本不得出现**在正式驱动里 —— 不是"排在后面"就行。"""
    source = DRIVER.read_text(encoding="utf-8")
    positions = _call_positions(source, "run_d2d_pilot")
    called = {name: positions[name] for name in PREPARATION_CALLS if name in positions}
    assert called == {}, (
        "正式驱动不得做任何准备动作(它们必须在执行之前由调用方完成):"
        f"{called}"
    )
    offenders = sorted(set(PREPARATION_CALLS) & _referenced_names(source, "run_d2d_pilot"))
    assert offenders == [], f"正式驱动不得引用准备工具:{offenders}"


def test_02c_no_forbidden_call_precedes_the_gate():
    """控制器要求 2 的字面口径:清单上任何一项都不得排在闸门之前。"""
    positions = _call_positions(DRIVER.read_text(encoding="utf-8"), "run_d2d_pilot")
    gate = positions[GATE]
    preceding = {
        name: positions[name]
        for name in FORBIDDEN_BEFORE_THE_GATE
        if name in positions and positions[name] < gate
    }
    assert preceding == {}, f"这些调用先于闸门出现:{preceding}"


def test_02d_datasets_manifest_and_plan_are_required_formal_inputs():
    """无隐式默认 —— 否则"闸门是第一件事"可以被默认值绕开。"""
    signature = inspect.signature(run_d2d_pilot)
    for name in FORMAL_DRIVER_REQUIRED_INPUTS:
        parameter = signature.parameters.get(name)
        assert parameter is not None, f"run_d2d_pilot() 必须接收 {name}"
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} 必须是仅关键字参数(避免位置参数错配)"
        )
        assert parameter.default is inspect.Parameter.empty, (
            f"{name} 必须**必填** —— 隐式默认值会让'闸门是第一件事'被绕开"
        )


# ---------------------------------------------------------------------------
# ④ 执行侧构造全部在闸门之后
# ---------------------------------------------------------------------------


def test_04b_every_construction_call_follows_the_gate():
    source = DRIVER.read_text(encoding="utf-8")
    gate = _first_substantive_statement(source, "run_d2d_pilot").lineno
    positions = _call_positions(source, "run_d2d_pilot")

    for required in REQUIRED_IN_DRIVER:
        assert required in positions, f"驱动入口应当构造 {required}"
    for name in MUST_FOLLOW_THE_GATE:
        if name not in positions:
            continue
        assert gate < positions[name], (
            f"冻结闸门(line {gate})必须**先于** {name}(line {positions[name]})"
        )


def test_04c_the_driver_never_calls_the_model_constructor_directly():
    """真实 provider 只能经 `make_llm_factory` 进入。

    直接调 `build_chat_model` 会绕开工厂 —— 于是"闸门在工厂之前"就不再
    蕴含"闸门在 provider 构造之前"。
    """
    positions = _call_positions(DRIVER.read_text(encoding="utf-8"), "run_d2d_pilot")
    assert "build_chat_model" not in positions


# ---------------------------------------------------------------------------
# ⑤ 检测器有牙:正/负对照
# ---------------------------------------------------------------------------

#: 负对照 A —— 准备动作排在闸门**之前**(F12 缺陷的忠实形状)。
_NEGATIVE_CONTROL_SOURCE = (
    "def run_d2d_pilot(*, datasets, manifest, plan, git_commit, api_key,\n"
    "                   experiment_id, workdir, allow_network=False):\n"
    "    datasets = datasets or load_datasets(workdir)\n"
    "    assert_frozen_pilot_manifest(manifest, datasets=datasets, plan=plan)\n"
    "    return None\n"
)

#: 正对照 —— 正确形状。
_POSITIVE_CONTROL_SOURCE = (
    "def run_d2d_pilot(*, datasets, manifest, plan, git_commit, api_key,\n"
    "                   experiment_id, workdir, allow_network=False):\n"
    "    assert_frozen_pilot_manifest(manifest, datasets=datasets, plan=plan)\n"
    "    factory = make_llm_factory(x=1)\n"
    "    return None\n"
)

#: 负对照 B —— 闸门被喂"现场计算值"而不是交来的对象。
_INDIRECT_INPUT_SOURCE = (
    "def run_d2d_pilot(*, datasets, manifest, plan, git_commit, api_key,\n"
    "                   experiment_id, workdir, allow_network=False):\n"
    "    assert_frozen_pilot_manifest(\n"
    "        manifest,\n"
    "        datasets=load_datasets(workdir),\n"
    "        plan=d2d_plan(),\n"
    "    )\n"
    "    return None\n"
)


def test_02_the_order_detector_has_teeth():
    """正/负对照:顺序反了必须被检出。"""
    bad = (
        "def run_d2d_pilot():\n"
        "    factory = make_llm_factory(x=1)\n"
        "    assert_frozen_pilot_manifest(m)\n"
    )
    positions = _call_positions(bad, "run_d2d_pilot")
    assert positions["make_llm_factory"] < positions["assert_frozen_pilot_manifest"]

    good = (
        "def run_d2d_pilot():\n"
        "    assert_frozen_pilot_manifest(m)\n"
        "    factory = make_llm_factory(x=1)\n"
    )
    positions = _call_positions(good, "run_d2d_pilot")
    assert positions["assert_frozen_pilot_manifest"] < positions["make_llm_factory"]


def test_02e_the_preparation_before_the_gate_detector_has_teeth():
    """**负对照(F12 专属)**:`load_datasets()` 排在闸门之前必须被检出。

    旧口径(只查执行侧构造)对这条**恒真** —— 这正是 F12。
    """
    first = _first_substantive_statement(_NEGATIVE_CONTROL_SOURCE, "run_d2d_pilot")
    assert not _is_call_to(first, GATE), "负对照的第一条语句就不该是闸门"

    positions = _call_positions(_NEGATIVE_CONTROL_SOURCE, "run_d2d_pilot")
    assert positions["load_datasets"] < positions[GATE]

    # test_02b 口径:准备动作出现在驱动里 ⇒ 检出。
    assert sorted(set(PREPARATION_CALLS) & set(positions)) == ["load_datasets"]
    # test_02c 口径:清单上有一项先于闸门 ⇒ 检出。
    preceding = {
        name: positions[name]
        for name in FORBIDDEN_BEFORE_THE_GATE
        if name in positions and positions[name] < positions[GATE]
    }
    assert set(preceding) == {"load_datasets"}, preceding
    # 旧口径对照:只查执行侧构造 ⇒ 什么都看不见。
    legacy = {
        name: positions[name]
        for name in MUST_FOLLOW_THE_GATE
        if name in positions and positions[name] < positions[GATE]
    }
    assert legacy == {}, "旧口径对 F12 恒真 —— 这就是它不够用的原因"


def test_02f_the_preparation_before_the_gate_detector_accepts_the_correct_shape():
    """**正对照**:正确形状必须通过全部结构断言。"""
    first = _first_substantive_statement(_POSITIVE_CONTROL_SOURCE, "run_d2d_pilot")
    assert _is_call_to(first, GATE)

    positions = _call_positions(_POSITIVE_CONTROL_SOURCE, "run_d2d_pilot")
    assert sorted(set(PREPARATION_CALLS) & set(positions)) == []
    assert positions[GATE] < positions["make_llm_factory"]


def test_02g_the_gate_input_detector_has_teeth():
    """**负对照(闸门输入专属)**:闸门被喂现场计算值必须被检出。

    这一条**不是** F12 的检出器 —— 旧实现确实传了 `datasets=datasets,
    plan=plan`,只是那两个名字在闸门之前被重绑了。它防的是另一类回归:
    把 `load_datasets(workdir)` / `d2d_plan()` 直接内联进闸门语句。
    """
    statement = _first_substantive_statement(_INDIRECT_INPUT_SOURCE, "run_d2d_pilot")
    assert _is_call_to(statement, GATE)
    call: ast.Call = statement.value  # type: ignore[assignment]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert keywords["datasets"] == "load_datasets(workdir)"
    assert keywords["plan"] == "d2d_plan()"
    # 与 test_01b 同一口径的检测器必须拒绝它。
    assert not (
        [ast.unparse(arg) for arg in call.args] == ["manifest"]
        and keywords.get("datasets") == "datasets"
        and keywords.get("plan") == "plan"
    )


# ---------------------------------------------------------------------------
# ⑥ 模块边界
# ---------------------------------------------------------------------------


def test_03_the_driver_lives_outside_the_guarded_package():
    """驱动不得落进被 provider 护栏覆盖的 `app/evaluation/llm/` 包内。"""
    assert DRIVER.exists()
    assert not (LLM_PKG / "d2d_pilot.py").exists()


def test_04_the_guarded_package_does_not_import_the_driver():
    offenders: list[str] = []
    for path in sorted(LLM_PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.endswith("d2d_pilot"):
                    offenders.append(path.name)
            elif isinstance(node, ast.Import):
                if any(alias.name.endswith("d2d_pilot") for alias in node.names):
                    offenders.append(path.name)
    assert offenders == [], offenders


def test_05_importing_the_driver_constructs_nothing():
    """导入驱动不得读环境、不得构造客户端、不得发请求。"""
    assert d2d_pilot.default_pilot_config is not None
    # 模块级没有 asyncio.run / build_chat_model / OfflineExecutor 的调用。
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    module_level_calls = {
        ast.unparse(node.func)
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    }
    assert not module_level_calls


# ---------------------------------------------------------------------------
# ⑦ 运行时:坏清单必须先于一切执行侧副作用失败
# ---------------------------------------------------------------------------


class _SpyFired(Exception):
    """探针被调用 —— 说明闸门**没有**拦住。"""


class _Spies:
    """记录并拒绝 provider 工厂 / 执行器 / 默认配置的构造尝试。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def provider_factory(self, *args, **kwargs):
        self.calls.append("make_llm_factory")
        raise _SpyFired("provider 工厂在闸门通过之前被构造")

    def executor(self, *args, **kwargs):
        self.calls.append("OfflineExecutor")
        raise _SpyFired("执行器在闸门通过之前被构造")

    def default_config(self, *args, **kwargs):
        self.calls.append("default_pilot_config")
        raise _SpyFired("默认配置在闸门通过之前被读取")


@pytest.fixture
def spies(monkeypatch) -> _Spies:
    import app.evaluation.llm.executor as executor_module
    import app.evaluation.real_provider as real_provider

    probe = _Spies()
    monkeypatch.setattr(real_provider, "make_llm_factory", probe.provider_factory)
    monkeypatch.setattr(executor_module, "OfflineExecutor", probe.executor)
    monkeypatch.setattr(d2d_pilot, "default_pilot_config", probe.default_config)

    # 探针本身必须是活的 —— 否则"没被调用"可能只是补丁没生效。
    for direct in (
        real_provider.make_llm_factory,
        executor_module.OfflineExecutor,
        d2d_pilot.default_pilot_config,
    ):
        with pytest.raises(_SpyFired):
            direct()
    probe.calls.clear()
    return probe


def _frozen_manifest(tmp_path):
    datasets = load_datasets(tmp_path / "fixtures")
    manifest = build_d2d_manifest(
        git_commit=GIT_COMMIT, datasets=datasets, created_at_utc=CREATED_AT
    )
    return datasets, manifest


def _run(tmp_path, datasets, manifest, plan, *, allow_network=False, workdir=None):
    return run_d2d_pilot(
        datasets=datasets,
        manifest=manifest,
        plan=plan,
        git_commit=GIT_COMMIT,
        api_key="synthetic-not-a-real-key",
        experiment_id="d2d-gate-test",
        workdir=workdir if workdir is not None else tmp_path / "run",
        allow_network=allow_network,
    )


def test_06_the_driver_refuses_without_explicit_authorisation(tmp_path):
    datasets, manifest = _frozen_manifest(tmp_path)
    with pytest.raises(D2DPilotDisabled):
        _run(tmp_path, datasets, manifest, d2d_plan())


def test_07_the_gate_runs_before_the_authorisation_check(tmp_path):
    """即使没有授权,闸门也要先跑 —— 坏清单必须在授权之前就被拒绝。

    这保证"授权打开的那一次"不会成为**第一次**校验清单的时刻。
    """
    from app.evaluation.llm.protocol import ManifestError

    datasets, manifest = _frozen_manifest(tmp_path)
    tampered = manifest.model_copy(update={"manifest_digest": "0" * 64})

    with pytest.raises(ManifestError):
        _run(tmp_path, datasets, tampered, d2d_plan())


def test_08_the_driver_writes_no_pilot_artifacts_when_refused(tmp_path):
    datasets, manifest = _frozen_manifest(tmp_path)
    workdir = tmp_path / "run"
    with pytest.raises(D2DPilotDisabled):
        _run(tmp_path, datasets, manifest, d2d_plan(), workdir=workdir)
    assert not workdir.exists() or list(workdir.rglob("d2d-pilot-*")) == []


def test_09_a_tampered_manifest_fails_before_any_execution_side_effect(
    tmp_path, spies
):
    """坏清单必须在 provider / 执行器 / 产物写入 / 实验准入**之前**失败。

    这里显式给 `allow_network=True` —— 证明拦住它的**不是**授权检查,
    而是闸门本身。
    """
    from app.evaluation.llm.protocol import ManifestError

    datasets, manifest = _frozen_manifest(tmp_path)
    tampered = manifest.model_copy(update={"manifest_digest": "0" * 64})
    workdir = tmp_path / "run"

    with pytest.raises(ManifestError):
        _run(tmp_path, datasets, tampered, d2d_plan(), allow_network=True, workdir=workdir)

    assert spies.calls == [], spies.calls
    assert not workdir.exists() or list(workdir.rglob("d2d-pilot-*")) == []


def test_10_a_plan_mismatch_fails_before_any_execution_side_effect(tmp_path, spies):
    """清单合法但**计划对不上** —— 同样必须在执行侧构造之前失败。"""
    from app.evaluation.llm.protocol import ManifestError

    datasets, manifest = _frozen_manifest(tmp_path)
    workdir = tmp_path / "run"
    stale_plan = d2d_plan().model_copy(update={"provider_http_attempt_ceiling": 972})

    with pytest.raises(ManifestError, match="provider_http_attempt_ceiling"):
        _run(
            tmp_path, datasets, manifest, stale_plan,
            allow_network=True, workdir=workdir,
        )

    assert spies.calls == [], spies.calls
    assert not workdir.exists() or list(workdir.rglob("d2d-pilot-*")) == []


def test_11_allow_network_false_still_refuses_after_the_gate(tmp_path, spies):
    """闸门通过之后,`allow_network=False` 仍然拒绝 —— 且不构造任何东西。"""
    datasets, manifest = _frozen_manifest(tmp_path)
    workdir = tmp_path / "run"

    with pytest.raises(D2DPilotDisabled):
        _run(tmp_path, datasets, manifest, d2d_plan(), workdir=workdir)

    assert spies.calls == [], spies.calls
    assert not workdir.exists() or list(workdir.rglob("d2d-pilot-*")) == []
