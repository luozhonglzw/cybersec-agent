"""Phase 9.2-D-2d / F13 —— 正式驱动的 **真实 provider 身份接线**。

F13 的形状
----------
`run_d2d_pilot()` 向 `OfflineExecutor` 注入真实 provider 工厂,却**不给**与之
对应的真实身份 ⇒ `OfflineExecutor` 正确地回落到 `scripted_identity()` 并
**正确地**拒绝构造(守卫:`llm_factory is not None` **且** `identity.is_scripted`
⇒ `ValueError`)。后果不是"记录贴错标签",而是**正式执行根本起不来**。

守卫本身是**对的**,因此本阶段修的是**接线**,不是守卫:
驱动在闸门之后从**同一份**冻结 `PilotProviderConfig` 派生身份。

F13 为什么从既有测试套件里漏了出去
----------------------------------
`test_d2d_manifest_gate.py` 的 `spies` 夹具把
`app.evaluation.llm.executor.OfflineExecutor` **整个类**替换成探针 ⇒ 真实的
`OfflineExecutor.__init__`(及其身份/工厂一致性守卫)**从未**被正式驱动路径触达。

因此本模块**刻意不替换 `OfflineExecutor`**:

    * `test_f13_02/03`  只拦截 `run()` —— **构造仍然是真的**;
    * `test_f13_04`     **完全不碰执行器**:真实 `run()` 跑起来,由**假工厂**
                        在"模型构造"这一步停下(零网络)。

`test_f13_05` 是**负对照**:它复现 F13 的原始形状,证明"真实工厂 + 省略身份"
确实会抛那条一致性 `ValueError` —— 否则"已修复"只是一句主张。
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest

from app.evaluation.d2d_pilot import build_d2d_manifest, run_d2d_pilot
from app.evaluation.llm.executor import HarnessAbort
from app.evaluation.llm.identity import EndpointCategory
from app.evaluation.llm.pilot import load_datasets
from app.evaluation.llm.protocol import ManifestError
from app.evaluation.pilot_config import d2d_plan, default_pilot_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "app" / "evaluation" / "d2d_pilot.py"

GIT_COMMIT = "ca91214c95513b54b2425f2e87a099446ed09161"
CREATED_AT = "2026-10-01T00:00:00+00:00"
GATE = "assert_frozen_pilot_manifest"

#: 身份派生**之后**才允许出现的东西 —— 闸门必须先于它们全部。
#: (名字收集器对 `ast.Attribute` 记的是末段属性名,因此这里不写 `asyncio.run`。)
MUST_FOLLOW_THE_GATE = (
    "default_pilot_config",
    "identity_for_candidate",
    "make_llm_factory",
    "OfflineExecutor",
)


# ---------------------------------------------------------------------------
# 零出网观测
# ---------------------------------------------------------------------------

_AUDITED = frozenset({
    "socket.connect", "socket.bind", "socket.sendto",
    "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
    "urllib.Request", "http.client.connect", "http.client.send",
    "ssl.SSLContext.wrap_socket",
})
_LOOPBACK = ("127.", "::1", "localhost")
_EGRESS: list[str] = []


def _audit(event: str, args: tuple) -> None:
    if event not in _AUDITED:
        return
    detail = ""
    try:
        if event in ("socket.connect", "socket.bind", "socket.sendto"):
            address = args[1] if len(args) > 1 else None
            detail = str(address[0]) if isinstance(address, tuple) and address else str(address)
        elif args:
            detail = str(args[0])
    except Exception:  # noqa: BLE001 —— 描述失败绝不影响被测流程
        detail = ""
    if any(token in detail for token in _LOOPBACK):
        return
    _EGRESS.append(f"{event}:{detail}")


sys.addaudithook(_audit)


@pytest.fixture
def no_egress():
    """断言本测试全程**零对外网络访问**。"""
    _EGRESS.clear()
    yield _EGRESS
    assert _EGRESS == [], f"测试期间检测到网络出口:{_EGRESS}"


# ---------------------------------------------------------------------------
# 工装
# ---------------------------------------------------------------------------


class _StoppedBeforeInvocation(RuntimeError):
    """`run()` 被有意拦截 —— 不执行任何模型调用。"""


def _frozen(tmp_path):
    datasets = load_datasets(tmp_path / "fixtures")
    manifest = build_d2d_manifest(
        git_commit=GIT_COMMIT, datasets=datasets, created_at_utc=CREATED_AT
    )
    return datasets, manifest, d2d_plan()


def _run(tmp_path, datasets, manifest, plan, *, allow_network=False, workdir=None):
    return run_d2d_pilot(
        datasets=datasets,
        manifest=manifest,
        plan=plan,
        git_commit=GIT_COMMIT,
        api_key="synthetic-not-a-real-key",
        experiment_id="d2d-f13-test",
        workdir=workdir if workdir is not None else tmp_path / "run",
        allow_network=allow_network,
    )


def _capture_executor(tmp_path, monkeypatch) -> dict:
    """让**真实** `OfflineExecutor.__init__` 跑完,只拦截 `run()`。

    **只替换 `run` 这一个方法,不替换类** —— 否则就重蹈了 F13 逃逸的老路。
    """
    import app.evaluation.llm.executor as executor_module

    captured: dict = {}

    async def _intercept_run(self, **_kwargs):
        captured["executor"] = self
        raise _StoppedBeforeInvocation("run() 被有意拦截")

    monkeypatch.setattr(executor_module.OfflineExecutor, "run", _intercept_run)
    datasets, manifest, plan = _frozen(tmp_path)
    with pytest.raises(_StoppedBeforeInvocation):
        _run(tmp_path, datasets, manifest, plan, allow_network=True)
    return captured


# ---------------------------------------------------------------------------
# ① F12 顺序不变量:身份派生必须**晚于**冻结闸门
# ---------------------------------------------------------------------------


def test_f13_01_identity_derivation_stays_after_the_frozen_manifest_gate():
    """F13 的修法**不得**破坏 F12:闸门仍是第一条实质语句,身份派生在其后。"""
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_d2d_pilot"
    )
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), ast.unparse(first)
    assert getattr(first.value.func, "id", None) == GATE, ast.unparse(first)
    gate_line = first.lineno

    positions: dict[str, int] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name is None:
            continue
        positions[name] = min(node.lineno, positions.get(name, node.lineno))

    # 闸门之前**零调用**。
    assert min(positions.values()) == gate_line, sorted(positions.items(), key=lambda kv: kv[1])

    for name in MUST_FOLLOW_THE_GATE:
        assert name in positions, f"{name} 必须出现在驱动里"
        assert positions[name] > gate_line, f"{name} 不得先于冻结闸门"


# ---------------------------------------------------------------------------
# ② 真实 `OfflineExecutor.__init__` 被触达,且身份不再是脚本化
# ---------------------------------------------------------------------------


def test_f13_02_the_driver_reaches_the_real_executor_constructor(tmp_path, monkeypatch, no_egress):
    captured = _capture_executor(tmp_path, monkeypatch)
    executor = captured["executor"]

    # 真实工厂被注入 —— 这正是 F13 之前触发守卫的那个条件。
    assert executor.llm_factory is not None
    # 身份**不再**是 SCRIPTED_OFFLINE ⇒ 守卫不再触发。
    assert executor.identity.is_scripted is False
    assert executor.identity.endpoint_category is not EndpointCategory.SCRIPTED_OFFLINE


# ---------------------------------------------------------------------------
# ③ 执行器身份 == 冻结的 D-2d 身份
# ---------------------------------------------------------------------------


def test_f13_03_executor_identity_equals_the_frozen_d2d_identity(
    tmp_path, monkeypatch, no_egress
):
    captured = _capture_executor(tmp_path, monkeypatch)
    identity = captured["executor"].identity

    config = default_pilot_config()

    # ---- 与**冻结清单**逐字段一致(非循环:清单是独立产物) ----
    _, manifest, _ = _frozen(tmp_path)
    assert identity.provider == manifest.provider == "deepseek"
    assert identity.model == manifest.model == "deepseek-flash"
    assert identity.endpoint_category.value == manifest.endpoint_category
    assert identity.endpoint_category is EndpointCategory.OPENAI_COMPATIBLE

    # ---- base_url:只留主机名摘要,且等于冻结 base URL 的主机 ----
    assert identity.base_url_host_sha256 == hashlib.sha256(b"api.deepseek.com").hexdigest()

    # ---- 温度语义:NOT_SET(记录侧)与请求侧**同源** ----
    assert identity.temperature is None
    assert config.model_kwargs()["temperature"] is None
    assert config.recorded_temperature() is None

    # ---- usage 来源:真实 provider ⇒ provider_reported ----
    assert identity.usage_source == "provider_reported"

    # ---- 与既有身份构造 API 的派生结果逐字段相同 ----
    from app.evaluation.real_provider import identity_for_candidate

    expected = identity_for_candidate(
        config.provider_candidate(),
        base_url=config.base_url,
        temperature=config.recorded_temperature(),
    )
    assert identity == expected


# ---------------------------------------------------------------------------
# ④ 端到端:真实 `run()` 启动,由假工厂在模型构造处停下(零网络)
# ---------------------------------------------------------------------------


def test_f13_04_the_real_executor_accepts_the_wired_identity_end_to_end(
    tmp_path, monkeypatch, no_egress
):
    """**不碰 `OfflineExecutor`**:真实构造 + 真实 `run()` 启动。

    假工厂在任何模型构造 / 网络调用之前抛错 ⇒ 执行器把该单元记为 incomplete
    并抛 `HarnessAbort`。这证明 F13 那条一致性 `ValueError` **已经消失**,
    而且整条路径**零网络**。
    """
    import app.evaluation.real_provider as real_provider

    class _FakeModelSeam(RuntimeError):
        """假工厂哨兵 —— 绝不构造任何客户端、绝不发请求。"""

    def _fake_factory(**_kwargs):
        raise _FakeModelSeam("假工厂:在模型构造之前停下")

    monkeypatch.setattr(real_provider, "make_llm_factory", lambda **_: _fake_factory)

    datasets, manifest, plan = _frozen(tmp_path)
    workdir = tmp_path / "run"

    with pytest.raises(HarnessAbort) as info:
        _run(tmp_path, datasets, manifest, plan, allow_network=True, workdir=workdir)

    message = str(info.value)
    # F13 已消失:不再是身份/工厂一致性拒绝。
    assert "SCRIPTED_OFFLINE" not in message
    # 是"工装异常"(执行真的跑起来了),而不是"构造被拒"。
    assert "工装异常" in message
    assert "_FakeModelSeam" in message

    # 没有任何 pilot 产物被创建。
    assert list(workdir.rglob("d2d-pilot-*")) == []


# ---------------------------------------------------------------------------
# ⑤ 负对照:复现 F13 的原始形状
# ---------------------------------------------------------------------------


def test_f13_05_negative_control_the_original_defect_still_reproduces(
    tmp_path, no_egress
):
    """**负对照**:真实工厂 + **省略** `identity` ⇒ 守卫抛出那条一致性 `ValueError`。

    修复前的 `run_d2d_pilot` 正是这样调用的。若这条不再成立,
    说明守卫被削弱了 —— 那是更糟的结果。
    """
    import app.evaluation.llm.executor as executor_module
    from app.evaluation.real_provider import make_llm_factory

    config = default_pilot_config()
    # 构造工厂**不**构造客户端:真正的 `ChatOpenAI` 由工厂被调用时才建。
    factory = make_llm_factory(
        candidate=config.provider_candidate(),
        api_key="synthetic-not-a-real-key",
        allow_network=True,
        **config.model_kwargs(),
    )

    with pytest.raises(ValueError, match="SCRIPTED_OFFLINE"):
        executor_module.OfflineExecutor(
            workdir=tmp_path / "negative-control",
            experiment_id="d2d-f13-negative-control",
            plan=d2d_plan(),
            manifest_digest="0" * 64,
            llm_factory=factory,
            # identity 刻意省略 —— 这正是 F13 的原始形状。
        )


# ---------------------------------------------------------------------------
# ⑥ 坏清单 / 未授权:必须在身份派生之前就失败
# ---------------------------------------------------------------------------


def test_f13_06_a_tampered_manifest_fails_before_identity_derivation(
    tmp_path, monkeypatch, no_egress
):
    import app.evaluation.llm.executor as executor_module
    import app.evaluation.real_provider as real_provider

    called: list[str] = []

    def _trip_identity(*_args, **_kwargs):
        called.append("identity_for_candidate")
        raise AssertionError("身份派生不得发生在闸门之前")

    def _trip_factory(*_args, **_kwargs):
        called.append("make_llm_factory")
        raise AssertionError("provider 工厂不得发生在闸门之前")

    class _TripExecutor:
        def __init__(self, *_args, **_kwargs):
            called.append("OfflineExecutor")
            raise AssertionError("执行器不得发生在闸门之前")

    monkeypatch.setattr(real_provider, "identity_for_candidate", _trip_identity)
    monkeypatch.setattr(real_provider, "make_llm_factory", _trip_factory)
    monkeypatch.setattr(executor_module, "OfflineExecutor", _TripExecutor)

    datasets, manifest, plan = _frozen(tmp_path)
    tampered = manifest.model_copy(update={"manifest_digest": "0" * 64})
    workdir = tmp_path / "run"

    with pytest.raises(ManifestError):
        _run(tmp_path, datasets, tampered, plan, allow_network=True, workdir=workdir)

    assert called == [], called
    assert not workdir.exists()


def test_f13_07_allow_network_false_refuses_before_identity_derivation(
    tmp_path, monkeypatch, no_egress
):
    """闸门通过之后,`allow_network=False` 仍然拒绝 —— 且**不派生身份**、不构造任何东西。"""
    from app.evaluation.d2d_pilot import D2DPilotDisabled
    import app.evaluation.llm.executor as executor_module
    import app.evaluation.real_provider as real_provider

    called: list[str] = []

    monkeypatch.setattr(
        real_provider, "identity_for_candidate",
        lambda *_a, **_k: called.append("identity_for_candidate"),
    )
    monkeypatch.setattr(
        real_provider, "make_llm_factory",
        lambda *_a, **_k: called.append("make_llm_factory"),
    )
    monkeypatch.setattr(
        executor_module, "OfflineExecutor",
        type("_Trip", (), {"__init__": lambda self, *_a, **_k: called.append("OfflineExecutor")}),
    )

    datasets, manifest, plan = _frozen(tmp_path)
    workdir = tmp_path / "run"

    with pytest.raises(D2DPilotDisabled):
        _run(tmp_path, datasets, manifest, plan, workdir=workdir)

    assert called == [], called
    assert not workdir.exists()
