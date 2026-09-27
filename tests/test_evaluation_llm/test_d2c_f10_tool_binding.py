"""Phase 9.2-D-2c F10 —— **C1 工具绑定路径**的聚焦测试。

F10 是什么
----------
C1 冻结的请求形状里**根本没有 `tools` 字段**:线上请求体的键集合是
`[max_tokens, messages, model, stream, thinking]`,工具契约从未到达 provider。
于是 C1 的判定建立在一条从未成立的前提上 —— 它回答的是"模型给的 tool call
形状合不合法",而模型压根没被告知有哪些工具。

本文件把修复钉住,分四层:

    1  **单一来源**   `ToolBinding` 持有工具对象;契约由它派生,不复制 schema
    2  **绑定**       `bind_declared_tools()` 只绑声明的东西
    3  **观测**       线上请求体的 `tools` 字段(经真实 SDK 序列化 + MockTransport)
    4  **判定**       前置核验五分支:不可观测 / 契约不同源 / 声明了却没发 /
                      观测与声明不一致 / 一致 ⇒ 落到形状判定

一条贯穿的纪律:**绝不从声明推断观测**
------------------------------------
`observed_tools_in_payload` 只从线上请求体读。请求体不可观测时记 `None`,
`binding_agreement` 随之记 `None` —— 不是 `True`。"没看见"永远不等于"一致"。

本文件全程离线
--------------
F7 已确认:Windows 异步路径上审计钩子的出口观测会漏报,因此 `egress_events == 0`
**两个方向都不作证据**。本文件的结构性证据是:

    1. 传输层**就是** `httpx.MockTransport` 本体(身份比较);
    2. 四处 tripwire 把 DNS 与两个真实 httpx transport 换成"调用即抛";
    3. base_url 指向 RFC 2606 保留的 `.invalid` 主机 —— 它**永远无法解析**,
       请求却仍然成功返回,所以不可能发生过 DNS 或 TCP。

零 provider 调用、零 API 额度消耗、零真实凭据。
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.evaluation import calibration as C
from app.evaluation.calibration import config as cal_config
from app.evaluation.calibration import stages as S
from app.evaluation.calibration.synthetic import (
    SyntheticProviderModel,
    SyntheticTurn,
    synthetic_ai_message,
    synthetic_tool_call,
)
from app.evaluation.calibration.verdict import StageVerdict
from app.evaluation.llm.dataset import LLM_TASKS, build_datasets
from app.evaluation.llm.offline_guard import NetworkEgressGuard
from app.evaluation.llm.raw import find_secret_patterns
from app.evaluation.real_provider import build_chat_model

#: 明显是合成物的凭据占位符。**不是**任何真实密钥。
SYNTHETIC_API_KEY = "sk-d2c-f10-synthetic-placeholder-0000000000"

#: RFC 2606 保留顶级域,永远无法解析。见模块 docstring 第 3 条。
UNRESOLVABLE_HOST = "f10-tool-binding-probe.invalid"
UNRESOLVABLE_BASE_URL = f"https://{UNRESOLVABLE_HOST}/v1"

TRIPWIRE_MARKER = "FAIL-CLOSED TRIPWIRE"

EXPERIMENT_ID = "d2c-cal-f10-20260923"

#: 本地合成响应(OpenAI-compatible 形状)。**不来自任何 provider。**
CANNED_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-f10-synthetic",
    "object": "chat.completion",
    "created": 0,
    "model": "DeepSeek-V4.1-Flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "SYNTHETIC-LOCAL"},
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
}

#: 一个**合法**的 tool call(参数符合 `analyze_risk_tool` 的契约)。
LEGAL_CALL = synthetic_tool_call(
    name="analyze_risk_tool", args={"indicator": "1.2.3.4"}, call_id="c1-1"
)


# ---------------------------------------------------------------------------
# 模块级离线证明(补充信号,不是主证据)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _module_runs_under_a_strict_egress_guard():
    """整个模块在**严格**出口守卫下运行 —— 第一次越界就在发生处炸掉。

    刻意说明它的**地位**:F7 已确认 Windows 异步路径上该守卫会漏报,所以
    `egress_events == 0` **不是**"没有出口"的证明。主证据是各测试内的四处
    tripwire 与不可解析主机;这个守卫只是一层补充,而且是有牙的补充
    (strict 模式在发生处抛错,不是事后检查)。
    """
    guard = NetworkEgressGuard(strict=True)
    with guard:
        yield
    guard.assert_clean()


# ---------------------------------------------------------------------------
# 工装(与 `test_d2c_f8_payload_seam.py` 同构)
# ---------------------------------------------------------------------------
#
# 刻意与 F8 文件**各自持有**一份:那个文件是已提交的冻结证据,不因 F10 而改动;
# 而"两个模块共用一份可变的测试工装"本身就是一个新的分叉点。


class _Capture:
    """MockTransport 的请求记录器。**不记录任何凭据材料。**"""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        try:
            body: Any = json.loads(request.content.decode("utf-8"))
        except Exception:  # noqa: BLE001 - 解析失败不该让测试崩
            body = {"__unparsed__": True}
        self.requests.append(
            {"method": request.method, "host": request.url.host, "body": body}
        )
        return httpx.Response(200, json=CANNED_RESPONSE)


def _arm_tripwires(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """把 DNS 与两个真实 httpx transport 换成"调用即抛"。

    `socket.socket` **刻意不动** —— 替换 socket 工厂会破坏 asyncio 事件循环
    (Windows 的 `socketpair()` 依赖它),守卫就变成了"测试跑不起来"。
    """
    fired: list[str] = []

    def _make(name: str):
        def _boom(*_args: Any, **_kwargs: Any):
            fired.append(name)
            raise RuntimeError(f"{TRIPWIRE_MARKER}: {name} invoked")

        return _boom

    monkeypatch.setattr(socket, "getaddrinfo", _make("socket.getaddrinfo"))
    monkeypatch.setattr(socket, "gethostbyname", _make("socket.gethostbyname"))
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        _make("httpx.AsyncHTTPTransport.handle_async_request"),
    )
    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        _make("httpx.HTTPTransport.handle_request"),
    )
    return fired


def _c1_shaped_kwargs() -> dict[str, Any]:
    """C1 形状的 `build_chat_model` kwargs(**与提交路径同源**)。

    唯一改动是 base_url 指向不可解析主机 —— 它不进请求体,因此不影响形状。
    """
    cfg = cal_config.default_calibration_config()
    kwargs = dict(cfg.model_kwargs())
    kwargs["base_url"] = UNRESOLVABLE_BASE_URL
    return kwargs


def _c1_messages() -> list[Any]:
    """与 `stages.run_c1` 逐字一致的探针消息。"""
    return [
        SystemMessage(content="D-2c 标定:工具形状探针。请调用一个工具。"),
        HumanMessage(content="请调用一个可用工具来完成一次查询。"),
    ]


def _seamed_model(
    capture: _Capture, *, kwargs: dict[str, Any] | None = None
) -> tuple[Any, httpx.MockTransport, httpx.AsyncClient]:
    """经**提交的** `build_chat_model` 路径构造一个带 MockTransport 的模型。"""
    transport = httpx.MockTransport(capture.handler)
    client = httpx.AsyncClient(transport=transport)
    cfg = cal_config.default_calibration_config()
    model = build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        http_async_client=client,
        allow_network=True,
        **(kwargs if kwargs is not None else _c1_shaped_kwargs()),
    )
    return model, transport, client


def _wire_tools_field(request: S.CalibrationRequest) -> list | None:
    """合成传输按**声明**折算出的 `tools` 字段。

    走 `S.bindable_tools()` —— 与 `bind_declared_tools()` 同一个接缝。
    声明为空 / 未声明时返回 `None`,即**不加该字段**。
    """
    binding = request.tool_binding
    if binding is None:
        return None
    tools = S.bindable_tools(binding)
    if not tools:
        return None
    return [
        {"type": "function", "function": {"name": getattr(tool, "name", "")}}
        for tool in tools
    ]


def _request(*, binding: S.ToolBinding | None) -> S.CalibrationRequest:
    """一条 C1 形状的请求(凭据无关,`model_kwargs` 来自冻结配置)。"""
    cfg = cal_config.default_calibration_config()
    return S.CalibrationRequest(
        stage=S.Stage.C1,
        messages=tuple(_c1_messages()),
        model_kwargs=cfg.model_kwargs(),
        output_token_cap=cfg.output_token_cap,
        tool_binding=binding,
    )


def _evidence(
    *,
    payload: Any,
    binding: S.ToolBinding,
    contract: S.ToolContract | None = None,
    supplied_contract: S.ToolContract | None = None,
    calls: tuple[dict[str, Any], ...] = (LEGAL_CALL,),
    invalid: tuple[Any, ...] = (),
) -> dict[str, Any]:
    """按给定请求体/绑定/契约装配一份 C1 证据。"""
    message = synthetic_ai_message(
        content="", tool_calls=list(calls), finish_reason="tool_calls"
    )
    if invalid:
        message = synthetic_ai_message(
            content="",
            tool_calls=list(calls),
            invalid_tool_calls=list(invalid),
            finish_reason="tool_calls",
        )
    return S.c1_evidence(
        request=_request(binding=binding),
        result=S.InvocationResult(message=message, request_payload=payload),
        contract=contract or binding.as_contract(),
        binding=binding,
        config=cal_config.default_calibration_config(),
        supplied_contract=supplied_contract,
    )


def _wire_payload_for(binding: S.ToolBinding) -> dict[str, Any]:
    """声明 `binding` 时,合成传输上"应当"出现的请求体。"""
    payload: dict[str, Any] = {"model": cal_config.CALIBRATION_REQUESTED_MODEL}
    payload.update(cal_config.default_calibration_config().model_kwargs()["extra_body"])
    tools_field = _wire_tools_field(_request(binding=binding))
    if tools_field is not None:
        payload["tools"] = tools_field
    return payload


# ===========================================================================
# 1. 单一来源:绑定持有工具对象,契约由它派生
# ===========================================================================


def test_00_binding_holds_tool_objects_not_schema_copies():
    """`ToolBinding` 只有 `tools` 与 `source` —— **没有**任何 schema 字段。

    若为了校验而复制一份 schema,同一件事就有了两个事实来源:工具对象改了
    而副本没改时,发出去的是新工具集、校验用的是旧契约,而两边看起来都合理。
    """
    field_names = {item.name for item in dataclasses.fields(S.ToolBinding)}
    assert field_names == {"tools", "source"}, field_names
    for forbidden in ("schemas", "arg_schemas", "schema", "contract"):
        assert forbidden not in field_names, f"{forbidden} 会让 schema 出现第二份事实来源"

    binding = S.production_tool_binding()
    assert isinstance(binding.tools, tuple)
    assert binding.tools, "生产绑定不得为空"
    for tool in binding.tools:
        assert hasattr(tool, "tool_call_schema"), (
            f"{tool!r} 不是工具对象 —— 绑定里必须存工具本身,而不是 schema"
        )


def test_01_binding_names_are_the_production_tool_names_in_order():
    """C1 与 C4 是**两个**绑定点;今天同源是巧合,因此必须有断言钉住。"""
    from app.tools import DEFAULT_TOOLS

    binding = S.production_tool_binding()
    assert binding.tool_names == tuple(tool.name for tool in DEFAULT_TOOLS)
    assert binding.tool_names == (
        "query_security_logs_tool",
        "query_threat_intel_tool",
        "analyze_risk_tool",
        "plan_response_tool",
    )
    assert binding.source == "app.tools.DEFAULT_TOOLS"


def test_02_contract_is_derived_from_the_binding():
    """验证契约 = `binding.as_contract()`,不是另一份独立派生的 schema。"""
    binding = S.production_tool_binding()
    derived = binding.as_contract()
    published = S.production_tool_contract()
    assert published.allowed_tool_names == binding.tool_names
    assert derived.allowed_tool_names == published.allowed_tool_names
    assert derived.arg_schemas == published.arg_schemas
    # 契约必须带 `required` —— 退化到 `tool.args` 会让"必填缺失"恒测不出来。
    assert any("required" in schema for schema in derived.arg_schemas.values())


def test_03_bindable_tools_returns_the_binding_tools_identically():
    """`bindable_tools()` 与 `as_contract()` 出自同一个元组。"""
    binding = S.production_tool_binding()
    assert S.bindable_tools(binding) == binding.tools
    assert all(a is b for a, b in zip(S.bindable_tools(binding), binding.tools))


# ===========================================================================
# 2. 绑定:只绑声明的东西
# ===========================================================================


class _RecordingModel:
    """记录 `bind_tools()` 调用次数的极小桩模型。"""

    def __init__(self) -> None:
        self.bind_calls = 0
        self.bound: tuple[str, ...] = ()

    def bind_tools(self, tools: Any) -> "_RecordingModel":
        self.bind_calls += 1
        self.bound = tuple(getattr(tool, "name", str(tool)) for tool in tools)
        return self


def test_04_bind_declared_tools_binds_exactly_what_is_declared():
    binding = S.production_tool_binding()
    model = _RecordingModel()
    returned = S.bind_declared_tools(model, _request(binding=binding))
    assert returned is model
    assert model.bind_calls == 1
    assert model.bound == binding.tool_names


def test_05_no_declaration_means_bind_tools_is_never_called():
    """未声明绑定时**不得**调用 `bind_tools([])`。

    线上"根本没有 `tools` 字段"与"`tools` 是空列表"是两件不同的事:
    后者是"我们绑了个空集",前者是"我们根本没绑"。记录里必须分得开。
    """
    model = _RecordingModel()
    returned = S.bind_declared_tools(model, _request(binding=None))
    assert returned is model
    assert model.bind_calls == 0, "未声明绑定时不得触碰 bind_tools"
    assert model.bound == ()


def test_06_an_empty_binding_also_skips_bind_tools():
    model = _RecordingModel()
    S.bind_declared_tools(model, _request(binding=S.ToolBinding(tools=(), source="empty")))
    assert model.bind_calls == 0


def test_07_the_synthetic_model_sees_the_declared_names():
    """合成模型与真实调用器走**同一个**接缝,因此不可能在"绑什么"上分叉。"""
    binding = S.production_tool_binding()
    model = SyntheticProviderModel(turns=(SyntheticTurn(content="x"),))
    S.bind_declared_tools(model, _request(binding=binding))
    assert model.bound_tool_names == binding.tool_names


# ===========================================================================
# 3. 观测:线上请求体里的 `tools` 字段(真实 SDK 序列化)
# ===========================================================================


def test_08_the_wire_tools_field_carries_exactly_the_bound_names(monkeypatch):
    """经**真实** `langchain_openai` 序列化后,线上 `tools` 名字逐个等于绑定。"""
    fired = _arm_tripwires(monkeypatch)
    binding = S.production_tool_binding()
    capture = _Capture()
    model, _transport, client = _seamed_model(capture)
    bound = S.bind_declared_tools(model, _request(binding=binding))
    try:
        asyncio.run(bound.ainvoke(_c1_messages()))
    finally:
        asyncio.run(client.aclose())

    assert fired == [], f"tripwire 被触发 —— 测试逃出了本地:{fired}"
    assert len(capture.requests) == 1, "恰好一次请求"
    request = capture.requests[0]
    assert request["host"] == UNRESOLVABLE_HOST, (
        "请求打到不可解析主机却仍然成功 —— 只有本地 mock 能解释这一点"
    )
    body = request["body"]
    assert "tools" in body, "绑定之后线上必须出现 tools 字段"
    observed = S._payload_tool_names(body)
    assert observed == list(binding.tool_names), observed
    assert S._payload_tools_present(body) is True


def test_09_binding_does_not_perturb_the_frozen_request_shape(monkeypatch):
    """绑定只在线上**新增** `tools`,冻结的四个字段逐字不变。"""
    _arm_tripwires(monkeypatch)
    kwargs = _c1_shaped_kwargs()
    binding = S.production_tool_binding()

    captures = []
    bodies = []
    for with_binding in (False, True):
        capture = _Capture()
        model, _transport, client = _seamed_model(capture, kwargs=kwargs)
        target = (
            S.bind_declared_tools(model, _request(binding=binding))
            if with_binding
            else model
        )
        try:
            asyncio.run(target.ainvoke(_c1_messages()))
        finally:
            asyncio.run(client.aclose())
        captures.append(capture)
        bodies.append(capture.requests[0]["body"])

    unbound, bound = bodies
    assert "tools" not in unbound, "未绑定时线上不得出现 tools 字段"
    assert "tools" in bound
    assert {k: v for k, v in bound.items() if k != "tools"} == unbound, (
        "绑定改变了冻结的请求形状 —— 除了 tools 之外任何差异都是回归"
    )
    for key in ("max_tokens", "model", "stream", "thinking"):
        assert key in bound, key
    assert "temperature" not in bound
    assert "tool_choice" not in bound


def test_10_an_unbound_request_yields_no_tools_field(monkeypatch):
    """未绑定时**线上没有** `tools` 字段 —— 这是"根本没绑"的结构证据。"""
    _arm_tripwires(monkeypatch)
    capture = _Capture()
    model, _transport, client = _seamed_model(capture)
    try:
        asyncio.run(model.ainvoke(_c1_messages()))
    finally:
        asyncio.run(client.aclose())
    body = capture.requests[0]["body"]
    assert "tools" not in body
    assert S._payload_tools_present(body) is False
    assert S._payload_tool_names(body) == []


def test_11_the_tripwires_have_teeth(monkeypatch):
    """恒不触发的 tripwire 与没有 tripwire 是一回事 —— 甚至更糟。"""
    fired = _arm_tripwires(monkeypatch)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        httpx.AsyncHTTPTransport.handle_async_request(None, None)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        httpx.HTTPTransport.handle_request(None)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        socket.gethostbyname("example.com")
    assert len(fired) == 4


def test_12_the_effective_transport_is_the_mock_itself(monkeypatch):
    """身份比较,不是"看起来像" —— 真实 transport 不得出现在生效路径上。"""
    _arm_tripwires(monkeypatch)
    capture = _Capture()
    model, transport, client = _seamed_model(capture)
    try:
        effective = model.root_async_client._client._transport
        assert effective is transport
        assert not isinstance(effective, httpx.AsyncHTTPTransport)
    finally:
        asyncio.run(client.aclose())


def test_13_the_real_wire_payload_satisfies_the_c1_precheck(monkeypatch):
    """把**真实序列化**出来的请求体喂进 C1 证据链 ⇒ 前置核验通过、判定 PASS。"""
    _arm_tripwires(monkeypatch)
    binding = S.production_tool_binding()
    capture = _Capture()
    model, _transport, client = _seamed_model(capture)
    bound = S.bind_declared_tools(model, _request(binding=binding))
    try:
        asyncio.run(bound.ainvoke(_c1_messages()))
    finally:
        asyncio.run(client.aclose())
    body = capture.requests[0]["body"]

    evidence = _evidence(payload=body, binding=binding)
    assert evidence["tools_field_present_in_payload"] is True
    assert evidence["observed_tools_in_payload"] == list(binding.tool_names)
    assert evidence["binding_agreement"] is True
    assert evidence["contract_agreement"] is True
    assert evidence["supplied_contract_agreement"] is None
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is StageVerdict.PASS, verdict.reason


# ===========================================================================
# 4. 判定:前置核验五分支
# ===========================================================================


def test_14_unobservable_payload_is_inconclusive_not_a_pass():
    """请求体不可观测 ⇒ INCONCLUSIVE —— 即使形状证据完美。"""
    binding = S.production_tool_binding()
    evidence = _evidence(payload=None, binding=binding)
    assert evidence["request_payload_available"] is False
    assert evidence["observed_tools_in_payload"] is None
    assert evidence["binding_agreement"] is None, "观测不到时**不得**记成一致"
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is StageVerdict.INCONCLUSIVE, verdict.reason
    assert "无法核验" in verdict.reason


def test_15_declared_tools_without_a_tools_field_is_a_failure():
    """声明了 4 个工具、线上却没有 `tools` 字段 ⇒ 工具契约从未到达 provider。"""
    binding = S.production_tool_binding()
    payload = _wire_payload_for(binding)
    del payload["tools"]
    evidence = _evidence(payload=payload, binding=binding)
    assert evidence["tools_field_present_in_payload"] is False
    assert evidence["binding_agreement"] is False
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is StageVerdict.FAIL, verdict.reason
    assert "tools" in verdict.reason


def test_16_observed_names_differing_from_the_declaration_is_a_failure():
    """线上工具名与声明不一致 ⇒ 请求在到达 provider 之前被改写过。"""
    binding = S.production_tool_binding()
    payload = _wire_payload_for(binding)
    payload["tools"] = [{"type": "function", "function": {"name": "some_other_tool"}}]
    evidence = _evidence(payload=payload, binding=binding)
    assert evidence["observed_tools_in_payload"] == ["some_other_tool"]
    assert evidence["binding_agreement"] is False
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is StageVerdict.FAIL, verdict.reason
    assert "不一致" in verdict.reason


def test_17_an_empty_tools_field_does_not_match_a_non_empty_declaration():
    """`tools: []` 与"声明了 4 个工具"不一致 —— 空集不是"一致"。"""
    binding = S.production_tool_binding()
    payload = _wire_payload_for(binding)
    payload["tools"] = []
    evidence = _evidence(payload=payload, binding=binding)
    assert evidence["tools_field_present_in_payload"] is True
    assert evidence["observed_tools_in_payload"] == []
    assert evidence["binding_agreement"] is False
    assert S.c1_evidence_verdict(evidence).verdict is StageVerdict.FAIL


def test_18_undeclared_tools_on_the_wire_are_a_failure():
    """反向分叉也要拦:未声明任何工具,线上却出现了工具。"""
    binding = S.ToolBinding(tools=(), source="empty")
    payload = {"model": "deepseek-flash", "tools": [{"function": {"name": "sneaky_tool"}}]}
    evidence = _evidence(payload=payload, binding=binding, calls=(LEGAL_CALL,))
    assert evidence["tool_binding_declared"] == []
    assert evidence["observed_tools_in_payload"] == ["sneaky_tool"]
    assert evidence["binding_agreement"] is False
    assert S.c1_evidence_verdict(evidence).verdict is StageVerdict.FAIL


def test_19_the_declaration_is_never_used_to_fill_the_observation():
    """**核心纪律**:观测侧只从请求体读,绝不拿声明回填。

    若回填,`binding_agreement` 会变成一个恒真的判断 —— 而它正是"工具契约
    到底有没有发出去"这个问题的唯一答案来源。
    """
    binding = S.production_tool_binding()
    payload = _wire_payload_for(binding)
    del payload["tools"]
    evidence = _evidence(payload=payload, binding=binding)
    # 声明侧照旧是 4 个;观测侧必须是"真的没看到",而不是"声明说有"。
    assert evidence["tool_binding_declared"] == list(binding.tool_names)
    assert evidence["observed_tools_in_payload"] == []
    assert evidence["observed_tools_in_payload"] != evidence["tool_binding_declared"]
    assert evidence["binding_agreement"] is False

    # 不可观测时更严格:`None` 不是 `True`,也不是 `False`。
    unobservable = _evidence(payload=None, binding=binding)
    assert unobservable["observed_tools_in_payload"] is None
    assert unobservable["binding_agreement"] is None


def test_20_a_verifier_contract_that_diverges_from_the_binding_is_a_failure():
    """判定用的契约必须由绑定派生;从别处拿来的契约一律 FAIL。"""
    binding = S.production_tool_binding()
    other = S.ToolContract(allowed_tool_names=("only_one_tool",), arg_schemas={})
    evidence = _evidence(payload=_wire_payload_for(binding), binding=binding, contract=other)
    assert evidence["allowed_tool_names"] == ["only_one_tool"]
    assert evidence["contract_agreement"] is False
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is StageVerdict.FAIL, verdict.reason
    assert "不同源" in verdict.reason


def test_21_a_supplied_contract_that_diverges_from_the_binding_is_a_failure():
    """调用方**另外**给一份契约时,它也必须与绑定同源。"""
    binding = S.production_tool_binding()
    other = S.ToolContract(allowed_tool_names=("only_one_tool",), arg_schemas={})
    payload = _wire_payload_for(binding)

    agreeing = _evidence(payload=payload, binding=binding, supplied_contract=binding.as_contract())
    assert agreeing["supplied_contract_agreement"] is True
    assert S.c1_evidence_verdict(agreeing).verdict is StageVerdict.PASS

    diverging = _evidence(payload=payload, binding=binding, supplied_contract=other)
    assert diverging["supplied_contract_agreement"] is False
    verdict = S.c1_evidence_verdict(diverging)
    assert verdict.verdict is StageVerdict.FAIL, verdict.reason
    assert "两个" in verdict.reason or "不同源" in verdict.reason


def test_22_agreeing_evidence_falls_through_to_the_shape_verdict():
    """前置核验通过之后,判定权**交还**给形状判定 —— 不是在这里替它下结论。"""
    binding = S.production_tool_binding()
    payload = _wire_payload_for(binding)

    legal = _evidence(payload=payload, binding=binding)
    assert S.c1_evidence_verdict(legal).verdict is StageVerdict.PASS

    no_call = _evidence(payload=payload, binding=binding, calls=())
    verdict = S.c1_evidence_verdict(no_call)
    assert verdict.verdict is StageVerdict.FAIL
    assert "没有给出任何 tool call" in verdict.reason
    assert "工具契约没有被检验到" in verdict.reason

    unknown = _evidence(
        payload=payload,
        binding=binding,
        calls=(
            synthetic_tool_call(name="delete_everything_tool", args={}, call_id="x"),
        ),
    )
    verdict = S.c1_evidence_verdict(unknown)
    assert verdict.verdict is StageVerdict.FAIL
    assert S.ToolCallDefect.UNKNOWN_TOOL.value in verdict.reason


def test_23_the_precheck_precedes_the_shape_verdict():
    """一个"没调工具"的响应在**工具没发出去**时,理由必须是前者。

    理由错位是最难发现的一类错误:结论碰巧对,记录却把人引向错误的现场。
    """
    binding = S.production_tool_binding()
    payload = _wire_payload_for(binding)
    del payload["tools"]
    evidence = _evidence(payload=payload, binding=binding, calls=())
    verdict = S.c1_evidence_verdict(evidence)
    assert verdict.verdict is StageVerdict.FAIL
    assert "没有给出任何 tool call" not in verdict.reason, (
        "理由落到了形状判定上 —— 前置核验没有排在最前面"
    )
    assert "没有" in verdict.reason and "tools" in verdict.reason


# ===========================================================================
# 5. 缺失绑定 ⇒ ABORT(配置错误必须表现成配置错误)
# ===========================================================================


def test_24_missing_tool_binding_is_a_dependency_error():
    deps = S.StageDeps(tool_contract=S.production_tool_contract())
    with pytest.raises(S.StageDependencyMissing):
        S.resolve_tool_binding(deps)


def test_25_missing_tool_binding_aborts_c1_through_the_harness(tmp_path):
    """端到端:缺绑定 ⇒ C1 **ABORT**,而不是 FAIL,更不是 PASS。"""
    cfg = cal_config.default_calibration_config()

    async def invoke(request: S.CalibrationRequest) -> S.InvocationResult:
        payload: dict[str, Any] = {"model": cfg.requested_model}
        payload.update(request.model_kwargs.get("extra_body") or {})
        if request.stage is S.Stage.C0A:
            message = synthetic_ai_message(content="OK")
        elif request.stage is S.Stage.C0B:
            message = synthetic_ai_message(
                content="x" * 64,
                finish_reason="length",
                normalized_usage={
                    "input_tokens": 20,
                    "output_tokens": request.output_token_cap,
                    "total_tokens": 20 + request.output_token_cap,
                },
            )
        else:  # pragma: no cover - C1 之前就 ABORT,调用器不会被触及
            raise AssertionError(f"调用器不应处理 {request.stage}")
        return S.InvocationResult(message=message, request_payload=payload, http_attempts=1)

    harness = C.CalibrationHarness(
        experiment_id=EXPERIMENT_ID,
        workdir_root=tmp_path / "wd",
        deps=S.StageDeps(invoke=invoke, tool_contract=S.production_tool_contract()),
    )
    run = asyncio.run(harness.run())

    assert run.verdict_of(S.Stage.C0A) is StageVerdict.PASS
    assert run.verdict_of(S.Stage.C0B) is StageVerdict.PASS
    assert run.verdict_of(S.Stage.C1) is StageVerdict.ABORT
    assert run.halted_at is S.Stage.C1
    assert run.skipped_stages == (S.Stage.C2, S.Stage.C3, S.Stage.C4)
    assert run.passed_all is False
    reason = run.records[S.STAGE_ORDER.index(S.Stage.C1)].reason
    assert "依赖缺失" in reason
    assert "tool_binding" in reason


def test_26_c1_passes_through_the_real_harness_with_a_faithful_transport(tmp_path):
    """端到端(正向):声明 → 绑定 → 线上 → 观测 → PASS。"""
    cfg = cal_config.default_calibration_config()
    binding = S.production_tool_binding()

    async def invoke(request: S.CalibrationRequest) -> S.InvocationResult:
        payload: dict[str, Any] = {"model": cfg.requested_model}
        payload.update(request.model_kwargs.get("extra_body") or {})
        tools_field = _wire_tools_field(request)
        if tools_field is not None:
            payload["tools"] = tools_field
        if request.stage is S.Stage.C0A:
            message = synthetic_ai_message(content="OK")
        elif request.stage is S.Stage.C0B:
            message = synthetic_ai_message(
                content="x" * 64,
                finish_reason="length",
                normalized_usage={
                    "input_tokens": 20,
                    "output_tokens": request.output_token_cap,
                    "total_tokens": 20 + request.output_token_cap,
                },
            )
        elif request.stage is S.Stage.C1:
            message = synthetic_ai_message(
                content="", tool_calls=[dict(LEGAL_CALL)], finish_reason="tool_calls"
            )
        else:  # pragma: no cover
            raise AssertionError(f"调用器不处理 {request.stage}")
        return S.InvocationResult(message=message, request_payload=payload, http_attempts=1)

    harness = C.CalibrationHarness(
        experiment_id=EXPERIMENT_ID,
        workdir_root=tmp_path / "wd",
        deps=S.StageDeps(
            invoke=invoke,
            tool_contract=S.production_tool_contract(),
            tool_binding=binding,
            roundtrip=[
                synthetic_ai_message(
                    content="",
                    tool_calls=[
                        synthetic_tool_call(
                            name="analyze_risk_tool",
                            args={"indicator": "1.2.3.4"},
                            call_id="rt-1",
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ToolMessage(content="{}", tool_call_id="rt-1"),
                synthetic_ai_message(content="结论:已评估。"),
            ],
        ),
    )
    run = asyncio.run(harness.run())

    assert run.verdict_of(S.Stage.C1) is StageVerdict.PASS, [
        (r.stage.value, r.verdict.value, r.reason) for r in run.records
    ]
    assert run.verdict_of(S.Stage.C2) is StageVerdict.PASS
    c1 = run.records[S.STAGE_ORDER.index(S.Stage.C1)]
    assert c1.evidence["observed_tools_in_payload"] == list(binding.tool_names)
    assert c1.evidence["binding_agreement"] is True
    assert c1.evidence["request_digest_covers_tool_binding"] is False


# ===========================================================================
# 6. digest 不变与覆盖缺口记账
# ===========================================================================

#: 冻结的 `request_digest`。工具绑定**不在**它的覆盖范围内。
FROZEN_REQUEST_DIGEST = (
    "35319071e5dd0eb1b2c3d8b9321d7d3613641b0dedf7f8b1c28312d43f6f3d35"
)


def test_27_the_request_digest_is_unchanged_by_the_tool_binding():
    cfg = cal_config.default_calibration_config()
    assert cfg.request_shape_digest() == FROZEN_REQUEST_DIGEST
    binding = S.production_tool_binding()
    for candidate in (None, binding, S.ToolBinding(tools=(), source="empty")):
        request = _request(binding=candidate)
        # 请求对象携带绑定,但 digest 只覆盖 `CalibrationConfig.request_shape()`。
        assert request.tool_binding is candidate
        assert cfg.request_shape_digest() == FROZEN_REQUEST_DIGEST


def test_28_the_digest_coverage_gap_is_recorded_not_silently_absent():
    """`request_digest` **不**覆盖工具绑定 —— 这是已知缺口,必须被记账。"""
    cfg = cal_config.default_calibration_config()
    shape = cfg.request_shape()
    assert len(shape) == 11
    assert "tools" not in shape
    assert S.REQUEST_DIGEST_COVERS_TOOL_BINDING is False

    binding = S.production_tool_binding()
    evidence = _evidence(payload=_wire_payload_for(binding), binding=binding)
    assert "request_digest_covers_tool_binding" in evidence
    assert evidence["request_digest_covers_tool_binding"] is False
    assert evidence["request_digest"] == FROZEN_REQUEST_DIGEST


def test_29_the_binding_is_recorded_alongside_the_gap():
    """绑定本身必须进证据(否则缺口就成了"什么都没记")。"""
    binding = S.production_tool_binding()
    evidence = _evidence(payload=_wire_payload_for(binding), binding=binding)
    for key in (
        "tool_binding_declared",
        "tool_binding_source",
        "tool_binding_size",
        "tools_field_present_in_payload",
        "observed_tools_in_payload",
        "binding_agreement",
        "contract_agreement",
        "supplied_contract_agreement",
        "request_digest_covers_tool_binding",
    ):
        assert key in evidence, key
    assert evidence["tool_binding_size"] == 4
    assert evidence["tool_binding_source"] == "app.tools.DEFAULT_TOOLS"


# ===========================================================================
# 7. 凭据安全
# ===========================================================================


def test_30_c1_evidence_carries_no_credential_material(monkeypatch):
    """证据里不得出现凭据,也不得出现完整 base_url。"""
    _arm_tripwires(monkeypatch)
    binding = S.production_tool_binding()
    capture = _Capture()
    model, _transport, client = _seamed_model(capture)
    bound = S.bind_declared_tools(model, _request(binding=binding))
    try:
        asyncio.run(bound.ainvoke(_c1_messages()))
    finally:
        asyncio.run(client.aclose())
    body = capture.requests[0]["body"]
    evidence = _evidence(payload=body, binding=binding)

    blob = json.dumps(evidence, ensure_ascii=False, default=str)
    assert SYNTHETIC_API_KEY not in blob
    assert UNRESOLVABLE_BASE_URL not in blob
    assert find_secret_patterns(evidence) == []
    lowered = {key.lower() for key in evidence}
    assert not any("key" in key or "authorization" in key for key in lowered), lowered


# ===========================================================================
# 8. 已知覆盖边界(**不是**"已覆盖")
# ===========================================================================


def test_31_the_synthetic_message_cannot_exercise_invalid_tool_calls():
    """合成消息的 `invalid_tool_calls` 恒为空 —— 这是一条**覆盖边界**。

    `AIMessage.invalid_tool_calls` 由 LangChain 从 `tool_calls` 的**解析过程**
    派生,**不读** `additional_kwargs["invalid_tool_calls"]`。因此
    `INVALID_TOOL_CALLS_PRESENT` 这条缺陷只能由**真实 SDK 消息**触发 ——
    本文件用合成消息覆盖不到它。

    该分支由 D-2c 聚焦套件的 `tool_call_findings` 直调用例覆盖(直接传
    `invalid_tool_calls=[...]`)。把它记在这里,是为了避免"F10 覆盖了 C1
    全部缺陷分支"被读成一个已被证明的结论。
    """
    message = synthetic_ai_message(
        content="",
        tool_calls=[dict(LEGAL_CALL)],
        invalid_tool_calls=[{"name": "analyze_risk_tool", "error": "parse error"}],
        finish_reason="tool_calls",
    )
    assert message.additional_kwargs["invalid_tool_calls"]
    assert S.invalid_tool_calls_of(message) == [], (
        "合成消息的形状变了 —— 若它开始能带 invalid_tool_calls,本测试要改成"
        "断言该分支被覆盖,而不是断言覆盖不到"
    )
