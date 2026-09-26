"""Phase 9.2-D-2c F8 —— 构造期 HTTP 客户端注入缝(最终请求体可观测性)。

F8 是什么
---------
已提交的标定路径能**声明** `extra_body["max_tokens"] = 16`,也能读到 provider
的响应证据,却无法核验**最终发往 provider 的请求体**。C0b 的判定因此卡在
"上限到底有没有发出去"这个前提上,只能得到 INCONCLUSIVE。

本缝(`build_chat_model(..., http_async_client=...)`)提供构造期注入点:
调用方传入一个自带记录钩子的真实 `httpx.AsyncClient` 即可观测最终序列化
请求体;传入 `httpx.MockTransport` 客户端即可做完全离线的形状验证。

为什么必须是构造期 —— 本文件把它钉住
------------------------------------
`langchain_openai` 的 `BaseChatOpenAI` 在**构造期**就把
`root_client` / `client` / `root_async_client` / `async_client` 全部物化,
并在这一刻读取 `self.http_async_client`。构造之后再赋值会被存进实例,却
**永远不会被读取** —— 拦截器"挂上了"而请求真的发出去,仪器什么也看不到。
`test_03` 就是防这一类事故重新引入的回归。

本文件全程离线:三层结构性证据,**不依赖** F7 已知失明的 egress observer
------------------------------------------------------------------------
F7 已确认:Windows 异步路径上,审计钩子的 egress observer 会漏报真实出口。
因此本文件**不**用 `egress_events == 0` 当证明,而是用结构:

    1. 传输层**就是** `httpx.MockTransport` 本体(身份比较,不是"看起来像");
    2. 四处 tripwire 把 DNS 与两个真实 httpx transport 换成"调用即抛";
    3. base_url 用 RFC 2606 保留的 `.invalid` 主机 —— 它**永远无法解析**,
       请求却仍然成功返回,所以不可能发生过 DNS 或 TCP。

零 provider 调用、零 API 额度消耗、零真实凭据。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import socket
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.evaluation.calibration import config as cal_config
from app.evaluation.calibration import stages as S
from app.evaluation.calibration.synthetic import (
    deepseek_shaped_raw_usage,
    synthetic_ai_message,
)
from app.evaluation.calibration.verdict import StageVerdict
from app.evaluation.llm.raw import find_secret_patterns
from app.evaluation.real_provider import build_chat_model, make_llm_factory

#: 明显是合成物的凭据占位符。**不是**任何真实密钥。
SYNTHETIC_API_KEY = "sk-d2c-f8-synthetic-placeholder-00000000"

#: RFC 2606 保留的顶级域,永远无法解析。用它做 base_url 是刻意的:
#: "用不可解析的主机却拿到了响应"本身就是"没有走网络"的结构性证据。
UNRESOLVABLE_HOST = "c0b-f8-probe.invalid"
UNRESOLVABLE_BASE_URL = f"https://{UNRESOLVABLE_HOST}/v1"

TRIPWIRE_MARKER = "FAIL-CLOSED TRIPWIRE"

#: 本地合成响应(OpenAI-compatible 形状)。**不来自任何 provider。**
CANNED_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-f8-synthetic",
    "object": "chat.completion",
    "created": 0,
    "model": "DeepSeek-V4.1-Flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "SYNTHETIC-LOCAL"},
            "finish_reason": "length",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 16, "total_tokens": 17},
}

#: 请求头里**一律不得进入产物**的字段名。
_DROPPED_HEADERS = frozenset({"authorization", "api-key", "x-api-key", "proxy-authorization"})


# ---------------------------------------------------------------------------
# 工装
# ---------------------------------------------------------------------------


class _Capture:
    """MockTransport 的请求记录器。

    **不记录任何凭据材料**:`Authorization` / `*-api-key` 一律丢弃,只留一个
    布尔值证明"线上确实带过它"—— 这样"丢弃"才不是一句空话。
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        try:
            body: Any = json.loads(request.content.decode("utf-8"))
        except Exception:  # noqa: BLE001 - 解析失败不该让测试崩
            body = {"__unparsed__": True}
        header_names = list(request.headers.keys())
        self.requests.append(
            {
                "method": request.method,
                "host": request.url.host,
                "path": request.url.path,
                "body": body,
                "headers": {
                    name: value
                    for name, value in request.headers.items()
                    if name.lower() not in _DROPPED_HEADERS
                },
                # 布尔值,不是值本身 —— 用来证明上面的丢弃不是恒真。
                "auth_header_present_on_wire": any(
                    name.lower() == "authorization" for name in header_names
                ),
            }
        )
        return httpx.Response(200, json=CANNED_RESPONSE)


def _arm_tripwires(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """把 DNS 与两个真实 httpx transport 换成"调用即抛"。

    `socket.socket` **刻意不动** —— 替换 socket 工厂会破坏 asyncio 事件循环
    (Windows 的 `socketpair()` 依赖它),于是"守卫"会变成"测试跑不起来"。
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


def _c0b_shaped_kwargs(*, cap: int | None = None) -> dict[str, Any]:
    """C0b 形状的 `build_chat_model` kwargs(**与提交路径同源**)。

    唯一改动是 base_url 指向不可解析主机 —— 它不进请求体,因此不影响形状。
    """
    cfg = cal_config.default_calibration_config()
    kwargs = dict(
        cfg.model_kwargs(output_token_cap=cal_config.CALIBRATION_C0B_TEMPORARY_CAP)
    )
    kwargs["base_url"] = UNRESOLVABLE_BASE_URL
    return kwargs


def _c0b_messages() -> list[Any]:
    """与 `stages.run_c0b` 逐字一致的探针消息。"""
    demanded = cal_config.CALIBRATION_C0B_DEMANDED_OUTPUT_TOKENS
    return [
        SystemMessage(
            content="D-2c 标定:上限探针。请**完整**列出一个长清单,不要提前收尾。"
        ),
        HumanMessage(
            content=(
                f"请输出至少 {demanded} 个 token 的内容:"
                "逐条列出你能想到的安全事件类型,每条一行。"
            )
        ),
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
        **(kwargs if kwargs is not None else _c0b_shaped_kwargs()),
    )
    return model, transport, client


# ---------------------------------------------------------------------------
# 缝本身
# ---------------------------------------------------------------------------


def test_00_seam_exists_on_both_constructors_and_defaults_to_none():
    """缝必须存在,且默认 `None` —— 未提供时不得改变任何既有行为。"""
    for fn in (build_chat_model, make_llm_factory):
        params = inspect.signature(fn).parameters
        assert "http_async_client" in params, f"{fn.__name__} 缺少构造期注入缝"
        assert params["http_async_client"].default is None, (
            f"{fn.__name__} 的缝默认值必须是 None,否则既有调用方行为会变"
        )


def test_01_seam_absent_does_not_inject_the_kwarg(monkeypatch):
    """默认 `None` ⇒ 该 kwarg **完全不传**给 SDK(与 D-2b/D-2c 逐字段一致)。"""
    import langchain_openai

    seen: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Recorder)
    cfg = cal_config.default_calibration_config()
    build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        allow_network=True,
        **cfg.model_kwargs(),
    )
    assert "http_async_client" not in seen


def test_02_seam_present_passes_the_exact_object(monkeypatch):
    """提供了就**原样**传下去 —— 身份相同,不是拷贝、不是包装。"""
    import langchain_openai

    seen: dict[str, Any] = {}
    sentinel = httpx.MockTransport(lambda _r: httpx.Response(200, json=CANNED_RESPONSE))

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Recorder)
    cfg = cal_config.default_calibration_config()
    build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        http_async_client=sentinel,
        allow_network=True,
        **cfg.model_kwargs(),
    )
    assert seen["http_async_client"] is sentinel


# ---------------------------------------------------------------------------
# TEST A —— 最终序列化请求体
# ---------------------------------------------------------------------------


def test_03_final_serialized_body_matches_frozen_c0b_semantics(monkeypatch):
    """TEST A:断言的是**最终序列化的请求体**,不是配置声明。"""
    fired = _arm_tripwires(monkeypatch)
    capture = _Capture()
    model, _transport, client = _seamed_model(capture)
    try:
        message = asyncio.run(model.ainvoke(_c0b_messages()))
    finally:
        asyncio.run(client.aclose())

    assert len(capture.requests) == 1, "恰好一次请求"
    request = capture.requests[0]
    assert request["method"] == "POST"
    assert request["host"] == UNRESOLVABLE_HOST, (
        "请求打到了不可解析主机却仍然成功 —— 只有本地 mock 能解释这一点"
    )

    body = request["body"]
    assert body["model"] == cal_config.CALIBRATION_REQUESTED_MODEL
    assert body["model"] == "deepseek-flash"
    assert body["stream"] is False
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == cal_config.CALIBRATION_C0B_TEMPORARY_CAP
    assert body["max_tokens"] == 16
    assert "temperature" not in body
    assert "max_completion_tokens" not in body

    assert fired == [], f"tripwire 被触发 —— 测试逃出了本地:{fired}"
    assert message.content == "SYNTHETIC-LOCAL"


# ---------------------------------------------------------------------------
# TEST B —— fail-closed 结构证明
# ---------------------------------------------------------------------------


def test_04_tripwires_have_teeth(monkeypatch):
    """护栏必须**有牙**:直接调用被替换的原语必须立刻炸掉。

    一个恒不触发的 tripwire 与没有 tripwire 是一回事 —— 甚至更糟。
    """
    fired = _arm_tripwires(monkeypatch)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        httpx.AsyncHTTPTransport.handle_async_request(None, None)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        httpx.HTTPTransport.handle_request(None)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(RuntimeError, match=TRIPWIRE_MARKER):
        socket.gethostbyname("example.com")
    assert len(fired) == 4, "四处 tripwire 都必须真的接过调用"


def test_05_mock_transport_is_the_effective_transport(monkeypatch):
    """结构证明:生效的异步传输层**就是** MockTransport 本体(身份比较)。"""
    _arm_tripwires(monkeypatch)
    capture = _Capture()
    model, transport, client = _seamed_model(capture)
    try:
        effective = model.root_async_client._client._transport
        assert effective is transport
        assert not isinstance(effective, httpx.AsyncHTTPTransport), (
            "真实 transport 不得出现在生效路径上 —— 否则存在真实出口"
        )
    finally:
        asyncio.run(client.aclose())


def test_06_request_succeeds_though_the_host_cannot_resolve(monkeypatch):
    """用不可解析主机却拿到响应 ⇒ 没有 DNS、没有 TCP。"""
    fired = _arm_tripwires(monkeypatch)
    capture = _Capture()
    model, _transport, client = _seamed_model(capture)
    try:
        message = asyncio.run(model.ainvoke(_c0b_messages()))
    finally:
        asyncio.run(client.aclose())
    assert message.content == "SYNTHETIC-LOCAL"
    assert capture.requests[0]["host"] == UNRESOLVABLE_HOST
    assert fired == []


# ---------------------------------------------------------------------------
# TEST C —— 默认行为保持
# ---------------------------------------------------------------------------


def test_07_default_behaviour_preserved_without_the_seam(monkeypatch):
    """不提供缝的调用方,构造行为与请求形状都必须不变。"""
    _arm_tripwires(monkeypatch)
    cfg = cal_config.default_calibration_config()
    kwargs = _c0b_shaped_kwargs()
    plain = build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        allow_network=True,
        **kwargs,
    )

    assert plain.http_async_client is None, "未提供缝时该字段必须仍是 None"
    effective = plain.root_async_client._client._transport
    assert isinstance(effective, httpx.AsyncHTTPTransport), (
        "未提供缝时应保持 SDK 默认传输层 —— 本测试只做内省,**不调用**它"
    )

    capture = _Capture()
    seamed, _transport, client = _seamed_model(capture, kwargs=kwargs)
    try:
        assert plain._get_request_payload(_c0b_messages()) == seamed._get_request_payload(
            _c0b_messages()
        ), "缝不得改变请求形状"
    finally:
        asyncio.run(client.aclose())


# ---------------------------------------------------------------------------
# TEST D —— 构造期要求(事故类回归)
# ---------------------------------------------------------------------------


def test_08_post_construction_assignment_is_not_honoured(monkeypatch):
    """回归:构造后赋值 `http_async_client` **不会**生效。

    这正是先前事故的机制:拦截器"挂上了"、请求却真的发出去,而仪器什么也
    看不到。本测试把该陷阱钉死 —— 受支持的机制**只有**构造期注入。
    """
    _arm_tripwires(monkeypatch)
    cfg = cal_config.default_calibration_config()
    plain = build_chat_model(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        allow_network=True,
        **_c0b_shaped_kwargs(),
    )

    late_transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, json=CANNED_RESPONSE)
    )
    late_client = httpx.AsyncClient(transport=late_transport)
    try:
        plain.http_async_client = late_client

        assert plain.http_async_client is late_client, (
            "值确实被存进了实例 —— 这正是陷阱所在,值得显式记录"
        )
        effective = plain.root_async_client._client._transport
        assert effective is not late_transport, (
            "构造后赋值不得生效:SDK 客户端在构造期已物化,不会再读该字段"
        )
        assert not isinstance(effective, httpx.MockTransport)
    finally:
        asyncio.run(late_client.aclose())


# ---------------------------------------------------------------------------
# make_llm_factory 转发
# ---------------------------------------------------------------------------


def test_09_make_llm_factory_forwards_the_seam(monkeypatch):
    """工厂必须把缝一起转发,否则它就成了第二个 F8 缺口。"""
    _arm_tripwires(monkeypatch)
    capture = _Capture()
    transport = httpx.MockTransport(capture.handler)
    client = httpx.AsyncClient(transport=transport)
    cfg = cal_config.default_calibration_config()
    factory = make_llm_factory(
        candidate=cfg.provider_candidate(),
        api_key=SYNTHETIC_API_KEY,
        http_async_client=client,
        allow_network=True,
        **_c0b_shaped_kwargs(),
    )
    try:
        model = factory(behavior="GOOD", task=None, dataset_paths={})
        assert model.root_async_client._client._transport is transport
    finally:
        asyncio.run(client.aclose())


# ---------------------------------------------------------------------------
# TEST E —— 凭据安全
# ---------------------------------------------------------------------------


def test_10_capture_persists_no_credential_material(monkeypatch):
    """记录器必须丢掉凭据材料,且该丢弃**不是恒真**。"""
    _arm_tripwires(monkeypatch)
    capture = _Capture()
    model, _transport, client = _seamed_model(capture)
    try:
        asyncio.run(model.ainvoke(_c0b_messages()))
    finally:
        asyncio.run(client.aclose())

    request = capture.requests[0]
    assert request["auth_header_present_on_wire"] is True, (
        "线上确实带过 Authorization —— 否则'丢弃'是恒真的,证明不了任何事"
    )
    header_names = {name.lower() for name in request["headers"]}
    assert "authorization" not in header_names
    assert not any("key" in name for name in header_names)

    blob = json.dumps(capture.requests, ensure_ascii=False, default=str)
    assert SYNTHETIC_API_KEY not in blob
    assert find_secret_patterns(capture.requests) == []


# ---------------------------------------------------------------------------
# TEST F —— 既有 C0b 判定语义回归
# ---------------------------------------------------------------------------


def test_11_c0b_unobservable_payload_still_inconclusive():
    """冻结语义:请求体不可观测 ⇒ **即使长度证据完美**,仍为 INCONCLUSIVE。

    这是 C0b 原始结果(INCONCLUSIVE)的语义护栏。本缝只让"未来能观测",
    **不**改变判定;一旦这条断言变红,说明有人把前提检查拆掉了。
    """
    cfg = cal_config.default_calibration_config()
    cap = cal_config.CALIBRATION_C0B_TEMPORARY_CAP
    demanded = cal_config.CALIBRATION_C0B_DEMANDED_OUTPUT_TOKENS

    request = S.CalibrationRequest(
        stage=S.Stage.C0B,
        messages=tuple(_c0b_messages()),
        model_kwargs=cfg.model_kwargs(output_token_cap=cap),
        output_token_cap=cap,
        demanded_output_tokens=demanded,
    )
    # 长度证据"完美":输出恰好等于上限、finish_reason=length、用量可得。
    result = S.InvocationResult(
        message=synthetic_ai_message(
            content="x" * 64,
            finish_reason="length",
            raw_usage=deepseek_shaped_raw_usage(
                prompt_tokens=55, completion_tokens=cap
            ),
            normalized_usage={
                "input_tokens": 55,
                "output_tokens": cap,
                "total_tokens": 55 + cap,
            },
        ),
        request_payload=None,
        http_attempts=None,
    )
    evidence = S.c0b_evidence(
        request=request,
        result=result,
        provider_rejected=False,
        provider_rejected_reason=None,
        config=cfg,
    )
    assert evidence["request_payload_available"] is False
    assert evidence["observed_cap_in_payload"] is None
    assert evidence["observed_output_tokens"] == cap, "长度证据本身是完美的"

    verdict = S.c0b_evidence_verdict(evidence)
    assert verdict.verdict is StageVerdict.INCONCLUSIVE, (
        "请求体不可观测时,长度证据不得单独支撑结论 —— 这是冻结语义"
    )
    assert verdict.verdict.blocks_progression is True
