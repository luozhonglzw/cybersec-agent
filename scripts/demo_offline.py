#!/usr/bin/env python
"""离线可复现 Demo —— 四个场景，零网络、零真实 provider、零付费模型。

这个脚本**不实现任何新能力**。它只是把仓库里已经存在的四条机制,用真实代码
跑一遍并把结果打印出来:

    DEMO 1  RBAC                真实 FastAPI 应用 + 真实 AuthKeyring → 401 / 403
    DEMO 2  Citation validation 真实语料 + 真实检索 + 真实校验器 → grounded 判定
    DEMO 3  Retrieval fail-closed  真实 Qdrant Local 装配路径 → unavailable,不回退
    DEMO 4  HITL               真实 pytest 子进程 → 暂停 / 恢复 / 线程隔离 / 归属约束

三条纪律
--------

**1. 每个 PASS 都必须来自真实执行的断言。**
没有任何一处"打印预设结果"。每个场景的每一条 `Check` 都由上一步的实际返回
值计算得出;`Check.ok` 为假即整个场景 FAIL,进程退出码非零。

**2. 脚本化的东西必须显式标注。**
DEMO 2 的**回答文本**是脚本化的(我们没有真实 provider),但检索、信封、
证据收集、引用校验全部是真实代码。DEMO 4 是**测试驱动**的演示,不是在线
业务请求。两者都在输出里写明 `MODE`。

**3. 不产生任何外部副作用。**
不使用真实凭据、不读 `.env`、不写仓库 `data/`(审计库与 checkpoint 都落在
临时目录,结束时删除)、不发起网络请求。占位的 `LLM_BASE_URL` 指向
`127.0.0.1:9`(不可路由),因此"如果真有模型调用,它会立刻失败"——这也让
"DEMO 1 只可能返回 401/403,不可能返回 502"成为可检验的陈述。

用法::

    uv run python scripts/demo_offline.py

退出码:全部场景 PASS → 0;任一场景 FAIL → 1。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: 让 `python scripts/demo_offline.py` 在没有 editable 安装时也能 import app。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging  # noqa: E402

import structlog  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from app.api.main import build_rag_ask_service, create_app  # noqa: E402
from app.core.agent import SecurityAgent  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.rag_ask import GroundedAskService, ask_grounded  # noqa: E402
from app.rag import (  # noqa: E402
    HashingEmbedder,
    IndexUnavailable,
    build_in_memory_retriever,
    chunk_corpus,
    load_corpus,
)
from app.rag.citations import assess_answer_citations, collect_evidence  # noqa: E402
from app.security.auth import AuthKeyring  # noqa: E402
from app.tools.search_knowledge import build_search_knowledge_tool  # noqa: E402
from app.vectorstore import build_qdrant_retriever  # noqa: E402

# ---------------------------------------------------------------------------
# Demo 专用占位凭据 —— 明确不是真实凭据,不得复用
# ---------------------------------------------------------------------------

#: 名字即用途:这是**演示用**明文,只为让真实密钥环能解析出两个身份。
#: 它们不指向任何真实系统,也不是任何服务的口令。
DEMO_ANALYST_KEY = "demo-offline-analyst-key-not-a-real-credential"
DEMO_VIEWER_KEY = "demo-offline-viewer-key-not-a-real-credential"
DEMO_UNKNOWN_KEY = "demo-offline-unknown-key-not-a-real-credential"

DEMO_ANALYST_SUBJECT = "demo-analyst"
DEMO_VIEWER_SUBJECT = "demo-viewer"

#: 不可路由的占位 provider 地址。任何真实模型调用都会在这里立刻失败,
#: 因此"没有发起网络请求"这一点是可以从结果反推的。
DEMO_PLACEHOLDER_BASE_URL = "http://127.0.0.1:9/offline-demo-no-provider"

#: 本脚本会改动的环境变量。运行结束时按快照恢复。
_MANAGED_ENV = (
    "AUTH_API_KEYS",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "AUDIT_DB_PATH",
    "CHECKPOINT_DB_PATH",
    "AUDIT_BACKEND",
    "RAG_BACKEND",
    "RAG_QDRANT_PATH",
)

QUESTION = "Log4j 的 JNDI 查找为什么能被用来执行代码?"

#: 一个**故意不存在**的 citation_id,用来演示编造引用。
FORGED_CITATION_ID = "kb-cve-2099-00001"


# ---------------------------------------------------------------------------
# 结果模型与打印
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """一条由真实返回值计算出的断言。"""

    label: str
    ok: bool
    observed: str


@dataclass
class ScenarioResult:
    key: str
    title: str
    mode: str
    inputs: tuple[str, ...]
    evidence: tuple[str, ...]
    checks: list[Check] = field(default_factory=list)
    notes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.checks) and all(c.ok for c in self.checks)


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _print_scenario(result: ScenarioResult) -> None:
    print()
    print(_rule("="))
    print(f"{result.key} — {result.title}")
    print(_rule("="))
    print(f"MODE     : {result.mode}")
    for index, line in enumerate(result.inputs):
        prefix = "INPUT    :" if index == 0 else "           "
        print(f"{prefix} {line}")
    for index, line in enumerate(result.evidence):
        prefix = "EVIDENCE :" if index == 0 else "           "
        print(f"{prefix} {line}")

    print("CHECKS   :")
    for check in result.checks:
        mark = "ok  " if check.ok else "FAIL"
        print(f"  [{mark}] {check.label}")
        print(f"         -> {check.observed}")

    for note in result.notes:
        print(f"NOTE     : {note}")

    if result.error is not None:
        print(f"ERROR    : {result.error}")

    print(f"RESULT   : {'PASS' if result.ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# 环境准备 / 恢复
# ---------------------------------------------------------------------------

def _snapshot_env() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _MANAGED_ENV}


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    get_settings.cache_clear()


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _apply_offline_env(workdir: Path) -> None:
    """装上**演示专用**的配置:真实密钥环结构 + 占位 provider + 临时存储路径。"""
    os.environ["AUTH_API_KEYS"] = json.dumps(
        [
            {
                "sha256": _digest(DEMO_ANALYST_KEY),
                "subject": DEMO_ANALYST_SUBJECT,
                "role": "analyst",
            },
            {
                "sha256": _digest(DEMO_VIEWER_KEY),
                "subject": DEMO_VIEWER_SUBJECT,
                "role": "viewer",
            },
        ]
    )
    os.environ["LLM_MODEL"] = "offline-demo-placeholder-model"
    os.environ["LLM_API_KEY"] = "offline-demo-placeholder-key-not-a-real-credential"
    os.environ["LLM_BASE_URL"] = DEMO_PLACEHOLDER_BASE_URL
    os.environ["AUDIT_BACKEND"] = "sqlite"
    os.environ["AUDIT_DB_PATH"] = str(workdir / "audit.db")
    os.environ["CHECKPOINT_DB_PATH"] = str(workdir / "checkpoints.db")
    os.environ.pop("RAG_BACKEND", None)  # 用默认值 memory
    os.environ.pop("RAG_QDRANT_PATH", None)
    get_settings.cache_clear()


class _ScriptedLLM:
    """按脚本重放工具调用与最终回答。

    **它不是模型。** 它不选工具、不理解问题,只按顺序吐出预先写好的
    `AIMessage`。存在的唯一理由:在没有 provider 的环境里,让真实的图
    (agent → tools → agent)跑起来,从而让**真实工具**执行**真实检索**,
    产出**真实 ToolMessage**。证据链是真的,只有"措辞"是脚本的。
    """

    def __init__(
        self,
        *,
        answer: str,
        tool_call_batches: tuple[list[dict[str, Any]], ...] = (),
    ) -> None:
        self.answer = answer
        self._batches = [list(batch) for batch in tool_call_batches]
        self.bound_tools: list[Any] = []
        self.seen_messages: list[list[Any]] = []

    def bind_tools(self, tools: Any) -> "_ScriptedLLM":
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.seen_messages.append(list(messages))
        if self._batches:
            return AIMessage(content="", tool_calls=self._batches.pop(0))
        return AIMessage(content=self.answer)


def _tool_call(query: str, call_id: str = "demo-call-1") -> list[dict[str, Any]]:
    return [{"name": "search_knowledge_tool", "args": {"query": query}, "id": call_id}]


def _last_tool_message_content(llm: _ScriptedLLM) -> str | None:
    """取出图里**真实执行**过的工具消息内容(不是模型文本)。"""
    for batch in reversed(llm.seen_messages):
        for message in batch:
            if getattr(message, "type", None) == "tool":
                content = getattr(message, "content", None)
                return content if isinstance(content, str) else None
    return None


# ===========================================================================
# DEMO 1 — RBAC
# ===========================================================================

def demo_1_rbac() -> ScenarioResult:
    """真实 FastAPI 应用 + 真实密钥环:未认证 401、角色不足 403。"""
    from fastapi.testclient import TestClient

    result = ScenarioResult(
        key="DEMO 1",
        title="RBAC — authentication and role admission",
        mode="REAL API BEHAVIOUR (in-process TestClient; real lifespan, real AuthKeyring)",
        inputs=(
            'POST /ask  {"question": "..."}',
            "request A: no Authorization header",
            "request B: Authorization: Bearer <demo viewer key>",
            "request C: Authorization: Bearer <unknown key>",
        ),
        evidence=(
            "app/api/main.py :: lifespan -> app.state.auth_keyring = AuthKeyring.from_entries(...)",
            "app/security/auth.py :: require_principal / require_analyst (_unauthorized / _forbidden)",
            "tests/test_api/test_object_authorization.py :: test_missing_credentials_are_401_on_every_business_route",
        ),
    )

    # create_app() 不带参数 ⇒ lifespan 真的跑起来 ⇒ 密钥环来自 AUTH_API_KEYS。
    app = create_app()
    with TestClient(app) as client:
        keyring_ok = isinstance(getattr(app.state, "auth_keyring", None), AuthKeyring)
        result.checks.append(
            Check(
                "lifespan assembled a real AuthKeyring",
                keyring_ok,
                f"app.state.auth_keyring is AuthKeyring = {keyring_ok}",
            )
        )

        anonymous = client.post("/ask", json={"question": QUESTION})
        result.checks.append(
            Check(
                "no credentials -> HTTP 401",
                anonymous.status_code == 401,
                f"HTTP {anonymous.status_code} {anonymous.text}",
            )
        )
        result.checks.append(
            Check(
                "401 carries the fixed detail and a Bearer challenge",
                anonymous.json().get("detail") == "authentication required"
                and anonymous.headers.get("www-authenticate") == "Bearer",
                f"detail={anonymous.json().get('detail')!r} "
                f"www-authenticate={anonymous.headers.get('www-authenticate')!r}",
            )
        )

        viewer = client.post(
            "/ask",
            json={"question": QUESTION},
            headers={"Authorization": f"Bearer {DEMO_VIEWER_KEY}"},
        )
        result.checks.append(
            Check(
                "authenticated viewer -> HTTP 403 (role admission)",
                viewer.status_code == 403,
                f"HTTP {viewer.status_code} {viewer.text}",
            )
        )

        unknown = client.post(
            "/ask",
            json={"question": QUESTION},
            headers={"Authorization": f"Bearer {DEMO_UNKNOWN_KEY}"},
        )
        result.checks.append(
            Check(
                "unknown key is indistinguishable from no key (no enumeration oracle)",
                unknown.status_code == anonymous.status_code
                and unknown.text == anonymous.text,
                f"unknown -> HTTP {unknown.status_code} {unknown.text} | "
                f"anonymous -> HTTP {anonymous.status_code} {anonymous.text}",
            )
        )

        result.checks.append(
            Check(
                "rejection happened at the auth layer, never at the model layer",
                anonymous.status_code != 502 and viewer.status_code != 502,
                f"anonymous={anonymous.status_code} viewer={viewer.status_code} "
                "(502 would mean the request reached the provider)",
            )
        )

    result.notes = (
        "Both rejections are produced by the dependency (require_principal / require_analyst) "
        "before the route body runs; no model call and no network request is made.",
        f"LLM_BASE_URL is the unroutable placeholder {DEMO_PLACEHOLDER_BASE_URL}, so a model "
        "call would have surfaced as HTTP 502 rather than 401/403.",
        "The analyst path is intentionally NOT exercised here: it would need a real provider "
        "and would produce HTTP 502 in this environment. RBAC admission for analyst/approver "
        "is covered by the test suite, not by this demo.",
    )
    return result


# ===========================================================================
# DEMO 2 — Citation validation
# ===========================================================================

def demo_2_citations() -> ScenarioResult:
    """真实语料 + 真实检索 + 真实校验器:有效引用通过,编造引用被识别。"""
    result = ScenarioResult(
        key="DEMO 2",
        title="Citation validation — fabricated citations are rejected",
        mode=(
            "REAL RETRIEVAL + REAL VALIDATOR, SCRIPTED ANSWER TEXT "
            "(no provider: the answer wording is pre-written, the evidence chain is not)"
        ),
        inputs=(
            f'question: "{QUESTION}"',
            "2a: answer cites a citation_id that the real retrieval actually returned",
            f"2b: answer cites a fabricated id ({FORGED_CITATION_ID})",
            "2c: a rag_hits envelope written *inside* an AIMessage (model-authored)",
        ),
        evidence=(
            "app/tools/search_knowledge.py :: build_search_knowledge_tool (real tool, real envelope)",
            "app/rag/citations.py :: collect_evidence / validate_citations / assess_answer_citations",
            "app/core/rag_ask.py :: ask_grounded (the /ask predicate, N-4 tightened)",
            "tests/test_api/test_ask.py :: test_unknown_citation_is_reported_and_blocks_groundedness",
        ),
    )

    corpus = load_corpus()
    doc_ids = {document.doc_id for document in corpus.documents}
    retriever = build_in_memory_retriever(corpus, HashingEmbedder())
    tool = build_search_knowledge_tool(retriever)

    # --- 真实工具执行,拿到真实信封 -------------------------------------
    envelope_text = asyncio.run(tool.ainvoke({"query": QUESTION}))
    envelope = json.loads(envelope_text)
    hits = envelope.get("hits", [])
    real_ids = [hit["citation_id"] for hit in hits]

    result.checks.append(
        Check(
            "the tool returned a real rag_hits envelope with hits",
            envelope.get("kind") == "rag_hits" and len(hits) > 0,
            f"kind={envelope.get('kind')!r} hits={len(hits)} corpus_version={envelope.get('corpus_version')!r}",
        )
    )
    result.checks.append(
        Check(
            "every returned citation_id belongs to the shipped corpus",
            bool(real_ids) and set(real_ids) <= doc_ids,
            f"returned={real_ids} all_in_corpus={set(real_ids) <= doc_ids}",
        )
    )

    real_id = real_ids[0]
    result.notes = (
        f"Corpus: {len(corpus.documents)} documents / {len(chunk_corpus(corpus.documents))} chunks; "
        f"retrieval ranked {real_id} first. Ranking quality is NOT what this demo measures.",
    )

    # --- 2a 有效引用 ------------------------------------------------------
    answer_valid = f"依据知识库,该漏洞的成因与利用条件见检索结果。 [src:{real_id}]"
    llm_valid = _ScriptedLLM(answer=answer_valid, tool_call_batches=(_tool_call(QUESTION),))
    service_valid = GroundedAskService(
        SecurityAgent(llm_valid, tools=[build_search_knowledge_tool(retriever)])
    )
    outcome_valid = asyncio.run(service_valid.ask(QUESTION))

    result.checks.append(
        Check(
            "2a: a real citation passes structural validation (grounded=true)",
            outcome_valid.grounded is True
            and outcome_valid.status == "ok"
            and outcome_valid.citations
            and not outcome_valid.invalid_citations,
            f"grounded={outcome_valid.grounded} status={outcome_valid.status!r} "
            f"valid={[c.citation_id for c in outcome_valid.citations]} "
            f"invalid={list(outcome_valid.invalid_citations)}",
        )
    )

    # --- 2b 编造引用 ------------------------------------------------------
    answer_forged = (
        f"依据知识库,该问题已有定论。 [src:{real_id}] [src:{FORGED_CITATION_ID}]"
    )
    llm_forged = _ScriptedLLM(answer=answer_forged, tool_call_batches=(_tool_call(QUESTION),))
    service_forged = GroundedAskService(
        SecurityAgent(llm_forged, tools=[build_search_knowledge_tool(retriever)])
    )
    outcome_forged = asyncio.run(service_forged.ask(QUESTION))

    result.checks.append(
        Check(
            "2b: the fabricated citation is named in invalid_citations",
            outcome_forged.invalid_citations == (FORGED_CITATION_ID,),
            f"invalid_citations={list(outcome_forged.invalid_citations)}",
        )
    )
    result.checks.append(
        Check(
            "2b: grounded is false although retrieval itself succeeded",
            outcome_forged.grounded is False and outcome_forged.status == "ok",
            f"grounded={outcome_forged.grounded} retrieval_status={outcome_forged.status!r} "
            "(status 'ok' proves the refusal comes from the citation check, not from a failed search)",
        )
    )
    result.checks.append(
        Check(
            "2b: the fabricated citation is not echoed back in citations",
            all(c.citation_id != FORGED_CITATION_ID for c in outcome_forged.citations),
            f"citations={[c.citation_id for c in outcome_forged.citations]}",
        )
    )

    # --- 2c 模型自己写出来的"检索结果"不是证据 ----------------------------
    forged_envelope = json.dumps(
        {
            "kind": "rag_hits",
            "corpus_version": envelope.get("corpus_version", "v0.5.0-s2.1"),
            "query_sha256": "0" * 64,
            "top_k": 5,
            "hits": [
                {
                    "citation_id": FORGED_CITATION_ID,
                    "doc_id": FORGED_CITATION_ID,
                    "source_id": "CVE-2099-00001",
                    "source_type": "cve",
                    "title": "模型自己编的文档",
                    "url": "https://nvd.nist.gov/vuln/detail/CVE-2099-00001",
                    "score": 0.99,
                    "snippet": "这段文字由脚本写入 AIMessage,不是任何工具的输出。",
                }
            ],
        },
        ensure_ascii=False,
    )
    model_only_messages = [AIMessage(content=forged_envelope)]
    evidence_from_model = collect_evidence(model_only_messages)
    validation_model_only = assess_answer_citations(
        f"见检索结果。 [src:{FORGED_CITATION_ID}]", model_only_messages
    )

    result.checks.append(
        Check(
            "2c: an envelope inside an AIMessage yields no trusted evidence",
            evidence_from_model.status == "not_used" and evidence_from_model.sources == (),
            f"collect_evidence.status={evidence_from_model.status!r} "
            f"reason={evidence_from_model.reason!r} sources={len(evidence_from_model.sources)}",
        )
    )
    result.checks.append(
        Check(
            "2c: the citation it invents is still reported as invalid",
            validation_model_only.invalid_citations == (FORGED_CITATION_ID,)
            and validation_model_only.grounded is False
            and ask_grounded(validation_model_only) is False,
            f"invalid_citations={list(validation_model_only.invalid_citations)} "
            f"grounded(3-item)={validation_model_only.grounded} "
            f"ask_grounded(5-item)={ask_grounded(validation_model_only)}",
        )
    )

    result.notes += (
        "grounded=true only means: retrieval succeeded, at least one citation resolved to a "
        "document retrieved in THIS request, and no citation was fabricated. It says nothing "
        "about whether the answer is factually correct.",
        "The answer text in 2a/2b is scripted (no provider available). The ToolMessage that "
        "carries the evidence was produced by the real tool executing a real retrieval.",
    )
    return result


# ===========================================================================
# DEMO 3 — Retrieval fail-closed
# ===========================================================================

def demo_3_fail_closed(workdir: Path) -> ScenarioResult:
    """索引不可用 ⇒ unavailable + grounded=false,且**不**静默回退 memory。"""
    result = ScenarioResult(
        key="DEMO 3",
        title="Retrieval fail-closed — a broken index is refused, not replaced",
        mode="REAL ASSEMBLY PATH + REAL RETRIEVER + SCRIPTED MODEL",
        inputs=(
            "RAG_BACKEND=qdrant-local",
            f"RAG_QDRANT_PATH={workdir / 'no-such-index'} (does not exist)",
            f'question: "{QUESTION}"',
        ),
        evidence=(
            "app/vectorstore/qdrant_local.py :: build_qdrant_retriever -> QdrantLocalVectorStore.open",
            "app/api/main.py :: build_rag_ask_service (IndexUnavailable -> build_stub_knowledge_tool)",
            "app/tools/search_knowledge.py :: build_stub_knowledge_tool (rag_error envelope)",
            "tests/test_vectorstore/test_qdrant_local.py :: test_missing_index_directory_is_unavailable",
        ),
    )

    corpus = load_corpus()
    missing_index = workdir / "no-such-index"

    # --- 直接打开缺失索引:必须抛 IndexUnavailable ------------------------
    raised: IndexUnavailable | None = None
    try:
        build_qdrant_retriever(corpus, HashingEmbedder(), missing_index)
    except IndexUnavailable as exc:
        raised = exc

    result.checks.append(
        Check(
            "opening a missing index raises IndexUnavailable (fail closed)",
            raised is not None,
            f"IndexUnavailable reason={getattr(raised, 'reason', None)!r} "
            f"message={str(raised) if raised else None!r}",
        )
    )
    reason = getattr(raised, "reason", "unknown")

    # --- 走组合根:装配成功,但查询时 unavailable -------------------------
    settings = Settings(rag_backend="qdrant-local", rag_qdrant_path=missing_index)
    llm = _ScriptedLLM(answer="这里不会走到回答阶段。", tool_call_batches=(_tool_call(QUESTION),))
    service = build_rag_ask_service(llm, settings)

    result.checks.append(
        Check(
            "the composition root degrades at query time instead of failing assembly",
            isinstance(service, GroundedAskService),
            f"build_rag_ask_service(...) -> {type(service).__name__} "
            "(None would mean a fixed HTTP 503 for /ask)",
        )
    )
    if service is None:  # pragma: no cover - 仅为让类型收窄
        result.error = "composition root returned None; cannot continue DEMO 3"
        return result

    outcome = asyncio.run(service.ask(QUESTION))

    result.checks.append(
        Check(
            "the answer is reported as unavailable with grounded=false",
            outcome.status == "unavailable"
            and outcome.grounded is False
            and outcome.citations == ()
            and outcome.hit_count == 0,
            f"status={outcome.status!r} grounded={outcome.grounded} "
            f"citations={len(outcome.citations)} hit_count={outcome.hit_count}",
        )
    )

    # 工具消息是**真实执行**的 stub 输出 —— 证明原因码被保留下来
    tool_content = _last_tool_message_content(llm)
    parsed_tool: dict[str, Any] = json.loads(tool_content) if tool_content else {}
    result.checks.append(
        Check(
            "the tool actually returned a rag_error envelope with the original reason",
            parsed_tool.get("kind") == "rag_error" and parsed_tool.get("reason") == reason,
            f"kind={parsed_tool.get('kind')!r} reason={parsed_tool.get('reason')!r} "
            f"(expected {reason!r}); type={parsed_tool.get('type')!r}",
        )
    )

    # --- 对照:同一问题在 memory 后端是有命中的 --------------------------
    memory_retriever = build_in_memory_retriever(corpus, HashingEmbedder())
    memory_result = memory_retriever.retrieve(QUESTION, top_k=5)
    result.checks.append(
        Check(
            "control: the same question does return hits on the memory backend",
            memory_result.status == "ok" and len(memory_result.hits) > 0,
            f"memory status={memory_result.status!r} hits={len(memory_result.hits)} "
            "=> 'unavailable' is a refusal, not an empty corpus",
        )
    )

    result.notes = (
        "No silent fallback: the backend is chosen once in the composition root "
        "(build_rag_ask_service) and a broken qdrant-local index never turns into memory results.",
        "The tool envelope is what carries the reason code to the citation layer; the stub tool "
        "exists precisely so that 'index unavailable' stays distinguishable from 'no hits'.",
    )
    return result


# ===========================================================================
# DEMO 4 — HITL (test-based)
# ===========================================================================

#: 逐条对应控制器要求的四项证据:暂停 / 恢复 / 线程隔离 / 审批人约束。
HITL_NODE_IDS: tuple[tuple[str, str], ...] = (
    (
        "tests/test_core/test_hitl_graph.py::test_requires_approval_pauses_at_human_approval",
        "requires_approval pauses at human_approval (interrupt, no decision yet)",
    ),
    (
        "tests/test_core/test_hitl_graph.py::test_resume_approved_completes",
        "an approved resume completes the run",
    ),
    (
        "tests/test_core/test_hitl_graph.py::test_thread_id_isolation",
        "a decision on one thread_id does not leak into another",
    ),
    (
        "tests/test_api/test_object_authorization.py::test_owner_cannot_self_approve",
        "the owner cannot approve their own thread",
    ),
    (
        "tests/test_api/test_object_authorization.py::test_approver_of_one_thread_cannot_resume_another",
        "an approver of one thread cannot resume another",
    ),
)


def demo_4_hitl(workdir: Path, baseline_env: dict[str, str | None]) -> ScenarioResult:
    """TEST-BASED DEMONSTRATION —— 真实运行既有 HITL 测试,不伪造在线请求。"""
    result = ScenarioResult(
        key="DEMO 4",
        title="HITL approval and resume — TEST-BASED DEMONSTRATION",
        mode=(
            "TEST-BASED DEMONSTRATION (real pytest subprocess; NOT an online business request "
            "and NOT a live LLM run)"
        ),
        inputs=tuple(f"pytest -v {node_id}" for node_id, _ in HITL_NODE_IDS),
        evidence=(
            "tests/test_core/test_hitl_graph.py (real graph + real checkpointer + real audit store)",
            "tests/test_api/test_object_authorization.py (real auth, real thread ownership)",
            "app/core/graph.py :: human_approval / policy_gate nodes",
        ),
    )

    basetemp = Path(tempfile.mkdtemp(prefix="cybersec-demo-pytest-", dir=str(workdir)))
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--no-header",
        "-p",
        "no:cacheprovider",
        f"--basetemp={basetemp}",
        *[node_id for node_id, _ in HITL_NODE_IDS],
    ]

    # 子进程用**运行本脚本之前**的环境:演示用的占位配置不得渗进测试。
    env = {name: value for name, value in os.environ.items() if name not in _MANAGED_ENV}
    for name, value in baseline_env.items():
        if value is not None:
            env[name] = value

    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = (completed.stdout or "") + (completed.stderr or "")

    result.checks.append(
        Check(
            "pytest exited with code 0",
            completed.returncode == 0,
            f"returncode={completed.returncode}",
        )
    )

    passed_nodes = [line for line in output.splitlines() if "PASSED" in line]
    for node_id, description in HITL_NODE_IDS:
        short = node_id.rsplit("::", 1)[-1]
        ok = any(short in line and "PASSED" in line for line in passed_nodes)
        result.checks.append(
            Check(f"{description} [{short}]", ok, "PASSED" if ok else "not reported as PASSED")
        )

    summary_lines = [
        line.strip()
        for line in output.splitlines()
        if (" passed" in line or " failed" in line or " error" in line) and "==" in line
    ]
    summary = summary_lines[-1] if summary_lines else "(no pytest summary line found)"
    result.checks.append(
        Check(
            "no test failed or errored",
            "failed" not in summary and "error" not in summary,
            summary,
        )
    )

    result.notes = (
        f"pytest summary: {summary}",
        "This is a test-based demonstration: the numbers come from running the repository's "
        "own HITL tests, not from a live /triage -> /resume round trip.",
        "A live round trip needs a real LLM provider: /triage enters the graph at the agent "
        "node, so the model must answer before the run can pause at human_approval.",
    )
    return result


# ===========================================================================
# main
# ===========================================================================

def _run(key: str, title: str, fn: Callable[[], ScenarioResult]) -> ScenarioResult:
    """执行一个场景;异常按 FAIL 上报(打印堆栈,不吞掉)。"""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - 场景隔离:一个坏掉不拖垮其余三个
        traceback.print_exc()
        return ScenarioResult(
            key=key,
            title=title,
            mode="(scenario aborted before completion)",
            inputs=(),
            evidence=(),
            error=f"{type(exc).__name__}: {exc}",
        )


def _configure_logging() -> None:
    """默认只显示 WARNING 以上的结构化日志,让 Demo 输出可读。

    `DEMO_VERBOSE=1` 时放开到 INFO —— 那些 `rag_ask_completed grounded=...`
    与 `http.request.completed status_code=...` 本身就是很好的旁证。
    这里只影响**本脚本进程**,不改应用本身的日志配置。
    """
    level = logging.INFO if os.environ.get("DEMO_VERBOSE") == "1" else logging.WARNING
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


def main() -> int:
    _configure_logging()
    print("CyberSec Agent — offline demo (no provider, no network, no real credentials)")
    print(f"repository : {REPO_ROOT}")
    print(f"python     : {sys.version.split()[0]}")

    workdir = Path(tempfile.mkdtemp(prefix="cybersec-demo-"))
    baseline_env = _snapshot_env()
    results: list[ScenarioResult] = []
    try:
        _apply_offline_env(workdir)
        results.append(_run("DEMO 1", "RBAC", demo_1_rbac))
        results.append(_run("DEMO 2", "Citation validation", demo_2_citations))
        results.append(_run("DEMO 3", "Retrieval fail-closed", lambda: demo_3_fail_closed(workdir)))
        results.append(
            _run("DEMO 4", "HITL (test-based)", lambda: demo_4_hitl(workdir, baseline_env))
        )
    finally:
        _restore_env(baseline_env)
        shutil.rmtree(workdir, ignore_errors=True)

    for result in results:
        _print_scenario(result)

    print()
    print(_rule("="))
    print("SUMMARY")
    print(_rule("="))
    for result in results:
        print(f"  {'PASS' if result.ok else 'FAIL'}  {result.key}  {result.title}")
    failed = [result.key for result in results if not result.ok]
    print()
    if failed:
        print(f"DEMO FAILED: {', '.join(failed)}")
        return 1
    print("DEMO PASSED: 4/4 scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
