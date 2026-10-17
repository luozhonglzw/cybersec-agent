"""引用解析与校验的离线测试。

这些测试只依赖构造出来的消息对象,不读语料、不建索引、不碰网络 ——
引用校验的正确性应当与"检索得准不准"完全解耦。一旦两者纠缠,一个检索
质量波动就会让引用逻辑的回归信号变得不可读。
"""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.rag.citations import (
    CITATION_MARKER_PATTERN,
    MAX_REPORTED_MARKERS,
    CitationValidation,
    EvidenceCollection,
    EvidenceSource,
    HitsEnvelope,
    assess_answer_citations,
    collect_evidence,
    looks_like_rag_hits,
    parse_citation_markers,
    parse_error_envelope,
    parse_hits_envelope,
    validate_citations,
)

CORPUS_VERSION = "v0.5.0-s2.1"
QUERY_SHA = "a" * 64


# ---------------------------------------------------------------------------
# 构造器:所有信封都走真实模型校验,不构造"绕过校验的裸字典"
# ---------------------------------------------------------------------------

def make_hit(**overrides):
    payload = {
        "citation_id": "kb-cve-2021-44228",
        "doc_id": "kb-cve-2021-44228",
        "source_id": "CVE-2021-44228",
        "source_type": "cve",
        "title": "合成标题 —— Log4j JNDI 注入",
        "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
        "score": 0.5,
        "snippet": "合成片段,仅用于测试。",
    }
    payload.update(overrides)
    return payload


def make_attack_hit(**overrides):
    payload = make_hit(
        citation_id="kb-attack-t1190",
        doc_id="kb-attack-t1190",
        source_id="T1190",
        source_type="attack-technique",
        title="合成标题 —— 利用对外暴露的应用",
        url="https://attack.mitre.org/techniques/T1190/",
        **overrides,
    )
    return payload


def make_envelope(hits=(), corpus_version=CORPUS_VERSION, **overrides) -> str:
    payload = {
        "kind": "rag_hits",
        "corpus_version": corpus_version,
        "query_sha256": QUERY_SHA,
        "top_k": 5,
        "hits": list(hits),
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def make_error_envelope(reason: str = "index-corpus-hash") -> str:
    return json.dumps(
        {
            "kind": "rag_error",
            "error": "knowledge index unavailable",
            "type": "IndexUnavailable",
            "reason": reason,
        },
        ensure_ascii=False,
    )


def tool_message(content: str, call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id)


# ---------------------------------------------------------------------------
# 标记语法
# ---------------------------------------------------------------------------

def test_the_marker_regex_matches_the_frozen_shape():
    """S1 §2.5 冻结的标记格式:只认 `[src:<id>]`。"""
    match = CITATION_MARKER_PATTERN.fullmatch("[src:kb-cve-2021-44228]")
    assert match is not None
    assert match.group(1) == "kb-cve-2021-44228"

    assert CITATION_MARKER_PATTERN.fullmatch("[src:kb-attack-t1190]") is not None
    assert CITATION_MARKER_PATTERN.fullmatch("[src:KB-CVE-2021-44228]") is not None
    # 不是合法标记的形态
    assert CITATION_MARKER_PATTERN.fullmatch("[src:]") is None
    assert CITATION_MARKER_PATTERN.fullmatch("[src: kb-cve-2021-44228]") is None
    assert CITATION_MARKER_PATTERN.fullmatch("[src:../../etc/passwd]") is None
    assert CITATION_MARKER_PATTERN.fullmatch("src:kb-cve-2021-44228") is None
    assert CITATION_MARKER_PATTERN.fullmatch("[src:kb-cve-2021-44228") is None


def test_parse_returns_ids_in_appearance_order_and_deduplicates():
    answer = (
        "先说结论。[src:kb-attack-t1190] 然后补充。[src:kb-cve-2021-44228] "
        "再引用一次同一篇。[src:kb-attack-t1190]"
    )
    ids, malformed = parse_citation_markers(answer)
    assert ids == ("kb-attack-t1190", "kb-cve-2021-44228")
    assert malformed == ()


def test_malformed_markers_are_reported_instead_of_vanishing():
    """写错的标记必须暴露 —— 静默丢弃会把"试图引用但写错"伪装成"没引用"。"""
    answer = "坏的:[src:] 带空格:[src: kb-cve-2021-44228] 穿越:[src:../../etc/passwd]"
    ids, malformed = parse_citation_markers(answer)
    assert ids == ()
    assert malformed == (
        "[src:]",
        "[src: kb-cve-2021-44228]",
        "[src:../../etc/passwd]",
    )


def test_malformed_marker_reporting_is_bounded():
    answer = "".join("[src:]" for _ in range(MAX_REPORTED_MARKERS + 15))
    ids, malformed = parse_citation_markers(answer)
    assert ids == ()
    assert len(malformed) == MAX_REPORTED_MARKERS


def test_markers_never_cross_a_line_break_or_a_bracket():
    answer = "[src:kb-cve-2021-44228\n] [src:[src:kb-attack-t1190]"
    ids, _ = parse_citation_markers(answer)
    assert ids == ("kb-attack-t1190",)


# ---------------------------------------------------------------------------
# 信封解析:严格拒绝
# ---------------------------------------------------------------------------

def test_parse_hits_envelope_accepts_the_documented_shape():
    envelope = parse_hits_envelope(make_envelope([make_hit()]))
    assert envelope is not None
    assert envelope.kind == "rag_hits"
    assert envelope.corpus_version == CORPUS_VERSION
    assert envelope.top_k == 5
    assert envelope.hits[0].citation_id == "kb-cve-2021-44228"


def test_parse_rejects_unknown_extra_fields():
    """多出一个字段就拒绝:信封只可能来自我们自己的工具或伪造。"""
    raw = make_envelope([make_hit()], injected=True)
    assert parse_hits_envelope(raw) is None

    raw = make_envelope([make_hit(extra="x")])
    assert parse_hits_envelope(raw) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"citation_id": "kb-cve-2021-44228x", "doc_id": "kb-attack-t1190"},
        {"citation_id": "not-a-doc-id", "doc_id": "not-a-doc-id"},
        {"citation_id": "kb-cve-2021-44228", "doc_id": "kb-cve-2021-44228", "url": "http://nvd.nist.gov/x"},
        {"citation_id": "kb-cve-2021-44228", "doc_id": "kb-cve-2021-44228", "url": "https://evil.example.com/x"},
        {"citation_id": "kb-cve-2021-44228", "doc_id": "kb-cve-2021-44228", "url": "https://nvd.nist.gov/"},
        {"citation_id": "kb-cve-2021-44228", "doc_id": "kb-cve-2021-44228", "snippet": ""},
        {"citation_id": "kb-cve-2021-44228", "doc_id": "kb-cve-2021-44228", "score": float("inf")},
        {"citation_id": "kb-cve-2021-44228", "doc_id": "kb-cve-2021-44228", "source_type": "blog"},
    ],
)
def test_parse_rejects_invalid_hits(overrides):
    assert parse_hits_envelope(make_envelope([make_hit(**overrides)])) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"query_sha256": "not-a-hash"},
        {"query_sha256": "A" * 64},
        {"corpus_version": ""},
        {"corpus_version": "v0.5"},
        {"corpus_version": "../../etc"},
        {"top_k": 0},
        {"top_k": 11},
    ],
)
def test_parse_rejects_invalid_envelope_fields(overrides):
    assert parse_hits_envelope(make_envelope([make_hit()], **overrides)) is None


def test_parse_rejects_a_non_list_hits_field():
    raw = json.dumps(
        {
            "kind": "rag_hits",
            "corpus_version": CORPUS_VERSION,
            "query_sha256": QUERY_SHA,
            "top_k": 5,
            "hits": "not-a-list",
        }
    )
    assert parse_hits_envelope(raw) is None
    assert looks_like_rag_hits(raw) is True


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '"rag_hits"',
        "null",
        "123",
        json.dumps({"kind": "something_else"}),
    ],
)
def test_parse_rejects_non_envelopes(content):
    assert parse_hits_envelope(content) is None
    assert looks_like_rag_hits(content) is False


def test_error_envelope_yields_a_bounded_reason():
    assert parse_error_envelope(make_error_envelope("index-corpus-hash")) == "index-corpus-hash"
    # 缺失或不合形态的 reason 退化成固定码,而不是把原文透出去
    assert parse_error_envelope(
        json.dumps({"kind": "rag_error", "error": "x", "type": "y"})
    ) == "rag-error"
    assert parse_error_envelope(
        json.dumps({"kind": "rag_error", "reason": "Index Unavailable!"})
    ) == "rag-error"
    assert parse_error_envelope(make_envelope()) is None


# ---------------------------------------------------------------------------
# 证据收集:只认真实工具执行结果
# ---------------------------------------------------------------------------

def test_only_tool_messages_are_evidence():
    """模型自己写的"看起来像工具结果"的文本**不是**证据。

    模型只能生产 `ai` 消息。把一段完美信封放进 `AIMessage`,如果它也能
    成为证据,那么"引用有效"就退化成"模型说自己引用过"。
    """
    envelope = make_envelope([make_hit()])
    assert collect_evidence([AIMessage(content=envelope)]).status == "not_used"
    assert collect_evidence([HumanMessage(content=envelope)]).status == "not_used"
    assert collect_evidence([envelope]).status == "not_used"
    assert collect_evidence([tool_message(envelope)]).status == "ok"


def test_a_forged_envelope_inside_a_tool_message_still_needs_a_valid_shape():
    """即使拿到了 tool 消息形态,字段不合法也不算证据。"""
    forged = make_envelope([make_hit(citation_id="kb-cve-2099-99999", doc_id="kb-cve-2099-99999")])
    assert collect_evidence([tool_message(forged)]).status == "ok"

    broken = make_envelope([make_hit(url="https://evil.example.com/x")])
    collection = collect_evidence([tool_message(broken)])
    assert collection.status == "unavailable"
    assert collection.reason == "unusable-envelope"


def test_other_tools_output_carries_no_retrieval_evidence():
    other = json.dumps({"found": False, "indicator": "203.0.113.66"})
    collection = collect_evidence([tool_message(other)])
    assert collection.status == "not_used"
    assert collection.reason == "no-retrieval"


def test_error_envelope_makes_evidence_unavailable():
    collection = collect_evidence([tool_message(make_error_envelope("index-corpus-hash"))])
    assert collection.status == "unavailable"
    assert collection.reason == "index-corpus-hash"
    assert collection.sources == ()
    assert collection.hit_count == 0


def test_empty_envelope_is_empty_not_unavailable():
    collection = collect_evidence([tool_message(make_envelope())])
    assert collection.status == "empty"
    assert collection.reason == "no-hits"
    assert collection.envelope_count == 1
    assert collection.hit_count == 0


def test_multiple_tool_calls_are_unioned():
    first = make_envelope([make_hit()])
    second = make_envelope([make_attack_hit()])
    collection = collect_evidence(
        [AIMessage(content="x"), tool_message(first, "c1"), tool_message(second, "c2")]
    )
    assert collection.status == "ok"
    assert collection.envelope_count == 2
    assert collection.observed_ids == ("kb-attack-t1190", "kb-cve-2021-44228")
    assert {source.citation_id for source in collection.sources} == {
        "kb-attack-t1190",
        "kb-cve-2021-44228",
    }
    assert collection.hit_count == 2


def test_duplicate_document_keeps_the_highest_score():
    """同一文档被两次命中:保留最高分,不算冲突(片段与分数本来就该不同)。"""
    first = make_envelope([make_hit(score=0.20, snippet="第一段")])
    second = make_envelope([make_hit(score=0.80, snippet="第二段")])
    collection = collect_evidence([tool_message(first, "c1"), tool_message(second, "c2")])
    assert collection.status == "ok"
    assert collection.hit_count == 1
    assert collection.sources[0].score == pytest.approx(0.80)
    assert collection.sources[0].snippet == "第二段"
    assert collection.conflicting_citations == ()


def test_metadata_conflict_for_the_same_id_fails_closed():
    """同一个 id 指向两份不同来源 → 注册表整体作废。"""
    first = make_envelope([make_hit(title="标题 A")])
    second = make_envelope([make_hit(title="标题 B")])
    collection = collect_evidence([tool_message(first, "c1"), tool_message(second, "c2")])

    assert collection.status == "unavailable"
    assert collection.reason == "citation-metadata-conflict"
    assert collection.conflicting_citations == ("kb-cve-2021-44228",)
    assert collection.sources == ()
    assert collection.hit_count == 0
    # 诊断信息仍然保留:哪些 id 被看到过,是诊断,不是信任
    assert collection.observed_ids == ("kb-cve-2021-44228",)


def test_conflict_on_the_source_url_alone_is_enough():
    first = make_envelope([make_hit()])
    second = make_envelope([make_hit(url="https://github.com/advisories/GHSA-x")])
    collection = collect_evidence([tool_message(first, "c1"), tool_message(second, "c2")])
    assert collection.status == "unavailable"
    assert collection.reason == "citation-metadata-conflict"


def test_multiple_corpus_versions_fail_closed():
    first = make_envelope([make_hit()], corpus_version="v0.5.0-s2.1")
    second = make_envelope([make_attack_hit()], corpus_version="v0.5.0-s2.2")
    collection = collect_evidence([tool_message(first, "c1"), tool_message(second, "c2")])

    assert collection.status == "unavailable"
    assert collection.reason == "corpus-version-conflict"
    assert collection.sources == ()
    assert collection.observed_ids == ("kb-attack-t1190", "kb-cve-2021-44228")


def test_non_string_tool_content_is_unusable():
    class WeirdToolMessage:
        type = "tool"
        content = [{"type": "text", "text": "{}"}]

    collection = collect_evidence([WeirdToolMessage()])
    assert collection.status == "unavailable"
    assert collection.reason == "unusable-envelope"


def test_evidence_collection_rejects_inconsistent_status():
    with pytest.raises(ValidationError):
        EvidenceCollection(status="ok", reason="ok")
    with pytest.raises(ValidationError):
        EvidenceCollection(status="empty", reason="no-hits", hit_count=1)
    with pytest.raises(ValidationError):
        EvidenceCollection(status="unavailable", reason="x", sources=(_source(),))


def _source() -> EvidenceSource:
    return EvidenceSource.model_validate(make_hit())


# ---------------------------------------------------------------------------
# 引用校验与 grounded
# ---------------------------------------------------------------------------

@pytest.fixture()
def ok_evidence() -> EvidenceCollection:
    return collect_evidence(
        [tool_message(make_envelope([make_hit(score=0.9), make_attack_hit(score=0.4)]))]
    )


def test_unknown_id_is_invalid_and_blocks_grounded(ok_evidence):
    answer = "有据可依。[src:kb-cve-2021-44228] 但这里编了一个。[src:kb-cve-2099-99999]"
    validation = validate_citations(answer, ok_evidence)

    assert validation.status == "ok"
    assert validation.valid_citations == ("kb-cve-2021-44228",)
    assert validation.invalid_citations == ("kb-cve-2099-99999",)
    assert validation.grounded is False


def test_valid_citations_carry_metadata_sorted_by_score(ok_evidence):
    answer = "[src:kb-attack-t1190] 与 [src:kb-cve-2021-44228]"
    validation = validate_citations(answer, ok_evidence)

    assert validation.valid_citations == ("kb-attack-t1190", "kb-cve-2021-44228")
    assert [citation.citation_id for citation in validation.citations] == [
        "kb-cve-2021-44228",
        "kb-attack-t1190",
    ]
    assert validation.citations[0].score == pytest.approx(0.9)
    assert validation.citations[0].source_id == "CVE-2021-44228"
    assert validation.citations[0].url.startswith("https://nvd.nist.gov/")
    assert validation.grounded is True


def test_no_marker_means_not_grounded(ok_evidence):
    validation = validate_citations("这段回答没有任何引用。", ok_evidence)
    assert validation.valid_citations == ()
    assert validation.grounded is False


def test_grounded_requires_a_successful_retrieval():
    """检索失败/未使用时,即使答案里写了真实存在的 id,也不算 grounded。"""
    for collection in (
        EvidenceCollection(status="not_used", reason="no-retrieval"),
        EvidenceCollection(status="unavailable", reason="index-corpus-hash"),
        EvidenceCollection(status="empty", reason="no-hits"),
    ):
        validation = validate_citations("引用 [src:kb-cve-2021-44228]", collection)
        assert validation.valid_citations == ()
        assert validation.invalid_citations == ("kb-cve-2021-44228",)
        assert validation.grounded is False


def test_conflicted_ids_are_unverified_not_silently_valid():
    first = make_envelope([make_hit(title="标题 A")])
    second = make_envelope([make_hit(title="标题 B")])
    collection = collect_evidence([tool_message(first, "c1"), tool_message(second, "c2")])
    validation = validate_citations("引用 [src:kb-cve-2021-44228]", collection)

    assert validation.status == "unavailable"
    assert validation.reason == "citation-metadata-conflict"
    assert validation.valid_citations == ()
    assert validation.invalid_citations == ()
    assert validation.unverified_citations == ("kb-cve-2021-44228",)
    assert validation.citations == ()
    assert validation.grounded is False


def test_version_conflict_ids_are_unverified_too():
    first = make_envelope([make_hit()], corpus_version="v0.5.0-s2.1")
    second = make_envelope([make_attack_hit()], corpus_version="v0.5.0-s2.2")
    collection = collect_evidence([tool_message(first, "c1"), tool_message(second, "c2")])
    validation = validate_citations("[src:kb-cve-2021-44228]", collection)

    assert validation.status == "unavailable"
    assert validation.reason == "corpus-version-conflict"
    assert validation.unverified_citations == ("kb-cve-2021-44228",)
    assert validation.grounded is False


def test_malformed_markers_are_reported_but_do_not_enter_the_frozen_formula(ok_evidence):
    """记录冻结公式的**当前**边界:畸形标记单独报告,不参与 `grounded`。

    S1 §2.5 冻结的公式只有三项。是否把畸形标记纳入判据属于契约变更,
    已作为 N-4 上报;这条测试把现状钉住 —— 一旦有人改了公式,它会立刻
    变红,从而迫使改动显式化,而不是悄悄发生。
    """
    answer = "有据可依 [src:kb-cve-2021-44228],另有一处写错 [src:]"
    validation = validate_citations(answer, ok_evidence)

    assert validation.malformed_markers == ("[src:]",)
    assert validation.invalid_citations == ()
    assert validation.grounded is True


def test_buckets_must_be_disjoint():
    with pytest.raises(ValidationError):
        CitationValidation(
            status="ok",
            reason="ok",
            valid_citations=("kb-cve-2021-44228",),
            invalid_citations=("kb-cve-2021-44228",),
        )


def test_assess_answer_citations_is_the_single_entry_point(ok_evidence):
    answer = "结论 [src:kb-cve-2021-44228]"
    messages = [tool_message(make_envelope([make_hit(score=0.9), make_attack_hit(score=0.4)]))]
    direct = assess_answer_citations(answer, messages)
    composed = validate_citations(answer, collect_evidence(messages))
    assert direct.model_dump() == composed.model_dump()


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------

def test_validation_is_deterministic():
    """同一输入两次运行 → 逐字段相同(排序键全部显式)。"""
    messages = [
        tool_message(make_envelope([make_hit(score=0.5), make_attack_hit(score=0.5)]), "c1"),
    ]
    answer = "[src:kb-attack-t1190] [src:kb-cve-2021-44228]"
    first = assess_answer_citations(answer, messages).model_dump()
    second = assess_answer_citations(answer, messages).model_dump()
    assert first == second
    # 同分时按 citation_id 升序,不依赖字典插入顺序
    assert [citation.citation_id for citation in assess_answer_citations(answer, messages).citations] == [
        "kb-attack-t1190",
        "kb-cve-2021-44228",
    ]


def test_hits_envelope_model_is_immutable():
    envelope = HitsEnvelope.model_validate(json.loads(make_envelope([make_hit()])))
    with pytest.raises(ValidationError):
        envelope.top_k = 3  # type: ignore[misc]
