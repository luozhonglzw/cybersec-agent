"""检索测试:确定性排序、去重、top-k 边界、索引完整性 fail closed。

`verify()` 是这一层的安全边界:索引与当前语料/嵌入器不一致时,宁可
"知识源不可用",也不用陈旧索引给出看似权威的答案。
"""
from __future__ import annotations

import pytest

from app.rag import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    HashingEmbedder,
    InMemoryVectorStore,
    IndexMetadata,
    IndexUnavailable,
    RetrievalError,
    Retriever,
    build_index_metadata,
    chunk_corpus,
    sha256_hex,
)
from app.rag.chunking import CHUNKER_VERSION

QUERY = "Log4j 的 JNDI 查找为什么能被用来执行代码?"


def build_retriever(
    corpus,
    embedder: HashingEmbedder,
    *,
    min_score: float = 0.0,
    **metadata_overrides,
) -> Retriever:
    """按指定的（可能被篡改的）索引身份装配检索器。

    向量维度跟随 `metadata.dim`,这样"维度不符"这条能走到 `verify()`
    而不是先在存储层被拦下 —— 要测的正是**校验顺序**。
    """
    chunks = chunk_corpus(corpus.documents)
    metadata = build_index_metadata(corpus, embedder, chunks).model_copy(
        update=metadata_overrides
    )
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    if metadata.dim != embedder.dim:
        vectors = [vector[: metadata.dim] for vector in vectors]
    store = InMemoryVectorStore.build(chunks, vectors, metadata)
    return Retriever(
        embedder,
        store,
        {document.doc_id: document for document in corpus.documents},
        manifest=corpus.manifest,
        min_score=min_score,
    )


# ---------- 确定性 ----------

def test_retrieval_is_deterministic(retriever):
    first = retriever.retrieve(QUERY)
    second = retriever.retrieve(QUERY)
    assert first.model_dump() == second.model_dump()


def test_hits_are_sorted_by_score_then_citation_id(retriever):
    hits = retriever.retrieve(QUERY, top_k=MAX_TOP_K).hits
    assert hits == sorted(hits, key=lambda hit: (-hit.score, hit.citation_id))


def test_query_hash_is_stable_and_is_not_the_raw_query(retriever):
    """工具结果会进日志与 checkpoint,没必要把问题原文一起带过去。"""
    result = retriever.retrieve(QUERY)
    assert result.query_sha256 == sha256_hex(QUERY)
    assert QUERY not in result.query_sha256
    assert len(result.query_sha256) == 64


def test_result_reports_the_corpus_version(retriever, corpus):
    assert retriever.retrieve(QUERY).corpus_version == corpus.manifest.corpus_version


# ---------- 去重与粒度 ----------

def test_documents_are_deduplicated(retriever):
    """引用粒度是文档:同一篇文档的多个块不能各占一个位置。"""
    hits = retriever.retrieve(QUERY, top_k=MAX_TOP_K).hits
    doc_ids = [hit.doc_id for hit in hits]
    assert len(set(doc_ids)) == len(doc_ids)


def test_citation_id_is_the_document_id(retriever):
    for hit in retriever.retrieve(QUERY).hits:
        assert hit.citation_id == hit.doc_id


def test_hits_carry_the_document_metadata(retriever, corpus):
    hit = retriever.retrieve(QUERY).hits[0]
    document = corpus.by_id(hit.doc_id)
    assert hit.source_id == document.source_id
    assert hit.source_type == document.source_type
    assert hit.title == document.title
    assert hit.url == document.source_url


# ---------- top-k 边界 ----------

def test_default_top_k_is_five(retriever):
    assert retriever.top_k == DEFAULT_TOP_K
    assert retriever.retrieve(QUERY).top_k == DEFAULT_TOP_K


@pytest.mark.parametrize(("requested", "expected"), [(1, 1), (3, 3), (MAX_TOP_K, MAX_TOP_K)])
def test_top_k_is_honoured_within_bounds(retriever, requested, expected):
    result = retriever.retrieve(QUERY, top_k=requested)
    assert result.top_k == expected
    assert len(result.hits) <= expected


def test_top_k_is_clamped_to_the_contract(retriever):
    result = retriever.retrieve(QUERY, top_k=999)
    assert result.top_k == MAX_TOP_K
    assert len(result.hits) <= MAX_TOP_K


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_empty_query_is_rejected(retriever, query):
    with pytest.raises(RetrievalError, match="non-empty"):
        retriever.retrieve(query)


def test_empty_index_returns_an_empty_result(embedder):
    """空索引是"检索成功但没有任何候选",不是错误,也不是 5xx。"""
    metadata = IndexMetadata(
        corpus_version="v0.5.0-s2.1",
        corpus_sha256=sha256_hex("empty"),
        embedder_id=embedder.embedder_id,
        dim=embedder.dim,
        chunker_version=CHUNKER_VERSION,
        doc_count=0,
        chunk_count=0,
    )
    retriever = Retriever(embedder, InMemoryVectorStore(metadata), {}, manifest=None)

    result = retriever.retrieve(QUERY)
    assert result.status == "empty"
    assert result.hits == []


def test_character_hashing_collisions_produce_spurious_similarity(retriever, corpus):
    """**已记录的限制**:零字符重合的查询仍可能得到正分。

    256 维特征哈希下,7 个查询 gram 与每篇文档约 250 个 gram 的期望碰撞
    数接近 7,所以"无关"很少表现为 0 分。这条把事实钉住,防止把
    `score > 0` 或 `status == "ok"` 误当成"找到了相关证据";无答案的判定
    必须靠引用校验(S2-B),不能靠检索为空。
    """
    query = "ζζζζ ξξξξ"
    assert not any(
        char in document.text
        for document in corpus.documents
        for char in query
        if char != " "
    )

    result = retriever.retrieve(query)
    assert result.status == "ok"
    assert all(hit.score > 0 for hit in result.hits)


# ---------- 索引完整性 fail closed ----------

@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("corpus_sha256", sha256_hex("tampered"), "index-corpus-hash"),
        ("corpus_version", "v0.5.0-s9.9", "index-corpus-version"),
        ("embedder_id", "hashing-v1-d256-n9", "index-embedder"),
        ("dim", 128, "index-dim"),
        ("chunker_version", "paragraph-window-v0", "index-chunker"),
        ("doc_count", 999, "index-doc-count"),
    ],
)
def test_stale_index_fails_closed(corpus, embedder, field, value, reason):
    retriever = build_retriever(corpus, embedder, **{field: value})
    with pytest.raises(IndexUnavailable) as excinfo:
        retriever.retrieve(QUERY)
    assert excinfo.value.reason == reason


def test_verify_returns_metadata_when_index_matches(retriever):
    assert retriever.verify().corpus_sha256


def test_index_unavailable_message_leaks_no_paths(corpus, embedder):
    retriever = build_retriever(corpus, embedder, doc_count=999)
    with pytest.raises(IndexUnavailable) as excinfo:
        retriever.retrieve(QUERY)
    message = str(excinfo.value)
    assert "\\" not in message
    assert "/" not in message
    assert ".jsonl" not in message


def test_missing_document_fails_closed(corpus, embedder):
    """索引里引用了语料中不存在的文档 → 不可用,而不是返回残缺的命中。"""
    chunks = chunk_corpus(corpus.documents)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store = InMemoryVectorStore.build(
        chunks, vectors, build_index_metadata(corpus, embedder, chunks)
    )
    retriever = Retriever(embedder, store, {}, manifest=None)
    with pytest.raises(IndexUnavailable) as excinfo:
        retriever.retrieve(QUERY)
    assert excinfo.value.reason == "index-unknown-document"


# ---------- 阈值 ----------

def test_min_score_filters_weak_candidates(corpus, embedder):
    """把阈值抬到 1.0 时,只有完全重合的向量才留下。"""
    strict = build_retriever(corpus, embedder, min_score=1.0)
    assert strict.retrieve(QUERY).status == "empty"


def test_unanswerable_queries_are_not_filtered_by_the_baseline(retriever, eval_queries):
    """**已记录的限制**,不是通过项:`HashingEmbedder` 下语料外的问题仍会得分。

    字符 n-gram 基线对中文虚词(的、是、在)天然敏感,因此"语料没有答案"
    不会表现为零命中。这条断言把这个事实钉住,防止后续有人把
    `status == "empty"` 当成"没有答案"的判据 —— 无答案的判定必须靠引用
    校验(S2-B),不能靠检索为空。
    """
    unanswerable = [item for item in eval_queries if not item["answerable"]]
    assert len(unanswerable) == 4

    results = [retriever.retrieve(item["query"]) for item in unanswerable]
    assert all(item["gold_doc_ids"] == [] for item in unanswerable)
    assert any(result.status == "ok" for result in results), (
        "基线的已知限制:语料外问题仍可能得分,因此 empty 不能当作无答案判据"
    )
    for result in results:
        assert all(hit.score > 0 for hit in result.hits)
