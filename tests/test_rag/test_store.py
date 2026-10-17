"""存储测试:排序、tie-break、边界与契约违例。

存储层**不**保证同分顺序(真实向量库不保证),但内存实现仍然给出稳定
顺序,以便测试能钉住"同一输入 → 同一输出"。确定性由检索层最终负责,
这里测的是"候选集合与分数是对的"。
"""
from __future__ import annotations

import pytest

from app.rag.chunking import CHUNKER_VERSION
from app.rag.errors import StoreError
from app.rag.store import InMemoryVectorStore, VectorStore
from app.rag.types import IndexMetadata, KnowledgeChunk, sha256_hex

DIM = 4


def make_chunk(doc_id: str, ordinal: int = 0, text: str | None = None) -> KnowledgeChunk:
    body = text or f"{doc_id} 的第 {ordinal} 段"
    return KnowledgeChunk(
        chunk_id=f"{doc_id}#{ordinal}",
        doc_id=doc_id,
        ordinal=ordinal,
        text=body,
        char_start=0,
        char_end=len(body),
    )


def make_metadata(**overrides) -> IndexMetadata:
    payload = {
        "corpus_version": "v0.5.0-s2.1",
        "corpus_sha256": sha256_hex("corpus"),
        "embedder_id": "hashing-v1-d4-n3",
        "dim": DIM,
        "chunker_version": CHUNKER_VERSION,
        "doc_count": 2,
        "chunk_count": 2,
    }
    payload.update(overrides)
    return IndexMetadata(**payload)


def make_store(chunks, vectors, **metadata_overrides) -> InMemoryVectorStore:
    return InMemoryVectorStore.build(chunks, vectors, make_metadata(**metadata_overrides))


# ---------- 排序与边界 ----------

def test_search_returns_at_most_limit():
    chunks = [make_chunk(f"kb-cve-2021-000{index}") for index in range(1, 4)]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store = make_store(chunks, vectors)

    assert len(store.search([1.0, 0.0, 0.0, 0.0], 1)) == 1
    assert len(store.search([1.0, 0.0, 0.0, 0.0], 3)) == 3
    assert len(store.search([1.0, 0.0, 0.0, 0.0], 99)) == 3


def test_search_is_sorted_by_score_descending():
    chunks = [make_chunk(f"kb-cve-2021-000{index}") for index in range(1, 4)]
    vectors = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
    store = make_store(chunks, vectors)

    scores = [item.score for item in store.search([1.0, 0.0, 0.0, 0.0], 3)]
    assert scores == sorted(scores, reverse=True)


def test_equal_scores_break_by_chunk_id():
    """同分顺序由 `chunk_id` 决定,而不是由插入顺序或哈希桶决定。"""
    chunks = [make_chunk("kb-cve-2021-0002"), make_chunk("kb-cve-2021-0001")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    store = make_store(chunks, vectors)

    result = store.search([1.0, 0.0, 0.0, 0.0], 2)
    assert [item.chunk.chunk_id for item in result] == [
        "kb-cve-2021-0001#0",
        "kb-cve-2021-0002#0",
    ]


def test_search_order_is_independent_of_insertion_order():
    chunks = [make_chunk(f"kb-cve-2021-000{index}") for index in range(1, 4)]
    vectors = [[0.3, 0.7, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]]
    forward = make_store(chunks, vectors)
    backward = make_store(list(reversed(chunks)), list(reversed(vectors)))

    query = [1.0, 0.0, 0.0, 0.0]
    assert [item.chunk.chunk_id for item in forward.search(query, 3)] == [
        item.chunk.chunk_id for item in backward.search(query, 3)
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_returns_nothing(limit):
    store = make_store([make_chunk("kb-cve-2021-0001")], [[1.0, 0.0, 0.0, 0.0]])
    assert store.search([1.0, 0.0, 0.0, 0.0], limit) == []


def test_empty_store_returns_nothing():
    store = InMemoryVectorStore(make_metadata())
    assert store.search([1.0, 0.0, 0.0, 0.0], 5) == []
    assert len(store) == 0


# ---------- 契约违例 ----------

def test_chunk_and_vector_length_must_match():
    with pytest.raises(StoreError, match="same length"):
        make_store([make_chunk("kb-cve-2021-0001")], [])


def test_dimension_mismatch_is_rejected_on_upsert():
    with pytest.raises(StoreError, match="dimension mismatch"):
        make_store([make_chunk("kb-cve-2021-0001")], [[1.0, 0.0]])


def test_dimension_mismatch_is_rejected_on_search():
    store = make_store([make_chunk("kb-cve-2021-0001")], [[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(StoreError, match="dimension mismatch"):
        store.search([1.0, 0.0], 1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_vectors_are_rejected(bad):
    """NaN 参与排序时与任何值比较都为 False,结果依赖实现细节。"""
    with pytest.raises(StoreError, match="non-finite"):
        make_store([make_chunk("kb-cve-2021-0001")], [[bad, 0.0, 0.0, 0.0]])


# ---------- 幂等与身份 ----------

def test_upsert_replaces_the_same_chunk_id():
    """重复构建必须是"替换"而不是"追加",否则重建索引会留下幽灵块。"""
    chunk = make_chunk("kb-cve-2021-0001")
    store = make_store([chunk], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([chunk], [[0.0, 1.0, 0.0, 0.0]])

    assert len(store) == 1
    assert store.search([0.0, 1.0, 0.0, 0.0], 1)[0].score == pytest.approx(1.0)


def test_metadata_is_returned_unchanged():
    metadata = make_metadata()
    store = InMemoryVectorStore(metadata)
    assert store.metadata() is metadata


def test_chunk_ids_are_sorted():
    store = make_store(
        [make_chunk("kb-cve-2021-0002"), make_chunk("kb-cve-2021-0001")],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    assert store.chunk_ids == ["kb-cve-2021-0001#0", "kb-cve-2021-0002#0"]


def test_store_satisfies_the_protocol():
    assert isinstance(InMemoryVectorStore(make_metadata()), VectorStore)
