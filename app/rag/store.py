"""向量存储契约与内存实现。

契约只有一个方法值得解释:`search(vector, limit) -> list[ScoredChunk]`。
它**不**保证同分候选的顺序 —— 真实向量库(包括 Qdrant)在分数相同时的
返回顺序取决于内部索引结构,不是可依赖的语义。因此确定性由**检索层**
在 Python 内重排完成(见 `app/rag/retriever.py`),存储层只负责"把候选
取出来"。这条分工让换后端不改变可观测行为。

内存实现刻意保持"笨":线性扫描 + 余弦。40 篇摘要、几百个块,线性扫描
比任何近似最近邻都快,而且没有近似误差 —— 在 S2-A 阶段引入 ANN 只会
让"检索结果不对"多出一种可能原因。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.rag.embedding import cosine_similarity
from app.rag.errors import StoreError
from app.rag.types import IndexMetadata, KnowledgeChunk, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """向量存储契约。"""

    def upsert(self, chunks: Sequence[KnowledgeChunk],
               vectors: Sequence[Sequence[float]]) -> None:
        """写入或替换切块向量。同一 `chunk_id` 重复写入按**替换**处理。"""
        ...

    def search(self, vector: Sequence[float], limit: int) -> list[ScoredChunk]:
        """返回至多 `limit` 个候选。**不保证同分顺序。**"""
        ...

    def metadata(self) -> IndexMetadata:
        """索引身份。"""
        ...


def _check_vector(vector: Sequence[float], dim: int, *, label: str) -> None:
    if len(vector) != dim:
        raise StoreError(
            f"{label} dimension mismatch: expected {dim}, got {len(vector)}",
            reason="dimension-mismatch",
        )
    for value in vector:
        if not math.isfinite(value):
            raise StoreError(
                f"{label} contains a non-finite value",
                reason="non-finite-vector",
            )


class InMemoryVectorStore:
    """纯 Python 内存实现 —— 默认测试与离线演示用。"""

    def __init__(self, metadata: IndexMetadata) -> None:
        self._metadata = metadata
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._vectors: dict[str, list[float]] = {}

    # ---- 契约实现 ----

    def upsert(self, chunks: Sequence[KnowledgeChunk],
               vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise StoreError(
                "chunks and vectors must have the same length",
                reason="length-mismatch",
            )
        for chunk, vector in zip(chunks, vectors):
            _check_vector(vector, self._metadata.dim, label=f"chunk {chunk.chunk_id}")
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vector)

    def search(self, vector: Sequence[float], limit: int) -> list[ScoredChunk]:
        _check_vector(vector, self._metadata.dim, label="query")
        if limit <= 0:
            return []
        scored = [
            ScoredChunk(
                chunk=self._chunks[chunk_id],
                score=cosine_similarity(vector, self._vectors[chunk_id]),
            )
            # 按 chunk_id 遍历:即使存储实现换成无序容器,候选集合也一致。
            for chunk_id in sorted(self._chunks)
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:limit]

    def metadata(self) -> IndexMetadata:
        return self._metadata

    # ---- 便利方法(非契约) ----

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunk_ids(self) -> list[str]:
        """已写入的 `chunk_id`(升序)—— 供测试核对重建幂等性。"""
        return sorted(self._chunks)

    @classmethod
    def build(
        cls,
        chunks: Sequence[KnowledgeChunk],
        vectors: Sequence[Sequence[float]],
        metadata: IndexMetadata,
    ) -> "InMemoryVectorStore":
        store = cls(metadata)
        store.upsert(chunks, vectors)
        return store
