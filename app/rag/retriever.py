"""检索器:索引完整性校验 + **确定性** top-k。

三条刻意的决定
------------

**1. 候选扩取后在 Python 内重排。**
存储层只保证"给出候选",不保证同分顺序(真实向量库的顺序取决于内部
索引结构)。所以这里先取 `top_k * CANDIDATE_MULTIPLIER` 个候选,再按
`(-score, chunk_id)` 重排并截断。于是"同一 query → 同一顺序"成为可测
性质,而不是对某个后端的善意假设。

**2. 按 `doc_id` 去重,而不是按 `chunk_id`。**
引用粒度是文档(`citation_id == doc_id`)。同一篇文档的两个块同时命中
时,上下文里出现两次来源只会挤占预算;保留最高分的那块即可。

**3. 零分不算命中。**
`HashingEmbedder` 下无关文本的余弦接近 0(带符号哈希还会给出负值)。
把 `score <= min_score` 的候选全部丢弃,于是"语料里没有答案"这件事
表现为 `status == "empty"`,而不是"返回 5 条毫不相关的文档但分数很低"。
这个阈值是**基线参数**,换真实嵌入器时必须重新标定 —— 它不是普适常数。

`verify()` 是 fail closed 的入口:语料版本/哈希、嵌入器身份、维度、
切块器版本、文档数,任一项与索引不符就抛 `IndexUnavailable`,而不是
用陈旧索引回答。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.rag.chunking import CHUNKER_VERSION, chunk_corpus
from app.rag.corpus import LoadedCorpus
from app.rag.embedding import Embedder
from app.rag.errors import IndexUnavailable, RetrievalError
from app.rag.store import InMemoryVectorStore, VectorStore
from app.rag.types import (
    DEFAULT_TOP_K,
    CorpusManifest,
    IndexMetadata,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalHit,
    RetrievalResult,
    ScoredChunk,
    clamp_top_k,
    normalize_query,
    sha256_hex,
    snippet_of,
    sorted_by_score,
)

#: 候选扩取倍数。去重与重排都在候选集上做,取太小的候选会让"扩取"
#: 失去意义;取太大则线性扫描成本上升。4 倍在几十个块的规模上无感。
CANDIDATE_MULTIPLIER = 4

#: 命中片段上限。片段会进模型上下文,不该把整块塞进去。
SNIPPET_MAX_CHARS = 400

#: 基线命中阈值:严格大于 0 才算命中(见模块文档第 3 条)。
MIN_HIT_SCORE = 0.0


def verify_index(
    manifest: CorpusManifest,
    metadata: IndexMetadata,
    *,
    embedder_id: str,
    dim: int,
    doc_count: int,
) -> None:
    """索引与当前语料/嵌入器是否仍然匹配。**任一不符即不可用。**"""
    checks: tuple[tuple[bool, str], ...] = (
        (metadata.corpus_sha256 != manifest.corpus_sha256, "corpus-hash"),
        (metadata.corpus_version != manifest.corpus_version, "corpus-version"),
        (metadata.embedder_id != embedder_id, "embedder"),
        (metadata.dim != dim, "dim"),
        (metadata.chunker_version != CHUNKER_VERSION, "chunker"),
        (metadata.doc_count != doc_count, "doc-count"),
    )
    for failed, reason in checks:
        if failed:
            # 消息里只写原因码:索引路径、集合名都不该离开进程。
            raise IndexUnavailable(
                f"knowledge index is unavailable ({reason})",
                reason=f"index-{reason}",
            )


class Retriever:
    """文档级检索器。"""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        documents: Mapping[str, KnowledgeDocument],
        *,
        manifest: CorpusManifest | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_HIT_SCORE,
        candidate_multiplier: int = CANDIDATE_MULTIPLIER,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        self._embedder = embedder
        self._store = store
        self._documents = dict(documents)
        self._manifest = manifest
        self._top_k = clamp_top_k(top_k)
        self._min_score = float(min_score)
        self._candidate_multiplier = candidate_multiplier

    # ---- 只读属性 ----

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def min_score(self) -> float:
        return self._min_score

    # ---- 完整性 ----

    def verify(self) -> IndexMetadata:
        """校验索引身份。不匹配 → `IndexUnavailable`。"""
        metadata = self._store.metadata()
        if self._manifest is None:
            # 没有清单就没有"当前语料"这个概念,无法证明索引新鲜。
            # 允许这种模式是为了单测能只测排序;生产装配必须带清单。
            return metadata
        verify_index(
            self._manifest,
            metadata,
            embedder_id=self._embedder.embedder_id,
            dim=self._embedder.dim,
            doc_count=len(self._documents),
        )
        return metadata

    # ---- 检索 ----

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        normalized = normalize_query(query)
        if not normalized:
            raise RetrievalError("query must be non-empty", reason="empty-query")

        metadata = self.verify()
        depth = clamp_top_k(top_k if top_k is not None else self._top_k)

        vector = self._embedder.embed_query(normalized)
        candidates = self._store.search(vector, depth * self._candidate_multiplier)
        hits = self._to_hits(candidates, depth)

        return RetrievalResult(
            status="ok" if hits else "empty",
            query_sha256=sha256_hex(normalized),
            top_k=depth,
            hits=hits,
            corpus_version=metadata.corpus_version,
        )

    # ---- 内部 ----

    def _to_hits(self, candidates: Sequence[ScoredChunk], depth: int) -> list[RetrievalHit]:
        ordered = sorted(
            candidates, key=lambda item: (-item.score, item.chunk.chunk_id)
        )
        best_per_document: dict[str, ScoredChunk] = {}
        for item in ordered:
            if item.score <= self._min_score:
                continue
            # ordered 已按分数降序,首次出现的即为该文档的最高分块。
            best_per_document.setdefault(item.chunk.doc_id, item)

        hits = [self._hit(item) for item in best_per_document.values()]
        return sorted_by_score(hits)[:depth]

    def _hit(self, item: ScoredChunk) -> RetrievalHit:
        chunk = item.chunk
        document = self._documents.get(chunk.doc_id)
        if document is None:
            raise IndexUnavailable(
                "knowledge index references an unknown document",
                reason="index-unknown-document",
            )
        return RetrievalHit(
            citation_id=document.doc_id,
            doc_id=document.doc_id,
            chunk_id=chunk.chunk_id,
            source_id=document.source_id,
            source_type=document.source_type,
            title=document.title,
            url=document.source_url,
            score=item.score,
            snippet=snippet_of(chunk.text, SNIPPET_MAX_CHARS),
        )


def build_index_metadata(
    corpus: LoadedCorpus, embedder: Embedder, chunks: Sequence[KnowledgeChunk]
) -> IndexMetadata:
    """由语料 + 嵌入器 + 切块结果构造索引身份。

    单独抽出来,是因为**索引身份只能有一处来源**。若测试各自拼
    `IndexMetadata`,就会出现"测试造的身份和生产者造的不一样"这类
    假绿 —— 而身份正是 fail closed 的判据。
    """
    return IndexMetadata(
        corpus_version=corpus.manifest.corpus_version,
        corpus_sha256=corpus.manifest.corpus_sha256,
        embedder_id=embedder.embedder_id,
        dim=embedder.dim,
        chunker_version=CHUNKER_VERSION,
        doc_count=len(corpus.documents),
        chunk_count=len(chunks),
    )


def build_in_memory_retriever(
    corpus: LoadedCorpus,
    embedder: Embedder,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> Retriever:
    """组装"语料 → 切块 → 嵌入 → 内存索引 → 检索器"的离线参考链路。

    S2-A 只有这一条装配路径(真实向量库在 S2-D)。把它放在这里而不是
    测试里,是为了让"索引身份怎么算"只有一份实现 —— 测试与被测代码
    走同一条装配,而不是各写一遍。
    """
    chunks = chunk_corpus(corpus.documents)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    metadata = build_index_metadata(corpus, embedder, chunks)
    store = InMemoryVectorStore.build(chunks, vectors, metadata)
    return Retriever(
        embedder,
        store,
        {document.doc_id: document for document in corpus.documents},
        manifest=corpus.manifest,
        top_k=top_k,
    )
