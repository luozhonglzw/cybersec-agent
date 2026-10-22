"""BM25 检索候选 —— 字符 3-gram、块级打分、文档级取最大块分。

为什么要有第二个检索器
--------------------
`HashingEmbedder` 把字符 3-gram 哈希进 256 维再算余弦:不同 gram 落进
同一个桶(碰撞)后就不再可区分。BM25 对每个 gram 单独记 IDF 与词频,
不做降维。两条路径共用同一种分词(字符 3-gram),所以分数可以直接对比:
S2-F2 在冻结的 20 条评测集上测得 BM25 `Recall@5 = 0.8125` /
`MRR@5 = 0.9375`,Hashing 基线是 `0.6562` / `0.6771`。

**它仍然只是词法匹配。** 查询与文档没有任何共享 3-gram 时 BM25 给不出
信号 —— 冻结集里 4 个零重合 gold 在 BM25 下同样未命中。把这里的结果读成
"语义检索效果"是错的。

三条刻意的决定
------------

**1. 块级打分,文档级取最大块分。**
一篇文档有多个段落。平均段落分数会稀释"有一段高度相关、其余无关"的
文档;把整篇拼成一块又会让长文档的词频被长度归一化压平。取最大块分
等于说"这篇文档里有一段直接回应了查询",与引用粒度(文档)一致。

**2. 查询词项去重,按首次出现顺序求和。**
重复的 gram 不重复计分(BM25 的查询词项是集合,不是词袋);顺序固定
意味着浮点求和顺序固定 —— 同一查询在任何一次运行里得到逐位相同的分数。

**3. 不做候选扩取。**
向量路径扩取候选,是因为真实向量库的 top-k 可能是近似的。BM25 对全部
块精确打分,没有近似误差,所以把所有正分块交给 `hits_from_candidates`
去重截断。若先截断再去重,同一文档的最高分块可能落在截断之外,结果就
不再是"文档级 top-k"。

本模块**不做 I/O、不联网、不读时钟、不用内置 `hash()`**:输入只有文档
与查询,输出只由它们决定。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from app.rag.chunking import chunk_corpus
from app.rag.corpus import LoadedCorpus
from app.rag.errors import CorpusError, RetrievalError
from app.rag.retriever import MIN_HIT_SCORE, hits_from_candidates
from app.rag.types import (
    DEFAULT_TOP_K,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalResult,
    ScoredChunk,
    clamp_top_k,
    normalize_query,
    sha256_hex,
)

#: BM25 的两个自由参数。取文献常用默认值,且**不针对冻结评测集调参**:
#: S2-F2 的敏感性检查显示 n=2 时 Recall 反而更高(0.90625),即当前配置
#: 不是网格最优点 —— 这正是不该按评测集调参的理由。
BM25_K1 = 1.5
BM25_B = 0.75

#: 字符 n-gram 长度。与 `HashingEmbedder` 的默认值一致:两条路径共享同一
#: 种分词,分数才可比。中文没有空格分词,字符窗口是唯一不需要词典的选择。
BM25_NGRAM = 3


def _normalize_lexical(text: str) -> str:
    """折叠大小写与空白。与嵌入层的规范化一致(由测试钉住)。"""
    return " ".join(text.lower().split())


def _char_ngrams(text: str, n: int) -> tuple[str, ...]:
    """字符 n-gram;短于 n 的文本整体作为一个 gram(不是空元组)。"""
    if not text:
        return ()
    if len(text) <= n:
        return (text,)
    return tuple(text[index : index + n] for index in range(len(text) - n + 1))


def _query_terms(query: str, n: int) -> tuple[str, ...]:
    """查询词项:**去重**且保留首次出现顺序。"""
    seen: dict[str, None] = {}
    for gram in _char_ngrams(_normalize_lexical(query), n):
        seen.setdefault(gram, None)
    return tuple(seen)


def _ordered_documents(
    documents: Iterable[KnowledgeDocument],
) -> list[KnowledgeDocument]:
    """按 `doc_id` 升序排列,并拒绝重复 id 与空集合。

    重复 `doc_id` 会产出两套相同的 `chunk_id`,文档级去重就会静默吞掉其中
    一篇 —— 这类"看起来正常但少了一篇"的结果比报错更难发现,所以在这里
    直接 fail closed。
    """
    ordered = sorted(documents, key=lambda document: document.doc_id)
    seen: set[str] = set()
    for document in ordered:
        if document.doc_id in seen:
            raise CorpusError(
                f"duplicate doc_id {document.doc_id!r}", reason="duplicate-doc-id"
            )
        seen.add(document.doc_id)
    if not ordered:
        raise CorpusError("knowledge corpus is empty", reason="empty-corpus")
    return ordered


class BM25Index:
    """块级 BM25 倒排索引(纯内存)。

    倒排表按块位置升序保存 `(position, tf)`,因此打分时的浮点累加顺序
    只由查询词项顺序与块顺序决定,不依赖字典遍历顺序。
    """

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        ngram: int = BM25_NGRAM,
    ) -> None:
        if k1 < 0.0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be within [0, 1]")
        if ngram < 1:
            raise ValueError("ngram must be >= 1")

        self._chunks = tuple(chunks)
        self._k1 = float(k1)
        self._b = float(b)
        self._ngram = ngram

        lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for position, chunk in enumerate(self._chunks):
            grams = _char_ngrams(_normalize_lexical(chunk.text), ngram)
            lengths.append(len(grams))
            counts: dict[str, int] = {}
            for gram in grams:
                counts[gram] = counts.get(gram, 0) + 1
            for gram, tf in counts.items():
                postings.setdefault(gram, []).append((position, tf))

        self._lengths = tuple(lengths)
        self._postings: dict[str, tuple[tuple[int, int], ...]] = {
            gram: tuple(entries) for gram, entries in postings.items()
        }
        self._chunk_count = len(self._chunks)
        self._avgdl = (
            sum(lengths) / self._chunk_count if self._chunk_count else 0.0
        )
        self._idf_cache: dict[str, float] = {}

    # ---- 只读属性 ----

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def avgdl(self) -> float:
        return self._avgdl

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    @property
    def ngram(self) -> int:
        return self._ngram

    # ---- 打分 ----

    def idf(self, gram: str) -> float:
        """`log(1 + (N - df + 0.5) / (df + 0.5))`。未见过的 gram 记 0。"""
        cached = self._idf_cache.get(gram)
        if cached is None:
            df = len(self._postings.get(gram, ()))
            if df == 0:
                cached = 0.0
            else:
                cached = math.log(
                    1.0 + (self._chunk_count - df + 0.5) / (df + 0.5)
                )
            self._idf_cache[gram] = cached
        return cached

    def search(self, query: str) -> list[ScoredChunk]:
        """对所有块精确打分,返回**正分**块(按 `(-score, chunk_id)`)。

        不设 `limit`:截断必须发生在文档级去重之后(见模块文档第 3 条)。
        """
        terms = _query_terms(query, self._ngram)
        if not terms:
            raise RetrievalError("query must be non-empty", reason="empty-query")

        scores = [0.0] * self._chunk_count
        for gram in terms:
            postings = self._postings.get(gram)
            if postings is None:
                continue
            idf = self.idf(gram)
            for position, tf in postings:
                length_norm = 1.0 - self._b + self._b * (
                    self._lengths[position] / self._avgdl
                )
                denominator = tf + self._k1 * length_norm
                scores[position] += idf * tf * (self._k1 + 1.0) / denominator

        scored = [
            ScoredChunk(chunk=self._chunks[position], score=score)
            for position, score in enumerate(scores)
            if score > 0.0
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored

    # ---- 装配 ----

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[KnowledgeDocument],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        ngram: int = BM25_NGRAM,
    ) -> "BM25Index":
        """切块 + 建倒排。切块走 `chunk_corpus`,与向量路径同一份实现。"""
        return cls(
            chunk_corpus(_ordered_documents(documents)),
            k1=k1,
            b=b,
            ngram=ngram,
        )


class BM25Retriever:
    """BM25 检索器 —— 与 `Retriever` 相同的结果契约,不同的打分方式。

    刻意**没有** `verify()`:BM25 索引不落盘,它在构造期由同一份文档现场
    建立,不存在"索引比语料旧"的窗口。为对称而造一个恒真的 `verify()`
    只会让"校验过了"这句话失去含义。
    """

    def __init__(
        self,
        documents: Iterable[KnowledgeDocument],
        *,
        corpus_version: str = "",
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_HIT_SCORE,
        k1: float = BM25_K1,
        b: float = BM25_B,
        ngram: int = BM25_NGRAM,
    ) -> None:
        ordered = _ordered_documents(documents)
        self._documents = {document.doc_id: document for document in ordered}
        self._index = BM25Index.from_documents(ordered, k1=k1, b=b, ngram=ngram)
        self._corpus_version = corpus_version
        self._top_k = clamp_top_k(top_k)
        self._min_score = float(min_score)

    # ---- 只读属性 ----

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def min_score(self) -> float:
        return self._min_score

    @property
    def chunk_count(self) -> int:
        return self._index.chunk_count

    # ---- 检索 ----

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        normalized = normalize_query(query)
        if not normalized:
            raise RetrievalError("query must be non-empty", reason="empty-query")

        depth = clamp_top_k(top_k if top_k is not None else self._top_k)
        candidates = self._index.search(normalized)
        hits = hits_from_candidates(
            candidates,
            self._documents,
            depth=depth,
            min_score=self._min_score,
        )

        return RetrievalResult(
            status="ok" if hits else "empty",
            query_sha256=sha256_hex(normalized),
            top_k=depth,
            hits=hits,
            corpus_version=self._corpus_version,
        )


def build_bm25_retriever(
    corpus: LoadedCorpus,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> BM25Retriever:
    """语料 → 切块 → 倒排 → 检索器 的离线装配。

    与 `build_in_memory_retriever` 并列存在而不是二选一:本阶段 BM25 只是
    **候选**,默认后端仍是向量路径,装配点也不在组合根。
    """
    return BM25Retriever(
        corpus.documents,
        corpus_version=corpus.manifest.corpus_version,
        top_k=top_k,
    )
