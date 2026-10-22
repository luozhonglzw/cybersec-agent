"""离线检索底座(Threat Intelligence RAG 的第一层)。

本包刻意**不依赖** `app.core` / `app.api` / `app.security` / `app.tools` /
`app.evaluation`:

- 检索不是授权路径。它不该能读到策略、审计、checkpoint 的任何符号 ——
  这是"检索内容无法影响安全判定"的**结构性**前提,而不是纪律要求;
- 反向依赖同样禁止:安全层不感知检索。工具层(S2-B)才是唯一接线点。

包内分层(自下而上):

| 模块 | 职责 | 有无 I/O |
|---|---|---|
| `types` | 数据契约与校验 | 无 |
| `errors` | 异常层级(通用文案 + 原因码) | 无 |
| `chunking` | 确定性切块 | 无 |
| `embedding` | 嵌入契约 + 离线确定性基线 | 无 |
| `store` | 向量存储契约 + 内存实现 | 无 |
| `corpus` | 语料与清单加载 | **只读本地文件** |
| `retriever` | 完整性校验 + 确定性 top-k | 无 |
| `bm25` | 词法检索候选(字符 3-gram BM25) | 无 |

`HashingEmbedder` 是可复现的**基线**,不是语义模型;它产出的质量数字
只描述该基线。`bm25` 同样只是**词法**候选:它不消除"零共享 3-gram"
这一类未命中,也**不是** `/ask` 的默认后端。
"""
from app.rag.bm25 import (
    BM25_B,
    BM25_K1,
    BM25_NGRAM,
    BM25Index,
    BM25Retriever,
    build_bm25_retriever,
)
from app.rag.chunking import (
    CHUNKER_VERSION,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    chunk_corpus,
    chunk_document,
)
from app.rag.corpus import (
    APPLICATION_ROOT,
    CORPUS_FILENAME,
    DEFAULT_KNOWLEDGE_DIR,
    MANIFEST_FILENAME,
    LoadedCorpus,
    corpus_sha256,
    load_corpus,
    parse_documents,
    read_manifest,
)
from app.rag.embedding import (
    DEFAULT_HASHING_DIM,
    DEFAULT_HASHING_NGRAM,
    Embedder,
    HashingEmbedder,
    cosine_similarity,
)
from app.rag.errors import (
    CorpusError,
    EmbeddingUnavailable,
    IndexUnavailable,
    RagError,
    RetrievalError,
    StoreError,
)
from app.rag.retriever import (
    CANDIDATE_MULTIPLIER,
    MIN_HIT_SCORE,
    SNIPPET_MAX_CHARS,
    Retriever,
    SupportsRetrieve,
    build_in_memory_retriever,
    build_index_metadata,
    hit_from_scored,
    hits_from_candidates,
    verify_index,
)
from app.rag.store import InMemoryVectorStore, VectorStore, check_vector
from app.rag.types import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    CorpusManifest,
    IndexMetadata,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalHit,
    RetrievalResult,
    ScoredChunk,
    SourceProvenance,
    clamp_top_k,
    derive_doc_id,
    normalize_query,
    sha256_hex,
    snippet_of,
    sorted_by_score,
)

__all__ = [
    "APPLICATION_ROOT",
    "BM25_B",
    "BM25_K1",
    "BM25_NGRAM",
    "BM25Index",
    "BM25Retriever",
    "CANDIDATE_MULTIPLIER",
    "CHUNKER_VERSION",
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "CORPUS_FILENAME",
    "DEFAULT_HASHING_DIM",
    "DEFAULT_HASHING_NGRAM",
    "DEFAULT_KNOWLEDGE_DIR",
    "DEFAULT_TOP_K",
    "MANIFEST_FILENAME",
    "MAX_TOP_K",
    "MIN_HIT_SCORE",
    "SNIPPET_MAX_CHARS",
    "CorpusError",
    "CorpusManifest",
    "Embedder",
    "EmbeddingUnavailable",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "IndexMetadata",
    "IndexUnavailable",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "LoadedCorpus",
    "RagError",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalResult",
    "Retriever",
    "ScoredChunk",
    "SourceProvenance",
    "StoreError",
    "SupportsRetrieve",
    "VectorStore",
    "build_bm25_retriever",
    "build_in_memory_retriever",
    "build_index_metadata",
    "check_vector",
    "chunk_corpus",
    "chunk_document",
    "clamp_top_k",
    "corpus_sha256",
    "cosine_similarity",
    "derive_doc_id",
    "hit_from_scored",
    "hits_from_candidates",
    "load_corpus",
    "normalize_query",
    "parse_documents",
    "read_manifest",
    "sha256_hex",
    "snippet_of",
    "sorted_by_score",
    "verify_index",
]
