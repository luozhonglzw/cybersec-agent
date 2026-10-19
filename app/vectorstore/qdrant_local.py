"""Qdrant Local 向量存储适配器(S2-D)。

为什么这个包**不**在 `app/rag/` 里面
------------------------------------
`app/rag/**` 有一条冻结的结构断言(`tests/test_rag/test_security.py`):

- 不 import 任何网络模块(`socket` / `httpx` / `urllib` / `grpc` …);
- 只有 `corpus.py` 碰文件系统;
- 不 import 授权层 / 工具层 / 评测层。

`qdrant-client` 是一个**网络能力齐备的 SDK**(内部依赖 `httpx[http2]` 与
`grpcio`,并且 Local 模式要读写存储目录)。把它放进 `app/rag/**`,那条
"检索底座不会发请求"的结构性证明就从"成立"退化成"暂时没人调用"。

因此适配器单独一层:它**依赖** `app.rag`(契约、类型、错误),而 `app.rag`
对它一无所知。方向是单向的,`app/rag` 的边界断言因此仍然成立且可机械验证。

它做什么
--------
实现 `app.rag.store.VectorStore` 协议(`upsert` / `search` / `metadata`),
让 `app.rag.retriever.Retriever` **一行不改**就能跑在真实 Qdrant Local 上。
确定性排序、按 `doc_id` 去重、零分不算命中、索引完整性校验,全部仍由
`Retriever` 负责 —— 换后端不改变可观测行为,这是 S2-A 就定下的分工。

三条硬约束
----------

**1. 固定集合名,没有选择器。** `COLLECTION_NAME` 是模块常量,构造函数
不接受集合名参数。模型与客户端都无法指定"从哪个集合读"。

**2. 显式存储目录,拒绝隐式网络。** 只构造 `QdrantClient(path=...)`。
代码里**不出现** `url=` / `host=` / `port=` / `api_key=` —— 没有远程模式
这条代码路径,也就没有"某次配置失误导致生产环境连上外部 Qdrant"的可能。

**3. 坏的持久索引绝不静默降级。** `open()` 的每一步失败都抛
`IndexUnavailable`:目录缺失、集合缺失、元数据缺失/非法、维度不符、
距离函数不符、点数与 `chunk_count` 不符、目录被另一实例锁定。**没有**
"打不开就退回内存索引"的分支 —— 那会让"检索用的是哪份数据"变成运行时
猜测。

Qdrant Local 的并发事实(实测,非假设)
--------------------------------------
`qdrant-client 1.19.1` 的 Local 模式对存储目录持有**排他锁**(portalocker)。
同一路径上开第二个 client 直接抛:

    RuntimeError: Storage folder ... is already accessed by another instance
    of Qdrant client.

因此:

- 一个存储目录在**任一时刻只能有一个** client;本适配器把这种冲突翻译成
  `IndexUnavailable(reason="index-locked")`,而不是重试或等待 —— 重试会把
  "配置错了"伪装成"有点慢";
- 客户端**必须** `close()`(或用作上下文管理器)。不关会让 `.lock` 残留到
  进程退出,下一个进程打不开;
- 正因为持有排他锁,`metadata()` 可以在 `open()` 时读一次并缓存:在锁的
  有效期内集合不可能被别的进程改掉。`metadata()` 返回的是**索引身份**,
  不是实时点数 —— 点数由 `count()` 单独提供,构建路径在写入后自己核对
  两者一致。
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from app.rag.chunking import chunk_corpus
from app.rag.corpus import LoadedCorpus
from app.rag.embedding import Embedder
from app.rag.errors import IndexUnavailable, StoreError
from app.rag.retriever import Retriever, build_index_metadata
from app.rag.store import check_vector
from app.rag.types import (
    DEFAULT_TOP_K,
    IndexMetadata,
    KnowledgeChunk,
    ScoredChunk,
)

#: 固定集合名。**不接受参数** —— 这是"没有集合选择器"的落地方式。
COLLECTION_NAME = "cybersec_knowledge"

#: 索引格式版本。进集合元数据;格式一变,旧索引立刻被判为不可用,
#: 而不是用"旧布局的 payload"拼出看起来合法的 `KnowledgeChunk`。
INDEX_FORMAT = "qdrant-local-v1"

#: `chunk_id → point id` 的 UUIDv5 命名空间。
#:
#: 为什么需要映射:Qdrant 的 point id 只接受无符号整数或 UUID,而
#: `chunk_id` 形如 `kb-cve-2021-44228#0`。用 UUIDv5(名字空间 + 名字)而不是
#: `hash()`:后者对 `str` 加了进程级随机盐,同一份语料两次启动会得到两套
#: point id —— "重建索引"就变成"换了一套索引",而语料哈希一模一样。
#: 这个常量**不得修改**,改了等于所有既有索引失效。
POINT_ID_NAMESPACE = uuid.UUID("1d0f9a52-6c4b-5e7a-9a10-2b8c3d4e5f60")

#: 单次 upsert 的批大小。Local 模式是内存 + 单文件写入,批太大只会让
#: 一次失败的爆炸半径变大;批太小则写放大。40 篇语料约 120 块,一批就完。
UPSERT_BATCH_SIZE = 256

#: 仓库根目录(应用相对)。`app/vectorstore/qdrant_local.py` → parents[2]。
APPLICATION_ROOT = Path(__file__).resolve().parents[2]

#: 默认索引目录 —— **应用相对**,不依赖进程 CWD。
#:
#: 与 `app.rag.corpus.DEFAULT_KNOWLEDGE_DIR` 同一理由:相对路径会让
#: "索引在哪"取决于从哪个目录启动 pytest / uvicorn,而这类差异只在
#: 换机器或换启动方式时暴露。
#:
#: 该目录落在 `data/` 下(`data/.gitignore` 是 `*`),因此生成物**不会**
#: 被提交。部署若要把索引放到数据卷上,用 `--index-dir` 显式指定。
DEFAULT_INDEX_DIR = APPLICATION_ROOT / "data" / "knowledge_index"

#: 集合元数据里的键。与 `IndexMetadata` 字段一一对应,另加 `index_format`。
_METADATA_FORMAT_KEY = "index_format"


def _metadata_payload(metadata: IndexMetadata) -> dict[str, Any]:
    """`IndexMetadata` → 可存进集合配置的字典(纯数据,可 JSON 序列化)。"""
    return {
        _METADATA_FORMAT_KEY: INDEX_FORMAT,
        "corpus_version": metadata.corpus_version,
        "corpus_sha256": metadata.corpus_sha256,
        "embedder_id": metadata.embedder_id,
        "dim": metadata.dim,
        "chunker_version": metadata.chunker_version,
        "doc_count": metadata.doc_count,
        "chunk_count": metadata.chunk_count,
    }


def _metadata_from_payload(payload: Any) -> IndexMetadata:
    """集合配置 → `IndexMetadata`。任何形态异常都 fail closed。

    刻意**不**做字段级容错:缺一个字段就说明这份索引不是本代码写的
    (或写坏了),补默认值只会让一份来路不明的索引看起来合法。
    """
    if not isinstance(payload, dict):
        raise IndexUnavailable(
            "knowledge index has no identity metadata",
            reason="index-metadata-missing",
        )
    if payload.get(_METADATA_FORMAT_KEY) != INDEX_FORMAT:
        raise IndexUnavailable(
            "knowledge index format is not supported",
            reason="index-format",
        )
    fields = {
        key: value
        for key, value in payload.items()
        if key != _METADATA_FORMAT_KEY
    }
    try:
        return IndexMetadata.model_validate(fields)
    except ValidationError as exc:
        raise IndexUnavailable(
            "knowledge index identity is invalid",
            reason="index-metadata-invalid",
        ) from exc


def point_id_of(chunk_id: str) -> str:
    """`chunk_id` → 确定性 point id(UUIDv5 字符串)。"""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def _chunk_payload(chunk: KnowledgeChunk) -> dict[str, Any]:
    """`KnowledgeChunk` → payload。

    存全量而不是只存 `chunk_id`:检索结果要重建 `KnowledgeChunk`(它的
    校验器会核对 `chunk_id == f"{doc_id}#{ordinal}"` 与偏移/长度一致),
    而这些字段只存在于构建期。只存 id 就得回查另一份数据源 —— 那又是一处
    可能漂移的真相源。
    """
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


def _chunk_from_payload(payload: Any) -> KnowledgeChunk:
    """payload → `KnowledgeChunk`。坏 payload 不跳过,而是整体 fail closed。"""
    if not isinstance(payload, dict):
        raise IndexUnavailable(
            "knowledge index contains a malformed chunk payload",
            reason="index-payload-invalid",
        )
    try:
        return KnowledgeChunk.model_validate(payload)
    except ValidationError as exc:
        raise IndexUnavailable(
            "knowledge index contains a malformed chunk payload",
            reason="index-payload-invalid",
        ) from exc


class QdrantLocalVectorStore:
    """`VectorStore` 协议的 Qdrant Local 实现。

    生命周期:一个存储目录同一时刻只能有一个实例(排他锁)。用完必须
    `close()`,或写成 `with QdrantLocalVectorStore.open(...) as store:`。
    """

    def __init__(
        self,
        client: QdrantClient,
        metadata: IndexMetadata,
        *,
        index_dir: Path,
    ) -> None:
        self._client = client
        self._metadata = metadata
        self._index_dir = index_dir
        self._closed = False

    # ---- 构造 ----

    @classmethod
    def create(
        cls,
        index_dir: Path | str,
        metadata: IndexMetadata,
        *,
        replace: bool = False,
    ) -> "QdrantLocalVectorStore":
        """新建(或按 `replace` 重建)一个空集合,身份元数据**暂不提交**。

        为什么不在这一步写身份(S2-G1 修正)
        ------------------------------------
        集合配置里先放**故意不合法**的哨兵(`index_format` 一项,缺全部身份
        字段),等调用方把点写完、`count` 自检通过之后再调 `commit()` 写入
        真正的身份。于是"元数据已更新、点还是旧的"这个窗口**不存在**:

        - 崩溃在 `commit()` 之前 ⇒ 配置里只有哨兵 ⇒ 下一次 `open()` 被
          既有 `_metadata_from_payload` 以 `index-metadata-invalid` 拒绝;
        - 崩溃在 `commit()` 之后 ⇒ 点已全部落盘且已核对,索引完整。

        修正前的顺序(先写身份、再清点、再写入)会让"只改正文、doc_id 不变"
        的重建在窗口内崩溃时留下**看起来完全合法**的索引:六项身份校验全过,
        而向量还是上一版语料的 —— 检索会把它当作有效证据返回。

        身份仍写进**集合配置**而不是旁边的 JSON 文件:配置与集合是同一份
        存储的组成部分,不存在"元数据说 N 块、实际只有 M 块"的文件级竞争。

        `replace=False` 时若集合已存在 → `IndexUnavailable`。**不**默认覆盖:
        覆盖是破坏性动作,必须由调用方(构建脚本)显式要求。
        """
        directory = Path(index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        client = cls._connect(directory)
        try:
            existing = {item.name for item in client.get_collections().collections}
            if COLLECTION_NAME in existing:
                if not replace:
                    raise IndexUnavailable(
                        "knowledge index already exists",
                        reason="index-already-exists",
                    )
                client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=metadata.dim, distance=Distance.COSINE
                ),
                metadata={_METADATA_FORMAT_KEY: INDEX_FORMAT},
            )
            cls._reset_points(client)
        except Exception:
            client.close()
            raise
        return cls(client, metadata, index_dir=directory)

    def commit(self) -> None:
        """把索引身份写进集合配置 —— 重建的**提交点**。

        只在"点已全部写入且 `count` 自检通过"之后调用。之前配置里只有
        哨兵,所以任何在此之前的崩溃都会让下一次 `open()` fail closed,
        而不会留下一份"身份说新语料、内容还是旧语料"的索引。

        `update_collection(metadata=...)` 在 Qdrant Local 实测可用,且是
        **整字典替换**(哨兵里的 `index_format` 会随新字典一起写回)。
        这不是事务提交:它只保证"身份最后可见",不保证"点与身份同时生效"。
        """
        self._client.update_collection(
            collection_name=COLLECTION_NAME,
            metadata=_metadata_payload(self._metadata),
        )

    @staticmethod
    def _reset_points(client: QdrantClient) -> None:
        """清空新建集合的点,并**验证**清空成功。

        为什么需要这一步(实测事实,不是防御性冗余)
        ------------------------------------------
        `qdrant-client 1.19.1` 的 Local 模式里,`delete_collection()` 只把集合
        从 `meta.json` 移除,**磁盘上的 `collection/<name>/` 数据目录会保留**。
        随后用同名 `create_collection()` 会把那份残留目录**重新挂上**,于是
        旧点全部"复活":

            建好 80 块 → delete_collection → create_collection → count() == 80

        对重建来说这是**真实的正确性危险**,不只是噪声:若新语料块数恰好与
        旧语料相同而文档集合不同,残留的旧点会留在索引里被检索出来 ——
        等于"删掉的文档还在回答问题"。

        所以重建路径必须显式清空。`Filter(must=[])` 是 Qdrant 语义下的
        "匹配全部点"。清空之后**再数一遍**:若不为 0,说明这条语义在某次
        升级后变了,此时响亮失败(`index-not-clean`)远好于静默带着残留点
        上线。
        """
        client.delete(collection_name=COLLECTION_NAME, points_selector=Filter(must=[]))
        remaining = client.count(COLLECTION_NAME, exact=True).count
        if remaining != 0:
            raise IndexUnavailable(
                "knowledge index is not empty after reset",
                reason="index-not-clean",
            )

    @classmethod
    def open(
        cls,
        index_dir: Path | str,
        *,
        expected_dim: int | None = None,
    ) -> "QdrantLocalVectorStore":
        """打开一个**已存在**的索引。任何不满足都 fail closed。

        刻意**不**自动创建:把"索引不存在"变成"悄悄建一个空索引"会让
        检索永远返回 `empty`,而调用方以为知识库是空的 —— 那是比报错
        更糟的失败。
        """
        directory = Path(index_dir)
        if not directory.is_dir():
            raise IndexUnavailable(
                "knowledge index is not present", reason="index-missing"
            )
        client = cls._connect(directory)
        try:
            names = {item.name for item in client.get_collections().collections}
            if COLLECTION_NAME not in names:
                raise IndexUnavailable(
                    "knowledge index collection is not present",
                    reason="index-missing",
                )
            info = client.get_collection(COLLECTION_NAME)
            metadata = _metadata_from_payload(info.config.metadata)
            cls._check_vector_params(info, metadata)
            if expected_dim is not None and metadata.dim != expected_dim:
                raise IndexUnavailable(
                    "knowledge index dimension does not match the embedder",
                    reason="index-dim",
                )
            stored = client.count(COLLECTION_NAME, exact=True).count
            if stored != metadata.chunk_count:
                raise IndexUnavailable(
                    "knowledge index is incomplete",
                    reason="index-incomplete",
                )
        except Exception:
            client.close()
            raise
        return cls(client, metadata, index_dir=directory)

    @staticmethod
    def _connect(directory: Path) -> QdrantClient:
        """只以 **Local 模式**连接。没有 url / host 分支。

        锁冲突被翻译成 `IndexUnavailable`:Qdrant Local 的排他锁是实测
        事实(见模块文档),并发写并不安全,所以不重试、不等待。
        """
        try:
            return QdrantClient(path=str(directory))
        except RuntimeError as exc:
            if "already accessed" in str(exc):
                raise IndexUnavailable(
                    "knowledge index is locked by another process",
                    reason="index-locked",
                ) from exc
            raise IndexUnavailable(
                "knowledge index cannot be opened",
                reason="index-open-failed",
            ) from exc
        except OSError as exc:
            raise IndexUnavailable(
                "knowledge index cannot be opened",
                reason="index-open-failed",
            ) from exc

    @staticmethod
    def _check_vector_params(info: Any, metadata: IndexMetadata) -> None:
        """集合的向量参数必须与元数据自洽。

        这一条挡的是"元数据被改过"与"集合是用别的参数建的"两类不一致 ——
        单看任何一方都发现不了。
        """
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        if size != metadata.dim:
            raise IndexUnavailable(
                "knowledge index vector size does not match its identity",
                reason="index-dim",
            )
        if distance != Distance.COSINE:
            raise IndexUnavailable(
                "knowledge index distance is not cosine",
                reason="index-distance",
            )

    # ---- 契约实现 ----

    def upsert(
        self,
        chunks: Sequence[KnowledgeChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """写入或替换切块向量。同一 `chunk_id` 的 point id 恒定,故为替换。"""
        if len(chunks) != len(vectors):
            raise StoreError(
                "chunks and vectors must have the same length",
                reason="length-mismatch",
            )
        for chunk, vector in zip(chunks, vectors):
            check_vector(vector, self._metadata.dim, label=f"chunk {chunk.chunk_id}")

        points = [
            PointStruct(
                id=point_id_of(chunk.chunk_id),
                vector=list(vector),
                payload=_chunk_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[start : start + UPSERT_BATCH_SIZE],
            )

    def search(self, vector: Sequence[float], limit: int) -> list[ScoredChunk]:
        """返回至多 `limit` 个候选。**不保证同分顺序**(契约如此)。

        这里仍然按 `(-score, chunk_id)` 排一遍:契约允许存储层无序,但
        让返回顺序稳定可以让"跨后端的差异"只可能来自候选集合,而不是
        来自返回顺序。真正的确定性截断仍在 `Retriever` 里。
        """
        check_vector(vector, self._metadata.dim, label="query")
        if limit <= 0:
            return []
        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=list(vector),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        scored = [
            ScoredChunk(chunk=_chunk_from_payload(point.payload), score=float(point.score))
            for point in response.points
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored

    def metadata(self) -> IndexMetadata:
        """索引身份。`open()` 时读一次并缓存 —— 持有排他锁期间不会漂移。"""
        return self._metadata

    # ---- 生命周期与便利方法(非契约) ----

    def close(self) -> None:
        """关闭客户端并释放存储目录锁。幂等。"""
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def __enter__(self) -> "QdrantLocalVectorStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def index_dir(self) -> Path:
        """本实例的存储目录。**仅供诊断**,不参与检索。"""
        return self._index_dir

    def count(self) -> int:
        """集合内 point 数。供构建脚本自检与测试使用。"""
        return self._client.count(COLLECTION_NAME, exact=True).count

    def chunk_ids(self) -> list[str]:
        """已写入的 `chunk_id`(升序)—— 供测试核对重建幂等性。"""
        ids: list[str] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=COLLECTION_NAME,
                limit=UPSERT_BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            ids.extend(_chunk_from_payload(point.payload).chunk_id for point in points)
            if offset is None:
                break
        return sorted(ids)


def build_qdrant_retriever(
    corpus: LoadedCorpus,
    embedder: Embedder,
    index_dir: Path | str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> Retriever:
    """打开既有 Qdrant Local 索引,组装成与内存后端同形的 `Retriever`。

    与 `build_in_memory_retriever` 的关系是**同构而非替换**:两者产出同一个
    `Retriever` 类型,走同一套 `verify()` 与同一套确定性 top-k。调用方的
    差别只有一处 —— 返回的检索器持有一个**必须关闭**的存储(排他锁)。
    用 `retriever.store.close()`,或优先用 `open_qdrant_retriever()`。

    校验失败时会把已打开的 client 关掉再抛:`open()` 成功、`verify()` 失败
    的路径如果不关,锁会残留到进程退出。
    """
    store = QdrantLocalVectorStore.open(index_dir, expected_dim=embedder.dim)
    try:
        retriever = Retriever(
            embedder,
            store,
            {document.doc_id: document for document in corpus.documents},
            manifest=corpus.manifest,
            top_k=top_k,
        )
        # 提前校验:语料版本/哈希、嵌入器身份、维度、切块器、文档数。
        retriever.verify()
    except Exception:
        store.close()
        raise
    return retriever


def build_qdrant_index(
    corpus: LoadedCorpus,
    embedder: Embedder,
    index_dir: Path | str,
    *,
    replace: bool = False,
) -> IndexMetadata:
    """把语料切块、嵌入并写入一个新的 Qdrant Local 索引,返回索引身份。

    为什么先把全部向量算完再动存储:嵌入是唯一可能"跑很久后失败"的步骤。
    先算完,失败时旧索引**原封不动**;随后才进入"删集合 → 建集合 → 写入 →
    提交身份"这段快速窗口。

    窗口内的顺序是 S2-G1 的**核心不变量**:
    `create(哨兵身份)` → `_reset_points` → `upsert` → `count` 自检 →
    `commit(真实身份)`。身份最后写,因此窗口内任何一步失败都留下一个
    **哨兵索引**,下一次 `open()` 被既有校验以 `index-metadata-invalid`
    拒绝 —— 包括"只改正文、doc_id 不变"这种否则会静默陈旧的场景。

    这不宣称事务原子性:点与身份仍是两次写入,只是**身份最后可见**。
    """
    chunks = chunk_corpus(corpus.documents)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    metadata = build_index_metadata(corpus, embedder, chunks)

    with QdrantLocalVectorStore.create(index_dir, metadata, replace=replace) as store:
        store.upsert(chunks, vectors)
        stored = store.count()
        if stored != metadata.chunk_count:
            # 写入后立刻自检:让"部分写入"在这里就暴露,而不是等到下一次
            # `open()` 才由别的进程发现。
            raise IndexUnavailable(
                "knowledge index write is incomplete",
                reason="index-incomplete",
            )
        # 提交点:至此点已全部落盘且数量核对通过,才让身份对外可见。
        store.commit()
    return metadata


__all__ = [
    "APPLICATION_ROOT",
    "COLLECTION_NAME",
    "DEFAULT_INDEX_DIR",
    "INDEX_FORMAT",
    "POINT_ID_NAMESPACE",
    "QdrantLocalVectorStore",
    "UPSERT_BATCH_SIZE",
    "build_qdrant_index",
    "build_qdrant_retriever",
    "point_id_of",
]
