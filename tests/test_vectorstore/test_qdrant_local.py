"""Qdrant Local 适配器测试 —— **真实** Qdrant Local,不是替身。

为什么不用 mock
---------------
本阶段要证明的是"真实向量库能承载既有契约":索引能建、能查、能关能开、
维度不符会拒、目录被锁会拒、坏索引不会静默降级。用 mock 换掉
`QdrantClient`,这些性质就全变成"我实现的 mock 同意我"。

为什么不需要服务端 / Docker / GPU
--------------------------------
`QdrantClient(path=...)` 是**嵌入式**模式:数据落在本地目录,进程内计算
余弦相似度。它不需要 Qdrant 服务端、不需要容器、不需要网络、不需要 GPU。

全部用例建在 `tmp_path`,不读也不写仓库 `data/`。
"""
from __future__ import annotations

import inspect
import uuid
from pathlib import Path

import pytest

from app.rag import (
    HashingEmbedder,
    IndexUnavailable,
    KnowledgeChunk,
    Retriever,
    StoreError,
    build_in_memory_retriever,
    chunk_corpus,
)
from app.rag.types import IndexMetadata
from app.vectorstore import (
    COLLECTION_NAME,
    QdrantLocalVectorStore,
    build_qdrant_index,
    build_qdrant_retriever,
    point_id_of,
)
from app.vectorstore import qdrant_local


def _recreate_collection(
    index_dir: Path,
    payload: dict | None,
    *,
    dim: int = 256,
    distance=None,
) -> None:
    """在既有存储目录上重建集合,用于制造各种"坏索引"。

    **必须显式清空点**:Qdrant Local 的 `delete_collection()` 只把集合从
    `meta.json` 移除,磁盘上的 `collection/<name>/` 目录会保留;同名
    `create_collection()` 会把残留目录重新挂上,旧点因此"复活"(实测:
    80 块 → 删除 → 同名重建 → `count() == 80`)。用例只想验证某一条
    元数据/维度不一致,不该被这个行为干扰。
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, Filter, VectorParams

    client = QdrantClient(path=str(index_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dim, distance=distance or Distance.COSINE
            ),
            metadata=payload,
        )
        client.delete(
            collection_name=COLLECTION_NAME, points_selector=Filter(must=[])
        )
    finally:
        client.close()


def _metadata_payload(metadata: IndexMetadata) -> dict:
    payload = dict(metadata.model_dump())
    payload["index_format"] = qdrant_local.INDEX_FORMAT
    return payload


# =====================================================================
# 1. 索引构建
# =====================================================================


def test_build_creates_the_fixed_collection(built_index):
    index_dir, metadata = built_index
    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.count() == metadata.chunk_count
        assert store.metadata() == metadata


def test_build_writes_every_chunk_exactly_once(built_index, corpus):
    index_dir, _ = built_index
    expected = [chunk.chunk_id for chunk in chunk_corpus(corpus.documents)]
    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.chunk_ids() == sorted(expected)


def test_index_identity_matches_the_shared_builder(built_index, corpus, embedder):
    """索引身份只能有一处来源 —— 与内存后端走同一个 `build_index_metadata`。"""
    _, metadata = built_index
    chunks = chunk_corpus(corpus.documents)
    assert metadata.doc_count == len(corpus.documents)
    assert metadata.chunk_count == len(chunks)
    assert metadata.dim == embedder.dim
    assert metadata.embedder_id == embedder.embedder_id
    assert metadata.corpus_sha256 == corpus.manifest.corpus_sha256


def test_point_ids_are_deterministic_and_unique(corpus):
    """UUIDv5 而不是 `hash()`:后者带进程级盐,两次启动会得到两套 id。"""
    chunks = chunk_corpus(corpus.documents)
    ids = [point_id_of(chunk.chunk_id) for chunk in chunks]
    assert len(set(ids)) == len(ids), "point id 冲突"
    # 同一次进程内重算必须相同
    assert ids == [point_id_of(chunk.chunk_id) for chunk in chunks]
    # 必须是合法 UUID(Qdrant 的 point id 只接受整数或 UUID)
    for value in ids:
        uuid.UUID(value)


def test_rebuilding_produces_the_same_point_ids(index_dir, corpus, embedder):
    """重复构建的 point id 必须逐字相同 —— 否则"重建"等于"换了一套索引"。"""
    build_qdrant_index(corpus, embedder, index_dir)
    with QdrantLocalVectorStore.open(index_dir) as store:
        first = store.chunk_ids()

    build_qdrant_index(corpus, embedder, index_dir, replace=True)
    with QdrantLocalVectorStore.open(index_dir) as store:
        second = store.chunk_ids()

    assert first == second


def test_rebuilding_does_not_resurrect_points_from_the_previous_index(
    index_dir, corpus, embedder
):
    """重建必须得到"恰好等于当前语料"的索引,不能残留上一份的点。

    这条防的是一个**实测存在**的 Qdrant Local 行为:`delete_collection()`
    只移除 `meta.json` 里的条目,磁盘数据目录保留;同名重建会把旧点挂回来。
    若不显式清空,重建后的索引会同时包含新旧两份内容 —— 被删掉的文档
    仍然会被检索出来,而所有元数据看起来都是对的。
    """
    from app.rag import LoadedCorpus

    build_qdrant_index(corpus, embedder, index_dir)
    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.count() == 80

    # 只留一篇文档的"新语料"
    smaller = LoadedCorpus(
        manifest=corpus.manifest.model_copy(
            update={"doc_count": 1, "corpus_version": "v0.5.0-s2.99"}
        ),
        documents=corpus.documents[:1],
    )
    build_qdrant_index(smaller, embedder, index_dir, replace=True)

    with QdrantLocalVectorStore.open(index_dir) as store:
        ids = store.chunk_ids()
        assert store.count() == len(ids)
        assert len(ids) < 80, "重建后仍带着旧语料的点"
        assert {chunk.split("#")[0] for chunk in ids} == {smaller.documents[0].doc_id}


# =====================================================================
# 2. 写入与检索
# =====================================================================


def test_search_returns_scored_chunks_with_intact_payload(built_index, embedder):
    index_dir, _ = built_index
    vector = embedder.embed_query("日志组件的查找注入")
    with QdrantLocalVectorStore.open(index_dir) as store:
        scored = store.search(vector, 5)

    assert len(scored) == 5
    assert all(isinstance(item.chunk, KnowledgeChunk) for item in scored)
    # 分数按降序,且 chunk_id 参与同分 tie-break
    keys = [(-item.score, item.chunk.chunk_id) for item in scored]
    assert keys == sorted(keys)
    # payload 里的偏移与原文必须自洽(KnowledgeChunk 的校验器已经保证,
    # 这里再核对一次"取回来的确实是原文")
    for item in scored:
        assert item.chunk.char_end - item.chunk.char_start == len(item.chunk.text)
        assert item.chunk.chunk_id == f"{item.chunk.doc_id}#{item.chunk.ordinal}"


def test_search_limit_zero_returns_nothing(built_index, embedder):
    index_dir, _ = built_index
    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.search(embedder.embed_query("x"), 0) == []


def test_search_rejects_a_wrong_dimension_query(built_index):
    index_dir, metadata = built_index
    with QdrantLocalVectorStore.open(index_dir) as store:
        with pytest.raises(StoreError) as info:
            store.search([0.0] * (metadata.dim - 1), 3)
        assert info.value.reason == "dimension-mismatch"


def test_upsert_rejects_mismatched_lengths(built_index):
    index_dir, _ = built_index
    chunk = KnowledgeChunk(
        chunk_id="kb-cve-2021-44228#0",
        doc_id="kb-cve-2021-44228",
        ordinal=0,
        text="abc",
        char_start=0,
        char_end=3,
    )
    with QdrantLocalVectorStore.open(index_dir) as store:
        with pytest.raises(StoreError) as info:
            store.upsert([chunk], [])
        assert info.value.reason == "length-mismatch"


def test_upsert_is_a_replace_not_an_append(built_index, embedder, corpus):
    """同一 `chunk_id` 重复写入按替换处理(契约要求)。"""
    index_dir, metadata = built_index
    chunk = chunk_corpus(corpus.documents)[0]
    vector = embedder.embed_documents([chunk.text])[0]

    with QdrantLocalVectorStore.open(index_dir) as store:
        before = store.count()
        store.upsert([chunk], [vector])
        assert store.count() == before == metadata.chunk_count


# =====================================================================
# 3. 走既有 Retriever
# =====================================================================


def test_retrieval_goes_through_the_existing_retriever(built_index, corpus, embedder):
    index_dir, _ = built_index
    retriever = build_qdrant_retriever(corpus, embedder, index_dir)
    try:
        assert isinstance(retriever, Retriever)
        result = retriever.retrieve("日志组件的查找注入如何触发")
        assert result.status == "ok"
        assert result.corpus_version == corpus.manifest.corpus_version
        assert 1 <= len(result.hits) <= result.top_k
        # 引用粒度是文档,`citation_id == doc_id`
        for hit in result.hits:
            assert hit.citation_id == hit.doc_id
            assert hit.doc_id in corpus.doc_ids
    finally:
        retriever.store.close()


def test_ranking_matches_the_in_memory_backend(built_index, corpus, embedder):
    """换后端不改变可观测行为 —— S2-A 定下的分工,这里实测。

    顺序与分数都要一致。分数允许极小误差(Qdrant Local 内部用 float32
    存储向量),但在这份语料上实测误差为 0。
    """
    index_dir, _ = built_index
    memory = build_in_memory_retriever(corpus, embedder)
    qdrant = build_qdrant_retriever(corpus, embedder, index_dir)
    try:
        for query in (
            "日志组件的查找注入如何触发",
            "对外暴露的应用如何被利用",
            "凭据转储的检测思路",
        ):
            left = memory.retrieve(query)
            right = qdrant.retrieve(query)
            assert [hit.citation_id for hit in left.hits] == [
                hit.citation_id for hit in right.hits
            ], query
            for a, b in zip(left.hits, right.hits):
                assert a.score == pytest.approx(b.score, abs=1e-6)
                assert a.chunk_id == b.chunk_id
                assert a.snippet == b.snippet
                assert a.title == b.title
                assert a.url == b.url
    finally:
        qdrant.store.close()


def test_retrieval_is_read_only(built_index, corpus, embedder):
    """检索不改变索引:点数与 chunk id 集合在多次查询后完全一致。"""
    index_dir, metadata = built_index
    retriever = build_qdrant_retriever(corpus, embedder, index_dir)
    try:
        before = retriever.store.chunk_ids()
        for _ in range(3):
            retriever.retrieve("日志组件的查找注入如何触发")
        assert retriever.store.count() == metadata.chunk_count
        assert retriever.store.chunk_ids() == before
    finally:
        retriever.store.close()


# =====================================================================
# 4. 关闭 / 重开
# =====================================================================


def test_index_survives_close_and_reopen(built_index, corpus, embedder):
    index_dir, metadata = built_index

    first = build_qdrant_retriever(corpus, embedder, index_dir)
    result_before = first.retrieve("日志组件的查找注入如何触发")
    first.store.close()

    second = build_qdrant_retriever(corpus, embedder, index_dir)
    try:
        result_after = second.retrieve("日志组件的查找注入如何触发")
        assert second.store.metadata() == metadata
    finally:
        second.store.close()

    assert [h.citation_id for h in result_before.hits] == [
        h.citation_id for h in result_after.hits
    ]
    assert [h.score for h in result_before.hits] == [
        h.score for h in result_after.hits
    ]


def test_a_second_client_on_the_same_directory_is_rejected(built_index):
    """Qdrant Local 对存储目录持**排他锁** —— 实测,不是假设。

    并发写不安全,因此这里不重试、不等待:把冲突翻译成
    `IndexUnavailable(index-locked)`。
    """
    index_dir, _ = built_index
    holder = QdrantLocalVectorStore.open(index_dir)
    try:
        with pytest.raises(IndexUnavailable) as info:
            QdrantLocalVectorStore.open(index_dir)
        assert info.value.reason == "index-locked"
    finally:
        holder.close()


def test_close_is_idempotent(built_index):
    index_dir, _ = built_index
    store = QdrantLocalVectorStore.open(index_dir)
    store.close()
    store.close()  # 不抛
    assert store._closed is True


def test_context_manager_closes_the_store(built_index):
    index_dir, _ = built_index
    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.count() > 0
    assert store._closed is True
    # 关掉之后锁释放,可以再开
    with QdrantLocalVectorStore.open(index_dir) as reopened:
        assert reopened.count() > 0


# =====================================================================
# 5. 缺失 / 损坏 / 陈旧 —— 一律 fail closed
# =====================================================================


def test_missing_index_directory_is_unavailable(tmp_path: Path):
    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(tmp_path / "nope")
    assert info.value.reason == "index-missing"


def test_open_does_not_create_a_missing_index(tmp_path: Path):
    """打不开就报错,不"顺手建一个空索引" —— 那会让检索永远 empty。"""
    target = tmp_path / "nope"
    with pytest.raises(IndexUnavailable):
        QdrantLocalVectorStore.open(target)
    assert not target.exists()


def test_directory_without_the_collection_is_unavailable(tmp_path: Path):
    """目录存在、Qdrant 能开,但没有我们的集合。"""
    empty = tmp_path / "empty_store"
    empty.mkdir()
    # 先用一个 client 初始化存储结构再关掉,得到一个"合法但空"的目录
    from qdrant_client import QdrantClient

    QdrantClient(path=str(empty)).close()

    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(empty)
    assert info.value.reason == "index-missing"


def test_corrupt_metadata_is_unavailable(built_index):
    """把集合元数据改成缺字段的形态 → 不可用,而不是补默认值。"""
    index_dir, _ = built_index
    _recreate_collection(index_dir, {"index_format": "qdrant-local-v1", "dim": 256})

    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(index_dir)
    assert info.value.reason == "index-metadata-invalid"


def test_unknown_index_format_is_unavailable(built_index):
    index_dir, metadata = built_index
    payload = _metadata_payload(metadata)
    payload["index_format"] = "some-future-format"
    _recreate_collection(index_dir, payload, dim=metadata.dim)

    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(index_dir)
    assert info.value.reason == "index-format"


def test_incomplete_index_is_unavailable(built_index):
    """元数据说 N 块、实际只有 0 块 → 不可用(挡住"写了一半")。"""
    index_dir, metadata = built_index
    _recreate_collection(index_dir, _metadata_payload(metadata), dim=metadata.dim)

    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(index_dir)
    assert info.value.reason == "index-incomplete"


def test_vector_size_mismatch_is_unavailable(built_index):
    """元数据说 256 维,集合实际是 8 维 —— 单看任何一方都发现不了。"""
    index_dir, metadata = built_index
    _recreate_collection(index_dir, _metadata_payload(metadata), dim=8)

    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(index_dir)
    assert info.value.reason == "index-dim"


def test_wrong_distance_is_unavailable(built_index):
    index_dir, metadata = built_index
    from qdrant_client.models import Distance

    _recreate_collection(
        index_dir, _metadata_payload(metadata), dim=metadata.dim, distance=Distance.EUCLID
    )

    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(index_dir)
    assert info.value.reason == "index-distance"


def test_embedder_dimension_mismatch_is_rejected_at_open(built_index):
    """用 128 维嵌入器去开 256 维索引 → 在 `open()` 就拒绝。"""
    index_dir, _ = built_index
    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.open(index_dir, expected_dim=128)
    assert info.value.reason == "index-dim"


# =====================================================================
# 6. 语料漂移
# =====================================================================


def test_corpus_version_drift_is_detected(built_index, corpus, embedder):
    """索引建好之后语料版本变了 → `verify()` fail closed。"""
    index_dir, _ = built_index
    drifted = corpus.__class__(
        manifest=corpus.manifest.model_copy(
            update={"corpus_version": "v0.5.0-s2.99"}
        ),
        documents=corpus.documents,
    )
    with pytest.raises(IndexUnavailable) as info:
        build_qdrant_retriever(drifted, embedder, index_dir)
    assert info.value.reason == "index-corpus-version"


def test_corpus_hash_drift_is_detected(built_index, corpus, embedder):
    index_dir, _ = built_index
    drifted = corpus.__class__(
        manifest=corpus.manifest.model_copy(update={"corpus_sha256": "0" * 64}),
        documents=corpus.documents,
    )
    with pytest.raises(IndexUnavailable) as info:
        build_qdrant_retriever(drifted, embedder, index_dir)
    assert info.value.reason == "index-corpus-hash"


def test_embedder_drift_is_detected(built_index, corpus):
    """换嵌入器(维度或参数不同)→ 索引不可用,而不是"凑合算个分"。"""
    index_dir, _ = built_index
    with pytest.raises(IndexUnavailable):
        build_qdrant_retriever(corpus, HashingEmbedder(ngram=4), index_dir)


def test_failed_open_releases_the_lock(built_index, corpus, embedder):
    """`open()` 成功但 `verify()` 失败时也必须放锁 —— 否则锁残留到进程退出。"""
    index_dir, _ = built_index
    drifted = corpus.__class__(
        manifest=corpus.manifest.model_copy(update={"corpus_sha256": "0" * 64}),
        documents=corpus.documents,
    )
    with pytest.raises(IndexUnavailable):
        build_qdrant_retriever(drifted, embedder, index_dir)

    # 能重新打开,说明上一轮确实关掉了
    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.count() > 0


# =====================================================================
# 7. 安全形状:没有选择器、没有远程模式
# =====================================================================


def test_the_adapter_exposes_no_remote_connection_parameters():
    """只构造 `QdrantClient(path=...)` —— 没有 url/host/port/api_key 分支。

    这条挡的不是"当前没人传 url",而是"以后有人加一个远程模式"。远程模式
    一旦存在,"生产环境连到外部 Qdrant"就成为一种可能的配置,而本阶段
    的授权边界是"本地、离线、无出网"。

    用 AST 而不是子串扫描:模块文档里**提到** `url=` / `host=` 这些词是
    为了解释"为什么没有它们",按子串判会把说明文字当成违规 —— 一个总在
    误报的护栏等于没有护栏。
    """
    import ast

    source = Path(qdrant_local.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden = {"url", "host", "port", "api_key", "https", "http", "grpc_port"}
    offenders: list[str] = []
    connect_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in forbidden:
                    offenders.append(keyword.arg)
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "QdrantClient":
                connect_calls += 1
                assert [kw.arg for kw in node.keywords] == ["path"], (
                    "QdrantClient 只允许以 path= 构造(Local 模式)"
                )
    assert offenders == [], offenders
    assert connect_calls == 1, "客户端构造点应当只有一处"

    signature = inspect.signature(QdrantLocalVectorStore.__init__)
    assert set(signature.parameters) == {
        "self",
        "client",
        "metadata",
        "index_dir",
    }


def test_collection_name_is_not_a_parameter():
    """集合名是模块常量 —— 没有"从哪个集合读"这个输入。"""
    for method in (
        QdrantLocalVectorStore.open,
        QdrantLocalVectorStore.create,
        build_qdrant_retriever,
        build_qdrant_index,
    ):
        parameters = set(inspect.signature(method).parameters)
        assert "collection" not in parameters
        assert "collection_name" not in parameters
    assert COLLECTION_NAME == "cybersec_knowledge"


def test_build_index_does_not_expose_paths_in_errors(tmp_path: Path, corpus, embedder):
    """异常文本不含路径 —— 它们会走到工具返回值与 HTTP 响应体。"""
    with pytest.raises(IndexUnavailable) as info:
        build_qdrant_retriever(corpus, embedder, tmp_path / "absent")
    rendered = str(info.value)
    assert str(tmp_path) not in rendered
    assert "absent" not in rendered
    assert info.value.reason == "index-missing"


def test_metadata_round_trips_through_the_collection_config(built_index):
    """`IndexMetadata` 的七个字段必须原样回来(含 dim 与 embedder_id)。"""
    index_dir, metadata = built_index
    with QdrantLocalVectorStore.open(index_dir) as store:
        restored = store.metadata()
    assert isinstance(restored, IndexMetadata)
    assert restored == metadata
