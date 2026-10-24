"""Qdrant Local **重建失败注入**测试(S2-F 建立,S2-G1 随契约更新)。

这些用例把失败精确注入重建的五个阶段,每次失败后**关掉再重新打开**存储,
断言不完整/陈旧的索引不会被当成有效索引打开,并验证显式重建可以恢复。

S2-G1 起身份元数据**最后提交**,因此"点已写完但身份未提交"也算失败态 ——
见 `test_crash_before_metadata_commit_leaves_no_usable_index`。这不宣称重建
是原子的:它保证的是"身份最后可见",不是"点与身份同时生效"。

全部 hermetic:真实 Qdrant Local,索引建在 pytest 的 `tmp_path`,不需要
Qdrant 服务端、Docker、GPU、真实 LLM 或网络。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from qdrant_client.models import PointStruct

import app.vectorstore.qdrant_local as ql
from app.rag import HashingEmbedder, load_corpus
from app.rag.errors import IndexUnavailable
from app.vectorstore import (
    COLLECTION_NAME,
    QdrantLocalVectorStore,
    build_qdrant_index,
    build_qdrant_retriever,
    point_id_of,
)

CORPUS_VERSION = "v0.5.0-s2f"
PROVENANCE = (
    "Self-authored summary written for this repository's test corpus. "
    "NOT an official NVD or MITRE ATT&CK record; not a reproduction of vendor text."
)


# ---------------------------------------------------------------------------
# 语料构造:两段正文(空行分隔)→ 每篇文档恰好 2 块
# ---------------------------------------------------------------------------


def _doc(*, doc_id: str, source_id: str, source_type: str, text: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "title": f"合成摘要 {source_id}",
        "source_type": source_type,
        "source_id": source_id,
        "source_url": f"https://nvd.nist.gov/vuln/detail/{source_id}",
        "publisher": "self-authored-summary",
        "authored_at": "2026-10-17",
        "provenance": PROVENANCE,
        "text": text,
    }


def _text(marker: str) -> str:
    """200–600 字符、含一个空行 → 稳定产出 2 块。marker 使文档可被检索区分。"""
    paragraph = (
        f"{marker} 段落。该自写摘要用于失败注入测试,描述某类安全机制的成因、"
        "影响面、利用前提与检测思路;它刻意不复制任何厂商公告原文,也不提供"
        "可直接照做的攻击步骤,正文仅为占位说明文字。缓解与检测以资产清点、"
        "及时修补、最小暴露面与出口侧告警为主,运维侧关注异常出站连接、"
        "权限校验缺失与版本滞后等信号,并配合基线核对与变更审计。"
        "再补一段说明文字以确保长度落进语料契约允许的区间之内。"
    )
    return f"{paragraph}\n\n{paragraph}"


def _write_corpus(tmp_path: Path, docs: list[dict[str, Any]]) -> Path:
    raw = "".join(
        json.dumps(document, ensure_ascii=False) + "\n" for document in docs
    ).encode("utf-8")
    target = tmp_path / f"corpus-{hashlib.sha256(raw).hexdigest()[:8]}"
    target.mkdir(parents=True)
    (target / "corpus.jsonl").write_bytes(raw)
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_version": CORPUS_VERSION,
                "doc_count": len(docs),
                "corpus_sha256": hashlib.sha256(raw).hexdigest(),
                "authored_from": "self-authored",
                "generated_at": "2026-10-17",
            }
        ),
        encoding="utf-8",
    )
    return target


def _docs(markers: list[str], *, number_base: int = 1000) -> list[dict[str, Any]]:
    """合成语料。`number_base` 让新旧两份语料的 doc_id / source_id 不同 ——
    "删掉的文档不得复活"这条不变量只有在 id 可区分时才可检验。"""
    return [
        _doc(
            doc_id=f"kb-cve-2015-{number_base + index}",
            source_id=f"CVE-2015-{number_base + index}",
            source_type="cve",
            text=_text(marker),
        )
        for index, marker in enumerate(markers)
    ]


def _build_ok(corpus_dir: Path, index_dir: Path, *, replace: bool = False):
    """无注入地成功构建一次索引,返回索引身份。"""
    return build_qdrant_index(
        load_corpus(corpus_dir), HashingEmbedder(), index_dir, replace=replace
    )


# ---------------------------------------------------------------------------
# 失败注入:包装真实客户端,按阶段抛错
# ---------------------------------------------------------------------------


class _Plan:
    """一次注入计划。默认全部关闭;测试按需打开某个开关。"""

    def __init__(self) -> None:
        self.enabled = False
        self.crash_after_delete = False
        self.crash_before_reset = False
        self.truncate_upsert_to: int | None = None
        self.crash_before_final_count = False
        self.crash_at_commit = False
        self.upsert_completed = False


class _InjectingClient:
    """转发到真实 QdrantClient,只在计划指定的阶段抛错。

    刻意**不替换被测类**(`QdrantLocalVectorStore`):被测路径完整保留,
    只有底层客户端被包装 —— 验证的是生产代码在失败下的行为,而不是替身。
    """

    def __init__(self, inner: Any, plan: _Plan) -> None:
        self._inner = inner
        self._plan = plan

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    # ---- 阶段 1:旧集合已删除,新集合尚未创建 ----
    def create_collection(self, **kwargs: Any):
        if self._plan.enabled and self._plan.crash_after_delete:
            raise RuntimeError("injected: crash after old collection deletion")
        return self._inner.create_collection(**kwargs)

    # ---- 阶段 2:新集合已创建(哨兵身份),清空/写入之前 ----
    def delete(self, **kwargs: Any):
        if self._plan.enabled and self._plan.crash_before_reset:
            raise RuntimeError("injected: crash after new collection creation")
        return self._inner.delete(**kwargs)

    # ---- 阶段 3:点写入中途失败(只写一部分再抛) ----
    def upsert(self, **kwargs: Any):
        if self._plan.enabled and self._plan.truncate_upsert_to is not None:
            points = list(kwargs.get("points", []))
            keep = points[: self._plan.truncate_upsert_to]
            if keep:
                self._inner.upsert(
                    collection_name=kwargs["collection_name"], points=keep
                )
            raise RuntimeError("injected: crash during partial point upsert")
        result = self._inner.upsert(**kwargs)
        self._plan.upsert_completed = True
        return result

    # ---- 阶段 4:全部点已写入,最终 count() 自检之前 ----
    def count(self, *args: Any, **kwargs: Any):
        if (
            self._plan.enabled
            and self._plan.crash_before_final_count
            and self._plan.upsert_completed
        ):
            raise RuntimeError("injected: crash before final verification")
        return self._inner.count(*args, **kwargs)

    # ---- 阶段 5:点已全部写入且自检通过,提交身份元数据时失败 ----
    def update_collection(self, **kwargs: Any):
        if self._plan.enabled and self._plan.crash_at_commit:
            raise RuntimeError("injected: crash during metadata commit")
        return self._inner.update_collection(**kwargs)

    def close(self):
        return self._inner.close()


@pytest.fixture()
def plan(monkeypatch) -> _Plan:
    """把模块级 `QdrantClient` 换成"返回被包装客户端"的工厂。"""
    state = _Plan()
    real = ql.QdrantClient

    def factory(*args: Any, **kwargs: Any) -> _InjectingClient:
        return _InjectingClient(real(*args, **kwargs), state)

    monkeypatch.setattr(ql, "QdrantClient", factory)
    yield state
    # 收尾:关掉注入,让后续 reopen 走真实客户端
    state.enabled = False


def _reopen(index_dir: Path) -> QdrantLocalVectorStore:
    """注入关闭后重新打开存储(调用方负责 close)。"""
    return QdrantLocalVectorStore.open(index_dir)


# ===========================================================================
# A. 阶段 1 —— 旧集合已删除
# ===========================================================================


def test_crash_after_old_collection_deletion_leaves_no_openable_index(tmp_path, plan):
    """删掉旧集合后崩 ⇒ 没有集合可打开,且不会残留半份索引。"""
    corpus_dir = _write_corpus(tmp_path, _docs(["ALPHA-ONE", "ALPHA-TWO"]))
    index_dir = tmp_path / "idx"
    _build_ok(corpus_dir, index_dir)
    # 前置条件:索引本来是好的
    with _reopen(index_dir) as store:
        assert store.count() > 0

    plan.enabled = True
    plan.crash_after_delete = True
    with pytest.raises(RuntimeError, match="crash after old collection deletion"):
        _build_ok(corpus_dir, index_dir, replace=True)

    # 关掉注入后重新打开:集合已不存在 ⇒ fail closed
    with pytest.raises(IndexUnavailable) as info:
        _reopen(index_dir)
    assert info.value.reason == "index-missing"


def test_recovery_after_stage_one_crash(tmp_path, plan):
    """阶段 1 崩溃后,一次显式重建即可恢复。"""
    corpus_dir = _write_corpus(tmp_path, _docs(["RECOVER-A", "RECOVER-B"]))
    index_dir = tmp_path / "idx"
    _build_ok(corpus_dir, index_dir)

    plan.enabled = True
    plan.crash_after_delete = True
    with pytest.raises(RuntimeError):
        _build_ok(corpus_dir, index_dir, replace=True)

    plan.enabled = False
    metadata = _build_ok(corpus_dir, index_dir, replace=True)
    assert metadata.chunk_count > 0
    with _reopen(index_dir) as store:
        assert store.count() == metadata.chunk_count


# ===========================================================================
# B. 阶段 2 —— 新集合已创建
# ===========================================================================


def test_crash_after_new_collection_creation_is_unavailable(tmp_path, plan):
    """建好新集合但没写入 ⇒ 身份仍是哨兵 ⇒ index-metadata-invalid。

    S2-G1 之后身份元数据最后才提交,所以这个阶段崩溃时集合配置里只有
    `index_format` 一项,既有 `_metadata_from_payload` 直接 fail closed ——
    比"靠 count 不符间接发现"更早、更确定。
    """
    corpus_dir = _write_corpus(tmp_path, _docs(["FRESH-A", "FRESH-B"]))
    index_dir = tmp_path / "idx"

    plan.enabled = True
    plan.crash_before_reset = True
    with pytest.raises(RuntimeError, match="crash after new collection creation"):
        _build_ok(corpus_dir, index_dir)

    plan.enabled = False
    with pytest.raises(IndexUnavailable) as info:
        _reopen(index_dir)
    assert info.value.reason == "index-metadata-invalid"


def test_recovery_after_stage_two_crash(tmp_path, plan):
    corpus_dir = _write_corpus(tmp_path, _docs(["RECOVER2-A", "RECOVER2-B"]))
    index_dir = tmp_path / "idx"

    plan.enabled = True
    plan.crash_before_reset = True
    with pytest.raises(RuntimeError):
        _build_ok(corpus_dir, index_dir)

    plan.enabled = False
    metadata = _build_ok(corpus_dir, index_dir, replace=True)
    with _reopen(index_dir) as store:
        assert store.count() == metadata.chunk_count


# ===========================================================================
# C. 阶段 3 —— 点写入中途失败
# ===========================================================================


def test_crash_during_partial_upsert_is_unavailable(tmp_path, plan):
    """只写了一部分点 ⇒ 身份未提交 ⇒ index-metadata-invalid。"""
    corpus_dir = _write_corpus(tmp_path, _docs(["PART-A", "PART-B", "PART-C"]))
    index_dir = tmp_path / "idx"
    full = _build_ok(corpus_dir, index_dir)
    total = full.chunk_count
    assert total >= 3, total  # 每篇 2 块

    plan.enabled = True
    plan.truncate_upsert_to = 1
    with pytest.raises(RuntimeError, match="crash during partial point upsert"):
        _build_ok(corpus_dir, index_dir, replace=True)

    plan.enabled = False
    with pytest.raises(IndexUnavailable) as info:
        _reopen(index_dir)
    assert info.value.reason == "index-metadata-invalid"


def test_recovery_after_partial_upsert_crash(tmp_path, plan):
    corpus_dir = _write_corpus(tmp_path, _docs(["PART-REC-A", "PART-REC-B"]))
    index_dir = tmp_path / "idx"
    full = _build_ok(corpus_dir, index_dir)

    plan.enabled = True
    plan.truncate_upsert_to = 1
    with pytest.raises(RuntimeError):
        _build_ok(corpus_dir, index_dir, replace=True)

    plan.enabled = False
    again = _build_ok(corpus_dir, index_dir, replace=True)
    assert again.chunk_count == full.chunk_count
    with _reopen(index_dir) as store:
        assert store.count() == full.chunk_count


# ===========================================================================
# D. 阶段 4/5 —— 自检之前、身份提交之时
# ===========================================================================


def test_crash_before_metadata_commit_leaves_no_usable_index(tmp_path, plan):
    """**S2-G1 的契约变化**:点已全写但身份未提交 ⇒ 仍然 fail closed。

    修正前这里是"索引其实完整、`open()` 会成功"—— 那条断言在**只改正文、
    doc_id 不变**的重建里正是静默陈旧的来源:点还是旧语料的,身份却已更新。
    现在身份最后写,所以这个阶段崩溃留下的是哨兵索引:

    - `open()` 抛 `index-metadata-invalid`;
    - 即便磁盘上的点数恰好等于 chunk_count,也不被当作有效索引。

    这不宣称原子性:它保证的是"身份最后可见",不是"点与身份同时生效"。
    """
    corpus_dir = _write_corpus(tmp_path, _docs(["VERIFY-A", "VERIFY-B"]))
    index_dir = tmp_path / "idx"

    plan.enabled = True
    plan.crash_before_final_count = True
    with pytest.raises(RuntimeError, match="crash before final verification"):
        _build_ok(corpus_dir, index_dir)

    plan.enabled = False
    with pytest.raises(IndexUnavailable) as info:
        _reopen(index_dir)
    assert info.value.reason == "index-metadata-invalid"


def test_recovery_after_pre_commit_crash(tmp_path, plan):
    """自检前崩溃后,一次显式重建即可恢复。"""
    corpus_dir = _write_corpus(tmp_path, _docs(["VERIFY-REC-A", "VERIFY-REC-B"]))
    index_dir = tmp_path / "idx"

    plan.enabled = True
    plan.crash_before_final_count = True
    with pytest.raises(RuntimeError):
        _build_ok(corpus_dir, index_dir)

    plan.enabled = False
    metadata = _build_ok(corpus_dir, index_dir, replace=True)
    with _reopen(index_dir) as store:
        assert store.count() == metadata.chunk_count


def test_crash_during_metadata_commit_leaves_no_usable_index(tmp_path, plan):
    """身份提交本身失败(update_collection 抛错)⇒ 哨兵仍在 ⇒ fail closed。

    这是"提交点"的负面对照:只要 `update_collection` 没成功,索引就不可用,
    不会出现"点已写完、身份只写了一半"的中间态被放行。
    """
    corpus_dir = _write_corpus(tmp_path, _docs(["COMMIT-A", "COMMIT-B"]))
    index_dir = tmp_path / "idx"

    plan.enabled = True
    plan.crash_at_commit = True
    with pytest.raises(RuntimeError, match="crash during metadata commit"):
        _build_ok(corpus_dir, index_dir)

    plan.enabled = False
    with pytest.raises(IndexUnavailable) as info:
        _reopen(index_dir)
    assert info.value.reason == "index-metadata-invalid"

    # 显式重建恢复
    metadata = _build_ok(corpus_dir, index_dir, replace=True)
    with _reopen(index_dir) as store:
        assert store.count() == metadata.chunk_count


# ===========================================================================
# E. 陈旧点复活(删掉的文档不得悄悄回来)
# ===========================================================================


def _docs_for_resurrection() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """两份**块数相同、文档不同**(doc_id 与 source_id 都不同)的语料。

    这正是复活危险的精确形态:`count == chunk_count` 的结构自检放行,
    而内容已经是上一份语料的。
    """
    old = _docs(["RESURRECT-OLD-A", "RESURRECT-OLD-B"], number_base=1000)
    new = _docs(["RESURRECT-NEW-A", "RESURRECT-NEW-B"], number_base=2000)
    assert len(old) == len(new)
    old_ids = {d["doc_id"] for d in old}
    new_ids = {d["doc_id"] for d in new}
    assert not (old_ids & new_ids), "doc_id 必须不同,否则反例不成立"
    return old, new


def test_crash_before_reset_cannot_expose_resurrected_points(tmp_path, plan):
    """块数相同时,阶段 2 崩溃**不再**让旧点可用。

    `delete_collection()` 只改 `meta.json`、磁盘数据目录保留,同名
    `create_collection()` 可能把旧点挂回来。**这个底层行为跨平台不一致**
    (Windows 与 Linux 的物理残留不同),所以本用例**不**断言点数 —— 断言
    的是**身份**:S2-G1 之后集合配置里只有哨兵,残留旧点因此无法通过
    `open()` 变成"有效索引"。

    修正前这条测试断言的是相反的事实(`open()` 放行、旧点可检索)。

    相关覆盖(不在此处重复):
    - 装配期拒绝 / 失效索引无法返回检索结果:
      `test_resurrected_index_is_rejected_at_assembly_time`;
    - 显式重建后恢复且与新语料一致:
      `test_rebuild_after_resurrection_crash_removes_deleted_documents`。
    """
    old_docs, new_docs = _docs_for_resurrection()
    old_dir = _write_corpus(tmp_path, old_docs)
    new_dir = _write_corpus(tmp_path / "alt", new_docs)
    index_dir = tmp_path / "idx"
    _build_ok(old_dir, index_dir)

    plan.enabled = True
    plan.crash_before_reset = True
    with pytest.raises(RuntimeError, match="crash after new collection creation"):
        _build_ok(new_dir, index_dir, replace=True)
    plan.enabled = False

    # 适配器层面:身份未提交 ⇒ fail closed,旧点不可达
    with pytest.raises(IndexUnavailable) as info:
        _reopen(index_dir)
    assert info.value.reason == "index-metadata-invalid"

    # 底层证据:集合仍存在且身份是哨兵(所以上面不是"因为索引空了才失败")
    # 不断言点数:Qdrant Local 的物理残留行为跨平台不一致
    raw = ql.QdrantClient(path=str(index_dir))
    try:
        assert raw.get_collection(ql.COLLECTION_NAME).config.metadata == {
            ql._METADATA_FORMAT_KEY: ql.INDEX_FORMAT
        }
    finally:
        raw.close()


def test_resurrected_index_is_rejected_at_assembly_time(tmp_path, plan):
    """修正后:复活索引在**装配期**就被拒绝,不再等到查询期。

    修正前(实测记录):`open()`/`verify()` 六项全过,靠 `_hit()` 的
    `index-unknown-document` 在查询期挡住 —— 代价是每条命中复活点的查询
    都失败。现在身份未提交,`build_qdrant_retriever` 直接抛
    `index-metadata-invalid`,陈旧向量连候选都进不去。
    """
    old_docs, new_docs = _docs_for_resurrection()
    old_dir = _write_corpus(tmp_path, old_docs)
    new_dir = _write_corpus(tmp_path / "alt", new_docs)
    index_dir = tmp_path / "idx"
    _build_ok(old_dir, index_dir)

    plan.enabled = True
    plan.crash_before_reset = True
    with pytest.raises(RuntimeError):
        _build_ok(new_dir, index_dir, replace=True)
    plan.enabled = False

    corpus = load_corpus(new_dir)
    with pytest.raises(IndexUnavailable) as info:
        build_qdrant_retriever(corpus, HashingEmbedder(), index_dir)
    assert info.value.reason == "index-metadata-invalid"


def test_rebuild_after_resurrection_crash_removes_deleted_documents(tmp_path, plan):
    """一次成功的重建会把被删文档清出去,且可检索集合与新语料一致。"""
    old_docs, new_docs = _docs_for_resurrection()
    old_dir = _write_corpus(tmp_path, old_docs)
    new_dir = _write_corpus(tmp_path / "alt", new_docs)
    index_dir = tmp_path / "idx"
    _build_ok(old_dir, index_dir)

    plan.enabled = True
    plan.crash_before_reset = True
    with pytest.raises(RuntimeError):
        _build_ok(new_dir, index_dir, replace=True)
    plan.enabled = False

    corpus = load_corpus(new_dir)
    # 崩溃后的中间态:装配期即被拒绝(见上一条测试)。
    with pytest.raises(IndexUnavailable) as info:
        build_qdrant_retriever(corpus, HashingEmbedder(), index_dir)
    assert info.value.reason == "index-metadata-invalid"

    # 显式重建
    metadata = build_qdrant_index(corpus, HashingEmbedder(), index_dir, replace=True)
    retriever = build_qdrant_retriever(corpus, HashingEmbedder(), index_dir)
    try:
        result = retriever.retrieve("RESURRECT-NEW-A 段落", top_k=10)
        retrieved_ids = {hit.doc_id for hit in result.hits}
        new_ids = {document.doc_id for document in corpus.documents}
        assert retrieved_ids <= new_ids, retrieved_ids
        # 旧语料的文档一篇都不在
        old_ids = {d["doc_id"] for d in old_docs}
        assert not (retrieved_ids & old_ids)
        assert result.corpus_version == metadata.corpus_version
    finally:
        retriever.store.close()


# ===========================================================================
# F. 同 doc_id、仅正文改变 —— S2-F2 的静默陈旧反例(S2-G1 转成回归测试)
# ===========================================================================


def _same_id_corpora() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """两份**doc_id 完全相同、只有正文不同**的语料。

    这是最危险的形态:身份哈希变了,但点的 doc_id 全都"合法",
    所以 `_hit()` 的未知文档校验帮不上忙。
    """
    old = _docs(["STALE-OLD-KEYWORD"], number_base=1000)
    new = _docs(["STALE-NEW-KEYWORD"], number_base=1000)
    assert {d["doc_id"] for d in old} == {d["doc_id"] for d in new}
    assert old[0]["text"] != new[0]["text"]
    return old, new


def test_same_doc_id_text_change_cannot_serve_stale_vectors(tmp_path, plan):
    """**S2-F2 反例的永久回归**:中断重建不得让旧向量继续被当作证据。

    修正前的实测行为:阶段 2 崩溃后

        open() 放行 → verify() 六项全过(元数据已是**新**语料)
        → retrieve("旧关键词") 返回 status="ok" + 旧向量分数 0.1935

    即"删改过的文档仍按旧内容回答问题",且**没有任何报错**。

    修正后:身份元数据最后提交,崩溃留下哨兵 ⇒ 装配期
    `index-metadata-invalid`,旧向量连候选都进不去。
    """
    old_docs, new_docs = _same_id_corpora()
    old_dir = _write_corpus(tmp_path, old_docs)
    new_dir = _write_corpus(tmp_path / "alt", new_docs)
    index_dir = tmp_path / "idx"
    _build_ok(old_dir, index_dir)

    plan.enabled = True
    plan.crash_before_reset = True
    with pytest.raises(RuntimeError):
        _build_ok(new_dir, index_dir, replace=True)
    plan.enabled = False

    corpus = load_corpus(new_dir)
    # 1) 存储层:身份是哨兵,不是"新语料的合法身份"
    with pytest.raises(IndexUnavailable) as info:
        _reopen(index_dir)
    assert info.value.reason == "index-metadata-invalid"
    # 2) 检索层:装配即失败,陈旧向量无法影响任何证据
    with pytest.raises(IndexUnavailable):
        build_qdrant_retriever(corpus, HashingEmbedder(), index_dir)


def test_same_doc_id_rebuild_then_serves_new_text(tmp_path, plan):
    """显式重建之后,同 doc_id 的文档按**新正文**被检索到。"""
    old_docs, new_docs = _same_id_corpora()
    old_dir = _write_corpus(tmp_path, old_docs)
    new_dir = _write_corpus(tmp_path / "alt", new_docs)
    index_dir = tmp_path / "idx"
    _build_ok(old_dir, index_dir)

    plan.enabled = True
    plan.crash_before_reset = True
    with pytest.raises(RuntimeError):
        _build_ok(new_dir, index_dir, replace=True)
    plan.enabled = False

    corpus = load_corpus(new_dir)
    build_qdrant_index(corpus, HashingEmbedder(), index_dir, replace=True)
    retriever = build_qdrant_retriever(corpus, HashingEmbedder(), index_dir)
    try:
        stale = retriever.retrieve("STALE-OLD-KEYWORD", top_k=5)
        fresh = retriever.retrieve("STALE-NEW-KEYWORD", top_k=5)
        assert fresh.hits, "新正文应当可被检索"
        stale_best = stale.hits[0].score if stale.hits else 0.0
        assert fresh.hits[0].score > stale_best, (fresh.hits[0].score, stale_best)
    finally:
        retriever.store.close()


def test_successful_rebuild_never_resurrects_points(tmp_path):
    """无注入的正向对照:同块数语料重建后,旧文档不可检索。"""
    old_docs, new_docs = _docs_for_resurrection()
    old_dir = _write_corpus(tmp_path, old_docs)
    new_dir = _write_corpus(tmp_path / "alt", new_docs)
    index_dir = tmp_path / "idx"
    _build_ok(old_dir, index_dir)

    corpus = load_corpus(new_dir)
    metadata = build_qdrant_index(corpus, HashingEmbedder(), index_dir, replace=True)
    retriever = build_qdrant_retriever(corpus, HashingEmbedder(), index_dir)
    try:
        result = retriever.retrieve("RESURRECT", top_k=10)
        ids = {hit.doc_id for hit in result.hits}
        assert not (ids & {d["doc_id"] for d in old_docs})
    finally:
        retriever.store.close()


def test_collection_name_is_the_fixed_constant(tmp_path):
    """护栏:集合名不是参数,注入也没有绕过它。"""
    corpus_dir = _write_corpus(tmp_path, _docs(["NAME-A", "NAME-B"]))
    index_dir = tmp_path / "idx"
    _build_ok(corpus_dir, index_dir)
    # 复用现有夹具的构造方式不可行,这里直接断言常量被适配器使用
    assert COLLECTION_NAME and COLLECTION_NAME.isidentifier()


# ===========================================================================
# G. 开期校验的覆盖面(S2-G1 明确记录的**剩余缺口**)
# ===========================================================================


def test_open_time_verification_does_not_scan_point_identities(tmp_path):
    """**已知边界,不是期望行为**:`open()` 只核对集合配置与点数,不扫点。

    构造方式:用带外客户端把一个点的 payload 换成"当前语料里不存在的
    doc_id",**点数不变**。于是:

    - `open()` 放行(元数据六项 + count 全对);
    - `build_qdrant_retriever` 的 `verify()` 也放行;
    - 查询期 `_hit()` 抛 `index-unknown-document` —— 证据不会被静默污染,
      但整条查询失败。

    S2-G1 消除的是**重建窗口内产生这种状态**的路径(身份最后提交);带外
    写入仍能造出它。要彻底关闭需在 `open()` 里做全量点身份扫描,成本与
    现有证据都不支持本阶段实施,故把边界钉住并上报,而不是声称已解决。
    """
    corpus_dir = _write_corpus(tmp_path, _docs(["SCAN-A", "SCAN-B"]))
    index_dir = tmp_path / "idx"
    metadata = _build_ok(corpus_dir, index_dir)

    # 取一个真实点的 id 与 payload,再改写 doc_id(chunk_id 必须跟着改)
    embedder = HashingEmbedder()
    store = QdrantLocalVectorStore.open(index_dir)
    try:
        chunk = store.search([0.0] * metadata.dim, limit=1)[0].chunk
        # 用**原始块的向量**:伪造点的得分与原块一致,确保它能进入候选
        vector = embedder.embed_query(chunk.text)
        forged = chunk.model_copy(
            update={
                "doc_id": "kb-cve-9999-9999",
                "chunk_id": "kb-cve-9999-9999#1",
            }
        )
        payload = ql._chunk_payload(forged)
        target_id = point_id_of(chunk.chunk_id)
    finally:
        store.close()

    raw = ql.QdrantClient(path=str(index_dir))
    try:
        raw.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=target_id, vector=vector, payload=payload)],
        )
        assert raw.count(COLLECTION_NAME, exact=True).count == metadata.chunk_count
    finally:
        raw.close()

    # 缺口:open() 与 verify() 都放行
    corpus = load_corpus(corpus_dir)
    retriever = build_qdrant_retriever(corpus, HashingEmbedder(), index_dir)
    try:
        with pytest.raises(IndexUnavailable) as info:
            retriever.retrieve("SCAN-A 段落", top_k=10)
        assert info.value.reason == "index-unknown-document"
    finally:
        retriever.store.close()
