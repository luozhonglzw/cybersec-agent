"""切块测试:确定性、分段、窗口与偏移。

切块的确定性不是风格问题:`chunk_id` 参与检索的 tie-break,也被写进索引。
只要切块依赖输入顺序或任何不稳定因素,同一份语料在两次构建后就会得到
两套 id,而语料哈希却一模一样 —— 这种漂移极难排查。
"""
from __future__ import annotations

import pytest

from app.rag.chunking import (
    CHUNKER_VERSION,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    chunk_corpus,
    chunk_document,
)
from app.rag.types import KnowledgeDocument


def _oversized_document(text: str, doc_id: str = "kb-cve-2021-44228") -> KnowledgeDocument:
    """构造一个**绕过长度校验**的文档,只为覆盖窗口硬切路径。

    真实语料永远落在 200-600 字符区间(由 `KnowledgeDocument` 强制),
    因此单段落不可能超过 `CHUNK_MAX_CHARS`。窗口路径必须能被测到,否则
    它只是"看起来存在"的代码。这里用 `model_construct` 显式绕过校验,
    并在断言里核对窗口几何 —— 绕过是有边界的,不是放松校验。
    """
    return KnowledgeDocument.model_construct(doc_id=doc_id, text=text)


def test_chunker_invariants_hold():
    """重叠必须严格小于窗口,否则切块不前进。"""
    assert CHUNK_OVERLAP_CHARS < CHUNK_MAX_CHARS
    assert CHUNKER_VERSION == "paragraph-window-v1"


def test_chunking_is_deterministic_across_calls(doc_factory):
    document = doc_factory()
    first = chunk_document(document)
    second = chunk_document(document)
    assert [chunk.model_dump() for chunk in first] == [
        chunk.model_dump() for chunk in second
    ]


def test_chunking_is_independent_of_input_order(doc_factory, attack_doc_factory):
    """语料来源(文件行序 / 数据库返回序)不该改变索引。"""
    documents = [doc_factory(), attack_doc_factory()]
    forward = chunk_corpus(documents)
    backward = chunk_corpus(list(reversed(documents)))
    assert [chunk.chunk_id for chunk in forward] == [
        chunk.chunk_id for chunk in backward
    ]


def test_paragraphs_become_separate_chunks(doc_factory):
    document = doc_factory()
    chunks = chunk_document(document)
    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert [chunk.chunk_id for chunk in chunks] == [
        f"{document.doc_id}#0",
        f"{document.doc_id}#1",
    ]
    assert chunks[0].text.strip() == document.text.split("\n\n")[0].strip()


def test_blank_spans_do_not_produce_chunks(doc_factory):
    """连续空行会产生"空白段落";它必须被丢弃,而不是变成空块。"""
    document = doc_factory(text="甲" * 100 + "\n\n   \n\n" + "乙" * 100)
    chunks = chunk_document(document)
    assert len(chunks) == 2
    assert all(chunk.text.strip() for chunk in chunks)


def test_long_paragraph_is_windowed_with_overlap():
    text = "甲" * (CHUNK_MAX_CHARS * 2 + 200)
    chunks = chunk_document(_oversized_document(text))

    step = CHUNK_MAX_CHARS - CHUNK_OVERLAP_CHARS
    assert [chunk.char_start for chunk in chunks] == [0, step, step * 2]
    assert chunks[0].char_end == CHUNK_MAX_CHARS
    # 相邻窗口重叠恰好 CHUNK_OVERLAP_CHARS
    assert chunks[0].char_end - chunks[1].char_start == CHUNK_OVERLAP_CHARS
    # 覆盖到正文末尾,不丢尾巴
    assert chunks[-1].char_end == len(text)
    assert "".join(chunk.text for chunk in chunks[:1]) == text[:CHUNK_MAX_CHARS]


def test_chunk_spans_stay_within_the_document(doc_factory):
    document = doc_factory()
    for chunk in chunk_document(document):
        assert 0 <= chunk.char_start < chunk.char_end <= len(document.text)
        assert document.text[chunk.char_start:chunk.char_end] == chunk.text


def test_every_corpus_document_is_chunked(chunks, documents):
    """语料里没有任何一篇文档会被切块器"吞掉"。"""
    assert {chunk.doc_id for chunk in chunks} == {doc.doc_id for doc in documents}
    assert len(chunks) >= len(documents)


def test_corpus_chunks_are_sorted_by_doc_id(chunks):
    doc_ids = [chunk.doc_id for chunk in chunks]
    assert doc_ids == sorted(doc_ids)


def test_corpus_chunk_ordinals_are_contiguous(chunks):
    by_doc: dict[str, list[int]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk.ordinal)
    for doc_id, ordinals in by_doc.items():
        assert ordinals == list(range(len(ordinals))), doc_id


def test_chunk_ids_are_unique(chunks):
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize("separator", ["\n\n", "\n\n\n\n"])
def test_separator_variants_do_not_change_semantics(doc_factory, separator):
    """多空行只是排版噪声,不应改变"按段落切"的语义。"""
    document = doc_factory(text="甲" * 100 + separator + "乙" * 100)
    chunks = chunk_document(document)
    assert [chunk.text.strip() for chunk in chunks] == ["甲" * 100, "乙" * 100]
