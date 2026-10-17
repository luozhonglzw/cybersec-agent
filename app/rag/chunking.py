"""确定性切块 —— 同一份语料永远产出同一组 `chunk_id`。

为什么切块必须确定性
------------------
`chunk_id` 参与检索的 tie-break(同分按 `chunk_id` 升序),也被写进索引。
只要切块依赖任何不稳定因素(字典序、随机、并发完成顺序),同一份语料在
两次构建后就会得到两套 id,于是"重建索引"变成"换了一套索引",而语料
哈希却一模一样 —— 这种漂移极难排查。所以这里的输入只有 `doc.text`,
输出只由 `doc.text` 决定。

切块规则(冻结,见 S1 §3.4)
--------------------------
1. 按空行(`\\n\\n`)切段;
2. 段落长度 ≤ `CHUNK_MAX_CHARS` → 该段落自成一块;
3. 超长段落 → 以 `CHUNK_MAX_CHARS` 为窗口硬切,相邻窗口重叠
   `CHUNK_OVERLAP_CHARS`(保证跨窗口边界的短语仍能被检索到);
4. 空白块丢弃;
5. 序号 `ordinal` 在**保留下来的块**上从 0 递增。

刻意不做"贪心打包":把多个短段落合并到接近上限会让块边界随语料微调而
移动,块的语义单位也会从"一个论点"变成"一段任意长度的文字"。段落是
作者划定的语义边界,尊重它比塞满窗口更重要。
"""
from __future__ import annotations

from collections.abc import Iterable

from app.rag.types import KnowledgeChunk, KnowledgeDocument

#: 切块器版本。它进 `IndexMetadata`:切法一变,旧索引立刻被判为不可用,
#: 而不是用"旧切法的向量"回答"新切法的语料"。
CHUNKER_VERSION = "paragraph-window-v1"

#: 单块最大字符数。
CHUNK_MAX_CHARS = 700

#: 硬切时的重叠字符数。必须严格小于窗口,否则切块不前进(死循环)。
CHUNK_OVERLAP_CHARS = 120

_PARAGRAPH_SEPARATOR = "\n\n"


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """返回非空段落的 `(start, end)` 列表,偏移相对 `text`。"""
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        if text.startswith(_PARAGRAPH_SEPARATOR, index):
            spans.append((start, index))
            index += len(_PARAGRAPH_SEPARATOR)
            start = index
        else:
            index += 1
    spans.append((start, length))
    return [(begin, end) for begin, end in spans if text[begin:end].strip()]


def _window_spans(start: int, end: int) -> list[tuple[int, int]]:
    """把 `[start, end)` 按窗口 + 重叠切成若干段。"""
    step = CHUNK_MAX_CHARS - CHUNK_OVERLAP_CHARS
    if step <= 0:  # pragma: no cover - 由导入期不变量与测试双重保证
        raise RuntimeError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_MAX_CHARS")
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + CHUNK_MAX_CHARS, end)
        spans.append((cursor, stop))
        if stop >= end:
            break
        cursor += step
    return spans


def chunk_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    """把一篇文档切成块。空文档不产生任何块(而不是产生一个空块)。"""
    spans: list[tuple[int, int]] = []
    for begin, end in _paragraph_spans(document.text):
        if end - begin <= CHUNK_MAX_CHARS:
            spans.append((begin, end))
        else:
            spans.extend(_window_spans(begin, end))

    chunks: list[KnowledgeChunk] = []
    for begin, end in spans:
        text = document.text[begin:end]
        if not text.strip():
            continue
        ordinal = len(chunks)
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{document.doc_id}#{ordinal}",
                doc_id=document.doc_id,
                ordinal=ordinal,
                text=text,
                char_start=begin,
                char_end=end,
            )
        )
    return chunks


def chunk_corpus(documents: Iterable[KnowledgeDocument]) -> list[KnowledgeChunk]:
    """切整个语料,**按 `doc_id` 升序**处理,不依赖输入顺序。

    这一点是刻意的:调用方可能从 JSONL 读入(文件行序)、从数据库读入
    (返回顺序未定义)或从测试夹具构造。若这里依赖输入顺序,上面那些
    来源的差异就会变成索引差异。
    """
    ordered = sorted(documents, key=lambda document: document.doc_id)
    chunks: list[KnowledgeChunk] = []
    for document in ordered:
        chunks.extend(chunk_document(document))
    return chunks
