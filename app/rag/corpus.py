"""语料加载与清单校验 —— `knowledge/` 的**唯一**读取入口。

为什么清单里要有 `corpus_sha256`
------------------------------
"索引和语料对不上"是 RAG 最典型的静默故障:改了语料没重建索引,检索
照样返回结果,只是返回的是**旧内容**的命中。这种错误没有任何异常信号,
只能靠内容指纹发现。所以 `manifest.json` 钉住 `corpus.jsonl` 的字节
哈希,加载时比对;不符即 `CorpusError`,不允许"先跑起来再说"。

为什么错误消息里没有路径
----------------------
这里的异常最终会走到工具返回值与 HTTP 响应体。路径一旦进去,模型与
客户端就知道部署布局了。因此消息只写**原因码 + 行号 + 文档 id**,
这三者足以定位问题,又不泄漏文件系统信息。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.rag.errors import CorpusError
from app.rag.types import CorpusManifest, KnowledgeDocument, sha256_hex

#: 应用根目录 —— 由**本文件位置**推导,不依赖进程 CWD。
#:
#: `app/rag/corpus.py` → `parents[0]=app/rag`、`parents[1]=app`、`parents[2]=仓库根`。
APPLICATION_ROOT = Path(__file__).resolve().parents[2]

#: 内置示例语料目录。**应用相对**,不是进程相对。
#:
#: 为什么改(v0.5.0-S2-D):原值是 `Path("knowledge")`,即"以启动目录为基准"。
#: 那让"语料在哪"取决于从哪个目录启动 pytest / uvicorn —— 从仓库根启动一切
#: 正常,从别的目录启动就 `CorpusError: missing-corpus`,而错误信息里刻意
#: 不含路径,于是故障现场只剩一句"语料不可用"。语料是**随应用分发**的输入,
#: 它相对的是应用,不是调用者的工作目录。
#:
#: 注意与 `Settings.audit_db_path` / `checkpoint_db_path` 的区别:那两个是
#: **运行时数据**,部署要把它们指到数据卷上,因此保持"以启动目录为基准"的
#: 相对路径是有意设计。语料与索引是**只读输入 / 可重建派生物**,不该跟着
#: CWD 漂移。
DEFAULT_KNOWLEDGE_DIR = APPLICATION_ROOT / "knowledge"

MANIFEST_FILENAME = "manifest.json"
CORPUS_FILENAME = "corpus.jsonl"


@dataclass(frozen=True)
class LoadedCorpus:
    """一份已校验的语料:清单 + 按 `doc_id` 升序排列的文档。"""

    manifest: CorpusManifest
    documents: tuple[KnowledgeDocument, ...]

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(document.doc_id for document in self.documents)

    def by_id(self, doc_id: str) -> KnowledgeDocument:
        for document in self.documents:
            if document.doc_id == doc_id:
                return document
        raise KeyError(doc_id)


def _summarize_validation_error(exc: ValidationError) -> str:
    """把 Pydantic 错误压成"字段路径 + 说明",**不**带原始输入值。

    原始输入值可能是一整段正文;把它拼进异常消息既冗长,又会让下游
    日志/响应体带上语料内容。
    """
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
    return f"{location}: {first.get('msg', 'invalid value')}"


def corpus_sha256(raw: bytes) -> str:
    """`corpus.jsonl` 的字节哈希 —— 与清单里的字段同一算法。"""
    return sha256_hex(raw)


def read_corpus_bytes(knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR) -> bytes:
    path = Path(knowledge_dir) / CORPUS_FILENAME
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CorpusError(
            "knowledge corpus is unavailable", reason="missing-corpus"
        ) from exc


def read_manifest(knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR) -> CorpusManifest:
    path = Path(knowledge_dir) / MANIFEST_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(
            "knowledge manifest is unavailable", reason="missing-manifest"
        ) from exc
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorpusError(
            "knowledge manifest is not valid JSON", reason="invalid-manifest"
        ) from exc
    try:
        return CorpusManifest.model_validate(payload)
    except ValidationError as exc:
        raise CorpusError(
            f"knowledge manifest is invalid ({_summarize_validation_error(exc)})",
            reason="invalid-manifest",
        ) from exc


def parse_documents(raw: bytes) -> list[KnowledgeDocument]:
    """逐行解析 `corpus.jsonl`。空行忽略,其余行必须是一篇合法文档。"""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusError(
            "knowledge corpus is not valid UTF-8", reason="invalid-encoding"
        ) from exc

    documents: list[KnowledgeDocument] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(
                f"knowledge corpus line {line_number} is not valid JSON",
                reason="invalid-document",
            ) from exc
        if not isinstance(payload, dict):
            raise CorpusError(
                f"knowledge corpus line {line_number} is not an object",
                reason="invalid-document",
            )
        try:
            document = KnowledgeDocument.model_validate(payload)
        except ValidationError as exc:
            raise CorpusError(
                f"knowledge corpus line {line_number} is invalid "
                f"({_summarize_validation_error(exc)})",
                reason="invalid-document",
            ) from exc
        if document.doc_id in seen:
            raise CorpusError(
                f"duplicate doc_id {document.doc_id!r} at line {line_number}",
                reason="duplicate-doc-id",
            )
        seen.add(document.doc_id)
        documents.append(document)

    if not documents:
        raise CorpusError("knowledge corpus is empty", reason="empty-corpus")
    return documents


def load_corpus(knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR) -> LoadedCorpus:
    """加载并校验语料。任何不一致都 fail closed。"""
    manifest = read_manifest(knowledge_dir)
    raw = read_corpus_bytes(knowledge_dir)

    actual_sha256 = corpus_sha256(raw)
    if actual_sha256 != manifest.corpus_sha256:
        raise CorpusError(
            "knowledge corpus does not match its manifest hash",
            reason="manifest-mismatch",
        )

    documents = parse_documents(raw)
    if len(documents) != manifest.doc_count:
        raise CorpusError(
            f"manifest declares {manifest.doc_count} documents "
            f"but the corpus has {len(documents)}",
            reason="doc-count-mismatch",
        )

    return LoadedCorpus(
        manifest=manifest,
        documents=tuple(sorted(documents, key=lambda document: document.doc_id)),
    )
