"""RAG 测试夹具。

三条约定:

1. **锚定仓库根**(`parents[2]`),不依赖当前工作目录 —— 否则从别的目录跑
   pytest 就找不到 `knowledge/`,而这条失败与实现无关。
2. **不读 `data/`**:语料在 `knowledge/`(受版本控制),索引是内存里的。
3. **合成文档走同一个校验路径**:`make_doc` 造出来的文档与真实语料一样要
   通过 `KnowledgeDocument` 的校验,不允许测试里出现"绕过校验的裸字典"。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from app.rag import (
    HashingEmbedder,
    KnowledgeDocument,
    KnowledgeChunk,
    LoadedCorpus,
    Retriever,
    build_in_memory_retriever,
    chunk_corpus,
    load_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"
EVAL_QUERIES_PATH = KNOWLEDGE_DIR / "eval" / "queries.jsonl"

#: 与语料一致的免责声明。测试里复制一份而不是从语料读,
#: 是为了让"声明被改动"这件事在语料测试里暴露出来,而不是被这里掩盖。
PROVENANCE = (
    "Self-authored summary written for this repository's demo corpus. "
    "NOT an official NVD or MITRE ATT&CK record; not a reproduction of vendor text."
)

#: 合成文档的默认正文:两段、总长落在 200-600 的合法区间内。
DEFAULT_TEXT = (
    "这是一段用于测试的合成摘要,描述某个公开漏洞的机制与影响面。"
    "它不复制任何厂商公告原文,只用于验证切块、嵌入与排序的确定性。"
    "正文长度需要落在契约允许的区间内,否则数据模型会直接拒绝;"
    "测试刻意走同一条校验路径,不构造绕过校验的裸字典,也不放松任何阈值。\n\n"
    "第二段用于验证按空行分段的行为:两段应当产生两个切块,"
    "序号从零开始递增,且偏移与原文可以互相核对。"
    "若切块器改动了段落边界或偏移算法,这里的断言会立刻变红,"
    "因为偏移错误会让片段与来源对不上。"
)


def make_document(**overrides: Any) -> KnowledgeDocument:
    """构造一篇通过完整校验的文档;`overrides` 用于制造**非法**变体。"""
    payload: dict[str, Any] = {
        "doc_id": "kb-cve-2021-44228",
        "title": "合成测试文档 —— Log4j JNDI 注入",
        "source_type": "cve",
        "source_id": "CVE-2021-44228",
        "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
        "publisher": "self-authored-summary",
        "authored_at": "2026-10-17",
        "provenance": PROVENANCE,
        "text": DEFAULT_TEXT,
    }
    payload.update(overrides)
    return KnowledgeDocument.model_validate(payload)


def make_attack_document(**overrides: Any) -> KnowledgeDocument:
    payload: dict[str, Any] = {
        "doc_id": "kb-attack-t1190",
        "title": "合成测试文档 —— 利用对外暴露的应用",
        "source_type": "attack-technique",
        "source_id": "T1190",
        "source_url": "https://attack.mitre.org/techniques/T1190/",
    }
    payload.update(overrides)
    return make_document(**payload)


@pytest.fixture(scope="session")
def knowledge_dir() -> Path:
    return KNOWLEDGE_DIR


@pytest.fixture(scope="session")
def corpus(knowledge_dir: Path) -> LoadedCorpus:
    return load_corpus(knowledge_dir)


@pytest.fixture(scope="session")
def documents(corpus: LoadedCorpus) -> tuple[KnowledgeDocument, ...]:
    return corpus.documents


@pytest.fixture(scope="session")
def chunks(corpus: LoadedCorpus) -> list[KnowledgeChunk]:
    return chunk_corpus(corpus.documents)


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture(scope="session")
def retriever(corpus: LoadedCorpus, embedder: HashingEmbedder) -> Retriever:
    return build_in_memory_retriever(corpus, embedder)


@pytest.fixture(scope="session")
def eval_queries() -> list[dict[str, Any]]:
    raw = EVAL_QUERIES_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


@pytest.fixture(scope="session")
def doc_factory() -> Callable[..., KnowledgeDocument]:
    return make_document


@pytest.fixture(scope="session")
def attack_doc_factory() -> Callable[..., KnowledgeDocument]:
    return make_attack_document
