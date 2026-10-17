"""语料与清单测试:完整性、来源声明、失败语义、CWD 无关。

这里测的不是"文件存在",而是"语料与清单不能各自漂移":
改了语料忘改哈希、改了条数忘改清单、重复了 doc_id —— 三种漂移都必须
在**加载时**响亮失败,而不是让索引悄悄落后于语料。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.corpus import (
    CORPUS_FILENAME,
    MANIFEST_FILENAME,
    corpus_sha256,
    load_corpus,
    parse_documents,
    read_manifest,
)
from app.rag.errors import CorpusError
from app.rag.types import (
    MAX_DOC_TEXT_CHARS,
    MIN_DOC_TEXT_CHARS,
    KnowledgeDocument,
    sha256_hex,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = _REPO_ROOT / "knowledge"


def _corpus_lines() -> list[str]:
    raw = (KNOWLEDGE_DIR / CORPUS_FILENAME).read_text(encoding="utf-8")
    return [line for line in raw.splitlines() if line.strip()]


def _write_corpus_dir(tmp_path: Path, lines: list[str], **manifest_overrides) -> Path:
    """把给定的语料行写成一份自洽的临时语料(哈希与条数都重新算)。"""
    target = tmp_path / "knowledge"
    target.mkdir()
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    (target / CORPUS_FILENAME).write_bytes(raw)

    manifest = {
        "corpus_version": "v0.5.0-s2.1",
        "doc_count": len(lines),
        "corpus_sha256": sha256_hex(raw),
        "authored_from": "self-authored",
        "generated_at": "2026-10-17",
    }
    manifest.update(manifest_overrides)
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return target


# ---------- 真实语料 ----------

def test_manifest_hash_matches_the_corpus_bytes(knowledge_dir):
    manifest = read_manifest(knowledge_dir)
    assert manifest.corpus_sha256 == corpus_sha256(
        (knowledge_dir / CORPUS_FILENAME).read_bytes()
    )


def test_manifest_doc_count_matches_the_corpus(corpus, knowledge_dir):
    assert corpus.manifest.doc_count == len(corpus)
    assert len(_corpus_lines()) == len(corpus)


def test_documents_are_sorted_by_doc_id(corpus):
    doc_ids = [document.doc_id for document in corpus.documents]
    assert doc_ids == sorted(doc_ids)
    assert corpus.doc_ids == tuple(doc_ids)


def test_doc_ids_are_unique(corpus):
    assert len(set(corpus.doc_ids)) == len(corpus)


def test_every_document_declares_self_authored_provenance(corpus):
    """B-3 的强制点:语料里不允许出现"看起来像官方数据"的文档。"""
    for document in corpus.documents:
        assert document.publisher == "self-authored-summary"
        assert "not an official" in document.provenance.lower()


def test_corpus_covers_both_source_types(corpus):
    source_types = {document.source_type for document in corpus.documents}
    assert source_types == {"cve", "attack-technique"}


def test_corpus_texts_are_within_the_contract_bounds(corpus):
    for document in corpus.documents:
        assert MIN_DOC_TEXT_CHARS <= len(document.text) <= MAX_DOC_TEXT_CHARS


def test_corpus_is_small_and_curated(corpus):
    """示例语料,不是数据集:规模刻意保持在"可人工评审"的范围。"""
    assert 20 <= len(corpus) <= 60


def test_every_document_has_at_least_one_paragraph_break(corpus):
    """多段落是切块路径的真实输入;没有它,切块器只在合成夹具上被验证。"""
    assert all("\n\n" in document.text for document in corpus.documents)


def test_loaded_corpus_supports_lookup(corpus):
    document = corpus.documents[0]
    assert corpus.by_id(document.doc_id) is document
    with pytest.raises(KeyError):
        corpus.by_id("kb-cve-1999-0000")


# ---------- 加载与 CWD 无关 ----------

def test_loading_does_not_depend_on_the_current_directory(tmp_path, monkeypatch):
    target = _write_corpus_dir(tmp_path, _corpus_lines())
    monkeypatch.chdir(tmp_path)
    loaded = load_corpus(target)
    assert len(loaded) == len(_corpus_lines())


# ---------- 漂移检测 ----------

def test_manifest_hash_mismatch_is_detected(tmp_path):
    target = _write_corpus_dir(tmp_path, _corpus_lines())
    corpus_path = target / CORPUS_FILENAME
    corpus_path.write_bytes(corpus_path.read_bytes().replace(b"Log4j", b"Log4X", 1))

    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "manifest-mismatch"


def test_doc_count_mismatch_is_detected(tmp_path):
    target = _write_corpus_dir(tmp_path, _corpus_lines(), doc_count=999)
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "doc-count-mismatch"


def test_duplicate_doc_id_is_detected(tmp_path):
    lines = _corpus_lines()
    target = _write_corpus_dir(tmp_path, [*lines, lines[0]])
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "duplicate-doc-id"


def test_invalid_document_reports_its_line_number(tmp_path):
    lines = _corpus_lines()
    payload = json.loads(lines[2])
    payload["doc_id"] = "log4shell"
    lines[2] = json.dumps(payload, ensure_ascii=False)
    target = _write_corpus_dir(tmp_path, lines)

    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "invalid-document"
    assert "line 3" in str(excinfo.value)


def test_non_json_line_is_rejected(tmp_path):
    lines = _corpus_lines()
    lines[0] = "{not json"
    target = _write_corpus_dir(tmp_path, lines)

    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "invalid-document"


def test_empty_corpus_is_rejected(tmp_path):
    target = _write_corpus_dir(tmp_path, ["   "])
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason in {"empty-corpus", "doc-count-mismatch"}


def test_missing_manifest_is_reported_generically(tmp_path):
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(tmp_path)
    assert excinfo.value.reason == "missing-manifest"


def test_missing_corpus_is_reported_generically(tmp_path):
    target = _write_corpus_dir(tmp_path, _corpus_lines())
    (target / CORPUS_FILENAME).unlink()
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "missing-corpus"


def test_invalid_manifest_json_is_rejected(tmp_path):
    target = _write_corpus_dir(tmp_path, _corpus_lines())
    (target / MANIFEST_FILENAME).write_text("{bad", encoding="utf-8")
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "invalid-manifest"


@pytest.mark.parametrize(
    "manifest_overrides",
    [
        {"corpus_sha256": "deadbeef"},
        {"corpus_version": "0.5.0"},
        {"authored_from": "nvd-dump"},
    ],
)
def test_invalid_manifest_fields_are_rejected(tmp_path, manifest_overrides):
    target = _write_corpus_dir(tmp_path, _corpus_lines(), **manifest_overrides)
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    assert excinfo.value.reason == "invalid-manifest"


def test_error_messages_never_contain_filesystem_paths(tmp_path):
    """异常消息会走到工具返回值与 HTTP 响应体,不能泄漏部署布局。"""
    target = _write_corpus_dir(tmp_path, _corpus_lines())
    corpus_path = target / CORPUS_FILENAME
    corpus_path.write_bytes(corpus_path.read_bytes().replace(b"Log4j", b"Log4X", 1))

    with pytest.raises(CorpusError) as excinfo:
        load_corpus(target)
    message = str(excinfo.value)
    for forbidden in ("\\", "/", ".jsonl", str(tmp_path)):
        assert forbidden not in message


# ---------- 标注查询集 ----------

def test_eval_queries_reference_existing_documents(eval_queries, corpus):
    known = set(corpus.doc_ids)
    for item in eval_queries:
        assert set(item["gold_doc_ids"]) <= known, item["query_id"]


def test_eval_query_ids_are_unique(eval_queries):
    query_ids = [item["query_id"] for item in eval_queries]
    assert len(set(query_ids)) == len(query_ids)


def test_eval_set_has_answerable_and_unanswerable_items(eval_queries):
    answerable = [item for item in eval_queries if item["answerable"]]
    unanswerable = [item for item in eval_queries if not item["answerable"]]
    assert len(eval_queries) == 20
    assert len(answerable) == 16
    assert len(unanswerable) == 4


def test_answerable_queries_have_gold_documents(eval_queries):
    for item in eval_queries:
        if item["answerable"]:
            assert item["gold_doc_ids"], item["query_id"]
        else:
            assert item["gold_doc_ids"] == [], item["query_id"]


def test_eval_queries_are_non_empty(eval_queries):
    assert all(item["query"].strip() for item in eval_queries)


# ---------- 解析器单元 ----------

def test_parse_documents_skips_blank_lines():
    raw = b"\n\n" + json.dumps(
        {
            "doc_id": "kb-cve-2021-44228",
            "title": "合成标题",
            "source_type": "cve",
            "source_id": "CVE-2021-44228",
            "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
            "publisher": "self-authored-summary",
            "authored_at": "2026-10-17",
            "provenance": (
                "Self-authored summary written for this repository's demo corpus. "
                "NOT an official NVD or MITRE ATT&CK record."
            ),
            "text": "甲" * 200,
        },
        ensure_ascii=False,
    ).encode("utf-8") + b"\n\n"
    documents = parse_documents(raw)
    assert len(documents) == 1
    assert isinstance(documents[0], KnowledgeDocument)
