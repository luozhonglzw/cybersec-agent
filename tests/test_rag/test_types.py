"""数据契约测试:ID 推导、元数据校验、状态不变式。

这些断言针对的是"脏数据会不会流进检索",而不是"字段能不能存下来"。
每一条失败路径都对应一种真实的语料漂移:自造编号、双重前缀、错误主机、
缺免责声明、块偏移与原文对不上。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.types import (
    MAX_DOC_TEXT_CHARS,
    MAX_TOP_K,
    MIN_DOC_TEXT_CHARS,
    CorpusManifest,
    IndexMetadata,
    KnowledgeChunk,
    RetrievalHit,
    RetrievalResult,
    clamp_top_k,
    derive_doc_id,
    normalize_query,
    sha256_hex,
    snippet_of,
)


# ---------- doc_id 推导 ----------

def test_cve_doc_id_is_the_lowercased_official_id():
    """CVE 的 doc_id 不再拼一次 `cve` 前缀 —— 否则会变成 kb-cve-cve-…。"""
    assert derive_doc_id("cve", "CVE-2021-44228") == "kb-cve-2021-44228"


def test_attack_doc_id_carries_the_attack_prefix():
    assert derive_doc_id("attack-technique", "T1190") == "kb-attack-t1190"


def test_advisory_doc_id_carries_the_adv_prefix():
    assert derive_doc_id("advisory", "GHSA-xxxx-yyyy-zzzz") == "kb-adv-ghsa-xxxx-yyyy-zzzz"


def test_unknown_source_type_cannot_derive_a_doc_id():
    """静默回退到某个默认前缀会让拼写错误产出"看起来合法"的 id。"""
    with pytest.raises(ValueError, match="unknown source_type"):
        derive_doc_id("cves", "CVE-2021-44228")


def test_doc_id_must_be_derived_from_its_source(doc_factory):
    with pytest.raises(ValidationError, match="is not derived from its source"):
        doc_factory(doc_id="kb-cve-2021-99999")


def test_doc_id_shape_is_enforced(doc_factory):
    with pytest.raises(ValidationError, match="required pattern"):
        doc_factory(doc_id="log4shell")


# ---------- 官方命名空间 ----------

@pytest.mark.parametrize(
    ("source_type", "source_id", "doc_id"),
    [
        ("cve", "CVE-21-1234", "kb-cve-21-1234"),
        ("cve", "CVE-2021-44", "kb-cve-2021-44"),
        ("attack-technique", "T119", "kb-attack-t119"),
    ],
)
def test_source_id_must_match_its_namespace(doc_factory, source_type, source_id, doc_id):
    """编号形态不对时,即使 doc_id 与编号自洽也必须拒绝。"""
    with pytest.raises(ValidationError, match="does not match"):
        doc_factory(source_type=source_type, source_id=source_id, doc_id=doc_id)


# ---------- 来源链接 ----------

@pytest.mark.parametrize(
    "url",
    [
        "http://nvd.nist.gov/vuln/detail/CVE-2021-44228",
        "https://evil.example.com/vuln/detail/CVE-2021-44228",
        "https://nvd.nist.gov",
        "https://user:pw@nvd.nist.gov/vuln/detail/CVE-2021-44228",
        "https://nvd.nist.gov:8443/vuln/detail/CVE-2021-44228",
    ],
)
def test_source_url_must_be_https_on_an_allowlisted_host(doc_factory, url):
    """`source_url` 只是展示数据;白名单让"来源链接"不可能指向任意地址。"""
    with pytest.raises(ValidationError):
        doc_factory(source_url=url)


# ---------- 出处 ----------

def test_provenance_must_state_that_it_is_not_official(doc_factory):
    with pytest.raises(ValidationError, match="must state"):
        doc_factory(
            provenance=(
                "这是一段自写摘要,用于演示语料格式,不涉及任何外部来源,"
                "也不声称覆盖完整或及时;外部数据集的许可条件尚未核验。"
            )
        )


def test_provenance_must_be_long_enough(doc_factory):
    with pytest.raises(ValidationError, match="at least"):
        doc_factory(provenance="NOT an official record")


def test_publisher_allowlist_is_enforced(doc_factory):
    """B-3:外部数据集许可未核验前,只允许自写摘要。"""
    with pytest.raises(ValidationError, match="publisher must be"):
        doc_factory(publisher="nvd-mirror")


def test_source_provenance_view_is_consistent(doc_factory):
    document = doc_factory()
    provenance = document.source_provenance
    assert provenance.publisher == document.publisher
    assert provenance.authored_at == document.authored_at
    assert provenance.note == document.provenance


# ---------- 未知字段与长度 ----------

def test_unknown_fields_are_rejected(doc_factory):
    """宽容解析会把"评审过的内容"变成"没评审过的内容"。"""
    with pytest.raises(ValidationError, match="Extra inputs"):
        doc_factory(confidence=0.9)


@pytest.mark.parametrize("length", [MIN_DOC_TEXT_CHARS - 1, MAX_DOC_TEXT_CHARS + 1])
def test_text_length_bounds(doc_factory, length):
    with pytest.raises(ValidationError, match="characters"):
        doc_factory(text="字" * length)


def test_documents_are_immutable(doc_factory):
    document = doc_factory()
    with pytest.raises(ValidationError):
        document.text = "改写"  # type: ignore[misc]


# ---------- 切块 ----------

def test_chunk_span_must_equal_its_text(doc_factory):
    """偏移与原文必须能互相核对,否则切块会静默错位。"""
    with pytest.raises(ValidationError, match="span must equal"):
        KnowledgeChunk(
            chunk_id="kb-cve-2021-44228#0",
            doc_id="kb-cve-2021-44228",
            ordinal=0,
            text="abc",
            char_start=0,
            char_end=99,
        )


def test_chunk_id_must_match_doc_and_ordinal():
    with pytest.raises(ValidationError, match="must equal"):
        KnowledgeChunk(
            chunk_id="kb-cve-2021-44228#7",
            doc_id="kb-cve-2021-44228",
            ordinal=0,
            text="abc",
            char_start=0,
            char_end=3,
        )


def test_chunk_rejects_empty_span():
    with pytest.raises(ValidationError, match="greater than char_start"):
        KnowledgeChunk(
            chunk_id="kb-cve-2021-44228#0",
            doc_id="kb-cve-2021-44228",
            ordinal=0,
            text="x",
            char_start=5,
            char_end=5,
        )


# ---------- 命中与结果 ----------

def _hit(**overrides):
    payload = {
        "citation_id": "kb-cve-2021-44228",
        "doc_id": "kb-cve-2021-44228",
        "chunk_id": "kb-cve-2021-44228#0",
        "source_id": "CVE-2021-44228",
        "source_type": "cve",
        "title": "合成标题",
        "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
        "score": 0.5,
        "snippet": "片段",
    }
    payload.update(overrides)
    return RetrievalHit.model_validate(payload)


def test_hit_citation_id_must_equal_doc_id():
    """引用粒度是文档;允许两者分叉就等于允许引用指向不存在的来源。"""
    with pytest.raises(ValidationError, match="citation_id must equal doc_id"):
        _hit(citation_id="S1")


def test_hit_is_immutable():
    hit = _hit()
    with pytest.raises(ValidationError):
        hit.score = 0.9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("status", "hits"),
    [("ok", []), ("empty", [_hit()]), ("unavailable", [_hit()])],
)
def test_result_status_must_match_its_hits(status, hits):
    with pytest.raises(ValidationError, match="must not carry hits|requires at least one hit"):
        RetrievalResult(
            status=status, query_sha256=sha256_hex("q"), top_k=5, hits=hits
        )


def test_result_query_hash_must_be_sha256_hex():
    with pytest.raises(ValidationError, match="64 lowercase hex"):
        RetrievalResult(status="empty", query_sha256="not-a-hash", top_k=5)


def test_result_top_k_is_bounded_by_the_contract():
    with pytest.raises(ValidationError):
        RetrievalResult(status="empty", query_sha256=sha256_hex("q"), top_k=MAX_TOP_K + 1)


# ---------- 清单与索引身份 ----------

def test_manifest_requires_a_versioned_corpus_id():
    with pytest.raises(ValidationError):
        CorpusManifest(
            corpus_version="0.5.0",
            doc_count=1,
            corpus_sha256=sha256_hex("x"),
            authored_from="self-authored",
            generated_at="2026-10-17",
        )


def test_manifest_rejects_non_self_authored_sources():
    with pytest.raises(ValidationError):
        CorpusManifest(
            corpus_version="v0.5.0-s2.1",
            doc_count=1,
            corpus_sha256=sha256_hex("x"),
            authored_from="nvd-dump",
            generated_at="2026-10-17",
        )


def test_manifest_hash_must_be_sha256_hex():
    with pytest.raises(ValidationError, match="64 lowercase hex"):
        CorpusManifest(
            corpus_version="v0.5.0-s2.1",
            doc_count=1,
            corpus_sha256="deadbeef",
            authored_from="self-authored",
            generated_at="2026-10-17",
        )


def test_index_metadata_rejects_short_hash():
    with pytest.raises(ValidationError, match="64 lowercase hex"):
        IndexMetadata(
            corpus_version="v0.5.0-s2.1",
            corpus_sha256="abc",
            embedder_id="hashing-v1-d256-n3",
            dim=256,
            chunker_version="paragraph-window-v1",
            doc_count=1,
            chunk_count=2,
        )


# ---------- 纯函数 ----------

@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, 5), (-3, 1), (0, 1), (1, 1), (5, 5), (MAX_TOP_K, MAX_TOP_K), (999, MAX_TOP_K)],
)
def test_clamp_top_k(requested, expected):
    """夹紧而不是报错:让一次越界调用失败只会诱发重试。"""
    assert clamp_top_k(requested) == expected


def test_normalize_query_collapses_whitespace_only():
    """大小写是 CVE / ATT&CK 编号语义的一部分,规范化层不折叠它。"""
    assert normalize_query("  Log4j \n  JNDI\t攻击  ") == "Log4j JNDI 攻击"


def test_snippet_is_single_line_and_bounded():
    assert snippet_of("a\n\nb   c", 100) == "a b c"
    snippet = snippet_of("字" * 50, 10)
    assert len(snippet) == 10
    assert snippet.endswith("…")
