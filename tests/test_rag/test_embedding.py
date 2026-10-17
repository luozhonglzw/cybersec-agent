"""嵌入测试:确定性(含跨进程)、归一化、相似度边界。

`HashingEmbedder` 是**离线基线**,不是语义模型。这里断言的是它的
*可复现性*,不是它的检索质量 —— 质量属于 L2 测量(`scripts/eval_rag.py`),
而且只描述这个基线。
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.embedding import (
    DEFAULT_HASHING_DIM,
    DEFAULT_HASHING_NGRAM,
    MIN_DIM,
    Embedder,
    HashingEmbedder,
    cosine_similarity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_SOURCE = REPO_ROOT / "app" / "rag" / "embedding.py"

SAMPLE_TEXT = "Log4j 的 JNDI 查找为什么能被用来执行代码?"


def _run_python(code: str, *, env_extra: dict[str, str] | None = None) -> str:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# ---------- 确定性 ----------

def test_embedding_is_deterministic_across_instances():
    first = HashingEmbedder()
    second = HashingEmbedder()
    assert first.embed_query(SAMPLE_TEXT) == second.embed_query(SAMPLE_TEXT)


def test_embedding_is_stable_across_processes():
    """跨进程一致 —— 这条排除了"用内置 hash()"这种实现。

    Python 对 `str` 的内置 `hash()` 带进程级随机盐,同一文本在两次启动后
    向量不同,索引就再也对不上了。`blake2b` 没有这个问题。
    """
    code = (
        "import json;"
        "from app.rag.embedding import HashingEmbedder;"
        f"print(json.dumps(HashingEmbedder().embed_query({SAMPLE_TEXT!r})))"
    )
    in_process = json.dumps(HashingEmbedder().embed_query(SAMPLE_TEXT))
    assert _run_python(code) == in_process


def test_the_cross_process_assertion_has_teeth():
    """负对照:内置 `hash()` **确实**跨进程不一致。

    没有这条,上面那条可能只是"两次都拿到全零向量"而恒真 ——
    一个恒真的确定性断言证明不了任何事。
    """
    code = "print(hash('abc'))"
    first = _run_python(code, env_extra={"PYTHONHASHSEED": "1"})
    second = _run_python(code, env_extra={"PYTHONHASHSEED": "2"})
    assert first != second


def test_embedding_does_not_call_builtin_hash():
    """结构断言:源码里不出现 `hash(...)` 调用(只允许 `hashlib`)。"""
    tree = ast.parse(EMBEDDING_SOURCE.read_text(encoding="utf-8"))
    offenders = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hash"
    ]
    assert offenders == []


def test_embed_documents_matches_embed_query(embedder):
    """文档与查询必须落在同一向量空间,否则"命中"无从谈起。"""
    assert embedder.embed_documents([SAMPLE_TEXT])[0] == embedder.embed_query(SAMPLE_TEXT)


def test_batch_embedding_preserves_order(embedder):
    texts = ["第一段文本", "第二段文本", "第三段文本"]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == len(texts)
    assert vectors[0] == embedder.embed_query(texts[0])
    assert vectors[2] == embedder.embed_query(texts[2])


# ---------- 向量形态 ----------

def test_vectors_are_unit_norm(embedder):
    vector = embedder.embed_query(SAMPLE_TEXT)
    assert len(vector) == embedder.dim
    norm = sum(value * value for value in vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-12)


def test_empty_text_yields_a_zero_vector(embedder):
    """零向量是诚实的"没有内容";检索层在嵌入之前就拒绝空查询。"""
    vector = embedder.embed_query("   \n\t ")
    assert len(vector) == embedder.dim
    assert set(vector) == {0.0}


def test_text_shorter_than_ngram_is_still_embedded():
    embedder = HashingEmbedder(ngram=3)
    vector = embedder.embed_query("ab")
    assert any(value != 0.0 for value in vector)


# ---------- 相似度 ----------

def test_identical_text_scores_one(embedder):
    vector = embedder.embed_query(SAMPLE_TEXT)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-12)


def test_related_text_scores_higher_than_unrelated(embedder):
    query = embedder.embed_query("Log4j 的 JNDI 查找导致远程代码执行")
    related = embedder.embed_query("Log4j 2 的 JNDI 查找允许加载远程对象,从而执行代码")
    unrelated = embedder.embed_query("蓝牙配对的密钥协商过程")
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_zero_vector_similarity_is_zero_not_nan():
    """NaN 会污染排序(`NaN` 与任何值比较都为 False),0.0 才是可排序的。"""
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# ---------- 参数与身份 ----------

def test_embedder_id_encodes_its_parameters():
    """换参数必须换身份 —— 否则 256 维索引会被 384 维查询"复用"。"""
    default = HashingEmbedder()
    assert default.embedder_id == (
        f"hashing-v1-d{DEFAULT_HASHING_DIM}-n{DEFAULT_HASHING_NGRAM}"
    )
    assert HashingEmbedder(dim=64).embedder_id != default.embedder_id
    assert HashingEmbedder(ngram=2).embedder_id != default.embedder_id


@pytest.mark.parametrize(
    ("dim", "ngram"),
    [(MIN_DIM - 1, 3), (256, 0), (256, -1)],
)
def test_invalid_parameters_are_rejected(dim, ngram):
    with pytest.raises(ValueError):
        HashingEmbedder(dim=dim, ngram=ngram)


def test_hashing_embedder_satisfies_the_protocol(embedder):
    assert isinstance(embedder, Embedder)
    assert embedder.dim == DEFAULT_HASHING_DIM
    assert embedder.ngram == DEFAULT_HASHING_NGRAM
