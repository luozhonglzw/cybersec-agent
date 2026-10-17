"""嵌入接口与**离线确定性**基线实现。

`HashingEmbedder` 是什么,不是什么
--------------------------------
它是**可离线复现的检索基线**:把文本切成字符 n-gram,用带符号的特征哈希
映射到固定维度,再 L2 归一化。相同的文本在任何进程、任何机器上都得到
逐位相同的向量(哈希用 `blake2b`,不是 Python 内置 `hash()` —— 后者对
`str` 加了进程级随机盐,同一份语料在两次启动后向量不同,索引就再也
对不上了)。

它**不是**语义模型:它衡量的是字符片段重合度,不理解"同义"或"蕴含"。
因此用它跑出来的 `recall@5` / `MRR@5` 只描述**这个基线**,不得表述为
"RAG 的语义检索效果"。真实语义质量需要外部嵌入服务,属于后续阶段。

为什么把 `embedder_id` 编进参数
----------------------------
索引只能被**同一向量空间**的嵌入器查询。把 `dim` 与 `ngram` 编进 id,
换参数就换身份 —— 旧索引立刻被判为不可用,而不是用 256 维索引去接
384 维查询然后得到一堆看似合理的垃圾分数。
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

#: 嵌入器族名与版本。版本参与 `embedder_id`,因此改算法必须改版本。
EMBEDDER_FAMILY_HASHING = "hashing"
HASHING_VERSION = "v1"

#: 默认维度。256 维对 40 篇摘要足够(冲突率低),又足够小到可以人工检查。
DEFAULT_HASHING_DIM = 256

#: 默认 n-gram 长度。中文用 3 字窗口近似"词",英文 CVE 编号也能覆盖到。
DEFAULT_HASHING_NGRAM = 3

#: 维度下限:再小就退化成"按首字母分桶",相似度失去意义。
MIN_DIM = 8


@runtime_checkable
class Embedder(Protocol):
    """嵌入器契约。

    `embed_documents` 与 `embed_query` **必须**在同一向量空间 —— 这是
    "文档能命中查询"的前提。把它们分成两个方法而不是一个,是因为真实
    服务商对两者常用不同的输入前缀/指令,提前留出这个缝。
    """

    @property
    def embedder_id(self) -> str:
        """参与索引身份的标识(含维度与算法参数)。"""
        ...

    @property
    def dim(self) -> int:
        """向量维度。"""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量嵌入文档文本。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """嵌入单条查询。"""
        ...


def _normalize(text: str) -> str:
    """折叠大小写与空白 —— 嵌入层承担这一步,查询规范化层不重复。"""
    return " ".join(text.lower().split())


def _char_ngrams(text: str, n: int) -> list[str]:
    """字符 n-gram。短于 n 的文本整体作为一个 gram(而不是空列表)。"""
    if not text:
        return []
    if len(text) <= n:
        return [text]
    return [text[index : index + n] for index in range(len(text) - n + 1)]


def _signed_index(gram: str, dim: int) -> tuple[int, float]:
    """把 gram 映射到 (桶下标, 符号)。

    带符号哈希(signed hashing)而不是只累加 1:无符号版本会让所有特征
    正相关,余弦相似度整体偏高且区分度差;符号把无关特征推向相反方向,
    在同样维度下显著降低碰撞带来的伪相似。
    """
    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    index = value % dim
    sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
    return index, sign


class HashingEmbedder:
    """确定性字符 n-gram 特征哈希嵌入器(**离线基线**)。"""

    def __init__(
        self,
        *,
        dim: int = DEFAULT_HASHING_DIM,
        ngram: int = DEFAULT_HASHING_NGRAM,
    ) -> None:
        if dim < MIN_DIM:
            raise ValueError(f"dim must be >= {MIN_DIM}")
        if ngram < 1:
            raise ValueError("ngram must be >= 1")
        self._dim = dim
        self._ngram = ngram

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ngram(self) -> int:
        return self._ngram

    @property
    def embedder_id(self) -> str:
        return f"{EMBEDDER_FAMILY_HASHING}-{HASHING_VERSION}-d{self._dim}-n{self._ngram}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for gram in _char_ngrams(_normalize(text), self._ngram):
            index, sign = _signed_index(gram, self._dim)
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # 空文本 / 全空白的向量是零向量。调用方必须自己决定怎么处理
            # —— 检索层选择在嵌入之前就拒绝空查询,而不是让它检索出 0 分。
            return vector
        return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度。零向量与任何向量的相似度定义为 `0.0`(不是 NaN)。

    NaN 会污染排序(`NaN` 与任何值比较都为 False,排序结果依赖实现),
    而 `0.0` 是一个诚实的"没有相似性"。
    """
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
