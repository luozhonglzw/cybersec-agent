"""RAG 数据契约:语料、切块、检索命中与索引元数据。

设计原则(与 `app/schemas/**` 同一风格):

- **枚举即校验**:`source_type` 用 `Literal` 收口,`source_id` 按类型分别
  正则校验。脏元数据在 Pydantic 层就失败,不会带着"看起来像 CVE 但不是"
  的来源流进检索结果。
- **id 由来源推导,不由人填**:`doc_id == f"kb-{prefix}-{source_id.lower()}"`。
  自由填写 id 的语料会随时间漂移(同一个 CVE 出现两个 id),而引用去重
  完全依赖 id 稳定 —— 推导把这条从"约定"变成"结构"。
- **`citation_id` 就是 `doc_id`**:文档级,不是排名级。排名级 id 在多次工具
  调用之间会冲突(第二次的 `S1` 与第一次的 `S1` 不是同一文档),而引用
  校验要在**并集**上做(见 S1 冻结契约 §2.5)。
- **hash 统一 sha256 小写十六进制**:语料完整性、索引身份、查询指纹共用
  一种表示,断言与比对不需要转换。
- **fail closed**:未知字段、越界长度、非白名单主机一律拒绝。语料是受
  版本控制的输入,任何"宽容解析"都会把评审过的内容变成没评审过的。

本模块**只做数据**,不做 I/O、不读文件、不建连接 —— 便于在纯内存里测。
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: 服务端对检索深度的硬上限。客户端不参与决定(请求体没有 `top_k`),
#: 工具侧可调但被夹在这个区间内 —— 与 S1 §2.3 的冻结值一致。
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

#: 文档正文长度区间。语料是人工撰写并评审的摘要,不是抓取正文:
#: 下界防止"一句话文档"污染检索,上界防止把整篇公告塞进上下文。
MIN_DOC_TEXT_CHARS = 200
MAX_DOC_TEXT_CHARS = 600

#: `doc_id` 形态。与 `derive_doc_id()` 的结果严格一致。
DOC_ID_PATTERN = re.compile(r"^kb-(cve|attack|adv)-[a-z0-9-]{1,40}$")

#: `source_type` → `doc_id` 前缀。注意 CVE 的前缀是**空串**:官方标识本身
#: 已带 `CVE-`,再拼一次会得到 `kb-cve-cve-2021-44228` 这种双重前缀。
SOURCE_TYPE_PREFIX: dict[str, str] = {
    "cve": "",
    "attack-technique": "attack-",
    "advisory": "adv-",
}

#: 各类型官方标识的形态。**只接受官方命名空间**,不接受自造 id。
SOURCE_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "cve": re.compile(r"^CVE-\d{4}-\d{4,7}$"),
    "attack-technique": re.compile(r"^T\d{4}$"),
    "advisory": re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{2,63}$"),
}

#: 允许出现在 `source_url` 里的主机。**只作为展示数据**,服务端与工具
#: 从不请求它;白名单的作用是让"来源链接"不可能指向任意地址。
ALLOWED_SOURCE_HOSTS: frozenset[str] = frozenset({
    "nvd.nist.gov",
    "attack.mitre.org",
    "github.com",
})

#: 允许的 `publisher`。B-3 决定:外部数据集许可未核验前,语料只允许
#: 自写摘要 —— 这不是风格偏好,是授权边界,所以写进校验。
ALLOWED_PUBLISHERS: frozenset[str] = frozenset({"self-authored-summary"})

#: `provenance` 必须包含的免责标记(大小写不敏感)。
PROVENANCE_MARKER = "not an official"

#: `provenance` 的最小长度:一句话免责声明不足以说明"这是自写的"。
MIN_PROVENANCE_CHARS = 40

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")

SourceType = Literal["cve", "attack-technique", "advisory"]
RetrievalStatus = Literal["ok", "empty", "unavailable"]


def sha256_hex(payload: str | bytes) -> str:
    """sha256 → 小写十六进制。str 按 UTF-8 编码。"""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(raw).hexdigest()


def is_sha256_hex(value: str) -> bool:
    return bool(SHA256_HEX_PATTERN.match(value))


def derive_doc_id(source_type: str, source_id: str) -> str:
    """由来源推导稳定的 `doc_id`。

    未知 `source_type` 直接报错而不是回退到某个默认前缀 —— 静默回退会让
    拼写错误(`"cves"`)产出看起来合法的 id。
    """
    prefix = SOURCE_TYPE_PREFIX.get(source_type)
    if prefix is None:
        raise ValueError(f"unknown source_type: {source_type!r}")
    return f"kb-{prefix}{source_id.lower()}"


def _split_https_url(url: str) -> tuple[str, str]:
    """把 `https://host/path` 拆成 (host, path),拒绝一切奇怪形态。

    刻意不 import `urllib.parse`:本包不允许出现任何看起来像"取数据"的
    模块(见 `tests/test_rag/test_security.py` 的导入边界断言),手写
    解析只有十几行,却让"这个包不会发请求"更容易被机械证明。
    """
    prefix = "https://"
    if not url.startswith(prefix):
        raise ValueError("source_url must use https")
    rest = url[len(prefix):]
    authority, sep, path = rest.partition("/")
    if not authority:
        raise ValueError("source_url must contain a host")
    if not sep or not path.strip():
        raise ValueError("source_url must contain a path")
    if "@" in authority:
        raise ValueError("source_url must not contain credentials")
    if ":" in authority:
        raise ValueError("source_url must not contain an explicit port")
    return authority.lower(), path


def _check_publisher(value: str) -> str:
    if value not in ALLOWED_PUBLISHERS:
        raise ValueError(
            "publisher must be one of "
            f"{sorted(ALLOWED_PUBLISHERS)} —— 外部数据集许可未核验前"
            "只允许自写摘要(B-3)"
        )
    return value


def _check_provenance_note(value: str) -> str:
    if len(value) < MIN_PROVENANCE_CHARS:
        raise ValueError(
            f"provenance must be at least {MIN_PROVENANCE_CHARS} characters"
        )
    if PROVENANCE_MARKER not in value.lower():
        raise ValueError(
            f"provenance must state {PROVENANCE_MARKER!r} explicitly"
        )
    return value


class SourceProvenance(BaseModel):
    """来源出处:谁写的、什么时候写的、以及"这不是官方数据"的明示。

    单独建模而不是散落成三个字段,是因为它会被**多处**要求:语料校验、
    文档展示、以及未来引用列表里的来源说明。三处若各自校验,迟早在某一
    处漏掉免责标记。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    publisher: str = Field(description="发布方;当前只允许自写摘要")
    authored_at: date = Field(description="撰写日期(ISO 8601)")
    note: str = Field(
        min_length=MIN_PROVENANCE_CHARS,
        description="免责声明原文,必须明示不是官方记录",
    )

    @field_validator("publisher")
    @classmethod
    def _validate_publisher(cls, value: str) -> str:
        return _check_publisher(value)

    @field_validator("note")
    @classmethod
    def _validate_note(cls, value: str) -> str:
        return _check_provenance_note(value)


class KnowledgeDocument(BaseModel):
    """一篇知识文档 —— `knowledge/corpus.jsonl` 的一行。

    字段是**扁平**的(而不是把 `SourceProvenance` 嵌成子对象),因为语料
    文件是人工评审的对象:扁平一行更容易 diff。需要结构化出处时用
    `source_provenance` 属性,校验逻辑仍然只有一份。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(description="稳定文档 id,由 source_type + source_id 推导")
    title: str = Field(min_length=4, max_length=120, description="人可读标题")
    source_type: SourceType = Field(description="来源类别(受限枚举)")
    source_id: str = Field(description="官方标识,如 CVE-2021-44228 / T1190")
    source_url: str = Field(description="官方页面链接;仅展示,从不请求")
    publisher: str = Field(description="发布方;当前只允许自写摘要")
    authored_at: date = Field(description="撰写日期(ISO 8601)")
    provenance: str = Field(
        min_length=MIN_PROVENANCE_CHARS,
        description="免责声明原文,必须明示不是官方记录",
    )
    text: str = Field(
        min_length=MIN_DOC_TEXT_CHARS,
        max_length=MAX_DOC_TEXT_CHARS,
        description="自写摘要正文;段落之间用空行分隔",
    )

    @field_validator("source_id")
    @classmethod
    def _validate_source_id(cls, value: str, info) -> str:
        source_type = info.data.get("source_type")
        if source_type is None:
            # source_type 校验失败时,这里不再叠加一条误导性的错误。
            return value
        pattern = SOURCE_ID_PATTERNS[source_type]
        if not pattern.match(value):
            raise ValueError(
                f"source_id {value!r} does not match {source_type} namespace"
            )
        return value

    @field_validator("doc_id")
    @classmethod
    def _validate_doc_id_shape(cls, value: str) -> str:
        if not DOC_ID_PATTERN.match(value):
            raise ValueError(f"doc_id {value!r} does not match required pattern")
        return value

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str) -> str:
        host, _ = _split_https_url(value)
        if host not in ALLOWED_SOURCE_HOSTS:
            raise ValueError(
                f"source_url host {host!r} is not in the allow-list "
                f"{sorted(ALLOWED_SOURCE_HOSTS)}"
            )
        return value

    @field_validator("publisher")
    @classmethod
    def _validate_publisher(cls, value: str) -> str:
        return _check_publisher(value)

    @field_validator("provenance")
    @classmethod
    def _validate_provenance(cls, value: str) -> str:
        return _check_provenance_note(value)

    @model_validator(mode="after")
    def _validate_doc_id_matches_source(self) -> "KnowledgeDocument":
        expected = derive_doc_id(self.source_type, self.source_id)
        if self.doc_id != expected:
            raise ValueError(
                f"doc_id {self.doc_id!r} is not derived from its source "
                f"(expected {expected!r}) —— 引用去重依赖 id 稳定"
            )
        return self

    @property
    def source_provenance(self) -> SourceProvenance:
        """结构化出处视图(与扁平字段同源,不重复校验逻辑)。"""
        return SourceProvenance(
            publisher=self.publisher,
            authored_at=self.authored_at,
            note=self.provenance,
        )


class KnowledgeChunk(BaseModel):
    """文档的一个切块。

    `chunk_id` 是 `{doc_id}#{ordinal}`,**ordinal 由切块器按文档内顺序
    分配**。它参与确定性 tie-break(同分按 `chunk_id` 升序),因此不能
    用随机串或内容哈希之外的东西。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=3, max_length=80, description="{doc_id}#{ordinal}")
    doc_id: str = Field(description="所属文档 id")
    ordinal: int = Field(ge=0, description="文档内序号,从 0 递增")
    text: str = Field(min_length=1, description="切块原文(未 strip,便于核对偏移)")
    char_start: int = Field(ge=0, description="相对文档正文的起始偏移")
    char_end: int = Field(ge=0, description="相对文档正文的结束偏移(不含)")

    @model_validator(mode="after")
    def _validate_identity_and_span(self) -> "KnowledgeChunk":
        expected_id = f"{self.doc_id}#{self.ordinal}"
        if self.chunk_id != expected_id:
            raise ValueError(
                f"chunk_id {self.chunk_id!r} must equal {expected_id!r}"
            )
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError(
                "chunk span must equal len(text) —— 偏移与原文必须可互相核对"
            )
        return self


class RetrievalHit(BaseModel):
    """一条检索命中(文档级)。

    `citation_id == doc_id`:回答里的 `[src:...]` 标记指向**文档**,而不是
    "本次排名的第几条"。这样同一文档被多次检索命中时,引用不会互相冲突。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str = Field(description="引用 id,等于 doc_id")
    doc_id: str = Field(description="文档 id")
    chunk_id: str = Field(description="命中的切块 id")
    source_id: str = Field(description="官方标识")
    source_type: SourceType = Field(description="来源类别")
    title: str = Field(description="文档标题")
    url: str = Field(description="来源链接;仅作为数据返回,从不请求")
    score: float = Field(description="相似度分数(越大越相关)")
    snippet: str = Field(min_length=1, description="命中片段(长度受限)")

    @model_validator(mode="after")
    def _validate_citation_id(self) -> "RetrievalHit":
        if self.citation_id != self.doc_id:
            raise ValueError("citation_id must equal doc_id")
        return self


class RetrievalResult(BaseModel):
    """一次检索的完整结果。

    `status` 的三态是**契约**,不是提示:
    - `ok`  —— 至少一条命中;
    - `empty` —— 检索成功但无相关命中(不是错误);
    - `unavailable` —— 索引/嵌入不可用(fail closed)。

    `query_sha256` 而非 query 原文:结果会被写进日志与 checkpoint,
    没必要把用户问题原文一起带过去;指纹足够做关联与去重。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RetrievalStatus = Field(description="ok / empty / unavailable")
    query_sha256: str = Field(description="规范化后查询的 sha256")
    top_k: int = Field(ge=1, le=MAX_TOP_K, description="本次生效的检索深度")
    hits: list[RetrievalHit] = Field(default_factory=list, description="按分数降序")
    corpus_version: str = Field(default="", description="产出本次结果的语料版本")

    @field_validator("query_sha256")
    @classmethod
    def _validate_query_hash(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("query_sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _validate_status_matches_hits(self) -> "RetrievalResult":
        if self.status == "ok" and not self.hits:
            raise ValueError("status 'ok' requires at least one hit")
        if self.status in ("empty", "unavailable") and self.hits:
            raise ValueError(f"status {self.status!r} must not carry hits")
        return self


class CorpusManifest(BaseModel):
    """语料清单 —— `knowledge/manifest.json`。

    `corpus_sha256` 钉住 `corpus.jsonl` 的**字节内容**。改语料忘改清单
    会立刻变红,而不是让索引悄悄落后于语料。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_version: str = Field(
        pattern=r"^v\d+\.\d+\.\d+-[a-z0-9]+(?:\.[a-z0-9]+)*$",
        description="语料版本,如 v0.5.0-s2.1",
    )
    doc_count: int = Field(ge=1, description="文档条数")
    corpus_sha256: str = Field(description="corpus.jsonl 的 sha256")
    authored_from: Literal["self-authored"] = Field(
        description="语料来源;只允许 self-authored"
    )
    generated_at: date = Field(description="清单生成日期")

    @field_validator("corpus_sha256")
    @classmethod
    def _validate_corpus_hash(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("corpus_sha256 must be 64 lowercase hex characters")
        return value


class IndexMetadata(BaseModel):
    """索引身份 —— 决定"这个索引还能不能用来回答当前语料"。

    五个字段缺一不可:语料版本/哈希(内容对不对)、embedder 身份与维度
    (向量空间对不对)、切块器版本(切法与 id 分配对不对)。任一项不匹配
    即 `IndexUnavailable`,而不是"凑合用旧索引"。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_version: str = Field(min_length=1, description="构建时的语料版本")
    corpus_sha256: str = Field(description="构建时的语料 sha256")
    embedder_id: str = Field(min_length=1, description="嵌入器身份(含参数)")
    dim: int = Field(ge=1, description="向量维度")
    chunker_version: str = Field(min_length=1, description="切块器版本")
    doc_count: int = Field(ge=0, description="构建时文档数")
    chunk_count: int = Field(ge=0, description="构建时切块数")

    @field_validator("corpus_sha256")
    @classmethod
    def _validate_corpus_hash(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("corpus_sha256 must be 64 lowercase hex characters")
        return value


class ScoredChunk(BaseModel):
    """存储层返回的"切块 + 分数"。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: KnowledgeChunk
    score: float


def normalize_query(text: str) -> str:
    """查询规范化:折叠空白并去首尾。

    **不做**大小写折叠 —— CVE 编号与 ATT&CK 编号的大小写是语义的一部分,
    而 `HashingEmbedder` 自己在向量化时会折叠。保持这一层可解释:
    规范化只处理"肉眼不可见"的差异。
    """
    return " ".join(text.split())


def clamp_top_k(value: int | None, *, default: int = DEFAULT_TOP_K) -> int:
    """把请求的 `top_k` 夹到 `[1, MAX_TOP_K]`。

    夹紧而不是报错:模型可能给出 `0` 或 `50`,让工具直接失败只会诱发
    重试;夹到合法区间既保留了意图,又不会让一次调用把整个语料拖进上下文。
    """
    if value is None:
        return default
    return max(1, min(MAX_TOP_K, int(value)))


def snippet_of(text: str, limit: int) -> str:
    """截取片段并折叠为单行 —— 片段会进上下文,不该带排版噪声。"""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def sorted_by_score(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    """按 `(-score, citation_id)` 排序 —— 确定性契约的唯一实现。"""
    return sorted(hits, key=lambda hit: (-hit.score, hit.citation_id))
