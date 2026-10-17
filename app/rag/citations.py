"""引用解析与校验 —— 把回答里的 `[src:...]` 标记对齐到"本次真实检索到的文档"。

这一层要回答的问题只有一个
------------------------
**"回答里引用的那份来源,是不是本次真的从知识库里检索出来的?"**

它**不**回答"每句话是否被来源支持" —— 那需要语义蕴含判断,属于离线标注
评测(见 S1 §4),本阶段明确不做。把这两件事混在一起,会让"引用有效"被
误读成"内容正确",而后者根本无法由结构计算得出。

三条不可退让的规则
----------------
**1. 证据只能来自真实工具执行结果。**
`collect_evidence()` 只认 `type == "tool"` 的消息,且其内容必须是一个
**合法**的 `kind == "rag_hits"` 信封。于是:

- 模型在 `AIMessage` 里写一段"看起来像工具结果"的 JSON —— 不是证据。
  模型**无法**生产 `ToolMessage`,这正是"伪造工具结果"不可行的原因。
- 普通字符串 / 其它工具的 JSON(没有 `kind`)—— 不是证据。
- `kind == "rag_hits"` 但字段不合法 —— 也**不是**证据(算"不可用")。

**2. 同一 `citation_id` 的元数据必须一致。**
两次工具调用命中同一篇文档,`score` 与 `snippet` 可以不同(命中的切块
不同),但 `doc_id / source_id / source_type / title / url` 必须完全相同。
任一不同即说明"同一个 id 指向了两份不同的来源",此时无法判断哪份可信
—— 整份证据按 `unavailable` 处理(fail closed),而不是挑一个看起来对的。

**3. 一次回答里只能有一个语料版本。**
不同 `corpus_version` 的证据混在一起,意味着索引在回答过程中被替换过。
没有经过验证的安全策略能合并两个版本的证据,因此同样 fail closed。

关于 `grounded`
--------------
`grounded` 是**结构性**属性,只由"检索状态 + 标记解析结果"计算,不读取
模型的任何自述。它的含义**仅**是:本次检索成功、至少有一条引用、且没有
任何编造引用。它**不**等价于"回答正确"。

`malformed_markers` 单独报告、但**不**参与 `grounded` 计算 —— 因为 S1
冻结的 `grounded` 公式只有三项(见 S1 §2.5)。这一点已作为待裁定项
N-4 上报,本模块不擅自修改冻结公式。
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from app.rag.retriever import SNIPPET_MAX_CHARS
from app.rag.types import (
    ALLOWED_SOURCE_HOSTS,
    DOC_ID_PATTERN,
    MAX_TOP_K,
    SourceType,
    is_sha256_hex,
)

#: 引用标记格式(S1 §2.5 冻结)。**只**接受这一种形态。
CITATION_MARKER_PATTERN = re.compile(r"\[src:([A-Za-z0-9][A-Za-z0-9._-]{0,63})\]")

#: "看起来像标记"的宽松扫描。用于把 `[src:]`、`[src: 带空格]`、
#: `[src:../../etc/passwd]` 这类**不是**合法标记的东西识别出来单独报告,
#: 而不是让它们静默消失。只作为诊断,永不产生引用。
MARKER_LIKE_PATTERN = re.compile(r"\[src:[^\[\]\n]*\]")

#: 语料版本形态 —— 与 `types.CorpusManifest.corpus_version` 同一约束。
#: 刻意复制一份而不是复用:信封是**外部输入**(来自工具消息文本),它的
#: 校验规则必须在本模块内可见,不能依赖"清单恰好也这么校验"。
CORPUS_VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+-[a-z0-9]+(?:\.[a-z0-9]+)*$")

#: 原因码形态。进日志、进断言,不进用户可见文案。
REASON_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

#: 信封 `kind` 词表。工具与校验器**共用**这两个常量,避免两侧各写一遍
#: 字符串后悄悄漂移。
RAG_HITS_KIND = "rag_hits"
RAG_ERROR_KIND = "rag_error"

#: 单条畸形标记在报告里的最大长度。原始文本可能很长(甚至是一整段),
#: 报告只需要"它长什么样"的线索。
MAX_REPORTED_MARKER_CHARS = 80

#: 畸形标记最多报告几条。再多也没有诊断价值,只会撑大响应体。
MAX_REPORTED_MARKERS = 20

#: 检索状态词表 —— 与 S1 §2.6 冻结的四值完全一致。**不**新增取值:
#: 元数据冲突表现为 `unavailable` + 具体原因码,而不是第五个状态。
EvidenceStatus = Literal["ok", "empty", "unavailable", "not_used"]


def _host_of(url: str) -> str:
    """取出 `https://host/path` 里的 host,并拒绝一切奇怪形态。

    与 `types._split_https_url` 同一规则,但刻意不 import 那个私有函数:
    信封来自**消息文本**,它的校验必须在本模块内可读。两处规则的一致性
    由测试保证(同一组坏 URL 在两处都必须被拒)。
    """
    prefix = "https://"
    if not url.startswith(prefix):
        raise ValueError("hit url must use https")
    rest = url[len(prefix):]
    authority, sep, path = rest.partition("/")
    if not authority or not sep or not path.strip():
        raise ValueError("hit url must contain a host and a path")
    if "@" in authority:
        raise ValueError("hit url must not contain credentials")
    if ":" in authority:
        raise ValueError("hit url must not contain an explicit port")
    return authority.lower()


def _check_reason(value: str) -> str:
    if not REASON_PATTERN.match(value):
        raise ValueError("reason must be a bounded lowercase code")
    return value


class EvidenceSource(BaseModel):
    """一条**可信**的检索命中 —— 同时是信封的 wire 模型与注册表条目。

    合并成一个模型而不是两个,是因为"能被解析"与"能被信任"必须是同一个
    判定:如果允许先解析再校验,就必然出现"解析成功但校验被跳过"的路径。

    `extra="forbid"` 是刻意的:信封多出一个字段就拒绝。工具是自己写的,
    多字段只可能来自伪造或版本错配 —— 两者都不该被"宽容解析"放过。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str = Field(description="引用 id,等于 doc_id")
    doc_id: str = Field(description="文档 id")
    source_id: str = Field(description="官方标识,如 CVE-2021-44228")
    source_type: SourceType = Field(description="来源类别")
    title: str = Field(min_length=1, max_length=200, description="文档标题")
    url: str = Field(min_length=1, max_length=300, description="来源链接;仅展示")
    score: float = Field(allow_inf_nan=False, description="相似度分数")
    snippet: str = Field(
        min_length=1,
        max_length=SNIPPET_MAX_CHARS,
        description="命中片段;长度受限,避免撑爆上下文",
    )

    @field_validator("citation_id", "doc_id")
    @classmethod
    def _validate_id_shape(cls, value: str) -> str:
        if not DOC_ID_PATTERN.match(value):
            raise ValueError("citation id does not match the frozen grammar")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        host = _host_of(value)
        if host not in ALLOWED_SOURCE_HOSTS:
            raise ValueError("hit url host is not in the allow-list")
        return value

    @model_validator(mode="after")
    def _validate_citation_id_equals_doc_id(self) -> "EvidenceSource":
        if self.citation_id != self.doc_id:
            raise ValueError("citation_id must equal doc_id")
        return self

    @property
    def provenance_key(self) -> tuple[str, str, str, str, str]:
        """判断"同一个 id 是否指向同一份来源"的比较键。

        **不含** `score` 与 `snippet`:同一文档在不同调用里命中不同切块,
        分数与片段本来就该不同。把它们算进身份,会把正常的多次检索误判成
        冲突,进而让每一条真实引用都变成不可信 —— 一个总是失败的护栏。
        """
        return (
            self.doc_id,
            self.source_id,
            self.source_type,
            self.title,
            self.url,
        )


class HitsEnvelope(BaseModel):
    """`kind == "rag_hits"` 的成功信封。

    `corpus_version` 是 S2-B 新增的必填字段(N-2):没有它就无法判断
    "同一次回答里的证据是否来自同一版语料"。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["rag_hits"]
    corpus_version: str = Field(min_length=1, max_length=64)
    query_sha256: str = Field(description="规范化查询的 sha256,不含查询原文")
    top_k: int = Field(ge=1, le=MAX_TOP_K)
    hits: tuple[EvidenceSource, ...] = Field(default_factory=tuple)

    @field_validator("corpus_version")
    @classmethod
    def _validate_corpus_version(cls, value: str) -> str:
        if not CORPUS_VERSION_PATTERN.match(value):
            raise ValueError("corpus_version does not match the required shape")
        return value

    @field_validator("query_sha256")
    @classmethod
    def _validate_query_hash(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("query_sha256 must be 64 lowercase hex characters")
        return value


class EvidenceCollection(BaseModel):
    """一次运行里"可用证据"的完整视图。

    `sources` 是**可信注册表**,只在 `status == "ok"` 时非空。冲突时注册表
    整体作废(`sources == ()`),但保留 `observed_ids` 与
    `conflicting_citations`,这样"哪些 id 被看到过、哪些自相矛盾"仍然
    可诊断 —— 诊断信息不是信任。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: EvidenceStatus = Field(description="ok / empty / unavailable / not_used")
    reason: str = Field(description="受限原因码,如 ok / citation-metadata-conflict")
    corpus_version: str = Field(default="", description="证据所属语料版本")
    sources: tuple[EvidenceSource, ...] = Field(
        default_factory=tuple, description="可信注册表,按 (-score, citation_id) 排序"
    )
    observed_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="在合法信封里出现过的全部 id(含冲突项)"
    )
    conflicting_citations: tuple[str, ...] = Field(
        default_factory=tuple, description="元数据自相矛盾的 id"
    )
    envelope_count: int = Field(default=0, ge=0, description="合法 rag_hits 信封数")
    hit_count: int = Field(default=0, ge=0, description="可信文档数")

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _check_reason(value)

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> "EvidenceCollection":
        if self.hit_count != len(self.sources):
            raise ValueError("hit_count must equal the number of trusted sources")
        if self.status == "ok":
            if not self.sources:
                raise ValueError("status 'ok' requires at least one source")
            if self.conflicting_citations:
                raise ValueError("status 'ok' must not carry conflicting citations")
        if self.status in ("empty", "not_used") and self.sources:
            raise ValueError(f"status {self.status!r} must not carry sources")
        if self.status == "unavailable" and self.sources:
            # 冲突时注册表整体作废:不可用状态不允许携带"看起来可用"的证据。
            raise ValueError("status 'unavailable' must not carry trusted sources")
        return self


class Citation(BaseModel):
    """响应里的一条引用 —— 字段与 S1 §2.2 冻结的 `Citation` 一致。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str
    doc_id: str
    source_id: str
    source_type: SourceType
    title: str
    url: str
    score: float = Field(allow_inf_nan=False)


class CitationValidation(BaseModel):
    """引用校验的完整结果 —— 五类信息彼此分离,不揉成一个布尔量。

    - `valid_citations`:标记 id 在可信注册表里;
    - `invalid_citations`:标记 id 从未被检索到(**编造**);
    - `unverified_citations`:标记 id 被看到过,但它的元数据自相矛盾,
      无法判定哪一份可信;
    - `malformed_markers`:长得像标记、但连标记语法都不满足的原文片段;
    - `citations` / `evidence`:可信来源的元数据。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: EvidenceStatus
    reason: str
    corpus_version: str = ""
    valid_citations: tuple[str, ...] = ()
    invalid_citations: tuple[str, ...] = ()
    unverified_citations: tuple[str, ...] = ()
    malformed_markers: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    evidence: tuple[EvidenceSource, ...] = ()
    hit_count: int = Field(default=0, ge=0)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _check_reason(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grounded(self) -> bool:
        """S1 §2.5 冻结公式:**只**看三项,不读模型自述。

        `malformed_markers` 不参与 —— 这是冻结公式的边界,不是疏忽;
        是否把它纳入判据已作为 N-4 上报,等待裁定。
        """
        return (
            self.status == "ok"
            and len(self.valid_citations) >= 1
            and len(self.invalid_citations) == 0
        )

    @model_validator(mode="after")
    def _validate_buckets(self) -> "CitationValidation":
        valid = set(self.valid_citations)
        invalid = set(self.invalid_citations)
        unverified = set(self.unverified_citations)
        if valid & invalid or valid & unverified or invalid & unverified:
            raise ValueError("citation buckets must be disjoint")
        if {citation.citation_id for citation in self.citations} != valid:
            raise ValueError("citations must describe exactly the valid ids")
        if self.hit_count != len(self.evidence):
            raise ValueError("hit_count must equal the number of evidence records")
        return self


# ---------------------------------------------------------------------------
# 信封解析
# ---------------------------------------------------------------------------


def _load_json_object(content: str) -> dict[str, Any] | None:
    """把工具消息文本解析成 JSON 对象;不是对象就返回 None。

    只接受 `dict`:JSON 数组、标量、`null` 都不是信封。解析失败同样返回
    None —— 工具消息文本不可解析,说明它不是我们认识的信封。
    """
    try:
        payload: Any = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_hits_envelope(content: str) -> HitsEnvelope | None:
    """解析成功信封。**任何**不合法都返回 None(不是"尽量解析")。"""
    payload = _load_json_object(content)
    if payload is None or payload.get("kind") != RAG_HITS_KIND:
        return None
    try:
        return HitsEnvelope.model_validate(payload)
    except ValidationError:
        return None


def looks_like_rag_hits(content: str) -> bool:
    """内容自称是检索结果(无论是否合法)。用于区分"坏信封"与"别的工具"。"""
    payload = _load_json_object(content)
    return payload is not None and payload.get("kind") == RAG_HITS_KIND


def parse_error_envelope(content: str) -> str | None:
    """解析失败信封,返回其中的受限原因码。

    只取 `reason`(短码),不取 `error` / `type` 原文:那些是给人看的文案,
    对"证据是否可用"没有增量信息,却可能携带上游想藏起来的细节。
    """
    payload = _load_json_object(content)
    if payload is None or payload.get("kind") != RAG_ERROR_KIND:
        return None
    reason = payload.get("reason")
    if isinstance(reason, str) and REASON_PATTERN.match(reason):
        return reason
    return "rag-error"


def _is_tool_message(message: object) -> bool:
    """LangChain 消息的 `type` 约定:`tool` / `ai` / `human` / `system`。

    刻意用属性判断而不是 `isinstance`:`app/rag/**` 不允许 import
    langchain(见 `tests/test_rag/test_security.py`),而这条边界是
    "检索底座不认识框架"的结构性前提。用 `type` 属性同样有效 —— 模型
    只能产出 `ai` 消息,伪造不出 `tool` 类型。
    """
    return getattr(message, "type", None) == "tool"


# ---------------------------------------------------------------------------
# 证据收集
# ---------------------------------------------------------------------------


def collect_evidence(messages: Iterable[object]) -> EvidenceCollection:
    """从**消息流**里收集可信证据。

    调用方必须传完整消息流(含 `AIMessage`),而不是自己先过滤好的
    字符串列表 —— 过滤发生在这里,"只有 tool 消息算证据"这条规则才有
    唯一的执行点。
    """
    envelopes: list[HitsEnvelope] = []
    error_reason: str | None = None
    saw_unusable = False

    for message in messages:
        if not _is_tool_message(message):
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            # 内容块列表等形态一律视为不可用证据,不尝试"猜"。
            saw_unusable = True
            continue

        envelope = parse_hits_envelope(content)
        if envelope is not None:
            envelopes.append(envelope)
            continue
        if looks_like_rag_hits(content):
            saw_unusable = True
            continue
        reason = parse_error_envelope(content)
        if reason is not None and error_reason is None:
            error_reason = reason
        # 其它工具的输出(没有 kind)与本次引用校验无关,直接忽略。

    if not envelopes:
        if error_reason is not None:
            return EvidenceCollection(status="unavailable", reason=error_reason)
        if saw_unusable:
            return EvidenceCollection(status="unavailable", reason="unusable-envelope")
        return EvidenceCollection(status="not_used", reason="no-retrieval")

    versions = {envelope.corpus_version for envelope in envelopes}
    if len(versions) > 1:
        # 索引在回答过程中被替换过 —— 没有经过验证的合并策略,只能拒绝。
        return EvidenceCollection(
            status="unavailable",
            reason="corpus-version-conflict",
            observed_ids=_observed_ids(envelopes),
            envelope_count=len(envelopes),
        )

    merged: dict[str, EvidenceSource] = {}
    conflicts: list[str] = []
    for envelope in envelopes:
        for hit in envelope.hits:
            previous = merged.get(hit.citation_id)
            if previous is None:
                merged[hit.citation_id] = hit
                continue
            if previous.provenance_key != hit.provenance_key:
                if hit.citation_id not in conflicts:
                    conflicts.append(hit.citation_id)
                continue
            if hit.score > previous.score:
                merged[hit.citation_id] = hit

    observed = tuple(sorted(merged))
    if conflicts:
        return EvidenceCollection(
            status="unavailable",
            reason="citation-metadata-conflict",
            corpus_version=next(iter(versions)),
            observed_ids=observed,
            conflicting_citations=tuple(sorted(conflicts)),
            envelope_count=len(envelopes),
        )

    sources = tuple(sorted(merged.values(), key=lambda source: (-source.score, source.citation_id)))
    return EvidenceCollection(
        status="ok" if sources else "empty",
        reason="ok" if sources else "no-hits",
        corpus_version=next(iter(versions)),
        sources=sources,
        observed_ids=observed,
        envelope_count=len(envelopes),
        hit_count=len(sources),
    )


def _observed_ids(envelopes: Sequence[HitsEnvelope]) -> tuple[str, ...]:
    seen: set[str] = set()
    for envelope in envelopes:
        seen.update(hit.citation_id for hit in envelope.hits)
    return tuple(sorted(seen))


# ---------------------------------------------------------------------------
# 标记解析与校验
# ---------------------------------------------------------------------------


def parse_citation_markers(answer: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """解析回答文本里的标记。

    返回 `(ids, malformed)`:前者按**出现顺序去重**,后者是畸形标记的
    原始片段(截断、限量、保序)。

    只用宽松模式扫一遍,再逐个判定是否满足冻结语法 —— 这样"看起来像
    标记但不是"的东西不会静默消失。如果直接用严格正则扫,`[src:]` 会
    变成空气,而它其实是模型**试图**引用却写错了,属于必须暴露的信号。
    """
    ids: list[str] = []
    malformed: list[str] = []
    for match in MARKER_LIKE_PATTERN.finditer(answer):
        span = match.group(0)
        strict = CITATION_MARKER_PATTERN.fullmatch(span)
        if strict is not None:
            citation_id = strict.group(1)
            if citation_id not in ids:
                ids.append(citation_id)
            continue
        if len(malformed) < MAX_REPORTED_MARKERS:
            malformed.append(span[:MAX_REPORTED_MARKER_CHARS])
    return tuple(ids), tuple(malformed)


def validate_citations(answer: str, evidence: EvidenceCollection) -> CitationValidation:
    """把标记分类到 valid / invalid / unverified,并附上来源元数据。"""
    marked, malformed = parse_citation_markers(answer)

    observed = set(evidence.observed_ids)
    trusted = {source.citation_id: source for source in evidence.sources}
    conflicting = set(evidence.conflicting_citations)

    valid: list[str] = []
    invalid: list[str] = []
    unverified: list[str] = []
    for citation_id in marked:
        if citation_id in trusted:
            valid.append(citation_id)
        elif citation_id in conflicting:
            unverified.append(citation_id)
        elif citation_id in observed:
            # 被看到过但注册表不可用(例如语料版本冲突),不能算有效引用。
            unverified.append(citation_id)
        else:
            invalid.append(citation_id)

    citations = tuple(
        sorted(
            (_to_citation(trusted[citation_id]) for citation_id in valid),
            key=lambda citation: (-citation.score, citation.citation_id),
        )
    )

    return CitationValidation(
        status=evidence.status,
        reason=evidence.reason,
        corpus_version=evidence.corpus_version,
        valid_citations=tuple(valid),
        invalid_citations=tuple(invalid),
        unverified_citations=tuple(unverified),
        malformed_markers=malformed,
        citations=citations,
        evidence=evidence.sources,
        hit_count=evidence.hit_count,
    )


def _to_citation(source: EvidenceSource) -> Citation:
    return Citation(
        citation_id=source.citation_id,
        doc_id=source.doc_id,
        source_id=source.source_id,
        source_type=source.source_type,
        title=source.title,
        url=source.url,
        score=source.score,
    )


def assess_answer_citations(answer: str, messages: Iterable[object]) -> CitationValidation:
    """一次完成"收集证据 → 校验引用"的便捷入口。

    S2-C 的 `/ask` 只需要这一条:把最终回答与完整消息流交进来。把两步
    合成一个函数,是为了不给调用方"只做后一步"的机会 —— 那正是"用一份
    手搓的注册表去校验引用"这类错误的入口。
    """
    return validate_citations(answer, collect_evidence(messages))


__all__ = [
    "CITATION_MARKER_PATTERN",
    "CORPUS_VERSION_PATTERN",
    "MAX_REPORTED_MARKERS",
    "MAX_REPORTED_MARKER_CHARS",
    "MARKER_LIKE_PATTERN",
    "RAG_ERROR_KIND",
    "RAG_HITS_KIND",
    "REASON_PATTERN",
    "Citation",
    "CitationValidation",
    "EvidenceCollection",
    "EvidenceSource",
    "EvidenceStatus",
    "HitsEnvelope",
    "assess_answer_citations",
    "collect_evidence",
    "looks_like_rag_hits",
    "parse_citation_markers",
    "parse_error_envelope",
    "parse_hits_envelope",
    "validate_citations",
]
