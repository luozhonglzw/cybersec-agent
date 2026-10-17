"""API 层的 Request / Response Schema。

为什么单独一个文件:API 的数据契约(contract)与业务层解耦,
未来 SecurityAgent 返回结构化结果(风险等级、处置建议)时,
只改这里和路由,不污染 core 层。

Phase 8.4 增加了 triage / resume 的契约。两条**刻意缺席**的字段:

    TriageRequest 没有 thread_id
        thread_id 由服务端生成(D3)。框架完全不校验 thread_id,允许客户端
        指定就等于允许它复用别人暂停中的 state(实测:indicator 被换掉、
        messages 被追加、agent 重跑)—— 这是劫持向量,不是便利。

    ResumeRequest 没有 interrupt_id
        服务端从 checkpoint 恢复它(D7)。客户端能指定 interrupt_id
        就等于能伪造"审批的是哪一次暂停"。
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.approval import ApprovalStatus, TriageOutcome

# 请求 DTO 的未知字段策略(Phase 8.5):**显式**声明为 ignore。
#
# 为什么显式:ignore 本来就是 Pydantic 的默认值,不写也生效 —— 但"没写"
# 意味着这个策略**没人决定过**,下一个 DTO 作者无从判断它是有意还是疏忽。
# 写出来,策略就从隐式默认变成可评审的决定。
#
# 为什么是 ignore 而不是 forbid:服务端自己生成 thread_id(D3),客户端多发
# 一个 thread_id 不影响执行;而 forbid 会让任何多发字段的客户端收到 422 ——
# 在没有 API 版本化机制时,这个兼容成本换不来对应的安全收益。
# 收紧到 forbid 留到 Phase 10(有认证与版本化之后)。
_REQUEST_EXTRA = ConfigDict(extra="ignore")


class ChatRequest(BaseModel):
    """POST /chat 的请求体。"""

    model_config = _REQUEST_EXTRA

    message: str = Field(min_length=1, description="用户输入的自然语言消息")


class ChatResponse(BaseModel):
    """POST /chat 的响应体。"""

    response: str = Field(description="Agent 的回复文本")


class TriageRequest(BaseModel):
    """POST /triage 的请求体。

    刻意不含 thread_id(见模块 docstring)。

    **v0.3.0-A3-3 新增 `approvers`(必填)**:发起一次判定的人必须**显式**
    说清"这条线程由谁来审批"。此前审批人是从角色推导的默认值(任何持有
    approver 角色的人都能审批任何线程)—— 那让"谁能审批"变成一个**没人
    记录过**的事实。现在它是一条被写下来的指派,并且:
    非空、无重复、每个都必须是一个**已配置的 approver**、且不得包含发起人
    自己(禁止自审批,D-7)。违反任一条 → 422,**在任何图调用或归属写入之前**。

    发起人身份**不**来自本请求体:属主取自已认证主体的 subject,客户端
    无法指定(与 thread_id 同一条理由 —— 能指定就等于能伪造)。
    """

    model_config = _REQUEST_EXTRA

    indicator: str = Field(
        min_length=1, description="判定对象(IP / 域名 / Hash)"
    )
    event_type: str | None = Field(
        default=None, description="可选:限定统计的日志事件类型,如 login_failed"
    )
    approvers: list[str] = Field(
        min_length=1,
        description=(
            "显式指派的审批人 subject 列表(必填、非空、无重复;"
            "每个都必须是已配置的 approver 角色主体,且不得是发起人自己)"
        ),
    )


class ApprovalDecisionRequest(BaseModel):
    """人工审批的决定字段(不含 thread_id,由 ResumeRequest 组合)。

    ⚠️ `operator` 自 **v0.3.0-A3-3** 起**不是**权威身份:审批的 `actor`
    一律取自已认证主体的 subject(见 `app/api/main.py` 的 `resume` 端点)。
    这个字段仍然**必填**,只为保持既有线上请求的**错误兼容性**
    (缺字段仍 422),它的值**被忽略** —— 服务端不会把它写进审计、
    也不会拿它做任何判定。客户端无法通过伪造它来改变权威 actor。
    """

    model_config = _REQUEST_EXTRA

    status: ApprovalStatus = Field(description="决定:approved / denied")
    operator: str = Field(
        min_length=1,
        description=(
            "审批人标识(**非权威**:服务端忽略其值,actor 取自已认证主体;"
            "字段保留仅为线上请求的错误兼容性)"
        ),
    )
    reason: str | None = Field(default=None, description="审批意见(可选)")


class ResumeRequest(ApprovalDecisionRequest):
    """POST /resume 的请求体。

    thread_id 是**必填**的 —— 它是客户端从 /triage 响应里拿到的恢复句柄。
    服务端会用校验门确认它确实停在待审批状态(未知 → 404,已完成 /
    checkpoint 丢失 → 409),不会静默从 START 重跑一轮。
    """

    # 父类已声明 extra 策略;此处再写一次是刻意的:策略必须在每个请求 DTO
    # 上肉眼可见,不能靠"读者去追继承链"。
    model_config = _REQUEST_EXTRA

    thread_id: str = Field(min_length=1, description="triage 返回的 thread_id")


class TriageResponse(TriageOutcome):
    """/triage 与 /resume 共用的响应体。

    直接继承领域模型 TriageOutcome(扁平结构 + 复用其校验),只补一个
    传输层字段 interrupt_id —— 它是框架运行时的恢复句柄,不属于判定结果,
    所以刻意**不**放进 TriageOutcome(D7),只在这一层暴露。
    """

    interrupt_id: str | None = Field(
        default=None, description="暂停时的 interrupt id;已跑完时为 None"
    )


# =====================================================================
# POST /ask —— 带引用的知识库问答(S1 §2.2 冻结契约)
# =====================================================================

#: 问题长度上限。超限 → 422(不是静默截断):截断会悄悄改变语义,而"问题
#: 太长"是客户端可以自行修正的错误。与工具层的 `MAX_QUERY_CHARS`(500)不是
#: 同一个量:模型会用问题派生出更短的检索词,两者不应互相绑定。
MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    """POST /ask 的请求体 —— **只有** `question`。

    刻意缺席的字段(每一条都是授权边界):

    - 没有 `top_k`:检索深度由服务端决定,客户端不参与;
    - 没有 `path` / `url` / `collection` / `index`:能指定"从哪读"就等于
      能读任意文件或访问任意地址(backlog R-1 记录的既有攻击面);
    - 没有 `corpus_version`:语料版本由部署决定,客户端指定它就能让回答
      引用一个并不存在的索引版本。
    """

    model_config = _REQUEST_EXTRA

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="用户问题;服务端据此检索知识库并给出带引用的回答",
    )


class Citation(BaseModel):
    """回答里的一条有效引用(S1 冻结的六个字段)。

    `citation_id` 就是文档 id(`kb-cve-…` / `kb-attack-…`),因此**不再**单独
    回传 `doc_id` —— 重复字段只会让客户端困惑该以哪个为准。

    `score` 是**检索相似度**,不是"这条引用支持该结论"的置信度:它只描述
    向量空间里的接近程度,与事实正确性无关。
    """

    citation_id: str = Field(description="引用 id(等于文档 id),来自本次检索结果")
    source_id: str = Field(description="官方标识,如 CVE-2021-44228 / T1190")
    source_type: str = Field(description="来源类别:cve / attack-technique / advisory")
    title: str = Field(description="文档标题")
    url: str = Field(description="来源链接;仅用于展示,服务端从不请求它")
    score: float = Field(description="检索相似度(不表示事实正确性)")


class RetrievalInfo(BaseModel):
    """本次检索的结构化状态。

    `status` 的四值来自 S1 冻结矩阵,含义严格区分:

    - `ok` —— 至少一条可信命中;
    - `empty` —— 检索成功但没有相关命中;
    - `unavailable` —— 索引/语料不可用,或证据自相矛盾(fail closed);
    - `not_used` —— 本次回答没有使用检索工具。

    只有 `ok` 才可能 `grounded=true`,但 `ok` **不**意味着"问题有答案"。
    """

    status: Literal["ok", "empty", "unavailable", "not_used"] = Field(
        description="检索状态"
    )
    hit_count: int = Field(ge=0, description="本次可用的可信文档数")
    top_k: int = Field(ge=1, description="本端点配置的检索深度(客户端不可影响)")


class AskResponse(BaseModel):
    """POST /ask 的响应体。

    `grounded` 是**结构性**判定,不是"回答正确"的证明:

    - `true` 仅表示:本次检索成功、至少一条引用、且没有任何编造 / 存疑 /
      写错的标记;
    - 它**不**表示每条断言都被来源支持,也**不**表示检索到的文档与问题相关。

    证据不足、索引不可用、工具未被使用时,服务端**不**返回 5xx —— 那是
    "服务坏了"的信号;这里返回 200 + `grounded=false` + 明确的 `status`。
    """

    answer: str = Field(description="Agent 的回答文本")
    grounded: bool = Field(description="结构性 groundedness(不代表事实正确)")
    citations: list[Citation] = Field(
        default_factory=list, description="有效引用,按 score 降序、同分按 id 升序"
    )
    invalid_citations: list[str] = Field(
        default_factory=list, description="回答里出现、但本次从未检索到的引用 id"
    )
    retrieval: RetrievalInfo = Field(description="本次检索的状态")
