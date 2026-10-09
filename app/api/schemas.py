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
