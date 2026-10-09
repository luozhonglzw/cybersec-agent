"""线程归属与审批指派的数据模型 —— Phase v0.3.0-A3-2(v0.3.0-A3-2-FIX2 修订)。

`ThreadOwnership` 是本阶段新增的**唯一**领域对象:一条线程的属主,以及
**显式指派**给它的审批人集合。它对应存储层的**单行**记录:

    thread_owners (thread_id, owner, approvers, created_at)   1 行

**为什么是单行(而不是"属主表 + 逐审批人行表")**(A3-2-FIX2,Controller
裁决 Option A):审批集合是**不可变的成员集合**,而"逐行可追加"的表在结构上
无法表达这件事 —— 只要存在一张可 INSERT 的 `thread_approvers`,
"注册之后再补一个审批人"就总是可能的,而任何基于 `EXISTS(owner)` 的触发器
守卫都同时面临两个无法回避的漏洞:对**尚未注册**的线程守卫为假(孤儿审批行
可写),以及在 READ COMMITTED 下看不见他人**未提交**的属主行(并发追加可写)。
把集合收进**同一行的一个列**,这三件事一次性由**结构**排除:

- 不存在可追加的独立行 ⇒ 不存在"事后补一个审批人"的写入面;
- 不存在"有审批行、无属主行"的形态 ⇒ 不存在孤儿;
- 一条 INSERT 即一个原子事实 ⇒ 不存在部分写入;
- 主键唯一性 ⇒ 并发注册只有一个胜者,集合**不会**被合并。

**为什么"显式"指派是重点**:本对象存在的全部意义,是让"谁能审批这条线程"
成为一个**被记录下来的事实**,而不是从角色推导出来的默认值。冻结设计明确
禁止任何全局回退 —— 没有 `thread_owners` 行 ⇒ **没有人**可以审批
(既不是"任何持有 approver 角色的人",也不是属主)。因此 `approvers`
**不能**用"所有 approver 角色的人"来填充,它必须来自逐线程的显式指派。

刻意不做的事(边界,勿越):

1. **不做任何授权判定。** 本模型只回答"这条线程的属主是谁、指派了谁",
   不回答"谁**有资格**被指派",更不回答"谁**可以**审批"。授权决策留在
   持久化模型之外(A5 的服务层),这里只校验**形态**。
2. **不 import `app.security`。** 分层的方向是 `app.security/**` →
   `app.schemas/**`(`store_protocol.py` / `audit.py` 都按这个方向导入),
   反向依赖会制造循环。因此本模块**不能**复用
   `app.security.auth.Principal` 的校验,而是自带一份等价的规范主体规则
   (`_require_canonical_subject`)—— 两处规则必须同步,由
   `tests/test_schemas/test_ownership.py` 的等价性测试钉住。
3. **不把调用方给的字符串当成已认证主体。** 这里的 `owner` / `approvers`
   是**声明值**:服务层会先确认它们确实存在于配置,但本模型自身**无法**
   证明这一点 —— 它只能校验形态。任何读到 `owner` 的地方都不应把它当作
   "已被认证的身份"。把审计 `actor` 换成已验证主体是 A6 的事。
"""
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


def _require_canonical_subject(value: str, *, field: str) -> str:
    """规范主体标识:非空、且不带前后空白。

    规则与 `app.security.auth.Principal.__post_init__` **逐条一致**。
    同一个字符串不能在认证侧合法、在归属侧非法(或反之),否则
    "配置里的 subject" 与 "被指派/被记录的 subject" 会变成两套口径。
    分层约束(见模块 docstring)不允许直接复用那边的函数,因此这里显式
    重复一次,并由等价性测试钉住。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty subject identifier")
    if value != value.strip():
        raise ValueError(f"{field} must not carry surrounding whitespace")
    return value


class ThreadOwnership(BaseModel):
    """一条线程的属主 + 显式审批指派(**不可变**聚合)。

    `frozen=True`:这个对象描述的是**已经发生**的指派事实。存储层是
    append-only 的,内存里的表示也必须是 —— 允许就地改 `approvers`
    等于让"改派"在内存里悄悄成立,而 v0.3.0 **不支持**改派(L-2)。

    `extra="forbid"`:拼错的字段名(例如 `approver=` 少一个 s)必须响亮
    失败,而不是被静默忽略后留下一个"以为指派了、其实没有"的对象 ——
    那正好会退化成 fail-open。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    thread_id: str = Field(min_length=1, description="图执行线程 id(服务端生成)")
    owner: str = Field(
        min_length=1, description="属主 subject(声明值,来自配置;本模型只校验形态)"
    )
    approvers: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "显式指派的审批人 subject(非空;语义为集合,顺序不承载意义;"
            "存储层以规范序 JSON 数组落在 thread_owners 同一行内)"
        ),
    )
    created_at: datetime = Field(description="指派建立时间(tz-aware UTC)")

    @field_validator("thread_id", "owner")
    @classmethod
    def _validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        return _require_canonical_subject(value, field=info.field_name or "subject")

    @field_validator("approvers")
    @classmethod
    def _validate_approvers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """逐个校验形态 + 拒绝重复,然后归一化为**字典序**。

        归一化的理由:`approvers` 语义上是**集合**(存储层把它序列化成一个
        规范序的 JSON 数组,不承载序),因此 `("a", "b")` 与 `("b", "a")` 是
        **同一个**指派。不归一化会有两个后果:

        - 同一指派有两种序列化结果,"是否同一指派"的比对与摘要失去意义;
        - `get_thread_ownership()` 读回的顺序会取决于存储里那一串字符的
          书写顺序,于是"写进去再读出来是否相等"这个最基本的往返性质
          就不成立。

        归一化同时是**规范序列化**的前提:存储层直接 `json.dumps(list(...))`,
        依赖这里已经把顺序钉成字典序。因此"先查重、再排序"的顺序不可颠倒 ——
        否则 `("bob","bob")` 会被排序悄悄去重成合法输入。
        """
        seen: set[str] = set()
        for index, subject in enumerate(value):
            _require_canonical_subject(subject, field=f"approvers[{index}]")
            if subject in seen:
                raise ValueError(
                    f"duplicate approver subject at index {index}: {subject!r}"
                )
            seen.add(subject)
        return tuple(sorted(seen))

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        """拒绝 naive datetime —— 与存储层 `_to_iso` 同一口径。

        混入本地时区会破坏"字典序 == 时间序",而审计流完全依赖顺序。
        在模型层就拒绝,比等到写库时才炸更早、更清楚。
        """
        if value.tzinfo is None:
            raise ValueError(
                "created_at must be timezone-aware (naive datetime breaks ordering)"
            )
        return value

    @model_validator(mode="after")
    def _validate_assignment(self) -> "ThreadOwnership":
        """禁止自审批(D-7):属主不得出现在自己的审批人集合里。

        与 `TriageOutcome._validate_pending` 同一纪律 —— 让一个畸形的归属
        对象**根本构造不出来**,而不是指望每个调用方都记得检查。

        这是 D-7 的**第一道**检查(在指派时);第二道在 `/resume` 的服务层
        (在审批时)。两道都要有:真正保护动作的是第二道,因为一条畸形的
        存储行不该有可利用的机会。
        """
        if self.owner in self.approvers:
            raise ValueError(
                "owner must not be an assigned approver (self-approval is prohibited)"
            )
        return self
