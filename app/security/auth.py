"""静态 API Key 认证 + 端点角色准入(Phase v0.3.0-A3-1)。

本模块**只**做两件事:

1. **认证** —— 把请求头里的 `Authorization: Bearer <key>` 解析成一个
   已认证的 `Principal(subject, role)`;失败一律 401。
2. **端点准入** —— 按角色判断"这个已认证主体能不能进这条路由";失败 403。

刻意**不做**(冻结设计把它们排在后续阶段,见 A2-FINAL §7.5 的 A4–A7):

- 审计读取的范围过滤由 **A3-3** 在 HTTP 边界实现(见 `app/api/main.py` 的
  `_authorize_audit_scope`),**不在本模块** —— 本模块只管"你是谁"与
  "这条路由准不准你进",对象级授权依赖 `AuditStore` 的归属读取,
  那不属于认证模块的依赖面。

**v0.3.0-A3-3 起,下面两条边界已经收口**(A3-1 时它们还是缺口):

- `/resume` 的**对象级授权**已在 API 边界实现:必须显式被指派到该 thread,
  属主不得自审批(见 `app/api/main.py` 的 `resume` 端点);
- 审计 `actor` 已经是**已验证主体**:`/resume` 不再把调用方自述的
  `operator` 当作actor,而是把 `principal.subject` 交给服务层
  (A6 会进一步同时记录 `verified_subject` 与 `claimed_operator`,并加签名)。

本模块**仍然只做**两件事:认证(`Authorization: Bearer` → `Principal`)
与端点角色准入。它**不**读线程归属,因此 `require_approver` 依然只是
"调用方是一个已认证的 approver",不是"他有权审批这一个 thread" ——
后者由 API 边界的对象级授权补齐。

为什么是 SHA-256 而不是 bcrypt / argon2(必须写下来,否则将来会有人
"顺手修好它"):API Key 是运维生成的高熵随机串(≥32 字节),不是人类
口令 —— 没有字典可防。慢 KDF 只会给**每一个请求**加上几十毫秒,换不来
任何安全性。这里要防的是"配置泄露后密钥还能直接用",SHA-256 已经足够。

为什么不用 JWT / OIDC:本阶段冻结的范围是"静态 API Key 映射到
Principal"。JWT 会引入签名密钥管理、过期与刷新语义、时钟依赖,以及
新的依赖 —— 全都不在本次授权内。认证是**始终开启**的:没有
`AUTH_ENABLED=false` 之类的开关,配置缺失或非法时进程拒绝启动。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request
from pydantic import AfterValidator, BaseModel, Field, SecretStr

#: 角色是**封闭集**,与项目既有的封闭枚举纪律一致
#: (`AuditBackend` / `AuditEvent` / `ApprovalStatus`)。
#: 配置里出现集合外的角色 → 启动期失败,绝不静默降级。
Role = Literal["viewer", "analyst", "approver"]

#: 运行时校验用的同集合视图(供 `Principal` 的严格校验使用)。
_ROLES: frozenset[str] = frozenset({"viewer", "analyst", "approver"})

#: 摘要形态:64 位**小写**十六进制。大写 / 长度不符一律拒绝 ——
#: 允许大小写混用会让"同一把密钥配了两遍"这种错误悄悄通过。
_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")

#: 401 / 403 的固定文案。**刻意**是常量而不是 f-string:
#: 缺失 / 畸形 / 未知凭据必须给出逐字节相同的响应体,否则响应差异就成了
#: "这把 key 存在吗"的枚举预言机。
_UNAUTHORIZED_DETAIL = "authentication required"
_FORBIDDEN_DETAIL = "insufficient role"


# =====================================================================
# 主体
# =====================================================================


@dataclass(frozen=True)
class Principal:
    """已认证主体:**身份**与**授权**是两个字段,刻意不合并。

    - `subject` 是身份(用于归属与指派);
    - `role` 是授权(用于路由准入)。

    合并它们会让"两个人共用 approver 角色"变得无法表达,而"至少两个
    独立主体"正是禁止自审批(D-7)能落地的前提。

    frozen:`Principal` 由解析器产出,下游不得篡改。

    `subject` **只**来自配置,永不来自请求头 —— `X-User:` 之类的头
    不是身份来源。
    """

    subject: str
    role: Role

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("Principal.subject must be a non-empty string")
        if self.subject != self.subject.strip():
            raise ValueError("Principal.subject must not carry surrounding whitespace")
        if self.role not in _ROLES:
            raise ValueError(
                f"Principal.role must be one of {sorted(_ROLES)}, got {self.role!r}"
            )


def _require_sha256_hex(value: SecretStr) -> SecretStr:
    """校验摘要是 64 位小写十六进制。

    用 `AfterValidator` 而不是 `Field(pattern=...)`:`SecretStr` 不是字符串
    schema,把 pattern 约束套上去会在**构造期**抛 `TypeError`
    (实测:"Unable to apply constraint 'pattern' to supplied value")。
    """
    if not _SHA256_HEX.match(value.get_secret_value()):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return value


#: 配置里的摘要类型。用 `SecretStr` 包一层是**类型层面**的防泄露:
#: 摘要本身不是可用凭据(不能拿它登录),但把 `repr` / `str` /
#: `model_dump(mode="json")` 统一变成 `**********`,可以保证它不会顺手
#: 被写进日志或异常文本 —— 与配置层既有的机密字段同一纪律。
Sha256Digest = Annotated[SecretStr, AfterValidator(_require_sha256_hex)]


class AuthKeyEntry(BaseModel):
    """配置里的一条主体记录:**只有摘要,没有原始密钥**。

    原始密钥由运维生成并只出示一次,配置里落的是它的 SHA-256。
    配置泄露因此不会直接给出可用凭据。
    """

    sha256: Sha256Digest
    subject: str = Field(min_length=1)
    role: Role

    @property
    def digest(self) -> str:
        """取回摘要明文,供常量时间比较使用(**不**用于打印)。"""
        return self.sha256.get_secret_value()


# =====================================================================
# 密钥环
# =====================================================================


class AuthKeyring:
    """一组 `(摘要, Principal)` 的不可变集合,提供常量时间认证。

    三条硬性质:

    1. **常量时间比较** —— 逐条用 `hmac.compare_digest`,不用 `==`
       (后者会在第一个不同字节处提前返回,泄漏前缀信息)。
    2. **全表扫描、不提前退出** —— 即使第 1 条就命中,也要把整张表比完。
       提前退出会让**响应时间**暴露"命中的是第几条",即暴露列表位置。
    3. **拒绝重复** —— 同一 subject 或同一摘要出现两次是配置错误,必须
       在启动期响亮失败,而不是"后面那条静默生效"。
    """

    __slots__ = ("_pairs",)

    def __init__(self, entries: Iterable[AuthKeyEntry]) -> None:
        pairs: list[tuple[str, Principal]] = []
        seen_subjects: set[str] = set()
        seen_digests: set[str] = set()
        for index, entry in enumerate(entries):
            digest = entry.digest
            if entry.subject in seen_subjects:
                raise ValueError(
                    f"duplicate subject in auth_api_keys at index {index}"
                )
            if digest in seen_digests:
                raise ValueError(
                    f"duplicate sha256 in auth_api_keys at index {index}"
                )
            seen_subjects.add(entry.subject)
            seen_digests.add(digest)
            pairs.append((digest, Principal(subject=entry.subject, role=entry.role)))
        if not pairs:
            raise ValueError("auth_api_keys must contain at least one entry")
        self._pairs: tuple[tuple[str, Principal], ...] = tuple(pairs)

    @classmethod
    def from_entries(cls, entries: Iterable[AuthKeyEntry]) -> "AuthKeyring":
        """从配置条目构建 —— 所有条目级规则(形态 / 角色 / 重复)的唯一实现处。"""
        return cls(entries)

    def authenticate(self, raw_key: str | None) -> Principal | None:
        """把原始密钥解析成主体;无法解析返回 `None`(**绝不**抛错、绝不放过)。

        返回 `None` 而不是抛异常,是为了让调用方(依赖)对
        "没给 / 给歪了 / 不认识" 三种情况走**同一条** 401 路径。
        """
        if not raw_key:
            return None
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        matched: Principal | None = None
        # 刻意**不** break:全表比完,时间不随命中位置变化。
        for stored_digest, principal in self._pairs:
            if hmac.compare_digest(stored_digest, digest):
                matched = principal
        return matched

    def subjects_with_role(self, role: str) -> frozenset[str]:
        """返回配置里**持有该角色**的全部 subject(A3-3 新增)。

        用途唯一:对象级授权必须回答"这个 subject 是不是一个**已配置的**
        approver"。这个问题只能由**配置**回答 —— 调用方自述的角色、请求体里
        的字段、任何请求头都不是答案。`/triage` 的审批指派校验用它,
        因此"指派了一个不存在或没有 approver 角色的人"会在**写任何东西之前**
        被拒绝,而不是等到审批时才失败。

        返回 `frozenset`:调用方只做成员判定,不需要顺序,也不得就地修改。
        与 `authenticate` 一样,这里**不**回显摘要,也不暴露任何凭据材料。
        """
        return frozenset(
            principal.subject for _, principal in self._pairs if principal.role == role
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        """绝不回显摘要(默认 dataclass/repr 会把它们打出来)。"""
        return f"AuthKeyring(entries={len(self._pairs)})"


# =====================================================================
# 请求头解析
# =====================================================================


def _bearer_token(request: Request) -> str | None:
    """从 `Authorization: Bearer <token>` 取出 token;不合规返回 `None`。

    只认一种方案、一个头 —— 不读 cookie、不读 query(查询串会进访问日志
    与 `Referer`),也不读任何自称身份的头(如 `X-User`)。
    """
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _unauthorized() -> HTTPException:
    """401:缺失 / 畸形 / 未知凭据共用同一条出口。"""
    return HTTPException(
        status_code=401,
        detail=_UNAUTHORIZED_DETAIL,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _forbidden() -> HTTPException:
    """403:已认证,但角色不足以进入这条路由。"""
    return HTTPException(status_code=403, detail=_FORBIDDEN_DETAIL)


# =====================================================================
# FastAPI 依赖
# =====================================================================


async def require_principal(request: Request) -> Principal:
    """认证依赖:返回已认证的 `Principal`,否则 401。

    这是**唯一**的认证缝(组合根注入的默认主体、以及新测试里的正负用例
    都通过 `app.dependency_overrides[require_principal]` 挂在这里)。

    fail-closed:密钥环取不到(例如应用不是经 `lifespan` 装配的)时也返回
    401,绝不"解析不出来就放行"。这个分支让"认证配置没接上"表现为拒绝
    服务,而不是悄悄变成一个开放 API。
    """
    keyring = getattr(request.app.state, "auth_keyring", None)
    if not isinstance(keyring, AuthKeyring):
        raise _unauthorized()
    principal = keyring.authenticate(_bearer_token(request))
    if principal is None:
        raise _unauthorized()
    return principal


#: 允许 `analyst` 与 `approver`(approver 是 analyst 的超集)。
_ANALYST_ROLES: frozenset[str] = frozenset({"analyst", "approver"})


def _admit(principal: Principal, allowed: frozenset[str]) -> Principal:
    if principal.role not in allowed:
        raise _forbidden()
    return principal


async def require_analyst(
    principal: Principal = Depends(require_principal),
) -> Principal:
    """准入:`analyst` 或 `approver`。用于 `/chat` 与 `/triage`。

    `viewer` 在这里拿 403 是**刻意**的:它只持只读审计权限,不是操作者。
    """
    return _admit(principal, _ANALYST_ROLES)


async def require_approver(
    principal: Principal = Depends(require_principal),
) -> Principal:
    """准入:仅 `approver`。用于 `/resume`。

    **这是角色准入,不是对象级授权** —— 它只回答"调用方是不是一个已认证的
    approver"。至于"他有没有被**指派**审批这一个 thread",由 `/resume`
    端点上的对象级授权补齐(Phase v0.3.0-A3-3):那里会在任何
    graph / checkpointer 访问之前读线程归属,未指派 → 404,属主自审批 → 403。
    两道闸门**都必须**有:角色准入挡"角色不对的人",对象授权挡
    "角色对但没被指派到这条线程的人"。
    """
    return _admit(principal, frozenset({"approver"}))
