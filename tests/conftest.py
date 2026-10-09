"""根 conftest:给**既有**测试套件配好认证,不改动它们。

## 为什么需要它

Phase v0.3.0-A3-1 给 4 条业务端点加了认证。在此之前,既有测试
(约 102 处 `create_app` 调用点 / 37 处 `TestClient` 构造点)从不带凭据。
冻结设计(A2-FINAL §5.4)明确要求"既有套件**不改动**地继续通过",机制是
FastAPI 的 `app.dependency_overrides` —— 这也是"认证必须是**依赖**而不是
中间件"这条冻结级约束的原因。

## 这里做三件事,都**不**削弱认证

1. **配一份真实的测试凭据**(`AUTH_API_KEYS`),让 `Settings` / `lifespan`
   能正常构造。这是"给测试发凭据",不是"关掉认证" —— 认证代码照常运行,
   凭据是真实配置。
2. **装默认主体的 `dependency_overrides`**,让既有用例不必逐个补
   `Authorization` 头。
3. **把测试密钥环挂到 `app.state.auth_keyring`**,使需要"配置里有哪些
   approver"的端点(A3-3 的 `/triage` 指派校验)在注入式装配下也能工作。

## 多身份(Phase v0.3.0-A3-3)

A3-3 引入了**逐线程审批指派**与**禁止自审批**(D-7)。这两条让
"一个主体既发起又审批"不再合法,于是"先 /triage 再 /resume"的端到端用例
**必须**用两个不同身份。因此这里定义了 4 个测试主体(owner / approver /
analyst / viewer),并提供对应的 `*_HEADERS`。

默认主体仍是**单个** owner(`approver` 角色,三条准入都通过)—— 不带
`Authorization` 的既有用例行为不变。**带上** `Authorization` 时,解析
委托给**真实的** `require_principal`(真密钥环、真 `hmac.compare_digest`、
真 401),因此"换个身份"走的是与生产完全相同的代码。

## 边界(必须说清楚)

第 2 项只替换**身份解析**那一个缝(`require_principal`)。**角色准入**
(`require_analyst` / `require_approver`)与**对象级授权**(A3-3 的
`_authorize_thread_read` / `_authorize_audit_scope`)仍然是真实代码、照常
执行 —— 这里**没有**任何 `AUTH_ENABLED=false` 之类的开关,也**没有**把守卫
从路由上摘掉,更**没有**把任何授权判定短路成"通过"。

真实的 401 / 403 / 404 行为由 `tests/test_api/test_auth_endpoints.py` 与
`tests/test_api/test_object_authorization.py` 覆盖:它们在构造完客户端后
`client.app.dependency_overrides.clear()`,把默认主体摘掉,走完整认证路径。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.security.auth import AuthKeyEntry, AuthKeyring, Principal, require_principal

#: 测试专用凭据。名字就是它的用途:**不是**真实凭据,不得复用,
#: 也不得指向任何真实系统。它只是让"认证配置有效"这件事在测试里成立。
TEST_OWNER_KEY = "test-owner-key-not-a-real-credential"
TEST_APPROVER_KEY = "test-approver-key-not-a-real-credential"
TEST_ANALYST_KEY = "test-analyst-key-not-a-real-credential"
TEST_VIEWER_KEY = "test-viewer-key-not-a-real-credential"

TEST_OWNER_SUBJECT = "test-owner"
TEST_APPROVER_SUBJECT = "test-approver"
TEST_ANALYST_SUBJECT = "test-analyst"
TEST_VIEWER_SUBJECT = "test-viewer"

#: 默认主体的角色取三者中最高的 `approver`,使三条准入守卫都能通过。
#: 它是"发起人"(owner)—— 因此**不能**同时被指派审批自己的线程(D-7)。
TEST_PRINCIPAL = Principal(subject=TEST_OWNER_SUBJECT, role="approver")
TEST_APPROVER_PRINCIPAL = Principal(subject=TEST_APPROVER_SUBJECT, role="approver")
TEST_ANALYST_PRINCIPAL = Principal(subject=TEST_ANALYST_SUBJECT, role="analyst")
TEST_VIEWER_PRINCIPAL = Principal(subject=TEST_VIEWER_SUBJECT, role="viewer")


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


#: 测试密钥环。与 `_test_auth_api_keys()` 编译出的 `AUTH_API_KEYS` **同源**,
#: 保证"配置里能解析出来的主体"与"override 能解析出来的主体"永远一致。
TEST_KEYRING = AuthKeyring.from_entries(
    [
        AuthKeyEntry(sha256=_digest(TEST_OWNER_KEY), subject=TEST_OWNER_SUBJECT, role="approver"),
        AuthKeyEntry(sha256=_digest(TEST_APPROVER_KEY), subject=TEST_APPROVER_SUBJECT, role="approver"),
        AuthKeyEntry(sha256=_digest(TEST_ANALYST_KEY), subject=TEST_ANALYST_SUBJECT, role="analyst"),
        AuthKeyEntry(sha256=_digest(TEST_VIEWER_KEY), subject=TEST_VIEWER_SUBJECT, role="viewer"),
    ]
)


def _auth(raw: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {raw}"}


#: 每个身份的请求头。端到端用例靠它在 /triage 与 /resume 之间切换身份。
OWNER_HEADERS = _auth(TEST_OWNER_KEY)
APPROVER_HEADERS = _auth(TEST_APPROVER_KEY)
ANALYST_HEADERS = _auth(TEST_ANALYST_KEY)
VIEWER_HEADERS = _auth(TEST_VIEWER_KEY)


def _test_auth_api_keys() -> str:
    """把测试凭据编译成 `AUTH_API_KEYS` 期望的 JSON(只含摘要)。"""
    return json.dumps(
        [
            {
                "sha256": _digest(TEST_OWNER_KEY),
                "subject": TEST_OWNER_SUBJECT,
                "role": "approver",
            },
            {
                "sha256": _digest(TEST_APPROVER_KEY),
                "subject": TEST_APPROVER_SUBJECT,
                "role": "approver",
            },
            {
                "sha256": _digest(TEST_ANALYST_KEY),
                "subject": TEST_ANALYST_SUBJECT,
                "role": "analyst",
            },
            {
                "sha256": _digest(TEST_VIEWER_KEY),
                "subject": TEST_VIEWER_SUBJECT,
                "role": "viewer",
            },
        ]
    )


async def _resolve_test_principal(request: Request) -> Principal:
    """默认身份解析缝:带 `Authorization` 走**真实**认证,不带则用默认主体。

    带凭据时委托给 `require_principal` 本身 —— 于是密钥环查找、常量时间
    比较、401 语义全部与生产一致;测试拿到的不是"另一个 mock",而是同一段
    代码。不带凭据时返回默认 owner,让 100+ 处既有用例不必逐个补头。

    这个分支**只**决定"你是谁";角色准入与对象级授权完全不在这里。
    """
    if request.headers.get("authorization"):
        return await require_principal(request)
    return TEST_PRINCIPAL


@pytest.fixture(autouse=True)
def auth_api_keys_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """为每个用例提供有效的 `AUTH_API_KEYS`(函数级,不污染其它用例)。

    `get_settings` 是 `lru_cache` 的,因此前后都要清缓存 —— 否则前一个
    用例缓存的 Settings 会渗进来,让"配置决定行为"的断言变成假的。
    """
    monkeypatch.setenv("AUTH_API_KEYS", _test_auth_api_keys())
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def default_test_principal(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """给测试里构造的每个应用装一个默认主体与一个测试密钥环。

    挂在 `TestClient.__init__` 而不是 `create_app` 上,是因为 6 个测试模块
    都用 `from app.api.main import create_app` 在**导入期**绑定了名字 ——
    事后 patch `app.api.main.create_app` 对它们不可见(导入期绑定坑)。
    `TestClient` 是所有这些用例的唯一共同入口。

    `setdefault` 而不是直接赋值:用例若自己装了 override(例如负向用例
    把 `require_principal` 换成"必定失败"),这里不得覆盖它。

    密钥环用 `hasattr` 守卫:用例若自己设了 `app.state.auth_keyring`
    (例如 `test_auth_endpoints` 用它自己的三主体密钥环),不得被这里覆盖。
    """
    original_init = TestClient.__init__

    def _init(self, app, *args, **kwargs):
        state = getattr(app, "state", None)
        if state is not None and not hasattr(state, "auth_keyring"):
            state.auth_keyring = TEST_KEYRING
        overrides = getattr(app, "dependency_overrides", None)
        if overrides is not None:
            overrides.setdefault(require_principal, _resolve_test_principal)
        original_init(self, app, *args, **kwargs)

    monkeypatch.setattr(TestClient, "__init__", _init)
    yield
