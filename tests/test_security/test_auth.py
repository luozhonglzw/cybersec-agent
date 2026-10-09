"""Phase v0.3.0-A3-1:认证原语与"配置 fail-closed"的离线测试。

覆盖冻结验收契约里的 **N-10**,以及 **N-8** 中"确实用了常量时间原语"
那一半(A5 才补齐 PG 侧的等价性):

1. `Principal` 的严格校验(身份非空、角色封闭集、不可变);
2. `AuthKeyEntry` 的条目级校验(摘要 64 位**小写**十六进制、角色封闭集);
3. `AuthKeyring` 的集合级规则(至少一条、subject / 摘要不重复)与认证行为;
4. 比较原语:`hmac.compare_digest`,且**全表扫描、不提前退出**;
5. 配置 fail-closed:`Settings` 在 `AUTH_API_KEYS` 缺失 / 空串 / 空数组 /
   坏 JSON / 角色非法 / 摘要非法 / 重复时抛 `ValueError`,且错误文本
   **不含**任何密钥材料;
6. 结构性护栏:模块内没有硬编码摘要,也没有任何"关闭认证"的旁路开关。

全部离线、hermetic:不读仓库 `data/`,不发网络请求,不调 provider。
"""
import ast
import hashlib
import hmac
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.security import auth as auth_module
from app.security.auth import AuthKeyEntry, AuthKeyring, Principal

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
AUTH_SOURCE_PATH = APP_ROOT / "security" / "auth.py"

#: 测试专用原始密钥。**不是**真实凭据,不得复用,也不得指向任何真实系统。
RAW_ALICE = "alice-test-key-not-a-real-credential"
RAW_BOB = "bob-test-key-not-a-real-credential"

#: 64 位小写十六进制 —— 摘要在配置里的唯一合法形态。
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _entry(raw: str, subject: str, role: str) -> dict:
    return {"sha256": _digest(raw), "subject": subject, "role": role}


def _settings(**overrides) -> Settings:
    """构造 Settings,不读 .env(避免本机 .env 影响断言)。"""
    base = {"llm_model": "m", "llm_api_key": "k", "_env_file": None}
    base.update(overrides)
    return Settings(**base)


# =====================================================================
# A. Principal:身份与授权是两个字段,且严格校验
# =====================================================================


def test_principal_keeps_identity_and_role_separate():
    """`subject` 是身份,`role` 是授权 —— 合并就无法表达"两人共用角色"。"""
    principal = Principal(subject="alice", role="analyst")
    assert principal.subject == "alice"
    assert principal.role == "analyst"


def test_principal_is_frozen():
    """`Principal` 由解析器产出,下游不得篡改。"""
    principal = Principal(subject="alice", role="analyst")
    with pytest.raises(Exception):
        principal.role = "approver"  # type: ignore[misc]


@pytest.mark.parametrize("subject", ["", "   ", "\t", None, 7])
def test_principal_rejects_a_missing_or_blank_subject(subject):
    """身份必须非空 —— 空身份会让"归属"变成一个没有主体的字段。"""
    with pytest.raises(ValueError):
        Principal(subject=subject, role="analyst")


@pytest.mark.parametrize("subject", [" alice", "alice ", "\talice\n"])
def test_principal_rejects_surrounding_whitespace(subject):
    """前后空白一律拒绝:否则 "alice" 与 "alice " 会成为两个不同的身份。"""
    with pytest.raises(ValueError):
        Principal(subject=subject, role="analyst")


@pytest.mark.parametrize("role", ["admin", "root", "", "VIEWER", "superuser"])
def test_principal_rejects_a_role_outside_the_closed_set(role):
    """角色是封闭集;集合外的取值绝不静默降级。"""
    with pytest.raises(ValueError):
        Principal(subject="alice", role=role)


# =====================================================================
# B. AuthKeyEntry:配置里只有摘要,且形态严格
# =====================================================================


def test_entry_accepts_a_wellformed_digest():
    entry = AuthKeyEntry(sha256=_digest(RAW_ALICE), subject="alice", role="viewer")
    assert entry.subject == "alice"
    assert entry.role == "viewer"
    assert _HEX64.match(entry.digest)


@pytest.mark.parametrize(
    "bad_digest",
    [
        "",
        "abc",
        "a" * 63,
        "a" * 65,
        "A" * 64,          # 大写:同一把密钥配两遍会因此悄悄通过,所以拒绝
        "g" * 64,          # 非十六进制
        "a" * 63 + "-",
        " " + "a" * 64,
    ],
)
def test_entry_rejects_a_malformed_digest(bad_digest):
    with pytest.raises(ValidationError):
        AuthKeyEntry(sha256=bad_digest, subject="alice", role="viewer")


@pytest.mark.parametrize("role", ["admin", "root", "", "VIEWER"])
def test_entry_rejects_a_role_outside_the_closed_set(role):
    with pytest.raises(ValidationError):
        AuthKeyEntry(sha256=_digest(RAW_ALICE), subject="alice", role=role)


def test_entry_rejects_an_empty_subject():
    with pytest.raises(ValidationError):
        AuthKeyEntry(sha256=_digest(RAW_ALICE), subject="", role="viewer")


def test_entry_rejects_a_missing_field():
    """缺字段 → 报错里带**下标路径**,便于运维定位是第几条配错了。"""
    with pytest.raises(ValidationError) as info:
        Settings(
            llm_model="m",
            llm_api_key="k",
            _env_file=None,
            auth_api_keys=[{"subject": "alice", "role": "viewer"}],
        )
    assert "auth_api_keys.0.sha256" in str(info.value)


def test_entry_never_renders_the_digest():
    """摘要不是可用凭据,但也不该被顺手写进日志 —— 用 SecretStr 在类型层挡住。"""
    digest = _digest(RAW_ALICE)
    entry = AuthKeyEntry(sha256=digest, subject="alice", role="viewer")
    for rendered in (repr(entry), str(entry)):
        assert digest not in rendered
        assert "**********" in rendered
    assert digest not in json.dumps(entry.model_dump(mode="json"))


# =====================================================================
# C. AuthKeyring:常量时间、全表扫描、拒绝重复
# =====================================================================


def _keyring(*specs: tuple[str, str, str]) -> AuthKeyring:
    return AuthKeyring.from_entries(
        [AuthKeyEntry(sha256=_digest(raw), subject=subject, role=role) for raw, subject, role in specs]
    )


def test_keyring_authenticates_a_known_key():
    keyring = _keyring((RAW_ALICE, "alice", "analyst"), (RAW_BOB, "bob", "approver"))
    assert keyring.authenticate(RAW_ALICE) == Principal(subject="alice", role="analyst")
    assert keyring.authenticate(RAW_BOB) == Principal(subject="bob", role="approver")
    assert len(keyring) == 2


@pytest.mark.parametrize("raw", [None, "", "   ", "unknown-key", RAW_ALICE + "x", RAW_ALICE[:-1]])
def test_keyring_returns_none_for_anything_it_cannot_resolve(raw):
    """解析不出来一律 `None` —— 调用方对"没给 / 给歪了 / 不认识"走同一条 401。"""
    keyring = _keyring((RAW_ALICE, "alice", "analyst"))
    assert keyring.authenticate(raw) is None


def test_keyring_rejects_a_duplicate_subject():
    """同一 subject 出现两次是配置错误,必须启动期响亮失败。"""
    with pytest.raises(ValueError, match="duplicate subject"):
        _keyring((RAW_ALICE, "alice", "analyst"), (RAW_BOB, "alice", "approver"))


def test_keyring_rejects_a_duplicate_digest():
    """同一把密钥配两遍:后者会静默覆盖前者,必须拒绝。"""
    with pytest.raises(ValueError, match="duplicate sha256"):
        _keyring((RAW_ALICE, "alice", "analyst"), (RAW_ALICE, "bob", "approver"))


def test_keyring_rejects_an_empty_entry_list():
    with pytest.raises(ValueError, match="at least one entry"):
        AuthKeyring.from_entries([])


def test_keyring_repr_does_not_leak_digests():
    keyring = _keyring((RAW_ALICE, "alice", "analyst"))
    assert _digest(RAW_ALICE) not in repr(keyring)
    assert RAW_ALICE not in repr(keyring)


def test_keyring_compares_in_constant_time_and_scans_the_whole_list(monkeypatch):
    """**全表扫描、不提前退出** —— 否则响应时间会暴露"命中的是第几条"。

    做法:替换 `hmac.compare_digest` 为记录调用的包装(仍委托真实现),
    先直接调用一次证明 spy 是活的,再清空记录跑被测路径。
    """
    calls: list[tuple[str, str]] = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth_module.hmac, "compare_digest", spy)
    assert spy("x", "x") is True, "spy 必须是活的,否则下面的断言毫无意义"
    calls.clear()

    keyring = _keyring((RAW_ALICE, "alice", "analyst"), (RAW_BOB, "bob", "approver"))

    # 第 1 条就命中 —— 但仍然必须把两条都比完。
    assert keyring.authenticate(RAW_ALICE) is not None
    assert len(calls) == 2, "命中第 1 条时也须比完全表(不得提前退出)"

    calls.clear()
    assert keyring.authenticate("definitely-not-configured") is None
    assert len(calls) == 2

    calls.clear()
    assert keyring.authenticate(None) is None
    assert calls == [], "没有密钥时不应进入比较"
    assert keyring.authenticate("") is None
    assert calls == [], "空密钥同样不应进入比较"


# =====================================================================
# D. 配置 fail-closed(N-10)
# =====================================================================


def test_valid_configuration_is_accepted(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEYS", json.dumps([_entry(RAW_ALICE, "alice", "approver")]))
    settings = Settings(llm_model="m", llm_api_key="k", _env_file=None)
    assert len(settings.auth_api_keys) == 1
    assert settings.auth_api_keys[0].subject == "alice"


def test_configuration_is_required(monkeypatch):
    """没配 = 不启动。把"没配"当成"关掉认证"会让"API 已认证"成为未核实的假设。"""
    monkeypatch.delenv("AUTH_API_KEYS", raising=False)
    with pytest.raises(ValueError):
        _settings()


def test_blank_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEYS", "")
    with pytest.raises(ValueError):
        Settings(llm_model="m", llm_api_key="k", _env_file=None)


def test_empty_key_list_is_rejected():
    with pytest.raises(ValueError):
        _settings(auth_api_keys=[])


def test_malformed_json_is_rejected_without_echoing_the_value(monkeypatch):
    """坏 JSON → 启动失败,且错误文本里**没有**原始配置内容。"""
    marker = "SECRET-MARKER-DO-NOT-ECHO"
    monkeypatch.setenv("AUTH_API_KEYS", '{"sha256": "' + marker + '"')
    with pytest.raises(ValueError) as info:
        Settings(llm_model="m", llm_api_key="k", _env_file=None)
    assert marker not in str(info.value)


@pytest.mark.parametrize(
    "entries",
    [
        [{"sha256": "a" * 64, "subject": "alice", "role": "admin"}],          # 角色越界
        [{"sha256": "abc", "subject": "alice", "role": "viewer"}],            # 摘要形态
        [{"sha256": "a" * 64, "subject": "", "role": "viewer"}],              # 空身份
        [{"sha256": "a" * 64, "subject": "alice"}],                           # 缺角色
        [_entry(RAW_ALICE, "alice", "viewer"), _entry(RAW_BOB, "alice", "approver")],
        [_entry(RAW_ALICE, "alice", "viewer"), _entry(RAW_ALICE, "bob", "approver")],
    ],
)
def test_invalid_configuration_is_rejected_at_construction(entries):
    """全部在**构造期**失败 —— 而不是运行到一半才发现认证配置是错的。"""
    with pytest.raises(ValueError):
        _settings(auth_api_keys=entries)


def test_validation_error_never_echoes_key_material():
    """校验失败文本会进日志,因此不得出现摘要原文。"""
    digest = _digest(RAW_ALICE)
    with pytest.raises(ValueError) as info:
        _settings(auth_api_keys=[{"sha256": digest, "subject": "alice", "role": "nope"}])
    assert digest not in str(info.value)


def test_settings_repr_hides_every_digest():
    settings = _settings(auth_api_keys=[_entry(RAW_ALICE, "alice", "approver")])
    for rendered in (repr(settings), str(settings)):
        assert _digest(RAW_ALICE) not in rendered
        assert RAW_ALICE not in rendered


# =====================================================================
# E. 结构性护栏:没有硬编码摘要,也没有旁路开关
# =====================================================================

#: 任何形式的"把认证关掉"的开关名。认证**始终开启**是冻结设计的一部分:
#: 一个能关掉认证的布尔量,配错一次就是开放 API,且在日志里与"有意为之"
#: 无从区分。
#:
#: 用**词边界**匹配而不是裸子串:裸子串会把
#: `NO_AUTHORITATIVE_PRICING_PROVENANCE` 里的 `no_auth` 判成命中(实测
#: 误报),而 `\b` 要求两侧是非单词字符,正好把这种粘连排除掉。
_BYPASS_PATTERN = re.compile(
    r"\b(?:auth[_-]?enabled"
    r"|disable[_-]?auth"
    r"|skip[_-]?auth(?:entication)?"
    r"|no[_-]?auth"
    r"|anonymous[_-]?ok)\b",
    re.IGNORECASE,
)


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    """找出所有 docstring 常量节点的 id(它们不算"代码里的标识符")。"""
    ids: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, holders):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _code_tokens(tree: ast.Module) -> set[str]:
    """收集**可执行代码**里的标识符与字符串字面量(刻意跳过 docstring)。

    只扫子串会误报:本模块的文档里**故意**提到"没有把认证关掉的开关"
    这句话本身包含那个开关名。跳过 docstring 之后,判据就变成
    "代码里有没有真的读/判断这个开关"。
    """
    skip = _docstring_node_ids(tree)
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                tokens.add(node.value)
        elif isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            tokens.add(node.arg)
    return tokens


def test_no_module_implements_a_bypass_switch():
    """全 `app/` 的可执行代码里不得出现任何"关闭认证"的开关名。"""
    offenders: dict[str, list[str]] = {}
    for path in sorted(APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = sorted(
            token for token in _code_tokens(tree) if _BYPASS_PATTERN.search(token)
        )
        if hits:
            offenders[path.relative_to(APP_ROOT).as_posix()] = hits
    assert offenders == {}, offenders


def test_the_bypass_switch_scan_has_teeth():
    """正对照:把开关**真的写进代码**时,上面那条必须变红。

    没有这条,`_code_tokens` 万一收不到任何 token,上面那条会永远为绿 ——
    一个恒真的护栏不是护栏。
    """
    planted = ast.parse(
        "import os\n"
        "def _guard():\n"
        "    return os.environ.get('AUTH_ENABLED', 'true') == 'false'\n"
    )
    assert [t for t in _code_tokens(planted) if _BYPASS_PATTERN.search(t)] == [
        "AUTH_ENABLED"
    ]


def test_the_bypass_switch_scan_ignores_docstring_mentions():
    """负对照:文档里**提到**这个开关名不算实现它。

    本模块的 docstring 就故意提到了它;判据必须只看可执行代码。
    """
    documented = ast.parse(
        '"""没有 AUTH_ENABLED=false 这种开关。"""\n'
        "VALUE = 1\n"
    )
    assert [t for t in _code_tokens(documented) if _BYPASS_PATTERN.search(t)] == []


def test_auth_module_hardcodes_no_digest():
    """模块里不得出现任何硬编码的 64 位十六进制摘要(即不得内置密钥)。"""
    tree = ast.parse(AUTH_SOURCE_PATH.read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert [value for value in literals if _HEX64.match(value)] == []


def test_authenticate_uses_the_constant_time_primitive_structurally():
    """结构性事实:`authenticate` 用 `hmac.compare_digest`,不用 `==`。

    行为测试(上面那条)证明"现在用了";这条证明"实现形态就是它" ——
    有人把 `compare_digest` 换回 `==` 时,行为测试仍可能过(两条都返回
    正确布尔值),但这条会立刻变红。
    """
    tree = ast.parse(AUTH_SOURCE_PATH.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "authenticate"
    )
    attributes = {
        node.func.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "compare_digest" in attributes
    assert not [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Eq) for op in node.ops)
    ], "摘要比较不得使用 `==`(非常量时间)"
