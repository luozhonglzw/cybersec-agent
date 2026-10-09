"""`ThreadOwnership` 领域模型测试(Phase v0.3.0-A3-2)。

这里断言的是**模型的形态与边界**,不是存储行为(那在
`tests/test_security/test_ownership_store.py` 与
`tests/test_postgres/test_store_contract.py`)。

三件必须被钉住的事:

1. **畸形归属对象构造不出来** —— 空属主、空审批集合、重复审批人、
   属主自己被指派(自审批 D-7)、naive 时间,一律在构造期拒绝;
2. **规范表示** —— `approvers` 语义是集合,`("a","b")` 与 `("b","a")` 必须
   归一化成**同一个**对象,否则"是否同一指派"的比对失去意义;
3. **模型里没有授权** —— 它不 import `app.security`,也不做任何
   "谁能被指派 / 谁能审批"的判定。授权在服务层(A5)。

第 3 点用**结构性断言**(AST 读源码)守着,而不是靠"我看了一遍觉得没有"。
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.ownership import ThreadOwnership
from app.security.auth import Principal

REPO_ROOT = Path(__file__).resolve().parents[2]
OWNERSHIP_PATH = REPO_ROOT / "app" / "schemas" / "ownership.py"

TS = datetime(2026, 10, 8, 9, 0, 0, tzinfo=timezone.utc)


def _ownership(**overrides) -> ThreadOwnership:
    base = {
        "thread_id": "th-1",
        "owner": "alice",
        "approvers": ("bob",),
        "created_at": TS,
    }
    base.update(overrides)
    return ThreadOwnership(**base)


# ---------------------------------------------------------------------------
# 1. 正常形态
# ---------------------------------------------------------------------------


def test_valid_ownership_round_trips_through_the_model() -> None:
    o = _ownership(approvers=("bob", "carol"))
    assert o.thread_id == "th-1"
    assert o.owner == "alice"
    assert o.approvers == ("bob", "carol")
    assert o.created_at == TS


def test_ownership_is_immutable() -> None:
    """描述"已经发生的指派"的对象不得被就地修改 —— 改派在 v0.3.0 不存在。"""
    o = _ownership()
    with pytest.raises(ValidationError):
        o.owner = "mallory"  # type: ignore[misc]


def test_ownership_is_hashable_and_frozen() -> None:
    assert hash(_ownership()) == hash(_ownership())
    assert _ownership() == _ownership()


def test_unknown_field_is_rejected_loudly() -> None:
    """拼错的字段名(例如 `approver` 少一个 s)必须响亮失败。

    静默忽略它会留下一个"以为指派了、其实没有"的对象 —— 那正好退化成
    fail-open(没有任何审批人却被当作已指派)。
    """
    with pytest.raises(ValidationError):
        ThreadOwnership(
            thread_id="th-1",
            owner="alice",
            approver=("bob",),  # type: ignore[call-arg]
            created_at=TS,
        )


def test_a_list_input_is_coerced_to_a_tuple() -> None:
    """请求体里是 JSON 数组,模型里必须是 tuple(不可变 + 顺序确定)。"""
    o = ThreadOwnership(
        thread_id="th-1", owner="alice", approvers=["bob"], created_at=TS
    )
    assert isinstance(o.approvers, tuple)
    assert o.approvers == ("bob",)


# ---------------------------------------------------------------------------
# 2. 拒绝畸形形态
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thread_id", ["", " ", " th-1", "th-1 "])
def test_thread_id_must_be_nonempty_and_unpadded(thread_id: str) -> None:
    with pytest.raises(ValidationError):
        _ownership(thread_id=thread_id)


@pytest.mark.parametrize("owner", ["", " ", "  alice", "alice  ", "\talice"])
def test_owner_must_be_a_canonical_subject(owner: str) -> None:
    with pytest.raises(ValidationError):
        _ownership(owner=owner)


def test_approvers_must_not_be_empty() -> None:
    """**可执行的指派至少一个审批人** —— 空集合不是"默认放行",是配置错误。

    冻结设计里"无归属行 ⇒ 没人能审批"讲的是**存储层缺行**(fail-closed);
    而这里禁止的是**构造**一个空集合的对象。两者不冲突:前者是运行时缺失,
    后者是不该被写进库的畸形输入。A3-2-FIX2 起 `approvers` 落在
    `thread_owners` 单行的 JSON 列里,空数组会在读取边界被拒绝。
    """
    with pytest.raises(ValidationError):
        _ownership(approvers=())


@pytest.mark.parametrize("approvers", [("bob", "bob"), ("bob", "carol", "bob")])
def test_duplicate_approvers_are_rejected(approvers: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        _ownership(approvers=approvers)


@pytest.mark.parametrize(
    "approvers", [("alice",), ("bob", "alice"), ("alice", "bob", "carol")]
)
def test_owner_cannot_be_an_approver(approvers: tuple[str, ...]) -> None:
    """**D-7:禁止自审批**。属主出现在自己的审批人集合里 ⇒ 构造期拒绝。

    这是 D-7 的**第一道**检查(指派时);第二道在 `/resume` 的服务层。
    两道都要有 —— 真正保护动作的是第二道。
    """
    with pytest.raises(ValidationError):
        _ownership(owner="alice", approvers=approvers)


@pytest.mark.parametrize("approvers", [("",), (" bob",), ("bob ",), ("bob", " ")])
def test_each_approver_must_be_a_canonical_subject(
    approvers: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        _ownership(approvers=approvers)


def test_naive_created_at_is_rejected() -> None:
    """与存储层 `_to_iso` 同一口径:naive datetime 会破坏"字典序 == 时间序"。"""
    with pytest.raises(ValidationError):
        _ownership(created_at=datetime(2026, 10, 8, 9, 0, 0))


def test_non_utc_offset_is_accepted_and_kept_as_an_instant() -> None:
    """带时区即可(不强制 UTC)—— 归一化到 UTC 是**存储层**的职责。"""
    plus8 = datetime(2026, 10, 8, 17, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    o = _ownership(created_at=plus8)
    assert o.created_at == TS


# ---------------------------------------------------------------------------
# 3. 规范表示(顺序不承载意义)
# ---------------------------------------------------------------------------


def test_approver_order_does_not_change_the_object() -> None:
    """语义是集合 ⇒ 同一指派只能有**一个**规范表示。

    否则同一份指派会有两种序列化结果,"是否同一指派"的比对与摘要失去意义,
    而且 `get_thread_ownership()` 的读回顺序会依赖存储里那串字符的书写顺序。
    """
    a = _ownership(approvers=("bob", "carol"))
    b = _ownership(approvers=("carol", "bob"))
    assert a.approvers == b.approvers == ("bob", "carol")
    assert a == b
    assert a.model_dump_json() == b.model_dump_json()


def test_canonical_form_is_lexicographic() -> None:
    o = _ownership(approvers=("zoe", "bob", "carol"))
    assert o.approvers == ("bob", "carol", "zoe")


def test_deduplication_check_runs_before_normalization() -> None:
    """先查重再排序 —— 否则 `("bob","bob")` 会被排序悄悄去重成合法输入。"""
    with pytest.raises(ValidationError):
        _ownership(approvers=("bob", "bob"))


# ---------------------------------------------------------------------------
# 4. 与认证层的**规范主体规则等价**
# ---------------------------------------------------------------------------
#
# 分层方向是 `app.security/**` → `app.schemas/**`,反向依赖会制造循环,
# 所以 `ownership.py` 自带一份与 `Principal.__post_init__` 等价的校验
# (见模块 docstring)。**重复本身是可接受的,漂移不是** —— 下面这组用例
# 就是防漂移的那颗钉子:同一个字符串不能在一侧合法、在另一侧非法。

_SUBJECT_CASES = [
    "alice",
    "a",
    "alice@corp.example",
    "svc-account_1",
    "用户甲",  # 两侧都不限制字符集(见下方说明)
    "",
    " ",
    "\t",
    "\n",
    " alice",
    "alice ",
    "  alice  ",
    "\talice",
]


@pytest.mark.parametrize("subject", _SUBJECT_CASES)
def test_subject_rule_matches_the_auth_layer(subject: str) -> None:
    """两侧对"什么算合法 subject"必须给出**同一个**判定。

    字符集不受限制是**当前事实**,不是安全声明:subject 的可信度来自它只
    出现在配置里(永不来自请求头),而不是来自字符集。这里把它钉住,是为了
    让"某一侧悄悄加了字符集限制"这种漂移立刻可见。
    """
    principal_ok = True
    try:
        Principal(subject=subject, role="approver")
    except ValueError:
        principal_ok = False

    ownership_ok = True
    try:
        _ownership(owner=subject)
    except ValidationError:
        ownership_ok = False

    assert principal_ok is ownership_ok, (
        f"subject={subject!r} 在认证侧为 {principal_ok}、在归属侧为 {ownership_ok}"
    )


@pytest.mark.parametrize("subject", _SUBJECT_CASES)
def test_approver_subject_uses_the_same_rule(subject: str) -> None:
    """审批人条目走**同一条**规范主体规则(空 / 带前后空白一律拒绝)。

    属主刻意换成一个**不在** `_SUBJECT_CASES` 里的名字 —— 否则 `"alice"` 这条
    会因为自审批(D-7)被拒,把"subject 规则"和"D-7"两件事混在一条断言里。
    """
    canonical = bool(subject.strip()) and subject == subject.strip()
    if canonical:
        o = _ownership(owner="owner-x", approvers=(subject,))
        assert o.approvers == (subject,)
    else:
        with pytest.raises(ValidationError):
            _ownership(owner="owner-x", approvers=(subject,))


# ---------------------------------------------------------------------------
# 5. 结构性:模型里没有授权,也没有反向依赖
# ---------------------------------------------------------------------------


def _module_imports() -> set[str]:
    tree = ast.parse(OWNERSHIP_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _code_tokens() -> set[str]:
    """模块里**非 docstring** 的标识符与字符串字面量(含导入名)。

    **刻意排除 docstring** —— 文档里*提到*某个名字(例如说明"本模块刻意
    不复用 `Principal` 的校验")不是使用它。这与
    `tests/test_security/test_store.py` 的 `_code_strings` 同一惯例:
    否则"写下为什么不这么做"反而会让测试失败,最后逼着人把解释删掉。
    """
    tree = ast.parse(OWNERSHIP_PATH.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))

    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tokens.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                tokens.add(node.module)
            tokens.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            tokens.add(node.value)
    return tokens


def test_ownership_module_does_not_import_the_security_layer() -> None:
    """分层:方向只能是 `app.security` → `app.schemas`,反向会制造循环。

    所以模型**不能**复用 `app.security.auth` 的校验函数 —— 它自带一份,
    由上面的等价性用例钉住。
    """
    imported = _module_imports()
    leaked = {name for name in imported if name.split(".")[:2] == ["app", "security"]}
    assert leaked == set(), f"归属模型反向依赖了安全层: {sorted(leaked)}"

    # 正对照:它**必须** import pydantic,否则上面的收集器是坏的
    assert "pydantic" in imported


def test_the_token_collector_ignores_docstrings_and_sees_real_code() -> None:
    """**收集器自身的正/负对照** —— 否则上面两条断言可能是恒真的。

    正对照:代码里确实存在的名字必须被收集到(`ThreadOwnership` / `pydantic`)。
    负对照:只出现在 docstring 里的名字**不得**被收集到 —— 用 `Principal`
    验证,它出现在模块 docstring 与函数 docstring 里,但代码里从不使用。
    """
    tokens = _code_tokens()
    assert "ThreadOwnership" in tokens
    assert "pydantic" in tokens
    assert "model_validator" in tokens
    # 负对照:docstring 提过,代码没用过
    source = OWNERSHIP_PATH.read_text(encoding="utf-8")
    assert "Principal" in source, "前置:docstring 里应当提到 Principal(否则负对照无意义)"
    assert "Principal" not in tokens, "收集器没有排除 docstring"


def test_ownership_module_contains_no_authorization_primitive() -> None:
    """模型不做授权判定:不得出现常量时间比较、摘要或密钥环。

    "谁能被指派 / 谁能审批"是**服务层**的判定(A5)。模型只回答
    "这条线程的属主是谁、指派了谁" —— 一旦它开始做判定,授权就有了
    第二个真相源。
    """
    tokens = _code_tokens()
    for forbidden in (
        "hmac",
        "compare_digest",
        "hashlib",
        "AuthKeyring",
        "AuthKeyEntry",
        "Principal",
        "from_entries",
        "sha256",
    ):
        assert forbidden not in tokens, f"归属模型里出现了授权原语: {forbidden}"


def test_ownership_module_does_not_consult_the_request_or_config() -> None:
    """模型不读请求、不读配置 —— 它只校验传进来的值。"""
    tokens = _code_tokens()
    for forbidden in (
        "fastapi",
        "Request",
        "get_settings",
        "environ",
        "getenv",
        "load_dotenv",
    ):
        assert forbidden not in tokens, f"归属模型触碰了请求/配置: {forbidden}"
