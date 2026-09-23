"""离线网络出口守卫(**基于 CPython 审计钩子**)。

为什么需要它
------------
"本次 E2E 全程离线"是一个**经验主张**。若它只写在文档里,那么任何一次
误引入的 `httpx` 调用都会静默地消耗额度、并把结果污染成"真实 provider 观测"。
所以这个主张必须由**运行时**证据支撑,而不是由代码 review 支撑。

为什么用 `sys.addaudithook` 而不是替换 socket 工厂
------------------------------------------------
两种常见做法都被刻意排除:

1. **替换 `socket.socket` / `socket.create_connection`。** 这会破坏 asyncio:
   Windows 上的 `ProactorEventLoop` 与 `SelectorEventLoop` 都依赖真实的
   socket 对象与 `socketpair()`;把它们换成假对象,事件循环会在启动阶段
   就崩掉 —— 于是"守卫"变成了"实验跑不起来"。
2. **在 `httpx` / `openai` 上打补丁。** 那只覆盖我们**想到**的客户端;
   而风险恰恰来自没想到的那一个。

审计钩子工作在**解释器层**:它不改任何对象,只在受审计操作发生时被调用。
它覆盖全部 socket 与 HTTP 客户端,包括将来才引入的那些。

两条刻意的设计决定
----------------
**环回不算出口。** Windows 上 `socket.socketpair()` 是通过 127.0.0.1 的真实
连接模拟的,`asyncio` 在部分配置下会用到它。把环回也算成"网络访问",
守卫就会在完全离线的情况下持续误报,最终被关掉 —— 一个总在误报的守卫
等于没有守卫。

**默认立即抛错。** 只在事后检查的守卫很容易被忽略(断言被删、被跳过、
或结果没被读)。`strict=True` 让第一次越界就在**发生处**炸掉。
"""
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

#: 对外网络操作的审计事件名。命中任意一条即为"发起了网络访问"。
EGRESS_AUDIT_EVENTS: frozenset[str] = frozenset({
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.sethostname",
    "urllib.Request",
    "http.client.connect",
    "http.client.send",
    "ftplib.connect",
    "smtplib.connect",
    "imaplib.open",
    "poplib.connect",
    "ssl.SSLContext.wrap_socket",
})

#: socket 对象创建。**不作为出口证据** —— 部分平台的事件循环会为自管道
#: 建 socket。单独记录,便于排查"到底是谁在建连接"。
SOCKET_CREATION_AUDIT_EVENT = "socket.__new__"

#: 环回主机名 / 地址。命中即视为**非出口**。
LOOPBACK_HOSTS: frozenset[str] = frozenset({
    "127.0.0.1",
    "::1",
    "localhost",
    "localhost.localdomain",
})

#: 通配地址。`bind` 到它 = 在所有网卡上开监听,是真实的网络暴露 —— 不豁免。
WILDCARD_HOSTS: frozenset[str] = frozenset({"0.0.0.0", "::", ""})


class NetworkEventKind(str, Enum):
    """审计事件的分类。"""

    EGRESS = "EGRESS"                    # 对外网络操作 —— **必须为 0**
    LOOPBACK = "LOOPBACK"                # 环回 —— 允许
    SOCKET_CREATION = "SOCKET_CREATION"  # socket 创建 —— 仅记录


class NetworkEgressError(AssertionError):
    """检测到对外网络访问。在 `strict=True` 时于**发生处**抛出。"""


@dataclass(frozen=True)
class NetworkEvent:
    """一条审计事件。"""

    event: str
    kind: NetworkEventKind
    detail: str = ""


@dataclass
class NetworkEgressGuard:
    """离线出口守卫。

    用法::

        with NetworkEgressGuard() as guard:
            ...  # 全程离线
        guard.assert_clean()

    **钩子不可卸载**(CPython 不提供移除 API),因此实现为"登记表":
    没有任何活动守卫时,钩子只做一次空列表判断就返回,开销可忽略。
    """

    strict: bool = True
    allow_loopback: bool = True
    events: list[NetworkEvent] = field(default_factory=list)
    _active: bool = field(default=False, repr=False)

    # ---- 上下文管理 ----

    def __enter__(self) -> "NetworkEgressGuard":
        _install_hook()
        _ACTIVE_GUARDS.append(self)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self in _ACTIVE_GUARDS:
            _ACTIVE_GUARDS.remove(self)
        self._active = False
        return False

    # ---- 查询 ----

    @property
    def egress_events(self) -> list[NetworkEvent]:
        return [item for item in self.events if item.kind is NetworkEventKind.EGRESS]

    @property
    def socket_creation_events(self) -> list[NetworkEvent]:
        return [
            item for item in self.events
            if item.kind is NetworkEventKind.SOCKET_CREATION
        ]

    @property
    def clean(self) -> bool:
        return not self.egress_events

    def assert_clean(self) -> None:
        offenders = self.egress_events
        if offenders:
            raise NetworkEgressError(
                f"检测到 {len(offenders)} 次对外网络访问:"
                f"{[(item.event, item.detail) for item in offenders[:5]]} —— "
                "离线阶段不允许任何网络出口"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "egress_events": len(self.egress_events),
            "socket_creation_events": len(self.socket_creation_events),
            "loopback_events": len(
                [i for i in self.events if i.kind is NetworkEventKind.LOOPBACK]
            ),
            "events": [
                {"event": item.event, "kind": item.kind.value, "detail": item.detail}
                for item in self.events
            ],
        }

    # ---- 内部:钩子回调 ----

    def _record(self, event: str, detail: str, kind: NetworkEventKind) -> None:
        self.events.append(NetworkEvent(event=event, kind=kind, detail=detail))
        if kind is NetworkEventKind.EGRESS and self.strict:
            raise NetworkEgressError(
                f"离线阶段检测到对外网络访问:{event}({detail}) —— 立即中止"
            )


#: 当前活动的守卫。钩子在空列表时立即返回。
_ACTIVE_GUARDS: list[NetworkEgressGuard] = []
_HOOK_INSTALLED = False


def _describe(event: str, args: tuple) -> str:
    """尽量从审计事件参数里掏出一个主机标识。**失败不抛错。**"""
    try:
        if event in ("socket.connect", "socket.bind", "socket.sendto"):
            address = args[1] if len(args) > 1 else None
            if isinstance(address, tuple) and address:
                return str(address[0])
            return str(address)
        if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
            return str(args[0]) if args else ""
        if event == "urllib.Request":
            return str(args[0]) if args else ""
        if event == "http.client.connect":
            return str(args[1]) if len(args) > 1 else ""
        return ""
    except Exception:  # pragma: no cover - 描述失败绝不影响被测流程
        return ""


def _classify(event: str, detail: str, *, allow_loopback: bool) -> NetworkEventKind:
    if event == SOCKET_CREATION_AUDIT_EVENT:
        return NetworkEventKind.SOCKET_CREATION
    if allow_loopback and detail in LOOPBACK_HOSTS:
        return NetworkEventKind.LOOPBACK
    if detail in WILDCARD_HOSTS and event == "socket.bind":
        # bind 通配地址 = 全网卡监听,是真实的网络暴露。
        return NetworkEventKind.EGRESS
    if detail in WILDCARD_HOSTS:
        return NetworkEventKind.LOOPBACK
    return NetworkEventKind.EGRESS


def _hook(event: str, args: tuple) -> None:
    """审计钩子。**先判空再判事件名** —— 这是热路径。"""
    if not _ACTIVE_GUARDS:
        return
    if event not in EGRESS_AUDIT_EVENTS and event != SOCKET_CREATION_AUDIT_EVENT:
        return
    detail = _describe(event, args)
    for guard in list(_ACTIVE_GUARDS):
        guard._record(
            event, detail, _classify(event, detail, allow_loopback=guard.allow_loopback)
        )


def _install_hook() -> None:
    """安装审计钩子(幂等)。

    CPython **不提供**移除审计钩子的 API,所以这里只安装一次;
    未激活时钩子会在第一行返回。
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.addaudithook(_hook)
    _HOOK_INSTALLED = True


def audit_hook_installed() -> bool:
    """钩子是否已安装 —— 供测试断言"守卫真的接上了"。"""
    return _HOOK_INSTALLED


def egress_events_of(guards: Iterable[NetworkEgressGuard]) -> list[NetworkEvent]:
    return [event for guard in guards for event in guard.egress_events]
