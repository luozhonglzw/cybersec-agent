"""持久化 HITL checkpoint 的**安全存储层** —— Phase v0.4.0 S1。

本模块只解决两件事,两件都与"checkpoint 从进程内内存搬到磁盘"直接相关。

--------------------------------------------------------------------------
1) 严格反序列化白名单(TASK 3)
--------------------------------------------------------------------------
langgraph 默认的 msgpack ext 解码器(`_msgpack_ext_hook`)对扩展对象执行
`getattr(importlib.import_module(tup[0]), tup[1])(...)` —— **没有任何白名单**,
并且每个分支都以 `except Exception: return` 收尾,即"解不出来就静默变成
`None`"。对本项目而言这是两个真实缺陷:

- 能写 checkpoint 库的一方 = 能在本进程里执行**任意可调用对象**
  (实测:`Ext(0, ("builtins", "eval", "6*7"))` → 42);
- 静默 `None` 把"载荷损坏"伪装成"状态本来就是空的",于是会被上层误判成
  "checkpoint 丢失" → 409 —— 那是对世界的**错误陈述**。

本模块用**静态注册表**替代动态 import:载荷里的 `(module, name)` 只用于
**查表**,命中才构造,否则抛错。因此不存在"由不可信类型标识符驱动的导入
或任意可调用对象执行"。嵌套载荷也走同一个 hook,所以不存在"外层校验通过、
内层扩展逃逸"的旁路。

白名单**不是猜的**,而是实测出来的:用真实 `TriageService`(真实节点、真实
工具、真实 `AsyncSqliteSaver`)跑遍全部可达场景(暂停 / 批准恢复 / 拒绝恢复 /
裸图 / 放宽 serde 恢复 / 惰性超时),记录交给 msgpack 编码器的每个类型与
解码器看到的每个 ext code。观测到 7 个 pydantic 模型 + 1 个构造器 + 1 个方法,
标签只有 `msgpack` 与 `null`,总共只用 3 个 code:

    code 5 EXT_PYDANTIC_V2          —— 7 个 pydantic 模型
    code 3 EXT_METHOD_SINGLE_ARG    —— datetime(仅 fromisoformat)
    code 2 EXT_CONSTRUCTOR_KW_ARGS  —— langgraph.types.Interrupt

code 0 / 1 / 4 / 6 本项目**从不产生**,一律拒绝。完整证据见
PHASE v0.4.0-I1 报告 §3。

⚠ 推导口径的教训(实测):白名单**必须**从 `TriageService` 路径推导。裸
`graph.ainvoke` **不是**有效证据来源 —— 该形状不写入 system prompt,因此会
漏掉 `SystemMessage`,使 6/19 个已落盘 blob 在真实服务路径上解不开。只有真实
调用形状才是有效口径。

刻意与上游不同的一点:上游在 pydantic 分支失败时会退到
`cls.model_construct(**dump)`(**跳过校验**地造对象)。本模块**不退**,
而是抛错 —— 载荷来自我们自己的编码器,校验失败就说明数据损坏或被篡改,
此时"造一个未校验的对象继续跑"比响亮失败更危险。

实测坑(两次踩到,故显式记录):`ormsgpack` 会**吞掉**回调抛出的异常。
`packb(default=...)` 抛出的异常被换成误导性的
`TypeError: Type is not msgpack serializable: <T>`;`unpackb(ext_hook=...)`
抛出的异常被换成不带原因的 `ValueError("ext_hook failed")`,且
`__cause__` / `__context__` **双双为 None**。后果是 hook 里抛出的
`CheckpointPayloadRejectedError` 到不了调用方。因此拒绝的翻译必须在
`StrictCheckpointSerde.loads_typed` 里再做一次(见该方法内的注释);
这也意味着 ormsgpack 会销毁原因链,诊断只能停在通用消息上。

--------------------------------------------------------------------------
2) 后端故障归因(TASK 5)
--------------------------------------------------------------------------
checkpoint 后端故障必须映射成 **503**,且**不得**被表述成 404(未知 thread)
或 409(checkpoint 丢失)—— 那是对世界的错误陈述(PHASE v0.4.0-D4 §5.4)。

归因放在**发生处**:只有 checkpoint 存储自己的异常被翻译成
`CheckpointStorageError`。审计存储的失败**不**经过这里,因此 v0.3.0 已文档化
的"读 503 / 写 500"不对称保持不变(D4 §5.4 明确要求它不变)。

--------------------------------------------------------------------------
本模块**不**声称(见 D4 §8)
--------------------------------------------------------------------------
不声称端到端 exactly-once;不声称与审计写入的原子性;不声称防篡改或
不可否认;不声称消除了 `PYSEC-2026-3636` / `PYSEC-2026-3635` 暴露(受影响
的是 Store 命名空间的 LIKE 读取路径,本项目未使用 —— 这是**缓解论据,不是
移除**);不声称生产就绪;不声称支持多副本分布式 checkpoint。
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_KW_ARGS,
    EXT_METHOD_SINGLE_ARG,
    EXT_PYDANTIC_V2,
    JsonPlusSerializer,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Interrupt
import ormsgpack

from app.schemas.approval import ApprovalDecision, ApprovalRequest
from app.schemas.policy import PolicyDecision
from app.schemas.response import ResponsePlan

# ---------------------------------------------------------------------------
# 错误族
# ---------------------------------------------------------------------------


class CheckpointStorageError(RuntimeError):
    """checkpoint 存储不可用的**基类**。上层据此映射成 503。

    刻意与审计存储的错误族分开:两者都是"持久化后端出问题",但只有
    checkpoint 的这一族应当改变 `/resume` 的失败语义。
    """


class CheckpointBackendUnavailableError(CheckpointStorageError):
    """checkpoint 后端(文件 / 连接 / 磁盘)不可用。"""


class CheckpointPayloadRejectedError(CheckpointStorageError):
    """checkpoint 载荷不在白名单内 —— 不支持,或被篡改,或已损坏。

    与"后端不可用"分开是为了让诊断能区分"存储坏了"与"数据坏了";
    两者都映射 503,但消息不同,且都**绝不**退化成 404 / 409。
    """


# ---------------------------------------------------------------------------
# 白名单:静态注册表(无 importlib、无动态导入)
# ---------------------------------------------------------------------------

#: ext code 5(EXT_PYDANTIC_V2)。键是载荷里的 `(module, name)`,值是**已经
#: 导入好的类对象** —— 载荷只能查表,不能决定导入什么。
_ALLOWED_PYDANTIC: dict[tuple[str, str], type] = {
    ("app.schemas.approval", "ApprovalDecision"): ApprovalDecision,
    ("app.schemas.approval", "ApprovalRequest"): ApprovalRequest,
    ("app.schemas.policy", "PolicyDecision"): PolicyDecision,
    ("app.schemas.response", "ResponsePlan"): ResponsePlan,
    ("langchain_core.messages.ai", "AIMessage"): AIMessage,
    ("langchain_core.messages.human", "HumanMessage"): HumanMessage,
    ("langchain_core.messages.system", "SystemMessage"): SystemMessage,
}

#: ext code 2(EXT_CONSTRUCTOR_KW_ARGS)。`Interrupt` 是 dataclass,字段
#: `value` / `id`,由上游以 `_asdict()` 形式编码。
_ALLOWED_CONSTRUCTOR_KWARGS: dict[tuple[str, str], type] = {
    ("langgraph.types", "Interrupt"): Interrupt,
}

#: ext code 3(EXT_METHOD_SINGLE_ARG)。本项目只用到 datetime 的
#: `fromisoformat`。**不接受**任意 "module.name.method" 三元组。
_ALLOWED_METHODS: frozenset[tuple[str, str, str]] = frozenset(
    {("datetime", "datetime", "fromisoformat")}
)

#: `loads_typed` 接受的序列化标签。本项目实测只产生 `msgpack` 与 `null`
#: (`bytes` / `bytearray` 是惰性直通,不涉及类型标识符,故一并接受)。
#: 刻意**拒绝** `json` 与 `pickle`:
#:   - `json` 会走到 langchain-core 的 reviver(另一条我们既不控制也不需要
#:     的动态导入面);
#:   - `pickle` 反序列化就是任意代码执行。
_ALLOWED_TAGS: frozenset[str] = frozenset({"msgpack", "null", "bytes", "bytearray"})


def _strict_unpack(data: bytes) -> Any:
    """用**本模块的** hook 解一层载荷 —— 嵌套扩展同样受白名单约束。"""
    return ormsgpack.unpackb(
        data, ext_hook=_strict_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
    )


def _shape(payload: Any, expected: int, code: int) -> list[Any]:
    """校验载荷形状。**不**把载荷内容放进错误消息(可能含敏感数据)。"""
    if not isinstance(payload, (list, tuple)) or len(payload) != expected:
        raise CheckpointPayloadRejectedError(
            f"checkpoint 载荷形状非法(扩展类型 {code})"
        )
    return list(payload)


def _key(*parts: Any) -> tuple[str, ...]:
    """类型标识必须是字符串 —— 同时挡住不可哈希元素导致的 TypeError。"""
    if not all(isinstance(part, str) for part in parts):
        raise CheckpointPayloadRejectedError("checkpoint 载荷的类型标识不是字符串")
    return tuple(parts)  # type: ignore[return-value]


def _strict_ext_hook(code: int, data: bytes) -> Any:
    """msgpack 扩展解码器 —— 白名单之外的**一律抛错**。

    绝不返回 `None`:静默 `None` 会让"载荷损坏"伪装成"状态为空",
    进而被上层误判为"checkpoint 丢失"(409)。
    """
    if code == EXT_PYDANTIC_V2:
        module, name, dump, _method = _shape(_strict_unpack(data), 4, code)
        cls = _ALLOWED_PYDANTIC.get(_key(module, name))
        if cls is None:
            raise CheckpointPayloadRejectedError(
                "checkpoint 载荷含未授权的 pydantic 类型"
            )
        try:
            return cls(**dump)
        except Exception as exc:
            raise CheckpointPayloadRejectedError(
                "checkpoint 载荷未通过模型校验(数据损坏或被篡改)"
            ) from exc

    if code == EXT_CONSTRUCTOR_KW_ARGS:
        module, name, kwargs = _shape(_strict_unpack(data), 3, code)
        cls = _ALLOWED_CONSTRUCTOR_KWARGS.get(_key(module, name))
        if cls is None:
            raise CheckpointPayloadRejectedError(
                "checkpoint 载荷含未授权的构造器类型"
            )
        try:
            return cls(**kwargs)
        except Exception as exc:
            raise CheckpointPayloadRejectedError(
                "checkpoint 载荷未通过构造校验(数据损坏或被篡改)"
            ) from exc

    if code == EXT_METHOD_SINGLE_ARG:
        module, name, arg, method = _shape(_strict_unpack(data), 4, code)
        if _key(module, name, method) not in _ALLOWED_METHODS:
            raise CheckpointPayloadRejectedError(
                "checkpoint 载荷含未授权的类方法调用"
            )
        try:
            return datetime.fromisoformat(arg)
        except Exception as exc:
            raise CheckpointPayloadRejectedError(
                "checkpoint 载荷的时间戳无法解析(数据损坏或被篡改)"
            ) from exc

    raise CheckpointPayloadRejectedError(
        f"checkpoint 载荷使用了未授权的扩展类型(扩展类型 {code})"
    )


# ---------------------------------------------------------------------------
# 序列化器
# ---------------------------------------------------------------------------


class StrictCheckpointSerde(JsonPlusSerializer):
    """`JsonPlusSerializer` + 窄白名单 + 显式拒绝。

    继承而不是重写:编码侧(`dumps_typed` / `_msgpack_default`)行为**不变**,
    只有解码侧被收紧 —— 我们写出去的东西仍然是标准 langgraph 格式。
    """

    def __init__(self) -> None:
        # pickle_fallback 必须保持 False:pickle 反序列化 = 任意代码执行。
        # 显式传入而不是依赖默认值,这样上游改默认值也不会静默放宽。
        super().__init__(pickle_fallback=False, __unpack_ext_hook__=_strict_ext_hook)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        if not isinstance(data, tuple) or len(data) != 2:
            raise CheckpointPayloadRejectedError(
                "checkpoint 记录的 (type, bytes) 形状非法"
            )
        type_, _payload = data
        # 不回显 `type_` 内容:它来自数据库,可能是超长字符串(日志注入)。
        if not isinstance(type_, str) or type_ not in _ALLOWED_TAGS:
            raise CheckpointPayloadRejectedError(
                "checkpoint 记录使用了未授权的序列化标签"
            )
        try:
            return super().loads_typed(data)
        except CheckpointPayloadRejectedError:
            raise
        except ValueError as exc:
            # ⚠ 实测坑(见下):`ormsgpack.unpackb` 会**吞掉** `ext_hook` 抛出的
            # 任何异常,改抛一个不带原因的 `ValueError("ext_hook failed")`
            # (`__cause__` / `__context__` 都是 None)。因此本模块 hook 里抛出的
            # `CheckpointPayloadRejectedError` 到不了调用方 —— 必须在这一层把
            # `ValueError` 重新翻译回来,否则"载荷被拒绝"会退化成裸 ValueError,
            # 既不是我们的错误族、也映射不到 503。
            # 顺带覆盖 `ormsgpack.MsgpackDecodeError`(ValueError 子类)与
            # pydantic 的 `ValidationError`(同样是 ValueError 子类,已在 hook
            # 内先行转换),即"任何解不出来的 msgpack 载荷"。
            # 代价:ormsgpack 销毁了原因链,因此这里只能给出通用消息。
            raise CheckpointPayloadRejectedError(
                "checkpoint 载荷无法解码(数据损坏或被篡改)"
            ) from exc


def build_checkpoint_serde() -> StrictCheckpointSerde:
    """组合根使用的工厂。"""
    return StrictCheckpointSerde()


# ---------------------------------------------------------------------------
# 后端故障归因
# ---------------------------------------------------------------------------


@contextmanager
def _translate_backend_errors() -> Iterator[None]:
    """把 checkpoint 后端自身的失败翻译成 `CheckpointBackendUnavailableError`。

    只翻译真正的后端/IO 故障。编码失败(`ormsgpack.MsgpackEncodeError`)刻意
    **不**翻译 —— 那是"有个对象不该被写进 checkpoint"的**程序缺陷**,
    把它说成"后端不可用"会掩盖 bug。
    """
    try:
        yield
    except CheckpointStorageError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise CheckpointBackendUnavailableError(
            "checkpoint 存储后端不可用"
        ) from exc


class SafeAsyncSqliteSaver(AsyncSqliteSaver):
    """`AsyncSqliteSaver`,但后端故障在**发生处**被归因。

    为什么继承而不是包一层代理:子类自动继承锁、事件循环绑定、`setup()` 与
    同步/异步两套接口(同步方法内部就是 `run_coroutine_threadsafe(self.aXXX)`),
    因此只需覆盖 5 个异步入口,不必重写整个 saver 协议 —— 重写协议是这类
    改造里最容易悄悄引入偏差的地方。

    归因之所以要精确到"只有 checkpoint",是因为审计存储的失败**不**经过这里:
    把两者混成一个 503 会改变 v0.3.0 已文档化的读写不对称(D4 §5.4)。
    """

    async def setup(self) -> None:
        with _translate_backend_errors():
            await super().setup()

    async def aget_tuple(self, config):  # type: ignore[override]
        with _translate_backend_errors():
            return await super().aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):  # type: ignore[override]
        with _translate_backend_errors():
            async for item in super().alist(
                config, filter=filter, before=before, limit=limit
            ):
                yield item

    async def aput(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        with _translate_backend_errors():
            return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):  # type: ignore[override]
        with _translate_backend_errors():
            return await super().aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        with _translate_backend_errors():
            await super().adelete_thread(thread_id)
