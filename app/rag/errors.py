"""RAG 异常层级 —— 只携带**通用文案 + 原因码**,不携带路径与堆栈。

为什么单独一层
------------
检索失败的原因会以三种方式离开进程:工具返回值、HTTP 响应体、日志。
只要有一处把 `str(exc)` 原样透出去,`C:\\...\\knowledge\\corpus.jsonl`
或索引目录就进了模型上下文 / 客户端响应 —— 这既是信息泄漏,也让攻击者
可以据此探测部署布局。

所以这里定两条规矩:

1. **消息是给人看的通用文案**,不含路径、DSN、集合名;
2. **原因是给程序看的短码**(`manifest-mismatch` / `duplicate-doc-id` …),
   它进日志与断言,但不进用户可见文案。

`IndexUnavailable` 与 `EmbeddingUnavailable` 都继承 `RagError`,而不是
继承 `AssertionError` 或 `RuntimeError`:上层(S2-B 的工具层)需要一个
可枚举的边界来把"知识源不可用"翻译成 `rag_error` 信封,而不是让它
冒成 500。
"""
from __future__ import annotations


class RagError(Exception):
    """RAG 层的基类异常。

    `reason` 是稳定的短码,用于日志与断言;`str(exc)` 是通用文案。
    """

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason or "unspecified"


class CorpusError(RagError):
    """语料本身不可用(缺失、清单不符、文档非法、id 重复)。"""


class StoreError(RagError):
    """存储层契约被违反(维度不符、长度不匹配、向量非有限数)。"""


class IndexUnavailable(RagError):
    """索引缺失 / 损坏 / 与当前语料或嵌入器不匹配。**fail closed**。"""


class EmbeddingUnavailable(RagError):
    """嵌入不可用(后续真实 provider 的失败出口;基线实现不会抛它)。"""


class RetrievalError(RagError):
    """检索请求本身非法(空查询、深度越界等)。"""
