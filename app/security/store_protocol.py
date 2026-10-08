"""审计持久化的后端契约(Phase v0.2.0-M1a)。

本模块**只声明契约**,不含任何实现、不 import 任何数据库驱动,也**不构造任何后端**。
它的作用是:把 `SqliteAuditStore` 已经存在的公开行为**写死成可校验的形状**,
使 PostgreSQL 后端能在同一个契约下被逐签名比对。

装配发生在**组合根**(`app/api/main.py` 的 `build_audit_store`,Phase v0.2.0-M1c):
那里按 `Settings` 的后端设置选择后端,并把实例注入图 / triage / 端点。
本模块只提供类型,不含任何"选哪个后端"的逻辑。

方法集与签名**逐字取自 `app/security/store.py` 的 `SqliteAuditStore`**,
不是另起一套设计。本文件与 `store.py` 的等价性由
`tests/test_security/test_store_protocol.py` 用 `inspect.signature` 结构性断言守着
—— 任一侧改了签名而另一侧没跟上,测试立刻失败。

刻意排除构造器(`__init__`):
    构造契约**不属于**行为契约。`SqliteAuditStore(db_path)` 与未来的
    `PostgresAuditStore(dsn)` 参数不同(文件路径 vs 连接串),把构造器塞进
    Protocol 会让两个后端都无法满足它 —— 那正是"抽象"要避免的事。
    连接参数由各后端自身与组合根负责;本 Protocol 只约束**实例方法**。

刻意排除私有方法(`_connect` / `_select_action_rows` / `_row_to_action` /
`_row_to_reasons` / `_project` / `_row_to_audit`):
    它们是实现细节。SQLite 与 PostgreSQL 的行对象完全不同
    (`sqlite3.Row` vs `psycopg.rows` 的 tuple/dict),强行统一私有辅助方法
    会把"实现自由"一起锁死。

契约本身不新增任何安全语义:它描述的是**已有的**读写能力,
不改变 `append-only`、不改变审批控制流、不改变异常语义。
"""
from typing import Protocol, runtime_checkable

from app.schemas.approval import ApprovalRequest
from app.schemas.audit import AuditRecord
from app.schemas.incident import Incident


@runtime_checkable
class AuditStore(Protocol):
    """incidents / action_requests / audit_logs 三张 append-only 表的读写契约。

    与 `app.security.store.SqliteAuditStore` 的公开方法**逐一对应**。
    调用方(组合根 / triage 服务 / 评测适配器)只应依赖本 Protocol,
    而不是具体后端。装配点**唯一**:`app/api/main.py` 的 `build_audit_store`。
    """

    # ---------- 写 ----------

    def record_incident(self, incident: Incident) -> None:
        """写入一条 incident 快照(append-only)。"""
        ...

    def record_action_request(
        self,
        request: ApprovalRequest,
        *,
        incident_id: str | None = None,
    ) -> list[str]:
        """写入一张审批单(每个 ResponseAction 一行)。

        返回写入的行 id 列表,顺序与 `request.actions` 一致。
        """
        ...

    def append_audit(self, record: AuditRecord) -> None:
        """追加一条审计记录(append-only)。"""
        ...

    # ---------- 读 ----------

    def get_incident(self, incident_id: str) -> Incident | None:
        """按 id 读回 incident;不存在返回 None。"""
        ...

    def get_approval_request(self, thread_id: str) -> ApprovalRequest | None:
        """按 thread_id 把逐动作的行重新组装成 request 级模型。"""
        ...

    def list_action_rows(
        self,
        *,
        thread_id: str | None = None,
        incident_id: str | None = None,
    ) -> list[dict]:
        """按动作粒度读出待审批动作的扁平投影。"""
        ...

    def pending_action_rows(self, *, thread_id: str | None = None) -> list[dict]:
        """派生查询:仍处于待审批状态的动作行(状态不落库)。"""
        ...

    def list_audit(
        self,
        *,
        thread_id: str | None = None,
        incident_id: str | None = None,
        event: str | None = None,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[AuditRecord]:
        """按条件读审计流,按 (ts, 存储层 tiebreaker) 升序 —— 顺序确定,不依赖存储顺序。

        tiebreaker 是**实现细节**,不属于契约:SQLite 用隐式 `rowid`,
        PostgreSQL 用 identity 列 `seq`。可主张的性质只有"对已提交且可见的
        行集合,同一份数据反复读得到同一顺序"。
        """
        ...
