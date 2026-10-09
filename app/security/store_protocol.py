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

**v0.3.0-A3-2 的增量:8 → 10 个方法,纯增量。**
新增 `record_thread_ownership` / `get_thread_ownership`,形状与
`record_incident` / `get_incident` 同款(一个聚合进、一个聚合出)。
**既有 8 个方法的签名与行为一个字都没动** —— 改既有签名会让所有现有调用方
的等价性保证静默失效,而加方法是可见、可评审的 diff。

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
from app.schemas.ownership import ThreadOwnership


@runtime_checkable
class AuditStore(Protocol):
    """incidents / action_requests / audit_logs 三张 append-only 审计表,加上
    thread_owners(线程归属,单行)的读写契约。

    与 `app.security.store.SqliteAuditStore` 的公开方法**逐一对应**。
    调用方(组合根 / triage 服务 / 评测适配器)只应依赖本 Protocol,
    而不是具体后端。装配点**唯一**:`app/api/main.py` 的 `build_audit_store`。

    归属表的两个方法**只做存储**:它们不校验"谁能被指派"、不校验"谁能审批",
    也不检查被指派的 subject 是否存在于配置。那些是**授权**判定,属于服务层
    (A5);本契约只保证"属主 + 完整审批集合"能落库并原样读回。
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

    def record_thread_ownership(self, ownership: ThreadOwnership) -> None:
        """写入一条线程归属(属主 + **完整**审批集合),**单条 INSERT**。

        为什么必须是一个方法而不是"先写属主、再逐个写审批人":部分写入
        (属主在、审批集合缺)会产生一条**看起来已指派、其实没人能审批**的
        线程 —— 而"没人能审批"正是冻结设计里的 fail-closed 终态,于是这个
        bug 会伪装成正常行为。A3-2-FIX2 起归属是**一行**(审批集合与属主同处
        一行),因此"部分写入"在结构上不可能存在,而不是靠调用方记得按顺序写。

        失败即抛(**不吞、不重试、不降级到另一个后端**):`/triage` 在生成
        任何东西之前调用本方法,所以这里抛出去就是 fail closed —— 没有线程、
        没有审计行。

        重复注册(同一 `thread_id` 再来一次)必须**响亮失败**:
        `thread_owners.thread_id` 是主键,重复插入由数据库拒绝,本方法不得把
        它吞掉或改成 upsert;并发注册同一 thread_id 时,主键保证**恰好一个**
        胜者,且落库的审批集合不会被合并。
        """
        ...

    # ---------- 读 ----------

    def get_incident(self, incident_id: str) -> Incident | None:
        """按 id 读回 incident;不存在返回 None。"""
        ...

    def get_thread_ownership(self, thread_id: str) -> ThreadOwnership | None:
        """按 thread_id 读回归属(属主 + 完整审批集合);不存在返回 None。

        **无归属 ⇒ 返回 None,绝不回退**。调用方拿到 `None` 只能拒绝
        (A5 的 `/resume` 对无归属线程一律 404),不得解释成"那就用任意
        approver"或"那就用属主"。

        审批人集合以**规范序**(字典序)返回,与 `ThreadOwnership` 的归一化
        一致 —— 否则"写进去再读出来是否相等"这个最基本的往返性质会依赖
        存储里那串字符的书写顺序。存储值畸形时**响亮失败**,不得静默读成
        "没有审批人"。
        """
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
