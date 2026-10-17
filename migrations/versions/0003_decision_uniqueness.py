"""durable decision uniqueness (at-most-once decision recording)

Revision ID: 0003_decision_uniqueness
Revises: 0002_thread_ownership
Create Date: 2026-10-12

Phase v0.4.0 S1 / TASK 4 —— 决策身份的**存储层**唯一性。

**范围**:只加**两条部分唯一索引**。
本迁移**不**改 `0001` / `0002` 的任何对象、**不**改任何列、**不**做数据
迁移、**不**新建表、**不**新建触发器、**不**引入新角色或权限种类。

**为什么必须有这个迁移(而不是像 SQLite 那样在 store 里建)**
------------------------------------------------------------------
SQLite 的 schema 由 `app/security/store.py` 在构造时建立;而 PostgreSQL 的
运行时角色(`cybersec_app`)**没有 DDL 权限**,`PostgresAuditStore` 也刻意
不做运行时 schema 变更。因此 PostgreSQL 侧的唯一约束只能来自迁移。

**为什么是部分(partial)唯一索引**
------------------------------------------------------------------
唯一性只对**决策事件**成立:

- `approval.decided` —— 身份是 `(thread_id, interrupt_id)`:同一次暂停只允许
  产生一个决定;
- `approval.timeout` —— 身份是 `thread_id`:一条线程只允许产生一次惰性超时。

其余事件(`plan.created` / `policy.evaluated` / `approval.requested` ...)天然
可以重复出现,不能纳入唯一约束 —— 普通唯一索引会把正常的多条审计行判成冲突。

**为什么唯一性必须落在存储层**
------------------------------------------------------------------
恢复路径可能被并发触发。任何"先 SELECT 再 INSERT"的检查都存在竞态窗口;
只有数据库在**提交点**判定的唯一约束才能保证"恰好一个胜者"。这是**结构性**
保证,不是调用方约定。

**⚠ 本迁移明确不承诺(见 PHASE v0.4.0-D4 §8)**
------------------------------------------------------------------
- **不**承诺端到端 exactly-once:审计行的落地与 checkpoint 的推进不在同一个
  事务里,两者的原子性本阶段未实现;
- **不**覆盖 `interrupt_id` 为 NULL 的身份:SQLite 与 PostgreSQL 都认为
  NULL 互不相等,因此 NULL 身份不受唯一索引保护(实测)。本项目的 `/resume`
  由服务端注入 `interrupt_id`(D7),经 API 的决策恒为非 NULL;直接调用图的
  旁路不在本阶段承诺范围内。

**⚠ 历史重复的处置(勿改)**
------------------------------------------------------------------
建唯一索引**之前**必须显式预检历史重复行。理由(实测,不是推断):SQLite 与
PostgreSQL 建唯一索引失败时给出的消息**只点名列、不点名索引**,运维无法据此
判断是哪条不变量、该查哪些行。预检把驱动报错换成一句能直接行动的诊断。

预检**只读**,发现重复时**响亮失败**并交给人处置 —— 本迁移**绝不**删除或
改写历史审计行。审计表是 append-only 证据,静默"清理重复"就是篡改证据。

**⚠ downgrade 的关键约束(勿改)**
------------------------------------------------------------------
`_STATEMENTS_DOWN` 里**恰好两条** `DROP INDEX IF EXISTS`,且:

- **绝不**加 `DROP FUNCTION cybersec_reject_append_only_mutation()` ——
  那个函数由 **0001** 创建并拥有;
- **绝不**加 `DROP TABLE` —— `audit_logs` 由 **0001** 创建,`thread_owners`
  由 **0002** 创建,都不是本迁移拥有的对象。

"照 up-path 反向抄一遍"在这里唯一会出错的地方,就是顺手删掉别人的对象。
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0003_decision_uniqueness"
down_revision: Union[str, None] = "0002_thread_ownership"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: 本迁移拥有的两条索引。down-path 只允许删这两个名字。
INDEX_DECIDED = "uq_audit_decided_per_interrupt"
INDEX_TIMEOUT = "uq_audit_timeout_per_thread"

#: 决策身份的**单一真相源**在本文件内。
#:
#: 刻意**不**从 `app.security.store` import:迁移是**历史快照**,必须永远
#: 表达"当时建了什么"。若去 import 应用代码,将来应用改了身份定义,这条
#: 已经执行过的迁移的语义会跟着变 —— 那是不可接受的。重复这十几行常量
#: 是刻意的取舍,不是疏漏。
_IDENTITIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (INDEX_DECIDED, "approval.decided", ("thread_id", "interrupt_id")),
    (INDEX_TIMEOUT, "approval.timeout", ("thread_id",)),
)


def _duplicate_probe_sql(event: str, columns: tuple[str, ...]) -> str:
    """按身份列生成"同一身份出现多次"的检测 SQL(**只读**)。

    预检口径必须与 `_STATEMENTS_UP` 里那条**部分唯一索引**一致:任一身份列
    为 NULL 的行,索引都不判为冲突(PostgreSQL 默认 NULL 互不相等 —— 实测),
    因此这里同样要排除。否则 `GROUP BY` 会把多行 NULL 归成一组而 `COUNT(*)
    > 1`,报出一个索引**本会接受**的假重复,把本可正常建立的迁移挡在门外。

    排除之后,预检扫描的行集 = 索引覆盖的行集:真重复(非 NULL 身份完全相同)
    仍然响亮失败。
    """
    joined = ", ".join(columns)
    non_null = " AND ".join(f"{column} IS NOT NULL" for column in columns)
    return (
        f"SELECT {joined}, COUNT(*) AS n FROM audit_logs"
        f" WHERE event = '{event}' AND {non_null}"
        f" GROUP BY {joined} HAVING COUNT(*) > 1"
    )


def _assert_no_duplicate_decisions() -> None:
    """建索引前预检历史重复行;**绝不**自动修复。

    只读,且发现重复即响亮失败 —— 交给人处置,不删行、不改行、不跳过约束。
    """
    bind = op.get_bind()
    for index_name, event, columns in _IDENTITIES:
        offenders = bind.exec_driver_sql(_duplicate_probe_sql(event, columns)).fetchall()
        if offenders:
            raise RuntimeError(
                f"无法建立唯一索引 {index_name}:历史审计行已存在 "
                f"{len(offenders)} 组重复的决策身份({event},身份列 "
                f"{', '.join(columns)})。本迁移不会删除或改写历史审计行"
                "(append-only 证据),请人工核对后处置。"
            )


# 每一条目 = **恰好一条** SQL 语句。逐条执行,而不是拼一个多语句字符串:
# 一来错误信息能精确指到出问题的那一条,二来避免依赖驱动的多语句行为。
_STATEMENTS_UP: tuple[str, ...] = (
    # 部分唯一索引:`WHERE` 子句把约束限制在决策事件上。
    f"""
    CREATE UNIQUE INDEX {INDEX_DECIDED}
        ON audit_logs (thread_id, interrupt_id)
        WHERE event = 'approval.decided'
    """,
    f"""
    CREATE UNIQUE INDEX {INDEX_TIMEOUT}
        ON audit_logs (thread_id)
        WHERE event = 'approval.timeout'
    """,
)

#: **恰好两条**,都只删本迁移自己创建的索引。详见模块 docstring 的 ⚠ 段。
_STATEMENTS_DOWN: tuple[str, ...] = (
    f"DROP INDEX IF EXISTS {INDEX_DECIDED}",
    f"DROP INDEX IF EXISTS {INDEX_TIMEOUT}",
)


def upgrade() -> None:
    """建立决策身份的两条部分唯一索引。"""
    # 预检必须在第一条 CREATE UNIQUE INDEX **之前** —— 否则索引自己先撞上
    # 历史重复行,抛出一句只点名列、不点索引名的驱动报错。
    _assert_no_duplicate_decisions()
    for statement in _STATEMENTS_UP:
        op.execute(statement)


def downgrade() -> None:
    """**破坏性**:删除决策唯一约束(数据不动)。

    只删 0003 **自己拥有的**两条索引。`audit_logs` 属于 0001、
    `thread_owners` 属于 0002、`cybersec_reject_append_only_mutation()` 属于
    0001 —— 本函数**一个都不碰**。

    降级之后"同一决策身份只落地一次"不再由存储层保证(回退到 v0.3.0 的
    纯 append-only 语义)。本函数只在一次性测试库上执行过。
    """
    for statement in _STATEMENTS_DOWN:
        op.execute(statement)
