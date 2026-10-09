"""thread ownership and explicit approver assignment

Revision ID: 0002_thread_ownership
Revises: 0001_baseline_audit_schema
Create Date: 2026-10-08

Phase v0.3.0-A3-2(v0.3.0-A3-2-FIX2 修订)—— 线程归属的**单行** append-only 表。

**范围**:只建**一张**表、三条触发器与运行时角色授权。
本迁移**不**改 `0001` 的任何对象、**不**改任何既有列、**不**做数据迁移
(历史线程刻意保持"无归属")、**不**引入新的角色或权限种类。

**为什么是单行而不是"属主表 + 逐审批人行表"(A3-2-FIX2 的设计修订)**:
审批集合是**不可变的成员集合**。一张可 INSERT 的 `thread_approvers` 在
结构上就无法表达"注册之后不能再加人":任何基于 `EXISTS(owner)` 的
`BEFORE INSERT` 守卫都同时有两个无法回避的漏洞 —— 对尚未注册的线程守卫为假
(孤儿审批行可写),以及在 READ COMMITTED 下看不见他人未提交的属主行
(并发追加可写)。把集合收进**同一行的一个列**,这些漏洞一次性由结构消除:
不存在可追加的独立行、不存在孤儿形态、一条 INSERT 即原子、主键唯一性即
并发胜者。审批集合以**规范序 JSON 数组**的 TEXT 落在 `approvers` 列。

**逐条对齐 SQLite 契约**(`app/security/store.py` 的 `_schema_statements()`):
- 表名、列名、列顺序与 SQLite 完全一致,以便按列名取值的读路径原样复用;
- 时间列是 **TEXT**,存 tz-aware UTC 的 ISO8601(与 0001 同口径,刻意
  **不用** `timestamptz`,否则驱动会返回 `datetime` 而 `_from_iso()` 期待字符串);
- `approvers` 也是 **TEXT**(规范序 JSON 数组),刻意**不用** `jsonb` /
  `text[]`:读侧 `json.loads` + 严格校验是单一解析路径,两侧同构;
- **不加外键**:归属与审计表之间没有可依赖的插入顺序(线程归属在
  `/triage` 生成 thread_id 之后、`graph.ainvoke` 之前写入,而审计行在其后),
  加 FK 会把生命周期提前绑定 —— 与 0001 "不加外键" 的决定一致;
- **不加** `seq` identity 列:本表从不按序读取(只按 `thread_id` 主键读),
  因此不需要 0001 那种 tiebreaker。

**为什么 `thread_owners.thread_id` 是主键**:"同一线程被注册两次"是必须
**响亮失败**的配置错误,不是可以静默覆盖的情况。主键把它变成**不可能**,
而不是"不太可能";并发注册同一线程时,它同时保证**恰好一个**胜者。

**⚠ downgrade 的关键约束(勿改)**:
`_STATEMENTS_DOWN` 里**恰好一条** `DROP TABLE IF EXISTS`,且**绝不**能加
`DROP FUNCTION cybersec_reject_append_only_mutation()`。那个函数由 **0001**
创建并拥有,0001 自己的 downgrade 会删它。如果 0002 也删,那么"降级 0002
但保留 0001"会留下三张原表的触发器引用一个不存在的函数 —— 一个坏掉的库。
这是"照着 up-path 反向抄一遍"唯一会出错的地方,由
`tests/test_postgres/test_pg_migration.py` 的 N-18 用例钉住。
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0002_thread_ownership"
down_revision: Union[str, None] = "0001_baseline_audit_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


MIGRATOR_ROLE = "cybersec_migrator"
APP_ROLE = "cybersec_app"

#: 0001 建立的共享 append-only 函数。本迁移**复用**它,**不**重新创建,
#: 因此也**不**拥有它 —— 所有权仍归 0001。
_APPEND_ONLY_FUNCTION = "cybersec_reject_append_only_mutation()"

#: 本迁移新增的表。down-path 只允许删这一个。
_OWNERSHIP_TABLES = ("thread_owners",)

_OWNERSHIP_TABLES_SQL = ", ".join(_OWNERSHIP_TABLES)


def _append_only_triggers(table: str) -> tuple[str, ...]:
    """一张表的 UPDATE / DELETE / TRUNCATE 三条触发器(与 0001 逐字同款)。

    TRUNCATE **不触发行触发器**,所以必须另外给一条语句级 TRUNCATE 触发器:
    否则持有 DML 权限的角色(例如表 owner)可以静默清空归属记录。
    运行时角色即便没有这条触发器也做不了 TRUNCATE(无权限),两层各自独立。
    """
    return (
        f"""
        CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {_APPEND_ONLY_FUNCTION}
        """,
        f"""
        CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {_APPEND_ONLY_FUNCTION}
        """,
        f"""
        CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table}
            FOR EACH STATEMENT EXECUTE FUNCTION {_APPEND_ONLY_FUNCTION}
        """,
    )


# 每一条目 = **恰好一条** SQL 语句。逐条执行,而不是拼一个多语句字符串:
# 一来错误信息能精确指到出问题的那一条,二来避免依赖驱动的多语句行为。
_STATEMENTS_UP: tuple[str, ...] = (
    # ------------------------------------------------------------------
    # 1. thread_owners —— 每线程一行:属主 + 规范序审批集合
    # ------------------------------------------------------------------
    """
    CREATE TABLE thread_owners (
        thread_id   TEXT PRIMARY KEY,
        owner       TEXT NOT NULL,
        approvers   TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )
    """,
    # ------------------------------------------------------------------
    # 2. append-only:三条触发器,复用 0001 的函数(不新建函数)
    # ------------------------------------------------------------------
    *_append_only_triggers("thread_owners"),
    # ------------------------------------------------------------------
    # 3. 所有权:对象必须属于 migration-owner,**绝不**属于运行时角色
    # ------------------------------------------------------------------
    f"ALTER TABLE thread_owners OWNER TO {MIGRATOR_ROLE}",
    # ------------------------------------------------------------------
    # 4-6. 运行时角色授权:只有 SELECT / INSERT
    # ------------------------------------------------------------------
    # 先 REVOKE ALL 再 GRANT,保证不残留任何历史授权。
    # 刻意**不**给:UPDATE / DELETE / TRUNCATE / REFERENCES / TRIGGER /
    # CREATE(在 schema 上)。TRUNCATE 不在默认授权里,因此运行时角色
    # 无法清空归属记录 —— 这一层不依赖触发器。
    f"REVOKE ALL ON TABLE {_OWNERSHIP_TABLES_SQL} FROM PUBLIC",
    f"REVOKE ALL ON TABLE {_OWNERSHIP_TABLES_SQL} FROM {APP_ROLE}",
    f"GRANT SELECT, INSERT ON TABLE {_OWNERSHIP_TABLES_SQL} TO {APP_ROLE}",
)

#: **恰好一条**。这里**绝不**出现
#: `DROP FUNCTION cybersec_reject_append_only_mutation()` —— 详见模块
#: docstring 的 ⚠ 段。
_STATEMENTS_DOWN: tuple[str, ...] = (
    "DROP TABLE IF EXISTS thread_owners",
)


def upgrade() -> None:
    """建立线程归属单行表、append-only 保护与最小授权。"""
    for statement in _STATEMENTS_UP:
        op.execute(statement)


def downgrade() -> None:
    """**破坏性**:删除归属表及其全部内容。

    append-only 约束保护的是"运行期的改写",不是"故意的整体拆除"。
    本函数只在一次性测试库上执行过,任何真实环境都不得运行它。

    只删 0002 **自己拥有的**对象。`cybersec_reject_append_only_mutation()`
    属于 0001,本函数**不**碰它 —— 降级到 0001 之后,三张原表的
    append-only 触发器必须依然有效。
    """
    for statement in _STATEMENTS_DOWN:
        op.execute(statement)
