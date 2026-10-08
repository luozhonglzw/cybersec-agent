"""安全层:policy / approval / audit(Phase 8)。

对应 docs/architecture.md §7 规划的 app/security/ 位置。

范围边界(刻意不做的事):
- 这里只承载"策略判定 + 审批 + 审计"这一件事,不做通用数据库抽象层。
  Phase 8 的 persistence(store.py)只服务 approval / audit / incident 三条
  生命周期;Phase v0.2.0 加入可选 PostgreSQL 后端时,抽出的也只是**针对这
  三条生命周期的窄契约**(store_protocol.AuditStore,恰好 8 个方法),
  不是通用持久化层 —— 仍然没有 Repository / UnitOfWork / ORM。
- 后端选择**只发生在组合根**(app/api/main.py 的 build_audit_store):
  图节点、triage 与端点只依赖 AuditStore 契约,不含任何后端分支。
  完整边界见 docs/architecture.md §11.6。
- 不做身份认证:actor 是调用方自称的标识,不具备不可否认性
  (见 app/schemas/audit.py 的已知局限)。
"""
