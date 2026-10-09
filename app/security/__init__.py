"""安全层:policy / approval / audit(Phase 8)。

对应 docs/architecture.md §7 规划的 app/security/ 位置。

范围边界(刻意不做的事):
- 这里只承载"策略判定 + 审批 + 审计"这一件事,不做通用数据库抽象层。
  Phase 8 的 persistence(store.py)只服务 approval / audit / incident 三条
  生命周期;Phase v0.2.0 加入可选 PostgreSQL 后端时,抽出的也只是**针对这
  三条生命周期的窄契约**(store_protocol.AuditStore;v0.3.0-A3-2 起为
  10 个方法 —— 新增的两个是线程归属与显式审批指派的读写),
  不是通用持久化层 —— 仍然没有 Repository / UnitOfWork / ORM。
- 后端选择**只发生在组合根**(app/api/main.py 的 build_audit_store):
  图节点、triage 与端点只依赖 AuditStore 契约,不含任何后端分支。
  完整边界见 docs/architecture.md §11.6。
- 认证与端点准入在 `app/security/auth.py`(Phase v0.3.0-A3-1):静态
  API Key → Principal(subject, role),按角色准入 4 条业务路由。
- **对象级授权与可信 actor 在 API 边界**(Phase v0.3.0-A3-3,见
  `app/api/main.py`):`/triage` 校验并登记逐线程审批指派、`/resume` 在
  任何图访问前做对象授权(未指派 404 / 属主自审批 403)、`/audit/events`
  按角色收敛读取范围。审计 `actor` 自 A3-3 起取自**已认证主体**,
  不再采信调用方自述的 `operator`。**仍未做**:同时记录
  `verified_subject` 与 `claimed_operator`、以及审计签名 —— 那在 A6。
  因此本阶段可以声称"actor 不可被客户端伪造",但**不得**声称
  "不可否认性已实现"。
"""
