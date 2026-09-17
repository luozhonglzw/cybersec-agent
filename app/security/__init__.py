"""安全层:policy / approval / audit(Phase 8)。

对应 docs/architecture.md §7 规划的 app/security/ 位置。

范围边界(刻意不做的事):
- 这里只承载"策略判定 + 审批 + 审计"这一件事,不做通用数据库抽象层。
  Phase 8 的 persistence(store.py)只服务 approval / audit / incident 三条
  生命周期;Phase 10 若引入 PostgreSQL,再重新抽象。
- 不做身份认证:Phase 8 的 actor 是调用方自称的标识,不具备不可否认性
  (见 app/schemas/audit.py 的已知局限)。
"""
