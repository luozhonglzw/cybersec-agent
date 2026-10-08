"""应用配置:所有配置项的唯一定义处。

读取优先级(后者覆盖前者):
1. 代码默认值
2. 项目根目录的 .env 文件
3. 操作系统环境变量

任何模块都不直接读 os.environ —— 全部通过 Settings。
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 审计后端取值集(Phase v0.2.0-M1c)。
#:
#: - `sqlite`   —— **默认**,与 v0.1.0 行为完全一致(本地文件库);
#: - `postgres` —— 显式 opt-in;由组合根装配 `PostgresAuditStore`。
#:
#: 只有这两个值。刻意**不提供** `auto` / `fallback` 之类的取值 ——
#: "显式选了 postgres 却在不可用时悄悄退回 sqlite" 会让审计落到另一个库,
#: 而调用方以为写进了 PostgreSQL。宁可响亮失败。
AuditBackend = Literal["sqlite", "postgres"]


class Settings(BaseSettings):
    """全局配置。

    - 进程启动时即完成校验:字段缺失/为空 → 直接 ValidationError,
      而不是运行到一半才发现配置错误;
    - llm_api_key 用 SecretStr:repr()/日志里显示 **********,
      从类型层面防止 key 泄露;
    - audit_postgres_dsn 同样用 SecretStr:连接串里带口令,任何
      repr / 日志 / 异常文本都不得回显它。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里多出的变量不影响启动
    )

    # ---- LLM(OpenAI-compatible API)----
    llm_provider: str = "openai-compatible"  # 仅作日志标签;业务代码不判断它
    llm_model: str = Field(min_length=1)     # 必填:缺失/空串 → ValidationError
    llm_base_url: str | None = None          # None = 用 SDK 默认地址
    llm_api_key: SecretStr = Field(min_length=1)

    # ---- 审计(Phase 8.4 D5)----
    # 只加这一个路径。logs_path / intel_path 刻意**不进 Settings**:
    # 它们已经由工具层持有默认值(app.tools.*.DEFAULT_DATA_PATH),测试通过
    # HitlConfig 注入 tmp_path。再放一份到 Settings 就会出现第二个真相源,
    # 且会让"测试是否 hermetic"取决于环境变量。
    audit_db_path: Path = Path("data/audit.db")

    # ---- 审计后端选择(Phase v0.2.0-M1c)----
    # 默认 sqlite ⇒ 不设任何环境变量的部署与 v0.1.0 逐字同行为。
    # 选择只发生在**组合根**(app/api/main.py 的 build_audit_store),
    # 图节点与端点里没有任何 backend 分支。
    audit_backend: AuditBackend = "sqlite"

    # PostgreSQL 连接串(仅 audit_backend="postgres" 时需要)。
    # - SecretStr:repr()/str() 都是 **********,从类型层面挡住口令外泄;
    # - 组合根取用时必须走 get_secret_value(),且**绝不**把它写进日志、
    #   异常消息或响应体(store 侧另有 redact_dsn 供诊断用);
    # - 运行期连接必须用受限角色(cybersec_app:只有 SELECT / INSERT),
    #   迁移与 DDL 由 migration-owner 角色在 Alembic 里做。
    audit_postgres_dsn: SecretStr | None = None

    @model_validator(mode="after")
    def _require_dsn_when_postgres(self) -> "Settings":
        """选了 postgres 却没给 DSN → 启动期就响亮失败。

        刻意**不**回退到 SQLite:那会让"审计写进了 PostgreSQL"变成一个
        未经核实的假设,而实际数据落在本地文件里。失败信息只说明缺哪个
        配置项,不回显任何连接串内容。
        """
        if self.audit_backend == "postgres":
            dsn = self.audit_postgres_dsn
            if dsn is None or not dsn.get_secret_value().strip():
                raise ValueError(
                    "audit_backend='postgres' 需要 AUDIT_POSTGRES_DSN;"
                    "未提供 DSN 时不会回退到 SQLite。"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """返回进程内唯一的 Settings 实例(.env 只读一次)。

    为什么缓存:Settings 创建会读 .env 并做校验,进程生命周期内配置不变,
    没必要每次调用都重读。若配置变了,重启进程即可。
    """
    return Settings()
