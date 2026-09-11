"""应用配置:所有配置项的唯一定义处。

读取优先级(后者覆盖前者):
1. 代码默认值
2. 项目根目录的 .env 文件
3. 操作系统环境变量

任何模块都不直接读 os.environ —— 全部通过 Settings。
"""
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。

    - 进程启动时即完成校验:字段缺失/为空 → 直接 ValidationError,
      而不是运行到一半才发现配置错误;
    - llm_api_key 用 SecretStr:repr()/日志里显示 **********,
      从类型层面防止 key 泄露。
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


@lru_cache
def get_settings() -> Settings:
    """返回进程内唯一的 Settings 实例(.env 只读一次)。

    为什么缓存:Settings 创建会读 .env 并做校验,进程生命周期内配置不变,
    没必要每次调用都重读。若配置变了,重启进程即可。
    """
    return Settings()
