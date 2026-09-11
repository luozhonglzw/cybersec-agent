"""Settings 的单元测试:不依赖 .env,不依赖真实 API Key。"""
import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


def test_settings_from_kwargs():
    """显式传参时,Settings 正确读取所有字段。"""
    s = Settings(
        llm_provider="deepseek",
        llm_model="deepseek-chat",
        llm_base_url="https://api.deepseek.com",
        llm_api_key="sk-test-123",
    )
    assert s.llm_provider == "deepseek"
    assert s.llm_model == "deepseek-chat"
    assert s.llm_base_url == "https://api.deepseek.com"
    assert isinstance(s.llm_api_key, SecretStr)
    assert s.llm_api_key.get_secret_value() == "sk-test-123"


def test_api_key_is_hidden_in_repr_and_str():
    """SecretStr 的核心价值:repr() 和 str() 都不泄露真实 key。"""
    s = Settings(llm_model="m", llm_api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(s)
    assert "sk-super-secret" not in str(s)
    assert "**********" in repr(s)


def test_missing_api_key_raises_validation_error():
    """缺少 API Key → 创建 Settings 时直接报 ValidationError(而非运行时才炸)。"""
    with pytest.raises(ValidationError):
        Settings(llm_model="m")


def test_missing_model_raises_validation_error():
    with pytest.raises(ValidationError):
        Settings(llm_api_key="sk-1")


def test_empty_api_key_raises_validation_error():
    with pytest.raises(ValidationError):
        Settings(llm_model="m", llm_api_key="")


def test_settings_from_env(monkeypatch):
    """环境变量优先级高于代码默认值(这里用不存在的 .env 隔离本地配置)。"""
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    s = Settings(_env_file=None)
    assert s.llm_model == "env-model"
    assert s.llm_api_key.get_secret_value() == "env-key"
    assert s.llm_base_url is None  # 未设置的环境变量 → 用默认值
