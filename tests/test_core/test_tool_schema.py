"""LangChain 工具 Schema 测试。

验证工具的 schema 定义与实际函数参数的一致性，
不涉及 LLM 调用，专注于工具定义和参数验证。

测试覆盖：
- tool name 和 description
- 参数 schema 类型
- required/optional 参数  
- 参数约束（如 limit 范围）
- datetime 参数的 schema 表示
"""
import json
import inspect
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
from pydantic import ValidationError

from app.tools.query_logs import query_security_logs, _create_tool_wrapper

# 锚定仓库根:seed 脚本用相对路径会随 CWD 变化。
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def logs_data(tmp_path_factory) -> Path:
    """hermetic 日志数据:跑 seed_logs.py 生成到 tmp_path。

    本文件里凡是**真正读文件**的调用都注入它。不注入的调用只有两类:
    参数校验在读取前就抛错(limit / severity / 时间范围),以及
    刻意测试"文件不存在"的用例 —— 它们不依赖仓库 data/ 是否存在。
    """
    out = tmp_path_factory.mktemp("logs") / "security_events.jsonl"
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "seed_logs.py"), str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


class TestToolSchema:
    """测试工具定义的 schema 一致性。"""
    
    def test_tool_name_and_description(self):
        """测试工具名称和描述。"""
        tool_instance = _create_tool_wrapper()
        
        assert tool_instance.name == "query_security_logs_tool"
        
        description = tool_instance.description
        assert "安全日志" in description or "security logs" in description
        assert "查询" in description or "query" in description
    
    def test_function_parameter_consistency(self):
        """测试函数参数定义的一致性。"""
        # 获取核心函数的参数签名
        core_sig = inspect.signature(query_security_logs)
        core_params = core_sig.parameters
        
        # 获取工具函数的参数签名  
        tool_instance = _create_tool_wrapper()
        tool_func = tool_instance.func
        tool_sig = inspect.signature(tool_func)
        tool_params = tool_sig.parameters
        
        # 验证参数名称一致
        core_param_names = set(core_params.keys())
        tool_param_names = set(tool_params.keys())
        
        assert core_param_names == tool_param_names
        
        # 验证参数类型注释一致（基本类型检查）
        for param_name in core_params:
            core_param = core_params[param_name]
            tool_param = tool_params[param_name]
            
            # 核心函数应该有类型注释
            if core_param.annotation != inspect.Parameter.empty:
                # 时间参数在工具中是字符串，核心函数中是 datetime
                if param_name in ["start_time", "end_time"]:
                    assert "str" in str(tool_param.annotation).lower()
                elif param_name == "limit":
                    # limit 是整数类型
                    assert "int" in str(tool_param.annotation).lower()
                else:
                    # 其他参数是字符串类型
                    assert "str" in str(tool_param.annotation).lower()
    
    def test_core_function_parameter_validation(self, logs_data):
        """测试核心函数的参数验证逻辑。"""
        # 测试 limit 参数验证(校验在读文件之前,无需 data_path)
        with pytest.raises(ValueError, match="limit 必须是"):
            query_security_logs(limit=0)  # 低于最小值
        
        with pytest.raises(ValueError, match="limit 必须是"):
            query_security_logs(limit=201)  # 超过最大值
        
        # 测试正常值
        result = query_security_logs(limit=50, data_path=logs_data)
        assert isinstance(result, list)
        
        # 测试 severity 参数验证
        valid_severities = ["info", "low", "medium", "high", "critical"]
        for severity in valid_severities:
            result = query_security_logs(min_severity=severity, limit=1, data_path=logs_data)
            assert isinstance(result, list)
        
        # 测试无效 severity(校验在读文件之前)
        with pytest.raises(ValueError, match="min_severity 必须是"):
            query_security_logs(min_severity="invalid_severity")
        
        # 测试时间范围验证
        start_time = datetime(2026, 9, 10, 8, 0, tzinfo=None)
        end_time = datetime(2026, 9, 10, 9, 0, tzinfo=None)
        
        # 正常时间范围
        result = query_security_logs(
            start_time=start_time, end_time=end_time, limit=5, data_path=logs_data
        )
        assert isinstance(result, list)
        
        # 错误时间范围（开始晚于结束;校验在读文件之前）
        invalid_start = datetime(2026, 9, 10, 10, 0)
        invalid_end = datetime(2026, 9, 10, 9, 0)
        with pytest.raises(ValueError, match="start_time 不能晚于 end_time"):
            query_security_logs(start_time=invalid_start, end_time=invalid_end)
    
    def test_function_parameter_defaults(self):
        """测试函数参数默认值。"""
        # 获取函数签名
        sig = inspect.signature(query_security_logs)
        params = sig.parameters
        
        # 验证默认值
        assert params["limit"].default == 50  # DEFAULT_LIMIT
        assert Path(params["data_path"].default).as_posix() == "data/security_events.jsonl"  # DEFAULT_DATA_PATH
        
        # 其他参数应该默认为 None（可选）
        optional_params = ["event_type", "source_ip", "username", "start_time", "end_time", "min_severity"]
        for param in optional_params:
            assert params[param].default is None
    
    def test_event_parameter_validation(self, logs_data):
        """测试事件相关参数的验证。"""
        # 空字符串应该有效（不过滤）
        result = query_security_logs(event_type="", limit=1, data_path=logs_data)
        assert isinstance(result, list)
        
        # 特定 event_type 应该有效
        result = query_security_logs(event_type="login_success", limit=1, data_path=logs_data)
        assert isinstance(result, list)
        
        # IP 地址验证（通过字符串传递）
        result = query_security_logs(source_ip="203.0.113.66", limit=1, data_path=logs_data)
        assert isinstance(result, list)


class TestToolFunctionConsistency:
    """测试工具函数与包装器的一致性。"""
    
    def test_core_function_exists(self, logs_data):
        """测试核心函数存在且可调用。"""
        assert callable(query_security_logs)
        
        # 测试基本调用
        result = query_security_logs(limit=5, data_path=logs_data)
        assert isinstance(result, list)
    
    def test_tool_wrapper_functionality(self, logs_data):
        """测试工具包装器的功能。"""
        tool_instance = _create_tool_wrapper()
        
        # 使用正确的LangChain工具调用方式
        result = tool_instance.invoke({
            "event_type": "login_failed", "limit": 10, "data_path": str(logs_data)
        })
        assert isinstance(result, str)
        
        # 解析 JSON 并验证结构
        parsed_result = json.loads(result)
        assert "count" in parsed_result
        assert "events" in parsed_result
        assert isinstance(parsed_result["count"], int)
        assert isinstance(parsed_result["events"], list)
    
    def test_error_handling_consistency(self):
        """测试错误处理的一致性。"""
        # 测试文件不存在的错误(刻意用不存在的路径)
        with pytest.raises(FileNotFoundError):
            query_security_logs(data_path="nonexistent_file.jsonl")
        
        # 工具包装器应该返回结构化的错误信息
        tool_instance = _create_tool_wrapper()
        tool_result = tool_instance.invoke({"data_path": "nonexistent_file.jsonl"})
        error_info = json.loads(tool_result)
        
        assert "error" in error_info
        assert "type" in error_info
        assert error_info["type"] == "FileNotFoundError"
    
    def test_result_structure_consistency(self, logs_data):
        """测试结果结构的一致性。"""
        # 使用核心函数
        core_result = query_security_logs(limit=5, data_path=logs_data)
        
        # 使用工具包装器
        tool_instance = _create_tool_wrapper()
        tool_result = tool_instance.invoke({"limit": 5, "data_path": str(logs_data)})
        parsed_tool_result = json.loads(tool_result)
        
        # 验证结果结构一致
        assert len(parsed_tool_result["events"]) == len(core_result)
        assert parsed_tool_result["count"] == len(core_result)
        
        # 验证事件结构
        for event in parsed_tool_result["events"]:
            assert isinstance(event, dict)
            assert "timestamp" in event
            assert "event_type" in event


class TestToolIntegration:
    """测试工具集成相关功能。"""
    
    def test_tool_wrapper_callable(self):
        """测试工具包装器可调用。"""
        tool_instance = _create_tool_wrapper()
        
        # 工具实例应该是可调用的
        assert callable(tool_instance)
        
        # 测试各种参数组合
        test_cases = [
            {},  # 无参数
            {"event_type": "login_failed"},
            {"source_ip": "203.0.113.66"},
            {"limit": 20},
            {"min_severity": "high"},
            {"event_type": "login_success", "limit": 5, "min_severity": "medium"}
        ]
        
        for test_args in test_cases:
            result = tool_instance.invoke(test_args)
            assert isinstance(result, str)
            
            # 验证 JSON 格式正确
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                pytest.fail(f"工具返回的不是有效的 JSON: {result}")
            
            # 如果没有错误，应该包含 count 和 events
            if "error" not in parsed:
                assert "count" in parsed
                assert "events" in parsed
    
    def test_tool_error_responses(self):
        """测试工具的错误响应格式。"""
        tool_instance = _create_tool_wrapper()
        
        # 测试参数错误(工具包装器捕获错误并返回 JSON,不抛异常)
        result = tool_instance.invoke({"limit": 0})  # 无效 limit
        error_info = json.loads(result)
        assert "error" in error_info
        assert error_info["type"] == "ValueError"
        assert "suggest_retry" in error_info
        
        # 测试文件不存在错误（工具包装器会捕获并返回 JSON）
        result = tool_instance.invoke({"data_path": "nonexistent_file.jsonl"})
        error_info = json.loads(result)
        
        assert "error" in error_info
        assert "type" in error_info
        assert error_info["type"] == "FileNotFoundError"
        assert "suggest_retry" in error_info
    
    def test_datetime_parameter_conversion(self, logs_data):
        """测试时间参数转换。"""
        tool_instance = _create_tool_wrapper()
        
        # 测试有效的 ISO 格式字符串
        valid_iso = "2026-09-10T08:00:00Z"
        result = tool_instance.invoke({
            "start_time": valid_iso, "limit": 1, "data_path": str(logs_data)
        })
        parsed_result = json.loads(result)
        
        # 应该成功执行
        assert "error" not in parsed_result
        assert "count" in parsed_result
        
        # 测试无效的时间格式(工具包装器捕获并返回 JSON 错误;转换在读文件之前)
        result = tool_instance.invoke({"start_time": "invalid-date"})
        error_info = json.loads(result)
        assert "error" in error_info
    
    def test_severity_parameter_enum(self, logs_data):
        """测试严重程度参数的枚举值。"""
        tool_instance = _create_tool_wrapper()
        
        valid_severities = ["info", "low", "medium", "high", "critical"]
        
        for severity in valid_severities:
            result = tool_instance.invoke({
                "min_severity": severity, "limit": 1, "data_path": str(logs_data)
            })
            parsed_result = json.loads(result)
            
            # 应该成功执行
            assert "error" not in parsed_result
            assert "count" in parsed_result