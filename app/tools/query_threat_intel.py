"""威胁情报精确查询工具。

分层与 query_logs.py 完全一致:
- 核心函数 query_threat_intel() 是纯 Python:读 JSONL + Exact Match,
  不依赖 LangChain,可独立测试;
- LangChain @tool 包装器 query_threat_intel_tool() 负责参数转换、
  JSON 序列化、错误归类,供 Agent / Graph 调用。

设计关键:IOC(IP / Domain / Hash)是唯一标识符,查询语义是
**精确匹配**(==),不做 contains / 模糊 / 相似度 ——
这不是 RAG 场景,引入 Embedding 只会引入误报。
"""
import json
from pathlib import Path

from app.schemas.threat_intel import ThreatIntelRecord

DEFAULT_DATA_PATH = Path("data/threat_intel.jsonl")

VALID_INDICATOR_TYPES = ("ip", "domain", "hash")


def _load_records(data_path: Path) -> list[ThreatIntelRecord]:
    """读取 JSONL 并逐行校验 —— 脏数据在此处就报错,不流进查询逻辑。"""
    records: list[ThreatIntelRecord] = []
    with open(data_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ThreatIntelRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"threat_intel 第 {line_no} 行校验失败") from exc
    return records


def query_threat_intel(
    indicator: str,
    indicator_type: str | None = None,
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> ThreatIntelRecord | None:
    """按 IOC 精确查询威胁情报。

    参数:
        indicator: IOC 原文(IP / 域名 / Hash),大小写敏感的精确匹配
        indicator_type: 可选过滤,必须是 ip / domain / hash 之一
        data_path: JSONL 数据文件路径

    返回:
        命中的 ThreatIntelRecord;不存在返回 None。

    异常:
        ValueError: indicator 为空 / indicator_type 非法 / 数据文件行损坏
        FileNotFoundError: 数据文件不存在
    """
    if not indicator or not indicator.strip():
        raise ValueError("indicator 不能为空")

    if indicator_type is not None and indicator_type not in VALID_INDICATOR_TYPES:
        raise ValueError(
            f"indicator_type 必须是 {' / '.join(VALID_INDICATOR_TYPES)} 之一"
        )

    records = _load_records(Path(data_path))
    for record in records:
        if record.indicator == indicator:
            if indicator_type is not None and record.indicator_type != indicator_type:
                return None  # indicator 相同但类型不符 → 视为未命中
            return record
    return None


def _create_tool_wrapper():
    """返回 LangChain 工具实例,职责与 query_logs 的包装器一致。"""
    from langchain_core.tools import tool

    @tool
    def query_threat_intel_tool(
        indicator: str,
        indicator_type: str | None = None,
        data_path: str = str(DEFAULT_DATA_PATH),
    ) -> str:
        """查询威胁情报库中的 IOC(Indicator of Compromise)。

        参数说明:
        - indicator: 要查询的 IOC 原文,精确匹配(如 "203.0.113.66"、
          "evil-example.com"、SHA256 hex)
        - indicator_type: 可选,IOC 类别("ip" / "domain" / "hash")
        - data_path: 数据文件路径

        返回:
        JSON:{"found": true, "record": {...}} 或
        {"found": false, "indicator": "...", "message": "No threat intelligence found"}。
        """
        try:
            record = query_threat_intel(indicator, indicator_type, data_path)
            if record is None:
                return json.dumps({
                    "found": False,
                    "indicator": indicator,
                    "message": "No threat intelligence found",
                })
            return json.dumps({
                "found": True,
                "record": record.model_dump(mode="json"),
            })
        except (ValueError, FileNotFoundError) as exc:
            return json.dumps({
                "error": str(exc),
                "type": type(exc).__name__,
                "suggest_retry": True,
            })
        except Exception:
            # 其他错误不暴露内部细节(traceback / 路径)
            return json.dumps({
                "error": "威胁情报查询失败",
                "type": "QueryExecutionError",
                "suggest_retry": True,
            })

    return query_threat_intel_tool


# 导出工具实例
query_threat_intel_tool = _create_tool_wrapper()
