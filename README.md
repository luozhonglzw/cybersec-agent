# CyberSec Agent

基于 LangGraph + MCP 的智能网络安全运营（SOC）Agent 平台。

## Overview

CyberSec Agent 是一个帮助安全分析师分析安全日志、关联威胁情报（IOC / CVE / MITRE ATT&CK）、进行风险分级，并在高风险操作时要求人工确认的 SOC 智能 Agent。

设计核心：**LLM 是不可信组件**——LLM 只产生意图与文本，真正执行操作的是经过 schema 校验的工具 + 安全策略 + 权限检查 + 审计。

## Goals

- 构建一个真实可运行、可用于求职与技术面试展示的 Agent 项目
- 覆盖：LLM API、Tool Calling、ReAct、LangGraph、RAG、MCP、Human-in-the-loop、Agent Security、Observability、Evaluation
- 形成清晰的 Git History，体现渐进构建过程

## Architecture

详见 [docs/architecture.md](docs/architecture.md)。

```
Client → FastAPI → LangGraph → Tools / Knowledge / Security → SQLite / ChromaDB / JSON
```

## Current Status

- [x] Phase 0：项目架构设计（architecture.md）
- [x] Phase 1 · Step 1：项目基础 + LLM Client + 最小 SecurityAgent
- [x] Phase 1 · Step 2：FastAPI `/chat` API
- [x] Phase 2：结构化模拟安全日志
- [ ] Phase 3 ~ 10：尚未开始

### 已实现

- 配置系统：`app/core/config.py`（pydantic-settings，API Key 用 SecretStr 保护）
- LLM Client：`app/core/llm.py`（OpenAI-compatible 统一封装，可切换 DeepSeek / Qwen / OpenAI）
- 最小 SecurityAgent：`app/core/agent.py`（User → LLM → Response，单轮对话）
- FastAPI API：`app/api/main.py`（`POST /chat`，Pydantic Request/Response 模型，依赖注入，LLM 异常 → 502）
- 测试：`tests/`（完全离线，Fake LLM，不需要真实 API Key）
- 结构化安全日志（Phase 2）：`app/schemas/log_event.py`（LogEvent 模型）+ `scripts/seed_logs.py`（固定 seed 生成 8 类安全场景的模拟日志 → `data/security_events.jsonl`）

### 生成模拟日志

```bash
uv run python scripts/seed_logs.py   # 生成 data/security_events.jsonl（可重复，固定 seed=42）
```

覆盖场景：正常登录、单次失败噪声、同用户多次失败（密码猜测）、SSH 撒网式爆破、爆破 IP 后续成功登录、权限提升、Web 攻击迹象、正常业务流量。

## Quick Start

```bash
uv sync
cp .env.example .env   # 填入 LLM_MODEL / LLM_BASE_URL / LLM_API_KEY
uv run uvicorn app.api.main:app --reload
```

交互：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我分析一下最近服务器有没有受到攻击"}'
```

Swagger UI：http://127.0.0.1:8000/docs

## Roadmap

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | Architecture | ✅ 完成 |
| 1 | Minimal Security Agent（LLM API + FastAPI） | ✅ 完成 |
| 2 | Security Logs（结构化日志） | ✅ 完成 |
| 3 | Tool Calling + 手写 ReAct | Planned |
| 4 | LangGraph 迁移 | Planned |
| 5 | Threat Intelligence + RAG | Planned |
| 6 | MCP Server | Planned |
| 7 | Risk Analysis + Response Planning | Planned |
| 8 | Agent Security + HITL | Planned |
| 9 | Observability + Evaluation | Planned |
| 10 | Engineering Hardening（Docker/PG/CI） | Planned |

## Planned Features

以下功能在 Roadmap 中规划，**当前尚未实现**：

- LangGraph 工作流（Phase 4）
- 威胁情报 RAG：IOC / CVE / MITRE ATT&CK（Phase 5）
- MCP Security Tools（Phase 6）
- 风险分级 LOW / MEDIUM / HIGH / CRITICAL（Phase 7）
- 人工审批 Human-in-the-loop（Phase 8）
- 工具权限策略与审计（Phase 8）
- Agent Evaluation（Phase 9）

## Security Design

- LLM 不直接持有 Shell / 文件系统 / 网络权限
- 所有工具调用经过统一安全层：Schema 校验 → Policy → Permission → Execution → Audit
- 高危操作（如 block_ip）必须人工审批
- 全程审计日志（append-only）

## Development

环境要求：Python 3.12+（uv 可自动管理）、[uv](https://docs.astral.sh/uv/)

```bash
uv sync          # 创建虚拟环境并安装依赖（含项目本身，editable 模式）
uv run pytest -v # 运行测试（不需要真实 LLM API Key；含 API 层测试）
```

配置：复制 `.env.example` 为 `.env`，填入 `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`（`.env` 不会进入 Git）。

## License

待定。
