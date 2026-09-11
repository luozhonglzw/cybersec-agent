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
- [ ] Phase 1 ~ 10：尚未开始

> **Phase 0 completed. Implementation not started yet.**

## Roadmap

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | Architecture | ✅ 完成 |
| 1 | Minimal Security Agent（LLM API + FastAPI） | Planned |
| 2 | Security Logs（结构化日志） | Planned |
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

环境要求：Python 3.12+、uv

> Phase 1 开始后补充安装与运行步骤。

## License

待定。
