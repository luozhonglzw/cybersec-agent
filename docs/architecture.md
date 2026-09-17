# CyberSec Agent 架构设计（Phase 0）

> 状态：已确认 | 日期：2026-09-11
> 对应里程碑：`chore: initialize CyberSec Agent project`
> 本文档是全部 10 个 Phase 的设计蓝图。实现细节随 Phase 推进沉淀到 `docs/learning/`，面试问答沉淀到 `docs/interview/`。

---

## 1. 项目定位

**CyberSec Agent —— 基于 LangGraph + MCP 的智能网络安全运营 Agent 平台。**

一个**单进程可运行、自带模拟数据、所有危险操作必须人工审批**的 SOC 智能分析助手：

```
自然语言请求
  → Agent（意图理解与决策）
  → Security Tools / Threat Intelligence（获取证据）
  → Risk Analysis（风险分级）
  → Response Planning（处置建议）
  → Human Approval（高危动作人工审批）
  → Audit（全程审计）
```

项目双重目标：

1. **求职展示**：一个真实、可运行、可解释的 Agent 项目。
2. **学习载体**：每个 Phase 对应一组可面试解释的概念（见 §12 Roadmap）。

## 2. 核心设计原则

### Principle 1：LLM 是不可信组件（Untrusted Component）

LLM 只能产生：

- 意图（Intent）
- 文本（Text）
- 结构化工具请求（Structured Tool Request）

真正执行操作的是：

```
经过 schema 校验的 Tool
  + Security Policy（策略引擎）
  + Permission Check（权限检查）
```

推论：

- LLM 永不直接接触 Shell / 文件系统 / 网络。禁止 `os.system(llm_output)` 类架构。
- 所有工具调用必须穿过同一个安全层；安全层是横切组件，不是某个 Phase 才打的补丁。
- 日志、网页、工具返回等一切外部内容都可能是 Prompt Injection 载体；LLM 的输出永远需要校验。

### Principle 2：简单优先（YAGNI）

- 不为"看起来企业级"提前引入 Repository / Factory / DI / 消息队列 / 微服务。
- 只有当真实痛点出现时才引入新模式（如 Phase 10 才上 PostgreSQL）。
- 第一版代码必须：简单、清晰、可运行、易 Debug。

## 3. 需求分析

### 3.1 用户角色

| 角色 | 典型诉求 |
|---|---|
| SOC 一级分析师（模拟） | "今天凌晨大量来自 203.0.113.x 的登录失败，帮我判断是不是暴力破解" |
| 威胁情报研究员（模拟） | "这个 hash 有没有关联 CVE / ATT&CK 技术" |
| 平台管理员 | 审批高危操作（如 block_ip）、查看审计日志 |
| 项目开发者 / 学习者（真实） | 每个 Phase 结束能独立回答对应的面试问题 |

### 3.2 核心功能

- **F1 对话式安全分析**：自然语言 → 有依据、可追溯的结构化结论
- **F2 日志查询与分析**：结构化模拟日志 + Tool Calling（query_logs）
- **F3 威胁情报检索**：IOC（IP/域名/hash）+ CVE + MITRE ATT&CK，RAG 语义检索
- **F4 风险分级**：LOW / MEDIUM / HIGH / CRITICAL + 判定依据
- **F5 响应规划**：处置建议；高危动作 → 人工审批（HITL）
- **F6 全链路审计**：每次工具调用、每个审批决策可查
- **F7 可观测性**：token 用量、延迟、每步 trace
- **F8 Agent Evaluation**：用指标证明"这是 Agent，不是套壳聊天机器人"

### 3.3 非功能需求

- 安全：最小权限、工具白名单、输入/输出校验、无 Shell 直通
- 可解释：每一步决策可回放（checkpoint）
- 可运行：一条命令启动，模拟数据内置，不依赖外部网络
- 成本可控：LLM 可切换（OpenAI 兼容协议），token 可计量

## 4. Non-Goals（明确不做）

- 不做真实网络攻击
- 不做真实第三方扫描
- 不对接生产 SIEM（只留接口）
- 不做多租户
- 不做微服务
- 不做 Kubernetes
- Phase 10 之前不引入 Docker / PostgreSQL / Redis

**实验环境边界**：所有扫描、命令执行、网络操作默认限制在本地实验环境、Docker Lab 或明确授权的测试环境。项目数据全部为模拟生成（scripts/seed_*.py），不针对任何真实第三方系统。

## 5. 系统架构

```
┌──────────────────────────────────────────────┐
│ 客户端：curl / Swagger UI / scripts/demo.py │
└───────────────────┬──────────────────────────┘
┌───────────────────▼──────────────────────────┐
│ API 层（app/api）  FastAPI                   │
│ /chat /approve /audit 路由、输入校验          │
└───────────────────┬──────────────────────────┘
┌───────────────────▼──────────────────────────┐
│ 编排层（app/graph）  LangGraph               │
│ State + Nodes + Conditional Edges            │
│ + Checkpointer（断点恢复/回放）               │
│ + Interrupt（人工审批暂停/恢复）              │
└───┬──────────────┬──────────────┬────────────┘
    │              │              │
┌───▼─────┐  ┌─────▼──────┐  ┌────▼───────────┐
│ 工具层   │  │ 知识层      │  │ 安全层（横切） │
│ tools/  │  │ rag/        │  │ security/     │
│ query_  │  │ knowledge/  │  │ 策略引擎       │
│ logs    │  │ 向量库+检索 │  │ 人工审批       │
│ search_ │  │            │  │ 审计日志       │
│ ioc ... │  │            │  │ 输入/输出校验   │
└───┬─────┘  └──────┬─────┘  └────┬───────────┘
┌───▼───────────────▼─────────────▼───────────┐
│ 数据层：SQLite（结构化+checkpoint+审计）      │
│         ChromaDB（向量）  data/（JSON 模拟数据）│
└─────────────────────────────────────────────┘
```

> 注：Phase 4 完成后，编排层实际实现为单文件 `app/core/graph.py`，未单独创建 `app/graph/` 包；`rag/` / `knowledge/` / `security/` / `evaluation/` 仍待对应 Phase 创建。上方框图与 §7 目录描述的是**终态蓝图**。

### 为什么是单进程 + FastAPI + SQLite + ChromaDB？

| 选择 | 原因 |
|---|---|
| 单进程部署 | 求职项目要求"一条命令跑起来"；微服务在此规模收益为零、成本是实的（网络调用、一致性、部署复杂度） |
| FastAPI | 类型驱动、自带 Swagger（=免费的产品 UI）、原生 async |
| SQLite | 零运维、单文件，够用到 Phase 9；Phase 10 遇到真实痛点再换 PostgreSQL |
| ChromaDB（embedded） | 持久化、元数据过滤、pip 即用；不引入 Docker 依赖 |

分层是**逻辑边界**，不是物理边界——边界清晰，Phase 10 想拆随时能拆。

## 6. 技术栈

> **Current**：Phase 0-6 所需依赖均已安装（实际版本以 `pyproject.toml` / `uv.lock` 为准）。下表"引入 Phase"记录该依赖首次加入的里程碑；Phase 7-10 的依赖仍为 Planned。

### 代码依赖

| 组件 | 用途 | 引入 Phase |
|---|---|---|
| Python 3.12 | 语言 | 1 |
| uv | 包管理与环境管理 | 1 |
| FastAPI + uvicorn | API 层 | 1 |
| langchain-openai | LLM 接入（OpenAI 兼容协议，provider 可切换） | 1 |
| Pydantic v2 | Schema 校验 | 1 |
| pydantic-settings | 配置管理（.env） | 1 |
| structlog | 结构化日志 | 1 |
| pytest / pytest-asyncio | 测试 | 1 |
| respx | Mock LLM 的 HTTP 调用 | 1 |
| LangGraph | Agent 编排（State/Node/Edge/Checkpoint/Interrupt） | 4 |
| ChromaDB | 向量库（RAG） | 5 |
| mcp（FastMCP） | MCP Server | 6 |
| OpenTelemetry | Trace | 9 |
| Langfuse | 可观测平台 | 9 |
| Docker / PostgreSQL / Redis | 工程化 | 10（可选，按需引入） |

### 安全领域知识（知识库内容，非代码依赖）

| 内容 | 用途 | 引入 Phase |
|---|---|---|
| MITRE ATT&CK | 战术/技术框架，知识库与关联分析 | 5 |
| CVE / NVD | 漏洞库，知识库 | 5 |
| STIX / TAXII | 威胁情报交换标准，作为 IOC 数据结构的参考 | 5 |
| Sigma | 检测规则格式，知识库扩展 | 10（可选） |
| YARA | 恶意样本规则，知识库扩展 | 10（可选） |

### LLM 接入

OpenAI 兼容协议（`LLM_PROVIDER / LLM_MODEL / LLM_BASE_URL / LLM_API_KEY` 来自 .env），一套代码可跑 DeepSeek / Qwen / OpenAI / 本地 Ollama。具体 provider 待定（见 §13）。

## 7. Repository 结构

```
cybersec-agent/
│
├── app/
│   ├── core/                    # Phase 1：config、logging、llm 客户端
│   ├── schemas/                 # Phase 2：LogEvent、IOC、CVE、RiskAssessment、AuditEvent...
│   ├── tools/                   # Phase 3：工具 = 描述 + 参数 schema + 实现，三者分离
│   ├── graph/                   # Phase 4：蓝图位置；实际实现为 app/core/graph.py
│   ├── security/                # Phase 8：policy、approval、audit（概念从 Phase 1 就存在）
│   ├── rag/                     # Phase 5：embedding、retriever、vector store
│   ├── knowledge/               # Phase 5：知识库加载器（data → SQLite + ChromaDB）
│   ├── api/                     # Phase 1：FastAPI 路由
│   └── evaluation/              # Phase 9
│
├── data/                        # 模拟日志、知识库 JSON；SQLite/Chroma 持久化文件（gitignore）
├── mcp_server/                  # Phase 6：MCP Server 入口（复用 app.tools + app.security）
├── scripts/                     # seed 数据生成、demo、evaluation 运行
├── tests/
├── docs/
│   ├── architecture.md          # 本文档
│   ├── learning/                # 每 phase 学习笔记（学到了什么/为什么这么设计/坑/面试怎么答）
│   └── interview/               # 面试问答（agent/langgraph/mcp/rag/security...）
├── docker/                      # Phase 10
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

规则：

- **目录随 Phase 创建**。Phase 0 只创建实际需要的文件（.gitignore / .env.example / README.md / docs/architecture.md），不创建空目录。
- **data/ 的 Git 策略**：git 只存生成脚本（scripts/seed_*.py），不存生成物；换台机器 `python scripts/seed.py` 一键复原。
- mcp_server/ 放顶层：MCP Server 是独立进程入口，生命周期与 API 服务不同。
- schemas/ 独立成包：Pydantic 模型被 tools、graph、api、knowledge 同时引用，单独放置避免循环 import。

## 8. 数据流设计

### Flow A — 对话分析（热路径，每次请求）

```
POST /chat {"message": "..."}
  → api 层：Pydantic 校验输入
  → graph.invoke(initial_state)
  → [节点] LLM 分析：决定"直接回答 / 查日志 / 查情报"
  → [条件边] 需要工具？
      ├─ 是 → 工具节点 → security 层：白名单 → 权限 → 审计 → 执行 → 结果写回 state
      │         └─ 回到 LLM 节点（这就是 ReAct 循环）
      └─ 否 → [节点] 风险分析 → 风险等级 + 依据
  → [节点] 响应规划 → 处置建议
  → [条件边] 含高危动作？
      ├─ 是 → human_approval 节点 → interrupt：图暂停，返回 pending 状态
      │         → 人通过 /approve 或 /deny → 图恢复 → 执行（或拒绝）→ 审计
      └─ 否 → final_response
  → 返回 {answer, risk, actions, trace}
```

### Flow B — 知识摄取（离线，seed 脚本或启动时）

```
data/knowledge/*.json（ATT&CK 子集、CVE 样例、IOC、威胁报告）
  → knowledge 加载器：解析 + Pydantic 校验 + 去重
  → SQLite（结构化，供精确查询）
  → 分块（chunk）→ embedding → ChromaDB（语义检索，metadata 保留来源）
```

### Flow C — MCP 工具调用（Phase 6）

```
外部 MCP Client（Claude Desktop / MCP Inspector / 我们自己的 Agent）
  → 我们写的 MCP Server（FastMCP）
  → 复用 app/tools 的工具实现
  → 复用 app/security 的同一套策略与审计   ← 关键：安全层只有一个入口
  → 返回结果
```

Flow C 的存在意义：Function Calling 是**进程内、厂商私有**的工具调用协议；MCP 是**跨进程、跨客户端**的标准协议。使用 MCP 的真实需求是"让第三方客户端也能安全地复用我们的安全工具"，而不是为简历硬加。

## 9. LangGraph Workflow 设计

### 9.1 图结构

```
START
  ↓
analyze_request（LLM：理解意图，决定下一步）
  ↓ ──条件边──
  │   需要工具？ ─是→ tool_router（按工具名路由）
  │                    ↓
  │                  tool_execute（安全层包裹：校验→权限→执行→审计）
  │                    ↓
  │              ──循环回 analyze_request（ReAct：用新证据再思考）
  └──否→ risk_analyze（风险分级 LOW/MEDIUM/HIGH/CRITICAL）
           ↓
         response_plan（生成处置建议）
           ↓ ──条件边──
           │  含高危动作？ ─是→ human_approval（interrupt！图暂停）
           │                    ↓  等待 /approve 或 /deny
           │                  execute_or_reject（批准→执行+审计；拒绝→记录）
           │                    ↓
           └──否────────────→ final_response
                                ↓
                               END
```

### 9.2 教学设计：为什么 Phase 3 先手写 ReAct，Phase 4 再迁移 LangGraph？

这是**故意的绕路**：

- Phase 3 手写 while 循环实现 ReAct，亲身体会三个痛点：循环逻辑与业务逻辑混在一起、调试要到处埋 print、无法中途暂停等待人工。
- Phase 4 迁移 LangGraph，才能理解它解决什么：State 显式化、节点职责单一、Checkpoint 免费获得（崩溃恢复/回放/审计）、Interrupt 原生支持 HITL、图可打印可单测。
- **先体会痛，再理解药**。这段演进过程也直接体现在 Git History 里，是面试最有说服力的材料。

代价也要承认：引入框架 = 多一层抽象、多一层调试栈。所以不是"越复杂越好"，而是"复杂度换来了可回放、可中断、可审计"。

### 9.3 为什么用 Graph 而不是 while 循环？

1. **循环逻辑和业务逻辑分离**：Graph 中每个节点职责单一，图本身就是流程图。
2. **免费获得 checkpoint**：每个节点落状态快照 → 崩溃恢复、中断续跑、事后回放审计。
3. **interrupt 是原生能力**：HITL 需要"跑到一半停住、人介入后继续"，手写循环得自己发明暂停/恢复协议。
4. **可测试、可打印**：节点可单测；`graph.get_graph().draw_mermaid()` 直接生成架构图。

## 10. State 设计

计划中的 `AgentState`（Phase 4 落实，实现时可能调整字段）：

```python
class AgentState(TypedDict):
    messages: list                      # 对话历史（LLM 上下文）
    tool_results: list                  # 本轮工具返回（结构化对象）
    retrieved_docs: list                # RAG 命中（带来源和分数）
    risk: RiskAssessment | None         # 风险等级 + 依据（Pydantic）
    plan: ResponsePlan | None           # 处置建议
    pending_approval: ActionRequest | None  # 等待人工的请求
    audit_entries: list                 # 本轮审计流水
```

**为什么用结构化 State，而不是把所有内容拼成一个字符串？**

- 每个节点只读写自己关心的字段，职责清晰，不会互相踩踏；
- 工具结果保留结构化对象（而非字符串拼贴），后续节点可以程序化判断（例如 risk_analyze 直接读 tool_results 里的失败登录计数）；
- checkpoint 回放时，每一步的状态都是可读、可验证的。

## 11. 数据库设计初稿

原则：**表随 Phase 增长，Phase 0 只定蓝图；审计类数据 append-only。**

| 表 | 用途 | 关键字段 | 引入 Phase |
|---|---|---|---|
| security_events | 模拟安全日志 | ts, src_ip, dst_ip, username, action, status, user_agent, raw | 2 |
| threat_intel | IOC 库 | ioc_type(ip/domain/hash), ioc_value, source, confidence, tags | 5 |
| knowledge_docs | RAG 源文档 | kind(cve/mitre/report), key, title, text, metadata | 5 |
| incidents | 分析结论沉淀 | summary, risk_level, linked_iocs, resolution | 8 |
| action_requests | HITL 审批单 | tool, params, risk, status(pending/approved/denied/executed), requester, approver, 各时间戳 | 8 |
| audit_logs | 审计流水 | ts, actor, event, resource, detail, outcome | 8（概念始于 1） |
| checkpoints | LangGraph 断点 | 框架自动管理 | 4 |

ChromaDB collections：`mitre_techniques` / `cve_entries` / `threat_reports`，每个 chunk 带 metadata（kind, id, tags）支持过滤检索。

设计细节：

- `audit_logs` 与 `action_requests` **只 INSERT 不 UPDATE**——状态变化 = 追加新记录（approved 不是把 pending 改掉，而是追加一条 decision 记录）。可变的"历史"不叫审计。
- `security_events` 保留 `raw` 原始字段——分析可能出错，原始数据永远可回溯。
- **incident 持久化延后至 Phase 8**：原计划 Phase 7 引入 `incidents` 表，实际 Phase 7 只交付 `ResponsePlan` 结构化契约与规则引擎，计划随 ToolMessage 流转、不落库。incident persistence 与 HITL / checkpoint / audit lifecycle 属同一条状态生命周期，拆开实现会产生两套状态语义，故统一延后到 Phase 8 一次性落地。

## 12. Phase Roadmap

| Phase | 交付物 | 核心概念（面试可讲） | 退出标准（能回答） | 最终 Commit |
|---|---|---|---|---|
| 0 | 本设计 + 仓库骨架 | 需求/架构/选型 | "为什么这个项目长这样" | chore: initialize CyberSec Agent project |
| 1 | 最小 FastAPI + LLM 对话 | Message/System Prompt/Token/async | "Agent 和普通 LLM 有什么区别" | feat: add basic security agent |
| 2 | 模拟日志 + 结构化分析 | Pydantic/结构化日志/Prompt | "如何从登录日志判断攻击" | feat: add structured security logs |
| 3 | query_logs 工具 + 手写 ReAct | Tool schema/Function Calling 全链路 | "LLM 是怎么调用工具的" | feat: add security log tool calling |
| 4 | 迁移 LangGraph | State/Node/Edge/Conditional/Checkpoint | "为什么不用 while 循环" | feat: introduce LangGraph workflow |
| 5 | 威胁情报 + RAG | Embedding/Chunk/Top-K/相似度 | "为什么威胁情报适合 RAG" | feat: add threat intelligence RAG |
| 6 | MCP Server | MCP 协议/Tool/Resource | "MCP 和 Function Calling 区别" | feat: add MCP security tools |
| 7 | 风险分析 + 响应规划 | 风险分级模型 | "什么操作应该人工确认" | feat: add risk analysis and response planning |
| 8 | 安全层 + HITL | Prompt Injection/最小权限/审批流 | "日志里有 Prompt Injection 怎么办" | security: add human approval and tool policies |
| 9 | 可观测 + 评估 | Trace/指标/LLM-as-Judge | "你怎么证明 Agent 比 Chatbot 好" | feat: add agent evaluation and observability |
| 10 | 工程化 | Docker/PG/Retry/CI/文档 | "10 万次运行怎么观察控制成本" | chore: production hardening |

依赖关系：Phase 3 → 4 → 8 是硬依赖（顺序不能乱）；Phase 5 与 6 可互换；评估集（golden set）从 Phase 2 起开始积累。

## 13. 待定决策

| # | 事项 | 说明 |
|---|---|---|
| 1 | LLM provider | DeepSeek（推荐，便宜、OpenAI 兼容）/ Qwen / OpenAI / 本地 Ollama；代码不变，只改 .env |
| 2 | GitHub 仓库名与可见性 | 已定：`cybersec-agent`（private，求职展示时可改 public） |
| 3 | License | 待定（Phase 10 前确定） |

## 14. 实现进度（随开发更新）

> 2026-09-17 · **Phase 0-7 全部完成**（Phase 8 未启动）。最新状态见文末"当前架构快照"。
> 下方按 Phase 顺序记录各阶段的交付物与设计决策。

历史快照（Phase 1-3 时期的调用链，已被 LangGraph 版取代，见文末）：

```
HTTP Client
 ↓
FastAPI（app/api/main.py：POST /chat，Pydantic 校验，依赖注入 Agent，LLM 错误 → 502）
 ↓
SecurityAgent（app/core/agent.py：当时为手写 ReAct 循环，Phase 4 起委托 LangGraph）
 ↓
LLMClient（app/core/llm.py：ChatOpenAI 统一封装，provider 由 .env 决定）
 ↓
OpenAI-compatible LLM
```

- Step 2（FastAPI API Layer，已完成）：
  - `app/api/schemas.py`：ChatRequest / ChatResponse（Pydantic 契约，与 core 层解耦）
  - `app/api/main.py`：`create_app()` 工厂 + lifespan 创建真实 Agent；测试通过 `create_app(agent=SecurityAgent(FakeLLMClient))` 注入假依赖
  - 错误处理：请求校验失败 → 422（Pydantic/FastAPI 自动）；LLM 初始化/调用失败（LLMClientError）→ 502，不泄露内部细节
  - 测试：`tests/test_api/test_chat.py`（正常 / 非法请求 / LLM 异常 / 消息透传，全部离线）

> 2026-09-14 · Phase 2 完成（结构化模拟安全日志）

- `app/schemas/log_event.py`：LogEvent 模型。字段：timestamp / event_type(Literal) / source / source_ip / destination_ip / source_port / destination_port / username / action / status(Literal) / severity(Literal，与 Phase 7 风险分级同一套词汇表) / message。
- 校验边界：IP 用 `str` + Pydantic validator 校验 IPv4（JSONL / 工具参数 / LLM 数据交换统一字符串形态）；port 校验 0~65535；可枚举字段全部 Literal，脏数据在校验边界被拒绝。
- 设计要点：**"SSH 暴力破解"不是 event_type，而是大量 login_failed 事件构成的模式**——数据层只记录原子事实，模式识别是 Agent（Phase 3+）的工作。
- `scripts/seed_logs.py`：固定 `random.Random(42)` + 固定基准时间（2026-09-10 08:00 UTC），输出逐字节可复现 → `data/security_events.jsonl`（144 条事件；data/ 生成物不进 git，见 §7 策略）。
- 8 个安全场景：正常登录 / 单次失败噪声 / 同用户多次失败（密码猜测）/ SSH 撒网式爆破 / 爆破 IP 后续成功登录 / 权限提升（sudo 失败→加入 sudo 组）/ Web 攻击迹象 / 正常防火墙流量。场景 3/4/5 构成递进攻击故事线，为 Phase 3 的多步推理（查失败→按 IP 聚合→查是否成功登录）准备真实问题。
- 测试：`tests/test_schemas/`（模型校验拒绝非法 IP/event_type/severity/status/port；生成可重复；JSONL 逐行 roundtrip；8 场景真实存在）。

> 2026-09-14 · Phase 3 Step 1 完成（安全日志查询工具）

- `app/tools/query_logs.py`：query_security_logs 工具，支持按 event_type、source_ip、username、start_time、end_time、min_severity、limit 等条件过滤，返回结构化 LogEvent 列表
- `scripts/seed_logs.py`：生成 144 条结构化安全日志，覆盖 8 个安全场景
- `tests/test_tools/test_query_logs.py`：20 个测试用例，覆盖各种查询条件和错误处理

> 2026-09-14 · Phase 3 Step 2 完成（Tool Calling + 手写 ReAct 循环）

- `app/core/agent.py`：SecurityAgent 支持 Tool Calling 和手写 ReAct 循环，默认 max_iterations=5
- `app/tools/query_logs.py`：添加 LangChain 工具包装器，保持核心查询逻辑独立
- `tests/test_core/test_react_agent.py`：完整的 ReAct 循环测试套件
- `app/core/llm.py`：FakeLLMClient 和 FakeChatModel 工具调用模拟框架
- System Prompt 增强：强调事实优先、证据驱动、不足证据时明确说明

> 2026-09-17 · Phase 3 Step 2 Review Fix 完成（测试完善 + 缺陷修复）

- 独立 LangChain Tool Schema 测试（`tests/test_core/test_tool_schema.py`）：参数一致性、错误响应契约、真实 `.invoke()` 路径
- 缺陷修复：工具包装器 JSON 序列化（`model_dump(mode="json")` 修复 datetime 不可序列化导致全部查询返回通用错误）；异常日志只记录 `error_type`/`tool_call_id`，不暴露参数与路径
- 77 tests（Phase 3 终态基线）

> 2026-09-17 · Phase 4 完成（LangGraph 迁移）

分四步实施（4.1 依赖与接口 → 4.2 graph 构建 → 4.3 接入 → 4.3-B 执行轨迹验证）：

- **4.1**：`langgraph` 依赖；`LLMClient.bind_tools()` 公开接口——API Key 注入边界收敛在 LLMClient 一处，调用方不再触碰内部模型
- **4.2**：新建 `app/core/graph.py`，图结构：
  ```
  START → agent → [should_continue] → tools → agent（循环）
                       ↓（无 tool_calls 或达 max_iterations）
                      END
  ```
  - `AgentState`：仅 `messages`（`add_messages` reducer 追加）+ `iteration_count`（业务层迭代上限）
  - agent / tools 节点手写（未用 ToolNode/create_react_agent）；工具执行严格 `tool.ainvoke(tc["args"])` + `ToolMessage(tool_call_id=tc["id"])`，修复了旧循环传整个 tool_call dict 的 bug
  - 业务层 `max_iterations` 与 LangGraph `recursion_limit` 分离，前者由 `should_continue` 控制
- **4.3**：`SecurityAgent.chat()` 切换为构造 `graph.ainvoke()` 委托（签名不变，API 零改动）；删除手写 ReAct for loop；达到迭代上限返回 Phase 3 相同的受限说明
- **4.3-B**：`astream(stream_mode="updates")` 原生执行轨迹测试——逐节点验证 agent→tools→agent 顺序、iteration_count 递增、tool_call_id 传递
- 测试覆盖：graph 构建/路由/多工具循环/tool_call_id/异常安全契约/消息顺序（`test_graph.py`）

> 2026-09-17 · Phase 5 完成（威胁情报 + Evidence Fusion）

- **5.1 数据层与工具**：
  - `app/schemas/threat_intel.py`：ThreatIntelRecord（indicator / indicator_type(ip/domain/hash) / malicious / confidence 0-100 / severity / tags / source / first_seen / last_seen）
  - `scripts/seed_threat_intel.py`：固定 seed + 固定时间，29 条 IOC（恶意 IP 11 / 域名 8 / Hash 6 / 可信对照 4），与 Phase 2 攻击场景对应（203.0.113.66）
  - `app/tools/query_threat_intel.py`：**Exact Match 查询**（见下方 Roadmap 偏离说明）+ @tool wrapper（found/not-found/error 三种 JSON 契约）
  - 跨数据集关联测试：Phase 2 日志中的攻击 IP 可在情报库精确命中（Evidence Fusion 的数据地基）
- **5.2 多工具串联**：SecurityAgent 默认注册双工具；prompt 增加融合指导（发现 IOC → 可查情报 → 综合证据并说明来源）；messages 是唯一证据容器，**未新增 State 字段**——日志 ToolMessage 与情报 ToolMessage 按 reducer 顺序自然进入 LLM 上下文
- `tests/test_core/test_evidence_fusion.py`：日志→情报→融合回答的完整链路验证

> 2026-09-17 · Phase 6 完成（Rule-based Risk Analyzer，Hybrid 架构）

- `app/schemas/risk.py`：RiskEvidence（证据快照：事件计数/失败登录数/情报命中与标签）+ RiskAssessment（risk_level 含 none / score 0-100 / confidence / reasons 逐条引用证据 / 内嵌可复现 evidence）
- `app/tools/risk_analyzer.py`：
  - `analyze_risk(evidence)` **纯函数规则引擎**：不读文件不查库，同样证据永远同样结果；权重表（恶意情报 +40、severity≥high +15、失败登录≥20 +30 / 5-19 +15、可信情报强制 ≤10 误报抑制），分数映射 none/low/medium/high/critical
  - `collect_evidence()` 便利采集器 + @tool wrapper；采集与分析分离，未来可支持直接传 Evidence
- **Hybrid 分工**：Rule-based Tool 输出可审计的结构化等级与分数；LLM 拿到 ToolMessage 后负责解释与汇报——数字与等级不经过 LLM
- SecurityAgent 默认注册三工具；prompt 第 9 条：优先用风险工具的结构化结果，LLM 职责是解释
- 159 tests（Phase 6 终态基线）

> 2026-09-17 · Phase 7 完成（Incident Response Planning，规则引擎侧）

- `app/schemas/response.py`：ResponseAction（action_type / priority / target / rationale / requires_approval / reversible）+ ResponsePlan（indicator / risk_level / summary / actions / **内嵌 assessment**）——plan → assessment → evidence 全链可追溯，Phase 9 golden set 可直接比对
- `app/tools/response_planner.py`：
  - `plan_response(assessment)` **纯函数规则引擎**：只消费 RiskAssessment，**不重新推导风险**——误报抑制等判定已在 risk_analyzer 完成，此处复用其结果而非重复实现，避免两个真相源
  - risk_level → 基础动作集，外加两条**修正项**：情报标记恶意 → 补 `block_ip`；失败登录 ≥ 20 → 补 `reset_credentials`。修正项不是冗余——仅情报恶意（40 分）与 20 次失败登录（30 分）都只落在 medium 档，而 medium 的基础动作集不含这两个动作
  - 动作属性表：破坏性动作（block_ip / isolate_host / reset_credentials）强制 `requires_approval=True`；`reset_credentials.reversible=False`（已改密不可逆）——**审批标记由规则引擎给出，不经过 LLM**
  - `plan_response_tool` 签名只接受 indicator 等查询意图参数，**不暴露 risk_level / score**：LLM 无法幻觉或篡改风险等级（守住 Phase 6「数字与等级不经过 LLM」）
- `app/tools/__init__.py`：`DEFAULT_TOOLS` 成为默认工具清单的**唯一真相源**，消除 graph 兜底默认（1 个）与 agent 默认（3 个）的漂移陷阱；刻意不引入 ToolRegistry 等注册机制
- 顺手统一 `risk_analyzer.py` 的 intel 路径常量，复用 `query_threat_intel.DEFAULT_DATA_PATH`（原为第三处硬编码）
- SecurityAgent 默认注册四工具；prompt 第 10 条：优先用规划工具的结构化计划，**是否需要人工审批由工具判定，LLM 不得自行推断**
- **采用 Tool 方案而非 Node**：graph.py 控制流零改动（tool_map 泛型路由对工具数量零假设），`AgentState` 仍为 2 字段；节点化、interrupt 与 State 扩展统一留到 Phase 8（见偏离说明）
- 198 tests（当前基线）

### Implementation Deviation Note（与 §12 Roadmap 的实现偏离说明）

> §12 Roadmap 的原始设计保持不变；本节只记录实际实现与蓝图之间的有意偏离及原因。

- **IOC 查询采用 Exact Match 而非 RAG**：IOC（IP/域名/Hash）是唯一标识符，查询语义是等值判断——Embedding 的语义近似性在此恰恰是缺陷（`203.0.113.66` 的向量近邻可能是 `203.0.113.65`，产生假阳性关联），且引入向量库违背精确查找的本质。§3.2 F3 的"RAG 语义检索"适用于 CVE/ATT&CK 知识库（自然语言文档），不适用于 IOC 库；RAG 仍按原计划留给知识库部分。
- **Risk Analyzer 提前实现**：原 §12 安排在 Phase 7，实际在 Phase 6 前置完成规则侧——因为它的输入（结构化证据）已由 Phase 5 的两个查询工具备齐，且 Rule-based 输出可离线确定性测试，是 Evidence Fusion 的自然收口。Response Planner 已在 Phase 7 补齐规则侧；LLM 风险复核层与计划节点化留给 Phase 8。
- LangGraph 实际形态（2 节点 + 条件边）比 §9.1 蓝图更小：checkpoint/interrupt/审批节点未引入（Phase 8），`AgentState` 仅 2 字段而非 §10 的 7 字段——蓝图描述终态，实现按最小必要演进。
- **Response Planner 采用 Tool 而非 Node（Phase 7）**：§9.1 蓝图把它画成 `response_plan` 节点，实际实现为第 4 个工具。原因：Node 需要从 messages 反解 `RiskAssessment` 或在 tools 节点特判风险工具，两者都会侵蚀 graph 的通用性；而 `interrupt()` 必须落在节点内，所以节点化与 State 扩展应和 Phase 8 的 HITL 一起做。Phase 8 的节点可直接复用同一个纯函数 `plan_response()`，不产生返工。
- **incident 持久化延后（Phase 7 → Phase 8）**：见 §11 说明。Phase 7 不引入数据库，`ResponsePlan` 仅作为结构化输出契约存在，随 ToolMessage 流转；落库与 HITL / checkpoint / audit lifecycle 一起在 Phase 8 实现。

## 当前架构快照（2026-09-17）

> 完成状态：**Phase 0-7 已完成**，Phase 8 未启动。

```
HTTP Client
 ↓
FastAPI（POST /chat，Pydantic 校验，LLM 错误 → 502）
 ↓
SecurityAgent（app/core/agent.py：组装 System+Human、构造持有 graph、提取最终回答）
 ↓
LangGraph StateGraph（app/core/graph.py：唯一控制流实现）
   agent 节点（LLM.bind_tools().ainvoke）⇄ tools 节点（tool_map 路由 + 安全错误契约）
   should_continue 条件边（无 tool_calls / iteration_count ≥ max_iterations → END）
 ↓
Tool layer（纯函数核心 + @tool wrapper 分层）：
   query_security_logs（144 条日志）/ query_threat_intel（29 条 IOC，Exact Match）/
   analyze_risk（风险规则引擎）/ plan_response（处置规划规则引擎）
 ↓
Structured evidence（messages 按 reducer 顺序累积：LogToolMsg → IntelToolMsg → RiskToolMsg → PlanToolMsg）
 ↓
LLM explanation（Hybrid 叙事侧：综合证据，说明来源，输出最终回答）
```

- 默认注册工具：`app.tools.DEFAULT_TOOLS` = `[query_security_logs_tool, query_threat_intel_tool, analyze_risk_tool, plan_response_tool]`（单一真相源，agent 与 graph 共用），graph 对工具数量零假设（加工具 = 加 map 条目，控制流不变）
- 测试基线：198 passed，全部离线（FakeLLMClient / FakeChatModel / ScriptedTraceModel 模式，无真实 API 调用）

## 尚未实现（按 §12 Roadmap）

- RAG / 知识库（CVE、ATT&CK）——Phase 5 剩余部分，检索对象是自然语言文档，与 IOC Exact Match 不冲突
- MCP Server——原 Phase 6（现顺延）
- Response Planner 节点化 / LLM 风险复核层——Phase 8（`interrupt()` 必须落在节点内，与 HITL 同步引入）
- incident 持久化 / HITL / 安全层 / 审批流——Phase 8（checkpoint / interrupt / audit lifecycle 同步引入）
- Observability / Evaluation——Phase 9

## Testing Framework

全部测试**离线运行**，不依赖真实 LLM API Key（`FakeLLMClient` / `FakeChatModel` / `ScriptedTraceModel`）。

### 1. 工具调用模拟（`FakeLLMClient` / `FakeChatModel`）
- 模拟 LLM 的工具调用行为，支持预设响应与预设工具结果
- 完整的异步支持；显式 Fake 而非 MagicMock，断言"收到了什么消息"一目了然

### 2. 图执行轨迹验证（`test_graph.py`）
- 用 `astream(stream_mode="updates")` 逐节点验证 `agent → tools → agent` 的执行顺序
- 覆盖 `iteration_count` 递增、`tool_call_id` 传递、多工具循环、异常安全契约、消息顺序

### 3. 核心流程验证
- 消息序列：`HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage`
- 错误处理：参数错误让 LLM 修正参数，执行错误返回结构化错误信息
- 工具结果：JSON 格式，包含 count 和 events，不暴露内部细节
- 终止条件：达到 max_iterations 时返回明确信息，为后续 Evaluation 和 Observability 做准备

### 4. 测试分层

| 目录 | 覆盖对象 |
|---|---|
| `tests/test_api/` | FastAPI 路由、错误码、消息透传 |
| `tests/test_core/` | agent / graph / llm / config / tool schema / evidence fusion / risk 集成 / response 集成 |
| `tests/test_schemas/` | LogEvent / ThreatIntelRecord / RiskAssessment / ResponsePlan 的校验边界、seed 可复现 |
| `tests/test_tools/` | 四个工具核心函数的过滤、排序、规则分支与错误契约 |

当前基线：**198 passed**（`pytest -q`，2026-09-17）。
`test_response_integration.py` 用 `tmp_path` 现场生成数据文件，不依赖 `data/*.jsonl`（该目录被 gitignore，新克隆下不存在），是可重复的离线测试。
