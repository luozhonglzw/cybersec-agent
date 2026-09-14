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

### 为什么是单进程 + FastAPI + SQLite + ChromaDB？

| 选择 | 原因 |
|---|---|
| 单进程部署 | 求职项目要求"一条命令跑起来"；微服务在此规模收益为零、成本是实的（网络调用、一致性、部署复杂度） |
| FastAPI | 类型驱动、自带 Swagger（=免费的产品 UI）、原生 async |
| SQLite | 零运维、单文件，够用到 Phase 9；Phase 10 遇到真实痛点再换 PostgreSQL |
| ChromaDB（embedded） | 持久化、元数据过滤、pip 即用；不引入 Docker 依赖 |

分层是**逻辑边界**，不是物理边界——边界清晰，Phase 10 想拆随时能拆。

## 6. 技术栈

> **Current**：Phase 0 未安装任何依赖，下表均为 **Planned**。装了什么、何时安装，以 pyproject.toml 与 Git History 为准。

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
│   ├── graph/                   # Phase 4：state、nodes、edges、checkpointer
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
| incidents | 分析结论沉淀 | summary, risk_level, linked_iocs, resolution | 7 |
| action_requests | HITL 审批单 | tool, params, risk, status(pending/approved/denied/executed), requester, approver, 各时间戳 | 8 |
| audit_logs | 审计流水 | ts, actor, event, resource, detail, outcome | 8（概念始于 1） |
| checkpoints | LangGraph 断点 | 框架自动管理 | 4 |

ChromaDB collections：`mitre_techniques` / `cve_entries` / `threat_reports`，每个 chunk 带 metadata（kind, id, tags）支持过滤检索。

设计细节：

- `audit_logs` 与 `action_requests` **只 INSERT 不 UPDATE**——状态变化 = 追加新记录（approved 不是把 pending 改掉，而是追加一条 decision 记录）。可变的"历史"不叫审计。
- `security_events` 保留 `raw` 原始字段——分析可能出错，原始数据永远可回溯。

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

> 2026-09-14 · Phase 1（Step 1+2）、Phase 2、Phase 3 Step 1 完成

当前实际实现（以代码为准）：

```
HTTP Client
 ↓
FastAPI（app/api/main.py：POST /chat，Pydantic 校验，依赖注入 Agent，LLM 错误 → 502）
 ↓
SecurityAgent（app/core/agent.py：支持 Tool Calling 和手写 ReAct 循环）
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

> 2026-09-14 · Phase 3 Step 1 完成（安全日志查询工具）

- `app/tools/query_logs.py`：query_security_logs 工具，支持按 event_type、source_ip、username、start_time、end_time、min_severity、limit 等条件过滤，返回结构化 LogEvent 列表
- `scripts/seed_logs.py`：生成 144 条结构化安全日志，覆盖 8 个安全场景
- `tests/test_tools/test_query_logs.py`：20 个测试用例，覆盖各种查询条件和错误处理

> 2026-09-14 · Phase 3 Step 2 完成（Tool Calling + 手写 ReAct 循环）

- `app/core/agent.py`：SecurityAgent 支持 Tool Calling 和手写 ReAct 循环，默认 max_iterations=5
- `app/tools/query_logs.py`：添加 LangChain 工具包装器，保持核心查询逻辑独立
- `tests/test_core/test_react_agent.py`：完整的 ReAct 循环测试套件，11/11 测试通过
- `app/core/llm.py`：FakeLLMClient 和 FakeChatModel 工具调用模拟框架
- System Prompt 增强：强调事实优先、证据驱动、不足证据时明确说明

## Testing Framework

已实现完整的测试框架，包括：

### 1. 工具调用模拟 (`FakeLLMClient` 和 `FakeChatModel`)
- 模拟 LLM 工具调用行为
- 支持预设响应和工具结果
- 智能工具检测和多关键词匹配
- 完整的异步支持

### 2. 异步测试套件 (`test_react_agent.py`)
- 11/11 测试通过（100%成功率）
- 覆盖所有 ReAct 循环场景：
  - ✅ LLM 直接回答
  - ✅ 单次工具调用 → 最终回答
  - ✅ 多次工具调用 → 最终回答
  - ✅ 未知工具处理
  - ✅ 无效工具参数
  - ✅ 工具执行异常
  - ✅ 最大迭代次数限制
  - ✅ 消息顺序正确性
- 独立测试实例，无 fixture 依赖

### 3. 核心流程验证
- 消息序列：`HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage`
- 错误处理：参数错误让 LLM 修正参数，执行错误返回结构化错误信息
- 工具结果：JSON 格式，包含 count 和 events，不暴露内部细节
- 终止条件：达到 max_iterations 时返回明确信息，为后续 Evaluation 和 Observability 做准备
  - IP 用 `str` + Pydantic validator 校验 IPv4（JSONL / 工具参数 / LLM 数据交换统一字符串形态）；port 校验 0~65535；可枚举字段全部 Literal，脏数据在校验边界被拒绝。
  - 设计要点：**"SSH 暴力破解"不是 event_type，而是大量 login_failed 事件构成的模式**——数据层只记录原子事实，模式识别是 Agent（Phase 3+）的工作。
- `scripts/seed_logs.py`：固定 `random.Random(42)` + 固定基准时间（2026-09-10 08:00 UTC），输出逐字节可复现 → `data/security_events.jsonl`（144 条事件；data/ 生成物不进 git，见 §7 策略）。
- 8 个安全场景：正常登录 / 单次失败噪声 / 同用户多次失败（密码猜测）/ SSH 撒网式爆破 / 爆破 IP 后续成功登录 / 权限提升（sudo 失败→加入 sudo 组）/ Web 攻击迹象 / 正常防火墙流量。场景 3/4/5 构成递进攻击故事线，为 Phase 3 的多步推理（查失败→按 IP 聚合→查是否成功登录）准备真实问题。
- 测试：`tests/test_schemas/`（模型校验拒绝非法 IP/event_type/severity/status/port；生成可重复；JSONL 逐行 roundtrip；8 场景真实存在）。
- LogEvent 查询工具（query_security_logs）：Not implemented yet（Phase 3）。

- Tools：Not implemented yet（Phase 3，先手写 ReAct 循环）
- LangGraph：Not implemented yet（Phase 4，原因见 §9.2）
- RAG：Not implemented yet（Phase 5）
- MCP：Not implemented yet（Phase 6）
- Risk Analyzer / Response Planner：Not implemented yet（Phase 7）
- HITL：Not implemented yet（Phase 8）
- Observability / Evaluation：Not implemented yet（Phase 9）
