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
│ /chat /triage /resume 路由、输入校验          │
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

> 注：上图与 §7 目录描述的是**终态蓝图**。截至 2026-09-19 的实际实现：编排层为单文件 `app/core/graph.py`（未创建 `app/graph/` 包）；`app/security/` 已创建（policy / audit / store）；`app/rag/`、`app/knowledge/`、`app/evaluation/`、`docker/` 尚未创建。MCP 接口已于 Phase 9.3-E 落地为 `app/mcp/`（见 §8 Flow C）。
>
> **路由命名的实现偏离**：蓝图写 `/chat /approve /audit`，实际实现为 `/chat /triage /resume`。审批不是独立端点——审批决定（`status` + `operator`）是 `/resume` 的请求载荷，与恢复句柄 `thread_id` 一起构成一次完整的恢复请求，拆成两个端点会引入"审批了但没恢复"的中间态。审计目前**没有读接口**（`/audit` 未实现），审计写入走 `app/security/store.py`。

### 为什么是单进程 + FastAPI + SQLite + ChromaDB？

| 选择 | 原因 |
|---|---|
| 单进程部署 | 求职项目要求"一条命令跑起来"；微服务在此规模收益为零、成本是实的（网络调用、一致性、部署复杂度） |
| FastAPI | 类型驱动、自带 Swagger（=免费的产品 UI）、原生 async |
| SQLite | 零运维、单文件，够用到 Phase 9；Phase 10 遇到真实痛点再换 PostgreSQL |
| ChromaDB（embedded） | 持久化、元数据过滤、pip 即用；不引入 Docker 依赖（**尚未引入**：Phase 5 只交付 IOC Exact Match，RAG 顺延，见 §6 / §14） |

分层是**逻辑边界**，不是物理边界——边界清晰，Phase 10 想拆随时能拆。

## 6. 技术栈

> **Current（2026-09-17）**：已装依赖以 `pyproject.toml` / `uv.lock` 为准。下表"引入 Phase"记录该依赖首次加入的**里程碑**，"当前状态"记录它**此刻是否真的装上了**——两者不是一回事。关键事实：**ChromaDB（Phase 5）仍未引入**（RAG 顺延，见 §14 偏离说明）；**mcp 已引入**（`mcp>=2.3.0`，Phase 9.3-E 落地为 `app/mcp/`，见 §8 Flow C）；`respx` 从未加入依赖，测试改用显式 Fake（`FakeLLMClient` / `FakeChatModel`）而非 mock HTTP 层。

### 代码依赖

| 组件 | 用途 | 引入 Phase | 当前状态 |
|---|---|---|---|
| Python 3.12 | 语言 | 1 | 已装（`.venv`） |
| uv | 包管理与环境管理 | 1 | 已用（`uv.lock` 存在） |
| FastAPI + uvicorn | API 层 | 1 | 已装 |
| langchain-openai | LLM 接入（OpenAI 兼容协议，provider 可切换） | 1 | 已装 |
| langchain-core | 消息 / 工具 / runnable 基础类型 | 1 | 已装 |
| Pydantic v2 | Schema 校验 | 1 | 已装 |
| pydantic-settings | 配置管理（.env） | 1 | 已装 |
| structlog | 结构化日志 | 1 | 已装 |
| pytest / pytest-asyncio | 测试 | 1 | 已装 |
| httpx | 测试用 HTTP 客户端（FastAPI `TestClient` 依赖） | 1 | 已装（dev） |
| respx | Mock LLM 的 HTTP 调用 | 1 | **未引入**（测试用显式 Fake 替身，不需要 mock HTTP 层） |
| LangGraph | Agent 编排（State/Node/Edge/Checkpoint/Interrupt） | 4 | 已装（1.0.1） |
| ChromaDB | 向量库（RAG） | 5 | **未引入**（Phase 5 只交付 IOC Exact Match） |
| mcp | MCP Server | 6 | 已装（2.3.0；该版本 `FastMCP` 已更名为 `MCPServer`）。Phase 9.3-E 落地为 `app/mcp/` |
| OpenTelemetry | Trace | 9 | 未引入 |
| Langfuse | 可观测平台 | 9 | 未引入 |
| Docker / PostgreSQL / Redis | 工程化 | 10（可选，按需引入） | 未引入 |

### 安全领域知识（知识库内容，非代码依赖）

| 内容 | 用途 | 引入 Phase |
|---|---|---|
| MITRE ATT&CK | 战术/技术框架，知识库与关联分析 | 5 |
| CVE / NVD | 漏洞库，知识库 | 5 |
| STIX / TAXII | 威胁情报交换标准，作为 IOC 数据结构的参考 | 5 |
| Sigma | 检测规则格式，知识库扩展 | 10（可选） |
| YARA | 恶意样本规则，知识库扩展 | 10（可选） |

> 上述知识库内容**均未实现**：Phase 5 只落地了 IOC 库（`ThreatIntelRecord`），且查询语义是 Exact Match 而非 RAG（见 §14 偏离说明）。STIX / TAXII 的"参考"作用体现在 `ThreatIntelRecord` 的字段设计上，未引入其数据格式。

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
│   ├── mcp/                     # Phase 6 / 9.3-E：只读 MCP 适配器 + 本地 stdio 入口
│   └── evaluation/              # Phase 9
│
├── data/                        # 模拟日志、知识库 JSON；SQLite/Chroma 持久化文件（gitignore）
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
- MCP 入口放 **`app/mcp/`**，不设顶层 `mcp_server/`：它复用 `app.tools` 的核心只读函数与 `app.schemas` 模型，放进 `app/` 包内才能沿用同一条导入边界与测试约定；进程入口仍由 `python -m app.mcp.server` 提供，独立于 API 服务。
- schemas/ 独立成包：Pydantic 模型被 tools、graph、api、knowledge 同时引用，单独放置避免循环 import。

### 7.1 当前实际结构（2026-09-19）

蓝图中的目录并非全部已创建。实际存在的是：

```
cybersec-agent/
├── app/
│   ├── core/       # config / logging / llm / agent / graph（编排层单文件）
│   ├── schemas/    # log_event / threat_intel / risk / response / approval / audit
│   ├── tools/      # query_logs / query_threat_intel / risk_analyzer / response_planner
│   ├── security/   # policy / audit / store（Phase 8）
│   ├── mcp/        # Phase 9.3-E：只读 MCP 适配器（tools.py）+ 本地 stdio 入口（server.py）
│   └── api/        # main（/chat /triage /resume）+ schemas
├── data/           # security_events.jsonl / threat_intel.jsonl（**仅 JSONL**）
├── scripts/        # seed_logs.py / seed_threat_intel.py
├── tests/          # test_api / test_core / test_schemas / test_security / test_tools / test_mcp
├── docs/           # **仅 architecture.md**
├── pyproject.toml
├── uv.lock
├── .env.example
└── .gitignore
```

**尚未创建**：`app/graph/`（编排层实现为 `app/core/graph.py`）、`app/rag/`、`app/knowledge/`、`app/evaluation/`、`docker/`、`docs/learning/`、`docs/interview/`。（MCP **不再**属于"尚未创建"：Phase 9.3-E 把它落在 `app/mcp/`，见 §8 Flow C。）

**data/ 的实际内容**：只有两个 seed 生成的 JSONL 文件，**没有** SQLite 或 Chroma 持久化文件。`audit.db` 由 `SqliteAuditStore` 在运行时按需创建（测试全部注入 `tmp_path`，仓库里不产生该文件）；checkpoint 走 `InMemorySaver`，**不落盘**。

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

### Flow C — MCP 只读接口（Phase 9.3-E 落地）

```
外部 MCP Client（Claude Desktop / MCP Inspector / 我们自己的 Agent）
  → 本地 stdio MCP Server（python -m app.mcp.server，app/mcp/server.py）
  → 只读适配器（app/mcp/tools.py：公开签名里没有任何路径形参）
  → 复用 app/tools 的核心只读函数（不经 LLM、不经 provider）
      ├─ query_security_logs   安全日志查询
      ├─ query_threat_intel    威胁情报 Exact Match 查询
      └─ analyze_risk          确定性风险分析
  → 返回结构化结果
```

没有 `plan_response` 分支、没有 provider/model 分支、没有写入/动作分支。

Flow C 的存在意义：Function Calling 是**进程内、厂商私有**的工具调用协议；MCP 是**跨进程、跨客户端**的标准协议。MCP 在本项目里的定位是**既有能力的另一种只读接口**：它不替代 LangGraph agent 路径，不替代 policy/HITL，不暴露响应规划，也不引入新的权威策略层或审计层——策略与审计仍然只有 `app/security/` 一个入口。

**MCP 信任边界**：

- **不可信**：MCP client 传入的调用参数（`tools/call` 的 `arguments`）。协议层守卫把 SDK 对未声明参数的"静默忽略"升级为**显式拒绝**。
- **服务端控制**：本地数据位置。客户端可见的 schema 里**没有** `data_path` / `logs_path` / `intel_path`；服务端在委派给既有只读函数之前绑定仓库受控的数据位置。
- **既有领域逻辑**：只读查询与风险分析函数（纯函数，不写盘）。
- **不在 MCP 暴露面内**：响应规划、审批变更、策略变更、checkpoint 变更、审计变更、provider/model 调用。

传输面只有**本地 stdio**：不实现 HTTP / SSE / Streamable HTTP 端点，也不做 MCP 鉴权服务器——身份由拉起该进程的宿主承担。

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

> 上图为**终态蓝图**。实际实现是 5 节点（`agent ⇄ tools` + `plan → policy_gate → human_approval`），差异与原因见 §14 偏离说明；审批端点并入 `/resume`，见 §5 注。

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

### 10.1 终态蓝图（Phase 0 设计，尚未落地）

计划中的 `AgentState`：

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

### 10.2 实际实现（`app/core/graph.py`，Phase 8.3 落地，8 字段）

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    indicator: str
    event_type: str | None
    plan: ResponsePlan | None
    policy_decision: PolicyDecision | None
    approval_request: ApprovalRequest | None
    approval_decision: ApprovalDecision | None
```

与蓝图的三处实质差异（记录事实，不改设计）：

- **没有 `tool_results` / `retrieved_docs` / `audit_entries`**：证据统一走 `messages`（`ToolMessage` 按 `add_messages` reducer 顺序累积），审计直接写 store。在 state 里再存一份就是第二个真相源。
- **`risk` 折叠进 `plan`**：`ResponsePlan` 内嵌 `RiskAssessment`（plan → assessment → evidence 全链可追溯），因此没有独立 `risk` 字段。
- **`thread_id` 刻意不进 state**：它是运行时 `configurable`，进 state 会制造第二个真相源；节点内用 `_thread_id()` 读取。
- **`indicator` / `event_type` 必须声明**：LangGraph 对**未声明**的初始 state 键是**静默丢弃**的（实测确认：`ainvoke` 传了不报错，节点里读不到），所以"调用方能传"就等于"这里必须有字段"。这也是 HITL 链路能按 `indicator` 自动短路的前提——自由对话（`/chat`）不传 `indicator`，`plan` 节点直接跳过。

**为什么用结构化 State，而不是把所有内容拼成一个字符串？**

- 每个节点只读写自己关心的字段，职责清晰，不会互相踩踏；
- 工具结果保留结构化对象（而非字符串拼贴），后续节点可以程序化判断（例如 risk_analyze 直接读 tool_results 里的失败登录计数）；
- checkpoint 回放时，每一步的状态都是可读、可验证的。

## 11. 数据库设计初稿

原则：**表随 Phase 增长，Phase 0 只定蓝图；审计类数据 append-only。**

### 11.1 实际落地的表（`app/security/store.py`，Phase 8.2）

三张 SQLite 表，字段以 DDL 为准：

| 表 | 用途 | 关键字段 | 引入 Phase |
|---|---|---|---|
| incidents | 分析结论沉淀 | id, created_at, indicator, risk_level, score, summary, plan_json | 8 |
| action_requests | HITL 审批单 | id, incident_id, thread_id, indicator, risk_level, score, summary, policy_reasons, action_type, priority, target, rationale, requires_approval, reversible, requested_at | 8 |
| audit_logs | 审计流水 | id, ts, actor, event, incident_id, thread_id, interrupt_id, outcome, reason, plan_digest, detail_json | 8 |

> **`action_requests` 没有 `status` 列** —— 这不是遗漏，见 §11.3。

### 11.2 蓝图 vs 实现（未被实现为 SQLite 表的部分）

| 蓝图项 | 蓝图字段 | 实际情况 |
|---|---|---|
| security_events | ts, src_ip, dst_ip, username, action, status, user_agent, raw | **不是 SQLite 表**：实现为 JSONL 文件 `data/security_events.jsonl`（144 条），按需全量读入内存过滤 |
| threat_intel | ioc_type, ioc_value, source, confidence, tags | **不是 SQLite 表**：实现为 JSONL 文件 `data/threat_intel.jsonl`（29 条 IOC） |
| knowledge_docs | kind, key, title, text, metadata | **未实现**（RAG 顺延，见 §14 偏离说明） |
| checkpoints | 框架自动管理 | 已实现，但用 `InMemorySaver` —— **进程内内存，不落盘**。`langgraph-checkpoint-sqlite` 未安装，所以"崩溃恢复"目前只在进程存活期内成立（跨进程恢复是 Phase 9/10 的事） |

ChromaDB collections（`mitre_techniques` / `cve_entries` / `threat_reports`）：**未创建**（ChromaDB 未引入）。

### 11.3 append-only 事件模型（Phase 8.1 落地）

`audit_logs` 与 `action_requests` **只 INSERT 不 UPDATE** —— 状态变化 = 追加新记录。这不是靠约定，而是靠**数据库强制**。完整的 5 个前提（缺一不可）：

1. **纯 INSERT 写入路径**：store 不提供任何 UPDATE / DELETE / INSERT OR REPLACE 接口；
2. **`id` 为 PRIMARY KEY**：重复写入抛 `IntegrityError`（响亮失败，不静默覆盖）；
3. **库层禁改触发器**：每张表 `BEFORE UPDATE` / `BEFORE DELETE` 各一条 `RAISE(ABORT, ...)` —— 绕过应用层直连 `sqlite3` 也改不动（共 6 条触发器）；
4. **表中不存在可变状态列**：`action_requests` 没有 `status`；
5. **待审批状态是派生值**：某 `thread_id` 在 `audit_logs` 里既没有对应的 `approval.decided`、也没有 `approval.timeout` 行 → 仍为 pending。状态是 `NOT EXISTS` 的查询结果，不是被改写的字段。
   - Phase 9.1-A 起有**两个终态事件**（人工决定 / 审批超时）。超时若不解除 pending，已过期的 thread 会每轮惰性清理都重复写一条 `approval.timeout` —— 审计是事实日志，重复计数就是失真。
   - 谓词只回答"是否已有终态事件"，**不做超时判定**（谁算过期由 `TriageService` 持有 `approval_timeout` 决定），职责边界不混。

> **与早期草稿的冲突及修正**：§11 初稿曾把 `action_requests.status(pending/approved/denied/executed)` 列为字段。那与 append-only 原则**直接矛盾** —— 可变状态列意味着"历史"会被原地改写，而可变的"历史"不叫审计。实现按 append-only 落地：**不设 status 列**，状态一律由审计事件推导。
>
> 其他细节：`security_events` 保留 `raw` 原始字段（分析可能出错，原始数据永远可回溯）；时间列一律存 **tz-aware UTC ISO8601 文本**（拒绝 naive datetime —— 混入本地时区会让"字典序 == 时间序"这个前提失效，而审计流完全依赖顺序）；读取一律重新过 Pydantic 校验，脏数据在读取边界报错而非静默跳过。

### 11.4 审计词汇表（封闭枚举）

```python
AuditEvent = Literal[
    "plan.created", "plan.failed", "policy.evaluated",
    "approval.requested", "approval.decided", "approval.timeout",
]
```

刻意包含**失败事件**（`plan.failed`）：失败若不留痕，"有多少次判定失败、为什么失败"就无从回答（F6「每次工具调用、每个审批决策可查」）。**只记成功的审计是幸存者偏差。**

`plan.failed` 的 `detail` 只含 `indicator` 与 `error_type`，**不记异常 message、不记绝对路径、不记 traceback** —— 审计库里的路径会永久留存，泄露内部目录结构。

> 已知局限（必须文档化，不得掩盖）：
> - 本阶段**没有身份认证** —— `actor` 只是调用方自称的字符串，**不具备不可否认性**。认证 / 签名留到 Phase 10（或后续引入最小 API key）。
> - `audit_logs.incident_id` 对图节点与 `approval.timeout` 均为 `NULL`（incident 在图跑完之后才创建，D4）；按 incident 查审计流查不到本次判定，须改用 `thread_id`。

### 11.5 incident 持久化为何延后到 Phase 8（历史决策）

原计划 Phase 7 引入 `incidents` 表，实际 Phase 7 只交付 `ResponsePlan` 结构化契约与规则引擎，计划随 `ToolMessage` 流转、不落库。理由：incident persistence 与 HITL / checkpoint / audit lifecycle 属**同一条状态生命周期**，拆开实现会产生两套状态语义，故统一延后到 Phase 8 一次性落地。Phase 8 已按此执行（见 §11.1）。



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

### 12.1 实际进度 vs 上表（2026-09-19）

上表是**设计蓝图**，保持不变。实际推进有两处顺序偏离（详见 §14 偏离说明）：

| Phase | 蓝图内容 | 实际状态 |
|---|---|---|
| 0-4 | 骨架 / API / 日志 / 工具+ReAct / LangGraph | **已完成** |
| 5 | 威胁情报 + RAG | **部分完成**：IOC 库与 Exact Match 查询已交付；RAG（CVE / ATT&CK 向量检索）**未实现**，顺延 |
| 6 | MCP Server | **未按蓝图在 Phase 6 执行**：该 Phase 实际交付的是 Rule-based Risk Analyzer（提前实现，见 §14），MCP Server 顺延；**已于 Phase 9.3-E 落地为 `app/mcp/`**（只读暴露，见 §8 Flow C） |
| 7 | 风险分析 + 响应规划 | **已完成**（规则引擎侧）：`RiskAssessment` + `ResponsePlan` 纯函数规则引擎 |
| 8 | 安全层 + HITL | **已完成（8.1-8.5）**：策略引擎 / 审批流 / append-only 审计 / SQLite 持久化 / triage 服务与 API |
| 9 | 可观测 + 评估 | **部分启动**：9.1-A 交付可靠性侧（审批超时生命周期 + 终态 checkpoint 清理 + `/chat` 依赖护栏对齐），**不属于**蓝图的可观测/评估内容 —— 蓝图的 Trace / 指标 / LLM-as-Judge 仍未启动 |
| 10 | 工程化 | 未启动 |


## 13. 待定决策

| # | 事项 | 说明 |
|---|---|---|
| 1 | LLM provider | DeepSeek（推荐，便宜、OpenAI 兼容）/ Qwen / OpenAI / 本地 Ollama；代码不变，只改 .env |
| 2 | GitHub 仓库名与可见性 | 已定：`cybersec-agent`（private，求职展示时可改 public） |
| 3 | License | 待定（Phase 10 前确定） |

## 14. 实现进度（随开发更新）

> 2026-09-19 · **Phase 0-7 全部完成；Phase 8 已完成（8.1-8.5）；Phase 9 已启动（9.1-A）**。最新状态见文末"当前架构快照"。
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
  - `scripts/seed_threat_intel.py`：固定 seed + 固定时间，29 条 IOC（恶意 IP 10 / 域名 8 / Hash 6 / 可信对照 5），与 Phase 2 攻击场景对应（203.0.113.66）
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
- 198 tests（Phase 7 终态基线）

> 2026-09-17 · Phase 8.1 完成（审计数据模型 + 策略引擎）

- `app/schemas/audit.py`：`AuditRecord`（append-only 流水，写入后不再修改）+ `AuditEvent` 受限枚举 + `SYSTEM_ACTOR`。`plan_digest` 用 `pattern` 校验 64 位小写 sha256 —— 摘要写错会让"计划是否被改动"的比对失效。
- `app/security/audit.py`：`compute_plan_digest`（规范化 sha256）+ `build_audit_record`（唯一的记录构造入口，统一生成 id / ts）。
- `app/security/policy.py`：`evaluate_policy(plan) -> PolicyDecision`（`allow` / `require_approval`），`POLICY_VERSION = "phase8.1"`。**策略门永不解析 messages**，只消费结构化的 `ResponsePlan`。
- 审计词汇表**刻意包含失败事件**：只记成功的审计是幸存者偏差。

> 2026-09-17 · Phase 8.2 完成（SQLite 持久化 + append-only 强制）

- `app/security/store.py`：`SqliteAuditStore`，三张表（incidents / action_requests / audit_logs）+ 索引 + **6 条禁改触发器**。
- append-only 的 5 个前提全部落实（纯 INSERT / PRIMARY KEY / 库层触发器 / 无可变状态列 / 状态派生），详见 §11.3。
- 只负责 persistence，**不承担任何判定**：不 import policy、不生成时间、不算 digest、不决定"该不该写"。
- 时间列一律 tz-aware UTC ISO8601，**拒绝 naive datetime**（混入本地时区会破坏"字典序 == 时间序"）；读取重新过 Pydantic 校验，脏数据在边界报错。
- 同步 sqlite3 而非 aiosqlite：写入频率极低（一次 triage 个位数行，且不在 ReAct 热循环上），async 是**调用点**的问题而非本模块的问题。

> 2026-09-17 · Phase 8.3 完成（HITL 图：plan / policy_gate / human_approval）

- `app/core/graph.py` 扩展为 5 节点：`agent ⇄ tools` + `plan → policy_gate → human_approval`。HITL 模式下把 `END` 用 `path_map` **重映射到 `plan`**，因此 `should_continue` 的返回值语义不变（控制流零改动）。
- `interrupt()` 落在 `human_approval` 节点内；`AgentState` 扩到 8 字段（见 §10.2）。
- **实测确认 interrupt 的重放语义**：`Command(resume=)` 会从被中断节点的**函数体顶部重放** —— `interrupt()` 之前的代码跑两次、之后跑一次（已完成节点不重放）。由此得到铁律：**`interrupt()` 之前不得有任何副作用**。
- **实测确认框架不校验 `thread_id`**（4 个静默行为）：未知 thread 上 `aget_state()` 返回空快照（不报错）；`Command(resume=)` 会**静默从 START 新起一轮**（看起来像成功）；对已完成的图重复 resume 会**静默返回旧 state**；复用 thread_id 会**覆盖暂停中的 state**（劫持向量）。这些静默行为在 Phase 8.4 被逐一变成明确错误。
- **实测确认 checkpoint 只序列化异常自身的 repr，不序列化 `__cause__` 链** —— 这条决定了 Phase 8.5 的失败处理写法（见下）。

> 2026-09-17 · Phase 8.4 完成（TriageService + /triage + /resume）

- `app/core/triage.py`：`TriageService.triage()` 生成 `thread_id` → 执行图 → 聚合 `TriageOutcome` → 落 incident（审批路径落 action_requests）；`resume()` 的判定树把框架的 4 个静默行为变成明确错误。
- 错误模型：`UnknownThreadError`（404 语义）/ `NotAwaitingApprovalError`（409）/ `CheckpointLostError`（继承前者）/ `TriageDataUnavailableError`（503）。
- `app/api/main.py`：`POST /triage` 与 `POST /resume`（响应体共用 `TriageResponse` = `TriageOutcome` + 传输层 `interrupt_id`）。
- 安全约束（D 系列，均有结构性或行为级测试守着）：
  - **D3**：`thread_id` 只能服务端生成 —— 客户端指定即可复用他人暂停中的 state（劫持向量）；
  - **D7**：`interrupt_id` 只能服务端恢复 —— 客户端能指定就等于能伪造"审批的是哪一次暂停"；`resume()` 签名里没有 `interrupt_id`，服务端自己从 `aget_state().tasks[*].interrupts[*].id` 恢复；
  - **D2**：单计划源 —— 规划工具（`plan_response_tool`）不进 HITL 工具集，`policy_gate` 只消费 state 里的 `plan`，永不解析 messages；
  - **D6**：`event_type` 必须声明进 `AgentState` 才能透传（未声明键被静默丢弃，实测确认）；
  - **D5**：错误消息一律净化 —— 不泄露绝对路径。
- 465 tests（Phase 8.4 终态基线）。

> 2026-09-17 · Phase 8.5 完成（失败留痕 + 护栏加固 + 文档同步）

本阶段是**收口**性质，不含新功能：

- **plan 失败留痕闭环**：新增 `AuditEvent "plan.failed"`；`plan_node` 用 try/except 包住"采集证据 + 生成计划"，失败时先写一条 append-only 审计（`detail` **只含 `indicator` 与 `error_type`**），再抛**通用消息**的 `PlanFailedError("计划生成失败")`。
  - 为什么要通用消息：实测确认 LangGraph 会把异常的 repr 写进 checkpoint，若消息里带路径就会**永久留存**。`from exc` 保留因果链供日志使用，但因果链**不进 checkpoint**。
  - 统一失败映射：`PlanFailedError` → `TriageDataUnavailableError` → HTTP 503。**刻意不做 `cause_type` 分类** —— 那会让"计划失败"长出第二套错误词汇表。
  - 失败路径**不留业务痕迹**：没有 incident、没有 action_requests、没有 `plan.created`，恰好一条 `plan.failed`。
- **HITL_TOOLS 护栏加固**：`PLANNER_TOOL_NAME` 从**工具对象**派生（`plan_response_tool.name`）而非手写字符串。原测试是**恒真**的（拿 `HITL_TOOLS` 与自身表达式比较），已重写为身份断言 + AST 结构断言 —— 用变异测试验证过"改回硬编码字面量会变红"。
- **API DTO extra 策略显式化**：4 个请求 DTO 各自显式声明 `ConfigDict(extra="ignore")`。**行为不变**（不加 `forbid`）—— 在没有 API 版本化机制时，`forbid` 会让任何多发字段的客户端吃 422，兼容成本换不来对应收益；收紧到 `forbid` 留到 Phase 10。
  - 护栏用 **AST** 判断"是否亲自声明"：pydantic v2 的元类**总会**往类 `__dict__` 里塞 `model_config`，且继承会把父类配置合并下来 —— 所以"在 `__dict__` 里"和"值等于 ignore"两条断言在"靠父类兜底"时**都为真**（实测确认），完全失明。
- **hermetic 测试收口**：`test_query_threat_intel.py` / `test_tool_schema.py` / `test_react_agent.py` 三个文件不再依赖仓库 `data/`。根因是 `DEFAULT_DATA_PATH` 是**相对路径**，测试从无 `data/` 的 CWD 运行就红；修法是注入 `tmp_path` 现场生成的 seed 数据，**不改生产默认值**。seed 脚本路径统一锚定到仓库根（原为 CWD 相对）。
- **文档同步**：本节 + §5 路由 / §6 依赖 / §7 结构 / §10 State / §11 数据库与事件模型 / §12 进度 / 快照 / Testing Framework 全部按**实际实现**校正（只改事实漂移，不改设计）。
- 486 tests（Phase 8.5 终态基线）。验证方式：仓库根全绿；**从无 `data/` 目录的 CWD 运行同样 486 passed**（hermetic 证明）。

> 2026-09-19 · Phase 9.1-A 完成（审批超时生命周期 + 终态 checkpoint 清理 + `/chat` 护栏对齐）

本阶段闭合的是**已被文档化的一致性缺口**：`approval.timeout` 从 Phase 8.1 起就在审计词汇表里声明，但生产代码从未产生它（§11.4 / 快照 / §12.1 三处都写着"没有超时触发机制"）。现在补上。

- **惰性超时（没有后台调度器）**：判定只在请求到达时发生 —— `triage()` 入口全量扫（`reap_expired()`）、`resume()` 校验门点检查。锚点 = 该 thread 的 `min(action_requests.requested_at)`，判定式 `utc_now() - 锚点 >= approval_timeout`（**闭区间**，让"窗口为 0"这种退化配置立即生效）。
  - 锚点取自 `action_requests` 而非 `approval.requested` 审计的 `ts`：pending 的**定义**已经是"action_requests 行 + audit 的 NOT EXISTS"，锚点必须落在同一处，否则会出现两套时间真相。
  - 窗口是 `APPROVAL_TIMEOUT = timedelta(hours=24)`（core 的**生命周期策略默认值**），`TriageService(approval_timeout=...)` keyword-only 可覆盖。**刻意不进 `Settings`** —— 不为一个参数扩大部署配置面；日后确需部署期可配，由组合根注入即可，生命周期语义不变。
  - 代价（文档化，不是 bug）：长时间没有新的判定请求时，已过期的 pending 会继续显示为 pending、checkpoint 继续占内存，直到下一个 `/triage` 到达。
- **超时的两个副作用，顺序固定**：先 append 一条 `approval.timeout`（终态事实），再删除该 thread 的 checkpoint。`detail` 记 `elapsed_seconds` / `timeout_seconds`，让"为什么算过期"可复算。
  - **超时绝不写 `approval.decided`** —— 没有人工决定，就不能留下决定的痕迹（否则审计会显示"有人批准了"，而实际无人做过决定，是最严重的一类审计失真）。
  - **超时不可逆**：`approval.timeout` 一旦落库，该 thread 永久 409（`ApprovalExpiredError`），不允许事后补批。不可逆性**不依赖清理是否成功** —— 校验门先看"是否已落库"再看窗口，所以即使 `adelete_thread` 没生效，第二次 `resume` 仍然拒绝。
  - 幂等：`_reap_one` 有 `_has_timed_out` 闸门，同一 thread 最多一条 timeout 审计（`reap_expired` 是"先查 pending、再写审计"，两步之间没有事务，并发请求可能都读到 pending）。
- **清理范围刻意收窄**：只删**终态** —— `failed`（plan 节点抛错后残留的 `next=('plan',)` 快照）与 `timed_out`。`pending_approval` **绝不删除**（删掉就是把正在等人批的请求凭空抹掉）。`completed` / `allowed` 本轮**不清理**（是"9.1-A 没做"，不是"必须保留"）。
  - 单一 choke point：`adelete_thread` 在 `triage.py` 里**恰好出现 1 次**（`_drop_checkpoint` 内），由 AST 护栏钉住 —— 清理散落多处时"绝不删 pending"就无法靠审查一处保证。
  - checkpointer 从图上取（`self._graph.checkpointer`）：它是 `CompiledStateGraph` 的**公开属性**且实测 `is` 组合根传入的同一个实例，所以本类**仍然不持有** checkpointer（不违背 Phase 8.4 的装配约束）。这条由身份断言 + 能力断言（有 `adelete_thread`）守着。
  - 顺带收益：`pending` 是纯 DB 派生，所以进程重启留下的**僵尸 pending 行**（无 checkpoint、`resume` 永远 409）也会在窗口过期后被收成 `timed_out`，不再无限期污染 pending 视图。
- **恢复守卫细化（消息准确性）**：`next == ()` 分支从 3 种情况细分为 5 种，新增 `ApprovalExpiredError`（继承 `NotAwaitingApprovalError` → 409，理由与 `CheckpointLostError` 同构）。新增的"有审计但无终态审批事件"一支修掉了一个**事实错误**：此前 `resume` 一个失败的 thread 会说"该 thread 的审批已完成，不能重复提交"（它从没完成，是失败了），现在说"未产生待审批项（规划失败或策略放行）"。
- **错误优先级规则（本阶段唯一放宽的清理路径）**：`triage()` 的 `except PlanFailedError` 分支里，清理失败**不得顶替已经确立的主失败** —— 降级为 warning 并保留原始 `TriageDataUnavailableError`（→ 503）与完整 `__cause__` 链（`FileNotFoundError → PlanFailedError → TriageDataUnavailableError`）。**只在这一个"已在处理主失败"的路径上放宽**；`reap_expired` / `resume` 侧的清理失败一律响亮抛出。刻意不做通用错误框架。
- **`/chat` 依赖护栏对齐**：新增 `_require_agent()`（镜像 `_require_service`）。此前 `/chat` 直接取 `request.app.state.agent`，缺 agent 时 `AttributeError` → 带 traceback 的 500，而同场景下 `/triage` 返回 503 —— 不对称，且让 `create_app` 的 docstring（"反之 /chat 不可用（503）"）成为假话。
- **7 条变异测试全部验证变红后完整还原**：pending 谓词去掉 timeout / 失败路径不清理 / 过期判定 `>=` 反成 `<` / reap 去掉过期过滤 / 去掉幂等闸门 / 去掉 `/chat` 护栏 / 超时检查移位。其中"M4 reap 去掉过期过滤"打的是本阶段最重要的不变量 —— **未过期的 pending 绝不能被删**。
- 522 tests（Phase 9.1-A 终态基线，8.5 的 486 → +36）。验证方式：仓库根全绿；**从无 `data/` 目录的 CWD 运行同样 522 passed**（hermetic 证明）。

### Implementation Deviation Note（与 §12 Roadmap 的实现偏离说明）

> §12 Roadmap 的原始设计保持不变；本节只记录实际实现与蓝图之间的有意偏离及原因。

- **IOC 查询采用 Exact Match 而非 RAG**：IOC（IP/域名/Hash）是唯一标识符，查询语义是等值判断——Embedding 的语义近似性在此恰恰是缺陷（`203.0.113.66` 的向量近邻可能是 `203.0.113.65`，产生假阳性关联），且引入向量库违背精确查找的本质。§3.2 F3 的"RAG 语义检索"适用于 CVE/ATT&CK 知识库（自然语言文档），不适用于 IOC 库；RAG 仍按原计划留给知识库部分。
- **Risk Analyzer 提前实现**：原 §12 安排在 Phase 7，实际在 Phase 6 前置完成规则侧——因为它的输入（结构化证据）已由 Phase 5 的两个查询工具备齐，且 Rule-based 输出可离线确定性测试，是 Evidence Fusion 的自然收口。Response Planner 已在 Phase 7 补齐规则侧；LLM 风险复核层与计划节点化留给 Phase 8。
- LangGraph 实际形态比 §9.1 蓝图更小：蓝图是 7 节点（含 `analyze_request` / `tool_router` / `tool_execute` / `risk_analyze` / `response_plan` / `execute_or_reject`），实际是 5 节点（`agent ⇄ tools` + `plan → policy_gate → human_approval`）。工具路由用 `tool_map` 泛型路由替代 `tool_router` 节点，风险分析与计划生成合并在 `plan` 节点内（规则引擎是纯函数，不需要单独节点）。`AgentState` 为 8 字段而非 §10.1 蓝图的 7 字段（差异原因见 §10.2）。蓝图描述终态，实现按最小必要演进。
- **Response Planner 采用 Tool 而非 Node（Phase 7）→ 节点化在 Phase 8 落地**：§9.1 蓝图把它画成 `response_plan` 节点，Phase 7 先实现为第 4 个工具（Node 需要从 messages 反解 `RiskAssessment` 或在 tools 节点特判风险工具，两者都会侵蚀 graph 的通用性）。Phase 8.3 按计划把 `plan_response()` 提升为 `plan` 节点 —— **直接复用同一个纯函数，零返工**，验证了当初"等 HITL 一起做"的判断。
- **incident 持久化延后（Phase 7 → Phase 8）**：见 §11.5。Phase 7 不引入数据库，`ResponsePlan` 仅作为结构化输出契约存在，随 ToolMessage 流转；落库与 HITL / checkpoint / audit lifecycle 一起在 Phase 8 实现（已完成）。
- **MCP Server 与 RAG 顺延（Phase 5 / 6）**：§12 把 Phase 5 定为"威胁情报 + RAG"、Phase 6 定为"MCP Server"，实际 Phase 5 只交付 IOC Exact Match（理由见上一条），Phase 6 则交付了提前实现的 Rule-based Risk Analyzer。结果是 **RAG 与 MCP Server 两项能力整体顺延**，`ChromaDB` 至今未引入。这不是"砍掉"，是排序调整：先做能用规则确定性验证的部分（风险分级 / 响应规划 / HITL 安全层），把依赖外部组件的能力留到后面。（后续进展：**MCP Server 已于 Phase 9.3-E 以只读形态落地为 `app/mcp/`**，`mcp` 依赖引入 2.3.0；RAG 仍顺延。）
- **`/approve` 未实现，审批并入 `/resume`**：见 §5 注。审计也**没有读接口** —— `/audit` 未实现，审计写入走 store，读取目前只在测试里通过 `list_audit()` 进行。

## 当前架构快照（2026-09-19）

> 完成状态：**Phase 0-8.5 已完成；Phase 9.1-A 已完成**（审批超时生命周期 + 终态 checkpoint 清理 + `/chat` 护栏对齐）。

```
HTTP Client
 ↓
FastAPI（POST /chat | POST /triage | POST /resume，Pydantic 校验，
         LLM 错误 → 502，领域错误 → 404 / 409 / 503）
 ↓
┌─────────────────────────── /chat ───────────────────────────┐
│ SecurityAgent（app/core/agent.py：组装 System+Human、        │
│                构造持有 graph、提取最终回答）                 │
└─────────────────────────────────────────────────────────────┘
┌────────────────── /triage & /resume ────────────────────────┐
│ TriageService（app/core/triage.py：thread_id 生成、         │
│ outcome 聚合、incident / action_requests 落库、             │
│ resume 校验门把框架的 4 个静默行为变成明确错误、            │
│ 惰性审批超时（9.1-A：approval.timeout + 终态清理）          │
└─────────────────────────────────────────────────────────────┘
 ↓
LangGraph StateGraph（app/core/graph.py：唯一控制流实现）
   agent 节点（LLM.bind_tools().ainvoke）⇄ tools 节点（tool_map 路由 + 安全错误契约）
   should_continue 条件边（无 tool_calls / iteration_count ≥ max_iterations）
   HITL 分支：plan → policy_gate → human_approval（interrupt）→ END
   （HITL 模式把 END 经 path_map 重映射到 plan，should_continue 语义不变）
 ↓
Tool layer（纯函数核心 + @tool wrapper 分层）：
   query_security_logs（144 条日志）/ query_threat_intel（29 条 IOC，Exact Match）/
   analyze_risk（风险规则引擎）/ plan_response（处置规划规则引擎，同时被 plan 节点复用）
 ↓
Structured evidence（messages 按 reducer 顺序累积：LogToolMsg → IntelToolMsg → RiskToolMsg → PlanToolMsg）
 ↓
LLM explanation（Hybrid 叙事侧：综合证据，说明来源，输出最终回答）
 ↓
安全层（app/security/，横切）
   policy.py  evaluate_policy(plan) → allow / require_approval（永不解析 messages）
   audit.py   compute_plan_digest / build_audit_record
   store.py   SqliteAuditStore：incidents / action_requests / audit_logs
              append-only（纯 INSERT + PRIMARY KEY + 6 条禁改触发器 + 无状态列 + 状态派生）
 ↓
持久化：SQLite（业务表 + 审计，append-only）
        checkpoint = InMemorySaver（进程内内存，**不落盘**）
```

- 默认注册工具：`app.tools.DEFAULT_TOOLS` = `[query_security_logs_tool, query_threat_intel_tool, analyze_risk_tool, plan_response_tool]`（单一真相源，agent 与 graph 共用），graph 对工具数量零假设（加工具 = 加 map 条目，控制流不变）
- **HITL 工具集 = `DEFAULT_TOOLS` 去掉规划工具**：`HITL_TOOLS = [t for t in DEFAULT_TOOLS if t.name != PLANNER_TOOL_NAME]`，其中 `PLANNER_TOOL_NAME` 从**工具对象**派生（不手写字符串）。规划工具不进 HITL 工具集，保证"单计划源"（D2）—— `policy_gate` 只消费 state 里的 `plan`。
- 审计事件：`plan.created` / `plan.failed` / `policy.evaluated` / `approval.requested` / `approval.decided` / `approval.timeout` —— 6 个全部有生产写入路径（`approval.timeout` 由 Phase 9.1-A 的惰性超时补齐）
- 测试基线：**601 passed**，全部离线（`FakeLLMClient` / `FakeChatModel` / `ScriptedTraceModel`，无真实 API 调用）；**从无 `data/` 目录的 CWD 运行同样 601 passed**（hermetic）

## 尚未实现（按 §12 Roadmap）

- RAG / 知识库（CVE、ATT&CK）——Phase 5 剩余部分，检索对象是自然语言文档，与 IOC Exact Match 不冲突（ChromaDB 未引入）
- LLM 风险复核层——Phase 8 只做了规则侧；LLM 复核未实现
- 超时的**主动**触发——9.1-A 只做惰性判定（挂在请求入口），没有后台调度器；长时间无请求时过期 pending 不会被及时收掉
- 超时的**观测面**——`TriageOutcome.status` 不新增 `timed_out`，也没有"查询 thread 状态"的端点，所以超时在 API 响应里只以 409 的形式出现，终态事实只能从 `audit_logs` 读
- 跨进程 checkpoint 恢复——当前 `InMemorySaver` 只在进程存活期内有效（`langgraph-checkpoint-sqlite` 未安装）
- `completed` / `allowed` 的 checkpoint 清理——9.1-A 的清理范围只有 `failed` 与 `timed_out`，这两个终态仍留在内存里
- 身份认证 / 不可否认性——`actor` 只是自称字符串（Phase 10）
- 审计读接口（`/audit`）——未实现
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

### 4. 结构性护栏（Phase 8 起大量使用）

行为断言挡不住"今天写对了、明天漂移了"。对**不变式**（而非功能）用两种更强的断言：

- **AST 断言**：直接解析源码语法树，断言结构而非行为。例：`PLANNER_TOOL_NAME` 的赋值右侧必须是属性访问（`ast.Attribute`）而非字符串常量；`_audit_plan_failed` 内不得出现 `str` / `repr` / `traceback` / `.args` 调用；请求 DTO 的类体里必须**亲自**出现 `model_config` 赋值。
  - 注意 AST 的坑：带注解的赋值是 `ast.AnnAssign` 而非 `ast.Assign`；`ast.dump` 里查字符串会误匹配文档字符串之外的文本。这类护栏本身也需要被验证。
- **变异测试**：临时把生产代码改坏（改回硬编码字面量 / 删掉审计写入 / 把通用消息换成 f-string 拼接异常），确认护栏**变红**，然后还原。没有这一步，护栏可能只是恒真断言 —— 本阶段就抓到过两条这样的护栏（见 §14 Phase 8.5）。

### 5. 测试分层

| 目录 | 覆盖对象 |
|---|---|
| `tests/test_api/` | FastAPI 路由、错误码、消息透传、DTO 契约 |
| `tests/test_core/` | agent / graph / HITL 图 / llm / config / tool schema / evidence fusion / risk 集成 / response 集成 / triage service / **审批生命周期（超时与终态清理）** |
| `tests/test_schemas/` | LogEvent / ThreatIntelRecord / RiskAssessment / ResponsePlan / ApprovalRequest / AuditRecord 的校验边界、seed 可复现、API DTO 策略 |
| `tests/test_security/` | 策略引擎 / 审计记录构造 / append-only store（含"源码里无 UPDATE/DELETE"的结构护栏） |
| `tests/test_tools/` | 四个工具核心函数的过滤、排序、规则分支与错误契约 |

当前基线：**601 passed**（`pytest -q`，2026-09-19）。

### 6. hermetic 约束（Phase 8.5 收口）

**全部 35 个测试文件都不依赖仓库 `data/`。** 判据是可执行的，不是承诺：

```bash
cd <任意不含 data/ 的目录>
<python> -m pytest <repo>/tests -q      # 期望:601 passed
```

根因说明：生产默认值 `DEFAULT_DATA_PATH` 是**相对路径**（`data/security_events.jsonl`），相对 CWD 解析 —— 这是**生产行为的正确设计**（部署时以启动目录为基准），但会让测试在换 CWD 时红。所以修的是**测试**（注入 `tmp_path` 现场生成的 seed 数据），不是生产默认值。

同类约定：seed 脚本路径统一用 `Path(__file__).resolve().parents[2] / "scripts" / ...` 锚定仓库根，不写 CWD 相对路径；需要审计库的用例注入 `tmp_path/audit.db`，仓库里不产生 `audit.db`。
