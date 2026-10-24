# CyberSec Agent — 中文项目材料

> 本文所有技术表述均对应仓库 `b3e79509ffe10391417dc7307eae896df0b1c65f` 上的真实代码、测试与已发布记录。
> 凡属"未验证"或"不声称"的内容，均在文中显式标注。

---

## 1. 项目定位与适用场景

**一句话定位**：一个基于 LangGraph 的安全运营 Agent **原型**，设计前提是**把大语言模型当作不可信组件**——风险评级、响应计划、审批要求与审计记录全部由不读取模型输出的确定性规则引擎产生。

**它做什么**：把原始安全事件与威胁情报，变成一条可审计的处置结论——这个指标是否恶意、有多严重、应该做什么、以及**谁批准了**。

**它适合用来做什么**：

- 演示"如何在 LLM Agent 中把安全判定与模型输出做结构性解耦"，并让这种解耦可被 CI 回归验证；
- 演示 append-only 审计在数据库层如何被强制（而不只靠应用层自觉）；
- 演示 RAG 场景下"引用校验"如何做成结构性属性，而不是提示词约束。

**它不适合用来做什么**（明确边界）：

- **不是生产级安全系统**，不是企业 SOC 产品；
- **不执行任何真实处置动作**——它只生成计划并请求审批；
- **不是自主 Agent**——所有高风险动作都需要人工批准；
- 所有安全数据由确定性脚本**合成生成**，不含真实生产数据。

适用场景是技术评审、架构讨论与教学演示；不是线上安全运营。

---

## 2. 技术栈

| 层次 | 选型 |
| --- | --- |
| 语言 / 运行时 | Python 3.12 |
| Web | FastAPI + Uvicorn |
| Agent 编排 | LangGraph（5 节点状态图） |
| 数据校验 | Pydantic v2（封闭词表 + 边界校验） |
| 审计存储 | SQLite（默认）/ PostgreSQL（可选，Alembic 迁移） |
| 检索 | 自实现 `HashingEmbedder`（离线哈希基线）/ BM25（字符 3-gram）/ Qdrant Local |
| 持久化 checkpoint | `AsyncSqliteSaver` + 严格反序列化白名单（msgpack ext hook） |
| 容器 / CI | Docker、Docker Compose、GitHub Actions（3 个 job） |
| 测试 | pytest（离线 2,283 例 + PostgreSQL 集成 373 例） |

规模参考：`app/**.py` 约 **29,082** 行，`tests/**.py` 约 **43,238** 行（测试/实现 ≈ 1.49 : 1）。

---

## 3. 四项最强技术贡献

### 贡献一：用 AST 结构断言把"模型不可达安全路径"变成可回归的约束

**问题**：如果"模型不能影响安全判定"只写在文档里，下一次重构就可能悄悄破坏它。

**做法**：权威安全路径的两个节点对消息历史零读取——`plan` 节点只读 `state["indicator"]` / `state["event_type"]`，`policy_gate` 节点只读 `state["plan"]` 并调用 `evaluate_policy(plan)`。这一点被写成 CI 断言：

```python
# tests/test_core/test_hitl_graph.py:703
assert "messages" not in ast.dump(_find_func(_graph_tree(), "policy_gate_node"))
```

```python
# tests/test_core/test_hitl_graph.py:711
dumped = ast.dump(_find_func(_graph_tree(), "plan_node"))
assert "collect_evidence" in dumped and "analyze_risk" in dumped and "plan_response" in dumped
assert "messages" not in dumped and "ToolMessage" not in dumped
```

**为什么可信**：第二条是**反向对照**——它同时要求三个规则引擎调用**必须存在**，因此断言不会退化成恒真。

**必须说清的口径**：这是**结构性回归约束**，**不是形式化安全证明**。它约束的是一个**语法性质**（某函数体内不出现某标识符），不做污点分析、不做模型检验，也不能证明经由间接调用路径不可达。覆盖范围也有限：只覆盖 `policy_gate_node` 与 `plan_node` 两个内层函数体，`_route_after_gate` 与 `evaluate_policy` 内部**不在** AST 断言范围内。

**另一层保障**：策略门是**图节点而不是工具**。工具对模型是"可选调用"的，那会让策略退化成建议；节点在唯一路径上，模型无法跳过。

---

### 贡献二：定位框架重放语义，用结构护栏防住重复写入

**问题**：LangGraph 的 `Command(resume=...)` 会**从被中断节点的函数体顶部重放**，而不是从 `interrupt()` 之后继续。这意味着放在 `interrupt()` 之前的任何副作用都会执行两次——一次 append-only 审计写入在恢复时会因主键冲突抛 `IntegrityError`，把恢复流程打挂。

**做法**：把这条框架语义固化成两条互补的 AST 断言：

```python
# tests/test_core/test_hitl_graph.py:677 — interrupt() 之前不得出现副作用
for stmt in func.body[:_interrupt_index(func)]:
    dumped = ast.dump(stmt)
    for forbidden in ("audit_store", "append_audit", "record_incident",
                      "record_action_request"):
        assert forbidden not in dumped
```

```python
# tests/test_core/test_hitl_graph.py:694 — 反向断言:decided 必须写在 interrupt() 之后
after = func.body[_interrupt_index(func):]
assert any("approval.decided" in ast.dump(stmt) for stmt in after)
```

**为什么必须用结构断言而不是行为测试**：行为测试只能覆盖"当前这一种异常"，而"在 `interrupt()` 前加一行写入"是一个**未来可能发生的**改动。结构断言让这个改动立刻变红。

**配套语义**：审批超时为 24 小时**惰性**判定（无后台调度器），锚点是 `min(action_requests.requested_at)`，超时**绝不**写入 `approval.decided`，且超时不可逆。

---

### 贡献三：append-only 审计由数据库强制，而不是由应用自觉

**做法**：

- 审计表**没有**可变状态列——"待审批"是通过查询"终端事件不存在"（`NOT EXISTS`）**派生**出来的，而不是存一个 `status` 字段；
- 存储层只暴露 `INSERT` 与参数化 `SELECT`，**没有** update / delete 路径；
- 主键约束让重复写入**响亮失败**；
- 库层触发器拦截绕过应用的写入：**SQLite 8 个**（4 张表 × update/delete），**PostgreSQL 12 个**（4 张表 × update/delete/**truncate**）。

**PostgreSQL 侧的角色分离**：迁移 owner 持有表所有权，运行时角色只有 `SELECT` / `INSERT`——它没有 DDL 权限，**连禁用触发器都做不到**（那需要表所有权）。有测试逐条验证：不能 update、不能 delete、不能 truncate、不能 alter、不能 drop、不能禁用触发器、连 `alembic_version` 都读不到。

**为什么用 12 个而不是 8 个触发器**：PostgreSQL 的 `TRUNCATE` 是语句级操作，**不会触发行级触发器**——所以必须单独加 `BEFORE TRUNCATE` 触发器，否则一次 `TRUNCATE` 就能静默清空审计历史。

---

### 贡献四：引用校验是结构性属性，不是提示词约束

**做法**：`POST /ask` 的回答带引用标记 `[c1]`，服务端从**本次真实检索结果**构造引用注册表，再做四桶校验（valid / invalid / unverified / malformed）。关键设计：

- 证据**只能**来自 `type == "tool"` 的消息，且其内容必须是合法的 `rag_hits` 信封；
- 模型只能产出 `ai` 消息——因此**伪造工具结果在结构上不可达**；
- 同一 `citation_id` 的 `(doc_id, source_id, source_type, title, url)` 任一不同即判定冲突，整个注册表**作废**（fail closed）；
- 检索后端不可用时返回 **200 + `status="unavailable"` + `grounded=false`**，既不静默回退到另一个检索器，也不伪装成"语料里没有答案"。

**必须说清的口径**：`grounded=true` **不等于事实正确**。它只意味"引用都指向本次真实检索到的文档，且没有编造"。事实正确性未验证。

---

## 4. 一键 Demo 使用方法

```bash
uv run python scripts/demo_offline.py
```

**不需要真实 LLM provider、不需要 API Key、不需要网络、不需要 Docker。**

| 场景 | 类型 | 演示内容 |
| --- | --- | --- |
| `DEMO 1` RBAC | 真实 FastAPI 应用 + 真实 `AuthKeyring` | 401 / 403；未知凭据与缺失凭据响应**逐字节相同** |
| `DEMO 2` 引用校验 | 真实语料 + 真实检索 + 真实校验器 | 编造引用 → `grounded=false` |
| `DEMO 3` 检索 fail-closed | 真实 Qdrant Local 装配路径 | 索引缺失 → `status="unavailable"`，不回退 |
| `DEMO 4` HITL | **test-based**（真实 pytest 子进程） | 暂停 / 恢复 / 线程隔离 / 审批人约束 |

**场景类型必须如实标注**：

- `DEMO 2` 的**回答文本是脚本化的**（本次运行没有 provider），但检索、证据信封、引用校验全部是真实代码；
- `DEMO 4` 是**测试驱动**的演示，不是在线 `/triage → /resume` 往返——那条路径从 `agent` 节点进入图，必须先有模型响应才能暂停，因此需要可达的 provider。

任一场景失败，脚本以非零码退出。实测本地运行结果：

```text
  PASS  DEMO 1  RBAC — authentication and role admission
  PASS  DEMO 2  Citation validation — fabricated citations are rejected
  PASS  DEMO 3  Retrieval fail-closed — a broken index is refused, not replaced
  PASS  DEMO 4  HITL approval and resume — TEST-BASED DEMONSTRATION

DEMO PASSED: 4/4 scenarios
```

---

## 5. 真实 CI 验证数据

GitHub Actions run [`40114943132`](https://github.com/luozhonglzw/cybersec-agent/actions/runs/40114943132)，commit `b3e7950`：

| Job | 范围 | 观测结果 |
| --- | --- | --- |
| `test` | 离线测试套件（无网络、无 API Key） | **2,283 passed**，随后 Demo **4/4** |
| `test-postgres` | PostgreSQL 集成测试（一次性服务） | **373 passed, 0 skipped** |
| `docker` | 真实镜像构建 + 容器内语料加载 + 路由检查 | 镜像构建成功；容器内 40 docs / 80 chunks |

**口径说明**：这是**一个 commit、一次运行、一个一次性数据库**的观测，**不是**对后续提交的持续保证。PostgreSQL job 另设 `PG_TEST_MIN_COLLECTED=250` 下限守卫，防止"用例被整体跳过却仍然绿灯"。

---

## 6. 技术限制（必须如实告知）

### 6.1 checkpoint：默认持久化，但是单写者

- **默认配置**：组合根 `lifespan` 用 `SafeAsyncSqliteSaver` 打开 `CHECKPOINT_DB_PATH`（默认 `data/checkpoints.db`），因此**暂停在 `human_approval` 的运行可以跨进程恢复**。测试证据：`test_pause_survives_a_real_process_restart_and_can_be_resumed` 启动**两个真实 OS 进程**（断言子进程 pid ≠ 父进程 pid），进程 A 暂停后退出，进程 B 从磁盘恢复并跑完；`test_restart_then_authorization_still_holds` 走真实 `create_app()` lifespan 验证重启后对象级授权与禁止自审批**仍然生效**。
- **可选 / 非默认路径**：`InMemorySaver` 只用于**评测工装**（`app/evaluation/adapters.py`、`app/evaluation/llm/adapters.py`）与**测试注入服务**的路径。该路径下重启无法恢复——这是刻意的测试夹具行为，有 `test_checkpoint_is_in_memory_so_a_restart_cannot_resume` 显式记录。
- **真实限制**：checkpoint 存储是**单写者**，**不是**分布式存储；SQLite 文件放在共享卷上对多副本**不安全**。审计库与 checkpoint 库**无共同事务**，因此"决定已追加"与"checkpoint 已推进"之间崩溃始终可能——**不声称端到端 exactly-once**。

### 6.2 认证与不可否认性

- 认证是**静态 API key**，配置只存 SHA-256 摘要；没有过期、轮换、吊销机制。
- 审计 `actor` 字段是**已认证**的主体，但**不可否认性不成立**——没有签名，持有有效 key 的主体仍可否认自己操作过。项目**不作任何不可否认性声明**。

### 6.3 评测边界

- 检索指标：20 条标注集上 Hash 基线 `Recall@5 = 0.6562` / `MRR@5 = 0.6771`，BM25 `0.8125 / 0.9375`。`HashingEmbedder` 是**离线哈希基线，不是语义模型**；4 个零重合 gold 文档两个排序器**都未命中**（词法信号缺失）。
- 独立 13 条集（BM25 `0.9545` / Hash `0.6818`）词面重合中位数 **0.500**，远高于原集的 **0.278**，零重合样本 **0** 个——它**更简单**，只验证方向、不验证量级，**禁止与原集合并或平均**。
- 真实 provider 试点：108 次运行、8 个任务、单 provider 单模型单窗口。注入测试中 12 个载荷样本只有 6 个进入模型可见上下文，**这 6 个全部服从了注入指令，无一抵抗**——因此**不作**提示词注入防御声明。
- 架构不变量（plan digest / policy outcome / 审计事件序列）在试点中不变，这是**回归护栏**，**不是安全得分**。

### 6.4 其他

- 单进程、单实例；无高可用、无备份方案、无限流；无 JWT/OIDC。
- 所有安全数据为合成数据。
- PostgreSQL 后端只存审计 / 事件 / 动作请求 / 归属四张表，**不存** LangGraph checkpoint。

---

## 7. 简历用项目介绍（119 字）

> 基于 LangGraph 的安全运营 Agent 原型。核心是把大模型视为不可信组件：风险评级、响应计划、审批要求与审计记录均由不读模型输出的规则引擎产生，并用 AST 断言固化为 CI 约束。含角色权限、人工审批、只追加审计与引用校验。

---

## 8. 简历用项目经历（四条）

1. **设计并实现"模型不可达安全路径"的结构性保障**：`plan` / `policy_gate` 两个权威节点对消息历史零读取，并以 AST 断言（`assert "messages" not in ast.dump(func)`）固化为 CI 回归约束，配套反向对照断言防止护栏退化成恒真；策略门实现为图节点而非工具，使模型无法跳过。
2. **解决 LangGraph 重放语义导致的重复写入缺陷**：定位到 `Command(resume=)` 从被中断节点函数体顶部重放的框架行为，用 AST 断言禁止 `interrupt()` 之前出现任何审计写入，并实现 24 小时惰性审批超时（超时绝不写入 `approval.decided`、不可逆、无后台调度器）。
3. **实现库层强制的 append-only 审计**：SQLite 8 个 / PostgreSQL 12 个反篡改触发器（PG 额外覆盖 `TRUNCATE`，因语句级操作不触发行级触发器）；PG 侧分离迁移 owner 与运行时角色，运行时角色无 DDL、无法禁用触发器；待审批状态由"终端事件不存在"派生，不设可变状态列。CI 独立 job 实测 373 passed / 0 skipped。
4. **实现结构性引用校验与三检索后端**：RAG 证据仅从 `type == "tool"` 消息的合法信封中收集（模型只能产出 `ai` 消息，故伪造工具结果不可达）；同 id 元数据冲突整体 fail closed；`memory` / `qdrant-local` / `bm25` 三后端在组合根唯一选择、无自动回退，索引不可用返回 200 + `unavailable` 而非静默降级。

---

## 9. 五个常见面试问题与参考回答

### Q1：你怎么证明模型不能影响安全判定？

**三层，且必须说明每层证明了什么、没证明什么。**

1. **数据依赖层**：`plan` 节点只读 `state["indicator"]` 与 `state["event_type"]`；`policy_gate` 节点只读 `state["plan"]` 并调用 `evaluate_policy(plan)`。两者都不读 `state["messages"]`。
2. **拓扑层**：策略门是图节点，不在工具集里。工具对模型是可选调用，那会让策略变成建议；节点在唯一路径上，模型没有"跳过"这个选项。
3. **回归层**：CI 解析 `app/core/graph.py` 的 AST，断言两个函数体内不出现 `messages` 标识符。

**但这是结构性回归约束，不是形式化安全证明**：它约束的是一个语法性质，不做污点分析、不做模型检验，也无法证明经由间接调用路径不可达。它能保证的是"这个性质不会因为一次重构而悄悄消失"。

### Q2：AST 断言会不会误伤，或者退化成恒真？

有反向对照：`test_plan_node_uses_rules_not_tool_output` 在断言 `messages` **不出现**的同时，要求 `collect_evidence` / `analyze_risk` / `plan_response` **必须出现**。如果哪天函数被重命名或清空，这条会先红，而不是让"不出现 messages"变成空真。

覆盖边界也要说清：只覆盖 `policy_gate_node` 与 `plan_node` 两个内层函数体。`_route_after_gate` 不在 AST 断言内（它只读 `state.get("approval_request")`，是纯路由）；`evaluate_policy` 内部由 `tests/test_security/test_policy.py` 的纯函数测试覆盖（确定性、无模块级可变状态、单输入签名、不暴露配置常量）。

### Q3：checkpoint 是进程内还是持久化的？

**默认持久化。** 组合根 `lifespan` 用 `SafeAsyncSqliteSaver` 打开 `CHECKPOINT_DB_PATH`（默认 `data/checkpoints.db`），所以暂停在 `human_approval` 的运行可以跨进程恢复。证据是一条起**两个真实 OS 进程**的测试：子进程暂停后退出，父进程用全新连接、全新 saver 只从磁盘读并完成恢复，并断言 `pid` 不同。

`InMemorySaver` 只出现在评测工装与测试注入路径，那条路径下重启不能恢复，也有测试显式记录。

**限制**：单写者，不是分布式；SQLite 文件放共享卷对多副本不安全；checkpoint 与审计无共同事务，不声称端到端 exactly-once。checkpoint 载荷走严格白名单反序列化（固定项目模型 + `datetime` + `Interrupt`），拒绝 `pickle` 与 `json` 标签——但这只关闭"由数据库里的类型标识符构造任意对象"这条路，**不能**让一个可写的 checkpoint 库变安全。

### Q4：`grounded=true` 是不是就说明答案正确？

**不是。** `grounded` 是**结构性**属性：它只意味"引用都指向本次真实检索到的文档，且没有编造"。

实现上是这样保证的：证据只从 `type == "tool"` 的消息里收集，且内容必须是合法的 `rag_hits` 信封；模型只能产出 `ai` 消息，所以"模型伪造一个工具结果"在结构上不可达。同一 `citation_id` 的元数据（`doc_id` / `source_id` / `source_type` / `title` / `url`）任一冲突，整个注册表作废。

事实正确性未验证；`malformed_markers` 单独报告，不参与冻结的三项公式。

### Q5：三种检索后端为什么不做自动回退？检索效果如何？

**不做回退的理由是可审计性**：自动回退会让"当前用的是哪个检索器"变成隐式状态，审计记录里的分数随之失去可复现性。所以取值集是封闭的（`memory` / `qdrant-local` / `bm25`），由组合根唯一选择，重启才能切换；索引不可用时返回 `200 + status="unavailable" + grounded=false`，既不回退到另一个检索器，也不把"索引坏了"说成"语料里没有答案"。

**效果必须带样本量说**：20 条标注集上 Hash 基线 `Recall@5 = 0.6562` / `MRR@5 = 0.6771`，BM25 `0.8125 / 0.9375`。独立 13 条集 BM25 `0.9545` / Hash `0.6818`，但它的词面重合中位数是 `0.500`，原集只有 `0.278`，零重合样本 0 个——**独立集更简单**，只验证方向、不验证量级，两者禁止合并或平均。`HashingEmbedder` 是离线哈希基线，**不是语义模型**；4 个零重合 gold 文档两个排序器都未命中，因为词法信号本身缺失。

---

## 10. 不应写入简历的表述（红线清单）

| 禁止表述 | 事实 |
| --- | --- |
| "实现 100% 安全隔离 / 安全得分 X" | 项目刻意不产合成总分；不变量是架构回归护栏，不是安全度量 |
| "生产级 / 企业级 SOC 产品" | README 明确否认；无高可用、无备份、无限流 |
| "防止了提示注入攻击" | 试点中 6 个可评估样本**全部**服从注入指令，无一抵抗 |
| "支持语义检索" | `HashingEmbedder` 是 256 维哈希基线，不是语义模型 |
| "审计不可抵赖 / 不可否认" | 静态 API key，`actor` 已认证但不可否认性不成立 |
| "三种后端自动选择最优" | 无 auto、无 fallback、无运行期切换 |
| "checkpoint 全部在进程内" | 默认是持久化 SQLite，可跨进程恢复 |
| "全链路端到端测试覆盖" | 无覆盖率门禁；跨进程链路仅 checkpoint 恢复一条有真实进程测试 |
| 任何性能数字（QPS / 延迟 / 吞吐） | 项目**未做**任何性能测量，无可引用数据 |
| 任何团队规模 / 协作经历 | 单人项目，不虚构团队经历 |
