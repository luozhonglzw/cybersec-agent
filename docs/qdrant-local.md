# Qdrant Local 向量索引

本文只讲 **v0.5.0-S2-D 引入的 Qdrant Local 适配器**。检索语义、语料契约、
引用校验分别见 `app/rag/**` 的模块文档与 `knowledge/README.md`。

---

## 1. 这是什么,不是什么

`QdrantClient(path=...)` 是 Qdrant 的**嵌入式(Local)模式**:数据落在本地
目录,余弦相似度在进程内用 numpy 计算。

- **不需要** Qdrant 服务端;
- **不需要** Docker / 容器;
- **不需要** 网络(构建与检索全程不出网);
- **不需要** GPU。

它**不是** Qdrant 服务端,也**不提供**任何高可用、复制、分片或分布式保证。
本仓库没有任何"连接远程 Qdrant"的代码路径:适配器只以 `path=` 构造客户端,
这一点由 `tests/test_vectorstore/test_qdrant_local.py` 里的 AST 断言守住。

### 单进程、排他锁

Qdrant Local 对存储目录持有**排他锁**(实测 `qdrant-client 1.19.1`)。同一
目录上开第二个客户端会直接失败:

```
RuntimeError: Storage folder ... is already accessed by another instance of Qdrant client.
```

适配器把这种冲突翻译成 `IndexUnavailable(reason="index-locked")`,**不重试、
不等待** —— 重试会把"配置错了"伪装成"有点慢"。

推论:

- 一个存储目录同一时刻只能被**一个**客户端使用;
- 用完必须 `close()`(或用作上下文管理器)。不关会让 `.lock` 残留到进程退出;
- **并发写不安全**,本阶段没有任何"多写者"假设。

---

## 2. 依赖

`qdrant-client` 是**运行时依赖**(`pyproject.toml`),因为部署若要打开
Qdrant Local 索引,`uv sync --locked --no-dev` 装出来的环境里必须有它。

```
qdrant-client>=1.12,<2      # 实际锁定 1.19.1
```

传递依赖(全部为 CPU 轮子,无 GPU / torch / transformers):

| 包 | 作用 |
|---|---|
| `numpy` | Local 模式的向量运算 |
| `grpcio` / `protobuf` | SDK 的 gRPC 传输(本地模式不用,但属于必需依赖) |
| `httpx[http2]` → `h2` / `hpack` / `hyperframe` | SDK 的 HTTP 传输 |
| `portalocker` | 存储目录的排他锁 |
| `urllib3` | SDK 的兼容传输 |

---

## 3. 构建索引

```bash
# 用内置语料构建到默认目录
python scripts/build_knowledge_index.py

# 指定语料与输出目录(部署把索引放数据卷时用这个)
python scripts/build_knowledge_index.py \
    --knowledge-dir knowledge \
    --index-dir /var/lib/cybersec/knowledge_index

# 只校验既有索引是否仍然匹配当前语料,不写任何东西
python scripts/build_knowledge_index.py --check

# 机器可读输出
python scripts/build_knowledge_index.py --json
```

退出码:

| 码 | 含义 |
|---|---|
| `0` | 成功(构建完成,或 `--check` 通过) |
| `2` | 语料/索引不可用(清单不符、索引陈旧、维度不符、目录被锁 …) |
| `3` | 用法错误(`--knowledge-dir` 不是目录) |

### 默认位置

| 项 | 默认值 | 是否入库 |
|---|---|---|
| 语料 | `knowledge/`(**应用相对**,由 `app/rag/corpus.py` 位置推导) | **是**(受版本控制) |
| 索引 | `data/knowledge_index/`(**应用相对**) | **否**(`data/.gitignore` 为 `*`) |

`data/` 下的一切都是运行时生成物,可随时删除重建;索引由上面的命令一键复原。

### `/ask` 的运行时配置(S2-E)

| 变量 | 取值 | 说明 |
|---|---|---|
| `RAG_BACKEND` | `memory`(默认)/ `qdrant-local` | 检索后端。取值集封闭,没有 `auto` / `fallback` |
| `RAG_QDRANT_PATH` | `data/knowledge_index` | 仅 `qdrant-local` 时使用的索引目录 |

两条硬约束:

- **选错就在启动期失败。** 设置了未知的后端名会在 `Settings` 构造期
  抛 `ValidationError`(进程拒绝启动),不会带着一个隐含默认往下跑;
- **没有静默回退。** 选了 `qdrant-local` 而索引不可用时,`/ask` 返回
  `200` + `retrieval.status = "unavailable"` + `grounded = false`。
  这是"这次没有依据"的业务结论,不是服务故障。

排他锁的推论适用于进程生命周期:组合根在 `lifespan` 的 `finally` 里调用
`GroundedAskService.close()`,因此一次正常退出会释放目录锁。被 `SIGKILL`
时锁残留属预期 —— 下一个进程会得到 `index-locked`,处理方式是确认没有
残留进程后重开,而不是绕过它。

### 重建语义

- 默认行为是**重建**(`replace=True`):让索引等于当前语料,可幂等重跑;
- 脚本**先算完全部向量**,再进入"删集合 → 建集合 → 写入"这段快速窗口。
  嵌入阶段失败时旧索引原封不动;
- 写入后立刻自检 `count() == chunk_count`。不等就报
  `index-incomplete` —— **半份索引永远不会被当成有效索引**;
- 构建失败时不保留任何"看起来有效"的产物:下一次 `open()` 会 fail closed。

### 一个必须知道的 Qdrant Local 行为

`delete_collection()` **只**把集合从 `meta.json` 移除,**磁盘上的
`collection/<name>/` 数据目录会保留**;随后用同名 `create_collection()`
会把那份残留目录重新挂上,旧点全部"复活"(实测:80 块 → 删除 → 同名重建
→ `count() == 80`)。

因此适配器在重建路径上显式清空点(`Filter(must=[])`)并**再数一遍**确认
为 0,否则抛 `index-not-clean`。这不是防御性冗余:若不处理,重建后的索引
会同时包含新旧两份内容,被删掉的文档仍然会被检索出来,而所有元数据看起来
都是对的。

---

## 4. 在代码里使用

```python
from app.rag import HashingEmbedder, load_corpus
from app.vectorstore import build_qdrant_retriever

corpus = load_corpus()                      # 应用相对,不依赖 CWD
retriever = build_qdrant_retriever(corpus, HashingEmbedder(), "data/knowledge_index")
try:
    result = retriever.retrieve("日志组件的查找注入如何触发")
finally:
    retriever.store.close()                 # 必须关闭:排他锁
```

`build_qdrant_retriever` 与 `build_in_memory_retriever` 产出**同一个**
`Retriever` 类型:同一套完整性校验、同一套确定性 top-k、同一套引用 id。
差别只有一处 —— 返回的检索器持有一个必须关闭的存储。

在本语料上,两个后端的检索结果**顺序与分数完全一致**(实测,见
`test_ranking_matches_the_in_memory_backend`)。

> **`/ask` 的后端由 `RAG_BACKEND` 显式选择(S2-E)。** 默认仍是内存索引,与
> v0.5.0-S2-C 行为一致;选 `qdrant-local` 时组合根在启动期打开
> `RAG_QDRANT_PATH` 指向的索引,并在进程退出时释放排他锁。
> **没有** `auto` / `fallback` 取值:索引打不开就是打不开 —— 端点返回
> 200 + `status="unavailable"`,不会悄悄退回内存索引(那会让"这次用的是哪份
> 数据"变成运行时猜测)。详见 README 的 "Knowledge-base Q&A (`/ask`)" 一节。

---

## 5. 本地限制

| 限制 | 说明 |
|---|---|
| 单进程 | 排他锁;不支持多进程共享同一目录 |
| 无并发写保证 | 没有多写者场景的验证,也不应假设安全 |
| 无高可用 | 没有复制、分片、故障转移;目录损坏即索引损坏,需重建 |
| 非 ANN | Local 模式是暴力扫描;语料规模(40 篇 / 80 块)下无差异,规模上去后需重新评估 |
| 索引与语料强绑定 | 语料版本/哈希、嵌入器身份、维度、切块器版本任一不符即 fail closed |

---

## 6. 嵌入质量限制(重要)

`HashingEmbedder` 是**可离线复现的检索基线**,不是语义模型:它把文本切成
字符 3-gram,用带符号特征哈希映射到 256 维。它衡量的是**字符片段重合度**,
不理解同义、蕴含或否定。

因此:

- 用它算出的 `recall@5` / `MRR@5` 只描述**这个基线**,不得表述为"RAG 的
  语义检索效果";
- 哈希碰撞使**零重合**的查询也能拿到 0.10–0.14 的分数,于是
  `status == "empty"` 几乎不可达 —— **"检索为空"不能当作"语料无答案"的
  判据**。无答案只能靠引用校验(`invalid_citations`)识别;
- 真实语义质量需要外部嵌入服务,属于后续阶段;届时 `embedder_id` 会变,
  所有既有索引自动被判为不可用(这是设计,不是故障)。

---

## 7. 跑真实的 Qdrant 集成测试

```bash
# 全部向量库相关用例(真实 Qdrant Local,建在 tmp_path)
./.venv/Scripts/python.exe -m pytest tests/test_vectorstore -q

# 只跑适配器
./.venv/Scripts/python.exe -m pytest tests/test_vectorstore/test_qdrant_local.py -q

# 只跑构建脚本
./.venv/Scripts/python.exe -m pytest tests/test_vectorstore/test_index_build.py -q

# 后端装配:/ask 的 qdrant-local 选择 + stub 工具降级
./.venv/Scripts/python.exe -m pytest tests/test_api/test_ask.py tests/test_api/test_backend_selection.py -q

# 未绑定工具的提权回归
./.venv/Scripts/python.exe -m pytest tests/test_api/test_unbound_tool.py -q
```

这些用例**不需要** Qdrant 服务端、Docker、GPU 或真实 LLM,也不读也不写
仓库 `data/` —— 每个用例的索引都建在 pytest 的 `tmp_path` 里。

`tests/test_rag/**` 仍然只测**内存**后端(`InMemoryVectorStore`),与本目录
的集成测试相互独立。

---

## 7.5 检索质量评估(S2-E)

```bash
# 评估内存后端
python scripts/eval_rag.py

# 同时评估 qdrant-local(需要已构建索引)
python scripts/eval_rag.py --qdrant-index-dir data/knowledge_index

# 机器可读输出
python scripts/eval_rag.py --json
```

语料是 `knowledge/eval/queries.jsonl` 里的 20 条标注查询(16 可答 / 4 不可答),
`gold_doc_ids` 是单人标注。脚本**不调用任何 LLM**,只跑确定性检索并计算:

| 指标 | 含义 |
|---|---|
| `Recall@5` | 可答查询中,至少一个黄金文档出现在 top-5 的比例 |
| `MRR@5` | 可答查询中,首个命中黄金文档的排名的倒数(无命中为 0) |
| `empty_rate` | 全部查询中检索返回零命中的比例 |

在 `HashingEmbedder` 基线下,两个后端的结果**逐位相同** —— 这是设计:
适配器复用同一个 `Retriever`、同一套确定性 top-k 与同一套引用 id,换后端
不该改变可观测行为。

这些数字**只描述哈希基线**,不得表述为"RAG 的语义检索效果"(见下一节)。

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `index-missing` | 索引目录不存在 / 目录里没有集合 | 先跑构建脚本 |
| `index-locked` | 另一个进程(或本进程另一个客户端)持有目录 | 关闭那个进程/客户端;确认没有并发写 |
| `index-incomplete` | 点数与 `chunk_count` 不符(写了一半) | 重新构建 |
| `index-corpus-hash` / `index-corpus-version` | 语料变了,索引没重建 | 重新构建 |
| `index-embedder` / `index-dim` | 换过嵌入器参数 | 重新构建 |
| `index-distance` | 集合不是 cosine | 重建(不要手工改集合) |
| `index-format` | 索引由不兼容的版本写出 | 重新构建 |
| `index-not-clean` | 重建时未能清空旧点 | 重建;若持续出现,说明 Qdrant 语义变了,需复核适配器 |

所有异常消息**只含原因码与通用文案**,不含路径、集合名或堆栈 —— 它们会
走到工具返回值与 HTTP 响应体。
