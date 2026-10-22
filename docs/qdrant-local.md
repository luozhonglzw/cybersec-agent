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
| `RAG_BACKEND` | `memory`(默认)/ `qdrant-local` / `bm25` | 检索后端。取值集封闭,没有 `auto` / `fallback` |
| `RAG_QDRANT_PATH` | `data/knowledge_index` | 仅 `qdrant-local` 时使用的索引目录 |

三条硬约束:

- **选错就在启动期失败。** 设置了未知的后端名会在 `Settings` 构造期
  抛 `ValidationError`(进程拒绝启动),不会带着一个隐含默认往下跑;
- **没有静默回退。** 选了 `qdrant-local` 而索引不可用时,`/ask` 返回
  `200` + `retrieval.status = "unavailable"` + `grounded = false`。
  这是"这次没有依据"的业务结论,不是服务故障;
- **`bm25` 的失败走 503,不走 `unavailable`。** 它没有持久化索引,构建期
  唯一可能的失败来自语料本身(缺失、清单不符、文档非法)—— 与"语料坏掉"
  同类,因此复用既有的固定 503,而不是伪造一个"索引不可用"的业务结论。

### `bm25`(S2-G3)

| 维度 | `memory` | `qdrant-local` | `bm25` |
|---|---|---|---|
| 打分 | 字符 3-gram 哈希向量余弦 | 同左 | 字符 3-gram BM25(`k1=1.5`,`b=0.75`) |
| 索引位置 | 进程内存 | 磁盘目录 + 排他锁 | 进程内存,**不落盘** |
| 启动成本 | 切块 + 嵌入 | 打开既有索引 | 切块 + 建倒排 |
| 索引身份校验 | 六项 | 六项 | **无**(没有可陈旧的索引) |
| 需要构建步骤 | 否 | **是** | 否 |
| 词法/语义 | 词法(哈希碰撞) | 词法 | 词法(IDF 加权) |

三点必须说清楚:

1. **`bm25` 不是语义检索。** 它衡量字符 3-gram 重合,不理解同义或蕴含。
   查询与语料没有共享 3-gram 时它给不出信号 —— 冻结集里 4 个零重合 gold
   在 `bm25` 下同样未命中;
2. **`bm25` 没有 `verify()`。** 索引在装配期由 `load_corpus()` 已校验的
   同一份语料现场构建,不存在"索引比语料旧"的窗口。造一个恒真的
   `verify()` 只会让日志里的"已校验"变成空话;
3. **`empty` 的含义随后端变化,但都不等于"知识库没有答案"。**
   `bm25` 下的 `empty` 是"这次查询与任何文档没有共享 3-gram";
   `memory` 下是"所有候选都低于阈值"。判据始终是 `invalid_citations`。

排他锁的推论适用于进程生命周期:组合根在 `lifespan` 的 `finally` 里调用
`GroundedAskService.close()`,因此一次正常退出会释放目录锁。被 `SIGKILL`
时锁残留属预期 —— 下一个进程会得到 `index-locked`,处理方式是确认没有
残留进程后重开,而不是绕过它。

### 重建语义

- 默认行为是**重建**(`replace=True`):让索引等于当前语料,可幂等重跑;
- 脚本**先算完全部向量**,再进入"删集合 → 建集合 → 写入 → 提交身份"这段
  快速窗口。嵌入阶段失败时旧索引原封不动;
- **索引身份最后提交(S2-G1)**。集合建出来时配置里只有一个哨兵
  (`index_format`,没有任何身份字段),等点全部写入且 `count() ==
  chunk_count` 自检通过之后,才用 `update_collection(metadata=...)` 写入
  真正的身份。因此窗口内任何一步失败 —— 包括"点已写完、身份还没提交" ——
  都会留下一个**哨兵索引**,下一次 `open()` 被既有校验以
  `index-metadata-invalid` 拒绝;
- 这解决的是**静默陈旧**:修正前元数据先写、点后写,若重建只改了正文而
  `doc_id` 不变,窗口内崩溃会留下"六项身份校验全过、向量还是上一版"的
  索引 —— 检索会把旧内容当作有效证据返回,且没有任何报错;
- 构建失败时不保留任何"看起来有效"的产物:下一次 `open()` 会 fail closed。
- **这不是事务原子性**:点与身份仍是两次写入,只保证"身份最后可见",不保证
  "点与身份同时生效"。代价是"点已全写但身份未提交"也判为失败,需要重跑。

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
  所有既有索引自动被判为不可用(这是设计,不是故障);
- `bm25` 也**不是**语义检索:它只是把"哈希碰撞导致的伪相似"换成了
  IDF 加权的精确 3-gram 重合。它的 `empty` 反而**可达**(零重合即空),
  但同样**不能**读成"语料无答案" —— 换个措辞就可能命中。

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

## 7.5 检索质量评估(S2-E / S2-G3)

```bash
# 评估内存后端
python scripts/eval_rag.py

# 同时评估 qdrant-local(需要已构建索引)
python scripts/eval_rag.py --qdrant-index-dir data/knowledge_index

# 同时评估词法候选 bm25(不落盘,无需构建索引)
python scripts/eval_rag.py --include-bm25

# 换用 S2-F2 的独立 13 条集
python scripts/eval_rag.py --include-bm25 \
    --queries knowledge/eval/queries-s2f2-independent.jsonl

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

在 `HashingEmbedder` 基线下,`memory` 与 `qdrant-local` 的结果**逐位相同**
—— 这是设计:适配器复用同一个 `Retriever`、同一套确定性 top-k 与同一套
引用 id,换后端不该改变可观测行为。

当前实测(描述性,不是调参目标):

| 集 | `memory`(哈希) | `bm25` |
|---|---|---|
| 原 20 条 | 0.6562 / 0.6771 | 0.8125 / 0.9375 |
| 独立 13 条 | 0.6818 / 0.7273 | 0.9545 / 0.9545 |

**两套标注集不可合并。** 原 20 条是**冻结基准**,后续阶段从未修改;独立 13 条是
**补充验证集** —— 标注在运行任何排序之前写入,但它是在 BM25 诊断**之后**
创建的并用于验证该诊断,因此**不是**完全盲测。它的查询↔文档字符 3-gram
重合中位数是 0.500(原集 0.278),零重合对 0 个(原集 4 个)⇒ 它更简单,
只验证 `bm25` 提升的**方向**,不验证量级,也不支持任何泛化结论;把两套
分数平均会得到一个既不对应任何一套数据的数字。原集里 4 个零重合 gold 在
**两个排序器下都未命中**,这是词法信号缺失,不是某个排序器的缺陷。

这些数字**只描述检索层**,不得表述为"RAG 的语义检索效果",也不得当作
答案层质量(见下一节)。

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `index-missing` | 索引目录不存在 / 目录里没有集合 | 先跑构建脚本 |
| `index-locked` | 另一个进程(或本进程另一个客户端)持有目录 | 关闭那个进程/客户端;确认没有并发写 |
| `index-incomplete` | 点数与 `chunk_count` 不符(写了一半) | 重新构建 |
| `index-metadata-invalid` | 集合里只有哨兵身份:上一次重建在**提交身份之前**中断 | 重新构建(点可能已写全,但身份未提交 ⇒ 按失败处理) |
| `index-corpus-hash` / `index-corpus-version` | 语料变了,索引没重建 | 重新构建 |
| `index-embedder` / `index-dim` | 换过嵌入器参数 | 重新构建 |
| `index-distance` | 集合不是 cosine | 重建(不要手工改集合) |
| `index-format` | 索引由不兼容的版本写出 | 重新构建 |
| `index-not-clean` | 重建时未能清空旧点 | 重建;若持续出现,说明 Qdrant 语义变了,需复核适配器 |

所有异常消息**只含原因码与通用文案**,不含路径、集合名或堆栈 —— 它们会
走到工具返回值与 HTTP 响应体。
