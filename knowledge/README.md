# `knowledge/` —— 自写示例语料

## 这是什么

这个目录是 Threat Intelligence RAG 垂直切片的**人工编写、受版本控制**的输入:

| 文件 | 作用 |
|---|---|
| `corpus.jsonl` | 40 篇自写摘要,每行一篇文档 |
| `manifest.json` | 语料版本、条数与 `corpus.jsonl` 的 sha256 |
| `eval/queries.jsonl` | 20 条标注查询(16 条可答 / 4 条不可答),**冻结基线**,供离线质量测量使用 |
| `eval/queries-s2f2-independent.jsonl` | 13 条标注查询(11 条可答 / 2 条不可答),**补充验证集**;先写标注再看排序,但在 BM25 诊断**之后**建立,因此**不是**盲测集 |

两套评测集**只能分别报告,不得合并、不得平均**。补充集针对基线集未覆盖的文档,
词面重合中位数明显更高(0.500 vs 0.278),因此更容易:它能验证方向,不能支撑
量级或泛化结论,也**不得**被当作"未使用的盲测集"。

## 这**不是**什么(重要)

- **不是 NVD / MITRE ATT&CK 官方数据集的副本、摘录或再分发。** 每篇文档的
  `provenance` 字段都写着 `NOT an official NVD or MITRE ATT&CK record`。
- **不声称覆盖完整或及时。** 40 篇文档覆盖 24 个公开 CVE 与 16 个 ATT&CK 技术,
  挑选标准是"便于演示引用与失败语义",不是"覆盖真实威胁情报面"。
- **不声称已完成许可核验。** 外部数据集的许可与再分发条件**未核验**;在核验
  完成之前,本目录只允许自写摘要 + 官方编号 + 官方链接(`publisher` 字段被
  代码强制为 `self-authored-summary`)。

正文是自写的中文摘要,用于说明漏洞的机制、影响面与处置/检测思路。它们刻意
**不复制**厂商公告原文,也**不**给出可直接照做的攻击步骤。

## 为什么语料放在这里,而不是 `data/`

`data/.gitignore` 的内容是 `*` —— `data/` 下的一切都是**运行时生成物**,可以
随时删除重建。语料相反:它是人工撰写并评审过的输入,必须被版本控制、被
review、被 diff。两者混在一起会让"`data/` 可随时清空"这条既有约定失效。

索引产物(向量、元数据)属于生成物,放在 `data/knowledge_index/`,不入库。
构建/重建/校验命令见 [`docs/qdrant-local.md`](../docs/qdrant-local.md)。

## 改语料时必须同步 `manifest.json`

`manifest.json` 钉住 `corpus.jsonl` 的字节哈希。改语料而忘改清单时,加载会
直接失败(`CorpusError: manifest-mismatch`),而不是让索引悄悄落后于语料。
重新计算:

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('knowledge/corpus.jsonl').read_bytes()).hexdigest())"
```

同时更新 `doc_count`;改语义要升 `corpus_version`,否则旧索引会被当成仍然有效。

## 字段约定

| 字段 | 约束 |
|---|---|
| `doc_id` | 由 `source_type` + `source_id` **推导**(`kb-` + 小写编号;ATT&CK 前缀 `kb-attack-`),不允许自由填写 |
| `source_type` | `cve` / `attack-technique` / `advisory`(当前语料只用了前两种) |
| `source_id` | 官方命名空间(`CVE-YYYY-NNNN` / `T####`),不自造 |
| `source_url` | 官方页面链接,主机限于 `nvd.nist.gov` / `attack.mitre.org` / `github.com`;**仅作为数据展示,服务端与工具从不请求它** |
| `publisher` | 只能是 `self-authored-summary` |
| `provenance` | 必填,且必须包含 `not an official` 的明示 |
| `text` | 200–600 字符,段落之间用空行分隔(切块器按空行分段) |

## 标注查询集

`eval/queries.jsonl` 的 `gold_doc_ids` 只写**确实能回答该问题**的文档。4 条
`answerable: false` 的查询用于检验"语料没有支撑时不应编造引用"。这些标注是
单人标注,因此由此算出的指标是 **descriptive**,不构成质量保证。

```bash
# 离线评估检索质量(Recall@5 / MRR@5 / empty_rate,不调用 LLM)
python scripts/eval_rag.py --qdrant-index-dir data/knowledge_index
```

可用确定性指标只覆盖**检索**层:黄金文档是否出现在 top-5、排在第几位。
"回答是否正确""引用是否恰当""不可答时是否说了不知道"这类**答案层**指标需要
模型参与判定,当前阶段**不评估** —— 没有把它换成哈希检索代理指标。
