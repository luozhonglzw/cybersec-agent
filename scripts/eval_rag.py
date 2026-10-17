#!/usr/bin/env python
"""RAG 检索质量离线评估(S2-E)。

用法
----

    # 评估内存后端(默认)
    python scripts/eval_rag.py

    # 同时评估 qdrant-local(需要已构建索引)
    python scripts/eval_rag.py --qdrant-index-dir data/knowledge_index

    # 机器可读输出
    python scripts/eval_rag.py --json

指标
----

- **Recall@5**: answerable 查询中,黄金文档出现在 top-5 检索结果中的比例。
- **MRR@5**: answerable 查询中,首个命中黄金文档的排名的倒数(无命中则为 0)。
- **empty_rate**: 所有查询中检索返回零命中的比例(探测索引覆盖漏洞)。

本脚本**不调用任何 LLM**, purely deterministic。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_APPLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(_APPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLICATION_ROOT))

from app.rag.corpus import DEFAULT_KNOWLEDGE_DIR, load_corpus  # noqa: E402
from app.rag.embedding import HashingEmbedder  # noqa: E402
from app.rag.errors import IndexUnavailable  # noqa: E402
from app.rag.retriever import build_in_memory_retriever  # noqa: E402
from app.rag.types import normalize_query  # noqa: E402
from app.vectorstore import build_qdrant_retriever  # noqa: E402

EVAL_PATH = Path("knowledge/eval/queries.jsonl")
DEFAULT_TOP_K = 5


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_rag",
        description="评估 RAG 检索质量(Recall@5 / MRR@5),纯离线,不调用 LLM。",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=DEFAULT_KNOWLEDGE_DIR,
        help=f"语料目录,默认 {DEFAULT_KNOWLEDGE_DIR}",
    )
    parser.add_argument(
        "--qdrant-index-dir",
        type=Path,
        default=None,
        help="Qdrant Local 索引目录;给定则同时评估 qdrant-local 后端",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="以 JSON 输出结果",
    )
    return parser.parse_args(argv)


def _load_queries(path: Path) -> list[dict]:
    queries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _evaluate(retriever, queries: list[dict]) -> dict:
    """对给定检索器计算指标。"""
    answerable = [q for q in queries if q["answerable"]]
    unanswerable = [q for q in queries if not q["answerable"]]

    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    empty_count = 0

    for q in queries:
        text = normalize_query(q["query"])
        result = retriever.retrieve(text, top_k=DEFAULT_TOP_K)
        retrieved_ids = [hit.doc_id for hit in result.hits]
        if not retrieved_ids:
            empty_count += 1

        gold = set(q.get("gold_doc_ids", []))
        if not gold:
            continue

        hits = [doc_id for doc_id in retrieved_ids if doc_id in gold]
        recall_scores.append(len(hits) / len(gold))

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gold:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

    return {
        "total_queries": len(queries),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "recall_at_5": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
        "mrr_at_5": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else 0.0,
        "empty_rate": round(empty_count / len(queries), 4) if queries else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not EVAL_PATH.exists():
        print(f"error: eval set not found at {EVAL_PATH}", file=sys.stderr)
        return 2

    queries = _load_queries(EVAL_PATH)
    corpus = load_corpus(args.knowledge_dir)
    embedder = HashingEmbedder()

    memory_retriever = build_in_memory_retriever(corpus, embedder)
    memory_result = _evaluate(memory_retriever, queries)
    memory_result["backend"] = "memory"

    results = [memory_result]

    if args.qdrant_index_dir is not None:
        try:
            qdrant_retriever = build_qdrant_retriever(
                corpus, embedder, args.qdrant_index_dir
            )
            qdrant_result = _evaluate(qdrant_retriever, queries)
            qdrant_result["backend"] = "qdrant-local"
            results.append(qdrant_result)
            qdrant_retriever.store.close()
        except IndexUnavailable as exc:
            print(
                f"warning: qdrant-local index unavailable (reason: {exc.reason})",
                file=sys.stderr,
            )

    if args.as_json:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"backend        : {r['backend']}")
            print(f"  queries      : {r['total_queries']}")
            print(f"  answerable   : {r['answerable_queries']}")
            print(f"  unanswerable : {r['unanswerable_queries']}")
            print(f"  Recall@5     : {r['recall_at_5']}")
            print(f"  MRR@5        : {r['mrr_at_5']}")
            print(f"  empty_rate   : {r['empty_rate']}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
