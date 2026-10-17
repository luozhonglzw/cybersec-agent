#!/usr/bin/env python
"""构建 / 校验 Qdrant Local 知识库索引(S2-D)。

用法
----

    # 用内置语料构建到默认目录(data/knowledge_index)
    python scripts/build_knowledge_index.py

    # 指定语料与索引目录(部署把索引放数据卷时用这个)
    python scripts/build_knowledge_index.py \
        --knowledge-dir knowledge \
        --index-dir /var/lib/cybersec/knowledge_index

    # 只校验既有索引是否仍然匹配当前语料,不写任何东西
    python scripts/build_knowledge_index.py --check

    # 机器可读输出
    python scripts/build_knowledge_index.py --json

退出码
------

    0  成功(构建完成,或 --check 通过)
    2  语料/索引不可用(清单不符、索引陈旧、维度不符、目录被锁 …)
    3  用法错误(路径不是目录等)

为什么是脚本而不是应用启动时自动建索引
--------------------------------------
建索引是**运维动作**,不是启动动作:它要读语料、算全部向量、写一个目录。
放进 `lifespan` 会让"应用启动"变成"可能悄悄重写索引",而索引是否最新
应当由一次显式命令 + 一次显式校验决定。应用侧只**打开**索引,打不开就
fail closed(`/ask` 返回 503),不会自动重建。

关于路径
--------
本脚本的**异常文本**只含原因码与通用文案(与 `app.rag.errors` 的约定
一致),不含绝对路径 —— 这些异常可能被上层直接打印到日志或响应里。
成功摘要会打印目标目录:那是给操作者的,不是给客户端的。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许 `python scripts/build_knowledge_index.py` 直接运行(不要求先 pip install -e)。
_APPLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(_APPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLICATION_ROOT))

from app.rag.chunking import chunk_corpus  # noqa: E402
from app.rag.corpus import DEFAULT_KNOWLEDGE_DIR, load_corpus  # noqa: E402
from app.rag.embedding import HashingEmbedder  # noqa: E402
from app.rag.errors import RagError  # noqa: E402
from app.rag.retriever import build_index_metadata, verify_index  # noqa: E402
from app.vectorstore.qdrant_local import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    QdrantLocalVectorStore,
    build_qdrant_index,
)

EXIT_OK = 0
EXIT_UNAVAILABLE = 2
EXIT_USAGE = 3


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_knowledge_index",
        description="构建或校验 Qdrant Local 知识库索引(离线,不需要服务端)。",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=DEFAULT_KNOWLEDGE_DIR,
        help=f"语料目录(含 corpus.jsonl 与 manifest.json),默认 {DEFAULT_KNOWLEDGE_DIR}",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"索引输出目录(生成物,不入库),默认 {DEFAULT_INDEX_DIR}",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验既有索引与当前语料是否匹配,不写入任何东西",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="以 JSON 输出结果(便于脚本消费)",
    )
    return parser.parse_args(argv)


def _describe(metadata) -> dict[str, object]:
    """索引身份的可打印摘要。不含路径。"""
    return {
        "corpus_version": metadata.corpus_version,
        "corpus_sha256": metadata.corpus_sha256,
        "embedder_id": metadata.embedder_id,
        "dim": metadata.dim,
        "chunker_version": metadata.chunker_version,
        "doc_count": metadata.doc_count,
        "chunk_count": metadata.chunk_count,
    }


def _check(knowledge_dir: Path, index_dir: Path) -> dict[str, object]:
    """只读校验:打开索引 → 与当前语料逐项比对。"""
    corpus = load_corpus(knowledge_dir)
    embedder = HashingEmbedder()

    with QdrantLocalVectorStore.open(index_dir, expected_dim=embedder.dim) as store:
        verify_index(
            corpus.manifest,
            store.metadata(),
            embedder_id=embedder.embedder_id,
            dim=embedder.dim,
            doc_count=len(corpus.documents),
        )
        return {
            "status": "ok",
            "mode": "check",
            "index_dir": str(index_dir),
            **_describe(store.metadata()),
        }


def _build(knowledge_dir: Path, index_dir: Path) -> dict[str, object]:
    """构建(重建)索引。

    先算完全部向量,再动存储 —— 见 `build_qdrant_index` 的说明。
    `replace=True`:构建脚本的语义就是"让索引等于当前语料",幂等重跑。
    """
    corpus = load_corpus(knowledge_dir)
    embedder = HashingEmbedder()
    metadata = build_qdrant_index(corpus, embedder, index_dir, replace=True)
    return {
        "status": "ok",
        "mode": "build",
        "index_dir": str(index_dir),
        **_describe(metadata),
    }


def _expected(knowledge_dir: Path) -> dict[str, object]:
    """`--check` 失败时用来对照的"当前语料应有的身份"。"""
    corpus = load_corpus(knowledge_dir)
    embedder = HashingEmbedder()
    chunks = chunk_corpus(corpus.documents)
    return _describe(build_index_metadata(corpus, embedder, chunks))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    knowledge_dir = Path(args.knowledge_dir).expanduser()
    index_dir = Path(args.index_dir).expanduser()

    if not knowledge_dir.is_dir():
        print(
            "error: --knowledge-dir is not a directory "
            f"(reason: knowledge-dir-missing)",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        result = (
            _check(knowledge_dir, index_dir)
            if args.check
            else _build(knowledge_dir, index_dir)
        )
    except RagError as exc:
        # 只打印原因码与通用文案:路径不进错误输出。
        print(f"error: {exc} (reason: {exc.reason})", file=sys.stderr)
        if args.check:
            try:
                print(
                    "expected: " + json.dumps(_expected(knowledge_dir), ensure_ascii=False),
                    file=sys.stderr,
                )
            except RagError:
                pass
        return EXIT_UNAVAILABLE

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{result['mode']}: {result['status']}")
        print(f"index_dir     : {result['index_dir']}")
        print(f"corpus_version: {result['corpus_version']}")
        print(f"corpus_sha256 : {result['corpus_sha256']}")
        print(f"embedder_id   : {result['embedder_id']} (dim={result['dim']})")
        print(f"chunker       : {result['chunker_version']}")
        print(f"documents     : {result['doc_count']}")
        print(f"chunks        : {result['chunk_count']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
