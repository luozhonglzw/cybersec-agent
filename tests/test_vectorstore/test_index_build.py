"""`scripts/build_knowledge_index.py` 的行为测试。

脚本被当模块加载(`spec_from_file_location`),不启子进程 —— 要验的是
返回码与输出契约,不是 shell 解析。

覆盖 TASK 4 的每一条:显式输出目录、不提交生成物、可重复重建、
拒绝语料/清单不符、阻止半份索引被当成有效、错误里不带路径。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.rag import IndexUnavailable, load_corpus
from app.rag.corpus import CORPUS_FILENAME, MANIFEST_FILENAME
from app.vectorstore import COLLECTION_NAME, QdrantLocalVectorStore
from tests.test_vectorstore.conftest import KNOWLEDGE_DIR


def _copy_knowledge(tmp_path: Path) -> Path:
    """把语料复制到 tmp_path —— 制造"清单不符"需要能改文件。"""
    target = tmp_path / "knowledge"
    target.mkdir()
    for name in (CORPUS_FILENAME, MANIFEST_FILENAME):
        shutil.copyfile(KNOWLEDGE_DIR / name, target / name)
    return target


# =====================================================================
# 1. 构建
# =====================================================================


def test_build_writes_an_index_and_reports_identity(build_script, tmp_path: Path, capsys):
    index_dir = tmp_path / "idx"
    code = build_script.main(
        ["--knowledge-dir", str(KNOWLEDGE_DIR), "--index-dir", str(index_dir)]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "build: ok" in out
    assert "documents     : 40" in out
    assert "chunks        : 80" in out

    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.count() == 80


def test_build_json_output_is_machine_readable(build_script, tmp_path: Path, capsys):
    index_dir = tmp_path / "idx"
    code = build_script.main(
        [
            "--knowledge-dir",
            str(KNOWLEDGE_DIR),
            "--index-dir",
            str(index_dir),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["mode"] == "build"
    assert payload["doc_count"] == 40
    assert payload["chunk_count"] == 80
    assert payload["dim"] == 256
    assert payload["embedder_id"].startswith("hashing-v1-d256")


def test_build_is_repeatable(build_script, tmp_path: Path):
    """同一语料重复构建 → 同一索引身份、同一 chunk 集合。"""
    index_dir = tmp_path / "idx"
    assert build_script.main(["--index-dir", str(index_dir)]) == 0
    with QdrantLocalVectorStore.open(index_dir) as store:
        first_meta = store.metadata()
        first_ids = store.chunk_ids()

    assert build_script.main(["--index-dir", str(index_dir)]) == 0
    with QdrantLocalVectorStore.open(index_dir) as store:
        second_meta = store.metadata()
        second_ids = store.chunk_ids()

    assert first_meta == second_meta
    assert first_ids == second_ids


def test_build_replaces_rather_than_appends(build_script, tmp_path: Path):
    """重跑不会把点数翻倍。"""
    index_dir = tmp_path / "idx"
    build_script.main(["--index-dir", str(index_dir)])
    build_script.main(["--index-dir", str(index_dir)])
    with QdrantLocalVectorStore.open(index_dir) as store:
        assert store.count() == 80


# =====================================================================
# 2. 校验
# =====================================================================


def test_check_passes_for_a_fresh_index(build_script, tmp_path: Path, capsys):
    index_dir = tmp_path / "idx"
    build_script.main(["--index-dir", str(index_dir)])
    capsys.readouterr()  # 丢掉构建输出

    assert build_script.main(["--index-dir", str(index_dir), "--check"]) == 0
    assert "check: ok" in capsys.readouterr().out


def test_check_does_not_create_anything(build_script, tmp_path: Path):
    index_dir = tmp_path / "absent"
    code = build_script.main(["--index-dir", str(index_dir), "--check"])
    assert code == 2
    assert not index_dir.exists(), "--check 不得顺手建索引"


def test_check_detects_corpus_drift(build_script, tmp_path: Path):
    """索引建好后语料变了 → `--check` 必须失败,而不是"看起来还行"。"""
    index_dir = tmp_path / "idx"
    knowledge = _copy_knowledge(tmp_path)
    build_script.main(
        ["--knowledge-dir", str(knowledge), "--index-dir", str(index_dir)]
    )

    # 改清单里的 corpus_version —— 内容没变,但版本漂移
    manifest = json.loads((knowledge / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["corpus_version"] = "v0.5.0-s2.99"
    (knowledge / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    assert (
        build_script.main(
            ["--knowledge-dir", str(knowledge), "--index-dir", str(index_dir), "--check"]
        )
        == 2
    )


# =====================================================================
# 3. 语料/清单不符 —— 拒绝构建
# =====================================================================


def test_manifest_hash_mismatch_is_rejected(build_script, tmp_path: Path, capsys):
    knowledge = _copy_knowledge(tmp_path)
    # 改正文但不动清单 → 哈希不符
    corpus_path = knowledge / CORPUS_FILENAME
    corpus_path.write_bytes(corpus_path.read_bytes() + b"\n")

    code = build_script.main(
        ["--knowledge-dir", str(knowledge), "--index-dir", str(tmp_path / "idx")]
    )
    assert code == 2
    assert "manifest-mismatch" in capsys.readouterr().err
    assert not (tmp_path / "idx").exists(), "校验失败不得留下索引目录"


def test_missing_corpus_is_rejected(build_script, tmp_path: Path, capsys):
    empty = tmp_path / "empty_knowledge"
    empty.mkdir()
    code = build_script.main(
        ["--knowledge-dir", str(empty), "--index-dir", str(tmp_path / "idx")]
    )
    assert code == 2
    assert "missing-manifest" in capsys.readouterr().err


def test_non_directory_knowledge_dir_is_a_usage_error(build_script, tmp_path: Path):
    bogus = tmp_path / "not_a_dir.txt"
    bogus.write_text("x", encoding="utf-8")
    assert (
        build_script.main(
            ["--knowledge-dir", str(bogus), "--index-dir", str(tmp_path / "idx")]
        )
        == 3
    )


# =====================================================================
# 4. 错误输出不含路径
# =====================================================================


def test_error_output_carries_no_paths(build_script, tmp_path: Path, capsys):
    """TASK 4:错误里不得暴露内部路径 —— 它们可能被直接打印到日志。"""
    empty = tmp_path / "empty_knowledge"
    empty.mkdir()
    build_script.main(
        ["--knowledge-dir", str(empty), "--index-dir", str(tmp_path / "idx")]
    )
    err = capsys.readouterr().err
    assert str(tmp_path) not in err
    assert "empty_knowledge" not in err
    assert "missing-manifest" in err


# =====================================================================
# 5. 半份索引 / 被锁目录
# =====================================================================


def test_check_fails_when_the_index_is_locked(build_script, tmp_path: Path, capsys):
    """索引正被另一个实例持有 → `--check` 报不可用,不重试、不等待。"""
    index_dir = tmp_path / "idx"
    build_script.main(["--index-dir", str(index_dir)])
    capsys.readouterr()

    holder = QdrantLocalVectorStore.open(index_dir)
    try:
        code = build_script.main(["--index-dir", str(index_dir), "--check"])
    finally:
        holder.close()

    assert code == 2
    assert "index-locked" in capsys.readouterr().err


def test_build_fails_when_the_index_is_locked(build_script, tmp_path: Path, capsys):
    index_dir = tmp_path / "idx"
    build_script.main(["--index-dir", str(index_dir)])
    capsys.readouterr()

    holder = QdrantLocalVectorStore.open(index_dir)
    try:
        code = build_script.main(["--index-dir", str(index_dir)])
    finally:
        holder.close()

    assert code == 2
    assert "index-locked" in capsys.readouterr().err


def test_create_refuses_to_replace_without_explicit_permission(built_index):
    """`create()` 默认不覆盖 —— 覆盖是破坏性动作,必须显式要求。"""
    index_dir, metadata = built_index
    with pytest.raises(IndexUnavailable) as info:
        QdrantLocalVectorStore.create(index_dir, metadata)
    assert info.value.reason == "index-already-exists"


def test_the_index_directory_is_not_inside_the_repository_by_default():
    """默认输出落在 `data/` 下(`data/.gitignore` 是 `*`),不会被提交。"""
    from app.vectorstore import DEFAULT_INDEX_DIR

    assert DEFAULT_INDEX_DIR.is_absolute()
    assert DEFAULT_INDEX_DIR.parent.name == "data"
    repo_root = Path(__file__).resolve().parents[2]
    assert DEFAULT_INDEX_DIR.parent.parent == repo_root


def test_default_index_dir_is_ignored_by_git():
    """生成物不得进入版本控制 —— 直接问 git,而不是相信注释。"""
    import subprocess

    from app.vectorstore import DEFAULT_INDEX_DIR

    repo = DEFAULT_INDEX_DIR.parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(DEFAULT_INDEX_DIR)],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, "默认索引目录未被 .gitignore 覆盖"
