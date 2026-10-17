"""语料定位的 CWD 无关性(v0.5.0-S2-D TASK 5)。

背景:`DEFAULT_KNOWLEDGE_DIR` 曾经是 `Path("knowledge")`,即"以进程启动目录
为基准"。于是同一份代码:

    从仓库根启动   → 正常
    从别的目录启动 → CorpusError: missing-corpus

而错误文案刻意不含路径(避免泄漏部署布局),所以现场只剩一句"语料不可用"。
语料是**随应用分发**的只读输入,它相对的是应用,不是调用者的工作目录。

这三个用例都不读 `data/`,只验证路径解析语义。
"""
from __future__ import annotations

from pathlib import Path

from app.rag import load_corpus
from app.rag.corpus import APPLICATION_ROOT, DEFAULT_KNOWLEDGE_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_default_knowledge_dir_is_absolute():
    """相对路径会把"语料在哪"变成启动目录的函数。"""
    assert DEFAULT_KNOWLEDGE_DIR.is_absolute()


def test_the_default_knowledge_dir_is_derived_from_the_package_location():
    """由 `app/rag/corpus.py` 的位置推导,不是由 CWD 推导。"""
    assert APPLICATION_ROOT == REPO_ROOT
    assert DEFAULT_KNOWLEDGE_DIR == REPO_ROOT / "knowledge"
    assert (DEFAULT_KNOWLEDGE_DIR / "corpus.jsonl").is_file()
    assert (DEFAULT_KNOWLEDGE_DIR / "manifest.json").is_file()


def test_loading_the_default_corpus_works_from_any_working_directory(tmp_path: Path, monkeypatch):
    """从任意 CWD 调用 `load_corpus()` 都必须成功。

    这是本阶段的**验收条件**:`/ask` 在启动装配时调用无参 `load_corpus()`,
    若它依赖 CWD,那么"从 / 启动的容器"就永远拿不到知识库,而失败信息
    不会告诉任何人原因。
    """
    monkeypatch.chdir(tmp_path)
    corpus = load_corpus()
    assert len(corpus.documents) == 40
    assert corpus.manifest.corpus_version == "v0.5.0-s2.1"


def test_explicit_relative_paths_still_resolve_against_the_cwd(tmp_path: Path, monkeypatch):
    """反向对照:**显式传入**的相对路径仍按 CWD 解释。

    只有"默认值"改成应用相对;显式参数不该被偷偷改写 —— 那会让
    `load_corpus("knowledge")` 的语义随实现变化,而调用方无从预期。
    """
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "knowledge").exists()

    from app.rag import CorpusError

    try:
        load_corpus("knowledge")
    except CorpusError as exc:
        assert exc.reason == "missing-manifest"
    else:  # pragma: no cover - 走到这里说明显式路径被改写了
        raise AssertionError("显式相对路径不应被解析到应用根目录")
