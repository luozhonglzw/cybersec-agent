"""向量存储后端的测试夹具。

三条约定(与 `tests/test_rag/conftest.py` 同一风格):

1. **锚定仓库根**(`parents[2]`),不依赖 CWD;
2. **索引一律建在 `tmp_path`**,不读也不写仓库 `data/`;
3. **用真实语料**而不是合成夹具 —— 这里要证明的是"真实 Qdrant Local 能
   承载既有语料",合成夹具会让"语料里恰好有 80 块"这类事实失去意义。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from app.rag import HashingEmbedder, LoadedCorpus, load_corpus
from app.vectorstore import build_qdrant_index

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_knowledge_index.py"


@pytest.fixture(scope="session")
def corpus() -> LoadedCorpus:
    return load_corpus(KNOWLEDGE_DIR)


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture()
def index_dir(tmp_path: Path) -> Path:
    """每个用例一个全新的存储目录 —— Qdrant Local 持排他锁,不能共享。"""
    return tmp_path / "knowledge_index"


@pytest.fixture()
def built_index(index_dir: Path, corpus: LoadedCorpus, embedder: HashingEmbedder):
    """建好的索引目录 + 它的索引身份。"""
    metadata = build_qdrant_index(corpus, embedder, index_dir)
    return index_dir, metadata


@pytest.fixture(scope="session")
def build_script() -> ModuleType:
    """把 `scripts/build_knowledge_index.py` 当模块加载。

    用 `spec_from_file_location` 而不是改 `sys.path`:`scripts/` 不是包,
    往 `sys.path` 里塞目录会影响整个会话的其他导入,而这条测试只需要
    一个模块对象。模块名带唯一后缀,避免与别处的同名模块冲突。
    """
    spec = importlib.util.spec_from_file_location(
        "_s2d_build_knowledge_index", BUILD_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
