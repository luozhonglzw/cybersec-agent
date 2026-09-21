"""Phase 9.2-D-1 测试的公共夹具。

hermetic 要求(与全项目一致):
    所有评测输入与产物都写到 pytest 的临时目录,**绝不读仓库 `data/`**。
    因此在一个没有 `data/` 目录的 CWD 下,本目录的测试同样能完整跑通。

为什么 `matrix` 是会话级夹具:
    完整矩阵是 8 任务 × 11 行为 × 4 基线标签 = 352 次脚本化运行。
    单次约 5 秒;若每个测试都重跑,整个文件会退化成一分钟级的等待。
    夹具只提供**只读**结果对象,测试不得就地修改它。
"""
import asyncio

import pytest

from app.evaluation.llm import LLM_TASKS, run_llm_evaluation


@pytest.fixture(scope="session")
def matrix_workdir(tmp_path_factory):
    """完整矩阵的工作目录(仓库之外)。"""
    return tmp_path_factory.mktemp("cs92d1-matrix")


@pytest.fixture(scope="session")
def matrix(matrix_workdir):
    """跑一次完整脚本化评测矩阵,返回 `LLMEvaluationResult`。

    **零网络调用、零 API 额度消耗。**
    """
    return asyncio.run(run_llm_evaluation(workdir=matrix_workdir))


@pytest.fixture
def fresh_workdir(tmp_path):
    """单个测试用的独立工作目录。"""
    path = tmp_path / "llm-eval"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def tasks():
    return LLM_TASKS
