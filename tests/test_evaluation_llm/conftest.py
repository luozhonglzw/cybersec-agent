"""Phase 9.2-D-1 / 9.2-D-2a 测试的公共夹具。

hermetic 要求(与全项目一致):
    所有评测输入与产物都写到 pytest 的临时目录,**绝不读仓库 `data/`**。
    因此在一个没有 `data/` 目录的 CWD 下,本目录的测试同样能完整跑通。

为什么 `matrix` 是会话级夹具:
    完整矩阵是 8 任务 × 11 行为 × 4 基线标签 = 352 次脚本化运行。
    单次约 5 秒;若每个测试都重跑,整个文件会退化成一分钟级的等待。
    夹具只提供**只读**结果对象,测试不得就地修改它。

D-2a 夹具(会话级,同样只读)
----------------------------
    完整 108 单元试点(96 treatment + 12 control)约 2.6 秒,
    因此同样只在会话级跑一次。全部运行由 `ScriptedLLM` 产生:
    **零网络出口、零 API 额度消耗、零真实 provider 客户端。**
"""
import asyncio

import pytest

from app.evaluation.llm import LLM_TASKS, run_llm_evaluation

#: D-2a 清单里记录的基线 commit(D-1 的冻结基线)。
D2A_GIT_COMMIT = "872a20b08848dc34f40e6a7fd5b6856af3fc5968"


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


# ---------------------------------------------------------------------------
# Phase 9.2-D-2a(离线试点基础设施)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def d2a_plan():
    """冻结的试点计划(108 单元 / 324 逻辑上界 / 972 物理上界)。"""
    from app.evaluation.llm.pilot import pilot_plan

    return pilot_plan()


@pytest.fixture(scope="session")
def d2a_datasets(tmp_path_factory):
    """四个数据集变体(仓库之外)。"""
    from app.evaluation.llm.pilot import load_datasets

    return load_datasets(tmp_path_factory.mktemp("cs92d2a-fixtures"))


@pytest.fixture(scope="session")
def d2a_manifest(d2a_datasets):
    """**候选**清单(D-2a 只允许产出 candidate)。"""
    from app.evaluation.llm.pilot import build_candidate_manifest

    return build_candidate_manifest(git_commit=D2A_GIT_COMMIT, datasets=d2a_datasets)


@pytest.fixture(scope="session")
def d2a_outcome(tmp_path_factory, d2a_manifest):
    """完整 108 单元离线 E2E 的产物(会话级,只跑一次)。

    守卫用 `strict=True`:一旦出现对外网络访问,**在发生处**就炸掉 ——
    只在事后检查的守卫很容易被忽略。
    """
    from app.evaluation.llm.executor import OfflineExecutor
    from app.evaluation.llm.offline_guard import NetworkEgressGuard

    workdir = tmp_path_factory.mktemp("cs92d2a-e2e")
    executor = OfflineExecutor(
        workdir=workdir,
        experiment_id="session-e2e",
        manifest_digest=d2a_manifest.manifest_digest,
        guard=NetworkEgressGuard(strict=True),
    )
    return asyncio.run(executor.run())


def _reduced_outcome(tmp_path_factory, behavior: str, experiment_id: str):
    """缩小规模的离线 E2E:单个行为、两个图基线、单重复。

    行为维度由 D-1 的离线矩阵覆盖;这里只把**行为相关**的语义单独跑一遍,
    因此不需要重跑全部 108 单元。
    """
    from app.evaluation.llm.executor import OfflineExecutor
    from app.evaluation.llm.offline_guard import NetworkEgressGuard
    from app.evaluation.llm.pilot import pilot_plan

    baselines = ("B2'", "B3")
    plan = pilot_plan(baselines=baselines, repetition_count=1)
    workdir = tmp_path_factory.mktemp(f"cs92d2a-{experiment_id}")
    executor = OfflineExecutor(
        workdir=workdir,
        experiment_id=experiment_id,
        plan=plan,
        baselines=baselines,
        behaviors=(behavior,),
        repetition_count=1,
        guard=NetworkEgressGuard(strict=True),
    )
    return asyncio.run(executor.run())


@pytest.fixture(scope="session")
def d2a_follow_outcome(tmp_path_factory):
    """`SAFE_PROMPT_INJECTION_FOLLOW`:处理组采纳虚假声明、对照不采纳。"""
    return _reduced_outcome(tmp_path_factory, "SAFE_PROMPT_INJECTION_FOLLOW", "follow")


@pytest.fixture(scope="session")
def d2a_contradict_outcome(tmp_path_factory):
    """`CONTRADICT_NARRATIVE`:两侧都命中同一句话,但**与载荷无关**。"""
    return _reduced_outcome(tmp_path_factory, "CONTRADICT_NARRATIVE", "contradict")
