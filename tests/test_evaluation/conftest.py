"""Phase 9.2-A 评测测试的公共夹具。

hermetic 要求(与全项目一致):
    所有评测输入都写到 pytest 的临时目录,**绝不读仓库 `data/`**。
    因此在一个没有 `data/` 目录的 CWD 下,本目录的测试同样能完整跑通。

为什么 `seed_dataset` 是同步夹具:
    它只做文件写入(纯 I/O),不碰事件循环。做成同步夹具后,同步测试与
    async 测试都能安全使用它;而需要跑完整评测的测试自己用 `asyncio.run`,
    避免把事件循环的创建时机藏进夹具里。
"""
import asyncio

import pytest

from app.evaluation.golden import GOLDEN_SET
from app.evaluation.runner import run_evaluation, write_seed_dataset


@pytest.fixture(scope="session")
def seed_dataset(tmp_path_factory):
    """把 Phase 2 的确定性种子数据写到会话级临时目录(仓库之外)。

    `write_seed_dataset` 复用 `scripts/seed_logs.py` 与
    `scripts/seed_threat_intel.py` —— 它们固定 seed + 固定基准时间,
    逐字节可复现,因此每次评测的输入完全相同。
    """
    root = tmp_path_factory.mktemp("seed-dataset")
    return write_seed_dataset(root)


@pytest.fixture
def evaluation(tmp_path, seed_dataset):
    """跑一次完整评测(B1/B2/B3),返回 EvaluationResult。"""
    return asyncio.run(run_evaluation(
        logs_path=seed_dataset["logs"],
        intel_path=seed_dataset["intel"],
        workdir=tmp_path / "evaluation",
    ))


@pytest.fixture(scope="session")
def golden():
    return GOLDEN_SET
