"""Phase 9.2-D-2d 正式试点的**驱动** —— 带 fail-closed 冻结闸门。

本模块存在的理由只有一个:

    **在 provider 被构造之前**,清单必须已经被证明是真正冻结的。

顺序即安全性质
--------------
`OfflineExecutor` 只收 `manifest_digest: str`,**看不到清单对象** ——
因此它**不可能**成为"先于 provider 构造"的那道闸门。
`verify_manifest(require_frozen=True)` 在应用代码里**零调用**,
这件事本身**不**豁免治理要求。

所以闸门只能放在**驱动入口**,而且必须是**第一件事**:
`tests/test_evaluation_llm/test_d2d_manifest_gate.py` 用 AST 断言
`run_d2d_pilot()` 的**第一条实质语句**就是 `assert_frozen_pilot_manifest(...)`,
并断言驱动里**不存在**任何准备动作(`load_datasets` / `build_d2d_manifest` /
`d2d_plan`),以及所有构造动作(`default_pilot_config` / `make_llm_factory` /
`build_chat_model` / `OfflineExecutor` / `asyncio.run`)都在闸门**之后**。

准备与执行分离(F12)
--------------------
"闸门是第一条语句"这件事**只有**在驱动**不做准备**时才成立。若驱动自己
`load_datasets()` / `build_d2d_manifest()` / `d2d_plan()` 取默认值,那么闸门
之前就存在实质动作,而且**输入是驱动自己挑的** —— 于是"闸门先于执行"变成
一句无法兑现的声明。

因此 `run_d2d_pilot()` 把 `datasets` / `manifest` / `plan` 作为**必填**输入,
**不提供**隐式默认:

    准备(执行之前,由调用方完成)          执行(本模块)
    --------------------------------    ------------------------------
    datasets = load_datasets(workdir)   run_d2d_pilot(
    plan     = d2d_plan()                   datasets=datasets,
    manifest = build_d2d_manifest(          manifest=manifest,
        git_commit=..., datasets=...)       plan=plan, ...)

两条**互不替代**的职责(不得合并):

    builder validity      `build_d2d_manifest()` / `build_candidate_manifest(..., frozen)`
                          可以校验"标着 frozen 的清单**自身**是否合法";
    execution eligibility `run_d2d_pilot()` **独立地**断言"交来的这份冻结清单"
                          与"交来的这份 D-2d 计划 / 身份 / 资源声明 / git commit /
                          数据集"**逐项一致** —— 之后才谈得上执行。

**DEFERRED / BACKLOG**:`OfflineExecutor` 自身加一个 `frozen_manifest`
构造参数的"纵深防御"改动**不在本阶段范围内** —— 驱动入口是唯一能真正保证
"pre-provider validation"的一层,改共享执行器只会不必要地扩大影响面。

本模块**不执行**任何东西:导入它不读环境变量、不构造客户端、不发请求。
`run_d2d_pilot()` 默认 `allow_network=False` ⇒ 直接拒绝。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.evaluation.llm.pilot import (
    PILOT_BASELINES,
    PILOT_BEHAVIORS,
    PILOT_REPETITION_COUNT,
    PilotPlan,
    assert_frozen_pilot_manifest,
    build_candidate_manifest,
)
from app.evaluation.llm.protocol import ExperimentManifest
from app.evaluation.pilot_config import (
    d2d_plan,
    d2d_resource_budgets,
    default_pilot_config,
    pilot_identity_fields,
)


class D2DPilotDisabled(RuntimeError):
    """D-2d 正式执行需要**显式授权**。默认拒绝。"""


def build_d2d_manifest(
    *,
    git_commit: str,
    datasets: dict[str, dict[str, str]],
    created_at_utc: str | None = None,
    manifest_status: str = "frozen",
) -> ExperimentManifest:
    """构建 D-2d 清单(**默认 frozen**)。

    与候选清单的四处差异,每一处都对应一条已授权的控制器决定:

        plan                  `d2d_plan()`(`sdk_max_retries = 0` ⇒ 配置包络 324)
        provider / model / 端点 `pilot_identity_fields()`
        token_budget          已决定:`NOT_NUMERICALLY_BOUNDED_BY_PROTOCOL` + 输出分量包络
        cost_budget           已决定:`NO_NUMERIC_BOUND_NO_PROVENANCE`

    `manifest_status="frozen"` 时,`build_candidate_manifest` 会**当场**跑
    冻结闸门 —— 因此本函数不可能产出一个"标着 frozen 但其实还没准备好"的清单。
    """
    return build_candidate_manifest(
        git_commit=git_commit,
        datasets=datasets,
        plan=d2d_plan(),
        created_at_utc=created_at_utc,
        manifest_status=manifest_status,  # type: ignore[arg-type]
        token_budget=d2d_resource_budgets()["token_budget"],
        cost_budget=d2d_resource_budgets()["cost_budget"],
        **pilot_identity_fields(),
    )


def run_d2d_pilot(
    *,
    datasets: dict[str, dict[str, str]],
    manifest: ExperimentManifest,
    plan: PilotPlan,
    git_commit: str,
    api_key: str,
    experiment_id: str,
    workdir: Path | str,
    baselines: tuple[str, ...] = PILOT_BASELINES,
    behaviors: tuple[str, ...] = PILOT_BEHAVIORS,
    repetition_count: int = PILOT_REPETITION_COUNT,
    allow_network: bool = False,
) -> Any:
    """正式 D-2d 驱动。**第一条语句就是冻结闸门。**

    **准备与执行分离**:`datasets` / `manifest` / `plan` 是**必填**输入,
    由调用方在**执行之前**用准备工具构造好 ——

        datasets = load_datasets(workdir)
        plan     = d2d_plan()
        manifest = build_d2d_manifest(git_commit=..., datasets=datasets)

    本函数**不做**任何准备:它不读数据集、不建清单、**不取默认计划**。
    这样"闸门是第一件事"才是一句可以兑现的话 —— 而不是"闸门之前还有
    三步准备,而且那三步挑的输入恰好也是闸门要核对的对象"。

    闸门通过之前,本函数**不**构造任何 provider 客户端、**不**构造执行器、
    **不**写入任何试点产物、**不**准入任何实验单元。

    `allow_network=False`(默认)⇒ 闸门之后立即拒绝。这样即使有人拿着
    一份合法的冻结清单调用本函数,也不会意外地发出一封真实请求。
    """
    # === 冻结闸门:先于任何 provider 构造 / 执行器构造 / 产物写入 ===
    # 这是本函数的**第一条**语句 —— 见
    # tests/test_evaluation_llm/test_d2d_manifest_gate.py 的结构性断言。
    assert_frozen_pilot_manifest(
        manifest,
        datasets=datasets,
        expected_git_commit=git_commit,
        plan=plan,
        expected_identity=pilot_identity_fields(),
        expected_budgets=d2d_resource_budgets(),
    )

    if not allow_network:
        raise D2DPilotDisabled(
            "D-2d 正式执行需要显式 allow_network=True —— "
            "默认拒绝,避免任何一次意外网络出口。"
        )

    # ---- 以下只有闸门通过 **且** 显式授权之后才可能到达 ----
    config = default_pilot_config()
    from app.evaluation.llm.executor import OfflineExecutor  # noqa: PLC0415
    from app.evaluation.llm.offline_guard import NetworkEgressGuard  # noqa: PLC0415
    from app.evaluation.real_provider import make_llm_factory  # noqa: PLC0415

    factory = make_llm_factory(
        candidate=config.provider_candidate(),
        api_key=api_key,
        allow_network=allow_network,
        **config.model_kwargs(),
    )
    executor = OfflineExecutor(
        workdir=Path(workdir),
        experiment_id=experiment_id,
        plan=plan,
        baselines=baselines,
        behaviors=behaviors,
        repetition_count=repetition_count,
        manifest_digest=manifest.manifest_digest,
        guard=NetworkEgressGuard(strict=True),
        llm_factory=factory,
    )
    import asyncio  # noqa: PLC0415

    return asyncio.run(executor.run())
