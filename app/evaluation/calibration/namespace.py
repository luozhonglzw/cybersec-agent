"""标定与试点之间的**命名空间**与**文件系统**隔离(fail closed)。

为什么需要这一层
----------------
`experiment_id` 的既有守卫(见 `app/evaluation/llm/raw.py`)比较的是
**字符串相等**:不同 id 的记录不能被当作完成态。这条守卫是**正确的**,
但它只回答"这两个 id 一样吗",不回答"这个 id 属于哪个实验类别"。

后果很具体:若有人把**试点**的 `experiment_id` 交给一次标定运行,
相等性守卫会欣然放行 —— 标定产物于是**合法地**满足了试点的完成条件。
隔离就此消失,而所有产物看起来都正常。

所以本模块补上缺的那一半:**命名空间成员资格**。判据不是"相等",
而是"属于哪一类",且**任何歧义一律拒绝**。

两个前缀(稳定,不得重命名)
--------------------------
    d2c-cal-      标定
    d2d-pilot-    正式试点

拒绝规则(**全部 fail closed,没有 warning 模式**)
--------------------------------------------------
    * 空 / 纯空白 / 首尾带空白          → 拒绝
    * 不以任何已知前缀开头              → 拒绝
    * 前缀之后没有内容                  → 拒绝
    * 同时含有两个前缀(歧义)          → 拒绝
    * 用"给试点 id 加后缀"的方式造标定 id → 拒绝(它仍以试点前缀开头)

最后一条是刻意的:标定 id 必须**独立构造**,不得由试点 id 派生 ——
派生出来的 id 与试点 id 共享前缀,一旦有人放宽前缀检查就会静默串味。
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# 稳定前缀
# ---------------------------------------------------------------------------

#: 标定命名空间前缀。
CALIBRATION_ID_PREFIX = "d2c-cal-"

#: 正式试点命名空间前缀。
PILOT_ID_PREFIX = "d2d-pilot-"

#: 工作目录下的类别子目录名。两者**必须不同**,这是文件系统层的兜底。
CALIBRATION_WORKDIR_DIRNAME = "d2c-calibration"
PILOT_WORKDIR_DIRNAME = "d2d-pilot"

#: 仓库根(锚定本文件位置,**不用 CWD 相对路径**)。
REPO_ROOT = Path(__file__).resolve().parents[3]

#: 仓库 `data/` —— 冻结的本地 fixture 目录,任何运行都不得写入。
REPO_DATA_DIR = REPO_ROOT / "data"


class CalibrationNamespaceError(ValueError):
    """命名空间违规。**fail closed** —— 没有"仅告警"的降级路径。"""


class ExperimentNamespace(str, Enum):
    """一次运行的实验类别。"""

    CALIBRATION = "CALIBRATION"
    PILOT = "PILOT"


# ---------------------------------------------------------------------------
# experiment_id 分类
# ---------------------------------------------------------------------------


def classify_experiment_id(value: object) -> ExperimentNamespace:
    """判定 `experiment_id` 属于哪个命名空间。**歧义即拒绝。**

    这是唯一的判定入口:`assert_*` 两个函数都走它,因此不可能出现
    "标定侧放行、试点侧拒绝"这类分叉。
    """
    if not isinstance(value, str):
        raise CalibrationNamespaceError(
            f"experiment_id 必须是字符串,收到 {type(value).__name__}"
        )
    if not value:
        raise CalibrationNamespaceError("experiment_id 不得为空")
    if value != value.strip():
        raise CalibrationNamespaceError(
            f"experiment_id 首尾不得有空白:{value!r} —— 空白会让相等性判断失真"
        )

    starts_calibration = value.startswith(CALIBRATION_ID_PREFIX)
    starts_pilot = value.startswith(PILOT_ID_PREFIX)
    contains_calibration = CALIBRATION_ID_PREFIX in value
    contains_pilot = PILOT_ID_PREFIX in value

    if contains_calibration and contains_pilot:
        raise CalibrationNamespaceError(
            f"experiment_id {value!r} 同时含有两个命名空间前缀 —— "
            "歧义 id 一律拒绝,不得靠「看起来像哪个」来裁决"
        )

    if starts_calibration:
        if not value[len(CALIBRATION_ID_PREFIX):]:
            raise CalibrationNamespaceError(
                f"experiment_id {value!r} 只有前缀,没有阶段/时间戳"
            )
        return ExperimentNamespace.CALIBRATION

    if starts_pilot:
        if not value[len(PILOT_ID_PREFIX):]:
            raise CalibrationNamespaceError(
                f"experiment_id {value!r} 只有前缀,没有时间戳"
            )
        return ExperimentNamespace.PILOT

    raise CalibrationNamespaceError(
        f"experiment_id {value!r} 不以任何已知前缀开头 —— "
        f"合法前缀为 {CALIBRATION_ID_PREFIX!r} / {PILOT_ID_PREFIX!r}"
    )


def assert_calibration_experiment_id(value: object) -> str:
    """断言这是**标定**命名空间的 id。

    在**构造执行器/写入任何产物之前**调用。用"给试点 id 加后缀"的方式
    造出来的 id 会在这里被拒 —— 它仍然以试点前缀开头。
    """
    namespace = classify_experiment_id(value)
    if namespace is not ExperimentNamespace.CALIBRATION:
        raise CalibrationNamespaceError(
            f"experiment_id {value!r} 属于 {namespace.value} 命名空间,"
            f"但这里要求 {ExperimentNamespace.CALIBRATION.value} —— "
            "标定 id 不得占用试点命名空间,也不得由试点 id 派生"
        )
    return str(value)


def assert_pilot_experiment_id(value: object) -> str:
    """断言这是**试点**命名空间的 id。"""
    namespace = classify_experiment_id(value)
    if namespace is not ExperimentNamespace.PILOT:
        raise CalibrationNamespaceError(
            f"experiment_id {value!r} 属于 {namespace.value} 命名空间,"
            f"但这里要求 {ExperimentNamespace.PILOT.value} —— "
            "试点 id 不得占用标定命名空间"
        )
    return str(value)


def calibration_experiment_id(*, stage: str, stamp: str) -> str:
    """构造一个**独立**的标定 id(不由试点 id 派生)。

    `stamp` 必须由调用方提供(通常是 UTC 时间戳),本函数**不读时钟** ——
    隐式时间源会让同一份配置产出不同 id,破坏可复现性。
    """
    stage_token = "".join(
        ch if (ch.isalnum() or ch in "-_") else "-" for ch in stage.strip()
    ).strip("-")
    stamp_token = "".join(
        ch if (ch.isalnum() or ch in "-_") else "-" for ch in stamp.strip()
    ).strip("-")
    if not stage_token:
        raise CalibrationNamespaceError("stage 不得为空")
    if not stamp_token:
        raise CalibrationNamespaceError("stamp 不得为空")
    value = f"{CALIBRATION_ID_PREFIX}{stage_token}-{stamp_token}"
    return assert_calibration_experiment_id(value)


# ---------------------------------------------------------------------------
# 文件系统隔离
# ---------------------------------------------------------------------------


def assert_outside_repo_data(path: Path | str) -> Path:
    """拒绝任何落在仓库 `data/` 里的路径。"""
    resolved = Path(path).resolve()
    if resolved == REPO_DATA_DIR or REPO_DATA_DIR in resolved.parents:
        raise CalibrationNamespaceError(
            f"路径 {resolved} 落在仓库 data/ 下 —— 那是冻结的本地 fixture 目录,"
            "任何运行都不得写入"
        )
    return resolved


def calibration_root(workdir_root: Path | str) -> Path:
    """标定工作目录**根**。"""
    return assert_outside_repo_data(Path(workdir_root) / CALIBRATION_WORKDIR_DIRNAME)


def pilot_root(workdir_root: Path | str) -> Path:
    """试点工作目录**根**。"""
    return assert_outside_repo_data(Path(workdir_root) / PILOT_WORKDIR_DIRNAME)


def calibration_stage_workdir(workdir_root: Path | str, experiment_id: str) -> Path:
    """某个标定阶段的**独立**工作目录。

    每个阶段一个目录:失败阶段的半成品因此不可能被下一个阶段续跑。
    """
    assert_calibration_experiment_id(experiment_id)
    root = calibration_root(workdir_root)
    candidate = (root / experiment_id).resolve()
    # 目录穿越防御:`experiment_id` 已被前缀规则限制,但仍显式核验包含关系。
    if root.resolve() != candidate and root.resolve() not in candidate.parents:
        raise CalibrationNamespaceError(
            f"标定工作目录 {candidate} 越出了标定根 {root}"
        )
    return assert_outside_repo_data(candidate)


def assert_workdirs_disjoint(calibration_workdir: Path | str, pilot_workdir: Path | str) -> None:
    """标定与试点的工作目录必须**互不包含**。

    这是 id 检查之外的第二道防线:即使有人绕过命名空间校验,
    两条路径也不可能互相覆盖对方的产物。
    """
    calibration_path = Path(calibration_workdir).resolve()
    pilot_path = Path(pilot_workdir).resolve()
    if calibration_path == pilot_path:
        raise CalibrationNamespaceError(
            f"标定与试点共用工作目录 {calibration_path} —— 产物会互相污染"
        )
    if calibration_path in pilot_path.parents or pilot_path in calibration_path.parents:
        raise CalibrationNamespaceError(
            f"标定工作目录 {calibration_path} 与试点工作目录 {pilot_path} "
            "互相包含 —— 必须完全分离"
        )
