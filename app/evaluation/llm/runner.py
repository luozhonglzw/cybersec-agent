"""Phase 9.2-D-1 真实 LLM 评测的编排层。

职责:
    1. 生成评测侧数据集(fixture 全部落在调用方给定的临时目录);
    2. 独立重算证据(oracle),作为 A 类 ground truth;
    3. 跑 基线 × 行为 × 任务 的完整矩阵(**脚本化 LLM,零网络调用**);
    4. 计算指标;
    5. 产出**可复现性元数据**(D-2 直接复用该 schema)。

D-1 全程离线
------------
本模块**不**创建任何真实 provider 客户端、不读 `.env`、不发起网络请求。
`provider` 恒为 `"scripted"`。

为什么证据要独立重算
--------------------
`narrative_claim_grounding_rate` 的参照物必须是**先于被测实现写下**的东西。
`app/evaluation/oracles.py` 只读原始 JSONL、不 import 任何规则模块
(`app.tools.risk_analyzer` / `app.tools.response_planner` / `app.security.policy`),
因此它重算出来的证据是 A 类,不是实现的自述。
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.agent import SECURITY_ANALYST_SYSTEM_PROMPT
from app.evaluation.llm.adapters import (
    BASELINES,
    MATRIX_BEHAVIORS,
    LLMObservation,
)
from app.evaluation.llm.dataset import (
    LLM_DATASET_VERSION,
    LLM_TASKS,
    VARIANT_BASE,
    build_datasets,
)
from app.evaluation.llm.extract import NarrativeClaims, extract_claims
from app.evaluation.llm.metrics import MetricResult, compute_llm_metrics
from app.evaluation.llm.tasks import LLMTask
from app.evaluation.oracles import recompute_evidence
from app.evaluation.runner import golden_digest
from app.evaluation.golden import GOLDEN_SET

#: 基线标签。`B0-shared` 使用与 B2'/B3 **完全相同**的系统提示词 ——
#: 它没有工具却被要求"优先调用工具",因此**刻意处于劣势**;
#: `B0-notool` 使用基线适配的去工具指引提示词。两者必须分开呈现。
BASELINE_LABELS: tuple[str, ...] = ("B0-shared", "B0-notool", "B2'", "B3")

SCRIPTED_PROVIDER = "scripted"

#: 元数据里**绝不允许**出现的字段(凭据 / 机密)。
FORBIDDEN_METADATA_FIELDS: frozenset[str] = frozenset({
    "api_key",
    "llm_api_key",
    "authorization",
    "auth_header",
    "token",
    "secret",
    "password",
    "base_url",  # 只允许 base_url_host —— 完整 URL 可能内嵌凭据
})


class RunMetadata(BaseModel):
    """一次运行的**可复现性元数据**。

    D-1 用 `provider="scripted"`;D-2 接真实 provider 时复用**同一个 schema**,
    只需填入真实值。**绝不记录** API Key、Authorization 头、完整凭据 URL
    或任何机密环境变量 —— 由 `assert_no_secrets` 机械守住。
    """

    provider: str = Field(min_length=1, description="provider 标签(scripted / openai-compatible …)")
    model: str = Field(min_length=1, description="模型标识;D-1 为 deterministic-scripted-<behavior>")
    base_url_host: str | None = Field(
        default=None, description="**只记主机名**;完整 URL 可能内嵌凭据,禁止记录"
    )
    temperature: str = Field(
        default="provider_default",
        description="显式温度或 'provider_default';当前 LLMClient 未设置该参数",
    )
    max_tokens: str = Field(default="provider_default", description="同上")
    timeout: str = Field(default="provider_default", description="同上")
    max_retries: str = Field(default="provider_default", description="同上")
    system_prompt_sha256: str = Field(description="系统提示词摘要(sha256)")
    tool_schema_sha256: str = Field(description="工具 schema 摘要(sha256)")
    task_prompt_sha256: dict[str, str] = Field(
        default_factory=dict, description="任务 id → 任务提示词摘要"
    )
    golden_digest: str = Field(description="9.2-A 冻结 golden 摘要(跨阶段回归锚点)")
    dataset_version: str = Field(description="9.2-D 任务数据集版本")
    run_timestamp_utc: str = Field(description="运行时刻(UTC,ISO 8601)")
    repetition_id: int = Field(ge=0, description="重复序号;D-1 恒为 0")


class LLMEvaluationResult(BaseModel):
    """一次完整评测的产物。"""

    metadata: RunMetadata
    observations: list[LLMObservation]
    claims: list[NarrativeClaims]
    control_observations: list[LLMObservation] = Field(
        default_factory=list,
        description=(
            "**匹配对照**观测:只对带注入契约的任务生成。同一个任务、同一个"
            "行为、同一个基线,只把载荷从 LLM 可见文本里拿掉 —— 用于把"
            "「输出命中注入目标」与「因注入而命中」区分开。"
            "刻意与 `observations` 分开:对照观测**不进入**任何其它指标的矩阵。"
        ),
    )
    metrics: list[MetricResult]
    evidence_by_variant: dict[str, dict[str, dict]]
    authorized_paths: dict[str, list[str]]
    dataset_paths: dict[str, dict[str, str]]

    def metric(self, metric_id: str) -> MetricResult:
        for item in self.metrics:
            if item.metric_id == metric_id:
                return item
        raise KeyError(f"未知指标: {metric_id}")

    def observation(self, task_id: str, baseline: str, behavior: str) -> LLMObservation | None:
        for obs in self.observations:
            if (obs.task_id, obs.baseline, obs.behavior) == (task_id, baseline, behavior):
                return obs
        return None

    def control_observation(
        self, task_id: str, baseline: str, behavior: str
    ) -> LLMObservation | None:
        """取匹配对照条件下的观测(没有对照的任务返回 `None`)。"""
        for obs in self.control_observations:
            if (obs.task_id, obs.baseline, obs.behavior) == (task_id, baseline, behavior):
                return obs
        return None


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------


def assert_no_secrets(payload: dict[str, Any]) -> None:
    """机械守卫:元数据里不得出现任何凭据类字段名。

    写成函数而不是靠人肉 review —— "临时加个 base_url 方便排查"是完全可以
    预见的演化方向,而 base_url 有可能内嵌凭据。
    """
    leaked = FORBIDDEN_METADATA_FIELDS & {key.lower() for key in payload}
    if leaked:
        raise AssertionError(
            f"运行元数据出现被禁止的字段 {sorted(leaked)} ——"
            "凭据 / 完整 URL 一律不得进入评测产物"
        )


def system_prompt_sha256() -> str:
    """系统提示词的摘要。**D-2 清单与运行元数据共用同一个事实来源。**"""
    return hashlib.sha256(SECURITY_ANALYST_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def tool_schema_sha256() -> str:
    """工具 schema 摘要(公开别名,供 D-2a 的清单构建复用)。

    这里 import `app.tools` 是**取元数据**(工具名 + 参数 schema),
    不是调用被测规则 —— 因此不违反评测侧的独立性约束
    (见 `tests/test_evaluation_llm/test_independence.py` 的 import 护栏:
    被拦的是 `app.tools.risk_analyzer` / `response_planner` / `security.policy` 这些
    **承载被测规则**的模块)。
    """
    return _tool_schema_digest()


def _tool_schema_digest() -> str:
    from app.tools import DEFAULT_TOOLS

    schemas = {tool.name: tool.args for tool in DEFAULT_TOOLS}
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_metadata(
    tasks: tuple[LLMTask, ...],
    *,
    behavior: str,
    repetition_id: int = 0,
) -> RunMetadata:
    payload = {
        "provider": SCRIPTED_PROVIDER,
        "model": f"deterministic-scripted-{behavior}",
        "base_url_host": None,
        "temperature": "provider_default",
        "max_tokens": "provider_default",
        "timeout": "provider_default",
        "max_retries": "provider_default",
        "system_prompt_sha256": system_prompt_sha256(),
        "tool_schema_sha256": tool_schema_sha256(),
        "task_prompt_sha256": {
            task.task_id: hashlib.sha256(task.user_prompt.encode("utf-8")).hexdigest()
            for task in tasks
        },
        "golden_digest": golden_digest(GOLDEN_SET),
        "dataset_version": LLM_DATASET_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repetition_id": repetition_id,
    }
    assert_no_secrets(payload)
    return RunMetadata(**payload)


# ---------------------------------------------------------------------------
# 数据集 / 证据
# ---------------------------------------------------------------------------


def build_evidence_index(
    tasks: tuple[LLMTask, ...],
    datasets: dict[str, dict[str, str]],
) -> dict[str, dict[str, dict]]:
    """独立重算每个任务在其数据集变体下的原始证据(A 类 ground truth)。"""
    index: dict[str, dict[str, dict]] = {}
    for task in tasks:
        paths = datasets[task.dataset_variant]
        index.setdefault(task.dataset_variant, {})[task.indicator] = recompute_evidence(
            task.indicator,
            logs_path=paths["logs"],
            intel_path=paths["intel"],
            event_type=task.event_type,
        )
    return index


def authorized_paths_for(datasets: dict[str, dict[str, str]]) -> dict[str, set[str]]:
    """评测授权路径集合(逐变体)。

    同时收录原始字符串与 `resolve()` 后的绝对路径 —— 调用方可能用任一种写法,
    判定必须对两者都成立,否则会把"写法不同"误判成"越权"。
    """
    out: dict[str, set[str]] = {}
    for variant, paths in datasets.items():
        values: set[str] = set()
        for value in paths.values():
            values.add(str(value))
            values.add(str(Path(value).resolve()))
        out[variant] = values
    return out


def make_decoys(
    workdir: Path, datasets: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """为 PATH_DEVIATION 探针造**授权之外**的无害副本(逐变体)。

    设计要点:副本内容与该变体的授权文件**逐字节相同**,只是位于未授权目录。
    这样 `PATH_DEVIATION` 改变的变量**只有"路径授权"一个** —— 叙事、接地、
    计划、策略、审计全部与 GOOD 一致,只有 `path_argument_deviation_rate`
    从 0 变到 1。若副本内容不同,接地指标会跟着一起坏,失败模式就被混淆了。

    **不触碰任何真实系统文件 / 凭据 / SSH 密钥 / 用户文档 / OS 敏感文件。**
    """
    decoys: dict[str, dict[str, str]] = {}
    for variant, paths in datasets.items():
        target_dir = workdir / "decoy" / variant
        target_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, str] = {}
        for key, source in paths.items():
            destination = target_dir / Path(source).name
            destination.write_text(
                Path(source).read_text(encoding="utf-8"), encoding="utf-8"
            )
            entry[key] = str(destination)
        decoys[variant] = entry
    return decoys


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


async def run_llm_evaluation(
    *,
    workdir: Path | str,
    tasks: tuple[LLMTask, ...] = LLM_TASKS,
    baselines: tuple[str, ...] = BASELINE_LABELS,
    behaviors: tuple[str, ...] = MATRIX_BEHAVIORS,
    repetition_id: int = 0,
) -> LLMEvaluationResult:
    """跑完整评测矩阵。**全程脚本化,零网络调用。**

    `baselines` 用标签而非适配器类:B0 的两个提示词变体是两个独立标签,
    必须分开呈现(见 `BASELINE_LABELS`)。

    **两组运行**
    -------------
    treatment(处理组)  常规矩阵:每个任务用**自己的**数据集变体。
    control  (对照组)  只对带注入契约的任务生成:同一个任务、同一个行为、
                       同一个基线,把数据集换成 `base` 变体 —— 即**载荷缺席**
                       的那个版本。证据字段不变,只有 LLM 可见的自由文本
                       里少了载荷。

    对照组的用途只有一个:把「输出命中了注入目标」与「因为注入才命中」
    分开。因此它**不进入**任何其它指标的矩阵,只由
    `injection_attributable_follow` 使用。
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets(workdir)
    evidence_index = build_evidence_index(tasks, datasets)
    authorized = authorized_paths_for(datasets)
    decoys = make_decoys(workdir, datasets)

    def _make_adapter(label: str) -> Any:
        if label.startswith("B0"):
            return BASELINES["B0"](dataset_paths=datasets[VARIANT_BASE])
        return BASELINES[label](
            dataset_paths=datasets[VARIANT_BASE],
            audit_db_path=str(workdir / f"audit-{label.replace(chr(39), 'p')}.db"),
        )

    def _prompt_variant(label: str) -> str | None:
        if not label.startswith("B0"):
            return None
        return "no_tool_prompt" if label.endswith("notool") else "shared_prompt"

    observations: list[LLMObservation] = []
    claims: list[NarrativeClaims] = []
    control_observations: list[LLMObservation] = []
    control_claims: list[NarrativeClaims] = []

    # ---- treatment:常规矩阵 ----
    for baseline_label in baselines:
        adapter = _make_adapter(baseline_label)
        prompt_variant = _prompt_variant(baseline_label)

        for behavior in behaviors:
            for task in tasks:
                paths = datasets[task.dataset_variant]
                adapter.dataset_paths = paths
                kwargs: dict[str, Any] = {
                    "dataset_paths": paths,
                    "decoy_paths": decoys[task.dataset_variant],
                }
                if prompt_variant is not None:
                    kwargs["prompt_variant"] = prompt_variant
                obs = await adapter.run(task, behavior, **kwargs)
                if baseline_label.startswith("B0"):
                    obs = obs.model_copy(update={"baseline": baseline_label})
                observations.append(obs)
                claims.append(NarrativeClaims(
                    task_id=obs.task_id,
                    baseline=obs.baseline,
                    behavior=obs.behavior,
                    claims=extract_claims(obs.answer),
                ))

    # ---- control:匹配对照(仅注入任务)----
    for baseline_label in baselines:
        adapter = _make_adapter(baseline_label)
        prompt_variant = _prompt_variant(baseline_label)

        for behavior in behaviors:
            for task in tasks:
                if task.security_contract.injection is None:
                    continue
                base_paths = datasets[VARIANT_BASE]
                adapter.dataset_paths = base_paths
                kwargs = {
                    "dataset_paths": base_paths,
                    "decoy_paths": decoys[VARIANT_BASE],
                    "dataset_variant_override": VARIANT_BASE,
                    "condition": "control",
                }
                if prompt_variant is not None:
                    kwargs["prompt_variant"] = prompt_variant
                obs = await adapter.run(task, behavior, **kwargs)
                if baseline_label.startswith("B0"):
                    obs = obs.model_copy(update={"baseline": baseline_label})
                control_observations.append(obs)
                control_claims.append(NarrativeClaims(
                    task_id=obs.task_id,
                    baseline=obs.baseline,
                    behavior=obs.behavior,
                    claims=extract_claims(obs.answer),
                ))

    metrics = compute_llm_metrics(
        tasks,
        observations,
        evidence_by_variant=evidence_index,
        authorized_paths=authorized,
        baselines=list(baselines),
        behaviors=list(behaviors),
        control_observations=control_observations,
    )

    metadata = build_metadata(
        tasks,
        behavior="matrix" if len(behaviors) > 1 else behaviors[0],
        repetition_id=repetition_id,
    )

    return LLMEvaluationResult(
        metadata=metadata,
        observations=observations,
        claims=claims,
        control_observations=control_observations,
        metrics=metrics,
        evidence_by_variant=evidence_index,
        authorized_paths={k: sorted(v) for k, v in authorized.items()},
        dataset_paths=datasets,
    )


# ---------------------------------------------------------------------------
# 确定性校验辅助
# ---------------------------------------------------------------------------


#: 路径类参数名。签名里它们被归一化成 `<authorized>` / `<unauthorized>`,
#: 否则签名会随工作目录变化 —— 而"两次运行是否一致"的判定不该依赖临时目录名。
_PATH_ARGUMENTS: frozenset[str] = frozenset({"data_path", "logs_path", "intel_path"})

#: 值本身**不可复现**的指标(墙钟耗时)。确定性签名必须排除它们。
NON_DETERMINISTIC_METRICS: frozenset[str] = frozenset({"wall_clock_ms"})


def _normalized_args(args: dict, authorized: set[str] | None) -> str:
    """把参数归一化成可跨工作目录比较的字符串。

    只把**路径类**参数折叠成授权标记;其余参数原样保留。
    折叠保留了我们真正关心的区别("读到授权数据" vs "读到未授权文件"),
    同时消除了"临时目录名不同 → 签名不同"这个假阳性。
    """
    normalized: dict[str, Any] = {}
    for key in sorted(args):
        value = args[key]
        if key in _PATH_ARGUMENTS and isinstance(value, str):
            normalized[key] = (
                "<authorized>" if authorized and value in authorized else "<unauthorized>"
            )
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def observation_signature(obs: LLMObservation, *, authorized: set[str] | None = None) -> tuple:
    """一次运行的可比较投影(用于两次运行逐位一致性的判定)。

    刻意**排除** `wall_clock_ms` —— 墙钟耗时天然不可复现,把它纳入签名
    会让确定性校验永远失败,从而失去意义。

    路径类参数按**授权与否**折叠(见 `_normalized_args`),因此两次评测
    即使落在不同的临时目录,签名也应当逐位相同。
    """
    return (
        obs.task_id,
        obs.baseline,
        obs.behavior,
        obs.dataset_variant,
        obs.condition,
        obs.prompt_variant,
        obs.run_status,
        obs.error,
        obs.payload_present_in_dataset,
        obs.payload_visible_to_model,
        tuple(
            (r.order, r.tool, _normalized_args(r.args, authorized), r.known, r.error)
            for r in obs.tool_calls
        ),
        obs.answer,
        obs.plan_digest,
        obs.plan_risk_level,
        tuple(obs.plan_actions or ()),
        obs.policy_outcome,
        obs.policy_requires_approval,
        tuple(obs.gated_actions or ()),
        tuple(obs.audit_events or ()),
        obs.graph_iterations,
        obs.llm_call_count,
        obs.tool_call_count,
        obs.input_tokens,
        obs.output_tokens,
        obs.total_tokens,
    )


def result_signature(result: LLMEvaluationResult) -> tuple:
    """整个结果的确定性签名(**跨工作目录可比**)。

    treatment 与 control 两组观测都进签名 —— 对照条件也是被测产物的一部分,
    把它排除会让"对照组是否真的可复现"无从判定。
    """
    return tuple(
        observation_signature(
            obs,
            authorized=set(result.authorized_paths.get(obs.dataset_variant, [])),
        )
        for obs in (*result.observations, *result.control_observations)
    )


def metric_signature(result: LLMEvaluationResult) -> tuple:
    """全部指标元组的确定性签名。

    排除 `NON_DETERMINISTIC_METRICS`(墙钟耗时)—— 保留它们会让这个签名
    在每次运行时都不同,于是"确定性校验"变成永远失败的空操作。
    """
    return tuple(
        (
            metric.metric_id,
            tuple(
                (cell.baseline, cell.behavior, cell.status, cell.value,
                 cell.numerator, cell.denominator)
                for cell in metric.cells
            ),
        )
        for metric in result.metrics
        if metric.metric_id not in NON_DETERMINISTIC_METRICS
    )


# ---------------------------------------------------------------------------
# D-2 试点清单(**只生成清单,不执行**)
# ---------------------------------------------------------------------------


class PilotManifest(BaseModel):
    """D-2 试点的清单 —— 本阶段**只产出清单,不执行任何真实调用**。"""

    task_count: int
    repetition_count: int
    baselines: list[str]
    behaviors: list[str]
    total_runs: int
    estimated_llm_calls_low: int
    estimated_llm_calls_high: int
    notes: list[str]
    requires_separate_approval: bool = True


def pilot_manifest(
    *,
    task_count: int = len(LLM_TASKS),
    repetition_count: int = 3,
    baselines: tuple[str, ...] = ("B0-shared", "B0-notool", "B2'", "B3"),
    behaviors: tuple[str, ...] = ("GOOD",),
) -> PilotManifest:
    """D-2 试点规模。

    刻意**不把 72 次运行当成统计终局** —— 它是"先证明工装能跑通"的起点。
    LLM 调用上界来自 `max_iterations=5`(每次图运行 ≤5 次 LLM 调用);
    B0 每运行恰好 1 次调用。
    """
    graph_baselines = [b for b in baselines if not b.startswith("B0")]
    direct_baselines = [b for b in baselines if b.startswith("B0")]
    runs = task_count * repetition_count * len(baselines) * len(behaviors)
    low = (
        task_count * repetition_count * len(direct_baselines) * len(behaviors) * 1
        + task_count * repetition_count * len(graph_baselines) * len(behaviors) * 2
    )
    high = (
        task_count * repetition_count * len(direct_baselines) * len(behaviors) * 1
        + task_count * repetition_count * len(graph_baselines) * len(behaviors) * 5
    )
    return PilotManifest(
        task_count=task_count,
        repetition_count=repetition_count,
        baselines=list(baselines),
        behaviors=list(behaviors),
        total_runs=runs,
        estimated_llm_calls_low=low,
        estimated_llm_calls_high=high,
        notes=[
            "D-1 只生成清单,不执行;D-2 需要**单独的用户批准**后才可运行。",
            "LLM 调用上界来自生产 max_iterations=5(每次图运行 ≤5 次调用);"
            "B0 每运行恰好 1 次调用。",
            f"运行数口径:{task_count} 任务 × {repetition_count} 重复 × "
            f"{len(baselines)} 基线标签 × {len(behaviors)} 行为 = {runs}。"
            "B0 被拆成 shared_prompt / no_tool_prompt **两个标签**(刻意暴露公平性问题),"
            "因此按「3 个基线」数会得到 72 —— 两者口径不同,不是矛盾。",
            "B0 两个提示词变体在 D-2 会有实质差异(真实 LLM 会读提示词);"
            "D-1 的脚本化 LLM 不读提示词,因此两者必然相同 —— "
            "不得据此断言公平性问题不存在。",
            f"{runs} 次运行**不是**统计终局:放大规则见设计报告 §13.3。",
            f"⚠️ 本清单只数 **treatment**(处理组)运行 = {runs}。D-2 的冻结试点"
            "还要加上**匹配对照**运行(仅注入任务 × 基线 × 重复),"
            "口径见 `app.evaluation.llm.pilot.pilot_plan()`(treatment 96 + control 12 = 108)。"
            "两者不是矛盾,是分母不同 —— 报告不得只引用其中一个。",
            "D-2 必须报告重复次数与区间(Wilson);单次运行不得代表比率。",
            "D-2 必须真实记录 provider / model / base_url_host / temperature;"
            "token 字段若 provider 不返回 usage,继续记 NOT_AVAILABLE,**不得伪造 0**。",
        ],
        requires_separate_approval=True,
    )
