"""Phase 9.2-A 评测运行器 —— 编排 dataset → adapters → oracles → metrics。

运行器契约(runner contract)
---------------------------
输入(全部显式传入,没有隐式默认路径):
    logs_path    原始日志 JSONL
    intel_path   原始威胁情报 JSONL
    workdir      本次运行的**私有目录**(审计数据库、合成数据都写在它下面)
    adapter_ids  要跑的基线集合,默认 B1/B2/B3

输出:
    EvaluationResult —— 观测 + 指标 + 变形关系 + 输入摘要(可序列化为 JSON)

可复现性:
    * 不读系统时间做判定(只有审计记录内部带时间戳,且不参与任何指标);
    * 不读环境变量;
    * 不写仓库目录 —— `workdir` 由调用方给出(测试给 tmp_path);
    * 同一输入重复运行,观测逐字段一致(由 deterministic_repeat_consistency 验证)。

hermetic 要求:
    本模块**从不**引用 `data/` 下的任何路径。`logs_path` / `intel_path` 缺失时
    由调用方负责(测试从 tmp_path 造数据),因此在一个没有 `data/` 目录的 CWD
    下也能完整跑通。

变形关系(Metamorphic Relations)
--------------------------------
数值断言会随实现常量漂移而失效,关系断言不会。本模块在 `workdir` 下构造
**合成数据**并检查三条关系:

    MR1 单调性    失败登录更多 → 风险分数不得下降
    MR2 支配性    同一份日志证据下:可信 ≤ 无情报 ≤ 恶意
    MR3 确定性    同一输入重复执行 → 观测一致(由 metrics.determinism_metric 承担)

这三条都**不引用任何权重常量**,因此"把权重从 30 改成 10"不会让它们变红,
而"逻辑倒置"会被抓住 —— 这正是它们与 D 类自洽检查的分工。
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import structlog
from pydantic import BaseModel, Field

from app.evaluation import oracles
from app.evaluation.adapters import ADAPTERS, BaseAdapter
from app.evaluation.cases import GoldenSet
from app.evaluation.golden import GOLDEN_SET
from app.evaluation.metrics import MetricResult, compute_metrics, determinism_metric

logger = structlog.get_logger(__name__)

#: 变形关系使用的合成指标(刻意用文档保留网段,且不在种子数据里)
_SYNTHETIC_IP = "203.0.113.200"
_SYNTHETIC_TS = "2026-09-10T08:00:00Z"


class RelationResult(BaseModel):
    """一条变形关系的检查结果。"""

    relation_id: str
    title: str
    definition: str
    status: str  # holds / violated / not_evaluable
    detail: str
    evidence: dict = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """一次完整评测的全部产物。"""

    golden_version: str
    golden_digest: str
    dataset_digests: dict[str, str]
    adapter_ids: list[str]
    observations: list[dict]
    metrics: list[MetricResult]
    relations: list[RelationResult]
    audit_db_paths: list[str]
    workdir: str

    def metrics_by_category(self, category: str) -> list[MetricResult]:
        return [m for m in self.metrics if m.category == category]

    def metric(self, metric_id: str) -> MetricResult:
        for item in self.metrics:
            if item.metric_id == metric_id:
                return item
        raise KeyError(f"未知指标 {metric_id}")


# ---------------------------------------------------------------------------
# 数据集工具
# ---------------------------------------------------------------------------


#: fixture 序列化的**规范换行序列**。
#:
#: 冻结摘要 `FROZEN_TASKSET_DIGEST` 是在 **CRLF** 字节上标定的。文本模式写文件会把
#: `\n` 翻译成 `os.linesep`(Windows → CRLF,Linux → LF),于是同一份逻辑 fixture 在
#: 不同平台上得到不同字节 → 不同 SHA256 → 不同 taskset digest,冻结常量只能在
#: 一个平台上成立。这里把规范换行**显式**固定为 CRLF。
FIXTURE_NEWLINE = "\r\n"


def write_fixture_lines(path: Path | str, lines: Iterable[str]) -> None:
    """以**规范字节**(UTF-8 + `FIXTURE_NEWLINE`,禁用平台换行翻译)写出 fixture。

    这是 fixture 序列化的**唯一边界**:所有生成器都必须经过它。绕开它,就等于把
    "同一 fixture → 同一字节 → 同一摘要"这条不变量降级成一句没有强制力的注释。
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for line in lines:
            handle.write(line + FIXTURE_NEWLINE)


def write_seed_dataset(dest_dir: Path | str) -> dict[str, Path]:
    """把 Phase 2 的**确定性**种子数据写进 dest_dir,返回两个 JSONL 的路径。

    刻意延迟 import `scripts/*`:它们是 Phase 2 的数据生成脚本,不是运行时依赖。
    延迟 import 让 `app.evaluation` 的导入链里没有 `scripts`,同时仍给需要造数据
    的调用方(以及评测测试)留一个入口。

    这两个生成器在 Phase 2 就被写成"固定 seed + 固定基准时间 → 逐字节一致",
    因此每次评测拿到的输入完全相同 —— 这是可复现评测的前提。
    """
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scripts.seed_logs import generate_events  # noqa: PLC0415 - 刻意延迟
    from scripts.seed_threat_intel import _records  # noqa: PLC0415 - 刻意延迟

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    logs_path = dest / "security_events.jsonl"
    write_fixture_lines(
        logs_path, (event.model_dump_json() for event in generate_events())
    )

    intel_path = dest / "threat_intel.jsonl"
    write_fixture_lines(
        intel_path, (record.model_dump_json() for record in _records())
    )

    return {"logs": logs_path, "intel": intel_path}


def _file_digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def golden_digest(golden_set: GoldenSet) -> str:
    """golden set 的规范化摘要 —— 冻结的机械凭据。

    任何用例内容变化都会改变这个摘要。把它写进报告,是为了让"这份结果对应
    哪一版用例"可被独立复核。
    """
    canonical = json.dumps(
        golden_set.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


async def run_evaluation(
    *,
    logs_path: Path | str,
    intel_path: Path | str,
    workdir: Path | str,
    adapter_ids: Sequence[str] = ("B1", "B2", "B3"),
    golden_set: GoldenSet | None = None,
    llm: Any | None = None,
    run_relations: bool = True,
) -> EvaluationResult:
    """跑一次完整评测。

    每次调用都**新建**适配器实例(B3 因此拿到全新的 checkpointer 与审计库)。
    复用实例会让 thread_id 撞车 —— LangGraph 对重复 thread_id 的语义是覆盖,
    这会把上一次暂停的线程悄悄顶掉。
    """
    golden = golden_set or GOLDEN_SET
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    audit_db_path = work / "audit.db"
    adapters: list[BaseAdapter] = [
        ADAPTERS[adapter_id](
            logs_path=str(logs_path),
            intel_path=str(intel_path),
            audit_db_path=str(audit_db_path) if adapter_id == "B3" else None,
            llm=llm,
        )
        for adapter_id in adapter_ids
    ]

    observations: list[dict] = []
    for case in golden.cases:
        expected = oracles.recompute_evidence(
            case.indicator, logs_path=logs_path, intel_path=intel_path
        )
        for adapter in adapters:
            observation = await adapter.run(case)
            payload = observation.model_dump(mode="json")
            # oracle 重算结果随观测一起传递,使 metrics 无需接触文件系统
            payload["_oracle_evidence"] = expected
            observations.append(payload)

    metrics = compute_metrics(golden, observations, adapter_ids=list(adapter_ids))

    relations: list[RelationResult] = []
    if run_relations:
        relations = await _run_metamorphic_relations(
            workdir=work / "relations",
            adapter_cls=ADAPTERS["B1"],
            llm=llm,
        )

    result = EvaluationResult(
        golden_version=golden.version,
        golden_digest=golden_digest(golden),
        dataset_digests={
            "logs": _file_digest(logs_path),
            "intel": _file_digest(intel_path),
        },
        adapter_ids=list(adapter_ids),
        observations=observations,
        metrics=metrics,
        relations=relations,
        audit_db_paths=[str(audit_db_path)] if "B3" in adapter_ids else [],
        workdir=str(work),
    )
    logger.info(
        "evaluation_completed",
        cases=len(golden.cases),
        adapters=len(adapters),
        metrics=len(metrics),
    )
    return result


# ---------------------------------------------------------------------------
# 变形关系
# ---------------------------------------------------------------------------


def _synthetic_event(event_type: str, severity: str, status: str, index: int) -> dict:
    """构造一条**最小合法**的 LogEvent 字典。

    字段刻意保持最少:变形关系只关心"失败登录条数"与"情报标记",
    其它字段不参与关系断言。
    """
    return {
        "timestamp": _SYNTHETIC_TS,
        "event_type": event_type,
        "source": "sshd",
        "source_ip": _SYNTHETIC_IP,
        "destination_ip": "10.0.1.20",
        "source_port": 50000,
        "destination_port": 22,
        "username": f"u{index:02d}",
        "action": None,
        "status": status,
        "severity": severity,
        "message": f"synthetic {event_type} #{index}",
    }


def _synthetic_intel(indicator: str, malicious: bool) -> dict:
    return {
        "indicator": indicator,
        "indicator_type": "ip",
        "malicious": malicious,
        "confidence": 90,
        "severity": "high" if malicious else "info",
        "tags": ["synthetic"],
        "source": "evaluation-fixture",
        "first_seen": "2026-09-01T00:00:00Z",
        "last_seen": "2026-09-02T00:00:00Z",
        "description": "评测合成情报(变形关系专用)",
    }


def _write_jsonl(path: Path, rows: Sequence[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _relation_view(observation: dict) -> dict:
    """变形关系用的扁平视图:把 evidence 里的失败登录数提到顶层。

    变形关系的谓词写在 `oracles.py` 里,而那里**不允许** import 任何生产类型
    (包括 `RiskEvidence`)。所以由 runner 负责把嵌套结构拍平,oracle 只看到
    普通 dict —— 依赖方向保持单向。
    """
    evidence = observation.get("evidence") or {}
    return {
        "failed_login_count": evidence.get("failed_login_count"),
        "score": observation.get("score"),
        "risk_level": observation.get("risk_level"),
    }


async def _run_metamorphic_relations(
    *, workdir: Path, adapter_cls: type[BaseAdapter], llm: Any | None
) -> list[RelationResult]:
    """构造合成数据并检查 MR1 / MR2。

    MR3(确定性)由 `metrics.determinism_metric` 在两次完整评测之间承担,
    因为它需要的是"整轮重复",而不是单点重复。
    """
    workdir.mkdir(parents=True, exist_ok=True)

    low_logs = _write_jsonl(
        workdir / "low.jsonl",
        [_synthetic_event("login_failed", "low", "failed", i) for i in range(3)],
    )
    high_logs = _write_jsonl(
        workdir / "high.jsonl",
        [_synthetic_event("login_failed", "medium", "failed", i) for i in range(25)],
    )
    empty_intel = _write_jsonl(workdir / "intel_empty.jsonl", [])
    malicious_intel = _write_jsonl(
        workdir / "intel_malicious.jsonl", [_synthetic_intel(_SYNTHETIC_IP, True)]
    )
    trusted_intel = _write_jsonl(
        workdir / "intel_trusted.jsonl", [_synthetic_intel(_SYNTHETIC_IP, False)]
    )

    class _Probe:
        """变形关系只关心分数与失败登录数,用一个最小 case 壳喂给适配器。"""

        case_id = "MR-PROBE"
        indicator = _SYNTHETIC_IP

    probe = _Probe()

    async def _observe(logs: Path, intel: Path) -> dict:
        adapter = adapter_cls(logs_path=str(logs), intel_path=str(intel), llm=llm)
        return _relation_view((await adapter.run(probe)).model_dump(mode="json"))

    low = await _observe(low_logs, empty_intel)
    high = await _observe(high_logs, empty_intel)
    trusted = await _observe(high_logs, trusted_intel)
    malicious = await _observe(high_logs, malicious_intel)

    mr1 = oracles.relation_monotonic_score(low, high)
    mr2 = oracles.relation_trusted_dominance(trusted, high, malicious)

    return [
        RelationResult(
            relation_id="MR1_monotonicity",
            title="失败登录更多 → 风险分数不得下降",
            definition=(
                "同一指标,合成 3 条 vs 25 条 login_failed,断言 score(3) ≤ score(25)。"
                "关系型断言,不引用任何权重常量。"
            ),
            status=mr1.status,
            detail=mr1.detail,
            evidence={
                "low": {"failed": low.get("failed_login_count"), "score": low.get("score")},
                "high": {"failed": high.get("failed_login_count"), "score": high.get("score")},
            },
        ),
        RelationResult(
            relation_id="MR2_trusted_dominance",
            title="同一份日志证据下:可信 ≤ 无情报 ≤ 恶意",
            definition=(
                "固定 25 条 login_failed,分别用\"情报标注恶意\"/\"无情报\"/\"情报标注可信\""
                "三份情报库跑同一条流水线,断言 score 的序关系。"
                "关系型断言,不引用任何权重常量。"
            ),
            status=mr2.status,
            detail=mr2.detail,
            evidence={
                "trusted": {"malicious": False, "score": trusted.get("score")},
                "absent": {"malicious": None, "score": high.get("score")},
                "malicious": {"malicious": True, "score": malicious.get("score")},
            },
        ),
    ]


__all__ = [
    "EvaluationResult",
    "RelationResult",
    "determinism_metric",
    "golden_digest",
    "run_evaluation",
    "write_seed_dataset",
]
