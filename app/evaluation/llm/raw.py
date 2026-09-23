"""原始结果持久化 + 崩溃/续跑语义(**离线**)。

三条硬规则
----------
1. **追加式 + flush + fsync。** 没有 fsync 的"写完了"只是写进了进程缓冲 ——
   崩溃后拿不到,而"拿不到"的实验数据比没有数据更糟:它会让人以为
   那个单元"跑了但结果不好"。
2. **写前过机密模式守卫。** 只匹配**真实凭据形态**(PEM / AWS / GitHub /
   `sk-` / Slack / JWT / Bearer),不按关键词扫 ——
   合成 fixture 里本来就有 `Failed password for ...`、`GET /.env`
   这类**攻击叙事文本**,按关键词扫会大量误报,最终被人直接关掉。
3. **摘要 sidecar。** 文件级 sha256 让"事后被改过"可被机械检出。

崩溃/续跑语义
-------------
    一个**已有完整记录**的单元永不重跑。

这是本模块唯一的不可协商条款。它的意义在于:**续跑的选择依据是
"记录是否完整",而不是"结果好不好"**。一旦允许按结果选择重跑,
实验就从"测量"退化成"挑样本",而所有下游统计都失去意义。

未执行(完全没有记录)的单元当然要跑 —— 那是恢复,不是挑样本。
两者在 `resume_eligibility()` 里被**分开报告**,不合并成一个布尔值。
"""
import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from app.evaluation.llm.budget import UNKNOWN
from app.evaluation.llm.protocol import canonical_json, sha256_hex

# ---------------------------------------------------------------------------
# 机密模式守卫
# ---------------------------------------------------------------------------

#: **真实凭据形态**。刻意不写 `password` / `token` / `.env` 这类关键词:
#: 合成 fixture 里的攻击叙事文本会被大量误报,而一个总在误报的守卫
#: 最后的结局一定是被关掉。
SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pem_private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("aws_access_key_id", r"AKIA[0-9A-Z]{16}"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9]{20,}"),
    ("openai_style_key", r"sk-[A-Za-z0-9]{20,}"),
    ("slack_token", r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ("bearer_header", r"Bearer\s+[A-Za-z0-9._\-]{20,}"),
    ("authorization_header", r"(?i)authorization\s*:"),
)

_COMPILED_SECRETS = tuple((name, re.compile(pattern)) for name, pattern in SECRET_PATTERNS)


def _iter_strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _iter_strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _iter_strings(value)


def find_secret_patterns(payload: Any) -> list[str]:
    """返回命中的**模式名**(去重、有序)。空列表 = 干净。"""
    hits: list[str] = []
    for text in _iter_strings(payload):
        for name, pattern in _COMPILED_SECRETS:
            if name not in hits and pattern.search(text):
                hits.append(name)
    return hits


class SecretLeakError(AssertionError):
    """原始记录里出现了凭据形态 —— 拒绝写入并中止。"""


def assert_no_secret_patterns(payload: Any) -> None:
    hits = find_secret_patterns(payload)
    if hits:
        raise SecretLeakError(
            f"原始记录命中凭据模式 {hits} —— 拒绝写入。"
            "凭据 / Authorization 头 / 含凭据 URL / cookie 一律不得进入评测产物。"
        )


# ---------------------------------------------------------------------------
# 记录状态与续跑
# ---------------------------------------------------------------------------


class RecordStatus(str, Enum):
    """一个实验单元的记录状态。

    `INCOMPLETE` 是**给基础设施失败用的**:进程在写完整记录之前挂了。
    它**不**表示"结果不好"。
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class ResumeEligibility(str, Enum):
    """续跑判定。刻意是**三态**而不是布尔值。"""

    FROZEN = "FROZEN"           # 已有完整记录 —— 永不重跑
    RESUMABLE = "RESUMABLE"     # 有记录但标记为 incomplete
    NEVER_EXECUTED = "NEVER_EXECUTED"  # 完全没有记录 —— 属于恢复,不是重跑


def resume_eligibility(record: "RawRecord | None") -> ResumeEligibility:
    if record is None:
        return ResumeEligibility.NEVER_EXECUTED
    if record.record_status == RecordStatus.COMPLETE:
        return ResumeEligibility.FROZEN
    return ResumeEligibility.RESUMABLE


def is_resumable(record: "RawRecord | None") -> bool:
    """只有 `FROZEN` 不可续跑。**结果好坏不参与判定。**"""
    return resume_eligibility(record) is not ResumeEligibility.FROZEN


def build_resume_metadata(
    *, origin_experiment_id: str, resumed_from_index: int
) -> dict[str, Any]:
    return {
        "origin_experiment_id": origin_experiment_id,
        "resumed_from_index": resumed_from_index,
    }


# ---------------------------------------------------------------------------
# 原始记录
# ---------------------------------------------------------------------------


class RawRecord(BaseModel):
    """一次运行的原始记录(**逐字段自解释**)。

    为什么把 `metric_inputs` 与 `metric_outputs` 都存下来:
    只存输出的话,事后无法回答"这个数是怎么来的";只存输入的话,
    无法回答"当时算出来是多少"。两者都存,派生层才能被独立重算并比对。
    """

    # ---- 身份 ----
    run_id: str
    experiment_id: str
    protocol_version: str
    manifest_digest: str
    task_id: str
    condition: str
    baseline_label: str
    dataset_variant: str
    repetition_id: int = Field(ge=1)
    execution_index: int = Field(ge=0)

    # ---- 四个计数器(语义见 budget.py)----
    experimental_run_attempts: int = Field(
        default=1, description="首试点恒为 1(HARNESS_LEVEL_RETRY = 0)",
    )
    logical_llm_invocations: int = Field(
        default=0, description="**与上一项不是同一个量**:图条件可能 1~5",
    )
    provider_http_attempts: int | str = Field(
        default=UNKNOWN, description="可观测时为 int,否则 UNKNOWN",
    )

    # ---- provider 元数据(D-2a 只能是 provider_default / UNKNOWN)----
    provider: str = "scripted"
    model: str = ""
    provider_reported_model_id: str | None = None
    base_url_host_sha256: str | None = Field(
        default=None, description="只记主机名摘要;完整 URL 可能内嵌凭据",
    )
    provider_default_parameters: dict[str, Any] = Field(default_factory=dict)

    # ---- 提示词 / 输入摘要 ----
    system_prompt_sha256: str = ""
    user_prompt_sha256: str = ""
    tool_schema_sha256: str = ""
    full_input_sha256: str = ""

    # ---- 工具轨迹 ----
    tool_call_trace: list[dict] = Field(default_factory=list)

    # ---- 叙事 ----
    final_narrative: str = ""
    final_narrative_sha256: str = ""

    # ---- 指标 ----
    metric_inputs: dict = Field(default_factory=dict)
    metric_outputs: dict = Field(default_factory=dict)

    # ---- 用量 / 性能 ----
    usage: dict = Field(default_factory=dict)
    latency_ms: float = 0.0

    # ---- 分类 ----
    failure: dict | None = None
    exposure: dict = Field(default_factory=dict)

    # ---- 状态 ----
    record_status: RecordStatus = RecordStatus.COMPLETE
    resume: dict | None = None
    timestamp_start_utc: str = ""
    timestamp_end_utc: str = ""

    def identity_key(self) -> tuple[str, str, str, int]:
        return (self.condition, self.task_id, self.baseline_label, self.repetition_id)

    # ---- 暴露(与 `LLMObservation` **同一套判定**,便于两处共用)----

    @property
    def payload_present_in_dataset(self) -> bool:
        return bool(self.exposure.get("payload_present_in_dataset", False))

    @property
    def payload_visible_to_model(self) -> bool | None:
        """`None` = 不可判定。**不把不可判定折成 False** —— 那会把"不知道"说成"没发生"。"""
        value = self.exposure.get("payload_visible_to_model", None)
        return value if isinstance(value, bool) else None

    @property
    def exposed(self) -> bool:
        """载荷真的进入过模型可见上下文 —— 归因的必要条件。"""
        return bool(self.payload_present_in_dataset and self.payload_visible_to_model)


class RawWriter:
    """追加式 JSONL 写入器 + 摘要 sidecar。"""

    def __init__(self, path: Path | str, *, experiment_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id

    @property
    def sidecar_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".sha256")

    def write(self, record: RawRecord) -> None:
        payload = record.model_dump(mode="json")
        assert_no_secret_patterns(payload)
        line = canonical_json(payload)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_sidecar(self) -> str:
        """写出文件级摘要。空运行(全部单元已冻结)时文件可能尚未创建 —— 补一个空文件。

        "空运行也要有 sidecar" 不是洁癖:没有 sidecar 时 `verify_sidecar()`
        返回 `False`,与"文件被改过"无法区分。
        """
        if not self.path.exists():
            self.path.touch()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.sidecar_path.write_text(digest + "\n", encoding="utf-8")
        return digest

    def verify_sidecar(self) -> bool:
        """sidecar 与实际文件是否一致 —— 检出事后改动。"""
        if not self.sidecar_path.exists():
            return False
        recorded = self.sidecar_path.read_text(encoding="utf-8").strip()
        return recorded == hashlib.sha256(self.path.read_bytes()).hexdigest()

    def read_all(self) -> list[RawRecord]:
        if not self.path.exists():
            return []
        records: list[RawRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(RawRecord(**json.loads(line)))
        return records


def index_by_unit(records: Iterable[RawRecord]) -> dict[tuple[str, str, str, int], RawRecord]:
    """按实验单元身份索引。同键后写覆盖先写(不应发生;发生了说明有重复)。"""
    return {record.identity_key(): record for record in records}


def digest_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: 路径类工具参数名。签名里它们被**归一化** —— 否则签名会随工作目录变化,
#: 而"两次运行是否一致"的判定不该依赖临时目录名。
#: 与 D-1 的 `runner.observation_signature` 同源,处理方式保持一致。
PATH_ARGUMENTS: frozenset[str] = frozenset({"data_path", "logs_path", "intel_path"})


def _normalized_trace(
    trace: list[dict], authorized_paths: set[str] | None
) -> list[dict]:
    out: list[dict] = []
    for entry in trace:
        args = entry.get("args") or {}
        normalized: dict[str, Any] = {}
        for key in sorted(args):
            value = args[key]
            if key in PATH_ARGUMENTS and isinstance(value, str):
                if authorized_paths is None:
                    normalized[key] = "<path>"
                else:
                    normalized[key] = (
                        "<authorized>" if value in authorized_paths else "<unauthorized>"
                    )
            else:
                normalized[key] = value
        out.append({**entry, "args": normalized})
    return out


def record_signature(
    record: "RawRecord", *, authorized_paths: set[str] | None = None
) -> str:
    """跨运行可比的**确定性投影**。

    排除 / 归一化两样东西:

    `latency_ms`      墙钟耗时天然不可复现。把它纳入投影会让"两次运行是否一致"
                      的判定永远失败,于是那条判定变成空操作。耗时仍然被**记录**
                      (它是真实测量),只是不参与可复现性判定 ——
                      与 `runner.NON_DETERMINISTIC_METRICS` 的处理一致。

    路径类工具参数      绝对路径随工作目录变化。未提供 `authorized_paths` 时一律
                      折叠成 `<path>`;提供时折叠成 `<authorized>` / `<unauthorized>`
                      (保留"读到授权数据 / 读到未授权文件"这一区别)。
                      ⚠️ 用 `<path>` 折叠的签名对**路径授权差异不敏感**,
                      因此它只用于不涉及路径偏差的比对(D-2a 试点只跑 GOOD 行为);
                      路径授权的差异由 D-1 的 `path_argument_deviation_rate`
                      与 `observation_signature` 负责。
    """
    payload = record.model_dump(mode="json")
    payload.pop("latency_ms", None)
    payload["tool_call_trace"] = _normalized_trace(
        payload.get("tool_call_trace") or [], authorized_paths
    )
    return canonical_json(payload)


def narrative_sha256(text: str) -> str:
    return sha256_hex(text)
