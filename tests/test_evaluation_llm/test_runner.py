"""编排层:确定性、D-2 清单、离线保证。

D-1 的核心承诺是"这套工装可以重复跑出同一个结果"。本文件把它做成可验证的
事实,而不是一句声明 —— 包括**跨工作目录**的比对(临时目录名不同不应改变结论)。
"""
import asyncio
import ast
from pathlib import Path

import pytest

from app.evaluation.llm import (
    LLM_TASKS,
    PilotManifest,
    metric_signature,
    pilot_manifest,
    result_signature,
    run_llm_evaluation,
)
from app.evaluation.llm.runner import NON_DETERMINISTIC_METRICS

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_PKG = REPO_ROOT / "app" / "evaluation" / "llm"


# ---------------------------------------------------------------------------
# 1. 确定性
# ---------------------------------------------------------------------------


def test_double_run_is_bit_identical_across_workdirs(matrix, tmp_path):
    """两次完整评测(不同临时目录)必须逐位一致。

    路径类参数在签名里被折叠成"授权 / 未授权",因此临时目录名不影响结论 ——
    这正是要检验的性质:结果只取决于脚本化行为,不取决于环境。
    """
    second = asyncio.run(run_llm_evaluation(workdir=tmp_path / "second"))
    assert len(second.observations) == len(matrix.observations)
    assert result_signature(second) == result_signature(matrix)
    assert metric_signature(second) == metric_signature(matrix)


def test_signature_ignores_wall_clock(matrix):
    """墙钟耗时天然不可复现,把它纳入签名会让确定性校验永远失败。"""
    assert "wall_clock_ms" in NON_DETERMINISTIC_METRICS
    signed = {metric_id for metric_id, _ in metric_signature(matrix)}
    assert "wall_clock_ms" not in signed


def test_signature_still_distinguishes_authorized_from_deviated_paths(matrix):
    """折叠路径**不能**把"读了授权文件"和"读了未授权文件"抹成一样。"""
    good = matrix.observation("T-BENIGN-01", "B3", "GOOD")
    deviated = matrix.observation("T-BENIGN-01", "B3", "PATH_DEVIATION")
    authorized = set(matrix.authorized_paths["base"])

    from app.evaluation.llm.runner import observation_signature

    assert observation_signature(good, authorized=authorized) != observation_signature(
        deviated, authorized=authorized
    )


def test_metric_values_are_identical_between_runs(matrix, tmp_path):
    second = asyncio.run(run_llm_evaluation(workdir=tmp_path / "second"))
    for metric in matrix.metrics:
        if metric.metric_id in NON_DETERMINISTIC_METRICS:
            continue
        other = second.metric(metric.metric_id)
        assert [(c.baseline, c.behavior, c.status, c.value) for c in metric.cells] == [
            (c.baseline, c.behavior, c.status, c.value) for c in other.cells
        ], metric.metric_id


# ---------------------------------------------------------------------------
# 2. D-2 清单(**只列清单,不执行**)
# ---------------------------------------------------------------------------


def test_pilot_manifest_requires_separate_approval():
    manifest = pilot_manifest()
    assert manifest.requires_separate_approval is True


def test_pilot_manifest_counts_match_its_own_dimensions():
    manifest = pilot_manifest()
    expected = (
        manifest.task_count
        * manifest.repetition_count
        * len(manifest.baselines)
        * len(manifest.behaviors)
    )
    assert manifest.total_runs == expected
    assert manifest.total_runs == 8 * 3 * 4 * 1


def test_pilot_manifest_llm_call_bounds_are_ordered_and_positive():
    manifest = pilot_manifest()
    assert 0 < manifest.estimated_llm_calls_low < manifest.estimated_llm_calls_high


def test_pilot_manifest_llm_call_bounds_follow_max_iterations():
    """图运行每次 ≤ max_iterations=5 次 LLM 调用;B0 每次恰好 1 次。"""
    manifest = pilot_manifest()
    direct_runs = manifest.task_count * manifest.repetition_count * 2  # B0 两个标签
    graph_runs = manifest.task_count * manifest.repetition_count * 2  # B2' 与 B3
    assert manifest.estimated_llm_calls_low == direct_runs + graph_runs * 2
    assert manifest.estimated_llm_calls_high == direct_runs + graph_runs * 5


def test_pilot_manifest_explains_the_72_vs_96_discrepancy():
    """B0 被拆成两个标签,所以是 96 而不是 72 —— 必须写明,不能留成疑点。"""
    notes = " ".join(pilot_manifest().notes)
    assert "72" in notes
    assert "B0" in notes


def test_pilot_manifest_states_it_is_not_statistically_final():
    notes = " ".join(pilot_manifest().notes)
    assert "不是" in notes and "统计终局" in notes


def test_pilot_manifest_is_a_pydantic_model():
    assert isinstance(pilot_manifest(), PilotManifest)


# ---------------------------------------------------------------------------
# 3. 离线保证:不得出现真实 provider
# ---------------------------------------------------------------------------

#: 真实 provider / 网络客户端 —— 评测包里出现任何一个都意味着 D-1 越界了。
FORBIDDEN_PROVIDER_MODULES = (
    "openai",
    "langchain_openai",
    "anthropic",
    "httpx",
    "requests",
    "aiohttp",
    "urllib.request",
)


def _collect_imports(path: Path) -> set[str]:
    """收集模块里**所有** import 目标,包括函数体内的局部 import。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("module_name", sorted(p.name for p in LLM_PKG.glob("*.py")))
def test_no_evaluation_module_imports_a_network_client(module_name):
    imported = _collect_imports(LLM_PKG / module_name)
    offenders = sorted(
        module
        for module in imported
        if any(
            module == target or module.startswith(target + ".")
            for target in FORBIDDEN_PROVIDER_MODULES
        )
    )
    assert offenders == [], f"{module_name} 引入了真实 provider / 网络客户端:{offenders}"


def test_runner_never_reads_dotenv_or_environment_secrets():
    """D-1 不得读 `.env` 或环境里的凭据。"""
    source = (LLM_PKG / "runner.py").read_text(encoding="utf-8")
    for forbidden in ("load_dotenv", "os.environ", "getenv", "Settings("):
        assert forbidden not in source, f"runner.py 出现了 {forbidden}"


def test_scripted_llm_exposes_only_the_two_methods_production_uses():
    """脚本化 LLM 只实现 `bind_tools` 与 `ainvoke`,不实现任何网络入口。"""
    from app.evaluation.llm.adapters import ScriptedLLM

    public = {name for name in vars(ScriptedLLM) if not name.startswith("_")}
    assert public == {"bind_tools", "ainvoke"}


def test_no_observation_carries_a_secret_like_value(matrix):
    """整份结果序列化后不得出现疑似凭据的片段。"""
    dumped = matrix.model_dump_json()
    for needle in ("sk-", "Bearer ", "api_key", "Authorization"):
        assert needle not in dumped, f"评测产物里出现了疑似凭据:{needle}"


# ---------------------------------------------------------------------------
# 4. 运行时离线证明(比 AST 检查更强)
# ---------------------------------------------------------------------------

#: 真实的网络出口审计事件。环回不算 —— Windows 上 `socketpair()` 用 127.0.0.1
#: 模拟,asyncio 的事件循环 self-pipe 依赖它,那不是对外流量。
_EXTERNAL_NETWORK_EVENTS = frozenset({
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.sendto",
    "urllib.Request",
    "http.client.connect",
})
_LOOPBACK_PREFIXES = ("127.", "::1", "localhost")


def _is_loopback(address) -> bool:
    if isinstance(address, tuple) and address:
        host = str(address[0])
        return any(host.startswith(prefix) for prefix in _LOOPBACK_PREFIXES)
    return False


def test_evaluation_performs_zero_outbound_network_activity(tmp_path):
    """用 CPython 审计钩子证明:评测全程**没有**任何出网活动。

    AST 检查"没 import 网络库"是不够的 —— 依赖链上任何一环都可能间接发起
    连接。审计钩子是解释器级的,绕过不了。事件列表为空即证明:
        - 没有真实 provider 调用
        - 因此**没有消耗任何 API 额度**

    为了控制耗时,这里只跑一个子集(B3 × 3 行为 × 8 任务);机制与全矩阵相同。
    """
    import sys as _sys

    events: list[str] = []

    def _audit(event: str, args) -> None:
        if event in _EXTERNAL_NETWORK_EVENTS:
            events.append(event)
        elif event in ("socket.connect", "socket.connect_ex"):
            address = args[1] if len(args) > 1 else None
            if not _is_loopback(address):
                events.append(f"{event}:{address}")

    # 审计钩子一旦装上就不可移除;这里只记录、不抛错,对其它测试无副作用。
    _sys.addaudithook(_audit)

    result = asyncio.run(run_llm_evaluation(
        workdir=tmp_path / "offline",
        baselines=("B3",),
        behaviors=("GOOD", "SAFE_PROMPT_INJECTION_FOLLOW", "PATH_DEVIATION"),
    ))
    assert len(result.observations) == 8 * 3
    assert events == [], f"评测过程中出现了出网活动:{events[:5]}"
    assert result.metadata.provider == "scripted"
    assert result.metadata.base_url_host is None
