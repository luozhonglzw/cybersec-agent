"""阶段判定语义:**PASS / INCONCLUSIVE / FAIL / ABORT**。

为什么 `INCONCLUSIVE` 必须是一个**独立**的判定
---------------------------------------------
"上限没生效"与"这次探针没能证明上限生效"是两个完全不同的结论。把它们
合并成一个"失败"会让人以为已经得到了答案;合并成一个"通过"则是伪造。

所以本模块有三档,且 `INCONCLUSIVE` **不算通过** —— 它与其他非 PASS
判定一样阻断后续阶段。

`finish_reason` 的地位(**刻意受限**)
------------------------------------
`finish_reason` **不参与判定**,只被记录。原因:

    finish_reason == "stop"   **不**能推出"上限被忽略"
        —— 模型完全可能在远低于上限处自然结束。

    finish_reason == "length" **不**能推出"上限生效"
        —— 它只说明"因为某种长度原因停了",不说明停在了我们请求的那个长度上。

判定的依据是**长度证据**:观测到的输出 token 数(首选)与字符长度(退而求其次),
与"请求的上限"和"要求模型产出的长度"两者比较。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# 阈值(全部为**判定参数**,不是实测值)
# ---------------------------------------------------------------------------

#: 观测输出超过 `请求上限 × 该系数` ⇒ 判定"显著超过上限"。
C0B_OVERRUN_TOLERANCE = 1.5

#: 观测输出低于 `请求上限 × 该系数` ⇒ 判定"上限从未被触及"(模型提前自然结束)。
C0B_ENFORCED_FLOOR = 0.5

#: 观测字符长度达到 `要求长度 × 该系数` ⇒ 判定"在上限远小于要求时仍产出了接近
#: 要求的长度",即上限**未生效**。仅在缺少 token usage 时使用。
C0B_DEMAND_RATIO = 0.5


class StageVerdict(str, Enum):
    """一个标定阶段的判定。"""

    PASS = "PASS"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAIL = "FAIL"
    ABORT = "ABORT"

    @property
    def is_pass(self) -> bool:
        return self is StageVerdict.PASS

    @property
    def blocks_progression(self) -> bool:
        """**只有 PASS 放行。** INCONCLUSIVE 与 FAIL/ABORT 一样阻断后续阶段。"""
        return not self.is_pass


class StageBlocked(RuntimeError):
    """某个阶段的判定不允许后续阶段开始。"""


@dataclass(frozen=True)
class Verdict:
    """判定 + 得出它的**具体理由**。理由不得为空。"""

    verdict: StageVerdict
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("判定理由不得为空 —— 没有理由的判定无法复核")

    @property
    def is_pass(self) -> bool:
        return self.verdict.is_pass


def passed(reason: str) -> Verdict:
    return Verdict(StageVerdict.PASS, reason)


def inconclusive(reason: str) -> Verdict:
    return Verdict(StageVerdict.INCONCLUSIVE, reason)


def failed(reason: str) -> Verdict:
    return Verdict(StageVerdict.FAIL, reason)


def aborted(reason: str) -> Verdict:
    return Verdict(StageVerdict.ABORT, reason)


def assert_stage_passed(verdict: Verdict, *, stage: str, next_stage: str) -> None:
    """门禁:非 PASS ⇒ 后续阶段不得开始。"""
    if verdict.is_pass:
        return
    raise StageBlocked(
        f"{stage} 判定为 {verdict.verdict.value},后续阶段 {next_stage} 不得开始。"
        f"理由:{verdict.reason}"
    )


# ---------------------------------------------------------------------------
# C0b:长输出上限探针的判定
# ---------------------------------------------------------------------------


def c0b_verdict(
    *,
    requested_cap: int,
    demanded_output_tokens: int,
    observed_output_tokens: int | None,
    observed_chars: int | None,
    demanded_chars: int | None,
    finish_reason: str | None,
    provider_rejected: bool,
) -> Verdict:
    """C0b 的判定。

    依据**长度证据**,不依据 `finish_reason`(后者只被记录,见模块 docstring)。

    判定顺序:
        1. provider 拒绝了上限参数            → FAIL
        2. 观测输出显著超过请求上限           → FAIL
        3. 观测输出远低于请求上限(自然结束) → INCONCLUSIVE(上限未被触及)
        4. 观测输出落在上限附近               → PASS
        5. 无 token usage 时:
             长度接近要求长度                 → FAIL(上限显然未生效)
             否则                             → INCONCLUSIVE(证据不足)

    `finish_reason` 入参**只用于记录**;本函数不对它做任何分支。
    """
    if requested_cap < 1:
        raise ValueError(f"requested_cap 至少为 1,收到 {requested_cap!r}")
    if demanded_output_tokens <= requested_cap:
        raise ValueError(
            f"要求产出的长度 {demanded_output_tokens} 必须**远大于**上限 "
            f"{requested_cap} —— 否则'模型自然停下'与'上限生效'无法区分"
        )

    if provider_rejected:
        return failed(
            f"provider 拒绝了上限参数(请求上限 {requested_cap}) —— "
            "参数未被接受,上限无从生效"
        )

    if observed_output_tokens is not None:
        if observed_output_tokens > requested_cap * C0B_OVERRUN_TOLERANCE:
            return failed(
                f"观测输出 {observed_output_tokens} tokens 显著超过请求上限 "
                f"{requested_cap}(容差系数 {C0B_OVERRUN_TOLERANCE}) —— "
                "上限未被 provider 执行"
            )
        if observed_output_tokens < requested_cap * C0B_ENFORCED_FLOOR:
            return inconclusive(
                f"观测输出仅 {observed_output_tokens} tokens,远低于请求上限 "
                f"{requested_cap} —— 模型在上限之前就自然结束了,"
                "本次探针**没有触及**上限,无法判定它是否生效"
                + (f"(finish_reason={finish_reason!r} 仅记录,不作为判据)" if finish_reason else "")
            )
        return passed(
            f"观测输出 {observed_output_tokens} tokens 落在请求上限 {requested_cap} "
            f"附近,且远低于要求产出的 {demanded_output_tokens} tokens —— "
            "证据与「上限被 provider 执行」一致"
        )

    # ---- 无 token usage:退到长度证据 ----
    if observed_chars is None or demanded_chars is None:
        return inconclusive(
            "既没有 token usage,也没有可用的长度证据 —— 无法判定上限是否生效"
        )
    if observed_chars >= demanded_chars * C0B_DEMAND_RATIO:
        return failed(
            f"上限为 {requested_cap} tokens(远小于要求产出),但响应长度 "
            f"{observed_chars} 字符已达到要求长度 {demanded_chars} 字符的 "
            f"{C0B_DEMAND_RATIO:.0%} 以上 —— 上限显然未生效"
        )
    return inconclusive(
        f"缺少 token usage;响应长度 {observed_chars} 字符虽然短于要求长度 "
        f"{demanded_chars} 字符,但**仅凭长度无法判定**它究竟是被上限截断的,"
        "还是模型自然提前结束的"
    )
