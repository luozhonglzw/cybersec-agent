"""Phase 9.2-A 报告渲染。

报告必须做到的几件事(都是硬要求,不是排版偏好)
------------------------------------------------
1. **按 A/B/C/D/E 五类分开**,并且**不合成任何"Agent 总分"**。
   一个把 D 类自洽检查和 A 类独立正确性加权平均的"总分",在数学上就是
   把"实现复述自己"算进了正确性 —— 那正是本阶段要避免的循环论证。
2. **NOT_EVALUABLE 显式打印**,并附上原因。留空会被读成 0。
3. **D 类必须自带免责声明**,让读者一眼看出"这些绿色不代表正确"。
4. **输入摘要(golden digest + dataset digest)必须打印**,否则结果无法被独立复核。
5. **基线对比必须带上诚实声明**:B1 与 B3 共享同一个确定性内核,
   它们之间的差异不是"Agent 优于非 Agent"。
"""
from pathlib import Path

from app.evaluation.metrics import MetricResult
from app.evaluation.runner import EvaluationResult

CATEGORY_TITLES: dict[str, str] = {
    "A": "A. 独立正确性(Independent Correctness)",
    "B": "B. 独立安全性(Independent Safety)",
    "C": "C. 可靠性 / 生命周期(Reliability & Lifecycle)",
    "D": "D. 实现自洽性(Implementation Consistency)—— **不能当作正确性证据**",
    "E": "E. 不可评测(Not Evaluable in 9.2-A)",
}

CATEGORY_CAVEATS: dict[str, str] = {
    "A": "证据层事实与**独立于实现**的来源比对(手工撰写的原始事实 / 独立重算)。",
    "B": "行为层是否满足**手工撰写**的安全性质。性质不从 scenario_intent 推导,"
         "也不把\"恶意\"等同于\"需人工审批\"。",
    "C": "审计完整性、终止性、以及评测自身的确定性。",
    "D": "**以下指标全部是\"实现复述自己的规则\"。** 一个把风险权重从 30 改成 10 "
         "的语义变异会让它们全部保持绿色。它们能抓内部漂移,但**不能**证明判断正确。",
    "E": "本阶段没有独立 oracle 的能力。**这些不是 0 分,是\"无法评测\"。**",
}


def _fmt(metric: MetricResult) -> str:
    if metric.status == "not_evaluable":
        return f"NOT_EVALUABLE — {metric.not_evaluable_reason}"
    assert metric.value is not None
    return (
        f"{metric.value:.4f}  "
        f"({metric.numerator}/{metric.denominator} {metric.unit})"
    )


def render_markdown(result: EvaluationResult) -> str:
    """把评测结果渲染成 Markdown。"""
    lines: list[str] = []
    add = lines.append

    add("# Phase 9.2-A 评测报告")
    add("")
    add("## 输入摘要(可独立复核)")
    add("")
    add(f"- golden set 版本:`{result.golden_version}`")
    add(f"- golden set 摘要(sha256):`{result.golden_digest}`")
    add(f"- 数据集摘要:`logs`=`{result.dataset_digests['logs']}`")
    add(f"- 数据集摘要:`intel`=`{result.dataset_digests['intel']}`")
    add(f"- 参与基线:{', '.join(result.adapter_ids)}")
    add(f"- 观测条数:{len(result.observations)}")
    add("")

    add("## 指标(按类别分开,不合成总分)")
    add("")
    add(
        "> 本报告**刻意不给出**任何单一\"Agent 分数\"。把 D 类自洽性与 A 类独立正确性"
        "加权平均,等于把\"实现复述自己\"计入正确性 —— 那正是本阶段要避免的循环论证。"
    )
    add("")

    for category in ("A", "B", "C", "D", "E"):
        items = result.metrics_by_category(category)
        if not items:
            continue
        add(f"### {CATEGORY_TITLES[category]}")
        add("")
        add(f"> {CATEGORY_CAVEATS[category]}")
        add("")
        add("| 指标 | ground truth | 方向 | 结果 |")
        add("| --- | --- | --- | --- |")
        for metric in items:
            add(
                f"| `{metric.metric_id}` | {metric.oracle_class} | "
                f"{metric.direction} | {_fmt(metric)} |"
            )
        add("")
        add("<details><summary>指标定义</summary>")
        add("")
        for metric in items:
            add(f"- **`{metric.metric_id}`** — {metric.title}")
            add(f"  - {metric.definition}")
        add("")
        add("</details>")
        add("")

    add("## 变形关系(不引用任何权重常量)")
    add("")
    if not result.relations:
        add("(本次运行未执行变形关系检查)")
    else:
        add("| 关系 | 状态 | 说明 |")
        add("| --- | --- | --- |")
        for relation in result.relations:
            add(f"| `{relation.relation_id}` | {relation.status} | {relation.detail} |")
    add("")

    add("## 基线对比的诚实声明")
    add("")
    add(
        "- **B1(规则直连)与 B3(完整 Agent)共享同一个确定性内核** —— 同一套"
        " collect_evidence / analyze_risk / plan_response / evaluate_policy。"
        "二者的差异**不是**\"Agent 优于非 Agent\",而是\"有没有策略门\"。"
    )
    add(
        "- **B2(无门图)与 B3** 的差异是\"有门 / 无门\",可作为安全性消融;"
        "无门基线在\"必须人工审批\"的用例上必然不满足 —— 这是结构性事实,不是缺陷计数。"
    )
    add(
        "- **没有任何基线执行真实处置动作。** Phase 8 只做审批,不接防火墙/EDR。"
        "因此消融实验度量的是**门控与留痕**,不是\"实际危害\"。"
    )
    add(
        "- **Agent vs Chatbot 对照在本阶段不可评测** —— 需要真实 LLM 才能构成有意义的对照。"
    )
    add("")
    return "\n".join(lines)


def write_report(result: EvaluationResult, path: Path | str) -> Path:
    """把 Markdown 报告写到 path,返回该路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(result), encoding="utf-8")
    return target
