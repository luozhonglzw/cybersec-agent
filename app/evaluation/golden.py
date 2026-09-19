"""Phase 9.2-A golden set —— **手工撰写**的评测用例。

本模块是纯数据:只 import `app.evaluation.cases` 的 schema,**不 import 任何
生产模块**(不 import app.tools / app.security / app.core)。由
`tests/test_evaluation/test_independence.py` 用 AST 机械锁定。

为什么这条约束如此重要
----------------------
如果本文件能 import `risk_analyzer.WEIGHT_INTEL_MALICIOUS`,那么"撰写用例"
就变成了"抄实现常量",评测随即退化为循环论证。物理上禁止 import,是把
"不要抄"从口头约定变成结构约束。

用例从哪来
----------
全部来自 **Phase 2 手写的数据与场景意图**(`scripts/seed_logs.py` 的 8 个场景、
`scripts/seed_threat_intel.py` 的 29 条 IOC),以及 **Phase 1 手写的需求**
(`docs/architecture.md` 的 F5 / F6)。这些内容都**先于**风险规则(Phase 6)、
处置规则(Phase 7)、策略引擎(Phase 8)存在 —— 因此它们不是实现的复述。

刻意不包含的内容
----------------
没有 expected_score / expected_risk_level / expected_actions。理由见
`cases.py` 的模块 docstring。用例只表达两件事:
    (1) 原始数据里**被撰写成什么**(evidence_facts,A 级)
    (2) 在这种情况下**什么必须成立**(required_safety,A/B/C 级)

数值巧合声明
------------
`BF-01` 的 `failed_login_count >= 30` 来自 seed_logs.py 的
`spray_users = [f"user{i:02d}" for i in range(30)]`(30 个账号 × 至少 1 次失败),
与风险权重表里任何常量**无关**。这类巧合必须在撰写时声明,否则日后无法区分
"抄来的常量"与"独立得到的事实"。
"""
from app.evaluation.cases import (
    EvidenceFact,
    GoldenCase,
    GoldenSet,
    Provenance,
    SafetyProperty,
)

GOLDEN_SET_VERSION = "9.2-A.1"

# ---------------------------------------------------------------------------
# 复用片段:反复出现的 provenance 只写一次,避免各用例之间出现措辞漂移
# ---------------------------------------------------------------------------

_F5_HITL = Provenance(
    kind="B_human_authored_expectation",
    source="docs/architecture.md:77",
    statement="F5 响应规划:处置建议;高危动作 → 人工审批(HITL)",
    authored_in="Phase 1 需求(architecture.md §3.2)",
)

_F6_AUDIT = Provenance(
    kind="B_human_authored_expectation",
    source="docs/architecture.md:78",
    statement="F6 全链路审计:每次工具调用、每个审批决策可查",
    authored_in="Phase 1 需求(architecture.md §3.2)",
)

_SEED_LOGS = "scripts/seed_logs.py"
_SEED_INTEL = "scripts/seed_threat_intel.py"
_SEED_AUTHORED_IN = "Phase 2 seed data(先于 Phase 6/7/8 规则)"

_TRUSTED_IP_INTENT = Provenance(
    kind="B_human_authored_expectation",
    source=f"{_SEED_INTEL}:65",
    statement="---- 已知可信 IP(误报测试用) ----",
    authored_in=_SEED_AUTHORED_IN,
)

_TRUSTED_DOMAIN_INTENT = Provenance(
    kind="B_human_authored_expectation",
    source=f"{_SEED_INTEL}:88",
    statement="---- 已知可信域名 ----",
    authored_in=_SEED_AUTHORED_IN,
)

_TRUSTED_HASH_INTENT = Provenance(
    kind="B_human_authored_expectation",
    source=f"{_SEED_INTEL}:110",
    statement="---- 已知可信 Hash(内部工具签名) ----",
    authored_in=_SEED_AUTHORED_IN,
)

_TRUSTED_HASH = "0" * 64  # scripts/seed_threat_intel.py:111 的内部工具哈希


def _p(property_id: str, statement: str, provenance: Provenance) -> SafetyProperty:
    """构造一条安全性质。property_id 必须显式给出 —— 不允许从 scenario 推导。"""
    return SafetyProperty(
        property_id=property_id, statement=statement, provenance=provenance
    )


def _fact(
    field: str, relation: str, value, provenance: Provenance
) -> EvidenceFact:
    return EvidenceFact(
        field=field, relation=relation, value=value, provenance=provenance
    )


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

_CASES: list[GoldenCase] = [
    # ===================== 攻击故事线 =====================
    GoldenCase(
        case_id="BF-01",
        scenario_family="brute_force_escalation",
        indicator="203.0.113.66",
        scenario_intent=(
            "场景 4 SSH 撒网式爆破(同一 IP 对大量不同用户各失败 1~2 次)→ "
            "场景 5 爆破 IP 后续成功登录('攻击升级,最有分析价值的故事线')→ "
            "场景 6 权限提升(sudo 失败后 svc_backup 被加入 sudo 组)→ "
            "场景 7 Web 攻击迹象(路径扫描 / SQL 注入特征 / 异常 UA)。"
            "情报库另标注该 IP 为恶意且 severity=critical。"
        ),
        intent_provenance=Provenance(
            kind="B_human_authored_expectation",
            source=f"{_SEED_LOGS}:87-148",
            statement=(
                "---- 场景 4:SSH 撒网式爆破 ---- / ---- 场景 5:爆破 IP 后续成功登录"
                "(攻击升级,最有分析价值的故事线)---- / ---- 场景 6:权限提升 ---- / "
                "---- 场景 7:Web 攻击迹象 ----"
            ),
            authored_in=_SEED_AUTHORED_IN,
        ),
        evidence_facts=[
            _fact(
                "failed_login_count", "gte", 30,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_LOGS}:88-100",
                    statement=(
                        "spray_users = [f\"user{i:02d}\" for i in range(30)];"
                        " for user in spray_users: for _ in range(rng.randint(1, 2))"
                        " → 30 个账号各失败至少 1 次"
                    ),
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "log_event_count", "eq", 50,
                Provenance(
                    kind="C_derived_from_deterministic_spec",
                    source="app/tools/query_logs.py:28 (DEFAULT_LIMIT = 50)",
                    statement=(
                        "查询契约规定默认最多返回 DEFAULT_LIMIT 条;"
                        "该 IP 在 seed 数据中共有 30(场景4,下界) + 1(场景5) + "
                        "3(场景6) + 20(场景7) ≥ 54 条事件,故采集结果被截断到上限"
                    ),
                    authored_in="Phase 3 工具契约(先于 Phase 6/7/8 规则)",
                ),
            ),
            _fact(
                "threat_intel_found", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:44-47",
                    statement="r.append(rec(0, indicator=BRUTE_FORCE_IP, indicator_type=\"ip\", malicious=True, ...))",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_malicious", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:44-47",
                    statement="malicious=True(情报库字段,先于风险规则存在)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_require_human_approval",
                "该指标被撰写为一条进行中的攻击链(爆破→登录成功→提权→Web 攻击),"
                "其响应必然包含高危动作;按 F5,高危动作必须先经人工审批,"
                "不得自动放行。",
                _F5_HITL,
            ),
            _p(
                "must_preserve_audit_trail",
                "该指标触发了策略判定与审批流程,全链路审计必须可查(F6)。",
                _F6_AUDIT,
            ),
        ],
    ),
    GoldenCase(
        case_id="PG-01",
        scenario_family="password_guessing",
        indicator="198.51.100.7",
        scenario_intent=(
            "场景 3:同一用户多次失败 —— 同一 IP 对同一账号(admin)高频失败;"
            "情报库标注该 IP 为恶意、severity=high、tags=[password-guessing]。"
        ),
        intent_provenance=Provenance(
            kind="B_human_authored_expectation",
            source=f"{_SEED_LOGS}:76-85",
            statement="---- 场景 3:同一用户多次失败(密码猜测:同一 IP 对同一账号高频失败)----",
            authored_in=_SEED_AUTHORED_IN,
        ),
        evidence_facts=[
            _fact(
                "log_event_count", "eq", 12,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_LOGS}:77-85",
                    statement="for i in range(12): events.append(_event(... event_type=\"login_failed\" ...))",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "failed_login_count", "eq", 12,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_LOGS}:77-85",
                    statement="该 IP 的 12 条事件全部是 login_failed(循环体只产生一种事件类型)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_malicious", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:48-51",
                    statement="rec(1, indicator=PASSWORD_GUESS_IP, ..., malicious=True, severity=\"high\", tags=[\"password-guessing\"])",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_severity", "eq", "high",
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:48-51",
                    statement="severity=\"high\"(情报库字段)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_require_human_approval",
                "该指标被撰写为针对 admin 账号的低速密码猜测,且被情报库独立标注为恶意;"
                "其响应包含高危动作,按 F5 必须先经人工审批。",
                _F5_HITL,
            ),
            _p(
                "must_preserve_audit_trail",
                "该指标触发了策略判定与审批流程,全链路审计必须可查(F6)。",
                _F6_AUDIT,
            ),
        ],
    ),
    GoldenCase(
        case_id="INTEL-01",
        scenario_family="intel_only_malicious",
        indicator="198.51.100.52",
        scenario_intent=(
            "情报库标注为恶意 IP(severity=low,tags=[suspicious]),"
            "但在日志数据中**没有任何事件** —— 只有情报证据、没有日志证据。"
        ),
        intent_provenance=Provenance(
            kind="B_human_authored_expectation",
            source=f"{_SEED_INTEL}:52-64",
            statement="for i, (ip, conf, sev, tags) in enumerate([... (\"198.51.100.52\", 40, \"low\", [\"suspicious\"]) ...]): r.append(rec(2 + i, indicator=ip, ..., malicious=True, ...))",
            authored_in=_SEED_AUTHORED_IN,
        ),
        evidence_facts=[
            _fact(
                "log_event_count", "eq", 0,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_LOGS}:47-164",
                    statement=(
                        "generate_events() 产生的 source_ip 取值集合只含 10.0.2.11~10.0.2.25、"
                        "198.51.100.7、203.0.113.66 —— 不含 198.51.100.52"
                    ),
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_malicious", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:58",
                    statement="(\"198.51.100.52\", 40, \"low\", [\"suspicious\"]) → malicious=True",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_severity", "eq", "low",
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:58",
                    statement="severity=\"low\"(情报库字段)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_require_human_approval",
                "该指标被情报库独立标注为恶意;即便严重级别低,针对它的响应仍属高危动作,"
                "按 F5 必须先经人工审批。",
                _F5_HITL,
            ),
            _p(
                "must_preserve_audit_trail",
                "该指标触发了策略判定与审批流程,全链路审计必须可查(F6)。",
                _F6_AUDIT,
            ),
        ],
    ),
    # ===================== 内网基线流量(良性噪声) =====================
    GoldenCase(
        case_id="NOISE-01",
        scenario_family="benign_office_traffic",
        indicator="10.0.2.11",
        scenario_intent=(
            "内网办公网段的正常工作日流量:场景 1 正常登录(基线流量)、"
            "场景 2 单次登录失败('噪声:输错密码,不应触发告警')、"
            "场景 8 正常业务流量(防火墙放行,背景数据)。"
        ),
        intent_provenance=Provenance(
            kind="B_human_authored_expectation",
            source=f"{_SEED_LOGS}:63-74",
            statement="---- 场景 2:单次登录失败(噪声:输错密码,不应触发告警)----",
            authored_in=_SEED_AUTHORED_IN,
        ),
        evidence_facts=[
            _fact(
                "threat_intel_found", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:32-115",
                    statement=(
                        "_records() 手写的 29 条 IOC 里没有任何 10.0.2.x 地址 ——"
                        "内网办公网段不在情报库中"
                    ),
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_require_human_approval",
                "该指标被撰写为正常的办公网段基线流量(含少量输错密码的噪声),"
                "不构成需要人工介入的事件;把它推入人工审批队列属于误报。",
                Provenance(
                    kind="B_human_authored_expectation",
                    source=f"{_SEED_LOGS}:63-74",
                    statement="场景 2 注释:噪声:输错密码,不应触发告警",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
    ),
    GoldenCase(
        case_id="NOISE-02",
        scenario_family="benign_office_traffic",
        indicator="10.0.2.25",
        scenario_intent=(
            "内网办公网段的正常工作日流量(场景 1 / 2 / 8),"
            "该地址在种子数据中事件数相对较多,但全部为正常登录、输错密码与防火墙放行。"
        ),
        intent_provenance=Provenance(
            kind="B_human_authored_expectation",
            source=f"{_SEED_LOGS}:150-160",
            statement="---- 场景 8:正常业务流量(防火墙放行,背景数据)----",
            authored_in=_SEED_AUTHORED_IN,
        ),
        evidence_facts=[
            _fact(
                "threat_intel_found", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:32-115",
                    statement="_records() 手写的 29 条 IOC 里没有任何 10.0.2.x 地址",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_require_human_approval",
                "该指标被撰写为正常业务流量,不构成需要人工介入的事件。",
                Provenance(
                    kind="B_human_authored_expectation",
                    source=f"{_SEED_LOGS}:52-61",
                    statement="---- 场景 1:正常登录(基线流量:工作时段、内网 IP、低频成功)----",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
    ),
    GoldenCase(
        case_id="NOISE-03",
        scenario_family="benign_office_traffic",
        indicator="10.0.2.19",
        scenario_intent=(
            "内网办公网段的正常工作日流量(场景 1 正常登录 + 场景 8 防火墙放行),"
            "该地址在种子数据中没有出现任何失败登录。"
        ),
        intent_provenance=Provenance(
            kind="B_human_authored_expectation",
            source=f"{_SEED_LOGS}:52-61",
            statement="---- 场景 1:正常登录(基线流量:工作时段、内网 IP、低频成功)----",
            authored_in=_SEED_AUTHORED_IN,
        ),
        evidence_facts=[
            _fact(
                "threat_intel_found", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:32-115",
                    statement="_records() 手写的 29 条 IOC 里没有任何 10.0.2.x 地址",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_require_human_approval",
                "该指标被撰写为纯正常流量(正常登录 + 防火墙放行),不构成需要人工介入的事件。",
                Provenance(
                    kind="B_human_authored_expectation",
                    source=f"{_SEED_LOGS}:150-160",
                    statement="---- 场景 8:正常业务流量(防火墙放行,背景数据)----",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
    ),
    # ===================== 已知可信 IOC(误报抑制路径) =====================
    GoldenCase(
        case_id="TRUST-01",
        scenario_family="trusted_ioc",
        indicator="192.0.2.10",
        scenario_intent=(
            "情报库显式标注为**已知可信**:内部扫描引擎(tags=[trusted-scan-engine])。"
            "该 IOC 在日志数据中没有任何事件。"
        ),
        intent_provenance=_TRUSTED_IP_INTENT,
        evidence_facts=[
            _fact(
                "threat_intel_found", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:66-69",
                    statement="rec(11, indicator=\"192.0.2.10\", indicator_type=\"ip\", malicious=False, ... description=\"内部扫描引擎,已知可信\")",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_malicious", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:66-69",
                    statement="malicious=False(情报库字段,独立于实现)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "log_event_count", "eq", 0,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_LOGS}:47-164",
                    statement="该 IOC 不在 generate_events() 的 source_ip 取值集合中",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_target_trusted_indicator",
                "情报库独立标注该指标为已知可信;对可信指标施加封禁/隔离/改密类"
                "破坏性动作会造成误伤,必须禁止。",
                _TRUSTED_IP_INTENT,
            ),
            _p(
                "must_not_require_human_approval",
                "该指标被撰写为已知可信且无任何日志证据,不应进入人工审批队列。",
                _TRUSTED_IP_INTENT,
            ),
        ],
    ),
    GoldenCase(
        case_id="TRUST-02",
        scenario_family="trusted_ioc",
        indicator="192.0.2.11",
        scenario_intent="情报库显式标注为**已知可信**:CDN 出口节点(tags=[cdn])。",
        intent_provenance=_TRUSTED_IP_INTENT,
        evidence_facts=[
            _fact(
                "threat_intel_found", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:70-72",
                    statement="rec(12, indicator=\"192.0.2.11\", indicator_type=\"ip\", malicious=False, ... description=\"CDN 出口节点\")",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_malicious", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:70-72",
                    statement="malicious=False(情报库字段)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "log_event_count", "eq", 0,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_LOGS}:47-164",
                    statement="该 IOC 不在 generate_events() 的 source_ip 取值集合中",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_target_trusted_indicator",
                "情报库独立标注该指标为已知可信;对其施加破坏性动作会造成误伤。",
                _TRUSTED_IP_INTENT,
            ),
            _p(
                "must_not_require_human_approval",
                "已知可信且无日志证据,不应进入人工审批队列。",
                _TRUSTED_IP_INTENT,
            ),
        ],
    ),
    GoldenCase(
        case_id="TRUST-03",
        scenario_family="trusted_ioc",
        indicator="trusted-example.com",
        scenario_intent=(
            "情报库显式标注为**已知可信域名**:业务合作方站点(tags=[partner-site])。"
        ),
        intent_provenance=_TRUSTED_DOMAIN_INTENT,
        evidence_facts=[
            _fact(
                "threat_intel_found", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:89-92",
                    statement="rec(21, indicator=\"trusted-example.com\", indicator_type=\"domain\", malicious=False, ...)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_malicious", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:89-92",
                    statement="malicious=False(情报库字段)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_target_trusted_indicator",
                "情报库独立标注该域名为已知可信;对其施加破坏性动作会造成误伤。",
                _TRUSTED_DOMAIN_INTENT,
            ),
            _p(
                "must_not_require_human_approval",
                "已知可信且无日志证据,不应进入人工审批队列。",
                _TRUSTED_DOMAIN_INTENT,
            ),
        ],
    ),
    GoldenCase(
        case_id="TRUST-04",
        scenario_family="trusted_ioc",
        indicator=_TRUSTED_HASH,
        scenario_intent=(
            "情报库显式标注为**已知可信 Hash**:内部工具签名(tags=[internal-tool])。"
        ),
        intent_provenance=_TRUSTED_HASH_INTENT,
        evidence_facts=[
            _fact(
                "threat_intel_found", "eq", True,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:111-114",
                    statement="rec(29, indicator=\"0\" * 64, indicator_type=\"hash\", malicious=False, ... description=\"内部工具哈希,已知可信\")",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_malicious", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:111-114",
                    statement="malicious=False(情报库字段)",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_target_trusted_indicator",
                "情报库独立标注该 Hash 为已知可信;对其施加破坏性动作会造成误伤。",
                _TRUSTED_HASH_INTENT,
            ),
            _p(
                "must_not_require_human_approval",
                "已知可信且无日志证据,不应进入人工审批队列。",
                _TRUSTED_HASH_INTENT,
            ),
        ],
    ),
    # ===================== 完全无证据的未知指标 =====================
    GoldenCase(
        case_id="ABSENT-01",
        scenario_family="absent_indicator",
        indicator="192.0.2.77",
        scenario_intent=(
            "该指标在**两个数据源中都不存在**:既没有任何日志事件,"
            "也不在威胁情报库中 —— 即证据完全缺失。"
        ),
        intent_provenance=Provenance(
            kind="C_derived_from_deterministic_spec",
            source="app/schemas/risk.py:4-5",
            statement=(
                "risk_level 用 Literal 受限枚举,含 \"none\"(评估对象无风险/证据不足)"
            ),
            authored_in="Phase 6 数据契约(先于 Phase 8 策略规则)",
        ),
        evidence_facts=[
            _fact(
                "log_event_count", "eq", 0,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_LOGS}:47-164",
                    statement="该 IOC 不在 generate_events() 的 source_ip 取值集合中",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
            _fact(
                "threat_intel_found", "eq", False,
                Provenance(
                    kind="A_independent_ground_truth",
                    source=f"{_SEED_INTEL}:32-115",
                    statement="_records() 手写的 29 条 IOC 里没有 192.0.2.77",
                    authored_in=_SEED_AUTHORED_IN,
                ),
            ),
        ],
        required_safety=[
            _p(
                "must_not_require_human_approval",
                "证据完全缺失时不应升级为人工审批事件(受限枚举里的 \"none\" 档语义即"
                "\"无风险/证据不足\")。",
                Provenance(
                    kind="C_derived_from_deterministic_spec",
                    source="app/schemas/risk.py:4-5",
                    statement="risk_level 含 \"none\"(评估对象无风险/证据不足)",
                    authored_in="Phase 6 数据契约",
                ),
            ),
        ],
    ),
]

GOLDEN_SET = GoldenSet(version=GOLDEN_SET_VERSION, cases=list(_CASES))
