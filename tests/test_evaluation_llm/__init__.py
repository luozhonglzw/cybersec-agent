"""Phase 9.2-D-1 真实 LLM 评测测试包。

本目录与 `tests/test_evaluation/`(Phase 9.2-A)**并列**:
    9.2-A  测确定性内核(11 个冻结 golden case)
    9.2-D  测 LLM 能力 / 叙事接地 / 合成注入 / 架构回归不变量

D-1 全程离线:所有 LLM 行为都由 `ScriptedLLM` 脚本化,
**不发起任何真实 provider 调用、不消耗任何 API 额度**。
"""
