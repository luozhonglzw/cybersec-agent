"""API 请求 DTO 的 extra 策略护栏(Phase 8.5)。

背景:Phase 8.4 Review 决定**不**改行为(不加 forbid),但要求把
"未知字段怎么办"从隐式默认变成**显式、可评审**的决定。这条护栏守三件事:

1. 每个请求 DTO 都在**自己身上**声明策略 —— 不靠继承链。
   否则"策略在哪"要靠读者追父类,下一个 DTO 作者无从判断它是有意还是疏忽。
2. 策略值是 ignore,不是 forbid —— forbid 会让任何多发字段的客户端吃 422;
   在没有 API 版本化机制之前,这个兼容成本换不来对应的安全收益。
3. 行为与策略一致:多发字段被**忽略**,不影响服务端生成的关键字段
   (thread_id 只能服务端生成,D3)。

全部 hermetic:只构造内存对象,不发请求、不读写数据文件。
"""
import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.api import schemas as api_schemas
from app.api.schemas import (
    ApprovalDecisionRequest,
    ChatRequest,
    ChatResponse,
    ResumeRequest,
    TriageRequest,
    TriageResponse,
)

# 显式列出,便于 parametrize 出可读的用例 id。
REQUEST_DTOS = (
    ChatRequest,
    TriageRequest,
    ApprovalDecisionRequest,
    ResumeRequest,
)

RESPONSE_DTOS = (ChatResponse, TriageResponse)

_SCHEMAS_SOURCE = Path(api_schemas.__file__)


def _class_bodies() -> dict[str, list[ast.stmt]]:
    """解析 schemas.py,返回 {类名: 类体语句}。

    为什么要查 AST 而不是 `"model_config" in cls.__dict__`:
    **pydantic v2 的元类总会往类 __dict__ 里塞一个 model_config**
    (实测:未声明的类拿到 `{}`)。所以"在 __dict__ 里"永远为真 ——
    那个断言是恒真的。而且继承会让 `model_config` 从父类合并过来,
    "删掉本类声明、靠父类兜底"时值依然正确。要判断"是否**亲自声明**",
    只有看源码语法树这一条路。
    """
    tree = ast.parse(_SCHEMAS_SOURCE.read_text(encoding="utf-8"))
    return {
        node.name: node.body
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def _declares_model_config(class_name: str) -> bool:
    """类体里是否**直接**出现 model_config 赋值(含带注解形式)。"""
    for stmt in _class_bodies().get(class_name, []):
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        else:
            continue
        if any(
            isinstance(t, ast.Name) and t.id == "model_config" for t in targets
        ):
            return True
    return False


# ---------- 策略显式性 ----------

@pytest.mark.parametrize("dto", REQUEST_DTOS, ids=lambda d: d.__name__)
def test_request_dto_declares_extra_policy_explicitly(dto):
    """策略必须写在类**自身**的类体里,不能只靠继承。

    只断言 `model_config["extra"] == "ignore"` 挡不住"删掉本类的声明、
    靠父类继承"—— ResumeRequest 继承 ApprovalDecisionRequest,父类的
    extra=ignore 会合并下来,值照样正确,但策略从"可见"退回"隐式"。
    所以用 AST 判断"是否亲自声明"。
    """
    assert _declares_model_config(dto.__name__), (
        f"{dto.__name__} 未在类体中显式声明 model_config —— "
        f"策略不可见即等于没策略(靠继承会静默漂移)"
    )
    assert dto.model_config["extra"] == "ignore"


def test_every_request_dto_in_module_is_covered():
    """模块里新增的 *Request 必须一起被覆盖,不能悄悄漏网。

    漏网的 DTO 会退回 pydantic 默认(extra=ignore),今天行为一样、
    明天改默认就静默漂移 —— 所以这里要求"名单 == 实际"。
    """
    declared = {
        name
        for name, obj in vars(api_schemas).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and name.endswith("Request")
    }
    assert declared == {d.__name__ for d in REQUEST_DTOS}


def test_response_dto_does_not_forbid_extra():
    """响应体由服务端构造,不该被"收紧"成 forbid —— 那是加固错了方向。"""
    for dto in RESPONSE_DTOS:
        assert dto.model_config.get("extra") != "forbid", dto.__name__


# ---------- 行为与策略一致 ----------

def test_extra_field_is_ignored_not_rejected():
    """行为层:多发字段被忽略(而不是抛 ValidationError)。"""
    req = TriageRequest.model_validate(
        {"indicator": "203.0.113.66", "junk": 1, "nested": {"a": 2}}
    )
    assert req.indicator == "203.0.113.66"
    assert not hasattr(req, "junk")
    assert not hasattr(req, "nested")


def test_client_supplied_thread_id_is_dropped():
    """D3 的 DTO 侧证据:客户端塞 thread_id 进 TriageRequest 不会生效。

    服务端自己生成 thread_id;若这个字段能进模型,"客户端指定恢复哪条
    thread"就成了劫持向量。这里锁住"进不来"。
    """
    req = TriageRequest.model_validate(
        {"indicator": "203.0.113.66", "thread_id": "attacker-chosen"}
    )
    assert not hasattr(req, "thread_id")


def test_resume_request_keeps_its_own_thread_id():
    """反向对照:ResumeRequest 的 thread_id 是**必填**的业务字段,
    不能被 extra 策略误伤成"被忽略"。"""
    req = ResumeRequest.model_validate(
        {"thread_id": "thread-abc", "status": "approved", "operator": "analyst-1"}
    )
    assert req.thread_id == "thread-abc"
