#!/usr/bin/env python3

"""Debug script to test the specific test scenario."""

import sys
sys.path.append('.')

import asyncio
from app.core.llm import FakeChatModel, FakeLLMClient
from app.core.agent import SecurityAgent

# Recreate the exact test scenario
async def test_one_tool_call_debug():
    print("=== Creating test scenario ===")
    
    # Create initial agent
    original_llm = FakeLLMClient("收到,正在分析。")
    agent = SecurityAgent(original_llm, max_iterations=3)
    
    print(f"Original agent LLM: {agent._llm}")
    print(f"Original agent LLM _model: {agent._llm._model}")
    print(f"Has bind_tools: {hasattr(agent._llm._model, 'bind_tools')}")
    
    # Simulate the test modification
    print("\n=== Modifying LLM as in test ===")
    
    # Mock tool result
    import json
    tool_result = {
        "args": {"limit": 5},
        "result": json.dumps({
            "count": 5,
            "events": [{"event_type": "login_success"}] * 5
        })
    }
    
    # Create new LLM as in test
    new_llm = FakeLLMClient(reply="我需要查询安全日志来回答这个问题")
    new_llm._model = FakeChatModel([
        "我需要查询安全日志来回答这个问题",  # 触发工具调用
        "根据查询结果,有 5 条登录成功事件"  # 最终回答
    ], [tool_result])
    
    print(f"New LLM: {new_llm}")
    print(f"New LLM _model: {new_llm._model}")
    print(f"New LLM _model type: {type(new_llm._model)}")
    print(f"New LLM _model has bind_tools: {hasattr(new_llm._model, 'bind_tools')}")
    
    # Set the new LLM
    agent._llm = new_llm
    print(f"Agent LLM after modification: {agent._llm}")
    print(f"Agent LLM _model after modification: {agent._llm._model}")
    
    # Test the problematic call
    print("\n=== Testing bind_tools call ===")
    try:
        result = agent._llm._model.bind_tools(agent._tools)
        print(f"bind_tools result: {result}")
        print("SUCCESS: bind_tools works!")
        
        # Now test the actual chat
        print("\n=== Testing actual chat ===")
        chat_result = await agent.chat("查看最近的登录事件")
        print(f"Chat result: {chat_result}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_one_tool_call_debug())