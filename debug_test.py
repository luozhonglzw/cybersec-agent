#!/usr/bin/env python3

"""Debug script to test bind_tools issue."""

import sys
sys.path.append('.')

from app.core.llm import FakeChatModel, FakeLLMClient
from app.core.agent import SecurityAgent

print("1. Creating FakeChatModel...")
model = FakeChatModel(["test response"])
print(f"FakeChatModel created: {model}")
print(f"Has bind_tools: {hasattr(model, 'bind_tools')}")

print("\n2. Creating FakeLLMClient...")
llm = FakeLLMClient("test reply")
print(f"FakeLLMClient created: {llm}")
print(f"Has _model: {hasattr(llm, '_model')}")
print(f"_model type: {type(llm._model)}")
print(f"_model has bind_tools: {hasattr(llm._model, 'bind_tools')}")

print("\n3. Testing bind_tools call...")
try:
    result = model.bind_tools([])
    print(f"bind_tools result: {result}")
except Exception as e:
    print(f"bind_tools error: {e}")

print("\n4. Creating SecurityAgent...")
agent = SecurityAgent(llm, max_iterations=3)
print(f"SecurityAgent created: {agent}")
print(f"Has _tools: {hasattr(agent, '_tools')}")
print(f"_tools type: {type(agent._tools)}")
print(f"_tools length: {len(agent._tools)}")

print("\n5. Testing bind_tools through agent...")
try:
    result = agent._llm._model.bind_tools(agent._tools)
    print(f"bind_tools through agent result: {result}")
except Exception as e:
    print(f"bind_tools through agent error: {e}")
    import traceback
    traceback.print_exc()