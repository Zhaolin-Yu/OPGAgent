"""
测试 AgentMemory
"""

import json
import tempfile
from pathlib import Path

from opgagent.memory import AgentMemory


def test_memory_initialization():
    """测试 Memory 初始化"""
    memory = AgentMemory(
        image_path="/path/to/image.jpg",
        question="测试问题"
    )
    
    assert memory.image_path == "/path/to/image.jpg"
    assert memory.question == "测试问题"
    assert len(memory.tool_calls) == 0
    assert len(memory.reasoning_steps) == 0


def test_add_tool_call():
    """测试添加工具调用"""
    memory = AgentMemory(
        image_path="/path/to/image.jpg",
        question="测试问题"
    )
    
    memory.add_tool_call(
        tool_name="test_tool",
        tool_input={"param": "value"},
        tool_output={"result": "success"},
        iteration=1
    )
    
    assert len(memory.tool_calls) == 1
    assert memory.tool_calls[0]["tool_name"] == "test_tool"
    assert memory.tool_calls[0]["iteration"] == 1


def test_add_reasoning_step():
    """测试添加推理步骤"""
    memory = AgentMemory(
        image_path="/path/to/image.jpg",
        question="测试问题"
    )
    
    memory.add_reasoning_step("第一步：检测象限")
    
    assert len(memory.reasoning_steps) == 1
    assert "第一步" in memory.reasoning_steps[0]["step"]


def test_get_summary():
    """测试获取摘要"""
    memory = AgentMemory(
        image_path="/path/to/image.jpg",
        question="测试问题"
    )
    
    memory.add_tool_call(
        tool_name="test_tool",
        tool_input={},
        tool_output={},
        iteration=1
    )
    
    summary = memory.get_summary()
    
    assert summary["total_tool_calls"] == 1
    assert summary["image_path"] == "/path/to/image.jpg"
    assert summary["question"] == "测试问题"


def test_save_to_file():
    """测试保存到文件"""
    memory = AgentMemory(
        image_path="/path/to/image.jpg",
        question="测试问题"
    )
    
    memory.add_tool_call(
        tool_name="test_tool",
        tool_input={},
        tool_output={},
        iteration=1
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
    
    try:
        memory.save_to_file(temp_path)
        
        # 验证文件存在且内容正确
        assert Path(temp_path).exists()
        
        with open(temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert data["image_path"] == "/path/to/image.jpg"
        assert len(data["tool_calls"]) == 1
    finally:
        Path(temp_path).unlink()
