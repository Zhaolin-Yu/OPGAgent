"""
测试工具定义
"""

from opgagent.tools import create_dental_tools


def test_create_dental_tools():
    """测试创建工具列表"""
    tools_config = {}
    
    # 测试包含所有工具
    tools = create_dental_tools(tools_config, analysis_only=False)
    
    assert len(tools) > 0
    
    # 检查是否包含高级检测工具（run_all_detections 不暴露为工具，由 Agent 启动时预加载）
    tool_names = [tool.name for tool in tools]
    assert "get_tooth_by_fdi" in tool_names
    assert "list_teeth_with_status" in tool_names
    assert "get_status_on_tooth" in tool_names

    # 检查是否包含 4 个 VLM 分析工具
    assert "dental_expert_analysis" in tool_names
    assert "oral_expert_analysis" in tool_names
    assert "llm_zoo_openai" in tool_names
    assert "llm_zoo_google" in tool_names


def test_create_dental_tools_analysis_only():
    """测试只创建分析工具"""
    tools_config = {}
    
    tools = create_dental_tools(tools_config, analysis_only=True)
    
    tool_names = [tool.name for tool in tools]
    
    # 应该只包含 4 个 VLM 分析工具
    assert "dental_expert_analysis" in tool_names
    assert "oral_expert_analysis" in tool_names
    assert "llm_zoo_openai" in tool_names
    assert "llm_zoo_google" in tool_names

    # 不应该包含检测/高级工具
    assert "get_tooth_by_fdi" not in tool_names
    assert "list_teeth_with_status" not in tool_names