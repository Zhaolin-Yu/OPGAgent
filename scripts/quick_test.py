"""
快速测试脚本 - 运行单个样例
"""

import json
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.agent import OPGReActAgent

def main():
    # 使用第一个测试样例
    test_data_dir = Path(__file__).parent.parent.parent / "test_data"
    test_dirs = [d for d in test_data_dir.iterdir() if d.is_dir()]
    
    if not test_dirs:
        print(f"未找到测试样例: {test_data_dir}")
        return
    
    test_dir = test_dirs[0]
    print(f"使用测试样例: {test_dir.name}")
    
    # 查找图像
    image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    if not image_files:
        print(f"未找到图像文件: {test_dir}")
        return
    
    image_path = image_files[0]
    print(f"图像路径: {image_path}")
    
    # 加载 GT（如果有）
    gt_path = test_dir / "structured_report.json"
    question = "请分析这张 OPG 全景片，生成完整的诊断报告。"
    
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        print(f"GT 已加载: {len(gt.get('teeth', {}))} 颗异常牙齿")
    
    # 初始化 Agent
    config_path = Path(__file__).parent.parent / "src" / "opgagent" / "config" / "agent_config.yaml"
    print("初始化 Agent...")
    agent = OPGReActAgent(config_path=str(config_path) if config_path.exists() else None)
    
    # 运行 Agent
    print(f"\n问题: {question}")
    print("开始运行 Agent...\n")
    
    result = agent.run(
        question=question,
        image_path=str(image_path)
    )
    
    # 输出结果
    print("\n" + "="*80)
    print("Agent 运行结果")
    print("="*80)
    print(f"\n答案:\n{result.get('answer', '')}\n")
    
    # 显示工具调用统计
    memory = result.get("memory")
    if memory:
        summary = memory.get_summary()
        print(f"工具调用统计:")
        print(f"  - 总调用次数: {summary['total_tool_calls']}")
        print(f"  - 检测到的象限数: {summary['detection_summary']['quadrants']}")
        print(f"  - 检测到的牙齿数: {summary['detection_summary']['teeth']}")
        print(f"  - 检测到的疾病数: {summary['detection_summary']['diseases']}")
    
    print("="*80)

if __name__ == "__main__":
    main()
