"""
简化的 Agent 测试脚本
用于快速验证 Agent 是否能正常调用工具并返回结果
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.agent import OPGReActAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化 Agent 测试")
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="../test_data",
        help="test_data 目录路径"
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default="4a2e27ba-991a-48bc-a7f6-0188fc41c52e",
        help="测试样例 ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/test_simple",
        help="输出目录"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="请简要分析这张 OPG 全景片，列出主要发现。",
        help="用户问题"
    )
    
    args = parser.parse_args()
    
    # 查找测试图像
    test_data_dir = Path(args.test_data_dir)
    sample_dir = test_data_dir / args.sample_id
    
    image_files = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
    if not image_files:
        logger.error(f"未找到图像文件: {sample_dir}")
        return
    
    image_path = str(image_files[0])
    logger.info(f"使用测试图像: {image_path}")
    
    # 初始化 Agent
    config_path = Path(__file__).parent.parent / "src" / "opgagent" / "config" / "agent_config.yaml"
    agent = OPGReActAgent(config_path=str(config_path) if config_path.exists() else None)
    
    # 运行 Agent
    logger.info("开始运行 Agent...")
    try:
        result = agent.run(
            question=args.question,
            image_path=image_path
        )
        
        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "question": args.question,
            "image_path": image_path,
            "answer": result.get("answer", ""),
            "tool_calls": result.get("tool_calls", []),
            "token_usage": result.get("token_usage", {}),
            "memory_summary": result.get("memory", {}).get_summary() if result.get("memory") else None
        }
        
        output_file = output_dir / "agent_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存: {output_file}")
        logger.info(f"工具调用数: {len(result.get('tool_calls', []))}")
        logger.info(f"答案长度: {len(result.get('answer', ''))} 字符")
        
        if result.get("token_usage"):
            token_usage = result.get("token_usage", {})
            if token_usage.get("enabled"):
                logger.info(f"Token 使用: {token_usage.get('total_tokens', 0):,} / {token_usage.get('token_limit', 0):,}")
        
        return output_data
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    main()
