"""
单工具连通性测试脚本
逐个测试所有工具的连通性，确保工具服务正常运行
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.tools.dental_tools import DentalToolkit
from opgagent.tools.dental_tools import current_image_path_ctx
from langchain_core.runnables import RunnableConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_tools_config() -> Dict[str, Any]:
    """加载工具配置"""
    config_path = Path(__file__).parent.parent / "src" / "opgagent" / "config" / "tools_config.yaml"
    if not config_path.exists():
        logger.warning(f"工具配置文件不存在: {config_path}，使用空配置")
        return {}
    
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_tool(
    toolkit: DentalToolkit,
    tool_name: str,
    image_path: str,
    test_func,
    description: str
) -> Dict[str, Any]:
    """
    测试单个工具
    
    Args:
        toolkit: DentalToolkit 实例
        tool_name: 工具名称
        image_path: 图像路径
        test_func: 测试函数（接受 toolkit, image_path, config 参数）
        description: 工具描述
        
    Returns:
        测试结果字典
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"测试工具: {tool_name} - {description}")
    logger.info(f"{'='*80}")
    
    abs_image_path = str(Path(image_path).resolve())
    config = RunnableConfig(configurable={"current_image_path": abs_image_path})
    token = current_image_path_ctx.set(abs_image_path)
    
    try:
        result = test_func(toolkit, abs_image_path, config)
        
        # 检查结果
        if isinstance(result, str):
            try:
                result_dict = json.loads(result)
            except:
                result_dict = {"raw_output": result}
        else:
            result_dict = result
        
        # 判断是否成功
        # 注意：有些工具返回 {"success": true, "error": null}，这种情况应该视为成功
        error_value = result_dict.get("error")
        has_error = error_value is not None and error_value != "null" and str(error_value).lower() != "none"
        # 如果 result_dict 有 "success" 字段，优先使用它
        if "success" in result_dict:
            is_success = result_dict.get("success") is True and not has_error
        else:
            is_success = not has_error
        
        if is_success:
            logger.info(f"✓ {tool_name} 测试通过")
            if "detections" in result_dict:
                num_detections = len(result_dict.get("detections", []))
                logger.info(f"  检测到 {num_detections} 个结果")
            elif "analysis" in result_dict:
                analysis_len = len(str(result_dict.get("analysis", "")))
                logger.info(f"  分析结果长度: {analysis_len} 字符")
        else:
            error_msg = result_dict.get("error", "未知错误")
            logger.error(f"✗ {tool_name} 测试失败: {error_msg}")
        
        return {
            "tool_name": tool_name,
            "description": description,
            "status": "success" if is_success else "failed",
            "result": result_dict,
            "error": result_dict.get("error") if has_error else None
        }
        
    except Exception as e:
        logger.error(f"✗ {tool_name} 测试异常: {e}", exc_info=True)
        return {
            "tool_name": tool_name,
            "description": description,
            "status": "error",
            "error": str(e),
            "result": None
        }
    finally:
        current_image_path_ctx.reset(token)


def test_quadrant_detection(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试象限检测"""
    return toolkit.quadrant_detection(image_path, confidence_threshold=0.5, config=config)


def test_tooth_enumeration(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试牙齿编号"""
    return toolkit.tooth_enumeration(image_path, config=config)


def test_disease_detection_tvem(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试 TVEM 疾病检测"""
    return toolkit.disease_detection_tvem(image_path, confidence=0.5, return_vis=False, config=config)


def test_bone_loss_detection(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试骨吸收检测"""
    return toolkit.bone_loss_detection(image_path, confidence=0.5, return_vis=False, config=config)


def test_anatomy_detection(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试解剖结构检测"""
    return toolkit.anatomy_detection(image_path, confidence=0.5, return_vis=False, config=config)


def test_segment_object(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试 MedSAM 分割（需要先有 bbox，这里使用示例 bbox）"""
    # 使用一个示例 bbox（图像中心区域）
    # 实际使用中应该从 quadrant_detection 或 tooth_enumeration 的结果中获取
    example_bbox = [[100, 100, 200, 200]]  # [x1, y1, x2, y2]
    return toolkit.segment_object(image_path, boxes=example_bbox, config=config)


def test_dental_expert_analysis(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试 DentalGPT 分析"""
    return toolkit.dental_expert_analysis(
        image_path,
        analysis_type="overall",
        config=config
    )


def test_oral_expert_analysis(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> str:
    """测试 OralGPT 分析"""
    return toolkit.oral_expert_analysis(
        image_path,
        analysis_type="overall",
        config=config
    )


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="单工具连通性测试")
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="../test_data",
        help="test_data 目录路径"
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="测试样例 ID（默认使用第一个）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/tools_test",
        help="输出目录"
    )
    parser.add_argument(
        "--skip_vlm",
        action="store_true",
        help="跳过 VLM 工具测试（dental_gpt, oral_gpt）"
    )
    
    args = parser.parse_args()
    
    # 加载工具配置
    tools_config = load_tools_config()
    toolkit = DentalToolkit(tools_config)
    
    # 查找测试图像
    test_data_dir = Path(args.test_data_dir)
    if args.sample_id:
        sample_dir = test_data_dir / args.sample_id
    else:
        sample_dirs = [d for d in test_data_dir.iterdir() if d.is_dir()]
        if not sample_dirs:
            logger.error(f"未找到测试样例: {test_data_dir}")
            return
        sample_dir = sample_dirs[0]
    
    image_files = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
    if not image_files:
        logger.error(f"未找到图像文件: {sample_dir}")
        return
    
    image_path = str(image_files[0])
    logger.info(f"使用测试图像: {image_path}")
    logger.info(f"测试样例: {sample_dir.name}")
    
    # 定义测试用例
    test_cases = [
        ("quadrant_detection", "象限检测（TVEM 4quadrants）", test_quadrant_detection),
        ("tooth_enumeration", "牙齿编号（YOLO enumeration）", test_tooth_enumeration),
        ("disease_detection_tvem", "TVEM 11diseases 检测", test_disease_detection_tvem),
        ("bone_loss_detection", "骨吸收检测", test_bone_loss_detection),
        ("anatomy_detection", "解剖结构检测", test_anatomy_detection),
        ("segment_object", "MedSAM 分割（示例 bbox）", test_segment_object),
    ]
    
    if not args.skip_vlm:
        test_cases.extend([
            ("dental_expert_analysis", "DentalGPT 分析", test_dental_expert_analysis),
            ("oral_expert_analysis", "OralGPT 分析", test_oral_expert_analysis),
        ])
    
    # 运行测试
    results = []
    for tool_name, description, test_func in test_cases:
        result = test_tool(toolkit, tool_name, image_path, test_func, description)
        results.append(result)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "test_image": image_path,
        "test_sample": sample_dir.name,
        "total_tests": len(results),
        "passed": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }
    
    output_file = output_dir / "single_tools_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("测试汇总")
    logger.info(f"{'='*80}")
    logger.info(f"总测试数: {summary['total_tests']}")
    logger.info(f"通过: {summary['passed']}")
    logger.info(f"失败: {summary['failed']}")
    logger.info(f"异常: {summary['errors']}")
    logger.info(f"\n结果已保存: {output_file}")
    
    # 返回退出码
    if summary["failed"] > 0 or summary["errors"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
