"""
多工具组合测试脚本
参考 Agent_refactor 预处理管道的工具调用顺序，测试多工具组合使用
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
from opgagent.tools.coordinate_utils import merge_detection_results, match_diseases_to_teeth
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


def convert_quadrants_to_dict(quadrants_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    将象限检测结果（列表格式）转换为字典格式
    
    Args:
        quadrants_result: 工具返回的结果 {"detections": [{"class_name": "...", "bbox": [...], ...}, ...]}
        
    Returns:
        字典格式 {quadrant_id: {"name": "...", "box": [...], "confidence": ...}}
    """
    quadrants_dict = {}
    detections = quadrants_result.get("detections", [])
    
    for idx, det in enumerate(detections):
        class_name = det.get("class_name") or det.get("class") or f"quadrant_{idx}"
        # 标准化象限名称
        name_mapping = {
            "Upper Right": "Upperright",
            "Upper Left": "Upperleft",
            "Lower Left": "Lowerleft",
            "Lower Right": "Lowerright",
            "class_0": "Upperleft",  # 根据 Agent_refactor 的映射
        }
        quadrant_name = name_mapping.get(class_name, class_name)
        
        bbox = det.get("bbox") or det.get("box") or []
        confidence = det.get("confidence", 0.0)
        
        quadrants_dict[quadrant_name] = {
            "name": quadrant_name,
            "box": bbox,
            "confidence": confidence
        }
    
    return quadrants_dict


def convert_teeth_to_dict(teeth_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    将牙齿检测结果（列表格式）转换为字典格式
    
    Args:
        teeth_result: 工具返回的结果 {"detections": [{"class_name": "1", "bbox": {...}, ...}, ...]}
        
    Returns:
        字典格式 {tooth_id: {"number": "1", "box": [...], "confidence": ...}}
    """
    teeth_dict = {}
    detections = teeth_result.get("detections", [])
    
    for idx, det in enumerate(detections):
        tooth_id = f"t{idx}"
        class_name = det.get("class_name") or det.get("class") or str(idx + 1)
        tooth_number = str(class_name)
        
        # 处理 bbox 格式（可能是 dict 或 list）
        bbox = det.get("bbox") or det.get("box") or []
        if isinstance(bbox, dict):
            # 转换为 [x1, y1, x2, y2] 格式
            bbox = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
        
        confidence = det.get("confidence", 0.0)
        
        teeth_dict[tooth_id] = {
            "number": tooth_number,
            "box": bbox,
            "confidence": confidence
        }
    
    return teeth_dict


def test_combo_a(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> Dict[str, Any]:
    """
    组合测试用例 A：参考预处理顺序
    1. quadrant_detection (TVEM 4quadrants)
    2. tooth_enumeration (YOLO enumeration)
    3. calculate_fdi (合并象限和牙齿结果)
    5. disease_detection_tvem (11diseases)
    6. bone_loss_detection
    7. anatomy_detection
    8. match_disease_to_tooth (匹配疾病到牙齿)
    """
    logger.info("\n" + "="*80)
    logger.info("组合测试用例 A：预处理管道顺序")
    logger.info("="*80)
    
    results = {}
    errors = []
    
    # 1. 象限检测
    logger.info("\n[1/8] 象限检测（TVEM 4quadrants）...")
    try:
        quadrants_result = toolkit.quadrant_detection(image_path, confidence_threshold=0.5, config=config)
        quadrants_dict = json.loads(quadrants_result)
        if "error" in quadrants_dict:
            errors.append(f"象限检测失败: {quadrants_dict['error']}")
        else:
            results["quadrants"] = quadrants_dict
            num_quadrants = len(quadrants_dict.get("detections", []))
            logger.info(f"  ✓ 检测到 {num_quadrants} 个象限")
    except Exception as e:
        errors.append(f"象限检测异常: {e}")
        logger.error(f"  ✗ 象限检测异常: {e}")
    
    # 2. 牙齿编号
    logger.info("\n[2/8] 牙齿编号（YOLO enumeration）...")
    try:
        teeth_result = toolkit.tooth_enumeration(image_path, config=config)
        teeth_dict = json.loads(teeth_result)
        if "error" in teeth_dict:
            errors.append(f"牙齿编号失败: {teeth_dict['error']}")
        else:
            results["teeth"] = teeth_dict
            num_teeth = len(teeth_dict.get("detections", []))
            logger.info(f"  ✓ 检测到 {num_teeth} 颗牙齿")
    except Exception as e:
        errors.append(f"牙齿编号异常: {e}")
        logger.error(f"  ✗ 牙齿编号异常: {e}")
    
    # 3. 计算 FDI（合并象限和牙齿结果）
    logger.info("\n[3/8] 计算 FDI（合并象限和牙齿结果）...")
    try:
        if "quadrants" in results and "teeth" in results:
            # 转换工具返回的列表格式为字典格式
            quadrants_dict = convert_quadrants_to_dict(results["quadrants"])
            teeth_dict = convert_teeth_to_dict(results["teeth"])
            
            fdi_result = toolkit.calculate_fdi(
                quadrants=quadrants_dict,
                teeth=teeth_dict
            )
            fdi_dict = json.loads(fdi_result)
            if "error" in fdi_dict:
                errors.append(f"FDI 计算失败: {fdi_dict['error']}")
            else:
                results["fdi"] = fdi_dict
                num_fdi = len(fdi_dict) if isinstance(fdi_dict, dict) else 0
                logger.info(f"  ✓ 计算出 {num_fdi} 个 FDI 编号")
        else:
            errors.append("FDI 计算跳过：缺少象限或牙齿检测结果")
            logger.warning("  ⚠ FDI 计算跳过：缺少前置结果")
    except Exception as e:
        errors.append(f"FDI 计算异常: {e}")
        logger.error(f"  ✗ FDI 计算异常: {e}", exc_info=True)
    
    # 5. TVEM 11diseases 检测
    logger.info("\n[5/8] TVEM 11diseases 检测...")
    try:
        tvem_disease_result = toolkit.disease_detection_tvem(
            image_path, confidence=0.5, return_vis=False, config=config
        )
        tvem_disease_dict = json.loads(tvem_disease_result)
        if "error" in tvem_disease_dict:
            errors.append(f"TVEM 疾病检测失败: {tvem_disease_dict['error']}")
        else:
            results["tvem_disease"] = tvem_disease_dict
            num_tvem = len(tvem_disease_dict.get("detections", []))
            logger.info(f"  ✓ 检测到 {num_tvem} 个 TVEM 疾病发现")
    except Exception as e:
        errors.append(f"TVEM 疾病检测异常: {e}")
        logger.error(f"  ✗ TVEM 疾病检测异常: {e}")
    
    # 6. 骨吸收检测
    logger.info("\n[6/8] 骨吸收检测...")
    try:
        bone_loss_result = toolkit.bone_loss_detection(
            image_path, confidence=0.5, return_vis=False, config=config
        )
        bone_loss_dict = json.loads(bone_loss_result)
        if "error" in bone_loss_dict:
            errors.append(f"骨吸收检测失败: {bone_loss_dict['error']}")
        else:
            results["bone_loss"] = bone_loss_dict
            num_bone = len(bone_loss_dict.get("detections", []))
            logger.info(f"  ✓ 检测到 {num_bone} 个骨吸收区域")
    except Exception as e:
        errors.append(f"骨吸收检测异常: {e}")
        logger.error(f"  ✗ 骨吸收检测异常: {e}")
    
    # 7. 解剖结构检测
    logger.info("\n[7/8] 解剖结构检测...")
    try:
        anatomy_result = toolkit.anatomy_detection(
            image_path, confidence=0.5, return_vis=False, config=config
        )
        anatomy_dict = json.loads(anatomy_result)
        if "error" in anatomy_dict:
            errors.append(f"解剖结构检测失败: {anatomy_dict['error']}")
        else:
            results["anatomy"] = anatomy_dict
            num_anatomy = len(anatomy_dict.get("detections", []))
            logger.info(f"  ✓ 检测到 {num_anatomy} 个解剖结构")
    except Exception as e:
        errors.append(f"解剖结构检测异常: {e}")
        logger.error(f"  ✗ 解剖结构检测异常: {e}")
    
    # 8. 匹配疾病到牙齿
    logger.info("\n[8/8] 匹配疾病到牙齿...")
    try:
        if "tvem_disease" in results and "fdi" in results:
            diseases = results["tvem_disease"].get("detections", [])
            # fdi_result 是字典，key 是 FDI 编号，value 是牙齿信息
            teeth = results["fdi"] if isinstance(results["fdi"], dict) else {}
            
            # 转换疾病 bbox 格式（如果是 dict 格式，转换为 list）
            for disease in diseases:
                bbox = disease.get("bbox") or disease.get("box")
                if isinstance(bbox, dict):
                    disease["box"] = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
                elif bbox:
                    disease["box"] = bbox
            
            match_result = toolkit.match_disease_to_tooth(
                diseases=diseases,
                teeth=teeth,
                iou_threshold=0.3
            )
            match_dict = json.loads(match_result)
            if "error" in match_dict:
                errors.append(f"疾病匹配失败: {match_dict['error']}")
            else:
                results["matched_diseases"] = match_dict
                # match_disease_to_tooth 返回的是 {tooth_id: [disease1, ...]} 格式
                num_matched = sum(len(diseases) for diseases in match_dict.values() if isinstance(diseases, list))
                logger.info(f"  ✓ 匹配了 {num_matched} 个疾病到牙齿")
        else:
            errors.append("疾病匹配跳过：缺少疾病或 FDI 结果")
            logger.warning("  ⚠ 疾病匹配跳过：缺少前置结果")
    except Exception as e:
        errors.append(f"疾病匹配异常: {e}")
        logger.error(f"  ✗ 疾病匹配异常: {e}", exc_info=True)
    
    # 汇总
    logger.info("\n" + "="*80)
    logger.info("组合测试用例 A 完成")
    logger.info("="*80)
    logger.info(f"成功步骤: {len(results)}/8")
    if errors:
        logger.warning(f"错误/警告: {len(errors)} 个")
        for err in errors:
            logger.warning(f"  - {err}")
    
    return {
        "test_case": "A",
        "description": "预处理管道顺序",
        "status": "success" if len(errors) == 0 else "partial",
        "results": results,
        "errors": errors,
        "summary": {
            "total_steps": 8,
            "completed_steps": len(results),
            "error_count": len(errors)
        }
    }


def test_combo_b(toolkit: DentalToolkit, image_path: str, config: RunnableConfig) -> Dict[str, Any]:
    """
    组合测试用例 B：在用例 A 基础上增加 VLM 分析
    """
    logger.info("\n" + "="*80)
    logger.info("组合测试用例 B：检测 + VLM 分析")
    logger.info("="*80)
    
    # 先运行用例 A
    combo_a_result = test_combo_a(toolkit, image_path, config)
    results = combo_a_result["results"]
    errors = combo_a_result["errors"]
    
    # 9. DentalGPT 分析
    logger.info("\n[9/10] DentalGPT 整体分析...")
    try:
        dental_result = toolkit.dental_expert_analysis(
            image_path,
            analysis_type="overall",
            config=config
        )
        dental_dict = json.loads(dental_result)
        if "error" in dental_dict:
            errors.append(f"DentalGPT 分析失败: {dental_dict['error']}")
        else:
            results["dental_gpt"] = dental_dict
            analysis_text = dental_dict.get("analysis", "")
            if analysis_text:
                logger.info(f"  ✓ DentalGPT 分析完成（{len(analysis_text)} 字符）")
            else:
                logger.warning("  ⚠ DentalGPT 分析结果为空")
    except Exception as e:
        errors.append(f"DentalGPT 分析异常: {e}")
        logger.error(f"  ✗ DentalGPT 分析异常: {e}")
    
    # 10. OralGPT 分析
    logger.info("\n[10/10] OralGPT 整体分析...")
    try:
        oral_result = toolkit.oral_expert_analysis(
            image_path,
            analysis_type="overall",
            config=config
        )
        oral_dict = json.loads(oral_result)
        if "error" in oral_dict:
            errors.append(f"OralGPT 分析失败: {oral_dict['error']}")
        else:
            results["oral_gpt"] = oral_dict
            analysis_text = oral_dict.get("analysis", "")
            if analysis_text:
                logger.info(f"  ✓ OralGPT 分析完成（{len(analysis_text)} 字符）")
            else:
                logger.warning("  ⚠ OralGPT 分析结果为空")
    except Exception as e:
        errors.append(f"OralGPT 分析异常: {e}")
        logger.error(f"  ✗ OralGPT 分析异常: {e}")
    
    # 汇总
    logger.info("\n" + "="*80)
    logger.info("组合测试用例 B 完成")
    logger.info("="*80)
    logger.info(f"成功步骤: {len(results)}/10")
    if errors:
        logger.warning(f"错误/警告: {len(errors)} 个")
        for err in errors:
            logger.warning(f"  - {err}")
    
    return {
        "test_case": "B",
        "description": "检测 + VLM 分析",
        "status": "success" if len(errors) == 0 else "partial",
        "results": results,
        "errors": errors,
        "summary": {
            "total_steps": 10,
            "completed_steps": len(results),
            "error_count": len(errors)
        }
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多工具组合测试")
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
        "--test_case",
        type=str,
        choices=["A", "B", "both"],
        default="both",
        help="测试用例：A（预处理顺序）、B（+VLM）、both（两者都测试）"
    )
    parser.add_argument(
        "--skip_vlm",
        action="store_true",
        help="跳过 VLM 工具（仅测试用例 A）"
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
    
    abs_image_path = str(Path(image_path).resolve())
    config = RunnableConfig(configurable={"current_image_path": abs_image_path})
    token = current_image_path_ctx.set(abs_image_path)
    
    try:
        all_results = []
        
        # 测试用例 A
        if args.test_case in ["A", "both"]:
            result_a = test_combo_a(toolkit, abs_image_path, config)
            all_results.append(result_a)
        
        # 测试用例 B
        if args.test_case in ["B", "both"] and not args.skip_vlm:
            result_b = test_combo_b(toolkit, abs_image_path, config)
            all_results.append(result_b)
        
        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "test_image": image_path,
            "test_sample": sample_dir.name,
            "test_cases": all_results,
            "overall_status": "success" if all(r["status"] == "success" for r in all_results) else "partial"
        }
        
        output_file = output_dir / "combo_tools_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n结果已保存: {output_file}")
        
        # 返回退出码
        if summary["overall_status"] != "success":
            sys.exit(1)
        else:
            sys.exit(0)
            
    finally:
        current_image_path_ctx.reset(token)


if __name__ == "__main__":
    main()
