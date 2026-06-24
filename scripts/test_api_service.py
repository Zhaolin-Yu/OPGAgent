#!/usr/bin/env python3
"""
测试 api_service 工具服务配置

用于验证新的 api_service 配置（端口从 6600 开始）是否正常工作。
可以指定使用哪套工具服务配置进行测试。

使用方式:
    # 测试新的 api_service 配置
    python test_api_service.py --tool-service api_service --image /path/to/image.jpg
    
    # 测试默认配置
    python test_api_service.py --tool-service default --image /path/to/image.jpg
    
    # 仅检查服务健康状态（不需要图像）
    python test_api_service.py --tool-service api_service --health-only
"""

import argparse
import json
import logging
import sys
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 工具服务端口配置
SERVICE_CONFIGS = {
    "default": {
        "yolo_enumeration": 8002,
        "tvem": 8003,
        "medsam": 8008,
        "oral_gpt": [8664, 8665, 8666, 8667],
        "dental_gpt": [8566, 8567, 8568, 8569],
    },
    "api_service": {
        "yolo_enumeration": 6600,
        "tvem": 6602,
        "medsam": 6603,
        "oral_gpt": [6604, 6605, 6606, 6607],
        "dental_gpt": [6608, 6609, 6610, 6611],
    }
}


def check_health(port: int, service_name: str) -> Dict[str, Any]:
    """检查服务健康状态"""
    url = f"http://localhost:{port}/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return {
                "service": service_name,
                "port": port,
                "status": "healthy",
                "response": response.json()
            }
        else:
            return {
                "service": service_name,
                "port": port,
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "service": service_name,
            "port": port,
            "status": "unreachable",
            "error": "Connection refused"
        }
    except Exception as e:
        return {
            "service": service_name,
            "port": port,
            "status": "error",
            "error": str(e)
        }


def check_all_services(tool_service: str) -> Dict[str, Any]:
    """检查指定配置下的所有服务"""
    config = SERVICE_CONFIGS.get(tool_service, SERVICE_CONFIGS["api_service"])
    
    results = {
        "config": tool_service,
        "services": [],
        "summary": {
            "total": 0,
            "healthy": 0,
            "unhealthy": 0,
            "unreachable": 0
        }
    }
    
    for service_name, ports in config.items():
        if isinstance(ports, list):
            # 多副本服务
            for i, port in enumerate(ports):
                result = check_health(port, f"{service_name}_{i}")
                results["services"].append(result)
                results["summary"]["total"] += 1
                if result["status"] == "healthy":
                    results["summary"]["healthy"] += 1
                elif result["status"] == "unreachable":
                    results["summary"]["unreachable"] += 1
                else:
                    results["summary"]["unhealthy"] += 1
        else:
            # 单实例服务
            result = check_health(ports, service_name)
            results["services"].append(result)
            results["summary"]["total"] += 1
            if result["status"] == "healthy":
                results["summary"]["healthy"] += 1
            elif result["status"] == "unreachable":
                results["summary"]["unreachable"] += 1
            else:
                results["summary"]["unhealthy"] += 1
    
    return results


def load_tools_config(tool_service: str) -> Dict[str, Any]:
    """加载工具配置"""
    opgagent_root = Path(__file__).parent.parent
    
    if tool_service == "api_service":
        config_path = opgagent_root / "api_service" / "tools_config.yaml"
    else:
        config_path = opgagent_root / "src" / "opgagent" / "config" / "tools_config.yaml"
    
    if not config_path.exists():
        logger.warning(f"工具配置文件不存在: {config_path}")
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_detection_tools(toolkit, image_path: str, config) -> Dict[str, Any]:
    """测试检测类工具"""
    from opgagent.tools.dental_tools import current_image_path_ctx
    
    results = {}
    
    # 测试 run_all_detections
    logger.info("\n=== 测试 run_all_detections ===")
    try:
        abs_image_path = str(Path(image_path).resolve())
        token = current_image_path_ctx.set(abs_image_path)
        try:
            result = toolkit.run_all_detections(abs_image_path, config=config)
            result_dict = json.loads(result)
            
            teeth_count = len(result_dict.get("teeth_fdi", {}))
            quadrant_count = len([k for k in result_dict.get("quadrants", {}).keys() if k not in ["error"]])
            
            results["run_all_detections"] = {
                "status": "success",
                "teeth_count": teeth_count,
                "quadrant_count": quadrant_count
            }
            logger.info(f"✓ run_all_detections: {teeth_count} 颗牙齿, {quadrant_count} 个象限")
        finally:
            current_image_path_ctx.reset(token)
    except Exception as e:
        results["run_all_detections"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"✗ run_all_detections: {e}")
    
    # 测试 get_tooth_by_fdi
    logger.info("\n=== 测试 get_tooth_by_fdi ===")
    try:
        abs_image_path = str(Path(image_path).resolve())
        token = current_image_path_ctx.set(abs_image_path)
        try:
            result = toolkit.get_tooth_by_fdi(abs_image_path, "11", config=config)
            result_dict = json.loads(result)
            
            if "error" not in result_dict or result_dict.get("error") is None:
                results["get_tooth_by_fdi"] = {
                    "status": "success",
                    "tooth": result_dict.get("number")
                }
                logger.info(f"✓ get_tooth_by_fdi: 牙齿 {result_dict.get('number')}")
            else:
                # 如果 11 号牙齿不存在，尝试其他牙齿
                available_fdi = result_dict.get("available_fdi", [])
                if available_fdi:
                    results["get_tooth_by_fdi"] = {
                        "status": "partial",
                        "available_fdi": available_fdi[:5]  # 只显示前5个
                    }
                    logger.info(f"△ get_tooth_by_fdi: 11号不存在，可用FDI: {available_fdi[:5]}")
                else:
                    results["get_tooth_by_fdi"] = {
                        "status": "error",
                        "error": result_dict.get("error")
                    }
                    logger.error(f"✗ get_tooth_by_fdi: {result_dict.get('error')}")
        finally:
            current_image_path_ctx.reset(token)
    except Exception as e:
        results["get_tooth_by_fdi"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"✗ get_tooth_by_fdi: {e}")
    
    # 测试 get_quadrant
    logger.info("\n=== 测试 get_quadrant ===")
    try:
        abs_image_path = str(Path(image_path).resolve())
        token = current_image_path_ctx.set(abs_image_path)
        try:
            result = toolkit.get_quadrant(abs_image_path, "Q1", config=config)
            result_dict = json.loads(result)
            
            if "error" not in result_dict or result_dict.get("error") is None:
                teeth_fdi = result_dict.get("teeth_fdi", [])
                results["get_quadrant"] = {
                    "status": "success",
                    "quadrant": "Q1",
                    "teeth_count": len(teeth_fdi)
                }
                logger.info(f"✓ get_quadrant: Q1 有 {len(teeth_fdi)} 颗牙齿")
            else:
                results["get_quadrant"] = {
                    "status": "error",
                    "error": result_dict.get("error")
                }
                logger.error(f"✗ get_quadrant: {result_dict.get('error')}")
        finally:
            current_image_path_ctx.reset(token)
    except Exception as e:
        results["get_quadrant"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"✗ get_quadrant: {e}")
    
    return results


def test_vlm_tools(toolkit, image_path: str, config) -> Dict[str, Any]:
    """测试 VLM 分析工具"""
    from opgagent.tools.dental_tools import current_image_path_ctx
    
    results = {}
    abs_image_path = str(Path(image_path).resolve())
    
    # 测试 dental_expert_analysis (DentalGPT)
    logger.info("\n=== 测试 dental_expert_analysis (DentalGPT) ===")
    try:
        token = current_image_path_ctx.set(abs_image_path)
        try:
            result = toolkit.dental_expert_analysis(
                image_path=abs_image_path,
                analysis_type="overall",
                custom_prompt="Please briefly describe this dental X-ray image.",
                config=config
            )
            result_dict = json.loads(result)
            
            if "error" not in result_dict or result_dict.get("error") is None:
                analysis_len = len(str(result_dict.get("analysis", "")))
                results["dental_expert_analysis"] = {
                    "status": "success",
                    "analysis_length": analysis_len
                }
                logger.info(f"✓ dental_expert_analysis: 分析结果 {analysis_len} 字符")
            else:
                results["dental_expert_analysis"] = {
                    "status": "error",
                    "error": result_dict.get("error")
                }
                logger.error(f"✗ dental_expert_analysis: {result_dict.get('error')}")
        finally:
            current_image_path_ctx.reset(token)
    except Exception as e:
        results["dental_expert_analysis"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"✗ dental_expert_analysis: {e}")
    
    # 测试 oral_expert_analysis (OralGPT)
    logger.info("\n=== 测试 oral_expert_analysis (OralGPT) ===")
    try:
        token = current_image_path_ctx.set(abs_image_path)
        try:
            result = toolkit.oral_expert_analysis(
                image_path=abs_image_path,
                analysis_type="overall",
                custom_prompt="Please briefly describe this dental X-ray image.",
                config=config
            )
            result_dict = json.loads(result)
            
            if "error" not in result_dict or result_dict.get("error") is None:
                analysis_len = len(str(result_dict.get("analysis", "")))
                results["oral_expert_analysis"] = {
                    "status": "success",
                    "analysis_length": analysis_len
                }
                logger.info(f"✓ oral_expert_analysis: 分析结果 {analysis_len} 字符")
            else:
                results["oral_expert_analysis"] = {
                    "status": "error",
                    "error": result_dict.get("error")
                }
                logger.error(f"✗ oral_expert_analysis: {result_dict.get('error')}")
        finally:
            current_image_path_ctx.reset(token)
    except Exception as e:
        results["oral_expert_analysis"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"✗ oral_expert_analysis: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="测试 api_service 工具服务配置")
    parser.add_argument(
        "--tool-service",
        type=str,
        default="api_service",
        choices=["default", "api_service"],
        help="工具服务配置: default (原端口 8xxx) 或 api_service (新端口 6600 开始)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="测试用图像路径"
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="仅检查服务健康状态，不进行工具测试"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"\n{'='*60}")
    print(f"测试工具服务配置: {args.tool_service}")
    print(f"{'='*60}\n")
    
    # 检查服务健康状态
    print("=== 服务健康检查 ===\n")
    health_results = check_all_services(args.tool_service)
    
    for service in health_results["services"]:
        status = service["status"]
        if status == "healthy":
            print(f"✓ {service['service']:20} (端口 {service['port']}) - 健康")
        elif status == "unreachable":
            print(f"✗ {service['service']:20} (端口 {service['port']}) - 不可达")
        else:
            print(f"△ {service['service']:20} (端口 {service['port']}) - {service.get('error', '未知错误')}")
    
    summary = health_results["summary"]
    print(f"\n总计: {summary['total']} | 健康: {summary['healthy']} | 不可达: {summary['unreachable']} | 不健康: {summary['unhealthy']}")
    
    if args.health_only:
        print("\n仅健康检查模式，跳过工具测试。")
        return
    
    if not args.image:
        print("\n未指定图像路径，跳过工具测试。使用 --image 参数指定图像路径。")
        return
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n错误: 图像文件不存在: {args.image}")
        return
    
    # 加载工具配置并创建 toolkit
    print(f"\n=== 加载工具配置: {args.tool_service} ===\n")
    tools_config = load_tools_config(args.tool_service)
    
    if not tools_config:
        print("错误: 无法加载工具配置")
        return
    
    from opgagent.tools.dental_tools import DentalToolkit
    from langchain_core.runnables import RunnableConfig
    
    toolkit = DentalToolkit(tools_config)
    abs_image_path = str(image_path.resolve())
    config = RunnableConfig(configurable={"current_image_path": abs_image_path})
    
    # 测试检测类工具
    print("\n=== 测试检测类工具 ===\n")
    detection_results = test_detection_tools(toolkit, abs_image_path, config)
    
    # 如果有可用的 VLM 服务，测试 VLM 工具
    vlm_healthy = any(
        s["status"] == "healthy" 
        for s in health_results["services"] 
        if "oral_gpt" in s["service"] or "dental_gpt" in s["service"]
    )
    
    if vlm_healthy:
        print("\n=== 测试 VLM 工具 ===\n")
        vlm_results = test_vlm_tools(toolkit, abs_image_path, config)
    else:
        print("\n跳过 VLM 工具测试（服务不可用）")
        vlm_results = {}
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    all_results = {**detection_results, **vlm_results}
    success_count = sum(1 for r in all_results.values() if r.get("status") == "success")
    partial_count = sum(1 for r in all_results.values() if r.get("status") == "partial")
    error_count = sum(1 for r in all_results.values() if r.get("status") == "error")
    
    print(f"\n工具测试: 成功 {success_count} | 部分成功 {partial_count} | 失败 {error_count}")
    print(f"服务健康: {summary['healthy']}/{summary['total']}")
    
    if summary["unreachable"] > 0:
        print(f"\n警告: {summary['unreachable']} 个服务不可达，请检查服务是否已启动")
        print("启动命令: cd Agent_v3/api_service && bash start_all_services.sh")


if __name__ == "__main__":
    main()
