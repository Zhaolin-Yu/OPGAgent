"""
牙齿枚举检测 API 客户端
提供 Python 接口和 Agent 函数
"""

import requests
import base64
from pathlib import Path
from typing import Optional, Dict, List, Union
import json


class ToothEnumerationClient:
    """牙齿枚举检测客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        初始化客户端
        
        Args:
            base_url: API 服务地址
        """
        self.base_url = base_url.rstrip('/')
        self._check_connection()
    
    def _check_connection(self):
        """检查连接"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ 成功连接到牙齿枚举检测服务: {self.base_url}")
            else:
                print(f"⚠️  服务响应异常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 无法连接到服务: {e}")
            print(f"   请确保 API 服务正在运行: {self.base_url}")
    
    def detect_from_file(
        self,
        image_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        return_image: bool = False
    ) -> Dict:
        """
        从文件检测牙齿
        
        Args:
            image_path: 图像文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
            return_image: 是否返回可视化图像
        
        Returns:
            检测结果字典
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            params = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'return_image': return_image
            }
            
            response = requests.post(
                f"{self.base_url}/detect",
                files=files,
                params=params,
                timeout=30
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"检测失败: {response.status_code} - {response.text}")
    
    def detect_from_base64(
        self,
        image_base64: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        return_image: bool = False
    ) -> Dict:
        """
        从 Base64 检测牙齿
        
        Args:
            image_base64: Base64 编码的图像
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
            return_image: 是否返回可视化图像
        
        Returns:
            检测结果字典
        """
        data = {
            'image_base64': image_base64,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'return_image': return_image
        }
        
        response = requests.post(
            f"{self.base_url}/detect_base64",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"检测失败: {response.status_code} - {response.text}")
    
    def detect_from_bytes(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        return_image: bool = False
    ) -> Dict:
        """
        从字节数据检测牙齿
        
        Args:
            image_bytes: 图像字节数据
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
            return_image: 是否返回可视化图像
        
        Returns:
            检测结果字典
        """
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return self.detect_from_base64(image_base64, conf_threshold, iou_threshold, return_image)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        response = requests.get(f"{self.base_url}/model/info", timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取模型信息失败: {response.status_code}")
    
    def format_result(self, result: Dict) -> str:
        """
        格式化检测结果为易读文本
        
        Args:
            result: 检测结果字典
        
        Returns:
            格式化的文本
        """
        lines = []
        lines.append("=" * 70)
        lines.append("🦷 牙齿枚举检测结果")
        lines.append("=" * 70)
        
        stats = result.get('statistics', {})
        total = stats.get('total_teeth', 0)
        
        lines.append(f"\n检测到 {total} 个牙齿\n")
        
        if total > 0:
            lines.append("详细信息:")
            lines.append("-" * 70)
            
            for idx, tooth in enumerate(result.get('detections', []), 1):
                cls_name = tooth['class_name']
                conf = tooth['confidence']
                bbox = tooth['bbox']
                
                lines.append(
                    f"{idx}. {cls_name} "
                    f"(置信度: {conf:.2f}, "
                    f"位置: [{bbox['x1']:.0f}, {bbox['y1']:.0f}, "
                    f"{bbox['x2']:.0f}, {bbox['y2']:.0f}])"
                )
            
            lines.append("-" * 70)
            lines.append("\n统计信息:")
            for tooth_type, count in stats.get('tooth_count_by_type', {}).items():
                lines.append(f"  {tooth_type}: {count} 个")
        else:
            lines.append("未检测到任何牙齿")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


# ============================================================
# Agent 函数 - 用于 AI Agent 调用
# ============================================================

def detect_tooth_enumeration(
    image_path: str,
    api_url: str = "http://localhost:8001",
    conf_threshold: float = 0.25
) -> str:
    """
    检测牙齿枚举（Agent 函数）
    
    Args:
        image_path: 图像文件路径
        api_url: API 服务地址
        conf_threshold: 置信度阈值
    
    Returns:
        格式化的检测结果文本
    """
    try:
        client = ToothEnumerationClient(base_url=api_url)
        result = client.detect_from_file(image_path, conf_threshold=conf_threshold)
        return client.format_result(result)
    except Exception as e:
        return f"❌ 检测失败: {str(e)}"


def get_tooth_enumeration_summary(
    image_path: str,
    api_url: str = "http://localhost:8001"
) -> Dict:
    """
    获取牙齿枚举统计摘要（Agent 函数）
    
    Args:
        image_path: 图像文件路径
        api_url: API 服务地址
    
    Returns:
        统计摘要字典
    """
    try:
        client = ToothEnumerationClient(base_url=api_url)
        result = client.detect_from_file(image_path)
        
        stats = result.get('statistics', {})
        
        return {
            "success": True,
            "total_teeth": stats.get('total_teeth', 0),
            "tooth_count_by_type": stats.get('tooth_count_by_type', {}),
            "num_detections": len(result.get('detections', []))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# 命令行接口
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="牙齿枚举检测客户端")
    parser.add_argument(
        "image",
        type=str,
        help="图像文件路径"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8001",
        help="API 服务地址"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存可视化结果"
    )
    
    args = parser.parse_args()
    
    # 创建客户端
    client = ToothEnumerationClient(base_url=args.url)
    
    # 检测
    print(f"\n🔍 检测图像: {args.image}")
    result = client.detect_from_file(
        args.image,
        conf_threshold=args.conf,
        return_image=args.save
    )
    
    # 显示结果
    print(client.format_result(result))
    
    # 保存可视化
    if args.save and 'visualized_image' in result:
        output_path = Path(args.image).stem + "_result.jpg"
        image_data = base64.b64decode(result['visualized_image'])
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"\n💾 可视化结果已保存: {output_path}")


if __name__ == "__main__":
    main()
