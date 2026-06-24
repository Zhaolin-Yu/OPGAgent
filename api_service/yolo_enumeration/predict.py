"""
牙齿枚举检测本地预测脚本
无需 API 服务，直接使用模型进行推理
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Optional, Dict, List


# 牙齿类别映射
TOOTH_LABELS = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "7",
    7: "8"
}

# 可视化颜色 (BGR格式)
COLORS = [
    (0, 0, 255),      # Red - Tooth 1
    (0, 255, 0),      # Green - Tooth 2
    (255, 0, 0),      # Blue - Tooth 3
    (0, 255, 255),    # Yellow - Tooth 4
    (255, 0, 255),    # Magenta - Tooth 5
    (255, 255, 0),    # Cyan - Tooth 6
    (128, 0, 128),    # Purple - Tooth 7
    (0, 165, 255),    # Orange - Tooth 8
]


class ToothEnumerationDetector:
    """牙齿枚举检测器"""
    
    def __init__(self, model_path: str = "model/best.pt"):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重路径
        """
        print(f"🔧 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
        print(f"✅ Model loaded successfully")
    
    def detect(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_result: bool = True,
        output_dir: str = "output"
    ) -> Dict:
        """
        检测牙齿枚举
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            save_result: 是否保存可视化结果
            output_dir: 输出目录
        
        Returns:
            检测结果字典
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n📷 Reading image: {image_path.name}")
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        print(f"🔍 Starting detection...")
        
        # 推理
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # 解析结果
        detections = []
        tooth_count = {i: 0 for i in range(8)}
        
        result = results[0]
        boxes = result.boxes
        
        print(f"✅ Detection completed! Found {len(boxes)} teeth (before filtering)\n")
        
        # 后处理：限制每个类别≤4个
        boxes = self._filter_max_per_class(boxes, max_per_class=4)
        print(f"📊 After filtering: {len(boxes)} teeth (max 4 per class)\n")
        
        if len(boxes) > 0:
            print("📊 Detection Results:")
            print("-" * 80)
            print(f"{'No.':<6}{'Class':<20}{'Confidence':<12}{'BBox (x1, y1, x2, y2)'}")
            print("-" * 80)
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                tooth_count[cls] += 1
                
                detection = {
                    "class_id": cls,
                    "class_name": TOOTH_LABELS[cls],
                    "confidence": conf,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                }
                detections.append(detection)
                
                print(f"{idx+1:<6}{TOOTH_LABELS[cls]:<20}{conf:<12.4f}"
                      f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
            
            print("-" * 80)
            
            # 统计信息
            print("\n📈 Tooth Type Statistics:")
            for cls_id, count in tooth_count.items():
                if count > 0:
                    print(f"   {TOOTH_LABELS[cls_id]}: {count}")
        else:
            print("⚠️  No teeth detected")
        
        # 构建结果
        result_dict = {
            "image_name": image_path.name,
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "detections": detections,
            "statistics": {
                "total_teeth": len(detections),
                "tooth_count_by_type": {
                    TOOTH_LABELS[cls_id]: count 
                    for cls_id, count in tooth_count.items() 
                    if count > 0
                }
            },
            "parameters": {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "model_path": str(self.model_path)
            }
        }
        
        # 保存结果
        if save_result:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 可视化
            vis_image = self._visualize(image, detections)
            vis_path = output_path / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            print(f"\n💾 Visualization saved: {vis_path}")
            
            # JSON结果
            json_path = output_path / f"{image_path.stem}_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            print(f"💾 JSON result saved: {json_path}")
        
        return result_dict
    
    def _visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
        
        Returns:
            可视化后的图像
        """
        vis_image = image.copy()
        
        for detection in detections:
            cls_id = detection['class_id']
            conf = detection['confidence']
            bbox = detection['bbox']
            
            x1 = int(bbox['x1'])
            y1 = int(bbox['y1'])
            x2 = int(bbox['x2'])
            y2 = int(bbox['y2'])
            
            color = COLORS[cls_id]
            label = f"{TOOTH_LABELS[cls_id]} {conf:.2f}"
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制文字
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return vis_image
    
    def _filter_max_per_class(self, boxes, max_per_class: int = 4):
        """
        限制每个类别的检测数量
        
        Args:
            boxes: YOLO检测框
            max_per_class: 每个类别的最大数量
        
        Returns:
            过滤后的检测框
        """
        if len(boxes) == 0:
            return boxes
        
        # 按类别分组
        class_boxes = {}
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in class_boxes:
                class_boxes[cls] = []
            class_boxes[cls].append(box)
        
        # 每个类别只保留置信度最高的max_per_class个
        filtered_boxes = []
        for cls, cls_box_list in class_boxes.items():
            # 按置信度排序
            cls_box_list.sort(key=lambda x: float(x.conf[0]), reverse=True)
            # 只保留前max_per_class个
            filtered_boxes.extend(cls_box_list[:max_per_class])
        
        # 按原始顺序(置信度)排序
        filtered_boxes.sort(key=lambda x: float(x.conf[0]), reverse=True)
        
        return filtered_boxes
    
    def batch_detect(
        self,
        image_dir: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        output_dir: str = "output"
    ) -> List[Dict]:
        """
        批量检测
        
        Args:
            image_dir: 图像目录
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            output_dir: 输出目录
        
        Returns:
            检测结果列表
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # 查找所有图像
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for pattern in image_patterns:
            image_files.extend(list(image_dir.glob(pattern)))
        
        if not image_files:
            print(f"⚠️  No image files found in {image_dir}")
            return []
        
        print(f"📂 Found {len(image_files)} images")
        print("=" * 80)
        
        results = []
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")
            print("-" * 80)
            
            try:
                result = self.detect(
                    str(image_file),
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    save_result=True,
                    output_dir=output_dir
                )
                results.append(result)
            except Exception as e:
                print(f"❌ Processing failed: {e}")
        
        print("\n" + "=" * 80)
        print(f"✅ Batch detection completed! Successfully processed {len(results)}/{len(image_files)} images")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="牙齿枚举检测本地预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图像检测
  python predict.py --image test.png
  
  # 批量检测
  python predict.py --dir test_images/
  
  # 自定义参数
  python predict.py --image test.png --conf 0.3 --output results/
  
  # 使用自定义模型
  python predict.py --image test.png --model custom_model.pt
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="model/best.pt",
        help="模型权重路径 (默认: model/best.pt)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="输入图像路径 (单张图像检测)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="输入图像目录 (批量检测)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值 (默认: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU阈值 (默认: 0.45)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="输出目录 (默认: output/)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存可视化结果"
    )
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.image and not args.dir:
        parser.error("Please specify --image or --dir")
    
    if args.image and args.dir:
        parser.error("--image and --dir cannot be used together")
    
    # 创建检测器
    print("=" * 80)
    print("🦷 Tooth Enumeration Detection - Local Prediction")
    print("=" * 80)
    
    detector = ToothEnumerationDetector(model_path=args.model)
    
    # 执行检测
    try:
        if args.image:
            detector.detect(
                image_path=args.image,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                save_result=not args.no_save,
                output_dir=args.output
            )
        else:
            detector.batch_detect(
                image_dir=args.dir,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                output_dir=args.output
            )
        
        print("\n" + "=" * 80)
        print("✨ Detection completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
