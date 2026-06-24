#!/usr/bin/env python3
"""
MaskDINO 推理脚本 - 自动保存 JSON 结果
"""

import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加 MaskDINO 路径
MASKDINO_PATH = os.path.join(os.path.dirname(__file__), "MaskDINO")
sys.path.insert(0, MASKDINO_PATH)

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from demo.predictor import VisualizationDemo
from maskdino import add_maskdino_config


def filter_predictions_by_confidence(predictions, confidence_threshold):
    """过滤低置信度的预测结果"""
    instances = predictions["instances"]
    
    if not instances.has("scores"):
        return predictions
    
    # 获取高置信度的索引
    scores = instances.scores
    keep = scores >= confidence_threshold
    
    # 过滤实例
    filtered_instances = instances[keep]
    
    # 创建新的预测字典
    filtered_predictions = {"instances": filtered_instances}
    
    return filtered_predictions


class OutlineOnlyVisualizer(Visualizer):
    """自定义可视化器，只显示轮廓不显示填充"""
    
    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        重写draw_polygon方法，设置fill=False只显示轮廓
        
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon (用于edge_color如果未提供)
            edge_color: color of the polygon edges
            alpha: 不使用（保留参数兼容性）
        """
        import matplotlib as mpl
        import matplotlib.colors as mplc
        
        if edge_color is None:
            edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)  # 完全不透明的边框
        
        polygon = mpl.patches.Polygon(
            segment,
            fill=False,  # 关键：不填充
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 2),  # 稍微加粗边框使其更明显
        )
        self.output.ax.add_patch(polygon)
        return self.output


def visualize_predictions(img, predictions, metadata, confidence_threshold, category_map=None):
    """自定义可视化，只显示高置信度检测"""
    # 先过滤预测
    filtered_predictions = filter_predictions_by_confidence(predictions, confidence_threshold)
    
    # 如果有类别映射，更新metadata的thing_classes
    if category_map:
        max_id = max(category_map.keys()) if category_map else 0
        thing_classes = [""] * (max_id + 1)
        for cat_id, cat_name in category_map.items():
            thing_classes[cat_id - 1] = cat_name
        metadata.thing_classes = thing_classes
    
    # 使用自定义可视化器（只显示轮廓）
    visualizer = OutlineOnlyVisualizer(
        img[:, :, ::-1],  # BGR to RGB
        metadata=metadata,
        instance_mode=ColorMode.IMAGE
    )
    
    # 绘制过滤后的实例
    vis_output = visualizer.draw_instance_predictions(filtered_predictions["instances"].to("cpu"))
    
    return vis_output


def setup_cfg(config_file, confidence_threshold=0.3, checkpoint_path=None):
    """设置配置"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    
    # 设置置信度阈值 - MaskDINO 特定配置
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    
    # 设置模型权重和设备
    if checkpoint_path:
        cfg.MODEL.WEIGHTS = checkpoint_path
    
    cfg.freeze()
    return cfg


def load_category_mapping(category_file):
    """加载类别映射文件"""
    if not os.path.exists(category_file):
        print(f"⚠️  警告: 类别文件不存在: {category_file}")
        return {}
    
    with open(category_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建 id -> name 的映射
    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    return category_map


def save_detection_results(predictions, category_map, output_path, image_name, confidence_threshold):
    """保存检测结果为 JSON"""
    instances = predictions["instances"].to("cpu")
    
    # 提取检测结果
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
    scores = instances.scores.numpy() if instances.has("scores") else None
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
    
    # 过滤低置信度检测
    valid_indices = []
    if scores is not None:
        for i in range(len(instances)):
            if scores[i] >= confidence_threshold:
                valid_indices.append(i)
    else:
        valid_indices = list(range(len(instances)))
    
    results = {
        "image": image_name,
        "timestamp": datetime.now().isoformat(),
        "confidence_threshold": confidence_threshold,
        "total_detections": len(valid_indices),
        "detections": []
    }
    
    for idx, i in enumerate(valid_indices):
        detection = {
            "id": idx + 1,
            "category_id": int(classes[i]) if classes is not None else -1,
            "category_name": category_map.get(int(classes[i]) + 1, f"class_{int(classes[i])}") if classes is not None else "unknown",
            "confidence": float(scores[i]) if scores is not None else 0.0,
        }
        
        # 添加边界框
        if boxes is not None:
            box = boxes[i]
            detection["bbox"] = {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1])
            }
        
        # 添加掩码统计信息（不保存完整掩码，文件太大）
        if masks is not None:
            mask = masks[i]
            detection["mask_area"] = int(mask.sum())
            detection["mask_shape"] = list(mask.shape)
        
        results["detections"].append(detection)
    
    # 按置信度排序
    results["detections"].sort(key=lambda x: x["confidence"], reverse=True)
    
    # 统计每个类别的数量
    category_counts = {}
    for det in results["detections"]:
        cat_name = det["category_name"]
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    results["category_counts"] = category_counts
    
    # 保存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 结果已保存: {output_path}")
    print(f"   - 总检测数: {results['total_detections']}")
    print(f"   - 类别分布: {category_counts}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MaskDINO 推理 - 自动保存 JSON 结果")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--output-dir", default="./inference_results", help="输出目录")
    parser.add_argument("--category-file", help="类别映射 JSON 文件路径")
    parser.add_argument("--confidence", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    setup_logger(name="maskdino")
    
    # 加载类别映射
    category_map = {}
    if args.category_file and os.path.exists(args.category_file):
        category_map = load_category_mapping(args.category_file)
        print(f"✅ 加载类别映射: {len(category_map)} 个类别")
    
    # 设置配置
    print(f"🔧 加载配置: {args.config}")
    print(f"⚙️  置信度阈值: {args.confidence}")
    cfg = setup_cfg(args.config, args.confidence, args.checkpoint)
    cfg.defrost()
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    
    # 创建预测器
    print(f"🦷 加载模型: {args.checkpoint}")
    demo = VisualizationDemo(cfg)
    
    # 读取图像
    print(f"📸 读取图像: {args.image}")
    img = read_image(args.image, format="BGR")
    
    # 运行推理
    print(f"🚀 开始推理 (置信度阈值: {args.confidence})...")
    predictions, _ = demo.run_on_image(img)  # 不使用默认可视化
    
    # 使用自定义可视化（应用置信度过滤）
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    visualized_output = visualize_predictions(
        img, 
        predictions, 
        metadata, 
        args.confidence,
        category_map=category_map
    )
    
    # 保存可视化结果
    image_name = Path(args.image).stem
    vis_output_path = output_dir / f"{image_name}_visualization.jpg"
    visualized_output.save(str(vis_output_path))
    print(f"✅ 可视化结果已保存: {vis_output_path}")
    
    # 保存 JSON 结果
    json_output_path = output_dir / f"{image_name}_results.json"
    results = save_detection_results(
        predictions, 
        category_map, 
        str(json_output_path),
        os.path.basename(args.image),
        args.confidence
    )
    
    print(f"\n🎉 推理完成！")
    print(f"   可视化: {vis_output_path}")
    print(f"   JSON:   {json_output_path}")


if __name__ == "__main__":
    main()
