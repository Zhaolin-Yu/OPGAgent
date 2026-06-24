#!/usr/bin/env python3
"""
TVEM 统一 API 服务
支持多种检测模型：4quadrants, 11diseases, bone_loss, mandibular_maxillary
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import sys
import os
from pathlib import Path
import tempfile
import json
import argparse
import base64
import cv2
import numpy as np

# 添加 TVEM 路径
TVEM_PATH = Path(__file__).parent
sys.path.insert(0, str(TVEM_PATH))
sys.path.insert(0, str(TVEM_PATH / "MaskDINO"))

# 延迟导入（在模型加载时）
demo_instances = {}
category_maps = {}

app = FastAPI(
    title="TVEM Detection API",
    description="TVEM 多模型检测服务 (MaskDINO)",
    version="1.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型配置（目录均可通过 env 覆盖：TVEM_CONFIG_DIR / TVEM_WEIGHTS_DIR / TVEM_CATEGORY_DIR）
_CFG_DIR = os.environ.get("TVEM_CONFIG_DIR", "configs")
_WTS_DIR = os.environ.get("TVEM_WEIGHTS_DIR", "weights")
_CAT_DIR = os.environ.get("TVEM_CATEGORY_DIR", "categories")
MODEL_CONFIGS = {
    "4quadrants": {
        "config": f"{_CFG_DIR}/4quadrants.yaml",
        "weights": f"{_WTS_DIR}/4quadrants.pth",
        "categories": f"{_CAT_DIR}/4quadrants_category.json",
        "description": "象限检测 (Q1-Q4)"
    },
    "11diseases": {
        "config": f"{_CFG_DIR}/11diseases.yaml",
        "weights": f"{_WTS_DIR}/11diseases.pth",
        "categories": f"{_CAT_DIR}/11diseases_category.json",
        "description": "11种牙科疾病检测"
    },
    "bone_loss": {
        "config": f"{_CFG_DIR}/bone_loss.yaml",
        "weights": f"{_WTS_DIR}/bone_loss.pth",
        "categories": f"{_CAT_DIR}/bone_loss_category.json",
        "description": "骨质流失检测"
    },
    "mandibular_maxillary": {
        "config": f"{_CFG_DIR}/mandibular_maxillary.yaml",
        "weights": f"{_WTS_DIR}/mandibular_maxillary.pth",
        "categories": f"{_CAT_DIR}/mandibular_maxillary_category.json",
        "description": "下颌管/上颌窦检测"
    }
}

# 服务状态
service_status = {
    "models_loaded": {},
    "device": "cuda:0"  # 使用 CUDA_VISIBLE_DEVICES=1 时，GPU 1 映射为 cuda:0
}


def load_category_mapping(category_file: str) -> Dict[int, str]:
    """加载类别映射"""
    if not os.path.exists(category_file):
        return {}
    with open(category_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式：
    # 格式1: {"categories": [{"id": 1, "name": "xxx"}, ...]}
    # 格式2: {"0": "xxx", "1": "yyy", ...}
    if "categories" in data:
        return {item["id"]: item["name"] for item in data["categories"]}
    else:
        return {int(k): v for k, v in data.items()}


def load_model(model_name: str, device: str = "cuda:0"):
    """加载指定模型"""
    global demo_instances, category_maps
    
    if model_name in demo_instances:
        return True
    
    if model_name not in MODEL_CONFIGS:
        return False
    
    try:
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.utils.logger import setup_logger
        from demo.predictor import VisualizationDemo
        from maskdino import add_maskdino_config
        
        config = MODEL_CONFIGS[model_name]
        config_path = str(TVEM_PATH / config["config"])
        weights_path = str(TVEM_PATH / config["weights"])
        category_path = str(TVEM_PATH / config["categories"])
        
        # 检查文件
        if not os.path.exists(weights_path):
            print(f"❌ 权重文件不存在: {weights_path}")
            return False
        
        # 设置配置
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        
        # 创建预测器
        print(f"🔧 加载模型: {model_name}")
        demo_instances[model_name] = VisualizationDemo(cfg)
        
        # 加载类别映射
        category_maps[model_name] = load_category_mapping(category_path)
        
        service_status["models_loaded"][model_name] = True
        print(f"✅ 模型 {model_name} 加载成功")
        return True
        
    except Exception as e:
        print(f"❌ 加载模型 {model_name} 失败: {e}")
        service_status["models_loaded"][model_name] = False
        return False


def create_visualizer(img, metadata):
    """创建只显示轮廓的可视化器"""
    from detectron2.utils.visualizer import Visualizer, ColorMode
    
    class OutlineOnlyVisualizer(Visualizer):
        """自定义可视化器，只显示轮廓不显示填充"""
        def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
            import matplotlib as mpl
            import matplotlib.colors as mplc
            
            if edge_color is None:
                edge_color = color
            edge_color = mplc.to_rgb(edge_color) + (1,)
            
            polygon = mpl.patches.Polygon(
                segment,
                fill=False,
                edgecolor=edge_color,
                linewidth=max(self._default_font_size // 15 * self.output.scale, 2),
            )
            self.output.ax.add_patch(polygon)
            return self.output
    
    return OutlineOnlyVisualizer(
        img[:, :, ::-1],  # BGR to RGB
        metadata=metadata,
        instance_mode=ColorMode.IMAGE
    )


def run_detection(model_name: str, image_path: str, confidence: float = 0.3, 
                  return_visualization: bool = False) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    运行检测
    
    Args:
        model_name: 模型名称
        image_path: 图像路径
        confidence: 置信度阈值
        return_visualization: 是否返回可视化图像路径
    
    Returns:
        (检测结果字典, 可视化图像路径或None)
    """
    from detectron2.data.detection_utils import read_image
    from detectron2.data import MetadataCatalog
    
    if model_name not in demo_instances:
        raise ValueError(f"模型 {model_name} 未加载")
    
    demo = demo_instances[model_name]
    category_map = category_maps.get(model_name, {})
    
    # 读取图像
    img = read_image(image_path, format="BGR")
    
    # 运行推理
    predictions, _ = demo.run_on_image(img)
    instances = predictions["instances"]
    
    # 过滤低置信度
    if instances.has("scores"):
        keep = instances.scores >= confidence
        instances = instances[keep]
    
    # 解析结果
    detections = []
    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor.cpu().numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.cpu().numpy() if instances.has("scores") else []
        classes = instances.pred_classes.cpu().numpy() if instances.has("pred_classes") else []
        masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else []
        
        for i in range(len(instances)):
            box = boxes[i].tolist() if len(boxes) > i else []
            score = float(scores[i]) if len(scores) > i else 0.0
            class_id = int(classes[i]) if len(classes) > i else -1
            class_name = category_map.get(class_id, f"class_{class_id}")
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(score, 4),
                "bbox": [round(x, 2) for x in box]  # 改名为 bbox 更清晰
            }
            
            # 提取 mask 轮廓（如果有）
            if len(masks) > i and masks[i] is not None:
                try:
                    import cv2
                    mask = masks[i].astype(np.uint8)
                    # 查找轮廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # 取最大的轮廓
                        largest_contour = max(contours, key=cv2.contourArea)
                        # 简化轮廓（减少点数）
                        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                        # 转换为列表格式 [[x, y], [x, y], ...]
                        detection["mask_contour"] = approx_contour.reshape(-1, 2).tolist()
                except Exception as e:
                    print(f"  ⚠️ Mask 轮廓提取失败: {e}")
                    detection["mask_contour"] = None
            
            detections.append(detection)
    
    result = {
        "model": model_name,
        "total_detections": len(detections),
        "confidence_threshold": confidence,
        "detections": detections
    }
    
    # 生成可视化图像
    vis_path = None
    if return_visualization and len(instances) > 0:
        try:
            # 获取 metadata
            metadata = MetadataCatalog.get("__unused")
            if category_map:
                max_id = max(category_map.keys()) if category_map else 0
                thing_classes = [""] * (max_id + 1)
                for cat_id, cat_name in category_map.items():
                    if cat_id < len(thing_classes):
                        thing_classes[cat_id] = cat_name
                metadata.thing_classes = thing_classes
            
            # 创建可视化器并绘制
            visualizer = create_visualizer(img, metadata)
            vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))
            
            # 保存可视化图像
            vis_path = tempfile.NamedTemporaryFile(delete=False, suffix="_vis.jpg").name
            vis_output.save(vis_path)
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
            vis_path = None
    
    return result, vis_path


# === API 端点 ===

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "tvem",
        "models_loaded": service_status["models_loaded"],
        "device": service_status["device"]
    }


@app.get("/models")
async def list_models():
    """列出可用模型"""
    models = []
    for name, config in MODEL_CONFIGS.items():
        models.append({
            "name": name,
            "description": config["description"],
            "loaded": service_status["models_loaded"].get(name, False)
        })
    return {"models": models}


@app.post("/load/{model_name}")
async def load_model_endpoint(model_name: str):
    """加载指定模型"""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
    
    success = load_model(model_name, service_status["device"])
    if success:
        return {"status": "success", "message": f"模型 {model_name} 加载成功"}
    else:
        raise HTTPException(status_code=500, detail=f"模型 {model_name} 加载失败")


@app.post("/unload/{model_name}")
async def unload_model_endpoint(model_name: str):
    """卸载指定模型以释放 GPU 内存"""
    global demo_instances, category_maps
    
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
    
    if model_name not in demo_instances:
        return {"status": "info", "message": f"模型 {model_name} 未加载"}
    
    try:
        import torch
        del demo_instances[model_name]
        if model_name in category_maps:
            del category_maps[model_name]
        service_status["models_loaded"][model_name] = False
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"🗑️ 模型 {model_name} 已卸载")
        return {"status": "success", "message": f"模型 {model_name} 已卸载"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")


@app.post("/unload_all")
async def unload_all_models():
    """卸载所有模型"""
    global demo_instances, category_maps
    import torch
    
    unloaded = list(demo_instances.keys())
    demo_instances.clear()
    category_maps.clear()
    for name in service_status["models_loaded"]:
        service_status["models_loaded"][name] = False
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"🗑️ 已卸载所有模型: {unloaded}")
    return {"status": "success", "message": f"已卸载模型: {unloaded}"}


@app.post("/detect/{model_name}")
async def detect(
    model_name: str,
    file: UploadFile = File(...),
    confidence: float = Form(0.3),
    return_vis: bool = Form(False)
):
    """
    使用指定模型进行检测
    
    Args:
        model_name: 模型名称 (4quadrants, 11diseases, bone_loss, mandibular_maxillary)
        file: 上传的图像文件
        confidence: 置信度阈值
        return_vis: 是否返回可视化图像（base64编码）
    """
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
    
    # 自动加载模型
    if model_name not in demo_instances:
        success = load_model(model_name, service_status["device"])
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {model_name} 加载失败")
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 运行检测
        result, vis_path = run_detection(model_name, tmp_path, confidence, return_vis)
        
        # 如果有可视化图像，添加到结果中
        if vis_path and os.path.exists(vis_path):
            with open(vis_path, "rb") as f:
                vis_base64 = base64.b64encode(f.read()).decode("utf-8")
            result["visualization_base64"] = vis_base64
            os.unlink(vis_path)
        
        # 清理输入临时文件
        os.unlink(tmp_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_with_vis/{model_name}")
async def detect_with_visualization(
    model_name: str,
    file: UploadFile = File(...),
    confidence: float = Form(0.3)
):
    """
    使用指定模型进行检测并返回可视化图像文件
    
    Args:
        model_name: 模型名称
        file: 上传的图像文件
        confidence: 置信度阈值
    
    Returns:
        可视化图像文件（直接下载）
    """
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
    
    # 自动加载模型
    if model_name not in demo_instances:
        success = load_model(model_name, service_status["device"])
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {model_name} 加载失败")
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 运行检测并生成可视化
        result, vis_path = run_detection(model_name, tmp_path, confidence, return_visualization=True)
        
        # 清理输入临时文件
        os.unlink(tmp_path)
        
        if vis_path and os.path.exists(vis_path):
            return FileResponse(
                vis_path,
                media_type="image/jpeg",
                filename=f"{model_name}_visualization.jpg"
            )
        else:
            raise HTTPException(status_code=500, detail="可视化生成失败")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === 便捷端点（兼容旧 API）===

@app.post("/detect/quadrants")
async def detect_quadrants(
    file: UploadFile = File(...),
    confidence: float = Form(0.3)
):
    """检测象限（兼容端点）"""
    return await detect("4quadrants", file, confidence)


@app.post("/detect/diseases")
async def detect_diseases(
    file: UploadFile = File(...),
    confidence: float = Form(0.3)
):
    """检测疾病（兼容端点）"""
    return await detect("11diseases", file, confidence)


@app.post("/detect/bone_loss")
async def detect_bone_loss(
    file: UploadFile = File(...),
    confidence: float = Form(0.3)
):
    """检测骨质流失（兼容端点）"""
    return await detect("bone_loss", file, confidence)


@app.post("/detect/mandibular")
async def detect_mandibular(
    file: UploadFile = File(...),
    confidence: float = Form(0.3)
):
    """检测下颌管/上颌窦（兼容端点）"""
    return await detect("mandibular_maxillary", file, confidence)


def main():
    parser = argparse.ArgumentParser(description="TVEM API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务地址")
    parser.add_argument("--port", type=int, default=8003, help="服务端口")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--preload", nargs="*", default=[], help="预加载的模型")
    
    args = parser.parse_args()
    
    service_status["device"] = args.device
    
    # 预加载模型
    for model_name in args.preload:
        if model_name in MODEL_CONFIGS:
            load_model(model_name, args.device)
    
    print(f"\n🚀 启动 TVEM API 服务")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   文档: http://{args.host}:{args.port}/docs")
    print(f"   设备: {args.device}")
    print(f"   可用模型: {list(MODEL_CONFIGS.keys())}\n")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
