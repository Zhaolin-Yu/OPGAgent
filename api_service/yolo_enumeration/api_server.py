"""
牙齿枚举检测 API 服务
FastAPI 服务，提供 RESTful API 接口
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import cv2
import io
import base64
from typing import List, Dict, Optional
import argparse
from pathlib import Path



# 全局模型实例
model = None

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

# 可视化颜色
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
]


def load_model(weights_path: str):
    """加载 YOLO 模型"""
    global model
    print(f"🔧 加载模型: {weights_path}")
    model = YOLO(weights_path)
    print("✅ 模型加载成功!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理器"""
    try:
        # 默认加载模型
        model_path = os.environ.get("YOLO_ENUM_MODEL_PATH", "model/best.pt")
        if Path(model_path).exists():
            load_model(model_path)
        else:
            print(f"⚠️ 模型文件未找到: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
    
    yield
    
    # 服务关闭时清理
    global model
    model = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="牙齿枚举检测 API",
    description="基于 YOLOv9c 的牙齿枚举检测服务，识别8种牙齿类型",
    version="1.0.0",
    lifespan=lifespan
)



def process_detections(results, image_np: np.ndarray, return_image: bool = False):
    """处理检测结果"""
    detections = results[0]
    boxes = detections.boxes
    
    # 解析检测结果
    tooth_list = []
    tooth_count = {i: 0 for i in range(8)}
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        tooth_count[cls] += 1
        
        tooth_list.append({
            "class_id": cls,
            "class_name": TOOTH_LABELS[cls],
            "confidence": round(conf, 4),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1)
            }
        })
    
    # 统计信息
    statistics = {
        "total_teeth": len(boxes),
        "tooth_count_by_type": {
            TOOTH_LABELS[cls]: count 
            for cls, count in tooth_count.items() if count > 0
        }
    }
    
    result = {
        "detections": tooth_list,
        "statistics": statistics
    }
    
    # 可视化
    if return_image and len(boxes) > 0:
        vis_image = image_np.copy()
        
        for tooth in tooth_list:
            cls = tooth["class_id"]
            bbox = tooth["bbox"]
            conf = tooth["confidence"]
            color = COLORS[cls]
            
            # 绘制边界框
            cv2.rectangle(
                vis_image,
                (int(bbox["x1"]), int(bbox["y1"])),
                (int(bbox["x2"]), int(bbox["y2"])),
                color, 2
            )
            
            # 绘制标签
            label_text = f"{TOOTH_LABELS[cls]} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                vis_image,
                (int(bbox["x1"]), int(bbox["y1"]) - text_height - baseline - 5),
                (int(bbox["x1"]) + text_width, int(bbox["y1"])),
                color, -1
            )
            
            cv2.putText(
                vis_image,
                label_text,
                (int(bbox["x1"]), int(bbox["y1"]) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        # 转换为 JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        result["visualized_image"] = base64.b64encode(buffer).decode('utf-8')
    
    return result


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "牙齿枚举检测 API",
        "version": "1.0.0",
        "model": "YOLOv9c",
        "classes": 8,
        "endpoints": {
            "POST /detect": "检测图像中的牙齿（multipart/form-data）",
            "POST /detect_base64": "检测 Base64 编码图像",
            "GET /health": "健康检查",
            "GET /model/info": "模型信息"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "service": "牙齿枚举检测"
    }


@app.get("/model/info")
async def model_info():
    """模型信息"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_type": "YOLOv9c",
        "task": "牙齿枚举检测",
        "num_classes": 8,
        "classes": TOOTH_LABELS,
        "input_size": 640
    }


# 模型加载/卸载相关变量
_model_path = None


@app.post("/load_model")
async def load_model_endpoint():
    """手动加载模型"""
    global model, _model_path
    
    if model is not None:
        return {"status": "info", "message": "模型已加载"}
    
    try:
        if _model_path is None:
            _model_path = os.environ.get("YOLO_ENUM_MODEL_PATH", "model/best.pt")  # 默认路径
        load_model(_model_path)
        return {"status": "success", "message": "模型加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.post("/unload_model")
async def unload_model_endpoint():
    """卸载模型以释放 GPU 内存"""
    global model
    
    if model is None:
        return {"status": "info", "message": "模型未加载"}
    
    try:
        import torch
        del model
        model = None
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🗑️ yolo_enumeration 模型已卸载")
        return {"status": "success", "message": "模型已卸载，GPU 内存已释放"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")


@app.post("/detect")
async def detect_teeth(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    return_image: bool = False
):
    """
    检测图像中的牙齿
    
    - **file**: 图像文件
    - **conf_threshold**: 置信度阈值 (0-1)
    - **iou_threshold**: NMS IoU 阈值 (0-1)
    - **return_image**: 是否返回可视化图像 (Base64)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 读取图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # 推理
        results = model.predict(
            source=image_np,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # 处理结果
        result = process_detections(results, image_np, return_image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/detect_base64")
async def detect_teeth_base64(
    image_base64: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    return_image: bool = False
):
    """
    检测 Base64 编码的图像
    
    - **image_base64**: Base64 编码的图像数据
    - **conf_threshold**: 置信度阈值 (0-1)
    - **iou_threshold**: NMS IoU 阈值 (0-1)
    - **return_image**: 是否返回可视化图像 (Base64)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 解码 Base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        # 推理
        results = model.predict(
            source=image_np,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # 处理结果
        result = process_detections(results, image_np, return_image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="牙齿枚举检测 API 服务")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.environ.get("YOLO_ENUM_MODEL_PATH", "model/best.pt"),
        help="模型权重路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="服务端口"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.weights)
    
    # 启动服务
    import uvicorn
    print(f"\n🚀 启动 API 服务:")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   文档: http://{args.host}:{args.port}/docs")
    print(f"   模型: {args.weights}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
