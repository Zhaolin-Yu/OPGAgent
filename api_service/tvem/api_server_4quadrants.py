#!/usr/bin/env python3
"""
TVEM 4Quadrants API 服务
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import tempfile
import json

# 添加 TVEM 路径
TVEM_PATH = Path(__file__).parent
sys.path.insert(0, str(TVEM_PATH))

from infer_with_json import run_inference

app = FastAPI(
    title="TVEM 4Quadrants Detection API",
    description="TVEM 象限检测服务",
    version="1.0.0"
)

# 配置
CONFIG_PATH = TVEM_PATH / "configs" / "4quadrants.yaml"
WEIGHTS_PATH = TVEM_PATH / "weights" / "4quadrants_model.pth"  # 根据实际情况修改


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "tvem_4quadrants"}


@app.post("/detect")
async def detect_quadrants(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    检测象限
    
    Args:
        file: 上传的图像文件
        confidence_threshold: 置信度阈值
        
    Returns:
        {
            "detections": [
                {
                    "class": "Q1",
                    "box": [x1, y1, x2, y2],
                    "confidence": 0.95,
                    "mask": [...]  # 可选
                },
                ...
            ]
        }
    """
    try:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 运行推理
        result = run_inference(
            image_path=tmp_path,
            config_path=str(CONFIG_PATH),
            weights_path=str(WEIGHTS_PATH),
            confidence_threshold=confidence_threshold
        )
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8001))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
