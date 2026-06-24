"""
MedSAM API Server
提供基于 MedSAM 的牙齿分割服务
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
from PIL import Image
import cv2
from skimage import transform
import torch.nn.functional as F

# 添加 MedSAM 路径
sys.path.insert(0, str(Path(__file__).parent / "MedSAM"))
from segment_anything import sam_model_registry

app = FastAPI(title="MedSAM API", description="Med SAM 分割服务", version="1.0.0")

# 全局模型变量
medsam_model = None
device = None


class SegmentationRequest(BaseModel):
    """分割请求"""
    bbox: List[float]  # [x1, y1, x2, y2]


class SegmentationResponse(BaseModel):
    """分割响应"""
    success: bool
    mask_contour: Optional[List[List[int]]] = None
    bbox: List[float]
    image_size: List[int]
    error: Optional[str] = None


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    """
    MedSAM 推理函数
    """
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def mask_to_contour(mask: np.ndarray) -> List[List[int]]:
    """
    将 mask 转换为轮廓点列表
    """
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # 获取最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 简化轮廓（减少点数）
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 转换为列表格式 [[x1, y1], [x2, y2], ...]
    contour_points = approx_contour.reshape(-1, 2).tolist()
    
    return contour_points


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global medsam_model, device
    
    print("🚀 Loading MedSAM model...")
    
    # 设置设备（使用 cuda:0，因为 CUDA_VISIBLE_DEVICES 会重新映射设备编号）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"📍 Using device: {device}")
    
    # 模型路径（env MEDSAM_MODEL_PATH 可覆盖）
    checkpoint_path = Path(os.environ.get(
        "MEDSAM_MODEL_PATH",
        str(Path(__file__).parent / "MedSAM" / "work_dir" / "MedSAM" / "medsam_vit_b.pth"),
    ))
    
    if not checkpoint_path.exists():
        print(f"⚠️  Warning: Model checkpoint not found at {checkpoint_path}")
        print("   Please download the model from: https://github.com/bowang-lab/MedSAM")
        medsam_model = None
        return
    
    # 加载模型
    medsam_model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    
    print("✓ MedSAM model loaded successfully!")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "MedSAM API",
        "version": "1.0.0",
        "status": "running" if medsam_model is not None else "model_not_loaded"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if medsam_model is not None else "model_not_loaded",
        "device": str(device) if device else "unknown"
    }


@app.post("/segment", response_model=SegmentationResponse)
async def segment_tooth(
    file: UploadFile = File(...),
    bbox: str = Form(...)  # 格式: "x1,y1,x2,y2"
):
    """
    对牙齿进行分割
    
    Args:
        file: 上传的图像文件
        bbox: 牙齿的检测框，格式为 "x1,y1,x2,y2"
    
    Returns:
        分割结果，包含 mask 轮廓
    """
    if medsam_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 解析 bbox
        bbox_values = [float(x.strip()) for x in bbox.split(',')]
        if len(bbox_values) != 4:
            raise ValueError("bbox must have 4 values: x1,y1,x2,y2")
        
        # 读取图像
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 numpy 数组
        img_np = np.array(image)
        
        # 确保是 RGB 格式
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif img_np.shape[2] == 4:  # RGBA
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np
        
        H, W, _ = img_3c.shape
        
        # 预处理图像
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )
        
        # 转换 bbox 到 1024x1024 尺度
        box_np = np.array([bbox_values])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        
        # 获取图像嵌入
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
        
        # 推理
        medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        
        # 转换 mask 为轮廓
        contour = mask_to_contour(medsam_seg)
        
        return SegmentationResponse(
            success=True,
            mask_contour=contour,
            bbox=bbox_values,
            image_size=[W, H]
        )
        
    except Exception as e:
        return SegmentationResponse(
            success=False,
            mask_contour=None,
            bbox=bbox_values if 'bbox_values' in locals() else [0, 0, 0, 0],
            image_size=[0, 0],
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8008))
    host = os.environ.get("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")
