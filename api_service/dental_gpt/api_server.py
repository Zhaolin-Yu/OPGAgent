"""
DentalGPT REST API Server
提供局域网访问的牙科图像分析 API 服务
"""

import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import uvicorn
from PIL import Image
import io
import base64
from typing import Optional
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="DentalGPT API",
    description="牙科图像分析 REST API 服务",
    version="1.0.0"
)

# 配置 CORS，允许局域网访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体的 IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型和处理器
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型配置（env DENTAL_GPT_MODEL_PATH 可覆盖；默认 HF repo id 以便首次自动下载）
LOCAL_MODEL_PATH = os.environ.get("DENTAL_GPT_MODEL_PATH", "Eric3200/DentalGPT-7B-1026")
PROCESSOR_MODEL = os.environ.get("DENTAL_GPT_PROCESSOR", "Qwen/Qwen2.5-VL-7B-Instruct")


def load_model():
    """加载模型和处理器"""
    global model, processor
    
    try:
        logger.info("正在加载 processor...")
        processor = AutoProcessor.from_pretrained(
            PROCESSOR_MODEL,
            trust_remote_code=True
        )
        logger.info("Processor 加载成功")
        
        logger.info("正在加载 DentalGPT 模型...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        logger.info("模型加载成功")
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    logger.info("启动 DentalGPT API 服务...")
    load_model()
    logger.info("服务启动完成")


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "service": "DentalGPT API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "analyze_base64": "/analyze/base64 (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    model_loaded = model is not None and processor is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(default="请分析图像中有问题的牙齿,并给出有问题的牙齿所在位置,牙齿类型"),
    max_new_tokens: int = Form(default=4096),
    temperature: float = Form(default=0.3),
    with_report: bool = Form(default=False),
):
    """
    分析上传的牙科图像
    
    参数:
        image: 上传的图像文件
        question: 要问的问题（可选）
        max_new_tokens: 最大生成token数（默认4096）
    
    返回:
        JSON 响应，包含分析结果
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 读取图像
        logger.info(f"接收到图像分析请求: {image.filename}")
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        def _generate_text(prompt_text: str) -> str:
            """单次生成：输入同一张图 + 文本提示，返回模型输出（用于 report 或 answer）。"""
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # 处理输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # 推理
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=1.0,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            return output_text[0]

        logger.info("开始推理...")
        if with_report:
            # 核心说明：为满足“同一次 API 调用里先生成 report，再回答问题”的约束，
            # 这里在服务端同一次请求内串行做两次生成，并拼接成固定结构返回（REPORT + FINAL_ANSWER）。
            report_prompt = "\n".join(
                [
                    "你是一名牙科放射科专家。",
                    "任务：请基于该 OPG 生成 1-4 行客观、简短的放射学报告。",
                    "规则：使用 FDI 牙号；避免臆测；仅输出报告正文。",
                ]
            )
            report_text = _generate_text(report_prompt)
            answer_text = _generate_text(question)
            output_text = [f"REPORT: {(report_text or '').strip()}\nFINAL_ANSWER: {(answer_text or '').strip()}"]
        else:
            output_text = [_generate_text(question)]
        
        logger.info("推理完成")
        
        return JSONResponse(content={
            "success": True,
            "question": question,
            "answer": output_text[0],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/analyze/base64")
async def analyze_image_base64(
    image_base64: str = Form(...),
    question: str = Form(default="请分析图像中有问题的牙齿,并给出有问题的牙齿所在位置,牙齿类型"),
    max_new_tokens: int = Form(default=4096)
):
    """
    使用 Base64 编码的图像进行分析
    
    参数:
        image_base64: Base64 编码的图像字符串
        question: 要问的问题（可选）
        max_new_tokens: 最大生成token数（默认4096）
    
    返回:
        JSON 响应，包含分析结果
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 解码 Base64 图像
        logger.info("接收到 Base64 图像分析请求")
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_data))
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {
                        "type": "text",
                        "text": question
                    },
                ],
            }
        ]
        
        # 处理输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # 推理
        logger.info("开始推理...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        logger.info("推理完成")
        
        return JSONResponse(content={
            "success": True,
            "question": question,
            "answer": output_text[0],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/load_model")
async def load_model_endpoint():
    """手动加载模型"""
    global model, processor
    
    if model is not None and processor is not None:
        return {"status": "info", "message": "模型已加载"}
    
    try:
        logger.info("手动加载模型...")
        load_model()
        return {"status": "success", "message": "模型加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.post("/unload_model")
async def unload_model_endpoint():
    """卸载模型以释放 GPU 内存"""
    global model, processor
    
    if model is None and processor is None:
        return {"status": "info", "message": "模型未加载"}
    
    try:
        logger.info("卸载模型...")
        
        # 删除模型和处理器
        del model
        del processor
        model = None
        processor = None
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🗑️ dentalGPT 模型已卸载")
        return {"status": "success", "message": "模型已卸载，GPU 内存已释放"}
    except Exception as e:
        logger.error(f"卸载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")


if __name__ == "__main__":
    # 启动服务器
    # host="0.0.0.0" 允许局域网访问
    # port 从环境变量读取，默认 8566
    import os
    port = int(os.getenv("API_PORT", "8566"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
