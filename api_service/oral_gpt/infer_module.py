"""
OralGPT-omni 推理模块
可被 api_server.py 导入使用
"""
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import Optional

# OralGPT-Omni 模型路径（env ORAL_GPT_MODEL_PATH 可覆盖；默认 HF repo id 以便首次自动下载）
MODEL_PATH = os.environ.get("ORAL_GPT_MODEL_PATH", "OralGPT/OralGPT-Omni-7B-Instruct")

# 全局变量
model = None
processor = None
_initialized = False


def initialize_model(model_path: Optional[str] = None):
    """
    初始化模型和处理器
    
    Args:
        model_path: 模型路径，默认使用 MODEL_PATH
    """
    global model, processor, _initialized
    
    if _initialized:
        return
    
    path = model_path or MODEL_PATH
    
    # 加载 processor
    print(f"Loading processor from {path}...")
    processor = AutoProcessor.from_pretrained(
        path,
        trust_remote_code=True
    )
    print("Processor loaded successfully.")
    
    # 加载模型
    print(f"Loading model from {path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded successfully.")
    
    _initialized = True


def run_inference(
    image_path: str,
    prompt: str = "Analyze this image",
    max_new_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """
    运行推理
    
    Args:
        image_path: 图像文件路径
        prompt: 提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度（Temperature 温度），越小越保守；为满足评测约束默认 0.3
        
    Returns:
        生成的文本
    """
    global model, processor, _initialized
    
    # 确保模型已初始化
    if not _initialized:
        initialize_model()
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    
    # 手动构建 Qwen2-VL 格式的文本提示
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 处理图像输入
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 准备输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # 推理生成
    print(f"Running inference on {image_path}...")
    # 说明：为了满足“统一温度=0.3”的约束，这里启用采样（do_sample=True）。
    # 如果你希望完全确定性输出，可把 do_sample=False（此时 temperature 将被忽略）。
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=float(temperature),
        top_p=1.0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def unload_model():
    """
    卸载模型以释放 GPU 内存
    """
    global model, processor, _initialized
    
    if not _initialized:
        return
    
    print("🗑️ 卸载 OralGPT-omni 模型...")
    
    # 删除模型和处理器
    del model
    del processor
    
    model = None
    processor = None
    _initialized = False
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✅ 模型已卸载")


def is_loaded() -> bool:
    """检查模型是否已加载"""
    return _initialized


# 兼容原有脚本的直接执行
if __name__ == "__main__":
    # 初始化模型
    initialize_model()
    
    # 测试推理
    test_image = "/mnt/hdd/sdc/zlyu/dentalGPT/images/4-No004.jpg"
    result = run_inference(test_image, "Analyze this image")
    
    print("\n===== Output =====")
    print(result)
