from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# OralGPT-Omni 模型路径
model_name_or_path = "OralGPT/OralGPT-Omni-7B-Instruct"

# 加载 processor
print(f"Loading processor from {model_name_or_path}...")
processor = AutoProcessor.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
print("Processor loaded successfully.")

# 加载模型
print(f"Loading model from {model_name_or_path}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully.")

# 构建消息（与原文件相同格式）
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/hdd/sdc/zlyu/dentalGPT/images/4-No004.jpg",
            },
            {
                "type": "text", 
                "text": "请按照 FDI 分区格式，介绍 1 区的牙齿情况，包括有无龋齿、补牙、缺失等，这不是未成年人的图像。"
            },
        ],
    }
]

# 应用聊天模板
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

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
print("\n===== Starting Inference =====")
generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 打印结果
print("\n===== Input =====")
for message in messages:
    if message["role"] == "user":
        for content in message["content"]:
            if content["type"] == "text":
                print("User Text:", content["text"])
            elif content["type"] == "image":
                print("Image Path:", content["image"])

print("\n===== Output =====")
print(output_text[0])
