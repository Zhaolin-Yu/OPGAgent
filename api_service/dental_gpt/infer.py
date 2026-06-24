from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

local_model_path = "/home/zhaolin/Projects/dentalGPT/models--Eric3200--DentalGPT-7B-1026/snapshots/dce584f3ec6cf929cc37f6156cc41c8ebc8178a9"

# 使用 Qwen2.5-VL 的 processor (DentalGPT 是基于它微调的,processor 完全兼容)
print("Loading processor from Qwen/Qwen2.5-VL-7B-Instruct...")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)
print("Processor loaded successfully.")

# 加载本地的 DentalGPT 模型权重
print("Loading DentalGPT model from local path...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
print("Model loaded successfully.")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/hdd/sdb/zlyu/Documents/dentalagent_test/tools/output.jpg",
            },
            {
                "type": "text", 
                # "text": "请分析图像中有问题的牙齿,并给出有问题的牙齿所在位置,牙齿类型"
                "text": "仅描述图中框选出来的牙齿特征，并给出诊断建议"
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("===== Input =====")
for message in messages:
    if message["role"] == "user":
        for content in message["content"]:
            if content["type"] == "text":
                print("User Text:", content["text"])
print("===== Output =====")
print(output_text)