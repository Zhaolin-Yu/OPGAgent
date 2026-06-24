#!/bin/bash

# 批量推理所有 MaskDINO 模型并生成 JSON 结果
# 使用方法: bash infer_all_with_json.sh <image_path>

set -e

IMAGE_PATH="/home/zhaolin/Projects/DentalXrayAgent/test_data/4a2e27ba-991a-48bc-a7f6-0188fc41c52e/image_1.png"
OUTPUT_BASE="./results"
CONFIDENCE=0.3

echo "🦷 批量推理所有 MaskDINO 模型（生成 JSON 结果）"
echo "输入图像: $IMAGE_PATH"
echo "输出目录: $OUTPUT_BASE"
echo "置信度阈值: $CONFIDENCE"
echo ""

# 激活虚拟环境
# 激活虚拟环境
PROJECT_VENV="/data/zhaolin/tool_env/TVEM/.venv"
if [ -f "$PROJECT_VENV/bin/activate" ]; then
    echo "使用独立虚拟环境: $PROJECT_VENV"
    source "$PROJECT_VENV/bin/activate"
elif [ -f "../../.venv/bin/activate" ]; then
     echo "使用项目根目录虚拟环境"
    source "../../.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "1/4: 11类疾病检测"
echo "=========================================="
python infer_with_json.py \
  --config configs/11diseases.yaml \
  --checkpoint weights/11diseases.pth \
  --image "$IMAGE_PATH" \
  --output-dir "$OUTPUT_BASE/11diseases" \
  --category-file categories/11diseases_category.json \
  --confidence $CONFIDENCE

echo ""
echo "=========================================="
echo "2/4: 4象限分割"
echo "=========================================="
python infer_with_json.py \
  --config configs/4quadrants.yaml \
  --checkpoint weights/4quadrants.pth \
  --image "$IMAGE_PATH" \
  --output-dir "$OUTPUT_BASE/4quadrants" \
  --category-file categories/4quadrants_category.json \
  --confidence $CONFIDENCE

echo ""
echo "=========================================="
echo "3/4: 下颌管和上颌窦"
echo "=========================================="
python infer_with_json.py \
  --config configs/mandibular_maxillary.yaml \
  --checkpoint weights/mandibular_maxillary.pth \
  --image "$IMAGE_PATH" \
  --output-dir "$OUTPUT_BASE/mandibular_maxillary" \
  --category-file categories/mandibular_maxillary_category.json \
  --confidence $CONFIDENCE

echo ""
echo "=========================================="
echo "4/4: 骨质流失检测"
echo "=========================================="
python infer_with_json.py \
  --config configs/bone_loss.yaml \
  --checkpoint weights/bone_loss.pth \
  --image "$IMAGE_PATH" \
  --output-dir "$OUTPUT_BASE/bone_loss" \
  --category-file categories/bone_loss_category.json \
  --confidence $CONFIDENCE

echo ""
echo "=========================================="
echo "✅ 所有推理完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  可视化图像: ls -lh $OUTPUT_BASE/*/*.jpg"
echo "  JSON 结果:   ls -lh $OUTPUT_BASE/*/*.json"
echo ""
echo "合并所有结果:"
cat << PYTHON > /tmp/merge_results_$$.py
import json
import glob
from pathlib import Path

results_dir = Path("$OUTPUT_BASE")
all_results = {}

for json_file in results_dir.glob("*/*_results.json"):
    model_name = json_file.parent.name
    with open(json_file, 'r') as f:
        data = json.load(f)
    all_results[model_name] = {
        "total_detections": data["total_detections"],
        "category_counts": data["category_counts"]
    }

with open(results_dir / "summary.json", 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"✅ 汇总结果已保存: {results_dir}/summary.json")
PYTHON

python /tmp/merge_results_$$.py
rm /tmp/merge_results_$$.py
