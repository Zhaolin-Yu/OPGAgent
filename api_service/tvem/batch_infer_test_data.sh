#!/bin/bash

# 批量处理 test_data 目录下所有数据的推理脚本
# 使用方法: bash batch_infer_test_data.sh <test_data_dir>

set -e

# 参数解析
TEST_DATA_DIR="${1:-/data/zhaolin/Agents/agent_v2/test_data}"
CONFIDENCE=0.3

# 脚本所在目录（用于定位配置文件和权重）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🦷 批量推理 test_data 目录下所有数据"
echo "数据目录: $TEST_DATA_DIR"
echo "置信度阈值: $CONFIDENCE"
echo ""

# 激活虚拟环境
PROJECT_VENV="/data/zhaolin/tool_env/TVEM/.venv"
if [ -f "$PROJECT_VENV/bin/activate" ]; then
    echo "使用独立虚拟环境: $PROJECT_VENV"
    source "$PROJECT_VENV/bin/activate"
elif [ -f "$SCRIPT_DIR/../../.venv/bin/activate" ]; then
    echo "使用项目根目录虚拟环境"
    source "$SCRIPT_DIR/../../.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# 切换到脚本目录（确保相对路径正确）
cd "$SCRIPT_DIR"

# 统计变量
TOTAL=0
SUCCESS=0
FAILED=0

# 遍历 test_data 目录下的所有子目录
for DATA_DIR in "$TEST_DATA_DIR"/*/; do
    # 获取目录名（去掉末尾斜杠）
    DATA_NAME=$(basename "$DATA_DIR")
    IMAGE_PATH="$DATA_DIR/image_1.png"
    OUTPUT_BASE="$DATA_DIR/tvem_results"
    
    # 检查图像是否存在
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "⚠️  跳过 $DATA_NAME: 未找到 image_1.png"
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    
    echo ""
    echo "=========================================="
    echo "📂 处理: $DATA_NAME ($TOTAL)"
    echo "=========================================="
    echo "输入图像: $IMAGE_PATH"
    echo "输出目录: $OUTPUT_BASE"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_BASE"
    
    # 执行4个模型的推理
    (
        echo ""
        echo "  [1/4] 11类疾病检测..."
        python infer_with_json.py \
            --config configs/11diseases.yaml \
            --checkpoint weights/11diseases.pth \
            --image "$IMAGE_PATH" \
            --output-dir "$OUTPUT_BASE/11diseases" \
            --category-file categories/11diseases_category.json \
            --confidence $CONFIDENCE
        
        echo "  [2/4] 4象限分割..."
        python infer_with_json.py \
            --config configs/4quadrants.yaml \
            --checkpoint weights/4quadrants.pth \
            --image "$IMAGE_PATH" \
            --output-dir "$OUTPUT_BASE/4quadrants" \
            --category-file categories/4quadrants_category.json \
            --confidence $CONFIDENCE
        
        echo "  [3/4] 下颌管和上颌窦..."
        python infer_with_json.py \
            --config configs/mandibular_maxillary.yaml \
            --checkpoint weights/mandibular_maxillary.pth \
            --image "$IMAGE_PATH" \
            --output-dir "$OUTPUT_BASE/mandibular_maxillary" \
            --category-file categories/mandibular_maxillary_category.json \
            --confidence $CONFIDENCE
        
        echo "  [4/4] 骨质流失检测..."
        python infer_with_json.py \
            --config configs/bone_loss.yaml \
            --checkpoint weights/bone_loss.pth \
            --image "$IMAGE_PATH" \
            --output-dir "$OUTPUT_BASE/bone_loss" \
            --category-file categories/bone_loss_category.json \
            --confidence $CONFIDENCE
    ) && {
        # 生成汇总 JSON
        python - "$OUTPUT_BASE" << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
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

print(f"  ✅ 汇总结果已保存: {results_dir}/summary.json")
PYTHON_SCRIPT
        
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ $DATA_NAME 处理完成"
    } || {
        FAILED=$((FAILED + 1))
        echo "  ❌ $DATA_NAME 处理失败"
    }
done

echo ""
echo "=========================================="
echo "📊 批量处理完成"
echo "=========================================="
echo "总计: $TOTAL"
echo "成功: $SUCCESS"
echo "失败: $FAILED"
echo ""
