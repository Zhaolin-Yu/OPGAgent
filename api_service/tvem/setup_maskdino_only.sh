#!/bin/bash

# 简化版安装 - 只设置 MaskDINO（稳定可用）

set -e

WORK_DIR="/mnt/hdd/sdb/zlyu/Documents/dentalagent_test/tools/TVEM"
cd "$WORK_DIR"

echo "=========================================="
echo "🦷 设置 MaskDINO 推理环境"
echo "=========================================="
echo ""

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source .venv/bin/activate
echo ""

# 克隆 MaskDINO
echo "📦 克隆 MaskDINO..."
if [ ! -d "MaskDINO" ]; then
    git clone https://github.com/IDEA-Research/MaskDINO.git
    echo "✅ MaskDINO 克隆完成"
else
    echo "✅ MaskDINO 已存在"
fi
echo ""

# 安装依赖
echo "📦 安装依赖..."
cd MaskDINO

echo "   安装 requirements..."
uv pip install numpy==2.2.6
uv pip install setuptools
uv pip install torch==2.9.1
echo "   安装 Detectron2..."
uv pip install 'git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2' --no-build-isolation
uv pip install 'git+https://github.com/cocodataset/cocoapi.git@8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9#subdirectory=PythonAPI' --no-build-isolation
uv pip install 'git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0'
uv pip install -r requirements.txt


echo "   编译 MaskDINO CUDA 算子..."
cd MaskDINO/maskdino/modeling/pixel_decoder/ops
sh make.sh
cd "$WORK_DIR"

echo ""
echo "=========================================="
echo "✅ MaskDINO 环境设置完成！"
echo "=========================================="
echo ""
echo "📋 可用的 MaskDINO 分割模型（4个）:"
echo ""
echo "   1. 11类疾病分割"
echo "      configs/11diseases.yaml + weights/11diseases.pth"
echo ""
echo "   2. 四象限分割"
echo "      configs/4quadrants.yaml + weights/4quadrants.pth"
echo ""
echo "   3. 下颌管和上颌窦"
echo "      configs/mandibular_maxillary.yaml + weights/mandibular_maxillary.pth"
echo ""
echo "   4. 骨质流失"
echo "      configs/bone_loss.yaml + weights/bone_loss.pth"
echo ""
echo "🚀 使用示例:"
echo ""
echo "# 单个模型推理"
echo "uv run python infer_with_json.py \\"
echo "  --config configs/11diseases.yaml \\"
echo "  --checkpoint weights/11diseases.pth \\"
echo "  --image examples/sample.jpg \\"
echo "  --output-dir ./results \\"
echo "  --category-file categories/11diseases_category.json \\"
echo "  --confidence 0.3"
echo ""
echo "# 批量推理所有模型"
echo "bash infer_all_with_json.sh examples/sample.jpg"
echo ""
echo "=========================================="
