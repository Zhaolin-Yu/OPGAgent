#!/bin/bash

# =================================================================
# MedSAM 分割 API 服务启动脚本
# 端口: 6603
# GPU: 通过 CUDA_VISIBLE_DEVICES 环境变量指定
# =================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置默认值
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6603}"
GPU="${GPU:-0}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MedSAM 分割 API 服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo "📍 地址: http://${HOST}:${PORT}"
echo "📚 文档: http://localhost:${PORT}/docs"
echo "🖥️  GPU: $GPU"
echo ""

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo -e "${GREEN}使用虚拟环境: $SCRIPT_DIR/.venv${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠️  虚拟环境不存在，请先运行: uv venv --python 3.10${NC}"
    exit 1
fi

# 检查模型文件
MODEL_PATH="MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  模型文件不存在: $MODEL_PATH${NC}"
    echo "请从以下链接下载模型："
    echo "https://github.com/bowang-lab/MedSAM"
    echo ""
fi

# 启动服务
echo "启动 FastAPI 服务..."
export API_PORT=$PORT
export API_HOST=$HOST
python api_server.py
