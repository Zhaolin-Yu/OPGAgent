#!/bin/bash

# =================================================================
# TVEM MaskDINO API 服务启动脚本
# 端口: 6602
# GPU: 通过 CUDA_VISIBLE_DEVICES 环境变量指定
# =================================================================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置默认值
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6602}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TVEM MaskDINO API 服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo "📍 地址: http://${HOST}:${PORT}"
echo "📚 文档: http://localhost:${PORT}/docs"
echo "🖥️  GPU: $GPU (Device: $DEVICE)"
echo ""

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU

# 尝试激活虚拟环境
PROJECT_VENV="/data/zhaolin/tool_env/TVEM/.venv"
if [ -f "$PROJECT_VENV/bin/activate" ]; then
    echo -e "${GREEN}使用独立虚拟环境: $PROJECT_VENV${NC}"
    source "$PROJECT_VENV/bin/activate"
    python api_server.py --host "$HOST" --port "$PORT" --device "$DEVICE"
else
    echo "未找到独立虚拟环境，尝试使用项目根目录环境..."
    if [ -f "../../.venv/bin/activate" ]; then
        source "../../.venv/bin/activate"
        python api_server.py --host "$HOST" --port "$PORT" --device "$DEVICE"
    else
        echo "未找到虚拟环境，尝试使用 uv run..."
        uv run python api_server.py --host "$HOST" --port "$PORT" --device "$DEVICE"
    fi
fi
