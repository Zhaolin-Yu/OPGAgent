#!/bin/bash

# =================================================================
# OralGPT-Omni API 服务启动脚本
# 端口: 6604-6607 (4个副本)
# GPU: 通过 CUDA_VISIBLE_DEVICES 环境变量指定 (测试用 GPU 0-3)
# =================================================================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置默认值
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6604}"
GPU="${GPU:-0}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  OralGPT-Omni API 服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo "📍 地址: http://${HOST}:${PORT}"
echo "📚 文档: http://localhost:${PORT}/docs"
echo "🖥️  GPU: $GPU"
echo ""

# 检查 uv 是否安装
if ! command -v uv > /dev/null 2>&1; then
    echo -e "${YELLOW}错误: uv 未安装，请先安装 uv${NC}"
    echo "安装命令: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU

# 启动服务
PROJECT_VENV="/data/zhaolin/tool_env/OralGPT-omni/.venv"
if [ -f "$PROJECT_VENV/bin/activate" ]; then
    echo -e "${GREEN}使用独立虚拟环境: $PROJECT_VENV${NC}"
    source "$PROJECT_VENV/bin/activate"
    python api_server.py --host "$HOST" --port "$PORT"
else
    echo "未找到独立虚拟环境，尝试使用项目根目录环境..."
    if [ -f "../../.venv/bin/activate" ]; then
        source "../../.venv/bin/activate"
        python api_server.py --host "$HOST" --port "$PORT"
    else
        echo "未找到虚拟环境，尝试使用 uv run..."
        uv run api_server.py --host "$HOST" --port "$PORT"
    fi
fi
