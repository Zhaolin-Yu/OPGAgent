#!/bin/bash

# =================================================================
# YOLO 牙齿编号检测 API 服务启动脚本
# 端口: 6600
# GPU: 通过 CUDA_VISIBLE_DEVICES 环境变量指定
# =================================================================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置默认值
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6600}"
WORKERS="${WORKERS:-1}"
GPU="${GPU:-0}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  YOLO 牙齿编号检测 API 服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo "GPU: $GPU"
echo "Workers: $WORKERS"
echo ""

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU

# 尝试激活虚拟环境
PROJECT_VENV="/data/zhaolin/tool_env/yolo_enumeration/.venv"
if [ -f "$PROJECT_VENV/bin/activate" ]; then
    echo -e "${GREEN}使用独立虚拟环境: $PROJECT_VENV${NC}"
    source "$PROJECT_VENV/bin/activate"
elif [ -f "../../.venv/bin/activate" ]; then
    echo -e "${GREEN}使用项目根目录虚拟环境${NC}"
    source "../../.venv/bin/activate"
fi

# 启动服务
if command -v uv > /dev/null 2>&1 && [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${GREEN}使用 uv 运行服务${NC}"
    uv run uvicorn api_server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info
else
    python -m uvicorn api_server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info
fi
