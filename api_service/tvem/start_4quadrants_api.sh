#!/bin/bash

# TVEM 4Quadrants API 服务启动脚本 (uv 版本)

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置端口
PORT=${API_PORT:-8001}
HOST=${API_HOST:-0.0.0.0}

echo "🚀 启动 TVEM 4Quadrants API 服务..."
echo "📍 地址: http://${HOST}:${PORT}"
echo "📚 文档: http://localhost:${PORT}/docs"
echo ""

# 使用 uv 运行（自动使用项目虚拟环境）
# 尝试激活项目根目录的虚拟环境
# 注意：使用特定的虚拟环境
PROJECT_VENV="/data/zhaolin/tool_env/TVEM/.venv"
if [ -f "$PROJECT_VENV/bin/activate" ]; then
    echo "使用独立虚拟环境: $PROJECT_VENV"
    source "$PROJECT_VENV/bin/activate"
    python api_server_4quadrants.py --host "$HOST" --port "$PORT"
else
    echo "未找到独立虚拟环境，尝试使用项目根目录环境..."
    if [ -f "../../.venv/bin/activate" ]; then
        source "../../.venv/bin/activate"
        python api_server_4quadrants.py --host "$HOST" --port "$PORT"
    else
        echo "未找到虚拟环境，尝试使用 uv run..."
        uv run python api_server_4quadrants.py --host "$HOST" --port "$PORT"
    fi
fi

