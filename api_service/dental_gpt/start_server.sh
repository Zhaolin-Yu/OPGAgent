#!/bin/bash

# DentalGPT API Server 启动脚本 (uv 版本)

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================="
echo "DentalGPT API Server 启动脚本"
echo "================================="

# 检查 uv 是否安装
echo ""
echo "检查 uv 环境..."
if ! command -v uv > /dev/null 2>&1; then
    echo "错误: uv 未安装，请先安装 uv"
    echo "安装命令: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 获取本机局域网 IP
echo ""
echo "正在获取本机 IP 地址..."
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "本机 IP: $LOCAL_IP"

# 设置端口
PORT=${API_PORT:-8566}
HOST=${API_HOST:-0.0.0.0}

echo ""
echo "服务配置:"
echo "  - Python 包管理: uv"
echo "  - 访问地址: http://$LOCAL_IP:$PORT"
echo "  - API 文档: http://$LOCAL_IP:$PORT/docs"
echo "  - 健康检查: http://$LOCAL_IP:$PORT/health"
echo ""

# 启动服务器
echo "正在启动服务器..."

# 尝试激活项目根目录的虚拟环境
# 注意：使用特定的虚拟环境
PROJECT_VENV="/data/zhaolin/tool_env/dentalGPT/.venv"
if [ -f "$PROJECT_VENV/bin/activate" ]; then
    echo "使用独立虚拟环境: $PROJECT_VENV"
    source "$PROJECT_VENV/bin/activate"
    python api_server.py
else
    echo "未找到独立虚拟环境，尝试使用项目根目录环境..."
    if [ -f "../../.venv/bin/activate" ]; then
        source "../../.venv/bin/activate"
        python api_server.py
    else
        echo "未找到虚拟环境，尝试使用 uv run..."
        uv run api_server.py
    fi
fi

