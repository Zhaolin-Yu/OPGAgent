#!/bin/bash

# DentalGPT API Server 依赖安装脚本
# 在 conda 环境 zl_dentalGPT 中安装所有必需的依赖

echo "================================="
echo "DentalGPT API 依赖安装脚本"
echo "================================="

# 激活 conda 环境
echo ""
echo "正在激活 conda 环境: zl_dentalGPT"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate zl_dentalGPT

# 检查是否成功激活
if [[ $CONDA_DEFAULT_ENV != "zl_dentalGPT" ]]; then
    echo "错误: 无法激活 conda 环境 zl_dentalGPT"
    echo "请确保该环境存在，可以使用以下命令检查:"
    echo "  conda env list"
    exit 1
fi

echo "✓ Conda 环境已激活: $CONDA_DEFAULT_ENV"

# 显示 Python 版本
echo ""
echo "Python 版本:"
python --version

# 安装依赖
echo ""
echo "正在安装依赖包..."
echo "================================="

pip install fastapi==0.109.0
pip install "uvicorn[standard]==0.27.0"
pip install python-multipart==0.0.6

echo ""
echo "================================="
echo "✓ 依赖安装完成！"
echo ""
echo "注意: 以下包应该已经在您的环境中安装："
echo "  - torch"
echo "  - transformers"
echo "  - qwen-vl-utils"
echo "  - Pillow"
echo ""
echo "如果缺少这些包，请手动安装："
echo "  pip install torch transformers qwen-vl-utils Pillow"
echo ""
echo "现在可以启动服务器了："
echo "  ./start_server.sh"
echo "================================="
