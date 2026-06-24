#!/bin/bash
# 测试 Agent_v3 脚本

set -e

# 进入 Agent_v3 目录
cd "$(dirname "$0")/.."

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "警告: 未检测到虚拟环境，建议先激活虚拟环境"
    echo "可以使用: cd Agent_v3 && uv sync"
fi

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 运行测试
python3 scripts/test_agent.py \
    --test_data_dir ../test_data \
    --output_dir runs/test \
    --question "请分析这张 OPG 全景片，生成完整的诊断报告。" \
    --generate_report

echo "测试完成！结果保存在 runs/test/ 目录"
