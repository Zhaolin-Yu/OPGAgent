#!/bin/bash

# =================================================================
# Agent_v3 全部 API 服务停止脚本
# =================================================================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Agent_v3 API 服务全部停止${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 所有服务端口
ALL_PORTS=(6600 6602 6603 6604 6605 6606 6607 6608 6609 6610 6611)

# 端口名称映射
declare -A PORT_NAMES=(
    [6600]="yolo_enumeration"
    [6602]="tvem"
    [6603]="medsam"
    [6604]="oral_gpt_0"
    [6605]="oral_gpt_1"
    [6606]="oral_gpt_2"
    [6607]="oral_gpt_3"
    [6608]="dental_gpt_0"
    [6609]="dental_gpt_1"
    [6610]="dental_gpt_2"
    [6611]="dental_gpt_3"
)

for port in "${ALL_PORTS[@]}"; do
    service_name="${PORT_NAMES[$port]}"
    pid=$(lsof -t -i:$port 2>/dev/null)
    
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}停止 $service_name (端口: $port, PID: $pid)${NC}"
        kill $pid 2>/dev/null
        sleep 0.5
        
        # 检查是否还在运行
        if kill -0 $pid 2>/dev/null; then
            echo -e "${RED}  强制终止 PID: $pid${NC}"
            kill -9 $pid 2>/dev/null
        fi
        
        echo -e "${GREEN}  ✓ 已停止${NC}"
    else
        echo "端口 $port ($service_name) 上没有运行的服务"
    fi
done

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  全部服务已停止！${NC}"
echo -e "${CYAN}========================================${NC}"
