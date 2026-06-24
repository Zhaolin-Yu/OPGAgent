#!/bin/bash

# =================================================================
# Agent_v3 全部 API 服务启动脚本
# 启动所有工具服务（测试环境，GPU 0-3）
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

# 日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Agent_v3 API 服务全部启动${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "端口分配:"
echo "  - yolo_enumeration:      6600 (GPU 0)"
echo "  - tvem:                   6602 (GPU 2)"
echo "  - medsam:                 6603 (GPU 3)"
echo "  - oral_gpt:               6604-6607 (GPU 0-3)"
echo "  - dental_gpt:             6608-6611 (GPU 0-3)"
echo ""

# 启动服务函数
start_service() {
    local name=$1
    local dir=$2
    local port=$3
    local gpu=$4
    local log_file="$LOG_DIR/${name}.log"
    
    echo -e "${GREEN}启动 $name (端口: $port, GPU: $gpu)...${NC}"
    cd "$SCRIPT_DIR/$dir"
    PORT=$port GPU=$gpu nohup bash start_api.sh > "$log_file" 2>&1 &
    echo "  PID: $!, 日志: $log_file"
    cd "$SCRIPT_DIR"
}

# 启动 YOLO 服务
echo -e "${YELLOW}=== 启动 YOLO 服务 ===${NC}"
start_service "yolo_enumeration" "yolo_enumeration" 6600 0
sleep 2

# 启动 TVEM
echo ""
echo -e "${YELLOW}=== 启动 TVEM 服务 ===${NC}"
start_service "tvem" "tvem" 6602 2
sleep 2

# 启动 MedSAM
echo ""
echo -e "${YELLOW}=== 启动 MedSAM 服务 ===${NC}"
start_service "medsam" "medsam" 6603 3
sleep 2

# 启动 VLM 服务（多副本）
echo ""
echo -e "${YELLOW}=== 启动 OralGPT 服务 (4副本) ===${NC}"
cd "$SCRIPT_DIR/oral_gpt"
bash start_all_replicas.sh
cd "$SCRIPT_DIR"
sleep 5

echo ""
echo -e "${YELLOW}=== 启动 DentalGPT 服务 (4副本) ===${NC}"
cd "$SCRIPT_DIR/dental_gpt"
bash start_all_replicas.sh
cd "$SCRIPT_DIR"
sleep 5

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  全部服务启动完成！${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "等待服务就绪后，可以运行健康检查："
echo "  bash check_health.sh"
echo ""
echo "停止所有服务："
echo "  bash stop_all_services.sh"
