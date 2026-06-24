#!/bin/bash

# =================================================================
# DentalGPT 全部副本启动脚本 (4个副本)
# 端口: 6608-6611
# GPU: 分配到 GPU 0-3
# =================================================================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 副本配置
declare -A REPLICAS=(
    [0]="6608:0"  # 副本0: 端口6608, GPU 0
    [1]="6609:1"  # 副本1: 端口6609, GPU 1
    [2]="6610:2"  # 副本2: 端口6610, GPU 2
    [3]="6611:3"  # 副本3: 端口6611, GPU 3
)

# 日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  DentalGPT 多副本启动${NC}"
echo -e "${GREEN}========================================${NC}"
echo "副本数量: ${#REPLICAS[@]}"
echo "端口范围: 6608-6611"
echo "GPU 分配: 0-3"
echo ""

# 启动函数
start_replica() {
    local replica_id=$1
    local config="${REPLICAS[$replica_id]}"
    local port="${config%:*}"
    local gpu="${config#*:}"
    local log_file="$LOG_DIR/dental_gpt_replica_${replica_id}.log"
    
    echo -e "${GREEN}启动副本 $replica_id: 端口=$port, GPU=$gpu${NC}"
    
    # 后台启动
    PORT=$port GPU=$gpu nohup bash start_api.sh > "$log_file" 2>&1 &
    echo "PID: $!, 日志: $log_file"
}

# 启动所有副本
for replica_id in "${!REPLICAS[@]}"; do
    start_replica $replica_id
    sleep 2  # 等待2秒再启动下一个副本
done

echo ""
echo -e "${GREEN}所有副本已启动！${NC}"
echo ""
echo "状态检查命令:"
for replica_id in "${!REPLICAS[@]}"; do
    config="${REPLICAS[$replica_id]}"
    port="${config%:*}"
    echo "  curl http://localhost:$port/health"
done
echo ""
echo "停止所有副本: bash stop_all_replicas.sh"
