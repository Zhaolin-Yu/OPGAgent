#!/bin/bash

# =================================================================
# DentalGPT 停止所有副本脚本
# =================================================================

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  停止 DentalGPT 所有副本${NC}"
echo -e "${GREEN}========================================${NC}"

# 端口列表
PORTS=(6608 6609 6610 6611)

for port in "${PORTS[@]}"; do
    # 查找占用该端口的进程
    pid=$(lsof -t -i:$port 2>/dev/null)
    
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}停止端口 $port 上的进程 (PID: $pid)${NC}"
        kill $pid 2>/dev/null
        sleep 1
        
        # 检查是否还在运行
        if kill -0 $pid 2>/dev/null; then
            echo -e "${RED}强制终止 PID: $pid${NC}"
            kill -9 $pid 2>/dev/null
        fi
        
        echo -e "${GREEN}✓ 端口 $port 已释放${NC}"
    else
        echo "端口 $port 上没有运行的服务"
    fi
done

echo ""
echo -e "${GREEN}所有副本已停止！${NC}"
