#!/bin/bash

# =================================================================
# Agent_v3 API 服务健康检查脚本
# =================================================================

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Agent_v3 API 服务健康检查${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 服务配置
declare -A SERVICES=(
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

# 统计
total=0
healthy=0
unhealthy=0

for port in "${!SERVICES[@]}"; do
    service_name="${SERVICES[$port]}"
    total=$((total + 1))
    
    # 检查健康状态
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://localhost:$port/health" 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ $service_name (端口: $port) - 健康${NC}"
        healthy=$((healthy + 1))
    else
        echo -e "${RED}✗ $service_name (端口: $port) - 不可用 (HTTP: $response)${NC}"
        unhealthy=$((unhealthy + 1))
    fi
done

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "总计: $total | ${GREEN}健康: $healthy${NC} | ${RED}不可用: $unhealthy${NC}"
echo -e "${CYAN}========================================${NC}"

# 返回错误码（如果有服务不可用）
if [ $unhealthy -gt 0 ]; then
    exit 1
fi
