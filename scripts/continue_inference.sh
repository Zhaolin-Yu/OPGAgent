#!/bin/bash
# 继续推理：在指定输出目录中只处理尚未有 answer.txt 的样本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${1:-/data/zhaolin/Agents/agent_v2/dataset}"
OUTPUT_DIR="${2:-$PROJECT_DIR/runs/inference_20260202_160416}"
QUESTION="Analyze this OPG image and provide a diagnostic report."

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Agent_v3 继续推理${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "数据目录: ${GREEN}$DATA_DIR${NC}"
echo -e "输出目录: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

SAMPLES=()
for sample_dir in "$DATA_DIR"/*/; do
    if [[ -d "$sample_dir" && -f "${sample_dir}image_1.png" ]]; then
        sample_id=$(basename "$sample_dir")
        if [[ -f "$OUTPUT_DIR/$sample_id/answer.txt" ]]; then
            continue
        fi
        SAMPLES+=("$sample_dir")
    fi
done

TOTAL_SAMPLES=${#SAMPLES[@]}
echo -e "待处理: ${GREEN}$TOTAL_SAMPLES${NC} 个样本"
echo -e "${BLUE}------------------------------------------------------------${NC}"

cd "$PROJECT_DIR"

SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!SAMPLES[@]}"; do
    sample_dir="${SAMPLES[$i]}"
    sample_id=$(basename "$sample_dir")
    image_path="${sample_dir}image_1.png"

    sample_output_dir="$OUTPUT_DIR/${sample_id}"
    mkdir -p "$sample_output_dir"
    output_file="$sample_output_dir/agent_io.json"

    current=$((i + 1))
    echo -e "\n${YELLOW}[$current/$TOTAL_SAMPLES]${NC} 处理: ${GREEN}$sample_id${NC}"

    START_TIME=$(date +%s)

    if uv run python -m opgagent.cli \
        --question "$QUESTION" \
        --image_path "$image_path" \
        --output "$output_file" \
        --tool-service api_service 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "  ${GREEN}✓ 成功${NC} (耗时: ${DURATION}s)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "  ${RED}✗ 失败${NC} (耗时: ${DURATION}s)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}推理完成${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "总样本数: ${YELLOW}$TOTAL_SAMPLES${NC}"
echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "失败: ${RED}$FAIL_COUNT${NC}"
