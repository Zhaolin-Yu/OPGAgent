#!/bin/bash
# =============================================================================
# Agent_v3 批量推理脚本
# 
# 功能: 对 test_data 目录下的所有样本进行 OPG 诊断分析
# 输出: runs/inference_<墨尔本时间>/ 目录
#
# 使用方法:
#   bash scripts/batch_inference.sh                    # 使用默认数据目录
#   bash scripts/batch_inference.sh /path/to/data      # 指定数据目录
#   bash scripts/batch_inference.sh --help             # 显示帮助
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_DATA_DIR="/data/zhaolin/Agents/agent_v2/test_data"
QUESTION="Analyze this OPG image and provide a diagnostic report."

# 显示帮助
show_help() {
    echo -e "${BLUE}Agent_v3 批量推理脚本${NC}"
    echo ""
    echo "使用方法:"
    echo "  bash $0 [数据目录] [选项]"
    echo ""
    echo "参数:"
    echo "  数据目录    包含样本子目录的路径 (默认: $DEFAULT_DATA_DIR)"
    echo ""
    echo "选项:"
    echo "  --help      显示此帮助信息"
    echo "  --no-langsmith  禁用 LangSmith 追踪"
    echo ""
    echo "示例:"
    echo "  bash $0                                # 使用默认数据目录"
    echo "  bash $0 /path/to/my_data               # 指定数据目录"
    echo "  bash $0 --no-langsmith                 # 禁用 LangSmith"
    echo ""
    echo "输出目录结构:"
    echo "  runs/inference_YYYYMMDD_HHMMSS/"
    echo "    ├── <sample_id_1>/"
    echo "    │   ├── agent_io.json"
    echo "    │   └── answer.txt"
    echo "    ├── <sample_id_2>/"
    echo "    │   ├── agent_io.json"
    echo "    │   └── answer.txt"
    echo "    ├── ..."
    echo "    └── summary.json"
}

# 解析参数
DATA_DIR="$DEFAULT_DATA_DIR"
USE_LANGSMITH=true

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            exit 0
            ;;
        --no-langsmith)
            USE_LANGSMITH=false
            ;;
        *)
            if [[ -d "$arg" ]]; then
                DATA_DIR="$arg"
            fi
            ;;
    esac
done

# 检查数据目录
if [[ ! -d "$DATA_DIR" ]]; then
    echo -e "${RED}错误: 数据目录不存在: $DATA_DIR${NC}"
    exit 1
fi

# 生成墨尔本时间戳
MELBOURNE_TIMESTAMP=$(TZ='Australia/Melbourne' date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$PROJECT_DIR/runs/inference_${MELBOURNE_TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Agent_v3 批量推理${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "数据目录: ${GREEN}$DATA_DIR${NC}"
echo -e "输出目录: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "墨尔本时间: ${YELLOW}$(TZ='Australia/Melbourne' date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "LangSmith: ${YELLOW}$USE_LANGSMITH${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# 收集所有样本目录
SAMPLES=()
for sample_dir in "$DATA_DIR"/*/; do
    if [[ -d "$sample_dir" ]]; then
        # 检查是否包含 image_1.png
        if [[ -f "${sample_dir}image_1.png" ]]; then
            SAMPLES+=("$sample_dir")
        fi
    fi
done

TOTAL_SAMPLES=${#SAMPLES[@]}
if [[ $TOTAL_SAMPLES -eq 0 ]]; then
    echo -e "${RED}错误: 未找到有效样本 (需要包含 image_1.png)${NC}"
    exit 1
fi

echo -e "发现 ${GREEN}$TOTAL_SAMPLES${NC} 个样本"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# 初始化统计
SUCCESS_COUNT=0
FAIL_COUNT=0
RESULTS=()

# 切换到项目目录
cd "$PROJECT_DIR"

# 遍历处理每个样本
for i in "${!SAMPLES[@]}"; do
    sample_dir="${SAMPLES[$i]}"
    sample_id=$(basename "$sample_dir")
    image_path="${sample_dir}image_1.png"
    
    # 为每个样本创建独立的输出目录
    sample_output_dir="$OUTPUT_DIR/${sample_id}"
    mkdir -p "$sample_output_dir"
    output_file="$sample_output_dir/agent_io.json"
    
    current=$((i + 1))
    echo -e "\n${YELLOW}[$current/$TOTAL_SAMPLES]${NC} 处理: ${GREEN}$sample_id${NC}"
    
    # 构建命令
    CMD="uv run python -m opgagent.cli"
    CMD+=" --question \"$QUESTION\""
    CMD+=" --image_path \"$image_path\""
    CMD+=" --output \"$output_file\""

    if [[ "$USE_LANGSMITH" == false ]]; then
        CMD+=" --no-langsmith"
    fi
    
    # 执行推理
    START_TIME=$(date +%s)
    
    if eval "$CMD" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        # CLI 现在会自动生成 answer.txt，无需额外处理
        echo -e "  ${GREEN}✓ 成功${NC} (耗时: ${DURATION}s)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        RESULTS+=("{\"sample_id\": \"$sample_id\", \"status\": \"success\", \"duration_s\": $DURATION}")
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "  ${RED}✗ 失败${NC} (耗时: ${DURATION}s)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("{\"sample_id\": \"$sample_id\", \"status\": \"failed\", \"duration_s\": $DURATION}")
    fi
done

# 生成汇总报告
SUMMARY_FILE="$OUTPUT_DIR/summary.json"
MELBOURNE_END_TIME=$(TZ='Australia/Melbourne' date '+%Y-%m-%d %H:%M:%S')

cat > "$SUMMARY_FILE" << EOF
{
  "batch_info": {
    "timestamp_melbourne": "$(TZ='Australia/Melbourne' date '+%Y-%m-%d %H:%M:%S')",
    "timestamp_utc": "$(date -u '+%Y-%m-%d %H:%M:%S')",
    "data_directory": "$DATA_DIR",
    "output_directory": "$OUTPUT_DIR",
    "question": "$QUESTION",
    "langsmith_enabled": $USE_LANGSMITH
  },
  "statistics": {
    "total_samples": $TOTAL_SAMPLES,
    "success": $SUCCESS_COUNT,
    "failed": $FAIL_COUNT,
    "success_rate": $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_SAMPLES" | bc)
  },
  "results": [
    $(IFS=,; echo "${RESULTS[*]}")
  ]
}
EOF

# 输出汇总
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}批量推理完成${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "总样本数: ${YELLOW}$TOTAL_SAMPLES${NC}"
echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "失败: ${RED}$FAIL_COUNT${NC}"
echo -e "成功率: ${YELLOW}$(echo "scale=1; $SUCCESS_COUNT * 100 / $TOTAL_SAMPLES" | bc)%${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"
echo -e "输出目录: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "汇总文件: ${GREEN}$SUMMARY_FILE${NC}"
echo -e "完成时间 (墨尔本): ${YELLOW}$MELBOURNE_END_TIME${NC}"
echo -e "${BLUE}============================================================${NC}"
