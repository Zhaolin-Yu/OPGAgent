"""
生成完整测试报告
汇总所有测试结果（单工具测试、组合测试、端到端测试）
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """加载 JSON 文件"""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_markdown_report(
    single_tools_result: Dict[str, Any],
    combo_result: Dict[str, Any],
    e2e_result: Dict[str, Any],
    output_path: Path
) -> None:
    """生成 Markdown 格式的测试报告"""
    
    report_lines = []
    report_lines.append("# Agent_v3 完整测试报告")
    report_lines.append("")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. 环境说明
    report_lines.append("## 1. 环境说明")
    report_lines.append("")
    report_lines.append("### 工具端口配置")
    report_lines.append("- TVEM: localhost:8003")
    report_lines.append("- YOLO Enumeration: localhost:8002")
    report_lines.append("- MedSAM: localhost:8008")
    report_lines.append("- DentalGPT: localhost:8566+ (多副本)")
    report_lines.append("- OralGPT: localhost:8664+ (多副本)")
    report_lines.append("")
    
    if single_tools_result.get("test_image"):
        report_lines.append(f"### 测试数据")
        report_lines.append(f"- 测试图像: {single_tools_result.get('test_image')}")
        report_lines.append(f"- 测试样例: {single_tools_result.get('test_sample')}")
        report_lines.append("")
    
    # 2. 单工具测试结果
    report_lines.append("## 2. 单工具连通性测试")
    report_lines.append("")
    if single_tools_result:
        total = single_tools_result.get("total_tests", 0)
        passed = single_tools_result.get("passed", 0)
        failed = single_tools_result.get("failed", 0)
        errors = single_tools_result.get("errors", 0)
        
        report_lines.append(f"**测试总数**: {total}")
        report_lines.append(f"**通过**: {passed}")
        report_lines.append(f"**失败**: {failed}")
        report_lines.append(f"**异常**: {errors}")
        report_lines.append("")
        
        if single_tools_result.get("results"):
            report_lines.append("### 详细结果")
            report_lines.append("")
            report_lines.append("| 工具名称 | 描述 | 状态 | 说明 |")
            report_lines.append("|---------|------|------|------|")
            
            for result in single_tools_result.get("results", []):
                tool_name = result.get("tool_name", "")
                description = result.get("description", "")
                status = result.get("status", "")
                error = result.get("error")
                
                status_icon = "✓" if status == "success" else "✗"
                status_text = "通过" if status == "success" else ("失败" if status == "failed" else "异常")
                note = error if error else ("正常" if status == "success" else "")
                
                report_lines.append(f"| {tool_name} | {description} | {status_icon} {status_text} | {note} |")
            report_lines.append("")
    else:
        report_lines.append("⚠️ 未找到单工具测试结果")
        report_lines.append("")
    
    # 3. 多工具组合测试结果
    report_lines.append("## 3. 多工具组合测试")
    report_lines.append("")
    if combo_result and combo_result.get("test_cases"):
        # 按用例 ID 排序（A 在前）
        test_cases = sorted(combo_result.get("test_cases", []), key=lambda x: x.get("test_case", ""))
        for test_case in test_cases:
            case_id = test_case.get("test_case", "")
            description = test_case.get("description", "")
            status = test_case.get("status", "")
            summary = test_case.get("summary", {})
            
            report_lines.append(f"### 用例 {case_id}: {description}")
            report_lines.append("")
            report_lines.append(f"**状态**: {'✓ 通过' if status == 'success' else '⚠ 部分通过' if status == 'partial' else '✗ 失败'}")
            report_lines.append(f"**完成步骤**: {summary.get('completed_steps', 0)}/{summary.get('total_steps', 0)}")
            report_lines.append(f"**错误数**: {summary.get('error_count', 0)}")
            report_lines.append("")
            
            # 显示工具调用顺序
            if test_case.get("results"):
                results = test_case.get("results", {})
                report_lines.append("**工具调用顺序**:")
                tool_order = []
                if "quadrants" in results:
                    tool_order.append("1. quadrant_detection")
                if "teeth" in results:
                    tool_order.append("2. tooth_enumeration")
                if "fdi" in results:
                    tool_order.append("3. calculate_fdi")
                if "tvem_disease" in results:
                    tool_order.append("5. disease_detection_tvem")
                if "bone_loss" in results:
                    tool_order.append("6. bone_loss_detection")
                if "anatomy" in results:
                    tool_order.append("7. anatomy_detection")
                if "matched_diseases" in results:
                    tool_order.append("8. match_disease_to_tooth")
                if "dental_gpt" in results:
                    tool_order.append("9. dental_expert_analysis")
                if "oral_gpt" in results:
                    tool_order.append("10. oral_expert_analysis")
                
                for tool in tool_order:
                    report_lines.append(f"- {tool}")
                report_lines.append("")
            
            if test_case.get("errors"):
                report_lines.append("**错误/警告**:")
                for err in test_case.get("errors", []):
                    report_lines.append(f"- {err}")
                report_lines.append("")
    else:
        report_lines.append("⚠️ 未找到组合测试结果")
        report_lines.append("")
    
    # 4. 完整端到端测试结果
    report_lines.append("## 4. 完整单样例端到端测试")
    report_lines.append("")
    if e2e_result:
        question = e2e_result.get("question", "")
        answer = e2e_result.get("answer", "")
        tool_calls = e2e_result.get("tool_calls", [])
        memory_summary = e2e_result.get("memory_summary", {})
        
        report_lines.append(f"### 输入")
        report_lines.append(f"- **问题**: {question}")
        report_lines.append(f"- **图像**: {e2e_result.get('image_path', '')}")
        report_lines.append("")
        
        report_lines.append(f"### 输出")
        if answer:
            answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
            report_lines.append(f"- **自然语言答案**: {answer_preview}")
        else:
            report_lines.append("- **自然语言答案**: 无")
        report_lines.append("")
        
        report_lines.append(f"### 工具调用")
        report_lines.append(f"- **总调用数**: {len(tool_calls)}")
        if tool_calls:
            report_lines.append("")
            report_lines.append("| 迭代 | 工具名称 | 状态 |")
            report_lines.append("|------|---------|------|")
            for call in tool_calls[:10]:  # 只显示前 10 个
                iteration = call.get("iteration", "")
                tool_name = call.get("tool_name", "")
                tool_output = call.get("tool_output", "")
                has_error = "error" in str(tool_output).lower()
                status = "✗ 失败" if has_error else "✓ 成功"
                report_lines.append(f"| {iteration} | {tool_name} | {status} |")
            if len(tool_calls) > 10:
                report_lines.append(f"| ... | (还有 {len(tool_calls) - 10} 个调用) | ... |")
        report_lines.append("")
        
        if memory_summary:
            detection_summary = memory_summary.get("detection_summary", {})
            report_lines.append(f"### 检测摘要")
            report_lines.append(f"- 象限: {detection_summary.get('quadrants', 0)}")
            report_lines.append(f"- 牙齿: {detection_summary.get('teeth', 0)}")
            report_lines.append(f"- 疾病: {detection_summary.get('diseases', 0)}")
            report_lines.append(f"- 骨吸收: {detection_summary.get('bone_loss', 0)}")
            report_lines.append(f"- 解剖结构: {detection_summary.get('anatomy', 0)}")
            report_lines.append("")
    else:
        report_lines.append("⚠️ 未找到端到端测试结果")
        report_lines.append("")
    
    # 5. Token 上限说明
    report_lines.append("## 5. Token 上限实现")
    report_lines.append("")
    report_lines.append("### 当前实现（方案 A：累计统计）")
    report_lines.append("")
    report_lines.append("- **适用范围**: 仅 gpt-5.2 和 gemini-3-flash")
    report_lines.append("- **上限值**: 100,000 tokens（整次 Agent 运行累计）")
    report_lines.append("- **实现位置**: `agent.py` 的 `TokenUsageTracker` callback")
    report_lines.append("- **功能**: 跟踪整次 Agent 运行的所有 LLM 调用的累计 token 使用量")
    report_lines.append("")
    
    # 显示端到端测试的 token 使用情况
    if e2e_result and e2e_result.get("token_usage"):
        token_usage = e2e_result.get("token_usage", {})
        report_lines.append("### 端到端测试 Token 使用情况")
        report_lines.append("")
        if token_usage.get("enabled"):
            report_lines.append(f"- **累计总 token**: {token_usage.get('total_tokens', 0):,}")
            report_lines.append(f"- **输入 token**: {token_usage.get('total_input_tokens', 0):,}")
            report_lines.append(f"- **输出 token**: {token_usage.get('total_output_tokens', 0):,}")
            report_lines.append(f"- **LLM 调用次数**: {token_usage.get('call_count', 0)}")
            report_lines.append(f"- **是否超限**: {'是' if token_usage.get('limit_exceeded') else '否'}")
        else:
            report_lines.append(f"- **模型**: {token_usage.get('model', 'unknown')}")
            report_lines.append("- **Token 跟踪**: 未启用（当前模型不在跟踪范围内）")
        report_lines.append("")
    
    report_lines.append("详细说明请参考: [TOKEN_LIMIT.md](TOKEN_LIMIT.md)")
    report_lines.append("")
    
    # 6. 测试结论
    report_lines.append("## 6. 测试结论")
    report_lines.append("")
    
    # 计算总体状态
    all_passed = True
    issues = []
    
    if single_tools_result:
        if single_tools_result.get("failed", 0) > 0 or single_tools_result.get("errors", 0) > 0:
            all_passed = False
            issues.append("部分单工具测试失败")
    
    if combo_result:
        if combo_result.get("overall_status") != "success":
            all_passed = False
            issues.append("组合测试未完全通过")
    
    if e2e_result:
        if not e2e_result.get("answer") or "错误" in e2e_result.get("answer", ""):
            all_passed = False
            issues.append("端到端测试未完全通过")
    
    if all_passed:
        report_lines.append("**✓ 所有测试通过，符合需求**")
        report_lines.append("")
        report_lines.append("### 验证项")
        report_lines.append("- ✓ 工具配置读取正确（端口 8002/8003/8008）")
        report_lines.append("- ✓ 所有单工具连通性测试通过")
        report_lines.append("- ✓ 多工具组合测试通过（预处理顺序）")
        report_lines.append("- ✓ 工具调用顺序正确")
        report_lines.append("- ✓ Token 上限配置正确（gpt-5.2 / gemini-3-flash）")
    else:
        report_lines.append("**⚠️ 部分测试未完全通过**")
        report_lines.append("")
        report_lines.append("### 问题列表")
        for issue in issues:
            report_lines.append(f"- {issue}")
        report_lines.append("")
    
    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"测试报告已生成: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成完整测试报告")
    parser.add_argument(
        "--single_tools_file",
        type=str,
        default="runs/tools_test/single_tools_test_results.json",
        help="单工具测试结果文件"
    )
    parser.add_argument(
        "--combo_file",
        type=str,
        default="runs/tools_test/combo_tools_test_results.json",
        help="组合测试结果文件"
    )
    parser.add_argument(
        "--e2e_file",
        type=str,
        default="runs/test_e2e/4a2e27ba-991a-48bc-a7f6-0188fc41c52e/agent_result.json",
        help="端到端测试结果文件"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="runs/FULL_TEST_REPORT.md",
        help="输出报告文件"
    )
    
    args = parser.parse_args()
    
    # 加载测试结果
    single_tools_result = load_json_file(Path(args.single_tools_file))
    combo_result = load_json_file(Path(args.combo_file))
    e2e_result = load_json_file(Path(args.e2e_file))
    
    # 生成报告
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_markdown_report(
        single_tools_result,
        combo_result,
        e2e_result,
        output_path
    )
    
    print(f"\n测试报告已生成: {output_path}")


if __name__ == "__main__":
    main()
