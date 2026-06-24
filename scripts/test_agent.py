"""
测试 Agent_v3 脚本
运行 test_data 中的样例，从 GT 中提取问题或生成问题
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.agent import OPGReActAgent
from opgagent.memory import AgentMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_questions_from_gt(gt_path: Path) -> List[str]:
    """
    从 GT（structured_report.json）中提取或生成问题
    
    基于 GT 中的发现，生成 10 个相关问题
    
    Args:
        gt_path: GT 文件路径
        
    Returns:
        问题列表（最多 10 个）
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    questions = []
    
    # 1. 基础问题
    questions.append("请分析这张 OPG 全景片，生成完整的诊断报告。")
    
    # 2. 从 GT 中提取具体发现，生成问题
    teeth = gt.get("teeth", {})
    if teeth:
        # 提取异常牙齿
        abnormal_teeth = []
        for fdi, findings in teeth.items():
            if findings:
                abnormal_teeth.append(fdi)
        
        if abnormal_teeth:
            questions.append(f"请分析以下牙齿的异常情况：{', '.join(abnormal_teeth)}")
        
        # 检查是否有阻生齿
        impacted_teeth = [fdi for fdi, f in teeth.items() if f.get("status") == "impacted"]
        if impacted_teeth:
            questions.append(f"请评估以下阻生齿的情况，包括阻生角度和与重要结构的关系：{', '.join(impacted_teeth)}")
        
        # 检查是否有龋齿
        caries_teeth = [fdi for fdi, f in teeth.items() if f.get("caries_location")]
        if caries_teeth:
            questions.append(f"请评估以下牙齿的龋齿情况：{', '.join(caries_teeth)}")
        
        # 检查是否有根尖周病变
        periapical_teeth = [fdi for fdi, f in teeth.items() if f.get("periapical_status") or f.get("pai_score")]
        if periapical_teeth:
            questions.append(f"请评估以下牙齿的根尖周状况：{', '.join(periapical_teeth)}")
        
        # 检查是否有修复体问题
        restoration_teeth = [fdi for fdi, f in teeth.items() if f.get("restoration_issue")]
        if restoration_teeth:
            questions.append(f"请评估以下牙齿的修复体状况：{', '.join(restoration_teeth)}")
    
    # 3. 检查牙周状况
    periodontium = gt.get("periodontium")
    if periodontium:
        severity = periodontium.get("severity")
        if severity:
            questions.append(f"请评估这张 OPG 的牙周状况，特别是骨吸收的严重程度（{severity}）和模式。")
    
    # 4. 检查鼻窦
    sinuses = gt.get("sinuses")
    if sinuses:
        maxillary_sinus = sinuses.get("maxillary_sinus")
        if maxillary_sinus:
            finding = maxillary_sinus.get("finding")
            if finding:
                questions.append(f"请评估上颌窦的状况，特别是 {finding}。")
    
    # 5. 检查缺失牙
    missing_teeth = gt.get("dentition_summary", {}).get("missing_teeth_fdi", [])
    if missing_teeth:
        questions.append(f"请确认以下牙齿是否缺失：{', '.join(missing_teeth)}")
    
    # 6. 检查 TMJ
    tmj = gt.get("tmj")
    if tmj:
        morphology = tmj.get("morphology")
        if morphology:
            questions.append("请评估颞下颌关节（TMJ）的形态。")
    
    # 7. 检查颌骨
    jaws = gt.get("jaws")
    if jaws:
        finding = jaws.get("finding")
        if finding:
            questions.append("请评估颌骨是否有异常发现。")
    
    # 8-10. 补充通用问题
    questions.extend([
        "请评估整体牙周状况，包括骨吸收程度（mild/moderate/severe）和模式（horizontal/vertical）。",
        "请检查是否有根尖周病变，并评估其严重程度。",
        "请评估第三磨牙的状况（如果有），包括是否阻生、阻生角度、与重要结构的关系。"
    ])
    
    # 返回前 10 个问题
    return questions[:10]


def generate_structured_report_prompt(gt: Dict[str, Any]) -> str:
    """
    生成结构化报告生成的 prompt（基于 Agent_refactor 的 Schema）
    
    Args:
        gt: GT 数据（用于参考格式）
        
    Returns:
        Prompt 字符串
    """
    patient = gt.get("patient", {})
    age = patient.get("age")
    sex = patient.get("sex")
    
    prompt = (
        "请基于你的分析结果，生成一个符合 Schema 标准的结构化 JSON 报告。\n\n"
        "报告要求：\n"
        "1. 遵循稀疏表示原则：只报告异常发现，不报告正常状态\n"
        "2. 使用 FDI 两 digit 编号作为牙齿键（如 \"11\", \"18\"）\n"
        "3. 只报告 OPG 可见的发现，不要提及咬合、气道、头影测量\n"
        "4. 缺失牙齿应列在 dentition_summary.missing_teeth_fdi 数组中\n"
        "5. 输出必须是有效的 JSON，不要包含 Markdown 代码块或解释文字\n"
        "6. 只包含已确认的发现（≥3 个来源支持），不确定的发现不要包含\n\n"
    )
    
    if age:
        prompt += f"患者年龄：{age} 岁\n"
    if sex:
        prompt += f"患者性别：{sex}\n"
    
    prompt += (
        "\n报告应包含以下字段（如果存在异常）：\n"
        "- patient: {age, sex}\n"
        "- dentition_summary: {missing_teeth_fdi: []}\n"
        "- teeth: {fdi: {status, caries_location, periapical_status, pai_score, restoration_issue, ...}}\n"
        "  * status: \"missing\" | \"impacted\" | \"residual_root\" | \"implant\"\n"
        "  * 阻生齿相关：winters_class (\"vertical\" | \"angled\" | \"horizontal\"), relationship (\"approximates_iac\" | \"approximates_sinus\")\n"
        "- periodontium: {severity (\"mild\" | \"moderate\" | \"severe\"), pattern (\"horizontal\" | \"vertical\"), findings: []}\n"
        "- sinuses: {maxillary_sinus: {finding, severity}}\n"
        "- tmj: {morphology}\n"
        "- jaws: {finding}\n\n"
        "重要约束：\n"
        "- 只有第三磨牙（FDI 以 8 结尾）可以是阻生齿\n"
        "- 只报告已确认的发现，不确定的发现不要包含\n"
        "- 遵循稀疏表示：只包含异常字段，正常状态完全省略\n\n"
        "请生成结构化 JSON 报告（仅 JSON，无其他文字）。"
    )
    
    return prompt


def run_test(
    test_dir: Path,
    output_dir: Path,
    question: Optional[str] = None,
    generate_report: bool = False
) -> Dict[str, Any]:
    """
    运行单个测试样例
    
    Args:
        test_dir: 测试样例目录
        output_dir: 输出目录
        question: 用户问题（如果为 None，则从 GT 提取）
        generate_report: 是否生成结构化报告
        
    Returns:
        测试结果
    """
    # 查找图像文件
    image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"未找到图像文件: {test_dir}")
    
    image_path = image_files[0]
    logger.info(f"使用图像: {image_path}")
    
    # 加载 GT
    gt_path = test_dir / "structured_report.json"
    gt = None
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        logger.info(f"加载 GT: {gt_path}")
    
    # 确定问题
    if question is None:
        if gt:
            questions = extract_questions_from_gt(gt_path)
            question = questions[0]  # 使用第一个问题
            logger.info(f"从 GT 提取问题: {question}")
        else:
            question = "请分析这张 OPG 全景片，生成完整的诊断报告。"
            logger.info(f"使用默认问题: {question}")
    
    # 初始化 Agent
    config_path = Path(__file__).parent.parent / "src" / "opgagent" / "config" / "agent_config.yaml"
    agent = OPGReActAgent(config_path=str(config_path) if config_path.exists() else None)
    
    # 运行 Agent
    logger.info("开始运行 Agent...")
    result = agent.run(
        question=question,
        image_path=str(image_path)
    )
    
    # 保存结果
    test_id = test_dir.name
    output_test_dir = output_dir / test_id
    output_test_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 Agent 输出
    output_data = {
        "question": question,
        "image_path": str(image_path),
        "answer": result.get("answer", ""),
        "tool_calls": result.get("tool_calls", []),
        "memory_summary": result.get("memory", {}).get_summary() if result.get("memory") else None
    }
    
    output_file = output_test_dir / "agent_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {output_file}")
    
    # 如果要求生成结构化报告
    if generate_report and gt:
        logger.info("生成结构化报告...")
        report_prompt = generate_structured_report_prompt(gt)
        
        # 使用 Agent 生成结构化报告
        report_result = agent.run(
            question=report_prompt,
            image_path=str(image_path),
            memory=result.get("memory")  # 复用之前的 memory
        )
        
        # 尝试解析 JSON
        answer_text = report_result.get("answer", "")
        try:
            # 尝试提取 JSON（可能包含在 Markdown 代码块中）
            import re
            json_match = re.search(r'\{[\s\S]*\}', answer_text)
            if json_match:
                structured_report = json.loads(json_match.group())
            else:
                structured_report = json.loads(answer_text)
        except:
            logger.warning("无法解析结构化报告 JSON，保存原始文本")
            structured_report = {"raw_text": answer_text}
        
        report_file = output_test_dir / "structured_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(structured_report, f, ensure_ascii=False, indent=2)
        logger.info(f"结构化报告已保存: {report_file}")
    
    return output_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 Agent_v3")
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="测试样例目录（默认：test_data 中的第一个）"
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="../test_data",
        help="test_data 目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/test",
        help="输出目录"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="用户问题（如果为 None，则从 GT 提取）"
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="是否生成结构化报告"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有测试样例"
    )
    
    args = parser.parse_args()
    
    test_data_dir = Path(args.test_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        # 运行所有测试样例
        test_dirs = [d for d in test_data_dir.iterdir() if d.is_dir()]
        logger.info(f"找到 {len(test_dirs)} 个测试样例")
        
        results = []
        for test_dir in test_dirs:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"测试样例: {test_dir.name}")
                logger.info(f"{'='*80}")
                
                result = run_test(
                    test_dir=test_dir,
                    output_dir=output_dir,
                    question=args.question,
                    generate_report=args.generate_report
                )
                results.append({
                    "test_id": test_dir.name,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                logger.error(f"测试失败 {test_dir.name}: {e}", exc_info=True)
                results.append({
                    "test_id": test_dir.name,
                    "status": "error",
                    "error": str(e)
                })
        
        # 保存汇总结果
        summary_file = output_dir / "test_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n测试完成，汇总结果已保存: {summary_file}")
    else:
        # 运行单个测试样例
        if args.test_dir:
            test_dir = Path(args.test_dir)
        else:
            # 使用第一个测试样例
            test_dirs = [d for d in test_data_dir.iterdir() if d.is_dir()]
            if not test_dirs:
                logger.error(f"未找到测试样例: {test_data_dir}")
                return
            test_dir = test_dirs[0]
            logger.info(f"使用第一个测试样例: {test_dir.name}")
        
        run_test(
            test_dir=test_dir,
            output_dir=output_dir,
            question=args.question,
            generate_report=args.generate_report
        )


if __name__ == "__main__":
    main()
