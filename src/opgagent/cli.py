"""
CLI 入口
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .agent import OPGReActAgent
from .memory import AgentMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_dotenv_if_present() -> None:
    """若存在 .env 则加载 LANGCHAIN_* 等变量（不覆盖已有环境变量）。便于用户在 .env 中配置 LangSmith API Key。"""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    key, value = k.strip(), v.strip().strip("'\"")
                    if key.startswith("LANGCHAIN_") and key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass


def main():
    """CLI 入口"""
    _load_dotenv_if_present()
    parser = argparse.ArgumentParser(
        description="Agent_v3: 基于 LangChain create_agent 的纯 ReAct OPG Agent"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="用户问题"
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="OPG 图像路径"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Agent 配置文件路径（默认: src/opgagent/config/agent_config.yaml）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（JSON 格式，包含答案和工具调用历史）"
    )
    
    parser.add_argument(
        "--save_memory",
        type=str,
        default=None,
        help="保存 Memory 到文件（JSON 格式）"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )
    
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="禁用 LangSmith 追踪（默认开启）"
    )
    parser.add_argument(
        "--langsmith-project",
        type=str,
        default=None,
        help="LangSmith 项目名（写入 LANGCHAIN_PROJECT）；默认使用环境变量或 opgagent_opg"
    )
    
    parser.add_argument(
        "--tool-service",
        type=str,
        default="default",
        choices=["default", "api_service"],
        help="工具服务配置: default (原端口 8xxx) 或 api_service (新端口 6600 开始)"
    )
    
    parser.add_argument(
        "--structured",
        action="store_true",
        help="报告生成后，调用 convert_to_structured 将自然语言报告转为结构化 JSON（打印；有 --output 时另存 structured_report.json）"
    )

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # LangSmith：默认开启，可用 --no-langsmith 关闭
    if not getattr(args, "no_langsmith", False):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if args.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = args.langsmith_project
        elif not os.environ.get("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "opgagent_opg"
        api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
        if api_key:
            os.environ["LANGCHAIN_API_KEY"] = api_key
            logger.info("LangSmith 追踪已开启，项目: %s", os.environ.get("LANGCHAIN_PROJECT", "opgagent_opg"))
        else:
            logger.warning("未设置 LANGCHAIN_API_KEY 或 LANGSMITH_API_KEY，trace 将不会上报到 LangSmith")
    
    # 检查图像文件是否存在
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"图像文件不存在: {args.image_path}")
        sys.exit(1)
    
    # 确定配置文件路径
    if args.config:
        config_path = Path(args.config)
    else:
        # 默认配置文件路径
        config_path = Path(__file__).parent / "config" / "agent_config.yaml"
    
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        config_path = None
    
    try:
        # 初始化 Agent
        logger.info(f"正在初始化 Agent (工具服务: {args.tool_service}, VLM: 4 VLM)...")
        agent = OPGReActAgent(
            config_path=str(config_path) if config_path else None,
            tool_service=args.tool_service,
        )
        
        # 运行 Agent
        logger.info(f"开始分析图像: {args.image_path}")
        logger.info(f"问题: {args.question}")
        
        # 运行 Agent（Agent 自主决定是否调用 convert_to_structured 工具）
        result = agent.run(
            question=args.question,
            image_path=str(image_path)
        )
        
        # 输出结果
        print("\n" + "="*80)
        print("分析结果")
        print("="*80)
        
        # 显示 Agent 的 ReAct 推理过程
        react_trace = result.get("react_trace", [])
        if react_trace:
            print("\n--- Agent ReAct Trace ---")
            for step in react_trace:
                iter_num = step.get("iteration", "?")
                thought = step.get("thought", "")
                actions = step.get("actions", [])
                observations = step.get("observations", [])
                
                print(f"\n[Iteration {iter_num}]")
                if thought and thought != "(no explicit thought)":
                    # 截取前 300 字符
                    preview = thought[:300] + ("..." if len(thought) > 300 else "")
                    print(f"  Thought: {preview}")
                
                for i, action in enumerate(actions):
                    tool_name = action.get("tool", "?")
                    print(f"  Action {i+1}: {tool_name}")
                
                for i, obs in enumerate(observations):
                    tool_name = obs.get("tool", "?")
                    output = obs.get("output", "")
                    # 截取前 200 字符
                    preview = output[:200] + ("..." if len(output) > 200 else "")
                    print(f"  Observation {i+1}: [{tool_name}] {preview}")
            print("\n" + "-"*40)
        
        print(f"\nAgent 输出:\n{result['answer']}\n")
        
        # 显示工具调用摘要
        memory = result.get("memory")
        if memory:
            summary = memory.get_summary()
            print(f"\n工具调用统计:")
            print(f"  - 总调用次数: {summary['total_tool_calls']}")
            print(f"  - 检测到的象限数: {summary['detection_summary']['quadrants']}")
            print(f"  - 检测到的牙齿数: {summary['detection_summary']['teeth']}")
            print(f"  - 检测到的疾病数: {summary['detection_summary']['diseases']}")
        
        # 保存结果到文件
        if args.output:
            output_data = {
                "question": args.question,
                "image_path": str(image_path),
                "answer": result["answer"],  # Agent 输出（可能包含结构化 JSON）
                "react_trace": react_trace,  # 完整的 ReAct 追踪
                "tool_calls": result.get("tool_calls", []),
                "memory_summary": memory.get_summary() if memory else None,
                "token_usage": result.get("token_usage", {}),
                "tool_service": args.tool_service  # 记录使用的工具服务配置
            }
            
            # 判断 output 是目录还是文件
            output_path = Path(args.output)
            if output_path.suffix == ".json":
                # 如果指定了 .json 文件，使用该文件名
                output_dir = output_path.parent
                agent_io_path = output_path
            else:
                # 如果指定的是目录，在目录下创建 agent_io.json
                output_dir = output_path
                agent_io_path = output_dir / "agent_io.json"
            
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(agent_io_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果已保存到: {agent_io_path}")
            
            # 同时生成 answer.txt（与 agent_io.json 同目录）
            # 只保留最终报告部分（从 "## " 或 "# " 标题开始，排除推理过程）
            answer_path = output_dir / "answer.txt"
            raw_answer = result["answer"]
            # 兼容 content 为 list 的情况（如部分模型返回 [{"type":"text","text":"..."}]）
            if isinstance(raw_answer, list):
                parts = []
                for block in raw_answer:
                    if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                        parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
                full_answer = "\n".join(parts) if parts else str(raw_answer)
            elif isinstance(raw_answer, str):
                full_answer = raw_answer
            else:
                full_answer = str(raw_answer)

            # 提取最终报告：查找第一个 markdown 标题（## 或 #）
            import re
            report_match = re.search(r'^(#{1,2}\s+.+)$', full_answer, re.MULTILINE)
            if report_match:
                report_start = report_match.start()
                final_report = full_answer[report_start:].strip()
            else:
                final_report = full_answer

            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            logger.info(f"答案已保存到: {answer_path}")

        # 结构化报告：将自然语言报告转换为 Schema&Enum JSON（--structured）
        if getattr(args, "structured", False):
            ans = result.get("answer", "")
            if isinstance(ans, list):
                ans = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in ans
                )
            schema_file = Path(__file__).parent / "config" / "schema_enum_standard.md"
            logger.info("转换为结构化 JSON ...")
            conv = json.loads(agent.toolkit.convert_to_structured(
                str(ans),
                schema_path=str(schema_file) if schema_file.exists() else None,
            ))
            structured = conv.get("structured_report")
            if conv.get("conversion_status") == "success" and structured is not None:
                print("\n结构化报告 (JSON):")
                print(json.dumps(structured, ensure_ascii=False, indent=2))
                if args.output:
                    structured_path = output_dir / "structured_report.json"
                    with open(structured_path, "w", encoding="utf-8") as f:
                        json.dump(structured, f, ensure_ascii=False, indent=2)
                    logger.info(f"结构化报告已保存到: {structured_path}")
            else:
                logger.warning(f"结构化转换失败: {conv.get('error') or conv.get('conversion_status')}")

        # 保存 Memory
        if args.save_memory and memory:
            memory.save_to_file(args.save_memory)
            logger.info(f"Memory 已保存到: {args.save_memory}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"运行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
