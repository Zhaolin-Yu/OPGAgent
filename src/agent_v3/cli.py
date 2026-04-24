"""
CLI entry point
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
    """If a .env file is present, load LANGCHAIN_* variables (without overriding existing env vars). Lets users put their LangSmith API key in .env."""
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
    """CLI entry point"""
    _load_dotenv_if_present()
    parser = argparse.ArgumentParser(
        description="Agent_v3: pure ReAct OPG Agent built on LangChain create_agent"
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="OPG image path"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Agent config file path (default: src/agent_v3/config/agent_config.yaml)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSON with answer and tool-call history)"
    )

    parser.add_argument(
        "--save_memory",
        type=str,
        default=None,
        help="Save memory to file (JSON format)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose logs"
    )

    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="Disable LangSmith tracing (enabled by default)"
    )
    parser.add_argument(
        "--langsmith-project",
        type=str,
        default=None,
        help="LangSmith project name (written to LANGCHAIN_PROJECT); defaults to env var or agent_v3_opg"
    )

    parser.add_argument(
        "--tool-service",
        type=str,
        default="default",
        choices=["default", "api_service"],
        help="Tool service config: default (legacy 8xxx ports) or api_service (new ports starting at 6600)"
    )

    parser.add_argument(
        "--local-vlm-only",
        action="store_true",
        help="Use 3-VLM mode (DentalGPT + OralGPT + GPT-5.2), excluding Gemini"
    )
    parser.add_argument(
        "--answer-mode",
        action="store_true",
        help="Answer mode: after report generation, if the patient directory contains a VQA file, run single-question inference per item and produce vqa_answer.txt (requires --patient-dir)"
    )
    parser.add_argument(
        "--patient-dir",
        type=str,
        default=None,
        help="Patient directory path (used to find VQA files); combined with --output in answer mode to look up VQA questions in that directory"
    )
    parser.add_argument(
        "--vqa-file",
        type=str,
        default="vqa.json",
        help="VQA file name used in answer mode (e.g. vqa.json or vqa_update.json); defaults to vqa.json"
    )
    # NOTE: --structured and --schema have been removed.
    # The agent now decides autonomously whether to call the convert_to_structured tool.
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # LangSmith: enabled by default; disable with --no-langsmith
    if not getattr(args, "no_langsmith", False):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if args.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = args.langsmith_project
        elif not os.environ.get("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "agent_v3_opg"
        api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
        if api_key:
            os.environ["LANGCHAIN_API_KEY"] = api_key
            logger.info("LangSmith tracing enabled, project: %s", os.environ.get("LANGCHAIN_PROJECT", "agent_v3_opg"))
        else:
            logger.warning("LANGCHAIN_API_KEY or LANGSMITH_API_KEY is not set; traces will not be reported to LangSmith")

    # Check that the image file exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image file not found: {args.image_path}")
        sys.exit(1)

    # Determine config file path
    if args.config:
        config_path = Path(args.config)
    else:
        # Default config file path
        config_path = Path(__file__).parent / "config" / "agent_config.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using default config")
        config_path = None

    try:
        # Initialize agent
        vlm_mode = "3 VLM (DentalGPT + OralGPT + GPT-5.2)" if args.local_vlm_only else "4 VLM (all)"
        logger.info(f"Initializing agent (tool service: {args.tool_service}, VLM mode: {vlm_mode})...")
        agent = OPGReActAgent(
            config_path=str(config_path) if config_path else None,
            tool_service=args.tool_service,
            local_vlm_only=args.local_vlm_only
        )

        # Run agent
        logger.info(f"Starting image analysis: {args.image_path}")
        logger.info(f"Question: {args.question}")

        # Run agent (agent decides autonomously whether to call convert_to_structured)
        result = agent.run(
            question=args.question,
            image_path=str(image_path)
        )

        # Output results
        print("\n" + "="*80)
        print("Analysis Result")
        print("="*80)

        # Show the agent's ReAct reasoning trace
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
                    # Truncate to first 300 chars
                    preview = thought[:300] + ("..." if len(thought) > 300 else "")
                    print(f"  Thought: {preview}")

                for i, action in enumerate(actions):
                    tool_name = action.get("tool", "?")
                    print(f"  Action {i+1}: {tool_name}")

                for i, obs in enumerate(observations):
                    tool_name = obs.get("tool", "?")
                    output = obs.get("output", "")
                    # Truncate to first 200 chars
                    preview = output[:200] + ("..." if len(output) > 200 else "")
                    print(f"  Observation {i+1}: [{tool_name}] {preview}")
            print("\n" + "-"*40)

        print(f"\nAgent output:\n{result['answer']}\n")

        # Show tool-call summary
        memory = result.get("memory")
        if memory:
            summary = memory.get_summary()
            print(f"\nTool-call stats:")
            print(f"  - Total calls: {summary['total_tool_calls']}")
            print(f"  - Quadrants detected: {summary['detection_summary']['quadrants']}")
            print(f"  - Teeth detected: {summary['detection_summary']['teeth']}")
            print(f"  - Diseases detected: {summary['detection_summary']['diseases']}")

        # Save result to file
        if args.output:
            output_data = {
                "question": args.question,
                "image_path": str(image_path),
                "answer": result["answer"],  # Agent output (may include structured JSON)
                "react_trace": react_trace,  # Full ReAct trace
                "tool_calls": result.get("tool_calls", []),
                "memory_summary": memory.get_summary() if memory else None,
                "token_usage": result.get("token_usage", {}),
                "tool_service": args.tool_service  # Record the tool service config used
            }

            # Decide whether output is a directory or file
            output_path = Path(args.output)
            if output_path.suffix == ".json":
                # If a .json file was specified, use that file name
                output_dir = output_path.parent
                agent_io_path = output_path
            else:
                # If a directory was specified, create agent_io.json in it
                output_dir = output_path
                agent_io_path = output_dir / "agent_io.json"

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(agent_io_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Result saved to: {agent_io_path}")

            # Also generate answer.txt (alongside agent_io.json)
            # Keep only the final report portion (from the first "## " or "# " heading, excluding reasoning)
            answer_path = output_dir / "answer.txt"
            raw_answer = result["answer"]
            # Handle the case where content is a list (e.g. some models return [{"type":"text","text":"..."}])
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

            # Extract the final report: look for the first markdown heading (## or #)
            import re
            report_match = re.search(r'^(#{1,2}\s+.+)$', full_answer, re.MULTILINE)
            if report_match:
                report_start = report_match.start()
                final_report = full_answer[report_start:].strip()
            else:
                final_report = full_answer

            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            logger.info(f"Answer saved to: {answer_path}")

            # Extract structured report from tool calls (if convert_to_structured was called)
            structured_report = None
            for tc in result.get("tool_calls", []):
                if tc.get("tool_name") == "convert_to_structured":
                    try:
                        tc_out = json.loads(tc.get("tool_output", "{}"))
                        if tc_out.get("conversion_status") == "success":
                            structured_report = tc_out.get("structured_report")
                    except (json.JSONDecodeError, TypeError):
                        pass
            if structured_report:
                structured_path = output_dir / "structured_report.json"
                with open(structured_path, "w", encoding="utf-8") as f:
                    json.dump(structured_report, f, ensure_ascii=False, indent=2)
                logger.info(f"Structured report saved to: {structured_path}")

            # Answer mode: if enabled and the patient directory has a vqa file, run single-question inference per item and write vqa_answer.txt
            if getattr(args, "answer_mode", False) and getattr(args, "patient_dir", None):
                patient_dir = Path(args.patient_dir)
                if patient_dir.exists():
                    from .vqa_runner import run_vqa_after_report
                    vqa_file = getattr(args, "vqa_file", "vqa.json")
                    run_vqa_after_report(
                        toolkit=agent.toolkit,
                        image_path=str(image_path),
                        report=final_report,
                        patient_dir=patient_dir,
                        output_dir=output_dir,
                        vqa_file_name=vqa_file,
                    )
                else:
                    logger.warning(f"Answer mode enabled but patient directory does not exist: {patient_dir}, skipping VQA")
            elif getattr(args, "answer_mode", False):
                logger.warning("Answer mode enabled but --patient-dir was not specified, skipping VQA")

        # Save memory
        if args.save_memory and memory:
            memory.save_to_file(args.save_memory)
            logger.info(f"Memory saved to: {args.save_memory}")

        print("="*80)

    except Exception as e:
        logger.error(f"Run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
