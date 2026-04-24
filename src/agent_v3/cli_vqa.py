#!/usr/bin/env python3
"""
Agent_v3 VQA CLI entry point

Batch-answers VQA questions on OPG images: only the question and options
are shown to the agent (answers are not provided).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_vqa_questions(vqa_path: str) -> list[dict[str, Any]]:
    """
    Load VQA questions, keeping only questions and options (discarding answers
    and explanations).

    Return format:
    [
        {
            "question_id": "sc_0",
            "question_type": "single_choice",
            "question": "question text",
            "options": ["A. xxx", "B. xxx", ...],
            "structure": "teeth",
            "finding": "implant",
            "feature": "status"
        },
        ...
    ]
    """
    with open(vqa_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    questions = []

    # Single-choice
    for i, item in enumerate(vqa_data.get("single_choice", [])):
        questions.append({
            "question_id": f"sc_{i}",
            "question_type": "single_choice",
            "question": item["question"],
            "options": item["options"],
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })

    # Multiple-choice
    for i, item in enumerate(vqa_data.get("multiple_choice", [])):
        questions.append({
            "question_id": f"mc_{i}",
            "question_type": "multiple_choice",
            "question": item["question"],
            "options": item["options"],
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })

    # True/false
    for i, item in enumerate(vqa_data.get("true_false", [])):
        questions.append({
            "question_id": f"tf_{i}",
            "question_type": "true_false",
            "question": item["question"],
            "options": ["True", "False"],
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })

    # Short-answer (if any)
    for i, item in enumerate(vqa_data.get("short_answer", [])):
        questions.append({
            "question_id": f"sa_{i}",
            "question_type": "short_answer",
            "question": item["question"],
            "options": None,  # short_answer has no options
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })

    return questions


# Enum value definitions (based on Schema&Enum_standard.md)
ENUM_DEFINITIONS = {
    # Teeth Level
    "teeth": {
        "status": ["present", "missing", "unerupted", "erupted", "impacted", "residual_root", "implant", "root_filled", "supernumerary", "exfoliating"],
        "winters_class": ["vertical", "angled", "horizontal"],
        "icdas_code": ["early", "moderate", "advanced"],
        "pai_score": ["normal", "mild_change", "moderate_change", "severe_change"],
        "relationship": ["approximates_iac", "approximates_sinus"],
        "root_anomaly": ["abnormal_morphology", "root_rounding"],
        "crown_anomaly": ["abnormal_morphology", "developmental_anomaly"],
        "position_anomaly": ["transposed", "insufficient_space", "ectopic_eruption"],
        "caries_location": ["recurrent", "root_surface"],
        "restoration_issue": ["defective"],
        "apical_status": ["scar", "active_pathology"],
    },
    # Periodontal Level
    "periodontium": {
        "severity": ["normal", "mild", "moderate", "severe"],
        "bone_loss_pattern": ["horizontal", "vertical"],
        "findings": ["normal", "calculus", "pericoronitis"],
    },
    # TMJ
    "tmj": {
        "morphology": ["normal", "degenerative", "developmental"],
    },
    # Jaws
    "jaws": {
        "finding": ["sclerotic_lesion", "lucent_lesion", "bone_variant", "osteopenia", "marrow_space_prominence", "surgical_hardware"],
    },
    # Sinuses
    "sinuses": {
        "finding": ["mucosal_change", "opacification", "air_fluid_level"],
        "severity": ["mild", "moderate", "severe"],
    },
    # Anatomical Variants
    "anatomical_variants": {
        "variant_present": ["bridged_sella", "stylohyoid_ossification", "retained_fragment"],
    },
}

# Mapping from finding to relevant enum values
FINDING_TO_ENUMS = {
    # General
    "normal": ["normal"],
    # Teeth findings
    "missing": ["missing"],
    "impacted": ["impacted", "vertical", "angled", "horizontal"],
    "implant": ["implant"],
    "caries": ["early", "moderate", "advanced", "recurrent", "root_surface"],
    "periapical": ["normal", "mild_change", "moderate_change", "severe_change", "scar", "active_pathology"],
    "root_canal": ["root_filled"],
    "restoration": ["defective"],
    "supernumerary": ["supernumerary"],
    "root": ["abnormal_morphology", "root_rounding"],
    "crown": ["abnormal_morphology", "developmental_anomaly"],
    "position": ["transposed", "insufficient_space", "ectopic_eruption"],
    # Periodontal findings
    "bone_loss": ["normal", "mild", "moderate", "severe", "horizontal", "vertical"],
    "calculus": ["calculus"],
    "pericoronitis": ["pericoronitis"],
    "periodontal": ["normal", "mild", "moderate", "severe", "horizontal", "vertical", "calculus", "pericoronitis"],
    # TMJ findings
    "degenerative": ["normal", "degenerative", "developmental"],
    "tmj": ["normal", "degenerative", "developmental"],
    # Sinus findings
    "sinus": ["normal", "mucosal_change", "opacification", "air_fluid_level", "mild", "moderate", "severe"],
    "mucosal": ["mucosal_change"],
    # Jaw findings
    "lesion": ["sclerotic_lesion", "lucent_lesion"],
    "bone": ["bone_variant", "osteopenia", "marrow_space_prominence"],
    "hardware": ["surgical_hardware"],
}


def get_relevant_enums(structure: str | None, finding: str | None, feature: str | None) -> list[str]:
    """
    Retrieve relevant enum values for a question based on its structure,
    finding, and feature fields.
    """
    enums = set()

    # Add all relevant enums by structure
    if structure and structure in ENUM_DEFINITIONS:
        for field_enums in ENUM_DEFINITIONS[structure].values():
            enums.update(field_enums)

    # Add relevant enums by finding
    if finding:
        finding_lower = finding.lower()
        for key, values in FINDING_TO_ENUMS.items():
            if key in finding_lower or finding_lower in key:
                enums.update(values)

    # If no relevant enums were found, fall back to common defaults
    if not enums:
        # Add generic enums
        enums.update(["mild", "moderate", "severe", "normal", "present", "missing", "impacted"])
        for field_enums in ENUM_DEFINITIONS.get("teeth", {}).values():
            enums.update(field_enums)

    return sorted(list(enums))


def format_question_prompt(q: dict[str, Any]) -> str:
    """
    Format a question as a prompt the agent can understand.
    """
    prompt_parts = [f"Question: {q['question']}"]

    if q["question_type"] == "single_choice":
        prompt_parts.append("\nThis is a SINGLE CHOICE question. Select ONE answer from:")
        for opt in q["options"]:
            prompt_parts.append(f"  {opt}")
        prompt_parts.append("\nAnswer with ONLY the letter (e.g., 'A' or 'B').")

    elif q["question_type"] == "multiple_choice":
        prompt_parts.append("\nThis is a MULTIPLE CHOICE question. Select ALL correct answers from:")
        for opt in q["options"]:
            prompt_parts.append(f"  {opt}")
        prompt_parts.append("\nAnswer with letters separated by commas (e.g., 'A, B' or 'A, C, D').")

    elif q["question_type"] == "true_false":
        prompt_parts.append("\nThis is a TRUE/FALSE question.")
        prompt_parts.append("Answer with ONLY 'True' or 'False'.")

    elif q["question_type"] == "short_answer":
        prompt_parts.append("\nThis is a SHORT ANSWER question.")

        # Retrieve relevant enum values
        relevant_enums = get_relevant_enums(
            q.get("structure"),
            q.get("finding"),
            q.get("feature")
        )

        if relevant_enums:
            prompt_parts.append("\nYour answer MUST be one of the following standardized enum values (snake_case):")
            prompt_parts.append(f"  {', '.join(relevant_enums)}")
            prompt_parts.append("\nAnswer with ONLY the enum value (e.g., 'mild' or 'impacted' or 'horizontal').")
        else:
            prompt_parts.append("Provide a concise answer based on the OPG image.")

    return "\n".join(prompt_parts)


def main():
    parser = argparse.ArgumentParser(description="Agent_v3 VQA CLI")
    parser.add_argument("--image_path", required=True, help="OPG image path")
    parser.add_argument("--vqa_path", required=True, help="Path to VQA question JSON file")
    parser.add_argument("--output", required=True, help="Path to output result JSON file")
    parser.add_argument("--no-langsmith", action="store_true", help="Disable LangSmith tracing (enabled by default)")
    parser.add_argument("--config", help="Agent config file path")
    parser.add_argument(
        "--tool-service",
        type=str,
        default="default",
        choices=["default", "api_service"],
        help="Tool service config: default (legacy 8xxx ports) or api_service (new ports starting at 6600)"
    )

    args = parser.parse_args()

    # Check that files exist
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        sys.exit(1)

    if not os.path.exists(args.vqa_path):
        logger.error(f"VQA file not found: {args.vqa_path}")
        sys.exit(1)

    # LangSmith: enabled by default; disable with --no-langsmith
    if not getattr(args, "no_langsmith", False):
        langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        if langsmith_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
            os.environ.setdefault("LANGCHAIN_PROJECT", "agent_v3_vqa")
            logger.info("LangSmith tracing enabled, project: agent_v3_vqa")
        else:
            logger.warning("LANGSMITH_API_KEY or LANGCHAIN_API_KEY not found; LangSmith tracing disabled")

    # Load questions
    logger.info(f"Loading VQA questions: {args.vqa_path}")
    questions = load_vqa_questions(args.vqa_path)
    logger.info(f"Total: {len(questions)} questions")

    # Initialize agent
    logger.info(f"Initializing agent (tool service: {args.tool_service})...")
    from agent_v3.agent import OPGReActAgent

    config_path = args.config
    if config_path and not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using default config")
        config_path = None

    agent = OPGReActAgent(config_path=config_path, tool_service=args.tool_service)

    # Store results
    results = {
        "image_path": str(args.image_path),
        "vqa_path": str(args.vqa_path),
        "total_questions": len(questions),
        "answers": []
    }

    # Answer questions one by one
    for i, q in enumerate(questions):
        logger.info(f"[{i+1}/{len(questions)}] Answering question: {q['question_id']} ({q['question_type']})")

        # Format the question
        prompt = format_question_prompt(q)

        try:
            # Invoke the agent
            result = agent.run(
                question=prompt,
                image_path=args.image_path,
            )

            answer = result.get("answer", "")

            # Parse answer
            parsed_answer = parse_answer(
                answer,
                q["question_type"],
                structure=q.get("structure"),
                finding=q.get("finding")
            )

            results["answers"].append({
                "question_id": q["question_id"],
                "question_type": q["question_type"],
                "question": q["question"],
                "options": q["options"],
                "agent_answer": parsed_answer,
                "agent_raw_output": answer,
                "structure": q.get("structure"),
                "finding": q.get("finding"),
                "feature": q.get("feature"),
            })

            logger.info(f"  Answer: {parsed_answer}")

        except Exception as e:
            logger.error(f"  Failed to answer question: {e}")
            results["answers"].append({
                "question_id": q["question_id"],
                "question_type": q["question_type"],
                "question": q["question"],
                "options": q["options"],
                "agent_answer": None,
                "error": str(e),
                "structure": q.get("structure"),
                "finding": q.get("finding"),
                "feature": q.get("feature"),
            })

    # Save results
    output_path = Path(args.output)

    # If using api_service, add a "new_" prefix to the output directory
    if args.tool_service == "api_service":
        # Add "new_" prefix before a key directory name
        path_str = str(output_path)
        # Preferred list of directory names (ordered by priority)
        key_dirs = ["Predictions", "Results", "output", "reports", "vqa"]
        for key_dir in key_dirs:
            if f"/{key_dir}/" in path_str or path_str.startswith(f"{key_dir}/"):
                path_str = path_str.replace(f"/{key_dir}/", f"/new_{key_dir}/", 1)
                path_str = path_str.replace(f"{key_dir}/", f"new_{key_dir}/", 1) if path_str.startswith(f"{key_dir}/") else path_str
                break
        else:
            # If no matching directory name found, prefix the second-to-last directory
            parts = list(output_path.parts)
            if len(parts) > 2:
                parts[-2] = f"new_{parts[-2]}"
                path_str = str(Path(*parts))
        output_path = Path(path_str)

    # Record the tool service config used
    results["tool_service"] = args.tool_service

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")

    # Stats
    answered = sum(1 for a in results["answers"] if a.get("agent_answer") is not None)
    logger.info(f"Done: {answered}/{len(questions)} questions answered")


def parse_answer(raw_answer: str, question_type: str, structure: str = None, finding: str = None) -> Any:
    """
    Parse an answer from the agent's raw output.
    """
    if not raw_answer:
        return None

    # Extract the last line or key answer portion
    lines = raw_answer.strip().split("\n")

    # Attempt to extract answer from output
    answer_text = raw_answer.strip()

    if question_type == "single_choice":
        # Look for a single-letter answer (A, B, C, D)
        import re
        # Prefer "Answer: X" or "The answer is X" style patterns
        match = re.search(r'(?:answer)[:\s]*([A-D])', answer_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Look for a standalone letter
        match = re.search(r'\b([A-D])\b', answer_text)
        if match:
            return match.group(1).upper()
        return answer_text[:1].upper() if answer_text else None

    elif question_type == "multiple_choice":
        # Look for multiple letter answers
        import re
        # Find all uppercase letters
        matches = re.findall(r'\b([A-D])\b', answer_text.upper())
        if matches:
            return sorted(list(set(matches)))
        return None

    elif question_type == "true_false":
        # Look for True/False
        lower_text = answer_text.lower()
        if "true" in lower_text and "false" not in lower_text:
            return True
        elif "false" in lower_text and "true" not in lower_text:
            return False
        # Also check for Yes/No
        if "yes" in lower_text:
            return True
        if "no" in lower_text:
            return False
        return None

    elif question_type == "short_answer":
        # Retrieve relevant enum values
        relevant_enums = get_relevant_enums(structure, finding, None)

        # Try to extract enum values from the answer
        lower_text = answer_text.lower()

        # Prefer the last line (typically the final answer)
        last_line = lines[-1].strip().lower() if lines else ""

        # Check if the last line contains an enum value
        for enum_val in relevant_enums:
            if enum_val in last_line:
                return enum_val

        # If not in the last line, search the entire text
        for enum_val in relevant_enums:
            if enum_val in lower_text:
                return enum_val

        # If no enum value found, return the last line or cleaned-up text.
        # Try to extract a snake_case answer.
        import re
        snake_case_match = re.search(r'\b([a-z]+(?:_[a-z]+)*)\b', last_line)
        if snake_case_match:
            return snake_case_match.group(1)

        return last_line if last_line else answer_text

    return answer_text


if __name__ == "__main__":
    main()
