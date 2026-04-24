"""
Answer mode: after the agent generates a report, run single-question inference
for each VQA item in the patient directory and write results to file.

Key constraints (user requirements):
- When the options contain tooth numbers (FDI, e.g. #36/#46/36/46), each
  candidate tooth number must be independently verified:
  - Use a quadrant/tooth crop image, or
  - Use DentalGPT/OralGPT on a bbox-overlay OPG image,
  in order to reduce tooth-position confusion and pick the correct option
  more reliably.

Tool selection rules (VLM relative strengths):
- When the question involves NONE of finding/structure/feature → call OralGPT
- GPT-5.2 (llm_zoo_openai) is stronger on: short answer; structure — periodontium, tmj, sinuses; finding — bone_loss; feature — bone_loss_pattern, findings, severity
- Gemini (llm_zoo_google) is stronger on: multiple choice, true/false; feature — relationship

Prompt design references Benchmark/Experiments/inference/run_vqa_task.py:
short_answer questions inject the full enum list.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Normalized output for single-choice (used to validate after "non-conforming answer → LLM + image re-answer")
VALID_SINGLE_CHOICE = frozenset({"A", "B", "C", "D"})


class SingleChoiceAnswer(BaseModel):
    """Single-choice normalized format: only A/B/C/D allowed."""
    answer: Literal["A", "B", "C", "D"] = Field(description="Exactly one letter: A, B, C, or D.")

# VQA file names supported in the patient directory
VQA_FILE_NAMES = ["vqa.json"]

# Enum values allowed for short_answer (aligned with Schema&Enum_standard Part II, injected into the prompt)
SHORT_ANSWER_ENUM_VALUES = [
    # teeth.status / presence
    "present", "missing", "unerupted", "erupted", "impacted", "residual_root",
    "implant", "root_filled", "supernumerary", "exfoliating",
    # teeth.winters_class
    "vertical", "angled", "horizontal",
    # teeth.icdas_code / caries
    "early", "moderate", "advanced",
    # teeth.pai_score / apical
    "normal", "mild_change", "moderate_change", "severe_change",
    # teeth.relationship
    "approximates_iac", "approximates_sinus",
    # teeth.root_anomaly, crown_anomaly, position_anomaly, caries_location, restoration_issue
    "abnormal_morphology", "root_rounding", "developmental_anomaly",
    "transposed", "insufficient_space", "ectopic_eruption", "recurrent", "root_surface", "defective",
    # periodontium
    "mild", "severe", "calculus", "pericoronitis",
    # tmj
    "degenerative", "developmental",
    # jaws
    "sclerotic_lesion", "lucent_lesion", "bone_variant", "osteopenia",
    "marrow_space_prominence", "surgical_hardware",
    # sinuses
    "mucosal_change", "opacification", "air_fluid_level",
    # apical_status
    "scar", "active_pathology",
]


def get_short_answer_enum_values() -> List[str]:
    """Return the list of allowed enum values for short_answer (aligned with run_vqa_task / Schema&Enum_standard)."""
    return list(SHORT_ANSWER_ENUM_VALUES)


def select_vqa_tool(
    question: str,
    q_type: str,
    structure: Optional[str] = None,
    finding: Optional[str] = None,
    feature: Optional[str] = None,
) -> str:
    """
    Select the model used to produce the final answer (simplified version that
    satisfies the user's constraints).

    Notes:
    - We no longer route tool priority by question/finding/structure/feature.
    - Default: first call DentalGPT + OralGPT to obtain a neutral, short context;
      then the "final answer model" produces the conclusion based on that
      context plus the report.

    Rules:
    - short_answer → GPT-5.2 (openai)
    - other question types (single/multiple/true_false) → Gemini (google)
    """
    _ = (question, structure, finding, feature)  # Kept for signature compatibility; not used for decisions
    return "openai" if q_type == "short_answer" else "google"


def _build_neutral_context_prompt(item: Dict[str, Any]) -> str:
    """
    Build a neutral, non-leading context request for DentalGPT / OralGPT:
    - Do not inject other tool / model results
    - Require an extremely concise output, easy for a downstream model to integrate
    """
    q = (item.get("question") or "").strip()
    q_type = (item.get("question_type") or "").strip()
    options = item.get("options") or []

    parts = [
        "You are an oral radiology expert. Analyze the given OPG image independently and objectively, and answer the question.",
        "Requirements: do not guess, do not use leading language, keep the output as short as possible.",
        "Forbidden: do not output the word \"error\" or any equivalent expression.",
        "",
        f"Question: {q}",
    ]
    if options:
        parts.append("")
        parts.append("Options:")
        for opt in options:
            parts.append(f"- {opt}")

    parts.append("")
    if q_type == "short_answer":
        parts.extend([
            "Output format:",
            "1) First line: the shortest possible answer (a phrase / an enum value / an FDI number, etc.)",
            "2) Second line: one-sentence justification (optional, keep it short)",
        ])
    elif q_type == "multiple_choice":
        parts.extend([
            "Output format:",
            "1) First line: only option letters, comma-separated (e.g. A, C)",
            "2) Second line: one-sentence justification (optional, keep it short)",
        ])
    elif q_type == "true_false":
        parts.extend([
            "Output format:",
            "1) First line: only True or False",
            "2) Second line: one-sentence justification (optional, keep it short)",
        ])
    else:
        parts.extend([
            "Output format:",
            "1) First line: only the option letter (e.g. A)",
            "2) Second line: one-sentence justification (optional, keep it short)",
        ])

    return "\n".join(parts)


def load_vqa_questions(patient_dir: Path, vqa_file_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load VQA questions from a patient directory (supports vqa.json / vqa_update.json).
    vqa_file_name: if provided, load only this file; otherwise search through VQA_FILE_NAMES in order.
    Returns a list where each item contains question, question_type, options (optional),
    structure, finding, feature (optional).
    """
    vqa_path = None
    if vqa_file_name:
        p = patient_dir / vqa_file_name
        if p.exists():
            vqa_path = p
    else:
        for name in VQA_FILE_NAMES:
            p = patient_dir / name
            if p.exists():
                vqa_path = p
                break
    if not vqa_path:
        return []

    with open(vqa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[Dict[str, Any]] = []
    # single_choice
    for i, item in enumerate(data.get("single_choice", [])):
        out.append({
            "question_id": f"sc_{i}",
            "question_type": "single_choice",
            "question": item["question"],
            "options": item.get("options"),
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })
    # multiple_choice
    for i, item in enumerate(data.get("multiple_choice", [])):
        out.append({
            "question_id": f"mc_{i}",
            "question_type": "multiple_choice",
            "question": item["question"],
            "options": item.get("options"),
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })
    # true_false
    for i, item in enumerate(data.get("true_false", [])):
        out.append({
            "question_id": f"tf_{i}",
            "question_type": "true_false",
            "question": item["question"],
            "options": ["True", "False"],
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })
    # short_answer
    for i, item in enumerate(data.get("short_answer", [])):
        out.append({
            "question_id": f"sa_{i}",
            "question_type": "short_answer",
            "question": item["question"],
            "options": None,
            "structure": item.get("structure"),
            "finding": item.get("finding"),
            "feature": item.get("feature"),
        })
    return out


def _build_gold_context_from_detections(detections_json: Optional[str]) -> str:
    """
    Parse tooth count (gold standard) and FDI positions (strong evidence) from
    the run_all_detections JSON; returns text suitable as sub-context, or an
    empty string on failure.
    FDI is strong evidence; when comparing with OralGPT output: if the positions
    are adjacent, defer to OralGPT, otherwise defer to this strong evidence.
    """
    if not detections_json or not detections_json.strip():
        return ""
    try:
        det = json.loads(detections_json)
    except json.JSONDecodeError:
        return ""
    lines: List[str] = []
    teeth_data = det.get("teeth") or {}
    if isinstance(teeth_data, dict) and "error" not in teeth_data:
        stats = teeth_data.get("statistics") or {}
        total = stats.get("total_teeth")
        by_type = stats.get("tooth_count_by_type") or {}
        if total is None:
            dets = teeth_data.get("detections") or []
            total = len(dets)
            if not by_type and dets:
                for d in dets:
                    cls_name = str(d.get("class_name") or d.get("class") or "")
                    if cls_name:
                        by_type[cls_name] = by_type.get(cls_name, 0) + 1
        if total is not None:
            lines.append(f"- Tooth count (gold standard): total {total}")
            if by_type:
                type_desc = "; ".join(f"type {k}: {v}" for k, v in sorted(by_type.items()))
                lines.append(f"- By type: {type_desc}")
            if "8" in by_type:
                lines.append(f"- Wisdom teeth (third molar, type 8): {by_type['8']}")
    teeth_fdi = det.get("teeth_fdi")
    if isinstance(teeth_fdi, dict) and "error" not in teeth_fdi:
        fdi_list = sorted(teeth_fdi.keys(), key=lambda x: (int(x) // 10, int(x) % 10))
        lines.append("- FDI positions (strong evidence, from detection + quadrant mapping):")
        lines.append("  " + ", ".join(fdi_list))
        lines.append("- When comparing against OralGPT: if the cited position is adjacent, defer to OralGPT; otherwise defer to this FDI strong evidence.")
    if not lines:
        return ""
    return "\n".join(lines)


def _is_bone_loss_related(item: Dict[str, Any]) -> bool:
    """Determine whether the question relates to bone loss (used for weighting: OralGPT > DentalGPT)."""
    q = (item.get("question") or "").lower()
    finding = (item.get("finding") or "").lower()
    if "bone" in finding or "bone_loss" in finding or "bone loss" in q or "bone loss" in finding:
        return True
    return False


def _is_impaction_related(item: Dict[str, Any]) -> bool:
    """Determine whether the question relates to impaction (used to inject impaction-vs-gold-standard consistency guidance)."""
    q = (item.get("question") or "").lower()
    return "impacted" in q or "impaction" in q


def _classify_image_fact_with_gpt4o_mini(
    toolkit: Any,
    image_path: str,
    question: str,
    q_type: str,
    config: Optional[Any],
) -> bool:
    """
    Use gpt-4o-mini to make a simple "is this asking about image visibility" judgment.
    Binary yes/no, low max_tokens, asks the model to return only yes/no.
    Called only for true_false / single_choice / multiple_choice; returns False on failure.
    """
    if q_type not in ("true_false", "single_choice", "multiple_choice"):
        return False
    prompt = (
        "You are a classifier. Only the question text below matters.\n\n"
        "Question: {q}\n\n"
        "Does this question ask whether something is visible or present on the OPG image "
        "(e.g. mucosal thickening, a finding, a sign, an abnormality visible on the radiograph)?\n\n"
        "Answer with exactly one word: yes or no."
    ).format(q=(question or "").strip())
    return _classify_yes_no_with_gpt4o_mini(
        toolkit, image_path, prompt, config, log_label="image-fact classifier"
    )


def _classify_yes_no_with_gpt4o_mini(
    toolkit: Any,
    image_path: str,
    prompt: str,
    config: Optional[Any],
    log_label: str = "gpt4o-mini",
) -> bool:
    """
    Generic helper: use gpt-4o-mini to perform a yes/no classification, asking
    the model to return only yes/no. Parses the first token of the first line;
    yes → True, anything else (including failure) → False.
    """
    try:
        result = toolkit.call_gpt4o_mini_binary(
            image_path=image_path,
            prompt=prompt,
            config=config,
        )
        if result.get("status") != "success":
            return False
        text = (result.get("response") or "").strip().lower()
        if not text:
            return False
        first = text.splitlines()[0].strip() if text.splitlines() else text
        first_word = (first.split() or [first])[0].strip(".,;:") if first else ""
        return first_word in ("yes",)
    except Exception as e:
        logger.debug("%s binary classification failed, defaulting to no: %s", log_label, e)
        return False


def _classify_anatomical_position_with_gpt4o_mini(
    toolkit: Any,
    image_path: str,
    question: str,
    config: Optional[Any],
) -> bool:
    """
    Use gpt-4o-mini to decide whether the question asks about "anatomical position"
    (i.e. whether one tooth occupies another tooth's position) rather than merely
    whether a given FDI number is present.
    """
    prompt = (
        "You are a classifier. Only the question text below matters.\n\n"
        "Question: {q}\n\n"
        "Does this question ask about anatomical position in the arch (e.g. one tooth has drifted into or "
        "occupies the space/position of another tooth), rather than merely which FDI number is present or missing?\n\n"
        "Answer with exactly one word: yes or no."
    ).format(q=(question or "").strip())
    return _classify_yes_no_with_gpt4o_mini(
        toolkit, image_path, prompt, config, log_label="anatomical-position classifier"
    )


def _classify_lucency_caries_with_gpt4o_mini(
    toolkit: Any,
    image_path: str,
    question: str,
    config: Optional[Any],
) -> bool:
    """
    Use gpt-4o-mini to decide whether the question asks "which teeth have
    radiolucencies/caries or morphology-related lucencies", where we must
    distinguish pathology from normal anatomy (e.g. pulp chambers).
    """
    prompt = (
        "You are a classifier. Only the question text below matters.\n\n"
        "Question: {q}\n\n"
        "Does this question ask which teeth show radiolucencies that may reflect caries or morphology, "
        "where we must exclude normal anatomy like pulp chambers?\n\n"
        "Answer with exactly one word: yes or no."
    ).format(q=(question or "").strip())
    return _classify_yes_no_with_gpt4o_mini(
        toolkit, image_path, prompt, config, log_label="lucency/caries classifier"
    )


def _build_vqa_prompt(
    report: str,
    item: Dict[str, Any],
    is_image_fact: Optional[bool] = None,
    is_anatomical_position: Optional[bool] = None,
    is_lucency_caries: Optional[bool] = None,
) -> str:
    """
    Build a per-question VQA prompt: includes the report and the question,
    requires the answer on the first line followed by an explanation.
    short_answer questions inject the full enum list (see run_vqa_task's SYSTEM_PROMPT).
    5.1/5.3/5.4: each prompt flag is produced by a Gemini binary classifier,
    no longer via keyword lists.
    """
    q = item["question"]
    q_type = item["question_type"]
    parts = [
        "Based on the following diagnostic report and the OPG image, answer this question.",
        "",
        "## Report",
        report.strip(),
        "",
        "## Question",
        q,
    ]
    if item.get("options"):
        parts.append("")
        parts.append("Options:")
        for opt in item["options"]:
            parts.append(f"  {opt}")
    parts.extend([
        "",
        "CRITICAL RULES:",
        "- You MUST NOT output the word \"error\" in any form.",
        "- Always provide the best possible answer in the required format, even if uncertain.",
        "",
    ])
    # 5.1: Gemini binary classifier produces is_image_fact; if True, inject "use image findings as primary evidence"
    if is_image_fact:
        parts.extend([
            "IMAGE-BASED QUESTION: If the question asks whether a finding is visible or present on the OPG image, "
            "use the image as the primary evidence. The report may be conservative and not list subtle findings; "
            "absence of confirmation in the report does not mean the finding is absent on the image.",
            "",
        ])
    # 5.3: Gemini binary classifier produces is_anatomical_position
    if is_anatomical_position:
        parts.extend([
            "ANATOMICAL POSITION: Here 'position' refers to anatomical position in the arch (the space a tooth occupies), "
            "not FDI number. If one tooth has drifted into or occupies the space of another tooth, answer according to "
            "anatomical relationship, not merely whether an FDI number is present or missing.",
            "",
        ])
    # 5.4: Gemini binary classifier produces is_lucency_caries; only injected for multiple_choice
    if q_type == "multiple_choice" and is_lucency_caries:
        parts.extend([
            "LUCENCIES/CARIES: Include only teeth where the radiolucency suggests caries or abnormal morphology. "
            "Exclude normal anatomical radiolucencies such as pulp chambers and root canals.",
            "",
        ])
    parts.extend([
        "IMPORTANT: Your response must have two parts:",
        "1. First line: direct answer only (e.g. option letter like A/B/C, or True/False, or one short phrase).",
        "2. Then a blank line, then your brief explanation.",
        "",
        "Format requirements by question type:",
        "- single_choice: output ONLY one letter among A/B/C/D on the first line.",
        "- multiple_choice: output ONLY letters among A/B/C/D separated by comma on the first line (e.g., A, C).",
        "- true_false: output ONLY True or False on the first line.",
        "- short_answer: output ONLY one concise phrase/value on the first line.",
        "",
    ])
    if q_type == "short_answer":
        enum_values = get_short_answer_enum_values()
        enum_str = ", ".join(enum_values)
        parts.extend([
            "Short answer rule: You MUST use ONLY these valid enum values (exact, lowercase with underscores):",
            enum_str,
            "",
        ])
        # 5.2: severity and enum calibration — provide concise definitions for easily confused enums
        parts.extend(_get_short_answer_enum_definitions(item))
        parts.extend([
            "You MUST NOT output the word \"error\" for short answer; always pick the best matching value from the list above.",
            "",
        ])
    return "\n".join(parts)


def _get_short_answer_enum_definitions(item: Dict[str, Any]) -> List[str]:
    """
    5.2: provide concise definitions for easily confused enums in short_answer,
    aligned with the Benchmark and OralGPT wording.
    For bone-related questions where "prefer OralGPT" already applies, the
    hints here map OralGPT's description to the same severity bucket.
    """
    q = (item.get("question") or "").lower()
    lines = []
    if "bone loss" in q or "bone level" in q or "periodontal" in q or "alveolar" in q:
        lines.extend([
            "Bone loss / periodontal severity (use when question asks about degree of bone loss):",
            "- normal: crestal bone ~1–2 mm below CEJ, no significant loss.",
            "- mild: bone loss limited to cervical third of root.",
            "- moderate: crestal bone typically ~3–4 mm apical to CEJs (generalized horizontal loss).",
            "- advanced / severe: bone loss to middle or apical third of roots.",
            "If OralGPT describes ~3–4 mm bone loss from CEJ, choose moderate.",
            "",
        ])
    if "bite" in q or "occlusion" in q:
        lines.extend([
            "Bite / occlusion:",
            "- normal: anterior and posterior teeth meet in typical relationship.",
            "- open_bite: vertical gap between maxillary and mandibular anterior teeth.",
            "- deep overbite, crossbite, underbite: use as per standard definitions.",
            "",
        ])
    if "periodontal" in q and ("status" in q or "level" in q):
        lines.extend([
            "Periodontal status: normal = healthy height; mild_change / moderate_change / severe_change = increasing bone loss from CEJ.",
            "",
        ])
    return lines


def _parse_first_line_answer(raw: str) -> str:
    """Extract the first line of the model's output as the answer."""
    if not raw or not isinstance(raw, str):
        return ""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    return lines[0] if lines else raw.strip()[:200]


def _normalize_final_answer(raw_text: str, q_type: str) -> str:
    """
    Normalize the model's final output to the required format for each
    question type, and make sure "error" never appears.

    Targets:
    - single_choice: A/B/C/D
    - multiple_choice: A, C (deduplicated in order of appearance)
    - true_false: True/False
    - short_answer: first-line phrase (whitespace trimmed)
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    # Uniformly forbid "error" (any case)
    if text.strip().lower() == "error":
        text = ""

    q_type = (q_type or "").strip()
    upper = text.upper()

    if q_type == "single_choice":
        m = re.search(r"\b([A-D])\b", upper)
        return m.group(1) if m else ""

    if q_type == "multiple_choice":
        letters = re.findall(r"\b([A-D])\b", upper)
        seen = set()
        out = []
        for x in letters:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return ", ".join(out)

    if q_type == "true_false":
        if re.search(r"\bTRUE\b", upper):
            return "True"
        if re.search(r"\bFALSE\b", upper):
            return "False"
        return ""

    # short_answer / unknown: take the first non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    return "" if first.lower() == "error" else first


_FDI_RE = re.compile(r"(?:#\s*)?([1-4][1-8])\b")
_OPT_LETTER_RE = re.compile(r"^\s*([A-D])\s*[.\)]\s*", re.IGNORECASE)


def _extract_option_letter(option_text: str) -> str:
    """Extract A/B/C/D from option text (e.g. 'A. #36'). Returns empty string if not found."""
    if not option_text:
        return ""
    m = _OPT_LETTER_RE.search(option_text.strip())
    return m.group(1).upper() if m else ""


def _extract_fdi_from_text(text: str) -> str:
    """Extract an FDI tooth number (two digits 11-48) from text. Returns empty string if not found."""
    if not text:
        return ""
    m = _FDI_RE.search(text)
    return m.group(1) if m else ""


def _fdi_to_quadrant(fdi: str) -> str:
    """FDI prefix to quadrant: 1->Q1, 2->Q2, 3->Q3, 4->Q4."""
    if not fdi or len(fdi) < 2:
        return ""
    return {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}.get(fdi[0], "")


def _are_fdi_adjacent(fdi1: str, fdi2: str) -> bool:
    """Check whether two FDI positions are adjacent (same quadrant, unit digits differ by 1)."""
    if not fdi1 or not fdi2 or len(fdi1) < 2 or len(fdi2) < 2:
        return False
    if fdi1[0] != fdi2[0]:
        return False
    try:
        p1, p2 = int(fdi1[1]), int(fdi2[1])
        return abs(p1 - p2) == 1
    except (ValueError, IndexError):
        return False


def _normalize_yes_no_unsure(raw: str) -> str:
    """
    Normalize the model output to one of YES / NO / UNSURE.
    Use case: strictly for "candidate tooth number verification" — avoids being
    derailed by long free-form text.
    """
    if not raw:
        return "UNSURE"
    s = raw.strip().upper()
    first_line = s.splitlines()[0].strip() if s.splitlines() else s
    first_token = first_line.split()[0].strip(" ,.;:()[]{}\\\"'") if first_line else ""
    if first_token in {"YES", "NO", "UNSURE"}:
        return first_token
    if "UNSURE" in s:
        return "UNSURE"
    if "YES" in s:
        return "YES"
    if "NO" in s:
        return "NO"
    return "UNSURE"


def _unwrap_vlm_text(raw: Any) -> str:
    """Handle the case where a tool returns a JSON string; prefer analysis/response fields."""
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return str(obj.get("analysis") or obj.get("response") or obj.get("answer") or raw)
    except json.JSONDecodeError:
        pass
    return raw


def _normalize_single_choice_with_llm(
    toolkit: Any,
    image_path: str,
    question: str,
    options: List[str],
    raw_answer: str,
    config: Optional[Any],
) -> str:
    """
    Answer-mode follow-up: when a single-choice answer is not in A/B/C/D, re-ask
    the agent's LLM with the image and use Pydantic to enforce the format.
    Returns the normalized letter (A/B/C/D); on failure, returns the first
    matching letter found in raw_answer or defaults to 'A'.
    """
    prompt_lines = [
        "You are a dental VQA answer normalizer. Based on the OPG image, the question, and the options below,",
        "the previous model output (possibly ill-formatted) is given. Output exactly one letter: A, B, C, or D.",
        "",
        f"Question: {question}",
        "",
        "Options:",
    ]
    for opt in options:
        prompt_lines.append(f"  {opt}")
    prompt_lines.extend([
        "",
        f"Previous model output: {raw_answer!r}",
        "",
        "Output only the single letter (A, B, C, or D) that best matches the intended answer. No explanation.",
    ])
    prompt = "\n".join(prompt_lines)
    try:
        raw = toolkit.llm_zoo_openai(
            image_path=image_path,
            custom_prompt=prompt,
            analysis_level="overall",
            config=config,
        )
        text = _unwrap_vlm_text(raw).strip() if raw else ""
        # Extract the first A/B/C/D from the response
        match = re.search(r"\b([A-D])\b", (text or "").upper())
        letter = match.group(1) if match else None
        if letter:
            parsed = SingleChoiceAnswer.model_validate({"answer": letter})
            return parsed.answer
    except Exception as e:
        logger.warning("Single-choice LLM normalization failed: %s", e)
    for c in "ABCD":
        if c in (raw_answer or "").upper():
            return c
    return "A"


def _verify_tooth_option_with_vlm(
    toolkit: Any,
    image_path: str,
    question: str,
    option_letter: str,
    fdi: str,
    config: Optional[Any],
    detections_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Per-tooth-number verification (hard requirement from the user):
    - Generate a "tooth bbox-overlay OPG" and a "quadrant bbox-overlay OPG"
    - Call OralGPT and DentalGPT respectively to return YES/NO/UNSURE
    """
    from pathlib import Path as _Path

    quadrant = _fdi_to_quadrant(fdi)
    tooth_overlay_path = ""
    quadrant_overlay_path = ""
    try:
        tooth_overlay = toolkit.get_annotated_image(
            image_path=image_path,
            target_type="tooth",
            target_id=fdi,
            output_mode="bbox_overlay",
            detections_json=detections_json,
            config=config,
        )
        tooth_obj = json.loads(tooth_overlay) if isinstance(tooth_overlay, str) else {}
        tooth_overlay_path = (tooth_obj.get("image_path") or "").strip()

        if quadrant:
            quad_overlay = toolkit.get_annotated_image(
                image_path=image_path,
                target_type="quadrant",
                target_id=quadrant,
                output_mode="bbox_overlay",
                detections_json=detections_json,
                config=config,
            )
            quad_obj = json.loads(quad_overlay) if isinstance(quad_overlay, str) else {}
            quadrant_overlay_path = (quad_obj.get("image_path") or "").strip()

        oral_text = ""
        dental_text = ""

        if tooth_overlay_path:
            oral_prompt = (
                "You are performing per-tooth-number VQA option verification.\n"
                f"- Candidate option: {option_letter}\n"
                f"- Candidate tooth number (FDI): {fdi}\n"
                f"- Quadrant: {quadrant or 'unknown'}\n\n"
                "The image is an OPG (panoramic radiograph). A red box marks the candidate tooth region.\n"
                "Based only on the red-boxed region, decide whether it matches the target described in the question below (e.g. caries / periapical lesion / filling / missing, etc.).\n\n"
                f"Question: {question}\n\n"
                "Output requirement: output exactly one word (no explanation): YES / NO / UNSURE"
            )
            oral_raw = toolkit.oral_expert_analysis(
                image_path=tooth_overlay_path,
                analysis_type="custom",
                custom_prompt=oral_prompt,
                config=config,
            )
            oral_text = _unwrap_vlm_text(oral_raw)

        if quadrant_overlay_path:
            dental_prompt = (
                "You are performing per-tooth-number VQA option verification (quadrant consistency check).\n"
                f"- Candidate option: {option_letter}\n"
                f"- Candidate tooth number (FDI): {fdi}\n"
                f"- Quadrant (red box): {quadrant}\n\n"
                "The image is an OPG (panoramic radiograph). A red box marks the quadrant region.\n"
                "Decide whether the tooth position corresponding to this FDI exists within the quadrant, and whether it is plausibly consistent with the target described in the question.\n\n"
                f"Question: {question}\n\n"
                "Output requirement: output exactly one word (no explanation): YES / NO / UNSURE"
            )
            dental_raw = toolkit.dental_expert_analysis(
                image_path=quadrant_overlay_path,
                analysis_type="custom",
                custom_prompt=dental_prompt,
                config=config,
            )
            dental_text = _unwrap_vlm_text(dental_raw)

        return {
            "option_letter": option_letter,
            "fdi": fdi,
            "quadrant": quadrant,
            "oral_verdict": _normalize_yes_no_unsure(oral_text),
            "dental_verdict": _normalize_yes_no_unsure(dental_text),
            "oral_raw": (oral_text or "")[:800],
            "dental_raw": (dental_text or "")[:800],
        }
    finally:
        # Clean up temp files created by get_annotated_image to avoid leaving garbage in /tmp
        for p in (tooth_overlay_path, quadrant_overlay_path):
            try:
                if p and _Path(p).exists():
                    _Path(p).unlink()
            except Exception:
                pass


def _run_one_vqa(
    toolkit: Any,
    image_path: str,
    report: str,
    item: Dict[str, Any],
    config: Optional[Any],
    detections_json: Optional[str] = None,
    gold_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference on a single question. Returns {question_id, question, answer_first_line, explanation, tool, error}."""
    question = item["question"]
    q_type = item["question_type"]
    options = item.get("options") or []

    # Pseudocode (key logic, satisfying the user's constraints):
    # - If options contain FDI tooth numbers:
    #   - Generate tooth/quadrant bbox-overlay images for each candidate FDI
    #   - Call OralGPT / DentalGPT separately to verify (YES/NO/UNSURE)
    #   - If exactly one YES → directly output that option letter
    #   - Otherwise inject the verification summary into the final prompt
    # - Default (all question types):
    #   - First call DentalGPT + OralGPT to obtain a neutral, short context
    #   - Final answer: GPT-5.2 for short_answer; Gemini for other types

    tooth_option_map: Dict[str, str] = {}
    if isinstance(options, list) and options:
        for i, opt in enumerate(options):
            opt_str = str(opt)
            letter = _extract_option_letter(opt_str)
            if not letter and 0 <= i < 4:
                # Handle the case where options do not carry an "A./B." prefix
                letter = chr(ord("A") + i)
            fdi = _extract_fdi_from_text(opt_str)
            if letter and fdi and letter not in tooth_option_map:
                tooth_option_map[letter] = fdi

    tooth_verification: List[Dict[str, Any]] = []
    forced_answer_letter = ""
    if tooth_option_map:
        for letter in sorted(tooth_option_map.keys()):
            tooth_verification.append(
                _verify_tooth_option_with_vlm(
                    toolkit=toolkit,
                    image_path=image_path,
                    question=question,
                    option_letter=letter,
                    fdi=tooth_option_map[letter],
                    config=config,
                    detections_json=detections_json,
                )
            )
        yes_letters = [
            v.get("option_letter", "")
            for v in tooth_verification
            if v.get("oral_verdict") == "YES" and v.get("dental_verdict") != "NO"
        ]
        yes_letters = [x for x in yes_letters if x]
        if len(yes_letters) == 1:
            forced_answer_letter = yes_letters[0]

    tool_name = select_vqa_tool(
        question,
        q_type,
        structure=item.get("structure"),
        finding=item.get("finding"),
        feature=item.get("feature"),
    )
    # 5.1/5.3/5.4: use gpt-4o-mini for each yes/no classification rather than keyword lists
    is_image_fact = _classify_image_fact_with_gpt4o_mini(toolkit, image_path, question, q_type, config)
    is_anatomical_position = _classify_anatomical_position_with_gpt4o_mini(toolkit, image_path, question, config)
    is_lucency_caries = _classify_lucency_caries_with_gpt4o_mini(toolkit, image_path, question, config)
    prompt = _build_vqa_prompt(
        report,
        item,
        is_image_fact=is_image_fact,
        is_anatomical_position=is_anatomical_position,
        is_lucency_caries=is_lucency_caries,
    )
    if gold_context:
        prompt += "\n\n## Tooth count and FDI reference (gold standard / strong evidence)\n"
        prompt += "Tooth count is gold standard; FDI positions are strong evidence. You may compare with the OralGPT context: if the cited position is adjacent, defer to OralGPT, otherwise defer to this FDI strong evidence.\n"
        prompt += gold_context
        prompt += "\n\nUse the above reference together with the Report and image to give your final answer."
        # Impaction consistency: teeth listed as not_detected (missing) in the gold standard cannot be chosen as impacted
        if _is_impaction_related(item):
            prompt += "\n\nIMPACTION CONSISTENCY: A tooth listed as not detected (missing) in the gold standard FDI reference cannot be impacted. Only include teeth that are present in the dentition in your impacted list or selection."
    if tooth_verification:
        prompt += "\n\n## Per-tooth-number verification (mandatory)\n"
        prompt += "Below are the verification results for each candidate tooth number (YES/NO/UNSURE). Prioritize these verifications when deciding the final option.\n"
        prompt += json.dumps(tooth_verification, ensure_ascii=False, indent=2)
        prompt += "\n\nOutput format is unchanged: first line must contain only the final answer."

    try:
        # If per-tooth-number verification already yielded a unique conclusion, skip the second main-model call
        if forced_answer_letter:
            return {
                "question_id": item["question_id"],
                "question": question,
                "answer_first_line": forced_answer_letter,
                "explanation": json.dumps(
                    {
                        "reason": "Per-tooth-number verification produced a unique YES; answer determined directly.",
                        "tooth_verification": tooth_verification,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "tool": "tooth_option_verification",
                "error": None,
            }

        # First call DentalGPT + OralGPT to produce a neutral, short context
        neutral_context_prompt = _build_neutral_context_prompt(item)
        dental_ctx_raw = toolkit.dental_expert_analysis(
            image_path=image_path,
            analysis_type="overall",
            custom_prompt=neutral_context_prompt,
            config=config,
        )
        oral_ctx_raw = toolkit.oral_expert_analysis(
            image_path=image_path,
            analysis_type="overall",
            custom_prompt=neutral_context_prompt,
            config=config,
        )
        dental_ctx = _unwrap_vlm_text(dental_ctx_raw).strip()
        oral_ctx = _unwrap_vlm_text(oral_ctx_raw).strip()

        prompt += "\n\n## DentalGPT context (neutral, short)\n"
        prompt += dental_ctx if dental_ctx else "(empty)"
        prompt += "\n\n## OralGPT context (neutral, short)\n"
        prompt += oral_ctx if oral_ctx else "(empty)"
        if _is_bone_loss_related(item):
            prompt += "\n\nNote: this question relates to bone loss; OralGPT's opinion takes priority over DentalGPT's. Defer to OralGPT's conclusion."
        prompt += "\n\nUsing the Report and the two contexts above, give your final answer."

        # Final answer model: short_answer -> openai, others -> google
        if tool_name == "openai":
            raw = toolkit.llm_zoo_openai(
                image_path=image_path,
                custom_prompt=prompt,
                analysis_level="overall",
                config=config,
            )
            if isinstance(raw, str) and raw:
                try:
                    obj = json.loads(raw)
                    raw_text = obj.get("response", obj.get("analysis", raw))
                except json.JSONDecodeError:
                    raw_text = raw
            else:
                raw_text = str(raw)
        else:  # google
            raw = toolkit.llm_zoo_google(
                image_path=image_path,
                custom_prompt=prompt,
                analysis_level="overall",
                config=config,
            )
            if isinstance(raw, str) and raw:
                try:
                    obj = json.loads(raw)
                    raw_text = obj.get("response", obj.get("analysis", raw))
                except json.JSONDecodeError:
                    raw_text = raw
            else:
                raw_text = str(raw)

        first_line = _normalize_final_answer(raw_text, q_type) or _parse_first_line_answer(raw_text)
        explanation = raw_text.strip()
        if first_line and explanation.startswith(first_line):
            explanation = explanation[len(first_line):].strip().lstrip("\n")
        return {
            "question_id": item["question_id"],
            "question": question,
            "answer_first_line": first_line,
            "explanation": explanation,
            "tool": tool_name,
            "error": None,
        }
    except Exception as e:
        logger.exception(f"VQA single-question failed {item.get('question_id')}: {e}")
        return {
            "question_id": item.get("question_id", ""),
            "question": question,
            "answer_first_line": "",
            "explanation": "",
            "tool": tool_name,
            "error": str(e),
        }


def run_vqa_after_report(
    toolkit: Any,
    image_path: str,
    report: str,
    patient_dir: Path,
    output_dir: Path,
    config: Optional[Any] = None,
    max_workers: int = 4,
    vqa_file_name: Optional[str] = None,
) -> None:
    """
    Run after report generation: read VQA items from the patient directory,
    run single-question inference per item, and write results to output_dir/vqa_answer.txt.
    vqa_file_name: if provided (e.g. vqa_update.json), only that file is loaded;
    otherwise search defaults like vqa.json.

    Notes:
    - No longer runs in parallel (avoids the uncertainty and poor traceability
      of "parallel subagents").
    - `max_workers` is a legacy parameter kept for API compatibility; it is no
      longer used.
    """
    questions = load_vqa_questions(patient_dir, vqa_file_name=vqa_file_name)
    if not questions:
        logger.info(f"No VQA file found in patient directory: {patient_dir}, skipping answer mode")
        return

    # Pre-run a full detection pass once to avoid re-triggering the detection pipeline per question/option
    # (important: per-tooth-number verification retrieves bboxes multiple times)
    detections_json = None
    try:
        detections_json = toolkit.run_all_detections(image_path, config=config)
    except Exception as e:
        logger.warning(f"Answer mode: run_all_detections preload failed; detection will run on demand. Error: {e}")

    gold_context = _build_gold_context_from_detections(detections_json)
    if gold_context:
        logger.info("Injected tooth count gold standard and FDI strong evidence into answer context")
    logger.info(f"Answer mode: {len(questions)} questions, single-question inference (single-question mode)")
    results: List[Dict[str, Any]] = []
    for item in questions:
        try:
            r = _run_one_vqa(
                toolkit,
                image_path,
                report,
                item,
                config,
                detections_json=detections_json,
                gold_context=gold_context,
            )
            # Answer-mode follow-up: if a single-choice answer is not A/B/C/D, re-answer with the agent's LLM + image and enforce format via Pydantic
            if item.get("question_type") == "single_choice":
                ans = (r.get("answer_first_line") or "").strip()
                if ans not in VALID_SINGLE_CHOICE:
                    normalized = _normalize_single_choice_with_llm(
                        toolkit=toolkit,
                        image_path=image_path,
                        question=item.get("question", ""),
                        options=item.get("options") or [],
                        raw_answer=ans,
                        config=config,
                    )
                    r["answer_first_line"] = normalized
                    logger.debug("Single-choice normalization: %s -> %s", ans[:50] if len(ans) > 50 else ans, normalized)
            results.append(r)
        except Exception as e:
            logger.exception(f"VQA single-question failed {item.get('question_id')}: {e}")
            results.append({
                "question_id": item.get("question_id", ""),
                "question": item.get("question", ""),
                "answer_first_line": "",
                "explanation": "",
                "tool": "unknown",
                "error": str(e),
            })

    # Sort by original question order
    qid_order = {q["question_id"]: i for i, q in enumerate(questions)}
    results.sort(key=lambda r: qid_order.get(r["question_id"], 999))

    # Write vqa_answer.txt: per-question Q + A
    out_path = output_dir / "vqa_answer.txt"
    lines = []
    for r in results:
        lines.append(f"Q: {r['question']}")
        lines.append(f"A: {r['answer_first_line']}")
        if r.get("explanation"):
            lines.append(f"Explanation: {r['explanation']}")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"VQA answers written to: {out_path}")
