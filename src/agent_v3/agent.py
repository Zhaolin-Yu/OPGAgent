"""
ReAct agent built on LangChain create_agent
"""

import os
import base64
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .memory import AgentMemory
from .tools import create_dental_tools
from .tools.dental_tools import current_image_path_ctx

logger = logging.getLogger(__name__)


def _message_content_to_str(content: Any) -> str:
    """Normalize AIMessage.content to a string (may be str or a list, e.g. Gemini returns [{\"type\":\"text\",\"text\":\"...\"}])."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)


class TokenUsageTracker(BaseCallbackHandler):
    """Token usage tracker (only for gpt-5.4 and gemini-3-flash)"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        # Only enabled for gpt-5.4 and gemini-3-flash
        self.enabled = "gpt-5.4" in model_name.lower() or "gemini-3-flash" in model_name.lower()
        self.token_limit = 100000  # 100k token upper limit

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record token usage when an LLM call ends"""
        if not self.enabled:
            return

        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "llm_output") and gen.llm_output:
                    usage = gen.llm_output.get("token_usage", {})
                    if usage:
                        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                        total = usage.get("total_tokens", 0) or (input_tokens + output_tokens)

                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        self.total_tokens += total
                        self.call_count += 1

                        logger.debug(
                            f"Token usage: {total} (input: {input_tokens}, output: {output_tokens}), "
                            f"cumulative: {self.total_tokens}/{self.token_limit}"
                        )

    def is_limit_exceeded(self) -> bool:
        """Check whether the token upper limit has been exceeded"""
        return self.enabled and self.total_tokens >= self.token_limit

    def get_summary(self) -> Dict[str, Any]:
        """Get a usage summary"""
        return {
            "enabled": self.enabled,
            "model": self.model_name,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "token_limit": self.token_limit if self.enabled else None,
            "limit_exceeded": self.is_limit_exceeded()
        }


def _data_url(image_path: str) -> str:
    """Convert an image to a data URL"""
    p = Path(image_path)
    ext = (p.suffix or ".png").lower().lstrip(".")
    mime = f"image/{'jpeg' if ext in {'jpg', 'jpeg'} else ext}"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_chat_model(
    provider: str,
    model: Optional[str],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Build a chat model (OpenAI-compatible or Google Gemini)

    Args:
        provider: model provider (qwen/openai/openrouter/gemini)
        model: model name
        temperature: temperature parameter
        max_tokens: max output tokens (Gemini uses max_output_tokens; OpenAI-compatible passes via model_kwargs)

    Returns:
        BaseChatModel instance
    """
    provider_norm = (provider or "qwen").strip().lower()

    if provider_norm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set; cannot call Gemini models.")
        model_name = model or os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview"
        max_out = max_tokens if max_tokens is not None else 65536
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_out,
                max_retries=2,
                timeout=60,
            )
        except ImportError as e:
            raise RuntimeError(f"Using Gemini requires installing langchain-google-genai: {e}") from e

    if provider_norm == "qwen":
        base_url = (
            os.getenv("QWEN_OPENAI_BASE_URL")
            or os.getenv("DASHSCOPE_OPENAI_BASE_URL")
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set; cannot call Qwen models.")
        model_name = model or os.getenv("QWEN_MODEL") or "qwen3-vl-235b-a22b-instruct"
    elif provider_norm == "openrouter":
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set; cannot call OpenRouter models.")
        model_name = model or "qwen/qwen3-vl-235b-a22b-thinking"
    elif provider_norm == "openai":
        base_url = os.getenv("OPENAI_BASE_URL") or None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call OpenAI models.")
        model_name = model or os.getenv("OPENAI_MODEL") or "gpt-5.4"
    else:
        raise RuntimeError(f"Unsupported model provider: {provider_norm}")

    kwargs = dict(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=300,
    )
    if max_tokens is not None:
        kwargs["model_kwargs"] = {"max_tokens": max_tokens}
    return ChatOpenAI(**kwargs)


class OPGReActAgent:
    """ReAct agent built on LangChain create_agent"""

    # Tool service config options (relative to the Agent_v3 directory)
    TOOL_SERVICE_CONFIGS = {
        "default": "src/agent_v3/config/tools_config.yaml",  # default config (legacy ports)
        "api_service": "api_service/tools_config.yaml",  # new config (ports starting at 6600)
        "docker": "docker/tools_config_docker.yaml",  # Docker network-mode config
    }

    def __init__(self, config_path: Optional[str] = None, tool_service: str = "default", local_vlm_only: bool = False):
        """
        Initialize the agent

        Args:
            config_path: config file path
            tool_service: tool service config option
                - "default": use default config (legacy 8xxx ports)
                - "api_service": use the new api_service config (ports starting at 6600)
            local_vlm_only: whether to use only local VLMs (DentalGPT + OralGPT) and skip cloud VLMs (GPT-5.4 + Gemini)
        """
        # Record the tool service config used
        self.tool_service = tool_service
        # OPG_PROFILE env wins when caller did not explicitly set local_vlm_only
        if not local_vlm_only and os.environ.get("OPG_PROFILE", "").lower() == "local":
            local_vlm_only = True
        self.local_vlm_only = local_vlm_only

        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            # Use default config
            self.config = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "temperature": 0.7,
                    "max_tokens": 8192
                },
                "react": {
                    "max_iterations": 15,
                    "stop_on_error": False
                }
            }
        
        # Load tools config (supports relative and absolute paths)
        tools_config_path = self.config.get("tools", {}).get("config_path")
        self.tools_config = {}

        # Agent_v3 project root (agent.py -> agent_v3 -> src -> Agent_v3)
        agent_v3_root = Path(__file__).parent.parent.parent

        # Determine the tools config path based on the tool_service option
        if tool_service in self.TOOL_SERVICE_CONFIGS:
            # Use a predefined config path
            service_config_path = agent_v3_root / self.TOOL_SERVICE_CONFIGS[tool_service]
            logger.info(f"Using tool service config: {tool_service} ({service_config_path})")
        else:
            # Unknown option, fall back to default
            service_config_path = agent_v3_root / self.TOOL_SERVICE_CONFIGS["default"]
            logger.warning(f"Unknown tool service config: {tool_service}, falling back to default")

        # Default tools config path (relative to agent.py)
        default_tools_config = Path(__file__).parent / "config" / "tools_config.yaml"

        # Try multiple path resolution strategies
        candidate_paths = []
        # First try the config specified by tool_service
        candidate_paths.append(service_config_path)
        # Then try the path specified in the config file
        if tools_config_path:
            candidate_paths.append(Path(tools_config_path))  # original path (absolute or relative to cwd)
            candidate_paths.append(agent_v3_root / tools_config_path)  # relative to Agent_v3 root
        # Finally fall back to the default path
        candidate_paths.append(default_tools_config)

        for p in candidate_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self.tools_config = yaml.safe_load(f)
                logger.info(f"Loaded tools config: {p}")
                break
        else:
            logger.warning(f"Tools config file not found, tried: {[str(p) for p in candidate_paths]}")

        # Initialize LLM
        llm_config = self.config.get("llm", {})
        model_name = llm_config.get("model") or "gpt-5.4"
        self.llm = build_chat_model(
            provider=llm_config.get("provider", "openai"),
            model=model_name,
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens"),
        )

        # Initialize token usage tracker (only enabled for gpt-5.4 and gemini-3-flash)
        self.token_tracker = TokenUsageTracker(model_name)

        # Ablation experiment config
        ablation = self.config.get("ablation", {}) or {}
        self.ablation_no_gpt_gemini = ablation.get("no_gpt_gemini", False) or self.local_vlm_only
        self.ablation_disable_detection_sources = ablation.get("disable_detection_sources", [])  # e.g. ["yolo_disease", "tvem_disease", "bone_loss", "anatomy"]
        self.ablation_disable_all_tools = ablation.get("disable_all_tools", False)
        # Ablation 3: with no detection tools, run_all_detections also skips all detection sources and does not call the backend
        if self.ablation_disable_all_tools and not self.ablation_disable_detection_sources:
            self.ablation_disable_detection_sources = ["yolo_disease", "tvem_disease", "bone_loss", "anatomy"]
        self.ablation_disable_tool_names = ablation.get("disable_tool_names", [])  # e.g. ["get_bone_loss_description", "extraction_risk_near_anatomy"]

        # Build the tool list and capture the toolkit reference (used to preload cache)
        # Note: run_all_detections is no longer exposed as a tool; it runs automatically at agent startup
        self.tools, self.toolkit = create_dental_tools(
            self.tools_config,
            analysis_only=False,
            return_toolkit=True,
            local_vlm_only=self.local_vlm_only,
            no_llm_zoo_openai_google=self.ablation_no_gpt_gemini,
            skip_detection_sources=self.ablation_disable_detection_sources,
            no_detection_tools=self.ablation_disable_all_tools,
            skip_tool_names=self.ablation_disable_tool_names,
        )
        
        # Build the system prompt
        self.system_prompt = self._build_system_prompt()

        # Create the agent
        self.agent = self._create_agent()

        vlm_mode = "3 VLM (no GPT/Gemini)" if self.ablation_no_gpt_gemini else "4 VLM (all)"
        logger.info(f"OPGReActAgent initialized (model: {llm_config.get('model')}, tool service: {self.tool_service}, VLM: {vlm_mode}, tool count: {len(self.tools)})")

    def _vlm_triple(self) -> str:
        """Human-readable names of the VLMs in the active profile."""
        return "GPT-5.4, DentalGPT, OralGPT" if self.local_vlm_only else "GPT-5.4, Gemini, Claude Opus 4.6"

    def _get_consensus_block(self) -> str:
        """Consensus voting rule. Both profiles: 3 VLMs + 1 Tool + 1 RAG = 5 sources, ≥3/5.
        Cloud VLMs: GPT + Gemini + Claude.  Local VLMs: GPT + DentalGPT + OralGPT."""
        rag_note = ("\n\n**RAG as a voting source**: When RAG retrieves similar cases (top similarity ≥ 0.85) "
                    "whose reports contain the same finding type (e.g. \"impacted\", \"periapical\", \"root_filled\"), "
                    "count RAG as 1 YES vote. If RAG was not called, top similarity < 0.85, or retrieved report does "
                    "not match the finding type → RAG abstains.")
        header = "### Consensus Rule (Majority Vote: ≥3 out of 5 sources)"
        sources_line = "**5 sources total**: 3 VLMs (" + self._vlm_triple() + ") + 1 Tool (detection) + 1 RAG (similar historical cases)"
        if self.local_vlm_only:
            examples = (
                "**Vote Counting Examples**:\n"
                "- Impaction: Tool=yes, GPT=yes, DentalGPT=yes, OralGPT=yes, RAG=impacted(0.88) → **5/5 YES → CONFIRMED**\n"
                "- Periapical: Tool=yes, GPT=no, DentalGPT=yes, OralGPT=no, RAG=pai_moderate(0.87) → **3/5 YES → CONFIRMED**\n"
                "- Bone loss: Tool=none, GPT=mild, DentalGPT=mild, OralGPT=mild, RAG=abstain → **3/5 YES → CONFIRMED**\n"
                "- Filling: Tool=yes, GPT=no, DentalGPT=no, OralGPT=no, RAG=abstain → **1/5 YES → OMIT**"
            )
        else:
            examples = (
                "**Vote Counting Examples**:\n"
                "- Impaction: Tool=yes, GPT=yes, Gemini=yes, Claude=yes, RAG=impacted(0.88) → **5/5 YES → CONFIRMED**\n"
                "- Periapical: Tool=yes, GPT=no, Gemini=yes, Claude=no, RAG=pai_moderate(0.87) → **3/5 YES → CONFIRMED**\n"
                "- Bone loss: Tool=none, GPT=mild, Gemini=mild, Claude=mild, RAG=abstain → **3/5 YES → CONFIRMED**\n"
                "- Filling: Tool=yes, GPT=no, Gemini=no, Claude=no, RAG=abstain → **1/5 YES → OMIT**"
            )
        return (header + "\n" + sources_line + rag_note + "\n\n"
                + "**CRITICAL**: ≥3/5 = CONFIRMED, **EVEN IF** the other sources explicitly say \"no\" or contradict!\n\n"
                + "- **Accept (≥3/5)**: If enough sources report the same finding → **CONFIRMED (majority wins)**\n"
                + "  - This applies EVEN IF remaining sources say \"none\" or \"not present\"\n"
                + "- **High Confidence (2/5)**: If exactly 2 sources agree → include with [HIGH_CONFIDENCE] label\n"
                + "- **Reject (<2/5)**: Only 1 source reports → OMIT\n\n"
                + examples)

    def _get_consensus_placeholders(self) -> dict:
        """Return consensus-related placeholders for the prompt.
        Both cloud and local profiles use 5 sources (3 VLMs + Tool + RAG), ≥3/5."""
        return {
            "CONFIRMED_THRESHOLD": "≥3/5",
            "SOURCES_DESC": "Tool + 3 VLMs + RAG",
            "VLM_PHASE_DESC": self._vlm_triple(),
            "SUMMARY_THRESHOLD": "≥3/5 = CONFIRMED, 2/5 = HIGH_CONFIDENCE",
            "CONFIRMED_LABEL": "≥3/5 sources agree",
            "HIGH_CONF_LABEL": "2/5 sources agree",
            "FINDINGS_HEADER": "(≥3/5 sources agree on presence)",
            "RESOLVE_WHEN": "≥3/5",
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt (English, based on best practices)"""
        ph = self._get_consensus_placeholders()
        prompt = """You are a professional dental OPG (panoramic radiograph) analysis assistant. Your task is to analyze the user-provided OPG image and answer questions.

## CRITICAL CONSTRAINTS

**Maximum __MAX_ITER__ iterations allowed.** Plan efficiently.

**Tool call batch limit: never emit more than 30 tool calls in a single iteration.**
Parallel tool calls are encouraged (3–10 at a time is typical), but batching 100+ tool calls per step exceeds the OpenAI platform limit (128) and will fail. If you need to analyze many teeth individually, split across iterations (≤30 per step) rather than dispatching all at once. Prefer aggregate tools (e.g. `list_teeth_with_status`, `get_quadrant`) over fan-out.

## OPG ORIENTATION (CRITICAL)

**Image RIGHT = Patient's LEFT side** | **Image LEFT = Patient's RIGHT side**

This is standard radiographic orientation (as if looking at the patient face-to-face).
- FDI Quadrant 1 (upper right) & 4 (lower right) = Patient's RIGHT = **Image LEFT side**
- FDI Quadrant 2 (upper left) & 3 (lower left) = Patient's LEFT = **Image RIGHT side**

**ALL descriptions MUST use PATIENT orientation, NOT image orientation!**

## ReAct Reasoning Format

**IMPORTANT**: You MUST follow the ReAct (Reasoning + Acting) format for EVERY step:

1. **Thought**: First explain your reasoning - what you've learned, what you need to find out, and why you're choosing a specific tool
2. **Action**: Then call the appropriate tool
3. **Observation**: After receiving tool results, analyze what you learned
4. **Summary (after VLM calls)**: Summarize VLM findings, note consistencies/conflicts, decide if more iteration needed

### Iteration End Protocol

After EACH iteration with VLM tool calls:
1. **Summarize** findings from all VLM sources in this iteration
2. **Identify consensus** - which findings are consistent across ≥2 VLMs?
3. **Identify conflicts** - which findings differ significantly?
4. **Decide**: 
   - If sufficient consensus → proceed to report generation
   - If key conflicts need resolution → plan targeted next iteration
   - **Do NOT exceed __MAX_ITER__ total iterations**

## Core Principles

1. **Three-Dimensional Analysis Framework** (Detection → Localization → Classification):
   - **Detection**: Was a finding detected? (Yes/No/Uncertain)
   - **Localization**: Where exactly? (FDI number, quadrant, region)
   - **Classification**: What type? (Use snake_case enum from schema)
   
2. **Consensus-Based Reporting (≥2 VLMs agree, no contradiction)**:
   - **Confirmed**: ≥2 VLMs agree on a finding AND no other VLM explicitly contradicts → Include in report
   - **High Confidence**: 1 tool + 1 VLM agree → Include with note
   - **Uncertain**: Only 1 source OR explicit contradiction → **OMIT from report** (Sparse Representation)
   
3. **Evidence Equality**: 
   - **GOLD STANDARD (absolute)**: Tooth counts, FDI numbering from detection tools
   - **EQUAL weight**: All other detection tool outputs = VLM opinions (neither is stronger)
   - VLMs provide independent analysis; tools provide structured detection results
   
4. **Unbiased Analysis**: DO NOT include tool detection results in VLM prompts. Let VLMs analyze independently.
5. **FDI Numbering**: Use two-digit FDI numbers (e.g., "11", "18", "28"). Quadrant 1/2/3/4 = upper-right/upper-left/lower-left/lower-right.
6. **OPG-only Constraint**: Only report OPG-visible findings. Do NOT mention occlusion, airway, cephalometry, etc.

## GOLD STANDARD Information

**IMPORTANT**: The following tool outputs are GOLD STANDARD:
- **Total tooth count** from detection tools is gold standard
- **Per-quadrant tooth count** is gold standard
- **FDI numbering** is nearly gold standard; at most 1-tooth offset error in missing tooth cases

### "Not Detected" Teeth
- **not_detected**: Tool did not detect this tooth on OPG. This could mean: extracted, congenitally absent, or unerupted.
- List ALL not-detected teeth in report (no special treatment for wisdom teeth)
- Example: "Not detected: 16, 18, 25, 28, 38, 46, 48"

## Hierarchical Analysis Workflow (OPG → Quadrant → Tooth)

**You MUST follow this top-down, three-level analysis pipeline.** This mirrors how a dentist reads an OPG: first the full panorama, then suspicious quadrants, then individual teeth.

### Phase 1: Full OPG Overview (Iterations 1–2)

**Goal**: Establish gold-standard dentition + collect independent VLM overall reads.

**Iteration 1** — run ALL of these in parallel:
- `get_quadrant(quadrant_names="Q1,Q2,Q3,Q4")` → gold-standard tooth counts & FDI lists
- `list_teeth_with_status(status_class="all")` → tool-detected abnormalities
- `get_bone_loss_description()` → bone loss regions
- `llm_zoo_openai(analysis_level="overall")` → GPT-5.4 full OPG read
- `llm_zoo_google(analysis_level="overall")` → Gemini full OPG read
- `llm_zoo_anthropic(analysis_level="overall")` → Claude Opus 4.6 full OPG read

**After Iteration 1** — Summarize:
- Which quadrants have abnormalities? (from tool + VLM consensus)
- Which specific findings are mentioned? (bone loss, impaction, caries, periapical, restorations, etc.)
- Identify **suspicious quadrants** that need focused analysis

### Phase 2: Quadrant-Level Analysis (Iterations 2–4)

**Goal**: Crop suspicious quadrants, have VLMs analyze them to narrow down specific teeth.

**Iteration 2** — for each suspicious quadrant, generate crop image:
- `get_annotated_image(target_type="quadrant", target_id="Q3", output_mode="crop")` → for GPT-5.4/Gemini

**Iteration 3** — use quadrant images for VLM analysis (3 VLMs):
- `llm_zoo_openai(image_path=<Q3_crop>, analysis_level="quadrant", custom_prompt="...")` — GPT-5.4 on crop
- `llm_zoo_google(image_path=<Q3_crop>, analysis_level="quadrant", custom_prompt="...")` — Gemini on crop
- `llm_zoo_anthropic(image_path=<Q3_crop>, analysis_level="quadrant", custom_prompt="...")` — Claude on crop

**After quadrant VLM reads** — identify **specific abnormal teeth** by cross-referencing:
- Tool-detected statuses (FDI-level)
- VLM quadrant reads from 3 VLMs (approximate FDI or region)
- Determine which teeth need single-tooth focused analysis

### Phase 3: Single-Tooth Analysis + RAG (Iterations 4–8)

**Goal**: Confirm specific findings on individual teeth with focused VLM + historical case retrieval.

**Step A** — Crop abnormal teeth:
- `get_annotated_image(target_type="tooth", target_id=FDI, output_mode="crop")` → for GPT-5.4/Gemini/RAG

**Step B** (next iteration) — 3 VLMs + RAG in parallel:
- `llm_zoo_openai(image_path=<tooth_crop>, analysis_level="tooth", custom_prompt="...")` — GPT-5.4 on crop
- `llm_zoo_google(image_path=<tooth_crop>, analysis_level="tooth", custom_prompt="...")` — Gemini on crop
- `llm_zoo_anthropic(image_path=<tooth_crop>, analysis_level="tooth", custom_prompt="...")` — Claude on crop
- `rag_similar_cases(image_path=<tooth_crop>)` — retrieve similar historical cases

**Step C** — For impacted teeth specifically:
- `extraction_risk_near_anatomy(fdi=FDI)` — assess anatomy proximity risk
- Report format: `Impacted tooth [FDI]: [direction] impaction, extraction risk: [level]`

**RAG Usage & Voting Guidelines**:
- **RAG ONLY accepts single-tooth crop images** — always use `get_annotated_image(target_type='tooth', output_mode='crop')` first, then pass the crop path to `rag_similar_cases`. Do NOT pass quadrant crops, bbox_overlay images, or the original OPG.
- Use RAG **only for teeth with detected abnormalities** (not for normal teeth with fillings)
- RAG is most effective for: **impacted teeth, periapical lesions, implants, unusual morphology**
- RAG is less effective for: **caries severity grading, periodontal assessment** (use VLM consensus instead)
- **RAG counts as 1 vote** in the consensus when: top similarity ≥ 0.85 AND retrieved report contains the same finding type
- **RAG abstains** when: not called, top similarity < 0.85, or retrieved finding type doesn't match
- Example: Tool=periapical(36), Gemini=periapical(36), RAG=pai_moderate(sim=0.87) → 3/4 = CONFIRMED

### Phase 4: Consensus & Report (Iterations 8–__MAX_ITER__)

**Goal**: Vote on all findings, resolve disagreements, write final report, then convert to structured JSON.

- Count votes per finding across sources (tool + VLMs + RAG)
- Call `resolve_finding_disagreement` if needed for position/classification conflicts
- Write final natural language diagnostic report
- **LAST STEP (MANDATORY)**: You MUST call the `convert_to_structured` tool with your full report text as input. Do NOT output the report as your final answer until AFTER you have called this tool. The workflow is: write report → call convert_to_structured → then output the report as final answer

### Report Structure
1. **Dentition Overview**: Total teeth detected, per-quadrant counts, list ALL not-detected teeth
2. **Confirmed Pathological Findings**: Only confirmed findings with FDI localization
3. **Restorations/Treatments**: Detected restorations, crowns, implants, RCT
4. **Periodontal Assessment**: If bone loss detected and confirmed
5. **Impacted Teeth Assessment** (if any): Direction + extraction risk
6. **RAG Support** (if used): Brief summary of similar historical cases that support confirmed findings
7. **Other Findings**: Sinuses, TMJ, anatomical variants (if any)

## Available Tools

### Detection & Structured Query Tools (High-level only)

> Note: Detection cache is preloaded at startup. All tools share the same cache.

1. **get_tooth_by_fdi**: Get single tooth info by FDI (box/confidence) - FDI is nearly gold standard
2. **get_quadrant**: Get one or more quadrant info (box + FDI list), comma-separated names supported - tooth count is gold standard
3. **get_quadrant_teeth**: Get all teeth in quadrant with full info - tooth count is gold standard
4. **get_tooth_mask**: Get tooth mask (MedSAM segmentation) by FDI
5. **get_status_on_tooth**: Get statuses on a tooth (TVEM + YOLO filtered)
   - YOLO Caries/Deep Caries are filtered out; only Impacted and Periapical Lesion from YOLO kept
   - 8th teeth (18/28/38/48) with TVEM "Root Piece" auto-changed to "impacted tooth"
6. **list_teeth_with_status**: List FDI of teeth with a given status class (YOLO Caries/Deep Caries filtered)
7. **extraction_risk_near_anatomy**: Assess proximity risk to maxillary sinus/mandibular canal
8. **get_bone_loss_description**: Describe bone loss regions by involved quadrants/teeth
9. **get_annotated_image**: Generate annotated image (crop or bbox_overlay)

### VLM Analysis Tools (Strong Evidence, for validation)

**VLM Tool Usage Guidelines:**
- **GPT-5.4 / Gemini**: Use for any level; prefer `crop` images for quadrant/tooth analysis
- **Token limits by analysis_level**: overall=2048, quadrant=1024, tooth=256

10. **llm_zoo_openai**: GPT-5.4 (temperature 0.3, high precision) - for all analysis levels
11. **llm_zoo_google**: Gemini 3 Flash (temperature 0.3) - for all analysis levels
12. **llm_zoo_anthropic**: Claude Opus 4.6 - for all analysis levels
13. **resolve_finding_disagreement**: Resolve position/classification disagreement for a confirmed finding
14. **convert_to_structured**: Convert natural language report to structured JSON (Schema&Enum standard) — **MUST call as last step**

### Disagreement Resolution Tool

**resolve_finding_disagreement**: Call when a confirmed finding has position or classification disagreement.
- Input: finding_type, disagreement_type, vlm_opinions (JSON), gold_standard_info (JSON)
- Output: resolved FDI position or classification with reasoning
- **Use to get specific FDI** when VLMs agree on presence but disagree on location

## Tool Usage Principles

**Follow the Hierarchical Analysis Workflow (OPG → Quadrant → Tooth) defined above.**

- Detection cache is preloaded; use high-level tools directly
- **VLM strategy**: GPT-5.4 + Gemini + Claude for overall OPG; use `crop` images for quadrant/tooth-level
- **Consensus**: count votes across tool + VLMs; threshold determines confirmed/high-confidence/omitted

**CRITICAL - Crop-then-Analyze Rule:**

For quadrant-level or tooth-level VLM analysis, you MUST:
1. **FIRST** call `get_annotated_image` to get the cropped image
2. **THEN** (in the NEXT iteration) call VLM tools with the returned `image_path`

**DO NOT**:
- Call VLM + `get_annotated_image` in PARALLEL (the image doesn't exist yet!)
- Use the original OPG path for tooth/quadrant VLM analysis
- Skip cropping when analyzing specific quadrants or teeth

**Image Type Selection**:
- `output_mode="crop"`: Cropped region — use for **GPT-5.4, Gemini, RAG** (focused, token-efficient)
- `output_mode="bbox_overlay"`: Full OPG with red bbox — available but less commonly needed
- For quadrant/tooth analysis, use `crop` mode for VLMs

## Key Constraints and Rules

### FDI Numbering Rules
- Use two-digit strings: "11", "18", "28" (not "1", "18", "28")
- Unit digit 8 = third molar
- Quadrant: 1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right

### Evidence Equality Rule
- **GOLD STANDARD (absolute truth)**: Tooth counts, FDI lists from detection tools
- **EQUAL weight (all others)**: Other detection tool outputs = VLM opinions
- Example: Tool says "Filling on 17" has SAME weight as "GPT says Filling on 17"

__CONSENSUS_BLOCK__

### Classification Disagreement Rule
If sources **agree on disease presence** but **disagree on classification/severity**:
- **KEEP the disease** in report
- **Use conservative classification** (least severe / most general)
- Example: 3 sources say "bone loss" but severity differs (severe/mild/mild) → Report as "bone loss present (mild to moderate)" or "bone loss (severity undetermined)"

### VLM Position Insensitivity Rule
**IMPORTANT**: VLMs are often inaccurate about exact tooth positions but reliable about disease presence.

When VLMs **agree on finding presence** but **disagree on location/FDI**:
1. **Confirm the finding EXISTS** (majority vote on presence)
2. **For position**: 
   - If Tool detected it → use Tool's FDI (gold standard for position)
   - If Tool did NOT detect → **CALL `resolve_finding_disagreement` tool** to determine FDI

### Resolving Disagreements with Subagent
**IMPORTANT**: For each confirmed finding (__RESOLVE_WHEN__) with position OR classification disagreement, call `resolve_finding_disagreement`:

**When to call**:
- Finding is CONFIRMED (__RESOLVE_WHEN__ agree on presence)
- BUT VLMs disagree on **position** (different FDI numbers) OR **classification** (different severity/type)

**How to call**:
1. Collect all VLM opinions mentioning this finding
2. Gather gold standard info (teeth_fdi, not_detected, quadrants from tool results)
3. Call `resolve_finding_disagreement` with:
   - `finding_type`: e.g., "implant", "bone_loss"
   - `disagreement_type`: "position" or "classification"
   - `vlm_opinions`: JSON array of each VLM's opinion and position/classification claim
   - `gold_standard_info`: JSON with teeth list and quadrants

**Example for implant position disagreement**:
```
vlm_opinions: [
  {"source": "DentalGPT", "opinion": "implant in lower left", "position_or_classification": "36"},
  {"source": "OralGPT", "opinion": "implant posterior mandible", "position_or_classification": "lower left"},
  {"source": "GPT-5.4", "opinion": "implant at 46 site", "position_or_classification": "46"},
  {"source": "Gemini", "opinion": "implant lower right", "position_or_classification": "46"}
]
gold_standard_info: {"teeth_fdi": ["31","32",...], "not_detected": ["38","46","48"], "quadrants": {...}}
```

The subagent will analyze and return a specific FDI or conservative region description.
   
**Examples**:
- 4 VLMs say "implant present" (Tool=none), locations: 3 say "lower left", 1 says "lower right"
  → Report: "**Implant present** [CONFIRMED by 4 VLMs] in lower posterior region (position varies across VLMs)"
- 3 VLMs say "periapical lesion" on different teeth (36/37/46)
  → Report: "**Periapical changes noted** [CONFIRMED] in lower posterior region"

### Prompt Design Principles (for VLM tools)
- **Unbiased**: Do NOT include tool detection results in VLM prompts
- **Neutral language**: Use "analyze", "assess" not "verify", "confirm"

### Uncertainty Handling
- **If uncertain → OMIT**: Do not report findings without sufficient consensus
- **Sparse principle**: Better to omit uncertain findings than include noise

### OPG-only Constraint
- Only report OPG-visible findings
- **DO NOT mention**: occlusion, airway, cephalometry, clinical exam results

## Output Requirements

### Analysis Workflow (within __MAX_ITER__ iterations)
Follow the **Hierarchical Analysis Workflow** defined above:
1. **Phase 1 (OPG)**: Tool detection + 4 VLM overall reads (parallel)
2. **Phase 2 (Quadrant)**: Crop suspicious quadrants → VLM quadrant analysis → identify abnormal teeth
3. **Phase 3 (Tooth)**: Crop abnormal teeth → VLM tooth analysis + RAG similar cases + extraction risk
4. **Phase 4 (Report)**: Consensus vote (__SUMMARY_THRESHOLD__) → resolve disagreements → write report
   - **For each finding**: Count how many sources (__SOURCES_DESC__) report it
   - **Presence vote**: Does the source say "yes/present" or "no/absent"? (silence = abstain)
   - **If threshold say YES**: CONFIRMED, regardless of whether others say NO

### Confidence Labels in Report
- **[CONFIRMED]**: __CONFIRMED_LABEL__ - definitive finding (majority vote wins)
- **[CONFIRMED, severity undetermined]**: __CONFIRMED_LABEL__ on disease presence, but severity/classification differs
- **[CONFIRMED, position varies]**: __CONFIRMED_LABEL__ on finding, but location/FDI differs - use general location
- **[HIGH_CONFIDENCE]**: __HIGH_CONF_LABEL__ - include with qualification
- **[OMITTED]**: below threshold - do not include

### Natural Language Report Structure

1. **Dentition Summary**
   - Total teeth detected (GOLD STANDARD count)
   - Per-quadrant breakdown
   - ALL not-detected teeth: List FDI (e.g., "Not detected: 16, 18, 25, 28, 38, 46, 48")

2. **Confirmed Pathological Findings** __FINDINGS_HEADER__
   - Specify FDI
   - Describe finding type
   - If classification differs: use conservative/general description (e.g., "bone loss present" instead of severity grade)

3. **Restorations/Treatments Detected**
   - List teeth with restorations (Filling, Crown, etc.)

4. **Periodontal Assessment** (if bone loss confirmed)

5. **Impacted Teeth** (if any): direction + extraction risk

6. **RAG Support** (if used): Brief note on similar historical cases that corroborate confirmed findings

7. **Other Findings** (if any and confirmed)
   - Sinus/TMJ findings

### Final Rules
- Use FDI notation (two-digit strings: "38", "47")
- Answer in user's language
- OMIT uncertain findings
- **BEFORE outputting your final answer**: you MUST call `convert_to_structured` with your complete report text. Do NOT skip this step.
- Output ONLY the final natural language diagnostic report (no reasoning process in final answer)

Now analyze the user-provided OPG image."""
        prompt = prompt.replace("__CONSENSUS_BLOCK__", self._get_consensus_block())
        max_iter = str(self.config.get("react", {}).get("max_iterations", 10))
        prompt = prompt.replace("__MAX_ITER__", max_iter)
        for k, v in ph.items():
            prompt = prompt.replace("__" + k + "__", v)
        # Both profiles now use COMBINED region tools that wrap get_annotated_image
        # + a specific VLM/RAG in one call. Rewrite the default prompt body's
        # references to the raw tool names so the Agent calls the right things.
        if self.local_vlm_only:
            prompt = (prompt
                .replace("llm_zoo_openai", "analyze_with_gpt")
                .replace("llm_zoo_google", "analyze_with_dentalgpt")
                .replace("llm_zoo_anthropic", "analyze_with_oralgpt")
                .replace("rag_similar_cases", "retrieve_similar_cases")
                .replace("Gemini 3 Flash", "DentalGPT (Qwen2.5-VL)")
                .replace("Gemini", "DentalGPT")
                .replace("Claude Opus 4.6", "OralGPT (Qwen2.5-VL)")
                .replace("Claude", "OralGPT"))
            addendum_table = (
                "| Tool | Underlying model | Image format it uses internally |\n"
                "|---|---|---|\n"
                "| `analyze_with_gpt`        | GPT-5.4             | tooth/quadrant crop |\n"
                "| `analyze_with_dentalgpt`  | DentalGPT (Qwen-VL) | full OPG + bbox overlay |\n"
                "| `analyze_with_oralgpt`    | OralGPT   (Qwen-VL) | full OPG + bbox overlay |\n"
                "| `retrieve_similar_cases`  | MedImageInsights RAG | tooth crop |\n"
            )
            parallel_line = ("call all four in parallel: `analyze_with_gpt`, `analyze_with_dentalgpt`, "
                             "`analyze_with_oralgpt`, `retrieve_similar_cases`")
        else:
            prompt = (prompt
                .replace("llm_zoo_openai", "analyze_with_gpt")
                .replace("llm_zoo_google", "analyze_with_gemini")
                .replace("llm_zoo_anthropic", "analyze_with_claude")
                .replace("rag_similar_cases", "retrieve_similar_cases"))
            addendum_table = (
                "| Tool | Underlying model | Image format it uses internally |\n"
                "|---|---|---|\n"
                "| `analyze_with_gpt`        | GPT-5.4              | tooth/quadrant crop |\n"
                "| `analyze_with_gemini`     | Gemini 3 Flash       | tooth/quadrant crop |\n"
                "| `analyze_with_claude`     | Claude Opus 4.6      | tooth/quadrant crop |\n"
                "| `retrieve_similar_cases`  | MedImageInsights RAG | tooth crop |\n"
            )
            parallel_line = ("call all four in parallel: `analyze_with_gpt`, `analyze_with_gemini`, "
                             "`analyze_with_claude`, `retrieve_similar_cases`")

        prompt += (
            "\n\n## Combined Region Tools\n"
            "The following tools each wrap `get_annotated_image` + a VLM/RAG internally.\n"
            "**Do NOT call `get_annotated_image` manually before them** — each tool prepares its own preferred image format:\n"
            + addendum_table +
            "\n**Input schema (identical for all 4)**: `target_type` ∈ {'tooth','quadrant','overall'}, "
            "`target_id` (FDI like '36' for tooth, 'Q1'-'Q4' for quadrant; omit for overall), "
            "optional `custom_prompt`, `focus_areas`, `detected_findings`.\n"
            "\n**Consensus source count**: 5 = Tool (detection) + RAG + 3 VLMs. Confirm when ≥3/5 agree.\n"
            "\n**Parallel dispatch pattern**: for each analysis target (overall / quadrant / single tooth), "
            + parallel_line + " — each tool fetches its own image independently and returns a vote. "
            "No separate `get_annotated_image` step is needed."
        )
        return prompt
    
    def _create_agent(self):
        """Create the LangChain agent"""
        react_config = self.config.get("react", {})
        max_iterations = react_config.get("max_iterations", 10)

        # The first argument of create_agent is the model (either a string or a BaseChatModel)
        agent = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
        
        return agent
    
    def run(
        self,
        question: str,
        image_path: str,
        memory: Optional[AgentMemory] = None
    ) -> Dict[str, Any]:
        """
        Run agent and return answer.
        
        Agent will generate a natural language diagnostic report.
        
        Args:
            question: User question (should be in English)
            image_path: OPG image path
            memory: Memory instance (optional)
            
        Returns:
            Dict containing answer and tool call history
        """
        # Initialize memory
        if memory is None:
            memory = AgentMemory(image_path=image_path, question=question)
        
        # STEP 0: Preload detection cache by running run_all_detections at startup (ablation 3 without detection tools skips this)
        abs_image_path = str(Path(image_path).resolve())
        token = current_image_path_ctx.set(abs_image_path)
        try:
            if not self.ablation_disable_all_tools:
                logger.info(f"Preloading detection cache for: {abs_image_path}")
                from langchain_core.runnables import RunnableConfig
                preload_config = RunnableConfig(configurable={"current_image_path": abs_image_path})
                preload_result = self.toolkit.run_all_detections(abs_image_path, config=preload_config)
                import json
                preload_data = json.loads(preload_result)
                teeth_fdi_count = len(preload_data.get("teeth_fdi", {})) if isinstance(preload_data.get("teeth_fdi"), dict) else 0
                quadrant_count = len([k for k in preload_data.get("quadrants", {}).keys() if k not in ["error"]]) if isinstance(preload_data.get("quadrants"), dict) else 0
                logger.info(f"✓ Detection cache preloaded: {teeth_fdi_count} teeth FDI, {quadrant_count} quadrants")
            else:
                logger.info("Ablation 3: skip detection preload (no detection tools)")
        except Exception as e:
            logger.warning(f"Detection cache preload failed: {e}")
        finally:
            current_image_path_ctx.reset(token)
        
        # Build user message (text only, image analysis handled by VLM tools)
        # NOTE: We do NOT send the image to Agent LLM to avoid token limits
        # VLM tools (dental_expert_analysis, oral_expert_analysis, llm_zoo_*) handle image analysis
        from langchain_core.messages import HumanMessage
        
        # Include detection summary in the question for context
        detection_summary = f"\n\n[Image path: {abs_image_path}]\n[Detection cache preloaded: ready for tool queries]"
        user_message = HumanMessage(
            content=question + detection_summary
        )
        
        # Prepare message history (from memory)
        chat_history = []
        for call in memory.tool_calls:
            # Append the tool call and its result to the history
            chat_history.append({
                "role": "assistant",
                "content": f"Tool call: {call['tool_name']}"
            })
            chat_history.append({
                "role": "user",
                "content": f"Tool result: {call['tool_output']}"
            })

        # Invoke the agent
        try:
            # Input format expected by create_agent: {"messages": [HumanMessage, ...]}
            # Build the message list
            messages = [user_message]

            # If chat_history is present, convert to message objects
            from langchain_core.messages import AIMessage
            for hist in chat_history:
                role = hist.get("role", "")
                content = hist.get("content", "")
                if role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))

            # Invoke the agent, passing the current image path for tools to inject from config (avoids the LLM passing wrong paths)
            abs_image_path = str(Path(image_path).resolve())
            run_config = {"configurable": {"current_image_path": abs_image_path}}

            # Reset token tracker
            self.token_tracker.total_tokens = 0
            self.token_tracker.total_input_tokens = 0
            self.token_tracker.total_output_tokens = 0
            self.token_tracker.call_count = 0

            token = current_image_path_ctx.set(abs_image_path)
            try:
                # Truncated exponential backoff retry to handle 429 (quota/rate limit)
                max_retries = 5
                initial_delay = 1.0
                max_delay = 60.0
                last_error = None
                for attempt in range(max_retries):
                    try:
                        # Set recursion_limit to allow enough tool-call iterations
                        react_config = self.config.get("react", {})
                        max_iterations = react_config.get("max_iterations", 15)
                        result = self.agent.invoke(
                            {"messages": messages},
                            config={
                                **run_config,
                                "callbacks": [self.token_tracker],
                                "recursion_limit": max_iterations * 2 + 5  # each iteration may have 2 steps (AI + Tool)
                            }
                        )
                        break
                    except Exception as e:
                        last_error = e
                        err_str = str(e).lower()
                        is_429 = (
                            "429" in err_str
                            or "rate" in err_str
                            or "quota" in err_str
                            or "insufficient_quota" in err_str
                            or (getattr(e, "status_code", None) == 429)
                        )
                        if is_429 and attempt < max_retries - 1:
                            delay = min(
                                initial_delay * (2 ** attempt) + random.uniform(0, 1),
                                max_delay
                            )
                            logger.warning(
                                "LLM call hit 429/rate limit (attempt %d/%d), retrying in %.1f s: %s",
                                attempt + 1, max_retries, delay, str(e)[:200]
                            )
                            time.sleep(delay)
                            continue
                        raise

                # Check token upper limit (only for gpt-5.4 and gemini-3-flash)
                if self.token_tracker.is_limit_exceeded():
                    logger.warning(
                        f"Token usage exceeded limit: {self.token_tracker.total_tokens}/{self.token_tracker.token_limit}"
                    )
            finally:
                current_image_path_ctx.reset(token)

            # Record the full ReAct reasoning process
            # Format: each iteration contains Thought -> Action(s) -> Observation(s)
            react_trace = []  # full ReAct trace
            final_answer = ""

            if isinstance(result, dict) and "messages" in result:
                from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

                # Build a tool_call_id -> ToolMessage mapping
                tool_results = {}
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_call_id:
                            tool_results[tool_call_id] = {
                                "name": getattr(msg, "name", "unknown"),
                                "output": getattr(msg, "content", "")
                            }

                # Process messages in order to build the full ReAct trace
                iteration = 0
                for msg in result["messages"]:
                    if isinstance(msg, HumanMessage):
                        # Skip user messages (already captured in `question`)
                        continue

                    if isinstance(msg, AIMessage):
                        iteration += 1
                        content = _message_content_to_str(getattr(msg, "content", ""))
                        tool_calls_in_msg = getattr(msg, "tool_calls", []) or []

                        # Build the trace for this iteration
                        step = {
                            "iteration": iteration,
                            "thought": content.strip() if content else "(no explicit thought)",
                            "actions": [],
                            "observations": []
                        }

                        # Handle all tool_calls in this AIMessage
                        for tc in tool_calls_in_msg:
                            tool_name = tc.get("name", "unknown")
                            tool_input = tc.get("args", {})
                            tool_call_id = tc.get("id")

                            action_entry = {
                                "tool": tool_name,
                                "input": tool_input
                            }
                            step["actions"].append(action_entry)

                            # Retrieve the corresponding observation
                            if tool_call_id and tool_call_id in tool_results:
                                tr = tool_results[tool_call_id]
                                observation = {
                                    "tool": tr["name"],
                                    "output": tr["output"]
                                }
                                step["observations"].append(observation)

                                # Record into memory
                                memory.add_tool_call(
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                    tool_output=tr["output"],
                                    iteration=iteration,
                                    reasoning=content.strip() if content else ""
                                )

                        # Record the last AIMessage with content as a candidate final_answer
                        if content and content.strip():
                            final_answer = content.strip()

                        # Only record steps that have a thought or actions
                        if step["thought"] != "(no explicit thought)" or step["actions"]:
                            react_trace.append(step)

            # Fallback logic
            if not final_answer:
                if isinstance(result, dict):
                    if "messages" in result and result["messages"]:
                        for msg in reversed(result["messages"]):
                            if hasattr(msg, "content") and msg.content:
                                final_answer = _message_content_to_str(msg.content)
                                if final_answer.strip():
                                    break
                        else:
                            last_msg = result.get("messages", [])[-1] if result.get("messages") else None
                            final_answer = _message_content_to_str(getattr(last_msg, "content", "")) if last_msg else ""
                    elif "output" in result:
                        final_answer = result["output"]
                    else:
                        final_answer = str(result)
                else:
                    final_answer = str(result)
            
            # Build the return value
            return {
                "answer": final_answer,
                "memory": memory,
                "react_trace": react_trace,  # full ReAct trace (Thought-Action-Observation)
                "tool_calls": memory.tool_calls,
                "token_usage": self.token_tracker.get_summary()
            }

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            return {
                "answer": f"Error during analysis: {str(e)}",
                "memory": memory,
                "tool_calls": memory.tool_calls,
                "error": str(e),
                "token_usage": self.token_tracker.get_summary()
            }

    def stream(
        self,
        question: str,
        image_path: str,
        memory: Optional[AgentMemory] = None
    ):
        """
        Run the agent in streaming mode (generator)

        Args:
            question: user question
            image_path: OPG image path
            memory: Memory instance (optional)

        Yields:
            Result of each step
        """
        if memory is None:
            memory = AgentMemory(image_path=image_path, question=question)
        
        # Build user message (text only, no image to avoid token limits)
        from langchain_core.messages import HumanMessage
        abs_image_path = str(Path(image_path).resolve())
        detection_summary = f"\n\n[Image path: {abs_image_path}]\n[Use VLM tools for image analysis]"
        user_message = HumanMessage(
            content=question + detection_summary
        )
        
        chat_history = []
        for call in memory.tool_calls:
            chat_history.append({
                "role": "assistant",
                "content": f"Tool call: {call['tool_name']}"
            })
            chat_history.append({
                "role": "user",
                "content": f"Tool result: {call['tool_output']}"
            })
        
        # Build message list
        messages = [user_message]

        # If chat_history is present, convert to message objects
        from langchain_core.messages import AIMessage
        for hist in chat_history:
            role = hist.get("role", "")
            content = hist.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))

        try:
            for chunk in self.agent.stream({"messages": messages}):
                yield chunk
        except Exception as e:
            logger.error(f"Agent streaming run failed: {e}", exc_info=True)
            yield {"error": str(e)}
