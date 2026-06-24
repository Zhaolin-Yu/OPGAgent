"""
基于 LangChain create_agent 的 ReAct Agent
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
    """将 AIMessage.content 规范化为字符串（可能是 str 或 list，如 Gemini 的 [{\"type\":\"text\",\"text\":\"...\"}]）。"""
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
    """
    Token 使用量跟踪器：凡 LangChain 在 llm_output 中带回 token_usage 即累计。
    多轮 ReAct 下 total_input_tokens 为各次请求 prompt_tokens 之和，同一对话历史会被重复计入，仅作上界/成本粗估；
    max_prompt_tokens_single_call 可反映单次请求上下文规模。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.per_call: List[Dict[str, int]] = []
        mn = (model_name or "").lower()
        # 仅对部分模型强制执行累计上限（避免误伤其他模型）
        self.enforce_token_limit = "gpt-5.2" in mn or "gemini-3-flash" in mn
        self.token_limit = 100000  # 10 万 token 上限

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用结束时记录 token 使用量"""
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "llm_output") and gen.llm_output:
                    usage = gen.llm_output.get("token_usage", {})
                    if not usage:
                        continue
                    input_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
                    output_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
                    total = int(usage.get("total_tokens", 0) or (input_tokens + output_tokens))

                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.total_tokens += total
                    self.call_count += 1
                    self.per_call.append(
                        {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": total,
                        }
                    )

                    logger.debug(
                        "Token 使用: %s (输入: %s, 输出: %s), 累计: %s",
                        total,
                        input_tokens,
                        output_tokens,
                        self.total_tokens,
                    )

    def is_limit_exceeded(self) -> bool:
        """检查是否超过 token 上限（仅 enforce_token_limit 为真时生效）"""
        return self.enforce_token_limit and self.total_tokens >= self.token_limit

    def get_summary(self) -> Dict[str, Any]:
        """获取使用量摘要"""
        max_prompt = max((c["prompt_tokens"] for c in self.per_call), default=0)
        return {
            "model": self.model_name,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "llm_call_count": self.call_count,
            "max_prompt_tokens_single_call": max_prompt,
            "per_call_token_usage": self.per_call,
            "input_tokens_sum_note": (
                "多轮对话下为各次 API 请求的 prompt_tokens 之和，含重复上下文，非唯一 token 数。"
            ),
            "enforce_token_limit": self.enforce_token_limit,
            "token_limit": self.token_limit if self.enforce_token_limit else None,
            "limit_exceeded": self.is_limit_exceeded(),
        }


def _data_url(image_path: str) -> str:
    """将图像转换为 data URL"""
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
    构建 Chat 模型（OpenAI 兼容或 Google Gemini）

    Args:
        provider: 模型提供商 (qwen/openai/openrouter/gemini)
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大输出 token 数（Gemini 使用 max_output_tokens；OpenAI 兼容通过 model_kwargs 传递）

    Returns:
        BaseChatModel 实例
    """
    provider_norm = (provider or "qwen").strip().lower()

    if provider_norm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 GEMINI_API_KEY 或 GOOGLE_API_KEY，无法调用 Gemini 模型。")
        model_name = model or os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview"
        max_out = max_tokens if max_tokens is not None else 65536
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_out,
            )
        except ImportError as e:
            raise RuntimeError(f"使用 Gemini 需安装 langchain-google-genai: {e}") from e

    if provider_norm == "qwen":
        base_url = (
            os.getenv("QWEN_OPENAI_BASE_URL")
            or os.getenv("DASHSCOPE_OPENAI_BASE_URL")
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 DASHSCOPE_API_KEY，无法调用 Qwen 模型。")
        model_name = model or os.getenv("QWEN_MODEL") or "qwen3-vl-235b-a22b-instruct"
    elif provider_norm == "openrouter":
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 OPENROUTER_API_KEY，无法调用 OpenRouter 模型。")
        model_name = model or "qwen/qwen3-vl-235b-a22b-thinking"
    elif provider_norm == "openai":
        base_url = os.getenv("OPENAI_BASE_URL") or None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 OPENAI_API_KEY，无法调用 OpenAI 模型。")
        model_name = model or os.getenv("OPENAI_MODEL") or "gpt-5.2"
    else:
        raise RuntimeError(f"不支持的 model provider: {provider_norm}")

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
    """基于 LangChain create_agent 的 ReAct Agent"""
    
    # 工具服务配置选项（相对于 Agent_v3 目录）
    TOOL_SERVICE_CONFIGS = {
        "default": "src/opgagent/config/tools_config.yaml",  # 默认配置（原端口）
        "api_service": "api_service/tools_config.yaml",  # 新配置（端口从 6600 开始）
    }
    
    def __init__(self, config_path: Optional[str] = None, tool_service: str = "default"):
        """
        初始化 Agent

        Args:
            config_path: 配置文件路径
            tool_service: 工具服务配置选项
                - "default": 使用默认配置（原端口 8xxx）
                - "api_service": 使用新 api_service 配置（端口从 6600 开始）
        """
        # 记录使用的工具服务配置
        self.tool_service = tool_service

        # 加载配置
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            # 使用默认配置
            self.config = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.2",
                    "temperature": 0.7,
                    "max_tokens": 8192
                },
                "react": {
                    "max_iterations": 15,
                    "stop_on_error": False
                }
            }
        
        # 加载工具配置（支持相对路径和绝对路径）
        tools_config_path = self.config.get("tools", {}).get("config_path")
        self.tools_config = {}
        
        # Agent_v3 根目录 (agent.py -> opgagent -> src -> Agent_v3)
        opgagent_root = Path(__file__).parent.parent.parent
        
        # 根据 tool_service 选项确定工具配置路径
        if tool_service in self.TOOL_SERVICE_CONFIGS:
            # 使用预定义的配置路径
            service_config_path = opgagent_root / self.TOOL_SERVICE_CONFIGS[tool_service]
            logger.info(f"使用工具服务配置: {tool_service} ({service_config_path})")
        else:
            # 未知选项，使用默认
            service_config_path = opgagent_root / self.TOOL_SERVICE_CONFIGS["default"]
            logger.warning(f"未知的工具服务配置: {tool_service}，使用默认配置")
        
        # 默认工具配置路径（相对于 agent.py）
        default_tools_config = Path(__file__).parent / "config" / "tools_config.yaml"
        
        # 尝试多种路径解析方式
        candidate_paths = []
        # 首先尝试 tool_service 指定的配置
        candidate_paths.append(service_config_path)
        # 然后尝试从配置文件中指定的路径
        if tools_config_path:
            candidate_paths.append(Path(tools_config_path))  # 原始路径（绝对或相对于 cwd）
            candidate_paths.append(opgagent_root / tools_config_path)  # 相对于 Agent_v3 根目录
        # 最后使用默认路径作为后备
        candidate_paths.append(default_tools_config)
        
        for p in candidate_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self.tools_config = yaml.safe_load(f)
                logger.info(f"✓ 已加载工具配置: {p}")
                break
        else:
            logger.warning(f"工具配置文件未找到，尝试过: {[str(p) for p in candidate_paths]}")
        
        # 初始化 LLM
        llm_config = self.config.get("llm", {})
        model_name = llm_config.get("model") or "gpt-5.2"
        self.llm = build_chat_model(
            provider=llm_config.get("provider", "openai"),
            model=model_name,
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens"),
        )
        
        # 初始化 Token 使用量跟踪器（仅对 gpt-5.2 和 gemini-3-flash 启用）
        self.token_tracker = TokenUsageTracker(model_name)
        
        # 创建工具列表并获取 toolkit 引用（用于预加载缓存）
        # 注意：run_all_detections 不再暴露为工具，而是在 Agent 启动时自动运行
        self.tools, self.toolkit = create_dental_tools(
            self.tools_config,
            analysis_only=False,
            return_toolkit=True,
        )

        # 构建 system prompt
        self.system_prompt = self._build_system_prompt()

        # 创建 agent
        self.agent = self._create_agent()

        logger.info(f"✓ OPGReActAgent 初始化完成 (模型: {llm_config.get('model')}, 工具服务: {self.tool_service}, VLM: 4 VLM, 工具数: {len(self.tools)})")
    
    def _get_consensus_block(self) -> str:
        """5 源 3/5 投票：4 VLMs (DentalGPT, OralGPT, GPT-5.2, Gemini) + 1 Tool。"""
        return r"""### Consensus Rule (Majority Vote: ≥3 out of 5 sources)
**5 sources total**: 4 VLMs (DentalGPT, OralGPT, GPT-5.2, Gemini) + 1 Tool (detection)

**CRITICAL**: ≥3/5 = CONFIRMED, **EVEN IF** the other 2 sources explicitly say "no" or contradict!

- **Accept (≥3/5)**: If ≥3 sources report the same finding → **CONFIRMED (majority wins)**
  - This applies EVEN IF the remaining 2 sources say "none" or "not present"
  - Example: 4 VLMs say "bone loss present", Tool says "none" → **CONFIRMED** (4/5 wins)
  - Example: 4 VLMs say "implant present", Tool says "no implant" → **CONFIRMED** (4/5 wins)
- **High Confidence (2/5)**: If exactly 2 sources agree → include with [HIGH_CONFIDENCE] label
- **Reject (<2/5)**: Only 1 source reports → OMIT

**Vote Counting Examples**:
- Bone loss: Tool=none, DentalGPT=severe, OralGPT=mild, GPT=mild, Gemini=mild → **4/5 say YES → CONFIRMED**
- Implant: Tool=none, DentalGPT=yes, OralGPT=yes, GPT=yes, Gemini=yes → **4/5 say YES → CONFIRMED**"""

    def _get_consensus_placeholders(self) -> dict:
        """返回 prompt 中与共识相关的占位符（用于后续段落）。"""
        return {
            "CONFIRMED_THRESHOLD": "≥3/5",
            "SOURCES_DESC": "Tool + 4 VLMs",
            "VLM_PHASE_DESC": "DentalGPT, OralGPT, GPT, Gemini",
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

**Maximum 5 iterations allowed.** Plan efficiently.

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
   - **Do NOT exceed 5 total iterations**

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

## Output Workflow

**Output:**
Generate a **natural language diagnostic report** with confirmed findings only.

### Natural Language Report Should Include:
1. **Dentition Overview**: Total teeth detected, per-quadrant counts, list ALL not-detected teeth
2. **Pathological Findings**: Only confirmed findings (≥3 sources) with FDI localization
3. **Restorations/Treatments**: Detected restorations, crowns, implants, RCT
4. **Periodontal Assessment**: If bone loss detected and confirmed
5. **Impacted Teeth Assessment** (if any): Direction + extraction risk (see below)
6. **Other Findings**: Sinuses, TMJ, anatomical variants (if any)

### Impacted Tooth Assessment Workflow

**When an impacted tooth is detected** (from tool or VLM), you MUST:

1. **Get focused image first** using `get_annotated_image`:
   - Call with `target_fdi` (e.g., "18"), `output_type="crop"` or `"bbox_overlay"`
   - WAIT for the returned image path

2. **Determine impaction direction** using VLM (llm_zoo_openai or llm_zoo_google) IN NEXT ITERATION:
   - Use the image_path from step 1
   - Call with `analysis_level="tooth"`, `custom_prompt` asking about impaction angle/direction
   - Common directions: **mesioangular** (tilted toward adjacent tooth), **distoangular** (tilted away), **horizontal**, **vertical**, **inverted**
   
3. **Assess extraction risk** using `extraction_risk_near_anatomy` tool (can run parallel with step 1):
   - For **upper teeth (18/28)**: checks proximity to **maxillary sinus**
   - For **lower teeth (38/48)**: checks proximity to **mandibular canal** (inferior alveolar nerve)
   - Risk levels: **high** (≤10px), **moderate** (10-20px), **low** (>20px)

4. **Report format for impacted teeth**:
   ```
   - **Impacted tooth [FDI]**: [direction] impaction
     - Extraction risk: [high/moderate/low] - [anatomy structure] proximity [distance]
   ```

**Example**:
```
- **Impacted tooth 38**: Mesioangular impaction
  - Extraction risk: HIGH - mandibular canal proximity <10px (nerve damage risk)
```

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
- **DentalGPT / OralGPT**: ONLY use for full OPG analysis (analysis_type='overall')
- **GPT-5.2 / Gemini**: Use for any level (overall/quadrant/tooth)
- **Token limits by analysis_level**: overall=2048, quadrant=1024, tooth=256

10. **dental_expert_analysis**: DentalGPT (RECOMMENDED: analysis_type='overall' ONLY)
11. **oral_expert_analysis**: OralGPT (RECOMMENDED: analysis_type='overall' ONLY)
12. **llm_zoo_openai**: GPT-5.2 (temperature 0.3, high precision) - for all analysis levels
13. **llm_zoo_google**: Gemini 3 Flash (temperature 1.0) - for all analysis levels
14. **resolve_finding_disagreement**: Resolve position/classification disagreement for a confirmed finding

### Disagreement Resolution Tool

**resolve_finding_disagreement**: Call when a confirmed finding has position or classification disagreement.
- Input: finding_type, disagreement_type, vlm_opinions (JSON), gold_standard_info (JSON)
- Output: resolved FDI position or classification with reasoning
- **Use to get specific FDI** when VLMs agree on presence but disagree on location

## Tool Usage Principles

**Autonomously decide** which tools to call and in what order.

- Detection cache is preloaded; use high-level tools directly
- **VLM strategy**: Call DentalGPT + OralGPT for overall analysis, GPT/Gemini for focused analysis
- **Consensus**: ≥2 VLMs agree AND no other VLM contradicts → accept finding

**CRITICAL - VLM Analysis with Target FDI or Quadrant:**

**MANDATORY WORKFLOW** for tooth-level or quadrant-level VLM analysis:
1. **FIRST** call `get_annotated_image` with `target_fdi` or `target_quadrant` to get the cropped/annotated image
2. **WAIT** for the tool to return the image path (e.g., `{"image_path": "/tmp/xxx.png"}`)
3. **THEN** (in the NEXT iteration) call VLM tools with:
   - `image_path` = the returned temp file path from step 2
   - `analysis_level` = "tooth" or "quadrant" accordingly

**DO NOT**:
- Call VLM tools in PARALLEL with `get_annotated_image` (the image doesn't exist yet!)
- Use the original OPG path directly for tooth/quadrant analysis (VLM needs focused image)
- Skip `get_annotated_image` when analyzing specific teeth or quadrants

**Correct Example** (2 iterations):
```
Iteration 1: get_annotated_image(target_fdi="18", output_type="crop")
             → returns {"image_path": "/tmp/tooth_18_crop.png"}

Iteration 2: llm_zoo_openai(image_path="/tmp/tooth_18_crop.png", 
                            analysis_level="tooth",
                            custom_prompt="Is this tooth impacted? What direction?")
```

**Image Type Selection**:
- `output_type="crop"`: Cropped image of specific tooth/quadrant (better for focused analysis)
- `output_type="bbox_overlay"`: Full OPG with bounding box highlight (better for context)
- **DentalGPT/OralGPT prefer**: `bbox_overlay` (they work better with full OPG context)
- **GPT-5.2/Gemini**: either works, but `crop` is more token-efficient

**Efficiency**: Max 5 iterations - plan tool calls carefully

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
  {"source": "GPT-5.2", "opinion": "implant at 46 site", "position_or_classification": "46"},
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

### Analysis Workflow (within 5 iterations)
1. **Detection Phase**: Query tools for teeth, quadrants, statuses
2. **VLM Phase**: Get independent VLM analyses (__VLM_PHASE_DESC__)
3. **Summary Phase**: Count votes for each finding (__SUMMARY_THRESHOLD__)
   - **For each finding**: Count how many sources (__SOURCES_DESC__) report it
   - **Presence vote**: Does the source say "yes/present" or "no/absent"? (silence = abstain)
   - **If threshold say YES**: CONFIRMED, regardless of whether others say NO
4. **Report Phase**: Write natural language report with confirmed findings

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

4. **Other Findings** (if any and confirmed)
   - Bone loss description
   - Sinus/TMJ findings

### Final Rules
- Use FDI notation (two-digit strings: "38", "47")
- Answer in user's language
- OMIT uncertain findings
- Output ONLY the final natural language diagnostic report (no reasoning process in final answer)

Now analyze the user-provided OPG image."""
        prompt = prompt.replace("__CONSENSUS_BLOCK__", self._get_consensus_block())
        for k, v in ph.items():
            prompt = prompt.replace("__" + k + "__", v)
        return prompt
    
    def _create_agent(self):
        """创建 LangChain agent"""
        react_config = self.config.get("react", {})
        max_iterations = react_config.get("max_iterations", 5)  # 最大 5 轮迭代
        
        # create_agent 第一个参数为 model（可传字符串或 BaseChatModel）
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
        
        # STEP 0: Preload detection cache by running run_all_detections at startup
        abs_image_path = str(Path(image_path).resolve())
        token = current_image_path_ctx.set(abs_image_path)
        try:
            logger.info(f"Preloading detection cache for: {abs_image_path}")
            from langchain_core.runnables import RunnableConfig
            preload_config = RunnableConfig(configurable={"current_image_path": abs_image_path})
            preload_result = self.toolkit.run_all_detections(abs_image_path, config=preload_config)
            import json
            preload_data = json.loads(preload_result)
            teeth_fdi_count = len(preload_data.get("teeth_fdi", {})) if isinstance(preload_data.get("teeth_fdi"), dict) else 0
            quadrant_count = len([k for k in preload_data.get("quadrants", {}).keys() if k not in ["error"]]) if isinstance(preload_data.get("quadrants"), dict) else 0
            logger.info(f"✓ Detection cache preloaded: {teeth_fdi_count} teeth FDI, {quadrant_count} quadrants")
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
        
        # 准备消息历史（从 memory 中获取）
        chat_history = []
        for call in memory.tool_calls:
            # 添加工具调用和结果到历史
            chat_history.append({
                "role": "assistant",
                "content": f"调用工具: {call['tool_name']}"
            })
            chat_history.append({
                "role": "user",
                "content": f"工具结果: {call['tool_output']}"
            })
        
        # 调用 agent
        try:
            # create_agent 期望的输入格式：{"messages": [HumanMessage, ...]}
            # 构建消息列表
            messages = [user_message]
            
            # 如果有 chat_history，转换为消息格式
            from langchain_core.messages import AIMessage
            for hist in chat_history:
                role = hist.get("role", "")
                content = hist.get("content", "")
                if role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))
            
            # 调用 agent，传入当前图像路径供工具从 config 注入（避免 LLM 传入错误路径）
            abs_image_path = str(Path(image_path).resolve())
            run_config = {"configurable": {"current_image_path": abs_image_path}}
            
            # 重置 token tracker
            self.token_tracker.total_tokens = 0
            self.token_tracker.total_input_tokens = 0
            self.token_tracker.total_output_tokens = 0
            self.token_tracker.call_count = 0
            self.token_tracker.per_call = []
            
            token = current_image_path_ctx.set(abs_image_path)
            try:
                # 带截断指数退避重试处理 429（配额/限流）
                max_retries = 5
                initial_delay = 1.0
                max_delay = 60.0
                last_error = None
                for attempt in range(max_retries):
                    try:
                        # 设置 recursion_limit 以支持足够的工具调用迭代
                        react_config = self.config.get("react", {})
                        max_iterations = react_config.get("max_iterations", 15)
                        result = self.agent.invoke(
                            {"messages": messages},
                            config={
                                **run_config,
                                "callbacks": [self.token_tracker],
                                "recursion_limit": max_iterations * 2 + 5  # 每次迭代可能有 2 步（AI + Tool）
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
                                "LLM 调用 429/限流（尝试 %d/%d），%.1f 秒后重试: %s",
                                attempt + 1, max_retries, delay, str(e)[:200]
                            )
                            time.sleep(delay)
                            continue
                        raise

                # 检查 token 上限（仅对 gpt-5.2 和 gemini-3-flash）
                if self.token_tracker.is_limit_exceeded():
                    logger.warning(
                        f"Token 使用量超过上限: {self.token_tracker.total_tokens}/{self.token_tracker.token_limit}"
                    )
            finally:
                current_image_path_ctx.reset(token)
            
            # 记录完整的 ReAct 推理过程
            # 格式：每个迭代包含 Thought -> Action(s) -> Observation(s)
            react_trace = []  # 完整的 ReAct 追踪
            final_answer = ""
            
            if isinstance(result, dict) and "messages" in result:
                from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
                
                # 构建 tool_call_id -> ToolMessage 的映射
                tool_results = {}
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_call_id:
                            tool_results[tool_call_id] = {
                                "name": getattr(msg, "name", "unknown"),
                                "output": getattr(msg, "content", "")
                            }
                
                # 按顺序处理消息，构建完整的 ReAct trace
                iteration = 0
                for msg in result["messages"]:
                    if isinstance(msg, HumanMessage):
                        # 跳过用户消息（已经记录在 question 中）
                        continue
                    
                    if isinstance(msg, AIMessage):
                        iteration += 1
                        content = _message_content_to_str(getattr(msg, "content", ""))
                        tool_calls_in_msg = getattr(msg, "tool_calls", []) or []
                        
                        # 构建这一轮的 trace
                        step = {
                            "iteration": iteration,
                            "thought": content.strip() if content else "(no explicit thought)",
                            "actions": [],
                            "observations": []
                        }
                        
                        # 处理该 AIMessage 中的所有 tool_calls
                        for tc in tool_calls_in_msg:
                            tool_name = tc.get("name", "unknown")
                            tool_input = tc.get("args", {})
                            tool_call_id = tc.get("id")
                            
                            action_entry = {
                                "tool": tool_name,
                                "input": tool_input
                            }
                            step["actions"].append(action_entry)
                            
                            # 获取对应的 observation
                            if tool_call_id and tool_call_id in tool_results:
                                tr = tool_results[tool_call_id]
                                observation = {
                                    "tool": tr["name"],
                                    "output": tr["output"]
                                }
                                step["observations"].append(observation)
                                
                                # 记录到 memory
                                memory.add_tool_call(
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                    tool_output=tr["output"],
                                    iteration=iteration,
                                    reasoning=content.strip() if content else ""
                                )
                        
                        # 记录最后一个有 content 的 AIMessage 作为候选 final_answer
                        if content and content.strip():
                            final_answer = content.strip()
                        
                        # 只有有 thought 或 actions 的步骤才记录
                        if step["thought"] != "(no explicit thought)" or step["actions"]:
                            react_trace.append(step)
            
            # 备用逻辑
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
            
            # 构建返回结果
            return {
                "answer": final_answer,
                "memory": memory,
                "react_trace": react_trace,  # 完整的 ReAct 追踪（Thought-Action-Observation）
                "tool_calls": memory.tool_calls,
                "token_usage": self.token_tracker.get_summary()
            }
        
        except Exception as e:
            logger.error(f"Agent 运行失败: {e}", exc_info=True)
            return {
                "answer": f"分析过程中出现错误: {str(e)}",
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
        流式运行 agent（生成器）
        
        Args:
            question: 用户问题
            image_path: OPG 图像路径
            memory: Memory 实例（可选）
            
        Yields:
            每个步骤的结果
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
        
        # 如果有 chat_history，转换为消息格式
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
            logger.error(f"Agent 流式运行失败: {e}", exc_info=True)
            yield {"error": str(e)}
