# OPGAgent

A ReAct-style Agent for OPG (panoramic dental radiograph) diagnosis. Given an uploaded OPG image, the Agent runs a self-directed 4-phase workflow — full-image overview → quadrant-level → single-tooth focus + historical retrieval → consensus voting — and produces a structured diagnostic report.

This repository contains **just the Agent source code**. The detection microservices (YOLO, MaskDINO/TVEM, MedSAM, OralGPT, DentalGPT, MedImageInsights RAG) are expected to be running separately; the Agent talks to them over HTTP using `tools_config.yaml`.

## Quick start

```bash
pip install uv
uv sync
cp .env.example .env      # add OPENAI_API_KEY + (optional) GEMINI_API_KEY / ANTHROPIC_API_KEY

# CLI: single image
uv run python -m agent_v3.cli \
  --question "Analyze this OPG and provide a diagnostic report" \
  --image_path /path/to/opg.png \
  --output result.json

# Python API
python - <<'PY'
from agent_v3.agent import OPGReActAgent
agent = OPGReActAgent(tool_service="default")
result = agent.run(question="Analyze this OPG…", image_path="/path/to/opg.png")
print(result["answer"])
PY
```

## Profiles

The Agent supports two consensus profiles via `OPG_PROFILE` env var (or `local_vlm_only=True/False` in Python):

| Profile | VLMs used | Consensus sources |
|---|---|---|
| **cloud** (default) | GPT-5.4 + Gemini 3 Flash + Claude Opus 4.6 | 3 VLMs + Tool + RAG (5 total, ≥3/5) |
| **local** | GPT-5.4 + DentalGPT + OralGPT | 3 VLMs + Tool + RAG (5 total, ≥3/5) |

Both profiles use the same combined region tools (`analyze_with_*`, `retrieve_similar_cases`) that wrap image preparation + VLM call in a single step.

## Project layout

```
src/agent_v3/
├── agent.py                  # Main ReAct loop, system prompt, consensus block
├── tools/
│   ├── dental_tools.py       # Tool registry + ServicePool queue scheduler
│   └── coordinate_utils.py   # IoU / FDI / geometry utilities
├── config/
│   ├── agent_config.yaml     # Agent model selection, iteration caps
│   ├── tools_config.yaml     # Service endpoints, load balancing
│   └── schema_enum_standard.md  # Structured JSON schema (FDI notation)
├── cli.py                    # Single-image CLI
├── cli_vqa.py                # VQA benchmark CLI
├── memory.py                 # ReAct trace recording
└── vqa_runner.py             # Batch VQA evaluation
```

## Tool inventory (combined region tools)

Per analysis target (tooth / quadrant / full OPG), the Agent dispatches:

| Tool | Underlying model | Image format used internally |
|---|---|---|
| `analyze_with_gpt`        | GPT-5.4              | tooth/quadrant crop |
| `analyze_with_gemini`     | Gemini 3 Flash Preview | tooth/quadrant crop |
| `analyze_with_claude`     | Claude Opus 4.6      | tooth/quadrant crop |
| `analyze_with_dentalgpt`  | DentalGPT (Qwen2.5-VL) | full OPG + bbox overlay |
| `analyze_with_oralgpt`    | OralGPT (Qwen2.5-VL)   | full OPG + bbox overlay |
| `retrieve_similar_cases`  | MedImageInsights RAG | tooth crop |

All six share the same input schema: `target_type` ∈ `{tooth, quadrant, overall}`, `target_id` (FDI like `36` or `Q1`-`Q4`, optional for overall).

## Output schema

Reports conform to `src/agent_v3/config/schema_enum_standard.md` — FDI-indexed structured JSON:

```json
{
  "dentition_summary": { "not_detected_fdi": ["16", "25"] },
  "teeth": {
    "46": { "status": "implant" },
    "36": { "status": "impacted", "winters_class": "mesial" }
  },
  "periodontium": { "severity": "moderate", "bone_loss_pattern": "horizontal" }
}
```

## Requirements

- Python 3.12
- Running microservices (see the parent deployment repo for `api_service/` setup)
- API keys:
  - `OPENAI_API_KEY` (always)
  - `GEMINI_API_KEY` (cloud profile)
  - `ANTHROPIC_API_KEY` (cloud profile)

## License

Internal use only.
