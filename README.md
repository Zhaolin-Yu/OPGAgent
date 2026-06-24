# OPGAgent: Auditable Dental Panoramic X-ray Interpretation

[![arXiv](https://img.shields.io/badge/arXiv-2603.00462-b31b1b.svg)](https://arxiv.org/abs/2603.00462)
[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026%20Accepted-1f6feb.svg)](https://conferences.miccai.org/2026/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![Model weights](https://img.shields.io/badge/%F0%9F%A4%97-Model%20weights-ff9d00.svg)](#models--weights)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

> 🎉 **Accepted at MICCAI 2026.**
> 📄 Paper: [**OPGAgent** (arXiv)](https://arxiv.org/abs/2603.00462)

OPGAgent is a fully autonomous dental **OPG (orthopantomogram / panoramic radiograph)**
diagnostic agent: a reasoning LLM drives a ReAct loop over a set of vision tools and writes a
consensus diagnostic report (natural language + structured JSON).

---

## Repository layout

```
OPGAgent/
├── src/opgagent/                  # the agent (Python package `opgagent`)
│   ├── agent.py                   # ReAct loop + system prompt
│   ├── cli.py                     # command-line entry point
│   ├── tools/dental_tools.py      # tools + DentalToolkit
│   └── config/                    # agent_config.yaml, schema_enum_standard.md
├── api_service/                   # 5 backend model services (FastAPI, per-service uv venv)
│   ├── yolo_enumeration/          # tooth enumeration  (weight bundled)
│   ├── tvem/                      # MaskDINO detection (weights from HF)
│   ├── medsam/                    # segmentation       (weight from HF)
│   ├── oral_gpt/  dental_gpt/     # VLM voters         (weights from HF)
│   ├── runner/unified_runner.py   # service orchestrator
│   └── start_all_services.sh / stop_all_services.sh / check_health.sh
├── scripts/                       # batch inference & evaluation helpers
└── tests/
```

---

## Dependencies

Everything is managed with **[uv](https://github.com/astral-sh/uv)** (one venv for the agent,
plus an isolated venv per backend service). Key libraries:

| Component | Main libraries |
|---|---|
| **Agent** (`src/opgagent`) | `langchain==1.2.7`, `langchain-openai`, `langchain-google-genai`, `pyyaml`, `pillow`, `requests` |
| **yolo_enumeration** | `ultralytics`, `fastapi`, `uvicorn` |
| **tvem** (MaskDINO) | `torch`, `torchvision`, `detectron2` (+ vendored MaskDINO), `fastapi` |
| **medsam** | `segment-anything`, `torch`, `fastapi` |
| **dental_gpt / oral_gpt** | `transformers` (Qwen2.5-VL), `qwen-vl-utils`, `torch`, `torchvision`, `fastapi` |

```bash
uv sync                                  # agent venv
cd api_service && bash setup_all_envs.sh # one venv per service
```

GPU: ≥1 NVIDIA GPU; DentalGPT/OralGPT need ~16–20 GB each, YOLO/MedSAM/TVEM are lighter.
GPT and Gemini run via cloud APIs (no local GPU).

---

## Models & weights

`yolo_enumeration` weights are **bundled in this repo** (committed directly, ~50 MB). The other four model
families are **not committed** — download them from Hugging Face and point each service at
them with an environment variable.

| Service | Weight source | Size | Env var to set |
|---|---|---|---|
| **yolo_enumeration** | **Bundled**: `api_service/yolo_enumeration/model/best.pt` | ~50 MB | — (in repo) |
| **TVEM** (MaskDINO) | [🤗 Bryceee/Teeth_Visual_Experts_Models](https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models) | ~2.5 GB × 4 | `TVEM_WEIGHTS_DIR` |
| **MedSAM** | [🤗 wanglab/medsam-vit-base](https://huggingface.co/wanglab/medsam-vit-base) (orig. `medsam_vit_b.pth`, [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)) | ~360 MB | `MEDSAM_MODEL_PATH` |
| **DentalGPT** | [🤗 Eric3200/DentalGPT-7B-1026](https://huggingface.co/Eric3200/DentalGPT-7B-1026) | ~16 GB | `DENTAL_GPT_MODEL_PATH` |
| **OralGPT** | [🤗 OralGPT/OralGPT-Omni-7B-Instruct](https://huggingface.co/OralGPT/OralGPT-Omni-7B-Instruct) | ~16 GB | `ORAL_GPT_MODEL_PATH` |

Every model path is **environment-overridable**; defaults are the HF repo id / in-repo path,
so a fresh clone resolves to a downloadable model even with nothing set:

| Env var | Default |
|---|---|
| `YOLO_ENUM_MODEL_PATH` | `model/best.pt` (bundled) |
| `TVEM_WEIGHTS_DIR` / `TVEM_CONFIG_DIR` / `TVEM_CATEGORY_DIR` | `weights` / `configs` / `categories` |
| `MEDSAM_MODEL_PATH` | bundled `medsam_vit_b.pth` path |
| `DENTAL_GPT_MODEL_PATH` / `DENTAL_GPT_PROCESSOR` | `Eric3200/DentalGPT-7B-1026` / `Qwen/Qwen2.5-VL-7B-Instruct` |
| `ORAL_GPT_MODEL_PATH` | `OralGPT/OralGPT-Omni-7B-Instruct` |

Download example (repeat per model):

```bash
huggingface-cli download Eric3200/DentalGPT-7B-1026 --local-dir ./weights/dental_gpt
huggingface-cli download OralGPT/OralGPT-Omni-7B-Instruct --local-dir ./weights/oral_gpt
huggingface-cli download Bryceee/Teeth_Visual_Experts_Models --local-dir ./weights/tvem
```

---

## Configuration: `.env` & API keys

The agent reads keys and model paths from environment variables. Copy the template and fill it
in (or `export` the variables in your shell):

```bash
cp .env.example .env
```

A complete `.env` looks like:

```dotenv
# --- LLM API keys ---
OPENAI_API_KEY=sk-...          # official OpenAI; brain (GPT-5.2) + llm_zoo_openai voter
GEMINI_API_KEY=AIza...         # Google Gemini; llm_zoo_google voter
# IMPORTANT: leave OPENAI_BASE_URL UNSET to hit the official api.openai.com
# OPENAI_BASE_URL=             # set only if you use an OpenAI-compatible relay

# --- local model paths (point at your downloaded HF weights) ---
TVEM_WEIGHTS_DIR=/abs/path/weights/tvem
MEDSAM_MODEL_PATH=/abs/path/medsam_vit_b.pth
DENTAL_GPT_MODEL_PATH=/abs/path/weights/dental_gpt
ORAL_GPT_MODEL_PATH=/abs/path/weights/oral_gpt
# YOLO_ENUM_MODEL_PATH defaults to the bundled weight — no need to set

# --- optional: LangSmith tracing ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=opgagent
```

- **OpenAI / Gemini keys are required** (the brain and 2 of the 4 VLM voters are cloud models).
- To load `.env` into your shell: `set -a; source .env; set +a`.
- The LLM profile (`provider`, `model`, `max_iterations`) is in
  `src/opgagent/config/agent_config.yaml` — default brain is `gpt-5.2`.

---

## Running

**1) Start the backend services** (assign GPUs freely; ports default to the 6600 range):

```bash
cd api_service
GPU=0 PORT=6600 bash yolo_enumeration/start_api.sh   &
GPU=1 PORT=6602 bash tvem/start_api.sh               &
GPU=1 PORT=6603 bash medsam/start_api.sh             &
GPU=2 PORT=6604 bash oral_gpt/start_api.sh           &
GPU=3 PORT=6608 bash dental_gpt/start_api.sh         &
bash check_health.sh        # wait until all report healthy
```

**2) Run the agent** on one OPG:

```bash
uv run python -m opgagent.cli \
  --question "Analyze this OPG and give a concise diagnostic summary." \
  --image_path /path/to/opg.png \
  --tool-service api_service \
  --output result.json \
  --structured           # also emit the structured JSON report
```

Or from Python:

```python
from opgagent.agent import OPGReActAgent
agent = OPGReActAgent(config_path="src/opgagent/config/agent_config.yaml",
                      tool_service="api_service")
result = agent.run(question="Analyze this OPG ...", image_path="/path/to/opg.png")
print(result["answer"])

# optional: convert the natural-language report to structured JSON
import json
structured = json.loads(agent.toolkit.convert_to_structured(result["answer"]))["structured_report"]
```

**Output:** a natural-language diagnostic report (always). With `--structured`, the report is
additionally converted to a **structured JSON** via `convert_to_structured` (schema:
`src/opgagent/config/schema_enum_standard.md`) — printed, and saved as `structured_report.json`
when `--output` is set.

---

## Citation

If you use OPGAgent, please cite:

```bibtex
@inproceedings{opgagent2026,
  title     = {OPGAgent: An Agent for Auditable Dental Panoramic X-ray Interpretation},
  author    = {Zhaolin Yu and Litao Yang and Ben Babicka and Ming Hu and Jing Hao and Anthony Huang and James Huang and Yueming Jin and Jiasong Wu and Zongyuan Ge},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
  note      = {arXiv:2603.00462}
}
```

---

## Acknowledgements

- **TVEM** — Teeth Visual Experts Models (MaskDINO) · [Bryceee/Teeth_Visual_Experts_Models](https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models)
- **MedSAM** — [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) · [wanglab/medsam-vit-base](https://huggingface.co/wanglab/medsam-vit-base)
- **DentalGPT** · [Eric3200/DentalGPT-7B-1026](https://huggingface.co/Eric3200/DentalGPT-7B-1026)
- **OralGPT** · [OralGPT/OralGPT-Omni-7B-Instruct](https://huggingface.co/OralGPT/OralGPT-Omni-7B-Instruct)
- Built on [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and LangChain.

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
