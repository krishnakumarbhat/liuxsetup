#!/bin/bash
set -euo pipefail

# Optional overrides:
#   export VLM_MODEL_ID="unsloth/Qwen3-VL-2B-Thinking-bnb-4bit"
#   export VLM_HOST=0.0.0.0
#   export VLM_PORT=8000

export VLM_MODEL_ID="${VLM_MODEL_ID:-/media/pope/projecteo/github_proj/sam_pdf/Qwen3-VL-2B-Thinking-bnb-4bit}"
export VLM_HOST="${VLM_HOST:-0.0.0.0}"
export VLM_PORT="${VLM_PORT:-8000}"

python3 -m uvicorn vlm_server:app --host "$VLM_HOST" --port "$VLM_PORT"
