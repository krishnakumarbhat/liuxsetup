#!/bin/bash

# Set cache to local folder
export HF_HOME=$(pwd)/temp_cache

# Run Server with BOTH model and vision adapter
python3 -m llama_cpp.server \
  --model "/media/pope/projecteo/github_proj/sam_pdf/gguf_model_qwen/Qwen3-VL-2b-Thinking-Q4_K_M.gguf" \
  --clip_model_path "/media/pope/projecteo/github_proj/sam_pdf/gguf_model_qwen/mmproj-BF16.gguf" \
  --n_gpu_layers 28 \
  --n_ctx 4096 \
  --host 0.0.0.0 \
  --port 8000
