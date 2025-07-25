#! /usr/bin/env bash

# 1. Please make sure vllm is installed and the model is downloaded.
# 2. Please make sure the model is downloaded and the path is correct.

# Vllm doc: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server

set -ex

# vllm will depends on pytorch-2.7.1, which now use numpy-2.xx.

model=/home/viseem/.cache/modelscope/hub/models/allenai/olmOCR-7B-0225-preview

VLLM_SERVER_PORT=30024

# --mem-fraction-static 0.8 : # For less-than 60G gpu memory.
vllm serve $model --port $VLLM_SERVER_PORT \
  --disable-log-requests \
  --uvicorn-log-level warning \
  --served-model-name olmocr \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32000
