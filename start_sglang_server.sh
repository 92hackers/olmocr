#! /usr/bin/env bash

# 1. Please make sure sglang is installed and the model is downloaded.
# 2. Please make sure the model is downloaded and the path is correct.

# Server arguments ref: https://docs.sglang.ai/backend/server_arguments.html

set -ex

# Make sure numpy version installed correctly.
# Otherwise, you should DELETE numpy related dirs in `site-packages` and reinstall.
python -c 'from transformers.generation.utils import *'
python -c 'import transformers.models.clip.modeling_clip'
python -c 'from numpy.lib.function_base import *'

model=/home/viseem/.cache/modelscope/hub/models/allenai/olmOCR-7B-0225-preview

SGLANG_SERVER_PORT=30024

# --mem-fraction-static 0.8 : # For less-than 60G gpu memory.
python3 -m sglang.launch_server \
  --model-path $model \
  --context-length 32000 \
  --chat-template qwen2-vl \
  --mem-fraction-static 0.8 \
  --host localhost \
  --port $SGLANG_SERVER_PORT
