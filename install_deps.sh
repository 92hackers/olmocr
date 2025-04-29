#! /usr/bin/env bash

#set -e

# Test if numpy version issues exists.
python -c 'from transformers.generation.utils import *'
python -c 'import transformers.models.clip.modeling_clip'

pip uninstall numpy scipy -y

# Intall gpu sglang deps.
pip install -e '.[gpu]' --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

pip install scipy==1.15.2
pip install numpy==1.26.4