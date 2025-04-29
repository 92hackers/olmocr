#! /usr/bin/env bash

#set -e

# Test if numpy version issues exists.
python -c 'from transformers.generation.utils import *'

pip uninstall numpy

# Intall gpu sglang deps.
pip install -e '.[gpu]' --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
