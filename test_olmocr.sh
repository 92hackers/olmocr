#! /usr/bin/env bash

set -ex

python -c 'from transformers.generation.utils import *'
python -c 'import transformers.models.clip.modeling_clip'

python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/horribleocr.pdf
