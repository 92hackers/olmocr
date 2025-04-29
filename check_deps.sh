#! /usr/bin/env bash

set -ex

python -c 'from transformers.generation.utils import *'
python -c 'import transformers.models.clip.modeling_clip'
python -c 'from numpy.lib.function_base import *'
