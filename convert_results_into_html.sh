#! /usr/bin/env bash

set -ex

python -m olmocr.viewer.dolmaviewer localworkspace/results/output_*.jsonl
