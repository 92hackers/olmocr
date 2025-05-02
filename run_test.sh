#! /usr/bin/env bash

# ref: https://github.com/allenai/olmocr/tree/main?tab=readme-ov-file#multi-node--cluster-usage
# olmocr, maintains a working queue of jobs to process, and a worker pool to process them.
# So, you can run this script on a single machine, or across a cluster, to process millions of pages simultaneously.

set -x

# Make sure the dependencies are correctly installed
bash ./check_deps.sh

# Kill any existing python processes, sglang .
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs kill -9

pdf_files=./tests/gnarly_pdfs/horribleocr.pdf
model_path=/root/.cache/modelscope/hub/models/allenai/olmOCR-7B-0225-preview

# --max_page_error_rate 0.01: means that if more than 1% of the pages in a document fail to process, the document will be skipped.
python -m olmocr.pipeline ./localworkspace \
  --pdfs $pdf_files \
  --max_page_retries 2 \
  --workers 2 \
  --model $model_path \
  --max_page_error_rate 0.01