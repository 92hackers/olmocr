#! /usr/bin/env bash

# ref: https://github.com/allenai/olmocr/tree/main?tab=readme-ov-file#multi-node--cluster-usage
# olmocr, maintains a working queue of jobs to process, and a worker pool to process them.
# So, you can run this script on a single machine, or across a cluster, to process millions of pages simultaneously.

set -x

SGLANG_SERVER_HOST=localhost
SGLANG_SERVER_PORT=30024

sglang_server_url=http://$SGLANG_SERVER_HOST:$SGLANG_SERVER_PORT

pdf_files=./tests/gnarly_pdfs/horribleocr.pdf

# Make sure the dependencies are correctly installed
bash ./check_deps.sh

localworkspace=./localworkspace

# Clear localworkspace to discard any previous cache.
rm -rf $localworkspace

# --max_page_error_rate 0.01: means that if more than 1% of the pages in a document fail to process, the document will be skipped.
python -m olmocr.pipeline $localworkspace \
  --sglang_server_url $sglang_server_url \
  --pdfs $pdf_files \
  --max_page_retries 8 \
  --workers 8 \
  --max_page_error_rate 0.5
