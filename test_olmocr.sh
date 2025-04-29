#! /usr/bin/env bash

set -ex

# Make sure the dependencies are correctly installed
bash ./check_deps.sh

# Kill any existing python processes
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

pdf_files=./tests/gnarly_pdfs/horribleocr.pdf

python -m olmocr.pipeline ./localworkspace \
  --pdfs $pdf_files \
  --max_page_retries 2 \
  --workers 2 \