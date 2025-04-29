#! /usr/bin/env bash

set -ex

bash ./check_deps.sh

python -m olmocr.pipeline ./localworkspace \
  --pdfs tests/gnarly_pdfs/horribleocr.pdf \
  --max_page_retries 2 \
  --workers 2 \