#! /usr/bin/env bash

set -ex

bash ./check_deps.sh

pdf_files=./tests/gnarly_pdfs/horribleocr.pdf

python -m olmocr.pipeline ./localworkspace \
  --pdfs $pdf_files \
  --max_page_retries 2 \
  --workers 2 \