#! /usr/bin/env bash

# ref: https://github.com/allenai/olmocr/tree/main?tab=readme-ov-file#multi-node--cluster-usage
# olmocr, maintains a working queue of jobs to process, and a worker pool to process them.
# So, you can run this script on a single machine, or across a cluster, to process millions of pages simultaneously.

set -x


#pdf_files=./olmocr_error_processing_files_20250503_160808.log
pdf_files=/home/cy/projects/large-language-model/open-webui/backend/extract_errors_pdf_filepaths_20250510_225826.log

# Make sure the dependencies are correctly installed
#bash ./check_deps.sh

localworkspace=./localworkspace_chatgpt

error_rate=0.5

# Clear localworkspace to discard any previous cache.
rm -rf $localworkspace

# --max_page_error_rate 0.01: means that if more than 1% of the pages in a document fail to process, the document will be skipped.
python -m olmocr.pipeline_chatgpt $localworkspace \
  --pdfs $pdf_files \
  --max_page_retries 8 \
  --workers 8 \
  --max_page_error_rate $error_rate
