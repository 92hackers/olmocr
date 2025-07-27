#! /usr/bin/env bash

set -ex

rm -rf ./localworkspace

# Resume
# python -m olmocr.pipeline ./localworkspace --markdown --port 8000 --pdfs /media/viseem/work_1t/doehler_dataset/hr-new/Kezhen-Guo_202501.pdf

python -m olmocr.pipeline ./localworkspace --markdown --port 8000 --pdfs ./tsinghua-math.pdf
