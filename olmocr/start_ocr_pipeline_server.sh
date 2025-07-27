#! /usr/bin/env bash

set -ex

uvicorn ocr_pipeline_server:app --port 18000
