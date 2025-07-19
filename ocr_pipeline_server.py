# ocr_pipeline_server.py

# This is a FastAPI server that receives a PDF file, saves it to a temporary file,
# and then runs the OCR pipeline on it.
# It returns the OCR result as a JSON object.

import os
import sys
import hashlib

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from olmocr.pipeline_api import main as pipeline_main

app = FastAPI(title="Olmocr Pipeline Server")

SGLANG_SERVER_URL = "http://localhost:30024"

@app.get("/health")
def get_status():
    return {"ok": 1}

@app.post("/ocr")
async def handle_ocr_pipeline(file: UploadFile = File(...)):
    """
    接收单个文件并进行OCR处理，返回OCR 内容提取结果
    """
    filename = os.path.basename(file.filename)
    content_type = file.content_type
    if not content_type:
        if filename.endswith('.pdf'):
            content_type = 'application/pdf'
        else:
            return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

    content = await file.read()
    if not content:
        return JSONResponse({"error": "File is empty"}, status_code=400)

    file_sha1 = hashlib.sha1(content).hexdigest()
    save_path = f"/tmp/{file_sha1}_{filename}"
    txt_save_path = f"/tmp/{file_sha1}_{filename}.txt"
    print(f"save_path: {save_path}")

    # Save the file to the temporary directory if it doesn't exist
    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            f.write(content)

    # Run the OCR pipeline
    sys.argv = [
        "olmocr.pipeline_api",
        "--sglang_server_url", SGLANG_SERVER_URL,
        "--pdfs", save_path,
        "--workers", "1",
        "--max_page_retries", "8",
        "--max_page_error_rate", "0.5",
    ]

    await pipeline_main()

    # Wait for the pipeline to finish
    txt_content = ""

    if os.path.exists(txt_save_path):
        with open(txt_save_path, "r", encoding="utf-8") as f:
            txt_content = f.read()

    return JSONResponse({
        "filename": filename,
        "content_type": content_type,
        "size": len(content),
        "save_path": save_path,
        "txt_save_path": txt_save_path,
        "txt_content": txt_content,
    })
