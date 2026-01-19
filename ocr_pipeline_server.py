# ocr_pipeline_server.py

# This is a FastAPI server that receives a PDF file,
# and then runs the PaddleOCR API on it.
# It returns the OCR result as a JSON object.

import os
import base64
import time
import logging
import requests

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="PaddleOCR Pipeline Server")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PADDLEOCR_API_URL = os.getenv("PADDLEOCR_API_URL", "https://m1p7w4i2u1w7q4o1.aistudio-app.com/layout-parsing")
PADDLEOCR_TOKEN = os.getenv("PADDLEOCR_TOKEN", "")

FILE_TYPE_MAP = {"pdf": 0, "image": 1}


def format_file_size(size_bytes: int) -> str:
    """
    Convert file size in bytes to human-readable format.
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def log_request_info(filename: str, content_type: str, file_size: int) -> None:
    """
    Log incoming request file information.
    """
    logger.info("=" * 60)
    logger.info("Incoming OCR Request")
    logger.info(f"  Filename:     {filename}")
    logger.info(f"  Content-Type: {content_type}")
    logger.info(f"  File Size:    {file_size} bytes ({format_file_size(file_size)})")
    logger.info("=" * 60)


def log_ocr_summary(filename: str, duration_seconds: float, extracted_length: int) -> None:
    """
    Log OCR processing summary.
    """
    logger.info("=" * 60)
    logger.info("OCR Processing Summary")
    logger.info(f"  Filename:        {filename}")
    logger.info(f"  Duration:        {duration_seconds:.2f} seconds")
    logger.info(f"  Extracted Chars: {extracted_length}")
    logger.info("=" * 60)


@app.get("/health")
def get_status():
    return {"ok": 1}

@app.post("/ocr")
async def handle_ocr_pipeline(file: UploadFile = File(...)):
    """
    接收单个文件并进行OCR处理，返回OCR 内容提取结果
    """
    start_time = time.time()
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

    # Log incoming request info
    log_request_info(filename, content_type, len(content))

    # Encode file to base64
    file_data = base64.b64encode(content).decode("ascii")

    # Determine file type
    file_type = FILE_TYPE_MAP["pdf"] if filename.endswith(".pdf") else FILE_TYPE_MAP["image"]

    # Prepare request headers and payload
    headers = {
        "Authorization": f"token {PADDLEOCR_TOKEN}",
        "Content-Type": "application/json"
    }

    required_payload = {
        "file": file_data,
        "fileType": file_type,
    }

    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }

    payload = {**required_payload, **optional_payload}

    # Call PaddleOCR API
    try:
        response = requests.post(PADDLEOCR_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()["result"]
    except requests.RequestException as e:
        return JSONResponse({"error": f"PaddleOCR API request failed: {str(e)}"}, status_code=500)
    except KeyError:
        return JSONResponse({"error": "Invalid response from PaddleOCR API"}, status_code=500)

    # Extract and concatenate markdown content from all parsing results
    txt_content_parts = []
    if result.get("layoutParsingResults"):
        for res in result["layoutParsingResults"]:
            markdown_text = res.get("markdown", {}).get("text", "")
            if markdown_text:
                txt_content_parts.append(markdown_text)
    txt_content = "\n\n".join(txt_content_parts)

    end_time = time.time()
    log_ocr_summary(filename, end_time - start_time, len(txt_content))

    return JSONResponse({
        "filename": filename,
        "content_type": content_type,
        "size": len(content),
        "txt_content": txt_content,
    })
