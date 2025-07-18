# upload_client.py
import requests
import sys

# 1. 要上传的文件路径
file_path = '/home/cy/Downloads/doehler_resumes/internal/pdfs/DS0514_Eric Xu_202505.pdf'

# 2. 构造 multipart/form-data 请求
url = "http://127.0.0.1:8000/ocr"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f)}
    resp = requests.post(url, files=files)

# 3. 打印结果
print("Status:", resp.status_code)
print("Response:", resp.json())
