import argparse
import asyncio
import base64
import datetime
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import cache, partial
from io import BytesIO
from urllib.parse import urlparse

import boto3
import httpx
from botocore.exceptions import ClientError
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from olmocr.check import (
    check_poppler_version,
)
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter.filter import Language, PdfFilter
from olmocr.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
from olmocr.metrics import MetricsKeeper, WorkerTracker
from olmocr.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from olmocr.s3_utils import (
    download_zstd_csv,
    expand_s3_glob,
    get_s3_bytes,
    get_s3_bytes_with_backoff,
    parse_s3_path,
)
from olmocr.version import VERSION
from olmocr.work_queue import LocalWorkQueue, S3WorkQueue, WorkQueue

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

file_handler = logging.FileHandler("olmocr-pipeline-debug.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global s3 clients fo the whole script, we have two separate ones in case your workspace and your pdfs are in different accounts
workspace_s3 = boto3.client("s3")
pdf_s3 = boto3.client("s3")

# Global variables for token statistics
metrics = MetricsKeeper(window=60 * 5)
tracker = WorkerTracker()

# Filter object, cached so it will only get loaded when/if you need it
get_pdf_filter = cache(lambda: PdfFilter(languages_to_keep={Language.ENGLISH, None}, apply_download_spam_check=True, apply_form_check=True))

# Specify the SGLang server URL
SGLANG_SERVER_URL = ''

# Record error processing files
ERROR_PROCESING_FILES_LOG_FILE = "olmocr_error_processing_files"

# Page delimiter, to concatenate the pages together.
PAGE_DELIMITER = "\n-\n"

# Global timestamp for logging
global_timestamp = time.strftime("%Y%m%d_%H%M%S")

# Global error-processing files counter
error_processing_files_counter = 0


def log_error_processing_file(file_path):
    """
    Log the file path of a PDF that failed to process.
    All error processing files will be logged to a single file.
    """
    print(f"Logging error processing file: {file_path}")
    error_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", f"{ERROR_PROCESING_FILES_LOG_FILE}_{global_timestamp}.log")
    with open(error_file, "a") as f:
        f.write(f"{file_path}\n")
    global error_processing_files_counter
    error_processing_files_counter += 1


@dataclass(frozen=True)
class PageResult:
    s3_path: str
    page_num: int
    response: PageResponse

    input_tokens: int
    output_tokens: int
    is_fallback: bool


async def build_page_query(local_pdf_path: str, page: int, target_longest_image_dim: int, target_anchor_text_len: int, image_rotation: int = 0) -> dict:
    """
    Build llm request body for single page.
    convert page content into an image firstly.
    """
    MAX_TOKENS = 3000
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    # Allow the page rendering to process in the background while we get the anchor text (which blocks the main thread)
    image_base64 = asyncio.to_thread(render_pdf_to_base64png, local_pdf_path, page, target_longest_image_dim=target_longest_image_dim)
    anchor_text = asyncio.to_thread(get_anchor_text, local_pdf_path, page, pdf_engine="pdfreport", target_length=target_anchor_text_len)

    image_base64, anchor_text = await asyncio.gather(image_base64, anchor_text)  # type: ignore
    if image_rotation != 0:
        image_bytes = base64.b64decode(image_base64)
        with Image.open(BytesIO(image_bytes)) as img:
            rotated_img = img.rotate(-image_rotation, expand=True)

            # Save the rotated image to a bytes buffer
            buffered = BytesIO()
            rotated_img.save(buffered, format="PNG")

        # Encode the rotated image back to base64
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # TODO: qwen2.5-vl-7b-instruct 更加强大，上下文更长，值得升级: total: 131,072, input: 129,024, output: 8192.
    return {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_finetuning_prompt(anchor_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }


# Manual simple implementation of HTTP Post
# It feels strange perhaps, but httpx and aiohttp are very complex beasts
# Ex. the sessionpool in httpcore has 4 different locks in it, and I've noticed
# that at the scale of 100M+ requests, that they deadlock in different strange ways
async def async_post(url, json_data):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or 80
    path = parsed_url.path or "/"

    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(json_payload)}\r\n"
            f"Connection: close\r\n\r\n"
            f"{json_payload}"
        )
        writer.write(request.encode())
        await writer.drain()

        # Read status line
        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        # Read response body
        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        else:
            raise ConnectionError("Anything other than fixed content length responses are not implemented yet")

        return status_code, response_body
    except Exception as e:
        # Pass through errors
        raise e
    finally:
        # But just make sure to close the socket on your way out
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass


# Actually requesting the page content to the SGLang server model api, to extract the anchor text and the page content
async def process_page(args, worker_id: int, pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    """
    Process a single page of a PDF document.
    """
    COMPLETION_URL = f"{SGLANG_SERVER_URL}/v1/chat/completions"
    MAX_RETRIES = args.max_page_retries
    TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    exponential_backoffs = 0
    local_anchor_text_len = args.target_anchor_text_len
    local_image_rotation = 0
    attempt = 0
    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "started")

    while attempt < MAX_RETRIES:
        query = await build_page_query(pdf_local_path, page_num, args.target_longest_image_dim, local_anchor_text_len, image_rotation=local_image_rotation)
        query["temperature"] = TEMPERATURE_BY_ATTEMPT[
            min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
        ]  # Change temperature as number of attempts increases to overcome repetition issues at expense of quality

        logger.info(f"Built page query for {pdf_orig_path}-{page_num}")
        response_body = None

        try:
            status_code, response_body = await async_post(COMPLETION_URL, json_data=query)

            if status_code == 400:
                raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
            elif status_code == 500:
                raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
            elif status_code != 200:
                raise ValueError(f"Error http status {status_code}")

            # Print response_body for debugging.
            # print(response_body)

            base_response_data = json.loads(response_body)

            if base_response_data["usage"]["total_tokens"] > args.model_max_context:
                local_anchor_text_len = max(1, local_anchor_text_len // 2)
                logger.info(f"Reducing anchor text len to {local_anchor_text_len} for {pdf_orig_path}-{page_num}")
                raise ValueError("Response exceeded model_max_context, cannot use this response")

            metrics.add_metrics(
                sglang_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                sglang_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
            )

            model_response_json = json.loads(base_response_data["choices"][0]["message"]["content"])
            page_response = PageResponse(**model_response_json)

            if not page_response.is_rotation_valid and attempt < MAX_RETRIES - 1:
                logger.info(
                    f"Got invalid_page rotation for {pdf_orig_path}-{page_num} attempt {attempt}, retrying with {page_response.rotation_correction} rotation"
                )
                local_image_rotation = page_response.rotation_correction
                raise ValueError(f"invalid_page rotation for {pdf_orig_path}-{page_num}")

            await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
            # Successfully processed the page, return the result.
            return PageResult(
                pdf_orig_path,
                page_num,
                page_response, # Page content.
                input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                output_tokens=base_response_data["usage"].get("completion_tokens", 0),
                is_fallback=False,
            )
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"Client error on attempt {attempt} for {pdf_orig_path}-{page_num}: {type(e)} {e}")

            # Now we want to do exponential backoff, and not count this as an actual page retry
            # Page retrys are supposed to be for fixing bad results from the model, but actual requests to sglang
            # are supposed to work. Probably this means that the server is just restarting
            sleep_delay = 10 * (2**exponential_backoffs)
            exponential_backoffs += 1
            logger.info(f"Sleeping for {sleep_delay} seconds on {pdf_orig_path}-{page_num} to allow server restart")
            await asyncio.sleep(sleep_delay)
        except asyncio.CancelledError:
            logger.info(f"Process page {pdf_orig_path}-{page_num} cancelled")
            await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "cancelled")
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt} for {pdf_orig_path}-{page_num}: {e}")
            attempt += 1
            # Debugging, view raw response
            print(f"----------------------------------------------------------------------------")
            print(f"Raw response: {response_body}")
            print(f"----------------------------------------------------------------------------")
        except ValueError as e:
            logger.warning(f"ValueError on attempt {attempt} for {pdf_orig_path}-{page_num}: {type(e)} - {e}")
            attempt += 1
            print(f"----------------------------------------------------------------------------")
            print(f"Raw response: {response_body}")
            print(f"----------------------------------------------------------------------------")
        except Exception as e:
            logger.exception(f"Unexpected error on attempt {attempt} for {pdf_orig_path}-{page_num}: {type(e)} - {e}")
            attempt += 1
            print(f"----------------------------------------------------------------------------")
            print(f"Raw response: {response_body}")
            print(f"----------------------------------------------------------------------------")

    logger.error(f"Failed to process {pdf_orig_path}-{page_num} after {MAX_RETRIES} attempts.")
    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")

    return PageResult(
        pdf_orig_path,
        page_num,
        PageResponse(
            natural_text=get_anchor_text(pdf_local_path, page_num, pdf_engine="pdftotext"),
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
        ),
        input_tokens=0,
        output_tokens=0,
        is_fallback=True,
    )


async def process_pdf(args, worker_id: int, pdf_orig_path: str):
    """
    Process a single PDF document, a document can be a single page or multiple pages.
    """
    # Check if txt file already exists in the same dir, If you still want to process it, you can delete the txt file firstly!
    if check_txt_file_exists(pdf_orig_path):
        logger.info(f"Found .txt file for: {pdf_orig_path}, skipping it")
        return None

    # Write the PDF to a temporary file, either a local file or an S3 file.
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
        try:
            data = await asyncio.to_thread(lambda: get_s3_bytes_with_backoff(pdf_s3, pdf_orig_path))
            tf.write(data)
            tf.flush()
        except ClientError as ex:
            if ex.response["Error"]["Code"] == "NoSuchKey":
                logger.info(f"S3 File Not found, skipping it completely {pdf_orig_path}")
                return None
            else:
                raise

        pdf_local_path = tf.name

        if is_png(pdf_local_path) or is_jpeg(pdf_local_path):
            logger.info(f"Converting {pdf_orig_path} from image to PDF format...")
            tf.seek(0)
            tf.write(convert_image_to_pdf_bytes(pdf_local_path))
            tf.flush()

        try:
            # Read the PDF file from the temporary file to count the number of pages
            reader = PdfReader(pdf_local_path)
            num_pages = reader.get_num_pages()
        except:
            logger.exception(f"Could not count number of pages for {pdf_orig_path}, aborting document")
            return None

        logger.info(f"Got {num_pages} pages to do for {pdf_orig_path} in worker {worker_id}")

        if args.apply_filter and get_pdf_filter().filter_out_pdf(pdf_local_path):
            logger.info(f"Filtering out pdf {pdf_orig_path}")
            return None

        # List to hold the tasks for processing each page
        page_tasks = []
        page_results = []

        try:
            async with asyncio.TaskGroup() as tg:
                # Iterate all pages and create tasks for each page.
                for page_num in range(1, num_pages + 1):
                    task = tg.create_task(process_page(args, worker_id, pdf_orig_path, pdf_local_path, page_num))
                    page_tasks.append(task)

            # Collect the results from the entire task group, assuming no exceptions
            page_results = [task.result() for task in page_tasks]

            num_fallback_pages = sum(page_result.is_fallback for page_result in page_results)

            # Check if the number of fallback pages exceeds the allowed error rate.
            if num_fallback_pages / num_pages > args.max_page_error_rate:
                logger.error(
                    f"Document {pdf_orig_path} has {num_fallback_pages} fallback pages out of {num_pages} exceeding max_page_error_rate of {args.max_page_error_rate}, discarding document."
                )
                # Log the error processing file
                log_error_processing_file(pdf_orig_path)
                return None
            elif num_fallback_pages > 0:
                logger.warning(
                    f"Document {pdf_orig_path} processed with {num_fallback_pages} fallback pages out of {num_pages}, proceeding to build Dolma document."
                )

            concate_doc_pages_and_save(pdf_orig_path, page_results)
            return None
            #return build_dolma_document(pdf_orig_path, page_results)
        except Exception as e:
            # Check for ExceptionGroup with BrokenProcessPool
            if isinstance(e, ExceptionGroup):
                broken_pool, other = e.split(BrokenProcessPool)
                if broken_pool is not None:  # Found at least one BrokenProcessPool
                    logger.critical("Encountered BrokenProcessPool, exiting process.")
                    sys.exit(1)

            logger.exception(f"Exception in process_pdf function for {pdf_orig_path}: {e}")
            # Log the error processing file
            log_error_processing_file(pdf_orig_path)
            # You can't build a dolma doc with even 1 failed page, so just get out of here
            # However, you don't want to propagate an exception higher up and cancel the entire work_group
            return None


def get_txt_file_path(file_path):
    """
    Get the text file path for a given PDF file path.
    """
    return os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}.txt")

def check_txt_file_exists(file_path):
    """
    Check if the text file exists for a given PDF file path.
    """
    txt_file_path = get_txt_file_path(file_path)
    return os.path.exists(txt_file_path)

def concate_doc_pages_and_save(pdf_orig_path, page_results):
    """
    Concatenate the pages together with a delimiter, and then save the document text.
    """
    document_text = ""
    for index, page in enumerate(page_results):
        if page.response.natural_text is not None:
            content = page.response.natural_text + ("\n" if index < len(page_results) - 1 else "")
        else:
            content = ""
        document_text += content + PAGE_DELIMITER if len(content) > 0 else "" # Only add the delimiter if there is actual content
    if not document_text:
        logger.info(f"No document text extracted for document: {pdf_orig_path}")
        # Record error file.
        log_error_processing_file(pdf_orig_path)
        return None  # Return None if the document text is empty
    # Save the document text to a file
    content_output_path = get_txt_file_path(pdf_orig_path)
    with open(content_output_path, 'w', encoding='utf-8') as f:
        f.write(document_text)
    logger.info(f"----------- Saved document text to {content_output_path} -----------")


def build_dolma_document(pdf_orig_path, page_results):
    # Build the document text content and page spans
    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, page_result in enumerate(page_results):
        if page_result.response.natural_text is not None:
            content = page_result.response.natural_text + ("\n" if index < len(page_results) - 1 else "")
        else:
            content = ""

        start_pos = current_char_pos
        document_text += content
        current_char_pos = len(document_text)
        pdf_page_spans.append([start_pos, current_char_pos, page_result.page_num])

    if not document_text:
        logger.info(f"No document text for {pdf_orig_path}")
        # Record error file.
        log_error_processing_file(pdf_orig_path)
        return None  # Return None if the document text is empty

    # Build the Dolma document
    metadata = {
        "Source-File": pdf_orig_path,
        "olmocr-version": VERSION,
        "pdf-total-pages": len(page_results),
        "total-input-tokens": sum(page.input_tokens for page in page_results),
        "total-output-tokens": sum(page.output_tokens for page in page_results),
        "total-fallback-pages": sum(page.is_fallback for page in page_results),
    }

    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    dolma_doc = {
        "id": id_,
        "text": document_text,
        "source": "olmocr",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {"pdf_page_numbers": pdf_page_spans},
    }
    return dolma_doc


async def worker(args, work_queue: WorkQueue, semaphore, worker_id):
    """
    Worker function to process work items from the work queue.
    """
    while True:
        # Wait until allowed to proceed
        await semaphore.acquire()

        work_item = await work_queue.get_work()

        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            semaphore.release()
            break

        logger.info(f"Worker {worker_id} processing work item {work_item.hash}")
        await tracker.clear_work(worker_id)

        try:
            async with asyncio.TaskGroup() as tg:
                dolma_tasks = [tg.create_task(process_pdf(args, worker_id, pdf)) for pdf in work_item.work_paths]
                logger.info(f"Created all tasks for {work_item.hash}")

            logger.info(f"Finished TaskGroup for worker on {work_item.hash}")

            dolma_docs = []
            result = None

            for task in dolma_tasks:
                try:
                    result = task.result()
                except Exception as e:
                    # some dolma doc creations may have failed
                    logger.error(f"Got exception {e} when processing work item {work_item.hash}, skipping this document")
                    logger.error(work_item.work_paths)
                if result is not None:
                    dolma_docs.append(result)

            logger.info(f"Got {len(dolma_docs)} docs for {work_item.hash}")

            if len(dolma_docs) > 0:
                # Write all Dolma documents to a single local temporary file in JSONL format
                with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
                    # Iterate over the dolma_docs and append each one to the file.
                    for doc in dolma_docs:
                        tf.write(json.dumps(doc))
                        tf.write("\n")
                    tf.flush()

                    # Define the output S3 path using the work_hash
                    output_final_path = os.path.join(args.workspace, "results", f"output_{work_item.hash}.jsonl")

                    if output_final_path.startswith("s3://"):
                        bucket, key = parse_s3_path(output_final_path)
                        workspace_s3.upload_file(tf.name, bucket, key)
                    else:
                        shutil.copyfile(tf.name, output_final_path)

                # Update finished token counts from successful documents
                metrics.add_metrics(
                    finished_input_tokens=sum(doc["metadata"]["total-input-tokens"] for doc in dolma_docs),
                    finished_output_tokens=sum(doc["metadata"]["total-output-tokens"] for doc in dolma_docs),
                )

            await work_queue.mark_done(work_item)
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")
        finally:
            semaphore.release()


async def sglang_server_ready():
    max_attempts = 10
    delay_sec = 1
    url = f"{SGLANG_SERVER_URL}/v1/models"

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)

                if response.status_code == 200:
                    logger.info("sglang server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception:
            logger.warning(f"Attempt {attempt}: Please wait for sglang server to become ready...")

        await asyncio.sleep(delay_sec)

    raise Exception("sglang server did not become ready after waiting.")


async def metrics_reporter(work_queue):
    while True:
        # Leading newlines preserve table formatting in logs
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)


async def main():
    parser = argparse.ArgumentParser(description="Manager for running millions of PDFs through a batch inference pipeline")
    parser.add_argument(
        "--pdfs",
        nargs="*",
        help="Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths",
        default=None,
    )
    parser.add_argument("--workspace_profile", help="S3 configuration profile for accessing the workspace", default=None)
    parser.add_argument("--pdf_profile", help="S3 configuration profile for accessing the raw pdf documents", default=None)
    parser.add_argument("--pages_per_group", type=int, default=500, help="Aiming for this many pdf pages per work item group")
    parser.add_argument("--max_page_retries", type=int, default=8, help="Max number of times we will retry rendering a page")
    parser.add_argument("--max_page_error_rate", type=float, default=0.004, help="Rate of allowable failed pages in a document, 1/250 by default")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to run at a time")
    parser.add_argument("--apply_filter", action="store_true", help="Apply basic filtering to English pdfs which are not forms, and not likely seo spam")
    parser.add_argument("--stats", action="store_true", help="Instead of running any job, reports some statistics about the current workspace")

    # Model parameters
    parser.add_argument(
        "--model",
        help="List of paths where you can find the model to convert this pdf. You can specify several different paths here, and the script will try to use the one which is fastest to access",
        default="allenai/olmOCR-7B-0225-preview",
    )
    parser.add_argument("--model_max_context", type=int, default="32000", help="Maximum context length that the model was fine tuned under")
    parser.add_argument("--model_chat_template", type=str, default="qwen2-vl", help="Chat template to pass to sglang server")
    parser.add_argument("--target_longest_image_dim", type=int, help="Dimension on longest side to use for rendering the pdf pages", default=1024)
    parser.add_argument("--target_anchor_text_len", type=int, help="Maximum amount of anchor text to use (characters)", default=6000)

    parser.add_argument("--port", type=int, default=30024, help="Port to use for the SGLang server")
    parser.add_argument("--sglang_server_url", type=str, default='http://localhost:30024', help="Url to use for the SGLang server")
    args = parser.parse_args()

    args.workspace = "./local_workspace"

    print(f"args: {args}")

    start_time = time.time()

    global workspace_s3, pdf_s3
    # set the global SGLANG_SERVER_PORT from args
    global SGLANG_SERVER_URL
    if args.sglang_server_url:
        SGLANG_SERVER_URL = args.sglang_server_url
    else:
        print(f"Sglang server is required, please set the --sglang_server_url argument")
        sys.exit(1)

    # Check if the sglang server is connected.
    await sglang_server_ready()

    if args.workspace_profile:
        workspace_session = boto3.Session(profile_name=args.workspace_profile)
        workspace_s3 = workspace_session.client("s3")

    if args.pdf_profile:
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3 = pdf_session.client("s3")

    # We need poppler to load the initial pdfs, even if we are not processing them here
    check_poppler_version()

    # Delete the workspace if it exists
    if os.path.exists(args.workspace):
        shutil.rmtree(args.workspace)

    # Create work queue
    if args.workspace.startswith("s3://"):
        work_queue = S3WorkQueue(workspace_s3, args.workspace)
    else:
        work_queue = LocalWorkQueue(args.workspace)

    pdf_work_paths = set()
    if args.pdfs:
        logger.info("Got --pdfs argument, going to add to the work queue")

        for pdf_path in args.pdfs:
            # Expand s3 paths
            if pdf_path.startswith("s3://"):
                logger.info(f"Expanding s3 glob at {pdf_path}")
                pdf_work_paths |= set(expand_s3_glob(pdf_s3, pdf_path))
            elif os.path.exists(pdf_path):
                # Check if the corresponding .txt file exists
                if check_txt_file_exists(pdf_path):
                    logger.info(f"Found .txt file for: {pdf_path}, skipping it")
                    continue

                if (
                    pdf_path.lower().endswith(".pdf")
                    or pdf_path.lower().endswith(".png")
                    or pdf_path.lower().endswith(".jpg")
                    or pdf_path.lower().endswith(".jpeg")
                ):
                    if open(pdf_path, "rb").read(4) == b"%PDF":
                        logger.info(f"Loading file at {pdf_path} as PDF document")
                        pdf_work_paths.add(pdf_path)
                    elif is_png(pdf_path) or is_jpeg(pdf_path):
                        logger.info(f"Loading file at {pdf_path} as image document")
                        pdf_work_paths.add(pdf_path)
                    else:
                        logger.warning(f"File at {pdf_path} is not a valid PDF")
                elif pdf_path.lower().endswith(".log"):
                    logger.info(f"Loading file at {pdf_path} as list of paths")
                    with open(pdf_path, "r") as f:
                        pdf_work_paths |= set(filter(None, (line.strip() for line in f)))
                else:
                    raise ValueError(f"Unsupported file extension for {pdf_path}")
            else:
                raise ValueError("pdfs argument needs to be either a local path, an s3 path, or an s3 glob pattern...")

        logger.info(f"Found {len(pdf_work_paths):,} total pdf paths to add")

        # Estimate average pages per pdf
        sample_size = min(100, len(pdf_work_paths))
        sampled_pdfs = random.sample(list(pdf_work_paths), sample_size)
        page_counts = []

        for pdf in tqdm(sampled_pdfs, desc="Sampling PDFs to calculate optimal length"):
            try:
                # Download the PDF to a temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                    tmp_file.write(get_s3_bytes(pdf_s3, pdf))
                    tmp_file.flush()
                    if is_png(tmp_file.name) or is_jpeg(tmp_file.name):
                        page_counts.append(1)
                    else:
                        reader = PdfReader(tmp_file.name)
                        page_counts.append(len(reader.pages))
            except Exception as e:
                logger.warning(f"Failed to read {pdf}: {e}")

        if page_counts:
            avg_pages_per_pdf = sum(page_counts) / len(page_counts)
        else:
            logger.warning("Could not read any PDFs to estimate average page count.")
            avg_pages_per_pdf = 10  # Default to 10 pages per PDF if sampling fails

        items_per_group = max(1, int(args.pages_per_group / avg_pages_per_pdf))
        logger.info(f"Calculated items_per_group: {items_per_group} based on average pages per PDF: {avg_pages_per_pdf:.2f}")

        # Now call populate_queue
        await work_queue.populate_queue(list(pdf_work_paths), items_per_group)

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    # Initialize the work queue
    qsize = await work_queue.initialize_queue()

    if qsize == 0:
        logger.info("No work to do, exiting")
        return
    # Create a semaphore to control worker access
    # We only allow one worker to move forward with requests, until the server has no more requests in its queue
    # This lets us get full utilization by having many workers, but also to be outputting dolma docs as soon as possible
    # As soon as one worker is no longer saturating the gpu, the next one can start sending requests
    semaphore = asyncio.Semaphore(1)

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue, semaphore, worker_id=i))
        worker_tasks.append(task)

    # Wait for all worker tasks to finish
    await asyncio.gather(*worker_tasks)

    global error_processing_files_counter

    metrics_task.cancel()
    logger.info("Work done")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
    logger.info(f"Total time taken: {elapsed_time / 60:.2f} minutes")
    logger.info(f"Total documents processed: {len(pdf_work_paths):,}")
    logger.info(f"Total error-processing documents count: {error_processing_files_counter}")
