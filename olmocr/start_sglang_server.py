## Start sglang server to run olmocr model.

# For reference usage only, use start_sglang_server.sh to start the server instead.

import os
import re
import logging
import torch
import asyncio
import atexit
import time
import sys

from huggingface_hub import snapshot_download

from olmocr.s3_utils import (
    download_directory,
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

sglang_logger = logging.getLogger("sglang")
sglang_logger.propagate = False

file_handler = logging.FileHandler("olmocr-sglang-server-running.log", mode="a")
sglang_logger.addHandler(file_handler)

# Specify a default port, but it can be overridden by args
SGLANG_SERVER_PORT = 30024

async def download_model(model_name_or_path: str):
    if model_name_or_path.startswith("s3://") or model_name_or_path.startswith("gs://") or model_name_or_path.startswith("weka://"):
        logger.info(f"Downloading model directory from '{model_name_or_path}'")
        model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "olmocr", "model")
        download_directory([model_name_or_path], model_cache_dir)
        return model_cache_dir
    elif os.path.isabs(model_name_or_path) and os.path.isdir(model_name_or_path):
        logger.info(f"Using local model path at '{model_name_or_path}'")
        return model_name_or_path
    else:
        logger.info(f"Downloading model with hugging face '{model_name_or_path}'")
        snapshot_download(repo_id=model_name_or_path)
        return model_name_or_path

async def start_sglang_server(model_name_or_path, args):
    """
    Start the SGLang server as a subprocess and handle its output.
    """
    # Check GPU memory, lower mem devices need a bit less KV cache space because the VLM takes additional memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    mem_fraction_arg = ["--mem-fraction-static", "0.80"] if gpu_memory < 60 else []

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_name_or_path,
        "--chat-template",
        args.model_chat_template,
        # "--context-length", str(args.model_max_context),  # Commented out due to crashes
        "--port",
        str(SGLANG_SERVER_PORT),
        "--log-level-http",
        "warning",
    ]
    cmd.extend(mem_fraction_arg)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Ensure the subprocess is terminated on exit
    def _kill_proc():
        proc.terminate()

    atexit.register(_kill_proc)

    # Shared variables between tasks
    last_running_req, last_queue_req = 0, 0
    server_printed_ready_message = False
    last_semaphore_release = time.time()

    async def process_line(line):
        nonlocal last_running_req, last_queue_req, last_semaphore_release, server_printed_ready_message
        sglang_logger.info(line)

        # if the server hasn't initialized yet, log all the lines to the main logger also, so that the user
        # can see any warnings/errors more easily
        if not server_printed_ready_message:
            logger.info(line)

        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        # TODO, need to trace down this issue in sglang itself, but it will otherwise cause the server to lock up
        if "IndexError: list index out of range" in line:
            logger.error("IndexError in model, restarting server")
            proc.terminate()

        if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
            server_printed_ready_message = True
            last_semaphore_release = time.time()

        match = re.search(r"#running-req: (\d+)", line)
        if match:
            last_running_req = int(match.group(1))

        match = re.search(r"#queue-req: (\d+)", line)
        if match:
            last_queue_req = int(match.group(1))
            logger.info(f"sglang running req: {last_running_req} queue req: {last_queue_req}")

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                line = line.decode("utf-8").rstrip()
                await process_line(line)
            except Exception as ex:
                logger.warning(f"Got {ex} when reading log line from inference server, skipping")

    # Start tasks to read stdout, stderr, and handle timeout logic
    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))

    try:
        await proc.wait()
    except asyncio.CancelledError:
        logger.info("Got cancellation request for SGLang server")
        proc.terminate()
        raise
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


async def sglang_server_host(model_name_or_path, args):
    MAX_RETRIES = 5
    retry = 0

    while retry < MAX_RETRIES:
        await start_sglang_server(model_name_or_path, args)
        logger.warning("Start sgLang server failed, retrying...")
        retry += 1

    if retry >= MAX_RETRIES:
        logger.error(f"Ended up starting the sglang server more than {retry} times, cancelling pipeline")
        logger.error("")
        logger.error("Please make sure sglang is installed according to the latest instructions here: https://docs.sglang.ai/start/install.html")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the SGLang server.")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Path to the model directory or Hugging Face model name.")
    parser.add_argument("--model-chat-template", type=str, default="default", help="Chat template for the model.")
    args = parser.parse_args()

    # Start the SGLang server
    asyncio.run(sglang_server_host(args.model_name_or_path, args))
    logger.info("SGLang server started.")
    logger.info("Waiting for SGLang server to finish...")