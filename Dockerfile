ARG CUDA_VERSION=12.8.1
FROM --platform=linux/amd64 nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Needs to be repeated below the FROM, or else it's not picked up
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.8.1

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# From original VLLM dockerfile https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile
# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo python3-apt \
    && for i in 1 2 3; do \
    add-apt-repository -y ppa:deadsnakes/ppa && break || \
    { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv

# olmOCR Specific Installs - Install fonts BEFORE changing Python version
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends poppler-utils fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools ttf-mscorefonts-installer

# Now update Python alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv python3 -m pip install uv

# Install some helper utilities for things like the benchmark
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    unzip

ENV PYTHONUNBUFFERED=1

# keep the build context clean
WORKDIR /build          
COPY . /build


# Needed to resolve setuptools dependencies
ENV UV_INDEX_STRATEGY="unsafe-best-match"
RUN uv pip install --system --no-cache ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl
RUN uv pip install --system --no-cache ".[bench]"

RUN playwright install-deps
RUN playwright install chromium

RUN python3 -m olmocr.pipeline --help