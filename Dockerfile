# Use NVIDIA's CUDA 12.1 base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" \
    MAX_JOBS=8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    ninja-build \
    pkg-config \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install base Python dependencies
RUN pip3 install --no-cache-dir packaging wheel setuptools

# Copy requirements and install them in two steps
COPY requirements.txt .

# First install PyTorch and other base packages
RUN pip3 install --no-cache-dir torch==2.1.2 ninja==1.11.1

# Then install the rest of the packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /data/qwen/logs /data/models/merged /tmp/model_checkpoints /tmp/offload

# Set up entry point script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default command
CMD ["/entrypoint.sh"]
