# Dockerfile
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.9" \
    CUDA_VISIBLE_DEVICES=0 \
    PROJECT_ROOT=/opt/deepseek

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create project structure
RUN mkdir -p ${PROJECT_ROOT}/{src,models,config,scripts,logs,tests}

# Set working directory
WORKDIR ${PROJECT_ROOT}

# Copy requirements first for better caching
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install Flash Attention 2
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    python3 setup.py install && \
    cd .. && \
    rm -rf flash-attention

# Copy project files
COPY . .

# Create directory for model weights
RUN mkdir -p ${PROJECT_ROOT}/models/merged

# Set up health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Start services
CMD ["bash", "scripts/startup.sh"]
