#!/bin/bash
set -e

echo "Starting pre-flight checks..."

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are properly installed."
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "ERROR: Unable to access GPU. Check GPU and driver installation."
    exit 1
fi

# Check for H100
if ! nvidia-smi -L | grep -q "H100"; then
    echo "WARNING: H100 GPU not detected. This may affect performance."
fi

# Check model files
MODEL_DIR="/data/models/merged"
REQUIRED_FILES=("config.json" "Deepseek-R1-Distill-Qwen-32B-Merged.safetensors")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${MODEL_DIR}/${file}" ]; then
        echo "ERROR: Required model file not found: ${MODEL_DIR}/${file}"
        exit 1
    fi
done

# Check directory permissions
REQUIRED_DIRS=(
    "/data/qwen/logs"
    "/data/models/merged"
    "/tmp/model_checkpoints"
    "/tmp/offload"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -w "$dir" ]; then
        echo "ERROR: Directory not writable: $dir"
        exit 1
    fi
done

echo "Pre-flight checks completed successfully"
echo "Starting API server..."

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
