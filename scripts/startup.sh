# scripts/startup.sh
#!/bin/bash
set -e

# Configure logging
exec 1> >(tee -a "/var/log/startup.log") 2>&1

echo "Starting service initialization at $(date)"

# Environment variables
export PROJECT_ROOT="/data/qwen"
export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_CUDA_ARCH_LIST="8.9"  # H100 architecture
export CUDA_LAUNCH_BLOCKING="0"
export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_AUTO_TUNE=1
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40

# GPU optimization
nvidia-smi -pm 1
nvidia-smi --auto-boost-default=0
nvidia-smi -ac 1215,1410

# Function to check GPU availability
check_gpu() {
    echo "Checking GPU status..."
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. GPU driver not installed?"
        exit 1
    fi
    
    nvidia-smi
    
    # Verify CUDA device availability through PyTorch
    python3 - <<EOF
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
print(f"Found {torch.cuda.device_count()} CUDA device(s)")
print(f"Using device: {torch.cuda.get_device_name(0)}")
EOF
}

# Function to optimize CUDA settings
optimize_cuda() {
    echo "Optimizing CUDA settings..."
    # Set optimal CUDA configurations for H100
    export CUDA_CACHE_PATH="${PROJECT_ROOT}/.cuda_cache"
    mkdir -p "${CUDA_CACHE_PATH}"
    
    # Configure PyTorch settings
    python3 - <<EOF
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("PyTorch CUDA settings optimized")
EOF
}

# Function to verify model files
verify_model() {
    echo "Verifying model files..."
    MODEL_DIR="${PROJECT_ROOT}/models/merged"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: Model directory not found at $MODEL_DIR"
        exit 1
    fi
    
    # Check model files
    python3 - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    model_path = "${MODEL_DIR}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Tokenizer verification successful")
except Exception as e:
    print(f"ERROR: Failed to load tokenizer: {e}")
    exit(1)
EOF
}

# Function to initialize services
init_services() {
    echo "Initializing services..."
    
    # Check if API is already running
    if pgrep -f "uvicorn src.api.main:app" > /dev/null; then
        echo "API service already running"
    else
        echo "Starting API service..."
        systemctl start deepseek-api
    fi
    
    # Check if monitor is already running
    if pgrep -f "scripts/azure/monitor.py" > /dev/null; then
        echo "Monitor service already running"
    else
        echo "Starting monitor service..."
        systemctl start deepseek-monitor
    fi
}

# Function to verify API health
verify_api() {
    echo "Verifying API health..."
    max_retries=30
    count=0
    while [ $count -lt $max_retries ]; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✓ API is healthy"
            return 0
        fi
        echo "Waiting for API to become healthy... ($count/$max_retries)"
        sleep 2
        count=$((count + 1))
    done
    echo "ERROR: API failed to become healthy"
    exit 1
}

# Function to cleanup on failure
cleanup() {
    echo "Error occurred, cleaning up..."
    systemctl stop deepseek-api deepseek-monitor || true
    pkill -f "uvicorn src.api.main:app" || true
    pkill -f "scripts/azure/monitor.py" || true
    torch.cuda.empty_cache || true
}

# Main execution
trap cleanup ERR

echo "Starting initialization sequence..."

# Run initialization steps
check_gpu
optimize_cuda
verify_model
init_services
verify_api

echo "Service initialization completed successfully at $(date)"
