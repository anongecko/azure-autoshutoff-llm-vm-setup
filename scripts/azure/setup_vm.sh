#!/bin/bash
set -e

# Log all output
exec 1> >(tee -a "/var/log/vm_setup.log") 2>&1

echo "Starting VM setup at $(date)"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install CUDA
install_cuda() {
    echo "Installing CUDA..."
    wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
    sudo sh cuda_12.3.0_545.23.06_linux.run --silent --toolkit
    echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
}

# Function to install Python dependencies
install_python_deps() {
    echo "Installing Python dependencies..."
    python3 -m pip install --upgrade pip
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123
    python3 -m pip install transformers accelerate fastapi uvicorn python-dotenv azure-cli flash-attn
}

# Update system
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install basic dependencies
echo "Installing basic dependencies..."
sudo apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    software-properties-common \
    nvidia-driver-535

# Install CUDA if not present
if ! command_exists nvcc; then
    install_cuda
fi

# Install Python dependencies
install_python_deps

# Setup project directory
PROJECT_ROOT="/data/qwen"
echo "Setting up project directory at ${PROJECT_ROOT}"
sudo mkdir -p ${PROJECT_ROOT}/{models,logs,config}
sudo chown -R $USER:$USER ${PROJECT_ROOT}

# Create systemd service for the API
echo "Creating systemd service..."
cat << EOF | sudo tee /etc/systemd/system/deepseek-api.service
[Unit]
Description=DeepSeek API Service
After=network.target

[Service]
User=$USER
WorkingDirectory=${PROJECT_ROOT}
Environment=PYTHONPATH=${PROJECT_ROOT}
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/usr/local/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
StandardOutput=append:${PROJECT_ROOT}/logs/api.log
StandardError=append:${PROJECT_ROOT}/logs/api.error.log

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for the monitor
echo "Creating monitor service..."
cat << EOF | sudo tee /etc/systemd/system/deepseek-monitor.service
[Unit]
Description=DeepSeek Monitor Service
After=deepseek-api.service

[Service]
User=$USER
WorkingDirectory=${PROJECT_ROOT}
Environment=PYTHONPATH=${PROJECT_ROOT}
ExecStart=/usr/bin/python3 scripts/azure/monitor.py
Restart=always
StandardOutput=append:${PROJECT_ROOT}/logs/monitor.log
StandardError=append:${PROJECT_ROOT}/logs/monitor.error.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
echo "Enabling and starting services..."
sudo systemctl daemon-reload
sudo systemctl enable deepseek-api deepseek-monitor
sudo systemctl start deepseek-api deepseek-monitor

# Setup monitoring
echo "Setting up monitoring..."
mkdir -p ${PROJECT_ROOT}/monitoring
cat << EOF > ${PROJECT_ROOT}/monitoring/check_gpu.sh
#!/bin/bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader > ${PROJECT_ROOT}/logs/gpu_stats.log
EOF
chmod +x ${PROJECT_ROOT}/monitoring/check_gpu.sh

# Add cron job for GPU monitoring
(crontab -l 2>/dev/null; echo "* * * * * ${PROJECT_ROOT}/monitoring/check_gpu.sh") | crontab -

echo "VM setup completed at $(date)"

# Final verification
echo "Verifying installation..."
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
systemctl status deepseek-api
systemctl status deepseek-monitor
