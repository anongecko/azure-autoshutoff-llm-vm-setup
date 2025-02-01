# Docker Deployment Instructions

## Prerequisites
- H100 GPU with 80GB VRAM
- NVIDIA Driver >= 525
- Docker >= 24.0
- docker-compose >= 2.0
- At least 320GB system RAM
- Ubuntu 22.04 LTS recommended

## Directory Structure
```
.
├── data/
│   ├── models/
│   │   └── merged/          # Model files
│   └── qwen/
│       └── logs/            # Application logs
├── config/                  # Configuration files
├── docker/
│   ├── entrypoint.sh       # Container entry point
│   └── README.md           # This file
├── src/                    # Application source code
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Service configuration
└── requirements.txt        # Python dependencies
```

## Setup Instructions

1. Prepare directories:
```bash
mkdir -p data/models/merged data/qwen/logs
chmod 777 data/qwen/logs
```

2. Place model files:
- Copy model files to `data/models/merged/`
- Required files:
  - config.json
  - model.safetensors

3. Configure settings:
- Review and adjust `config/api_config.json`
- Review and adjust `config/model_config.json`

4. Build and start:
```bash
docker-compose build
docker-compose up -d
```

## Monitoring

- Logs: `docker-compose logs -f`
- Health check: `curl http://localhost:8000/health`
- GPU stats: `nvidia-smi`

## Resource Management

The container is configured with:
- GPU: Single H100 (device 0)
- CPU: 40 threads
- Memory: 320GB limit
- Disk mounts:
  - Model files (read-only)
  - Logs (read-write)
  - Checkpoints (ephemeral)
  - Offload space (ephemeral)

## Troubleshooting

1. GPU Issues:
```bash
# Check GPU visibility
docker exec deepseek-api nvidia-smi

# Verify CUDA
docker exec deepseek-api python3 -c "import torch; print(torch.cuda.is_available())"
```

2. Memory Issues:
```bash
# Check memory usage
docker stats deepseek-api

# Clean checkpoints
docker exec deepseek-api rm -rf /tmp/model_checkpoints/*
```

3. Common Issues:
- "GPU not found": Check NVIDIA driver and docker-nvidia installation
- "Out of memory": Verify system has sufficient RAM
- "Permission denied": Check directory permissions
- "Model not loaded": Verify model files exist and are readable

## Security Notes

The container runs with:
- No root access
- Limited capabilities
- Read-only root filesystem
- Memory limits
- CPU limits
- Resource monitoring

## Maintenance

1. Updates:
```bash
docker-compose pull
docker-compose up -d
```

2. Cleanup:
```bash
# Remove old containers
docker-compose down

# Clean volumes
docker volume prune -f
```

3. Logs:
```bash
# View logs
docker-compose logs -f

# Rotate logs
docker-compose restart
```
