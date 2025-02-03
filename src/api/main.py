import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
import sys
import os
import torch
import gc
import psutil
import asyncio
import signal
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path
import json
import time

from .middleware import setup_middleware
from .routes import setup_routes
from ..model import ModelManager
from ..model.model_loader import EnhancedModelLoader
from ..utils.memory_manager import OptimizedMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("/data/qwen/logs/api.log")])
logger = logging.getLogger(__name__)

# Global state
model_manager: Optional[ModelManager] = None
memory_manager: Optional[OptimizedMemoryManager] = None


def load_configs():
    """Load application configurations"""
    try:
        config_dir = Path("config")
        with open(config_dir / "api_config.json") as f:
            api_config = json.load(f)
        with open(config_dir / "model_config.json") as f:
            model_config = json.load(f)
        return api_config, model_config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise RuntimeError("Failed to load configuration files")


API_CONFIG, MODEL_CONFIG = load_configs()


async def setup_cuda_environment():
    """Initialize CUDA environment"""
    try:
        # Set optimal thread settings
        os.environ["OMP_NUM_THREADS"] = "40"
        os.environ["MKL_NUM_THREADS"] = "40"
        torch.set_num_threads(40)
        torch.set_num_interop_threads(40)

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name} with {total_memory:.2f}GB total memory")
        else:
            raise RuntimeError("CUDA is not available")

    except Exception as e:
        logger.error(f"Failed to setup CUDA environment: {e}")
        raise


async def initialize_resources():
    """Initialize all required resources with proper ordering and timeout"""
    global model_manager, memory_manager

    try:
        logger.info("Starting resource initialization...")

        # Set up CUDA environment first
        await setup_cuda_environment()

        # Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = OptimizedMemoryManager()

        # Initial GPU memory check
        initial_gpu_free, initial_gpu_total = torch.cuda.mem_get_info()
        initial_gpu_free_gb = initial_gpu_free / (1024**3)
        initial_gpu_total_gb = initial_gpu_total / (1024**3)
        logger.info(f"Initial GPU memory - Free: {initial_gpu_free_gb:.2f}GB, Total: {initial_gpu_total_gb:.2f}GB")

        # Initialize model loader
        logger.info("Initializing model loader...")
        model_loader = EnhancedModelLoader(memory_manager=memory_manager)

        # Load model and tokenizer with timeout
        try:
            # Use wait_for instead of timeout context manager
            model, tokenizer = await asyncio.wait_for(
                model_loader.load_model(),
                timeout=600  # 10 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error("Model loading timed out after 10 minutes")
            await memory_manager._aggressive_cleanup()
            raise RuntimeError("Model loading timed out")

        # Initialize model manager
        logger.info("Initializing model manager...")
        model_manager = ModelManager(
            model=model,
            tokenizer=tokenizer,
            memory_manager=memory_manager,
            model_path=MODEL_CONFIG["model_path"]
        )

        # Verify loading was successful
        if not model_manager.is_loaded:
            raise RuntimeError("Model failed to initialize properly")

        # Log final memory status
        final_memory = memory_manager.get_memory_info()
        logger.info(
            f"Resource initialization completed:\n"
            f"- Allocated: {final_memory['allocated_gb']:.2f}GB\n"
            f"- Reserved: {final_memory['reserved_gb']:.2f}GB\n"
            f"- Free: {final_memory['free_gb']:.2f}GB"
        )

    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        if memory_manager:
            await memory_manager._aggressive_cleanup()
        raise

async def cleanup_resources():
    """Cleanup resources"""
    global model_manager, memory_manager

    try:
        logger.info("Starting resource cleanup...")

        if model_manager:
            logger.info("Cleaning up model manager...")
            if hasattr(model_manager, "model"):
                del model_manager.model
            if hasattr(model_manager, "tokenizer"):
                del model_manager.tokenizer
            del model_manager
            model_manager = None

        if memory_manager:
            logger.info("Running memory cleanup...")
            await memory_manager._aggressive_cleanup()

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Resource cleanup completed")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def setup_signal_handlers(loop):
    """Setup signal handlers for graceful shutdown"""

    def handle_signal():
        logger.info("Received shutdown signal. Initiating graceful shutdown...")
        loop.create_task(cleanup_resources())
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI application"""
    try:
        loop = asyncio.get_event_loop()
        setup_signal_handlers(loop)
        await initialize_resources()

        # Store managers in app state
        app.state.model_manager = model_manager
        app.state.memory_manager = memory_manager

        yield
    finally:
        await cleanup_resources()


# Create FastAPI app with lifespan manager
app = FastAPI(title="DeepSeek API", description="High-performance API for DeepSeek model inference", version="1.0.0", docs_url="/docs", redoc_url="/redoc", lifespan=lifespan)

# Setup middleware and routes
setup_middleware(app, memory_manager)
setup_routes(app)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Get current system state from components
        memory_info = memory_manager.get_memory_info()
        cpu_info = psutil.virtual_memory()
        model_info = model_manager.get_model_info()

        return {
            "status": "healthy",
            "model": {"loaded": True, "name": MODEL_CONFIG["model_name"], "info": model_info},
            "memory": {"gpu": memory_info, "cpu": {"total_gb": cpu_info.total / (1024**3), "available_gb": cpu_info.available / (1024**3), "percent": cpu_info.percent}},
            "config": {"max_sequence_length": MODEL_CONFIG["max_sequence_length"], "max_batch_size": MODEL_CONFIG["inference"]["max_batch_size"], "rate_limit": API_CONFIG["rate_limit"]},
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        workers=API_CONFIG["workers"],
        loop="uvloop",
        http="httptools",
        limit_concurrency=MODEL_CONFIG["inference"]["max_concurrent_requests"],
        backlog=1,
        timeout_keep_alive=API_CONFIG["timeout"],
        timeout_graceful_shutdown=300,
        proxy_headers=True,
        forwarded_allow_ips="*",
        server_header=False,
        access_log=True,
        log_level="info",
        reload=False,
    )

