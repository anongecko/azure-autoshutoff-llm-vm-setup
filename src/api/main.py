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
from typing import Optional
import signal
from contextlib import asynccontextmanager
from .middleware import setup_middleware
from .routes import setup_routes
from ..model import ModelManager
from ..utils import OptimizedMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("/data/qwen/logs/api.log")])
logger = logging.getLogger(__name__)

# Global state
model_manager: Optional[ModelManager] = None
memory_manager: Optional[OptimizedMemoryManager] = None


async def initialize_resources():
    """Initialize all required resources"""
    global model_manager, memory_manager

    try:
        # Set optimal thread settings before anything else
        torch.set_num_threads(40)
        torch.set_num_interop_threads(40)

        # Log initial memory status
        logger.info("Initializing resources...")

        # Initialize memory manager
        memory_manager = OptimizedMemoryManager()

        if torch.cuda.is_available():
            # Initial GPU memory check
            initial_gpu_free, initial_gpu_total = torch.cuda.mem_get_info()
            initial_gpu_free_gb = initial_gpu_free / (1024**3)
            initial_gpu_total_gb = initial_gpu_total / (1024**3)
            logger.info(f"Initial GPU memory - Free: {initial_gpu_free_gb:.2f}GB, Total: {initial_gpu_total_gb:.2f}GB")
        else:
            logger.warning("CUDA not available")

        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

            # Let memory manager handle optimizations and reservation
            memory_manager.setup_memory_optimizations()

            # Initialize model manager
            model_manager = ModelManager()
            await model_manager.load_model()

            # Verify model is loaded correctly
            if not model_manager.is_loaded:
                raise RuntimeError("Model failed to load properly")

            # Detailed post-initialization memory logging
            post_load_gpu_free, post_load_gpu_total = torch.cuda.mem_get_info()
            post_load_gpu_free_gb = post_load_gpu_free / (1024**3)
            post_load_gpu_total_gb = post_load_gpu_total / (1024**3)
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            logger.info(f"Final GPU memory - Free: {post_load_gpu_free_gb:.2f}GB, Total: {post_load_gpu_total_gb:.2f}GB")
            logger.info(f"GPU Memory Details - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            logger.info(f"GPU Memory Change - Used: {(initial_gpu_free_gb - post_load_gpu_free_gb):.2f}GB")
            logger.info(f"Peak GPU Memory - Allocated: {peak_memory_allocated:.2f}GB, Reserved: {peak_memory_reserved:.2f}GB")

            # Log successful initialization
            logger.info("Resources initialized successfully")

        else:
            raise RuntimeError("CUDA is not available")

    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        raise


async def cleanup_resources():
    """Cleanup all resources"""
    global model_manager, memory_manager

    try:
        if memory_manager:
            logger.info("Running final memory cleanup...")
            memory_manager.force_cleanup()
            logger.info("Memory cleanup completed successfully.")

        if model_manager:
            logger.info("Deleting model manager...")
            # Ensure model and tokenizer are deleted and their memory is released
            if hasattr(model_manager, "model") and model_manager.model is not None:
                del model_manager.model
            if hasattr(model_manager, "tokenizer") and model_manager.tokenizer is not None:
                del model_manager.tokenizer
            if hasattr(model_manager, "inference") and model_manager.inference is not None:
                # Clean up inference-specific resources
                model_manager.inference.force_cleanup()
                del model_manager.inference
            del model_manager

        # Force garbage collection to release memory held by deleted objects
        gc.collect()

        logger.info("Resource cleanup completed")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def setup_signal_handlers(loop):
    """Setup signal handlers for graceful shutdown"""

    def sigterm_handler():
        logger.info("Received SIGTERM. Initiating graceful shutdown...")
        loop.create_task(cleanup_resources())
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, sigterm_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI application"""
    # Startup
    try:
        loop = asyncio.get_event_loop()
        setup_signal_handlers(loop)
        await initialize_resources()
        yield
    finally:
        # Shutdown
        await cleanup_resources()


# Create FastAPI app with lifespan manager
app = FastAPI(title="DeepSeek API", description="High-performance API for DeepSeek model inference", version="1.0.0", docs_url="/docs", redoc_url="/redoc", lifespan=lifespan)

# Setup middleware and routes
setup_middleware(app)
setup_routes(app)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Get GPU memory info
        gpu_info = torch.cuda.mem_get_info()
        memory_allocated = torch.cuda.memory_allocated()
        peak_memory_allocated = torch.cuda.max_memory_allocated()  # Updated
        peak_memory_reserved = torch.cuda.max_memory_reserved()  # Updated

        # Get CPU memory info
        cpu_info = psutil.virtual_memory()

        return {
            "status": "healthy",
            "model_loaded": True,
            "gpu_memory": {
                "free": gpu_info[0] / (1024**3),
                "total": gpu_info[1] / (1024**3),
                "used": memory_allocated / (1024**3),
                "peak_used": peak_memory_allocated / (1024**3),
                "peak_reserved": peak_memory_reserved / (1024**3),  # Added
            },
            "cpu_memory": {"total": cpu_info.total / (1024**3), "available": cpu_info.available / (1024**3), "percent": cpu_info.percent},
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,  # Single worker for GPU access
        loop="uvloop",  # Fastest event loop implementation
        http="httptools",  # Faster HTTP protocol implementation
        limit_concurrency=1,  # Process one request at a time for optimal GPU utilization
        backlog=1,  # Minimize connection queuing for better resource management
        timeout_keep_alive=300,  # 5 minutes keep-alive for long-running requests
        timeout_graceful_shutdown=300,  # 5 minutes graceful shutdown period
        proxy_headers=True,  # Handle proxy headers for Azure deployment
        forwarded_allow_ips="*",  # Accept forwarded IPs from Azure load balancer
        server_header=False,  # Don't send server header for security
        limit_max_requests=0,  # No request limit as we handle this in middleware
        access_log=True,  # Keep access logging for monitoring
        log_level="info",
        reload=False,  # Disable auto-reload in production
        h11_max_incomplete_size=0,  # No limit on request size for large prompts
        ws_ping_interval=None,  # Disable WebSocket ping for our HTTP-only API
        ws_ping_timeout=None,  # Disable WebSocket timeout
    )

