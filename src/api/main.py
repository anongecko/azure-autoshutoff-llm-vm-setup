import uvicorn
from .middleware import setup_middleware
from .routes import setup_routes
from ..model import ModelManager
from ..utils import OptimizedMemoryManager
import os

# Create FastAPI app
app = FastAPI(
    title="DeepSeek-Qwen API",
    description="High-performance API for DeepSeek-Qwen model inference",
    version="1.0.0"
)

# Setup middleware and routes
setup_middleware(app)
setup_routes(app)

# Initialize model manager
memory_manager = OptimizedMemoryManager()
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    # Optimize memory
    memory_manager.setup_memory_optimizations()
    
    # Load model with optimized settings
    model_manager.load_model({
        "max_batch_size": 32,
        "prefill_chunk_size": 8192,
        "stream_chunk_size": 16,
        "max_concurrent_requests": 10
    })

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    memory_manager.cleanup()
    model_manager.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1,  # Single worker for GPU
        limit_concurrency=1,  # Focus on one request at a time
        timeout_keep_alive=300,
        loop="uvloop",
        http="httptools",
        proxy_headers=True,
        forwarded_allow_ips="*",
        server_header=False,
        reload=False,
        access_log=True,
        backlog=1,  # Minimize queuing
        limit_max_requests=0,  # No limit
        timeout_graceful_shutdown=300
    )
