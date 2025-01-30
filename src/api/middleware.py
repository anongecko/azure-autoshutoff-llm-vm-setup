from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import asyncio
from typing import Callable, Dict
import logging
from threading import Lock
import torch
import contextlib
import gc

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_size: int = 60):
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests: Dict[str, list] = {}
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        with self._lock:
            now = time.time()
            if key not in self.requests:
                self.requests[key] = []
            
            # Clean old requests
            self.requests[key] = [t for t in self.requests[key] 
                                if now - t < self.window_size]
            
            # Check rate limit
            if len(self.requests[key]) >= self.max_requests:
                return False
            
            self.requests[key].append(now)
            return True

class MemoryMonitor:
    def __init__(self, threshold_gb: float = 75.0):  # Leave 4GB buffer
        self.threshold_bytes = threshold_gb * (1024**3)
        self._lock = Lock()

    def check_memory(self) -> bool:
        with self._lock:
            memory_used = torch.cuda.memory_allocated()
            return memory_used < self.threshold_bytes

    async def wait_for_memory(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_memory():
                return True
            await asyncio.sleep(0.1)
        return False

class RequestTracker:
    def __init__(self):
        self.active_requests = 0
        self._lock = Lock()

    @contextlib.contextmanager
    def track(self):
        with self._lock:
            self.active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self.active_requests -= 1

class OptimizedAPIMiddleware:
    def __init__(
        self,
        memory_threshold_gb: float = 300.0,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60
    ):
        self.memory_monitor = MemoryMonitor(memory_threshold_gb)
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.request_tracker = RequestTracker()
        
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        try:
            # Track GPU memory before request
            initial_gpu_mem = torch.cuda.memory_allocated()
            
            # Process request
            response = await call_next(request)
            
            # Check memory after request
            final_gpu_mem = torch.cuda.memory_allocated()
            if final_gpu_mem - initial_gpu_mem > 1e9:  # 1GB threshold
                logger.warning("High memory usage detected")
                torch.cuda.empty_cache()
                gc.collect()
            
            return response
            
        except Exception as e:
            # Handle OOM
            if "out of memory" in str(e).lower():
                logger.error("OOM detected, cleaning memory")
                torch.cuda.empty_cache()
                gc.collect()
                return Response(
                    content="Server is temporarily overloaded",
                    status_code=503
                )
            raise

def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application"""
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add optimized API middleware
    app.add_middleware(OptimizedAPIMiddleware)


