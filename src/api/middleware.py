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
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

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
            self.requests[key] = [t for t in self.requests[key] if now - t < self.window_size]

            # Check rate limit
            if len(self.requests[key]) >= self.max_requests:
                return False

            self.requests[key].append(now)
            return True


class MemoryMonitor:
    def __init__(self, threshold_gb: float = 90.0):  # Increased threshold for H100
        # Ensure the threshold is a float
        assert isinstance(threshold_gb, float), "threshold_gb must be a float"
        self.threshold_bytes = int(threshold_gb * (1024**3))
        self._lock = Lock()

    def check_memory(self) -> bool:
        with self._lock:
            memory_used = torch.cuda.memory_allocated()
            memory_used_gb = memory_used / (1024**3)

            logger.info(f"Current GPU memory usage: {memory_used_gb:.2f} GB")

            if memory_used >= self.threshold_bytes:
                logger.warning(f"Memory threshold ({self.threshold_bytes / (1024**3):.2f} GB) exceeded")

                # Attempt cleanup if threshold exceeded
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
                gc.collect()

                memory_used = torch.cuda.memory_allocated()
                memory_used_gb = memory_used / (1024**3)
                logger.info(f"Memory after cleanup: {memory_used_gb:.2f} GB")

            return memory_used < self.threshold_bytes

    async def wait_for_memory(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_memory():
                return True
            await asyncio.sleep(0.1)
        return False

    async def force_cleanup(self):
        """Aggressive memory cleanup"""
        with self._lock:
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                gc.collect()


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


class OptimizedAPIMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        memory_threshold_gb: float = 75.0,  # Adjusted default value
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
    ):
        super().__init__(app)
        self.memory_monitor = MemoryMonitor(memory_threshold_gb)
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.request_tracker = RequestTracker()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        try:
            # Check rate limit
            if not self.rate_limiter.is_allowed(request.client.host):
                return Response(content="Too many requests", status_code=429)

            # Wait for memory to be available
            if not await self.memory_monitor.wait_for_memory():
                return Response(content="Server is temporarily overloaded", status_code=503)

            # Track the request
            with self.request_tracker.track():
                # Track GPU memory before request
                initial_gpu_mem = torch.cuda.memory_allocated()
                initial_gpu_mem_gb = initial_gpu_mem / (1024**3)
                logger.info(f"Initial GPU memory before request: {initial_gpu_mem_gb:.2f} GB")

                # Process request
                response = await call_next(request)

                # Check memory after request
                final_gpu_mem = torch.cuda.memory_allocated()
                final_gpu_mem_gb = final_gpu_mem / (1024**3)
                logger.info(f"Final GPU memory after request: {final_gpu_mem_gb:.2f} GB")

                memory_diff_gb = (final_gpu_mem - initial_gpu_mem) / (1024**3)
                if memory_diff_gb > 1.0:  # 1GB threshold
                    logger.warning(f"High memory usage detected: {memory_diff_gb:.2f} GB increase")
                    torch.cuda.empty_cache()
                    gc.collect()

                return response

        except Exception as e:
            # Handle OOM
            if "out of memory" in str(e).lower():
                logger.error("OOM detected, cleaning memory")
                torch.cuda.empty_cache()
                gc.collect()
                return Response(content="Server is temporarily overloaded", status_code=503)
            raise


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application"""

    # Add optimized API middleware first
    app.add_middleware(OptimizedAPIMiddleware)

    # Add Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


