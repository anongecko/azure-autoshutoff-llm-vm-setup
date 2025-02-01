from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing import Dict, Optional, Callable
import asyncio
import time
import logging
from dataclasses import dataclass
from ..utils import OptimizedMemoryManager
import json
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Enhanced request metrics tracking"""

    request_id: str
    start_time: float
    initial_memory: float
    client_ip: str
    endpoint: str
    method: str
    content_length: Optional[int] = None


class AsyncTokenBucket:
    """Enhanced token bucket rate limiter with improved fairness"""

    def __init__(self, rate_limit: int, burst_limit: int):
        self.rate_limit = rate_limit
        self.burst_limit = burst_limit
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Clean up stale token entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                now = time.time()
                async with self._lock:
                    # Remove entries older than 1 hour
                    stale_keys = [k for k, v in self.last_update.items() if now - v > 3600]
                    for k in stale_keys:
                        self.tokens.pop(k, None)
                        self.last_update.pop(k, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token bucket cleanup error: {e}")

    async def _add_tokens(self, key: str, now: float) -> None:
        """Add tokens with rate smoothing"""
        if key not in self.last_update:
            self.tokens[key] = self.burst_limit
            self.last_update[key] = now
            return

        time_passed = now - self.last_update[key]
        # Smooth token addition for more consistent rate limiting
        new_tokens = min(self.burst_limit - self.tokens.get(key, 0), time_passed * (self.rate_limit / 60.0))
        self.tokens[key] = min(self.burst_limit, self.tokens.get(key, 0) + new_tokens)
        self.last_update[key] = now

    async def acquire(self, key: str, tokens: float = 1.0) -> bool:
        """Try to acquire tokens with overload protection"""
        async with self._lock:
            now = time.time()
            await self._add_tokens(key, now)
            if self.tokens.get(key, 0) >= tokens:
                self.tokens[key] -= tokens
                return True
            return False

    async def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "_cleanup_task"):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class RequestTracker:
    """Enhanced concurrent request tracking"""

    def __init__(self, max_concurrent: int = 1):
        self.active_requests = 0
        self.max_concurrent = max_concurrent
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.request_times: Dict[str, float] = {}
        self._request_queue = asyncio.Queue()

    async def start_request(self, request_id: str) -> bool:
        """Start tracking with queue management"""
        try:
            async with self._lock:
                if self.active_requests >= self.max_concurrent:
                    # Add to queue with timeout
                    try:
                        await asyncio.wait_for(self._request_queue.put(request_id), timeout=5.0)
                    except asyncio.TimeoutError:
                        return False
                self.active_requests += 1
                self.request_times[request_id] = time.time()
                return True
        except Exception as e:
            logger.error(f"Error starting request tracking: {e}")
            return False

    async def end_request(self, request_id: str) -> float:
        """End request with queue processing"""
        try:
            async with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
                duration = time.time() - self.request_times.pop(request_id, time.time())

                # Process queue
                if not self._request_queue.empty():
                    next_request = await self._request_queue.get()
                    self.request_times[next_request] = time.time()
                    self.active_requests += 1

                return duration
        except Exception as e:
            logger.error(f"Error ending request tracking: {e}")
            return 0.0


class OptimizedAPIMiddleware(BaseHTTPMiddleware):
    """Enhanced API middleware with improved resource management"""

    def __init__(self, app: FastAPI, memory_manager: "OptimizedMemoryManager"):
        super().__init__(app)
        self.config_dir = Path("config")
        self.load_configs()

        self.memory_manager = memory_manager
        self.rate_limiter = AsyncTokenBucket(rate_limit=self.api_config["rate_limit"]["requests_per_minute"], burst_limit=self.api_config["rate_limit"]["burst_limit"])
        self.request_tracker = RequestTracker(max_concurrent=self.model_config["inference"]["max_concurrent_requests"])

    def load_configs(self):
        """Load and validate configurations"""
        try:
            with open(self.config_dir / "api_config.json") as f:
                self.api_config = json.load(f)
            with open(self.config_dir / "model_config.json") as f:
                self.model_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load middleware configs: {e}")
            raise

    async def check_api_key(self, request: Request) -> bool:
        """Enhanced API key validation"""
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header:
                return False

            api_key = auth_header.replace("Bearer ", "")
            if not api_key.startswith("sk-"):
                return False

            return True
        except Exception:
            return False

    async def _handle_memory_status(self, metrics: RequestMetrics) -> Optional[Response]:
        """Enhanced memory status handling"""
        try:
            memory_info = self.memory_manager.get_memory_info()
            current_usage = memory_info.get("allocated_gb", 0)
            threshold = self.api_config["memory_threshold_gb"]

            if current_usage > threshold:
                logger.warning(f"Memory usage ({current_usage:.2f}GB) exceeds threshold ({threshold}GB) for request {metrics.request_id}")

                # Let memory manager handle cleanup
                memory_ok = await self.memory_manager.verify_memory_state()
                if not memory_ok:
                    return Response(content="Server is at capacity", status_code=503, headers={"Retry-After": "30"})

            return None

        except Exception as e:
            logger.error(f"Memory status check failed: {e}")
            return Response(content="Internal server error", status_code=500)

    def _create_metrics(self, request: Request) -> RequestMetrics:
        """Create enhanced request metrics"""
        return RequestMetrics(
            request_id=f"req_{int(time.time() * 1000)}",
            start_time=time.time(),
            initial_memory=torch.cuda.memory_allocated() / (1024**3),
            client_ip=request.client.host,
            endpoint=request.url.path,
            method=request.method,
            content_length=request.headers.get("content-length"),
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Enhanced request dispatch with proper resource management"""
        metrics = self._create_metrics(request)

        try:
            # Validate API key
            if not await self.check_api_key(request):
                return Response(content="Invalid API key", status_code=401)

            # Check rate limit
            if not await self.rate_limiter.acquire(metrics.client_ip):
                return Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": "60", "X-RateLimit-Limit": str(self.api_config["rate_limit"]["requests_per_minute"]), "X-RateLimit-Remaining": "0"},
                )

            # Check memory status
            memory_response = await self._handle_memory_status(metrics)
            if memory_response:
                return memory_response

            # Track request
            if not await self.request_tracker.start_request(metrics.request_id):
                return Response(content="Too many concurrent requests", status_code=429, headers={"Retry-After": "10"})

            try:
                # Add request context
                request.state.request_id = metrics.request_id
                request.state.metrics = metrics

                # Process request
                response = await call_next(request)

                # Add response headers
                response.headers.update(
                    {
                        "X-Request-ID": metrics.request_id,
                        "X-Process-Time": f"{time.time() - metrics.start_time:.3f}",
                        "X-Memory-Usage": f"{torch.cuda.memory_allocated() / (1024**3):.2f}GB",
                        "X-Model-Name": self.model_config["model_name"],
                    }
                )

                return response

            finally:
                # End request tracking
                duration = await self.request_tracker.end_request(metrics.request_id)
                logger.info(f"Request {metrics.request_id} completed in {duration:.3f}s - Method: {metrics.method} Endpoint: {metrics.endpoint}")

        except Exception as e:
            logger.error(f"Request {metrics.request_id} failed: {e}")
            return Response(content="Internal server error", status_code=500, headers={"X-Request-ID": metrics.request_id})

        finally:
            # Check memory impact
            final_memory = torch.cuda.memory_allocated() / (1024**3)
            memory_diff = final_memory - metrics.initial_memory
            if memory_diff > 1.0:  # More than 1GB increase
                logger.warning(f"High memory impact detected: {memory_diff:.2f}GB for request {metrics.request_id}")
                await self.memory_manager.verify_memory_state()


def setup_middleware(app: FastAPI, memory_manager: "OptimizedMemoryManager") -> None:
    """Enhanced middleware setup"""
    with open("config/api_config.json") as f:
        api_config = json.load(f)

    # Add core middleware
    app.add_middleware(OptimizedAPIMiddleware, memory_manager=memory_manager)
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config["cors"]["allow_origins"],
        allow_credentials=True,
        allow_methods=api_config["cors"]["allow_methods"],
        allow_headers=api_config["cors"]["allow_headers"],
        expose_headers=["X-Request-ID", "X-Process-Time", "X-Memory-Usage", "X-Model-Name", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses"""
        response = await call_next(request)
        response.headers.update(
            {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Content-Security-Policy": "default-src 'self'", "Strict-Transport-Security": "max-age=31536000; includeSubDomains"}
        )
        return response

    logger.info(f"Middleware configured - Rate limit: {api_config['rate_limit']['requests_per_minute']} rpm, Memory threshold: {api_config['memory_threshold_gb']}GB")

