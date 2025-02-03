import torch
import gc
import os
import psutil
import logging
import asyncio
from typing import Dict, Optional, List, Tuple
from threading import Lock
import subprocess
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
import statistics
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MemoryError(Exception):
    """Custom exception for memory-related errors"""

    pass


@dataclass
class MemorySnapshot:
    """Track memory usage over time"""

    timestamp: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization: float


class OptimizedMemoryManager:
    """
    Enhanced singleton memory manager optimized for H100 GPU running DeepSeek model.
    Includes dynamic thresholds, predictive cleanup, and advanced monitoring.
    """

    _instance = None
    _initialization_lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._initialization_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            # Load configurations
            with open("config/api_config.json") as f:
                self.api_config = json.load(f)
            with open("config/model_config.json") as f:
                self.model_config = json.load(f)

            self._initialized = True
            self.cuda_available = torch.cuda.is_available()
            if not self.cuda_available:
                raise RuntimeError("CUDA must be available for memory management")

            # Core configuration
            self.device = torch.device("cuda:0")
            self.model_size_gb = 62  # Fixed model size
            self.model_memory_bytes = self.model_size_gb * (1024**3)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory

            # Enhanced memory settings
            self.base_memory_threshold = self.api_config["memory_threshold_gb"]
            self.memory_threshold = self.base_memory_threshold
            self.emergency_threshold = self.model_config.get("memory_management", {}).get("emergency_threshold_gb", 70)
            self.cleanup_interval = self.model_config.get("memory_management", {}).get("cleanup_interval_seconds", 30)
            self.workspace_size = int(self.model_config["optimization"]["workspace_size"].replace("GB", ""))

            # Memory monitoring
            self.memory_snapshots: List[MemorySnapshot] = []
            self.max_snapshots = 100
            self.monitoring_enabled = self.model_config.get("memory_management", {}).get("monitoring_enabled", True)

            # Configure memory pools
            if self.model_config["optimization"]["cpu_memory_pool"]["enabled"]:
                self.cpu_pool_size = int(self.model_config["optimization"]["cpu_memory_pool"]["size"].replace("GB", ""))
                self._setup_memory_pools()

            # Enhanced thread safety
            self.memory_lock = asyncio.Lock()
            self.allocation_lock = Lock()
            self.monitoring_lock = Lock()

            # Status tracking
            self.model_loaded = False
            self.last_cleanup_time = 0
            self.memory_pressure_detected = False
            self.last_snapshot_time = 0

            # Initialize monitoring and background tasks
            self._setup_cuda_environment()
            if self.monitoring_enabled:
                self._start_monitoring()

            logger.info("Enhanced memory manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise

    def _setup_cuda_environment(self):
        """Configure enhanced CUDA environment for optimal H100 performance"""
        try:
            # Set CUDA device configuration
            torch.cuda.set_device(self.device)

            # Configure PyTorch CUDA settings
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

            # Enhanced memory allocation strategy
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"expandable_segments:True,"
                f"garbage_collection_threshold:{self.model_config['hardware']['offload_config']['cpu_offload_threshold']},"
                f"max_split_size_mb:512,"
                f"roundup_power2_divisions:16,"
                f"backend:native,"
                f"max_split_size_mb:512"
            )

            # Enable optimizations
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except:
                logger.warning("Some CUDA optimizations not available")

            # Configure workspace with safety margin
            safe_workspace = min(0.90, (self.workspace_size * (1024**3)) / self.total_memory)
            torch.cuda.set_per_process_memory_fraction(safe_workspace)

            logger.info("Enhanced CUDA environment configured successfully")

        except Exception as e:
            logger.error(f"Failed to setup CUDA environment: {e}")
            raise

    def _setup_memory_pools(self):
        """Configure enhanced memory pools with monitoring"""
        try:
            # Configure CPU memory pool with optimization
            if hasattr(torch.cuda, "memory_pool"):
                torch.cuda.memory_pool.set_roundup_power2_divisions(16)
                if hasattr(torch.cuda.memory_pool, "set_memory_fraction"):
                    torch.cuda.memory_pool.set_memory_fraction(0.95)

            # Set memory limits with safety checks
            import resource

            system_memory = psutil.virtual_memory().total
            safe_pool_size = min(self.cpu_pool_size * (1024**3), system_memory * 0.9)
            resource.setrlimit(resource.RLIMIT_AS, (int(safe_pool_size), -1))

            logger.info(f"Enhanced memory pools configured - CPU pool size: {self.cpu_pool_size}GB")

        except Exception as e:
            logger.warning(f"Memory pool setup failed: {e}")

    def _start_monitoring(self):
        """Start background memory monitoring"""

        async def monitoring_task():
            while True:
                try:
                    await self._update_memory_snapshot()
                    await self._check_memory_pressure()
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    await asyncio.sleep(5)

        asyncio.create_task(monitoring_task())

    async def _update_memory_snapshot(self):
        """Update memory usage snapshots"""
        async with self.memory_lock:
            current_time = time.time()
            if current_time - self.last_snapshot_time < 1:  # Limit snapshot frequency
                return

            memory_info = self.get_memory_info()
            snapshot = MemorySnapshot(
                timestamp=current_time, allocated_gb=memory_info["allocated_gb"], reserved_gb=memory_info["reserved_gb"], free_gb=memory_info["free_gb"], utilization=memory_info["utilization"]
            )

            self.memory_snapshots.append(snapshot)
            if len(self.memory_snapshots) > self.max_snapshots:
                self.memory_snapshots.pop(0)

            self.last_snapshot_time = current_time

    async def _check_memory_pressure(self):
        """Check for memory pressure conditions"""
        if not self.memory_snapshots:
            return

        recent_snapshots = self.memory_snapshots[-5:]  # Last 5 snapshots
        avg_utilization = statistics.mean(s.utilization for s in recent_snapshots)
        avg_free_gb = statistics.mean(s.free_gb for s in recent_snapshots)

        if avg_utilization > 85 or avg_free_gb < self.emergency_threshold:
            self.memory_pressure_detected = True
            await self.emergency_cleanup()
        else:
            self.memory_pressure_detected = False

    async def emergency_cleanup(self):
        """Emergency memory cleanup procedure"""
        async with self.memory_lock:
            try:
                logger.warning("Initiating emergency cleanup")

                # Force cache clearing with multiple passes
                for _ in range(2):
                    # Clear CUDA caches
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "memory"):
                        torch.cuda.memory.empty_cache()

                    # Synchronize and clear again
                    with torch.cuda.device(self.device):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Clear autograd graphs
                    if hasattr(torch, "autograd"):
                        if hasattr(torch.autograd, "set_grad_enabled"):
                            torch.autograd.set_grad_enabled(False)
                        if hasattr(torch.autograd, "grad_mode"):
                            torch.autograd.grad_mode.set_grad_enabled(False)

                    # Multiple GC passes
                    gc.collect()

                # Fragment consolidation and stats reset
                if hasattr(torch.cuda, "memory_stats"):
                    torch.cuda.memory_stats(self.device)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

                # Final cleanup pass
                gc.collect()
                torch.cuda.empty_cache()

                # Verify cleanup effectiveness
                memory_info = self.get_memory_info()
                if memory_info["free_gb"] < self.emergency_threshold:
                    logger.error(f"Emergency cleanup insufficient - Free: {memory_info['free_gb']:.2f}GB, Need: {self.emergency_threshold:.2f}GB")
                    return False

                logger.info(f"Emergency cleanup successful - Free memory: {memory_info['free_gb']:.2f}GB")
                return True

            except Exception as e:
                logger.error(f"Emergency cleanup failed: {e}")
                return False

    @asynccontextmanager
    async def allocation_context(self):
        """Enhanced context manager for memory allocation operations"""
        async with self.memory_lock:
            try:
                # Pre-allocation cleanup
                await self._aggressive_cleanup()
                # Monitor memory pressure
                monitoring_task = None
                if self.monitoring_enabled:
                    monitoring_task = asyncio.create_task(self._monitor_allocation())
                yield
            finally:
                # Post-allocation cleanup
                if monitoring_task:
                    monitoring_task.cancel()
                torch.cuda.empty_cache()
                gc.collect()
                await self._update_memory_thresholds()

    async def _monitor_allocation(self):
        """Monitor memory during allocation operations"""
        while True:
            try:
                if self.memory_pressure_detected:
                    await self.emergency_cleanup()
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

    async def prepare_for_model_load(self) -> bool:
        """Enhanced memory preparation for model loading"""
        async with self.allocation_context():
            try:
                logger.info("Preparing memory for model load...")

                # Calculate required memory with enhanced buffer
                required_memory = self.model_size_gb * (1024**3)  # Model size
                runtime_buffer = 3 * (1024**3)  # 3GB runtime buffer
                workspace_buffer = 2 * (1024**3)  # 2GB workspace buffer
                total_required = required_memory + runtime_buffer + workspace_buffer

                # Get current memory status
                memory_info = self.get_memory_info()
                available_memory = memory_info["free_gb"] * (1024**3)

                if available_memory < total_required:
                    # Attempt emergency cleanup
                    await self.emergency_cleanup()
                    memory_info = self.get_memory_info()
                    available_memory = memory_info["free_gb"] * (1024**3)

                    if available_memory < total_required:
                        logger.error(f"Insufficient memory after cleanup: {available_memory / (1024**3):.2f}GB available, {total_required / (1024**3):.2f}GB required")
                        return False

                # Reserve memory with safety margin
                reservation_fraction = min(0.95, (total_required / self.total_memory) + 0.07)
                torch.cuda.set_per_process_memory_fraction(reservation_fraction)

                # Verify reservation
                final_memory = self.get_memory_info()
                logger.info(f"Memory prepared - Available: {final_memory['free_gb']:.2f}GB, Reserved: {final_memory['reserved_gb']:.2f}GB")

                return True

            except Exception as e:
                logger.error(f"Failed to prepare memory: {e}")
                return False

    async def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        try:
            # Force garbage collection first
            gc.collect()

            with torch.cuda.device(self.device):
                # Clear CUDA caches
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "memory"):
                    torch.cuda.memory.empty_cache()

                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                # Synchronize CUDA
                torch.cuda.synchronize()

                # Clear JIT cache if available
                if hasattr(torch.jit, "flush_memory_caches"):
                    torch.jit.flush_memory_caches()

                # Multiple GC passes
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()

            # Log cleanup results with detailed info
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

            logger.info(
                f"Memory cleanup completed:\n- Free: {free / (1024**3):.2f}GB\n- Total: {total / (1024**3):.2f}GB\n- Allocated: {allocated / (1024**3):.2f}GB\n- Reserved: {reserved / (1024**3):.2f}GB"
            )

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            # Don't raise the error, just log it

    async def _update_memory_thresholds(self):
        """Update memory thresholds based on usage patterns"""
        if not self.memory_snapshots:
            return

        recent_snapshots = self.memory_snapshots[-10:]  # Last 10 snapshots
        avg_usage = statistics.mean(s.allocated_gb for s in recent_snapshots)
        usage_stddev = statistics.stdev(s.allocated_gb for s in recent_snapshots) if len(recent_snapshots) > 1 else 0

        # Dynamic threshold calculation
        self.memory_threshold = max(
            self.base_memory_threshold,
            min(75, avg_usage * 1.2 + usage_stddev),  # 20% buffer plus standard deviation
        )

        logger.debug(f"Updated memory threshold to {self.memory_threshold:.2f}GB")

    async def verify_memory_state(self) -> bool:
        """Enhanced memory state verification"""
        try:
            memory_info = self.get_memory_info()
            available_gb = memory_info.get("free_gb", 0)

            # Check against current threshold
            if available_gb < self.memory_threshold:
                logger.warning(f"Available memory ({available_gb:.2f}GB) below threshold ({self.memory_threshold}GB)")

                # Attempt aggressive cleanup
                await self._aggressive_cleanup()

                # If still insufficient, try emergency cleanup
                memory_info = self.get_memory_info()
                available_gb = memory_info.get("free_gb", 0)

                if available_gb < self.memory_threshold:
                    await self.emergency_cleanup()
                    memory_info = self.get_memory_info()
                    available_gb = memory_info.get("free_gb", 0)

            # Update thresholds based on current state
            await self._update_memory_thresholds()

            # Analyze memory fragmentation
            fragmentation = self._analyze_fragmentation()
            if fragmentation > 0.3:  # More than 30% fragmentation
                logger.warning(f"High memory fragmentation detected: {fragmentation:.2%}")
                await self._defragment_memory()

            return available_gb >= self.memory_threshold

        except Exception as e:
            logger.error(f"Memory state verification failed: {e}")
            return False

    def _analyze_fragmentation(self) -> float:
        """Analyze memory fragmentation level"""
        try:
            if hasattr(torch.cuda, "memory_stats"):
                stats = torch.cuda.memory_stats(self.device)
                allocated = stats.get("allocated_bytes.all.current", 0)
                reserved = stats.get("reserved_bytes.all.current", 0)
                if reserved > 0:
                    return 1.0 - (allocated / reserved)
            return 0.0
        except Exception as e:
            logger.error(f"Error analyzing fragmentation: {e}")
            return 0.0

    async def _defragment_memory(self):
        """Attempt to defragment GPU memory"""
        try:
            # Force contiguous memory allocation
            current_tensors = []
            async with self.allocation_context():
                # Collect all existing tensors
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            current_tensors.append((obj.size(), obj.dtype, obj.clone()))
                    except:
                        continue

                # Clear existing tensors
                del current_tensors
                await self._aggressive_cleanup()

            logger.info("Memory defragmentation completed")
        except Exception as e:
            logger.error(f"Memory defragmentation failed: {e}")

    def get_memory_info(self) -> Dict[str, float]:
        """Enhanced memory statistics with detailed metrics"""
        try:
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

            # Additional metrics
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()

            # Calculate memory metrics
            fragmentation = 1.0 - (allocated / reserved) if reserved > 0 else 0.0
            utilization = (allocated / total) * 100

            return {
                "free_gb": free / (1024**3),
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "peak_allocated_gb": peak_allocated / (1024**3),
                "peak_reserved_gb": peak_reserved / (1024**3),
                "fragmentation": fragmentation,
                "utilization": utilization,
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    def get_memory_history(self) -> List[Dict[str, float]]:
        """Get historical memory usage data"""
        return [
            {"timestamp": snapshot.timestamp, "allocated_gb": snapshot.allocated_gb, "reserved_gb": snapshot.reserved_gb, "free_gb": snapshot.free_gb, "utilization": snapshot.utilization}
            for snapshot in self.memory_snapshots
        ]

    async def predict_memory_requirement(self, sequence_length: int) -> float:
        """Predict memory requirement for a given sequence length"""
        try:
            # Base memory requirement for model
            base_memory = self.model_size_gb

            # KV cache memory for attention
            heads = 40  # From model config
            head_dim = 128  # From model config
            kv_cache = (sequence_length * heads * head_dim * 2 * 2) / (1024**3)  # Times 2 for key and value

            # Additional workspace memory
            workspace = (sequence_length * 64) / (1024**3)  # Approximate workspace per token

            total_required = base_memory + kv_cache + workspace
            return total_required

        except Exception as e:
            logger.error(f"Error predicting memory requirement: {e}")
            return float("inf")

    def __del__(self):
        """Enhanced cleanup on deletion"""
        try:
            if hasattr(self, "cuda_available") and self.cuda_available:
                with torch.cuda.device(self.device):
                    # Clear CUDA caches
                    torch.cuda.empty_cache()
                    torch.cuda.memory.empty_cache()

                    # Reset stats
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

                    # Multiple cleanup passes
                    for _ in range(3):
                        gc.collect()

            logger.info("Memory manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during memory manager cleanup: {e}")
