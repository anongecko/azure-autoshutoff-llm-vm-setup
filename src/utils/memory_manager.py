import torch
import gc
import psutil
import logging
import time
from typing import Dict, Optional
import numpy as np
import os

# Configure logging
logger = logging.getLogger(__name__)


class OptimizedMemoryManager:
    def __init__(self, gpu_memory_threshold: float = 0.9):
        self.gpu_memory_threshold = gpu_memory_threshold
        self.inference_buffers = {}
        self._verify_cuda_installation()

    def _verify_cuda_installation(self):
        """Verify CUDA installation and requirements"""
        try:
            import subprocess
            import os

            # Check CUDA availability through PyTorch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available through PyTorch")

            # Check CUDA version
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if not cuda_home:
                logger.warning("CUDA_HOME not found. CUDA toolkit might not be installed properly")

            # Try to get CUDA version
            try:
                nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode()
                logger.info(f"Found CUDA: {nvcc_output.split('release')[-1].strip()}")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("nvcc not found. CUDA toolkit is not installed or not in PATH")
                logger.warning("Some optimizations will be disabled")
                self.cuda_toolkit_available = False
                return

            self.cuda_toolkit_available = True
            logger.info("CUDA toolkit verification completed successfully")

        except Exception as e:
            logger.error(f"CUDA verification error: {e}")
            self.cuda_toolkit_available = False

    def setup_memory_optimizations(self):
        """Configure memory optimizations for H100 and reserve memory"""
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            # Set CUDA device
            torch.cuda.set_device(0)

            # Single cleanup pass
            self.force_cleanup()

            # Configure GPU Memory Settings
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()

            # Reserve 95% of GPU memory via set_per_process_memory_fraction
            # Remove the dummy tensor approach, as it can lead to unnecessary fragmentation
            torch.cuda.set_per_process_memory_fraction(0.95, device=0)
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory_bytes = int(total_gpu_memory * 0.95)
            reserved_memory_gb = reserved_memory_bytes / (1024**3)
            logger.info(f"Reserved 95% of GPU memory for the model: {reserved_memory_gb:.2f}GB")

            # Configure H100-specific CUDA optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            # Enable flash attention for H100
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except Exception as e:
                logger.warning(f"Could not enable Flash SDP: {e}")

            # Configure System RAM
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (360 * 1024 * 1024 * 1024, -1))  # Increased to 360GB

            # Enhanced logging for GPU memory
            gpu_info = torch.cuda.mem_get_info()
            total_mem = gpu_info[1] / 1024**3
            free_mem = gpu_info[0] / 1024**3
            logger.info(f"H100 GPU Memory - Free: {free_mem:.2f}GB, Total: {total_mem:.2f}GB")
            logger.info(f"Memory Utilization: {((total_mem - free_mem) / total_mem * 100):.2f}%")

        except Exception as e:
            logger.error(f"Memory optimization setup failed: {e}")
            raise

    def optimize_for_inference(self):
        """Optimize memory settings for inference on H100"""
        try:
            # Set CUDA allocation settings with expanded configuration
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.85,max_split_size_mb:512"

            # Aggressive cleanup before allocation
            self.force_cleanup()

            # H100-specific CUDA configuration
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except Exception as e:
                logger.warning(f"Flash SDP not available: {e}")

            # Comprehensive memory logging
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"H100 Inference Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

            # Initialize inference buffers
            self._initialize_inference_buffers()

            return True

        except Exception as e:
            logger.error(f"H100 Inference Optimization Failed: {e}")
            self.force_cleanup()
            raise

    def _initialize_inference_buffers(self):
        """Initialize inference buffers with fallback options"""
        try:
            # Get GPU memory info
            gpu_info = torch.cuda.mem_get_info()
            available_mem = gpu_info[0]  # Free memory in bytes
            total_mem = gpu_info[1]  # Total memory in bytes

            # Calculate safe allocation size (80% of available memory)
            safe_buffer_size = int(available_mem * 0.8)

            logger.info(f"Available GPU memory: {available_mem / 1024**3:.2f}GB of {total_mem / 1024**3:.2f}GB total")
            logger.info(f"Allocating buffers with {safe_buffer_size / 1024**3:.2f}GB size")

            # Try to allocate buffers with error handling
            try:
                # Calculate sizes based on model dimensions
                hidden_dim = 4096  # Model's hidden dimension
                seq_len = min(131072, safe_buffer_size // (hidden_dim * 2))  # 2 bytes per element for bfloat16

                self.inference_buffers = {
                    "attention": torch.zeros((1, seq_len, hidden_dim), device="cuda:0", dtype=torch.bfloat16),
                    "hidden": torch.zeros((1, seq_len, hidden_dim), device="cuda:0", dtype=torch.bfloat16),
                }
                logger.info(f"Successfully allocated inference buffers with sequence length {seq_len}")

            except RuntimeError as e:
                logger.warning(f"Failed to allocate full-size buffers: {e}")
                # Try with reduced sequence length
                try:
                    seq_len = seq_len // 2
                    self.inference_buffers = {
                        "attention": torch.zeros((1, seq_len, hidden_dim), device="cuda:0", dtype=torch.bfloat16),
                        "hidden": torch.zeros((1, seq_len, hidden_dim), device="cuda:0", dtype=torch.bfloat16),
                    }
                    logger.info(f"Successfully allocated reduced-size inference buffers with sequence length {seq_len}")
                except RuntimeError as e2:
                    logger.error(f"Failed to allocate even reduced buffers: {e2}")
                    self.inference_buffers = {}  # Empty but valid state

        except Exception as e:
            logger.error(f"Buffer initialization failed: {e}")
            self.inference_buffers = {}  # Ensure valid state

    def force_cleanup(self):
        """Aggressive memory cleanup"""
        try:
            # Clear any existing inference buffers
            if hasattr(self, "inference_buffers"):
                for buffer in self.inference_buffers.values():
                    del buffer
                self.inference_buffers = {}

            # Clear CUDA cache multiple ways
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            if hasattr(torch.cuda, "memory_stats"):
                torch.cuda.memory_stats(device=None)
            if hasattr(torch.cuda, "reset_max_memory_cached"):
                torch.cuda.reset_max_memory_cached()

            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            # Reset pytorch caches
            if hasattr(torch, "_dynamo"):
                try:
                    torch._dynamo.reset()
                    logger.info("Dynamo cache reset successful")
                except Exception as e:
                    logger.warning(f"Error resetting Dynamo cache: {e}")

            # Force garbage collection multiple times
            for _ in range(5):  # Increased from 3 to 5 cycles
                gc.collect()

            # Additional GPU memory operations
            if hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()

            # Get current memory usage for logging
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"After cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")

    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory status"""
        try:
            return {
                "allocated": torch.cuda.memory_allocated() / (1024**3),
                "reserved": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated": torch.cuda.max_memory_allocated() / (1024**3),
            }
        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Public cleanup method"""
        self.force_cleanup()


