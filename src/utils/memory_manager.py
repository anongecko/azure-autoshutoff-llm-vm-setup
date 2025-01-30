import torch
import gc
import psutil
from typing import Dict, Optional
import numpy as np

class OptimizedMemoryManager:
    def __init__(self, gpu_memory_threshold: float = 0.9, ram_threshold: float = 0.95):
        self.gpu_memory_threshold = gpu_memory_threshold
        self.ram_threshold = ram_threshold
        self.setup_memory_optimizations()

    def setup_memory_optimizations(self):
        """Configure memory optimizations"""
        # GPU Memory
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.99)
        
        # System RAM
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (320 * 1024 * 1024 * 1024, -1))  # 320GB
        
        # Pre-allocate memory pool
        self.memory_pool = torch.cuda.CUDAPlace(0).allocate(75 * 1024 * 1024 * 1024)  # 75GB pool

    def get_memory_status(self) -> Dict[str, Dict[str, float]]:
        """Get current memory usage status"""
        gpu_memory = torch.cuda.memory_stats(0)
        system_memory = psutil.virtual_memory()
        
        return {
            "gpu": {
                "used": gpu_memory["allocated_bytes.all.current"] / 1024**3,
                "cached": gpu_memory["reserved_bytes.all.current"] / 1024**3,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            },
            "ram": {
                "used": system_memory.used / 1024**3,
                "available": system_memory.available / 1024**3,
                "total": system_memory.total / 1024**3
            }
        }

    def optimize_for_inference(self):
        """Optimize memory for inference"""
        gc.collect()
        torch.cuda.empty_cache()
        
        # Pin critical memory
        torch.cuda.memory.pin_memory(size=32 * 1024 * 1024 * 1024)  # 32GB pinned
        
        # Pre-allocate buffers
        self.inference_buffers = {
            "attention": torch.zeros(1, 1024, 4096, device="cuda", dtype=torch.bfloat16),
            "hidden": torch.zeros(1, 1024, 4096, device="cuda", dtype=torch.bfloat16)
        }

    def cleanup(self):
        """Cleanup memory"""
        for buffer in self.inference_buffers.values():
            del buffer
        torch.cuda.empty_cache()
        gc.collect()
