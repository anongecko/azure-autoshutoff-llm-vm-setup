import os
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import logging
from ..utils import OptimizedMemoryManager
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from .model_loader import EnhancedModelLoader
from .inference import OptimizedInference
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Track model performance metrics"""

    load_time: float = 0.0
    last_inference_time: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    peak_memory: float = 0.0
    supported_languages: List[str] = None


class ModelManager:
    """Enhanced model lifecycle and inference manager"""

    def __init__(
        self, model: Optional[PreTrainedModel] = None, tokenizer: Optional[PreTrainedTokenizer] = None, memory_manager: Optional["OptimizedMemoryManager"] = None, model_path: Optional[str] = None
    ):
        """Enhanced model manager initialization"""
        start_time = time.time()

        # Load and cache configuration
        with open("config/model_config.json") as f:
            self.model_config = json.load(f)

        self.base_path = Path(model_path or self.model_config["model_path"])
        self.model = model
        self.tokenizer = tokenizer
        self.memory_manager = memory_manager
        self.inference: Optional[OptimizedInference] = None

        # Initialize metrics
        self.metrics = ModelMetrics(load_time=0.0, supported_languages=["python", "javascript", "typescript", "html", "css", "sql"])

        if model is not None and tokenizer is not None:
            self.inference = OptimizedInference(model=self.model, tokenizer=self.tokenizer, memory_manager=self.memory_manager)
            self.metrics.load_time = time.time() - start_time

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return all([self.model is not None, self.tokenizer is not None, self.inference is not None])

    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model information and metrics"""
        if not self.is_loaded:
            return {"status": "not_loaded"}

        # Get current memory stats
        memory_stats = torch.cuda.memory_stats()
        current_memory = torch.cuda.memory_allocated() / (1024**3)

        # Update peak memory
        self.metrics.peak_memory = max(self.metrics.peak_memory, current_memory)

        return {
            "status": "loaded",
            "model_path": str(self.base_path),
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "max_sequence_length": self.model_config["max_sequence_length"],
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "metrics": {
                "load_time": self.metrics.load_time,
                "total_requests": self.metrics.total_requests,
                "total_tokens": self.metrics.total_tokens,
                "last_inference": self.metrics.last_inference_time,
                "supported_languages": self.metrics.supported_languages,
            },
            "memory": {
                "allocated_gb": current_memory,
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "peak_gb": self.metrics.peak_memory,
                "fragmentation": memory_stats.get("allocated_bytes.all.current", 0) / max(1, memory_stats.get("reserved_bytes.all.current", 1)),
            },
        }

    def __del__(self):
        """Enhanced cleanup on deletion"""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
            if hasattr(self, "inference") and self.inference is not None:
                del self.inference
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")


__all__ = ["ModelManager", "ModelMetrics"]

