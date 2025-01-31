# src/model/__init__.py
import os
import gc
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import logging
from transformers import AutoTokenizer, PreTrainedModel
from .model_loader import EnhancedModelLoader
from .inference import OptimizedInference, InferenceConfig

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_path: Optional[str] = None):
        self.base_path = Path(model_path or os.getenv("MODEL_PATH", "models/merged"))
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.inference: Optional[OptimizedInference] = None
        self._setup_cuda_optimizations()

    def _setup_cuda_optimizations(self):
        """Configure CUDA for optimal performance"""
        # Set CUDA device properties
        torch.cuda.set_device(0)

        # Enable TF32 for faster matrix multiplications on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    async def load_model(self, inference_config: Optional[Dict[str, Any]] = None) -> None:
        """Load model with optimized settings"""
        try:
            loader = EnhancedModelLoader(str(self.base_path))
            self.model = loader.load_model()

            # Load tokenizer with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_path), use_fast=True, model_max_length=131072, padding_side="left", truncation_side="left")

            # Set up special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Initialize inference engine
            config = InferenceConfig(**(inference_config or {}))
            self.inference = OptimizedInference(model=self.model, tokenizer=self.tokenizer, config=config)

            # Warm up the model
            await self._warmup()

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    async def _warmup(self, num_warmup_steps: int = 3):
        """Warm up the model to optimize compilation and caching"""
        if not self.inference:
            return

        warmup_inputs = [
            "def factorial(n):",  # Short input
            "# Implementation of QuickSort algorithm\ndef quicksort(arr):",  # Medium input
            "# Explain the implementation of a transformer architecture\n",  # Long input
        ]

        for _ in range(num_warmup_steps):
            for prompt in warmup_inputs:
                try:
                    async for _ in self.inference.generate_stream(prompt):
                        break
                except Exception as e:
                    print(f"Error during warmup with prompt '{prompt}': {e}")
                    # Handle the error appropriately, e.g., by logging, retrying, or stopping

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return all([self.model is not None, self.tokenizer is not None, self.inference is not None])

    def generate(self, *args, **kwargs):
        """Generate text using the loaded model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.inference is None:
            raise RuntimeError("Inference engine not initialized.")
        return self.inference(*args, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and stats"""
        if not self.is_loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_path": str(self.base_path),
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "max_sequence_length": self.tokenizer.model_max_length,
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "memory_usage": {
                "allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
                "cached": torch.cuda.memory_reserved() / (1024**3),  # GB
                "max_allocated": torch.cuda.max_memory_allocated() / (1024**3),  # GB
            },
        }


