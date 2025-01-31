from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict
import contextlib
import gc
import os
from ..utils import OptimizedMemoryManager

logger = logging.getLogger(__name__)


class EnhancedModelLoader:
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(config_path or model_path, trust_remote_code=True)
        self.device = torch.device("cuda:0")
        from ..api.main import memory_manager

        self.memory_manager = memory_manager

    def load_model(self, **kwargs):
        """Load model with optimized settings"""
        try:
            logger.info("Loading model with optimized settings...")

            # Configure model loading kwargs
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "device_map": "cuda:0",
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "offload_state_dict": False,  # Keep everything in GPU
            }

            # Only enable Flash Attention if CUDA toolkit is available
            if getattr(self.memory_manager, "cuda_toolkit_available", False):
                model_kwargs.update({"attn_implementation": "flash_attention_2"})

            # Update with any additional kwargs
            model_kwargs.update(kwargs)

            # Load config first to enable optimizations
            config = self.config
            config.pretraining_tp = 1

            # Load model with optimized settings
            logger.info(f"Starting model load with memory stats - Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f}GB")
            model = AutoModelForCausalLM.from_pretrained(self.model_path, config=config, **model_kwargs)

            # Enable caching at the config level
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True

            # Log memory usage after model load
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"Model loaded - GPU Memory Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

            # Apply optimizations
            model = self._apply_model_optimizations(model)

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _apply_model_optimizations(self, model: nn.Module):
        """Apply post-loading optimizations"""
        try:
            # Set model to eval mode
            model.eval()

            # Enable model-specific optimizations
            if hasattr(model.config, "use_memory_efficient_attention"):
                model.config.use_memory_efficient_attention = True

            if hasattr(model.config, "max_position_embeddings"):
                model.config.max_position_embeddings = 131072

            # Only apply CUDA-specific optimizations if toolkit is available
            if getattr(self.memory_manager, "cuda_toolkit_available", False):
                logger.info("Applying CUDA toolkit optimizations")

                # Configure compilation settings
                compiler_config = {"dynamic": True, "fullgraph": False}

                # Log memory before compilation
                logger.info(f"Memory before compilation - Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f}GB, Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f}GB")

                # Compile model with optimized settings
                model = torch.compile(
                    model,
                    backend="inductor",
                    mode="max-autotune",  # Use max-autotune for max optimization
                    **compiler_config,
                )

                # Log memory after compilation
                logger.info(f"Memory after compilation - Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f}GB, Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f}GB")

                logger.info("Model compilation completed successfully")
            else:
                logger.warning("Skipping CUDA toolkit optimizations (toolkit not available)")

            return model

        except Exception as e:
            logger.warning(f"Non-critical optimization error: {e}")
            return model

    def prepare_inference_settings(self) -> Dict:
        """Get optimized inference settings"""
        return {
            "pad_token_id": self.config.pad_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "use_cache": True,
            "repetition_penalty": 1.1,
            "num_return_sequences": 1,
            "early_stopping": True,
            "max_length": 131072,  # Support full context window
            "attention_mask": None,  # Will be created dynamically
        }


