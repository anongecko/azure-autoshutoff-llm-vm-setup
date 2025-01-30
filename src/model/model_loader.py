from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
from typing import Optional, Dict
import contextlib
import gc
from ..utils import OptimizedMemoryManager

class EnhancedModelLoader:
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(config_path or model_path)
        self.device = torch.device("cuda:0")
        self.memory_manager = OptimizedMemoryManager()  # Initialize memory manager
 
    @contextlib.contextmanager
    def _optimize_cuda_memory(self):
        """Context manager for optimizing CUDA memory operations"""
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated()
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.memory_allocated() > initial_memory:
                torch.cuda.reset_peak_memory_stats()

    def _configure_rope(self) -> Dict:
        """Configure rotary position embedding for 128k context"""
        return {
            "rope_scaling": {
                "type": "dynamic",
                "factor": 4.0,
                "original_max_position_embeddings": 32768
            },
            "max_position_embeddings": 131072
        }

    def _optimize_attention_settings(self) -> Dict:
        """Configure optimal attention settings for H100"""
        return {
            "use_flash_attention_2": True,
            "attention_dropout": 0.0,
            "pretraining_tp": 1,  # Ensure tensor parallelism is disabled for merged weights
            "window_size": 8192,  # Optimal for H100 memory access patterns
            "batch_size_hint": 1,  # Help compiler optimize for typical use case
        }

    def _setup_model_optimizations(self):
        """Setup H100-specific optimizations"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Pin memory for faster CPU-GPU transfer
        torch.cuda.set_per_process_memory_fraction(0.99)  # Use most of VRAM
        
        # Set optimal thread settings for CPU operations
        torch.set_num_threads(40)  # Match vCPU count
        torch.set_num_interop_threads(40)  # Match vCPU count

    def load_model(self):
        """Load model with optimized settings"""
        # Extend existing load_model with:
        try:
            # Optimize memory before loading
            self.memory_manager.optimize_for_inference()

        model_kwargs = {
            "device_map": "auto",
            "max_memory": {
                0: "79GB",
                "cpu": "300GB"
            },
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": False,  # We have plenty of RAM
            "offload_folder": "/tmp/model_offload"
        }
        
        # Add compiler optimizations
        self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,
            backend="inductor"
        )
        finally:
            # Cleanup if needed
            self.memory_manager.cleanup()

    def _apply_model_optimizations(self, model: nn.Module, streaming_mode: bool):
        """Apply post-loading optimizations"""
        # Enable fused operations where possible
        if hasattr(model, 'enable_fused_ops'):
            model.enable_fused_ops()

        # Optimize for inference
        model.eval()
        
        # Configure for streaming if needed
        if streaming_mode:
            model.config.use_cache = True
            if hasattr(model, 'enable_streaming'):
                model.enable_streaming()

        # Apply torch.compile with H100-specific optimizations
        model_config = {
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": True,
            "backend": "inductor",  # Best backend for H100
            "options": {
                "epilogue_fusion": True,
                "max_autotune": True,
                "workspace_size": "74GB"  # Reserve some VRAM for runtime
            }
        }
        
        return torch.compile(model, **model_config)

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

    def optimize_for_batch_size(self, batch_size: int):
        """Adjust settings for specific batch sizes"""
        max_vram = 79 * 1024 * 1024 * 1024  # 79GB in bytes
        seq_length = 131072
        
        # Calculate optimal chunk size based on available memory
        hidden_size = self.config.hidden_size
        bytes_per_token = hidden_size * 2  # bfloat16 = 2 bytes
        
        total_tokens = batch_size * seq_length
        tokens_per_chunk = max_vram // (bytes_per_token * batch_size)
        
        return min(8192, tokens_per_chunk)  # Cap at 8192 for efficiency

    @staticmethod
    def clear_memory():
        """Utility method to clear GPU memory"""
        with torch.cuda.device("cuda:0"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
