from typing import Dict, List, Optional, Union, Generator
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from contextlib import contextmanager
import gc
import threading
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    max_batch_size: int = 32
    max_sequence_length: int = 131072
    prefill_chunk_size: int = 8192
    decode_chunk_size: int = 2048
    kv_cache_threshold: float = 0.7  # 70% of available memory
    stream_chunk_size: int = 16
    max_concurrent_requests: int = 10
    timeout_seconds: float = 300.0

class CacheManager:
    def __init__(self, total_memory: int = 79 * (1024**3)):  # 79GB for H100
        self.total_memory = total_memory
        self.allocated_memory = 0
        self.cache_lock = Lock()
        self.kv_caches = {}
        
    def allocate(self, request_id: str, size: int) -> bool:
        with self.cache_lock:
            if self.allocated_memory + size > self.total_memory:
                return False
            self.allocated_memory += size
            self.kv_caches[request_id] = size
            return True
            
    def free(self, request_id: str):
        with self.cache_lock:
            if request_id in self.kv_caches:
                self.allocated_memory -= self.kv_caches[request_id]
                del self.kv_caches[request_id]

class OptimizedInference:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize optimized inference with H100-specific optimizations
        
        Args:
            model: The pre-trained model
            tokenizer: The model's tokenizer
            config: Optional inference configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig(
            max_batch_size=1,  # Optimized for single complex requests
            max_sequence_length=131072,
            prefill_chunk_size=16384,  # Larger chunks for better throughput
            decode_chunk_size=2048,
            kv_cache_threshold=0.9,  # Higher threshold since we have plenty of memory
            stream_chunk_size=32,
            max_concurrent_requests=1,  # Focus resources on one request
            timeout_seconds=600.0  # Longer timeout for complex tasks
        )
        self.device = torch.device("cuda:0")
        
        # Initialize memory management
        self.cache_manager = CacheManager(
            total_memory=79 * (1024**3)  # 79GB for H100
        )
        
        # Initialize request handling
        self.request_queue = Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_requests
        )
        
        # Set device defaults
        torch.cuda.set_device(self.device)
        
        # Apply model optimizations
        self._setup_model_optimizations()
        
        # Initialize cuda streams for concurrent operations
        self.cuda_streams = {
            "main": torch.cuda.Stream(),
            "prefetch": torch.cuda.Stream()
        }
        
        # Pre-allocate workspace memory
        self.workspace = torch.cuda.caching_allocator_init(
            max_split_size_mb=128
        )
        
        logger.info(
            f"Initialized OptimizedInference with {torch.cuda.get_device_name()} "
            f"({torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f}GB VRAM)"
        )

    def _setup_model_optimizations(self):
        """Apply advanced optimizations to the model"""
        # Enable memory efficient attention
        if hasattr(self.model, "config"):
            self.model.config.use_memory_efficient_attention = True
            self.model.config.use_flash_attention_2 = True
            self.model.config.pretraining_tp = 1  # Disable tensor parallelism for merged weights
            self.model.config.max_position_embeddings = 131072
            self.model.config.rope_scaling = {"type": "dynamic", "factor": 4.0}

        # Optimize for H100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Set optimal dtypes for different operations
        self.dtype_config = {
            "kv_cache": torch.bfloat16,
            "attention": torch.bfloat16,
            "intermediate": torch.bfloat16,
            "output": torch.float32
        }

        # Pre-allocate buffers for common operations with pinned memory
        self.static_buffers = {
            "position_ids": torch.arange(self.config.max_sequence_length, device=self.device).pin_memory(),
            "attention_mask": torch.ones((1, self.config.max_sequence_length), device=self.device).pin_memory()
        }

        # Set thread optimizations
        torch.set_num_threads(40)  # Match vCPU count
        torch.set_num_interop_threads(40)  # Match vCPU count

        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.99)  # Use most of VRAM

    @contextmanager
    def _inference_context(self):
        """Context manager for inference optimizations"""
        torch.cuda.empty_cache()
        gc.collect()
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, cache_enabled=True):
                with torch.no_grad():
                    yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def _calculate_memory_requirements(self, batch_size: int, seq_length: int) -> int:
        """Update memory calculations for better H100 utilization"""
        # Add to existing calculations:
        extra_buffer = 2 * (1024**3)  # 2GB extra buffer for H100
        return min(
            self.cache_manager.total_memory - extra_buffer,
            super()._calculate_memory_requirements(batch_size, seq_length)
        )


    def _optimize_batch_size(self, input_lengths: List[int]) -> int:
        """Dynamically optimize batch size based on input lengths"""
        max_length = max(input_lengths)
        memory_per_sequence = self._calculate_memory_requirements(1, max_length)
        available_memory = int(self.cache_manager.total_memory * self.config.kv_cache_threshold)
        
        optimal_batch_size = min(
            available_memory // memory_per_sequence,
            self.config.max_batch_size,
            len(input_lengths)
        )
        
        return max(1, optimal_batch_size)

    async def _process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: Dict
    ) -> Generator[torch.Tensor, None, None]:
        """Process a batch of inputs with optimized memory usage"""
        batch_size, seq_length = input_ids.shape
        
        # Optimize memory allocation
        memory_required = self._calculate_memory_requirements(batch_size, seq_length)
        if not self.cache_manager.allocate("batch", memory_required):
            raise RuntimeError("Insufficient memory for batch processing")

        try:
            # Chunked processing for long sequences
            for chunk_start in range(0, seq_length, self.config.prefill_chunk_size):
                chunk_end = min(chunk_start + self.config.prefill_chunk_size, seq_length)
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]
                chunk_attention_mask = attention_mask[:, :chunk_end]

                # Process chunk
                with self._inference_context():
                    outputs = self.model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        use_cache=True,
                        return_dict=True
                    )
                
                yield outputs.logits[:, -1:]

        finally:
            self.cache_manager.free("batch")

    async def generate_stream(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[Dict] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generate text with optimized batch processing"""
        if isinstance(prompt, str):
            prompt = [prompt]

        # Tokenize inputs
        tokenized_inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        ).to(self.device)

        # Optimize batch size
        input_lengths = tokenized_inputs.input_ids.shape[1]
        batch_size = self._optimize_batch_size([input_lengths])

        # Set up generation config
        default_config = {
            "max_new_tokens": 4096,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stream_chunk_size": self.config.stream_chunk_size
        }
        generation_config = {**default_config, **(generation_config or {})}

        # Stream generation
        generated = []
        async for logits in self._process_batch(
            tokenized_inputs.input_ids,
            tokenized_inputs.attention_mask,
            generation_config
        ):
            # Sample from logits
            next_token_logits = logits[:, -1, :]
            if generation_config["do_sample"]:
                probs = F.softmax(next_token_logits / generation_config["temperature"], dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated.append(next_tokens)

            # Decode and yield new tokens
            if len(generated) >= generation_config["stream_chunk_size"]:
                tokens = torch.cat(generated, dim=-1)
                text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
                generated = []
                yield text

        # Yield any remaining tokens
        if generated:
            tokens = torch.cat(generated, dim=-1)
            text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            yield text

    def __call__(self, *args, **kwargs):
        """Synchronous interface for generation"""
        return asyncio.run(self.generate_stream(*args, **kwargs))
