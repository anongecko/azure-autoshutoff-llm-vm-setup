from typing import Dict, List, Optional, Union, Generator, AsyncGenerator, Any
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
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
    def __init__(self, total_memory: int = 90 * (1024**3)):  # 90GB for H100
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
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Optional[InferenceConfig] = None):
        """Initialize optimized inference with H100-specific optimizations"""
        # Force cleanup before initialization
        self.force_cleanup()

        # Set CUDA device first
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # Ensure model is on CUDA
        self.model = model.to(self.device)

        # Verify model device placement
        if not next(self.model.parameters()).is_cuda:
            raise RuntimeError("Model not properly loaded on CUDA")

        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

        # Initialize memory management with H100 optimizations
        self.cache_manager = CacheManager(
            total_memory=90 * (1024**3)  # 90GB for H100
        )

        # Initialize request handling
        self.request_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)

        # Initialize CUDA streams for concurrent operations
        self.cuda_streams = {
            "main": torch.cuda.Stream(),
            "prefetch": torch.cuda.Stream(priority=-1),
        }

        # Set CUDA memory configurations
        torch.cuda.set_per_process_memory_fraction(0.99, self.device)  # Use most of VRAM

        # Apply model optimizations after device setup
        self._setup_model_optimizations()

        logger.info(f"Initialized OptimizedInference with {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f}GB VRAM)")

    def _setup_model_optimizations(self):
        """Apply advanced optimizations to the model"""
        # Set thread optimizations
        torch.set_num_threads(40)  # Match vCPU count
        torch.set_num_interop_threads(40)  # Match vCPU count

        self.model.eval()
        self.model = torch.compile(self.model, fullgraph=False, dynamic=True, backend="inductor", mode="max-autotune")

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
        self.dtype_config = {"kv_cache": torch.bfloat16, "attention": torch.bfloat16, "intermediate": torch.bfloat16, "output": torch.float32}

        # Pre-allocate buffers for common operations with pinned memory
        # Create tensors on CPU first, then pin them for efficient transfer
        self.static_buffers = {
            "position_ids": torch.arange(self.config.max_sequence_length, device=self.device),
            "attention_mask": torch.ones((1, self.config.max_sequence_length), device=self.device),
        }

        # Configure memory pools for optimal H100 performance
        torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Reserve some for system
        torch.cuda.memory.set_per_process_memory_fraction(0.99, self.device)  # Device-specific setting

    @contextmanager
    def _inference_context(self):
        """Context manager for inference optimizations"""
        try:
            # Ensure model is on correct device
            if hasattr(self.model, "device") and self.model.device != self.device:
                self.model.to(self.device)

            torch.cuda.empty_cache()
            gc.collect()

            # Log memory before
            allocated_before = torch.cuda.memory_allocated() / (1024**3)
            reserved_before = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"Memory before inference - Allocated: {allocated_before:.2f}GB, Reserved: {reserved_before:.2f}GB")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                with torch.no_grad():
                    yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            # Log memory after
            allocated_after = torch.cuda.memory_allocated() / (1024**3)
            reserved_after = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"Memory after inference - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB")

    def _calculate_memory_requirements(self, batch_size: int, seq_length: int) -> int:
        """Update memory calculations for better H100 utilization"""
        # Add to existing calculations:
        extra_buffer = 2 * (1024**3)  # 2GB extra buffer for H100

        # Calculate base memory requirements using total sequence length
        base_memory = self._calculate_base_memory(batch_size, seq_length)

        return min(self.cache_manager.total_memory - extra_buffer, base_memory)

    def _calculate_base_memory(self, batch_size: int, seq_length: int) -> int:
        """Calculate base memory requirements per sequence"""
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        bytes_per_param = 2  # Assuming bfloat16

        # Memory for KV cache
        kv_cache_memory = batch_size * seq_length * num_layers * hidden_size * bytes_per_param * 2  # x2 for K and V

        # Memory for intermediate activations (estimate)
        activation_memory = batch_size * seq_length * hidden_size * num_layers * bytes_per_param

        return kv_cache_memory + activation_memory

    def _optimize_batch_size(self, input_lengths: List[int]) -> int:
        """Dynamically optimize batch size based on input lengths"""
        max_length = max(input_lengths)
        memory_per_sequence = self._calculate_memory_requirements(1, max_length)
        available_memory = int(self.cache_manager.total_memory * self.config.kv_cache_threshold)

        optimal_batch_size = min(available_memory // memory_per_sequence, self.config.max_batch_size, len(input_lengths))

        return max(1, optimal_batch_size)

    def force_cleanup(self):
        """Aggressive memory cleanup"""
        try:
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
            for _ in range(5):
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

    async def _process_batch(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], generation_config: Dict[str, Any]) -> AsyncGenerator[torch.Tensor, None]:
        """Process a batch of inputs with optimized memory usage"""
        batch_size, seq_length = input_ids.shape

        # Ensure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

            # Handle attention mask with experimental control flow
            from functorch.experimental import control_flow

            def process_attention_mask(mask):
                return mask if mask is not None else torch.ones_like(input_ids)

            attention_mask = control_flow.cond(attention_mask is not None, lambda: attention_mask, lambda: torch.ones_like(input_ids))

        # Calculate memory requirements
        memory_required = self._calculate_memory_requirements(batch_size, seq_length)
        memory_required = self._calculate_memory_requirements(batch_size, seq_length)
        if not self.cache_manager.allocate("batch", memory_required):
            raise RuntimeError("Insufficient memory for batch processing")

        # Process in chunks for long sequences
        for chunk_start in range(0, seq_length, self.config.prefill_chunk_size):
            chunk_end = min(chunk_start + self.config.prefill_chunk_size, seq_length)
            chunk_input_ids = input_ids[:, chunk_start:chunk_end]

            chunk_attention_mask = None
            if attention_mask is not None:
                chunk_attention_mask = attention_mask[:, chunk_start:chunk_end]

            try:
                # Process chunk with experimental control flow handling
                with self._inference_context():
                    outputs = self.model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, use_cache=True, return_dict=True)
                chunk_attention_mask = None
                if attention_mask is not None:
                    chunk_attention_mask = attention_mask[:, chunk_start:chunk_end]
            except Exception as e:
                # Process chunk with experimental control flow handling
                with self._inference_context():
                    outputs = self.model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, use_cache=True, return_dict=True)
                torch.cuda.empty_cache()
                yield outputs.logits[:, -1:].to(self.device)

        self.cache_manager.free("batch")

    async def generate_stream(self, prompt: Union[str, List[str]], generation_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Stream generate text with optimized batch processing"""
        if isinstance(prompt, str):
            prompt = [prompt]

        # Tokenization try-except block
        try:
            # Validate input prompt
            if not prompt:
                logger.warning("Empty prompt provided")
                return

            # Check maximum concurrent requests
            if self.request_queue.qsize() >= self.config.max_concurrent_requests:
                raise RuntimeError("Maximum concurrent requests exceeded")

            # Tokenize with explicit device placement
            tokenized = self.tokenizer(prompt, padding=True, truncation=True, max_length=self.config.max_sequence_length, return_tensors="pt")

            # Ensure all tensors are on GPU
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device) if "attention_mask" in tokenized else None

            # Verify tensor placement
            if not input_ids.is_cuda:
                raise RuntimeError("input_ids not on CUDA device")
            if attention_mask is not None and not attention_mask.is_cuda:
                raise RuntimeError("attention_mask not on CUDA device")

            tokenized_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        except RuntimeError as e:
            logger.error(f"Runtime error during tokenization: {e}")
            yield f"Error: {str(e)}"
            return
        except Exception as e:
            logger.error(f"Error in tokenization: {e}")
            yield f"Tokenization error: {str(e)}"
            return

        # Generation try-except block
        try:
            # Optimize batch size
            input_lengths = tokenized_inputs["input_ids"].shape[1]
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
                "stream_chunk_size": self.config.stream_chunk_size,
            }
            generation_config = {**default_config, **(generation_config or {})}

            # Stream generation
            generated = []
            generated_texts = [[] for _ in range(tokenized_inputs["input_ids"].size(0))]

            async for logits in self._process_batch(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], generation_config):
                try:
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

                        # Decode each sequence in the batch individually
                        for i in range(tokens.size(0)):
                            text = self.tokenizer.decode(tokens[i], skip_special_tokens=True)

                            # Remove previously generated text to yield only the new part
                            text_diff = text[len("".join(generated_texts[i])) :]
                            generated_texts[i].append(text_diff)
                            yield text_diff
                        generated = []

                except torch.cuda.OutOfMemoryError:
                    logger.error("CUDA out of memory during token generation")
                    torch.cuda.empty_cache()
                    yield "Error: GPU memory exhausted"
                    return
                except Exception as e:
                    logger.error(f"Error during token generation: {e}")
                    yield f"Generation error: {str(e)}"
                    return

            # Yield any remaining tokens
            if generated:
                try:
                    tokens = torch.cat(generated, dim=-1)

                    # Decode each sequence in the batch individually
                    for i in range(tokens.size(0)):
                        text = self.tokenizer.decode(tokens[i], skip_special_tokens=True)

                        # Remove previously generated text to yield only the new part
                        text_diff = text[len("".join(generated_texts[i])) :]
                        generated_texts[i].append(text_diff)
                        yield text_diff
                except Exception as e:
                    logger.error(f"Error decoding final tokens: {e}")
                    yield f"Error in final decoding: {str(e)}"
                    return

        except asyncio.TimeoutError:
            logger.error("Generation request timed out")
            yield "Generation request timed out"
        except RuntimeError as memory_error:
            logger.error(f"Memory allocation error: {memory_error}")
            yield f"Memory allocation error: {str(memory_error)}"
        except Exception as e:
            logger.error(f"Unexpected error in generation: {e}")
            yield f"Unexpected generation error: {str(e)}"

    def __call__(self, *args, **kwargs):
        """Synchronous interface for generation"""

        async def run_async():
            results = []
            async for result in self.generate_stream(*args, **kwargs):
                results.append(result)
            return "".join(results)

        return asyncio.run(run_async())


