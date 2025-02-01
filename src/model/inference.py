from typing import Dict, List, Optional, Union, AsyncGenerator, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import asyncio
import logging
from ..utils import OptimizedMemoryManager
from dataclasses import dataclass, field
import gc
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
import statistics
from enum import Enum

logger = logging.getLogger(__name__)


class StoppingCriteria(Enum):
    """Enumeration of stopping criteria for generation"""

    MAX_TOKENS = "max_tokens"
    SEQUENCE_LENGTH = "sequence_length"
    STOP_SEQUENCE = "stop_sequence"
    ERROR = "error"
    MEMORY_PRESSURE = "memory_pressure"


@dataclass
class KVCache:
    """Enhanced KV Cache management for efficient inference"""

    key_states: Optional[torch.Tensor] = None
    value_states: Optional[torch.Tensor] = None
    current_length: int = 0
    max_length: int = field(default_factory=lambda: 0)
    position_offset: int = 0

    def update(self, key: torch.Tensor, value: torch.Tensor, position: int):
        """Update cache with new key-value pairs and memory management"""
        try:
            # Prune before update if needed
            if self.current_length > self.max_length * 0.9:  # 90% threshold
                self.prune()

            if self.key_states is None:
                self.key_states = key
                self.value_states = value
            else:
                # Memory-efficient concatenation
                new_key = torch.cat([self.key_states, key], dim=2)
                if torch.cuda.memory_allocated() > torch.cuda.max_memory_allocated() * 0.95:
                    # Memory pressure - clear and restart
                    self.clear()
                    self.key_states = key
                    self.value_states = value
                else:
                    self.key_states = new_key
                    self.value_states = torch.cat([self.value_states, value], dim=2)

            self.current_length = position + key.size(2)

        except RuntimeError as e:
            logger.error(f"KV cache update failed: {e}")
            self.clear()
            raise

    def prune(self, keep_last_n: int = 4096):
        """Prune cache to prevent memory growth"""
        if self.current_length > keep_last_n:
            if self.key_states is not None:
                self.key_states = self.key_states[:, :, -keep_last_n:]
            if self.value_states is not None:
                self.value_states = self.value_states[:, :, -keep_last_n:]
            self.position_offset += self.current_length - keep_last_n
            self.current_length = keep_last_n

    def clear(self):
        """Enhanced cache clearing with memory management"""
        if self.key_states is not None:
            del self.key_states
        if self.value_states is not None:
            del self.value_states
        self.key_states = None
        self.value_states = None
        self.current_length = 0
        self.position_offset = 0
        torch.cuda.empty_cache()


@dataclass
class GenerationState:
    """Enhanced state tracking during text generation"""

    batch_size: int = 1
    current_tokens: List[List[int]] = field(default_factory=lambda: [[]])
    kv_cache: KVCache = field(default_factory=KVCache)
    attention_mask: Optional[torch.Tensor] = None
    last_token_logits: Optional[torch.Tensor] = None
    generated_tokens: int = 0
    sequence_length: int = 0
    max_sequence_length: int = 131072
    max_new_tokens: int = 0
    stop_sequences: List[str] = field(default_factory=list)
    token_count: int = 0
    generation_start_time: float = field(default_factory=time.time)
    max_time: Optional[float] = None
    memory_pressure_count: int = 0

    def update_state(self, new_tokens: torch.Tensor):
        """Update generation state with memory monitoring"""
        for i, token in enumerate(new_tokens):
            self.current_tokens[i].extend(token.tolist())
        self.generated_tokens += new_tokens.size(1)
        self.token_count += new_tokens.numel()
        self.sequence_length += new_tokens.size(1)

        # Monitor memory pressure
        if torch.cuda.memory_allocated() > torch.cuda.max_memory_allocated() * 0.9:
            self.memory_pressure_count += 1
        else:
            self.memory_pressure_count = max(0, self.memory_pressure_count - 1)

    def should_stop(self) -> Tuple[bool, Optional[StoppingCriteria]]:
        """Enhanced stopping criteria check"""
        if self.memory_pressure_count >= 3:
            return True, StoppingCriteria.MEMORY_PRESSURE

        if self.generated_tokens >= self.max_new_tokens:
            return True, StoppingCriteria.MAX_TOKENS

        if self.sequence_length >= self.max_sequence_length:
            return True, StoppingCriteria.SEQUENCE_LENGTH

        if self.max_time and (time.time() - self.generation_start_time) > self.max_time:
            return True, StoppingCriteria.MAX_TOKENS

        return False, None

    def clear(self):
        """Enhanced state clearing"""
        self.current_tokens = [[] for _ in range(self.batch_size)]
        self.kv_cache.clear()
        if self.attention_mask is not None:
            del self.attention_mask
            self.attention_mask = None
        if self.last_token_logits is not None:
            del self.last_token_logits
            self.last_token_logits = None
        self.generated_tokens = 0
        self.sequence_length = 0
        self.token_count = 0
        self.memory_pressure_count = 0
        gc.collect()
        torch.cuda.empty_cache()


class OptimizedInference:
    """
    Enhanced inference engine for DeepSeek model with improved context handling
    and memory management for 131k context window.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, memory_manager: "OptimizedMemoryManager"):
        # Load and validate configurations
        with open("config/model_config.json") as f:
            self.config = json.load(f)

        self.model = model
        self.tokenizer = tokenizer
        self.memory_manager = memory_manager

        # Enhanced configuration
        self.max_sequence_length = self.config["max_sequence_length"]
        self.chunk_size = self.config["inference"]["prefill_chunk_size"]
        self.decode_chunk_size = self.config["inference"]["decode_chunk_size"]
        self.streaming_chunk_size = self.config["inference"]["streaming_chunk_size"]

        # Context window settings
        self.context_config = self.config["context_window"]
        self.chunk_overlap = self.context_config["chunk_overlap"]
        self.attention_sink = self.context_config.get("attention_sink", True)
        self.compression_ratio = self.context_config.get("compression_factor", 4)
        self.max_chunks = self.context_config.get("max_chunks", 8)

        # Initialize enhanced state tracking
        self.state: Optional[GenerationState] = None
        self.inference_lock = asyncio.Lock()
        self.generation_count = 0
        self.last_memory_check = 0

        # Setup and optimization
        self._setup_model_settings()
        self._setup_tokenizer()
        self._optimize_for_inference()

        logger.info(f"Initialized enhanced inference engine - Context window: {self.max_sequence_length}, Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")

    def _setup_model_settings(self):
        """Enhanced model-specific settings"""
        self.model.eval()

        # Configure advanced attention settings
        if hasattr(self.model.config, "use_flash_attention_2"):
            self.model.config.use_flash_attention_2 = self.config["attention"]["use_flash_attention"]

        # Set optimized sliding window
        if self.config["attention"]["sliding_window"]:
            window_size = self.config["attention"]["sliding_window"]
            self.model.config.sliding_window = window_size

        # Optimize KV cache
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self.config["optimization"]["use_cache"]

        # Set attention implementation
        if hasattr(self.model.config, "attention_implementation"):
            impl = self.config["attention"]["attention_implementation"]
            self.model.config.attention_implementation = impl

    def _setup_tokenizer(self):
        """Enhanced tokenizer configuration"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.model_max_length = self.max_sequence_length

        # Enhanced code-specific tokenization
        if self.config["code_specific"]["preserve_indentation"]:
            indent_tokens = [" " * i for i in range(2, 17, 2)]
            self.tokenizer.add_special_tokens({"additional_special_tokens": indent_tokens})

    def _optimize_for_inference(self):
        """Apply inference optimizations"""
        # Enable memory-efficient attention if available
        if hasattr(self.model.config, "use_memory_efficient_attention"):
            self.model.config.use_memory_efficient_attention = True

        # Configure attention sinks for long sequences
        if self.attention_sink and hasattr(self.model.config, "attention_sink"):
            self.model.config.attention_sink = True
            self.model.config.attention_sink_size = min(2048, self.chunk_size // 8)

        # Enable fused operations if available
        if hasattr(torch._C, "_jit_set_profiling_executor"):
            torch._C._jit_set_profiling_executor(True)

    def _create_attention_pattern(self, sequence_length: int, sliding_window: Optional[int] = None) -> torch.Tensor:
        """Create optimized attention pattern with sink tokens"""
        device = next(self.model.parameters()).device

        if sliding_window and sequence_length > sliding_window:
            # Enhanced sliding window with sink tokens
            mask = torch.zeros((1, sequence_length, sequence_length), device=device)
            sink_size = min(256, sliding_window // 4)

            for i in range(sequence_length):
                # Add sink token attention
                mask[0, i, :sink_size] = 1

                # Add sliding window attention
                start = max(sink_size, i - sliding_window // 2)
                end = min(sequence_length, i + sliding_window // 2)
                mask[0, i, start:end] = 1

                # Add local attention
                local_start = max(sink_size, i - 128)
                local_end = min(sequence_length, i + 128)
                mask[0, i, local_start:local_end] = 1
        else:
            mask = torch.ones((1, sequence_length, sequence_length), device=device)

        return mask

    async def _handle_memory_pressure(self, required_memory: int) -> bool:
        """Enhanced memory pressure handling"""
        try:
            current_free = torch.cuda.mem_get_info()[0]
            if current_free < required_memory:
                # Try aggressive cleanup first
                await self.memory_manager._aggressive_cleanup()
                new_free = torch.cuda.mem_get_info()[0]

                if new_free < required_memory:
                    # If still not enough, try emergency cleanup
                    await self.memory_manager.emergency_cleanup()
                    new_free = torch.cuda.mem_get_info()[0]

                return new_free >= required_memory
            return True
        except Exception as e:
            logger.error(f"Error handling memory pressure: {e}")
            return False

    async def _process_long_input(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> AsyncGenerator[torch.Tensor, None]:
        """Enhanced processing for long inputs"""
        total_length = input_ids.size(1)

        # Process normally if within chunk size
        if total_length <= self.chunk_size:
            if not await self._handle_memory_pressure(self.chunk_size * 4 * input_ids.size(0)):
                raise RuntimeError("Insufficient memory for processing")

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            yield outputs.logits[:, -1:, :]
            return

        # Enhanced chunked processing with compression
        compressed_chunks = []
        overlap_size = self.chunk_overlap

        for chunk_start in range(0, total_length, self.chunk_size - overlap_size):
            # Check memory before processing chunk
            if not await self._handle_memory_pressure(self.chunk_size * 8):
                logger.warning("Memory pressure detected during chunk processing")
                await self.memory_manager.emergency_cleanup()

            chunk_end = min(chunk_start + self.chunk_size, total_length)

            # Extract chunk
            chunk_input_ids = input_ids[:, chunk_start:chunk_end]
            chunk_attention = attention_mask[:, chunk_start:chunk_end] if attention_mask is not None else None

            # Process chunk
            try:
                outputs = self.model(input_ids=chunk_input_ids, attention_mask=chunk_attention, use_cache=True, return_dict=True)

                # Compress chunk if needed
                if len(compressed_chunks) >= self.max_chunks:
                    compressed_chunks = self._compress_chunks(compressed_chunks)

                compressed_chunks.append(outputs.logits[:, -1:, :])

                # Yield last token logits
                yield outputs.logits[:, -1:, :]

            except RuntimeError as e:
                logger.error(f"Error processing chunk: {e}")
                await self.memory_manager.emergency_cleanup()
                raise

    def _compress_chunks(self, chunks: List[torch.Tensor], compression_factor: int = 4) -> List[torch.Tensor]:
        """Compress chunks to manage memory for long sequences"""
        if len(chunks) <= 1:
            return chunks

        compressed = []
        chunk_size = chunks[0].size(1)

        for i in range(0, len(chunks), compression_factor):
            group = chunks[i : i + compression_factor]
            if len(group) > 1:
                # Average the chunks in the group
                compressed_chunk = torch.mean(torch.cat(group, dim=1), dim=1, keepdim=True)
                compressed.append(compressed_chunk)
            else:
                compressed.extend(group)

        return compressed

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_tokens: List[int], penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits with dynamic scaling"""
        if not generated_tokens:
            return logits

        # Create penalty tensor with dynamic scaling
        penalty_tensor = torch.ones_like(logits)
        unique_tokens = list(set(generated_tokens[-1000:]))  # Consider last 1000 tokens
        penalty_tensor[unique_tokens] = penalty

        # Scale penalty based on token frequency
        token_counts = {}
        for token in generated_tokens[-1000:]:
            token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            if count > 3:  # Increase penalty for frequently repeated tokens
                penalty_tensor[token] *= 1.0 + 0.1 * min(count - 3, 5)

        # Apply penalty
        logits = torch.where(logits > 0, logits / penalty_tensor, logits * penalty_tensor)
        return logits

    async def _handle_code_specific_generation(self, logits: torch.Tensor, generated: List[int], generation_config: Dict) -> torch.Tensor:
        """Enhanced code-specific generation with advanced features"""
        # Apply repetition penalty with dynamic scaling
        penalty = generation_config["repetition_penalty"]
        if len(generated) > 100:  # Increase penalty for longer sequences
            penalty *= 1.2

        logits = self._apply_repetition_penalty(logits, generated, penalty)

        if self.config["code_specific"]["language_specific_prompting"]:
            # Enhanced code token biasing
            if hasattr(self.tokenizer, "code_tokens"):
                code_tokens = self.tokenizer.code_tokens
                logits[:, code_tokens] *= 1.2  # Increased bias

            # Bias for structural tokens
            structural_tokens = self._get_structural_tokens()
            if structural_tokens:
                logits[:, structural_tokens] *= 1.1

            # Handle indentation
            if self.config["code_specific"]["preserve_indentation"]:
                indent_level = self._get_current_indent_level(generated)
                indent_tokens = self._get_indent_tokens(indent_level)
                logits[:, indent_tokens] *= 1.15

        return logits

    def _get_structural_tokens(self) -> List[int]:
        """Get structural code tokens"""
        structural = []
        for token in ["def", "class", "if", "for", "while", "return", "{", "}", "(", ")"]:
            try:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                structural.extend(token_ids)
            except:
                continue
        return structural

    def _get_current_indent_level(self, generated_tokens: List[int]) -> int:
        """Analyze current indentation level"""
        if not generated_tokens:
            return 0

        # Get last newline position
        text = self.tokenizer.decode(generated_tokens[-50:])  # Last 50 tokens
        lines = text.split("\n")
        if not lines:
            return 0

        # Count leading spaces
        last_line = lines[-1]
        indent_level = 0
        for char in last_line:
            if char == " ":
                indent_level += 1
            else:
                break
        return indent_level // 2  # Assuming 2 spaces per level

    def _get_indent_tokens(self, level: int) -> List[int]:
        """Get tokens for given indent level"""
        indent_str = " " * (level * 2)
        try:
            return self.tokenizer.encode(indent_str, add_special_tokens=False)
        except:
            return []

    @asynccontextmanager
    async def _inference_context(self):
        """Enhanced context manager for inference operations"""
        async with self.inference_lock:
            async with self.memory_manager.allocation_context():
                monitoring_task = None
                try:
                    # Start memory monitoring
                    monitoring_task = asyncio.create_task(self._monitor_memory())

                    with torch.inference_mode():
                        yield

                finally:
                    # Cancel monitoring and cleanup
                    if monitoring_task:
                        monitoring_task.cancel()
                    torch.cuda.empty_cache()
                    gc.collect()

    async def _monitor_memory(self):
        """Monitor memory usage during generation"""
        while True:
            try:
                memory_info = self.memory_manager.get_memory_info()
                if memory_info["utilization"] > 90:
                    logger.warning("High memory utilization detected")
                    await self.memory_manager.emergency_cleanup()
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")

    async def generate_stream(self, prompt: Union[str, List[str]], generation_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Enhanced streaming text generation with improved handling"""
        if isinstance(prompt, str):
            prompt = [prompt]

        try:
            async with self._inference_context():
                # Prepare enhanced generation config
                gen_config = {
                    **self.config["generation"],
                    **(generation_config or {}),
                    "max_new_tokens": min(
                        generation_config.get("max_new_tokens", self.config["attention"]["max_new_tokens"]),
                        self.max_sequence_length - 100,  # Reserve tokens for prompt
                    ),
                }

                # Enhanced tokenization with length check
                try:
                    tokenized = self.tokenizer(prompt, padding=True, truncation=True, max_length=self.max_sequence_length - gen_config["max_new_tokens"], return_tensors="pt")
                except Exception as e:
                    logger.error(f"Tokenization error: {e}")
                    yield f"Error: Input too long or invalid"
                    return

                # Move tensors to GPU with memory check
                if not await self._handle_memory_pressure(tokenized["input_ids"].numel() * 8):
                    yield f"Error: Insufficient memory for input"
                    return

                input_ids = tokenized["input_ids"].cuda()
                attention_mask = tokenized.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()

                # Initialize enhanced generation state
                self.state = GenerationState(
                    batch_size=len(prompt), max_sequence_length=self.max_sequence_length, max_new_tokens=gen_config["max_new_tokens"], max_time=gen_config.get("max_time", 600)
                )
                self.state.kv_cache.max_length = self.max_sequence_length

                # Create optimized attention pattern
                attention_pattern = self._create_attention_pattern(input_ids.size(1), self.config["attention"]["sliding_window"])

                # Enhanced generation loop
                async for logits in self._process_long_input(input_ids, attention_mask):
                    # Check stopping criteria
                    should_stop, reason = self.state.should_stop()
                    if should_stop:
                        logger.info(f"Generation stopped: {reason}")
                        break

                    # Apply enhanced code-specific optimizations
                    next_token_logits = await self._handle_code_specific_generation(logits[:, -1, :], self.state.current_tokens[0], gen_config)

                    # Enhanced token sampling
                    if gen_config["do_sample"]:
                        # Apply temperature with dynamic scaling
                        temperature = gen_config["temperature"]
                        if self.state.memory_pressure_count > 0:
                            temperature *= 1.1  # Increase randomness under memory pressure

                        probs = F.softmax(next_token_logits / temperature, dim=-1)

                        # Apply top-k if configured
                        if gen_config.get("top_k", 0) > 0:
                            indices_to_remove = probs < torch.topk(probs, gen_config["top_k"])[0][..., -1, None]
                            probs[indices_to_remove] = 0.0
                            probs = probs / probs.sum(dim=-1, keepdim=True)

                        next_tokens = torch.multinomial(probs, num_samples=1)
                    else:
                        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    # Update state with memory monitoring
                    self.state.update_state(next_tokens)

                    # Memory-efficient token processing
                    tokens = next_tokens.squeeze()

                    # Decode with cleanup
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    if text.strip():
                        yield text

                    # Periodic memory check
                    if self.state.generated_tokens % 100 == 0:
                        if not await self._handle_memory_pressure(1024 * 1024):  # 1MB buffer
                            logger.warning("Memory pressure detected during generation")
                            break

            # Enhanced cleanup
            if self.state:
                self.state.clear()
                self.state = None

        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"\nError during generation: {str(e)}"

        finally:
            # Final cleanup
            await self.memory_manager.emergency_cleanup()

    async def __call__(self, prompt: Union[str, List[str]], **kwargs) -> str:
        """Enhanced synchronous interface for generation"""
        start_time = time.time()
        result = []

        try:
            async for chunk in self.generate_stream(prompt, kwargs.get("generation_config")):
                result.append(chunk)

                # Check time limit
                if time.time() - start_time > kwargs.get("timeout", 600):
                    logger.warning("Generation timeout reached")
                    break

        except Exception as e:
            logger.error(f"Generation call error: {e}")
            return f"Error: {str(e)}"

        return "".join(result)

