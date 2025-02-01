from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
import torch
import logging
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import gc
import asyncio
from ..utils import OptimizedMemoryManager
from contextlib import asynccontextmanager
import json
import os
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""

    pass


@dataclass
class ModelCheckpoint:
    """Track model loading checkpoints"""

    path: Path
    config_hash: str
    weights_hash: str
    last_loaded: float
    load_time: float
    memory_usage: float


class EnhancedModelLoader:
    """
    Enhanced model loader optimized for DeepSeek model with fast startup.
    Implements checkpoint tracking, parallel loading, and memory optimization.
    """

    def __init__(self, memory_manager: "OptimizedMemoryManager", device: str = "cuda:0"):
        self.start_time = time.time()
        self._initialize_configs()

        self.model_path = Path(self.model_config["model_path"])
        self.memory_manager = memory_manager
        self.device = torch.device(device)
        self.model: Optional[PreTrainedModel] = None
        self.config = None
        self.checkpoint_dir = Path("/tmp/model_checkpoints")
        self.checkpoint_info: Optional[ModelCheckpoint] = None

        # Enhanced config loading
        self.hardware_config = self.model_config["hardware"]
        self.optimization_config = self.model_config["optimization"]
        self.attention_config = self.model_config["attention"]

        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Initialize environment and settings
        self._validate_environment()
        self._configure_torch_settings()
        self._setup_checkpoint_dir()

        logger.info(f"Model loader initialized in {time.time() - self.start_time:.2f}s")

    def _initialize_configs(self):
        """Load configurations with caching"""
        config_path = Path("config")

        try:
            # Load configs in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                api_future = executor.submit(self._load_json, config_path / "api_config.json")
                model_future = executor.submit(self._load_json, config_path / "model_config.json")

                self.api_config = api_future.result()
                self.model_config = model_future.result()

        except Exception as e:
            raise ModelLoadError(f"Failed to load configurations: {e}")

    @staticmethod
    def _load_json(path: Path) -> Dict:
        """Load JSON with error handling"""
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            raise ModelLoadError(f"Failed to load {path}: {e}")

    def _validate_environment(self) -> None:
        """Enhanced environment validation with parallel checks"""
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA must be available for model loading")

        try:
            # Run checks in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                gpu_future = executor.submit(self._check_gpu)
                path_future = executor.submit(self._check_model_path)
                memory_future = executor.submit(self._check_memory)

                # Get results and handle errors
                gpu_error = gpu_future.result()
                path_error = path_future.result()
                memory_error = memory_future.result()

                if any([gpu_error, path_error, memory_error]):
                    raise ModelLoadError("\n".join(filter(None, [gpu_error, path_error, memory_error])))

        except Exception as e:
            raise ModelLoadError(f"Environment validation failed: {e}")

    def _check_gpu(self) -> Optional[str]:
        """Check GPU compatibility"""
        gpu_name = torch.cuda.get_device_name(0)
        if "H100" not in gpu_name:
            return f"Warning: Expected H100 GPU, found: {gpu_name}"
        return None

    def _check_model_path(self) -> Optional[str]:
        """Validate model path and files"""
        if not self.model_path.exists():
            return f"Model path does not exist: {self.model_path}"

        required_files = ["config.json", "model.safetensors"]
        missing_files = [f for f in required_files if not (self.model_path / f).exists()]
        if missing_files:
            return f"Missing required files: {', '.join(missing_files)}"

        return None

    def _check_memory(self) -> Optional[str]:
        """Validate memory requirements"""
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory < 75 * (1024**3):
            return f"Insufficient GPU memory: {total_memory / (1024**3):.2f}GB (minimum 75GB required)"
        return None

    def _configure_torch_settings(self) -> None:
        """Configure optimized PyTorch settings"""
        # Enhanced CUDA configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"expandable_segments:True,"
            f"garbage_collection_threshold:{self.hardware_config['offload_config']['cpu_offload_threshold']},"
            f"max_split_size_mb:512,"
            f"roundup_power2_divisions:16,"
            f"backend:native,"
            f"max_split_size_mb:512"
        )

        # Optimal thread configuration
        os.environ["OMP_NUM_THREADS"] = "40"
        os.environ["MKL_NUM_THREADS"] = "40"
        torch.set_num_threads(40)
        torch.set_num_interop_threads(40)

        # Configure CUDA settings
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("high")

        # Optimize backends
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def _setup_checkpoint_dir(self):
        """Setup checkpoint directory with cleanup"""
        try:
            self.checkpoint_dir.mkdir(exist_ok=True)
            # Cleanup old checkpoints
            for path in self.checkpoint_dir.glob("*"):
                if path.stat().st_mtime < time.time() - 86400:  # 24 hours
                    shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Checkpoint directory setup failed: {e}")

    def _compute_model_hash(self) -> str:
        """Compute hash of model configuration and weights"""
        try:
            config_path = self.model_path / "config.json"
            weights_path = self.model_path / "model.safetensors"

            hasher = hashlib.sha256()

            with open(config_path, "rb") as f:
                hasher.update(f.read())

            # Only hash first 1MB of weights for speed
            with open(weights_path, "rb") as f:
                hasher.update(f.read(1024 * 1024))

            return hasher.hexdigest()

        except Exception as e:
            logger.warning(f"Failed to compute model hash: {e}")
            return ""

    async def _load_and_validate_config(self) -> None:
        """Enhanced configuration loading and validation"""
        try:
            # Load config with fast tokenizer settings
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, use_fast=True, use_auth_token=None)

            # Validate architecture
            if self.config.architectures[0] != "Qwen2ForCausalLM":
                raise ModelLoadError(f"Unexpected model architecture: {self.config.architectures[0]}")

            # Enhanced configuration
            self.config.use_cache = self.optimization_config["use_cache"]
            self.config.pretraining_tp = 1
            self.config.max_position_embeddings = self.model_config["max_sequence_length"]
            self.config.sliding_window = self.attention_config["sliding_window"]
            self.config.rope_theta = 1000000.0  # Optimize for long sequences
            self.config.use_flash_attention = self.attention_config["use_flash_attention"]
            self.config.attention_sink_size = min(2048, self.model_config["max_sequence_length"] // 64)

            if self.attention_config["use_flash_attention"]:
                self.config.use_flash_attention_2 = True
                self.config.attention_implementation = "flash_attention_2"

            logger.info("Model configuration optimized successfully")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model configuration: {str(e)}")

    async def _prepare_model_loading(self) -> None:
        """Enhanced model loading preparation"""
        try:
            logger.info("Preparing for model loading...")

            # Check for existing checkpoint
            model_hash = self._compute_model_hash()
            checkpoint_path = self.checkpoint_dir / model_hash

            if checkpoint_path.exists():
                logger.info("Found existing checkpoint")
                self.checkpoint_info = self._load_checkpoint_info(checkpoint_path)
                if self._validate_checkpoint():
                    return

            # Request memory cleanup
            if not await self.memory_manager.prepare_for_model_load():
                raise ModelLoadError("Failed to prepare memory for model loading")

            # Configure offload
            if self.hardware_config["offload_config"]["enabled"]:
                offload_path = Path(self.hardware_config["offload_config"]["offload_folder"])
                offload_path.mkdir(exist_ok=True)
                logger.info(f"Configured offload directory: {offload_path}")

        except Exception as e:
            raise ModelLoadError(f"Failed to prepare for model loading: {str(e)}")

    def _validate_checkpoint(self) -> bool:
        """Validate existing checkpoint"""
        try:
            if not self.checkpoint_info:
                return False

            # Check if checkpoint is recent and valid
            if time.time() - self.checkpoint_info.last_loaded > 3600:  # 1 hour
                return False

            # Verify hashes
            current_hash = self._compute_model_hash()
            if current_hash != self.checkpoint_info.config_hash:
                return False

            return True
        except Exception:
            return False

    def _optimize_loaded_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Enhanced model optimization with parallel processing"""
        try:
            # Basic setup
            model.eval()
            model = model.to(dtype=torch.bfloat16)

            # Apply optimizations in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                # Memory optimization
                futures.append(executor.submit(self._apply_memory_optimizations, model))

                # Attention optimization
                futures.append(executor.submit(self._apply_attention_optimizations, model))

                # Sequence optimization
                futures.append(executor.submit(self._apply_sequence_optimizations, model))

                # Wait for all optimizations
                for future in futures:
                    future.result()

            # Apply torch compile if enabled
            if self.optimization_config["torch_compile"]:
                model = self._apply_torch_compile(model)

            return model

        except Exception as e:
            raise ModelLoadError(f"Model optimization failed: {str(e)}")

    def _apply_memory_optimizations(self, model: PreTrainedModel):
        """Apply memory-specific optimizations"""
        model.config.use_cache = self.optimization_config["use_cache"]
        if hasattr(model.config, "use_memory_efficient_attention"):
            model.config.use_memory_efficient_attention = True

    def _apply_attention_optimizations(self, model: PreTrainedModel):
        """Apply attention-specific optimizations"""
        if self.attention_config["use_flash_attention"]:
            model.config.attention_implementation = "flash_attention_2"
            if hasattr(model.config, "use_flash_attention_2"):
                model.config.use_flash_attention_2 = True

    def _apply_sequence_optimizations(self, model: PreTrainedModel):
        """Apply sequence-specific optimizations"""
        if hasattr(model.config, "sequence_parallel_enabled"):
            model.config.sequence_parallel_enabled = True
        model.config.max_position_embeddings = self.model_config["max_sequence_length"]

    def _apply_torch_compile(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply torch compile with optimal settings"""
        try:
            compiled_model = torch.compile(
                model,
                backend=self.optimization_config["compile_backend"],
                mode=self.optimization_config["compile_mode"],
                fullgraph=False,
                dynamic=self.optimization_config["compile_dynamic"],
                options={"max_autotune": True, "trace_only": False, "aot_autograd": True},
            )
            return compiled_model
        except Exception as e:
            logger.warning(f"Torch compile failed, using original model: {e}")
            return model

    async def _setup_tokenizer(self) -> AutoTokenizer:
        """Enhanced tokenizer setup with fast initialization"""
        try:
            # Initialize tokenizer with optimal settings
            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                self.model_path,
                trust_remote_code=True,
                use_fast=True,
                model_max_length=self.model_config["max_sequence_length"],
                use_auth_token=None,
                local_files_only=True,
            )

            # Configure padding
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # Add code-specific tokens
            if self.model_config["code_specific"]["preserve_indentation"]:
                indent_tokens = [" " * i for i in range(2, 17, 2)]  # 2-16 spaces
                tokenizer.add_special_tokens({"additional_special_tokens": indent_tokens})

            return tokenizer

        except Exception as e:
            raise ModelLoadError(f"Failed to setup tokenizer: {str(e)}")

    async def _warmup_model(self, model: PreTrainedModel, tokenizer: AutoTokenizer) -> None:
        """Enhanced model warmup with progressive sequence lengths"""
        try:
            logger.info("Starting model warmup...")
            warmup_start = time.time()

            # Progressive warmup sequences
            warmup_sequences = [
                ("def add(x, y):", 32),  # Short
                ("def quicksort(arr):\n    if len(arr) <= 1:\n        return arr", 128),  # Medium
                ("# Implementing a transformer with attention\nclass TransformerBlock:\n    def __init__(self):", 256),  # Long
                ("# Complex system implementation\n" + "def process_batch(self, inputs):\n" * 4, 512),  # Very long
            ]

            async def run_warmup(prompt: str, max_length: int):
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=max_length, truncation=True).to(self.device)

                    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                        outputs = model(**inputs)
                        del outputs

                    torch.cuda.synchronize()
                    await self.memory_manager._aggressive_cleanup()

                except Exception as e:
                    logger.warning(f"Warmup sequence failed: {e}")

            # Run warmup sequences in order
            for prompt, length in warmup_sequences:
                await run_warmup(prompt, length)

            # Run parallel inference test
            try:
                parallel_prompt = "def test(): pass"
                tasks = [run_warmup(parallel_prompt, 64) for _ in range(2)]
                await asyncio.gather(*tasks)
            except Exception as e:
                logger.warning(f"Parallel warmup test failed: {e}")

            warmup_time = time.time() - warmup_start
            logger.info(f"Model warmup completed in {warmup_time:.2f}s")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _create_checkpoint(self, model: PreTrainedModel) -> None:
        """Create model checkpoint for faster loading"""
        try:
            model_hash = self._compute_model_hash()
            checkpoint_path = self.checkpoint_dir / model_hash
            checkpoint_path.mkdir(exist_ok=True)

            # Save checkpoint info
            info = ModelCheckpoint(
                path=checkpoint_path,
                config_hash=model_hash,
                weights_hash=model_hash,
                last_loaded=time.time(),
                load_time=time.time() - self.start_time,
                memory_usage=torch.cuda.max_memory_allocated() / (1024**3),
            )

            with open(checkpoint_path / "info.json", "w") as f:
                json.dump(dataclass.asdict(info), f)

            # Save model config
            model.config.save_pretrained(checkpoint_path)

            logger.info(f"Created checkpoint at {checkpoint_path}")

        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")

    def _load_checkpoint_info(self, path: Path) -> Optional[ModelCheckpoint]:
        """Load checkpoint information"""
        try:
            with open(path / "info.json") as f:
                data = json.load(f)
            return ModelCheckpoint(**data)
        except Exception:
            return None

    async def load_model(self) -> Tuple[PreTrainedModel, AutoTokenizer]:
        """
        Enhanced model loading with optimizations and checkpointing.

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelLoadError: If model loading fails
        """
        load_start = time.time()

        try:
            # Initial setup
            await self._load_and_validate_config()
            await self._prepare_model_loading()

            logger.info(f"Loading model from {self.model_path}...")

            # Optimize memory before loading
            torch.cuda.empty_cache()
            gc.collect()

            # Enhanced model loading with error handling
            try:
                model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    self.model_path,
                    config=self.config,
                    torch_dtype=torch.bfloat16,
                    device_map=self.hardware_config["device_map"],
                    max_memory=self.hardware_config["max_memory"],
                    offload_folder=self.hardware_config["offload_config"]["offload_folder"],
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    offload_state_dict=True,
                )
            except Exception as e:
                raise ModelLoadError(f"Model loading failed: {e}")

            if model is None:
                raise ModelLoadError("Model failed to load (returned None)")

            # Load tokenizer in parallel with model optimization
            tokenizer_task = asyncio.create_task(self._setup_tokenizer())

            # Apply optimizations
            logger.info("Applying model optimizations...")
            model = self._optimize_loaded_model(model)

            # Wait for tokenizer
            tokenizer = await tokenizer_task

            # Verify model location
            device_param = next(model.parameters()).device
            if device_param != self.device:
                raise ModelLoadError(f"Model not on correct device. Expected {self.device}, got {device_param}")

            # Create checkpoint for faster future loading
            self._create_checkpoint(model)

            # Perform optimized warmup
            await self._warmup_model(model, tokenizer)

            # Log performance metrics
            load_time = time.time() - load_start
            model_size = sum(p.numel() for p in model.parameters()) / 1e9
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

            logger.info(f"Model loaded successfully:\n- Load time: {load_time:.2f}s\n- Parameters: {model_size:.2f}B\n- Memory usage: {memory_used:.2f}GB\n- Peak memory: {peak_memory:.2f}GB")

            return model, tokenizer

        except Exception as e:
            # Enhanced error cleanup
            gc.collect()
            torch.cuda.empty_cache()
            await self.memory_manager.emergency_cleanup()

            error_context = {"load_time": time.time() - load_start, "memory_allocated": torch.cuda.memory_allocated() / (1024**3), "memory_reserved": torch.cuda.memory_reserved() / (1024**3)}
            logger.error(f"Model loading failed with context: {error_context}")

            raise ModelLoadError(f"Model loading failed: {str(e)}")

    def get_model_info(self, model: Optional[PreTrainedModel] = None) -> Dict[str, Any]:
        """Enhanced model information with detailed metrics"""
        try:
            if model is None:
                model = self.model

            if model is None:
                return {"status": "not_loaded"}

            memory_stats = torch.cuda.memory_stats(self.device)

            return {
                "status": "loaded",
                "model_path": str(self.model_path),
                "architecture": model.config.architectures[0],
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
                "memory": {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
                    "memory_segments": memory_stats.get("num_alloc_retries", 0),
                    "fragmentation": memory_stats.get("allocated_bytes.all.current", 0) / max(1, memory_stats.get("reserved_bytes.all.current", 1)),
                },
                "model_config": {
                    "context_window": model.config.max_position_embeddings,
                    "vocab_size": model.config.vocab_size,
                    "hidden_size": model.config.hidden_size,
                    "num_attention_heads": model.config.num_attention_heads,
                    "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
                    "intermediate_size": model.config.intermediate_size,
                },
                "optimization": {
                    "torch_compile": self.optimization_config["torch_compile"],
                    "compile_mode": self.optimization_config["compile_mode"],
                    "flash_attention": self.attention_config["use_flash_attention"],
                    "memory_efficient": self.optimization_config["memory_efficient_attention"],
                    "sequence_parallel": getattr(model.config, "sequence_parallel_enabled", False),
                    "device_map": self.hardware_config["device_map"],
                },
                "checkpoint": {
                    "exists": bool(self.checkpoint_info),
                    "last_loaded": self.checkpoint_info.last_loaded if self.checkpoint_info else None,
                    "load_time": self.checkpoint_info.load_time if self.checkpoint_info else None,
                },
            }

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "message": str(e)}

    def __del__(self):
        """Enhanced cleanup on deletion"""
        try:
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=False)

            if hasattr(self, "model") and self.model is not None:
                del self.model

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

