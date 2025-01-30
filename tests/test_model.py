import pytest
import torch
from src.model import ModelManager
from src.utils import MemoryTracker, TokenizerOptimizer
import asyncio
from typing import Generator
import gc

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    torch.cuda.empty_cache()
    gc.collect()

@pytest.fixture
def model_manager():
    """Initialize model manager for testing"""
    manager = ModelManager()
    manager.load_model({
        "max_batch_size": 4,  # Smaller batch size for testing
        "prefill_chunk_size": 2048,
        "stream_chunk_size": 8
    })
    yield manager
    manager.cleanup()

@pytest.mark.asyncio
async def test_model_initialization(model_manager):
    """Test model initialization"""
    assert model_manager.is_loaded
    assert model_manager.model is not None
    assert model_manager.tokenizer is not None
    assert model_manager.inference is not None

@pytest.mark.asyncio
async def test_inference_generation(model_manager):
    """Test text generation"""
    prompt = "def quicksort(arr):"
    response = []
    async for chunk in model_manager.inference.generate_stream(prompt):
        response.append(chunk)
    assert len(response) > 0
    assert isinstance("".join(response), str)

@pytest.mark.asyncio
async def test_long_context_handling(model_manager):
    """Test handling of long context"""
    long_prompt = "# " * 65536  # 64K tokens
    async for chunk in model_manager.inference.generate_stream(long_prompt, max_length=131072):
        assert isinstance(chunk, str)

@pytest.mark.asyncio
async def test_memory_management(model_manager):
    """Test memory management during inference"""
    memory_tracker = MemoryTracker()
    initial_stats = memory_tracker.get_memory_stats()
    
    # Generate multiple responses
    prompts = ["def fibonacci(n):" for _ in range(5)]
    for prompt in prompts:
        async for _ in model_manager.inference.generate_stream(prompt):
            pass
    
    final_stats = memory_tracker.get_memory_stats()
    assert abs(final_stats.allocated - initial_stats.allocated) < 1e9  # Less than 1GB diff

