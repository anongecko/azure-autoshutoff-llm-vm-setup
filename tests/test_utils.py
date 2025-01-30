import pytest
from src.utils import (
    MemoryTracker,
    TokenizerOptimizer,
    CacheManager,
    BatchProcessor,
    ModelOptimizer
)
import torch
import asyncio

@pytest.fixture
def memory_tracker():
    return MemoryTracker()

@pytest.fixture
def cache_manager():
    return CacheManager(max_size_gb=4.0)  # Small size for testing

@pytest.fixture
def batch_processor():
    return BatchProcessor(optimal_batch_size=4)

def test_memory_tracking(memory_tracker):
    """Test memory tracking functionality"""
    initial_stats = memory_tracker.get_memory_stats()
    
    # Allocate some tensors
    tensors = [torch.randn(1000, 1000, device="cuda") for _ in range(5)]
    
    current_stats = memory_tracker.get_memory_stats()
    assert current_stats.allocated > initial_stats.allocated
    
    # Cleanup
    del tensors
    torch.cuda.empty_cache()

@pytest.mark.asyncio
async def test_cache_management(cache_manager):
    """Test cache management"""
    # Allocate cache entries
    success = await cache_manager.allocate("test1", 1 * (1024**3), priority=1.0)
    assert success
    
    # Test eviction
    success = await cache_manager.allocate("test2", 3.5 * (1024**3), priority=2.0)
    assert not success  # Should fail due to size limit
    
    # Update priority
    cache_manager.update_priority("test1", 3.0)
    assert cache_manager.kv_cache["test1"]["priority"] == 3.0

def test_batch_processing(batch_processor):
    """Test batch processing optimization"""
    input_lengths = [100, 200, 300, 400, 500]
    available_memory = 8 * (1024**3)  # 8GB
    
    batch_size = batch_processor.optimize_batch_size(
        input_lengths,
        available_memory
    )
    assert 1 <= batch_size <= batch_processor.optimal_batch_size

def test_model_optimization():
    """Test model optimization utilities"""
    optimizer = ModelOptimizer()
    
    # Test configuration optimization
    config = optimizer.optimize_model_config({})
    assert config["use_cache"]
    assert config["flash_attention"]
    
    # Test compilation
    model = torch.nn.Linear(10, 10).cuda()
    compiled_model = optimizer.compile_model(model)
    assert compiled_model is not None

