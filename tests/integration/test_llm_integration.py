"""
Test Suite for LLM Integration (v0.4.12)

Tests for Llama, Mistral, Phi optimization wrappers and KV-cache.
Validates optimization, quantization, and backend integration.

Version: 0.4.12
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLLMTypes:
    """Tests for LLMType enum."""

    def test_llm_types_exist(self):
        """Verify all LLM types are defined."""
        from torchbridge.models.llm.llm_optimizer import LLMType

        assert hasattr(LLMType, 'LLAMA')
        assert hasattr(LLMType, 'MISTRAL')
        assert hasattr(LLMType, 'PHI')
        assert hasattr(LLMType, 'QWEN')
        assert hasattr(LLMType, 'GEMMA')
        assert hasattr(LLMType, 'CUSTOM')

    def test_llm_type_values(self):
        """Verify LLM type values."""
        from torchbridge.models.llm.llm_optimizer import LLMType

        assert LLMType.LLAMA.value == "llama"
        assert LLMType.MISTRAL.value == "mistral"
        assert LLMType.PHI.value == "phi"


class TestQuantizationMode:
    """Tests for QuantizationMode enum."""

    def test_quantization_modes_exist(self):
        """Verify all quantization modes are defined."""
        from torchbridge.models.llm.llm_optimizer import QuantizationMode

        assert hasattr(QuantizationMode, 'NONE')
        assert hasattr(QuantizationMode, 'INT8')
        assert hasattr(QuantizationMode, 'INT4')
        assert hasattr(QuantizationMode, 'FP8')
        assert hasattr(QuantizationMode, 'BNBT4')

    def test_quantization_mode_values(self):
        """Verify quantization mode values."""
        from torchbridge.models.llm.llm_optimizer import QuantizationMode

        assert QuantizationMode.NONE.value == "none"
        assert QuantizationMode.INT8.value == "int8"
        assert QuantizationMode.INT4.value == "int4"
        assert QuantizationMode.FP8.value == "fp8"
        assert QuantizationMode.BNBT4.value == "bnb_4bit"


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_config(self):
        """Test default generation config values."""
        from torchbridge.models.llm.llm_optimizer import GenerationConfig

        config = GenerationConfig()

        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.1
        assert config.do_sample is True
        assert config.use_cache is True

    def test_custom_config(self):
        """Test custom generation config."""
        from torchbridge.models.llm.llm_optimizer import GenerationConfig

        config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.9,
            do_sample=False
        )

        assert config.max_new_tokens == 512
        assert config.temperature == 0.9
        assert config.do_sample is False


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_llm_config(self):
        """Test default LLM config values."""
        from torchbridge.models.llm.llm_optimizer import (
            LLMConfig,
            LLMType,
            QuantizationMode,
        )

        config = LLMConfig()

        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.model_type == LLMType.LLAMA
        assert config.max_sequence_length == 4096
        assert config.quantization == QuantizationMode.NONE
        assert config.use_flash_attention is True
        assert config.use_torch_compile is True
        assert config.compile_mode == "reduce-overhead"
        assert config.use_kv_cache is True
        assert config.device == "auto"
        assert config.device_map == "auto"

    def test_custom_llm_config(self):
        """Test custom LLM config."""
        from torchbridge.models.llm.llm_optimizer import (
            LLMConfig,
            LLMType,
            QuantizationMode,
        )

        config = LLMConfig(
            model_name="mistralai/Mistral-7B-v0.1",
            model_type=LLMType.MISTRAL,
            quantization=QuantizationMode.INT8,
            max_sequence_length=8192,
        )

        assert config.model_name == "mistralai/Mistral-7B-v0.1"
        assert config.model_type == LLMType.MISTRAL
        assert config.quantization == QuantizationMode.INT8
        assert config.max_sequence_length == 8192


class TestLLMOptimizer:
    """Tests for LLMOptimizer class."""

    def test_optimizer_creation(self):
        """Test optimizer instantiation."""
        from torchbridge.models.llm.llm_optimizer import LLMOptimizer

        optimizer = LLMOptimizer()

        assert optimizer.device is not None
        assert optimizer.dtype is not None
        assert optimizer.config is not None

    def test_optimizer_with_custom_config(self):
        """Test optimizer with custom configuration."""
        from torchbridge.models.llm.llm_optimizer import (
            LLMConfig,
            LLMOptimizer,
            QuantizationMode,
        )

        config = LLMConfig(
            quantization=QuantizationMode.INT8,
            use_torch_compile=False,
        )

        optimizer = LLMOptimizer(config)

        assert optimizer.config.quantization == QuantizationMode.INT8
        assert optimizer.config.use_torch_compile is False

    def test_get_optimization_info(self):
        """Test optimization info retrieval."""
        from torchbridge.models.llm.llm_optimizer import LLMOptimizer

        optimizer = LLMOptimizer()
        info = optimizer.get_optimization_info()

        assert "device" in info
        assert "dtype" in info
        assert "backend" in info
        assert "quantization" in info
        assert "flash_attention" in info
        assert "torch_compile" in info
        assert "kv_cache" in info

    def test_estimate_memory_7b(self):
        """Test memory estimation for 7B model."""
        from torchbridge.models.llm.llm_optimizer import LLMConfig, LLMOptimizer

        config = LLMConfig()
        optimizer = LLMOptimizer(config)

        estimate = optimizer.estimate_memory("llama-7b")

        assert "model_memory_gb" in estimate
        assert "kv_cache_gb" in estimate
        assert "total_gb" in estimate
        assert estimate["model_memory_gb"] > 0

    def test_estimate_memory_with_quantization(self):
        """Test memory estimation with quantization."""
        from torchbridge.models.llm.llm_optimizer import (
            LLMConfig,
            LLMOptimizer,
            QuantizationMode,
        )

        # Test that INT4 uses less memory than INT8
        config_int8 = LLMConfig(quantization=QuantizationMode.INT8)
        optimizer_int8 = LLMOptimizer(config_int8)
        estimate_int8 = optimizer_int8.estimate_memory("llama-7b")

        config_int4 = LLMConfig(quantization=QuantizationMode.INT4)
        optimizer_int4 = LLMOptimizer(config_int4)
        estimate_int4 = optimizer_int4.estimate_memory("llama-7b")

        # INT4 should use less memory than INT8
        assert estimate_int4["model_memory_gb"] < estimate_int8["model_memory_gb"]
        # Both should have valid memory estimates
        assert estimate_int8["model_memory_gb"] > 0
        assert estimate_int4["model_memory_gb"] > 0

    def test_model_type_detection_llama(self):
        """Test Llama model type detection."""
        from torchbridge.models.llm.llm_optimizer import LLMOptimizer, LLMType

        optimizer = LLMOptimizer()

        class MockLlamaModel(nn.Module):
            pass

        MockLlamaModel.__name__ = "LlamaForCausalLM"
        model = MockLlamaModel()

        detected = optimizer._detect_model_type(model)
        assert detected == LLMType.LLAMA

    def test_model_type_detection_mistral(self):
        """Test Mistral model type detection."""
        from torchbridge.models.llm.llm_optimizer import LLMOptimizer, LLMType

        optimizer = LLMOptimizer()

        class MockMistralModel(nn.Module):
            pass

        MockMistralModel.__name__ = "MistralForCausalLM"
        model = MockMistralModel()

        detected = optimizer._detect_model_type(model)
        assert detected == LLMType.MISTRAL

    def test_model_type_detection_phi(self):
        """Test Phi model type detection."""
        from torchbridge.models.llm.llm_optimizer import LLMOptimizer, LLMType

        optimizer = LLMOptimizer()

        class MockPhiModel(nn.Module):
            pass

        MockPhiModel.__name__ = "PhiForCausalLM"
        model = MockPhiModel()

        detected = optimizer._detect_model_type(model)
        assert detected == LLMType.PHI


class TestKVCacheManager:
    """Tests for KVCacheManager class."""

    def test_cache_manager_creation(self):
        """Test cache manager creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            max_length=2048,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            dtype=torch.float16,
            device="cpu"
        )

        manager = KVCacheManager(config)
        assert manager.config == config

    def test_create_cache(self):
        """Test cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu"
        )

        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=2)

        assert len(cache) == 4  # num_layers
        for key_cache, value_cache in cache:
            assert key_cache.shape == (2, 8, 0, 64)
            assert value_cache.shape == (2, 8, 0, 64)

    def test_update_cache(self):
        """Test cache update."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            max_length=100,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        # Add some keys/values
        new_keys = torch.randn(1, 4, 10, 32)
        new_values = torch.randn(1, 4, 10, 32)

        cache = manager.update_cache(cache, new_keys, new_values, layer_idx=0)

        assert cache[0][0].shape[2] == 10
        assert cache[0][1].shape[2] == 10

    def test_cache_truncation(self):
        """Test that cache truncates when exceeding max length."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            max_length=20,  # Small max length
            num_layers=1,
            num_heads=2,
            head_dim=16,
            device="cpu"
        )

        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        # Add more than max_length
        for _ in range(5):
            new_keys = torch.randn(1, 2, 10, 16)
            new_values = torch.randn(1, 2, 10, 16)
            cache = manager.update_cache(cache, new_keys, new_values, 0)

        # Should be truncated to max_length
        assert cache[0][0].shape[2] == 20

    def test_get_cache_length(self):
        """Test cache length retrieval."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(num_layers=1, num_heads=2, head_dim=16, device="cpu")
        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        assert manager.get_cache_length(cache) == 0

        new_keys = torch.randn(1, 2, 5, 16)
        new_values = torch.randn(1, 2, 5, 16)
        cache = manager.update_cache(cache, new_keys, new_values, 0)

        assert manager.get_cache_length(cache) == 5

    def test_get_memory_usage(self):
        """Test memory usage calculation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(num_layers=2, num_heads=4, head_dim=32, device="cpu")
        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        new_keys = torch.randn(1, 4, 10, 32)
        new_values = torch.randn(1, 4, 10, 32)
        cache = manager.update_cache(cache, new_keys, new_values, 0)

        usage = manager.get_memory_usage(cache)

        assert "cache_memory_mb" in usage
        assert "cache_length" in usage
        assert "num_layers" in usage
        assert usage["cache_memory_mb"] > 0


class TestPagedKVCache:
    """Tests for PagedKVCache class."""

    def test_paged_cache_creation(self):
        """Test paged cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=16,
            page_size=8,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        assert paged_cache.config == config

    def test_initialize_pages(self):
        """Test page initialization."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=8,
            page_size=4,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        paged_cache.initialize_pages()

        assert paged_cache.physical_pages is not None
        assert len(paged_cache.free_pages) == 8

    def test_allocate_pages(self):
        """Test page allocation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=16,
            page_size=4,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        page_table = paged_cache.allocate_pages(batch_size=2, num_pages_per_seq=4)

        assert page_table.shape == (2, 4)
        assert len(paged_cache.free_pages) == 8  # 16 - 8 allocated

    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=8,
            page_size=4,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        paged_cache.initialize_pages()
        paged_cache.allocate_pages(batch_size=1, num_pages_per_seq=2)

        usage = paged_cache.get_memory_usage()

        assert "total_mb" in usage
        assert "used_pages" in usage
        assert "free_pages" in usage
        assert "utilization" in usage
        assert usage["used_pages"] == 2
        assert usage["free_pages"] == 6


class TestSlidingWindowCache:
    """Tests for SlidingWindowCache class."""

    def test_sliding_window_creation(self):
        """Test sliding window cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=256,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        assert sw_cache.window_size == 256

    def test_create_cache(self):
        """Test cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=128,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        cache = sw_cache.create_cache(batch_size=1)

        assert len(cache) == 2
        assert cache[0][0].shape[2] == 0

    def test_update_within_window(self):
        """Test update within window size."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=100,
            num_layers=1,
            num_heads=2,
            head_dim=16,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        cache = sw_cache.create_cache(batch_size=1)

        new_keys = torch.randn(1, 2, 50, 16)
        new_values = torch.randn(1, 2, 50, 16)

        cache = sw_cache.update_cache(cache, new_keys, new_values, 0)

        assert cache[0][0].shape[2] == 50

    def test_sliding_window_truncation(self):
        """Test that cache slides when exceeding window size."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=50,
            num_layers=1,
            num_heads=2,
            head_dim=16,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        cache = sw_cache.create_cache(batch_size=1)

        # Add more than window size
        for _ in range(3):
            new_keys = torch.randn(1, 2, 30, 16)
            new_values = torch.randn(1, 2, 30, 16)
            cache = sw_cache.update_cache(cache, new_keys, new_values, 0)

        # Should be truncated to window_size
        assert cache[0][0].shape[2] == 50

    def test_get_window_mask(self):
        """Test window mask generation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(window_size=10, device="cpu")
        sw_cache = SlidingWindowCache(config)

        mask = sw_cache.get_window_mask(seq_len=5, cache_len=20)

        assert mask.shape == (5, 25)  # seq_len x (cache_len + seq_len)


class TestFactoryFunction:
    """Tests for create_optimized_llm factory function."""

    def test_factory_function_exists(self):
        """Test factory function is importable."""
        from torchbridge.models.llm import create_optimized_llm

        assert callable(create_optimized_llm)

    def test_factory_quantization_mapping(self):
        """Test quantization string mapping."""
        from torchbridge.models.llm.llm_optimizer import LLMConfig, QuantizationMode

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.INT4,
            "fp8": QuantizationMode.FP8,
            "bnb_4bit": QuantizationMode.BNBT4,
        }

        for quant_str, quant_enum in quant_map.items():
            config = LLMConfig(
                quantization=quant_map.get(quant_str, QuantizationMode.NONE)
            )
            assert config.quantization == quant_enum


class TestModuleExports:
    """Tests for module exports."""

    def test_llm_module_exports(self):
        """Test LLM module exports all required classes."""
        from torchbridge.models.llm import (
            GenerationConfig,
            KVCacheManager,
            LLMConfig,
            LLMOptimizer,
            OptimizedLlama,
            OptimizedMistral,
            OptimizedPhi,
            PagedKVCache,
            QuantizationMode,
            SlidingWindowCache,
            create_optimized_llm,
        )

        assert LLMOptimizer is not None
        assert LLMConfig is not None
        assert OptimizedLlama is not None
        assert OptimizedMistral is not None
        assert OptimizedPhi is not None
        assert create_optimized_llm is not None
        assert QuantizationMode is not None
        assert GenerationConfig is not None
        assert KVCacheManager is not None
        assert PagedKVCache is not None
        assert SlidingWindowCache is not None

    def test_models_module_exports_llm(self):
        """Test models module exports LLM components."""
        from torchbridge.models import (
            LLMConfig,
            LLMOptimizer,
            create_optimized_llm,
        )

        assert LLMOptimizer is not None
        assert LLMConfig is not None
        assert create_optimized_llm is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
