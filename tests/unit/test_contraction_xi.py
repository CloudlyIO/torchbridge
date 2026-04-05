"""
Regression tests for v0.5.77 Contraction XI — models/ + inference/ cleanup.

These tests verify:
- Deleted classes are no longer importable
- QuantizedKVCache retains its dtype-resolution capability
- Wrapper methods removed from QuantizedKVCache
- to_json() wrappers removed from DisaggregatedFleetConfig and KVHandoffSpec
- PhaseDetector not importable
"""

import importlib

import pytest

from torchbridge.models.llm.kv.cache_dtype import KVCacheDtype

# ---------------------------------------------------------------------------
# Deleted classes must not be importable (models/)
# ---------------------------------------------------------------------------


class TestDeletedModelClasses:
    def test_kv_cache_manager_not_importable(self):
        """KVCacheManager must be deleted — it was a pure torch.zeros() wrapper."""
        with pytest.raises(ImportError):
            importlib.import_module("torchbridge.models.llm.kv_cache")

    def test_kv_cache_manager_not_in_models_init(self):
        import torchbridge.models as m

        assert not hasattr(m, "KVCacheManager")

    def test_paged_kv_cache_not_in_models_init(self):
        import torchbridge.models as m

        assert not hasattr(m, "PagedKVCache")

    def test_sliding_window_cache_not_in_models_init(self):
        import torchbridge.models as m

        assert not hasattr(m, "SlidingWindowCache")

    def test_prefix_cache_not_importable(self):
        """PrefixCache must be deleted — it was never activated in production."""
        from torchbridge.models.llm.kv import quantized_cache

        assert not hasattr(quantized_cache, "PrefixCache")

    def test_prefix_cache_entry_not_importable(self):
        from torchbridge.models.llm.kv import quantized_cache

        assert not hasattr(quantized_cache, "PrefixCacheEntry")


# ---------------------------------------------------------------------------
# Deleted class must not be importable (inference/)
# ---------------------------------------------------------------------------


class TestDeletedInferenceClasses:
    def test_phase_detector_not_importable(self):
        """phase_detection.py must be deleted — never called in production."""
        with pytest.raises(ImportError):
            importlib.import_module("torchbridge.inference.phase_detection")

    def test_phase_detector_not_in_inference_init(self):
        import torchbridge.inference as inf

        assert not hasattr(inf, "PhaseDetector")

    def test_phase_type_not_in_inference_init(self):
        import torchbridge.inference as inf

        assert not hasattr(inf, "PhaseType")


# ---------------------------------------------------------------------------
# QuantizedKVCache must not carry wrapper methods
# ---------------------------------------------------------------------------


class TestQuantizedKVCacheNoWrappers:
    def test_no_create_cache_method(self):
        from torchbridge.models.llm.kv.quantized_cache import QuantizedKVCache

        assert not hasattr(QuantizedKVCache, "create_cache"), (
            "create_cache() was a passthrough wrapper — deleted in v0.5.77"
        )

    def test_no_update_cache_method(self):
        from torchbridge.models.llm.kv.quantized_cache import QuantizedKVCache

        assert not hasattr(QuantizedKVCache, "update_cache")

    def test_no_quantize_tensor_method(self):
        from torchbridge.models.llm.kv.quantized_cache import QuantizedKVCache

        assert not hasattr(QuantizedKVCache, "_quantize_tensor")

    def test_no_get_memory_usage_method(self):
        from torchbridge.models.llm.kv.quantized_cache import QuantizedKVCache

        assert not hasattr(QuantizedKVCache, "get_memory_usage")

    def test_no_lookup_prefix_method(self):
        from torchbridge.models.llm.kv.quantized_cache import QuantizedKVCache

        assert not hasattr(QuantizedKVCache, "lookup_prefix")

    def test_no_store_prefix_method(self):
        from torchbridge.models.llm.kv.quantized_cache import QuantizedKVCache

        assert not hasattr(QuantizedKVCache, "store_prefix")


# ---------------------------------------------------------------------------
# QuantizedKVCache genuine value must be preserved
# ---------------------------------------------------------------------------


class TestQuantizedKVCacheDtypeResolution:
    def test_cpu_resolves_to_passthrough(self):
        from torchbridge.models.llm.kv.quantized_cache import (
            QuantizedCacheConfig,
            QuantizedKVCache,
        )

        config = QuantizedCacheConfig()
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_explicit_supported_dtype_honoured(self):
        from torchbridge.models.llm.kv.quantized_cache import (
            QuantizedCacheConfig,
            QuantizedKVCache,
        )

        config = QuantizedCacheConfig(kv_cache_dtype=KVCacheDtype.PASSTHROUGH)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_unsupported_dtype_falls_back(self):
        """Requesting NVFP4 on CPU must fall back to PASSTHROUGH."""
        from torchbridge.models.llm.kv.quantized_cache import (
            QuantizedCacheConfig,
            QuantizedKVCache,
        )

        config = QuantizedCacheConfig(kv_cache_dtype=KVCacheDtype.NVFP4)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_unknown_backend_defaults_to_passthrough(self):
        from torchbridge.models.llm.kv.quantized_cache import (
            QuantizedCacheConfig,
            QuantizedKVCache,
        )

        config = QuantizedCacheConfig()
        cache = QuantizedKVCache(config, backend_name="unknown_hw")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH


# ---------------------------------------------------------------------------
# to_json() must be removed from inference dataclasses
# ---------------------------------------------------------------------------


class TestNoToJsonWrappers:
    def test_kv_handoff_spec_has_no_to_json(self):
        from torchbridge.inference.kv_handoff import KVHandoffSpec

        assert not hasattr(KVHandoffSpec, "to_json"), (
            "KVHandoffSpec.to_json() was a single json.dumps() wrapper — deleted in v0.5.77"
        )

    def test_disaggregated_fleet_config_has_no_to_json(self):
        from torchbridge.inference.disaggregated import DisaggregatedFleetConfig

        assert not hasattr(DisaggregatedFleetConfig, "to_json")

    def test_kv_handoff_spec_to_dict_is_json_serialisable(self):
        """to_dict() must still work for callers that do json.dumps(spec.to_dict())."""
        import json

        from torchbridge.inference.kv_handoff import KVHandoffSpec

        spec = KVHandoffSpec(
            dtype="float16", layout="separate", page_size_tokens=16, alignment_bytes=128
        )
        assert json.loads(json.dumps(spec.to_dict()))["dtype"] == "float16"
