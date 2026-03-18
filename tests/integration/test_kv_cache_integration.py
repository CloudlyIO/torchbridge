"""
Integration Tests for KV-Cache Pipeline

End-to-end tests for KV-cache compatibility matrix lookups and
dtype resolution via the QuantizedKVCache config selector.
"""

from torchbridge.models.llm.kv.cache_compatibility import KVCacheCompatibilityMatrix
from torchbridge.models.llm.kv.cache_dtype import KV_DTYPE_SPECS, KVCacheDtype
from torchbridge.models.llm.kv.quantized_cache import (
    QuantizedCacheConfig,
    QuantizedKVCache,
)


class TestKVCacheCompatibilityPipeline:
    """Integration tests for the compatibility matrix → dtype resolution pipeline."""

    def test_compatibility_to_quantized_cache_dtype(self):
        """Matrix optimal dtype flows correctly through QuantizedKVCache resolver."""
        from torchbridge.core.config import HardwareBackend

        optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(HardwareBackend.CPU)
        config = QuantizedCacheConfig(kv_cache_dtype=optimal)
        qcache = QuantizedKVCache(config, backend_name="cpu")
        assert qcache.kv_dtype == optimal

    def test_all_kv_dtypes_have_consistent_specs(self):
        """Every dtype in every compatibility table should have a KV_DTYPE_SPECS entry."""
        from torchbridge.core.config import (
            AMDArchitecture,
            HardwareBackend,
            NVIDIAArchitecture,
            TPUVersion,
            TrainiumArchitecture,
        )

        backends = [
            (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
            (HardwareBackend.TPU, TPUVersion.V5E),
            (HardwareBackend.CPU, None),
        ]
        for backend, arch in backends:
            for dtype in KVCacheCompatibilityMatrix.get_supported_dtypes(backend, arch):
                assert dtype in KV_DTYPE_SPECS, (
                    f"dtype {dtype.value} from {backend.value} not in KV_DTYPE_SPECS"
                )

    def test_memory_factor_reduces_with_quantization(self):
        """FP8 should have lower memory factor than FP16/BF16."""
        fp16_spec = KV_DTYPE_SPECS[KVCacheDtype.FP16]
        fp8_spec = KV_DTYPE_SPECS[KVCacheDtype.FP8_E4M3]
        nvfp4_spec = KV_DTYPE_SPECS[KVCacheDtype.NVFP4]
        assert fp8_spec.memory_factor < fp16_spec.memory_factor
        assert nvfp4_spec.memory_factor < fp8_spec.memory_factor

    def test_fallback_chain_resolves_valid_dtype(self):
        """get_fallback_chain must return a non-empty list with all valid dtypes."""
        from torchbridge.core.config import HardwareBackend

        chain = KVCacheCompatibilityMatrix.get_fallback_chain(
            KVCacheDtype.FP8_E4M3, HardwareBackend.CPU
        )
        assert isinstance(chain, list)
        assert len(chain) > 0
        for dtype in chain:
            assert isinstance(dtype, KVCacheDtype)
