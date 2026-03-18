"""
Integration tests for LLM KV-Cache backend compatibility pipeline.

Tests the end-to-end flow from hardware backend → compatibility matrix →
QuantizedKVCache dtype resolution across all supported backends.
"""

import pytest

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.models.llm.kv.cache_compatibility import KVCacheCompatibilityMatrix
from torchbridge.models.llm.kv.cache_dtype import KV_DTYPE_SPECS, KVCacheDtype
from torchbridge.models.llm.kv.quantized_cache import (
    QuantizedCacheConfig,
    QuantizedKVCache,
)


class TestCompatibilityMatrixAcrossBackends:
    """End-to-end: compatibility matrix returns valid dtypes for every backend."""

    @pytest.mark.parametrize("backend,arch", [
        (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
        (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
        (HardwareBackend.AMD, AMDArchitecture.CDNA3),
        (HardwareBackend.AMD, AMDArchitecture.CDNA2),
        (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
        (HardwareBackend.TPU, TPUVersion.V5E),
        (HardwareBackend.CPU, None),
    ])
    def test_get_supported_dtypes_non_empty(self, backend, arch):
        dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(backend, arch)
        assert len(dtypes) > 0, f"No dtypes for {backend.value}/{arch}"

    @pytest.mark.parametrize("backend,arch", [
        (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
        (HardwareBackend.AMD, AMDArchitecture.CDNA3),
        (HardwareBackend.CPU, None),
    ])
    def test_optimal_dtype_has_spec(self, backend, arch):
        """Optimal dtype from the matrix must have a KV_DTYPE_SPECS entry."""
        optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(backend, arch)
        assert optimal in KV_DTYPE_SPECS

    def test_all_matrix_dtypes_have_specs(self):
        """Every dtype in every table must map to KV_DTYPE_SPECS."""
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
                    f"{dtype.value} from {backend.value} missing from KV_DTYPE_SPECS"
                )


class TestQuantizedKVCacheResolutionPipeline:
    """Integration: backend string → HardwareBackend → matrix → resolved dtype."""

    @pytest.mark.parametrize("backend_name,expected_dtype", [
        ("cpu", KVCacheDtype.PASSTHROUGH),
        ("unknown_hw", KVCacheDtype.PASSTHROUGH),
    ])
    def test_cpu_and_fallback_resolve_to_passthrough(self, backend_name, expected_dtype):
        config = QuantizedCacheConfig()
        cache = QuantizedKVCache(config, backend_name=backend_name)
        assert cache.kv_dtype == expected_dtype

    def test_explicit_unsupported_dtype_falls_back(self):
        """Requesting NVFP4 on CPU must fall back via the matrix."""
        config = QuantizedCacheConfig(kv_cache_dtype=KVCacheDtype.NVFP4)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_dtype_spec_accessible_from_resolved_dtype(self):
        """Resolved dtype must be in KV_DTYPE_SPECS (memory factor accessible)."""
        config = QuantizedCacheConfig()
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype in KV_DTYPE_SPECS
        spec = KV_DTYPE_SPECS[cache.kv_dtype]
        assert spec.memory_factor > 0


class TestMemoryFactorOrdering:
    """Verify memory factor ordering matches quantization depth."""

    def test_fp8_lower_than_fp16(self):
        assert KV_DTYPE_SPECS[KVCacheDtype.FP8_E4M3].memory_factor < \
               KV_DTYPE_SPECS[KVCacheDtype.FP16].memory_factor

    def test_nvfp4_lower_than_fp8(self):
        assert KV_DTYPE_SPECS[KVCacheDtype.NVFP4].memory_factor < \
               KV_DTYPE_SPECS[KVCacheDtype.FP8_E4M3].memory_factor
