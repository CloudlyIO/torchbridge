"""Tests for collective backend abstraction."""


from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
)
from torchbridge.distributed.collective_backend import (
    COLLECTIVE_BACKEND_SPECS,
    CollectiveBackendMatrix,
    CollectiveBackendType,
    CollectiveConfig,
)


class TestCollectiveBackendType:
    """Tests for CollectiveBackendType enum."""

    def test_all_types(self):
        expected = {"nccl", "rccl", "neuron_cc", "xla", "gloo"}
        actual = {t.value for t in CollectiveBackendType}
        assert actual == expected

    def test_specs_cover_all_types(self):
        for btype in CollectiveBackendType:
            assert btype in COLLECTIVE_BACKEND_SPECS


class TestCollectiveBackendSpec:
    """Tests for CollectiveBackendSpec."""

    def test_nccl_features(self):
        spec = COLLECTIVE_BACKEND_SPECS[CollectiveBackendType.NCCL]
        assert spec.supports_gpu_direct is True
        assert spec.supports_fp8_reduce is True
        assert spec.supports_symmetric_memory is True

    def test_rccl_features(self):
        spec = COLLECTIVE_BACKEND_SPECS[CollectiveBackendType.RCCL]
        assert spec.supports_gpu_direct is True
        assert spec.supports_fp8_reduce is False

    def test_gloo_features(self):
        spec = COLLECTIVE_BACKEND_SPECS[CollectiveBackendType.GLOO]
        assert spec.supports_gpu_direct is False
        assert spec.supports_fp8_reduce is False
        assert spec.supports_symmetric_memory is False


class TestCollectiveBackendMatrix:
    """Tests for CollectiveBackendMatrix."""

    def test_cuda_uses_nccl(self):
        result = CollectiveBackendMatrix.get_optimal_backend(HardwareBackend.CUDA)
        assert result == CollectiveBackendType.NCCL

    def test_amd_uses_rccl(self):
        result = CollectiveBackendMatrix.get_optimal_backend(HardwareBackend.AMD)
        assert result == CollectiveBackendType.RCCL

    def test_trainium_uses_neuron_cc(self):
        result = CollectiveBackendMatrix.get_optimal_backend(HardwareBackend.TRAINIUM)
        assert result == CollectiveBackendType.NEURON_CC

    def test_tpu_uses_xla(self):
        result = CollectiveBackendMatrix.get_optimal_backend(HardwareBackend.TPU)
        assert result == CollectiveBackendType.XLA_COLLECTIVES

    def test_cpu_uses_gloo(self):
        result = CollectiveBackendMatrix.get_optimal_backend(HardwareBackend.CPU)
        assert result == CollectiveBackendType.GLOO

    def test_symmetric_memory_hopper(self):
        assert CollectiveBackendMatrix.supports_symmetric_memory(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        ) is True

    def test_symmetric_memory_blackwell(self):
        assert CollectiveBackendMatrix.supports_symmetric_memory(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        ) is True

    def test_no_symmetric_memory_ampere(self):
        assert CollectiveBackendMatrix.supports_symmetric_memory(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        ) is False

    def test_no_symmetric_memory_amd(self):
        assert CollectiveBackendMatrix.supports_symmetric_memory(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        ) is False

    def test_fp8_reduce_hopper(self):
        assert CollectiveBackendMatrix.supports_fp8_reduce(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        ) is True

    def test_fp8_reduce_ada(self):
        assert CollectiveBackendMatrix.supports_fp8_reduce(
            HardwareBackend.CUDA, NVIDIAArchitecture.ADA
        ) is True

    def test_no_fp8_reduce_ampere(self):
        assert CollectiveBackendMatrix.supports_fp8_reduce(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        ) is False

    def test_no_fp8_reduce_amd(self):
        assert CollectiveBackendMatrix.supports_fp8_reduce(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        ) is False

    def test_torch_backend_string_cuda(self):
        result = CollectiveBackendMatrix.get_torch_backend_string(HardwareBackend.CUDA)
        assert result == "nccl"

    def test_torch_backend_string_amd(self):
        result = CollectiveBackendMatrix.get_torch_backend_string(HardwareBackend.AMD)
        assert result == "nccl"  # RCCL uses "nccl" in PyTorch

    def test_torch_backend_string_cpu(self):
        result = CollectiveBackendMatrix.get_torch_backend_string(HardwareBackend.CPU)
        assert result == "gloo"

    def test_torch_backend_string_tpu(self):
        result = CollectiveBackendMatrix.get_torch_backend_string(HardwareBackend.TPU)
        assert result == "gloo"  # XLA doesn't map to a torch string, fallback

    def test_get_all_backends(self):
        backends = CollectiveBackendMatrix.get_all_backends()
        assert len(backends) == 5
        assert all("backend" in b for b in backends)
        assert all("display_name" in b for b in backends)


class TestCollectiveConfig:
    """Tests for CollectiveConfig dataclass."""

    def test_defaults(self):
        config = CollectiveConfig()
        assert config.backend is None
        assert config.symmetric_memory is False
        assert config.float8_reduce is False

    def test_resolve_cuda_hopper(self):
        config = CollectiveConfig()
        resolved = config.resolve(HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER)
        assert resolved.backend == CollectiveBackendType.NCCL
        assert resolved.symmetric_memory is True
        assert resolved.float8_reduce is True

    def test_resolve_amd(self):
        config = CollectiveConfig()
        resolved = config.resolve(HardwareBackend.AMD, AMDArchitecture.CDNA3)
        assert resolved.backend == CollectiveBackendType.RCCL
        assert resolved.symmetric_memory is False
        assert resolved.float8_reduce is False

    def test_resolve_cpu(self):
        config = CollectiveConfig()
        resolved = config.resolve(HardwareBackend.CPU)
        assert resolved.backend == CollectiveBackendType.GLOO

    def test_explicit_backend_preserved(self):
        config = CollectiveConfig(backend=CollectiveBackendType.GLOO)
        resolved = config.resolve(HardwareBackend.CUDA)
        assert resolved.backend == CollectiveBackendType.GLOO

    def test_to_dict(self):
        config = CollectiveConfig(
            backend=CollectiveBackendType.NCCL,
            symmetric_memory=True,
            float8_reduce=True,
        )
        d = config.to_dict()
        assert d["backend"] == "nccl"
        assert d["symmetric_memory"] is True
        assert d["float8_reduce"] is True

    def test_to_dict_auto_backend(self):
        config = CollectiveConfig()
        d = config.to_dict()
        assert d["backend"] == "auto"
