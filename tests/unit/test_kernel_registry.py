"""
Tests for Kernel Registry System

This module tests the kernel registry functionality including:
- Kernel registration and retrieval
- Hardware-aware kernel selection
- Precision filtering
- Fallback chain management
- Version management
"""

from unittest.mock import Mock

import pytest
import torch

from torchbridge.core.config import PrecisionFormat
from torchbridge.core.kernel_registry import (
    KernelBackend,
    KernelMetadata,
    KernelRegistry,
    KernelType,
    get_kernel_registry,
    register_default_kernels,
)


class TestKernelMetadata:
    """Tests for KernelMetadata dataclass."""

    def test_metadata_creation(self):
        """Test basic kernel metadata creation."""
        metadata = KernelMetadata(
            kernel_id="test_kernel",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            kernel_fn=lambda x: x
        )

        assert metadata.kernel_id == "test_kernel"
        assert metadata.kernel_type == KernelType.ATTENTION
        assert metadata.version == "1.0"
        assert metadata.backend == KernelBackend.CUDA
        assert metadata.kernel_fn is not None

    def test_metadata_defaults(self):
        """Test default values in metadata."""
        metadata = KernelMetadata(
            kernel_id="test",
            kernel_type=KernelType.ACTIVATION,
            version="1.0",
            backend=KernelBackend.PYTORCH
        )

        assert metadata.min_compute_capability == (7, 0)
        assert metadata.expected_speedup == 1.0
        assert metadata.requires_compilation is True
        assert PrecisionFormat.FP32 in metadata.precision_support
        assert PrecisionFormat.FP16 in metadata.precision_support


class TestKernelRegistry:
    """Tests for KernelRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        reg = KernelRegistry()
        reg._kernels.clear()
        reg._cache.clear()
        return reg

    @pytest.fixture
    def sample_kernel(self):
        """Create a sample kernel metadata."""
        return KernelMetadata(
            kernel_id="flash_attention_v2",
            kernel_type=KernelType.ATTENTION,
            version="2.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(8, 0),
            precision_support=[PrecisionFormat.FP16, PrecisionFormat.BF16],
            expected_speedup=2.5,
            kernel_fn=Mock()
        )

    def test_singleton_pattern(self):
        """Test that KernelRegistry is a singleton."""
        reg1 = KernelRegistry()
        reg2 = KernelRegistry()
        assert reg1 is reg2

    def test_register_kernel(self, registry, sample_kernel):
        """Test kernel registration."""
        registry.register_kernel(sample_kernel)

        kernels = registry.list_kernels(kernel_type=KernelType.ATTENTION)
        assert len(kernels) == 1
        assert kernels[0].kernel_id == "flash_attention_v2"

    def test_register_duplicate_kernel_warning(self, registry, sample_kernel):
        """Test that registering duplicate kernel issues warning."""
        registry.register_kernel(sample_kernel)

        # Register again - should warn but succeed
        with pytest.warns(UserWarning, match="Overwriting existing kernel"):
            registry.register_kernel(sample_kernel)

    def test_list_kernels_by_type(self, registry):
        """Test listing kernels filtered by type."""
        # Register different kernel types
        registry.register_kernel(KernelMetadata(
            kernel_id="attention_1",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="norm_1",
            kernel_type=KernelType.NORMALIZATION,
            version="1.0",
            backend=KernelBackend.CUDA
        ))

        attention_kernels = registry.list_kernels(kernel_type=KernelType.ATTENTION)
        norm_kernels = registry.list_kernels(kernel_type=KernelType.NORMALIZATION)

        assert len(attention_kernels) == 1
        assert len(norm_kernels) == 1
        assert attention_kernels[0].kernel_id == "attention_1"
        assert norm_kernels[0].kernel_id == "norm_1"

    def test_list_kernels_by_backend(self, registry):
        """Test listing kernels filtered by backend."""
        registry.register_kernel(KernelMetadata(
            kernel_id="cuda_kernel",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="triton_kernel",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.TRITON
        ))

        cuda_kernels = registry.list_kernels(backend=KernelBackend.CUDA)
        triton_kernels = registry.list_kernels(backend=KernelBackend.TRITON)

        assert len(cuda_kernels) == 1
        assert len(triton_kernels) == 1
        assert cuda_kernels[0].backend == KernelBackend.CUDA
        assert triton_kernels[0].backend == KernelBackend.TRITON

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimal_kernel_selection_cuda_preferred(self, registry):
        """Test that CUDA kernel is preferred over Triton/PyTorch."""
        # Register kernels with different backends
        registry.register_kernel(KernelMetadata(
            kernel_id="attention_pytorch",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.PYTORCH,
            precision_support=[PrecisionFormat.FP16]
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="attention_triton",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.TRITON,
            precision_support=[PrecisionFormat.FP16]
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="attention_cuda",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),  # Low requirement
            precision_support=[PrecisionFormat.FP16]
        ))

        kernel = registry.get_optimal_kernel(
            kernel_type=KernelType.ATTENTION,
            device=torch.device('cuda'),
            precision=PrecisionFormat.FP16
        )

        assert kernel is not None
        assert kernel.backend == KernelBackend.CUDA

    def test_optimal_kernel_version_preference(self, registry):
        """Test that higher version is preferred."""
        registry.register_kernel(KernelMetadata(
            kernel_id="attention",
            kernel_type=KernelType.ATTENTION,
            version="2.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            precision_support=[PrecisionFormat.FP16]
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="attention",
            kernel_type=KernelType.ATTENTION,
            version="3.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            precision_support=[PrecisionFormat.FP16]
        ))

        if torch.cuda.is_available():
            kernel = registry.get_optimal_kernel(
                kernel_type=KernelType.ATTENTION,
                device=torch.device('cuda'),
                precision=PrecisionFormat.FP16
            )

            assert kernel is not None
            assert kernel.version == "3.0"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimal_kernel_compute_capability_filtering(self, registry):
        """Test that kernels are filtered by compute capability."""
        # Get actual compute capability
        compute_cap = torch.cuda.get_device_capability()

        # Register kernel requiring higher capability than available
        registry.register_kernel(KernelMetadata(
            kernel_id="h100_only",
            kernel_type=KernelType.ATTENTION,
            version="3.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(9, 0),  # H100
            precision_support=[PrecisionFormat.FP16]
        ))

        # Register kernel with lower requirement
        registry.register_kernel(KernelMetadata(
            kernel_id="general",
            kernel_type=KernelType.ATTENTION,
            version="2.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),  # V100+
            precision_support=[PrecisionFormat.FP16]
        ))

        kernel = registry.get_optimal_kernel(
            kernel_type=KernelType.ATTENTION,
            device=torch.device('cuda'),
            precision=PrecisionFormat.FP16
        )

        # If we have H100 (9.0+), we should get v3.0
        # Otherwise, we should get v2.0
        assert kernel is not None
        if compute_cap >= (9, 0):
            assert kernel.version == "3.0"
        else:
            assert kernel.version == "2.0"

    def test_optimal_kernel_precision_filtering(self, registry):
        """Test that kernels are filtered by precision support."""
        registry.register_kernel(KernelMetadata(
            kernel_id="fp16_only",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            precision_support=[PrecisionFormat.FP16]
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="fp32_only",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            precision_support=[PrecisionFormat.FP32]
        ))

        if torch.cuda.is_available():
            # Request FP16 - should get fp16_only
            kernel_fp16 = registry.get_optimal_kernel(
                kernel_type=KernelType.ATTENTION,
                device=torch.device('cuda'),
                precision=PrecisionFormat.FP16
            )
            assert kernel_fp16 is not None
            assert kernel_fp16.kernel_id == "fp16_only"

            # Request FP32 - should get fp32_only
            kernel_fp32 = registry.get_optimal_kernel(
                kernel_type=KernelType.ATTENTION,
                device=torch.device('cuda'),
                precision=PrecisionFormat.FP32
            )
            assert kernel_fp32 is not None
            assert kernel_fp32.kernel_id == "fp32_only"

    def test_optimal_kernel_sequence_length_constraint(self, registry):
        """Test sequence length constraint filtering."""
        registry.register_kernel(KernelMetadata(
            kernel_id="short_seq",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            max_sequence_length=1024,
            precision_support=[PrecisionFormat.FP16]
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="long_seq",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            max_sequence_length=None,  # No limit
            precision_support=[PrecisionFormat.FP16]
        ))

        if torch.cuda.is_available():
            # Request short sequence - could get either
            kernel_short = registry.get_optimal_kernel(
                kernel_type=KernelType.ATTENTION,
                device=torch.device('cuda'),
                precision=PrecisionFormat.FP16,
                sequence_length=512
            )
            assert kernel_short is not None

            # Request long sequence - should only get long_seq
            kernel_long = registry.get_optimal_kernel(
                kernel_type=KernelType.ATTENTION,
                device=torch.device('cuda'),
                precision=PrecisionFormat.FP16,
                sequence_length=2048
            )
            assert kernel_long is not None
            assert kernel_long.kernel_id == "long_seq"

    def test_fallback_chain(self, registry):
        """Test fallback chain ordering."""
        registry.register_kernel(KernelMetadata(
            kernel_id="pytorch_impl",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.PYTORCH,
            precision_support=[PrecisionFormat.FP16]
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="triton_impl",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.TRITON,
            precision_support=[PrecisionFormat.FP16]
        ))
        registry.register_kernel(KernelMetadata(
            kernel_id="cuda_impl",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            precision_support=[PrecisionFormat.FP16]
        ))

        fallback_chain = registry.get_fallback_chain(
            kernel_type=KernelType.ATTENTION,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            precision=PrecisionFormat.FP16
        )

        assert len(fallback_chain) == 3
        # Should be ordered: CUDA → Triton → PyTorch
        assert fallback_chain[0].backend == KernelBackend.CUDA
        assert fallback_chain[1].backend == KernelBackend.TRITON
        assert fallback_chain[2].backend == KernelBackend.PYTORCH

    def test_kernel_caching(self, registry):
        """Test that kernel selection is cached."""
        registry.register_kernel(KernelMetadata(
            kernel_id="test_kernel",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            precision_support=[PrecisionFormat.FP16]
        ))

        if torch.cuda.is_available():
            # First call
            kernel1 = registry.get_optimal_kernel(
                kernel_type=KernelType.ATTENTION,
                device=torch.device('cuda'),
                precision=PrecisionFormat.FP16
            )

            # Cache should have 1 entry
            assert len(registry._cache) == 1

            # Second call with same parameters
            kernel2 = registry.get_optimal_kernel(
                kernel_type=KernelType.ATTENTION,
                device=torch.device('cuda'),
                precision=PrecisionFormat.FP16
            )

            # Should return same cached kernel
            assert kernel1 is kernel2

    def test_clear_cache(self, registry):
        """Test cache clearing."""
        registry.register_kernel(KernelMetadata(
            kernel_id="test",
            kernel_type=KernelType.ATTENTION,
            version="1.0",
            backend=KernelBackend.PYTORCH,
            precision_support=[PrecisionFormat.FP16]
        ))

        # Build cache
        registry.get_optimal_kernel(
            kernel_type=KernelType.ATTENTION,
            device=torch.device('cpu'),
            precision=PrecisionFormat.FP16
        )

        assert len(registry._cache) > 0

        # Clear cache
        registry.clear_cache()
        assert len(registry._cache) == 0

    def test_unregister_kernel(self, registry, sample_kernel):
        """Test kernel unregistration."""
        registry.register_kernel(sample_kernel)
        assert len(registry.list_kernels()) == 1

        success = registry.unregister_kernel(
            kernel_type=KernelType.ATTENTION,
            kernel_id="flash_attention_v2",
            version="2.0"
        )

        assert success is True
        assert len(registry.list_kernels()) == 0

    def test_unregister_nonexistent_kernel(self, registry):
        """Test unregistering kernel that doesn't exist."""
        success = registry.unregister_kernel(
            kernel_type=KernelType.ATTENTION,
            kernel_id="nonexistent",
            version="1.0"
        )

        assert success is False

    def test_no_matching_kernel_returns_none(self, registry):
        """Test that get_optimal_kernel returns None if no match."""
        kernel = registry.get_optimal_kernel(
            kernel_type=KernelType.ATTENTION,
            device=torch.device('cpu'),
            precision=PrecisionFormat.FP16
        )

        assert kernel is None


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_kernel_registry_singleton(self):
        """Test global registry is singleton."""
        reg1 = get_kernel_registry()
        reg2 = get_kernel_registry()
        assert reg1 is reg2

    def test_register_default_kernels(self):
        """Test registering default kernels."""
        registry = KernelRegistry()
        registry._kernels.clear()

        register_default_kernels(registry)

        # Should have registered at least PyTorch reference implementations
        kernels = registry.list_kernels()
        assert len(kernels) > 0

        # Should have PyTorch attention
        pytorch_kernels = registry.list_kernels(backend=KernelBackend.PYTORCH)
        assert len(pytorch_kernels) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
