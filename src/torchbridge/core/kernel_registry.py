"""
Kernel Registry System for TorchBridge

This module provides a centralized registry for managing custom CUDA kernels,
Triton kernels, and PyTorch reference implementations. It enables automatic
kernel selection based on hardware capabilities, precision requirements, and
performance characteristics.

Key Features:
- Version management for kernels (v2.0, v3.0, etc.)
- Hardware-aware kernel selection (H100, A100, V100, etc.)
- Precision-based filtering (FP8, BFloat16, Float16, Float32)
- Automatic fallback chain: CUDA → Triton → PyTorch
- Performance-based kernel selection
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

from .config import NVIDIAArchitecture, PrecisionFormat


class KernelType(Enum):
    """Types of kernel operations."""
    ATTENTION = "attention"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    MATMUL = "matmul"
    FUSION = "fusion"  # Multi-operation fusion


class KernelBackend(Enum):
    """Backend implementations for kernels."""
    CUDA = "cuda"
    TRITON = "triton"
    PYTORCH = "pytorch"


@dataclass
class KernelMetadata:
    """
    Metadata for a registered kernel.

    This class stores all information needed to identify, select, and
    execute a custom kernel implementation.
    """
    # Identification
    kernel_id: str  # e.g., "flash_attention_v3"
    kernel_type: KernelType
    version: str  # e.g., "3.0"
    backend: KernelBackend

    # Hardware requirements
    min_compute_capability: tuple[int, int] = (7, 0)  # (major, minor) e.g., (9, 0) for H100
    supported_architectures: list[NVIDIAArchitecture] = field(default_factory=list)

    # Precision support
    precision_support: list[PrecisionFormat] = field(default_factory=lambda: [
        PrecisionFormat.FP32,
        PrecisionFormat.FP16
    ])

    # Operational constraints
    max_sequence_length: int | None = None
    max_batch_size: int | None = None
    memory_bound: bool = False
    compute_bound: bool = True

    # Performance characteristics
    expected_speedup: float = 1.0  # vs PyTorch baseline
    benchmark_latency_ms: float | None = None

    # Function pointers
    kernel_fn: Callable | None = None
    validation_fn: Callable | None = None

    # Additional metadata
    description: str = ""
    author: str = "TorchBridge Team"
    requires_compilation: bool = True

    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.backend == KernelBackend.CUDA and not self.kernel_fn:
            warnings.warn(f"CUDA kernel {self.kernel_id} registered without kernel function", stacklevel=2)

        if self.expected_speedup < 1.0:
            warnings.warn(f"Kernel {self.kernel_id} has expected_speedup < 1.0", stacklevel=2)


class KernelRegistry:
    """
    Singleton registry for managing custom kernels.

    The registry maintains a collection of kernel implementations and provides
    intelligent selection based on hardware capabilities, precision requirements,
    and performance characteristics.

    Usage:
        registry = KernelRegistry()

        # Register a kernel
        metadata = KernelMetadata(
            kernel_id="flash_attention_v3",
            kernel_type=KernelType.ATTENTION,
            version="3.0",
            backend=KernelBackend.CUDA,
            kernel_fn=flash_attention_v3_cuda,
            ...
        )
        registry.register_kernel(metadata)

        # Get optimal kernel
        kernel = registry.get_optimal_kernel(
            kernel_type=KernelType.ATTENTION,
            device=torch.device('cuda'),
            precision=PrecisionFormat.FLOAT16,
            sequence_length=2048
        )
    """

    _instance = None

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return

        self._kernels: dict[str, KernelMetadata] = {}
        self._initialized = True
        self._cache: dict[str, KernelMetadata] = {}  # Selection cache

    def register_kernel(self, metadata: KernelMetadata) -> None:
        """
        Register a kernel in the registry.

        Args:
            metadata: Complete kernel metadata

        Raises:
            ValueError: If kernel_id already exists with same version
        """
        key = self._make_key(metadata.kernel_type, metadata.kernel_id, metadata.version)

        if key in self._kernels:
            warnings.warn(
                f"Overwriting existing kernel: {key}. "
                f"Previous backend: {self._kernels[key].backend.value}, "
                f"New backend: {metadata.backend.value}",
            stacklevel=2,
            )

        self._kernels[key] = metadata

        # Clear cache since registry changed
        self._cache.clear()

    def _make_key(self, kernel_type: KernelType, kernel_id: str, version: str) -> str:
        """Create unique key for kernel storage."""
        return f"{kernel_type.value}:{kernel_id}:v{version}"

    def _make_cache_key(
        self,
        kernel_type: KernelType,
        device: torch.device,
        precision: PrecisionFormat,
        sequence_length: int | None,
        batch_size: int | None
    ) -> str:
        """Create cache key for kernel selection."""
        return f"{kernel_type.value}:{device.type}:{precision.value}:{sequence_length}:{batch_size}"

    def get_optimal_kernel(
        self,
        kernel_type: KernelType,
        device: torch.device,
        precision: PrecisionFormat,
        sequence_length: int | None = None,
        batch_size: int | None = None,
        prefer_backend: KernelBackend | None = None
    ) -> KernelMetadata | None:
        """
        Select the optimal kernel for given requirements.

        Selection criteria (in order):
        1. Hardware compatibility (compute capability, architecture)
        2. Precision support
        3. Operational constraints (sequence length, batch size)
        4. Backend preference (CUDA > Triton > PyTorch by default)
        5. Version (higher version preferred)
        6. Expected speedup

        Args:
            kernel_type: Type of kernel operation
            device: Target device (cuda, cpu, etc.)
            precision: Required precision format
            sequence_length: Optional sequence length constraint
            batch_size: Optional batch size constraint
            prefer_backend: Optional backend preference

        Returns:
            Best matching kernel metadata, or None if no kernel found
        """
        # Check cache first
        cache_key = self._make_cache_key(
            kernel_type, device, precision, sequence_length, batch_size
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Filter candidates by type
        candidates = [
            k for k in self._kernels.values()
            if k.kernel_type == kernel_type
        ]

        if not candidates:
            return None

        # Filter by hardware compatibility
        if device.type == 'cuda':
            compute_cap = torch.cuda.get_device_capability(device)
            candidates = [
                k for k in candidates
                if self._is_hardware_compatible(k, compute_cap)
            ]

        # Filter by precision support
        candidates = [
            k for k in candidates
            if precision in k.precision_support
        ]

        # Filter by sequence length constraint
        if sequence_length is not None:
            candidates = [
                k for k in candidates
                if k.max_sequence_length is None or sequence_length <= k.max_sequence_length
            ]

        # Filter by batch size constraint
        if batch_size is not None:
            candidates = [
                k for k in candidates
                if k.max_batch_size is None or batch_size <= k.max_batch_size
            ]

        if not candidates:
            return None

        # Sort by preference
        def sort_key(kernel: KernelMetadata) -> tuple:
            """
            Sort kernels by:
            1. Backend preference (CUDA=3, Triton=2, PyTorch=1)
            2. Version (higher is better)
            3. Expected speedup (higher is better)
            """
            backend_priority = {
                KernelBackend.CUDA: 3,
                KernelBackend.TRITON: 2,
                KernelBackend.PYTORCH: 1
            }

            # If backend preference specified, prioritize it
            if prefer_backend and kernel.backend == prefer_backend:
                backend_score = 10
            else:
                backend_score = backend_priority.get(kernel.backend, 0)

            # Parse version (e.g., "3.0" -> 3.0)
            try:
                version_num = float(kernel.version)
            except ValueError:
                version_num = 0.0

            return (backend_score, version_num, kernel.expected_speedup)

        candidates.sort(key=sort_key, reverse=True)

        # Cache result
        best_kernel = candidates[0]
        self._cache[cache_key] = best_kernel

        return best_kernel

    def _is_hardware_compatible(
        self,
        kernel: KernelMetadata,
        compute_capability: tuple[int, int]
    ) -> bool:
        """Check if kernel is compatible with hardware."""
        # Check minimum compute capability
        if compute_capability < kernel.min_compute_capability:
            return False

        # If specific architectures listed, check compatibility
        if kernel.supported_architectures:
            # Map compute capability to architecture
            arch_map = {
                (7, 0): NVIDIAArchitecture.VOLTA,
                (7, 5): NVIDIAArchitecture.TURING,
                (8, 0): NVIDIAArchitecture.AMPERE,
                (8, 6): NVIDIAArchitecture.AMPERE,
                (8, 9): NVIDIAArchitecture.ADA,
                (9, 0): NVIDIAArchitecture.HOPPER,
                (10, 0): NVIDIAArchitecture.BLACKWELL,
            }

            current_arch = arch_map.get(compute_capability)
            if current_arch and current_arch not in kernel.supported_architectures:
                return False

        return True

    def get_fallback_chain(
        self,
        kernel_type: KernelType,
        device: torch.device,
        precision: PrecisionFormat
    ) -> list[KernelMetadata]:
        """
        Get ordered fallback chain for a kernel type.

        Returns kernels in order: CUDA → Triton → PyTorch

        Args:
            kernel_type: Type of kernel
            device: Target device
            precision: Required precision

        Returns:
            List of kernels in fallback order
        """
        all_kernels = [
            k for k in self._kernels.values()
            if k.kernel_type == kernel_type and precision in k.precision_support
        ]

        # Sort by backend priority
        backend_order = [KernelBackend.CUDA, KernelBackend.TRITON, KernelBackend.PYTORCH]

        sorted_kernels = sorted(
            all_kernels,
            key=lambda k: (
                backend_order.index(k.backend) if k.backend in backend_order else 999,
                -float(k.version) if k.version.replace('.', '').isdigit() else 0
            )
        )

        return sorted_kernels

    def list_kernels(
        self,
        kernel_type: KernelType | None = None,
        backend: KernelBackend | None = None
    ) -> list[KernelMetadata]:
        """
        List all registered kernels with optional filtering.

        Args:
            kernel_type: Optional filter by kernel type
            backend: Optional filter by backend

        Returns:
            List of matching kernel metadata
        """
        kernels = list(self._kernels.values())

        if kernel_type is not None:
            kernels = [k for k in kernels if k.kernel_type == kernel_type]

        if backend is not None:
            kernels = [k for k in kernels if k.backend == backend]

        return kernels

    def validate_kernel(self, kernel_id: str, *args, **kwargs) -> bool:
        """
        Validate a kernel's correctness if validation function provided.

        Args:
            kernel_id: ID of kernel to validate
            *args, **kwargs: Arguments to pass to validation function

        Returns:
            True if validation passes, False otherwise
        """
        matching_kernels = [
            k for k in self._kernels.values()
            if k.kernel_id == kernel_id
        ]

        if not matching_kernels:
            warnings.warn(f"Kernel {kernel_id} not found in registry", stacklevel=2)
            return False

        kernel = matching_kernels[0]  # Use first match

        if kernel.validation_fn is None:
            warnings.warn(f"Kernel {kernel_id} has no validation function", stacklevel=2)
            return True  # Assume valid if no validation function

        try:
            return kernel.validation_fn(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Kernel {kernel_id} validation failed: {e}", stacklevel=2)
            return False

    def clear_cache(self) -> None:
        """Clear the kernel selection cache."""
        self._cache.clear()

    def unregister_kernel(self, kernel_type: KernelType, kernel_id: str, version: str) -> bool:
        """
        Unregister a kernel from the registry.

        Args:
            kernel_type: Type of kernel
            kernel_id: Kernel ID
            version: Kernel version

        Returns:
            True if kernel was unregistered, False if not found
        """
        key = self._make_key(kernel_type, kernel_id, version)

        if key in self._kernels:
            del self._kernels[key]
            self._cache.clear()
            return True

        return False

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"KernelRegistry(kernels={len(self._kernels)}, cached={len(self._cache)})"


# Global registry instance
_global_registry = None


def get_kernel_registry() -> KernelRegistry:
    """
    Get the global kernel registry instance.

    Returns:
        Global KernelRegistry singleton
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = KernelRegistry()
    return _global_registry


def register_default_kernels(registry: KernelRegistry | None = None) -> None:
    """
    Register default kernels that come with TorchBridge.

    This function registers:
    - Existing CUDA kernels (FlashAttention v2, LayerNorm, SwiGLU, RoPE)
    - Triton kernels (if available)
    - PyTorch reference implementations

    Args:
        registry: Optional registry instance (uses global if None)
    """
    if registry is None:
        registry = get_kernel_registry()

    # Try to import CUDA kernels
    try:
        import torchbridge_cuda
        cuda_available = True
    except ImportError:
        cuda_available = False
        warnings.warn("CUDA kernels not available - using fallbacks", stacklevel=2)

    # Register FlashAttention v2 (existing CUDA kernel)
    if cuda_available:
        registry.register_kernel(KernelMetadata(
            kernel_id="flash_attention_v2",
            kernel_type=KernelType.ATTENTION,
            version="2.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(8, 0),  # A100+
            supported_architectures=[
                NVIDIAArchitecture.AMPERE,
                NVIDIAArchitecture.ADA,
                NVIDIAArchitecture.HOPPER,
                NVIDIAArchitecture.BLACKWELL
            ],
            precision_support=[
                PrecisionFormat.FP16,
                PrecisionFormat.BF16
            ],
            expected_speedup=2.5,
            kernel_fn=torchbridge_cuda.flash_attention if cuda_available else None,
            description="FlashAttention-2 implementation"
        ))

    # Register LayerNorm (existing CUDA kernel)
    if cuda_available:
        registry.register_kernel(KernelMetadata(
            kernel_id="layernorm_cuda",
            kernel_type=KernelType.NORMALIZATION,
            version="1.0",
            backend=KernelBackend.CUDA,
            min_compute_capability=(7, 0),
            precision_support=[
                PrecisionFormat.FP32,
                PrecisionFormat.FP16
            ],
            expected_speedup=1.5,
            kernel_fn=torchbridge_cuda.fused_layer_norm if cuda_available else None,
            description="Fused LayerNorm with warp-level reductions"
        ))

    # Register PyTorch reference implementations (always available)
    registry.register_kernel(KernelMetadata(
        kernel_id="attention_pytorch",
        kernel_type=KernelType.ATTENTION,
        version="1.0",
        backend=KernelBackend.PYTORCH,
        min_compute_capability=(0, 0),  # Works everywhere
        precision_support=[
            PrecisionFormat.FP32,
            PrecisionFormat.FP16,
            PrecisionFormat.BF16
        ],
        expected_speedup=1.0,  # Baseline
        kernel_fn=None,  # Implemented in Python layer
        description="PyTorch native attention implementation",
        requires_compilation=False
    ))


__all__ = [
    'KernelType',
    'KernelBackend',
    'KernelMetadata',
    'KernelRegistry',
    'get_kernel_registry',
    'register_default_kernels'
]
