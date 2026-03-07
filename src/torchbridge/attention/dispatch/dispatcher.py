"""
Attention Kernel Dispatcher

Compatibility matrix that selects the best available attention kernel for the
given hardware backend. PyTorch SDPA handles runtime dispatch; TorchBridge
selects which backend context (FlashAttention, SDPA, math fallback) to use
based on (backend, dtype, sequence_length) via the compatibility matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    HardwareConfig,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)

from .benchmark_cache import KernelBenchmarkCache
from .compatibility import AttentionDispatchMatrix
from .kernel_types import AttentionKernelType

logger = logging.getLogger(__name__)

# Mapping from AttentionKernelType → attention registry name
_KERNEL_REGISTRY_MAP: dict[AttentionKernelType, str] = {
    AttentionKernelType.FLEX_ATTENTION: "flex_attention",
    AttentionKernelType.FLASH_ATTENTION_3: "flash_attention3",
    AttentionKernelType.FLASH_ATTENTION_2: "flash_attention2",
    AttentionKernelType.FLASH_ATTENTION_CK: "flash_attention2",  # CK uses same interface
    AttentionKernelType.NEURONX_SDPA: "memory_efficient_attention",
    AttentionKernelType.PALLAS_ATTENTION: "memory_efficient_attention",
    AttentionKernelType.PYTORCH_SDPA: "memory_efficient_attention",
}


@dataclass
class AttentionDispatchResult:
    """Result of kernel dispatch."""

    kernel_type: AttentionKernelType
    implementation_name: str
    used_fallback: bool = False
    fallback_chain: list[AttentionKernelType] = field(default_factory=list)
    benchmark_latency_ms: float | None = None
    warnings: list[str] = field(default_factory=list)


class AttentionDispatcher:
    """
    Backend-aware attention kernel dispatcher.

    Detects hardware via HardwareConfig, consults the compatibility matrix
    to find the optimal kernel, verifies runtime availability, and falls
    back along the chain until a working kernel is found.
    """

    def __init__(
        self,
        backend: HardwareBackend | None = None,
        architecture: NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None = None,
        use_benchmark_cache: bool = True,
    ) -> None:
        self._hw = HardwareConfig()
        self._backend = backend or self._hw.backend
        self._architecture = architecture or self._detect_architecture()
        self._cache = KernelBenchmarkCache() if use_benchmark_cache else None

    # ── public API ───────────────────────────────────────────────────

    def select_kernel(
        self,
        seq_length: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
    ) -> AttentionDispatchResult:
        """Select the best available attention kernel for current hardware."""
        supported = AttentionDispatchMatrix.get_supported_kernels(
            self._backend, self._architecture
        )

        chosen: AttentionKernelType | None = None
        used_fallback = False
        result_warnings: list[str] = []

        for i, kernel in enumerate(supported):
            if self._check_kernel_availability(kernel):
                chosen = kernel
                used_fallback = i > 0
                break
            result_warnings.append(
                f"{kernel.value} not available at runtime, trying next"
            )

        if chosen is None:
            chosen = AttentionKernelType.PYTORCH_SDPA
            used_fallback = True
            result_warnings.append(
                "No preferred kernel available — falling back to PyTorch SDPA"
            )

        # Benchmark lookup — lazy-warm cache on first miss (B4 fix)
        # Skip benchmarking for extreme dimensions to avoid OOM/hangs
        _MAX_BENCH_SEQ = 32_768
        latency: float | None = None
        if self._cache is not None:
            latency = self._cache.get_cached_latency(
                chosen, seq_length, num_heads, head_dim
            )
            if latency is None and seq_length <= _MAX_BENCH_SEQ:
                try:
                    entry = self._cache.run_benchmark(
                        chosen, seq_length, num_heads, head_dim,
                        warmup=1, iterations=5,
                    )
                    latency = entry.latency_ms
                    logger.debug(
                        "Lazy benchmark: %s at %s×%s×%s → %.2f ms",
                        chosen.value, seq_length, num_heads, head_dim, latency,
                    )
                except Exception as _e:
                    logger.debug("Lazy benchmark failed (non-fatal): %s", _e)

        fallback_chain = AttentionDispatchMatrix.get_fallback_chain(
            chosen, self._backend, self._architecture
        )

        return AttentionDispatchResult(
            kernel_type=chosen,
            implementation_name=self._kernel_to_registry_name(chosen),
            used_fallback=used_fallback,
            fallback_chain=fallback_chain,
            benchmark_latency_ms=latency,
            warnings=result_warnings,
        )

    def create_attention(self, config, **kwargs):
        """End-to-end: dispatch kernel then create attention layer.

        Args:
            config: AttentionModuleConfig instance.
            **kwargs: Extra arguments forwarded to the attention constructor.

        Returns:
            A BaseAttention subclass instance.
        """
        from torchbridge.attention.core.registry import (
            _ATTENTION_REGISTRY,
            create_attention,
        )

        result = self.select_kernel(
            seq_length=config.max_sequence_length,
            num_heads=config.num_heads,
            head_dim=config.head_dim or (config.embed_dim // config.num_heads),
        )

        # Try the dispatched implementation first (B1 fix)
        impl_name = result.implementation_name
        if impl_name in _ATTENTION_REGISTRY:
            logger.debug("Using dispatched kernel: %s → %s", result.kernel_type.value, impl_name)
            return create_attention(config, implementation=impl_name, **kwargs)

        # Walk the fallback chain before giving up (B1 fix)
        logger.warning(
            "Dispatched impl '%s' not in registry; walking fallback chain (%d candidates)",
            impl_name,
            len(result.fallback_chain),
        )
        for fallback_kt in result.fallback_chain:
            fb_impl = self._kernel_to_registry_name(fallback_kt)
            if fb_impl in _ATTENTION_REGISTRY:
                logger.info("Using fallback: %s → %s", fallback_kt.value, fb_impl)
                return create_attention(config, implementation=fb_impl, **kwargs)

        # Final fallback: auto-select (last resort)
        logger.info("No dispatch or fallback matched registry; using auto-select")
        return create_attention(config, **kwargs)

    # ── properties ───────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return self._backend.value

    @property
    def architecture_name(self) -> str:
        if self._architecture is not None:
            return self._architecture.value
        return "unknown"

    # ── runtime availability checks ──────────────────────────────────

    def _check_kernel_availability(self, kernel_type: AttentionKernelType) -> bool:
        """Check if a kernel is actually usable at runtime."""
        if kernel_type == AttentionKernelType.PYTORCH_SDPA:
            return True  # always available

        if kernel_type == AttentionKernelType.FLEX_ATTENTION:
            return self._check_flex_attention()

        if kernel_type in (
            AttentionKernelType.FLASH_ATTENTION_3,
            AttentionKernelType.FLASH_ATTENTION_2,
        ):
            return self._check_flash_attention()

        if kernel_type == AttentionKernelType.FLASH_ATTENTION_CK:
            return self._check_flash_attention_ck()

        if kernel_type == AttentionKernelType.NEURONX_SDPA:
            return self._check_neuronx()

        if kernel_type == AttentionKernelType.PALLAS_ATTENTION:
            return self._check_pallas()

        return False

    @staticmethod
    def _check_flex_attention() -> bool:
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

            return True
        except (ImportError, ModuleNotFoundError):
            return False

    @staticmethod
    def _check_flash_attention() -> bool:
        try:
            import flash_attn  # noqa: F401

            return True
        except (ImportError, ModuleNotFoundError):
            return False

    @staticmethod
    def _check_flash_attention_ck() -> bool:
        """Check CK FlashAttention availability.

        Requires both the flash_attn package AND a ROCm runtime.
        On ROCm, flash_attn ships Composable Kernel-backed kernels.
        On CUDA, flash_attn uses NVIDIA kernels — CK is not relevant.
        """
        try:
            import flash_attn  # noqa: F401
            import torch

            # CK kernels only activate on ROCm (torch.version.hip is set)
            return getattr(torch.version, "hip", None) is not None
        except (ImportError, ModuleNotFoundError):
            return False

    @staticmethod
    def _check_neuronx() -> bool:
        try:
            import torch_neuronx  # noqa: F401

            return True
        except (ImportError, ModuleNotFoundError):
            return False

    @staticmethod
    def _check_pallas() -> bool:
        try:
            import jax  # noqa: F401

            return True
        except (ImportError, ModuleNotFoundError):
            return False

    # ── helpers ──────────────────────────────────────────────────────

    def _detect_architecture(
        self,
    ) -> NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None:
        if self._backend == HardwareBackend.CUDA:
            return self._hw.nvidia.architecture
        elif self._backend == HardwareBackend.AMD:
            return self._hw.amd.architecture
        elif self._backend == HardwareBackend.TRAINIUM:
            return self._hw.trainium.architecture
        elif self._backend == HardwareBackend.TPU:
            return self._hw.tpu.version
        return None

    @staticmethod
    def _kernel_to_registry_name(kernel_type: AttentionKernelType) -> str:
        return _KERNEL_REGISTRY_MAP.get(kernel_type, "memory_efficient_attention")
