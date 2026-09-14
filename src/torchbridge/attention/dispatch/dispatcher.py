# SPDX-License-Identifier: Apache-2.0
"""
Attention Kernel Dispatcher

Compatibility matrix that selects the best available attention kernel for the
given hardware backend. PyTorch SDPA handles runtime dispatch; TorchBridge
selects which backend context (FlashAttention, SDPA, math fallback) to use
based on (backend, dtype, sequence_length) via the compatibility matrix.
"""

from __future__ import annotations

import importlib
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
        architecture: NVIDIAArchitecture
        | AMDArchitecture
        | TrainiumArchitecture
        | TPUVersion
        | None = None,
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
        if seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {seq_length}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

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
            msg = f"{kernel.value} not available at runtime, trying next"
            result_warnings.append(msg)
            logger.warning(msg)

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
            _key = f"{chosen.value}_{seq_length}_{num_heads}_{head_dim}"
            _entry = self._cache._entries.get(_key)
            latency = _entry.latency_ms if _entry else None
            if latency is None and seq_length <= _MAX_BENCH_SEQ:
                try:
                    entry = self._cache.run_benchmark(
                        chosen,
                        seq_length,
                        num_heads,
                        head_dim,
                        warmup=1,
                        iterations=5,
                    )
                    latency = entry.latency_ms
                    logger.debug(
                        "Lazy benchmark: %s at %s×%s×%s → %.2f ms",
                        chosen.value,
                        seq_length,
                        num_heads,
                        head_dim,
                        latency,
                    )
                except Exception as _e:
                    logger.debug("Lazy benchmark failed (non-fatal): %s", _e)

        fallback_chain = AttentionDispatchMatrix.get_fallback_chain(
            chosen, self._backend, self._architecture
        )

        return AttentionDispatchResult(
            kernel_type=chosen,
            implementation_name=chosen.value,
            used_fallback=used_fallback,
            fallback_chain=fallback_chain,
            benchmark_latency_ms=latency,
            warnings=result_warnings,
        )

    # ── runtime availability checks ──────────────────────────────────

    # Module paths for kernels whose availability is determined by a single import.
    # FLASH_ATTENTION_CK is excluded — it requires both flash_attn AND ROCm detection.
    _IMPORT_CHECKS: dict[AttentionKernelType, str] = {
        AttentionKernelType.FLEX_ATTENTION: "torch.nn.attention.flex_attention",
        AttentionKernelType.FLASH_ATTENTION_2: "flash_attn",
        AttentionKernelType.FLASH_ATTENTION_3: "flash_attn",
        AttentionKernelType.NEURONX_SDPA: "torch_neuronx",
        AttentionKernelType.PALLAS_ATTENTION: "jax",
    }

    def _check_kernel_availability(self, kernel_type: AttentionKernelType) -> bool:
        """Check if a kernel is actually usable at runtime."""
        if kernel_type == AttentionKernelType.PYTORCH_SDPA:
            return True  # always available

        if kernel_type == AttentionKernelType.FLASH_ATTENTION_CK:
            return self._check_flash_attention_ck()

        module_path = self._IMPORT_CHECKS.get(kernel_type)
        if module_path is None:
            return False
        try:
            importlib.import_module(module_path)
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

            # CK kernels only activate on ROCm. Asked through the shared
            # helper rather than inspected here: an empty HIP version string
            # has been seen in the wild, and an `is not None` test called that
            # ROCm while the hardware detector called it CUDA. One question,
            # two answers, depending on which module you asked.
            from torchbridge.core.hardware_detector import is_rocm_build

            return is_rocm_build()
        except (ImportError, ModuleNotFoundError):
            return False

    # ── helpers ──────────────────────────────────────────────────────

    def _detect_architecture(
        self,
    ) -> (
        NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
    ):
        if self._backend == HardwareBackend.CUDA:
            return self._hw.nvidia.architecture
        elif self._backend == HardwareBackend.AMD:
            return self._hw.amd.architecture
        elif self._backend == HardwareBackend.TRAINIUM:
            return self._hw.trainium.architecture
        elif self._backend == HardwareBackend.TPU:
            return self._hw.tpu.version
        return None
