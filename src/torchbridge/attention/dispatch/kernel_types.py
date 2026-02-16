"""
Attention Kernel Type Definitions

Enum of dispatchable attention algorithms, distinct from KernelBackend
(which represents CUDA/Triton/PyTorch execution backends).
"""

from enum import Enum


class AttentionKernelType(Enum):
    """Attention kernel implementations available for dispatch."""

    FLEX_ATTENTION = "flex_attention"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_CK = "flash_attention_ck"  # AMD Composable Kernel
    TRITON_ATTENTION = "triton_attention"
    NEURONX_SDPA = "neuronx_sdpa"
    PALLAS_ATTENTION = "pallas_attention"
    PYTORCH_SDPA = "pytorch_sdpa"
