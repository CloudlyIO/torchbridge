"""
Attention Kernel Type Definitions

Enum of dispatchable attention algorithms, distinct from KernelBackend
(which represents CUDA/Triton/PyTorch execution backends).
"""

from enum import Enum


class AttentionKernelType(Enum):
    """Attention kernel implementations available for dispatch.

    Each member represents a distinct runtime dependency and availability
    check, even when some share the same underlying registry entry:

    - FLEX_ATTENTION: PyTorch 2.5+ torch.nn.attention.flex_attention
    - FLASH_ATTENTION_3/2: Dao-AILab flash-attn library
    - FLASH_ATTENTION_CK: AMD Composable Kernel (flash-attn interface)
    - TRITON_ATTENTION: OpenAI Triton kernel
    - NEURONX_SDPA: AWS torch-neuronx SDPA
    - PALLAS_ATTENTION: Google JAX/Pallas for TPU
    - PYTORCH_SDPA: PyTorch F.scaled_dot_product_attention (always available)
    """

    FLEX_ATTENTION = "flex_attention"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_CK = "flash_attention_ck"  # AMD Composable Kernel
    TRITON_ATTENTION = "triton_attention"
    NEURONX_SDPA = "neuronx_sdpa"
    PALLAS_ATTENTION = "pallas_attention"
    PYTORCH_SDPA = "pytorch_sdpa"
