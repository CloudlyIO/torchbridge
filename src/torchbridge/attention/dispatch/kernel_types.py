# SPDX-License-Identifier: Apache-2.0
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
    - NEURONX_SDPA: AWS torch-neuronx SDPA
    - PALLAS_ATTENTION: Google JAX/Pallas for TPU
    - PYTORCH_SDPA: PyTorch F.scaled_dot_product_attention (always available)
    - DEEPSEEK_CSA_HCA: DeepSeek V4 Combined Sliding-window + Hybrid Chunk Attention (sm_89+, gfx942+)
    - MINIMAX_MSA: MiniMax M3 Lightning/Sparse Attention (sm_89+)
    - GLM_INDEX_SHARE: GLM 5.2 IndexShare expert-routing + sparse attention hybrid (sm_89+)
    - MAMBA2_HYBRID: Nemotron Ultra Mamba-2 + Transformer MoE (sm_89+; NOT XLA/Trainium)
    """

    FLEX_ATTENTION = "flex_attention"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_CK = "flash_attention_ck"  # AMD Composable Kernel
    NEURONX_SDPA = "neuronx_sdpa"
    PALLAS_ATTENTION = "pallas_attention"
    PYTORCH_SDPA = "pytorch_sdpa"
    # ── 2026 MoE attention variants (v0.5.99) ────────────────────────────────
    # DEEPSEEK_CSA_HCA: DeepSeek V4 Combined Sliding-window + Hybrid Chunk Attention
    # MINIMAX_MSA:      MiniMax M3 Lightning/Sparse Attention
    # GLM_INDEX_SHARE:  GLM 5.2 IndexShare expert-routing + sparse attention hybrid
    # MAMBA2_HYBRID:    Nemotron 3 Ultra Mamba-2 + Transformer MoE; NOT XLA-compatible
    DEEPSEEK_CSA_HCA = "deepseek_csa_hca"
    MINIMAX_MSA = "minimax_msa"
    GLM_INDEX_SHARE = "glm_index_share"
    MAMBA2_HYBRID = "mamba2_hybrid"
