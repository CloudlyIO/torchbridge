#!/usr/bin/env python3
"""
Attention Cross-Backend Dispatch Example

Demonstrates TorchBridge's backend-aware attention kernel dispatch:
1. Auto-detecting hardware and selecting the optimal attention kernel
2. GQA (Grouped-Query Attention) in Llama 3 style
3. Forward pass with dispatched attention
4. Querying kernel statistics
"""

import torch

from torchbridge.attention import (
    AttentionDispatcher,
    AttentionDispatchMatrix,
    AttentionModuleConfig,
)
from torchbridge.core.config import HardwareBackend


def main():
    print("=" * 60)
    print("TorchBridge — Attention Cross-Backend Dispatch")
    print("=" * 60)

    # ── 1. Auto-detect and dispatch ──────────────────────────────
    dispatcher = AttentionDispatcher(use_benchmark_cache=False)
    print(f"\nDetected backend: {dispatcher.backend_name}")
    print(f"Architecture:     {dispatcher.architecture_name}")

    result = dispatcher.select_kernel(seq_length=2048, num_heads=32, head_dim=128)
    print(f"Selected kernel:  {result.kernel_type.value}")
    print(f"Registry impl:    {result.implementation_name}")
    print(f"Used fallback:    {result.used_fallback}")
    if result.fallback_chain:
        print(f"Fallback chain:   {[k.value for k in result.fallback_chain]}")
    if result.warnings:
        for w in result.warnings:
            print(f"  Warning: {w}")

    # ── 2. Standard MHA forward pass ─────────────────────────────
    print("\n--- Standard MHA ---")
    config_mha = AttentionModuleConfig(embed_dim=128, num_heads=8)
    attn_mha = dispatcher.create_attention(config_mha)
    x = torch.randn(1, 32, 128)
    with torch.no_grad():
        out = attn_mha(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Stats:  {attn_mha.get_attention_stats()}")

    # ── 3. GQA forward pass (Llama 3 style) ──────────────────────
    print("\n--- GQA (32 Q heads, 8 KV heads) ---")
    config_gqa = AttentionModuleConfig(
        embed_dim=256, num_heads=32, num_kv_heads=8
    )
    attn_gqa = dispatcher.create_attention(config_gqa)
    x_gqa = torch.randn(1, 32, 256)
    with torch.no_grad():
        out_gqa = attn_gqa(x_gqa)
    print(f"Input:            {x_gqa.shape}")
    print(f"Output:           {out_gqa.shape}")
    print(f"KV repeat factor: {config_gqa.kv_head_repeat_factor}")

    # ── 4. Query compatibility matrix ─────────────────────────────
    print("\n--- Compatibility Matrix Queries ---")
    for backend in [
        HardwareBackend.CUDA,
        HardwareBackend.AMD,
        HardwareBackend.TRAINIUM,
        HardwareBackend.TPU,
        HardwareBackend.CPU,
    ]:
        optimal = AttentionDispatchMatrix.get_optimal_kernel(backend)
        print(f"  {backend.value:10s} → {optimal.value}")

    print("\nDone.")


if __name__ == "__main__":
    main()
