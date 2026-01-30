#!/usr/bin/env python3
"""
FlexAttention Demo (v0.4.4)

Demonstrates FlexAttention integration with PyTorch 2.5+ for flexible
attention patterns with FlashAttention-like performance.

Features:
- Multiple attention patterns (causal, sliding window, ALiBi)
- Custom score modifications
- Performance comparison with standard attention
- Integration with TorchBridge attention system
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

# Import shared utilities
try:
    from shared.utils import print_section
except ImportError:
    def print_section(title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)

from torchbridge.attention import (
    AttentionConfig,
    AttentionPatterns,
    create_attention,
)
from torchbridge.attention.implementations.flex_attention import (
    FlexAttentionLayer,
    FlexAttentionCausal,
    FlexAttentionSlidingWindow,
    FlexAttentionScoreMods,
    create_flex_attention,
    is_flex_attention_available,
    get_flex_attention_info,
)


def demo_flex_attention_availability():
    """Demo: Check FlexAttention availability"""
    print_section("FlexAttention Availability Check")

    info = get_flex_attention_info()
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"FlexAttention Available: {info['available']}")
    print(f"torch.compile Available: {info['torch_compile_available']}")
    print(f"\nSupported Patterns:")
    for pattern in info['supported_patterns']:
        print(f"  - {pattern}")

    return info['available']


def demo_basic_flex_attention():
    """Demo: Basic FlexAttention usage"""
    print_section("Basic FlexAttention")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create FlexAttention layer
    layer = create_flex_attention(
        embed_dim=256,
        num_heads=4,
        pattern='full'
    ).to(device)

    # Sample input
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, 256, device=device)

    # Forward pass
    output = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Stats: {layer.get_attention_stats()}")

    return True


def demo_causal_attention():
    """Demo: Causal/autoregressive attention"""
    print_section("Causal Attention Pattern")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create causal attention
    config = AttentionConfig(
        embed_dim=256,
        num_heads=4,
        pattern=AttentionPatterns.CAUSAL,
        causal=True
    )
    layer = FlexAttentionCausal(config).to(device)

    # Sample input
    x = torch.randn(2, 64, 256, device=device)
    output = layer(x)

    print(f"Pattern: Causal (autoregressive)")
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Use case: Language modeling, text generation")

    return True


def demo_sliding_window_attention():
    """Demo: Sliding window attention"""
    print_section("Sliding Window Attention")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create sliding window attention
    window_size = 32
    layer = FlexAttentionSlidingWindow(
        AttentionConfig(embed_dim=256, num_heads=4),
        window_size=window_size
    ).to(device)

    x = torch.randn(2, 128, 256, device=device)
    output = layer(x)

    print(f"Pattern: Sliding Window (size={window_size})")
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Use case: Long sequence processing with local context")
    print(f"Memory: O(N*W) vs O(N^2) for full attention")

    return True


def demo_custom_score_mod():
    """Demo: Custom score modification"""
    print_section("Custom Score Modification")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example: Soft capping for Gemma 2 style attention
    def soft_cap_score_mod(score, b, h, q_idx, kv_idx):
        cap_value = 30.0
        return cap_value * torch.tanh(score / cap_value)

    layer = create_flex_attention(
        embed_dim=256,
        num_heads=4,
        score_mod=soft_cap_score_mod
    ).to(device)

    x = torch.randn(2, 64, 256, device=device)
    output = layer(x)

    print("Custom score_mod: Soft capping (Gemma 2 style)")
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print("Use case: Improved training stability for large models")

    return True


def demo_alibi_attention():
    """Demo: ALiBi positional bias"""
    print_section("ALiBi Attention")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create attention with ALiBi
    alibi_mod = FlexAttentionScoreMods.alibi(num_heads=4)
    layer = create_flex_attention(
        embed_dim=256,
        num_heads=4,
        score_mod=alibi_mod
    ).to(device)

    x = torch.randn(2, 128, 256, device=device)
    output = layer(x)

    print("Pattern: ALiBi (Attention with Linear Biases)")
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print("Use case: Length extrapolation without positional embeddings")

    return True


def demo_performance_comparison():
    """Demo: Performance comparison"""
    print_section("Performance Comparison")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup
    embed_dim, num_heads = 512, 8
    seq_lengths = [64, 128, 256, 512]
    batch_size = 4

    # Create layers
    flex_layer = create_flex_attention(
        embed_dim, num_heads, pattern='causal'
    ).to(device)

    standard_config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        pattern=AttentionPatterns.CAUSAL
    )
    standard_layer = create_attention(
        standard_config,
        implementation='flash_attention3'
    ).to(device)

    print(f"Comparing FlexAttention vs FlashAttention-3")
    print(f"Device: {device}, Batch: {batch_size}, Embed: {embed_dim}, Heads: {num_heads}")
    print()

    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Warmup
        for _ in range(3):
            _ = flex_layer(x)
            _ = standard_layer(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark FlexAttention
        start = time.perf_counter()
        for _ in range(10):
            _ = flex_layer(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        flex_time = (time.perf_counter() - start) / 10 * 1000

        # Benchmark Standard
        start = time.perf_counter()
        for _ in range(10):
            _ = standard_layer(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        standard_time = (time.perf_counter() - start) / 10 * 1000

        speedup = standard_time / flex_time if flex_time > 0 else 1.0
        print(f"  Seq={seq_len:4d}: Flex={flex_time:.2f}ms, Standard={standard_time:.2f}ms, Ratio={speedup:.2f}x")

    return True


def demo_transformer_block():
    """Demo: FlexAttention in a transformer block"""
    print_section("Transformer Block with FlexAttention")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class FlexTransformerBlock(nn.Module):
        """Transformer block using FlexAttention"""

        def __init__(self, embed_dim, num_heads, ffn_dim, pattern='causal'):
            super().__init__()
            self.attention = create_flex_attention(
                embed_dim, num_heads, pattern=pattern
            )
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, embed_dim)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        def forward(self, x):
            x = x + self.attention(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    # Create and test block
    block = FlexTransformerBlock(
        embed_dim=256,
        num_heads=4,
        ffn_dim=1024,
        pattern='causal'
    ).to(device)

    x = torch.randn(2, 64, 256, device=device)
    output = block(x)

    print(f"Transformer block with FlexAttention causal pattern")
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Components: LayerNorm -> FlexAttention -> LayerNorm -> FFN")

    # Test gradient flow
    x.requires_grad = True
    output = block(x)
    loss = output.sum()
    loss.backward()

    print(f"Gradient flow: OK (grad shape: {x.grad.shape})")

    return True


def demo_registry_integration():
    """Demo: Registry integration"""
    print_section("Registry Integration")

    from torchbridge.attention.core.registry import (
        get_attention_registry,
        list_available_attention
    )

    # List available implementations
    available = list_available_attention()
    print("Registered attention implementations:")
    for name in sorted(available):
        if 'flex' in name.lower():
            print(f"  * {name} (FlexAttention)")
        else:
            print(f"    {name}")

    # Create via registry
    config = AttentionConfig(embed_dim=256, num_heads=4)
    layer = create_attention(config, implementation='flex_attention')
    print(f"\nCreated via registry: {type(layer).__name__}")

    return True


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("  FlexAttention Demo - TorchBridge v0.4.4")
    print("="*60)

    results = {}

    # Check availability first
    available = demo_flex_attention_availability()
    results['availability'] = available

    # Run demos
    demos = [
        ('basic', demo_basic_flex_attention),
        ('causal', demo_causal_attention),
        ('sliding_window', demo_sliding_window_attention),
        ('custom_score_mod', demo_custom_score_mod),
        ('alibi', demo_alibi_attention),
        ('performance', demo_performance_comparison),
        ('transformer_block', demo_transformer_block),
        ('registry', demo_registry_integration),
    ]

    for name, demo_fn in demos:
        try:
            success = demo_fn()
            results[name] = 'PASSED' if success else 'FAILED'
        except Exception as e:
            results[name] = f'ERROR: {e}'
            print(f"Error in {name}: {e}")

    # Summary
    print_section("Demo Summary")
    passed = sum(1 for v in results.values() if v == 'PASSED' or v == True)
    total = len(results)
    print(f"Results: {passed}/{total} demos passed")
    print()
    for name, result in results.items():
        status = "PASS" if result == 'PASSED' or result == True else "FAIL"
        print(f"  [{status}] {name}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
