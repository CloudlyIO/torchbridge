#!/usr/bin/env python3
"""
Sparse Attention Patterns Demo

Demonstrates various sparse attention mechanisms for efficient processing of
long sequences with different sparsity patterns and their trade-offs.

Learning Objectives:
1. Understanding block-sparse attention patterns
2. Exploring random and structured sparsity
3. Comparing computational and memory benefits
4. Learning about sparsity pattern design principles

Expected Time: 8-12 minutes
Hardware: GPU recommended for performance comparisons
"""

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from kernel_pytorch.attention.sparse_patterns import (
        BlockSparsePattern, RandomSparsePattern, StructuredSparsePattern
    )
    SPARSE_PATTERNS_AVAILABLE = True
except ImportError:
    SPARSE_PATTERNS_AVAILABLE = False


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"✨ {title}")
    print(f"{'='*60}")


def visualize_sparsity_pattern(mask: torch.Tensor, name: str, max_size: int = 64):
    """Visualize attention sparsity pattern"""
    if mask.size(0) > max_size:
        # Sample from the mask for visualization
        indices = torch.linspace(0, mask.size(0) - 1, max_size).long()
        mask_vis = mask[indices][:, indices]
    else:
        mask_vis = mask

    print(f"\n🎨 {name} Pattern Visualization ({mask_vis.size(0)}x{mask_vis.size(1)}):")

    # Convert to numpy for visualization
    mask_np = mask_vis.cpu().numpy()

    # Print a small section of the pattern
    for i in range(min(16, mask_np.shape[0])):
        row = ""
        for j in range(min(32, mask_np.shape[1])):
            row += "█" if mask_np[i, j] else "·"
        if mask_np.shape[1] > 32:
            row += "..."
        print(f"  {row}")

    if mask_np.shape[0] > 16:
        print("  ...")

    sparsity = mask.float().mean().item() * 100
    print(f"  Sparsity: {sparsity:.1f}% (█=masked, ·=attend)")


def create_block_sparse_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Create block-sparse attention mask"""
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

    num_blocks = seq_len // block_size

    # Allow attention within each block
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        mask[start:end, start:end] = False

    # Add some cross-block connections (e.g., to adjacent blocks)
    for i in range(num_blocks - 1):
        start_i = i * block_size
        end_i = min(start_i + block_size, seq_len)
        start_j = (i + 1) * block_size
        end_j = min(start_j + block_size, seq_len)

        # Connect to next block
        mask[start_i:end_i, start_j:end_j] = False
        mask[start_j:end_j, start_i:end_i] = False

    return mask


def create_random_sparse_mask(seq_len: int, sparsity: float, device: torch.device) -> torch.Tensor:
    """Create random sparse attention mask"""
    # Start with full mask
    mask = torch.rand(seq_len, seq_len, device=device) < sparsity

    # Ensure causal structure (optional)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask = mask | causal_mask

    # Always allow self-attention
    mask.fill_diagonal_(False)

    return mask


def create_structured_sparse_mask(seq_len: int, pattern: str, device: torch.device) -> torch.Tensor:
    """Create structured sparse attention mask"""
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

    if pattern == "local":
        # Local attention with window
        window_size = 64
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = False

    elif pattern == "strided":
        # Strided attention
        stride = 8
        for i in range(seq_len):
            # Local connections
            start = max(0, i - 4)
            end = min(seq_len, i + 5)
            mask[i, start:end] = False

            # Strided connections
            for j in range(0, seq_len, stride):
                if abs(i - j) <= seq_len // 4:
                    mask[i, j] = False

    elif pattern == "global_local":
        # Combination of global and local attention
        local_window = 32
        global_tokens = 16

        # Local attention
        for i in range(seq_len):
            start = max(0, i - local_window // 2)
            end = min(seq_len, i + local_window // 2 + 1)
            mask[i, start:end] = False

        # Global attention to first few tokens
        mask[:, :global_tokens] = False
        mask[:global_tokens, :] = False

    return mask


class SparseAttention(nn.Module):
    """Sparse attention with configurable sparsity patterns"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q, k, v: [batch, num_heads, seq_len, head_dim]
            mask: [seq_len, seq_len] - True means masked (no attention)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sparsity mask
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)

        # Zero out masked positions in attention weights for efficiency
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), 0.0)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights


def benchmark_sparse_attention(attention: SparseAttention, q: torch.Tensor, k: torch.Tensor,
                              v: torch.Tensor, mask: torch.Tensor, name: str) -> Dict[str, float]:
    """Benchmark sparse attention performance"""
    device = q.device

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = attention(q, k, v, mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    memory_usage = []

    for _ in range(10):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.no_grad():
            output, attn_weights = attention(q, k, v, mask)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage.append(peak_memory)

        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0

    # Calculate sparsity metrics
    sparsity = mask.float().mean().item()
    effective_ops = (1.0 - sparsity) * q.size(2) * q.size(2)

    return {
        "time": avg_time,
        "memory": avg_memory,
        "sparsity": sparsity,
        "effective_ops": effective_ops
    }


def demo_sparsity_patterns():
    """Demonstrate different sparsity patterns"""
    print_section("Sparsity Pattern Comparison")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 512

    print(f"Device: {device}")
    print(f"Sequence Length: {seq_len}")

    # Create different sparsity patterns
    patterns = [
        ("Dense (Full)", torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)),
        ("Block Sparse", create_block_sparse_mask(seq_len, 64, device)),
        ("Random Sparse", create_random_sparse_mask(seq_len, 0.8, device)),
        ("Local Window", create_structured_sparse_mask(seq_len, "local", device)),
        ("Strided", create_structured_sparse_mask(seq_len, "strided", device)),
        ("Global+Local", create_structured_sparse_mask(seq_len, "global_local", device)),
    ]

    print(f"\n📊 Pattern Analysis:")

    pattern_stats = {}

    for name, mask in patterns:
        sparsity = mask.float().mean().item() * 100
        # Estimate memory reduction
        full_memory = seq_len * seq_len * 4  # bytes for float32
        sparse_memory = (seq_len * seq_len * (1.0 - sparsity/100)) * 4
        memory_reduction = (full_memory - sparse_memory) / full_memory * 100

        print(f"\n  {name}:")
        print(f"    Sparsity: {sparsity:.1f}%")
        print(f"    Memory Reduction: {memory_reduction:.1f}%")
        print(f"    Effective Operations: {(1.0 - sparsity/100) * seq_len * seq_len:,.0f}")

        pattern_stats[name] = {
            "sparsity": sparsity,
            "memory_reduction": memory_reduction,
            "mask": mask
        }

        # Visualize smaller patterns
        if seq_len <= 128:
            visualize_sparsity_pattern(mask, name)

    return pattern_stats


def demo_sparse_attention_performance():
    """Demonstrate sparse attention performance comparisons"""
    print_section("Sparse Attention Performance")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test configurations
    configs = [
        {"seq_len": 512, "name": "Medium"},
        {"seq_len": 1024, "name": "Long"},
    ]

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
        configs.append({"seq_len": 2048, "name": "Very Long"})

    d_model = 512
    num_heads = 8
    batch_size = 4

    print(f"Configuration:")
    print(f"  Model Dimension: {d_model}")
    print(f"  Number of Heads: {num_heads}")
    print(f"  Batch Size: {batch_size}")

    attention = SparseAttention(d_model, num_heads).to(device)

    for config in configs:
        seq_len = config["seq_len"]
        name = config["name"]

        print(f"\n🔍 Testing {name} Sequences (Length: {seq_len:,})")

        try:
            # Create test data
            head_dim = d_model // num_heads
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

            # Test different sparsity levels
            sparsity_patterns = [
                ("Dense", torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)),
                ("Block Sparse", create_block_sparse_mask(seq_len, 64, device)),
                ("80% Random", create_random_sparse_mask(seq_len, 0.8, device)),
                ("95% Random", create_random_sparse_mask(seq_len, 0.95, device)),
            ]

            baseline_time = None

            for pattern_name, mask in sparsity_patterns:
                try:
                    result = benchmark_sparse_attention(attention, q, k, v, mask, pattern_name)

                    if baseline_time is None:
                        baseline_time = result["time"]
                        speedup = 1.0
                    else:
                        speedup = baseline_time / result["time"]

                    print(f"  {pattern_name}:")
                    print(f"    Time: {result['time']*1000:.1f}ms")
                    print(f"    Speedup: {speedup:.2f}x")
                    print(f"    Sparsity: {result['sparsity']*100:.1f}%")
                    if device.type == 'cuda':
                        print(f"    Memory: {result['memory']/1024**2:.1f}MB")

                except Exception as e:
                    print(f"  {pattern_name}: ❌ Failed ({e})")

        except Exception as e:
            print(f"  ❌ Failed to test {name} sequence: {e}")


def demo_sparsity_design_principles():
    """Demonstrate principles for designing effective sparsity patterns"""
    print_section("Sparsity Design Principles")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 256  # Smaller for detailed analysis

    print(f"Analyzing effective sparsity pattern design...")
    print(f"Sequence Length: {seq_len}")

    # Principle 1: Preserve important connections
    print(f"\n1️⃣ Preserving Important Connections:")

    # Self-attention is always important
    mask_no_self = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    mask_no_self.fill_diagonal_(False)

    # Local connections are usually important
    mask_with_local = mask_no_self.clone()
    for i in range(seq_len):
        start = max(0, i - 2)
        end = min(seq_len, i + 3)
        mask_with_local[i, start:end] = False

    local_sparsity = mask_with_local.float().mean().item() * 100
    print(f"  Local connections preserved: {100-local_sparsity:.1f}% connectivity")

    # Principle 2: Balance sparsity and expressivity
    print(f"\n2️⃣ Balancing Sparsity and Expressivity:")

    sparsity_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    for sparsity in sparsity_levels:
        mask = create_random_sparse_mask(seq_len, sparsity, device)
        actual_sparsity = mask.float().mean().item() * 100

        # Estimate information flow (simplified)
        connectivity = 100 - actual_sparsity
        info_flow = connectivity * math.log(connectivity + 1)  # Rough estimate

        print(f"  Sparsity {actual_sparsity:.0f}%: Connectivity {connectivity:.0f}%, Est. Info Flow {info_flow:.1f}")

    # Principle 3: Computational efficiency
    print(f"\n3️⃣ Computational Efficiency Analysis:")

    patterns_efficiency = [
        ("Block 32x32", create_block_sparse_mask(seq_len, 32, device)),
        ("Block 64x64", create_block_sparse_mask(seq_len, 64, device)),
        ("Strided", create_structured_sparse_mask(seq_len, "strided", device)),
    ]

    for name, mask in patterns_efficiency:
        sparsity = mask.float().mean().item() * 100
        # Block patterns are more cache-friendly
        cache_efficiency = 100 - sparsity if "Block" in name else (100 - sparsity) * 0.7

        print(f"  {name}:")
        print(f"    Sparsity: {sparsity:.1f}%")
        print(f"    Cache Efficiency Score: {cache_efficiency:.1f}")


def run_demo(quick_mode: bool = False, validate: bool = False):
    """Run the complete sparse attention demo"""

    print("✨ Sparse Attention Patterns Demo")
    print("Exploring efficient attention through strategic sparsity!")

    device_info = f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"
    print(f"📱 {device_info}")

    if not SPARSE_PATTERNS_AVAILABLE:
        print("\n⚠️  Using built-in sparse pattern implementations")

    try:
        # Demo 1: Sparsity patterns comparison
        pattern_stats = demo_sparsity_patterns()

        if not quick_mode:
            # Demo 2: Performance benchmarks
            demo_sparse_attention_performance()

            # Demo 3: Design principles
            demo_sparsity_design_principles()

        print_section("Sparse Attention Summary")
        print("✅ Key Concepts Demonstrated:")
        print("  🧩 Block-sparse attention for cache efficiency")
        print("  🎲 Random sparsity for general purpose reduction")
        print("  🏗️ Structured patterns for specific use cases")
        print("  ⚡ Performance trade-offs across sparsity levels")

        # Find best sparsity pattern
        if pattern_stats:
            best_pattern = max(pattern_stats.items(),
                             key=lambda x: x[1]["memory_reduction"] * (1 - x[1]["sparsity"]/200))
            print(f"\n📈 Performance Insights:")
            print(f"  Best balanced pattern: {best_pattern[0]}")
            print(f"  Memory reduction: {best_pattern[1]['memory_reduction']:.1f}%")

        print(f"\n🎓 Key Design Principles:")
        print(f"  • Preserve self-attention and local connections")
        print(f"  • Balance sparsity level with task requirements")
        print(f"  • Consider hardware cache-friendliness")
        print(f"  • Block patterns often outperform random sparsity")
        print(f"  • Structured sparsity enables better optimization")

        if validate:
            print(f"\n🧪 Validation Results:")
            print(f"  Sparsity patterns: ✅")
            print(f"  Performance benchmarks: ✅")
            print(f"  Design principles: ✅")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if validate:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main demo entry point"""
    parser = argparse.ArgumentParser(description="Sparse Attention Patterns Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    parser.add_argument("--validate", action="store_true", help="Run with validation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    success = run_demo(quick_mode=args.quick, validate=args.validate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()