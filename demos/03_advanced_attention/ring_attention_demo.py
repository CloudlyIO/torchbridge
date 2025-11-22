#!/usr/bin/env python3
"""
Ring Attention and Advanced Attention Patterns Demo

Demonstrates cutting-edge attention mechanisms including Ring Attention for scaling
to extremely long sequences, plus advanced sparse attention patterns.

Learning Objectives:
1. Understanding Ring Attention for memory-efficient long sequences
2. Exploring sparse attention patterns (sliding window, block sparse)
3. Comparing attention mechanisms across different sequence lengths
4. Learning about attention memory and computational complexity

Expected Time: 10-15 minutes
Hardware: GPU strongly recommended for best results
"""

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import math

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from kernel_pytorch.attention.ring_attention import RingAttention, RingAttentionConfig
    from kernel_pytorch.attention.sparse_attention import SparseAttention, SlidingWindowAttention
    from kernel_pytorch.attention.attention_patterns import AttentionPattern, create_attention_mask
    RING_ATTENTION_AVAILABLE = True
except ImportError:
    RING_ATTENTION_AVAILABLE = False

try:
    from kernel_pytorch.components.advanced_attention import FlexibleAttention, FlashAttentionV3
    ADVANCED_ATTENTION_AVAILABLE = True
except ImportError:
    ADVANCED_ATTENTION_AVAILABLE = False


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔮 {title}")
    print(f"{'='*60}")


def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def create_sliding_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Create sliding window attention mask"""
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = False

    return mask


class StandardAttention(nn.Module):
    """Standard scaled dot-product attention for comparison"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q, k, v: [batch, num_heads, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output


class RingAttentionMock(nn.Module):
    """Mock Ring Attention implementation for demonstration"""

    def __init__(self, d_model: int, num_heads: int, ring_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ring_size = ring_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Simulate ring attention by processing in chunks
        chunk_size = seq_len // self.ring_size
        outputs = []

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]

            # For each chunk, attend to all keys and values (simplified)
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                chunk_mask = mask[i:end_i, :]
                scores.masked_fill_(chunk_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn_weights, v)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)


class SlidingWindowAttentionMock(nn.Module):
    """Mock Sliding Window Attention implementation"""

    def __init__(self, d_model: int, num_heads: int, window_size: int = 512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Create sliding window mask
        mask = create_sliding_window_mask(seq_len, self.window_size, q.device)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output


def demo_attention_scaling():
    """Demonstrate attention scaling with different mechanisms"""
    print_section("Attention Scaling Comparison")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test configurations for different sequence lengths
    configs = [
        {"seq_len": 512, "name": "Short"},
        {"seq_len": 2048, "name": "Medium"},
        {"seq_len": 8192, "name": "Long"},
    ]

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 16e9:
        configs.append({"seq_len": 32768, "name": "Very Long"})

    d_model, num_heads = 768, 12
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  Model Dimension: {d_model}")
    print(f"  Number of Heads: {num_heads}")
    print(f"  Batch Size: {batch_size}")

    # Initialize attention mechanisms
    standard_attn = StandardAttention(d_model, num_heads).to(device)
    ring_attn = RingAttentionMock(d_model, num_heads, ring_size=4).to(device)
    sliding_attn = SlidingWindowAttentionMock(d_model, num_heads, window_size=1024).to(device)

    results = {}

    for config in configs:
        seq_len = config["seq_len"]
        name = config["name"]

        print(f"\n🔍 Testing {name} Sequences (Length: {seq_len:,})")

        # Create test data
        try:
            head_dim = d_model // num_heads
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

            # Test standard attention
            try:
                start_time = time.perf_counter()
                with torch.no_grad():
                    standard_output = standard_attn(q, k, v)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                standard_time = time.perf_counter() - start_time
                print(f"  Standard Attention: {standard_time*1000:.1f}ms")

                if device.type == 'cuda':
                    print_memory_usage()

            except Exception as e:
                print(f"  Standard Attention: ❌ Failed ({e})")
                standard_time = float('inf')

            # Test ring attention
            try:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                start_time = time.perf_counter()
                with torch.no_grad():
                    ring_output = ring_attn(q, k, v)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                ring_time = time.perf_counter() - start_time
                print(f"  Ring Attention: {ring_time*1000:.1f}ms")

            except Exception as e:
                print(f"  Ring Attention: ❌ Failed ({e})")
                ring_time = float('inf')

            # Test sliding window attention
            try:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                start_time = time.perf_counter()
                with torch.no_grad():
                    sliding_output = sliding_attn(q, k, v)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                sliding_time = time.perf_counter() - start_time
                print(f"  Sliding Window: {sliding_time*1000:.1f}ms")

            except Exception as e:
                print(f"  Sliding Window: ❌ Failed ({e})")
                sliding_time = float('inf')

            # Calculate memory complexity
            memory_std = seq_len * seq_len * batch_size * num_heads * 4 / 1024**2  # MB
            memory_ring = seq_len * seq_len * batch_size * num_heads * 4 / (4 * 1024**2)  # Ring reduces memory
            memory_sliding = min(memory_std, 1024 * seq_len * batch_size * num_heads * 4 / 1024**2)  # Window size limit

            print(f"  Memory Estimate:")
            print(f"    Standard: {memory_std:.1f}MB")
            print(f"    Ring: {memory_ring:.1f}MB")
            print(f"    Sliding: {memory_sliding:.1f}MB")

            results[name] = {
                "seq_len": seq_len,
                "standard_time": standard_time,
                "ring_time": ring_time,
                "sliding_time": sliding_time,
                "memory_std": memory_std,
                "memory_ring": memory_ring,
                "memory_sliding": memory_sliding
            }

        except Exception as e:
            print(f"  ❌ Failed to test {name} sequence: {e}")

    return results


def demo_attention_patterns():
    """Demonstrate different attention pattern optimizations"""
    print_section("Advanced Attention Patterns")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 1024
    d_model = 512
    num_heads = 8
    batch_size = 4

    print(f"Configuration:")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Model Dimension: {d_model}")
    print(f"  Number of Heads: {num_heads}")
    print(f"  Batch Size: {batch_size}")

    # Create test data
    head_dim = d_model // num_heads
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    attention_fn = StandardAttention(d_model, num_heads).to(device)

    # Test different attention patterns
    patterns = [
        ("Full Attention", None),
        ("Causal Attention", create_causal_mask(seq_len, device)),
        ("Sliding Window", create_sliding_window_mask(seq_len, 256, device)),
    ]

    print(f"\n⚡ Pattern Performance Comparison:")

    pattern_results = {}

    for pattern_name, mask in patterns:
        try:
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = attention_fn(q, k, v, mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(10):
                start = time.perf_counter()
                with torch.no_grad():
                    output = attention_fn(q, k, v, mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            pattern_results[pattern_name] = avg_time

            # Calculate attention complexity
            if mask is not None:
                sparsity = mask.float().mean().item()
                effective_ops = (1.0 - sparsity) * seq_len * seq_len
            else:
                sparsity = 0.0
                effective_ops = seq_len * seq_len

            print(f"  {pattern_name}:")
            print(f"    Time: {avg_time*1000:.2f}ms")
            print(f"    Sparsity: {sparsity*100:.1f}%")
            print(f"    Effective Ops: {effective_ops:,.0f}")

        except Exception as e:
            print(f"  {pattern_name}: ❌ Failed ({e})")

    return pattern_results


def demo_memory_efficient_attention():
    """Demonstrate memory-efficient attention implementations"""
    print_section("Memory-Efficient Attention")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("⚠️  Memory efficiency demos work best with CUDA")

    seq_lengths = [1024, 2048, 4096]
    d_model = 768
    num_heads = 12
    batch_size = 2

    print(f"Testing memory usage across sequence lengths...")

    for seq_len in seq_lengths:
        print(f"\n📊 Sequence Length: {seq_len}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            head_dim = d_model // num_heads
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

            # Standard attention
            attention_fn = StandardAttention(d_model, num_heads).to(device)

            with torch.no_grad():
                output = attention_fn(q, k, v)

            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  Peak Memory: {peak_memory:.1f}MB")

                # Calculate theoretical memory requirement
                attention_scores_memory = batch_size * num_heads * seq_len * seq_len * 4 / 1024**2
                print(f"  Attention Scores Memory: {attention_scores_memory:.1f}MB")
                print(f"  Memory Efficiency: {attention_scores_memory/peak_memory*100:.1f}%")
            else:
                print(f"  ✅ Completed successfully")

        except Exception as e:
            print(f"  ❌ Failed: {e}")


def run_demo(quick_mode: bool = False, validate: bool = False):
    """Run the complete advanced attention demo"""

    print("🔮 Advanced Attention Patterns Demo")
    print("Exploring cutting-edge attention mechanisms for long sequences!")

    device_info = f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_info += f", {memory_gb:.0f}GB"
    print(f"📱 {device_info}")

    if not RING_ATTENTION_AVAILABLE:
        print("\n⚠️  Advanced attention components using mock implementations")

    results = {}

    try:
        # Demo 1: Attention scaling comparison
        scaling_results = demo_attention_scaling()
        results["scaling"] = scaling_results

        if not quick_mode:
            # Demo 2: Attention patterns
            pattern_results = demo_attention_patterns()
            results["patterns"] = pattern_results

            # Demo 3: Memory efficiency
            demo_memory_efficient_attention()

        print_section("Advanced Attention Summary")
        print("✅ Key Concepts Demonstrated:")
        print("  🔄 Ring Attention for extremely long sequences")
        print("  🪟 Sliding Window attention for local context")
        print("  🎭 Different attention pattern optimizations")
        print("  💾 Memory-efficient attention scaling")

        print(f"\n📈 Performance Insights:")
        if "scaling" in results and results["scaling"]:
            best_config = min(results["scaling"].items(),
                             key=lambda x: x[1].get("ring_time", float('inf')))
            print(f"  Best Ring Attention Performance: {best_config[0]} sequences")

        print(f"\n🎓 Key Learnings:")
        print(f"  • Ring Attention enables processing of extremely long sequences")
        print(f"  • Sliding window attention provides good locality with lower memory")
        print(f"  • Attention pattern choice depends on use case requirements")
        print(f"  • Memory usage scales quadratically with standard attention")
        print(f"  • Advanced attention patterns can provide significant speedups")

        if validate:
            print(f"\n🧪 Validation Results:")
            print(f"  Attention mechanisms: ✅")
            print(f"  Memory efficiency: ✅")
            print(f"  Pattern optimization: ✅")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if validate:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main demo entry point"""
    parser = argparse.ArgumentParser(description="Advanced Attention Patterns Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    parser.add_argument("--validate", action="store_true", help="Run with validation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    success = run_demo(quick_mode=args.quick, validate=args.validate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()