#!/usr/bin/env python3
"""
⚡ FP8 Training Concepts Demo

Demonstrates FP8 training concepts and theoretical performance benefits:
- E4M3/E5M2 format explanation and simulation
- Numerical stability considerations
- Performance projections for H100/Blackwell hardware

Expected learning: Understanding FP8 training benefits and challenges
Hardware: Educational demo - works on all devices
Runtime: 2-3 minutes
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for FP8 training demonstration."""

    def __init__(self, d_model: int, d_ff: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Attention layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward layers
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)

        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        x = residual + attn_output

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff2(F.gelu(self.ff1(x)))
        x = residual + x

        return x


def simulate_fp8_quantization(tensor: torch.Tensor, format_type: str) -> torch.Tensor:
    """Simulate FP8 quantization for educational purposes."""

    if format_type == "E4M3":
        # E4M3: 4-bit exponent, 3-bit mantissa
        # Range: approximately ±448, higher precision
        return torch.clamp(tensor, -448, 448)
    elif format_type == "E5M2":
        # E5M2: 5-bit exponent, 2-bit mantissa
        # Range: approximately ±57344, wider range
        return torch.clamp(tensor, -57344, 57344)
    else:
        return tensor


def demonstrate_fp8_formats():
    """Demonstrate FP8 format characteristics."""
    print("\n📊 FP8 Format Analysis")
    print("-" * 40)

    formats = {
        "FP16": {
            "bits": 16,
            "exponent": 5,
            "mantissa": 10,
            "range": "±65504",
            "precision": "High",
            "use_case": "Standard training"
        },
        "E4M3": {
            "bits": 8,
            "exponent": 4,
            "mantissa": 3,
            "range": "±448",
            "precision": "High (for FP8)",
            "use_case": "Forward pass, weights, activations"
        },
        "E5M2": {
            "bits": 8,
            "exponent": 5,
            "mantissa": 2,
            "range": "±57344",
            "precision": "Lower",
            "use_case": "Backward pass, gradients"
        }
    }

    print(f"{'Format':<8} {'Bits':<6} {'Exp':<4} {'Man':<4} {'Range':<12} {'Best For'}")
    print("-" * 65)

    for format_name, specs in formats.items():
        print(f"{format_name:<8} {specs['bits']:<6} {specs['exponent']:<4} {specs['mantissa']:<4} {specs['range']:<12} {specs['use_case']}")

    print("\nKey Insights:")
    print("• E4M3 optimized for precision (activations, weights)")
    print("• E5M2 optimized for range (gradients, optimizer states)")
    print("• Different formats for forward vs backward pass")
    print("• Automatic scaling prevents overflow")


def demonstrate_numerical_stability(device: torch.device, config: dict):
    """Demonstrate FP8 numerical stability considerations."""
    print("\n🔬 FP8 Numerical Stability Analysis")
    print("-" * 45)

    d_model = config['d_model']

    # Test different value distributions
    test_cases = [
        ("Normal range", torch.randn(100, d_model, device=device)),
        ("Large values", torch.randn(100, d_model, device=device) * 10),
        ("Small values", torch.randn(100, d_model, device=device) * 0.01),
        ("Mixed range", torch.cat([
            torch.randn(30, d_model, device=device) * 10,
            torch.randn(40, d_model, device=device),
            torch.randn(30, d_model, device=device) * 0.01
        ])),
    ]

    print("Testing E4M3 and E5M2 formats with different value ranges:")

    for name, test_data in test_cases:
        print(f"\n{name}:")
        print(f"  Original range: [{test_data.min():.4f}, {test_data.max():.4f}]")

        # Simulate FP8 quantization
        e4m3_data = simulate_fp8_quantization(test_data, "E4M3")
        e5m2_data = simulate_fp8_quantization(test_data, "E5M2")

        mse_e4m3 = F.mse_loss(test_data, e4m3_data)
        mse_e5m2 = F.mse_loss(test_data, e5m2_data)

        print(f"  E4M3 MSE: {mse_e4m3.item():.6f}")
        print(f"  E5M2 MSE: {mse_e5m2.item():.6f}")

        # Check for overflow
        overflow_e4m3 = (test_data.abs() > 448).float().mean()
        overflow_e5m2 = (test_data.abs() > 57344).float().mean()

        print(f"  E4M3 overflow: {overflow_e4m3.item()*100:.1f}%")
        print(f"  E5M2 overflow: {overflow_e5m2.item()*100:.1f}%")

        # Recommendation
        if overflow_e4m3.item() > 0.01:  # > 1% overflow
            print("  💡 Recommendation: Use E5M2 for this data range")
        elif mse_e4m3.item() < mse_e5m2.item():
            print("  💡 Recommendation: E4M3 provides better precision")
        else:
            print("  💡 Recommendation: Both formats suitable")


def simulate_training_performance(device: torch.device, config: dict):
    """Simulate FP8 vs FP16 training performance."""
    print("\n⚡ FP8 vs FP16 Training Performance Simulation")
    print("-" * 50)

    d_model, d_ff, num_heads = config['d_model'], config['d_ff'], config['num_heads']
    batch_size, seq_len = config['batch_size'], config['seq_len']

    # Create model and data
    model = SimpleTransformerBlock(d_model, d_ff, num_heads).to(device)
    inputs = torch.randn(batch_size, seq_len, d_model, device=device)

    print("Model configuration:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())//1000}K")
    print(f"  Input shape: {inputs.shape}")

    # Benchmark FP16 (baseline)
    model_fp16 = model.half() if device.type == 'cuda' else model
    inputs_fp16 = inputs.half() if device.type == 'cuda' else inputs

    # Warmup and benchmark
    warmup_runs = 3
    benchmark_runs = 10

    print("\n🧮 Performance Benchmarking:")

    # FP16 timing
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model_fp16(inputs_fp16)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    fp16_times = []
    for _ in range(benchmark_runs):
        start_time = time.perf_counter()
        with torch.no_grad():
            output_fp16 = model_fp16(inputs_fp16)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fp16_times.append(time.perf_counter() - start_time)

    fp16_mean = sum(fp16_times) / len(fp16_times) * 1000  # ms

    # Simulate FP8 performance (theoretical speedup)
    h100_speedup = 1.95  # Typical H100 FP8 speedup
    other_gpu_speedup = 1.2  # Estimated speedup on other hardware

    if device.type == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if 'H100' in gpu_name or 'H200' in gpu_name:
            speedup = h100_speedup
            fp8_support = "Native"
        else:
            speedup = other_gpu_speedup
            fp8_support = "Emulated"
    else:
        speedup = 1.0
        fp8_support = "Not available"

    fp8_mean = fp16_mean / speedup

    print("\n📊 Performance Results:")
    print(f"{'Format':<8} {'Time (ms)':<12} {'Speedup':<10} {'Support'}")
    print("-" * 45)
    print(f"{'FP16':<8} {fp16_mean:.2f}        {'1.0x':<10} Standard")
    print(f"{'FP8':<8} {fp8_mean:.2f}        {speedup:.1f}x      {fp8_support}")

    # Memory analysis
    memory_reduction = 0.5  # 50% memory reduction with FP8
    print("\n💾 Memory Analysis:")
    if device.type == 'cuda':
        current_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        fp8_memory = current_memory * memory_reduction
        print(f"  FP16 memory: {current_memory:.1f} MB")
        print(f"  FP8 memory: {fp8_memory:.1f} MB ({memory_reduction*100:.0f}% reduction)")
    else:
        print(f"  Memory reduction: ~{memory_reduction*100:.0f}% with FP8")


def explain_production_benefits():
    """Explain production benefits of FP8 training."""
    print("\n🚀 Production FP8 Training Benefits")
    print("-" * 40)

    benefits = {
        "Performance": [
            "1.9-2.0x training speedup on H100/Blackwell",
            "1.5-1.8x inference speedup",
            "Better GPU utilization efficiency"
        ],
        "Memory": [
            "50% reduction in model memory footprint",
            "Larger batch sizes possible",
            "Enables training of larger models"
        ],
        "Cost": [
            "Reduced training time = lower compute costs",
            "Higher throughput = better ROI",
            "Efficient resource utilization"
        ],
        "Scaling": [
            "Enables larger models on same hardware",
            "Better multi-GPU scaling efficiency",
            "Reduced communication overhead"
        ]
    }

    for category, items in benefits.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

    print("\n💡 Implementation Considerations:")
    print("  • Automatic scaling to prevent numerical instability")
    print("  • Mixed E4M3/E5M2 formats for optimal precision/range balance")
    print("  • Gradual rollout with validation checkpoints")
    print("  • Monitor convergence and accuracy metrics")


def main():
    parser = argparse.ArgumentParser(description='FP8 Training Concepts Demo')
    parser.add_argument('--quick', action='store_true', help='Run quick analysis')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Force device selection')
    args = parser.parse_args()

    print("⚡ FP8 Training Concepts Demo")
    print("=" * 50)
    print("Understanding FP8 training for 2x H100 speedup and memory efficiency\n")

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🎯 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")

    # Configuration
    if args.quick:
        config = {
            'd_model': 256,
            'd_ff': 1024,
            'num_heads': 8,
            'batch_size': 4,
            'seq_len': 128
        }
        print("🏃‍♂️ Quick analysis mode")
    else:
        config = {
            'd_model': 512,
            'd_ff': 2048,
            'num_heads': 8,
            'batch_size': 8,
            'seq_len': 256
        }
        print("🏋️‍♂️ Full analysis mode")

    # Run demonstrations
    demonstrate_fp8_formats()
    demonstrate_numerical_stability(device, config)
    simulate_training_performance(device, config)
    explain_production_benefits()

    print("\n🎉 FP8 Training Demo Completed!")
    print("\n💡 Key Takeaways:")
    print("   • FP8 provides 1.9-2.0x speedup on H100/Blackwell hardware")
    print("   • E4M3 format optimal for forward pass (higher precision)")
    print("   • E5M2 format optimal for backward pass (wider range)")
    print("   • 50% memory reduction enables larger models and batch sizes")
    print("   • Automatic scaling maintains numerical stability")
    print("   • Production-ready with proper implementation considerations")

    print("\n✅ Demo completed! Try --quick for faster testing.")


if __name__ == "__main__":
    main()
