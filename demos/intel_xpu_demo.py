#!/usr/bin/env python3
"""
Intel XPU Backend Demo

Demonstrates the Intel XPU backend capabilities including:
- Device detection and information
- Model preparation and optimization
- Memory management
- IPEX integration (when available)

Usage:
    python demos/intel_xpu_demo.py
"""

import time

import torch
import torch.nn as nn


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def demo_device_detection():
    """Demonstrate Intel XPU device detection."""
    print_section("1. Device Detection")

    from torchbridge.backends.intel import (
        IPEX_AVAILABLE,
        XPU_AVAILABLE,
        get_ipex_version,
        get_xpu_device_count,
        is_ipex_available,
        is_xpu_available,
    )

    print(f"XPU Available: {XPU_AVAILABLE}")
    print(f"IPEX Available: {IPEX_AVAILABLE}")
    print(f"IPEX Version: {get_ipex_version() or 'Not installed'}")
    print(f"XPU Device Count: {get_xpu_device_count()}")

    # Check using functions
    print(f"\nUsing is_xpu_available(): {is_xpu_available()}")
    print(f"Using is_ipex_available(): {is_ipex_available()}")


def demo_backend_initialization():
    """Demonstrate Intel backend initialization."""
    print_section("2. Backend Initialization")

    from torchbridge.backends.intel import IntelBackend
    from torchbridge.core.config import TorchBridgeConfig

    # Initialize without config
    backend = IntelBackend()
    print(f"Backend device: {backend.device}")
    print(f"XPU available: {backend.is_xpu_available}")
    print(f"Device name: {backend.device_name or 'N/A'}")
    print(f"Device type: {backend.device_type or 'N/A'}")

    # Initialize with config
    config = TorchBridgeConfig()
    backend_with_config = IntelBackend(config=config)
    print(f"\nBackend with config: {backend_with_config.device}")

    return backend


def demo_device_info(backend):
    """Demonstrate device information retrieval."""
    print_section("3. Device Information")

    info = backend.get_device_info_dict()
    for key, value in info.items():
        print(f"  {key}: {value}")


def demo_memory_management(backend):
    """Demonstrate memory management."""
    print_section("4. Memory Management")

    # Get memory statistics
    stats = backend.get_memory_stats()
    print("Memory Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Print memory summary
    print(f"\nMemory Summary:\n{backend.get_memory_summary()}")


def demo_model_preparation():
    """Demonstrate model preparation."""
    print_section("5. Model Preparation")

    from torchbridge.backends.intel import IntelBackend

    # Create a simple model
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=256, num_heads=4, d_ff=512):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            # FFN
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x

    # Create model
    model = SimpleTransformer()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize backend
    backend = IntelBackend()

    # Prepare model
    prepared_model = backend.prepare_model(model)
    print(f"Model device after preparation: {next(prepared_model.parameters()).device}")

    return prepared_model


def demo_inference_optimization():
    """Demonstrate inference optimization."""
    print_section("6. Inference Optimization")

    from torchbridge.backends.intel import XPU_AVAILABLE, IntelBackend

    # Create model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.GELU(),
        nn.Linear(1024, 512),
    )

    backend = IntelBackend()

    # Optimize for inference
    print("Optimizing model for inference...")
    optimized_model = backend.optimize_for_inference(model)

    # Test inference
    x = torch.randn(32, 512)
    if XPU_AVAILABLE:
        x = x.to("xpu")

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = optimized_model(x)

        # Benchmark
        start = time.perf_counter()
        num_iters = 100
        for _ in range(num_iters):
            _ = optimized_model(x)
        elapsed = time.perf_counter() - start

    print(f"Inference time: {elapsed/num_iters*1000:.3f} ms per iteration")
    print(f"Throughput: {num_iters*32/elapsed:.0f} samples/sec")


def demo_optimizer():
    """Demonstrate Intel optimizer."""
    print_section("7. Intel Optimizer")

    from torchbridge.backends.intel import (
        IntelOptimizationLevel,
        IntelOptimizer,
    )

    # Create model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )

    # Test different optimization levels
    for level in [IntelOptimizationLevel.O0, IntelOptimizationLevel.O1, IntelOptimizationLevel.O2]:
        print(f"\nOptimization Level: {level.value}")
        optimizer = IntelOptimizer(optimization_level=level)
        _, result = optimizer.optimize(model)

        print(f"  Success: {result.success}")
        print(f"  Optimizations applied: {len(result.optimizations_applied)}")
        if result.optimizations_applied:
            for opt in result.optimizations_applied[:3]:
                print(f"    - {opt}")
        if result.warnings:
            print(f"  Warnings: {len(result.warnings)}")


def demo_kernel_optimizer():
    """Demonstrate kernel optimizer."""
    print_section("8. Kernel Optimizer")

    from torchbridge.backends.intel import IntelKernelOptimizer

    optimizer = IntelKernelOptimizer(device_type="auto")

    # GEMM configuration
    print("GEMM Configuration (1024x1024x1024, FP32):")
    gemm_config = optimizer.get_optimal_gemm_config(1024, 1024, 1024)
    for key, value in gemm_config.items():
        print(f"  {key}: {value}")

    # GEMM with BF16
    print("\nGEMM Configuration (1024x1024x1024, BF16):")
    gemm_bf16 = optimizer.get_optimal_gemm_config(1024, 1024, 1024, dtype=torch.bfloat16)
    for key, value in gemm_bf16.items():
        print(f"  {key}: {value}")

    # Convolution configuration
    print("\nConvolution Configuration (64->128, 3x3):")
    conv_config = optimizer.get_optimal_conv_config(64, 128, (3, 3))
    for key, value in conv_config.items():
        print(f"  {key}: {value}")

    # Attention configuration
    print("\nAttention Configuration (seq=2048, head_dim=64):")
    attn_config = optimizer.get_optimal_attention_config(2048, 64, 8)
    for key, value in attn_config.items():
        print(f"  {key}: {value}")

    print("\nAttention Configuration (seq=8192, head_dim=64):")
    attn_long = optimizer.get_optimal_attention_config(8192, 64, 8)
    for key, value in attn_long.items():
        print(f"  {key}: {value}")


def demo_config():
    """Demonstrate Intel configuration."""
    print_section("9. Configuration")

    from torchbridge.core.config import (
        IntelArchitecture,
        IntelConfig,
        TorchBridgeConfig,
    )

    # Intel architecture options
    print("Intel Architecture Options:")
    for arch in IntelArchitecture:
        print(f"  - {arch.name}: {arch.value}")

    # Intel config
    print("\nDefault Intel Config:")
    config = IntelConfig()
    print(f"  Architecture: {config.architecture}")
    print(f"  IPEX Enabled: {config.ipex_enabled}")
    print(f"  oneDNN Enabled: {config.onednn_enabled}")
    print(f"  Mixed Precision: {config.enable_mixed_precision}")
    print(f"  Allow BF16: {config.allow_bf16}")

    # PVC config
    print("\nPonte Vecchio (PVC) Config:")
    pvc_config = IntelConfig(architecture=IntelArchitecture.PVC)
    print(f"  AMX Enabled: {pvc_config.enable_amx}")
    print(f"  Allow BF16: {pvc_config.allow_bf16}")

    # Full TorchBridge config
    print("\nFull TorchBridge Config:")
    full_config = TorchBridgeConfig()
    print(f"  Device: {full_config.device}")
    print(f"  Hardware Backend: {full_config.hardware.backend}")
    print(f"  Intel Enabled: {full_config.hardware.intel.enabled}")


def demo_benchmark():
    """Run a simple benchmark."""
    print_section("10. Simple Benchmark")

    from torchbridge.backends.intel import IntelBackend

    # Create larger model for benchmarking
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
        nn.GELU(),
        nn.Linear(2048, 1024),
    )

    backend = IntelBackend()
    model = backend.prepare_model(model)

    # Create input
    batch_sizes = [1, 8, 32, 64]
    results = []

    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Throughput':<12}")
    print("-" * 36)

    for bs in batch_sizes:
        x = torch.randn(bs, 1024)
        if backend.is_xpu_available:
            x = x.to("xpu")

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(x)

            # Benchmark
            backend.synchronize()
            start = time.perf_counter()
            num_iters = 50
            for _ in range(num_iters):
                _ = model(x)
            backend.synchronize()
            elapsed = time.perf_counter() - start

        time_ms = elapsed / num_iters * 1000
        throughput = num_iters * bs / elapsed

        print(f"{bs:<12} {time_ms:<12.3f} {throughput:<12.0f}")
        results.append((bs, time_ms, throughput))

    return results


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("       Intel XPU Backend Demo")
    print("="*60)

    try:
        # Run demos
        demo_device_detection()
        backend = demo_backend_initialization()
        demo_device_info(backend)
        demo_memory_management(backend)
        demo_model_preparation()
        demo_inference_optimization()
        demo_optimizer()
        demo_kernel_optimizer()
        demo_config()
        demo_benchmark()

        print_section("Demo Complete!")
        print("All Intel XPU backend features demonstrated successfully.")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
