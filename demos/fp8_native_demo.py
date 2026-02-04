#!/usr/bin/env python3
"""
Native FP8 Demo (v0.4.5)

Demonstrates full FP8 capabilities including:
- Native PyTorch FP8 types (float8_e4m3fn, float8_e5m2)
- Real FP8 quantization and dequantization
- FP8 linear layers with actual FP8 GEMM operations
- FP8 inference engine for model serving
- Performance benchmarking FP8 vs standard layers

Features:
- Native FP8 when PyTorch 2.1+ available
- Simulated FP8 fallback for older PyTorch
- Dynamic scaling for numerical stability
- Memory savings analysis

Hardware:
- Best performance on H100/Blackwell with native FP8
- Works on all hardware with simulation fallback
"""

import sys
import os
import time
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared utilities
try:
    from shared.utils import print_section
except ImportError:
    def print_section(title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)

from torchbridge.precision import (
    # Native FP8
    FP8Dtype,
    NativeFP8Linear,
    FP8InferenceEngine,
    is_fp8_available,
    get_fp8_info,
    compute_fp8_scale,
    quantize_to_fp8,
    dequantize_from_fp8,
    convert_model_to_native_fp8,
    benchmark_fp8_layer,
    FP8_NATIVE_AVAILABLE,
    FP8_DTYPES_AVAILABLE,
    # Training
    FP8TrainingEngine,
    FP8Config,
    FP8Format,
    create_fp8_trainer,
)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for FP8 demonstration"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Self-attention
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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


def demo_fp8_availability():
    """Demo: Check FP8 availability and capabilities"""
    print_section("FP8 Availability Check")

    info = get_fp8_info()

    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"Native FP8 Available: {info['fp8_native_available']}")
    print(f"FP8 scaled_mm Available: {info['fp8_scaled_mm_available']}")

    if info['supported_formats']:
        print(f"\nSupported FP8 Formats:")
        for fmt in info['supported_formats']:
            print(f"  - {fmt}")

    print(f"\nFP8 Format Specifications:")
    print(f"  E4M3FN max value: {info['e4m3_max_value']}")
    print(f"  E5M2 max value: {info['e5m2_max_value']}")

    print(f"\nRecommended Usage:")
    for fmt, usage in info['recommended_use'].items():
        print(f"  {fmt}: {usage}")

    return info['fp8_native_available']


def demo_fp8_quantization():
    """Demo: FP8 quantization and dequantization"""
    print_section("FP8 Quantization")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create test tensor
    x = torch.randn(4, 256, device=device)
    print(f"\nOriginal tensor shape: {x.shape}")
    print(f"Original range: [{x.min():.4f}, {x.max():.4f}]")

    # Test E4M3 quantization
    scale_e4m3 = compute_fp8_scale(x, FP8Dtype.E4M3)
    quantized_e4m3, _ = quantize_to_fp8(x, scale_e4m3, FP8Dtype.E4M3)
    dequantized_e4m3 = dequantize_from_fp8(quantized_e4m3, scale_e4m3)

    mse_e4m3 = F.mse_loss(x, dequantized_e4m3)
    print(f"\nE4M3 Quantization:")
    print(f"  Scale: {scale_e4m3.item():.4f}")
    print(f"  MSE: {mse_e4m3.item():.6f}")
    print(f"  Relative error: {(mse_e4m3.sqrt() / x.std()).item()*100:.4f}%")

    # Test E5M2 quantization
    scale_e5m2 = compute_fp8_scale(x, FP8Dtype.E5M2)
    quantized_e5m2, _ = quantize_to_fp8(x, scale_e5m2, FP8Dtype.E5M2)
    dequantized_e5m2 = dequantize_from_fp8(quantized_e5m2, scale_e5m2)

    mse_e5m2 = F.mse_loss(x, dequantized_e5m2)
    print(f"\nE5M2 Quantization:")
    print(f"  Scale: {scale_e5m2.item():.4f}")
    print(f"  MSE: {mse_e5m2.item():.6f}")
    print(f"  Relative error: {(mse_e5m2.sqrt() / x.std()).item()*100:.4f}%")

    print(f"\nConclusion: E4M3 provides {mse_e5m2.item()/mse_e4m3.item():.1f}x lower error than E5M2")

    return True


def demo_native_fp8_linear():
    """Demo: Native FP8 Linear Layer"""
    print_section("Native FP8 Linear Layer")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create FP8 layer
    layer = NativeFP8Linear(
        in_features=512,
        out_features=256,
        weight_format=FP8Dtype.E4M3,
        activation_format=FP8Dtype.E4M3,
        device=device
    )

    print(f"\nLayer configuration:")
    print(f"  {layer}")

    # Get FP8 info
    info = layer.get_fp8_info()
    print(f"\nFP8 Layer Info:")
    print(f"  Weight format: {info['weight_format']}")
    print(f"  Activation format: {info['activation_format']}")
    print(f"  Weight scale: {info['weight_scale']:.4f}")
    print(f"  FP8 native: {info['fp8_native']}")
    print(f"  FP8 active: {info['fp8_active']}")

    # Forward pass
    x = torch.randn(4, 512, device=device)
    output = layer(x)

    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output finite: {torch.isfinite(output).all()}")

    # Gradient test
    x.requires_grad = True
    output = layer(x)
    loss = output.sum()
    loss.backward()

    print(f"  Gradient flow: OK (grad shape: {x.grad.shape})")

    return True


def demo_fp8_inference_engine():
    """Demo: FP8 Inference Engine"""
    print_section("FP8 Inference Engine")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = SimpleTransformerBlock(d_model=256, num_heads=4, d_ff=1024).to(device)

    # Count original linear layers
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"\nOriginal model: {linear_count} Linear layers")

    # Create FP8 inference engine
    engine = FP8InferenceEngine(
        model,
        weight_format=FP8Dtype.E4M3,
        activation_format=FP8Dtype.E4M3
    )

    # Prepare for inference
    engine.prepare(device)

    # Get memory savings
    savings = engine.get_memory_savings()
    print(f"\nMemory Analysis:")
    print(f"  FP32 memory: {savings['fp32_memory_mb']:.2f} MB")
    print(f"  FP8 memory: {savings['fp8_memory_mb']:.2f} MB")
    print(f"  Savings: {savings['savings_percent']:.1f}%")
    print(f"  FP8 layers: {savings['fp8_layers_count']}")

    # Run inference
    x = torch.randn(2, 32, 256, device=device)
    output = engine.infer(x)

    print(f"\nInference:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output finite: {torch.isfinite(output).all()}")

    return True


def demo_model_conversion():
    """Demo: Convert model to native FP8"""
    print_section("Model Conversion to Native FP8")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = SimpleTransformerBlock(d_model=256, num_heads=4, d_ff=1024).to(device)

    # Convert to FP8
    print("Converting model to native FP8...")
    fp8_model = convert_model_to_native_fp8(
        model,
        weight_format=FP8Dtype.E4M3,
        activation_format=FP8Dtype.E4M3
    )

    # Count FP8 layers
    fp8_count = sum(1 for m in fp8_model.modules() if isinstance(m, NativeFP8Linear))
    print(f"Converted {fp8_count} layers to NativeFP8Linear")

    # Test forward pass
    x = torch.randn(2, 32, 256, device=device)
    output = fp8_model(x)

    print(f"\nForward pass after conversion:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Output finite: {torch.isfinite(output).all()}")

    return True


def demo_fp8_training():
    """Demo: FP8 Training with dynamic scaling"""
    print_section("FP8 Training")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = SimpleTransformerBlock(d_model=128, num_heads=4, d_ff=512).to(device)

    # Create FP8 config
    config = FP8Config(
        forward_format=FP8Format.E4M3,
        backward_format=FP8Format.E5M2,
        scaling_strategy="dynamic",
        initial_scale=1024.0,
        use_te_linear=False  # Use our native implementation
    )

    # Create trainer
    trainer = create_fp8_trainer(model, device, **config.__dict__)
    trainer.setup_fp8_training()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training data
    batch_size, seq_len = 4, 32
    num_steps = 10

    print(f"\nTraining for {num_steps} steps...")
    losses = []

    for step in range(num_steps):
        # Generate batch
        x = torch.randn(batch_size, seq_len, 128, device=device)
        targets = torch.randint(0, 128, (batch_size,), device=device)

        # Forward
        optimizer.zero_grad()
        outputs = model(x)
        loss = F.cross_entropy(outputs.mean(dim=1), targets)

        # Backward
        loss.backward()

        # Optimizer step
        success = trainer.optimizer_step(optimizer)

        if success:
            losses.append(loss.item())

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}, success={success}")

    # Get stats
    stats = trainer.get_training_statistics()
    print(f"\nTraining Statistics:")
    print(f"  Steps: {stats['steps']}")
    print(f"  Overflows: {stats['overflows']}")
    print(f"  FP8 enabled: {stats['fp8_enabled']}")
    print(f"  Final scale: {stats['scale_info']['scale']:.2f}")

    return True


def demo_benchmark():
    """Demo: Benchmark FP8 vs standard layers"""
    print_section("FP8 Performance Benchmark")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Benchmark configurations
    configs = [
        (512, 512, 32),
        (1024, 1024, 32),
        (2048, 2048, 16),
    ]

    print(f"\nBenchmarking FP8 vs Standard Linear layers:")
    print(f"{'In':>6} {'Out':>6} {'Batch':>6} {'FP8 (ms)':>10} {'Std (ms)':>10} {'Speedup':>8}")
    print("-" * 50)

    for in_feat, out_feat, batch in configs:
        results = benchmark_fp8_layer(
            in_features=in_feat,
            out_features=out_feat,
            batch_size=batch,
            num_iterations=100,
            device=device
        )

        print(f"{in_feat:>6} {out_feat:>6} {batch:>6} "
              f"{results['fp8_time_ms']:>10.3f} {results['standard_time_ms']:>10.3f} "
              f"{results['speedup']:>7.2f}x")

    print(f"\nNote: FP8 native = {FP8_DTYPES_AVAILABLE}")
    if not FP8_DTYPES_AVAILABLE:
        print("  (Speedups reflect simulation overhead - native FP8 would be faster)")

    return True


def demo_numerical_stability():
    """Demo: FP8 numerical stability analysis"""
    print_section("FP8 Numerical Stability")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test with different value ranges
    test_cases = [
        ("Normal range", 1.0),
        ("Large values", 10.0),
        ("Small values", 0.01),
        ("Very large", 100.0),
    ]

    print(f"Testing FP8 quantization stability:")
    print(f"{'Case':<15} {'Scale':>8} {'E4M3 Error':>12} {'E5M2 Error':>12} {'Recommended':<10}")
    print("-" * 60)

    for name, magnitude in test_cases:
        x = torch.randn(100, 256, device=device) * magnitude

        # E4M3
        scale_e4m3 = compute_fp8_scale(x, FP8Dtype.E4M3)
        quant_e4m3, _ = quantize_to_fp8(x, scale_e4m3, FP8Dtype.E4M3)
        dequant_e4m3 = dequantize_from_fp8(quant_e4m3, scale_e4m3)
        error_e4m3 = F.mse_loss(x, dequant_e4m3).item()

        # E5M2
        scale_e5m2 = compute_fp8_scale(x, FP8Dtype.E5M2)
        quant_e5m2, _ = quantize_to_fp8(x, scale_e5m2, FP8Dtype.E5M2)
        dequant_e5m2 = dequantize_from_fp8(quant_e5m2, scale_e5m2)
        error_e5m2 = F.mse_loss(x, dequant_e5m2).item()

        recommended = "E4M3" if error_e4m3 < error_e5m2 else "E5M2"

        print(f"{name:<15} {magnitude:>8.2f} {error_e4m3:>12.6f} {error_e5m2:>12.6f} {recommended:<10}")

    print(f"\nConclusion: E4M3 is generally preferred for activations (higher precision)")
    print(f"            E5M2 is better for gradients (wider dynamic range)")

    return True


def main():
    parser = argparse.ArgumentParser(description='Native FP8 Demo')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Force device')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Native FP8 Demo - TorchBridge v0.4.5")
    print("  Full FP8 with Native PyTorch Types")
    print("=" * 60)

    results = {}

    # Run demos
    demos = [
        ('availability', demo_fp8_availability),
        ('quantization', demo_fp8_quantization),
        ('native_linear', demo_native_fp8_linear),
        ('inference_engine', demo_fp8_inference_engine),
        ('model_conversion', demo_model_conversion),
        ('training', demo_fp8_training),
        ('benchmark', demo_benchmark),
        ('numerical_stability', demo_numerical_stability),
    ]

    if args.quick:
        demos = demos[:4]  # Only first 4 demos in quick mode

    for name, demo_fn in demos:
        try:
            success = demo_fn()
            results[name] = 'PASSED' if success else 'FAILED'
        except Exception as e:
            results[name] = f'ERROR: {e}'
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_section("Demo Summary")
    passed = sum(1 for v in results.values() if v == 'PASSED' or v is True)
    total = len(results)

    print(f"Results: {passed}/{total} demos passed")
    print()

    for name, result in results.items():
        status = "PASS" if result == 'PASSED' or result is True else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nFP8 Support Status:")
    print(f"  Native FP8 types: {'Available' if FP8_DTYPES_AVAILABLE else 'Simulated'}")
    print(f"  PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 9:
            print(f"  Hardware FP8: Supported (Hopper/Blackwell)")
        else:
            print(f"  Hardware FP8: Limited (Compute {capability[0]}.{capability[1]})")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
