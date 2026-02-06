"""
Custom CUDA Kernel Demo

Demonstrates the custom kernel system including:
- FlashAttention-3 for memory-efficient attention
- Fused Linear+Activation kernels
- Kernel registry and auto-selection
- NVIDIA backend integration
- Performance comparisons

Usage:
    python demos/custom_kernel_demo.py [--device cuda|cpu] [--quick]
"""

import argparse
import sys

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, 'src')

# Use shared utilities
from demos.shared.utils import print_section

from torchbridge.backends.nvidia.nvidia_backend import NVIDIABackend
from torchbridge.core.config import PrecisionFormat, TorchBridgeConfig


def demo_kernel_registry():
    """Demonstrate kernel registry system."""
    print_section("1. Kernel Registry System")

    # Create config and backend
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    backend = NVIDIABackend(config)

    print(f"Backend initialized: {backend.__class__.__name__}")
    print(f"CUDA available: {backend.is_cuda_available}")
    print(f"Device: {backend.device}")

    # List registered kernels
    registry = backend.kernel_registry
    kernels = registry.list_kernels()

    if kernels:
        print(f"\nRegistered kernels ({len(kernels)} total):")
        for kernel in kernels:
            print(f"  - {kernel.kernel_id} v{kernel.version}")
            print(f"    Type: {kernel.kernel_type.value}")
            print(f"    Backend: {kernel.backend.value}")
            print(f"    Precision: {[p.value for p in kernel.precision_support]}")
            print()
    else:
        print("\n⚠️  No kernels registered (CUDA may not be available)")
        print("   Custom kernels are only available with CUDA GPU")

    return backend


def demo_flash_attention(backend: NVIDIABackend, quick: bool = False):
    """Demonstrate FlashAttention-3 kernel."""
    print_section("2. FlashAttention-3 Demo")

    if not backend.is_cuda_available:
        print("⚠️  FlashAttention demo requires CUDA GPU")
        print("   Running simplified CPU version for illustration...")

    # Parameters
    batch_size = 2
    num_heads = 8
    seq_len = 128 if quick else 512
    head_dim = 64

    print("Attention parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")

    # Create inputs
    device = backend.device
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nInput shapes:")
    print(f"  Q, K, V: {Q.shape}")

    # Get optimal kernel
    kernel_class = backend.get_optimal_attention_kernel(
        head_dim=head_dim,
        precision=PrecisionFormat.FP16 if backend.is_cuda_available else PrecisionFormat.FP32
    )

    if kernel_class:
        print(f"\nSelected kernel: {kernel_class.__name__}")

        # Create and run kernel
        scale = 1.0 / (head_dim ** 0.5)
        fa_layer = kernel_class(scale=scale).to(device)

        with torch.no_grad():
            output = fa_layer(Q, K, V)

        print(f"Output shape: {output.shape}")
        print("✅ FlashAttention-3 executed successfully!")

    else:
        print("\n⚠️  No optimal kernel available")
        print("   Using PyTorch fallback")

        # PyTorch fallback
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        print(f"Output shape: {output.shape}")


def demo_fused_linear_activation(backend: NVIDIABackend):
    """Demonstrate Fused Linear+Activation kernels."""
    print_section("3. Fused Linear+Activation Demo")

    if not backend.is_cuda_available:
        print("⚠️  Fused kernels optimized for CUDA GPU")
        print("   Running CPU version for illustration...")

    try:
        from torchbridge.hardware.gpu.custom_kernels import (
            FusedLinearGELU,
            FusedLinearSiLU,
            create_fused_ffn_layer,
        )

        # Parameters
        batch_size = 32
        seq_len = 64
        in_features = 512
        hidden_features = 2048

        print("FFN parameters:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Input features: {in_features}")
        print(f"  Hidden features: {hidden_features}")

        # Create input
        device = backend.device
        x = torch.randn(batch_size, seq_len, in_features, device=device)
        x_flat = x.view(-1, in_features)  # Flatten for linear layer

        print(f"\nInput shape: {x.shape}")
        print(f"Flattened: {x_flat.shape}")

        # Demo 1: Fused Linear+GELU
        print("\n--- Fused Linear+GELU ---")
        fused_gelu = FusedLinearGELU(in_features, hidden_features).to(device)

        with torch.no_grad():
            output_gelu = fused_gelu(x_flat)

        print(f"Output shape: {output_gelu.shape}")
        print("✅ Fused Linear+GELU executed successfully!")

        # Demo 2: Fused Linear+SiLU
        print("\n--- Fused Linear+SiLU ---")
        fused_silu = FusedLinearSiLU(in_features, hidden_features).to(device)

        with torch.no_grad():
            output_silu = fused_silu(x_flat)

        print(f"Output shape: {output_silu.shape}")
        print("✅ Fused Linear+SiLU executed successfully!")

        # Demo 3: Complete FFN with factory function
        print("\n--- Complete FFN Layer ---")
        ffn = create_fused_ffn_layer(
            in_features=in_features,
            hidden_features=hidden_features,
            activation="gelu"
        ).to(device)

        with torch.no_grad():
            output_ffn = ffn(x_flat)

        print(f"Output shape: {output_ffn.shape}")
        print("✅ Complete FFN layer executed successfully!")

    except ImportError as e:
        print(f"⚠️  Fused kernels not available: {e}")
        print("   Compile CUDA kernels for full functionality")


def demo_model_optimization(backend: NVIDIABackend):
    """Demonstrate automatic model optimization with custom kernels."""
    print_section("4. Automatic Model Optimization")

    # Create a simple model with fusible patterns
    class SimpleFFN(nn.Module):
        def __init__(self, dim=512):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(dim * 4, dim),
                nn.SiLU()
            )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    print("Original model:")
    model = SimpleFFN()
    print(model)

    print("\nApplying custom kernel optimizations...")

    # Prepare model with backend
    model = backend.prepare_model(model)
    model = backend.prepare_model_with_custom_kernels(model)

    print("\nOptimized model:")
    print(model)

    # Test forward pass
    device = backend.device
    x = torch.randn(16, 512, device=device)

    with torch.no_grad():
        output = model(x)

    print("\nForward pass successful!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def demo_validation():
    """Demonstrate kernel validation."""
    print_section("5. Kernel Validation")

    from torchbridge.validation.unified_validator import validate_custom_kernels

    config = TorchBridgeConfig()
    config.kernel.enabled = True

    print("Running kernel validation...")
    result = validate_custom_kernels(config)

    print("\nValidation results:")
    print(f"  Total tests: {result.total_tests}")
    print(f"  Passed: {result.passed}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Failed: {result.failed}")
    print(f"  Success rate: {result.success_rate:.1%}")

    if result.failed == 0:
        print("\n✅ All validation checks passed!")
    else:
        print("\n⚠️  Some validation checks failed")


def main():
    """Run all demos."""
    parser = argparse.ArgumentParser(description="Custom CUDA Kernel Demo")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version with smaller tensors"
    )
    args = parser.parse_args()

    print("#" * 80)
    print("# Custom CUDA Kernel Demo")
    print("# Demonstrates FlashAttention-3, Fused kernels, and Kernel Registry")
    print("#" * 80)

    # Run all demos
    backend = demo_kernel_registry()

    if args.device != "auto":
        print(f"\n⚠️  Device override: {args.device}")
        if args.device == "cuda" and not torch.cuda.is_available():
            print("   CUDA not available, falling back to CPU")

    demo_flash_attention(backend, quick=args.quick)
    demo_fused_linear_activation(backend)
    demo_model_optimization(backend)
    demo_validation()

    print_section("Demo Complete!")
    print("Summary:")
    print("  ✓ Kernel registry system")
    print("  ✓ FlashAttention-3 kernel")
    print("  ✓ Fused Linear+Activation kernels")
    print("  ✓ Automatic model optimization")
    print("  ✓ Kernel validation")

    if backend.is_cuda_available:
        print("\n🚀 All demos completed successfully on CUDA!")
    else:
        print("\n💡 Demos completed on CPU")
        print("   For best performance, run with CUDA GPU")

    print("\n" + "#" * 80 + "\n")


if __name__ == "__main__":
    main()
