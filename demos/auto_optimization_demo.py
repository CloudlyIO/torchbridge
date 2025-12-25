"""
Auto-Optimization Demo - Stage 3A: Intelligent Optimization Selection

Demonstrates automatic hardware detection and optimization selection
without requiring users to manually configure backend settings.

This demo shows:
1. Automatic hardware detection
2. Intelligent backend selection (NVIDIA/TPU/CPU)
3. Automatic optimization level selection
4. Getting optimization recommendations
5. Simple one-line model optimization
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any

from kernel_pytorch.core.management import get_manager
from kernel_pytorch.core.hardware_detector import detect_hardware, get_optimal_backend


# Simple test models
class SimpleLinearModel(nn.Module):
    """Simple linear model for testing."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""
    def __init__(self, d_model: int = 512, nhead: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_hardware_info(profile):
    """Print hardware profile information."""
    print(f"Hardware Type: {profile.hardware_type.value}")
    print(f"Device Name: {profile.device_name}")
    print(f"Device Count: {profile.device_count}")

    if profile.nvidia_architecture:
        print(f"NVIDIA Architecture: {profile.nvidia_architecture.value}")
    if profile.tpu_version:
        print(f"TPU Version: {profile.tpu_version.value}")
    if profile.compute_capability:
        print(f"Compute Capability: {profile.compute_capability[0]}.{profile.compute_capability[1]}")
    if profile.total_memory_gb > 0:
        print(f"Total Memory: {profile.total_memory_gb:.2f} GB")

    print(f"\nCapabilities:")
    if profile.capabilities:
        for cap in profile.capabilities:
            print(f"  ✓ {cap.value}")
    else:
        print("  None (CPU fallback)")


def print_recommendations(recommendations: Dict[str, Any]):
    """Print optimization recommendations."""
    print(f"Detected Hardware: {recommendations['hardware_type']}")
    print(f"Device: {recommendations['device_name']}")
    print(f"Recommended Backend: {recommendations['backend']}")
    print(f"Recommended Optimization Level: {recommendations['optimization_level']}")

    print(f"\nCapabilities:")
    if recommendations['capabilities']:
        for cap in recommendations['capabilities']:
            print(f"  ✓ {cap}")
    else:
        print("  None (CPU fallback)")

    if recommendations['optimizations']:
        print(f"\nAvailable Optimizations:")
        for opt in recommendations['optimizations']:
            print(f"  • {opt['type']}: {opt['benefit']}")
            print(f"    Requirement: {opt['requirement']}")


def benchmark_model(model: nn.Module, inputs: torch.Tensor, num_iterations: int = 50) -> float:
    """Benchmark model inference time."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(inputs)

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(inputs)
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    return avg_time_ms


def demo_1_hardware_detection():
    """Demo 1: Automatic Hardware Detection."""
    print_section("Demo 1: Automatic Hardware Detection")

    print("Detecting available hardware...\n")

    # Detect hardware
    profile = detect_hardware()
    print_hardware_info(profile)

    # Get optimal backend
    optimal_backend = get_optimal_backend()
    print(f"\nOptimal Backend: {optimal_backend}")

    print("\n✅ Hardware detection complete!")


def demo_2_optimization_recommendations():
    """Demo 2: Get Optimization Recommendations."""
    print_section("Demo 2: Optimization Recommendations")

    print("Getting optimization recommendations...\n")

    # Get manager and recommendations
    manager = get_manager()
    recommendations = manager.get_optimization_recommendations()

    print_recommendations(recommendations)

    print("\n✅ Recommendations generated!")


def demo_3_auto_optimize_simple():
    """Demo 3: Simple Auto-Optimization."""
    print_section("Demo 3: Simple Auto-Optimization (One-Line)")

    print("Creating a simple model...")
    model = SimpleLinearModel(input_dim=512, hidden_dim=1024, output_dim=512)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nAuto-optimizing with one line of code:")
    print(">>> optimized_model = get_manager().auto_optimize(model)")

    # Auto-optimize (one line!)
    manager = get_manager()
    optimized_model = manager.auto_optimize(model)

    print(f"\n✅ Model optimized!")
    print(f"Backend used: {manager.get_hardware_profile().hardware_type.value}")

    # Test the optimized model
    print("\nTesting optimized model...")
    sample_input = torch.randn(8, 512)
    with torch.no_grad():
        output = optimized_model(sample_input)
    print(f"Output shape: {output.shape}")
    print("✅ Model works correctly!")


def demo_4_auto_optimize_with_options():
    """Demo 4: Auto-Optimization with Custom Options."""
    print_section("Demo 4: Auto-Optimization with Custom Options")

    print("Creating a transformer model...")
    model = SimpleTransformerBlock(d_model=512, nhead=8)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nAuto-optimizing with custom options:")

    # Create sample inputs
    sample_inputs = torch.randn(8, 32, 512)  # batch=8, seq_len=32, d_model=512

    # Auto-optimize for inference with aggressive optimization
    manager = get_manager()
    optimized_model = manager.auto_optimize(
        model,
        sample_inputs=sample_inputs,
        optimization_level='aggressive',
        for_inference=True
    )

    print(f"✅ Model optimized for inference!")
    print(f"Optimization level: aggressive")
    print(f"Training mode: {optimized_model.training}")

    # Test the optimized model
    print("\nTesting optimized model...")
    with torch.no_grad():
        output = optimized_model(sample_inputs)
    print(f"Output shape: {output.shape}")
    print("✅ Model works correctly!")


def demo_5_performance_comparison():
    """Demo 5: Performance Comparison."""
    print_section("Demo 5: Performance Comparison")

    print("Creating a model for benchmarking...")
    original_model = SimpleLinearModel(input_dim=512, hidden_dim=1024, output_dim=512)
    sample_inputs = torch.randn(32, 512)

    print("Benchmarking original model...")
    original_time = benchmark_model(original_model, sample_inputs, num_iterations=100)
    print(f"Original model: {original_time:.3f} ms/iteration")

    print("\nAuto-optimizing model...")
    manager = get_manager()
    optimized_model = manager.auto_optimize(
        original_model,
        sample_inputs=sample_inputs,
        for_inference=True
    )

    print("Benchmarking optimized model...")
    optimized_time = benchmark_model(optimized_model, sample_inputs, num_iterations=100)
    print(f"Optimized model: {optimized_time:.3f} ms/iteration")

    speedup = original_time / optimized_time if optimized_time > 0 else 1.0
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup > 1.1:
        print("✅ Optimization improved performance!")
    else:
        print("ℹ️  Performance similar (expected on CPU)")


def demo_6_multiple_models():
    """Demo 6: Auto-Optimizing Multiple Models."""
    print_section("Demo 6: Auto-Optimizing Multiple Models")

    print("Creating multiple models...")

    models = {
        "Linear (small)": SimpleLinearModel(256, 512, 256),
        "Linear (medium)": SimpleLinearModel(512, 1024, 512),
        "Transformer": SimpleTransformerBlock(512, 8),
    }

    manager = get_manager()
    optimized_models = {}

    print("\nAuto-optimizing all models...")
    for name, model in models.items():
        print(f"\n  Optimizing {name}...")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {param_count:,}")

        optimized_model = manager.auto_optimize(model)
        optimized_models[name] = optimized_model

        print(f"    ✅ Optimized!")

    print(f"\n✅ All {len(models)} models optimized successfully!")
    print(f"Backend used: {manager.get_hardware_profile().hardware_type.value}")


def demo_7_inference_mode():
    """Demo 7: Inference-Specific Optimization."""
    print_section("Demo 7: Inference-Specific Optimization")

    print("Creating a model for inference deployment...")
    model = SimpleTransformerBlock(d_model=256, nhead=4)
    sample_inputs = torch.randn(1, 16, 256)  # Small batch for inference

    print("\nOptimizing for inference:")
    print("  • Disables gradient computation")
    print("  • Sets model to eval mode")
    print("  • Applies inference-specific optimizations")

    manager = get_manager()
    optimized_model = manager.auto_optimize(
        model,
        sample_inputs=sample_inputs,
        for_inference=True
    )

    print(f"\n✅ Model optimized for inference!")
    print(f"Training mode: {optimized_model.training}")

    # Verify inference works
    print("\nRunning inference...")
    with torch.no_grad():
        output = optimized_model(sample_inputs)
    print(f"Output shape: {output.shape}")
    print("✅ Inference successful!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  Auto-Optimization Demo - Stage 3A")
    print("  Intelligent Hardware Detection & Optimization Selection")
    print("="*70)

    try:
        # Run all demos
        demo_1_hardware_detection()
        demo_2_optimization_recommendations()
        demo_3_auto_optimize_simple()
        demo_4_auto_optimize_with_options()
        demo_5_performance_comparison()
        demo_6_multiple_models()
        demo_7_inference_mode()

        # Final summary
        print_section("Summary")
        print("✅ All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  1. Automatic hardware detection")
        print("  2. Intelligent backend selection (NVIDIA/TPU/CPU)")
        print("  3. Automatic optimization level selection")
        print("  4. One-line model optimization")
        print("  5. Custom optimization options")
        print("  6. Performance comparison")
        print("  7. Inference-specific optimization")

        print("\nUsage Summary:")
        print("  Simple:   optimized = get_manager().auto_optimize(model)")
        print("  Advanced: optimized = get_manager().auto_optimize(")
        print("                model,")
        print("                sample_inputs=inputs,")
        print("                optimization_level='aggressive',")
        print("                for_inference=True")
        print("            )")

        manager = get_manager()
        profile = manager.get_hardware_profile()
        print(f"\nYour Hardware: {profile.hardware_type.value} - {profile.device_name}")
        print(f"Selected Backend: {get_optimal_backend()}")

        print("\n" + "="*70)
        print("  Demo Complete!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
