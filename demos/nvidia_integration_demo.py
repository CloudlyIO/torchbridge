"""
NVIDIA Integration Demo

Demonstrates comprehensive NVIDIA backend functionality including:
- Backend initialization and device management
- Multi-level optimization (conservative/balanced/aggressive)
- FP8 training support for H100/Blackwell
- FlashAttention-3 integration
- Memory management and optimization
- CUDA utilities and profiling
"""

import torch
import torch.nn as nn
import argparse

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAArchitecture
from kernel_pytorch.backends.nvidia import (
    NVIDIABackend,
    NVIDIAOptimizer,
    FP8Compiler,
    NVIDIAMemoryManager,
    FlashAttention3,
    create_flash_attention_3,
    CUDADeviceManager,
    CUDAOptimizations,
    create_cuda_integration
)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_nvidia_backend():
    """Demonstrate NVIDIA backend functionality."""
    print_section("1. NVIDIA Backend Initialization")

    # Create backend
    backend = NVIDIABackend()

    # Display device information
    info = backend.get_device_info()
    print(f"\n✅ Backend initialized successfully")
    print(f"   CUDA available: {info['cuda_available']}")
    print(f"   Device: {info['device']}")
    print(f"   Device count: {info['device_count']}")

    if backend.is_cuda_available:
        print(f"   Device name: {info['device_name']}")
        print(f"   Compute capability: {info['compute_capability']}")
        print(f"   Architecture: {info['architecture']}")
        print(f"   FP8 supported: {info['fp8_supported']}")
        print(f"   FlashAttention enabled: {info['flash_attention_enabled']}")

    # Test model preparation
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )

    prepared_model = backend.prepare_model(model)
    print(f"\n✅ Model prepared for NVIDIA GPU")
    print(f"   Model device: {next(prepared_model.parameters()).device}")

    # Get memory stats
    memory_stats = backend.get_memory_stats()
    print(f"\n📊 Memory Statistics:")
    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")


def demo_nvidia_optimizer():
    """Demonstrate NVIDIA optimizer with multiple optimization levels."""
    print_section("2. NVIDIA Optimizer - Multi-Level Optimization")

    optimizer = NVIDIAOptimizer()
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

    # Conservative optimization
    print("\n🔧 Conservative Optimization:")
    result_conservative = optimizer.optimize(model, optimization_level="conservative")
    print(f"   Optimizations applied: {', '.join(result_conservative.optimizations_applied)}")
    print(f"   Compilation time: {result_conservative.compilation_time:.4f}s")
    print(f"   Warnings: {len(result_conservative.warnings)}")

    # Balanced optimization
    print("\n⚖️  Balanced Optimization:")
    result_balanced = optimizer.optimize(model, optimization_level="balanced")
    print(f"   Optimizations applied: {', '.join(result_balanced.optimizations_applied)}")
    print(f"   Compilation time: {result_balanced.compilation_time:.4f}s")
    print(f"   Warnings: {len(result_balanced.warnings)}")

    # Aggressive optimization
    print("\n⚡ Aggressive Optimization:")
    result_aggressive = optimizer.optimize(model, optimization_level="aggressive")
    print(f"   Optimizations applied: {', '.join(result_aggressive.optimizations_applied)}")
    print(f"   Compilation time: {result_aggressive.compilation_time:.4f}s")
    print(f"   Warnings: {len(result_aggressive.warnings)}")

    # Optimization recommendations
    print("\n📋 Optimization Recommendations:")
    recommendations = optimizer.get_optimization_recommendations(model)
    print(f"   Architecture: {recommendations['architecture']}")
    print(f"   Suggested level: {recommendations['suggested_level']}")
    print(f"   Available optimizations:")
    for opt in recommendations['optimizations'][:3]:  # Show top 3
        print(f"      - {opt['type']}: {opt['benefit']}")


def demo_fp8_compiler():
    """Demonstrate FP8 compiler for H100/Blackwell."""
    print_section("3. FP8 Compiler - H100/Blackwell Optimization")

    config = KernelPyTorchConfig()
    config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
    compiler = FP8Compiler(config)

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )

    print(f"\n✅ FP8 Compiler initialized")
    print(f"   FP8 supported: {compiler._fp8_supported}")
    print(f"   Architecture: {compiler.nvidia_config.architecture.value}")

    # Prepare for FP8 inference
    print("\n🔧 Preparing model for FP8 inference:")
    prepared_model = compiler.prepare_for_fp8(model, for_inference=True)

    # Get FP8 statistics
    stats = compiler.get_fp8_stats(prepared_model)
    print(f"   Total layers: {stats['total_layers']}")
    print(f"   FP8 layers: {stats['fp8_layers']}")
    print(f"   FP8 coverage: {stats['fp8_coverage']*100:.1f}%")

    # Estimate speedup
    speedup = compiler.estimate_speedup(prepared_model)
    print(f"\n📈 Performance Estimate:")
    print(f"   Base speedup: {speedup['base_speedup']}x")
    print(f"   Estimated speedup: {speedup['estimated_speedup']:.2f}x")
    print(f"   Architecture: {speedup['architecture']}")


def demo_memory_manager():
    """Demonstrate NVIDIA memory manager."""
    print_section("4. NVIDIA Memory Manager")

    manager = NVIDIAMemoryManager()

    print("\n✅ Memory Manager initialized")

    # Allocate tensors
    print("\n🔧 Tensor Allocation:")
    tensor1 = manager.allocate_tensor((100, 100), pool_id="demo_pool")
    print(f"   Allocated tensor: {tensor1.shape}")

    # Optimize tensor layout
    print("\n⚡ Tensor Layout Optimization:")
    test_tensor = torch.randn(127, 127)  # Non-optimal dimensions
    optimized_tensor = manager.optimize_tensor_layout(test_tensor)
    print(f"   Original shape: {test_tensor.shape}")
    print(f"   Optimized shape: {optimized_tensor.shape}")

    # Model memory optimization
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )

    print("\n📊 Model Memory Analysis:")
    memory_results = manager.optimize_model_memory(model)
    print(f"   Parameter memory: {memory_results['parameter_memory_mb']:.2f} MB")
    print(f"   Buffer memory: {memory_results['buffer_memory_mb']:.2f} MB")
    print(f"   Total memory: {memory_results['total_memory_mb']:.2f} MB")

    if memory_results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in memory_results['recommendations'][:3]:  # Show top 3
            print(f"      - {rec['type']}: {rec.get('reason', 'N/A')}")

    # Memory statistics
    print("\n📈 Memory Statistics:")
    stats = manager.get_memory_stats()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")


def demo_flash_attention():
    """Demonstrate FlashAttention-3 integration."""
    print_section("5. FlashAttention-3 Integration")

    # Create FlashAttention module
    attn = FlashAttention3(embed_dim=512, num_heads=8, dropout=0.1)

    print(f"\n✅ FlashAttention-3 initialized")
    print(f"   Embed dim: {attn.embed_dim}")
    print(f"   Num heads: {attn.num_heads}")
    print(f"   Head dim: {attn.head_dim}")
    print(f"   Flash available: {attn._flash_available}")
    print(f"   Using FlashAttention: {attn.use_flash_attention}")

    # Test forward pass
    print("\n🔧 Forward Pass Test:")
    batch_size, seq_len = 4, 128
    x = torch.randn(batch_size, seq_len, 512)

    output, attn_weights = attn(x, return_attention_weights=False)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights: {'returned' if attn_weights is not None else 'not returned (FlashAttention used)'}")

    # Test with attention mask
    print("\n🎭 Forward Pass with Attention Mask:")
    attention_mask = torch.zeros(batch_size, attn.num_heads, seq_len, seq_len)
    output_masked, _ = attn(x, attention_mask=attention_mask)
    print(f"   Output shape with mask: {output_masked.shape}")

    # Factory function
    print("\n🏭 Using Factory Function:")
    attn2 = create_flash_attention_3(embed_dim=256, num_heads=4)
    print(f"   Created via factory: embed_dim={attn2.embed_dim}, num_heads={attn2.num_heads}")


def demo_cuda_utilities():
    """Demonstrate CUDA utilities."""
    print_section("6. CUDA Utilities and Integration")

    # CUDA device manager
    print("\n🔧 CUDA Device Manager:")
    device_manager = CUDADeviceManager()
    print(f"   Device count: {device_manager.device_count}")
    print(f"   Current device: {device_manager.device}")

    if device_manager.device_count > 0:
        props = device_manager.get_device_properties(0)
        print(f"   Device 0 properties:")
        for key, value in props.items():
            if isinstance(value, tuple):
                print(f"      {key}: {value}")
            elif isinstance(value, float):
                print(f"      {key}: {value:.2f}")
            else:
                print(f"      {key}: {value}")

    # CUDA optimizations
    print("\n⚡ CUDA Optimizations:")
    cuda_opts = CUDAOptimizations()
    config = cuda_opts.get_cuda_optimization_config()
    print(f"   Configuration:")
    for key, value in config.items():
        print(f"      {key}: {value}")

    # Environment information
    print("\n🌍 CUDA Environment:")
    from kernel_pytorch.backends.nvidia.cuda_utilities import CUDAUtilities
    env_info = CUDAUtilities.get_cuda_env_info()
    for key, value in env_info.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")


def demo_integration():
    """Demonstrate full integration pipeline."""
    print_section("7. Full Integration Pipeline")

    print("\n🔧 Creating integrated setup:")
    config = KernelPyTorchConfig()
    config.hardware.nvidia.fp8_enabled = True
    config.hardware.nvidia.flash_attention_enabled = True

    # Create components
    backend = NVIDIABackend(config)
    optimizer = NVIDIAOptimizer(config)
    memory_manager = NVIDIAMemoryManager(config)

    # Create model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        FlashAttention3(embed_dim=256, num_heads=8, config=config),
        nn.Linear(256, 128)
    )

    print(f"   Model created with FlashAttention-3")

    # Prepare model
    prepared_model = backend.prepare_model(model)
    print(f"   Model prepared on device: {next(prepared_model.parameters()).device}")

    # Optimize model
    result = optimizer.optimize(prepared_model, optimization_level="aggressive")
    print(f"   Optimizations applied: {', '.join(result.optimizations_applied)}")

    # Memory analysis
    memory_analysis = memory_manager.optimize_model_memory(model)
    print(f"   Total model memory: {memory_analysis['total_memory_mb']:.2f} MB")

    print(f"\n✅ Full integration pipeline completed successfully")


def main():
    """Run all NVIDIA integration demos."""
    parser = argparse.ArgumentParser(description="NVIDIA Integration Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  NVIDIA INTEGRATION COMPREHENSIVE DEMO")
    print("  Demonstrating complete NVIDIA backend functionality")
    print("=" * 70)

    try:
        demo_nvidia_backend()
        demo_nvidia_optimizer()
        demo_fp8_compiler()
        demo_memory_manager()
        demo_flash_attention()
        demo_cuda_utilities()
        demo_integration()

        print("\n" + "=" * 70)
        print("  ✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nNVIDIA backend is ready for:")
        print("  • Production H100/Blackwell deployments")
        print("  • FP8 training for 2x speedup")
        print("  • FlashAttention-3 for 3x memory reduction")
        print("  • Multi-level optimization strategies")
        print("  • Advanced memory management")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
