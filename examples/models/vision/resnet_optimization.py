"""
ResNet Optimization Example

This example demonstrates how to optimize ResNet models for efficient inference
using TorchBridge vision model optimization.

Demonstrates:
- Loading pre-trained ResNet models
- Applying multi-level optimizations
- Benchmarking inference performance
- Memory usage analysis
"""

import torch
from torchbridge.models.vision import (
    create_resnet50_optimized,
    create_resnet152_optimized,
    ResNetBenchmark,
    OptimizationLevel,
)


def example_basic_optimization():
    """Basic ResNet-50 optimization example."""
    print("=" * 80)
    print("Example 1: Basic ResNet-50 Optimization")
    print("=" * 80)

    # Create optimized ResNet-50 with O2 optimization level
    model, optimizer = create_resnet50_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\nModel optimized successfully!")
    print(f"\nOptimization Summary:")
    summary = optimizer.get_optimization_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Create benchmark
    benchmark = ResNetBenchmark(model, optimizer)

    # Get model info
    print("\nModel Information:")
    model_info = benchmark.get_model_info()
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"\nMemory Estimate:")
    for key, value in model_info['memory_estimate'].items():
        print(f"  {key}: {value:.2f}")

    # Run inference benchmark
    print("\n" + "=" * 80)
    print("Running Inference Benchmark (batch_size=8, 100 iterations)...")
    print("=" * 80)

    results = benchmark.benchmark_inference(
        batch_size=8,
        num_iterations=100,
        warmup_iterations=10,
        image_size=224,
    )

    print("\nBenchmark Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def example_optimization_levels():
    """Compare different optimization levels."""
    print("\n" + "=" * 80)
    print("Example 2: Comparing Optimization Levels")
    print("=" * 80)

    levels = [
        OptimizationLevel.O0,
        OptimizationLevel.O1,
        OptimizationLevel.O2,
        OptimizationLevel.O3,
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("\nNote: Running on CPU (CUDA not available)")
        print("GPU optimizations will be skipped.\n")

    results = {}

    for level in levels:
        print(f"\nTesting {level.value}...")

        # Create optimized model
        model, optimizer = create_resnet50_optimized(
            optimization_level=level,
            batch_size=8,
            device=device,
        )

        # Run benchmark
        benchmark = ResNetBenchmark(model, optimizer)
        bench_results = benchmark.benchmark_inference(
            batch_size=8,
            num_iterations=50,
            warmup_iterations=5,
            image_size=224,
        )

        results[level.value] = bench_results

        print(f"  Throughput: {bench_results['throughput_images_per_second']:.2f} images/sec")
        print(f"  Time per image: {bench_results['time_per_image_ms']:.2f} ms")
        print(f"  Optimizations: {optimizer.optimizations_applied}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Optimization Level Comparison")
    print("=" * 80)
    print(f"{'Level':<10} {'Throughput (img/s)':<20} {'Time per Image (ms)':<20}")
    print("-" * 80)

    for level_name, result in results.items():
        throughput = result['throughput_images_per_second']
        time_per_img = result['time_per_image_ms']
        print(f"{level_name:<10} {throughput:<20.2f} {time_per_img:<20.2f}")


def example_large_model():
    """ResNet-152 optimization example."""
    print("\n" + "=" * 80)
    print("Example 3: Large Model (ResNet-152) Optimization")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized ResNet-152
    model, optimizer = create_resnet152_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=16,
        device=device,
    )

    print("\nResNet-152 optimized successfully!")

    # Get model info
    benchmark = ResNetBenchmark(model, optimizer)
    model_info = benchmark.get_model_info()

    print(f"\nModel Information:")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Memory (Inference): {model_info['memory_estimate']['total_inference_mb']:.2f} MB")

    # Run benchmark with different batch sizes
    print("\n" + "=" * 80)
    print("Benchmarking Different Batch Sizes")
    print("=" * 80)

    batch_sizes = [1, 4, 8, 16] if device == "cuda" else [1, 2, 4]

    for batch_size in batch_sizes:
        results = benchmark.benchmark_inference(
            batch_size=batch_size,
            num_iterations=50,
            warmup_iterations=5,
            image_size=224,
        )

        print(f"\nBatch Size: {batch_size}")
        print(f"  Throughput: {results['throughput_images_per_second']:.2f} images/sec")
        print(f"  Time per batch: {results['time_per_batch_seconds']*1000:.2f} ms")
        print(f"  Time per image: {results['time_per_image_ms']:.2f} ms")


def example_batch_inference():
    """Batch inference example."""
    print("\n" + "=" * 80)
    print("Example 4: Optimized Batch Inference")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized model
    model, optimizer = create_resnet50_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=32,
        device=device,
    )

    print("\nModel optimized successfully!")

    # Create dummy images
    num_images = 100
    images = torch.randn(num_images, 3, 224, 224)

    print(f"\nRunning batch inference on {num_images} images...")
    print(f"Configured batch size: {optimizer.config.batch_size}")

    import time
    start_time = time.time()

    # Run optimized batch inference
    outputs = optimizer.optimize_batch_inference(model, images)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nInference completed!")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Throughput: {num_images / total_time:.2f} images/sec")
    print(f"  Output shape: {outputs.shape}")


def example_custom_config():
    """Custom configuration example."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Configuration")
    print("=" * 80)

    from torchbridge.models.vision import (
        VisionOptimizationConfig,
        VisionModelType,
        create_resnet_optimizer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create custom configuration
    config = VisionOptimizationConfig(
        model_type=VisionModelType.RESNET,
        optimization_level=OptimizationLevel.O2,
        batch_size=16,
        enable_fusion=True,
        enable_cudnn_benchmark=True,
        enable_tf32=True,
        use_fp16=True if device == "cuda" else False,
        channels_last=True,
        compile_model=False,  # Disable compile for faster startup
        device=device,
    )

    print(f"\nCustom Configuration:")
    print(f"  Model Type: {config.model_type.value}")
    print(f"  Optimization Level: {config.optimization_level.value}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Fusion: {config.enable_fusion}")
    print(f"  cuDNN Benchmark: {config.enable_cudnn_benchmark}")
    print(f"  TF32: {config.enable_tf32}")
    print(f"  FP16: {config.use_fp16}")
    print(f"  Channels Last: {config.channels_last}")
    print(f"  Device: {config.device}")

    # Create model with custom config
    from torchbridge.models.vision import ResNetOptimizer

    model, optimizer_instance = create_resnet_optimizer(
        model_name="resnet50",
        optimization_level=config.optimization_level,
        batch_size=config.batch_size,
        device=config.device,
        enable_fusion=config.enable_fusion,
        use_fp16=config.use_fp16,
    )

    print("\nModel created with custom configuration!")
    print(f"Applied optimizations: {optimizer_instance.optimizations_applied}")


def main():
    """Run all examples."""
    print("ResNet Optimization Examples")
    print("=" * 80)
    print("\nThese examples demonstrate optimizing ResNet models for inference.")
    print("Requirements: torchvision")
    print("\nInstall with: pip install torchvision")
    print("=" * 80)

    try:
        import torchvision
    except ImportError:
        print("\nERROR: torchvision not installed!")
        print("Install with: pip install torchvision")
        return

    # Run examples
    example_basic_optimization()
    example_optimization_levels()
    example_large_model()
    example_batch_inference()
    example_custom_config()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
