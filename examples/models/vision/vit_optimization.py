"""
Vision Transformer (ViT) Optimization Example

This example demonstrates how to optimize ViT models for efficient inference
using KernelPyTorch vision model optimization.

Demonstrates:
- Loading pre-trained ViT models
- Applying multi-level optimizations
- Attention slicing for memory efficiency
- Benchmarking inference performance
"""

import torch
from kernel_pytorch.models.vision import (
    create_vit_base_optimized,
    create_vit_large_optimized,
    ViTBenchmark,
    OptimizationLevel,
)


def example_basic_optimization():
    """Basic ViT-Base optimization example."""
    print("=" * 80)
    print("Example 1: Basic ViT-Base Optimization")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized ViT-Base with O2 optimization level
    model, optimizer = create_vit_base_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=32,
        device=device,
    )

    print("\nViT-Base optimized successfully!")
    print(f"\nOptimization Summary:")
    summary = optimizer.get_optimization_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Create benchmark
    benchmark = ViTBenchmark(model, optimizer)

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


def example_attention_slicing():
    """Attention slicing for memory efficiency."""
    print("\n" + "=" * 80)
    print("Example 2: Attention Slicing for Memory Efficiency")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Without attention slicing
    print("\nWithout Attention Slicing:")
    model1, optimizer1 = create_vit_base_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=16,
        device=device,
        enable_attention_slicing=False,
    )

    benchmark1 = ViTBenchmark(model1, optimizer1)
    results1 = benchmark1.benchmark_inference(
        batch_size=16,
        num_iterations=50,
        warmup_iterations=5,
    )

    print(f"  Throughput: {results1['throughput_images_per_second']:.2f} images/sec")
    print(f"  Time per image: {results1['time_per_image_ms']:.2f} ms")

    # With attention slicing
    print("\nWith Attention Slicing:")
    model2, optimizer2 = create_vit_base_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=16,
        device=device,
        enable_attention_slicing=True,
        attention_slice_size=8,
    )

    benchmark2 = ViTBenchmark(model2, optimizer2)
    results2 = benchmark2.benchmark_inference(
        batch_size=16,
        num_iterations=50,
        warmup_iterations=5,
    )

    print(f"  Throughput: {results2['throughput_images_per_second']:.2f} images/sec")
    print(f"  Time per image: {results2['time_per_image_ms']:.2f} ms")

    print("\nNote: Attention slicing reduces memory usage at the cost of slight throughput reduction.")


def example_large_model():
    """ViT-Large optimization example."""
    print("\n" + "=" * 80)
    print("Example 3: Large Model (ViT-Large) Optimization")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized ViT-Large
    model, optimizer = create_vit_large_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=8,
        device=device,
        enable_attention_slicing=True,
    )

    print("\nViT-Large optimized successfully!")

    # Get model info
    benchmark = ViTBenchmark(model, optimizer)
    model_info = benchmark.get_model_info()

    print(f"\nModel Information:")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Memory (Inference): {model_info['memory_estimate']['total_inference_mb']:.2f} MB")

    # Run benchmark with different batch sizes
    print("\n" + "=" * 80)
    print("Benchmarking Different Batch Sizes")
    print("=" * 80)

    batch_sizes = [1, 2, 4, 8] if device == "cuda" else [1, 2]

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
    model, optimizer = create_vit_base_optimized(
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


def example_optimization_levels():
    """Compare different optimization levels."""
    print("\n" + "=" * 80)
    print("Example 5: Comparing Optimization Levels")
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
        model, optimizer = create_vit_base_optimized(
            optimization_level=level,
            batch_size=8,
            device=device,
        )

        # Run benchmark
        benchmark = ViTBenchmark(model, optimizer)
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


def example_real_inference():
    """Real image classification example."""
    print("\n" + "=" * 80)
    print("Example 6: Real Image Classification")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized model
    model, optimizer = create_vit_base_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=1,
        device=device,
    )

    print("\nModel optimized successfully!")
    print("\nNote: This example uses a random image for demonstration.")
    print("In practice, you would load real images using PIL or OpenCV.")

    # Create dummy image (in practice, load from file)
    image = torch.randn(1, 3, 224, 224)

    # Run inference
    with torch.no_grad():
        if device == "cuda":
            image = image.cuda()
            if optimizer.config.use_fp16:
                image = image.half()

        output = model(image)

    print(f"\nInference completed!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output type: {output.dtype}")

    # Get top-5 predictions (indices)
    top5_indices = torch.topk(output, k=5, dim=1).indices[0]
    print(f"\nTop-5 predicted class indices: {top5_indices.tolist()}")

    print("\nTo get actual class names, use a pre-trained model with a label mapping.")


def main():
    """Run all examples."""
    print("Vision Transformer (ViT) Optimization Examples")
    print("=" * 80)
    print("\nThese examples demonstrate optimizing ViT models for inference.")
    print("Requirements: timm or torchvision")
    print("\nInstall with: pip install timm")
    print("=" * 80)

    try:
        import timm
    except ImportError:
        try:
            import torchvision
        except ImportError:
            print("\nERROR: Neither timm nor torchvision installed!")
            print("Install with: pip install timm")
            return

    # Run examples
    example_basic_optimization()
    example_attention_slicing()
    example_large_model()
    example_batch_inference()
    example_optimization_levels()
    example_real_inference()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
