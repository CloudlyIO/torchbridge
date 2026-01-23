"""
CLIP Optimization Example

This example demonstrates how to optimize CLIP models for efficient
vision-language embedding using KernelPyTorch multi-modal optimization.

Demonstrates:
- Loading pre-trained CLIP models
- Optimizing for image and text encoding
- Computing image-text similarities
- Benchmarking performance
"""

import torch
from kernel_pytorch.models.multimodal import (
    create_clip_vit_b_optimized,
    create_clip_vit_l_optimized,
    CLIPBenchmark,
    OptimizationLevel,
)


def example_basic_optimization():
    """Basic CLIP ViT-B/32 optimization example."""
    print("=" * 80)
    print("Example 1: Basic CLIP ViT-B/32 Optimization")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized CLIP ViT-B/32
    model, optimizer = create_clip_vit_b_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=32,
        device=device,
    )

    print("\nCLIP ViT-B/32 optimized successfully!")
    print(f"\nOptimization Summary:")
    summary = optimizer.get_optimization_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Create benchmark
    benchmark = CLIPBenchmark(model, optimizer)

    # Get model info
    print("\nModel Information:")
    model_info = benchmark.get_model_info()
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    if "vision_parameters" in model_info:
        print(f"  Vision Parameters: {model_info['vision_parameters']:,}")
    if "text_parameters" in model_info:
        print(f"  Text Parameters: {model_info['text_parameters']:,}")

    print("\nNote: This creates an optimized CLIP model ready for embedding generation.")


def example_image_encoding():
    """Image encoding example."""
    print("\n" + "=" * 80)
    print("Example 2: Image Encoding")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized model
    model, optimizer = create_clip_vit_b_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=32,
        device=device,
    )

    print("\nModel optimized successfully!")

    # Create dummy images (in practice, load real images)
    num_images = 10
    images = torch.randn(num_images, 3, 224, 224)

    print(f"\nEncoding {num_images} images...")

    import time
    start_time = time.time()

    # Encode images
    image_embeddings = optimizer.encode_images(model, images, normalize=True)

    end_time = time.time()

    print(f"\nEncoding completed!")
    print(f"  Time: {end_time - start_time:.2f} seconds")
    print(f"  Throughput: {num_images / (end_time - start_time):.2f} images/sec")
    print(f"  Embedding shape: {image_embeddings.shape}")
    print(f"  Embedding dtype: {image_embeddings.dtype}")


def example_text_encoding():
    """Text encoding example."""
    print("\n" + "=" * 80)
    print("Example 3: Text Encoding")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized model
    model, optimizer = create_clip_vit_b_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=32,
        device=device,
    )

    print("\nModel optimized successfully!")

    # Create dummy text inputs (in practice, use real text)
    # For demonstration, use random token IDs
    num_texts = 10
    text_inputs = torch.randint(0, 49407, (num_texts, 77))  # CLIP vocab size, max length

    print(f"\nEncoding {num_texts} texts...")

    import time
    start_time = time.time()

    # Encode texts
    text_embeddings = optimizer.encode_text(model, text_inputs, normalize=True)

    end_time = time.time()

    print(f"\nEncoding completed!")
    print(f"  Time: {end_time - start_time:.2f} seconds")
    print(f"  Throughput: {num_texts / (end_time - start_time):.2f} texts/sec")
    print(f"  Embedding shape: {text_embeddings.shape}")
    print(f"  Embedding dtype: {text_embeddings.dtype}")


def example_similarity_computation():
    """Image-text similarity computation example."""
    print("\n" + "=" * 80)
    print("Example 4: Image-Text Similarity Computation")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized model
    model, optimizer = create_clip_vit_b_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
    )

    print("\nModel optimized successfully!")

    # Create dummy embeddings
    num_images = 5
    num_texts = 3

    images = torch.randn(num_images, 3, 224, 224)
    text_inputs = torch.randint(0, 49407, (num_texts, 77))

    print(f"\nEncoding {num_images} images and {num_texts} texts...")

    # Encode
    image_embeddings = optimizer.encode_images(model, images)
    text_embeddings = optimizer.encode_text(model, text_inputs)

    # Compute similarity
    similarity = optimizer.compute_similarity(image_embeddings, text_embeddings)

    print(f"\nSimilarity matrix computed!")
    print(f"  Shape: {similarity.shape} ({num_images} images × {num_texts} texts)")
    print(f"\nSimilarity scores:")
    print(similarity)

    # Find best matches
    best_text_per_image = similarity.argmax(dim=1)
    best_image_per_text = similarity.argmax(dim=0)

    print(f"\nBest text match for each image: {best_text_per_image.tolist()}")
    print(f"Best image match for each text: {best_image_per_text.tolist()}")


def example_benchmarking():
    """Benchmarking example."""
    print("\n" + "=" * 80)
    print("Example 5: Performance Benchmarking")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("\nNote: Running on CPU. Benchmarks will be slower.\n")

    # Create optimized model
    model, optimizer = create_clip_vit_b_optimized(
        optimization_level=OptimizationLevel.O2,
        batch_size=32,
        device=device,
    )

    print("\nModel optimized successfully!")

    # Create benchmark
    benchmark = CLIPBenchmark(model, optimizer)

    # Benchmark image encoding
    print("\n" + "=" * 80)
    print("Benchmarking Image Encoding...")
    print("=" * 80)

    image_results = benchmark.benchmark_image_encoding(
        batch_size=32,
        num_iterations=50,
        warmup_iterations=5,
    )

    print("\nImage Encoding Results:")
    for key, value in image_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Benchmark text encoding
    print("\n" + "=" * 80)
    print("Benchmarking Text Encoding...")
    print("=" * 80)

    text_results = benchmark.benchmark_text_encoding(
        batch_size=32,
        num_iterations=50,
        warmup_iterations=5,
    )

    print("\nText Encoding Results:")
    for key, value in text_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def example_optimization_levels():
    """Compare optimization levels."""
    print("\n" + "=" * 80)
    print("Example 6: Comparing Optimization Levels")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    levels = [OptimizationLevel.O0, OptimizationLevel.O1, OptimizationLevel.O2]

    results = {}

    for level in levels:
        print(f"\nTesting {level.value}...")

        model, optimizer = create_clip_vit_b_optimized(
            optimization_level=level,
            batch_size=16,
            device=device,
        )

        # Quick benchmark
        benchmark = CLIPBenchmark(model, optimizer)
        bench_results = benchmark.benchmark_image_encoding(
            batch_size=16,
            num_iterations=20,
            warmup_iterations=3,
        )

        results[level.value] = bench_results

        print(f"  Throughput: {bench_results['throughput_images_per_second']:.2f} img/s")
        print(f"  Optimizations: {optimizer.optimizations_applied}")

    # Print comparison
    print("\n" + "=" * 80)
    print("Optimization Level Comparison")
    print("=" * 80)
    print(f"{'Level':<10} {'Throughput (img/s)':<20} {'Time per Image (ms)':<20}")
    print("-" * 80)

    for level_name, result in results.items():
        throughput = result['throughput_images_per_second']
        time_per_img = result['time_per_image_ms']
        print(f"{level_name:<10} {throughput:<20.2f} {time_per_img:<20.2f}")


def main():
    """Run all examples."""
    print("CLIP Optimization Examples")
    print("=" * 80)
    print("\nThese examples demonstrate optimizing CLIP models for vision-language tasks.")
    print("Requirements: transformers")
    print("\nInstall with: pip install transformers")
    print("=" * 80)

    try:
        import transformers
    except ImportError:
        print("\nERROR: transformers not installed!")
        print("Install with: pip install transformers")
        return

    # Run examples
    example_basic_optimization()
    example_image_encoding()
    example_text_encoding()
    example_similarity_computation()
    example_benchmarking()
    example_optimization_levels()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
