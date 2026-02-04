"""
Stable Diffusion Optimization Example

This example demonstrates how to optimize Stable Diffusion models for efficient
image generation using TorchBridge vision model optimization.

Demonstrates:
- Loading pre-trained Stable Diffusion models
- Applying memory-efficient optimizations
- VAE tiling for large images
- Attention slicing
- Benchmarking generation performance
"""

import torch
from torchbridge.models.vision import (
    create_sd_1_5_optimized,
    create_sd_2_1_optimized,
    StableDiffusionBenchmark,
    OptimizationLevel,
)


def example_basic_optimization():
    """Basic Stable Diffusion 1.5 optimization example."""
    print("=" * 80)
    print("Example 1: Basic Stable Diffusion 1.5 Optimization")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("\nWARNING: Running Stable Diffusion on CPU is very slow!")
        print("A CUDA-enabled GPU is strongly recommended.\n")

    # Create optimized Stable Diffusion 1.5 pipeline
    pipeline, optimizer = create_sd_1_5_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
        num_inference_steps=50,
        enable_attention_slicing=True,
    )

    print("\nStable Diffusion 1.5 optimized successfully!")
    print(f"\nOptimization Summary:")
    summary = optimizer.get_optimization_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Create benchmark
    benchmark = StableDiffusionBenchmark(pipeline, optimizer)

    # Get model info
    print("\nModel Information:")
    model_info = benchmark.get_model_info()
    for key, value in model_info.items():
        if isinstance(value, dict):
            continue
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    # Generate sample image
    print("\n" + "=" * 80)
    print("Generating sample image...")
    print("=" * 80)

    prompt = "A beautiful sunset over mountains, high quality, detailed"
    print(f"\nPrompt: {prompt}")

    output = optimizer.generate_optimized(
        prompt=prompt,
        num_inference_steps=50,
        height=512,
        width=512,
    )

    print("\nImage generated successfully!")
    print(f"  Output type: {type(output)}")

    # In practice, save the image:
    # output.images[0].save("generated_image.png")


def example_memory_optimizations():
    """Memory optimization techniques."""
    print("\n" + "=" * 80)
    print("Example 2: Memory Optimization Techniques")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("\nNote: Memory optimizations are most beneficial on GPU.\n")

    # Create pipeline with aggressive memory optimizations
    print("\nCreating pipeline with memory optimizations:")
    print("  - Attention slicing")
    print("  - VAE tiling")
    print("  - FP16 precision")

    pipeline, optimizer = create_sd_1_5_optimized(
        optimization_level=OptimizationLevel.O3,
        device=device,
        enable_attention_slicing=True,
        attention_slice_size="auto",
        enable_vae_tiling=True,
        use_fp16=True if device == "cuda" else False,
    )

    print("\nMemory-optimized pipeline created!")
    print(f"Applied optimizations: {optimizer.optimizations_applied}")

    # Generate larger image (demonstrates VAE tiling benefit)
    prompt = "A fantasy castle in the clouds, highly detailed"
    print(f"\nGenerating 768x768 image...")
    print(f"Prompt: {prompt}")

    output = optimizer.generate_optimized(
        prompt=prompt,
        num_inference_steps=30,
        height=768,
        width=768,
    )

    print("\nLarge image generated successfully!")
    print("VAE tiling allows generation of large images without OOM errors.")


def example_batch_generation():
    """Batch image generation."""
    print("\n" + "=" * 80)
    print("Example 3: Batch Image Generation")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized pipeline
    pipeline, optimizer = create_sd_1_5_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
        enable_attention_slicing=True,
    )

    print("\nPipeline optimized successfully!")

    # Generate multiple images from single prompt
    prompt = "A cute robot playing with a ball, cartoon style"
    num_images = 4

    print(f"\nGenerating {num_images} images from prompt:")
    print(f"  {prompt}")

    import time
    start_time = time.time()

    output = optimizer.generate_optimized(
        prompt=prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=30,
        height=512,
        width=512,
    )

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{num_images} images generated!")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Time per image: {total_time / num_images:.2f} seconds")


def example_guided_generation():
    """Classifier-free guidance example."""
    print("\n" + "=" * 80)
    print("Example 4: Classifier-Free Guidance")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized pipeline
    pipeline, optimizer = create_sd_1_5_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
    )

    prompt = "A professional photograph of a vintage car"
    negative_prompt = "blurry, low quality, distorted"

    guidance_scales = [3.0, 7.5, 12.0]

    print(f"\nPrompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print("\nTesting different guidance scales...")

    for guidance_scale in guidance_scales:
        print(f"\n  Guidance Scale: {guidance_scale}")

        import time
        start_time = time.time()

        output = optimizer.generate_optimized(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
            height=512,
            width=512,
        )

        end_time = time.time()

        print(f"    Generation time: {end_time - start_time:.2f} seconds")

    print("\nNote: Higher guidance scale follows prompt more closely but may reduce diversity.")


def example_benchmark():
    """Benchmark generation performance."""
    print("\n" + "=" * 80)
    print("Example 5: Performance Benchmarking")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create optimized pipeline
    pipeline, optimizer = create_sd_1_5_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
        enable_attention_slicing=True,
    )

    print("\nPipeline optimized successfully!")

    # Create benchmark
    benchmark = StableDiffusionBenchmark(pipeline, optimizer)

    # Benchmark different configurations
    configs = [
        {"num_inference_steps": 25, "height": 512, "width": 512},
        {"num_inference_steps": 50, "height": 512, "width": 512},
        {"num_inference_steps": 30, "height": 768, "width": 768},
    ]

    print("\n" + "=" * 80)
    print("Benchmarking different configurations...")
    print("=" * 80)

    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}:")
        print(f"  Steps: {config['num_inference_steps']}")
        print(f"  Size: {config['height']}x{config['width']}")

        results = benchmark.benchmark_generation(
            prompt="A test image for benchmarking",
            num_iterations=3,
            **config
        )

        print(f"\nResults:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


def example_sd2_comparison():
    """Compare SD 1.5 vs SD 2.1."""
    print("\n" + "=" * 80)
    print("Example 6: Comparing SD 1.5 vs SD 2.1")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = "A majestic lion in the savanna at sunset"

    # SD 1.5
    print("\nLoading Stable Diffusion 1.5...")
    pipeline_15, optimizer_15 = create_sd_1_5_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
    )

    benchmark_15 = StableDiffusionBenchmark(pipeline_15, optimizer_15)
    model_info_15 = benchmark_15.get_model_info()

    print(f"  Total Parameters: {model_info_15['total_parameters']:,}")
    print(f"  Optimizations: {optimizer_15.optimizations_applied}")

    # SD 2.1
    print("\nLoading Stable Diffusion 2.1...")
    pipeline_21, optimizer_21 = create_sd_2_1_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
    )

    benchmark_21 = StableDiffusionBenchmark(pipeline_21, optimizer_21)
    model_info_21 = benchmark_21.get_model_info()

    print(f"  Total Parameters: {model_info_21['total_parameters']:,}")
    print(f"  Optimizations: {optimizer_21.optimizations_applied}")

    print("\n" + "=" * 80)
    print("Generating images with both models...")
    print("=" * 80)
    print(f"Prompt: {prompt}")

    import time

    # Generate with SD 1.5
    print("\nGenerating with SD 1.5...")
    start_time = time.time()
    output_15 = optimizer_15.generate_optimized(
        prompt=prompt,
        num_inference_steps=30,
    )
    time_15 = time.time() - start_time
    print(f"  Generation time: {time_15:.2f} seconds")

    # Generate with SD 2.1
    print("\nGenerating with SD 2.1...")
    start_time = time.time()
    output_21 = optimizer_21.generate_optimized(
        prompt=prompt,
        num_inference_steps=30,
    )
    time_21 = time.time() - start_time
    print(f"  Generation time: {time_21:.2f} seconds")

    print("\nBoth images generated successfully!")


def example_custom_config():
    """Custom configuration example."""
    print("\n" + "=" * 80)
    print("Example 7: Custom Configuration")
    print("=" * 80)

    from torchbridge.models.vision import (
        VisionOptimizationConfig,
        VisionModelType,
        StableDiffusionOptimizer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create custom configuration
    config = VisionOptimizationConfig(
        model_type=VisionModelType.STABLE_DIFFUSION,
        optimization_level=OptimizationLevel.O3,
        enable_cudnn_benchmark=True,
        enable_tf32=True,
        enable_attention_slicing=True,
        attention_slice_size=8,
        enable_vae_tiling=True,
        vae_tile_size=512,
        use_fp16=True if device == "cuda" else False,
        num_inference_steps=30,
        device=device,
    )

    print(f"\nCustom Configuration:")
    print(f"  Model Type: {config.model_type.value}")
    print(f"  Optimization Level: {config.optimization_level.value}")
    print(f"  Attention Slicing: {config.enable_attention_slicing}")
    print(f"  Attention Slice Size: {config.attention_slice_size}")
    print(f"  VAE Tiling: {config.enable_vae_tiling}")
    print(f"  VAE Tile Size: {config.vae_tile_size}")
    print(f"  FP16: {config.use_fp16}")
    print(f"  Inference Steps: {config.num_inference_steps}")
    print(f"  Device: {config.device}")

    print("\nLoading pipeline with custom configuration...")
    pipeline, optimizer = create_sd_1_5_optimized(
        optimization_level=config.optimization_level,
        device=config.device,
        enable_attention_slicing=config.enable_attention_slicing,
        attention_slice_size=config.attention_slice_size,
        enable_vae_tiling=config.enable_vae_tiling,
        use_fp16=config.use_fp16,
        num_inference_steps=config.num_inference_steps,
    )

    print("\nPipeline created with custom configuration!")
    print(f"Applied optimizations: {optimizer.optimizations_applied}")


def main():
    """Run all examples."""
    print("Stable Diffusion Optimization Examples")
    print("=" * 80)
    print("\nThese examples demonstrate optimizing Stable Diffusion for image generation.")
    print("Requirements: diffusers, transformers")
    print("\nInstall with: pip install diffusers transformers")
    print("\nNote: These examples will download models (~4-7GB per model)")
    print("=" * 80)

    try:
        import diffusers
        import transformers
    except ImportError:
        print("\nERROR: diffusers and/or transformers not installed!")
        print("Install with: pip install diffusers transformers")
        return

    # Run examples
    example_basic_optimization()
    example_memory_optimizations()
    example_batch_generation()
    example_guided_generation()
    example_benchmark()
    example_sd2_comparison()
    example_custom_config()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nNote: To save generated images, use:")
    print("  output.images[0].save('image.png')")


if __name__ == "__main__":
    main()
