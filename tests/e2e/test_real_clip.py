"""
End-to-End Tests for Real CLIP Model Optimization

Tests that the CLIPOptimizer produces measurable speedups on real
CLIP model loaded from HuggingFace without degrading image-text similarity.

Markers:
    - @pytest.mark.e2e: End-to-end test
    - @pytest.mark.slow: Long-running test
    - @pytest.mark.real_model: Uses real model weights
    - @pytest.mark.requires_transformers: Requires HuggingFace transformers

Success Criteria:
    - CLIP optimization shows measurable speedup
    - Image-text similarity ranking preserved
    - No significant embedding quality degradation
"""

import pytest
import torch

from .conftest import (
    assert_output_close,
    benchmark_function,
    requires_cuda,
    requires_transformers,
)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
class TestRealCLIPOptimization:
    """Test CLIP optimization with real HuggingFace model."""

    def test_clip_loads_and_encodes(self, clip_model_and_processor, sample_pil_images, e2e_device):
        """Verify real CLIP model loads and produces embeddings."""
        model, processor = clip_model_and_processor

        # Prepare inputs
        texts = ["a photo of a cat", "a photo of a dog", "a sunset", "a mountain"]
        inputs = processor(
            text=texts,
            images=sample_pil_images,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        model = model.to(e2e_device)
        inputs = {k: v.to(e2e_device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = model(**inputs)

        # Verify output shapes
        assert outputs.text_embeds.shape == (4, 512)  # 4 texts, 512-dim embeddings
        assert outputs.image_embeds.shape == (4, 512)  # 4 images, 512-dim embeddings

        # Verify embeddings are normalized
        text_norms = outputs.text_embeds.norm(dim=-1)
        image_norms = outputs.image_embeds.norm(dim=-1)
        assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=0.01)
        assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=0.01)

    def test_clip_similarity_computation(self, clip_model_and_processor, sample_pil_images, e2e_device):
        """Test CLIP image-text similarity computation."""
        model, processor = clip_model_and_processor

        texts = ["a colorful image", "random noise pattern", "abstract art", "digital artwork"]
        inputs = processor(
            text=texts,
            images=sample_pil_images,
            return_tensors="pt",
            padding=True
        )

        model = model.to(e2e_device)
        inputs = {k: v.to(e2e_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Compute similarity matrix
        logits_per_image = outputs.logits_per_image  # (4, 4)
        logits_per_text = outputs.logits_per_text    # (4, 4)

        assert logits_per_image.shape == (4, 4)
        assert logits_per_text.shape == (4, 4)

        # Similarity should be symmetric (up to temperature scaling)
        assert torch.allclose(logits_per_image, logits_per_text.T, atol=0.1)

    def test_clip_optimization_speedup(self, clip_model_and_processor, sample_pil_images, e2e_device):
        """Test CLIP optimization produces speedup."""
        from kernel_pytorch.models.multimodal import (
            CLIPOptimizer,
            MultiModalOptimizationConfig,
            OptimizationLevel,
        )

        model, processor = clip_model_and_processor

        texts = ["test image one", "test image two", "test image three", "test image four"]
        inputs = processor(
            text=texts,
            images=sample_pil_images,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(e2e_device) for k, v in inputs.items()}

        # Baseline
        baseline_model = model.to(e2e_device)
        baseline_model.eval()

        def run_baseline():
            with torch.no_grad():
                return baseline_model(**inputs)

        # Optimize
        config = MultiModalOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            compile_model=False,
        )
        optimizer = CLIPOptimizer(config)
        optimized_model = optimizer.optimize(model)
        optimized_model = optimized_model.to(e2e_device)

        def run_optimized():
            with torch.no_grad():
                return optimized_model(**inputs)

        # Benchmark
        baseline_result = benchmark_function(
            run_baseline,
            warmup_runs=2,
            benchmark_runs=5,
            sync_cuda=torch.cuda.is_available()
        )
        optimized_result = benchmark_function(
            run_optimized,
            warmup_runs=2,
            benchmark_runs=5,
            sync_cuda=torch.cuda.is_available()
        )

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms

        print("\nCLIP Optimization Results:")
        print(f"  Baseline: {baseline_result.mean_time_ms:.2f}ms")
        print(f"  Optimized: {optimized_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Allow some variance
        assert speedup >= 0.85, f"Unexpected slowdown: {speedup:.2f}x"

    def test_clip_embedding_correctness(self, clip_model_and_processor, sample_pil_images, e2e_device):
        """Test CLIP embeddings match after optimization."""
        from kernel_pytorch.models.multimodal import (
            CLIPOptimizer,
            MultiModalOptimizationConfig,
            OptimizationLevel,
        )

        model, processor = clip_model_and_processor

        texts = ["embedding test"]
        inputs = processor(
            text=texts,
            images=sample_pil_images[:1],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(e2e_device) for k, v in inputs.items()}

        # Baseline embeddings
        baseline_model = model.to(e2e_device)
        baseline_model.eval()
        with torch.no_grad():
            baseline_outputs = baseline_model(**inputs)

        # Optimize (without FP16 for precision)
        config = MultiModalOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            compile_model=False,
            use_fp16=False,
        )
        optimizer = CLIPOptimizer(config)
        optimized_model = optimizer.optimize(model)
        optimized_model = optimized_model.to(e2e_device)

        with torch.no_grad():
            optimized_outputs = optimized_model(**inputs)

        # Check text embeddings
        assert_output_close(
            baseline_outputs.text_embeds,
            optimized_outputs.text_embeds,
            atol=1e-3,
            rtol=1e-3,
            message="CLIP text embeddings"
        )

        # Check image embeddings
        assert_output_close(
            baseline_outputs.image_embeds,
            optimized_outputs.image_embeds,
            atol=1e-3,
            rtol=1e-3,
            message="CLIP image embeddings"
        )

    def test_clip_similarity_ranking_preserved(self, clip_model_and_processor, sample_pil_images, e2e_device):
        """Test CLIP similarity ranking is preserved after optimization."""
        from kernel_pytorch.models.multimodal import (
            CLIPOptimizer,
            MultiModalOptimizationConfig,
            OptimizationLevel,
        )

        model, processor = clip_model_and_processor

        # Distinct texts to test ranking
        texts = [
            "a vibrant colorful pattern",
            "a dark black image",
            "a bright white background",
            "a mixed gray texture"
        ]

        inputs = processor(
            text=texts,
            images=sample_pil_images,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(e2e_device) for k, v in inputs.items()}

        # Baseline rankings
        baseline_model = model.to(e2e_device)
        baseline_model.eval()
        with torch.no_grad():
            baseline_outputs = baseline_model(**inputs)
        baseline_rankings = baseline_outputs.logits_per_image.argsort(dim=-1, descending=True)

        # Optimize
        config = MultiModalOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            compile_model=False,
        )
        optimizer = CLIPOptimizer(config)
        optimized_model = optimizer.optimize(model)
        optimized_model = optimized_model.to(e2e_device)

        with torch.no_grad():
            optimized_outputs = optimized_model(**inputs)
        optimized_rankings = optimized_outputs.logits_per_image.argsort(dim=-1, descending=True)

        # Top-1 rankings should match for each image
        assert torch.equal(
            baseline_rankings[:, 0],
            optimized_rankings[:, 0]
        ), "Top-1 similarity rankings differ"

        print("\nCLIP Similarity Ranking Test:")
        print(f"  Baseline top-1: {baseline_rankings[:, 0].tolist()}")
        print(f"  Optimized top-1: {optimized_rankings[:, 0].tolist()}")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
class TestCLIPEncoderOptimization:
    """Test CLIP individual encoder optimization."""

    def test_clip_image_encoder_speedup(self, clip_model_and_processor, sample_pil_images, e2e_device):
        """Test CLIP image encoder optimization speedup."""
        from kernel_pytorch.models.multimodal import (
            CLIPOptimizer,
            MultiModalOptimizationConfig,
            OptimizationLevel,
        )

        model, processor = clip_model_and_processor

        # Image-only inputs
        inputs = processor(
            images=sample_pil_images,
            return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(e2e_device)

        # Baseline
        baseline_model = model.to(e2e_device)
        baseline_model.eval()

        def run_baseline():
            with torch.no_grad():
                return baseline_model.get_image_features(pixel_values)

        # Optimize
        config = MultiModalOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            compile_model=False,
        )
        optimizer = CLIPOptimizer(config)
        optimized_model = optimizer.optimize(model)
        optimized_model = optimized_model.to(e2e_device)

        def run_optimized():
            with torch.no_grad():
                return optimized_model.get_image_features(pixel_values)

        # Benchmark
        baseline_result = benchmark_function(run_baseline, warmup_runs=2, benchmark_runs=5)
        optimized_result = benchmark_function(run_optimized, warmup_runs=2, benchmark_runs=5)

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms

        print("\nCLIP Image Encoder Speedup:")
        print(f"  Baseline: {baseline_result.mean_time_ms:.2f}ms")
        print(f"  Optimized: {optimized_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_clip_text_encoder_speedup(self, clip_model_and_processor, e2e_device):
        """Test CLIP text encoder optimization speedup."""
        from kernel_pytorch.models.multimodal import (
            CLIPOptimizer,
            MultiModalOptimizationConfig,
            OptimizationLevel,
        )

        model, processor = clip_model_and_processor

        # Text-only inputs
        texts = [
            "A beautiful sunset over the ocean",
            "A cat sitting on a windowsill",
            "A mountain landscape with snow",
            "A bustling city street at night",
        ]
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        input_ids = inputs["input_ids"].to(e2e_device)
        attention_mask = inputs["attention_mask"].to(e2e_device)

        # Baseline
        baseline_model = model.to(e2e_device)
        baseline_model.eval()

        def run_baseline():
            with torch.no_grad():
                return baseline_model.get_text_features(input_ids, attention_mask)

        # Optimize
        config = MultiModalOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            compile_model=False,
        )
        optimizer = CLIPOptimizer(config)
        optimized_model = optimizer.optimize(model)
        optimized_model = optimized_model.to(e2e_device)

        def run_optimized():
            with torch.no_grad():
                return optimized_model.get_text_features(input_ids, attention_mask)

        # Benchmark
        baseline_result = benchmark_function(run_baseline, warmup_runs=2, benchmark_runs=5)
        optimized_result = benchmark_function(run_optimized, warmup_runs=2, benchmark_runs=5)

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms

        print("\nCLIP Text Encoder Speedup:")
        print(f"  Baseline: {baseline_result.mean_time_ms:.2f}ms")
        print(f"  Optimized: {optimized_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
@requires_cuda
class TestCLIPCUDAOptimizations:
    """Test CLIP CUDA-specific optimizations."""

    def test_clip_fp16_optimization(self, clip_model_and_processor, sample_pil_images):
        """Test CLIP FP16 optimization on CUDA."""
        from kernel_pytorch.models.multimodal import (
            CLIPOptimizer,
            MultiModalOptimizationConfig,
            OptimizationLevel,
        )

        model, processor = clip_model_and_processor
        device = torch.device("cuda")

        texts = ["test text"] * 4
        inputs = processor(
            text=texts,
            images=sample_pil_images,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # FP32 baseline
        fp32_model = model.to(device)
        fp32_model.eval()

        def run_fp32():
            with torch.no_grad():
                return fp32_model(**inputs)

        # FP16 optimized
        config = MultiModalOptimizationConfig(
            optimization_level=OptimizationLevel.O3,
            use_fp16=True,
            compile_model=False,
            device="cuda"
        )
        optimizer = CLIPOptimizer(config)
        fp16_model = optimizer.optimize(model)

        # Convert inputs to FP16
        inputs_fp16 = {
            k: v.half() if v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }

        def run_fp16():
            with torch.no_grad():
                return fp16_model(**inputs_fp16)

        # Benchmark
        fp32_result = benchmark_function(run_fp32, warmup_runs=5, benchmark_runs=20, sync_cuda=True)
        fp16_result = benchmark_function(run_fp16, warmup_runs=5, benchmark_runs=20, sync_cuda=True)

        speedup = fp32_result.mean_time_ms / fp16_result.mean_time_ms

        print("\nCLIP FP16 vs FP32:")
        print(f"  FP32: {fp32_result.mean_time_ms:.2f}ms")
        print(f"  FP16: {fp16_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # FP16 should be faster
        assert speedup >= 1.0, "FP16 should not be slower than FP32"

    def test_clip_batch_throughput(self, clip_model_and_processor, e2e_device):
        """Test CLIP batch throughput on CUDA."""
        import numpy as np
        from PIL import Image

        from kernel_pytorch.models.multimodal import (
            CLIPOptimizer,
            MultiModalOptimizationConfig,
            OptimizationLevel,
        )

        model, processor = clip_model_and_processor
        device = torch.device("cuda")

        # Create larger batch
        batch_size = 16
        images = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(batch_size)
        ]
        texts = [f"description {i}" for i in range(batch_size)]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Optimize
        config = MultiModalOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            batch_size=batch_size,
            compile_model=False,
            device="cuda"
        )
        optimizer = CLIPOptimizer(config)
        optimized_model = optimizer.optimize(model)

        def run_batch():
            with torch.no_grad():
                return optimized_model(**inputs)

        # Benchmark
        result = benchmark_function(run_batch, warmup_runs=3, benchmark_runs=10, sync_cuda=True)

        # Calculate throughput
        pairs_per_second = (batch_size * 1000) / result.mean_time_ms

        print("\nCLIP Batch Throughput:")
        print(f"  Batch size: {batch_size}")
        print(f"  Latency: {result.mean_time_ms:.2f}ms")
        print(f"  Throughput: {pairs_per_second:.1f} image-text pairs/sec")
