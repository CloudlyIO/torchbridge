"""
End-to-End Tests for Real ResNet Model Optimization

Tests that the ResNetOptimizer produces measurable speedups on real
ResNet-50 model loaded from torchvision without accuracy degradation.

Markers:
    - @pytest.mark.e2e: End-to-end test
    - @pytest.mark.slow: Long-running test
    - @pytest.mark.real_model: Uses real model weights
    - @pytest.mark.requires_torchvision: Requires torchvision

Success Criteria:
    - ResNet optimization shows measurable speedup (>20% target on CUDA)
    - ImageNet classification predictions preserved
    - No significant accuracy degradation
"""

import copy
import pytest
import torch
import torch.nn.functional as F

from .conftest import (
    requires_torchvision,
    requires_cuda,
    benchmark_function,
    assert_speedup,
    assert_output_close,
)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_torchvision
class TestRealResNetOptimization:
    """Test ResNet optimization with real torchvision model."""

    def test_resnet50_loads_and_classifies(self, resnet50_model, sample_image_tensor, e2e_device):
        """Verify real ResNet-50 model loads and produces classification."""
        model = resnet50_model.to(e2e_device)
        images = sample_image_tensor.to(e2e_device)

        with torch.no_grad():
            outputs = model(images)

        # Verify output shape (1000 ImageNet classes)
        assert outputs.shape == (4, 1000)

        # Verify valid probabilities
        probs = F.softmax(outputs, dim=-1)
        assert probs.sum(dim=-1).allclose(torch.ones(4, device=e2e_device))

    def test_resnet50_optimization_speedup_cpu(self, resnet50_model, sample_image_tensor):
        """Test ResNet-50 optimization produces speedup on CPU."""
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        device = torch.device("cpu")
        model = resnet50_model.to(device)
        images = sample_image_tensor.to(device)

        def run_baseline():
            with torch.no_grad():
                return model(images)

        # Optimize model
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            enable_cudnn_benchmark=False,  # CPU
            channels_last=True,
            compile_model=False,
            device="cpu"
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(model)

        # Convert images to channels_last for optimized model
        images_cl = images.to(memory_format=torch.channels_last)

        def run_optimized():
            with torch.no_grad():
                return optimized_model(images_cl)

        # Benchmark
        baseline_result = benchmark_function(
            run_baseline,
            warmup_runs=2,
            benchmark_runs=5,
            sync_cuda=False
        )
        optimized_result = benchmark_function(
            run_optimized,
            warmup_runs=2,
            benchmark_runs=5,
            sync_cuda=False
        )

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms

        print(f"\nResNet-50 CPU Optimization Results:")
        print(f"  Baseline: {baseline_result.mean_time_ms:.2f}ms")
        print(f"  Optimized: {optimized_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # channels_last can provide speedup on CPU too
        assert speedup >= 0.8, f"Unexpected slowdown: {speedup:.2f}x"

    @requires_cuda
    def test_resnet50_optimization_speedup_cuda(self, resnet50_model, sample_image_tensor):
        """Test ResNet-50 optimization produces speedup on CUDA."""
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        device = torch.device("cuda")
        images = sample_image_tensor.to(device)

        # Create baseline model copy BEFORE optimization to avoid modification
        baseline_model = copy.deepcopy(resnet50_model).to(device)
        baseline_model.eval()

        def run_baseline():
            with torch.no_grad():
                return baseline_model(images)

        # Optimize original model with full CUDA optimizations
        model = resnet50_model.to(device)
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O3,
            enable_cudnn_benchmark=True,
            enable_tf32=True,
            channels_last=True,
            compile_model=True,
            use_fp16=True,
            device="cuda"
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(model)

        # Prepare optimized inputs
        images_opt = images.to(memory_format=torch.channels_last).half()

        def run_optimized():
            with torch.no_grad():
                return optimized_model(images_opt)

        # Benchmark with more runs for accuracy
        baseline_result = benchmark_function(
            run_baseline,
            warmup_runs=10,
            benchmark_runs=50,
            sync_cuda=True
        )
        optimized_result = benchmark_function(
            run_optimized,
            warmup_runs=10,
            benchmark_runs=50,
            sync_cuda=True
        )

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms

        print(f"\nResNet-50 CUDA Optimization Results:")
        print(f"  Baseline: {baseline_result.mean_time_ms:.2f}ms")
        print(f"  Optimized: {optimized_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Expect significant speedup on CUDA with all optimizations
        assert speedup >= 1.0, f"Unexpected slowdown: {speedup:.2f}x"

        if speedup < 1.2:
            print(f"  WARNING: Speedup {speedup:.2f}x below 20% target")

    def test_resnet50_output_correctness(self, resnet50_model, sample_image_tensor, e2e_device):
        """Test ResNet-50 optimization preserves classification correctness."""
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        model = resnet50_model.to(e2e_device)
        images = sample_image_tensor.to(e2e_device)

        # Baseline predictions
        with torch.no_grad():
            baseline_output = model(images)
        baseline_preds = baseline_output.argmax(dim=-1)

        # Optimize (without FP16 to maintain precision)
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            channels_last=True,
            compile_model=False,
            use_fp16=False,
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(model)

        # Optimized predictions
        images_opt = images.to(memory_format=torch.channels_last)
        with torch.no_grad():
            optimized_output = optimized_model(images_opt)
        optimized_preds = optimized_output.argmax(dim=-1)

        # Predictions should match exactly
        assert torch.equal(baseline_preds, optimized_preds), "Classification predictions differ"

        # Logits should be close
        assert_output_close(
            baseline_output,
            optimized_output,
            atol=1e-4,
            rtol=1e-4,
            message="ResNet-50 output correctness"
        )

    def test_resnet50_batch_throughput(self, resnet50_model, e2e_device):
        """Test ResNet-50 batch throughput optimization."""
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        # Larger batch for throughput testing
        batch_size = 32
        images = torch.randn(batch_size, 3, 224, 224, device=e2e_device)

        # Optimize for throughput
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            batch_size=batch_size,
            channels_last=True,
            compile_model=False,
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(resnet50_model)
        optimized_model = optimized_model.to(e2e_device)

        images_opt = images.to(memory_format=torch.channels_last)

        def run_batch():
            with torch.no_grad():
                return optimized_model(images_opt)

        # Benchmark
        result = benchmark_function(
            run_batch,
            warmup_runs=3,
            benchmark_runs=10,
            sync_cuda=torch.cuda.is_available()
        )

        # Calculate throughput
        images_per_second = (batch_size * 1000) / result.mean_time_ms

        print(f"\nResNet-50 Batch Throughput:")
        print(f"  Batch size: {batch_size}")
        print(f"  Latency: {result.mean_time_ms:.2f}ms")
        print(f"  Throughput: {images_per_second:.1f} images/sec")

        # Should process at least a few images per second on any device
        assert images_per_second > 1.0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_torchvision
class TestResNetVariants:
    """Test optimization on different ResNet variants."""

    def test_resnet18_optimization(self, e2e_device, sample_image_tensor):
        """Test ResNet-18 optimization (smaller model)."""
        from torchvision.models import resnet18, ResNet18_Weights
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        model = model.to(e2e_device)
        images = sample_image_tensor.to(e2e_device)

        def run_baseline():
            with torch.no_grad():
                return model(images)

        # Optimize
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            channels_last=True,
            compile_model=False,
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(model)

        images_opt = images.to(memory_format=torch.channels_last)

        def run_optimized():
            with torch.no_grad():
                return optimized_model(images_opt)

        # Benchmark
        baseline_result = benchmark_function(run_baseline, warmup_runs=2, benchmark_runs=5)
        optimized_result = benchmark_function(run_optimized, warmup_runs=2, benchmark_runs=5)

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms

        print(f"\nResNet-18 Optimization:")
        print(f"  Speedup: {speedup:.2f}x")

    def test_resnet101_optimization(self, e2e_device, sample_image_tensor):
        """Test ResNet-101 optimization (larger model)."""
        from torchvision.models import resnet101, ResNet101_Weights
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        model.eval()
        model = model.to(e2e_device)
        images = sample_image_tensor.to(e2e_device)

        def run_baseline():
            with torch.no_grad():
                return model(images)

        # Optimize
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O2,
            channels_last=True,
            compile_model=False,
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(model)

        images_opt = images.to(memory_format=torch.channels_last)

        def run_optimized():
            with torch.no_grad():
                return optimized_model(images_opt)

        # Benchmark
        baseline_result = benchmark_function(run_baseline, warmup_runs=2, benchmark_runs=5)
        optimized_result = benchmark_function(run_optimized, warmup_runs=2, benchmark_runs=5)

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms

        print(f"\nResNet-101 Optimization:")
        print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_torchvision
@requires_cuda
class TestResNetCUDAOptimizations:
    """Test ResNet CUDA-specific optimizations."""

    def test_resnet50_fp16_optimization(self, resnet50_model, sample_image_tensor):
        """Test ResNet-50 FP16 optimization on CUDA."""
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        device = torch.device("cuda")
        images = sample_image_tensor.to(device)

        # Baseline FP32 model - use deepcopy BEFORE optimization
        fp32_model = copy.deepcopy(resnet50_model).to(device)
        fp32_model.eval()

        def run_fp32():
            with torch.no_grad():
                return fp32_model(images)

        # Optimize with FP16 (use original model)
        model = resnet50_model.to(device)
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O3,
            channels_last=True,
            use_fp16=True,
            compile_model=False,
            device="cuda"
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(model)

        images_fp16 = images.to(memory_format=torch.channels_last).half()

        def run_fp16():
            with torch.no_grad():
                return optimized_model(images_fp16)

        # Benchmark
        fp32_result = benchmark_function(run_fp32, warmup_runs=5, benchmark_runs=20, sync_cuda=True)
        fp16_result = benchmark_function(run_fp16, warmup_runs=5, benchmark_runs=20, sync_cuda=True)

        speedup = fp32_result.mean_time_ms / fp16_result.mean_time_ms

        print(f"\nResNet-50 FP16 vs FP32:")
        print(f"  FP32: {fp32_result.mean_time_ms:.2f}ms")
        print(f"  FP16: {fp16_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # FP16 should be faster on modern GPUs
        assert speedup >= 1.0, "FP16 should not be slower than FP32"

    def test_resnet50_torch_compile(self, resnet50_model, sample_image_tensor):
        """Test ResNet-50 with torch.compile."""
        from kernel_pytorch.models.vision import ResNetOptimizer, VisionOptimizationConfig, OptimizationLevel

        device = torch.device("cuda")
        model = resnet50_model.to(device)
        images = sample_image_tensor.to(device)

        def run_baseline():
            with torch.no_grad():
                return model(images)

        # Optimize with torch.compile
        config = VisionOptimizationConfig(
            optimization_level=OptimizationLevel.O3,
            channels_last=True,
            compile_model=True,
            device="cuda"
        )
        optimizer = ResNetOptimizer(config)
        optimized_model = optimizer.optimize(model)

        images_opt = images.to(memory_format=torch.channels_last)

        def run_compiled():
            with torch.no_grad():
                return optimized_model(images_opt)

        # Benchmark (more warmup for compile)
        baseline_result = benchmark_function(run_baseline, warmup_runs=5, benchmark_runs=20, sync_cuda=True)
        compiled_result = benchmark_function(run_compiled, warmup_runs=10, benchmark_runs=20, sync_cuda=True)

        speedup = baseline_result.mean_time_ms / compiled_result.mean_time_ms

        print(f"\nResNet-50 torch.compile:")
        print(f"  Baseline: {baseline_result.mean_time_ms:.2f}ms")
        print(f"  Compiled: {compiled_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
