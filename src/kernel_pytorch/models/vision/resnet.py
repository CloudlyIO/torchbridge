"""
ResNet optimization for efficient inference.

This module provides optimizations for ResNet models including:
- Operator fusion (Conv+BN+ReLU)
- Memory layout optimization
- Batch inference optimization
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from .base import (
    BaseVisionOptimizer,
    VisionOptimizationConfig,
    VisionModelType,
    OptimizationLevel,
    count_parameters,
    estimate_model_memory,
)


class ResNetOptimizer(BaseVisionOptimizer):
    """Optimizer for ResNet models."""

    def __init__(self, config: Optional[VisionOptimizationConfig] = None):
        """Initialize ResNet optimizer.

        Args:
            config: Optimization configuration
        """
        if config is None:
            config = VisionOptimizationConfig(model_type=VisionModelType.RESNET)
        elif config.model_type != VisionModelType.RESNET:
            config.model_type = VisionModelType.RESNET

        super().__init__(config)

    def optimize(self, model: nn.Module) -> nn.Module:
        """Optimize ResNet model for inference.

        Args:
            model: ResNet model to optimize

        Returns:
            Optimized ResNet model
        """
        model.eval()  # Set to eval mode

        # Apply cuDNN optimizations
        self.apply_cudnn_optimization()

        # Apply operator fusion (Conv+BN+ReLU)
        if self.config.enable_fusion:
            model = self.apply_operator_fusion(model)

        # Apply memory format optimization
        model = self.apply_memory_format_optimization(model)

        # Apply precision optimization
        model = self.apply_precision_optimization(model)

        # Move to device
        model = model.to(self.device)

        # Apply torch.compile
        model = self.apply_compilation(model)

        return model

    def optimize_batch_inference(
        self,
        model: nn.Module,
        images: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Run optimized batch inference.

        Args:
            model: Optimized model
            images: Input images tensor (B, C, H, W)
            batch_size: Batch size for inference (default: config.batch_size)

        Returns:
            Model predictions
        """
        batch_size = batch_size or self.config.batch_size

        # Ensure correct memory format
        if self.config.channels_last:
            images = images.to(memory_format=torch.channels_last)

        # Ensure correct device
        images = images.to(self.device)

        # Ensure correct precision
        if self.config.use_fp16:
            images = images.half()
        elif self.config.use_bf16:
            images = images.bfloat16()

        # Run inference
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.config.use_fp16 or self.config.use_bf16
        ):
            if images.size(0) <= batch_size:
                # Single batch
                outputs = model(images)
            else:
                # Multiple batches
                outputs = []
                for i in range(0, images.size(0), batch_size):
                    batch = images[i:i + batch_size]
                    batch_output = model(batch)
                    outputs.append(batch_output)
                outputs = torch.cat(outputs, dim=0)

        return outputs


def create_resnet_optimizer(
    model_name: str = "resnet50",
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    batch_size: int = 32,
    device: str = "cuda",
    **kwargs
) -> tuple[nn.Module, ResNetOptimizer]:
    """Create and optimize a ResNet model.

    Args:
        model_name: ResNet variant ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        optimization_level: Optimization level
        batch_size: Batch size for inference
        device: Device for inference
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    # Import torchvision
    try:
        import torchvision.models as models
    except ImportError:
        raise ImportError("torchvision is required for ResNet models. Install with: pip install torchvision")

    # Create model
    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        raise ValueError(f"Unknown ResNet variant: {model_name}")

    model = model_fn(weights="DEFAULT")

    # Create config
    config = VisionOptimizationConfig.from_optimization_level(
        optimization_level,
        model_type=VisionModelType.RESNET,
        batch_size=batch_size,
        device=device,
        **kwargs
    )

    # Create optimizer
    optimizer = ResNetOptimizer(config)

    # Optimize model
    model = optimizer.optimize(model)

    return model, optimizer


class ResNetBenchmark:
    """Benchmark ResNet models."""

    def __init__(self, model: nn.Module, optimizer: ResNetOptimizer):
        """Initialize benchmark.

        Args:
            model: ResNet model
            optimizer: ResNet optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.device = optimizer.device

    def benchmark_inference(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        image_size: int = 224,
    ) -> Dict[str, float]:
        """Benchmark inference performance.

        Args:
            batch_size: Batch size for inference
            num_iterations: Number of iterations to benchmark
            warmup_iterations: Number of warmup iterations
            image_size: Input image size

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Create dummy input
        dummy_input = torch.randn(
            batch_size, 3, image_size, image_size,
            device=self.device
        )

        if self.optimizer.config.channels_last:
            dummy_input = dummy_input.to(memory_format=torch.channels_last)

        if self.optimizer.config.use_fp16:
            dummy_input = dummy_input.half()
        elif self.optimizer.config.use_bf16:
            dummy_input = dummy_input.bfloat16()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        time_per_batch = total_time / num_iterations
        throughput = batch_size / time_per_batch

        return {
            "total_time_seconds": total_time,
            "time_per_batch_seconds": time_per_batch,
            "time_per_image_ms": (time_per_batch / batch_size) * 1000,
            "throughput_images_per_second": throughput,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information
        """
        total_params, trainable_params = count_parameters(self.model)

        memory_estimate = estimate_model_memory(
            self.model,
            batch_size=self.optimizer.config.batch_size,
            input_size=(3, 224, 224),
            precision=self.optimizer._get_precision_string(),
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_estimate": memory_estimate,
            "optimization_summary": self.optimizer.get_optimization_summary(),
        }


# Pre-configured optimizers for common ResNet variants
def create_resnet50_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[nn.Module, ResNetOptimizer]:
    """Create optimized ResNet-50.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_resnet_optimizer("resnet50", optimization_level, **kwargs)


def create_resnet152_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[nn.Module, ResNetOptimizer]:
    """Create optimized ResNet-152.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_resnet_optimizer("resnet152", optimization_level, **kwargs)
