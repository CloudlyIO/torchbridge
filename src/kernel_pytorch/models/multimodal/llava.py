"""
LLaVA (Large Language and Vision Assistant) optimization.

This module provides optimizations for LLaVA models including:
- Vision-language instruction following optimization
- Cross-modal fusion optimization
- Memory-efficient inference
"""

from typing import Any

import torch

from .base import (
    BaseMultiModalOptimizer,
    ModalityType,
    MultiModalOptimizationConfig,
    MultiModalType,
    OptimizationLevel,
    count_parameters,
)


class LLaVAOptimizer(BaseMultiModalOptimizer):
    """Optimizer for LLaVA models."""

    def __init__(self, config: MultiModalOptimizationConfig | None = None):
        """Initialize LLaVA optimizer.

        Args:
            config: Optimization configuration
        """
        if config is None:
            config = MultiModalOptimizationConfig(
                model_type=MultiModalType.LLAVA,
                modalities=[ModalityType.VISION, ModalityType.TEXT]
            )
        elif config.model_type != MultiModalType.LLAVA:
            config.model_type = MultiModalType.LLAVA

        super().__init__(config)

    def optimize(self, model: Any) -> Any:
        """Optimize LLaVA model for inference.

        Args:
            model: LLaVA model to optimize

        Returns:
            Optimized LLaVA model
        """
        # Apply cuDNN optimizations
        self.apply_cudnn_optimization()

        # Move to device
        model = model.to(self.device)

        # Apply precision optimization
        if self.config.use_fp16:
            model = model.to(torch.float16)
            self.optimizations_applied.append("fp16")
        elif self.config.use_bf16:
            model = model.to(torch.bfloat16)
            self.optimizations_applied.append("bf16")

        # Apply gradient checkpointing if enabled
        if self.config.enable_gradient_checkpointing:
            model = self.apply_gradient_checkpointing(model)

        # Apply attention slicing if enabled
        if self.config.enable_attention_slicing:
            self.apply_attention_slicing(model)

        return model

    def apply_attention_slicing(self, model: Any) -> None:
        """Apply attention slicing for memory efficiency.

        Args:
            model: Model to optimize
        """
        # Check if model supports attention slicing
        if hasattr(model, "enable_attention_slicing"):
            slice_size = self.config.attention_slice_size or "auto"
            model.enable_attention_slicing(slice_size)
            self.optimizations_applied.append("attention_slicing")

    def generate(
        self,
        model: Any,
        images: torch.Tensor,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> list[str]:
        """Generate text responses from images and prompts.

        Args:
            model: LLaVA model
            images: Image tensors (B, C, H, W)
            prompts: Text prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation arguments

        Returns:
            Generated text responses
        """
        # Ensure correct device
        images = images.to(self.device)

        # Ensure correct precision
        if self.config.use_fp16:
            images = images.half()
        elif self.config.use_bf16:
            images = images.bfloat16()

        # Generate
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.config.use_fp16 or self.config.use_bf16
        ):
            if hasattr(model, "generate"):
                outputs = model.generate(
                    images=images,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
            else:
                raise AttributeError("Model must have generate method")

        return outputs


def create_llava_optimizer(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    batch_size: int = 1,
    device: str = "cuda",
    **kwargs
) -> tuple[Any, LLaVAOptimizer]:
    """Create and optimize a LLaVA model.

    Args:
        model_name: LLaVA model name or path
        optimization_level: Optimization level
        batch_size: Batch size for generation
        device: Device for inference
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    # Import transformers
    try:
        from transformers import LlavaForConditionalGeneration
    except ImportError:
        raise ImportError(
            "transformers is required for LLaVA models. "
            "Install with: pip install transformers"
        ) from None

    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Create config
    config = MultiModalOptimizationConfig.from_optimization_level(
        optimization_level,
        model_type=MultiModalType.LLAVA,
        batch_size=batch_size,
        device=device,
        modalities=[ModalityType.VISION, ModalityType.TEXT],
        **kwargs
    )

    # Create optimizer
    optimizer = LLaVAOptimizer(config)

    # Optimize model
    model = optimizer.optimize(model)

    return model, optimizer


class LLaVABenchmark:
    """Benchmark LLaVA models."""

    def __init__(self, model: Any, optimizer: LLaVAOptimizer):
        """Initialize benchmark.

        Args:
            model: LLaVA model
            optimizer: LLaVA optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.device = optimizer.device

    def benchmark_generation(
        self,
        num_iterations: int = 10,
        max_new_tokens: int = 256,
        image_size: int = 336,
    ) -> dict[str, float]:
        """Benchmark generation performance.

        Args:
            num_iterations: Number of iterations to benchmark
            max_new_tokens: Maximum tokens to generate
            image_size: Input image size

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Create dummy inputs
        dummy_images = torch.randn(
            1, 3, image_size, image_size,
            device=self.device
        )

        if self.optimizer.config.use_fp16:
            dummy_images = dummy_images.half()
        elif self.optimizer.config.use_bf16:
            dummy_images = dummy_images.bfloat16()

        dummy_prompt = ["Describe this image in detail."]

        # Warmup (1 iteration)
        _ = self.optimizer.generate(
            self.model,
            dummy_images,
            dummy_prompt,
            max_new_tokens=max_new_tokens
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.optimizer.generate(
                self.model,
                dummy_images,
                dummy_prompt,
                max_new_tokens=max_new_tokens
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        time_per_generation = total_time / num_iterations

        return {
            "total_time_seconds": total_time,
            "time_per_generation_seconds": time_per_generation,
            "num_iterations": num_iterations,
            "max_new_tokens": max_new_tokens,
            "generations_per_minute": 60 / time_per_generation,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information
        """
        total_params, trainable_params = count_parameters(self.model)

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "optimization_summary": self.optimizer.get_optimization_summary(),
        }

        # Get component parameters if available
        if hasattr(self.model, "vision_tower"):
            vision_params = sum(p.numel() for p in self.model.vision_tower.parameters())
            info["vision_parameters"] = vision_params

        if hasattr(self.model, "language_model"):
            llm_params = sum(p.numel() for p in self.model.language_model.parameters())
            info["language_model_parameters"] = llm_params

        return info


# Pre-configured optimizers for common LLaVA variants
def create_llava_7b_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, LLaVAOptimizer]:
    """Create optimized LLaVA-1.5-7B.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_llava_optimizer(
        "llava-hf/llava-1.5-7b-hf",
        optimization_level,
        **kwargs
    )


def create_llava_13b_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, LLaVAOptimizer]:
    """Create optimized LLaVA-1.5-13B.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_llava_optimizer(
        "llava-hf/llava-1.5-13b-hf",
        optimization_level,
        **kwargs
    )
