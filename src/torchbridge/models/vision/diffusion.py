"""
Stable Diffusion optimization for efficient image generation.

This module provides optimizations for Stable Diffusion models including:
- UNet optimization
- VAE tiling for memory efficiency
- Attention slicing
- Memory-efficient generation
"""

from typing import Any

import torch

from .base import (
    BaseVisionOptimizer,
    OptimizationLevel,
    VisionModelType,
    VisionOptimizationConfig,
)


class StableDiffusionOptimizer(BaseVisionOptimizer):
    """Optimizer for Stable Diffusion models."""

    def __init__(self, config: VisionOptimizationConfig | None = None):
        """Initialize Stable Diffusion optimizer.

        Args:
            config: Optimization configuration
        """
        if config is None:
            config = VisionOptimizationConfig(model_type=VisionModelType.STABLE_DIFFUSION)
        elif config.model_type != VisionModelType.STABLE_DIFFUSION:
            config.model_type = VisionModelType.STABLE_DIFFUSION

        super().__init__(config)
        self.pipeline = None

    def optimize(self, pipeline: Any) -> Any:
        """Optimize Stable Diffusion pipeline for inference.

        Args:
            pipeline: Stable Diffusion pipeline to optimize

        Returns:
            Optimized pipeline
        """
        self.pipeline = pipeline

        # Apply cuDNN optimizations
        self.apply_cudnn_optimization()

        # Move to device
        pipeline = pipeline.to(self.device)

        # Apply precision optimization
        if self.config.use_fp16:
            pipeline = pipeline.to(torch.float16)
            self.optimizations_applied.append("fp16")
        elif self.config.use_bf16:
            pipeline = pipeline.to(torch.bfloat16)
            self.optimizations_applied.append("bf16")

        # Apply attention slicing
        if self.config.enable_attention_slicing:
            self.apply_attention_slicing_to_pipeline(pipeline)

        # Apply VAE tiling
        if self.config.enable_vae_tiling:
            self.apply_vae_tiling_to_pipeline(pipeline)

        # Enable memory efficient attention if available
        self.enable_memory_efficient_attention(pipeline)

        # Apply xformers if available
        self.enable_xformers(pipeline)

        return pipeline

    def apply_attention_slicing_to_pipeline(self, pipeline: Any) -> None:
        """Apply attention slicing to reduce memory.

        Args:
            pipeline: Diffusion pipeline
        """
        if hasattr(pipeline, "enable_attention_slicing"):
            slice_size = self.config.attention_slice_size or "auto"
            pipeline.enable_attention_slicing(slice_size)
            self.optimizations_applied.append("attention_slicing")

    def apply_vae_tiling_to_pipeline(self, pipeline: Any) -> None:
        """Apply VAE tiling for large images.

        Args:
            pipeline: Diffusion pipeline
        """
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
            self.optimizations_applied.append("vae_tiling")
        elif hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
            self.optimizations_applied.append("vae_tiling")

    def enable_memory_efficient_attention(self, pipeline: Any) -> None:
        """Enable memory efficient attention.

        Args:
            pipeline: Diffusion pipeline
        """
        if hasattr(pipeline, "enable_memory_efficient_attention"):
            try:
                pipeline.enable_memory_efficient_attention()
                self.optimizations_applied.append("memory_efficient_attention")
            except Exception:
                pass  # Not available

    def enable_xformers(self, pipeline: Any) -> None:
        """Enable xformers memory efficient attention.

        Args:
            pipeline: Diffusion pipeline
        """
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                self.optimizations_applied.append("xformers")
            except Exception:
                pass  # xformers not installed

    def generate_optimized(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        **kwargs
    ) -> Any:
        """Generate images with optimized settings.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            height: Image height
            width: Image width
            num_images_per_prompt: Number of images per prompt
            **kwargs: Additional pipeline arguments

        Returns:
            Generated images
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not optimized. Call optimize() first.")

        num_inference_steps = num_inference_steps or self.config.num_inference_steps

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.config.use_fp16 or self.config.use_bf16
        ):
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
                **kwargs
            )

        return output


def create_stable_diffusion_optimizer(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    device: str = "cuda",
    **kwargs
) -> tuple[Any, StableDiffusionOptimizer]:
    """Create and optimize a Stable Diffusion pipeline.

    Args:
        model_id: Hugging Face model ID
        optimization_level: Optimization level
        device: Device for inference
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_pipeline, optimizer)
    """
    # Import diffusers
    try:
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    except ImportError:
        raise ImportError(
            "diffusers is required for Stable Diffusion models. "
            "Install with: pip install diffusers transformers"
        ) from None

    # Load pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    # Use faster scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    # Create config
    config = VisionOptimizationConfig.from_optimization_level(
        optimization_level,
        model_type=VisionModelType.STABLE_DIFFUSION,
        device=device,
        **kwargs
    )

    # Create optimizer
    optimizer = StableDiffusionOptimizer(config)

    # Optimize pipeline
    pipeline = optimizer.optimize(pipeline)

    return pipeline, optimizer


class StableDiffusionBenchmark:
    """Benchmark Stable Diffusion models."""

    def __init__(self, pipeline: Any, optimizer: StableDiffusionOptimizer):
        """Initialize benchmark.

        Args:
            pipeline: Stable Diffusion pipeline
            optimizer: Stable Diffusion optimizer
        """
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.device = optimizer.device

    def benchmark_generation(
        self,
        prompt: str = "A photo of an astronaut riding a horse on mars",
        num_iterations: int = 10,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
    ) -> dict[str, float]:
        """Benchmark image generation performance.

        Args:
            prompt: Text prompt
            num_iterations: Number of iterations to benchmark
            num_inference_steps: Number of denoising steps
            height: Image height
            width: Image width

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Warmup (1 iteration)
        _ = self.optimizer.generate_optimized(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.optimizer.generate_optimized(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        time_per_image = total_time / num_iterations

        return {
            "total_time_seconds": total_time,
            "time_per_image_seconds": time_per_image,
            "num_iterations": num_iterations,
            "num_inference_steps": num_inference_steps,
            "image_size": f"{height}x{width}",
            "images_per_minute": 60 / time_per_image,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information
        """
        info = {
            "optimization_summary": self.optimizer.get_optimization_summary(),
        }

        # Get component info
        if hasattr(self.pipeline, "unet"):
            unet_params = sum(p.numel() for p in self.pipeline.unet.parameters())
            info["unet_parameters"] = unet_params

        if hasattr(self.pipeline, "vae"):
            vae_params = sum(p.numel() for p in self.pipeline.vae.parameters())
            info["vae_parameters"] = vae_params

        if hasattr(self.pipeline, "text_encoder"):
            text_encoder_params = sum(
                p.numel() for p in self.pipeline.text_encoder.parameters()
            )
            info["text_encoder_parameters"] = text_encoder_params

        total_params = sum(
            v for k, v in info.items() if k.endswith("_parameters")
        )
        info["total_parameters"] = total_params

        return info


# Pre-configured optimizers for common Stable Diffusion variants
def create_sd_1_5_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, StableDiffusionOptimizer]:
    """Create optimized Stable Diffusion 1.5.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_pipeline, optimizer)
    """
    return create_stable_diffusion_optimizer(
        "runwayml/stable-diffusion-v1-5",
        optimization_level,
        **kwargs
    )


def create_sd_2_1_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, StableDiffusionOptimizer]:
    """Create optimized Stable Diffusion 2.1.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_pipeline, optimizer)
    """
    return create_stable_diffusion_optimizer(
        "stabilityai/stable-diffusion-2-1",
        optimization_level,
        **kwargs
    )


def create_sdxl_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, StableDiffusionOptimizer]:
    """Create optimized Stable Diffusion XL.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_pipeline, optimizer)
    """
    try:
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
    except ImportError:
        raise ImportError(
            "diffusers is required for Stable Diffusion XL. "
            "Install with: pip install diffusers transformers"
        ) from None

    device = kwargs.get("device", "cuda")

    # Load SDXL pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )

    # Use faster scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    # Create config
    config = VisionOptimizationConfig.from_optimization_level(
        optimization_level,
        model_type=VisionModelType.STABLE_DIFFUSION,
        device=device,
        **kwargs
    )

    # Create optimizer
    optimizer = StableDiffusionOptimizer(config)

    # Optimize pipeline
    pipeline = optimizer.optimize(pipeline)

    return pipeline, optimizer
