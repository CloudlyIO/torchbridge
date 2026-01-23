"""
CLIP (Contrastive Language-Image Pre-training) optimization.

This module provides optimizations for CLIP models including:
- Vision-language embedding optimization
- Cross-modal attention optimization
- Batch embedding generation
"""

from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
from .base import (
    BaseMultiModalOptimizer,
    MultiModalOptimizationConfig,
    MultiModalType,
    OptimizationLevel,
    ModalityType,
    count_parameters,
    estimate_model_memory,
)


class CLIPOptimizer(BaseMultiModalOptimizer):
    """Optimizer for CLIP models."""

    def __init__(self, config: Optional[MultiModalOptimizationConfig] = None):
        """Initialize CLIP optimizer.

        Args:
            config: Optimization configuration
        """
        if config is None:
            config = MultiModalOptimizationConfig(
                model_type=MultiModalType.CLIP,
                modalities=[ModalityType.VISION, ModalityType.TEXT]
            )
        elif config.model_type != MultiModalType.CLIP:
            config.model_type = MultiModalType.CLIP

        super().__init__(config)

    def optimize(self, model: Any) -> Any:
        """Optimize CLIP model for inference.

        Args:
            model: CLIP model or processor to optimize

        Returns:
            Optimized CLIP model
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

        # Apply torch.compile
        if self.config.compile_model:
            try:
                # Compile vision and text encoders separately
                if hasattr(model, "vision_model"):
                    model.vision_model = torch.compile(model.vision_model, mode="reduce-overhead")
                if hasattr(model, "text_model"):
                    model.text_model = torch.compile(model.text_model, mode="reduce-overhead")
                self.optimizations_applied.append("torch_compile")
            except Exception:
                pass

        return model

    def encode_images(
        self,
        model: Any,
        images: torch.Tensor,
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """Encode images to embeddings.

        Args:
            model: CLIP model
            images: Image tensors (B, C, H, W)
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings

        Returns:
            Image embeddings (B, D)
        """
        batch_size = batch_size or self.config.batch_size

        # Ensure correct device
        images = images.to(self.device)

        # Ensure correct precision
        if self.config.use_fp16:
            images = images.half()
        elif self.config.use_bf16:
            images = images.bfloat16()

        # Encode in batches
        embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.config.use_fp16 or self.config.use_bf16
        ):
            for i in range(0, images.size(0), batch_size):
                batch = images[i:i + batch_size]

                if hasattr(model, "get_image_features"):
                    batch_emb = model.get_image_features(batch)
                elif hasattr(model, "encode_image"):
                    batch_emb = model.encode_image(batch)
                else:
                    raise AttributeError("Model must have get_image_features or encode_image method")

                if normalize:
                    batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)

                embeddings.append(batch_emb)

        return torch.cat(embeddings, dim=0)

    def encode_text(
        self,
        model: Any,
        text_inputs: Union[torch.Tensor, List[str]],
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """Encode text to embeddings.

        Args:
            model: CLIP model
            text_inputs: Text inputs (tokenized or strings)
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings

        Returns:
            Text embeddings (B, D)
        """
        batch_size = batch_size or self.config.batch_size

        # Handle string inputs
        if isinstance(text_inputs, list) and isinstance(text_inputs[0], str):
            # Need tokenizer - assume it's available
            try:
                from transformers import CLIPProcessor
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                text_inputs = processor(
                    text=text_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )["input_ids"]
            except Exception:
                raise ValueError("String inputs require transformers library with CLIPProcessor")

        # Ensure correct device
        text_inputs = text_inputs.to(self.device)

        # Encode in batches
        embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.config.use_fp16 or self.config.use_bf16
        ):
            for i in range(0, text_inputs.size(0), batch_size):
                batch = text_inputs[i:i + batch_size]

                if hasattr(model, "get_text_features"):
                    batch_emb = model.get_text_features(batch)
                elif hasattr(model, "encode_text"):
                    batch_emb = model.encode_text(batch)
                else:
                    raise AttributeError("Model must have get_text_features or encode_text method")

                if normalize:
                    batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)

                embeddings.append(batch_emb)

        return torch.cat(embeddings, dim=0)

    def compute_similarity(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute similarity between image and text embeddings.

        Args:
            image_embeddings: Image embeddings (N, D)
            text_embeddings: Text embeddings (M, D)
            temperature: Temperature scaling factor

        Returns:
            Similarity matrix (N, M)
        """
        # Normalize if not already normalized
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = (image_embeddings @ text_embeddings.T) / temperature

        return similarity


def create_clip_optimizer(
    model_name: str = "openai/clip-vit-base-patch32",
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    batch_size: int = 32,
    device: str = "cuda",
    **kwargs
) -> tuple[Any, CLIPOptimizer]:
    """Create and optimize a CLIP model.

    Args:
        model_name: CLIP model name or path
        optimization_level: Optimization level
        batch_size: Batch size for encoding
        device: Device for inference
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    # Import transformers
    try:
        from transformers import CLIPModel
    except ImportError:
        raise ImportError(
            "transformers is required for CLIP models. "
            "Install with: pip install transformers"
        )

    # Load model
    model = CLIPModel.from_pretrained(model_name)

    # Create config
    config = MultiModalOptimizationConfig.from_optimization_level(
        optimization_level,
        model_type=MultiModalType.CLIP,
        batch_size=batch_size,
        device=device,
        modalities=[ModalityType.VISION, ModalityType.TEXT],
        **kwargs
    )

    # Create optimizer
    optimizer = CLIPOptimizer(config)

    # Optimize model
    model = optimizer.optimize(model)

    return model, optimizer


class CLIPBenchmark:
    """Benchmark CLIP models."""

    def __init__(self, model: Any, optimizer: CLIPOptimizer):
        """Initialize benchmark.

        Args:
            model: CLIP model
            optimizer: CLIP optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.device = optimizer.device

    def benchmark_image_encoding(
        self,
        batch_size: int = 32,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        image_size: int = 224,
    ) -> Dict[str, float]:
        """Benchmark image encoding performance.

        Args:
            batch_size: Batch size for encoding
            num_iterations: Number of iterations to benchmark
            warmup_iterations: Number of warmup iterations
            image_size: Input image size

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Create dummy images
        dummy_images = torch.randn(
            batch_size, 3, image_size, image_size,
            device=self.device
        )

        if self.optimizer.config.use_fp16:
            dummy_images = dummy_images.half()
        elif self.optimizer.config.use_bf16:
            dummy_images = dummy_images.bfloat16()

        # Warmup
        for _ in range(warmup_iterations):
            _ = self.optimizer.encode_images(self.model, dummy_images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.optimizer.encode_images(self.model, dummy_images)

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

    def benchmark_text_encoding(
        self,
        batch_size: int = 32,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        sequence_length: int = 77,
    ) -> Dict[str, float]:
        """Benchmark text encoding performance.

        Args:
            batch_size: Batch size for encoding
            num_iterations: Number of iterations to benchmark
            warmup_iterations: Number of warmup iterations
            sequence_length: Text sequence length

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Create dummy text inputs
        dummy_text = torch.randint(
            0, 49407,  # CLIP vocab size
            (batch_size, sequence_length),
            device=self.device
        )

        # Warmup
        for _ in range(warmup_iterations):
            _ = self.optimizer.encode_text(self.model, dummy_text)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.optimizer.encode_text(self.model, dummy_text)

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
            "time_per_text_ms": (time_per_batch / batch_size) * 1000,
            "throughput_texts_per_second": throughput,
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
            input_sizes={
                "vision": (3, 224, 224),
                "text": (77,),
            },
            precision=self.optimizer._get_precision_string(),
        )

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_estimate": memory_estimate,
            "optimization_summary": self.optimizer.get_optimization_summary(),
        }

        # Get component parameters if available
        if hasattr(self.model, "vision_model"):
            vision_params = sum(p.numel() for p in self.model.vision_model.parameters())
            info["vision_parameters"] = vision_params

        if hasattr(self.model, "text_model"):
            text_params = sum(p.numel() for p in self.model.text_model.parameters())
            info["text_parameters"] = text_params

        return info


# Pre-configured optimizers for common CLIP variants
def create_clip_vit_b_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, CLIPOptimizer]:
    """Create optimized CLIP ViT-B/32.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_clip_optimizer(
        "openai/clip-vit-base-patch32",
        optimization_level,
        **kwargs
    )


def create_clip_vit_l_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, CLIPOptimizer]:
    """Create optimized CLIP ViT-L/14.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_clip_optimizer(
        "openai/clip-vit-large-patch14",
        optimization_level,
        **kwargs
    )
