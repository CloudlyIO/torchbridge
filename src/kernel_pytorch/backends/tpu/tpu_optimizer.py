"""
TPU Model Optimizer

High-level optimizer for TPU models that combines backend preparation,
XLA compilation, and TPU-specific optimizations.
"""

import logging
import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

from kernel_pytorch.core.config import KernelPyTorchConfig, TPUConfig
from .tpu_backend import TPUBackend
from .xla_compiler import XLACompiler
from .tpu_exceptions import TPUOptimizationError, TPUValidationError, raise_or_warn

logger = logging.getLogger(__name__)


@dataclass
class TPUOptimizationResult:
    """Result of TPU optimization process."""
    optimized_model: nn.Module
    backend: TPUBackend
    compiler: XLACompiler
    optimization_time: float
    memory_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class TPUOptimizer:
    """
    High-level TPU model optimizer.

    Combines TPU backend preparation, XLA compilation, and optimization
    strategies to provide the best performance for TPU deployments.
    """

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize TPU optimizer.

        Args:
            config: Optional configuration. If None, creates default config.
        """
        self.config = config or KernelPyTorchConfig()
        self.tpu_config = self.config.hardware.tpu

        # Initialize components
        self.backend = TPUBackend(self.config)
        self.compiler = XLACompiler(self.tpu_config)

        # Optimization tracking
        self._optimization_history = []

    def optimize(self, model: nn.Module,
                sample_inputs: Optional[Union[torch.Tensor, tuple]] = None,
                optimization_level: str = "balanced") -> TPUOptimizationResult:
        """
        Optimize model for TPU execution.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Sample inputs for optimization
            optimization_level: Optimization level ('conservative', 'balanced', 'aggressive')

        Returns:
            Optimization result with optimized model and metrics
        """
        import time
        start_time = time.time()

        logger.info("Starting TPU optimization: level=%s", optimization_level)

        # Step 1: Prepare model for TPU backend
        logger.debug("Step 1: Preparing model for TPU")
        prepared_model = self.backend.prepare_model(model)

        # Step 2: Apply optimization-level specific changes
        logger.debug("Step 2: Applying %s optimizations", optimization_level)
        optimized_model = self._apply_optimization_level(prepared_model, optimization_level)

        # Step 3: Compile with XLA
        logger.debug("Step 3: Compiling with XLA")
        compiled_model = self.compiler.compile_model(optimized_model, sample_inputs)

        # Step 4: Validate optimization
        logger.debug("Step 4: Validating optimization")
        self._validate_optimization(compiled_model, sample_inputs)

        optimization_time = time.time() - start_time

        # Gather metrics
        memory_usage = self.backend.get_memory_stats()
        performance_metrics = {
            'optimization_time': optimization_time,
            'compilation_stats': self.compiler.get_compilation_stats(),
            'optimization_level': optimization_level,
            'tpu_version': self.tpu_config.version.value,
            'world_size': self.backend.world_size
        }

        # Create result
        result = TPUOptimizationResult(
            optimized_model=compiled_model,
            backend=self.backend,
            compiler=self.compiler,
            optimization_time=optimization_time,
            memory_usage=memory_usage,
            performance_metrics=performance_metrics
        )

        # Track optimization
        self._optimization_history.append({
            'timestamp': time.time(),
            'model_type': type(model).__name__,
            'optimization_level': optimization_level,
            'optimization_time': optimization_time
        })

        logger.info("TPU optimization completed: time=%.2fs, level=%s", optimization_time, optimization_level)
        return result

    def _apply_optimization_level(self, model: nn.Module, level: str) -> nn.Module:
        """Apply optimization strategies based on level."""

        if level == "conservative":
            return self._apply_conservative_optimizations(model)
        elif level == "balanced":
            return self._apply_balanced_optimizations(model)
        elif level == "aggressive":
            return self._apply_aggressive_optimizations(model)
        else:
            raise ValueError(f"Unknown optimization level: {level}")

    def _apply_conservative_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply conservative optimizations (safety first)."""

        # Basic TPU preparation only
        model = model.to(self.backend.device)

        # Enable basic mixed precision if configured
        if self.tpu_config.mixed_precision and self.tpu_config.precision == "bfloat16":
            # Only convert Linear layers to bfloat16
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.to(dtype=torch.bfloat16)

        return model

    def _apply_balanced_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply balanced optimizations (performance + safety)."""

        # Start with conservative optimizations
        model = self._apply_conservative_optimizations(model)

        # Add gradient checkpointing if enabled
        if self.tpu_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        # Apply layer fusion for common patterns
        model = self._apply_layer_fusion(model)

        # Optimize attention layers if present
        model = self._optimize_attention_layers(model)

        return model

    def _apply_aggressive_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply aggressive optimizations (maximum performance)."""

        # Start with balanced optimizations
        model = self._apply_balanced_optimizations(model)

        # Additional aggressive optimizations
        if self.tpu_config.mixed_precision:
            # Convert more layers to bfloat16
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding)):
                    module.to(dtype=torch.bfloat16)

        # Apply model-specific optimizations
        model = self._apply_model_specific_optimizations(model)

        return model

    def _apply_layer_fusion(self, model: nn.Module) -> nn.Module:
        """
        Apply layer fusion optimizations.

        Note: XLA compiler automatically fuses operations during compilation,
        including Linear+Activation patterns, conv+batch_norm, and other
        common patterns. No explicit marking is required.

        This method serves as a placeholder for future manual fusion hints
        if needed, but currently relies on XLA's automatic fusion capabilities.
        """
        # XLA handles layer fusion automatically during compilation
        # Common fusions include:
        # - Linear + Activation (ReLU, GELU, SiLU)
        # - Conv + BatchNorm
        # - ElementWise operations
        return model

    def _optimize_attention_layers(self, model: nn.Module) -> nn.Module:
        """Optimize attention layers for TPU."""

        for module in model.modules():
            # Look for attention patterns
            if hasattr(module, 'attention') or 'attention' in module.__class__.__name__.lower():
                # Enable flash attention if available
                if hasattr(module, 'flash_attention'):
                    module.flash_attention = True

                # Optimize for TPU memory layout
                if hasattr(module, 'num_heads'):
                    # Ensure head dimension is divisible by 8 for TPU efficiency
                    head_dim = getattr(module, 'head_dim', None)
                    if head_dim and head_dim % 8 != 0:
                        warnings.warn(
                            f"Attention head dimension {head_dim} not optimal for TPU. "
                            "Consider using dimensions divisible by 8."
                        )

        return model

    def _apply_model_specific_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply optimizations specific to model architectures."""

        model_type = type(model).__name__.lower()

        if 'bert' in model_type or 'transformer' in model_type:
            # Transformer optimizations
            model = self._optimize_transformer_model(model)
        elif 'resnet' in model_type or 'convnet' in model_type:
            # CNN optimizations
            model = self._optimize_cnn_model(model)

        return model

    def _optimize_transformer_model(self, model: nn.Module) -> nn.Module:
        """
        Apply Transformer-specific optimizations.

        Note: For Transformer models, XLA automatically optimizes:
        - Attention computation patterns
        - Matrix multiplications in feed-forward layers
        - Layer normalization operations
        - Residual connections

        For sequence length optimizations, use XLA's dynamic shape support
        which handles variable-length sequences efficiently.
        """
        # XLA handles Transformer optimizations automatically
        # Additional optimizations can be enabled via:
        # - config.enable_xla_dynamic_shapes for variable sequences
        # - config.gradient_checkpointing for memory efficiency
        # - config.mixed_precision for compute efficiency

        return model

    def _optimize_cnn_model(self, model: nn.Module) -> nn.Module:
        """Apply CNN-specific optimizations."""

        # Optimize convolution patterns for TPU
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                # Ensure channel dimensions are optimal for TPU
                if module.in_channels % 8 != 0 or module.out_channels % 8 != 0:
                    warnings.warn(
                        f"Convolution channels ({module.in_channels}→{module.out_channels}) "
                        "not optimal for TPU. Consider using multiples of 8."
                    )

        return model

    def _validate_optimization(self, model: nn.Module,
                             sample_inputs: Optional[Union[torch.Tensor, tuple]]) -> None:
        """Validate that optimization was successful."""

        if sample_inputs is None:
            logger.warning("No sample inputs provided, skipping validation")
            return

        try:
            # Move inputs to TPU
            if isinstance(sample_inputs, torch.Tensor):
                sample_inputs = sample_inputs.to(self.backend.device)
            elif isinstance(sample_inputs, (list, tuple)):
                sample_inputs = tuple(
                    inp.to(self.backend.device) if isinstance(inp, torch.Tensor) else inp
                    for inp in sample_inputs
                )

            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(sample_inputs)

            # Synchronize TPU operations
            self.backend.synchronize()

            logger.info("Optimization validation passed")

        except Exception as e:
            error_msg = f"Optimization validation failed: {str(e)}"
            raise_or_warn(
                error_msg,
                TPUValidationError,
                strict_mode=self.tpu_config.enable_strict_validation,
                logger=logger
            )

    def optimize_for_inference(self, model: nn.Module,
                             sample_inputs: Optional[Union[torch.Tensor, tuple]] = None) -> TPUOptimizationResult:
        """
        Optimize model specifically for inference.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs

        Returns:
            Optimization result
        """
        # Set model to eval mode
        model.eval()

        # Use conservative optimization for stability
        result = self.optimize(model, sample_inputs, optimization_level="conservative")

        # Additional inference optimizations
        result.optimized_model = self.compiler.optimize_for_inference(
            result.optimized_model, sample_inputs
        )

        return result

    def optimize_for_training(self, model: nn.Module,
                            sample_inputs: Optional[Union[torch.Tensor, tuple]] = None) -> TPUOptimizationResult:
        """
        Optimize model specifically for training.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs

        Returns:
            Optimization result
        """
        # Use balanced optimization for training
        result = self.optimize(model, sample_inputs, optimization_level="balanced")

        # Additional training optimizations
        result.optimized_model = self.compiler.optimize_for_training(
            result.optimized_model, sample_inputs
        )

        return result

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        total_optimizations = len(self._optimization_history)

        if total_optimizations == 0:
            return {'total_optimizations': 0}

        total_time = sum(opt['optimization_time'] for opt in self._optimization_history)
        avg_time = total_time / total_optimizations

        model_types = {}
        for opt in self._optimization_history:
            model_type = opt['model_type']
            model_types[model_type] = model_types.get(model_type, 0) + 1

        return {
            'total_optimizations': total_optimizations,
            'total_optimization_time': total_time,
            'average_optimization_time': avg_time,
            'model_types': model_types,
            'backend_stats': self.backend.get_memory_stats(),
            'compiler_stats': self.compiler.get_compilation_stats()
        }

    def clear_cache(self) -> None:
        """Clear all optimization caches."""
        self.backend.clear_cache()
        self.compiler.clear_cache()

    def __repr__(self) -> str:
        """String representation of TPU optimizer."""
        return (
            f"TPUOptimizer(backend={self.backend}, "
            f"compiler={self.compiler}, "
            f"optimizations={len(self._optimization_history)})"
        )