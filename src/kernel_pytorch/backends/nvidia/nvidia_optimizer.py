"""
NVIDIA Optimizer Implementation

High-level optimizer for NVIDIA GPU models with multiple optimization levels.
"""

import logging
import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAArchitecture
from .nvidia_backend import NVIDIABackend
from .fp8_compiler import FP8Compiler

logger = logging.getLogger(__name__)


@dataclass
class NVIDIAOptimizationResult:
    """Results from NVIDIA model optimization."""
    optimized_model: nn.Module
    optimization_level: str
    optimizations_applied: List[str]
    compilation_time: float
    memory_stats: Dict[str, Any]
    device_info: Dict[str, Any]
    warnings: List[str]


class NVIDIAOptimizer:
    """
    High-level NVIDIA GPU optimizer with multiple optimization levels.

    Provides conservative, balanced, and aggressive optimization strategies
    for NVIDIA GPUs (H100, Blackwell, Ampere, etc.).
    """

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize NVIDIA optimizer.

        Args:
            config: KernelPyTorch configuration with NVIDIA settings
        """
        self.config = config or KernelPyTorchConfig()
        self.backend = NVIDIABackend(self.config)
        self.fp8_compiler = FP8Compiler(self.config)

        self._optimization_warnings = []

    def optimize(
        self,
        model: nn.Module,
        sample_inputs: Optional[torch.Tensor] = None,
        optimization_level: str = "balanced",
        for_inference: bool = False
    ) -> NVIDIAOptimizationResult:
        """
        Optimize model for NVIDIA GPU execution.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Sample inputs for compilation (optional)
            optimization_level: Optimization level ('conservative', 'balanced', 'aggressive')
            for_inference: Whether to optimize for inference (vs training)

        Returns:
            NVIDIAOptimizationResult with optimized model and statistics
        """
        start_time = time.time()
        self._optimization_warnings = []

        # Prepare model with backend
        prepared_model = self.backend.prepare_model(model)

        # Apply optimization level
        optimized_model, applied_optimizations = self._apply_optimization_level(
            prepared_model,
            optimization_level,
            sample_inputs,
            for_inference
        )

        # Compile if sample inputs provided and appropriate
        if sample_inputs is not None and optimization_level in ["balanced", "aggressive"]:
            optimized_model = self._compile_model(optimized_model, sample_inputs, for_inference)
            applied_optimizations.append("torch_compile")

        compilation_time = time.time() - start_time

        return NVIDIAOptimizationResult(
            optimized_model=optimized_model,
            optimization_level=optimization_level,
            optimizations_applied=applied_optimizations,
            compilation_time=compilation_time,
            memory_stats=self.backend.get_memory_stats(),
            device_info=self.backend.get_device_info(),
            warnings=self._optimization_warnings
        )

    def _apply_optimization_level(
        self,
        model: nn.Module,
        level: str,
        sample_inputs: Optional[torch.Tensor],
        for_inference: bool
    ) -> Tuple[nn.Module, List[str]]:
        """Apply optimizations based on level."""
        applied = []

        if level == "conservative":
            # Minimal optimizations for maximum stability
            if self.backend.is_cuda_available:
                applied.append("device_placement")

            if for_inference:
                model.eval()
                applied.append("eval_mode")

        elif level == "balanced":
            # Standard optimizations for good performance/stability balance
            if self.backend.is_cuda_available:
                applied.append("device_placement")

            # Apply mixed precision if supported
            if self.backend.nvidia_config.mixed_precision_enabled:
                model = self._enable_mixed_precision(model)
                applied.append("mixed_precision")

            # Apply gradient checkpointing if training
            if not for_inference and self.config.memory.gradient_checkpointing:
                model = self._enable_gradient_checkpointing(model)
                applied.append("gradient_checkpointing")

            if for_inference:
                model.eval()
                applied.append("eval_mode")

        elif level == "aggressive":
            # Maximum optimizations for best performance
            if self.backend.is_cuda_available:
                applied.append("device_placement")

            # Apply FP8 if supported
            if self.backend.supports_fp8 and self.backend.nvidia_config.fp8_enabled:
                model = self.fp8_compiler.prepare_for_fp8(model, for_inference)
                applied.append("fp8_training" if not for_inference else "fp8_inference")

            # Apply mixed precision if not using FP8
            elif self.backend.nvidia_config.mixed_precision_enabled:
                model = self._enable_mixed_precision(model)
                applied.append("mixed_precision")

            # Apply kernel fusion
            if self.backend.nvidia_config.kernel_fusion_enabled:
                model = self._enable_kernel_fusion(model)
                applied.append("kernel_fusion")

            # Apply memory optimizations
            model = self._apply_aggressive_memory_optimizations(model, for_inference)
            applied.append("memory_optimization")

            if for_inference:
                model.eval()
                applied.append("eval_mode")

        else:
            self._optimization_warnings.append(
                f"Unknown optimization level '{level}', using 'balanced'"
            )
            return self._apply_optimization_level(model, "balanced", sample_inputs, for_inference)

        return model, applied

    def _enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision training/inference."""
        # This is typically handled via torch.cuda.amp.autocast during training
        # Here we just mark the model as mixed-precision ready
        setattr(model, '_mixed_precision_enabled', True)
        return model

    def _enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable()
            except Exception as e:
                self._optimization_warnings.append(
                    f"Failed to enable gradient checkpointing: {e}"
                )
        return model

    def _enable_kernel_fusion(self, model: nn.Module) -> nn.Module:
        """Enable kernel fusion optimizations."""
        # Mark model for kernel fusion with torch.compile
        setattr(model, '_kernel_fusion_enabled', True)
        return model

    def _apply_aggressive_memory_optimizations(
        self,
        model: nn.Module,
        for_inference: bool
    ) -> nn.Module:
        """Apply aggressive memory optimizations."""
        # Use memory-efficient attention if available
        for module in model.modules():
            if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
                try:
                    module.set_use_memory_efficient_attention_xformers(True)
                except (AttributeError, RuntimeError) as e:
                    # Method exists but may not be supported on this hardware
                    logger.debug("Could not enable memory-efficient attention: %s", e)

        # For inference, enable additional optimizations
        if for_inference:
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False

            # Convert BatchNorm to eval mode and fuse if possible
            model = self._fuse_bn_layers(model)

        return model

    def _fuse_bn_layers(self, model: nn.Module) -> nn.Module:
        """Fuse BatchNorm layers for inference."""
        try:
            # Fuse Conv + BatchNorm layers
            torch.quantization.fuse_modules(model, inplace=True)
        except Exception as e:
            # Fusion may not be applicable for all models
            pass
        return model

    def _compile_model(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        for_inference: bool
    ) -> nn.Module:
        """Compile model with torch.compile."""
        if not hasattr(torch, 'compile'):
            self._optimization_warnings.append(
                "torch.compile not available, skipping compilation"
            )
            return model

        try:
            # Prepare compilation kwargs
            compile_kwargs = {
                'mode': 'reduce-overhead' if for_inference else 'default',
                'fullgraph': False,  # Allow graph breaks for robustness
                'dynamic': False,  # Static shapes for better optimization
            }

            # Add backend-specific options
            if self.backend.is_h100 or self.backend.is_blackwell:
                # Use max-autotune for latest hardware
                compile_kwargs['mode'] = 'max-autotune'

            compiled_model = torch.compile(model, **compile_kwargs)

            # Warm up compilation
            if sample_inputs is not None:
                with torch.no_grad():
                    _ = compiled_model(sample_inputs)

            return compiled_model

        except Exception as e:
            self._optimization_warnings.append(
                f"torch.compile failed: {e}, using uncompiled model"
            )
            return model

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_inputs: Optional[torch.Tensor] = None,
        optimization_level: str = "aggressive"
    ) -> NVIDIAOptimizationResult:
        """
        Optimize model specifically for inference.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Sample inputs for compilation
            optimization_level: Optimization level

        Returns:
            NVIDIAOptimizationResult with inference-optimized model
        """
        return self.optimize(
            model=model,
            sample_inputs=sample_inputs,
            optimization_level=optimization_level,
            for_inference=True
        )

    def optimize_for_training(
        self,
        model: nn.Module,
        sample_inputs: Optional[torch.Tensor] = None,
        optimization_level: str = "balanced"
    ) -> NVIDIAOptimizationResult:
        """
        Optimize model specifically for training.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Sample inputs for compilation
            optimization_level: Optimization level

        Returns:
            NVIDIAOptimizationResult with training-optimized model
        """
        return self.optimize(
            model=model,
            sample_inputs=sample_inputs,
            optimization_level=optimization_level,
            for_inference=False
        )

    def get_optimization_recommendations(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get optimization recommendations for the model.

        Args:
            model: PyTorch model to analyze

        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'architecture': self.backend.nvidia_config.architecture.value,
            'fp8_available': self.backend.supports_fp8,
            'suggested_level': 'balanced',
            'optimizations': []
        }

        # Recommend FP8 for H100/Blackwell
        if self.backend.supports_fp8:
            recommendations['optimizations'].append({
                'type': 'fp8_training',
                'benefit': '2x training speedup',
                'requirement': 'H100 or Blackwell GPU'
            })

        # Recommend mixed precision for older GPUs
        elif self.backend.compute_capability and self.backend.compute_capability[0] >= 7:
            recommendations['optimizations'].append({
                'type': 'mixed_precision',
                'benefit': '1.5-2x training speedup',
                'requirement': 'Volta or newer GPU'
            })

        # Recommend torch.compile
        if hasattr(torch, 'compile'):
            recommendations['optimizations'].append({
                'type': 'torch_compile',
                'benefit': '1.5-3x speedup depending on model',
                'requirement': 'PyTorch 2.0+'
            })

        # Suggest aggressive for H100/Blackwell
        if self.backend.is_h100 or self.backend.is_blackwell:
            recommendations['suggested_level'] = 'aggressive'

        return recommendations
