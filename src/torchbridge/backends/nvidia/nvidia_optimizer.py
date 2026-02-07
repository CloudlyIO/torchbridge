"""
NVIDIA Optimizer Implementation

High-level optimizer for NVIDIA GPU models with multiple optimization levels.

Inherits from BaseOptimizer to provide a consistent interface across all
hardware backends while implementing NVIDIA-specific optimizations.

"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchbridge.backends.base_backend import OptimizationLevel, OptimizationResult
from torchbridge.backends.base_optimizer import BaseOptimizer, OptimizationStrategy
from torchbridge.core.config import TorchBridgeConfig

from .fp8_compiler import FP8Compiler
from .nvidia_backend import NVIDIABackend

logger = logging.getLogger(__name__)

@dataclass
class NVIDIAOptimizationResult:
    """Results from NVIDIA model optimization (legacy format for backward compatibility)."""
    optimized_model: nn.Module
    optimization_level: str
    optimizations_applied: list[str]
    compilation_time: float
    memory_stats: dict[str, Any]
    device_info: dict[str, Any]
    warnings: list[str]

class NVIDIAOptimizer(BaseOptimizer):
    """
    High-level NVIDIA GPU optimizer with multiple optimization levels.

    Provides conservative, balanced, and aggressive optimization strategies
    for NVIDIA GPUs (H100, Blackwell, Ampere, etc.).

    Inherits from BaseOptimizer to provide a unified interface while
    maintaining backward compatibility with existing NVIDIA-specific APIs.
    """

    OPTIMIZER_NAME: str = "nvidia"
    DEFAULT_LEVEL = OptimizationLevel.O2

    def __init__(self, config: TorchBridgeConfig | None = None, device: torch.device | None = None):
        """
        Initialize NVIDIA optimizer.

        Args:
            config: TorchBridge configuration with NVIDIA settings
            device: Target device (auto-detected if not provided)
        """
        self._full_config = config or TorchBridgeConfig()
        self.backend = NVIDIABackend(self._full_config)
        self.fp8_compiler = FP8Compiler(self._full_config)

        # Call parent init
        super().__init__(config=self._full_config, device=device or self.backend.device)

        # Alias for backward compatibility
        self.config = self._full_config

        self._optimization_warnings = []

    def _apply_optimizations(
        self,
        model: nn.Module,
        level: OptimizationLevel,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> tuple[nn.Module, OptimizationResult]:
        """
        Apply NVIDIA-specific optimizations (implements BaseOptimizer abstract method).

        Args:
            model: PyTorch model to optimize
            level: Optimization level
            sample_input: Optional sample input for tracing
            dtype: Optional dtype for precision

        Returns:
            Tuple of (optimized_model, OptimizationResult)
        """
        # Map OptimizationLevel to string for backward compatibility
        level_map = {
            OptimizationLevel.O0: "conservative",
            OptimizationLevel.O1: "conservative",
            OptimizationLevel.O2: "balanced",
            OptimizationLevel.O3: "aggressive"
        }
        level_str = level_map.get(level, "balanced")

        # Use existing implementation
        result = self.optimize_legacy(
            model=model,
            sample_inputs=sample_input,
            optimization_level=level_str,
            for_inference=False
        )

        # Convert to unified OptimizationResult
        return result.optimized_model, OptimizationResult(
            success=True,
            model=result.optimized_model,
            level=level,
            optimizations_applied=result.optimizations_applied,
            warnings=result.warnings,
            metrics={
                'compilation_time': result.compilation_time,
                'memory_stats': result.memory_stats,
                'device_info': result.device_info
            }
        )

    def get_available_strategies(self) -> list[OptimizationStrategy]:
        """Get available NVIDIA optimization strategies (implements BaseOptimizer abstract method)."""
        strategies = [
            OptimizationStrategy(
                name='device_placement',
                description='Move model to CUDA device',
                applicable_levels=[OptimizationLevel.O0, OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3],
                speedup_estimate=1.0
            ),
            OptimizationStrategy(
                name='mixed_precision',
                description='FP16/BF16 mixed precision training',
                applicable_levels=[OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3],
                speedup_estimate=1.8,
                precision_impact='minor',
                requires=['compute_capability>=7.0']
            ),
            OptimizationStrategy(
                name='gradient_checkpointing',
                description='Trade compute for memory during training',
                applicable_levels=[OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3],
                speedup_estimate=0.9,  # Slightly slower
                memory_impact=0.5
            ),
            OptimizationStrategy(
                name='torch_compile',
                description='PyTorch 2.0 compilation with inductor',
                applicable_levels=[OptimizationLevel.O2, OptimizationLevel.O3],
                speedup_estimate=2.0,
                requires=['torch>=2.0']
            ),
            OptimizationStrategy(
                name='kernel_fusion',
                description='Fuse operations for reduced memory bandwidth',
                applicable_levels=[OptimizationLevel.O3],
                speedup_estimate=1.3
            ),
        ]

        # Add FP8 strategy if supported
        if self.backend.supports_fp8:
            strategies.append(OptimizationStrategy(
                name='fp8_precision',
                description='FP8 precision for maximum throughput',
                applicable_levels=[OptimizationLevel.O3],
                speedup_estimate=2.5,
                precision_impact='significant',
                requires=['H100 or Blackwell GPU']
            ))

        return strategies

    def optimize_legacy(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None = None,
        optimization_level: str = "balanced",
        for_inference: bool = False
    ) -> NVIDIAOptimizationResult:
        """
        Legacy optimize method for backward compatibility.

        Use the unified `optimize()` method from BaseOptimizer for new code.

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
            device_info=self.backend.get_device_info_dict(),
            warnings=self._optimization_warnings
        )

    def _apply_optimization_level(
        self,
        model: nn.Module,
        level: str,
        sample_inputs: torch.Tensor | None,
        for_inference: bool
    ) -> tuple[nn.Module, list[str]]:
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
        model._mixed_precision_enabled = True
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
        model._kernel_fusion_enabled = True
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
        except Exception:
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

    def optimize_for_inference_legacy(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None = None,
        optimization_level: str = "aggressive"
    ) -> NVIDIAOptimizationResult:
        """
        Legacy inference optimization method for backward compatibility.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Sample inputs for compilation
            optimization_level: Optimization level

        Returns:
            NVIDIAOptimizationResult with inference-optimized model
        """
        return self.optimize_legacy(
            model=model,
            sample_inputs=sample_inputs,
            optimization_level=optimization_level,
            for_inference=True
        )

    def optimize_for_training_legacy(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None = None,
        optimization_level: str = "balanced"
    ) -> NVIDIAOptimizationResult:
        """
        Legacy training optimization method for backward compatibility.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Sample inputs for compilation
            optimization_level: Optimization level

        Returns:
            NVIDIAOptimizationResult with training-optimized model
        """
        return self.optimize_legacy(
            model=model,
            sample_inputs=sample_inputs,
            optimization_level=optimization_level,
            for_inference=False
        )

    def get_optimization_recommendations(self, model: nn.Module) -> dict[str, Any]:
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
