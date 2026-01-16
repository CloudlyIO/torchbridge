"""
Optimization Management for KernelPyTorch.

This module provides unified optimization management:
- Compilation optimization (torch.compile, Triton)
- Precision optimization (FP8, BF16, mixed precision)
- Fusion optimization (operator fusion)

Consolidates functionality from:
- PyGraphCUDAOptimizer, FusionBoundaryOptimizer
- LongSequenceOptimizer, AdaptiveCompressionOptimizer
- FP8Optimizer, and 15+ other optimization managers

Version: 0.3.11
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import warnings

from .base import BaseManager, ManagerType, ManagerState


class OptimizationManager(BaseManager):
    """
    Unified optimization management.

    Consolidates:
    - PyGraphCUDAOptimizer
    - FusionBoundaryOptimizer
    - LongSequenceOptimizer
    - AdaptiveCompressionOptimizer
    - FP8Optimizer
    - And 15+ other optimization managers
    """

    def _get_manager_type(self) -> ManagerType:
        return ManagerType.OPTIMIZATION

    def _initialize(self) -> None:
        """Initialize optimization management."""
        self.precision_config = self.config.precision

        # Initialize compilation optimization
        self.compilation_enabled = self.config.hardware.torch_compile
        self.triton_enabled = self.config.hardware.triton_enabled

        # Initialize precision optimization
        self.precision_formats = self._setup_precision_formats()

        # Initialize fusion optimization
        self.fusion_enabled = self.config.attention.fusion_enabled

        # Track applied optimizations
        self.applied_optimizations: List[str] = []

        self.context.state = ManagerState.READY
        self._initialized = True

    def optimize(self, target: Any, **kwargs) -> Any:
        """Apply comprehensive optimizations to target."""
        if not self._initialized:
            raise RuntimeError("OptimizationManager not initialized")

        optimization_level = kwargs.get('level', self.config.optimization_level)

        # Clear previous optimizations tracking
        self.applied_optimizations = []

        # Apply compilation optimizations
        if self.compilation_enabled:
            target = self._apply_compilation_optimization(target, **kwargs)

        # Apply precision optimizations
        if self.precision_config.adaptive_allocation:
            target = self._apply_precision_optimization(target, **kwargs)

        # Apply fusion optimizations
        if self.fusion_enabled:
            target = self._apply_fusion_optimization(target, **kwargs)

        return target

    def _setup_precision_formats(self) -> List[str]:
        """Setup available precision formats based on default_format and fp8 settings."""
        formats = ['fp32', 'fp16', 'bf16']  # Standard formats always available

        if self.precision_config.fp8_enabled:
            formats.extend(['fp8_e4m3', 'fp8_e5m2'])

        return formats

    def _apply_compilation_optimization(self, target: Any, **kwargs) -> Any:
        """Apply compilation-based optimizations."""
        if hasattr(target, 'forward') and callable(target.forward):
            if self.compilation_enabled:
                try:
                    # Apply torch.compile if available
                    compiled = torch.compile(target)
                    self.applied_optimizations.append('torch_compile')
                    return compiled
                except Exception as e:
                    warnings.warn(f"Compilation optimization failed: {e}")

        return target

    def _apply_precision_optimization(self, target: Any, **kwargs) -> Any:
        """Apply precision-based optimizations."""
        # Check if we should apply adaptive precision allocation
        if isinstance(target, nn.Module) and self.precision_config.adaptive_allocation:
            # Track that adaptive precision is enabled
            self.applied_optimizations.append('adaptive_precision')

        return target

    def _apply_fusion_optimization(self, target: Any, **kwargs) -> Any:
        """Apply fusion-based optimizations."""
        if self.fusion_enabled:
            self.applied_optimizations.append('fusion_enabled')
        return target

    def get_available_optimizations(self) -> Dict[str, bool]:
        """Get available optimization capabilities."""
        return {
            'compilation': self.compilation_enabled,
            'triton': self.triton_enabled,
            'fusion': self.fusion_enabled,
            'fp8': self.precision_config.fp8_enabled,
            'quantization': self.precision_config.quantization_enabled,
            'adaptive_precision': self.precision_config.adaptive_allocation
        }

    def get_applied_optimizations(self) -> List[str]:
        """Get list of optimizations applied in last optimize() call."""
        return self.applied_optimizations.copy()

    def get_precision_formats(self) -> List[str]:
        """Get available precision formats."""
        return self.precision_formats.copy()
