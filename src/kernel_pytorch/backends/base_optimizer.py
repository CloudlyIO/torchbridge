"""
Base Optimizer for All Hardware Backends

This module provides the abstract base class for all optimizer implementations,
defining the common interface and shared functionality for model optimization.

Backends (NVIDIA, AMD, TPU, Intel) implement optimizers inheriting from this base
while providing device-specific optimization strategies.

Version: 0.4.8
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .base_backend import OptimizationLevel, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class KernelConfig:
    """
    Configuration for a specific kernel operation.

    Used by kernel optimizers to tune operations for specific hardware.
    """
    algorithm: str = "auto"
    tile_sizes: tuple[int, ...] = (32, 32, 32)
    num_warps: int = 4
    num_stages: int = 2
    use_tensor_cores: bool = True
    memory_format: str = "contiguous"
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm,
            'tile_sizes': self.tile_sizes,
            'num_warps': self.num_warps,
            'num_stages': self.num_stages,
            'use_tensor_cores': self.use_tensor_cores,
            'memory_format': self.memory_format,
            **self.extra_params
        }


@dataclass
class OptimizationStrategy:
    """
    Describes an optimization strategy with its applicability and effects.
    """
    name: str
    description: str
    applicable_levels: list[OptimizationLevel]
    speedup_estimate: float = 1.0  # 1.0 = no speedup
    memory_impact: float = 1.0  # < 1.0 = less memory, > 1.0 = more memory
    precision_impact: str = "none"  # "none", "minor", "significant"
    requires: list[str] = field(default_factory=list)  # Required features

    def is_applicable(self, level: OptimizationLevel) -> bool:
        """Check if this strategy is applicable for the given level."""
        return level in self.applicable_levels


class BaseOptimizer(ABC):
    """
    Abstract base class for model optimizers.

    This class defines the common interface for optimizing PyTorch models
    for different hardware backends.

    Subclasses must implement:
    - _apply_optimizations(model, level) -> Tuple[nn.Module, OptimizationResult]
    - get_available_strategies() -> List[OptimizationStrategy]

    Optional overrides for device-specific behavior:
    - optimize_for_inference()
    - optimize_for_training()
    - get_optimization_recommendations()
    """

    # Optimizer name - should be overridden by subclasses
    OPTIMIZER_NAME: str = "base"

    # Default optimization level
    DEFAULT_LEVEL: OptimizationLevel = OptimizationLevel.O2

    def __init__(self, config: Any = None, device: torch.device | None = None):
        """
        Initialize the optimizer.

        Args:
            config: Backend-specific configuration object
            device: Target device for optimization
        """
        self.config = config
        self.device = device or torch.device('cpu')

        # Optimization history tracking
        self._optimization_history: list[dict[str, Any]] = []

        # Cache for optimization results
        self._cache: dict[str, OptimizationResult] = {}

        logger.debug(
            "%s initialized: device=%s",
            self.__class__.__name__,
            self.device
        )

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _apply_optimizations(
        self,
        model: nn.Module,
        level: OptimizationLevel,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> tuple[nn.Module, OptimizationResult]:
        """
        Apply optimizations for the given level.

        Args:
            model: PyTorch model to optimize
            level: Optimization level
            sample_input: Optional sample input for tracing
            dtype: Optional dtype for precision

        Returns:
            Tuple of (optimized_model, OptimizationResult)
        """
        pass

    @abstractmethod
    def get_available_strategies(self) -> list[OptimizationStrategy]:
        """
        Get list of available optimization strategies.

        Returns:
            List of OptimizationStrategy objects
        """
        pass

    # =========================================================================
    # Common implementations
    # =========================================================================

    def optimize(
        self,
        model: nn.Module,
        level: str | OptimizationLevel | None = None,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        for_inference: bool = False,
        for_training: bool = False
    ) -> tuple[nn.Module, OptimizationResult]:
        """
        Optimize a model for the given level.

        This is the main entry point for optimization.

        Args:
            model: PyTorch model to optimize
            level: Optimization level (O0, O1, O2, O3) or string
            sample_input: Optional sample input for tracing
            dtype: Optional dtype for precision
            for_inference: If True, apply inference-specific optimizations
            for_training: If True, apply training-specific optimizations

        Returns:
            Tuple of (optimized_model, OptimizationResult)
        """
        # Parse optimization level
        if level is None:
            level = self.DEFAULT_LEVEL
        elif isinstance(level, str):
            level = OptimizationLevel.from_string(level)

        logger.info(
            f"Optimizing model with {self.OPTIMIZER_NAME} at level {level.value}"
        )

        try:
            # Apply base optimizations
            optimized_model, result = self._apply_optimizations(
                model, level, sample_input, dtype
            )

            # Apply mode-specific optimizations
            if for_inference:
                optimized_model = self._apply_inference_optimizations(
                    optimized_model, sample_input, dtype
                )
                result.optimizations_applied.append('inference_mode')

            if for_training:
                optimized_model = self._apply_training_optimizations(
                    optimized_model, dtype
                )
                result.optimizations_applied.append('training_mode')

            # Record in history
            self._record_optimization(model, result)

            return optimized_model, result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return model, OptimizationResult(
                success=False,
                model=model,
                level=level,
                errors=[str(e)]
            )

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | None = None,
        level: str | OptimizationLevel | None = None,
        dtype: torch.dtype | None = None
    ) -> tuple[nn.Module, OptimizationResult]:
        """
        Optimize a model specifically for inference.

        Args:
            model: PyTorch model
            sample_input: Optional sample input for tracing
            level: Optimization level
            dtype: Optional dtype for precision

        Returns:
            Tuple of (optimized_model, OptimizationResult)
        """
        return self.optimize(
            model=model,
            level=level or OptimizationLevel.O2,
            sample_input=sample_input,
            dtype=dtype,
            for_inference=True
        )

    def optimize_for_training(
        self,
        model: nn.Module,
        level: str | OptimizationLevel | None = None,
        dtype: torch.dtype | None = None
    ) -> tuple[nn.Module, OptimizationResult]:
        """
        Optimize a model specifically for training.

        Args:
            model: PyTorch model
            level: Optimization level
            dtype: Optional dtype for precision

        Returns:
            Tuple of (optimized_model, OptimizationResult)
        """
        return self.optimize(
            model=model,
            level=level or OptimizationLevel.O1,
            dtype=dtype,
            for_training=True
        )

    def _apply_inference_optimizations(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module:
        """
        Apply inference-specific optimizations.

        Override in subclasses for device-specific inference optimizations.

        Args:
            model: Model to optimize
            sample_input: Optional sample input
            dtype: Optional dtype

        Returns:
            Optimized model
        """
        # Set to eval mode
        model = model.eval()

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _apply_training_optimizations(
        self,
        model: nn.Module,
        dtype: torch.dtype | None = None
    ) -> nn.Module:
        """
        Apply training-specific optimizations.

        Override in subclasses for device-specific training optimizations.

        Args:
            model: Model to optimize
            dtype: Optional dtype

        Returns:
            Optimized model
        """
        # Ensure train mode
        model = model.train()

        return model

    def get_optimization_recommendations(
        self,
        model: nn.Module,
        target_use: str = "inference"
    ) -> list[dict[str, Any]]:
        """
        Get optimization recommendations for a model.

        Args:
            model: Model to analyze
            target_use: "inference" or "training"

        Returns:
            List of recommendations with details
        """
        recommendations = []

        # Check model size
        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.element_size() * p.numel() for p in model.parameters())

        if param_count > 1_000_000_000:  # > 1B params
            recommendations.append({
                'type': 'model_size',
                'priority': 'high',
                'message': 'Large model detected. Consider gradient checkpointing.',
                'param_count': param_count,
                'param_bytes': param_bytes
            })

        # Check for suboptimal layers
        for name, module in model.named_modules():
            # Check Linear layers
            if isinstance(module, nn.Linear):
                if module.in_features % 8 != 0 or module.out_features % 8 != 0:
                    recommendations.append({
                        'type': 'dimension_alignment',
                        'priority': 'medium',
                        'layer': name,
                        'message': 'Linear layer dimensions not aligned for tensor cores',
                        'current': f'{module.in_features}x{module.out_features}',
                        'suggestion': 'Pad to multiples of 8'
                    })

            # Check for BatchNorm after Conv (can be fused)
            if isinstance(module, nn.BatchNorm2d):
                recommendations.append({
                    'type': 'layer_fusion',
                    'priority': 'low',
                    'layer': name,
                    'message': 'BatchNorm can potentially be fused with preceding Conv2d'
                })

        # Add device-specific recommendations
        strategies = self.get_available_strategies()
        for strategy in strategies:
            if strategy.speedup_estimate > 1.1:  # >10% speedup
                recommendations.append({
                    'type': 'optimization_strategy',
                    'priority': 'medium',
                    'name': strategy.name,
                    'description': strategy.description,
                    'speedup_estimate': f'{strategy.speedup_estimate:.1f}x',
                    'applicable_levels': [l.value for l in strategy.applicable_levels]  # noqa: E741
                })

        return recommendations

    def get_strategies_for_level(
        self,
        level: OptimizationLevel
    ) -> list[OptimizationStrategy]:
        """
        Get strategies applicable for a specific level.

        Args:
            level: Optimization level

        Returns:
            List of applicable strategies
        """
        return [s for s in self.get_available_strategies() if s.is_applicable(level)]

    def _record_optimization(
        self,
        model: nn.Module,
        result: OptimizationResult
    ) -> None:
        """Record optimization in history."""
        import time

        self._optimization_history.append({
            'timestamp': time.time(),
            'model_class': model.__class__.__name__,
            'level': result.level.value,
            'success': result.success,
            'optimizations': result.optimizations_applied,
            'warnings': len(result.warnings),
            'errors': len(result.errors)
        })

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get optimization history."""
        return self._optimization_history.copy()

    def clear_cache(self) -> None:
        """Clear optimization cache."""
        self._cache.clear()
        logger.debug(f"{self.__class__.__name__} cache cleared")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"optimizations_performed={len(self._optimization_history)})"
        )


class BaseKernelOptimizer(ABC):
    """
    Abstract base class for kernel-level optimizers.

    This class provides an interface for tuning specific operations
    (GEMM, Convolution, Attention, etc.) for different hardware.
    """

    OPTIMIZER_NAME: str = "base_kernel"

    def __init__(self, device: torch.device | None = None):
        """
        Initialize the kernel optimizer.

        Args:
            device: Target device for optimization
        """
        self.device = device or torch.device('cpu')
        self._config_cache: dict[str, KernelConfig] = {}

    @abstractmethod
    def get_optimal_gemm_config(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype = torch.float32
    ) -> KernelConfig:
        """
        Get optimal configuration for GEMM operation.

        Args:
            m, n, k: Matrix dimensions (M x K) @ (K x N)
            dtype: Data type

        Returns:
            KernelConfig with optimal parameters
        """
        pass

    @abstractmethod
    def get_optimal_conv_config(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        dtype: torch.dtype = torch.float32
    ) -> KernelConfig:
        """
        Get optimal configuration for convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            dtype: Data type

        Returns:
            KernelConfig with optimal parameters
        """
        pass

    @abstractmethod
    def get_optimal_attention_config(
        self,
        seq_len: int,
        head_dim: int,
        num_heads: int,
        dtype: torch.dtype = torch.float32
    ) -> KernelConfig:
        """
        Get optimal configuration for attention operation.

        Args:
            seq_len: Sequence length
            head_dim: Head dimension
            num_heads: Number of attention heads
            dtype: Data type

        Returns:
            KernelConfig with optimal parameters
        """
        pass

    def get_cached_config(self, key: str) -> KernelConfig | None:
        """Get cached configuration."""
        return self._config_cache.get(key)

    def cache_config(self, key: str, config: KernelConfig) -> None:
        """Cache a configuration."""
        self._config_cache[key] = config

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()


class CPUOptimizer(BaseOptimizer):
    """
    CPU optimizer implementation.

    Provides basic optimizations for CPU execution.
    """

    OPTIMIZER_NAME = "cpu"

    def _apply_optimizations(
        self,
        model: nn.Module,
        level: OptimizationLevel,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> tuple[nn.Module, OptimizationResult]:
        """Apply CPU optimizations."""
        optimizations = []
        warnings = []

        # O0: No optimizations
        if level == OptimizationLevel.O0:
            return model, OptimizationResult(
                success=True,
                model=model,
                level=level,
                optimizations_applied=['none']
            )

        # O1+: Basic optimizations
        if level.value >= OptimizationLevel.O1.value:
            # Set number of threads
            if hasattr(torch, 'set_num_threads'):
                import os
                num_threads = os.cpu_count() or 4
                torch.set_num_threads(num_threads)
                optimizations.append(f'set_num_threads({num_threads})')

        # O2+: Torch compile if available
        if level.value >= OptimizationLevel.O2.value:
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    optimizations.append('torch_compile')
                except Exception as e:
                    warnings.append(f'torch.compile failed: {e}')

        # O3: More aggressive optimizations
        if level == OptimizationLevel.O3:
            # Enable MKL optimizations if available
            if torch.backends.mkl.is_available():
                optimizations.append('mkl_enabled')

        return model, OptimizationResult(
            success=True,
            model=model,
            level=level,
            optimizations_applied=optimizations,
            warnings=warnings
        )

    def get_available_strategies(self) -> list[OptimizationStrategy]:
        """Get CPU optimization strategies."""
        return [
            OptimizationStrategy(
                name='threading',
                description='Optimize CPU thread count',
                applicable_levels=[
                    OptimizationLevel.O1,
                    OptimizationLevel.O2,
                    OptimizationLevel.O3
                ],
                speedup_estimate=1.2
            ),
            OptimizationStrategy(
                name='torch_compile',
                description='Use torch.compile for JIT optimization',
                applicable_levels=[
                    OptimizationLevel.O2,
                    OptimizationLevel.O3
                ],
                speedup_estimate=1.5,
                requires=['torch>=2.0']
            ),
            OptimizationStrategy(
                name='mkl',
                description='Intel MKL optimizations',
                applicable_levels=[
                    OptimizationLevel.O3
                ],
                speedup_estimate=1.3,
                requires=['mkl']
            )
        ]


__all__ = [
    'BaseOptimizer',
    'BaseKernelOptimizer',
    'CPUOptimizer',
    'KernelConfig',
    'OptimizationStrategy',
]
