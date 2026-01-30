"""
Unified Management System for KernelPyTorch

This module provides the main UnifiedManager class that orchestrates
all management subsystems:

- Hardware management (GPU, memory, distributed)
- Optimization management (compilation, precision, performance)
- Infrastructure management (testing, deprecation, lifecycle)

The original 38+ scattered Manager/Optimizer classes have been consolidated
into three focused managers coordinated by UnifiedManager.

Version: 0.3.11
"""

import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..config import KernelPyTorchConfig
from ..hardware_detector import HardwareDetector, HardwareProfile

# Import management components
from .base import ManagerType
from .hardware_manager import HardwareManager
from .infrastructure_manager import InfrastructureManager
from .optimization_manager import OptimizationManager


class UnifiedManager:
    """
    Unified management system that orchestrates all managers.

    Provides a single interface to replace all 38+ scattered managers:

    Hardware Managers (11):
    - PrivateUse1Manager, MemoryOptimizer, TensorCoreOptimizer,
    - MixedPrecisionManager, DistributedManager, DataParallelOptimizer,
    - ModelParallelOptimizer, CommunicationOptimizer, HardwareTopologyManager,
    - PowerEfficiencyOptimizer, DeviceMeshOptimizer

    Optimization Managers (18):
    - PyGraphCUDAOptimizer, FusionBoundaryOptimizer, LongSequenceOptimizer,
    - MemoryFragmentationOptimizer, AdaptiveCompressionOptimizer,
    - AdaptiveMemoryManager, MXFPOptimizer, DynamicSparsityOptimizer,
    - HybridShardingOptimizer, FSDP2Manager, FP8ScaleManager, FP8Optimizer,
    - NetworkTopologyOptimizer, AutoScalingManager, FaultToleranceManager,
    - AdvancedFSDPManager, HeterogenousClusterManager, MultiNodeTrainingManager

    Infrastructure Managers (9):
    - TestEnvironmentManager, CIPipelineManager, DeprecationManager,
    - SLURMClusterManager, CUDAGraphManager (multiple), OptimizerStateGroup,
    - DeepOptimizerStates, InterleaveOffloadingOptimizer, CPUGPUHybridOptimizer
    """

    def __init__(self, config: KernelPyTorchConfig | None = None):
        self.config = config or KernelPyTorchConfig()

        # Initialize sub-managers
        self.hardware_manager = HardwareManager(self.config)
        self.optimization_manager = OptimizationManager(self.config)
        self.infrastructure_manager = InfrastructureManager(self.config)

        self._managers = {
            ManagerType.HARDWARE: self.hardware_manager,
            ManagerType.OPTIMIZATION: self.optimization_manager,
            ManagerType.INFRASTRUCTURE: self.infrastructure_manager
        }

        # Initialize hardware detector for auto-optimization
        self.hardware_detector = HardwareDetector()
        self._hardware_profile: HardwareProfile | None = None

        # Backend optimizer instances (lazy-loaded)
        self._nvidia_optimizer = None
        self._tpu_optimizer = None
        self._amd_optimizer = None

        self._initialized = True

    def optimize(self, target: Any, **kwargs) -> Any:
        """
        Apply comprehensive optimization across all management domains.

        Replaces the need to call multiple individual managers.
        """
        if not self._initialized:
            raise RuntimeError("UnifiedManager not initialized")

        # Apply optimizations in order
        target = self.hardware_manager.optimize(target, **kwargs)
        target = self.optimization_manager.optimize(target, **kwargs)
        target = self.infrastructure_manager.optimize(target, **kwargs)

        return target

    def auto_optimize(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None = None,
        optimization_level: str | None = None,
        for_inference: bool = False
    ) -> nn.Module:
        """
        Automatically optimize model based on detected hardware.

        This method detects available hardware and applies the best
        optimization strategy automatically.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Optional sample inputs for compilation
            optimization_level: Override optimization level (conservative/balanced/aggressive)
                              If None, automatically determined based on hardware
            for_inference: Whether optimizing for inference (vs training)

        Returns:
            Optimized PyTorch model

        Example:
            >>> manager = UnifiedManager()
            >>> optimized_model = manager.auto_optimize(model)
            >>> # Automatically uses NVIDIA/TPU/AMD/CPU based on available hardware
        """
        # Detect hardware if not cached
        if self._hardware_profile is None:
            self._hardware_profile = self.hardware_detector.detect()

        # Determine optimization level if not specified
        if optimization_level is None:
            optimization_level = self.hardware_detector.get_recommended_optimization_level(
                self._hardware_profile
            )

        # Route to appropriate backend
        backend_name = self.hardware_detector.get_optimal_backend(self._hardware_profile)

        if backend_name == 'nvidia':
            result = self._optimize_with_nvidia(
                model, sample_inputs, optimization_level, for_inference
            )
        elif backend_name == 'tpu':
            result = self._optimize_with_tpu(
                model, sample_inputs, optimization_level, for_inference
            )
        elif backend_name == 'amd':
            result = self._optimize_with_amd(
                model, sample_inputs, optimization_level, for_inference
            )
        else:
            result = self._optimize_with_cpu(
                model, sample_inputs, optimization_level, for_inference
            )

        # Extract model from result (backends return result objects)
        if hasattr(result, 'optimized_model'):
            return result.optimized_model
        elif hasattr(result, 'model'):
            return result.model
        else:
            # Fallback: result is already the model
            return result

    def _optimize_with_nvidia(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool
    ) -> Any:
        """Optimize model using NVIDIA backend."""
        try:
            from ...backends.nvidia import NVIDIAOptimizer

            if self._nvidia_optimizer is None:
                self._nvidia_optimizer = NVIDIAOptimizer(self.config)

            if for_inference:
                result = self._nvidia_optimizer.optimize_for_inference(
                    model, sample_inputs, optimization_level
                )
            else:
                result = self._nvidia_optimizer.optimize_for_training(
                    model, sample_inputs, optimization_level
                )

            return result

        except ImportError as e:
            warnings.warn(f"NVIDIA backend not available: {e}. Using CPU fallback.", stacklevel=2)
            return self._optimize_with_cpu(model, sample_inputs, optimization_level, for_inference)

    def _optimize_with_tpu(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool
    ) -> Any:
        """Optimize model using TPU backend."""
        try:
            from ...backends.tpu import TPUOptimizer

            if self._tpu_optimizer is None:
                self._tpu_optimizer = TPUOptimizer(self.config)

            result = self._tpu_optimizer.optimize(
                model, sample_inputs, optimization_level, for_inference
            )

            return result

        except ImportError as e:
            warnings.warn(f"TPU backend not available: {e}. Using CPU fallback.", stacklevel=2)
            return self._optimize_with_cpu(model, sample_inputs, optimization_level, for_inference)

    def _optimize_with_amd(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool
    ) -> Any:
        """Optimize model using AMD ROCm backend."""
        try:
            from ...backends.amd import AMDOptimizer

            if self._amd_optimizer is None:
                self._amd_optimizer = AMDOptimizer(self.config)

            result = self._amd_optimizer.optimize(
                model, sample_inputs, optimization_level, for_inference
            )

            return result

        except ImportError as e:
            warnings.warn(f"AMD backend not available: {e}. Using CPU fallback.", stacklevel=2)
            return self._optimize_with_cpu(model, sample_inputs, optimization_level, for_inference)

    def _optimize_with_cpu(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool
    ) -> Any:
        """Optimize model for CPU execution (minimal optimization)."""
        @dataclass
        class CPUOptimizationResult:
            optimized_model: nn.Module
            optimization_level: str
            optimizations_applied: list[str]
            backend: str = "cpu"

        # For CPU, just set to eval mode if for inference
        if for_inference:
            model.eval()
            optimizations = ["eval_mode"]
        else:
            optimizations = []

        return CPUOptimizationResult(
            optimized_model=model,
            optimization_level=optimization_level,
            optimizations_applied=optimizations
        )

    def get_hardware_profile(self, force_redetect: bool = False) -> HardwareProfile:
        """
        Get detected hardware profile.

        Args:
            force_redetect: Force re-detection of hardware

        Returns:
            HardwareProfile with detected capabilities
        """
        if self._hardware_profile is None or force_redetect:
            self._hardware_profile = self.hardware_detector.detect(force_redetect)
        return self._hardware_profile

    def get_optimization_recommendations(self, model: nn.Module | None = None) -> dict[str, Any]:
        """
        Get optimization recommendations based on detected hardware.

        Args:
            model: Optional PyTorch model to analyze (currently unused)

        Returns:
            Dictionary with recommendations
        """
        profile = self.get_hardware_profile()

        recommendations = {
            'hardware_type': profile.hardware_type.value,
            'device_name': profile.device_name,
            'backend': self.hardware_detector.get_optimal_backend(profile),
            'optimization_level': self.hardware_detector.get_recommended_optimization_level(profile),
            'capabilities': [cap.value for cap in profile.capabilities],
            'optimizations': []
        }

        # Add specific recommendations based on hardware
        if profile.is_nvidia_h100_or_better():
            recommendations['optimizations'].append({
                'type': 'fp8_training',
                'benefit': '2x training speedup',
                'requirement': 'H100 or Blackwell GPU'
            })
            recommendations['optimizations'].append({
                'type': 'flash_attention_3',
                'benefit': '3x memory reduction',
                'requirement': 'H100 or Blackwell GPU'
            })

        if profile.is_high_end_tpu():
            recommendations['optimizations'].append({
                'type': 'xla_compilation',
                'benefit': 'Optimized TPU execution',
                'requirement': f'TPU {profile.tpu_version.value}'
            })

        return recommendations

    def get_status(self) -> dict[str, Any]:
        """Get status of all managers."""
        return {
            manager_type.value: manager.get_status()
            for manager_type, manager in self._managers.items()
        }

    def suspend_all(self) -> None:
        """Suspend all managers."""
        for manager in self._managers.values():
            manager.suspend()

    def resume_all(self) -> None:
        """Resume all managers."""
        for manager in self._managers.values():
            manager.resume()

    def shutdown_all(self) -> None:
        """Shutdown all managers."""
        for manager in self._managers.values():
            manager.shutdown()


# Global unified manager instance for convenience
default_manager: UnifiedManager | None = None


def get_manager(config: KernelPyTorchConfig | None = None) -> UnifiedManager:
    """Get the global unified manager."""
    global default_manager
    if default_manager is None or config is not None:
        default_manager = UnifiedManager(config)
    return default_manager


def optimize_with_unified_manager(target: Any, **kwargs) -> Any:
    """Convenience function for unified optimization."""
    return get_manager().optimize(target, **kwargs)


def reset_manager() -> None:
    """Reset the global manager."""
    global default_manager
    if default_manager:
        default_manager.shutdown_all()
    default_manager = None
