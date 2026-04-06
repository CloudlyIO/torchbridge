"""
Unified Management System for TorchBridge

This module provides the main UnifiedManager class that orchestrates
backend-aware model optimization through hardware detection and
infrastructure lifecycle management.

"""

import warnings
from typing import Any

import torch
import torch.nn as nn

from ..config import TorchBridgeConfig
from ..hardware_detector import HardwareDetector, HardwareProfile
from .infrastructure_manager import InfrastructureManager


class UnifiedManager:
    """
    Unified management system for TorchBridge.

    Provides hardware-aware model optimization through:
    - Automatic backend detection and selection
    - Backend adapter dispatch (NVIDIA, AMD, TPU, CPU)
    - Infrastructure lifecycle management (deprecation tracking, validation)

    Primary entry point: auto_optimize(model)
    """

    def __init__(self, config: TorchBridgeConfig | None = None):
        self.config = config or TorchBridgeConfig()

        self.infrastructure_manager = InfrastructureManager(self.config)

        # Hardware detection
        self.hardware_detector = HardwareDetector()
        self._hardware_profile: HardwareProfile | None = None

        # Backend adapter instances (lazy-loaded)
        self._nvidia_adapter: Any = None
        self._tpu_adapter: Any = None
        self._amd_adapter: Any = None

        self._initialized = True

    def optimize(self, target: Any, **kwargs) -> Any:
        """
        Optimize target. For nn.Module, delegates to auto_optimize().
        For other targets, runs infrastructure checks and returns unchanged.
        """
        if not self._initialized:
            raise RuntimeError("UnifiedManager not initialized")

        if isinstance(target, nn.Module):
            return self.auto_optimize(target, **kwargs)

        self.infrastructure_manager.optimize(target, **kwargs)
        return target

    def auto_optimize(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None = None,
        optimization_level: str | None = None,
        for_inference: bool = False,
    ) -> nn.Module:
        """
        Automatically optimize model based on detected hardware.

        Detects available hardware and routes to the best backend adapter:
        NVIDIA → NVIDIAAdapter, AMD → AMDAdapter, TPU → TPUAdapter, else CPU.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Optional sample inputs for compilation
            optimization_level: Override level (conservative/balanced/aggressive).
                                 Auto-determined from hardware if None.
            for_inference: Whether optimizing for inference (vs training)

        Returns:
            Optimized PyTorch model

        Example:
            >>> manager = UnifiedManager()
            >>> optimized = manager.auto_optimize(model)
        """
        if self._hardware_profile is None:
            self._hardware_profile = self.hardware_detector.detect()

        if optimization_level is None:
            optimization_level = (
                self.hardware_detector.get_recommended_optimization_level(
                    self._hardware_profile
                )
            )

        backend_name = self.hardware_detector.get_optimal_backend(
            self._hardware_profile
        )

        if backend_name == "nvidia":
            result = self._optimize_with_nvidia(
                model, sample_inputs, optimization_level, for_inference
            )
        elif backend_name == "tpu":
            result = self._optimize_with_tpu(
                model, sample_inputs, optimization_level, for_inference
            )
        elif backend_name == "amd":
            result = self._optimize_with_amd(
                model, sample_inputs, optimization_level, for_inference
            )
        else:
            result = self._optimize_with_cpu(
                model, sample_inputs, optimization_level, for_inference
            )

        # Normalize result — backends return different types
        if isinstance(result, tuple):
            return result[0]
        elif isinstance(result, nn.Module):
            return result
        elif hasattr(result, "optimized_model"):
            return result.optimized_model
        elif hasattr(result, "model"):
            return result.model
        return result

    def _optimize_with_nvidia(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool,
    ) -> Any:
        try:
            from ...backends.nvidia import NVIDIAAdapter

            if self._nvidia_adapter is None:
                self._nvidia_adapter = NVIDIAAdapter(self.config)

            if for_inference:
                return self._nvidia_adapter.optimize_for_inference(
                    model, sample_input=sample_inputs
                )
            else:
                return self._nvidia_adapter.optimize_for_training(model)

        except ImportError as e:
            warnings.warn(
                f"NVIDIA backend not available: {e}. Using CPU fallback.", stacklevel=2
            )
            return self._optimize_with_cpu(
                model, sample_inputs, optimization_level, for_inference
            )

    def _optimize_with_tpu(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool,
    ) -> Any:
        try:
            from ...backends.tpu import TPUAdapter

            if self._tpu_adapter is None:
                self._tpu_adapter = TPUAdapter(self.config)

            if for_inference:
                return self._tpu_adapter.optimize_for_inference(
                    model, sample_inputs=sample_inputs
                )
            else:
                return self._tpu_adapter.optimize_for_training(
                    model, sample_inputs=sample_inputs
                )

        except ImportError as e:
            warnings.warn(
                f"TPU backend not available: {e}. Using CPU fallback.", stacklevel=2
            )
            return self._optimize_with_cpu(
                model, sample_inputs, optimization_level, for_inference
            )

    def _optimize_with_amd(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool,
    ) -> Any:
        try:
            from ...backends.amd import AMDAdapter
            from ...core.config import AMDConfig

            if self._amd_adapter is None:
                self._amd_adapter = AMDAdapter(AMDConfig())

            result = self._amd_adapter.optimize(model, level=optimization_level)
            if for_inference and isinstance(result, nn.Module):
                result.eval()
            return result

        except ImportError as e:
            warnings.warn(
                f"AMD backend not available: {e}. Using CPU fallback.", stacklevel=2
            )
            return self._optimize_with_cpu(
                model, sample_inputs, optimization_level, for_inference
            )

    def _optimize_with_cpu(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None,
        optimization_level: str,
        for_inference: bool,
    ) -> nn.Module:
        if for_inference:
            model.eval()
        return model

    def get_hardware_profile(self, force_redetect: bool = False) -> HardwareProfile:
        """Get detected hardware profile."""
        if self._hardware_profile is None or force_redetect:
            self._hardware_profile = self.hardware_detector.detect(force_redetect)
        return self._hardware_profile

    def get_optimization_recommendations(
        self, model: nn.Module | None = None
    ) -> dict[str, Any]:
        """Get optimization recommendations based on detected hardware."""
        profile = self.get_hardware_profile()

        recommendations: dict[str, Any] = {
            "hardware_type": profile.hardware_type.value,
            "device_name": profile.device_name,
            "backend": self.hardware_detector.get_optimal_backend(profile),
            "optimization_level": self.hardware_detector.get_recommended_optimization_level(
                profile
            ),
            "capabilities": [cap.value for cap in profile.capabilities],
            "optimizations": [],
        }

        if profile.is_nvidia_h100_or_better():
            recommendations["optimizations"].append(
                {
                    "type": "fp8_training",
                    "benefit": "2x training speedup",
                    "requirement": "H100 or Blackwell GPU",
                }
            )
            recommendations["optimizations"].append(
                {
                    "type": "flash_attention_3",
                    "benefit": "3x memory reduction",
                    "requirement": "H100 or Blackwell GPU",
                }
            )

        if profile.is_high_end_tpu():
            recommendations["optimizations"].append(
                {
                    "type": "xla_compilation",
                    "benefit": "Optimized TPU execution",
                    "requirement": f"TPU {profile.tpu_version.value}",
                }
            )

        return recommendations

    def get_status(self) -> dict[str, Any]:
        """Get status of active managers."""
        return {"infrastructure": self.infrastructure_manager.get_status()}

    def suspend_all(self) -> None:
        """Suspend all managers."""
        self.infrastructure_manager.suspend()

    def resume_all(self) -> None:
        """Resume all managers."""
        self.infrastructure_manager.resume()

    def shutdown_all(self) -> None:
        """Shutdown all managers."""
        self.infrastructure_manager.shutdown()


# Module-level helpers

default_manager: UnifiedManager | None = None


def get_manager(config: TorchBridgeConfig | None = None) -> UnifiedManager:
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
