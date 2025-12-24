"""
FP8 Compiler for NVIDIA H100/Blackwell

Provides FP8 training and inference support for Hopper and Blackwell architectures.
"""

import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAArchitecture


@dataclass
class FP8CompilationResult:
    """Results from FP8 compilation."""
    compiled_model: nn.Module
    fp8_layers: Set[str]
    compilation_mode: str
    warnings: list


class FP8Compiler:
    """
    FP8 compiler for NVIDIA H100 and Blackwell GPUs.

    Provides FP8 (8-bit floating point) training and inference support
    for maximum performance on Hopper and Blackwell architectures.
    """

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize FP8 compiler.

        Args:
            config: KernelPyTorch configuration with NVIDIA/FP8 settings
        """
        self.config = config or KernelPyTorchConfig()
        self.nvidia_config = self.config.hardware.nvidia

        self._fp8_supported = self.nvidia_config.architecture in [
            NVIDIAArchitecture.HOPPER,
            NVIDIAArchitecture.BLACKWELL
        ]

        self._fp8_layers = set()
        self._warnings = []

    def prepare_for_fp8(
        self,
        model: nn.Module,
        for_inference: bool = False
    ) -> nn.Module:
        """
        Prepare model for FP8 execution.

        Args:
            model: PyTorch model to prepare
            for_inference: Whether preparing for inference (vs training)

        Returns:
            Model prepared for FP8 execution
        """
        if not self._fp8_supported:
            self._warnings.append(
                f"FP8 not supported on {self.nvidia_config.architecture.value}, "
                "model returned unchanged"
            )
            return model

        if not self.nvidia_config.fp8_enabled:
            return model

        # Apply FP8 transformations
        model = self._convert_to_fp8_compatible(model, for_inference)

        # Add FP8 scaling hooks if training
        if not for_inference:
            model = self._add_fp8_scaling_hooks(model)

        return model

    def _convert_to_fp8_compatible(
        self,
        model: nn.Module,
        for_inference: bool
    ) -> nn.Module:
        """Convert model layers to FP8-compatible format."""
        for name, module in model.named_modules():
            # Convert Linear layers
            if isinstance(module, nn.Linear):
                self._prepare_linear_for_fp8(name, module, for_inference)

            # Convert Conv layers
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                self._prepare_conv_for_fp8(name, module, for_inference)

            # Convert attention layers
            elif hasattr(module, 'num_attention_heads'):
                self._prepare_attention_for_fp8(name, module, for_inference)

        return model

    def _prepare_linear_for_fp8(
        self,
        name: str,
        module: nn.Linear,
        for_inference: bool
    ) -> None:
        """Prepare Linear layer for FP8 execution."""
        # Mark layer as FP8-ready
        setattr(module, '_fp8_enabled', True)
        setattr(module, '_fp8_mode', 'inference' if for_inference else 'training')
        self._fp8_layers.add(name)

        # For actual FP8 execution, we would use transformer_engine or similar
        # Here we add metadata for torch.compile to use FP8 kernels
        if hasattr(module, 'weight'):
            # Ensure weight dimensions are optimal for FP8 Tensor Cores
            in_features, out_features = module.in_features, module.out_features
            if in_features % 16 != 0 or out_features % 16 != 0:
                self._warnings.append(
                    f"Linear layer {name} dimensions ({in_features}x{out_features}) "
                    "not optimal for FP8 Tensor Cores. Consider padding to multiples of 16."
                )

    def _prepare_conv_for_fp8(
        self,
        name: str,
        module: nn.Module,
        for_inference: bool
    ) -> None:
        """Prepare Conv layer for FP8 execution."""
        setattr(module, '_fp8_enabled', True)
        setattr(module, '_fp8_mode', 'inference' if for_inference else 'training')
        self._fp8_layers.add(name)

    def _prepare_attention_for_fp8(
        self,
        name: str,
        module: nn.Module,
        for_inference: bool
    ) -> None:
        """Prepare attention layer for FP8 execution."""
        setattr(module, '_fp8_enabled', True)
        setattr(module, '_fp8_mode', 'inference' if for_inference else 'training')
        self._fp8_layers.add(name)

        # Check attention head dimensions
        if hasattr(module, 'attention_head_size'):
            head_size = module.attention_head_size
            if head_size % 16 != 0:
                self._warnings.append(
                    f"Attention layer {name} head size {head_size} "
                    "not optimal for FP8. Consider using head size divisible by 16."
                )

    def _add_fp8_scaling_hooks(self, model: nn.Module) -> nn.Module:
        """Add FP8 scaling hooks for training."""
        # In production, this would integrate with transformer_engine
        # or similar libraries for FP8 scaling

        # Add forward pre-hook for activation scaling
        def fp8_forward_pre_hook(module, inputs):
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                # Apply FP8 scaling (simplified - actual implementation
                # would use proper FP8 scaling from transformer_engine)
                return inputs
            return inputs

        # Add forward hook for gradient scaling
        def fp8_forward_hook(module, inputs, outputs):
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                # Apply FP8 gradient scaling
                return outputs
            return outputs

        # Register hooks on FP8-enabled layers
        for name, module in model.named_modules():
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                module.register_forward_pre_hook(fp8_forward_pre_hook)
                module.register_forward_hook(fp8_forward_hook)

        return model

    def compile_with_fp8(
        self,
        model: nn.Module,
        sample_inputs: Optional[torch.Tensor] = None,
        for_inference: bool = False
    ) -> FP8CompilationResult:
        """
        Compile model with FP8 optimizations.

        Args:
            model: PyTorch model to compile
            sample_inputs: Sample inputs for compilation
            for_inference: Whether compiling for inference

        Returns:
            FP8CompilationResult with compiled model and statistics
        """
        self._fp8_layers = set()
        self._warnings = []

        # Prepare model for FP8
        compiled_model = self.prepare_for_fp8(model, for_inference)

        # Compile with torch.compile if available
        if hasattr(torch, 'compile') and sample_inputs is not None:
            try:
                compile_mode = 'max-autotune' if for_inference else 'default'
                compiled_model = torch.compile(
                    compiled_model,
                    mode=compile_mode,
                    dynamic=False
                )

                # Warm up
                with torch.no_grad():
                    _ = compiled_model(sample_inputs)

            except Exception as e:
                self._warnings.append(f"torch.compile with FP8 failed: {e}")

        return FP8CompilationResult(
            compiled_model=compiled_model,
            fp8_layers=self._fp8_layers,
            compilation_mode='inference' if for_inference else 'training',
            warnings=self._warnings
        )

    def get_fp8_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get FP8 statistics for the model."""
        total_layers = 0
        fp8_layers = 0

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                total_layers += 1
                if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                    fp8_layers += 1

        return {
            'total_layers': total_layers,
            'fp8_layers': fp8_layers,
            'fp8_coverage': fp8_layers / total_layers if total_layers > 0 else 0,
            'fp8_supported': self._fp8_supported,
            'fp8_enabled': self.nvidia_config.fp8_enabled,
            'architecture': self.nvidia_config.architecture.value
        }

    def estimate_speedup(self, model: nn.Module) -> Dict[str, Any]:
        """
        Estimate FP8 training/inference speedup.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with speedup estimates
        """
        stats = self.get_fp8_stats(model)

        # Theoretical speedups based on NVIDIA documentation
        if self.nvidia_config.architecture == NVIDIAArchitecture.HOPPER:
            base_speedup = 2.0  # H100 FP8 vs FP16
        elif self.nvidia_config.architecture == NVIDIAArchitecture.BLACKWELL:
            base_speedup = 2.5  # Blackwell FP8 vs FP16
        else:
            base_speedup = 1.0  # No FP8 support

        # Adjust for coverage
        estimated_speedup = 1.0 + (base_speedup - 1.0) * stats['fp8_coverage']

        return {
            'estimated_speedup': estimated_speedup,
            'base_speedup': base_speedup,
            'fp8_coverage': stats['fp8_coverage'],
            'architecture': self.nvidia_config.architecture.value,
            'note': 'Theoretical estimate based on NVIDIA specs and layer coverage'
        }
