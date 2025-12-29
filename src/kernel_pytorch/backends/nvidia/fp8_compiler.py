"""
FP8 Compiler for NVIDIA H100/Blackwell

**v0.4.0 STATUS: METADATA-ONLY - NOT PRODUCTION FP8**

This module provides FP8 layer identification and metadata marking for H100/Blackwell
GPUs. In v0.4.0, it does NOT perform actual FP8 quantization or arithmetic.
Full FP8 implementation with NVIDIA Transformer Engine is planned for v0.5.0.

Use this module for:
- Planning FP8 deployments (layer identification, performance estimation)
- Preparing models for future FP8 optimization (v0.5.0)
- Infrastructure testing and validation

For ACTUAL FP8 training in production, use NVIDIA Transformer Engine directly.
"""

import logging
import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAArchitecture

logger = logging.getLogger(__name__)


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

    **v0.4.0 LIMITATION - METADATA-ONLY**:
    This version provides FP8 METADATA marking only. Layers are identified
    and marked for FP8, but actual FP8 quantization and arithmetic are NOT
    performed. Full FP8 implementation with NVIDIA Transformer Engine
    integration is planned for v0.5.0.

    Current capabilities (v0.4.0):
    - Identifies FP8-capable layers (Linear, Conv, Attention)
    - Marks layers with '_fp8_enabled' attribute
    - Validates hardware support (H100/Blackwell)
    - Estimates performance improvements (~2x theoretical)
    - Prepares infrastructure for v0.5.0

    Future capabilities (v0.5.0):
    - Actual FP8 weight quantization
    - FP8 activation scaling
    - FP8 gradient scaling with Transformer Engine
    - Calibration and accuracy validation

    For production FP8 training NOW, use NVIDIA Transformer Engine directly.
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
        """Add FP8 scaling hooks for training.

        **IMPORTANT - v0.4.0 LIMITATION**:
        FP8 support in v0.4.0 is METADATA-ONLY. The FP8Compiler marks layers as
        FP8-enabled and estimates performance, but does NOT perform actual FP8
        operations. This is intentional - full FP8 implementation is planned for v0.5.0.

        Current behavior:
        - Marks layers with '_fp8_enabled' attribute
        - Estimates 2x speedup for performance planning
        - Does NOT quantize weights or activations
        - Does NOT perform FP8 arithmetic

        For actual FP8 training on H100/Blackwell, use NVIDIA Transformer Engine
        directly or wait for v0.5.0 integration.

        Roadmap:
        - v0.4.0 (current): Metadata-only, layer marking
        - v0.5.0 (planned): Full FP8 with Transformer Engine integration

        Raises:
            DeprecationWarning: FP8 hooks are metadata-only in v0.4.0
        """
        import warnings
        warnings.warn(
            "FP8 support is metadata-only in v0.4.0. Layers are marked for FP8 but "
            "no actual FP8 operations are performed. Full FP8 implementation with "
            "NVIDIA Transformer Engine integration is planned for v0.5.0. "
            "For production FP8 training, use Transformer Engine directly.",
            DeprecationWarning,
            stacklevel=2
        )

        # Add forward pre-hook for activation scaling (NO-OP in v0.4.0)
        def fp8_forward_pre_hook(module, inputs):
            """
            Metadata-only hook (v0.4.0).
            In v0.5.0, this will apply FP8 scaling from transformer_engine.
            """
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                # TODO (v0.5.0): Integrate with transformer_engine for actual FP8 scaling
                # return scale_to_fp8(inputs)  # Future implementation
                return inputs  # NO-OP in v0.4.0
            return inputs

        # Add forward hook for gradient scaling (NO-OP in v0.4.0)
        def fp8_forward_hook(module, inputs, outputs):
            """
            Metadata-only hook (v0.4.0).
            In v0.5.0, this will apply FP8 gradient scaling.
            """
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                # TODO (v0.5.0): Apply FP8 gradient scaling
                # return scale_fp8_gradients(outputs)  # Future implementation
                return outputs  # NO-OP in v0.4.0
            return outputs

        # Register hooks on FP8-enabled layers (for v0.5.0 compatibility)
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
