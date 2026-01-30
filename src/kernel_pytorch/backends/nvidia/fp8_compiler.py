"""
FP8 Compiler for NVIDIA H100/Blackwell

**STATUS: METADATA-ONLY - NOT PRODUCTION FP8**

This module provides FP8 layer identification and metadata marking for H100/Blackwell
GPUs. It does NOT perform actual FP8 quantization or arithmetic - this is by design
to keep the module lightweight and avoid duplicating NVIDIA Transformer Engine.

Use this module for:
- Planning FP8 deployments (layer identification, performance estimation)
- Preparing models for FP8 optimization analysis
- Infrastructure testing and validation

For ACTUAL FP8 training in production, use NVIDIA Transformer Engine directly:
  pip install transformer-engine
  from transformer_engine.pytorch import fp8_autocast
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAArchitecture

logger = logging.getLogger(__name__)


@dataclass
class FP8CompilationResult:
    """Results from FP8 compilation."""
    compiled_model: nn.Module
    fp8_layers: set[str]
    compilation_mode: str
    warnings: list


class FP8Compiler:
    """
    FP8 compiler for NVIDIA H100 and Blackwell GPUs.

    **METADATA-ONLY BY DESIGN**:
    This module provides FP8 METADATA marking only. Layers are identified
    and marked for FP8, but actual FP8 quantization and arithmetic are NOT
    performed. This is intentional to avoid duplicating NVIDIA Transformer Engine.

    Capabilities:
    - Identifies FP8-capable layers (Linear, Conv, Attention)
    - Marks layers with '_fp8_enabled' attribute
    - Validates hardware support (H100/Blackwell)
    - Estimates performance improvements (~2x theoretical)
    - Reports dimension alignment issues for Tensor Cores

    For production FP8 training, use NVIDIA Transformer Engine directly:
        pip install transformer-engine
        from transformer_engine.pytorch import fp8_autocast
    """

    def __init__(self, config: KernelPyTorchConfig | None = None):
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
        module._fp8_enabled = True
        module._fp8_mode = 'inference' if for_inference else 'training'
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
        module._fp8_enabled = True
        module._fp8_mode = 'inference' if for_inference else 'training'
        self._fp8_layers.add(name)

    def _prepare_attention_for_fp8(
        self,
        name: str,
        module: nn.Module,
        for_inference: bool
    ) -> None:
        """Prepare attention layer for FP8 execution."""
        module._fp8_enabled = True
        module._fp8_mode = 'inference' if for_inference else 'training'
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

        **METADATA-ONLY BY DESIGN**:
        FP8 support is METADATA-ONLY. The FP8Compiler marks layers as FP8-enabled
        and estimates performance, but does NOT perform actual FP8 operations.
        This is intentional to avoid duplicating NVIDIA Transformer Engine.

        Current behavior:
        - Marks layers with '_fp8_enabled' attribute
        - Estimates 2x speedup for performance planning
        - Does NOT quantize weights or activations
        - Does NOT perform FP8 arithmetic

        For actual FP8 training on H100/Blackwell, use NVIDIA Transformer Engine directly.

        Raises:
            UserWarning: FP8 hooks are metadata-only
        """
        import warnings
        warnings.warn(
            "FP8 support is metadata-only. Layers are marked for FP8 but "
            "no actual FP8 operations are performed. For production FP8 training, "
            "use NVIDIA Transformer Engine directly: pip install transformer-engine",
            UserWarning,
            stacklevel=2
        )

        # Add forward pre-hook for activation scaling (NO-OP - metadata only)
        def fp8_forward_pre_hook(module, inputs):
            """
            Metadata-only hook - no actual FP8 operations.
            For production FP8, use NVIDIA Transformer Engine directly.
            """
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                # DESIGN_NOTE: Actual FP8 scaling requires Transformer Engine integration.
                # This module intentionally stays metadata-only to avoid duplication.
                return inputs  # NO-OP by design
            return inputs

        # Add forward hook for gradient scaling (NO-OP - metadata only)
        def fp8_forward_hook(module, inputs, outputs):
            """
            Metadata-only hook - no actual FP8 operations.
            For production FP8, use NVIDIA Transformer Engine directly.
            """
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                # DESIGN_NOTE: Actual FP8 gradient scaling requires Transformer Engine.
                # This module intentionally stays metadata-only to avoid duplication.
                return outputs  # NO-OP by design
            return outputs

        # Register hooks on FP8-enabled layers
        for _name, module in model.named_modules():
            if hasattr(module, '_fp8_enabled') and module._fp8_enabled:
                module.register_forward_pre_hook(fp8_forward_pre_hook)
                module.register_forward_hook(fp8_forward_hook)

        return model

    def compile_with_fp8(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | None = None,
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

    def get_fp8_stats(self, model: nn.Module) -> dict[str, Any]:
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

    def estimate_speedup(self, model: nn.Module) -> dict[str, Any]:
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
