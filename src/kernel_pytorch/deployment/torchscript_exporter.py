"""
TorchScript Exporter with Optimization Preservation

This module provides TorchScript export functionality for PyTorch models,
supporting both tracing and scripting methods with optimization metadata.

Features:
- Export via torch.jit.trace or torch.jit.script
- Preserve optimization metadata as model attributes
- Support for traced and scripted models
- Export validation and verification
- Mobile optimization support (torch.utils.mobile_optimizer)

Example:
    ```python
    from kernel_pytorch.deployment import TorchScriptExporter

    exporter = TorchScriptExporter()
    result = exporter.export(
        model=optimized_model,
        output_path="model.pt",
        sample_input=torch.randn(1, 512),
        method="trace"
    )

    print(f"Exported to: {result.output_path}")
    print(f"Method: {result.export_method}")
    ```
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn

from .optimization_metadata import (
    OptimizationMetadata,
    ExportFormat,
    create_metadata
)

logger = logging.getLogger(__name__)


class ExportMethod:
    """TorchScript export methods."""
    TRACE = "trace"
    SCRIPT = "script"
    AUTO = "auto"


@dataclass
class TorchScriptExportConfig:
    """Configuration for TorchScript export."""
    # Export method
    method: str = ExportMethod.TRACE  # trace, script, auto

    # Tracing options
    check_trace: bool = True
    check_tolerance: float = 1e-5
    strict: bool = True

    # Optimization options
    optimize_for_inference: bool = True
    optimize_for_mobile: bool = False
    freeze_model: bool = True

    # Validation options
    validate_export: bool = True
    validation_tolerance: float = 1e-5
    validation_iterations: int = 3

    # Metadata options
    include_metadata: bool = True
    metadata_path: Optional[str] = None

    # Save options
    save_extra_files: bool = True
    compression: bool = False


@dataclass
class TorchScriptExportResult:
    """Result of TorchScript export operation."""
    success: bool = False
    output_path: str = ""
    metadata_path: Optional[str] = None
    metadata: Optional[OptimizationMetadata] = None

    # Export details
    export_method: str = "trace"
    model_size_mb: float = 0.0

    # Validation results
    validated: bool = False
    validation_error: Optional[float] = None
    validation_message: str = ""

    # Optimization flags
    frozen: bool = False
    optimized_for_inference: bool = False
    optimized_for_mobile: bool = False

    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TorchScriptExporter:
    """
    TorchScript Exporter with optimization preservation.

    This class handles exporting PyTorch models to TorchScript format
    using either tracing or scripting methods.
    """

    def __init__(self, config: Optional[TorchScriptExportConfig] = None):
        """
        Initialize TorchScript exporter.

        Args:
            config: Export configuration. If None, uses defaults.
        """
        self.config = config or TorchScriptExportConfig()

    def export(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        sample_input: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        method: Optional[str] = None,
        optimization_level: str = "balanced",
        backend: str = "auto",
        benchmark: bool = True,
        **kwargs
    ) -> TorchScriptExportResult:
        """
        Export model to TorchScript format.

        Args:
            model: PyTorch model to export
            output_path: Path for the output .pt file
            sample_input: Sample input for tracing (required for trace method)
            method: Export method (trace, script, auto). Overrides config.
            optimization_level: Optimization level applied to model
            backend: Hardware backend (auto, cuda, tpu, amd, cpu)
            benchmark: Whether to run benchmarks for metadata
            **kwargs: Additional arguments

        Returns:
            TorchScriptExportResult with export status and details
        """
        result = TorchScriptExportResult()
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine export method
        export_method = method or self.config.method
        if export_method == ExportMethod.AUTO:
            export_method = self._determine_best_method(model, sample_input)

        result.export_method = export_method

        try:
            # Prepare model for export
            model = self._prepare_model(model)

            # Create optimization metadata
            if self.config.include_metadata and sample_input is not None:
                first_input = sample_input[0] if isinstance(sample_input, tuple) else sample_input
                metadata = create_metadata(
                    model=model,
                    backend=backend,
                    optimization_level=optimization_level,
                    export_format="torchscript",
                    sample_input=first_input,
                    benchmark=benchmark
                )
                result.metadata = metadata

            # Perform export based on method
            logger.info(f"Exporting model to TorchScript ({export_method}): {output_path}")

            if export_method == ExportMethod.TRACE:
                if sample_input is None:
                    raise ValueError("sample_input is required for trace export method")
                scripted_model = self._export_trace(model, sample_input)
            else:
                scripted_model = self._export_script(model)

            # Apply optimizations
            if self.config.freeze_model:
                scripted_model = self._freeze_model(scripted_model)
                result.frozen = True

            if self.config.optimize_for_inference:
                scripted_model = self._optimize_for_inference(scripted_model)
                result.optimized_for_inference = True

            if self.config.optimize_for_mobile:
                scripted_model = self._optimize_for_mobile(scripted_model)
                result.optimized_for_mobile = True

            # Save model
            extra_files = {}
            if self.config.include_metadata and result.metadata:
                import json
                extra_files['metadata.json'] = json.dumps(result.metadata.to_dict())

            if self.config.save_extra_files and extra_files:
                torch.jit.save(scripted_model, str(output_path), _extra_files=extra_files)
            else:
                torch.jit.save(scripted_model, str(output_path))

            result.success = True
            result.output_path = str(output_path)

            # Get file size
            if output_path.exists():
                result.model_size_mb = output_path.stat().st_size / (1024**2)

            # Save separate metadata file if configured
            if self.config.include_metadata and result.metadata:
                if self.config.metadata_path:
                    result.metadata.save(self.config.metadata_path)
                    result.metadata_path = self.config.metadata_path
                else:
                    metadata_path = output_path.with_suffix('.metadata.json')
                    result.metadata.save(str(metadata_path))
                    result.metadata_path = str(metadata_path)

            # Validate export
            if self.config.validate_export and sample_input is not None:
                self._validate_export(model, scripted_model, sample_input, result)

            logger.info(f"TorchScript export successful: {result.model_size_mb:.2f} MB")

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"TorchScript export failed: {e}")

        return result

    def _determine_best_method(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor]
    ) -> str:
        """Determine the best export method for a model."""
        # Try scripting first as it's more robust
        try:
            test_scripted = torch.jit.script(model)
            return ExportMethod.SCRIPT
        except Exception as script_error:
            logger.debug(f"Scripting not possible: {script_error}")

        # Fall back to tracing if sample_input is available
        if sample_input is not None:
            return ExportMethod.TRACE

        # Default to script (may fail later with better error)
        return ExportMethod.SCRIPT

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for export."""
        model.eval()

        # Handle DataParallel wrapper
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        return model

    def _export_trace(
        self,
        model: nn.Module,
        sample_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.jit.ScriptModule:
        """Export using torch.jit.trace."""
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                sample_input,
                check_trace=self.config.check_trace,
                strict=self.config.strict
            )
        return traced_model

    def _export_script(self, model: nn.Module) -> torch.jit.ScriptModule:
        """Export using torch.jit.script."""
        scripted_model = torch.jit.script(model)
        return scripted_model

    def _freeze_model(
        self,
        scripted_model: torch.jit.ScriptModule
    ) -> torch.jit.ScriptModule:
        """Freeze model for inference optimization."""
        try:
            frozen_model = torch.jit.freeze(scripted_model)
            logger.debug("Model frozen for inference")
            return frozen_model
        except Exception as e:
            logger.warning(f"Failed to freeze model: {e}")
            return scripted_model

    def _optimize_for_inference(
        self,
        scripted_model: torch.jit.ScriptModule
    ) -> torch.jit.ScriptModule:
        """Apply inference optimizations."""
        try:
            optimized_model = torch.jit.optimize_for_inference(scripted_model)
            logger.debug("Applied inference optimizations")
            return optimized_model
        except Exception as e:
            logger.warning(f"Failed to optimize for inference: {e}")
            return scripted_model

    def _optimize_for_mobile(
        self,
        scripted_model: torch.jit.ScriptModule
    ) -> torch.jit.ScriptModule:
        """Apply mobile optimizations."""
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            mobile_model = optimize_for_mobile(scripted_model)
            logger.debug("Applied mobile optimizations")
            return mobile_model
        except ImportError:
            logger.warning("Mobile optimizer not available")
            return scripted_model
        except Exception as e:
            logger.warning(f"Failed to optimize for mobile: {e}")
            return scripted_model

    def _validate_export(
        self,
        original_model: nn.Module,
        scripted_model: torch.jit.ScriptModule,
        sample_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        result: TorchScriptExportResult
    ) -> None:
        """Validate exported model against original."""
        try:
            import numpy as np

            # Get outputs
            with torch.no_grad():
                original_output = original_model(sample_input)
                scripted_output = scripted_model(sample_input)

            # Handle tuple outputs
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            if isinstance(scripted_output, tuple):
                scripted_output = scripted_output[0]

            # Compare
            original_np = original_output.cpu().numpy()
            scripted_np = scripted_output.cpu().numpy()

            max_error = np.max(np.abs(original_np - scripted_np))
            result.validation_error = float(max_error)

            if max_error <= self.config.validation_tolerance:
                result.validated = True
                result.validation_message = f"Validation PASSED (max error: {max_error:.2e})"
                logger.info(result.validation_message)
            else:
                result.validated = False
                result.validation_message = f"Validation FAILED (max error: {max_error:.2e} > {self.config.validation_tolerance})"
                result.warnings.append(result.validation_message)
                logger.warning(result.validation_message)

        except Exception as e:
            result.validated = False
            result.validation_message = f"Validation error: {e}"
            result.warnings.append(result.validation_message)
            logger.warning(f"TorchScript validation failed: {e}")

    def load(
        self,
        model_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Tuple[torch.jit.ScriptModule, Optional[OptimizationMetadata]]:
        """
        Load a TorchScript model with its metadata.

        Args:
            model_path: Path to the saved model
            map_location: Device mapping for loading

        Returns:
            Tuple of (loaded_model, metadata)
        """
        model_path = Path(model_path)

        # Load model with extra files
        extra_files = {'metadata.json': ''}

        try:
            model = torch.jit.load(
                str(model_path),
                map_location=map_location,
                _extra_files=extra_files
            )
        except Exception:
            # Try loading without extra files
            model = torch.jit.load(str(model_path), map_location=map_location)
            extra_files = {}

        # Parse metadata
        metadata = None
        if extra_files.get('metadata.json'):
            import json
            metadata_dict = json.loads(extra_files['metadata.json'])
            metadata = OptimizationMetadata.from_dict(metadata_dict)
        elif model_path.with_suffix('.metadata.json').exists():
            metadata = OptimizationMetadata.load(str(model_path.with_suffix('.metadata.json')))

        return model, metadata


def export_to_torchscript(
    model: nn.Module,
    output_path: Union[str, Path],
    sample_input: Optional[torch.Tensor] = None,
    method: str = "trace",
    optimization_level: str = "balanced",
    **kwargs
) -> TorchScriptExportResult:
    """
    Convenience function for TorchScript export.

    Args:
        model: PyTorch model to export
        output_path: Output .pt file path
        sample_input: Sample input for tracing
        method: Export method (trace, script, auto)
        optimization_level: Optimization level applied
        **kwargs: Additional export arguments

    Returns:
        TorchScriptExportResult with export status
    """
    exporter = TorchScriptExporter()
    return exporter.export(
        model=model,
        output_path=output_path,
        sample_input=sample_input,
        method=method,
        optimization_level=optimization_level,
        **kwargs
    )


def load_torchscript(
    model_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.jit.ScriptModule, Optional[OptimizationMetadata]]:
    """
    Convenience function to load TorchScript model with metadata.

    Args:
        model_path: Path to saved model
        map_location: Device mapping

    Returns:
        Tuple of (model, metadata)
    """
    exporter = TorchScriptExporter()
    return exporter.load(model_path, map_location)
