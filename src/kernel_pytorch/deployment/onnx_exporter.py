"""
ONNX Exporter with Optimization Preservation

This module provides ONNX export functionality that preserves optimization
metadata and validates exported models.

Features:
- Export PyTorch models to ONNX format
- Preserve optimization metadata as ONNX metadata properties
- Support for dynamic axes (variable batch size, sequence length)
- Automatic input/output shape detection
- Export validation and verification
- Quantization-aware export

Example:
    ```python
    from kernel_pytorch.deployment import ONNXExporter

    exporter = ONNXExporter()
    result = exporter.export(
        model=optimized_model,
        output_path="model.onnx",
        sample_input=torch.randn(1, 512),
        optimization_level="aggressive"
    )

    print(f"Exported to: {result.output_path}")
    print(f"Validation: {'PASSED' if result.validated else 'FAILED'}")
    ```
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .optimization_metadata import OptimizationMetadata, create_metadata

logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""
    # ONNX export settings
    opset_version: int = 17  # ONNX opset version
    do_constant_folding: bool = True
    export_params: bool = True
    verbose: bool = False

    # Dynamic axes configuration
    dynamic_axes: dict[str, dict[int, str]] | None = None
    dynamic_batch: bool = True
    dynamic_sequence: bool = True

    # Validation settings
    validate_export: bool = True
    validation_tolerance: float = 1e-5
    validation_iterations: int = 3

    # Optimization settings
    optimize_for_inference: bool = True
    fp16_export: bool = False
    quantize_export: bool = False

    # Metadata settings
    include_metadata: bool = True
    metadata_path: str | None = None  # Separate metadata file path

    # Input/output names
    input_names: list[str] | None = None
    output_names: list[str] | None = None


@dataclass
class ONNXExportResult:
    """Result of ONNX export operation."""
    success: bool = False
    output_path: str = ""
    metadata_path: str | None = None
    metadata: OptimizationMetadata | None = None

    # Validation results
    validated: bool = False
    validation_error: float | None = None
    validation_message: str = ""

    # Export details
    opset_version: int = 17
    model_size_mb: float = 0.0
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    output_shapes: dict[str, list[int]] = field(default_factory=dict)
    dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)

    # Warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ONNXExporter:
    """
    ONNX Exporter with optimization preservation.

    This class handles exporting PyTorch models to ONNX format while
    preserving optimization metadata and validating the exported model.
    """

    def __init__(self, config: ONNXExportConfig | None = None):
        """
        Initialize ONNX exporter.

        Args:
            config: Export configuration. If None, uses defaults.
        """
        self.config = config or ONNXExportConfig()
        self._onnx_available = self._check_onnx_available()
        self._onnxruntime_available = self._check_onnxruntime_available()

    def _check_onnx_available(self) -> bool:
        """Check if ONNX is available."""
        try:
            import onnx  # noqa: F401
            return True
        except ImportError:
            logger.warning("ONNX not available. Install with: pip install onnx")
            return False

    def _check_onnxruntime_available(self) -> bool:
        """Check if ONNX Runtime is available for validation."""
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            logger.warning("ONNX Runtime not available. Validation disabled. Install with: pip install onnxruntime")
            return False

    def export(
        self,
        model: nn.Module,
        output_path: str | Path,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor],
        optimization_level: str = "balanced",
        backend: str = "auto",
        benchmark: bool = True,
        **kwargs
    ) -> ONNXExportResult:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            output_path: Path for the output ONNX file
            sample_input: Sample input for tracing (tensor, tuple, or dict)
            optimization_level: Optimization level applied to model
            backend: Hardware backend (auto, cuda, tpu, amd, cpu)
            benchmark: Whether to run benchmarks for metadata
            **kwargs: Additional arguments passed to torch.onnx.export

        Returns:
            ONNXExportResult with export status and details
        """
        result = ONNXExportResult()
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare model for export
            model = self._prepare_model(model)

            # Prepare sample input
            sample_input, input_names = self._prepare_input(sample_input)

            # Setup dynamic axes
            dynamic_axes = self._setup_dynamic_axes(sample_input, input_names)

            # Create optimization metadata
            if self.config.include_metadata:
                first_input = sample_input[0] if isinstance(sample_input, tuple) else sample_input
                metadata = create_metadata(
                    model=model,
                    backend=backend,
                    optimization_level=optimization_level,
                    export_format="onnx",
                    sample_input=first_input,
                    benchmark=benchmark
                )
                result.metadata = metadata

            # Perform ONNX export
            logger.info(f"Exporting model to ONNX: {output_path}")
            self._export_to_onnx(
                model=model,
                sample_input=sample_input,
                output_path=str(output_path),
                input_names=input_names,
                dynamic_axes=dynamic_axes,
                **kwargs
            )

            result.success = True
            result.output_path = str(output_path)
            result.opset_version = self.config.opset_version
            result.dynamic_axes = dynamic_axes

            # Get file size
            if output_path.exists():
                result.model_size_mb = output_path.stat().st_size / (1024**2)

            # Add metadata to ONNX model
            if self.config.include_metadata and result.metadata:
                self._add_metadata_to_onnx(output_path, result.metadata)

                # Save separate metadata file if configured
                if self.config.metadata_path:
                    result.metadata.save(self.config.metadata_path)
                    result.metadata_path = self.config.metadata_path
                else:
                    # Default: save alongside ONNX file
                    metadata_path = output_path.with_suffix('.metadata.json')
                    result.metadata.save(str(metadata_path))
                    result.metadata_path = str(metadata_path)

            # Validate export
            if self.config.validate_export:
                self._validate_export(model, sample_input, output_path, result)

            # Optimize for inference if requested
            if self.config.optimize_for_inference and self._onnx_available:
                self._optimize_onnx(output_path)

            logger.info(f"ONNX export successful: {result.model_size_mb:.2f} MB")

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"ONNX export failed: {e}")

        return result

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for export."""
        model.eval()

        # Handle DataParallel wrapper
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        return model

    def _prepare_input(
        self,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | tuple, list[str]]:
        """Prepare sample input and generate input names."""
        if isinstance(sample_input, dict):
            # Dict input: use keys as names
            input_names = list(sample_input.keys())
            sample_input = tuple(sample_input.values())
        elif isinstance(sample_input, tuple):
            # Tuple input: generate names
            input_names = [f"input_{i}" for i in range(len(sample_input))]
        else:
            # Single tensor
            input_names = ["input"]
            sample_input = (sample_input,)

        # Use configured names if provided
        if self.config.input_names:
            input_names = self.config.input_names

        return sample_input, input_names

    def _setup_dynamic_axes(
        self,
        sample_input: torch.Tensor | tuple,
        input_names: list[str]
    ) -> dict[str, dict[int, str]]:
        """Setup dynamic axes configuration."""
        if self.config.dynamic_axes:
            return self.config.dynamic_axes

        dynamic_axes = {}

        # Get input tensors
        inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)

        for name, tensor in zip(input_names, inputs):
            axes = {}
            if self.config.dynamic_batch and tensor.dim() > 0:
                axes[0] = "batch_size"
            if self.config.dynamic_sequence and tensor.dim() > 1:
                axes[1] = "sequence_length"
            if axes:
                dynamic_axes[name] = axes

        # Add output dynamic axes
        output_names = self.config.output_names or ["output"]
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

        return dynamic_axes

    def _export_to_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple,
        output_path: str,
        input_names: list[str],
        dynamic_axes: dict[str, dict[int, str]],
        **kwargs
    ) -> None:
        """Perform the actual ONNX export."""
        output_names = self.config.output_names or ["output"]

        # Merge kwargs with config
        export_kwargs = {
            'f': output_path,
            'input_names': input_names,
            'output_names': output_names,
            'dynamic_axes': dynamic_axes,
            'opset_version': self.config.opset_version,
            'do_constant_folding': self.config.do_constant_folding,
            'export_params': self.config.export_params,
            'verbose': self.config.verbose,
        }
        export_kwargs.update(kwargs)

        # Suppress export warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            torch.onnx.export(model, sample_input, **export_kwargs)

    def _add_metadata_to_onnx(
        self,
        output_path: Path,
        metadata: OptimizationMetadata
    ) -> None:
        """Add optimization metadata to ONNX model properties."""
        if not self._onnx_available:
            return

        try:
            import onnx

            model = onnx.load(str(output_path))

            # Add metadata as model properties
            metadata.to_dict()

            # Add key metadata fields
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="kernel_pytorch_version", value=metadata.kernel_pytorch_version)
            )
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="optimization_level", value=metadata.optimization_level)
            )
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="export_timestamp", value=metadata.export_timestamp)
            )
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="hardware_backend", value=metadata.hardware.backend)
            )
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="precision", value=metadata.precision.default_dtype)
            )

            # Save updated model
            onnx.save(model, str(output_path))
            logger.debug("Added optimization metadata to ONNX model")

        except Exception as e:
            logger.warning(f"Failed to add metadata to ONNX model: {e}")

    def _validate_export(
        self,
        original_model: nn.Module,
        sample_input: torch.Tensor | tuple,
        output_path: Path,
        result: ONNXExportResult
    ) -> None:
        """Validate exported ONNX model against original."""
        if not self._onnxruntime_available:
            result.validation_message = "ONNX Runtime not available for validation"
            return

        try:
            import numpy as np
            import onnxruntime as ort

            # Get PyTorch output
            inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)
            with torch.no_grad():
                pytorch_output = original_model(*inputs)

            if isinstance(pytorch_output, tuple):
                pytorch_output = pytorch_output[0]

            pytorch_output = pytorch_output.cpu().numpy()

            # Get ONNX Runtime output
            session = ort.InferenceSession(str(output_path))
            input_names = [inp.name for inp in session.get_inputs()]

            ort_inputs = {}
            for name, tensor in zip(input_names, inputs):
                ort_inputs[name] = tensor.cpu().numpy()

            ort_output = session.run(None, ort_inputs)[0]

            # Compare outputs
            max_error = np.max(np.abs(pytorch_output - ort_output))
            np.mean(np.abs(pytorch_output - ort_output))

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
            logger.warning(f"ONNX validation failed: {e}")

    def _optimize_onnx(self, output_path: Path) -> None:
        """Apply ONNX optimizations for inference."""
        if not self._onnx_available:
            return

        try:
            import onnx
            from onnx import optimizer

            model = onnx.load(str(output_path))

            # Apply standard optimizations
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
            ]

            optimized_model = optimizer.optimize(model, passes)
            onnx.save(optimized_model, str(output_path))
            logger.debug("Applied ONNX inference optimizations")

        except ImportError:
            # onnx.optimizer may not be available in all versions
            logger.debug("ONNX optimizer not available, skipping optimizations")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    def validate_model(
        self,
        onnx_path: str | Path,
        sample_input: torch.Tensor,
        expected_output: torch.Tensor | None = None
    ) -> tuple[bool, str]:
        """
        Validate an existing ONNX model.

        Args:
            onnx_path: Path to ONNX model
            sample_input: Sample input tensor
            expected_output: Optional expected output for comparison

        Returns:
            Tuple of (is_valid, message)
        """
        if not self._onnx_available:
            return False, "ONNX not available"

        try:
            import onnx
            import onnxruntime as ort

            # Check model validity
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)

            # Run inference
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            ort_output = session.run(None, {input_name: sample_input.numpy()})[0]

            if expected_output is not None:
                import numpy as np
                max_error = np.max(np.abs(expected_output.numpy() - ort_output))
                if max_error > self.config.validation_tolerance:
                    return False, f"Output mismatch (max error: {max_error:.2e})"

            return True, "Model is valid"

        except Exception as e:
            return False, f"Validation failed: {e}"


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    sample_input: torch.Tensor,
    optimization_level: str = "balanced",
    **kwargs
) -> ONNXExportResult:
    """
    Convenience function for ONNX export.

    Args:
        model: PyTorch model to export
        output_path: Output ONNX file path
        sample_input: Sample input tensor
        optimization_level: Optimization level applied
        **kwargs: Additional export arguments

    Returns:
        ONNXExportResult with export status
    """
    exporter = ONNXExporter()
    return exporter.export(
        model=model,
        output_path=output_path,
        sample_input=sample_input,
        optimization_level=optimization_level,
        **kwargs
    )
