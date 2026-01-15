"""
Model Deployment Module for KernelPyTorch

This module provides export functionality for deploying optimized PyTorch models
to various formats (ONNX, TorchScript) while preserving optimization metadata.

Key Components:
- ONNX Export: Export models to ONNX format with optimization metadata
- TorchScript Export: Export models to TorchScript (trace/script) format
- Optimization Metadata: Schema for preserving optimization information

Example:
    ```python
    from kernel_pytorch.deployment import (
        ONNXExporter,
        TorchScriptExporter,
        export_to_onnx,
        export_to_torchscript
    )

    # ONNX export
    result = export_to_onnx(
        model=optimized_model,
        output_path="model.onnx",
        sample_input=torch.randn(1, 512)
    )

    # TorchScript export
    result = export_to_torchscript(
        model=optimized_model,
        output_path="model.pt",
        sample_input=torch.randn(1, 512),
        method="trace"
    )
    ```

Version: 0.3.8
"""

# Optimization metadata
from .optimization_metadata import (
    OptimizationMetadata,
    HardwareMetadata,
    PrecisionMetadata,
    FusionMetadata,
    PerformanceMetadata,
    ModelMetadata,
    ExportFormat,
    create_metadata,
)

# ONNX export
from .onnx_exporter import (
    ONNXExporter,
    ONNXExportConfig,
    ONNXExportResult,
    export_to_onnx,
)

# TorchScript export
from .torchscript_exporter import (
    TorchScriptExporter,
    TorchScriptExportConfig,
    TorchScriptExportResult,
    ExportMethod,
    export_to_torchscript,
    load_torchscript,
)

__version__ = "0.3.8"

__all__ = [
    # Metadata
    "OptimizationMetadata",
    "HardwareMetadata",
    "PrecisionMetadata",
    "FusionMetadata",
    "PerformanceMetadata",
    "ModelMetadata",
    "ExportFormat",
    "create_metadata",
    # ONNX
    "ONNXExporter",
    "ONNXExportConfig",
    "ONNXExportResult",
    "export_to_onnx",
    # TorchScript
    "TorchScriptExporter",
    "TorchScriptExportConfig",
    "TorchScriptExportResult",
    "ExportMethod",
    "export_to_torchscript",
    "load_torchscript",
]
