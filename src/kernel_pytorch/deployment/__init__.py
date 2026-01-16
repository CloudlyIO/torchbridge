"""
Model Deployment Module for KernelPyTorch

This module provides export and serving functionality for deploying optimized
PyTorch models to various formats and inference platforms.

Key Components:
- ONNX Export: Export models to ONNX format with optimization metadata
- TorchScript Export: Export models to TorchScript (trace/script) format
- Optimization Metadata: Schema for preserving optimization information
- Inference Serving: TorchServe, Triton, and FastAPI integrations (v0.3.9)

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

    # FastAPI inference server (v0.3.9)
    from kernel_pytorch.deployment.serving import create_fastapi_server
    server = create_fastapi_server(model, model_name="my_model")
    ```

Version: 0.3.9
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

__version__ = "0.3.11"

# Serving module (v0.3.9)
from .serving import (
    # TorchServe
    KernelPyTorchHandler,
    BaseHandler,
    create_torchserve_handler,
    package_for_torchserve,
    # Triton
    TritonModelConfig,
    TritonBackend,
    TritonDataType,
    create_triton_config,
    generate_triton_model_repository,
    # FastAPI
    InferenceServer,
    ServerConfig,
    create_fastapi_server,
    run_server,
)

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
    # Serving - TorchServe (v0.3.9)
    "KernelPyTorchHandler",
    "BaseHandler",
    "create_torchserve_handler",
    "package_for_torchserve",
    # Serving - Triton (v0.3.9)
    "TritonModelConfig",
    "TritonBackend",
    "TritonDataType",
    "create_triton_config",
    "generate_triton_model_repository",
    # Serving - FastAPI (v0.3.9)
    "InferenceServer",
    "ServerConfig",
    "create_fastapi_server",
    "run_server",
]
