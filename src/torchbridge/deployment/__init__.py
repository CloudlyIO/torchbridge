"""
Model Deployment Module for TorchBridge

This module provides export and serving functionality for deploying optimized
PyTorch models to various formats and inference platforms.

Key Components:
- ONNX Export: Export models to ONNX format with optimization metadata
- TorchScript Export: Export models to TorchScript (trace/script) format
- Optimization Metadata: Schema for preserving optimization information
- Inference Serving: TorchServe, Triton, and FastAPI integrations (v0.3.9)

Example:
    ```python
    from torchbridge.deployment import (
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
    from torchbridge.deployment.serving import create_fastapi_server
    server = create_fastapi_server(model, model_name="my_model")
    ```

Version: 0.3.9
"""

# Optimization metadata
# ONNX export
from .onnx_exporter import (
    ONNXExportConfig,
    ONNXExporter,
    ONNXExportResult,
    export_to_onnx,
)
from .optimization_metadata import (
    ExportFormat,
    FusionMetadata,
    HardwareMetadata,
    ModelMetadata,
    OptimizationMetadata,
    PerformanceMetadata,
    PrecisionMetadata,
    create_metadata,
)

# Production validation (v0.4.25)
from .production_validator import (
    ProductionRequirements,
    ProductionValidationResult,
    ProductionValidator,
    ValidationCheck,
    ValidationSeverity,
    ValidationStatus,
    validate_production_readiness,
)

# SafeTensors export (v0.4.25)
from .safetensors_exporter import (
    SafeTensorsExportConfig,
    SafeTensorsExporter,
    SafeTensorsExportResult,
    export_to_safetensors,
    load_model_safetensors,
    load_safetensors,
)

# TorchScript export
from .torchscript_exporter import (
    ExportMethod,
    TorchScriptExportConfig,
    TorchScriptExporter,
    TorchScriptExportResult,
    export_to_torchscript,
    load_torchscript,
)

__version__ = "0.4.41"

# Serving module (v0.3.9)
from .serving import (
    BaseHandler,
    # FastAPI
    InferenceServer,
    ServerConfig,
    # TorchServe
    TorchBridgeHandler,
    TritonBackend,
    TritonDataType,
    # Triton
    TritonModelConfig,
    create_fastapi_server,
    create_torchserve_handler,
    create_triton_config,
    generate_triton_model_repository,
    package_for_torchserve,
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
    # SafeTensors (v0.4.25)
    "SafeTensorsExporter",
    "SafeTensorsExportConfig",
    "SafeTensorsExportResult",
    "export_to_safetensors",
    "load_safetensors",
    "load_model_safetensors",
    # Production Validation (v0.4.25)
    "ProductionValidator",
    "ProductionRequirements",
    "ProductionValidationResult",
    "ValidationCheck",
    "ValidationSeverity",
    "ValidationStatus",
    "validate_production_readiness",
    # Serving - TorchServe (v0.3.9)
    "TorchBridgeHandler",
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
