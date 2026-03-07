"""
Model Deployment Module for TorchBridge

This module provides export and serving functionality for deploying HAL-managed
PyTorch models to various formats and inference platforms.

Key Components:
- ONNX Export: Export models to ONNX format with backend metadata
- TorchScript Export: Export models to TorchScript (trace/script) format
- Backend Metadata: Schema for preserving hardware abstraction information
- Inference Serving: TorchServe, Triton, and FastAPI integrations

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

    # FastAPI inference server
    from torchbridge.deployment.serving import create_fastapi_server
    server = create_fastapi_server(model, model_name="my_model")
    ```

"""

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

# Production validation
from .production_validator import (
    ProductionRequirements,
    ProductionValidationResult,
    ProductionValidator,
    ValidationCheck,
    ValidationSeverity,
    ValidationStatus,
    validate_production_readiness,
)

# Serving module
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
    # Production Validation
    "ProductionValidator",
    "ProductionRequirements",
    "ProductionValidationResult",
    "ValidationCheck",
    "ValidationSeverity",
    "ValidationStatus",
    "validate_production_readiness",
    # Serving - TorchServe
    "TorchBridgeHandler",
    "BaseHandler",
    "create_torchserve_handler",
    "package_for_torchserve",
    # Serving - Triton
    "TritonModelConfig",
    "TritonBackend",
    "TritonDataType",
    "create_triton_config",
    "generate_triton_model_repository",
    # Serving - FastAPI
    "InferenceServer",
    "ServerConfig",
    "create_fastapi_server",
    "run_server",
]
