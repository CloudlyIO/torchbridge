"""
Inference Serving Module for KernelPyTorch

This module provides integration with popular inference serving platforms:
- TorchServe: PyTorch's native model serving solution
- Triton Inference Server: NVIDIA's multi-framework serving platform
- FastAPI: Lightweight REST API server with health checks

Key Components:
- TorchServe Handler: Custom handler for TorchServe deployment
- Triton Configuration: Model configuration for Triton deployment
- FastAPI Server: REST API server with batching and health checks

Example:
    ```python
    from kernel_pytorch.deployment.serving import (
        KernelPyTorchHandler,
        TritonModelConfig,
        create_fastapi_server
    )

    # TorchServe: Use KernelPyTorchHandler in your model archive
    # Triton: Generate config with TritonModelConfig
    # FastAPI: Create server with create_fastapi_server(model)
    ```

Version: 0.3.9
"""

from .torchserve_handler import (
    KernelPyTorchHandler,
    BaseHandler,
    create_torchserve_handler,
    package_for_torchserve,
)

from .triton_config import (
    TritonModelConfig,
    TritonBackend,
    TritonDataType,
    TritonInput,
    TritonOutput,
    TritonInstanceGroup,
    TritonDynamicBatching,
    create_triton_config,
    generate_triton_model_repository,
)

from .fastapi_server import (
    InferenceServer,
    ServerConfig,
    HealthStatus,
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    create_fastapi_server,
    run_server,
)

__version__ = "0.3.9"

__all__ = [
    # TorchServe
    "KernelPyTorchHandler",
    "BaseHandler",
    "create_torchserve_handler",
    "package_for_torchserve",
    # Triton
    "TritonModelConfig",
    "TritonBackend",
    "TritonDataType",
    "TritonInput",
    "TritonOutput",
    "TritonInstanceGroup",
    "TritonDynamicBatching",
    "create_triton_config",
    "generate_triton_model_repository",
    # FastAPI
    "InferenceServer",
    "ServerConfig",
    "HealthStatus",
    "InferenceRequest",
    "InferenceResponse",
    "BatchInferenceRequest",
    "create_fastapi_server",
    "run_server",
]
