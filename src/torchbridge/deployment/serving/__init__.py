"""
Inference Serving Module for TorchBridge

This module provides integration with popular inference serving platforms:
- TorchServe: PyTorch's native model serving solution
- Triton Inference Server: NVIDIA's multi-framework serving platform
- FastAPI: Lightweight REST API server with health checks
- LLM Server: Production LLM inference with streaming and batching

Key Components:
- TorchServe Handler: Custom handler for TorchServe deployment
- Triton Configuration: Model configuration for Triton deployment
- FastAPI Server: REST API server with batching and health checks
- LLM Server: Specialized server for LLMs with streaming and dynamic batching

Example:
    ```python
    from torchbridge.deployment.serving import (
        TorchBridgeHandler,
        TritonModelConfig,
        create_fastapi_server,
        create_llm_server
    )

    # TorchServe: Use TorchBridgeHandler in your model archive
    # Triton: Generate config with TritonModelConfig
    # FastAPI: Create server with create_fastapi_server(model)
    # LLM Server: Create LLM server with create_llm_server(model, tokenizer)
    ```

Version: 0.4.22
"""

from .fastapi_server import (
    BatchInferenceRequest,
    HealthStatus,
    InferenceRequest,
    InferenceResponse,
    InferenceServer,
    ServerConfig,
    create_fastapi_server,
    run_server,
)
from .llm_server import (
    ChatCompletionRequest,
    ChatMessage,
    GenerateRequest,
    GenerateResponse,
    LLMInferenceServer,
    LLMServerConfig,
    TokenCountRequest,
    TokenCountResponse,
    create_llm_server,
    run_llm_server,
)
from .torchserve_handler import (
    BaseHandler,
    TorchBridgeHandler,
    create_torchserve_handler,
    package_for_torchserve,
)
from .triton_config import (
    TritonBackend,
    TritonDataType,
    TritonDynamicBatching,
    TritonInput,
    TritonInstanceGroup,
    TritonModelConfig,
    TritonOutput,
    create_triton_config,
    generate_triton_model_repository,
)

__version__ = "0.5.0"

__all__ = [
    # TorchServe
    "TorchBridgeHandler",
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
    # LLM Server
    "LLMInferenceServer",
    "LLMServerConfig",
    "create_llm_server",
    "run_llm_server",
    "GenerateRequest",
    "GenerateResponse",
    "ChatMessage",
    "ChatCompletionRequest",
    "TokenCountRequest",
    "TokenCountResponse",
]
