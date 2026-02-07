"""
FastAPI Inference Server for TorchBridge

This module provides a lightweight REST API server for serving TorchBridge-
optimized models with health checks, metrics, and batch inference support.

Features:
- REST API for model inference
- Health check endpoints (liveness/readiness)
- Prometheus metrics endpoint
- Batch inference support
- Async request handling
- Model hot-reloading support

Example:
    ```python
    from torchbridge.deployment.serving import create_fastapi_server, run_server

    # Create and run server
    server = create_fastapi_server(model, model_name="my_model")
    run_server(server, host="0.0.0.0", port=8000)
    ```

"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Optional imports - server will work without these
try:
    from fastapi import FastAPI, HTTPException, Request, Response  # noqa: F401
    from fastapi.responses import JSONResponse  # noqa: F401
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create stub classes for type hints
    class BaseModel:
        pass
    FastAPI = None

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False

class HealthStatus(Enum):
    """Server health status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class ServerConfig:
    """Configuration for the inference server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Model settings
    model_name: str = "model"
    model_version: str = "1.0"
    device: str = "auto"  # auto, cuda, cpu

    # Inference settings
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    enable_fp16: bool = True

    # Health check settings
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30

    # Metrics settings
    enable_metrics: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# Pydantic models for request/response (only if FastAPI available)
if FASTAPI_AVAILABLE:
    class InferenceRequest(BaseModel):
        """Request model for single inference."""

        input: list[float] = Field(..., description="Input tensor data as flat list")
        dtype: str = Field(default="float32", description="Data type")
        shape: list[int] | None = Field(default=None, description="Optional tensor shape")

    class BatchInferenceRequest(BaseModel):
        """Request model for batch inference."""

        inputs: list[list[float]] = Field(..., description="Batch of input tensors")
        dtype: str = Field(default="float32", description="Data type")
        shape: list[int] | None = Field(default=None, description="Shape per input")

    class InferenceResponse(BaseModel):
        """Response model for inference."""

        output: list[float] = Field(..., description="Output tensor data")
        inference_time_ms: float = Field(..., description="Inference time in milliseconds")
        model_name: str = Field(..., description="Model name")
        model_version: str = Field(..., description="Model version")

    class HealthResponse(BaseModel):
        """Response model for health checks."""

        status: str = Field(..., description="Health status")
        model_name: str = Field(..., description="Model name")
        model_loaded: bool = Field(..., description="Whether model is loaded")
        device: str = Field(..., description="Device being used")
        uptime_seconds: float = Field(..., description="Server uptime")
        inference_count: int = Field(..., description="Total inference count")
        average_latency_ms: float = Field(..., description="Average inference latency")

    class MetricsResponse(BaseModel):
        """Response model for metrics."""

        inference_count: int
        total_inference_time_ms: float
        average_inference_time_ms: float
        last_inference_time_ms: float
        model_name: str
        device: str
        memory_allocated_mb: float
        memory_reserved_mb: float
else:
    # Stub classes when FastAPI not available
    InferenceRequest = dict[str, Any]
    BatchInferenceRequest = dict[str, Any]
    InferenceResponse = dict[str, Any]
    HealthResponse = dict[str, Any]
    MetricsResponse = dict[str, Any]

class InferenceServer:
    """
    FastAPI-based inference server for PyTorch models.

    Provides REST endpoints for:
    - POST /predict: Single inference
    - POST /predict/batch: Batch inference
    - GET /health: Health check
    - GET /health/live: Liveness probe
    - GET /health/ready: Readiness probe
    - GET /metrics: Prometheus-style metrics
    """

    def __init__(
        self,
        model: nn.Module,
        config: ServerConfig | None = None,
    ):
        """
        Initialize the inference server.

        Args:
            model: PyTorch model to serve
            config: Server configuration
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for InferenceServer. "
                "Install with: pip install fastapi uvicorn"
            )

        self.config = config or ServerConfig()
        self.model = model
        self.device: torch.device | None = None
        self.status = HealthStatus.STARTING
        self.start_time = time.time()

        # Metrics
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._last_inference_time = 0.0
        self._lock = threading.Lock()

        # Initialize device and model
        self._setup_device()
        self._setup_model()

        # Create FastAPI app
        self.app = self._create_app()

    def _setup_device(self) -> None:
        """Set up the inference device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(f"Using device: {self.device}")

    def _setup_model(self) -> None:
        """Set up the model for inference."""
        self.model = self.model.to(self.device)
        self.model.eval()

        # Apply FP16 optimization
        if self.config.enable_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Model converted to FP16")

        # Try torch.compile
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        self.status = HealthStatus.HEALTHY
        logger.info("Model setup complete")

    def _create_app(self) -> "FastAPI":
        """Create the FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Lifespan context manager for startup/shutdown."""
            logger.info(f"Starting inference server for {self.config.model_name}")
            self.status = HealthStatus.HEALTHY
            yield
            logger.info("Shutting down inference server")
            self.status = HealthStatus.SHUTTING_DOWN

        app = FastAPI(
            title=f"TorchBridge Inference Server - {self.config.model_name}",
            description="REST API for model inference with TorchBridge optimizations",
            version=self.config.model_version,
            lifespan=lifespan,
        )

        # Register routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: "FastAPI") -> None:
        """Register API routes."""

        @app.post("/predict", response_model=InferenceResponse)
        async def predict(request: InferenceRequest) -> InferenceResponse:
            """Single inference endpoint."""
            return await self._handle_inference(request)

        @app.post("/predict/batch")
        async def predict_batch(request: BatchInferenceRequest) -> list[InferenceResponse]:
            """Batch inference endpoint."""
            return await self._handle_batch_inference(request)

        @app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            """Health check endpoint."""
            return self._get_health_response()

        @app.get("/health/live")
        async def liveness() -> dict[str, str]:
            """Kubernetes liveness probe."""
            return {"status": "alive"}

        @app.get("/health/ready")
        async def readiness() -> dict[str, Any]:
            """Kubernetes readiness probe."""
            if self.status == HealthStatus.HEALTHY:
                return {"status": "ready", "model_loaded": True}
            raise HTTPException(status_code=503, detail="Service not ready")

        @app.get("/metrics")
        async def metrics() -> MetricsResponse:
            """Prometheus-style metrics endpoint."""
            return self._get_metrics_response()

        @app.get("/")
        async def root() -> dict[str, Any]:
            """Root endpoint with server info."""
            return {
                "service": "TorchBridge Inference Server",
                "model_name": self.config.model_name,
                "model_version": self.config.model_version,
                "status": self.status.value,
                "endpoints": [
                    "POST /predict",
                    "POST /predict/batch",
                    "GET /health",
                    "GET /health/live",
                    "GET /health/ready",
                    "GET /metrics",
                ],
            }

    async def _handle_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Handle single inference request."""
        start_time = time.time()

        try:
            # Convert input to tensor
            input_data = request.input
            if request.shape:
                tensor = torch.tensor(input_data, dtype=torch.float32).reshape(request.shape)
            else:
                tensor = torch.tensor(input_data, dtype=torch.float32)

            # Add batch dimension if needed
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            # Run inference
            output = await self._run_inference(tensor)

            # Calculate timing
            inference_time = (time.time() - start_time) * 1000

            # Update metrics
            with self._lock:
                self._inference_count += 1
                self._total_inference_time += inference_time
                self._last_inference_time = inference_time

            return InferenceResponse(
                output=output.squeeze(0).tolist(),
                inference_time_ms=inference_time,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def _handle_batch_inference(
        self, request: BatchInferenceRequest
    ) -> list[InferenceResponse]:
        """Handle batch inference request."""
        start_time = time.time()

        try:
            # Convert inputs to tensor batch
            batch = []
            for input_data in request.inputs:
                if request.shape:
                    tensor = torch.tensor(input_data, dtype=torch.float32).reshape(request.shape)
                else:
                    tensor = torch.tensor(input_data, dtype=torch.float32)
                batch.append(tensor)

            batch_tensor = torch.stack(batch)

            # Run inference
            outputs = await self._run_inference(batch_tensor)

            # Calculate timing
            inference_time = (time.time() - start_time) * 1000
            per_sample_time = inference_time / len(request.inputs)

            # Update metrics
            with self._lock:
                self._inference_count += len(request.inputs)
                self._total_inference_time += inference_time
                self._last_inference_time = inference_time

            # Build responses
            responses = []
            for i in range(outputs.size(0)):
                responses.append(
                    InferenceResponse(
                        output=outputs[i].tolist(),
                        inference_time_ms=per_sample_time,
                        model_name=self.config.model_name,
                        model_version=self.config.model_version,
                    )
                )

            return responses

        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def _run_inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference."""
        # Move to device and convert dtype
        tensor = tensor.to(self.device)
        if self.config.enable_fp16 and self.device.type == "cuda":
            tensor = tensor.half()

        # Run inference in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self._sync_inference, tensor)

        return output.cpu().float()

    def _sync_inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """Synchronous inference (called from thread pool)."""
        with torch.no_grad():
            return self.model(tensor)

    def _get_health_response(self) -> HealthResponse:
        """Get health check response."""
        uptime = time.time() - self.start_time
        avg_latency = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0.0
        )

        return HealthResponse(
            status=self.status.value,
            model_name=self.config.model_name,
            model_loaded=self.model is not None,
            device=str(self.device),
            uptime_seconds=uptime,
            inference_count=self._inference_count,
            average_latency_ms=avg_latency,
        )

    def _get_metrics_response(self) -> MetricsResponse:
        """Get metrics response."""
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0.0
        )

        # Get memory stats
        if self.device and self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0

        return MetricsResponse(
            inference_count=self._inference_count,
            total_inference_time_ms=self._total_inference_time,
            average_inference_time_ms=avg_time,
            last_inference_time_ms=self._last_inference_time,
            model_name=self.config.model_name,
            device=str(self.device),
            memory_allocated_mb=memory_allocated,
            memory_reserved_mb=memory_reserved,
        )

def create_fastapi_server(
    model: nn.Module,
    model_name: str = "model",
    model_version: str = "1.0",
    device: str = "auto",
    enable_fp16: bool = True,
    max_batch_size: int = 32,
) -> InferenceServer:
    """
    Create a FastAPI inference server.

    Args:
        model: PyTorch model to serve
        model_name: Name of the model
        model_version: Version of the model
        device: Device to use (auto, cuda, cpu)
        enable_fp16: Enable FP16 inference
        max_batch_size: Maximum batch size

    Returns:
        Configured InferenceServer instance
    """
    config = ServerConfig(
        model_name=model_name,
        model_version=model_version,
        device=device,
        enable_fp16=enable_fp16,
        max_batch_size=max_batch_size,
    )

    return InferenceServer(model=model, config=config)

def run_server(
    server: InferenceServer,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """
    Run the inference server.

    Args:
        server: InferenceServer instance
        host: Host to bind to
        port: Port to bind to
        workers: Number of workers
        log_level: Logging level
    """
    if not UVICORN_AVAILABLE:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install uvicorn"
        )

    logger.info(f"Starting server at {host}:{port}")

    uvicorn.run(
        server.app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
    )
