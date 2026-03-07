"""
LLM-Specific FastAPI Inference Server for TorchBridge

This module provides a production-grade REST API server for serving LLMs
managed with TorchBridge's LLMOptimizer. Includes streaming, dynamic
batching, and specialized endpoints for text generation.

Features:
- Text generation endpoint with streaming support (SSE)
- Chat completion endpoint
- Dynamic batching for efficient throughput
- Token counting utilities
- Health checks and metrics
- Integration with LLMOptimizer for quantization

Supported Models:
- Qwen (Qwen3-0.6B, Qwen2.5 variants)
- LLaMA (2, 3)
- Mistral
- Phi-2, Phi-3
- DeepSeek

Example:
    ```python
    from torchbridge.deployment.serving import create_llm_server, run_llm_server
    from torchbridge.models.llm import LLMOptimizer, LLMConfig

    # Create optimizer and load model
    config = LLMConfig(model_name="Qwen/Qwen3-0.6B", quantization="int8")
    optimizer = LLMOptimizer(config)
    model, tokenizer = optimizer.optimize("Qwen/Qwen3-0.6B")

    # Create and run server
    server = create_llm_server(model, tokenizer, model_name="Qwen/Qwen3-0.6B")
    run_llm_server(server, host="0.0.0.0", port=8000)
    ```

"""

import asyncio
import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Optional imports - server will work without these
try:
    from fastapi import FastAPI, HTTPException, Request, Response  # noqa: F401
    from fastapi.responses import JSONResponse, StreamingResponse  # noqa: F401
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create stub classes for type hints
    class BaseModel:
        pass
    FastAPI: type | None = None

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False

try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    TRANSFORMERS_AVAILABLE = True
    TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]  # noqa: UP007
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TokenizerType = Any

class HealthStatus(Enum):
    """Server health status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class LLMServerConfig:
    """Configuration for the LLM inference server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Model settings
    model_name: str = "Qwen/Qwen3-0.6B"
    model_version: str = "1.0"
    device: str = "auto"  # auto, cuda, cpu

    # Generation defaults
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Batching settings
    enable_dynamic_batching: bool = True
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    # Requests within a batch are LEFT-padded so all sequence ends align,
    # which is correct for causal (decoder-only) LMs. max_batch_size controls
    # how many requests are packed into a single model.generate() call.

    # Performance settings
    enable_streaming: bool = True
    stream_interval_tokens: int = 1

    # Speculative decoding
    enable_speculative_decoding: bool = False
    speculative_method: str | None = None  # None = auto-select
    draft_model_name: str | None = None
    num_speculative_tokens: int = 5

    # Structured output
    enable_structured_output: bool = False

    # Health check settings
    enable_health_checks: bool = True
    enable_metrics: bool = True
    enable_llm_metrics: bool = True

    # Security settings
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("LLM_SERVER_API_KEY")
    )
    """If set, all data endpoints require ``Authorization: Bearer <api_key>``.
    Defaults to the ``LLM_SERVER_API_KEY`` environment variable if present."""
    rate_limit_rpm: int | None = None
    """Max requests per minute per client IP. ``None`` disables rate limiting."""
    cors_origins: list = field(default_factory=lambda: ["*"])
    """List of allowed CORS origins. Defaults to ``["*"]`` (all origins)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# Pydantic models for request/response (only if FastAPI available)
if FASTAPI_AVAILABLE:
    class GenerateRequest(BaseModel):
        """Request model for text generation."""

        prompt: str = Field(..., min_length=1, description="Input text prompt")
        max_new_tokens: int | None = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
        temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")
        top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling probability")
        top_k: int | None = Field(None, ge=1, description="Top-k sampling")
        repetition_penalty: float | None = Field(None, ge=0.1, le=10.0, description="Repetition penalty")
        do_sample: bool | None = Field(True, description="Enable sampling")
        stream: bool | None = Field(False, description="Enable streaming response")
        stop_sequences: list[str] | None = Field(None, description="Stop generation sequences")

    class ChatMessage(BaseModel):
        """Chat message format."""

        role: Literal["system", "user", "assistant"] = Field(..., description="Message role (system, user, assistant)")
        content: str = Field(..., description="Message content")

    class ChatCompletionRequest(BaseModel):
        """Request model for chat completion."""

        messages: list[ChatMessage] = Field(..., description="List of chat messages")
        max_new_tokens: int | None = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
        temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
        top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling")
        stream: bool | None = Field(False, description="Enable streaming")

    class GenerateResponse(BaseModel):
        """Response model for text generation."""

        generated_text: str = Field(..., description="Generated text")
        prompt: str = Field(..., description="Original prompt")
        num_tokens: int = Field(..., description="Number of tokens generated")
        inference_time_ms: float = Field(..., description="Inference time in milliseconds")
        model_name: str = Field(..., description="Model name")

    class TokenCountRequest(BaseModel):
        """Request model for token counting."""

        text: str = Field(..., description="Text to count tokens")

    class TokenCountResponse(BaseModel):
        """Response model for token counting."""

        text: str = Field(..., description="Input text")
        num_tokens: int = Field(..., description="Number of tokens")
        tokens: list[str] | None = Field(None, description="Token list (if available)")

    class HealthResponse(BaseModel):
        """Response model for health checks."""

        status: str = Field(..., description="Health status")
        model_name: str = Field(..., description="Model name")
        model_loaded: bool = Field(..., description="Whether model is loaded")
        device: str = Field(..., description="Device being used")
        uptime_seconds: float = Field(..., description="Server uptime")
        generation_count: int = Field(..., description="Total generation count")
        average_latency_ms: float = Field(..., description="Average generation latency")

    class MetricsResponse(BaseModel):
        """Response model for metrics."""

        generation_count: int
        total_generation_time_ms: float
        average_generation_time_ms: float
        last_generation_time_ms: float
        total_tokens_generated: int
        average_tokens_per_second: float
        model_name: str
        device: str
        memory_allocated_mb: float
        memory_reserved_mb: float
        ttft_p50_ms: float = 0.0
        ttft_p95_ms: float = 0.0
        ttft_p99_ms: float = 0.0
        tpot_p50_ms: float = 0.0
        tpot_p95_ms: float = 0.0
        tpot_p99_ms: float = 0.0
        cache_hit_rate: float = 0.0
        tokens_per_second: float = 0.0
        avg_batch_size: float = 0.0
else:
    # Stub classes when FastAPI not available
    GenerateRequest = dict[str, Any]  # type: ignore[assignment]
    ChatMessage = dict[str, Any]  # type: ignore[assignment]
    ChatCompletionRequest = dict[str, Any]  # type: ignore[assignment]
    GenerateResponse = dict[str, Any]  # type: ignore[assignment]
    TokenCountRequest = dict[str, Any]  # type: ignore[assignment]
    TokenCountResponse = dict[str, Any]  # type: ignore[assignment]
    HealthResponse = dict[str, Any]  # type: ignore[assignment]
    MetricsResponse = dict[str, Any]  # type: ignore[assignment]

class _RateLimiter:
    """Sliding-window (1 minute) in-memory rate limiter, keyed by client IP."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._window = 60.0
        self._counts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """Return True if the request is within the per-IP rate limit."""
        now = time.time()
        cutoff = now - self._window
        with self._lock:
            times = [t for t in self._counts.get(client_ip, []) if t > cutoff]
            if len(times) >= self._rpm:
                self._counts[client_ip] = times
                return False
            times.append(now)
            self._counts[client_ip] = times
            return True


@dataclass
class BatchItem:
    """Item in the dynamic batching queue."""
    request_id: str
    prompt: str
    input_ids: torch.Tensor
    generation_kwargs: dict[str, Any]
    result_queue: queue.Queue
    timestamp: float = field(default_factory=time.time)

class LLMInferenceServer:
    """
    FastAPI-based LLM inference server with streaming and dynamic batching.

    Provides REST endpoints for:
    - POST /generate: Text generation (with optional streaming)
    - POST /chat: Chat completion
    - POST /tokenize: Token counting
    - GET /health: Health check
    - GET /metrics: Prometheus-style metrics
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerType,
        config: LLMServerConfig | None = None,
    ):
        """
        Initialize the LLM inference server.

        Args:
            model: PyTorch LLM model (optimized with LLMOptimizer)
            tokenizer: HuggingFace tokenizer
            config: Server configuration
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for LLMInferenceServer. "
                "Install with: pip install fastapi uvicorn"
            )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for LLMInferenceServer. "
                "Install with: pip install transformers"
            )

        self.config = config or LLMServerConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.device: torch.device | None = None
        self.status = HealthStatus.STARTING
        self.start_time = time.time()

        # Metrics
        self._generation_count = 0
        self._total_generation_time = 0.0
        self._last_generation_time = 0.0
        self._total_tokens_generated = 0
        self._total_batch_requests = 0
        self._total_batches_processed = 0
        self._lock = threading.Lock()

        # Rate limiter (None when disabled)
        self._rate_limiter: _RateLimiter | None = (
            _RateLimiter(self.config.rate_limit_rpm)
            if self.config.rate_limit_rpm is not None
            else None
        )

        # Speculative decoding engine (initialized after _setup_device)
        self._speculation_engine: Any = None

        # LLM serving metrics
        self._llm_metrics = None
        if self.config.enable_llm_metrics:
            from torchbridge.monitoring.llm_metrics import LLMMetricsCollector
            self._llm_metrics = LLMMetricsCollector()

        # Dynamic batching
        self._batch_queue: deque = deque()
        self._batch_lock = threading.Lock()
        self._batch_thread: threading.Thread | None = None
        self._stop_batching = threading.Event()

        # Initialize device and model
        self._setup_device()
        self._setup_model()

        # Speculative decoding engine (after device is known)
        if self.config.enable_speculative_decoding:
            self._setup_speculation_engine()

        # Start batching thread if enabled
        if self.config.enable_dynamic_batching:
            self._start_batch_processor()

        # Create FastAPI app
        self.app = self._create_app()

        # Warn when binding to all interfaces without auth (easy to expose publicly)
        if self.config.api_key is None and self.config.host == "0.0.0.0":
            logger.warning(
                "LLM server binding to 0.0.0.0 (all interfaces) with no API key. "
                "Set LLMServerConfig(api_key=...) or the LLM_SERVER_API_KEY "
                "environment variable before deploying to a shared or public network."
            )

        # Warn when CORS allows all origins on an authenticated server
        if self.config.cors_origins == ["*"] and self.config.api_key is not None:
            logger.warning(
                "CORS allow_origins=['*'] permits any website to call this API. "
                "Set LLMServerConfig(cors_origins=[...]) with explicit origins "
                "for production deployments."
            )

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
        # Model should already be on correct device from LLMOptimizer
        if hasattr(self.model, 'device'):
            self.device = self.model.device  # type: ignore[assignment]
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.status = HealthStatus.HEALTHY
        logger.info("LLM server setup complete")

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------

    def _authenticate(self, request: "Request") -> None:
        """Raise ``HTTPException(401)`` when auth is enabled and key is wrong.

        Health and root endpoints are intentionally excluded from auth
        requirements — they must remain accessible for infrastructure probes.
        """
        if self.config.api_key is None:
            return
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authorization header required: Bearer <api_key>",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_header[len("Bearer "):]
        if not token or token != self.config.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _check_rate_limit(self, request: "Request") -> None:
        """Raise ``HTTPException(429)`` when the per-IP rate limit is exceeded."""
        if self._rate_limiter is None:
            return
        client_ip = request.client.host if request.client else "unknown"
        if not self._rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded: max {self.config.rate_limit_rpm} "
                    "requests/minute"
                ),
                headers={"Retry-After": "60"},
            )

    def _setup_speculation_engine(self) -> None:
        """Set up speculative decoding engine with detected backend."""
        from torchbridge.core.config import HardwareBackend
        from torchbridge.inference.speculative.engine import (
            SpeculationConfig,
            SpeculationEngine,
        )
        from torchbridge.inference.speculative.methods import SpeculativeMethod

        # Map device type to HardwareBackend
        if self.device is not None and self.device.type == "cuda":
            backend = HardwareBackend.CUDA
        else:
            backend = HardwareBackend.CPU

        method = None
        if self.config.speculative_method:
            method = SpeculativeMethod.from_string(self.config.speculative_method)
        spec_config = SpeculationConfig(
            method=method,
            draft_model_name=self.config.draft_model_name,
            num_speculative_tokens=self.config.num_speculative_tokens,
        )
        self._speculation_engine = SpeculationEngine(
            config=spec_config,
            backend=backend,
        )

    def _start_batch_processor(self) -> None:
        """Start the background batch processing thread."""
        self._batch_thread = threading.Thread(
            target=self._batch_processor_loop,
            daemon=True
        )
        self._batch_thread.start()
        logger.info("Started dynamic batching processor")

    def _batch_processor_loop(self) -> None:
        """Background loop for processing batched requests."""
        while not self._stop_batching.is_set():
            try:
                # Wait for timeout or until we have items
                time.sleep(self.config.batch_timeout_ms / 1000.0)

                with self._batch_lock:
                    if not self._batch_queue:
                        continue

                    # Get batch items (up to max_batch_size)
                    batch_items = []
                    while len(batch_items) < self.config.max_batch_size and self._batch_queue:
                        batch_items.append(self._batch_queue.popleft())

                if batch_items:
                    self._process_batch(batch_items)

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")

    def _process_batch(self, batch_items: list[BatchItem]) -> None:
        """Process a batch of generation requests.

        Pads all inputs LEFT so sequence ends align — required for causal LMs.
        Generates once for the whole batch, then distributes and truncates
        per-item outputs to each request's individual max_new_tokens limit.
        """
        try:
            input_ids_list = [item.input_ids for item in batch_items]

            # Resolve pad token — many tokenizers (e.g. GPT-2) have pad_token_id=None.
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            # LEFT-pad to the same length so all sequence ends are aligned.
            # Causal LMs generate from the last real token; right-padding would
            # shift that position and produce garbage for shorter sequences.
            max_len = max(ids.size(1) for ids in input_ids_list)
            padded_inputs = []
            attention_masks = []

            for ids in input_ids_list:
                pad_len = max_len - ids.size(1)
                if pad_len > 0:
                    padded = torch.nn.functional.pad(
                        ids, (pad_len, 0), value=pad_token_id
                    )
                else:
                    padded = ids
                padded_inputs.append(padded)
                # 1 for real tokens (right side), 0 for left-side padding
                attention_masks.append((padded != pad_token_id).long())

            batch_input_ids = torch.cat(padded_inputs, dim=0).to(self.device)
            batch_attention_mask = torch.cat(attention_masks, dim=0).to(self.device)

            # Build shared gen_kwargs from first item, then maximize max_new_tokens
            # so the item requesting the most tokens gets its full output.
            # Each item's output is truncated to its individual limit below.
            gen_kwargs = batch_items[0].generation_kwargs.copy()
            max_new_tokens_values = [
                item.generation_kwargs.get('max_new_tokens')
                for item in batch_items
                if item.generation_kwargs.get('max_new_tokens') is not None
            ]
            if max_new_tokens_values:
                gen_kwargs['max_new_tokens'] = max(max_new_tokens_values)
            gen_kwargs['attention_mask'] = batch_attention_mask

            # Single generate() call for the whole batch
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    batch_input_ids,
                    **gen_kwargs
                )

            generation_time = (time.time() - start_time) * 1000

            # Distribute and truncate results per item
            for i, item in enumerate(batch_items):
                output_ids = outputs[i:i+1]
                # Strip prompt tokens (original unpadded length)
                generated_ids = output_ids[:, input_ids_list[i].size(1):]
                # Truncate to this item's individually requested max_new_tokens
                item_max = item.generation_kwargs.get('max_new_tokens')
                if item_max is not None:
                    generated_ids = generated_ids[:, :item_max]
                generated_text = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                item.result_queue.put({
                    'generated_text': generated_text,
                    'num_tokens': generated_ids.size(1),
                    'inference_time_ms': generation_time / len(batch_items),
                })

            # Update metrics
            with self._lock:
                self._generation_count += len(batch_items)
                self._total_generation_time += generation_time
                self._last_generation_time = generation_time
                self._total_batch_requests += len(batch_items)
                self._total_batches_processed += 1
                total_tokens = sum(
                    outputs[i, input_ids_list[i].size(1):].numel()
                    for i in range(len(batch_items))
                )
                self._total_tokens_generated += total_tokens

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for item in batch_items:
                item.result_queue.put({'error': str(e)})

    def _create_app(self) -> "FastAPI":
        """Create the FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Lifespan context manager for startup/shutdown."""
            logger.info(f"Starting LLM inference server for {self.config.model_name}")
            self.status = HealthStatus.HEALTHY
            yield
            logger.info("Shutting down LLM inference server")
            self.status = HealthStatus.SHUTTING_DOWN
            self._stop_batching.set()
            if self._batch_thread:
                self._batch_thread.join(timeout=2.0)

        app = FastAPI(
            title=f"TorchBridge LLM Server - {self.config.model_name}",
            description="Production LLM inference server with streaming and batching",
            version=self.config.model_version,
            lifespan=lifespan,
        )

        # CORS middleware
        try:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=list(self.config.cors_origins),
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["Authorization", "Content-Type"],
            )
        except ImportError:
            pass

        # Register routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: "FastAPI") -> None:
        """Register API routes."""

        @app.post("/generate")
        async def generate(body: GenerateRequest, request: Request):
            """Text generation endpoint with optional streaming."""
            self._authenticate(request)
            self._check_rate_limit(request)
            if body.stream and self.config.enable_streaming:
                return StreamingResponse(
                    self._generate_stream(body),
                    media_type="text/event-stream"
                )
            else:
                return await self._generate_non_stream(body)

        @app.post("/chat")
        async def chat_completion(body: ChatCompletionRequest, request: Request):
            """Chat completion endpoint."""
            self._authenticate(request)
            self._check_rate_limit(request)
            # Convert chat messages to prompt
            prompt = self._format_chat_prompt(body.messages)

            # Create generation request
            gen_request = GenerateRequest(  # type: ignore[call-arg]
                prompt=prompt,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                stream=body.stream,
            )

            if body.stream and self.config.enable_streaming:
                return StreamingResponse(
                    self._generate_stream(gen_request),
                    media_type="text/event-stream"
                )
            else:
                return await self._generate_non_stream(gen_request)

        @app.post("/tokenize", response_model=TokenCountResponse)
        async def tokenize(body: TokenCountRequest, request: Request) -> TokenCountResponse:
            """Token counting endpoint."""
            self._authenticate(request)
            self._check_rate_limit(request)
            return self._count_tokens(body.text)

        @app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            """Health check endpoint (auth-exempt)."""
            return self._get_health_response()

        @app.get("/health/live")
        async def liveness() -> dict[str, str]:
            """Kubernetes liveness probe (auth-exempt)."""
            return {"status": "alive"}

        @app.get("/health/ready")
        async def readiness() -> dict[str, Any]:
            """Kubernetes readiness probe (auth-exempt)."""
            if self.status == HealthStatus.HEALTHY:
                return {"status": "ready", "model_loaded": True}
            raise HTTPException(status_code=503, detail="Service not ready")

        @app.get("/metrics")
        async def metrics(request: Request) -> MetricsResponse:
            """Prometheus-style metrics endpoint."""
            self._authenticate(request)
            self._check_rate_limit(request)
            return self._get_metrics_response()

        @app.get("/")
        async def root() -> dict[str, Any]:
            """Root endpoint with server info."""
            return {
                "service": "TorchBridge LLM Inference Server",
                "model_name": self.config.model_name,
                "model_version": self.config.model_version,
                "status": self.status.value,
                "features": {
                    "streaming": self.config.enable_streaming,
                    "dynamic_batching": self.config.enable_dynamic_batching,
                    "max_batch_size": self.config.max_batch_size,
                },
                "endpoints": [
                    "POST /generate",
                    "POST /chat",
                    "POST /tokenize",
                    "GET /health",
                    "GET /metrics",
                ],
            }

    def _format_chat_prompt(self, messages: list[ChatMessage]) -> str:
        """Format chat messages into a prompt."""
        # Simple formatting - can be customized per model
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n\n"

        formatted += "Assistant: "
        return formatted

    async def _generate_non_stream(self, request: GenerateRequest) -> GenerateResponse:
        """Handle non-streaming generation request."""
        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = inputs.input_ids

            # Prepare generation kwargs
            gen_kwargs = self._prepare_generation_kwargs(request)

            if self.config.enable_dynamic_batching:
                # Use dynamic batching
                result_queue = queue.Queue()
                batch_item = BatchItem(
                    request_id=str(time.time()),
                    prompt=request.prompt,
                    input_ids=input_ids,
                    generation_kwargs=gen_kwargs,
                    result_queue=result_queue,
                )

                with self._batch_lock:
                    self._batch_queue.append(batch_item)

                # Wait for result
                result = result_queue.get(timeout=30.0)

                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])

                return GenerateResponse(
                    generated_text=result['generated_text'],
                    prompt=request.prompt,
                    num_tokens=result['num_tokens'],
                    inference_time_ms=result['inference_time_ms'],
                    model_name=self.config.model_name,
                )
            else:
                # Direct generation (no batching)
                input_ids = input_ids.to(self.device)
                prompt_tokens = input_ids.size(1)

                # Use GenerationTimer for LLM metrics
                timer = None
                if self._llm_metrics is not None:
                    from torchbridge.monitoring.llm_metrics import GenerationTimer
                    timer = GenerationTimer(prompt_tokens=prompt_tokens)

                if timer is not None:
                    with timer:
                        with torch.no_grad():
                            outputs = self.model.generate(
                                input_ids,
                                **gen_kwargs
                            )
                else:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            **gen_kwargs
                        )

                # Decode output
                generated_ids = outputs[:, input_ids.size(1):]
                generated_text = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                num_generated = generated_ids.size(1)

                # Calculate timing
                inference_time = (time.time() - start_time) * 1000

                # Update metrics
                with self._lock:
                    self._generation_count += 1
                    self._total_generation_time += inference_time
                    self._last_generation_time = inference_time
                    self._total_tokens_generated += num_generated

                # Record LLM metrics
                if timer is not None and self._llm_metrics is not None:
                    req_metrics = timer.finalize(num_generated)
                    self._llm_metrics.record_request(
                        req_metrics,
                        model_name=self.config.model_name,
                    )

                return GenerateResponse(
                    generated_text=generated_text,
                    prompt=request.prompt,
                    num_tokens=num_generated,
                    inference_time_ms=inference_time,
                    model_name=self.config.model_name,
                )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def _generate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Handle streaming generation request using SSE."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = inputs.input_ids.to(self.device)

            # Prepare generation kwargs
            gen_kwargs = self._prepare_generation_kwargs(request)
            gen_kwargs['max_new_tokens'] = min(
                gen_kwargs.get('max_new_tokens', 256), 512
            )  # Limit for streaming

            # Stream generation token by token
            generated_tokens = []

            with torch.no_grad():
                for _ in range(gen_kwargs['max_new_tokens']):
                    outputs = self.model(input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                    # Get next token
                    next_token_logits = logits[:, -1, :]

                    # Apply temperature
                    if gen_kwargs.get('temperature', 1.0) != 1.0:
                        next_token_logits = next_token_logits / gen_kwargs['temperature']

                    # Apply top-k
                    if gen_kwargs.get('top_k', 0) > 0:
                        top_k = gen_kwargs['top_k']
                        indices_to_remove = next_token_logits < torch.topk(
                            next_token_logits, top_k
                        )[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Sample
                    if gen_kwargs.get('do_sample', True):
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    # Append to input for next iteration
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    generated_tokens.append(next_token.item())

                    # Decode and stream
                    if len(generated_tokens) % self.config.stream_interval_tokens == 0:
                        token_text = self.tokenizer.decode(
                            generated_tokens[-self.config.stream_interval_tokens:],
                            skip_special_tokens=True
                        )
                        yield f"data: {token_text}\n\n"
                        await asyncio.sleep(0)  # Allow other tasks to run

                    # Check for stop sequences
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            # Final flush
            if len(generated_tokens) % self.config.stream_interval_tokens != 0:
                token_text = self.tokenizer.decode(
                    generated_tokens[-(len(generated_tokens) % self.config.stream_interval_tokens):],
                    skip_special_tokens=True
                )
                yield f"data: {token_text}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    def _prepare_generation_kwargs(self, request: GenerateRequest) -> dict[str, Any]:
        """Prepare generation kwargs from request."""
        kwargs = {
            'max_new_tokens': request.max_new_tokens or self.config.max_new_tokens,
            'temperature': request.temperature or self.config.temperature,
            'top_p': request.top_p or self.config.top_p,
            'top_k': request.top_k or self.config.top_k,
            'repetition_penalty': request.repetition_penalty or self.config.repetition_penalty,
            'do_sample': request.do_sample if request.do_sample is not None else True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        # Merge speculative decoding kwargs
        if self._speculation_engine is not None:
            kwargs.update(self._speculation_engine.get_generation_kwargs())

        return kwargs

    def _count_tokens(self, text: str) -> TokenCountResponse:
        """Count tokens in text."""
        try:
            tokens = self.tokenizer.encode(text)
            token_texts = None

            # Try to get token strings
            try:
                token_texts = [
                    self.tokenizer.decode([t]) for t in tokens
                ]
            except Exception:
                logger.debug("Token text decoding failed", exc_info=True)
                pass

            return TokenCountResponse(
                text=text,
                num_tokens=len(tokens),
                tokens=token_texts,
            )
        except Exception as e:
            logger.error(f"Token counting error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    def _get_health_response(self) -> HealthResponse:
        """Get health check response."""
        uptime = time.time() - self.start_time
        avg_latency = (
            self._total_generation_time / self._generation_count
            if self._generation_count > 0
            else 0.0
        )

        return HealthResponse(
            status=self.status.value,
            model_name=self.config.model_name,
            model_loaded=self.model is not None,
            device=str(self.device),
            uptime_seconds=uptime,
            generation_count=self._generation_count,
            average_latency_ms=avg_latency,
        )

    def _get_metrics_response(self) -> MetricsResponse:
        """Get metrics response."""
        avg_time = (
            self._total_generation_time / self._generation_count
            if self._generation_count > 0
            else 0.0
        )

        avg_tokens_per_sec = (
            self._total_tokens_generated / (self._total_generation_time / 1000.0)
            if self._total_generation_time > 0
            else 0.0
        )

        # Get memory stats
        if self.device and self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0

        # Compute avg_batch_size from tracked counters (real value, not always 0)
        avg_batch_size = (
            self._total_batch_requests / self._total_batches_processed
            if self._total_batches_processed > 0 else 0.0
        )

        # LLM metrics snapshot
        llm_kwargs: dict = {}
        if self._llm_metrics is not None:
            snap = self._llm_metrics.get_snapshot()
            llm_kwargs = {
                "ttft_p50_ms": snap.ttft_p50_ms,
                "ttft_p95_ms": snap.ttft_p95_ms,
                "ttft_p99_ms": snap.ttft_p99_ms,
                "tpot_p50_ms": snap.tpot_p50_ms,
                "tpot_p95_ms": snap.tpot_p95_ms,
                "tpot_p99_ms": snap.tpot_p99_ms,
                "cache_hit_rate": snap.cache_hit_rate,
                "tokens_per_second": snap.tokens_per_second,
                "avg_batch_size": avg_batch_size,
            }

        return MetricsResponse(
            generation_count=self._generation_count,
            total_generation_time_ms=self._total_generation_time,
            average_generation_time_ms=avg_time,
            last_generation_time_ms=self._last_generation_time,
            total_tokens_generated=self._total_tokens_generated,
            average_tokens_per_second=avg_tokens_per_sec,
            model_name=self.config.model_name,
            device=str(self.device),
            memory_allocated_mb=memory_allocated,
            memory_reserved_mb=memory_reserved,
            **llm_kwargs,
        )

def create_llm_server(
    model: nn.Module,
    tokenizer: TokenizerType,
    model_name: str = "Qwen/Qwen3-0.6B",
    model_version: str = "1.0",
    enable_streaming: bool = True,
    enable_dynamic_batching: bool = True,
    max_batch_size: int = 8,
) -> LLMInferenceServer:
    """
    Create an LLM inference server.

    Args:
        model: PyTorch LLM model (optimized with LLMOptimizer)
        tokenizer: HuggingFace tokenizer
        model_name: Name of the model
        model_version: Version of the model
        enable_streaming: Enable streaming responses
        enable_dynamic_batching: Enable dynamic batching
        max_batch_size: Maximum batch size

    Returns:
        Configured LLMInferenceServer instance

    Example:
        >>> from torchbridge.models.llm import LLMOptimizer
        >>> optimizer = LLMOptimizer()
        >>> model, tokenizer = optimizer.optimize("Qwen/Qwen3-0.6B")
        >>> server = create_llm_server(model, tokenizer, model_name="Qwen/Qwen3-0.6B")
    """
    config = LLMServerConfig(
        model_name=model_name,
        model_version=model_version,
        enable_streaming=enable_streaming,
        enable_dynamic_batching=enable_dynamic_batching,
        max_batch_size=max_batch_size,
    )

    return LLMInferenceServer(model=model, tokenizer=tokenizer, config=config)

def run_llm_server(
    server: LLMInferenceServer,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """
    Run the LLM inference server.

    Args:
        server: LLMInferenceServer instance
        host: Host to bind to
        port: Port to bind to
        workers: Number of workers (should be 1 for GPU models)
        log_level: Logging level

    Example:
        >>> from torchbridge.deployment.serving import create_llm_server, run_llm_server
        >>> server = create_llm_server(model, tokenizer)
        >>> run_llm_server(server, host="0.0.0.0", port=8000)
    """
    if not UVICORN_AVAILABLE:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install uvicorn"
        )

    logger.info(f"Starting LLM server at {host}:{port}")

    uvicorn.run(
        server.app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
    )

__all__ = [
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
