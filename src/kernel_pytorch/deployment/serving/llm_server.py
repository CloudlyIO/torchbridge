"""
LLM-Specific FastAPI Inference Server for KernelPyTorch

This module provides a production-grade REST API server for serving LLMs
optimized with KernelPyTorch's LLMOptimizer. Includes streaming, dynamic
batching, and specialized endpoints for text generation.

Features:
- Text generation endpoint with streaming support (SSE)
- Chat completion endpoint
- Dynamic batching for efficient throughput
- Token counting utilities
- Health checks and metrics
- Integration with LLMOptimizer for quantization

Supported Models:
- GPT-2 (all variants)
- LLaMA (2, 3)
- Mistral
- Phi-2, Phi-3

Example:
    ```python
    from kernel_pytorch.deployment.serving import create_llm_server, run_llm_server
    from kernel_pytorch.models.llm import LLMOptimizer, LLMConfig

    # Create optimizer and load model
    config = LLMConfig(model_name="gpt2", quantization="int8")
    optimizer = LLMOptimizer(config)
    model, tokenizer = optimizer.optimize("gpt2")

    # Create and run server
    server = create_llm_server(model, tokenizer, model_name="gpt2")
    run_llm_server(server, host="0.0.0.0", port=8000)
    ```

Version: 0.4.22
"""

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Union

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
    FastAPI = None

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
    model_name: str = "gpt2"
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

    # Performance settings
    enable_streaming: bool = True
    stream_interval_tokens: int = 1

    # Health check settings
    enable_health_checks: bool = True
    enable_metrics: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Pydantic models for request/response (only if FastAPI available)
if FASTAPI_AVAILABLE:
    class GenerateRequest(BaseModel):
        """Request model for text generation."""

        prompt: str = Field(..., description="Input text prompt")
        max_new_tokens: int | None = Field(None, description="Maximum tokens to generate")
        temperature: float | None = Field(None, description="Sampling temperature (0.0-2.0)")
        top_p: float | None = Field(None, description="Nucleus sampling probability")
        top_k: int | None = Field(None, description="Top-k sampling")
        repetition_penalty: float | None = Field(None, description="Repetition penalty")
        do_sample: bool | None = Field(True, description="Enable sampling")
        stream: bool | None = Field(False, description="Enable streaming response")
        stop_sequences: list[str] | None = Field(None, description="Stop generation sequences")

    class ChatMessage(BaseModel):
        """Chat message format."""

        role: str = Field(..., description="Message role (system, user, assistant)")
        content: str = Field(..., description="Message content")

    class ChatCompletionRequest(BaseModel):
        """Request model for chat completion."""

        messages: list[ChatMessage] = Field(..., description="List of chat messages")
        max_new_tokens: int | None = Field(None, description="Maximum tokens to generate")
        temperature: float | None = Field(None, description="Sampling temperature")
        top_p: float | None = Field(None, description="Nucleus sampling")
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
else:
    # Stub classes when FastAPI not available
    GenerateRequest = dict[str, Any]
    ChatMessage = dict[str, Any]
    ChatCompletionRequest = dict[str, Any]
    GenerateResponse = dict[str, Any]
    TokenCountRequest = dict[str, Any]
    TokenCountResponse = dict[str, Any]
    HealthResponse = dict[str, Any]
    MetricsResponse = dict[str, Any]


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
        self._lock = threading.Lock()

        # Dynamic batching
        self._batch_queue: deque = deque()
        self._batch_lock = threading.Lock()
        self._batch_thread: threading.Thread | None = None
        self._stop_batching = threading.Event()

        # Initialize device and model
        self._setup_device()
        self._setup_model()

        # Start batching thread if enabled
        if self.config.enable_dynamic_batching:
            self._start_batch_processor()

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
        # Model should already be on correct device from LLMOptimizer
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.status = HealthStatus.HEALTHY
        logger.info("LLM server setup complete")

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
        """Process a batch of generation requests."""
        try:
            # Prepare batch inputs
            input_ids_list = [item.input_ids for item in batch_items]

            # Pad to same length
            max_len = max(ids.size(1) for ids in input_ids_list)
            padded_inputs = []
            attention_masks = []

            for ids in input_ids_list:
                pad_len = max_len - ids.size(1)
                if pad_len > 0:
                    padded = torch.nn.functional.pad(
                        ids, (0, pad_len), value=self.tokenizer.pad_token_id
                    )
                else:
                    padded = ids
                padded_inputs.append(padded)
                attention_masks.append((padded != self.tokenizer.pad_token_id).long())

            batch_input_ids = torch.cat(padded_inputs, dim=0).to(self.device)
            batch_attention_mask = torch.cat(attention_masks, dim=0).to(self.device)

            # Use first item's kwargs as template (merge common settings)
            gen_kwargs = batch_items[0].generation_kwargs.copy()
            gen_kwargs['attention_mask'] = batch_attention_mask

            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    batch_input_ids,
                    **gen_kwargs
                )

            generation_time = (time.time() - start_time) * 1000

            # Distribute results
            for i, item in enumerate(batch_items):
                output_ids = outputs[i:i+1]
                generated_ids = output_ids[:, input_ids_list[i].size(1):]
                generated_text = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )

                result = {
                    'generated_text': generated_text,
                    'num_tokens': generated_ids.size(1),
                    'inference_time_ms': generation_time / len(batch_items),
                }

                item.result_queue.put(result)

            # Update metrics
            with self._lock:
                self._generation_count += len(batch_items)
                self._total_generation_time += generation_time
                self._last_generation_time = generation_time
                total_tokens = sum(
                    outputs[i, input_ids_list[i].size(1):].numel()
                    for i in range(len(batch_items))
                )
                self._total_tokens_generated += total_tokens

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Send error to all waiting requests
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
            title=f"KernelPyTorch LLM Server - {self.config.model_name}",
            description="Production LLM inference server with streaming and batching",
            version=self.config.model_version,
            lifespan=lifespan,
        )

        # Register routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: "FastAPI") -> None:
        """Register API routes."""

        @app.post("/generate")
        async def generate(request: GenerateRequest):
            """Text generation endpoint with optional streaming."""
            if request.stream and self.config.enable_streaming:
                return StreamingResponse(
                    self._generate_stream(request),
                    media_type="text/event-stream"
                )
            else:
                return await self._generate_non_stream(request)

        @app.post("/chat")
        async def chat_completion(request: ChatCompletionRequest):
            """Chat completion endpoint."""
            # Convert chat messages to prompt
            prompt = self._format_chat_prompt(request.messages)

            # Create generation request
            gen_request = GenerateRequest(
                prompt=prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
            )

            if request.stream and self.config.enable_streaming:
                return StreamingResponse(
                    self._generate_stream(gen_request),
                    media_type="text/event-stream"
                )
            else:
                return await self._generate_non_stream(gen_request)

        @app.post("/tokenize", response_model=TokenCountResponse)
        async def tokenize(request: TokenCountRequest) -> TokenCountResponse:
            """Token counting endpoint."""
            return self._count_tokens(request.text)

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
                "service": "KernelPyTorch LLM Inference Server",
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

                # Calculate timing
                inference_time = (time.time() - start_time) * 1000

                # Update metrics
                with self._lock:
                    self._generation_count += 1
                    self._total_generation_time += inference_time
                    self._last_generation_time = inference_time
                    self._total_tokens_generated += generated_ids.size(1)

                return GenerateResponse(
                    generated_text=generated_text,
                    prompt=request.prompt,
                    num_tokens=generated_ids.size(1),
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
        return {
            'max_new_tokens': request.max_new_tokens or self.config.max_new_tokens,
            'temperature': request.temperature or self.config.temperature,
            'top_p': request.top_p or self.config.top_p,
            'top_k': request.top_k or self.config.top_k,
            'repetition_penalty': request.repetition_penalty or self.config.repetition_penalty,
            'do_sample': request.do_sample if request.do_sample is not None else True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

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
        )


def create_llm_server(
    model: nn.Module,
    tokenizer: TokenizerType,
    model_name: str = "gpt2",
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
        >>> from kernel_pytorch.models.llm import LLMOptimizer
        >>> optimizer = LLMOptimizer()
        >>> model, tokenizer = optimizer.optimize("gpt2")
        >>> server = create_llm_server(model, tokenizer, model_name="gpt2")
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
        >>> from kernel_pytorch.deployment.serving import create_llm_server, run_llm_server
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
