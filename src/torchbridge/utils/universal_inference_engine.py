"""
Universal Inference Engine

High-performance inference engine supporting multiple hardware vendors
with intelligent request routing and real-time optimization.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..hardware.abstraction.hal_core import (
    DeviceSpec,
    HardwareAbstractionLayer,
    HardwareVendor,
)

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class InferenceMode(Enum):
    """Inference execution modes"""
    EAGER = "eager"
    COMPILED = "compiled"
    QUANTIZED = "quantized"
    SPECULATIVE = "speculative"


@dataclass
class InferenceRequest:
    """Request for model inference"""
    request_id: str
    model_id: str
    inputs: dict[str, torch.Tensor]
    priority: RequestPriority = RequestPriority.NORMAL
    max_latency_ms: float | None = None
    preferred_vendors: list[HardwareVendor] | None = None
    precision_requirements: list[str] | None = None
    batch_compatible: bool = True
    callback_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """Response from model inference"""
    request_id: str
    outputs: dict[str, torch.Tensor]
    latency_ms: float
    device_used: DeviceSpec
    model_version: str
    confidence_scores: dict[str, float] | None = None
    execution_stats: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: str | None = None


@dataclass
class RequestProfile:
    """Profile characteristics of inference request"""
    estimated_memory_gb: float
    estimated_compute_tflops: float
    expected_latency_ms: float
    input_shapes: list[tuple[int, ...]]
    batch_size: int
    sequence_length: int
    complexity_score: float
    can_batch: bool = True
    preferred_precision: str = "fp16"


@dataclass
class ModelVariant:
    """Model variant optimized for specific hardware"""
    model_id: str
    variant_id: str
    model: nn.Module
    target_hardware: DeviceSpec
    optimization_level: str
    precision: str
    expected_latency_ms: float
    memory_requirement_gb: float
    throughput_rps: float
    accuracy_metrics: dict[str, float] = field(default_factory=dict)


class ModelRegistry:
    """Registry for managing model variants across hardware"""

    def __init__(self):
        self.models: dict[str, dict[str, ModelVariant]] = {}  # model_id -> variant_id -> variant
        self.hardware_compatibility: dict[str, list[HardwareVendor]] = {}
        self._lock = threading.RLock()

    def register_model_variant(self, variant: ModelVariant) -> None:
        """Register model variant for specific hardware"""
        with self._lock:
            if variant.model_id not in self.models:
                self.models[variant.model_id] = {}
                self.hardware_compatibility[variant.model_id] = []

            self.models[variant.model_id][variant.variant_id] = variant

            # Update hardware compatibility
            if variant.target_hardware.vendor not in self.hardware_compatibility[variant.model_id]:
                self.hardware_compatibility[variant.model_id].append(variant.target_hardware.vendor)

        logger.info(f"Registered model variant {variant.variant_id} for {variant.model_id}")

    def get_optimal_variant(self,
                           model_id: str,
                           device: DeviceSpec,
                           latency_requirement: float | None = None,
                           memory_limit: float | None = None) -> ModelVariant | None:
        """Get optimal model variant for device and requirements"""
        if model_id not in self.models:
            return None

        candidates = []
        for _variant_id, variant in self.models[model_id].items():
            # Check hardware compatibility
            if variant.target_hardware.vendor != device.vendor:
                continue

            # Check latency requirement
            if latency_requirement and variant.expected_latency_ms > latency_requirement:
                continue

            # Check memory requirement
            if memory_limit and variant.memory_requirement_gb > memory_limit:
                continue

            # Calculate suitability score
            score = self._calculate_variant_score(variant, device, latency_requirement)
            candidates.append((variant, score))

        if not candidates:
            return None

        # Return variant with highest score
        return max(candidates, key=lambda x: x[1])[0]

    def _calculate_variant_score(self,
                                variant: ModelVariant,
                                device: DeviceSpec,
                                latency_requirement: float | None) -> float:
        """Calculate variant suitability score"""
        score = 0.0

        # Latency score
        if latency_requirement:
            latency_ratio = variant.expected_latency_ms / latency_requirement
            score += max(0.0, 1.0 - latency_ratio) * 0.4
        else:
            # Prefer faster variants
            score += (1000.0 / variant.expected_latency_ms) * 0.1

        # Throughput score
        score += variant.throughput_rps / 1000.0 * 0.3

        # Memory efficiency score
        memory_efficiency = 1.0 - (variant.memory_requirement_gb / device.capabilities.memory_gb)
        score += max(0.0, memory_efficiency) * 0.2

        # Accuracy score (if available)
        if "accuracy" in variant.accuracy_metrics:
            score += variant.accuracy_metrics["accuracy"] * 0.1

        return score

    def get_available_models(self) -> list[str]:
        """Get list of available model IDs"""
        return list(self.models.keys())

    def get_model_variants(self, model_id: str) -> list[ModelVariant]:
        """Get all variants for a model"""
        if model_id not in self.models:
            return []
        return list(self.models[model_id].values())


class HardwarePool:
    """Pool of available hardware devices"""

    def __init__(self, hal: HardwareAbstractionLayer):
        self.hal = hal
        self.available_devices: dict[int, DeviceSpec] = {}
        self.device_queues: dict[int, asyncio.Queue] = {}
        self.device_metrics: dict[int, dict[str, float]] = {}
        self._lock = threading.RLock()

    async def initialize(self) -> None:
        """Initialize hardware pool"""
        # Discover available devices
        hardware_inventory = self.hal.discover_all_hardware()
        device_counter = 0

        with self._lock:
            for _vendor, devices in hardware_inventory.items():
                for device in devices:
                    device.device_id = device_counter
                    self.available_devices[device_counter] = device
                    self.device_queues[device_counter] = asyncio.Queue()
                    self.device_metrics[device_counter] = {}
                    device_counter += 1

        logger.info(f"Initialized hardware pool with {len(self.available_devices)} devices")

    def get_available_devices(self) -> list[DeviceSpec]:
        """Get list of available devices"""
        return [device for device in self.available_devices.values() if device.is_available]

    def get_device_by_id(self, device_id: int) -> DeviceSpec | None:
        """Get device by ID"""
        return self.available_devices.get(device_id)

    async def allocate_device(self, requirements: RequestProfile) -> DeviceSpec | None:
        """Allocate optimal device for request"""
        optimal_device = self.hal.get_optimal_device(
            memory_requirement_gb=requirements.estimated_memory_gb,
            compute_requirement_tflops=requirements.estimated_compute_tflops,
            precision_requirements=None
        )

        if optimal_device and optimal_device.is_available:
            # Mark device as busy
            optimal_device.current_utilization += 0.1  # Approximate utilization increase
            return optimal_device

        return None

    async def release_device(self, device: DeviceSpec) -> None:
        """Release device back to pool"""
        device.current_utilization = max(0.0, device.current_utilization - 0.1)

    def update_device_metrics(self, device_id: int, metrics: dict[str, float]) -> None:
        """Update real-time device metrics"""
        if device_id in self.device_metrics:
            self.device_metrics[device_id].update(metrics)


class RequestProfiler:
    """Profiler for analyzing inference request characteristics"""

    def __init__(self):
        self.profile_cache: dict[str, RequestProfile] = {}
        self.model_stats: dict[str, dict[str, float]] = {}

    def profile_request(self, request: InferenceRequest) -> RequestProfile:
        """Profile inference request to estimate resource requirements"""
        cache_key = self._get_cache_key(request)
        if cache_key in self.profile_cache:
            return self.profile_cache[cache_key]

        # Calculate input characteristics
        total_elements = 0
        max_sequence_length = 0
        input_shapes = []

        for _name, tensor in request.inputs.items():
            elements = tensor.numel()
            total_elements += elements
            input_shapes.append(tensor.shape)

            # Estimate sequence length for language models
            if len(tensor.shape) >= 2:
                max_sequence_length = max(max_sequence_length, tensor.shape[-1])

        # Estimate memory requirements
        estimated_memory_gb = self._estimate_memory_usage(request.model_id, input_shapes)

        # Estimate compute requirements
        estimated_compute_tflops = self._estimate_compute_requirements(request.model_id, input_shapes)

        # Estimate latency
        expected_latency_ms = self._estimate_latency(request.model_id, input_shapes)

        # Calculate complexity score
        complexity_score = np.log10(total_elements + 1) * max_sequence_length

        batch_size = input_shapes[0][0] if input_shapes else 1

        profile = RequestProfile(
            estimated_memory_gb=estimated_memory_gb,
            estimated_compute_tflops=estimated_compute_tflops,
            expected_latency_ms=expected_latency_ms,
            input_shapes=input_shapes,
            batch_size=batch_size,
            sequence_length=max_sequence_length,
            complexity_score=complexity_score,
            can_batch=request.batch_compatible,
            preferred_precision=request.precision_requirements[0] if request.precision_requirements else "fp16"
        )

        # Cache profile
        self.profile_cache[cache_key] = profile
        return profile

    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request profile"""
        shapes_str = ",".join([str(tensor.shape) for tensor in request.inputs.values()])
        return f"{request.model_id}:{shapes_str}"

    def _estimate_memory_usage(self, model_id: str, input_shapes: list[tuple[int, ...]]) -> float:
        """Estimate memory usage in GB"""
        # Simple estimation based on model size and input size
        if model_id in self.model_stats and 'memory_gb' in self.model_stats[model_id]:
            base_memory = self.model_stats[model_id]['memory_gb']
        else:
            # Default estimation based on model name patterns
            base_memory = self._estimate_model_memory(model_id)

        # Add input and activation memory
        total_input_elements = sum(np.prod(shape) for shape in input_shapes)
        input_memory_gb = total_input_elements * 4 / (1024**3)  # 4 bytes per float32

        # Estimate activation memory (typically 2-4x input size)
        activation_memory_gb = input_memory_gb * 3

        return base_memory + input_memory_gb + activation_memory_gb

    def _estimate_model_memory(self, model_id: str) -> float:
        """Estimate base model memory from model ID"""
        # Simple heuristics based on model naming
        model_lower = model_id.lower()

        if "7b" in model_lower or "7-billion" in model_lower:
            return 14.0  # ~14GB for 7B parameter model
        elif "13b" in model_lower or "13-billion" in model_lower:
            return 26.0  # ~26GB for 13B parameter model
        elif "70b" in model_lower or "70-billion" in model_lower:
            return 140.0  # ~140GB for 70B parameter model
        elif "small" in model_lower or "base" in model_lower:
            return 1.0   # ~1GB for small models
        elif "large" in model_lower:
            return 5.0   # ~5GB for large models
        else:
            return 2.0   # Default 2GB

    def _estimate_compute_requirements(self, model_id: str, input_shapes: list[tuple[int, ...]]) -> float:
        """Estimate compute requirements in TFLOPS"""
        # Estimate based on model size and sequence length
        if input_shapes:
            sequence_length = input_shapes[0][-1] if len(input_shapes[0]) > 1 else 1
            batch_size = input_shapes[0][0]
        else:
            sequence_length = 512  # Default
            batch_size = 1

        # Rough FLOP estimation for transformer models
        if "7b" in model_id.lower():
            model_flops_per_token = 14e9  # ~14 GFLOPs per token for 7B model
        elif "13b" in model_id.lower():
            model_flops_per_token = 26e9
        elif "70b" in model_id.lower():
            model_flops_per_token = 140e9
        else:
            model_flops_per_token = 2e9  # Default

        total_flops = model_flops_per_token * sequence_length * batch_size
        return total_flops / 1e12  # Convert to TFLOPS

    def _estimate_latency(self, model_id: str, input_shapes: list[tuple[int, ...]]) -> float:
        """Estimate latency in milliseconds"""
        # Base latency from model stats or defaults
        if model_id in self.model_stats and 'latency_ms' in self.model_stats[model_id]:
            base_latency = self.model_stats[model_id]['latency_ms']
        else:
            base_latency = self._estimate_base_latency(model_id)

        # Scale by sequence length
        if input_shapes:
            sequence_length = input_shapes[0][-1] if len(input_shapes[0]) > 1 else 1
            batch_size = input_shapes[0][0]
        else:
            sequence_length = 1
            batch_size = 1

        # Latency scales roughly linearly with sequence length and batch size
        scaling_factor = np.sqrt(sequence_length * batch_size / 512.0)  # Normalized to 512 tokens
        return base_latency * scaling_factor

    def _estimate_base_latency(self, model_id: str) -> float:
        """Estimate base latency from model ID"""
        model_lower = model_id.lower()

        if "7b" in model_lower:
            return 50.0   # ~50ms for 7B model
        elif "13b" in model_lower:
            return 100.0  # ~100ms for 13B model
        elif "70b" in model_lower:
            return 500.0  # ~500ms for 70B model
        elif "small" in model_lower:
            return 10.0   # ~10ms for small model
        else:
            return 30.0   # Default 30ms


class UniversalInferenceEngine:
    """
    Universal inference engine supporting multiple hardware vendors

    Features:
    - Intelligent request routing based on hardware capabilities
    - Real-time performance optimization
    - Automatic model variant selection
    - Dynamic load balancing
    """

    def __init__(self,
                 model_registry: ModelRegistry,
                 hardware_pool: HardwarePool,
                 max_concurrent_requests: int = 1000):
        self.model_registry = model_registry
        self.hardware_pool = hardware_pool
        self.max_concurrent_requests = max_concurrent_requests

        # Request management
        self.active_requests: dict[str, InferenceRequest] = {}
        self.request_queue = asyncio.Queue(maxsize=max_concurrent_requests)
        self.profiler = RequestProfiler()

        # Performance tracking
        self.request_metrics: dict[str, list[float]] = {
            'latency_ms': deque(maxlen=1000),
            'throughput_rps': deque(maxlen=100),
            'queue_size': deque(maxlen=1000)
        }

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the inference engine"""
        await self.hardware_pool.initialize()

        # Start background workers
        for i in range(4):  # 4 worker tasks
            task = asyncio.create_task(self._request_worker(i))
            self._background_tasks.append(task)

        # Start metrics collector
        metrics_task = asyncio.create_task(self._metrics_collector())
        self._background_tasks.append(metrics_task)

        logger.info("Universal inference engine initialized")

    async def serve_request(self, request: InferenceRequest) -> InferenceResponse:
        """Serve inference request"""
        start_time = time.time()

        try:
            # Validate request
            if request.model_id not in self.model_registry.get_available_models():
                return InferenceResponse(
                    request_id=request.request_id,
                    outputs={},
                    latency_ms=0.0,
                    device_used=None,
                    model_version="unknown",
                    success=False,
                    error_message=f"Model {request.model_id} not available"
                )

            # Profile request
            profile = self.profiler.profile_request(request)

            # Allocate device
            device = await self.hardware_pool.allocate_device(profile)
            if not device:
                return InferenceResponse(
                    request_id=request.request_id,
                    outputs={},
                    latency_ms=0.0,
                    device_used=None,
                    model_version="unknown",
                    success=False,
                    error_message="No available devices"
                )

            # Get optimal model variant
            available_memory = device.capabilities.memory_gb - device.current_memory_usage_gb
            model_variant = self.model_registry.get_optimal_variant(
                model_id=request.model_id,
                device=device,
                latency_requirement=request.max_latency_ms,
                memory_limit=available_memory
            )

            if not model_variant:
                await self.hardware_pool.release_device(device)
                return InferenceResponse(
                    request_id=request.request_id,
                    outputs={},
                    latency_ms=0.0,
                    device_used=device,
                    model_version="unknown",
                    success=False,
                    error_message="No suitable model variant found"
                )

            # Execute inference
            inference_start = time.time()
            outputs = await self._execute_inference(model_variant, request.inputs, device)
            inference_time = (time.time() - inference_start) * 1000

            # Release device
            await self.hardware_pool.release_device(device)

            # Create response
            total_latency = (time.time() - start_time) * 1000
            response = InferenceResponse(
                request_id=request.request_id,
                outputs=outputs,
                latency_ms=total_latency,
                device_used=device,
                model_version=model_variant.variant_id,
                execution_stats={
                    'inference_time_ms': inference_time,
                    'queue_time_ms': total_latency - inference_time,
                    'memory_used_gb': model_variant.memory_requirement_gb
                }
            )

            # Record metrics
            self.request_metrics['latency_ms'].append(total_latency)

            return response

        except Exception as e:
            logger.error(f"Error serving request {request.request_id}: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                outputs={},
                latency_ms=(time.time() - start_time) * 1000,
                device_used=None,
                model_version="unknown",
                success=False,
                error_message=str(e)
            )

    async def _execute_inference(self,
                                model_variant: ModelVariant,
                                inputs: dict[str, torch.Tensor],
                                device: DeviceSpec) -> dict[str, torch.Tensor]:
        """Execute inference on specific device"""
        # Move model and inputs to device
        device_str = f"cuda:{device.device_id}"

        # Move inputs to device
        device_inputs = {}
        for key, tensor in inputs.items():
            device_inputs[key] = tensor.to(device_str)

        # Execute model
        model_variant.model.eval()
        with torch.no_grad():
            if hasattr(model_variant.model, 'forward'):
                if len(device_inputs) == 1:
                    # Single input
                    input_tensor = list(device_inputs.values())[0]
                    outputs = model_variant.model(input_tensor)
                else:
                    # Multiple inputs
                    outputs = model_variant.model(**device_inputs)
            else:
                raise ValueError("Model does not have forward method")

        # Handle different output types
        if isinstance(outputs, torch.Tensor):
            return {"output": outputs.cpu()}
        elif isinstance(outputs, (list, tuple)):
            return {f"output_{i}": out.cpu() for i, out in enumerate(outputs)}
        elif isinstance(outputs, dict):
            return {key: out.cpu() if isinstance(out, torch.Tensor) else out
                   for key, out in outputs.items()}
        else:
            return {"output": outputs}

    async def _request_worker(self, worker_id: int) -> None:
        """Background worker for processing requests"""
        logger.info(f"Started request worker {worker_id}")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get request from queue with timeout
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )

                    # Process request
                    await self.serve_request(request)

                    # Mark task as done
                    self.request_queue.task_done()

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")

        except asyncio.CancelledError:
            logger.info(f"Request worker {worker_id} cancelled")

    async def _metrics_collector(self) -> None:
        """Background metrics collection"""
        logger.info("Started metrics collector")

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(10.0)  # Collect every 10 seconds

                # Calculate current metrics
                queue_size = self.request_queue.qsize()
                self.request_metrics['queue_size'].append(queue_size)

                # Calculate throughput (requests per second)
                time.time()
                recent_latencies = [lat for lat in self.request_metrics['latency_ms']
                                  if lat > 0]  # Filter out invalid latencies

                if recent_latencies:
                    avg_latency = np.mean(recent_latencies[-100:])  # Last 100 requests
                    throughput = min(1000.0 / avg_latency, 1000.0)  # Cap at 1000 RPS
                    self.request_metrics['throughput_rps'].append(throughput)

        except asyncio.CancelledError:
            logger.info("Metrics collector cancelled")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics"""
        metrics = {}

        # Latency metrics
        if self.request_metrics['latency_ms']:
            latencies = list(self.request_metrics['latency_ms'])
            metrics['latency'] = {
                'avg_ms': np.mean(latencies),
                'p50_ms': np.percentile(latencies, 50),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'max_ms': np.max(latencies)
            }

        # Throughput metrics
        if self.request_metrics['throughput_rps']:
            throughputs = list(self.request_metrics['throughput_rps'])
            metrics['throughput'] = {
                'current_rps': throughputs[-1] if throughputs else 0.0,
                'avg_rps': np.mean(throughputs),
                'max_rps': np.max(throughputs) if throughputs else 0.0
            }

        # Queue metrics
        if self.request_metrics['queue_size']:
            queue_sizes = list(self.request_metrics['queue_size'])
            metrics['queue'] = {
                'current_size': queue_sizes[-1] if queue_sizes else 0,
                'avg_size': np.mean(queue_sizes),
                'max_size': np.max(queue_sizes) if queue_sizes else 0
            }

        # Hardware metrics
        available_devices = self.hardware_pool.get_available_devices()
        metrics['hardware'] = {
            'total_devices': len(self.hardware_pool.available_devices),
            'available_devices': len(available_devices),
            'avg_utilization': np.mean([d.current_utilization for d in available_devices]) if available_devices else 0.0
        }

        return metrics

    async def shutdown(self) -> None:
        """Shutdown the inference engine"""
        logger.info("Shutting down inference engine")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("Inference engine shutdown completed")
