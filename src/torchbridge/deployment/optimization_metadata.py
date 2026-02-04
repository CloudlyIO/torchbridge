"""
Optimization Metadata Schema for Model Export

This module provides metadata classes for preserving optimization information
when exporting models to ONNX, TorchScript, or other formats.

The metadata schema captures:
- Hardware-specific optimizations applied
- Precision configurations (FP8, FP16, BF16, etc.)
- Kernel fusion information
- Performance characteristics
- Deployment recommendations

Example:
    ```python
    from torchbridge.deployment import OptimizationMetadata, create_metadata

    # Create metadata from optimized model
    metadata = create_metadata(
        model=optimized_model,
        backend="nvidia",
        optimization_level="aggressive"
    )

    # Export with metadata
    metadata.save("model_metadata.json")
    ```
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TORCH_PACKAGE = "torch_package"
    SAFETENSORS = "safetensors"


@dataclass
class HardwareMetadata:
    """Hardware-specific optimization metadata."""
    backend: str = "auto"  # cuda, tpu, amd, cpu
    architecture: str = "auto"  # hopper, ampere, cdna3, v5e, etc.
    compute_capability: tuple[int, int] | None = None
    tensor_cores: bool = False
    fp8_support: bool = False
    memory_gb: float = 0.0

    # Optimization flags
    flash_attention_enabled: bool = False
    fused_kernels_enabled: bool = False
    custom_kernels_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert tuple to list for JSON serialization
        if result['compute_capability']:
            result['compute_capability'] = list(result['compute_capability'])
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'HardwareMetadata':
        """Create from dictionary."""
        if data.get('compute_capability'):
            data['compute_capability'] = tuple(data['compute_capability'])
        return cls(**data)


@dataclass
class PrecisionMetadata:
    """Precision configuration metadata."""
    default_dtype: str = "float32"  # float32, float16, bfloat16, fp8_e4m3
    mixed_precision: bool = False
    autocast_enabled: bool = False

    # FP8 specific
    fp8_enabled: bool = False
    fp8_format: str = "e4m3"
    fp8_layers: list[str] = field(default_factory=list)

    # Quantization
    quantized: bool = False
    quantization_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PrecisionMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FusionMetadata:
    """Kernel fusion information."""
    fused_operations: list[dict[str, Any]] = field(default_factory=list)
    attention_fusion: bool = False
    ffn_fusion: bool = False
    layernorm_fusion: bool = False

    # Fusion details
    fusion_count: int = 0
    estimated_speedup: float = 1.0

    def add_fusion(
        self,
        fusion_type: str,
        layers: list[str],
        estimated_speedup: float = 1.0
    ) -> None:
        """Add a fusion record."""
        self.fused_operations.append({
            'type': fusion_type,
            'layers': layers,
            'speedup': estimated_speedup
        })
        self.fusion_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'FusionMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PerformanceMetadata:
    """Performance characteristics metadata."""
    # Latency metrics (milliseconds)
    inference_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput metrics
    throughput_tokens_per_sec: float = 0.0
    throughput_samples_per_sec: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    model_size_mb: float = 0.0

    # Benchmark info
    benchmark_batch_size: int = 1
    benchmark_sequence_length: int = 512
    benchmark_iterations: int = 100
    benchmark_device: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PerformanceMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelMetadata:
    """Model architecture metadata."""
    model_type: str = "unknown"  # transformer, cnn, mlp, etc.
    num_parameters: int = 0
    num_layers: int = 0

    # Architecture details
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_hidden_layers: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0

    # Input/output shapes
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    output_shapes: dict[str, list[int]] = field(default_factory=dict)
    dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OptimizationMetadata:
    """
    Complete optimization metadata for exported models.

    This is the top-level metadata class that aggregates all optimization
    information for a model export.
    """
    # Version and identification
    schema_version: str = "1.0.0"
    torchbridge_version: str = "0.3.8"
    export_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    export_format: str = "onnx"

    # Optimization level
    optimization_level: str = "balanced"  # conservative, balanced, aggressive

    # Component metadata
    hardware: HardwareMetadata = field(default_factory=HardwareMetadata)
    precision: PrecisionMetadata = field(default_factory=PrecisionMetadata)
    fusion: FusionMetadata = field(default_factory=FusionMetadata)
    performance: PerformanceMetadata = field(default_factory=PerformanceMetadata)
    model: ModelMetadata = field(default_factory=ModelMetadata)

    # Deployment recommendations
    recommended_batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 16, 32])
    recommended_backends: list[str] = field(default_factory=lambda: ["cuda", "cpu"])
    minimum_compute_capability: tuple[int, int] | None = None

    # Custom metadata
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'schema_version': self.schema_version,
            'torchbridge_version': self.torchbridge_version,
            'export_timestamp': self.export_timestamp,
            'export_format': self.export_format,
            'optimization_level': self.optimization_level,
            'hardware': self.hardware.to_dict(),
            'precision': self.precision.to_dict(),
            'fusion': self.fusion.to_dict(),
            'performance': self.performance.to_dict(),
            'model': self.model.to_dict(),
            'recommended_batch_sizes': self.recommended_batch_sizes,
            'recommended_backends': self.recommended_backends,
            'minimum_compute_capability': list(self.minimum_compute_capability) if self.minimum_compute_capability else None,
            'custom': self.custom
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'OptimizationMetadata':
        """Create from dictionary."""
        # Parse nested dataclasses
        hardware = HardwareMetadata.from_dict(data.get('hardware', {}))
        precision = PrecisionMetadata.from_dict(data.get('precision', {}))
        fusion = FusionMetadata.from_dict(data.get('fusion', {}))
        performance = PerformanceMetadata.from_dict(data.get('performance', {}))
        model = ModelMetadata.from_dict(data.get('model', {}))

        min_cc = data.get('minimum_compute_capability')
        if min_cc:
            min_cc = tuple(min_cc)

        return cls(
            schema_version=data.get('schema_version', '1.0.0'),
            torchbridge_version=data.get('torchbridge_version', '0.3.8'),
            export_timestamp=data.get('export_timestamp', datetime.now().isoformat()),
            export_format=data.get('export_format', 'onnx'),
            optimization_level=data.get('optimization_level', 'balanced'),
            hardware=hardware,
            precision=precision,
            fusion=fusion,
            performance=performance,
            model=model,
            recommended_batch_sizes=data.get('recommended_batch_sizes', [1, 8, 16, 32]),
            recommended_backends=data.get('recommended_backends', ['cuda', 'cpu']),
            minimum_compute_capability=min_cc,
            custom=data.get('custom', {})
        )

    def save(self, path: str) -> None:
        """Save metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved optimization metadata to {path}")

    @classmethod
    def load(cls, path: str) -> 'OptimizationMetadata':
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded optimization metadata from {path}")
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Optimization Metadata Summary",
            "=" * 60,
            f"Export Format: {self.export_format}",
            f"Optimization Level: {self.optimization_level}",
            f"Timestamp: {self.export_timestamp}",
            "",
            "Hardware:",
            f"  Backend: {self.hardware.backend}",
            f"  Architecture: {self.hardware.architecture}",
            f"  Flash Attention: {self.hardware.flash_attention_enabled}",
            f"  Fused Kernels: {self.hardware.fused_kernels_enabled}",
            "",
            "Precision:",
            f"  Default dtype: {self.precision.default_dtype}",
            f"  Mixed Precision: {self.precision.mixed_precision}",
            f"  FP8 Enabled: {self.precision.fp8_enabled}",
            "",
            "Model:",
            f"  Type: {self.model.model_type}",
            f"  Parameters: {self.model.num_parameters:,}",
            f"  Size: {self.performance.model_size_mb:.2f} MB",
            "",
            "Performance:",
            f"  Latency (p50): {self.performance.p50_latency_ms:.2f} ms",
            f"  Throughput: {self.performance.throughput_samples_per_sec:.2f} samples/sec",
            f"  Peak Memory: {self.performance.peak_memory_mb:.2f} MB",
            "",
            "Fusion:",
            f"  Fusion Count: {self.fusion.fusion_count}",
            f"  Estimated Speedup: {self.fusion.estimated_speedup:.2f}x",
            "=" * 60
        ]
        return "\n".join(lines)


def create_metadata(
    model: nn.Module,
    backend: str = "auto",
    optimization_level: str = "balanced",
    export_format: str = "onnx",
    sample_input: torch.Tensor | None = None,
    benchmark: bool = False,
    benchmark_iterations: int = 100
) -> OptimizationMetadata:
    """
    Create optimization metadata from a model.

    Args:
        model: PyTorch model to analyze
        backend: Hardware backend (auto, cuda, tpu, amd, cpu)
        optimization_level: Optimization level applied
        export_format: Target export format
        sample_input: Sample input for shape inference and benchmarking
        benchmark: Whether to run performance benchmarks
        benchmark_iterations: Number of benchmark iterations

    Returns:
        OptimizationMetadata with all information captured
    """
    metadata = OptimizationMetadata(
        export_format=export_format,
        optimization_level=optimization_level
    )

    # Detect hardware
    _populate_hardware_metadata(metadata, backend)

    # Analyze model
    _populate_model_metadata(metadata, model)

    # Detect precision settings
    _populate_precision_metadata(metadata, model)

    # Detect fusion information
    _populate_fusion_metadata(metadata, model)

    # Run benchmarks if requested
    if benchmark and sample_input is not None:
        _run_benchmarks(metadata, model, sample_input, benchmark_iterations)

    return metadata


def _populate_hardware_metadata(metadata: OptimizationMetadata, backend: str) -> None:
    """Populate hardware-specific metadata."""
    if backend == "auto":
        if torch.cuda.is_available():
            backend = "cuda"
        else:
            backend = "cpu"

    metadata.hardware.backend = backend

    if backend == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata.hardware.compute_capability = (props.major, props.minor)
        metadata.hardware.memory_gb = props.total_memory / (1024**3)
        metadata.hardware.tensor_cores = props.major >= 7

        # Detect architecture
        if props.major >= 9:
            metadata.hardware.architecture = "hopper"
            metadata.hardware.fp8_support = True
        elif props.major >= 8:
            metadata.hardware.architecture = "ampere"
        elif props.major >= 7:
            metadata.hardware.architecture = "volta"
        else:
            metadata.hardware.architecture = "pascal"

        metadata.recommended_backends = ["cuda", "tensorrt", "cpu"]
        metadata.minimum_compute_capability = (7, 0)  # Volta minimum for tensor cores


def _populate_model_metadata(metadata: OptimizationMetadata, model: nn.Module) -> None:
    """Populate model architecture metadata."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    metadata.model.num_parameters = total_params
    metadata.performance.model_size_mb = total_params * 4 / (1024**2)  # Assuming FP32

    # Count layers
    num_layers = sum(1 for _ in model.modules())
    metadata.model.num_layers = num_layers

    # Try to detect model type
    model_class = model.__class__.__name__.lower()
    if 'transformer' in model_class or 'bert' in model_class or 'gpt' in model_class:
        metadata.model.model_type = 'transformer'
    elif 'resnet' in model_class or 'conv' in model_class or 'vgg' in model_class:
        metadata.model.model_type = 'cnn'
    elif 'linear' in model_class or 'mlp' in model_class:
        metadata.model.model_type = 'mlp'
    else:
        metadata.model.model_type = model_class

    # Try to extract architecture details from config if available
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'hidden_size'):
            metadata.model.hidden_size = config.hidden_size
        if hasattr(config, 'num_attention_heads'):
            metadata.model.num_attention_heads = config.num_attention_heads
        if hasattr(config, 'num_hidden_layers'):
            metadata.model.num_hidden_layers = config.num_hidden_layers
        if hasattr(config, 'intermediate_size'):
            metadata.model.intermediate_size = config.intermediate_size
        if hasattr(config, 'vocab_size'):
            metadata.model.vocab_size = config.vocab_size
        if hasattr(config, 'max_position_embeddings'):
            metadata.model.max_position_embeddings = config.max_position_embeddings


def _populate_precision_metadata(metadata: OptimizationMetadata, model: nn.Module) -> None:
    """Populate precision configuration metadata."""
    # Check parameter dtypes
    dtypes = set()
    for param in model.parameters():
        dtypes.add(str(param.dtype))

    if len(dtypes) > 1:
        metadata.precision.mixed_precision = True

    # Set default dtype
    if 'torch.float16' in dtypes:
        metadata.precision.default_dtype = 'float16'
    elif 'torch.bfloat16' in dtypes:
        metadata.precision.default_dtype = 'bfloat16'
    elif 'torch.float8_e4m3fn' in dtypes:
        metadata.precision.default_dtype = 'fp8_e4m3'
        metadata.precision.fp8_enabled = True
    else:
        metadata.precision.default_dtype = 'float32'


def _populate_fusion_metadata(metadata: OptimizationMetadata, model: nn.Module) -> None:
    """Populate kernel fusion information."""
    # Check for common fusion patterns
    module_types = [type(m).__name__ for m in model.modules()]

    # Detect fused operations
    if any('Fused' in name for name in module_types):
        metadata.fusion.ffn_fusion = True
        metadata.hardware.fused_kernels_enabled = True

    if any('FlashAttention' in name for name in module_types):
        metadata.fusion.attention_fusion = True
        metadata.hardware.flash_attention_enabled = True

    # Count fusion patterns
    fusion_count = sum(1 for name in module_types if 'Fused' in name or 'Flash' in name)
    metadata.fusion.fusion_count = fusion_count

    # Estimate speedup based on fusion count
    if fusion_count > 0:
        metadata.fusion.estimated_speedup = 1.0 + (fusion_count * 0.1)  # ~10% per fusion


def _run_benchmarks(
    metadata: OptimizationMetadata,
    model: nn.Module,
    sample_input: torch.Tensor,
    iterations: int
) -> None:
    """Run performance benchmarks and populate metadata."""
    import time

    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    model.eval()
    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)

    # Benchmark
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(sample_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    # Calculate metrics
    latencies.sort()
    metadata.performance.inference_latency_ms = sum(latencies) / len(latencies)
    metadata.performance.p50_latency_ms = latencies[len(latencies) // 2]
    metadata.performance.p95_latency_ms = latencies[int(len(latencies) * 0.95)]
    metadata.performance.p99_latency_ms = latencies[int(len(latencies) * 0.99)]

    # Calculate throughput
    avg_latency_sec = metadata.performance.inference_latency_ms / 1000
    batch_size = sample_input.shape[0] if sample_input.dim() > 0 else 1
    metadata.performance.throughput_samples_per_sec = batch_size / avg_latency_sec

    # Record benchmark parameters
    metadata.performance.benchmark_batch_size = batch_size
    metadata.performance.benchmark_iterations = iterations
    metadata.performance.benchmark_device = str(device)

    # Get peak memory
    if device.type == 'cuda':
        metadata.performance.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    logger.info(f"Benchmark complete: {metadata.performance.p50_latency_ms:.2f}ms p50 latency")
