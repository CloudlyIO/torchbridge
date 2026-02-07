"""
Triton Inference Server Integration for TorchBridge

This module provides configuration generation and model repository setup
for deploying TorchBridge-optimized models to NVIDIA Triton Inference Server.

Features:
- Model configuration generation (config.pbtxt)
- Dynamic batching configuration
- Instance group management (multi-GPU)
- Model repository structure generation
- PyTorch, TorchScript, and ONNX backend support

Example:
    ```python
    from torchbridge.deployment.serving import (
        TritonModelConfig,
        create_triton_config,
        generate_triton_model_repository
    )

    # Generate configuration
    config = create_triton_config(
        model_name="my_model",
        inputs=[("input", "FP32", [1, 512])],
        outputs=[("output", "FP32", [1, 10])],
        max_batch_size=32
    )

    # Create model repository
    generate_triton_model_repository(
        model=model,
        output_dir="model_repository",
        config=config
    )
    ```

"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class TritonBackend(Enum):
    """Supported Triton backends."""

    PYTORCH = "pytorch"  # PyTorch backend (requires LibTorch)
    ONNXRUNTIME = "onnxruntime"  # ONNX Runtime backend
    TENSORRT = "tensorrt"  # TensorRT backend (NVIDIA GPUs)
    PYTHON = "python"  # Python backend (flexible but slower)

class TritonDataType(Enum):
    """Triton data types."""

    BOOL = "TYPE_BOOL"
    UINT8 = "TYPE_UINT8"
    UINT16 = "TYPE_UINT16"
    UINT32 = "TYPE_UINT32"
    UINT64 = "TYPE_UINT64"
    INT8 = "TYPE_INT8"
    INT16 = "TYPE_INT16"
    INT32 = "TYPE_INT32"
    INT64 = "TYPE_INT64"
    FP16 = "TYPE_FP16"
    FP32 = "TYPE_FP32"
    FP64 = "TYPE_FP64"
    STRING = "TYPE_STRING"
    BF16 = "TYPE_BF16"

    @classmethod
    def from_torch_dtype(cls, dtype: torch.dtype) -> "TritonDataType":
        """Convert PyTorch dtype to Triton data type."""
        mapping = {
            torch.bool: cls.BOOL,
            torch.uint8: cls.UINT8,
            torch.int8: cls.INT8,
            torch.int16: cls.INT16,
            torch.int32: cls.INT32,
            torch.int64: cls.INT64,
            torch.float16: cls.FP16,
            torch.float32: cls.FP32,
            torch.float64: cls.FP64,
            torch.bfloat16: cls.BF16,
        }
        return mapping.get(dtype, cls.FP32)

    @classmethod
    def from_string(cls, dtype_str: str) -> "TritonDataType":
        """Convert string to Triton data type."""
        dtype_str = dtype_str.upper()
        if not dtype_str.startswith("TYPE_"):
            dtype_str = f"TYPE_{dtype_str}"

        for member in cls:
            if member.value == dtype_str:
                return member
        return cls.FP32

@dataclass
class TritonInput:
    """Triton model input specification."""

    name: str
    data_type: TritonDataType
    dims: list[int]
    optional: bool = False
    reshape: list[int] | None = None

    def to_config_str(self) -> str:
        """Convert to config.pbtxt format."""
        lines = [
            "input {",
            f'  name: "{self.name}"',
            f"  data_type: {self.data_type.value}",
            f"  dims: [{', '.join(str(d) for d in self.dims)}]",
        ]

        if self.optional:
            lines.append("  optional: true")

        if self.reshape:
            lines.append(f"  reshape {{ shape: [{', '.join(str(d) for d in self.reshape)}] }}")

        lines.append("}")
        return "\n".join(lines)

@dataclass
class TritonOutput:
    """Triton model output specification."""

    name: str
    data_type: TritonDataType
    dims: list[int]
    reshape: list[int] | None = None
    label_filename: str | None = None

    def to_config_str(self) -> str:
        """Convert to config.pbtxt format."""
        lines = [
            "output {",
            f'  name: "{self.name}"',
            f"  data_type: {self.data_type.value}",
            f"  dims: [{', '.join(str(d) for d in self.dims)}]",
        ]

        if self.reshape:
            lines.append(f"  reshape {{ shape: [{', '.join(str(d) for d in self.reshape)}] }}")

        if self.label_filename:
            lines.append(f'  label_filename: "{self.label_filename}"')

        lines.append("}")
        return "\n".join(lines)

@dataclass
class TritonInstanceGroup:
    """Triton instance group configuration."""

    count: int = 1
    kind: str = "KIND_GPU"  # KIND_GPU, KIND_CPU, KIND_MODEL
    gpus: list[int] = field(default_factory=list)

    def to_config_str(self) -> str:
        """Convert to config.pbtxt format."""
        lines = [
            "instance_group {",
            f"  count: {self.count}",
            f"  kind: {self.kind}",
        ]

        if self.gpus and self.kind == "KIND_GPU":
            lines.append(f"  gpus: [{', '.join(str(g) for g in self.gpus)}]")

        lines.append("}")
        return "\n".join(lines)

@dataclass
class TritonDynamicBatching:
    """Triton dynamic batching configuration."""

    preferred_batch_size: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    max_queue_delay_microseconds: int = 100000  # 100ms
    preserve_ordering: bool = False
    priority_levels: int = 0
    default_priority_level: int = 0

    def to_config_str(self) -> str:
        """Convert to config.pbtxt format."""
        lines = ["dynamic_batching {"]

        if self.preferred_batch_size:
            lines.append(
                f"  preferred_batch_size: [{', '.join(str(s) for s in self.preferred_batch_size)}]"
            )

        lines.append(f"  max_queue_delay_microseconds: {self.max_queue_delay_microseconds}")

        if self.preserve_ordering:
            lines.append("  preserve_ordering: true")

        if self.priority_levels > 0:
            lines.append(f"  priority_levels: {self.priority_levels}")
            lines.append(f"  default_priority_level: {self.default_priority_level}")

        lines.append("}")
        return "\n".join(lines)

@dataclass
class TritonModelConfig:
    """Complete Triton model configuration."""

    name: str
    platform: str = ""  # Empty for backend specification
    backend: TritonBackend = TritonBackend.PYTORCH
    max_batch_size: int = 32
    inputs: list[TritonInput] = field(default_factory=list)
    outputs: list[TritonOutput] = field(default_factory=list)
    instance_groups: list[TritonInstanceGroup] = field(default_factory=list)
    dynamic_batching: TritonDynamicBatching | None = None
    version_policy: str = "latest"  # latest, all, specific
    default_model_filename: str = "model.pt"

    # Optimization settings
    optimization: dict[str, Any] | None = None

    # Parameters for Python backend
    parameters: dict[str, str] = field(default_factory=dict)

    def to_config_str(self) -> str:
        """Generate config.pbtxt content."""
        lines = [
            f'name: "{self.name}"',
        ]

        # Backend or platform
        if self.backend:
            lines.append(f'backend: "{self.backend.value}"')
        if self.platform:
            lines.append(f'platform: "{self.platform}"')

        lines.append(f"max_batch_size: {self.max_batch_size}")

        # Default model filename
        if self.default_model_filename:
            lines.append(f'default_model_filename: "{self.default_model_filename}"')

        # Inputs
        for inp in self.inputs:
            lines.append(inp.to_config_str())

        # Outputs
        for out in self.outputs:
            lines.append(out.to_config_str())

        # Instance groups
        for ig in self.instance_groups:
            lines.append(ig.to_config_str())

        # Dynamic batching
        if self.dynamic_batching:
            lines.append(self.dynamic_batching.to_config_str())

        # Version policy
        if self.version_policy == "latest":
            lines.append("version_policy { latest { num_versions: 1 } }")
        elif self.version_policy == "all":
            lines.append("version_policy { all { } }")

        # Optimization
        if self.optimization:
            opt_lines = ["optimization {"]
            if "cuda" in self.optimization:
                cuda_opt = self.optimization["cuda"]
                opt_lines.append("  cuda {")
                if "graphs" in cuda_opt:
                    opt_lines.append(f"    graphs: {str(cuda_opt['graphs']).lower()}")
                opt_lines.append("  }")
            opt_lines.append("}")
            lines.append("\n".join(opt_lines))

        # Parameters
        for key, value in self.parameters.items():
            lines.append(f'parameters {{ key: "{key}" value: {{ string_value: "{value}" }} }}')

        return "\n\n".join(lines)

    def save(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            f.write(self.to_config_str())
        logger.info(f"Saved Triton config to {path}")

def create_triton_config(
    model_name: str,
    inputs: list[tuple[str, str, list[int]]],
    outputs: list[tuple[str, str, list[int]]],
    max_batch_size: int = 32,
    backend: str | TritonBackend = TritonBackend.PYTORCH,
    enable_dynamic_batching: bool = True,
    gpu_count: int = 1,
    instances_per_gpu: int = 1,
) -> TritonModelConfig:
    """
    Create a Triton model configuration.

    Args:
        model_name: Name of the model
        inputs: List of (name, dtype, dims) tuples
        outputs: List of (name, dtype, dims) tuples
        max_batch_size: Maximum batch size
        backend: Triton backend to use
        enable_dynamic_batching: Enable dynamic batching
        gpu_count: Number of GPUs to use
        instances_per_gpu: Model instances per GPU

    Returns:
        TritonModelConfig instance
    """
    # Convert backend string to enum
    if isinstance(backend, str):
        backend = TritonBackend(backend.lower())

    # Create input specifications
    input_specs = []
    for name, dtype, dims in inputs:
        triton_dtype = TritonDataType.from_string(dtype)
        input_specs.append(TritonInput(name=name, data_type=triton_dtype, dims=dims))

    # Create output specifications
    output_specs = []
    for name, dtype, dims in outputs:
        triton_dtype = TritonDataType.from_string(dtype)
        output_specs.append(TritonOutput(name=name, data_type=triton_dtype, dims=dims))

    # Create instance groups
    instance_groups = []
    if gpu_count > 0:
        for i in range(gpu_count):
            instance_groups.append(
                TritonInstanceGroup(count=instances_per_gpu, kind="KIND_GPU", gpus=[i])
            )
    else:
        instance_groups.append(TritonInstanceGroup(count=instances_per_gpu, kind="KIND_CPU"))

    # Create dynamic batching config
    dynamic_batching = None
    if enable_dynamic_batching:
        # Create preferred batch sizes based on max_batch_size
        preferred = [s for s in [1, 2, 4, 8, 16, 32, 64] if s <= max_batch_size]
        dynamic_batching = TritonDynamicBatching(preferred_batch_size=preferred)

    # Determine default model filename based on backend
    if backend == TritonBackend.ONNXRUNTIME:
        default_filename = "model.onnx"
    elif backend == TritonBackend.TENSORRT:
        default_filename = "model.plan"
    else:
        default_filename = "model.pt"

    return TritonModelConfig(
        name=model_name,
        backend=backend,
        max_batch_size=max_batch_size,
        inputs=input_specs,
        outputs=output_specs,
        instance_groups=instance_groups,
        dynamic_batching=dynamic_batching,
        default_model_filename=default_filename,
    )

def generate_triton_model_repository(
    model: nn.Module,
    output_dir: str,
    config: TritonModelConfig,
    sample_input: torch.Tensor | None = None,
    version: str = "1",
    export_format: str = "torchscript",  # torchscript, onnx
) -> str:
    """
    Generate a complete Triton model repository.

    Creates the directory structure expected by Triton:
        model_repository/
        └── model_name/
            ├── config.pbtxt
            └── 1/
                └── model.pt (or model.onnx)

    Args:
        model: PyTorch model to deploy
        output_dir: Base directory for the model repository
        config: Triton model configuration
        sample_input: Sample input for model tracing
        version: Model version (default "1")
        export_format: Export format (torchscript or onnx)

    Returns:
        Path to the created model directory
    """
    output_path = Path(output_dir)
    model_dir = output_path / config.name
    version_dir = model_dir / version

    # Create directories
    version_dir.mkdir(parents=True, exist_ok=True)

    # Export model
    model.eval()

    if export_format == "onnx" or config.backend == TritonBackend.ONNXRUNTIME:
        # ONNX export
        model_path = version_dir / "model.onnx"
        config.default_model_filename = "model.onnx"
        config.backend = TritonBackend.ONNXRUNTIME

        if sample_input is None:
            raise ValueError("sample_input is required for ONNX export")

        # Build input/output names
        input_names = [inp.name for inp in config.inputs]
        output_names = [out.name for out in config.outputs]

        # Build dynamic axes
        dynamic_axes = {}
        for inp in config.inputs:
            if config.max_batch_size > 0:
                dynamic_axes[inp.name] = {0: "batch_size"}
        for out in config.outputs:
            if config.max_batch_size > 0:
                dynamic_axes[out.name] = {0: "batch_size"}

        torch.onnx.export(
            model,
            sample_input,
            str(model_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            opset_version=17,
        )
        logger.info(f"Exported ONNX model to {model_path}")

    else:
        # TorchScript export
        model_path = version_dir / "model.pt"

        if sample_input is not None:
            # Trace the model
            traced = torch.jit.trace(model, sample_input)
            torch.jit.save(traced, model_path)
        else:
            # Try scripting
            try:
                scripted = torch.jit.script(model)
                torch.jit.save(scripted, model_path)
            except Exception as e:
                logger.warning(f"Script failed: {e}, saving state dict")
                torch.save(model.state_dict(), model_path)

        logger.info(f"Exported TorchScript model to {model_path}")

    # Save configuration
    config_path = model_dir / "config.pbtxt"
    config.save(str(config_path))

    # Create metadata file with TorchBridge info
    metadata = {
        "framework": "torchbridge",
        "version": "0.3.9",
        "export_format": export_format,
        "model_config": {
            "name": config.name,
            "backend": config.backend.value,
            "max_batch_size": config.max_batch_size,
        },
    }

    metadata_path = model_dir / "torchbridge_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created Triton model repository at {model_dir}")
    return str(model_dir)
