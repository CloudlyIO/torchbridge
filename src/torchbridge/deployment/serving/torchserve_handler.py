"""
TorchServe Integration for TorchBridge

This module provides a custom TorchServe handler that leverages TorchBridge
optimizations for production inference serving.

Features:
- Automatic model optimization on load
- TorchBridge optimization metadata support
- Batch inference with dynamic batching
- Health monitoring and metrics
- FP8/FP16 precision support

Example:
    ```python
    # In your handler file for TorchServe:
    from torchbridge.deployment.serving import TorchBridgeHandler

    class MyModelHandler(TorchBridgeHandler):
        def preprocess(self, data):
            # Custom preprocessing
            return super().preprocess(data)
    ```

"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class HandlerConfig:
    """Configuration for the TorchServe handler."""

    # Optimization settings
    optimization_level: str = "balanced"  # conservative, balanced, aggressive
    enable_fp16: bool = True
    enable_fp8: bool = False  # Requires H100/Blackwell

    # Batching settings
    max_batch_size: int = 32
    batch_timeout_ms: int = 100

    # Device settings
    device: str = "auto"  # auto, cuda, cpu
    device_ids: list[int] = field(default_factory=list)

    # Monitoring
    enable_metrics: bool = True
    log_inference_time: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandlerConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

class BaseHandler(ABC):
    """
    Abstract base handler for TorchServe integration.

    Provides the interface expected by TorchServe with TorchBridge
    optimization capabilities.
    """

    def __init__(self):
        self.model: nn.Module | None = None
        self.device: torch.device | None = None
        self.config: HandlerConfig = HandlerConfig()
        self.initialized: bool = False
        self.context = None

        # Metrics
        self._inference_count: int = 0
        self._total_inference_time: float = 0.0
        self._last_inference_time: float = 0.0

    def initialize(self, context) -> None:
        """
        Initialize the handler with model and context.

        This method is called by TorchServe when the model is loaded.

        Args:
            context: TorchServe context containing model information
        """
        self.context = context
        properties = context.system_properties if context else {}
        model_dir = properties.get("model_dir", ".")

        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                gpu_id = properties.get("gpu_id", 0)
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(f"Initializing handler on device: {self.device}")

        # Load configuration if available
        config_path = os.path.join(model_dir, "handler_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = HandlerConfig.from_dict(config_data)

        # Load model
        self.model = self._load_model(model_dir)

        # Apply optimizations
        self.model = self._optimize_model(self.model)

        self.initialized = True
        logger.info("Handler initialization complete")

    def _load_model(self, model_dir: str) -> nn.Module:
        """
        Load the model from the model directory.

        Supports TorchScript (.pt) and state dict formats.
        """
        # Try TorchScript first
        ts_path = os.path.join(model_dir, "model.pt")
        if os.path.exists(ts_path):
            logger.info(f"Loading TorchScript model from {ts_path}")
            model = torch.jit.load(ts_path, map_location=self.device)
            return model

        # Try serialized model
        serialized_path = os.path.join(model_dir, "model.pth")
        if os.path.exists(serialized_path):
            logger.info(f"Loading serialized model from {serialized_path}")
            model = torch.load(serialized_path, map_location=self.device)
            return model

        raise FileNotFoundError(
            f"No model file found in {model_dir}. "
            "Expected 'model.pt' (TorchScript) or 'model.pth' (serialized)."
        )

    def _optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply TorchBridge optimizations to the model.
        """
        model = model.to(self.device)
        model.eval()

        # Apply precision optimization
        if self.config.enable_fp8 and torch.cuda.is_available():
            try:
                # FP8 requires specific hardware
                from torchbridge.precision import FP8Config  # noqa: F401
                logger.info("FP8 optimization requested (requires H100/Blackwell)")
            except ImportError:
                logger.warning("FP8 not available, falling back to FP16")
                self.config.enable_fp8 = False

        if self.config.enable_fp16 and not self.config.enable_fp8:
            if torch.cuda.is_available():
                model = model.half()
                logger.info("Model converted to FP16")

        # Try to apply torch.compile for additional optimization
        if hasattr(torch, "compile") and self.config.optimization_level != "conservative":
            try:
                mode = "reduce-overhead" if self.config.optimization_level == "aggressive" else "default"
                model = torch.compile(model, mode=mode)
                logger.info(f"Applied torch.compile with mode={mode}")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        return model

    @abstractmethod
    def preprocess(self, data: list[dict[str, Any]]) -> torch.Tensor:
        """
        Preprocess input data into model-ready tensors.

        Args:
            data: List of input data dictionaries from the request

        Returns:
            Preprocessed tensor batch
        """
        pass

    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> list[dict[str, Any]]:
        """
        Postprocess model output into response format.

        Args:
            output: Model output tensor

        Returns:
            List of response dictionaries
        """
        pass

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        """
        Run inference on preprocessed data.

        Args:
            data: Preprocessed input tensor

        Returns:
            Model output tensor
        """
        with torch.no_grad():
            if self.config.enable_fp16 and data.dtype == torch.float32:
                data = data.half()

            data = data.to(self.device)
            output = self.model(data)

        return output

    def handle(self, data: list[dict[str, Any]], context) -> list[dict[str, Any]]:
        """
        Main entry point for TorchServe inference requests.

        Args:
            data: List of input data from request
            context: TorchServe context

        Returns:
            List of response data
        """
        if not self.initialized:
            self.initialize(context)

        start_time = time.time()

        try:
            # Preprocess
            model_input = self.preprocess(data)

            # Inference
            model_output = self.inference(model_input)

            # Postprocess
            result = self.postprocess(model_output)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            result = [{"error": str(e)}] * len(data)

        # Update metrics
        inference_time = time.time() - start_time
        self._inference_count += 1
        self._total_inference_time += inference_time
        self._last_inference_time = inference_time

        if self.config.log_inference_time:
            logger.info(f"Inference completed in {inference_time*1000:.2f}ms")

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get handler metrics."""
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0.0
        )
        return {
            "inference_count": self._inference_count,
            "total_inference_time_ms": self._total_inference_time * 1000,
            "average_inference_time_ms": avg_time * 1000,
            "last_inference_time_ms": self._last_inference_time * 1000,
        }

class TorchBridgeHandler(BaseHandler):
    """
    Default TorchBridge handler for TorchServe.

    Provides generic preprocessing and postprocessing for tensor-based models.
    Extend this class for model-specific customization.

    Example:
        ```python
        class MyHandler(TorchBridgeHandler):
            def preprocess(self, data):
                # Custom preprocessing
                tensors = []
                for item in data:
                    # Parse your input format
                    tensor = torch.tensor(item.get("input", []))
                    tensors.append(tensor)
                return torch.stack(tensors)
        ```
    """

    def preprocess(self, data: list[dict[str, Any]]) -> torch.Tensor:
        """
        Default preprocessing: expects 'input' key with tensor data.

        Input format:
            [{"input": [1.0, 2.0, 3.0, ...]}, ...]
        """
        tensors = []
        for item in data:
            if isinstance(item, dict):
                # Handle body wrapper from TorchServe
                if "body" in item:
                    item = item["body"]
                    if isinstance(item, bytes):
                        item = json.loads(item.decode("utf-8"))

                input_data = item.get("input", item.get("data", []))
            elif isinstance(item, (list, tuple)):
                input_data = item
            else:
                input_data = [item]

            tensor = torch.tensor(input_data, dtype=torch.float32)
            tensors.append(tensor)

        if not tensors:
            raise ValueError("No valid input data found")

        return torch.stack(tensors)

    def postprocess(self, output: torch.Tensor) -> list[dict[str, Any]]:
        """
        Default postprocessing: returns output tensor as list.

        Output format:
            [{"output": [1.0, 2.0, 3.0, ...]}, ...]
        """
        output = output.cpu()

        if output.dim() == 1:
            output = output.unsqueeze(0)

        results = []
        for i in range(output.size(0)):
            item_output = output[i].tolist()
            results.append({
                "output": item_output,
                "inference_time_ms": self._last_inference_time * 1000,
            })

        return results

def create_torchserve_handler(
    handler_class: type = TorchBridgeHandler,
    config: HandlerConfig | None = None,
) -> BaseHandler:
    """
    Create a TorchServe handler instance.

    Args:
        handler_class: Handler class to instantiate
        config: Handler configuration

    Returns:
        Configured handler instance
    """
    handler = handler_class()
    if config:
        handler.config = config
    return handler

@dataclass
class TorchServePackageConfig:
    """Configuration for TorchServe model packaging."""

    model_name: str
    version: str = "1.0"
    handler: str = "torchbridge.deployment.serving.torchserve_handler:TorchBridgeHandler"
    runtime: str = "python"
    requirements_file: str | None = None
    extra_files: list[str] = field(default_factory=list)

    # Handler config to embed
    handler_config: HandlerConfig | None = None

def package_for_torchserve(
    model: nn.Module,
    output_path: str,
    config: TorchServePackageConfig,
    sample_input: torch.Tensor | None = None,
) -> str:
    """
    Package a model for TorchServe deployment.

    Creates a .mar (Model Archive) file that can be deployed to TorchServe.

    Args:
        model: PyTorch model to package
        output_path: Path for the output .mar file
        config: Packaging configuration
        sample_input: Optional sample input for TorchScript tracing

    Returns:
        Path to the created .mar file
    """
    # Create temporary directory for model files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save model
        model_path = temp_path / "model.pt"
        if sample_input is not None:
            # Use tracing if sample input provided
            traced = torch.jit.trace(model.eval(), sample_input)
            torch.jit.save(traced, model_path)
        else:
            # Try scripting, fall back to state dict
            try:
                scripted = torch.jit.script(model.eval())
                torch.jit.save(scripted, model_path)
            except Exception:
                # Save as serialized model
                model_path = temp_path / "model.pth"
                torch.save(model, model_path)

        # Save handler config if provided
        if config.handler_config:
            config_path = temp_path / "handler_config.json"
            with open(config_path, "w") as f:
                json.dump(config.handler_config.to_dict(), f, indent=2)

        # Copy extra files
        for extra_file in config.extra_files:
            if os.path.exists(extra_file):
                shutil.copy(extra_file, temp_path)

        # Check if torch-model-archiver is available
        try:
            result = subprocess.run(
                ["torch-model-archiver", "--help"],
                capture_output=True,
                text=True
            )
            archiver_available = result.returncode == 0
        except FileNotFoundError:
            archiver_available = False

        if archiver_available:
            # Use torch-model-archiver
            cmd = [
                "torch-model-archiver",
                "--model-name", config.model_name,
                "--version", config.version,
                "--serialized-file", str(model_path),
                "--handler", config.handler,
                "--runtime", config.runtime,
                "--export-path", str(Path(output_path).parent),
            ]

            if config.requirements_file:
                cmd.extend(["--requirements-file", config.requirements_file])

            extra_paths = [str(p) for p in temp_path.glob("*") if p != model_path]
            if extra_paths:
                cmd.extend(["--extra-files", ",".join(extra_paths)])

            subprocess.run(cmd, check=True)
            mar_path = str(Path(output_path).parent / f"{config.model_name}.mar")
        else:
            # Create a simple archive manually
            logger.warning(
                "torch-model-archiver not found. Creating simple archive. "
                "Install with: pip install torchserve torch-model-archiver"
            )

            import zipfile
            mar_path = output_path if output_path.endswith(".mar") else f"{output_path}.mar"

            with zipfile.ZipFile(mar_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in temp_path.glob("*"):
                    zf.write(file_path, file_path.name)

                # Add manifest
                manifest = {
                    "model": {
                        "modelName": config.model_name,
                        "modelVersion": config.version,
                        "handler": config.handler,
                        "runtime": config.runtime,
                    }
                }
                zf.writestr("MAR-INF/MANIFEST.json", json.dumps(manifest, indent=2))

        logger.info(f"Created TorchServe archive: {mar_path}")
        return mar_path
