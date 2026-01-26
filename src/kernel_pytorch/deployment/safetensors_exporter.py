"""
SafeTensors Export for KernelPyTorch (v0.4.25)

Provides export functionality to the SafeTensors format, which offers:
- Safer loading (no pickle execution)
- Faster loading (memory-mapped access)
- Size efficiency (no redundant metadata)
- HuggingFace ecosystem compatibility

Example:
    ```python
    from kernel_pytorch.deployment import export_to_safetensors

    result = export_to_safetensors(
        model=optimized_model,
        output_path="model.safetensors",
        metadata={"description": "My optimized model"}
    )
    ```
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SafeTensorsExportConfig:
    """Configuration for SafeTensors export.

    Args:
        output_path: Path for output .safetensors file
        include_optimizer: Whether to include optimizer state
        half_precision: Convert weights to FP16
        metadata: Custom metadata to embed in file
        strict: Fail on non-tensor model attributes
    """
    output_path: str = "model.safetensors"
    include_optimizer: bool = False
    half_precision: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    strict: bool = True


@dataclass
class SafeTensorsExportResult:
    """Result of SafeTensors export operation.

    Attributes:
        success: Whether export succeeded
        output_path: Path to exported file
        file_size_mb: Size of output file in MB
        num_tensors: Number of tensors exported
        metadata: Metadata embedded in file
        error_message: Error message if failed
    """
    success: bool
    output_path: str
    file_size_mb: float = 0.0
    num_tensors: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None


# =============================================================================
# SafeTensors Exporter
# =============================================================================

class SafeTensorsExporter:
    """Export PyTorch models to SafeTensors format.

    SafeTensors is a safe, fast, and efficient format for storing tensors.
    It's widely used in the HuggingFace ecosystem.

    Features:
    - No pickle vulnerabilities (safe loading)
    - Memory-mapped loading (fast)
    - Metadata embedding
    - FP16 conversion support

    Example:
        ```python
        exporter = SafeTensorsExporter()
        result = exporter.export(
            model=my_model,
            output_path="model.safetensors"
        )
        print(f"Exported {result.num_tensors} tensors")
        ```
    """

    def __init__(self):
        """Initialize SafeTensors exporter."""
        self._check_safetensors_available()

    def _check_safetensors_available(self) -> bool:
        """Check if safetensors library is available."""
        try:
            import safetensors
            return True
        except ImportError:
            logger.warning(
                "safetensors library not installed. "
                "Install with: pip install safetensors"
            )
            return False

    def export(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        config: Optional[SafeTensorsExportConfig] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> SafeTensorsExportResult:
        """Export model to SafeTensors format.

        Args:
            model: PyTorch model to export
            output_path: Path for output file
            config: Export configuration
            metadata: Additional metadata to embed

        Returns:
            SafeTensorsExportResult with export details
        """
        try:
            from safetensors.torch import save_file
        except ImportError:
            return SafeTensorsExportResult(
                success=False,
                output_path=str(output_path),
                error_message="safetensors library not installed. Install with: pip install safetensors"
            )

        if config is None:
            config = SafeTensorsExportConfig(output_path=str(output_path))

        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get model state dict
            state_dict = model.state_dict()

            # Convert to FP16 if requested
            if config.half_precision:
                state_dict = self._convert_to_half(state_dict)

            # Build metadata
            combined_metadata = {
                "format": "pt",  # PyTorch format
                "framework": "kernel-pytorch",
            }

            # Add config metadata
            if config.metadata:
                combined_metadata.update(config.metadata)

            # Add provided metadata
            if metadata:
                combined_metadata.update(metadata)

            # Add model info
            combined_metadata["num_parameters"] = str(
                sum(p.numel() for p in model.parameters())
            )
            combined_metadata["num_tensors"] = str(len(state_dict))

            # Convert metadata values to strings (required by safetensors)
            string_metadata = {
                k: str(v) for k, v in combined_metadata.items()
            }

            # Save to safetensors format
            save_file(state_dict, str(output_path), metadata=string_metadata)

            # Get file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

            logger.info(f"Exported model to {output_path} ({file_size_mb:.2f} MB)")

            return SafeTensorsExportResult(
                success=True,
                output_path=str(output_path),
                file_size_mb=file_size_mb,
                num_tensors=len(state_dict),
                metadata=string_metadata,
            )

        except Exception as e:
            logger.error(f"SafeTensors export failed: {e}")
            return SafeTensorsExportResult(
                success=False,
                output_path=str(output_path),
                error_message=str(e),
            )

    def _convert_to_half(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert state dict tensors to FP16.

        Args:
            state_dict: Model state dictionary

        Returns:
            State dict with FP16 tensors
        """
        converted = {}
        for key, tensor in state_dict.items():
            if tensor.dtype in (torch.float32, torch.float64):
                converted[key] = tensor.half()
            else:
                converted[key] = tensor
        return converted

    def load(
        self,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Load tensors from SafeTensors file.

        Args:
            path: Path to .safetensors file
            device: Device to load tensors to

        Returns:
            Dictionary of tensors
        """
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors library not installed. "
                "Install with: pip install safetensors"
            )

        return load_file(str(path), device=str(device))

    def load_model(
        self,
        model: nn.Module,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> nn.Module:
        """Load weights from SafeTensors file into model.

        Args:
            model: PyTorch model to load weights into
            path: Path to .safetensors file
            device: Device to load tensors to
            strict: Require all keys to match

        Returns:
            Model with loaded weights
        """
        state_dict = self.load(path, device)
        model.load_state_dict(state_dict, strict=strict)
        return model

    def get_metadata(self, path: Union[str, Path]) -> Dict[str, str]:
        """Get metadata from SafeTensors file without loading tensors.

        Args:
            path: Path to .safetensors file

        Returns:
            Metadata dictionary
        """
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError(
                "safetensors library not installed. "
                "Install with: pip install safetensors"
            )

        with safe_open(str(path), framework="pt") as f:
            return dict(f.metadata()) if f.metadata() else {}


# =============================================================================
# Convenience Functions
# =============================================================================

def export_to_safetensors(
    model: nn.Module,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
    half_precision: bool = False,
) -> SafeTensorsExportResult:
    """Export model to SafeTensors format.

    Convenience function for quick exports.

    Args:
        model: PyTorch model to export
        output_path: Path for output .safetensors file
        metadata: Custom metadata to embed
        half_precision: Convert to FP16

    Returns:
        SafeTensorsExportResult with export details

    Example:
        ```python
        result = export_to_safetensors(
            model=my_model,
            output_path="model.safetensors",
            metadata={"description": "Optimized ResNet50"}
        )
        if result.success:
            print(f"Exported {result.file_size_mb:.2f} MB")
        ```
    """
    exporter = SafeTensorsExporter()
    config = SafeTensorsExportConfig(
        output_path=str(output_path),
        half_precision=half_precision,
        metadata=metadata or {},
    )
    return exporter.export(model, output_path, config, metadata)


def load_safetensors(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load tensors from SafeTensors file.

    Args:
        path: Path to .safetensors file
        device: Device to load tensors to

    Returns:
        Dictionary of tensors
    """
    exporter = SafeTensorsExporter()
    return exporter.load(path, device)


def load_model_safetensors(
    model: nn.Module,
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load weights from SafeTensors file into model.

    Args:
        model: PyTorch model to load weights into
        path: Path to .safetensors file
        device: Device to load tensors to
        strict: Require all keys to match

    Returns:
        Model with loaded weights
    """
    exporter = SafeTensorsExporter()
    return exporter.load_model(model, path, device, strict)
