"""
Shared utility functions for TorchBridge demos.

This module provides common utility functions used across demo scripts.

Version: 0.4.3
"""

import argparse
from typing import Any

import torch

# ============================================================================
# Printing Utilities
# ============================================================================

def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subsection(title: str, width: int = 50) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def print_success(message: str) -> None:
    """Print a success message with checkmark."""
    print(f"✅ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def print_metrics(metrics: dict[str, Any], indent: int = 3) -> None:
    """Print a dictionary of metrics in a formatted way."""
    prefix = " " * indent
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_metrics(value, indent + 3)
        else:
            print(f"{prefix}{key}: {value}")


# ============================================================================
# Device Utilities
# ============================================================================

def get_device(prefer: str = "auto") -> torch.device:
    """
    Get the best available device.

    Args:
        prefer: Preference for device ("auto", "cuda", "mps", "cpu")

    Returns:
        torch.device for the selected device
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print_warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    elif prefer == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print_warning("MPS requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device("cpu")


def get_device_info(device: torch.device | None = None) -> dict[str, Any]:
    """
    Get detailed information about a device.

    Args:
        device: Device to get info for (None = current device)

    Returns:
        Dictionary with device information
    """
    if device is None:
        device = get_device()

    info = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "cuda":
        info.update({
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(device),
        })

        # Get compute capability
        props = torch.cuda.get_device_properties(device)
        info.update({
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / 1024 / 1024 / 1024,
            "multi_processor_count": props.multi_processor_count,
        })
    elif device.type == "mps":
        info["mps_available"] = True
    else:
        info["cpu_only"] = True

    return info


# ============================================================================
# Formatting Utilities
# ============================================================================

def format_memory(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        bytes_value: Memory in bytes

    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_value) < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} PB"


def format_time(milliseconds: float) -> str:
    """
    Format milliseconds to human-readable string.

    Args:
        milliseconds: Time in milliseconds

    Returns:
        Formatted string (e.g., "1.23 ms", "1.23 s")
    """
    if milliseconds < 1:
        return f"{milliseconds * 1000:.2f} μs"
    elif milliseconds < 1000:
        return f"{milliseconds:.2f} ms"
    else:
        return f"{milliseconds / 1000:.2f} s"


# ============================================================================
# Argument Parsing
# ============================================================================

def setup_demo_args(
    description: str = "TorchBridge Demo",
    add_quick: bool = True,
    add_device: bool = True,
    add_verbose: bool = True,
) -> argparse.ArgumentParser:
    """
    Create a standard argument parser for demos.

    Args:
        description: Description for the demo
        add_quick: Add --quick flag
        add_device: Add --device option
        add_verbose: Add --verbose flag

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)

    if add_quick:
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Run quick version of demo (fewer iterations)",
        )

    if add_device:
        parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cuda", "mps", "cpu"],
            help="Device to run demo on (default: auto)",
        )

    if add_verbose:
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

    return parser
