"""
Shared utilities and models for TorchBridge demos.

This module provides reusable components for demonstration scripts:
- Printing and formatting utilities
- Common model architectures
- Device detection helpers
- Sample data generators

Version: 0.3.6
"""

from .data import (
    create_linear_input,
    create_sample_batch,
    create_transformer_input,
    create_vision_input,
)
from .models import (
    SimpleAttention,
    SimpleCNN,
    SimpleLinear,
    SimpleMLP,
    SimpleTransformer,
    create_model,
)
from .utils import (
    format_memory,
    format_time,
    get_device,
    get_device_info,
    print_error,
    print_info,
    print_metrics,
    print_section,
    print_subsection,
    print_success,
    print_warning,
    setup_demo_args,
)

__version__ = "0.4.2"

__all__ = [
    # Printing utilities
    "print_section",
    "print_subsection",
    "print_success",
    "print_info",
    "print_warning",
    "print_error",
    "print_metrics",
    # Device utilities
    "get_device",
    "get_device_info",
    # Formatting
    "format_memory",
    "format_time",
    # Argument parsing
    "setup_demo_args",
    # Models
    "SimpleLinear",
    "SimpleMLP",
    "SimpleTransformer",
    "SimpleCNN",
    "SimpleAttention",
    "create_model",
    # Data
    "create_linear_input",
    "create_transformer_input",
    "create_vision_input",
    "create_sample_batch",
]
