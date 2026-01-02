"""
Shared utilities and models for KernelPyTorch demos.

This module provides reusable components for demonstration scripts:
- Printing and formatting utilities
- Common model architectures
- Device detection helpers
- Sample data generators

Version: 0.3.6
"""

from .utils import (
    print_section,
    print_subsection,
    print_success,
    print_info,
    print_warning,
    print_error,
    print_metrics,
    get_device,
    get_device_info,
    format_memory,
    format_time,
    setup_demo_args,
)

from .models import (
    SimpleLinear,
    SimpleMLP,
    SimpleTransformer,
    SimpleCNN,
    SimpleAttention,
    create_model,
)

from .data import (
    create_linear_input,
    create_transformer_input,
    create_vision_input,
    create_sample_batch,
)

__version__ = "0.3.6"

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
