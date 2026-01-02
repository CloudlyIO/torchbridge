"""
Cloud Monitoring Dashboards for KernelPyTorch.

This module provides dashboard configurations and comparison tools
for monitoring KernelPyTorch performance across cloud platforms.

Version: 0.3.7
"""

from .cross_platform_comparison import (
    CrossPlatformComparison,
    PlatformMetrics,
    generate_comparison_report,
    create_comparison_chart,
)

__version__ = "0.3.7"

__all__ = [
    "CrossPlatformComparison",
    "PlatformMetrics",
    "generate_comparison_report",
    "create_comparison_chart",
]
