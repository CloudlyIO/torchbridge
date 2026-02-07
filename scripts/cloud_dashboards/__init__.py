"""
Cloud Monitoring Dashboards for TorchBridge.

This module provides dashboard configurations and comparison tools
for monitoring TorchBridge performance across cloud platforms.

Version: 0.5.3
"""

from .cross_platform_comparison import (
    CrossPlatformComparison,
    PlatformMetrics,
    generate_comparison_report,
    create_comparison_chart,
)

__version__ = "0.5.3"

__all__ = [
    "CrossPlatformComparison",
    "PlatformMetrics",
    "generate_comparison_report",
    "create_comparison_chart",
]
