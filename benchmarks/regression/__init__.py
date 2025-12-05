#!/usr/bin/env python3
"""
Performance Regression Testing Framework

Enterprise-grade performance regression detection and baseline management
for PyTorch optimization frameworks. Provides statistical analysis,
automated threshold management, and CI integration.
"""

from .baseline_manager import BaselineManager, BaselineMetrics
from .regression_detector import RegressionDetector, RegressionResult, RegressionSeverity
from .threshold_manager import ThresholdManager, ThresholdConfig

__all__ = [
    'BaselineManager',
    'BaselineMetrics',
    'RegressionDetector',
    'RegressionResult',
    'RegressionSeverity',
    'ThresholdManager',
    'ThresholdConfig'
]

__version__ = "0.1.58"