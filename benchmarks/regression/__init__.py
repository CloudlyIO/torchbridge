#!/usr/bin/env python3
"""
Performance Regression Testing Framework

Enterprise-grade performance regression detection and baseline management
for PyTorch workloads. Provides statistical analysis,
automated threshold management, historical analysis, reporting, and CI integration.

Phase 1: Core Infrastructure (✅ Complete)
- BaselineManager: Historical baseline establishment and management
- RegressionDetector: Statistical regression detection with severity classification
- ThresholdManager: Adaptive threshold configuration with environment awareness

Phase 2: Historical Analysis & Reporting (✅ Complete)
- HistoricalAnalyzer: Long-term performance trend analysis and anomaly detection
- RegressionReporter: Automated report generation in multiple formats
- DashboardGenerator: Interactive performance dashboards and visualizations
"""

from .baseline_manager import BaselineManager, BaselineMetrics
from .regression_detector import RegressionDetector, RegressionResult, RegressionSeverity
from .threshold_manager import ThresholdManager, ThresholdConfig
from .historical_analyzer import (
    HistoricalAnalyzer,
    TrendAnalysis,
    AnomalyReport,
    PerformanceSummary,
    TrendDirection,
    AnomalyType
)
from .reporting import (
    RegressionReporter,
    DashboardGenerator,
    Report,
    CISummary,
    ExecutiveSummary,
    Dashboard,
    ReportFormat,
    ChartType
)

__all__ = [
    # Phase 1 - Core Infrastructure
    'BaselineManager',
    'BaselineMetrics',
    'RegressionDetector',
    'RegressionResult',
    'RegressionSeverity',
    'ThresholdManager',
    'ThresholdConfig',

    # Phase 2 - Historical Analysis & Reporting
    'HistoricalAnalyzer',
    'TrendAnalysis',
    'AnomalyReport',
    'PerformanceSummary',
    'TrendDirection',
    'AnomalyType',
    'RegressionReporter',
    'DashboardGenerator',
    'Report',
    'CISummary',
    'ExecutiveSummary',
    'Dashboard',
    'ReportFormat',
    'ChartType'
]

__version__ = "0.1.59"