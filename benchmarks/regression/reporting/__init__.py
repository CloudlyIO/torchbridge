"""
Performance Regression Reporting Module

Provides comprehensive reporting and visualization capabilities for
performance regression testing results.
"""

from .dashboard_generator import ChartType, Dashboard, DashboardGenerator
from .regression_reporter import (
    CISummary,
    ExecutiveSummary,
    RegressionReporter,
    Report,
    ReportFormat,
)

__all__ = [
    'RegressionReporter',
    'ReportFormat',
    'Report',
    'CISummary',
    'ExecutiveSummary',
    'DashboardGenerator',
    'Dashboard',
    'ChartType'
]

__version__ = "0.1.59"
