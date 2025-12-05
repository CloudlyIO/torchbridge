"""
Performance Regression Reporting Module

Provides comprehensive reporting and visualization capabilities for
performance regression testing results.
"""

from .regression_reporter import (
    RegressionReporter,
    ReportFormat,
    Report,
    CISummary,
    ExecutiveSummary
)

from .dashboard_generator import (
    DashboardGenerator,
    Dashboard,
    ChartType
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