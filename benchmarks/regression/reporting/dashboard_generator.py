#!/usr/bin/env python3
"""
Performance Dashboard Generator

Creates interactive performance dashboards and visualizations for
regression testing results and historical performance analysis.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import statistics
import warnings

# Import from existing components
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from ..regression_detector import RegressionResult, RegressionSeverity
from ..historical_analyzer import HistoricalAnalyzer, PerformanceSummary, TrendAnalysis, AnomalyReport
from ..baseline_manager import BaselineManager
from .regression_reporter import Report


class ChartType(Enum):
    """Supported chart types for dashboard"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    PIE = "pie"


@dataclass
class ChartData:
    """Data structure for dashboard charts"""
    chart_id: str
    title: str
    chart_type: ChartType
    data: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Complete performance dashboard"""
    title: str
    subtitle: str
    generated_at: datetime
    time_period: str
    charts: List[ChartData]
    summary_stats: Dict[str, Any]
    alerts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to dictionary for serialization"""
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "generated_at": self.generated_at.isoformat(),
            "time_period": self.time_period,
            "charts": [
                {
                    "chart_id": chart.chart_id,
                    "title": chart.title,
                    "chart_type": chart.chart_type.value,
                    "data": chart.data,
                    "config": chart.config
                }
                for chart in self.charts
            ],
            "summary_stats": self.summary_stats,
            "alerts": self.alerts,
            "metadata": self.metadata
        }


class DashboardGenerator:
    """
    Generates interactive performance dashboards and visualizations.

    Provides functionality to:
    - Create comprehensive performance dashboards
    - Generate trend visualization charts
    - Build regression analysis displays
    - Export dashboards in multiple formats
    - Create real-time monitoring views
    """

    def __init__(self, historical_analyzer: HistoricalAnalyzer = None, baseline_manager: BaselineManager = None):
        self.historical_analyzer = historical_analyzer or HistoricalAnalyzer()
        self.baseline_manager = baseline_manager or BaselineManager()
        self.version = "0.1.59"

    def create_performance_dashboard(
        self,
        models: List[str],
        time_period: str = "30d",
        include_trends: bool = True,
        include_anomalies: bool = True
    ) -> Dashboard:
        """
        Create comprehensive performance dashboard.

        Args:
            models: List of model names to include
            time_period: Time period for analysis
            include_trends: Include trend analysis charts
            include_anomalies: Include anomaly detection charts

        Returns:
            Dashboard with performance visualizations
        """
        print(f"📊 Creating performance dashboard for {len(models)} models ({time_period})...")

        charts = []
        all_summaries = []
        alerts = []

        # Generate performance summaries for each model
        for model_name in models:
            try:
                summary = self.historical_analyzer.generate_performance_summary(
                    model_name, time_range=time_period
                )
                all_summaries.append(summary)

                # Add model-specific alerts
                if summary.stability_score < 0.7:
                    alerts.append(f"⚠️ {model_name}: Low stability score ({summary.stability_score:.1%})")
                if summary.anomaly_count > 5:
                    alerts.append(f"🔍 {model_name}: High anomaly count ({summary.anomaly_count})")

            except Exception as e:
                warnings.warn(f"Failed to analyze {model_name}: {e}")
                alerts.append(f"❌ {model_name}: Analysis failed - {str(e)}")

        if not all_summaries:
            warnings.warn("No performance summaries available for dashboard")
            return Dashboard(
                title="Performance Dashboard",
                subtitle="No Data Available",
                generated_at=datetime.now(),
                time_period=time_period,
                charts=[],
                summary_stats={},
                alerts=["No performance data available for analysis"]
            )

        # Create overview charts
        charts.append(self._create_performance_overview_chart(all_summaries))
        charts.append(self._create_stability_gauge_chart(all_summaries))

        # Create trend charts if requested
        if include_trends:
            charts.append(self._create_latency_trends_chart(all_summaries))
            charts.append(self._create_throughput_trends_chart(all_summaries))

        # Create anomaly charts if requested
        if include_anomalies:
            charts.append(self._create_anomaly_heatmap(all_summaries))

        # Create regression severity chart
        charts.append(self._create_model_health_chart(all_summaries))

        # Generate summary statistics
        summary_stats = self._calculate_dashboard_summary(all_summaries)

        # Create metadata
        metadata = {
            "generator": "DashboardGenerator",
            "version": self.version,
            "models_analyzed": len(all_summaries),
            "models_requested": len(models),
            "include_trends": include_trends,
            "include_anomalies": include_anomalies
        }

        return Dashboard(
            title="Performance Regression Dashboard",
            subtitle=f"Analysis of {len(all_summaries)} models over {time_period}",
            generated_at=datetime.now(),
            time_period=time_period,
            charts=charts,
            summary_stats=summary_stats,
            alerts=alerts,
            metadata=metadata
        )

    def create_regression_dashboard(
        self,
        regression_results: List[RegressionResult],
        title: str = "Regression Analysis Dashboard"
    ) -> Dashboard:
        """
        Create dashboard focused on regression analysis results.

        Args:
            regression_results: List of regression analysis results
            title: Dashboard title

        Returns:
            Dashboard with regression-focused visualizations
        """
        print(f"🔍 Creating regression dashboard for {len(regression_results)} results...")

        charts = []
        alerts = []

        # Regression severity distribution
        charts.append(self._create_regression_severity_chart(regression_results))

        # Performance delta scatter plot
        charts.append(self._create_performance_delta_scatter(regression_results))

        # Model comparison bar chart
        charts.append(self._create_model_comparison_chart(regression_results))

        # Time series of regressions (if timestamps available)
        charts.append(self._create_regression_timeline(regression_results))

        # Generate alerts from regression results
        blocking_count = sum(1 for r in regression_results if r.is_blocking())
        if blocking_count > 0:
            alerts.append(f"🚨 {blocking_count} blocking regressions detected!")

        critical_count = sum(1 for r in regression_results if r.severity == RegressionSeverity.CRITICAL)
        if critical_count > 0:
            alerts.append(f"🚨 {critical_count} critical regressions require immediate attention")

        # Summary statistics
        summary_stats = {
            "total_models": len(set(r.model_name for r in regression_results)),
            "total_regressions": sum(1 for r in regression_results if r.is_regression()),
            "blocking_regressions": blocking_count,
            "average_delta": statistics.mean([abs(r.performance_delta_percent) for r in regression_results if r.is_regression()]) if any(r.is_regression() for r in regression_results) else 0.0
        }

        return Dashboard(
            title=title,
            subtitle=f"Regression analysis of {len(set(r.model_name for r in regression_results))} models",
            generated_at=datetime.now(),
            time_period="Current Analysis",
            charts=charts,
            summary_stats=summary_stats,
            alerts=alerts,
            metadata={
                "generator": "DashboardGenerator",
                "version": self.version,
                "analysis_type": "regression"
            }
        )

    def export_dashboard(
        self,
        dashboard: Dashboard,
        output_path: str,
        format: str = "html"
    ) -> bool:
        """
        Export dashboard to file.

        Args:
            dashboard: Dashboard to export
            output_path: Output file path
            format: Export format ("html", "json")

        Returns:
            True if export was successful
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                with open(output_file, 'w') as f:
                    json.dump(dashboard.to_dict(), f, indent=2, default=str)

            elif format.lower() == "html":
                html_content = self._generate_html_dashboard(dashboard)
                with open(output_file, 'w') as f:
                    f.write(html_content)

            else:
                raise ValueError(f"Unsupported format: {format}")

            print(f"✅ Dashboard exported to {output_file} ({format})")
            return True

        except Exception as e:
            warnings.warn(f"Failed to export dashboard: {e}")
            return False

    def _create_performance_overview_chart(self, summaries: List[PerformanceSummary]) -> ChartData:
        """Create performance overview bar chart"""
        models = [s.model_name for s in summaries]
        latencies = [s.mean_latency_ms for s in summaries]
        throughputs = [s.mean_throughput for s in summaries]

        return ChartData(
            chart_id="performance_overview",
            title="Performance Overview",
            chart_type=ChartType.BAR,
            data={
                "labels": models,
                "datasets": [
                    {
                        "label": "Average Latency (ms)",
                        "data": latencies,
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "borderColor": "rgba(255, 99, 132, 1)",
                        "yAxisID": "y"
                    },
                    {
                        "label": "Average Throughput",
                        "data": throughputs,
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "yAxisID": "y1"
                    }
                ]
            },
            config={
                "responsive": True,
                "scales": {
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "title": {"display": True, "text": "Latency (ms)"}
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "title": {"display": True, "text": "Throughput"},
                        "grid": {"drawOnChartArea": False}
                    }
                }
            }
        )

    def _create_stability_gauge_chart(self, summaries: List[PerformanceSummary]) -> ChartData:
        """Create stability gauge chart"""
        avg_stability = statistics.mean([s.stability_score for s in summaries]) * 100

        return ChartData(
            chart_id="stability_gauge",
            title="Overall Stability Score",
            chart_type=ChartType.GAUGE,
            data={
                "datasets": [{
                    "data": [avg_stability],
                    "backgroundColor": [
                        "red" if avg_stability < 60 else
                        "orange" if avg_stability < 80 else
                        "green"
                    ]
                }]
            },
            config={
                "circumference": 180,
                "rotation": 270,
                "cutoutPercentage": 80,
                "plugins": {
                    "datalabels": {
                        "display": True,
                        "formatter": lambda value: f"{value:.1f}%"
                    }
                }
            }
        )

    def _create_latency_trends_chart(self, summaries: List[PerformanceSummary]) -> ChartData:
        """Create latency trends line chart"""
        models = []
        trend_data = []

        for summary in summaries:
            if summary.latency_trend and summary.latency_trend.raw_data:
                models.append(summary.model_name)
                # Extract time series data
                dates = [point[0].strftime('%Y-%m-%d') for point in summary.latency_trend.raw_data]
                values = [point[1] for point in summary.latency_trend.raw_data]
                trend_data.append({
                    "label": summary.model_name,
                    "data": [{"x": date, "y": value} for date, value in zip(dates, values)],
                    "fill": False,
                    "borderColor": f"hsl({hash(summary.model_name) % 360}, 70%, 50%)",
                    "tension": 0.1
                })

        return ChartData(
            chart_id="latency_trends",
            title="Latency Trends Over Time",
            chart_type=ChartType.LINE,
            data={
                "datasets": trend_data
            },
            config={
                "responsive": True,
                "scales": {
                    "x": {
                        "type": "time",
                        "time": {"unit": "day"},
                        "title": {"display": True, "text": "Date"}
                    },
                    "y": {
                        "title": {"display": True, "text": "Latency (ms)"}
                    }
                }
            }
        )

    def _create_throughput_trends_chart(self, summaries: List[PerformanceSummary]) -> ChartData:
        """Create throughput trends line chart"""
        models = []
        trend_data = []

        for summary in summaries:
            if summary.throughput_trend and summary.throughput_trend.raw_data:
                models.append(summary.model_name)
                dates = [point[0].strftime('%Y-%m-%d') for point in summary.throughput_trend.raw_data]
                values = [point[1] for point in summary.throughput_trend.raw_data]
                trend_data.append({
                    "label": summary.model_name,
                    "data": [{"x": date, "y": value} for date, value in zip(dates, values)],
                    "fill": False,
                    "borderColor": f"hsl({hash(summary.model_name) % 360}, 70%, 50%)",
                    "tension": 0.1
                })

        return ChartData(
            chart_id="throughput_trends",
            title="Throughput Trends Over Time",
            chart_type=ChartType.LINE,
            data={
                "datasets": trend_data
            },
            config={
                "responsive": True,
                "scales": {
                    "x": {
                        "type": "time",
                        "time": {"unit": "day"},
                        "title": {"display": True, "text": "Date"}
                    },
                    "y": {
                        "title": {"display": True, "text": "Throughput"}
                    }
                }
            }
        )

    def _create_anomaly_heatmap(self, summaries: List[PerformanceSummary]) -> ChartData:
        """Create anomaly count heatmap"""
        models = [s.model_name for s in summaries]
        anomaly_counts = [s.anomaly_count for s in summaries]

        # Create heatmap data
        data = []
        for i, (model, count) in enumerate(zip(models, anomaly_counts)):
            data.append({
                "x": model,
                "y": "Anomalies",
                "v": count
            })

        return ChartData(
            chart_id="anomaly_heatmap",
            title="Anomaly Count by Model",
            chart_type=ChartType.HEATMAP,
            data={
                "datasets": [{
                    "label": "Anomaly Count",
                    "data": data,
                    "backgroundColor": lambda ctx: f"rgba(255, 99, 132, {min(ctx.parsed.v / 10, 1)})"
                }]
            }
        )

    def _create_model_health_chart(self, summaries: List[PerformanceSummary]) -> ChartData:
        """Create model health comparison chart"""
        models = [s.model_name for s in summaries]
        stability_scores = [s.stability_score * 100 for s in summaries]
        reliability_scores = [s.reliability_score * 100 for s in summaries]

        return ChartData(
            chart_id="model_health",
            title="Model Health Comparison",
            chart_type=ChartType.BAR,
            data={
                "labels": models,
                "datasets": [
                    {
                        "label": "Stability Score (%)",
                        "data": stability_scores,
                        "backgroundColor": "rgba(75, 192, 192, 0.6)"
                    },
                    {
                        "label": "Reliability Score (%)",
                        "data": reliability_scores,
                        "backgroundColor": "rgba(153, 102, 255, 0.6)"
                    }
                ]
            },
            config={
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 100,
                        "title": {"display": True, "text": "Score (%)"}
                    }
                }
            }
        )

    def _create_regression_severity_chart(self, regression_results: List[RegressionResult]) -> ChartData:
        """Create regression severity distribution pie chart"""
        severity_counts = {}
        for severity in RegressionSeverity:
            severity_counts[severity.value] = sum(1 for r in regression_results if r.severity == severity)

        return ChartData(
            chart_id="regression_severity",
            title="Regression Severity Distribution",
            chart_type=ChartType.PIE,
            data={
                "labels": list(severity_counts.keys()),
                "datasets": [{
                    "data": list(severity_counts.values()),
                    "backgroundColor": [
                        "rgba(75, 192, 192, 0.6)",  # None - green
                        "rgba(255, 205, 86, 0.6)",  # Minor - yellow
                        "rgba(255, 99, 132, 0.6)",  # Major - red
                        "rgba(201, 203, 207, 0.6)"  # Critical - dark red
                    ]
                }]
            }
        )

    def _create_performance_delta_scatter(self, regression_results: List[RegressionResult]) -> ChartData:
        """Create performance delta scatter plot"""
        data = []
        for result in regression_results:
            data.append({
                "x": result.performance_delta_percent,
                "y": result.throughput_delta_percent,
                "label": result.model_name,
                "backgroundColor": self._get_severity_color(result.severity)
            })

        return ChartData(
            chart_id="performance_delta_scatter",
            title="Performance vs Throughput Delta",
            chart_type=ChartType.SCATTER,
            data={
                "datasets": [{
                    "label": "Performance Delta",
                    "data": data,
                    "backgroundColor": [point["backgroundColor"] for point in data]
                }]
            },
            config={
                "scales": {
                    "x": {"title": {"display": True, "text": "Performance Delta (%)"}},
                    "y": {"title": {"display": True, "text": "Throughput Delta (%)"}}
                }
            }
        )

    def _create_model_comparison_chart(self, regression_results: List[RegressionResult]) -> ChartData:
        """Create model comparison bar chart"""
        model_data = {}
        for result in regression_results:
            if result.model_name not in model_data:
                model_data[result.model_name] = {
                    "performance_delta": result.performance_delta_percent,
                    "severity": result.severity.value
                }

        models = list(model_data.keys())
        deltas = [model_data[model]["performance_delta"] for model in models]

        return ChartData(
            chart_id="model_comparison",
            title="Performance Delta by Model",
            chart_type=ChartType.BAR,
            data={
                "labels": models,
                "datasets": [{
                    "label": "Performance Delta (%)",
                    "data": deltas,
                    "backgroundColor": [
                        self._get_severity_color_by_value(delta) for delta in deltas
                    ]
                }]
            },
            config={
                "scales": {
                    "y": {"title": {"display": True, "text": "Performance Delta (%)"}}
                }
            }
        )

    def _create_regression_timeline(self, regression_results: List[RegressionResult]) -> ChartData:
        """Create regression timeline chart"""
        # Group by day and count regressions
        daily_counts = {}
        for result in regression_results:
            date_str = result.timestamp.strftime('%Y-%m-%d')
            if date_str not in daily_counts:
                daily_counts[date_str] = 0
            if result.is_regression():
                daily_counts[date_str] += 1

        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]

        return ChartData(
            chart_id="regression_timeline",
            title="Regressions Over Time",
            chart_type=ChartType.LINE,
            data={
                "labels": dates,
                "datasets": [{
                    "label": "Regression Count",
                    "data": counts,
                    "fill": False,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "tension": 0.1
                }]
            },
            config={
                "scales": {
                    "x": {"title": {"display": True, "text": "Date"}},
                    "y": {"title": {"display": True, "text": "Regression Count"}}
                }
            }
        )

    def _calculate_dashboard_summary(self, summaries: List[PerformanceSummary]) -> Dict[str, Any]:
        """Calculate summary statistics for dashboard"""
        if not summaries:
            return {}

        return {
            "total_models": len(summaries),
            "average_latency_ms": statistics.mean([s.mean_latency_ms for s in summaries if s.mean_latency_ms > 0]),
            "average_throughput": statistics.mean([s.mean_throughput for s in summaries if s.mean_throughput > 0]),
            "average_stability": statistics.mean([s.stability_score for s in summaries]),
            "average_reliability": statistics.mean([s.reliability_score for s in summaries]),
            "total_anomalies": sum([s.anomaly_count for s in summaries]),
            "models_with_issues": sum(1 for s in summaries if s.stability_score < 0.8 or s.reliability_score < 0.8)
        }

    def _get_severity_color(self, severity: RegressionSeverity) -> str:
        """Get color for severity level"""
        colors = {
            RegressionSeverity.NONE: "rgba(75, 192, 192, 0.6)",
            RegressionSeverity.MINOR: "rgba(255, 205, 86, 0.6)",
            RegressionSeverity.MAJOR: "rgba(255, 159, 64, 0.6)",
            RegressionSeverity.CRITICAL: "rgba(255, 99, 132, 0.6)"
        }
        return colors.get(severity, "rgba(201, 203, 207, 0.6)")

    def _get_severity_color_by_value(self, delta: float) -> str:
        """Get color based on delta value"""
        abs_delta = abs(delta)
        if abs_delta >= 10:
            return "rgba(255, 99, 132, 0.6)"  # Critical
        elif abs_delta >= 5:
            return "rgba(255, 159, 64, 0.6)"   # Major
        elif abs_delta >= 2:
            return "rgba(255, 205, 86, 0.6)"   # Minor
        else:
            return "rgba(75, 192, 192, 0.6)"   # None

    def _generate_html_dashboard(self, dashboard: Dashboard) -> str:
        """Generate HTML dashboard with charts"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{dashboard.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }}
        .header {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .alerts {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .alert {{ margin: 5px 0; }}
        canvas {{ max-height: 400px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{dashboard.title}</h1>
        <p>{dashboard.subtitle}</p>
        <p>Generated: {dashboard.generated_at.strftime('%Y-%m-%d %H:%M:%S')} | Period: {dashboard.time_period}</p>
    </div>

    {'<div class="alerts">' + ''.join(f'<div class="alert">{alert}</div>' for alert in dashboard.alerts) + '</div>' if dashboard.alerts else ''}

    <div class="summary">
        {''.join(f'''
        <div class="stat">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{key.replace('_', ' ').title()}</div>
        </div>
        ''' for key, value in dashboard.summary_stats.items())}
    </div>

    <div class="charts">
        {''.join(f'''
        <div class="chart-container">
            <h3>{chart.title}</h3>
            <canvas id="{chart.chart_id}"></canvas>
        </div>
        ''' for chart in dashboard.charts)}
    </div>

    <script>
        {''.join(f'''
        new Chart(document.getElementById('{chart.chart_id}'), {{
            type: '{chart.chart_type.value}',
            data: {json.dumps(chart.data)},
            options: {json.dumps(chart.config)}
        }});
        ''' for chart in dashboard.charts)}
    </script>
</body>
</html>
        """
        return html