#!/usr/bin/env python3
"""
Regression Reporting Engine

Generates comprehensive reports for performance regression testing results,
including human-readable summaries, CI integration reports, and executive dashboards.
"""

import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union, Tuple
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


class ReportFormat(Enum):
    """Supported report formats"""
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    TEXT = "text"


@dataclass
class Report:
    """Comprehensive regression testing report"""
    title: str
    generated_at: datetime
    time_period: str
    summary: Dict[str, Any]
    regression_results: List[RegressionResult]
    performance_summaries: List[PerformanceSummary]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "time_period": self.time_period,
            "summary": self.summary,
            "regression_results": [r.to_dict() for r in self.regression_results],
            "performance_summaries": [self._summary_to_dict(s) for s in self.performance_summaries],
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }

    def _summary_to_dict(self, summary: PerformanceSummary) -> Dict[str, Any]:
        """Convert PerformanceSummary to dictionary"""
        return {
            "model_name": summary.model_name,
            "time_period": summary.time_period,
            "start_date": summary.start_date.isoformat(),
            "end_date": summary.end_date.isoformat(),
            "mean_latency_ms": summary.mean_latency_ms,
            "mean_throughput": summary.mean_throughput,
            "mean_memory_mb": summary.mean_memory_mb,
            "total_measurements": summary.total_measurements,
            "anomaly_count": summary.anomaly_count,
            "stability_score": summary.stability_score,
            "reliability_score": summary.reliability_score,
            "recommendations": summary.recommendations
        }


@dataclass
class CISummary:
    """Compact summary for CI/CD integration"""
    overall_status: str  # "PASS", "WARNING", "FAIL"
    blocking_regressions: int
    total_regressions: int
    models_tested: List[str]
    recommendations: List[str]
    execution_time_seconds: float
    generated_at: datetime

    def is_passing(self) -> bool:
        """Check if CI should pass"""
        return self.overall_status == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "blocking_regressions": self.blocking_regressions,
            "total_regressions": self.total_regressions,
            "models_tested": self.models_tested,
            "recommendations": self.recommendations,
            "execution_time_seconds": self.execution_time_seconds,
            "generated_at": self.generated_at.isoformat()
        }


@dataclass
class ExecutiveSummary:
    """High-level executive summary"""
    period: str
    performance_health: str  # "EXCELLENT", "GOOD", "CONCERNING", "CRITICAL"
    key_metrics: Dict[str, float]
    trends: Dict[str, str]
    action_items: List[str]
    success_highlights: List[str]
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "performance_health": self.performance_health,
            "key_metrics": self.key_metrics,
            "trends": self.trends,
            "action_items": self.action_items,
            "success_highlights": self.success_highlights,
            "generated_at": self.generated_at.isoformat()
        }


class RegressionReporter:
    """
    Generates comprehensive regression testing reports.

    Provides functionality to:
    - Generate human-readable regression reports
    - Create CI/CD integration summaries
    - Produce executive dashboards
    - Export reports in multiple formats
    - Track performance trends over time
    """

    def __init__(self, historical_analyzer: HistoricalAnalyzer = None, baseline_manager: BaselineManager = None):
        self.historical_analyzer = historical_analyzer or HistoricalAnalyzer()
        self.baseline_manager = baseline_manager or BaselineManager()
        self.version = "0.1.59"

    def generate_regression_report(
        self,
        regression_results: List[RegressionResult],
        time_period: str = "Current",
        include_historical: bool = True,
        include_trends: bool = True
    ) -> Report:
        """
        Generate comprehensive regression report.

        Args:
            regression_results: List of regression analysis results
            time_period: Time period for the report
            include_historical: Include historical performance analysis
            include_trends: Include trend analysis

        Returns:
            Comprehensive regression report
        """
        print(f"📊 Generating regression report for {len(regression_results)} results...")

        # Generate summary statistics
        summary = self._generate_summary_statistics(regression_results)

        # Get performance summaries for each model if requested
        performance_summaries = []
        if include_historical:
            model_names = list(set(r.model_name for r in regression_results))
            for model_name in model_names:
                try:
                    perf_summary = self.historical_analyzer.generate_performance_summary(
                        model_name, time_range="30d"
                    )
                    performance_summaries.append(perf_summary)
                except Exception as e:
                    warnings.warn(f"Failed to generate performance summary for {model_name}: {e}")

        # Generate overall recommendations
        recommendations = self._generate_overall_recommendations(
            regression_results, performance_summaries
        )

        # Create metadata
        metadata = {
            "generator": "RegressionReporter",
            "version": self.version,
            "include_historical": include_historical,
            "include_trends": include_trends,
            "models_analyzed": list(set(r.model_name for r in regression_results))
        }

        return Report(
            title=f"Performance Regression Report - {time_period}",
            generated_at=datetime.now(),
            time_period=time_period,
            summary=summary,
            regression_results=regression_results,
            performance_summaries=performance_summaries,
            recommendations=recommendations,
            metadata=metadata
        )

    def export_ci_summary(
        self,
        regression_results: List[RegressionResult],
        execution_time_seconds: float = 0.0
    ) -> CISummary:
        """
        Export compact CI/CD summary.

        Args:
            regression_results: Regression analysis results
            execution_time_seconds: Time taken to run analysis

        Returns:
            CISummary for CI/CD integration
        """
        print(f"🔄 Generating CI summary for {len(regression_results)} results...")

        # Count regressions by severity
        blocking_regressions = sum(1 for r in regression_results if r.is_blocking())
        total_regressions = sum(1 for r in regression_results if r.is_regression())

        # Determine overall status
        if blocking_regressions > 0:
            overall_status = "FAIL"
        elif total_regressions > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        # Generate CI-specific recommendations
        recommendations = []
        if blocking_regressions > 0:
            recommendations.append(f"🚫 {blocking_regressions} blocking regressions detected - review required before merge")
        if total_regressions > blocking_regressions:
            non_blocking = total_regressions - blocking_regressions
            recommendations.append(f"⚠️ {non_blocking} non-blocking regressions detected - monitor closely")
        if total_regressions == 0:
            recommendations.append("✅ No performance regressions detected - safe to merge")

        models_tested = list(set(r.model_name for r in regression_results))

        return CISummary(
            overall_status=overall_status,
            blocking_regressions=blocking_regressions,
            total_regressions=total_regressions,
            models_tested=models_tested,
            recommendations=recommendations,
            execution_time_seconds=execution_time_seconds,
            generated_at=datetime.now()
        )

    def generate_executive_summary(
        self,
        time_period: str = "30d",
        models: Optional[List[str]] = None
    ) -> ExecutiveSummary:
        """
        Generate executive-level performance summary.

        Args:
            time_period: Time period for analysis
            models: List of models to analyze (None for all)

        Returns:
            ExecutiveSummary for leadership consumption
        """
        print(f"📈 Generating executive summary for {time_period}...")

        if models is None:
            models = self.baseline_manager.list_available_models()

        if not models:
            warnings.warn("No models available for executive summary")
            return ExecutiveSummary(
                period=time_period,
                performance_health="UNKNOWN",
                key_metrics={},
                trends={},
                action_items=["Establish performance baselines for models"],
                success_highlights=[],
                generated_at=datetime.now()
            )

        # Gather performance summaries
        performance_summaries = []
        for model_name in models:
            try:
                summary = self.historical_analyzer.generate_performance_summary(
                    model_name, time_range=time_period
                )
                performance_summaries.append(summary)
            except Exception as e:
                warnings.warn(f"Failed to analyze {model_name}: {e}")

        if not performance_summaries:
            return ExecutiveSummary(
                period=time_period,
                performance_health="UNKNOWN",
                key_metrics={},
                trends={},
                action_items=["Unable to analyze model performance - check data availability"],
                success_highlights=[],
                generated_at=datetime.now()
            )

        # Calculate key metrics
        key_metrics = self._calculate_executive_metrics(performance_summaries)

        # Determine overall health
        health = self._determine_performance_health(performance_summaries)

        # Extract trends
        trends = self._extract_executive_trends(performance_summaries)

        # Generate action items and highlights
        action_items, success_highlights = self._generate_executive_insights(performance_summaries)

        return ExecutiveSummary(
            period=time_period,
            performance_health=health,
            key_metrics=key_metrics,
            trends=trends,
            action_items=action_items,
            success_highlights=success_highlights,
            generated_at=datetime.now()
        )

    def export_report(
        self,
        report: Report,
        output_path: str,
        format: ReportFormat = ReportFormat.HTML
    ) -> bool:
        """
        Export report to file in specified format.

        Args:
            report: Report to export
            output_path: Output file path
            format: Export format

        Returns:
            True if export was successful
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if format == ReportFormat.JSON:
                with open(output_file, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2, default=str)

            elif format == ReportFormat.HTML:
                html_content = self._generate_html_report(report)
                with open(output_file, 'w') as f:
                    f.write(html_content)

            elif format == ReportFormat.MARKDOWN:
                markdown_content = self._generate_markdown_report(report)
                with open(output_file, 'w') as f:
                    f.write(markdown_content)

            elif format == ReportFormat.TEXT:
                text_content = self._generate_text_report(report)
                with open(output_file, 'w') as f:
                    f.write(text_content)

            elif format == ReportFormat.CSV:
                self._export_csv_report(report, output_file)

            else:
                raise ValueError(f"Unsupported format: {format}")

            print(f"✅ Report exported to {output_file} ({format.value})")
            return True

        except Exception as e:
            warnings.warn(f"Failed to export report: {e}")
            return False

    def _generate_summary_statistics(self, regression_results: List[RegressionResult]) -> Dict[str, Any]:
        """Generate summary statistics from regression results"""
        if not regression_results:
            return {
                "total_models": 0,
                "total_regressions": 0,
                "severity_breakdown": {},
                "average_performance_delta": 0.0,
                "models_with_regressions": []
            }

        # Count by severity
        severity_counts = {}
        for severity in RegressionSeverity:
            severity_counts[severity.value] = sum(
                1 for r in regression_results if r.severity == severity
            )

        # Calculate average performance delta
        deltas = [abs(r.performance_delta_percent) for r in regression_results if r.is_regression()]
        avg_delta = statistics.mean(deltas) if deltas else 0.0

        # Identify models with regressions
        models_with_regressions = list(set(
            r.model_name for r in regression_results if r.is_regression()
        ))

        return {
            "total_models": len(set(r.model_name for r in regression_results)),
            "total_regressions": sum(1 for r in regression_results if r.is_regression()),
            "blocking_regressions": sum(1 for r in regression_results if r.is_blocking()),
            "severity_breakdown": severity_counts,
            "average_performance_delta": avg_delta,
            "models_with_regressions": models_with_regressions,
            "statistical_significance_rate": sum(1 for r in regression_results if r.statistical_significance) / len(regression_results) * 100
        }

    def _generate_overall_recommendations(
        self,
        regression_results: List[RegressionResult],
        performance_summaries: List[PerformanceSummary]
    ) -> List[str]:
        """Generate overall recommendations from analysis results"""
        recommendations = []

        # Regression-based recommendations
        blocking_count = sum(1 for r in regression_results if r.is_blocking())
        if blocking_count > 0:
            recommendations.append(f"🚨 IMMEDIATE ACTION: {blocking_count} blocking regressions require investigation before deployment")

        # Performance summary recommendations
        if performance_summaries:
            low_stability_models = [s.model_name for s in performance_summaries if s.stability_score < 0.7]
            if low_stability_models:
                recommendations.append(f"⚠️ Performance stability issues detected in: {', '.join(low_stability_models)}")

            high_anomaly_models = [s.model_name for s in performance_summaries if s.anomaly_count > 3]
            if high_anomaly_models:
                recommendations.append(f"🔍 High anomaly count in: {', '.join(high_anomaly_models)} - investigate environmental factors")

        # Positive recommendations
        stable_models = [s.model_name for s in performance_summaries if s.stability_score > 0.9]
        if stable_models:
            recommendations.append(f"✅ Excellent performance stability: {', '.join(stable_models)}")

        if not recommendations:
            recommendations.append("✅ No significant performance issues detected")

        return recommendations

    def _calculate_executive_metrics(self, summaries: List[PerformanceSummary]) -> Dict[str, float]:
        """Calculate key metrics for executive summary"""
        if not summaries:
            return {}

        # Average performance metrics
        avg_latency = statistics.mean([s.mean_latency_ms for s in summaries if s.mean_latency_ms > 0])
        avg_throughput = statistics.mean([s.mean_throughput for s in summaries if s.mean_throughput > 0])
        avg_stability = statistics.mean([s.stability_score for s in summaries])
        avg_reliability = statistics.mean([s.reliability_score for s in summaries])

        # Count models with issues
        models_with_issues = sum(1 for s in summaries if s.stability_score < 0.8 or s.reliability_score < 0.8)
        total_models = len(summaries)

        return {
            "average_latency_ms": avg_latency,
            "average_throughput": avg_throughput,
            "overall_stability_score": avg_stability,
            "overall_reliability_score": avg_reliability,
            "models_healthy_percent": ((total_models - models_with_issues) / total_models * 100) if total_models > 0 else 0,
            "total_models_tracked": total_models
        }

    def _determine_performance_health(self, summaries: List[PerformanceSummary]) -> str:
        """Determine overall performance health level"""
        if not summaries:
            return "UNKNOWN"

        # Calculate health score based on stability and reliability
        health_scores = []
        for summary in summaries:
            # Weight stability and reliability equally
            health_score = (summary.stability_score + summary.reliability_score) / 2
            health_scores.append(health_score)

        avg_health = statistics.mean(health_scores)

        if avg_health >= 0.9:
            return "EXCELLENT"
        elif avg_health >= 0.75:
            return "GOOD"
        elif avg_health >= 0.6:
            return "CONCERNING"
        else:
            return "CRITICAL"

    def _extract_executive_trends(self, summaries: List[PerformanceSummary]) -> Dict[str, str]:
        """Extract trend information for executive summary"""
        trends = {}

        # Count trend directions
        latency_trends = []
        throughput_trends = []

        for summary in summaries:
            if summary.latency_trend and summary.latency_trend.is_significant_trend():
                latency_trends.append(summary.latency_trend.trend_direction.value)
            if summary.throughput_trend and summary.throughput_trend.is_significant_trend():
                throughput_trends.append(summary.throughput_trend.trend_direction.value)

        # Determine overall trend patterns
        if latency_trends:
            most_common_latency = max(set(latency_trends), key=latency_trends.count)
            trends["latency"] = f"Mostly {most_common_latency} across models"

        if throughput_trends:
            most_common_throughput = max(set(throughput_trends), key=throughput_trends.count)
            trends["throughput"] = f"Mostly {most_common_throughput} across models"

        if not trends:
            trends["overall"] = "Performance trends are stable"

        return trends

    def _generate_executive_insights(self, summaries: List[PerformanceSummary]) -> Tuple[List[str], List[str]]:
        """Generate action items and success highlights for executives"""
        action_items = []
        success_highlights = []

        # Analyze each summary for insights
        critical_models = []
        stable_models = []

        for summary in summaries:
            if summary.stability_score < 0.6 or summary.reliability_score < 0.6:
                critical_models.append(summary.model_name)
            elif summary.stability_score > 0.9 and summary.reliability_score > 0.9:
                stable_models.append(summary.model_name)

        # Generate action items
        if critical_models:
            action_items.append(f"Investigate performance issues in critical models: {', '.join(critical_models)}")

        # Generate success highlights
        if stable_models:
            success_highlights.append(f"Models showing excellent performance: {', '.join(stable_models)}")

        # Overall insights
        if len(stable_models) > len(critical_models):
            success_highlights.append("Overall performance trend is positive with more stable than problematic models")
        elif critical_models:
            action_items.append("Performance reliability requires attention across multiple models")

        # Default items
        if not action_items:
            action_items.append("Continue monitoring performance trends")
        if not success_highlights:
            success_highlights.append("Performance monitoring system is operational")

        return action_items, success_highlights

    def _generate_html_report(self, report: Report) -> str:
        """Generate HTML formatted report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        .summary {{ background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        .regression {{ background: #fff5f5; padding: 10px; margin: 10px 0; border-left: 4px solid #ff6b6b; }}
        .recommendation {{ background: #f0f8ff; padding: 10px; margin: 5px 0; border-left: 4px solid #4dabf7; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report.title}</h1>
        <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Period: {report.time_period}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">Total Models: {report.summary.get('total_models', 0)}</div>
        <div class="metric">Regressions: {report.summary.get('total_regressions', 0)}</div>
        <div class="metric">Blocking: {report.summary.get('blocking_regressions', 0)}</div>
        <div class="metric">Avg Delta: {report.summary.get('average_performance_delta', 0):.2f}%</div>
    </div>

    <h2>Recommendations</h2>
    {"".join(f'<div class="recommendation">{rec}</div>' for rec in report.recommendations)}

    <h2>Regression Results</h2>
    {"".join(f'''
    <div class="regression">
        <strong>{r.model_name}</strong> - {r.severity.value.upper()}<br>
        Performance: {r.performance_delta_percent:+.1f}%<br>
        {r.recommendation}
    </div>
    ''' for r in report.regression_results if r.is_regression())}

</body>
</html>
        """
        return html

    def _generate_markdown_report(self, report: Report) -> str:
        """Generate Markdown formatted report"""
        # Build recommendations section
        recommendations_section = ""
        for rec in report.recommendations:
            recommendations_section += f"- {rec}\n"

        md = f"""# {report.title}

**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {report.time_period}

## Summary

- **Total Models:** {report.summary.get('total_models', 0)}
- **Regressions:** {report.summary.get('total_regressions', 0)}
- **Blocking:** {report.summary.get('blocking_regressions', 0)}
- **Average Delta:** {report.summary.get('average_performance_delta', 0):.2f}%

## Recommendations

{recommendations_section}

## Regression Results

| Model | Severity | Performance Delta | Recommendation |
|-------|----------|------------------|----------------|
"""

        # Add regression results
        for r in report.regression_results:
            if r.is_regression():
                md += f"| {r.model_name} | {r.severity.value.upper()} | {r.performance_delta_percent:+.1f}% | {r.recommendation[:50]}... |\n"

        md += """

## Performance Summaries

"""

        # Add performance summaries
        for s in report.performance_summaries:
            md += f"""
### {s.model_name}
- **Stability Score:** {s.stability_score:.2%}
- **Reliability Score:** {s.reliability_score:.2%}
- **Total Measurements:** {s.total_measurements}
- **Anomalies:** {s.anomaly_count}
"""

        return md

    def _generate_text_report(self, report: Report) -> str:
        """Generate plain text report"""
        title_separator = '=' * len(report.title)

        text = f"""{report.title}
{title_separator}

Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Period: {report.time_period}

SUMMARY
-------
Total Models: {report.summary.get('total_models', 0)}
Regressions: {report.summary.get('total_regressions', 0)}
Blocking: {report.summary.get('blocking_regressions', 0)}
Average Delta: {report.summary.get('average_performance_delta', 0):.2f}%

RECOMMENDATIONS
---------------
"""

        # Add recommendations
        for rec in report.recommendations:
            text += f"• {rec}\n"

        text += """

REGRESSION RESULTS
------------------
"""

        # Add regression results
        for r in report.regression_results:
            if r.is_regression():
                text += f"{r.model_name} ({r.severity.value.upper()}): {r.performance_delta_percent:+.1f}%\n"
                text += f"  {r.recommendation}\n\n"

        return text

    def _export_csv_report(self, report: Report, output_file: Path):
        """Export report data to CSV format"""
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(['Model', 'Severity', 'Performance_Delta_Percent', 'Throughput_Delta_Percent',
                           'Memory_Delta_Percent', 'Statistical_Significance', 'Blocking', 'Recommendation'])

            # Data rows
            for r in report.regression_results:
                writer.writerow([
                    r.model_name,
                    r.severity.value,
                    r.performance_delta_percent,
                    r.throughput_delta_percent,
                    r.memory_delta_percent,
                    r.statistical_significance,
                    r.is_blocking(),
                    r.recommendation
                ])