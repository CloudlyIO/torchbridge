#!/usr/bin/env python3
"""
Historical Analysis & Reporting Demo (Phase 2)

Demonstrates the advanced historical analysis and reporting capabilities
including trend analysis, anomaly detection, and automated dashboard generation.

Usage:
    python historical_analysis_demo.py [--quick] [--generate-reports] [--export-dashboards]
"""

import argparse
import sys
import os
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime, timedelta

# Add src and root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Phase 2 regression testing framework
from benchmarks.regression.historical_analyzer import HistoricalAnalyzer, TrendDirection, AnomalyType
from benchmarks.regression.reporting import RegressionReporter, DashboardGenerator, ReportFormat
from benchmarks.regression.baseline_manager import BaselineManager
from benchmarks.regression.regression_detector import RegressionDetector, RegressionSeverity

# Import existing benchmark infrastructure
from benchmarks.framework.benchmark_runner import PerformanceMetrics


def create_comprehensive_historical_data(results_dir: Path, models: list, days_back: int = 60):
    """Create comprehensive historical data with realistic trends and anomalies"""
    print(f"📊 Creating {days_back} days of historical data for {len(models)} models...")

    results_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        base_latency = 10.0 + hash(model_name) % 5  # Different base performance per model
        base_throughput = 100.0 - hash(model_name) % 20
        base_memory = 256.0 + hash(model_name) % 128

        # Create performance trends
        trend_factor = 1.0
        if "degrading" in model_name:
            trend_factor = 1.002  # 0.2% degradation per day
        elif "improving" in model_name:
            trend_factor = 0.998  # 0.2% improvement per day

        for day_offset in range(days_back):
            timestamp = datetime.now() - timedelta(days=day_offset, hours=day_offset % 24)
            filename = f"{model_name}_inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

            # Apply trend
            daily_trend = trend_factor ** day_offset

            # Add realistic noise
            noise_factor = 1.0 + ((day_offset * 7 + hash(model_name)) % 100 - 50) / 1000  # ±5% noise

            # Add occasional anomalies
            anomaly_factor = 1.0
            if day_offset % 15 == 0:  # Anomaly every 15 days
                if "spike" in model_name:
                    anomaly_factor = 1.5  # 50% performance spike
                elif "unstable" in model_name:
                    anomaly_factor = 0.7 + (day_offset % 5) * 0.1  # Variable performance

            # Calculate final values
            final_latency = base_latency * daily_trend * noise_factor * anomaly_factor
            final_throughput = base_throughput / (daily_trend * noise_factor * anomaly_factor)
            final_memory = base_memory * (1.0 + day_offset * 0.001)  # Gradual memory increase

            result_data = {
                "config": {"name": model_name},
                "timestamp": timestamp.isoformat(),
                "results": {
                    "PyTorch Native": {
                        "latency_ms": final_latency,
                        "throughput_samples_per_sec": final_throughput,
                        "peak_memory_mb": final_memory,
                        "memory_efficiency": 0.85 - day_offset * 0.001,
                        "accuracy_loss": 0.01 + day_offset * 0.0001,
                        "statistical_significance": True,
                        "confidence_interval_95": [final_latency * 0.95, final_latency * 1.05]
                    }
                }
            }

            with open(results_dir / filename, 'w') as f:
                json.dump(result_data, f, indent=2)

    print(f"✅ Created {days_back * len(models)} historical benchmark files")


def demonstrate_historical_analysis(analyzer: HistoricalAnalyzer, models: list, results_dir: str):
    """Demonstrate historical analysis capabilities"""
    print("\n📈 HISTORICAL ANALYSIS DEMONSTRATION")
    print("=" * 60)

    for model_name in models[:3]:  # Analyze first 3 models for demo
        print(f"\n🔍 Analyzing {model_name}...")

        # Perform trend analysis
        trends = analyzer.analyze_performance_trends(
            model_name=model_name,
            days=30,
            results_dir=results_dir
        )

        print(f"  📊 Trend Analysis Results:")
        for metric_name, trend in trends.items():
            direction_emoji = {
                TrendDirection.IMPROVING: "📈",
                TrendDirection.DEGRADING: "📉",
                TrendDirection.STABLE: "➡️",
                TrendDirection.VOLATILE: "〰️"
            }

            emoji = direction_emoji.get(trend.trend_direction, "❓")
            print(f"    {emoji} {metric_name}: {trend.trend_direction.value} "
                  f"({trend.trend_strength:+.2f}% over {trend.time_period_days} days, "
                  f"confidence: {trend.confidence_score:.2f})")

            if trend.is_significant_trend():
                print(f"      ⚠️ Significant trend detected!")

        # Detect performance drift
        drift_analysis = analyzer.detect_performance_drift(
            model_name=model_name,
            window_days=10,
            drift_threshold=3.0,
            results_dir=results_dir
        )

        if "error" not in drift_analysis:
            print(f"  🔄 Performance Drift Analysis:")
            for metric_name, drift_data in drift_analysis.get("metrics", {}).items():
                if drift_data.get("is_significant"):
                    print(f"    ⚠️ {metric_name}: {drift_data['drift_percent']:+.1f}% drift "
                          f"({drift_data['direction']})")
                else:
                    print(f"    ✅ {metric_name}: Stable ({drift_data['drift_percent']:+.1f}%)")

        # Identify anomalies
        anomalies = analyzer.identify_performance_anomalies(
            model_name=model_name,
            days=30,
            sensitivity=2.0,
            results_dir=results_dir
        )

        if anomalies:
            print(f"  🚨 Detected {len(anomalies)} performance anomalies:")
            for anomaly in anomalies[:3]:  # Show top 3
                anomaly_emoji = {
                    AnomalyType.SPIKE: "🔺",
                    AnomalyType.DROP: "🔻",
                    AnomalyType.PLATEAU: "➖",
                    AnomalyType.OSCILLATION: "〜"
                }
                emoji = anomaly_emoji.get(anomaly.anomaly_type, "❓")
                print(f"    {emoji} {anomaly.detection_date.strftime('%m-%d')}: "
                      f"{anomaly.anomaly_type.value} in {anomaly.metric_name} "
                      f"({anomaly.deviation_percent:+.1f}%, severity: {anomaly.severity_score:.2f})")
        else:
            print(f"  ✅ No significant anomalies detected")


def demonstrate_performance_summaries(analyzer: HistoricalAnalyzer, models: list, results_dir: str):
    """Demonstrate comprehensive performance summaries"""
    print("\n📊 PERFORMANCE SUMMARY DEMONSTRATION")
    print("=" * 60)

    summaries = []

    for model_name in models:
        try:
            summary = analyzer.generate_performance_summary(
                model_name=model_name,
                time_range="30d"
            )
            summaries.append(summary)

            print(f"\n📈 {model_name} Performance Summary:")
            print(f"  Time Period: {summary.start_date.strftime('%Y-%m-%d')} to {summary.end_date.strftime('%Y-%m-%d')}")
            print(f"  Measurements: {summary.total_measurements}")
            print(f"  Mean Latency: {summary.mean_latency_ms:.2f}ms (volatility: {summary.latency_volatility:.1%})")
            print(f"  Mean Throughput: {summary.mean_throughput:.1f} (volatility: {summary.throughput_volatility:.1%})")
            print(f"  Mean Memory: {summary.mean_memory_mb:.1f}MB (volatility: {summary.memory_volatility:.1%})")
            print(f"  Stability Score: {summary.stability_score:.1%}")
            print(f"  Reliability Score: {summary.reliability_score:.1%}")
            print(f"  Anomalies: {summary.anomaly_count}")

            if summary.recommendations:
                print(f"  Recommendations:")
                for rec in summary.recommendations[:2]:  # Show top 2
                    print(f"    • {rec}")

        except Exception as e:
            print(f"  ❌ Failed to generate summary for {model_name}: {e}")

    return summaries


def demonstrate_reporting(reporter: RegressionReporter, summaries: list, temp_dir: Path):
    """Demonstrate automated reporting capabilities"""
    print("\n📄 AUTOMATED REPORTING DEMONSTRATION")
    print("=" * 60)

    # Create some regression results for demonstration
    regression_results = []
    detector = RegressionDetector()

    for summary in summaries[:3]:  # Use first 3 summaries
        # Simulate current performance with some regressions
        multiplier = 1.0 + len(regression_results) * 0.03  # Increasing regression

        current_metrics = PerformanceMetrics(
            latency_ms=summary.mean_latency_ms * multiplier,
            throughput_samples_per_sec=summary.mean_throughput / multiplier,
            peak_memory_mb=summary.mean_memory_mb * multiplier,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(summary.mean_latency_ms * 0.95, summary.mean_latency_ms * 1.05)
        )

        # Create a baseline from the summary
        baseline_metrics = type('BaselineMetrics', (), {
            'model_name': summary.model_name,
            'mean_latency_ms': summary.mean_latency_ms,
            'std_latency_ms': summary.mean_latency_ms * summary.latency_volatility,
            'mean_throughput': summary.mean_throughput,
            'std_throughput': summary.mean_throughput * summary.throughput_volatility,
            'mean_memory_mb': summary.mean_memory_mb,
            'std_memory_mb': summary.mean_memory_mb * summary.memory_volatility,
            'sample_count': summary.total_measurements,
            'confidence_interval_95': (summary.mean_latency_ms * 0.9, summary.mean_latency_ms * 1.1),
            'established_date': summary.start_date,
            'last_validated_date': summary.end_date
        })()

        result = detector.detect_regression(current_metrics, baseline_metrics)
        regression_results.append(result)

    # Generate comprehensive report
    print("📊 Generating comprehensive regression report...")
    report = reporter.generate_regression_report(
        regression_results=regression_results,
        time_period="Demo Analysis",
        include_historical=True,
        include_trends=True
    )

    print(f"  ✅ Generated report with {len(report.regression_results)} regression results")
    print(f"  📈 Included {len(report.performance_summaries)} performance summaries")
    print(f"  💡 {len(report.recommendations)} recommendations generated")

    # Generate CI summary
    print("\n🔄 Generating CI/CD integration summary...")
    ci_summary = reporter.export_ci_summary(
        regression_results=regression_results,
        execution_time_seconds=2.5
    )

    print(f"  Status: {ci_summary.overall_status}")
    print(f"  Blocking regressions: {ci_summary.blocking_regressions}")
    print(f"  Total regressions: {ci_summary.total_regressions}")
    print(f"  Models tested: {len(ci_summary.models_tested)}")

    # Generate executive summary
    print("\n👔 Generating executive summary...")
    exec_summary = reporter.generate_executive_summary(
        time_period="30d",
        models=[s.model_name for s in summaries]
    )

    print(f"  Performance Health: {exec_summary.performance_health}")
    print(f"  Key Metrics: {len(exec_summary.key_metrics)} tracked")
    print(f"  Action Items: {len(exec_summary.action_items)}")
    print(f"  Success Highlights: {len(exec_summary.success_highlights)}")

    # Export reports in multiple formats
    reports_dir = temp_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    print(f"\n💾 Exporting reports to {reports_dir}...")

    # Export main report
    formats_to_test = [ReportFormat.HTML, ReportFormat.JSON, ReportFormat.MARKDOWN]
    for fmt in formats_to_test:
        output_file = reports_dir / f"regression_report.{fmt.value}"
        success = reporter.export_report(report, str(output_file), fmt)
        if success:
            print(f"  ✅ Exported {fmt.value.upper()} report")
        else:
            print(f"  ❌ Failed to export {fmt.value.upper()} report")

    return report, ci_summary, exec_summary


def demonstrate_dashboards(dashboard_gen: DashboardGenerator, summaries: list, regression_results: list, temp_dir: Path):
    """Demonstrate interactive dashboard generation"""
    print("\n📊 DASHBOARD GENERATION DEMONSTRATION")
    print("=" * 60)

    # Create performance dashboard
    print("🎨 Creating comprehensive performance dashboard...")

    model_names = [s.model_name for s in summaries]

    performance_dashboard = dashboard_gen.create_performance_dashboard(
        models=model_names,
        time_period="30d",
        include_trends=True,
        include_anomalies=True
    )

    print(f"  ✅ Created dashboard with {len(performance_dashboard.charts)} charts")
    print(f"  📊 Summary stats: {len(performance_dashboard.summary_stats)} metrics")
    print(f"  ⚠️ Alerts: {len(performance_dashboard.alerts)}")

    # Create regression-focused dashboard
    print("\n🔍 Creating regression analysis dashboard...")

    regression_dashboard = dashboard_gen.create_regression_dashboard(
        regression_results=regression_results,
        title="Regression Analysis Dashboard"
    )

    print(f"  ✅ Created regression dashboard with {len(regression_dashboard.charts)} charts")
    print(f"  🎯 Focused on {len(set(r.model_name for r in regression_results))} models")

    # Export dashboards
    dashboards_dir = temp_dir / "dashboards"
    dashboards_dir.mkdir(exist_ok=True)

    print(f"\n💾 Exporting dashboards to {dashboards_dir}...")

    # Export performance dashboard
    perf_success = dashboard_gen.export_dashboard(
        performance_dashboard,
        str(dashboards_dir / "performance_dashboard.html"),
        format="html"
    )

    # Export regression dashboard
    reg_success = dashboard_gen.export_dashboard(
        regression_dashboard,
        str(dashboards_dir / "regression_dashboard.html"),
        format="html"
    )

    # Export JSON versions for programmatic access
    dashboard_gen.export_dashboard(
        performance_dashboard,
        str(dashboards_dir / "performance_dashboard.json"),
        format="json"
    )

    if perf_success and reg_success:
        print("  ✅ All dashboards exported successfully")
        print(f"  🌐 Open {dashboards_dir}/performance_dashboard.html in browser")
        print(f"  🌐 Open {dashboards_dir}/regression_dashboard.html in browser")
    else:
        print("  ⚠️ Some dashboard exports may have failed")

    return performance_dashboard, regression_dashboard


def main():
    parser = argparse.ArgumentParser(description="Historical Analysis & Reporting Demo (Phase 2)")
    parser.add_argument("--quick", action="store_true", help="Run quick demo with minimal data")
    parser.add_argument("--generate-reports", action="store_true", help="Generate and export reports")
    parser.add_argument("--export-dashboards", action="store_true", help="Export interactive dashboards")
    parser.add_argument("--days-back", type=int, default=60, help="Days of historical data to create")

    args = parser.parse_args()

    print("🎯 HISTORICAL ANALYSIS & REPORTING FRAMEWORK DEMO (PHASE 2)")
    print("=" * 70)
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print(f"Generate reports: {'ON' if args.generate_reports else 'OFF'}")
    print(f"Export dashboards: {'ON' if args.export_dashboards else 'OFF'}")

    # Create temporary directory for demo
    temp_dir = Path(tempfile.mkdtemp(prefix="historical_demo_"))

    try:
        # Define test models with different characteristics
        if args.quick:
            models = [
                "stable_model",
                "degrading_model",
                "unstable_model"
            ]
            days_back = 30
        else:
            models = [
                "stable_model",
                "degrading_model",
                "improving_model",
                "spike_model",
                "unstable_model",
                "memory_intensive_model"
            ]
            days_back = args.days_back

        # Phase 1: Create comprehensive historical data
        results_dir = temp_dir / "results"
        create_comprehensive_historical_data(results_dir, models, days_back)

        # Phase 2: Initialize analysis components
        print(f"\n🔧 Initializing Phase 2 components...")
        baseline_manager = BaselineManager(baselines_dir=str(temp_dir / "baselines"))
        analyzer = HistoricalAnalyzer(baseline_manager=baseline_manager)
        reporter = RegressionReporter(historical_analyzer=analyzer, baseline_manager=baseline_manager)
        dashboard_gen = DashboardGenerator(historical_analyzer=analyzer, baseline_manager=baseline_manager)

        # Phase 3: Demonstrate historical analysis
        demonstrate_historical_analysis(analyzer, models, str(results_dir))

        # Phase 4: Demonstrate performance summaries
        summaries = demonstrate_performance_summaries(analyzer, models, str(results_dir))

        # Phase 5: Demonstrate reporting (if requested)
        report = None
        ci_summary = None
        exec_summary = None
        regression_results = []

        if args.generate_reports or not args.quick:
            report, ci_summary, exec_summary = demonstrate_reporting(reporter, summaries, temp_dir)
            regression_results = report.regression_results

        # Phase 6: Demonstrate dashboards (if requested)
        if args.export_dashboards or not args.quick:
            if not regression_results:
                # Create minimal regression results for dashboard demo
                detector = RegressionDetector()
                for i, summary in enumerate(summaries[:3]):
                    current_metrics = PerformanceMetrics(
                        latency_ms=summary.mean_latency_ms * 1.05,
                        throughput_samples_per_sec=summary.mean_throughput * 0.95,
                        peak_memory_mb=summary.mean_memory_mb * 1.02,
                        memory_efficiency=0.85,
                        accuracy_loss=0.01,
                        statistical_significance=True,
                        confidence_interval_95=(10.0, 15.0)
                    )

                    baseline_metrics = type('BaselineMetrics', (), {
                        'model_name': summary.model_name,
                        'mean_latency_ms': summary.mean_latency_ms,
                        'std_latency_ms': 1.0,
                        'mean_throughput': summary.mean_throughput,
                        'std_throughput': 5.0,
                        'mean_memory_mb': summary.mean_memory_mb,
                        'std_memory_mb': 10.0,
                        'sample_count': 20,
                        'confidence_interval_95': (summary.mean_latency_ms * 0.9, summary.mean_latency_ms * 1.1),
                        'established_date': datetime.now(),
                        'last_validated_date': datetime.now()
                    })()

                    result = detector.detect_regression(current_metrics, baseline_metrics)
                    regression_results.append(result)

            performance_dashboard, regression_dashboard = demonstrate_dashboards(
                dashboard_gen, summaries, regression_results, temp_dir
            )

        # Final summary
        print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 Demo files created in: {temp_dir}")
        print(f"📊 Models analyzed: {len(models)}")
        print(f"📈 Performance summaries: {len(summaries)}")

        if args.generate_reports:
            print(f"📄 Reports generated: HTML, JSON, Markdown formats")
            print(f"🔄 CI summary status: {ci_summary.overall_status if ci_summary else 'Not generated'}")

        if args.export_dashboards:
            print(f"📊 Interactive dashboards: HTML and JSON formats")
            print(f"🌐 View dashboards by opening HTML files in browser")

        print(f"\n💡 Key Phase 2 Features Demonstrated:")
        print(f"   • Historical trend analysis with statistical confidence")
        print(f"   • Performance drift detection with configurable windows")
        print(f"   • Anomaly detection using statistical methods")
        print(f"   • Comprehensive performance summaries with recommendations")
        print(f"   • Multi-format automated report generation")
        print(f"   • Interactive dashboards with Chart.js visualizations")
        print(f"   • CI/CD integration summaries")
        print(f"   • Executive-level performance insights")

        if not args.quick:
            print(f"\n📋 Files generated:")
            print(f"   Historical data: {len(models) * days_back} benchmark files")
            if args.generate_reports:
                print(f"   Reports: regression_report.html/.json/.md")
            if args.export_dashboards:
                print(f"   Dashboards: performance_dashboard.html, regression_dashboard.html")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup (comment out to inspect generated files)
        if temp_dir.exists():
            print(f"\n🧹 Cleaned up temporary files from {temp_dir}")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()