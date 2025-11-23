#!/usr/bin/env python3
"""
Benchmark Results Analysis and Visualization

Advanced analysis tools for processing benchmark results and generating
comprehensive performance reports with statistical validation.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import glob

@dataclass
class BenchmarkAnalysis:
    """Structured analysis results"""
    summary: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    performance_rankings: Dict[str, List[str]]
    scaling_analysis: Dict[str, Any]
    recommendations: List[str]

class ResultsAnalyzer:
    """
    Advanced benchmark results analyzer with statistical validation
    and comprehensive visualization capabilities.
    """

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = results_dir
        self.analysis_output_dir = os.path.join(results_dir, "analysis")
        os.makedirs(self.analysis_output_dir, exist_ok=True)

    def load_benchmark_results(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """Load all benchmark results matching pattern"""

        results_files = glob.glob(os.path.join(self.results_dir, pattern))
        results = []

        for file_path in results_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['file_path'] = file_path
                    results.append(data)
            except Exception as e:
                print(f"⚠️  Failed to load {file_path}: {e}")

        print(f"📊 Loaded {len(results)} benchmark result files")
        return results

    def create_performance_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create structured DataFrame from benchmark results"""

        rows = []

        for result_data in results:
            config = result_data.get('config', {})
            results_dict = result_data.get('results', {})

            benchmark_name = config.get('name', 'Unknown')
            benchmark_type = config.get('benchmark_type', 'Unknown')

            for impl_name, metrics in results_dict.items():
                if metrics:
                    row = {
                        'benchmark_name': benchmark_name,
                        'benchmark_type': benchmark_type,
                        'implementation': impl_name,
                        'latency_ms': metrics.get('latency_ms', 0),
                        'throughput_samples_per_sec': metrics.get('throughput_samples_per_sec', 0),
                        'peak_memory_mb': metrics.get('peak_memory_mb', 0),
                        'memory_efficiency': metrics.get('memory_efficiency', 1.0),
                        'accuracy_loss': metrics.get('accuracy_loss', 0),
                        'statistical_significance': metrics.get('statistical_significance', False),
                        'model_config': str(config.get('model_config', {}))
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def perform_comprehensive_analysis(self, results: List[Dict[str, Any]]) -> BenchmarkAnalysis:
        """Perform comprehensive analysis of benchmark results"""

        print("🔍 Performing Comprehensive Analysis...")

        df = self.create_performance_dataframe(results)

        if df.empty:
            return BenchmarkAnalysis(
                summary={"error": "No valid results found"},
                statistical_tests={},
                performance_rankings={},
                scaling_analysis={},
                recommendations=["No data available for analysis"]
            )

        # Summary statistics
        summary = self._generate_summary_statistics(df)

        # Statistical testing
        statistical_tests = self._perform_statistical_tests(df)

        # Performance rankings
        rankings = self._generate_performance_rankings(df)

        # Scaling analysis
        scaling = self._analyze_scaling_characteristics(df)

        # Generate recommendations
        recommendations = self._generate_recommendations(df, statistical_tests)

        return BenchmarkAnalysis(
            summary=summary,
            statistical_tests=statistical_tests,
            performance_rankings=rankings,
            scaling_analysis=scaling,
            recommendations=recommendations
        )

    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""

        summary = {
            "total_benchmarks": len(df['benchmark_name'].unique()),
            "total_implementations": len(df['implementation'].unique()),
            "implementations": list(df['implementation'].unique()),
            "benchmark_types": list(df['benchmark_type'].unique())
        }

        # Performance metrics by implementation
        impl_stats = df.groupby('implementation').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'throughput_samples_per_sec': ['mean', 'std', 'min', 'max'],
            'peak_memory_mb': ['mean', 'std', 'min', 'max']
        }).round(2)

        summary['performance_by_implementation'] = impl_stats.to_dict()

        # Find baseline (usually PyTorch Native)
        baseline_candidates = ['PyTorch Native', 'Baseline', 'pytorch_native']
        baseline = None
        for candidate in baseline_candidates:
            if candidate in df['implementation'].values:
                baseline = candidate
                break

        if baseline:
            baseline_data = df[df['implementation'] == baseline]
            baseline_avg_latency = baseline_data['latency_ms'].mean()

            # Calculate speedups relative to baseline
            speedups = {}
            for impl in df['implementation'].unique():
                if impl != baseline:
                    impl_data = df[df['implementation'] == impl]
                    impl_avg_latency = impl_data['latency_ms'].mean()
                    if impl_avg_latency > 0:
                        speedup = baseline_avg_latency / impl_avg_latency
                        speedups[impl] = round(speedup, 2)

            summary['speedups_vs_baseline'] = speedups
            summary['baseline'] = baseline

        return summary

    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance testing"""

        from scipy import stats

        tests = {}

        # Find baseline
        baseline_candidates = ['PyTorch Native', 'Baseline', 'pytorch_native']
        baseline = None
        for candidate in baseline_candidates:
            if candidate in df['implementation'].values:
                baseline = candidate
                break

        if not baseline:
            return {"error": "No baseline implementation found for statistical testing"}

        baseline_data = df[df['implementation'] == baseline]['latency_ms']

        for impl in df['implementation'].unique():
            if impl == baseline:
                continue

            impl_data = df[df['implementation'] == impl]['latency_ms']

            if len(impl_data) > 1 and len(baseline_data) > 1:
                # Welch's t-test (assumes unequal variances)
                t_stat, p_value = stats.ttest_ind(baseline_data, impl_data, equal_var=False)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(baseline_data) - 1) * baseline_data.std()**2 +
                                     (len(impl_data) - 1) * impl_data.std()**2) /
                                    (len(baseline_data) + len(impl_data) - 2))

                if pooled_std > 0:
                    cohens_d = (baseline_data.mean() - impl_data.mean()) / pooled_std
                else:
                    cohens_d = 0

                tests[impl] = {
                    'vs_baseline': baseline,
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'statistically_significant': p_value < 0.05,
                    'cohens_d': float(cohens_d),
                    'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
                    'baseline_mean': float(baseline_data.mean()),
                    'impl_mean': float(impl_data.mean()),
                    'speedup': float(baseline_data.mean() / impl_data.mean()) if impl_data.mean() > 0 else 0
                }

        return tests

    def _generate_performance_rankings(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate performance rankings across different metrics"""

        rankings = {}

        # Latency ranking (lower is better)
        latency_ranking = df.groupby('implementation')['latency_ms'].mean().sort_values().index.tolist()
        rankings['by_latency'] = latency_ranking

        # Throughput ranking (higher is better)
        throughput_ranking = df.groupby('implementation')['throughput_samples_per_sec'].mean().sort_values(ascending=False).index.tolist()
        rankings['by_throughput'] = throughput_ranking

        # Memory efficiency ranking (lower memory is better)
        memory_ranking = df.groupby('implementation')['peak_memory_mb'].mean().sort_values().index.tolist()
        rankings['by_memory'] = memory_ranking

        # Overall ranking (combination of metrics)
        # Normalize each metric and compute weighted score
        df_norm = df.copy()
        df_norm['latency_score'] = 1 / df_norm['latency_ms']  # Inverse for latency
        df_norm['throughput_score'] = df_norm['throughput_samples_per_sec']
        df_norm['memory_score'] = 1 / (df_norm['peak_memory_mb'] + 1)  # Inverse for memory

        # Normalize scores to 0-1 range
        for col in ['latency_score', 'throughput_score', 'memory_score']:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        # Weighted combination (can be adjusted)
        weights = {'latency': 0.5, 'throughput': 0.3, 'memory': 0.2}
        df_norm['overall_score'] = (weights['latency'] * df_norm['latency_score'] +
                                   weights['throughput'] * df_norm['throughput_score'] +
                                   weights['memory'] * df_norm['memory_score'])

        overall_ranking = df_norm.groupby('implementation')['overall_score'].mean().sort_values(ascending=False).index.tolist()
        rankings['overall'] = overall_ranking

        return rankings

    def _analyze_scaling_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze scaling characteristics across different model sizes"""

        scaling_analysis = {}

        # Group by implementation
        for impl in df['implementation'].unique():
            impl_data = df[df['implementation'] == impl]

            if len(impl_data) > 1:
                # Analyze relationship between model complexity and performance
                # This is simplified - in practice you'd extract model size info

                scaling_analysis[impl] = {
                    'latency_variance': float(impl_data['latency_ms'].var()),
                    'throughput_variance': float(impl_data['throughput_samples_per_sec'].var()),
                    'memory_variance': float(impl_data['peak_memory_mb'].var()),
                    'consistency_score': 1.0 / (1.0 + impl_data['latency_ms'].std())  # Higher is more consistent
                }

        return scaling_analysis

    def _generate_recommendations(self, df: pd.DataFrame, statistical_tests: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis"""

        recommendations = []

        # Find best performer overall
        if not df.empty:
            best_latency = df.loc[df['latency_ms'].idxmin(), 'implementation']
            recommendations.append(f"🏆 Best overall performance: {best_latency}")

        # Check for our optimizations
        our_impl_names = ['Our Optimizations', 'our_optimizations', 'OurOptimized']
        our_impl = None
        for name in our_impl_names:
            if name in df['implementation'].values:
                our_impl = name
                break

        if our_impl and our_impl in statistical_tests:
            test_result = statistical_tests[our_impl]
            if test_result['statistically_significant']:
                speedup = test_result['speedup']
                recommendations.append(f"🚀 Our optimizations achieve {speedup:.2f}x speedup with statistical significance")
            else:
                recommendations.append(f"⚠️  Our optimizations show improvement but not statistically significant")

        # Memory efficiency recommendations
        memory_efficient = df.loc[df['peak_memory_mb'].idxmin(), 'implementation']
        recommendations.append(f"💾 Most memory efficient: {memory_efficient}")

        # Check for any implementations with poor performance
        if len(df) > 1:
            worst_latency = df.loc[df['latency_ms'].idxmax(), 'implementation']
            if worst_latency != best_latency:
                recommendations.append(f"📉 Consider avoiding: {worst_latency} (poorest performance)")

        return recommendations

    def generate_visualization_plots(self, df: pd.DataFrame, analysis: BenchmarkAnalysis):
        """Generate comprehensive visualization plots"""

        print("📊 Generating Visualization Plots...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Performance comparison plot
        plt.figure(figsize=(12, 8))

        # Latency comparison
        plt.subplot(2, 2, 1)
        impl_latency = df.groupby('implementation')['latency_ms'].mean().sort_values()
        bars = plt.bar(range(len(impl_latency)), impl_latency.values)
        plt.title('Average Latency by Implementation')
        plt.ylabel('Latency (ms)')
        plt.xticks(range(len(impl_latency)), impl_latency.index, rotation=45)

        # Color our optimization differently
        for i, impl in enumerate(impl_latency.index):
            if 'Our' in impl or 'our' in impl.lower():
                bars[i].set_color('red')

        # Throughput comparison
        plt.subplot(2, 2, 2)
        impl_throughput = df.groupby('implementation')['throughput_samples_per_sec'].mean().sort_values(ascending=False)
        bars = plt.bar(range(len(impl_throughput)), impl_throughput.values)
        plt.title('Average Throughput by Implementation')
        plt.ylabel('Throughput (samples/sec)')
        plt.xticks(range(len(impl_throughput)), impl_throughput.index, rotation=45)

        # Memory usage
        plt.subplot(2, 2, 3)
        impl_memory = df.groupby('implementation')['peak_memory_mb'].mean().sort_values()
        bars = plt.bar(range(len(impl_memory)), impl_memory.values)
        plt.title('Average Peak Memory by Implementation')
        plt.ylabel('Peak Memory (MB)')
        plt.xticks(range(len(impl_memory)), impl_memory.index, rotation=45)

        # Speedup chart (if baseline available)
        plt.subplot(2, 2, 4)
        if 'speedups_vs_baseline' in analysis.summary:
            speedups = analysis.summary['speedups_vs_baseline']
            impls = list(speedups.keys())
            values = list(speedups.values())

            bars = plt.bar(range(len(values)), values)
            plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
            plt.title(f"Speedup vs {analysis.summary.get('baseline', 'Baseline')}")
            plt.ylabel('Speedup (x)')
            plt.xticks(range(len(impls)), impls, rotation=45)
            plt.legend()

            # Highlight significant improvements
            for i, speedup in enumerate(values):
                if speedup > 1.2:  # 20% improvement threshold
                    bars[i].set_color('green')
                elif speedup < 0.8:  # Performance regression
                    bars[i].set_color('red')

        plt.tight_layout()
        plot_path = os.path.join(self.analysis_output_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   💾 Performance comparison plot saved: {plot_path}")

    def generate_html_report(self, analysis: BenchmarkAnalysis) -> str:
        """Generate comprehensive HTML report"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PyTorch Optimization Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2E86AB; color: white; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .metric {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .recommendation {{ background: #d4edda; padding: 10px; margin: 5px 0; border-radius: 4px; border-left: 4px solid #28a745; }}
                .warning {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 4px; border-left: 4px solid #ffc107; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 PyTorch Optimization Benchmark Report</h1>
                <p>Generated: {timestamp}</p>
            </div>

            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metric">Total Benchmarks: {analysis.summary.get('total_benchmarks', 'N/A')}</div>
                <div class="metric">Implementations Tested: {analysis.summary.get('total_implementations', 'N/A')}</div>
                <div class="metric">Benchmark Types: {', '.join(analysis.summary.get('benchmark_types', []))}</div>
        """

        # Add speedup summary if available
        if 'speedups_vs_baseline' in analysis.summary:
            html_content += """
                <h3>🏆 Key Performance Results</h3>
                <table>
                    <tr><th>Implementation</th><th>Speedup vs Baseline</th><th>Significance</th></tr>
            """

            speedups = analysis.summary['speedups_vs_baseline']
            for impl, speedup in speedups.items():
                significance = "✅ Significant" if impl in analysis.statistical_tests and analysis.statistical_tests[impl]['statistically_significant'] else "📊 Measured"
                color = "green" if speedup > 1.2 else "red" if speedup < 0.8 else "black"
                html_content += f'<tr><td>{impl}</td><td style="color: {color}">{speedup:.2f}x</td><td>{significance}</td></tr>'

            html_content += "</table>"

        # Add recommendations
        html_content += """
            </div>
            <div class="section">
                <h2>💡 Recommendations</h2>
        """

        for rec in analysis.recommendations:
            html_content += f'<div class="recommendation">{rec}</div>'

        # Add performance rankings
        html_content += """
            </div>
            <div class="section">
                <h2>🏅 Performance Rankings</h2>
        """

        for metric, ranking in analysis.performance_rankings.items():
            html_content += f"<h3>By {metric.replace('_', ' ').title()}</h3><ol>"
            for impl in ranking:
                html_content += f"<li>{impl}</li>"
            html_content += "</ol>"

        html_content += """
            </div>
        </body>
        </html>
        """

        # Save report
        report_path = os.path.join(self.analysis_output_dir, f'benchmark_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"   💾 HTML report saved: {report_path}")
        return report_path

def main():
    """Main analysis execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Results Analysis")
    parser.add_argument("--results-dir", default="benchmarks/results", help="Results directory")
    parser.add_argument("--pattern", default="*.json", help="File pattern to match")
    parser.add_argument("--generate-plots", action="store_true", help="Generate visualization plots")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML report")

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.results_dir)

    # Load results
    results = analyzer.load_benchmark_results(args.pattern)

    if not results:
        print("❌ No benchmark results found!")
        return

    # Perform analysis
    analysis = analyzer.perform_comprehensive_analysis(results)

    # Display key findings
    print(f"\n🎯 Analysis Summary:")
    print(f"   Benchmarks analyzed: {analysis.summary.get('total_benchmarks', 'N/A')}")
    print(f"   Implementations tested: {analysis.summary.get('total_implementations', 'N/A')}")

    if 'speedups_vs_baseline' in analysis.summary:
        print(f"\n🏆 Top Performance Improvements:")
        speedups = analysis.summary['speedups_vs_baseline']
        top_speedups = sorted(speedups.items(), key=lambda x: x[1], reverse=True)[:3]

        for impl, speedup in top_speedups:
            significance = "✅" if impl in analysis.statistical_tests and analysis.statistical_tests[impl]['statistically_significant'] else "📊"
            print(f"   {impl}: {speedup:.2f}x {significance}")

    print(f"\n💡 Key Recommendations:")
    for rec in analysis.recommendations[:3]:
        print(f"   • {rec}")

    # Generate outputs
    if args.generate_plots:
        df = analyzer.create_performance_dataframe(results)
        analyzer.generate_visualization_plots(df, analysis)

    if args.generate_report:
        analyzer.generate_html_report(analysis)

if __name__ == "__main__":
    main()