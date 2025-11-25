"""
Communication Profiling and Analysis

Advanced profiling tools for communication operations:
- Performance monitoring and bottleneck identification
- Operation-specific profiling and analysis
- Optimization recommendations based on profiling data
- Comprehensive performance reporting
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)


class CommunicationProfiler:
    """
    Profiler for communication operations

    Features:
    - Performance monitoring
    - Bottleneck identification
    - Optimization recommendations
    """

    def __init__(self):
        self.operation_history: List[Dict] = []
        self.performance_stats: Dict[str, List[float]] = {}
        self.bottlenecks: List[Dict] = []

    @contextmanager
    def profile_operation(self, operation_type: str, participants: List[int]):
        """Context manager for profiling communication operations"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            duration_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)

            # Record operation
            operation_record = {
                'type': operation_type,
                'participants': participants,
                'duration_ms': duration_ms,
                'memory_delta_mb': memory_delta_mb,
                'timestamp': end_time
            }

            self.operation_history.append(operation_record)

            # Update statistics
            if operation_type not in self.performance_stats:
                self.performance_stats[operation_type] = []
            self.performance_stats[operation_type].append(duration_ms)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            return psutil.Process().memory_info().rss

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify communication bottlenecks"""
        bottlenecks = []

        for op_type, durations in self.performance_stats.items():
            if len(durations) < 5:  # Need enough samples
                continue

            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            p95_duration = np.percentile(durations, 95)

            # Identify if operation is consistently slow
            if p95_duration > avg_duration * 2:
                bottleneck = {
                    'operation_type': op_type,
                    'avg_duration_ms': avg_duration,
                    'p95_duration_ms': p95_duration,
                    'variability': std_duration / avg_duration,
                    'severity': 'high' if p95_duration > avg_duration * 3 else 'medium'
                }
                bottlenecks.append(bottleneck)

        self.bottlenecks = bottlenecks
        return bottlenecks

    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling data"""
        recommendations = []

        bottlenecks = self.identify_bottlenecks()

        for bottleneck in bottlenecks:
            op_type = bottleneck['operation_type']

            if bottleneck['variability'] > 0.5:
                recommendations.append(
                    f"High variability in {op_type} operations suggests network congestion. "
                    f"Consider using bandwidth-aware scheduling."
                )

            if bottleneck['p95_duration_ms'] > 100:
                recommendations.append(
                    f"{op_type} operations are slow. Consider compression or pattern optimization."
                )

        # General recommendations based on operation patterns
        if 'allreduce' in self.performance_stats and len(self.performance_stats['allreduce']) > 10:
            avg_allreduce = np.mean(self.performance_stats['allreduce'])
            if avg_allreduce > 50:
                recommendations.append(
                    "AllReduce operations are slow. Consider hierarchical patterns for large clusters."
                )

        return recommendations

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'total_operations': len(self.operation_history),
            'operation_breakdown': {},
            'performance_stats': {},
            'bottlenecks': self.identify_bottlenecks(),
            'recommendations': self.get_optimization_recommendations(),
            'report_timestamp': time.time()
        }

        # Operation breakdown
        for record in self.operation_history:
            op_type = record['type']
            if op_type not in report['operation_breakdown']:
                report['operation_breakdown'][op_type] = 0
            report['operation_breakdown'][op_type] += 1

        # Performance statistics
        for op_type, durations in self.performance_stats.items():
            report['performance_stats'][op_type] = {
                'count': len(durations),
                'avg_duration_ms': np.mean(durations),
                'min_duration_ms': np.min(durations),
                'max_duration_ms': np.max(durations),
                'p95_duration_ms': np.percentile(durations, 95),
                'std_duration_ms': np.std(durations)
            }

        return report

    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns over time"""
        if not self.operation_history:
            return {'error': 'No operation history available'}

        # Time-based analysis
        timestamps = [op['timestamp'] for op in self.operation_history]
        durations = [op['duration_ms'] for op in self.operation_history]

        time_span = max(timestamps) - min(timestamps)
        operations_per_second = len(self.operation_history) / max(time_span, 1.0)

        # Pattern analysis
        operation_types = [op['type'] for op in self.operation_history]
        type_counts = {}
        for op_type in operation_types:
            type_counts[op_type] = type_counts.get(op_type, 0) + 1

        # Identify hotspots (most active participants)
        participant_activity = {}
        for op in self.operation_history:
            for participant in op['participants']:
                participant_activity[participant] = participant_activity.get(participant, 0) + 1

        # Find temporal patterns
        temporal_patterns = self._analyze_temporal_patterns()

        return {
            'time_analysis': {
                'time_span_seconds': time_span,
                'operations_per_second': operations_per_second,
                'avg_duration_ms': np.mean(durations),
                'duration_trend': self._calculate_trend(durations)
            },
            'operation_patterns': {
                'most_common_type': max(type_counts, key=type_counts.get) if type_counts else None,
                'type_distribution': type_counts
            },
            'participant_analysis': {
                'most_active_participants': sorted(
                    participant_activity.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
                'total_unique_participants': len(participant_activity)
            },
            'temporal_patterns': temporal_patterns
        }

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in communications"""
        if len(self.operation_history) < 10:
            return {'insufficient_data': True}

        # Group operations by time windows
        window_size = 10  # seconds
        time_windows = {}

        min_time = min(op['timestamp'] for op in self.operation_history)

        for op in self.operation_history:
            window_idx = int((op['timestamp'] - min_time) // window_size)
            if window_idx not in time_windows:
                time_windows[window_idx] = []
            time_windows[window_idx].append(op)

        # Analyze each window
        window_stats = []
        for window_idx, ops in time_windows.items():
            window_duration_avg = np.mean([op['duration_ms'] for op in ops])
            window_count = len(ops)

            window_stats.append({
                'window_start': min_time + window_idx * window_size,
                'operation_count': window_count,
                'avg_duration_ms': window_duration_avg
            })

        # Identify patterns
        operation_counts = [w['operation_count'] for w in window_stats]
        avg_durations = [w['avg_duration_ms'] for w in window_stats]

        return {
            'window_size_seconds': window_size,
            'total_windows': len(window_stats),
            'avg_operations_per_window': np.mean(operation_counts),
            'operation_count_variance': np.var(operation_counts),
            'duration_stability': 1.0 / (1.0 + np.var(avg_durations) / np.mean(avg_durations)) if avg_durations else 0.0,
            'peak_activity_window': max(window_stats, key=lambda w: w['operation_count']) if window_stats else None
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return 'insufficient_data'

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if abs(slope) < 0.01:  # Very small slope
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'

    def get_memory_profile(self) -> Dict[str, Any]:
        """Get memory usage profile from operations"""
        if not self.operation_history:
            return {'error': 'No operation history available'}

        memory_deltas = [op['memory_delta_mb'] for op in self.operation_history]

        # Filter out outliers (operations with very large memory changes)
        q75, q25 = np.percentile(memory_deltas, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr

        normal_deltas = [delta for delta in memory_deltas if delta < outlier_threshold]

        return {
            'total_operations': len(self.operation_history),
            'memory_statistics': {
                'avg_delta_mb': np.mean(memory_deltas),
                'max_delta_mb': max(memory_deltas),
                'min_delta_mb': min(memory_deltas),
                'std_delta_mb': np.std(memory_deltas)
            },
            'memory_efficiency': {
                'operations_with_negative_delta': sum(1 for delta in memory_deltas if delta < 0),
                'operations_with_large_delta': sum(1 for delta in memory_deltas if delta > 10.0),
                'avg_normal_delta_mb': np.mean(normal_deltas) if normal_deltas else 0.0
            },
            'memory_trend': self._calculate_trend(memory_deltas)
        }

    def export_detailed_profile(self, filepath: str, include_history: bool = False):
        """Export detailed profiling data to file"""
        profile_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_operations': len(self.operation_history),
                'profiling_duration_seconds': (
                    max(op['timestamp'] for op in self.operation_history) -
                    min(op['timestamp'] for op in self.operation_history)
                ) if self.operation_history else 0
            },
            'performance_report': self.generate_performance_report(),
            'pattern_analysis': self.analyze_communication_patterns(),
            'memory_profile': self.get_memory_profile()
        }

        if include_history:
            profile_data['operation_history'] = self.operation_history

        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)

            logger.info(f"Exported detailed profile to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export profile to {filepath}: {e}")
            return False

    def clear_history(self, keep_recent_hours: float = 1.0):
        """Clear operation history, optionally keeping recent operations"""
        if keep_recent_hours <= 0:
            self.operation_history.clear()
            self.performance_stats.clear()
            self.bottlenecks.clear()
        else:
            cutoff_time = time.time() - (keep_recent_hours * 3600)

            # Keep recent operations
            self.operation_history = [
                op for op in self.operation_history
                if op['timestamp'] > cutoff_time
            ]

            # Rebuild performance stats from remaining history
            self.performance_stats.clear()
            for op in self.operation_history:
                op_type = op['type']
                if op_type not in self.performance_stats:
                    self.performance_stats[op_type] = []
                self.performance_stats[op_type].append(op['duration_ms'])

            # Clear bottlenecks (will be recalculated when needed)
            self.bottlenecks.clear()

    def compare_with_baseline(self, baseline_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current performance with baseline statistics"""
        current_stats = self.generate_performance_report()['performance_stats']

        comparison = {
            'timestamp': time.time(),
            'operations_compared': [],
            'overall_performance_change': 'unknown'
        }

        performance_changes = []

        for op_type in set(current_stats.keys()) | set(baseline_stats.keys()):
            if op_type in current_stats and op_type in baseline_stats:
                current_avg = current_stats[op_type]['avg_duration_ms']
                baseline_avg = baseline_stats[op_type]['avg_duration_ms']

                change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100
                performance_changes.append(change_percent)

                comparison['operations_compared'].append({
                    'operation_type': op_type,
                    'current_avg_ms': current_avg,
                    'baseline_avg_ms': baseline_avg,
                    'change_percent': change_percent,
                    'status': 'improved' if change_percent < -5 else 'degraded' if change_percent > 5 else 'stable'
                })
            elif op_type in current_stats:
                comparison['operations_compared'].append({
                    'operation_type': op_type,
                    'current_avg_ms': current_stats[op_type]['avg_duration_ms'],
                    'baseline_avg_ms': None,
                    'status': 'new_operation'
                })
            else:
                comparison['operations_compared'].append({
                    'operation_type': op_type,
                    'current_avg_ms': None,
                    'baseline_avg_ms': baseline_stats[op_type]['avg_duration_ms'],
                    'status': 'removed_operation'
                })

        # Overall performance assessment
        if performance_changes:
            avg_change = np.mean(performance_changes)
            if avg_change < -10:
                comparison['overall_performance_change'] = 'significantly_improved'
            elif avg_change < -2:
                comparison['overall_performance_change'] = 'improved'
            elif avg_change > 10:
                comparison['overall_performance_change'] = 'significantly_degraded'
            elif avg_change > 2:
                comparison['overall_performance_change'] = 'degraded'
            else:
                comparison['overall_performance_change'] = 'stable'

        return comparison