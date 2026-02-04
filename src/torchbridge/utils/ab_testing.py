"""
Hardware A/B Testing Framework

Advanced A/B testing system for comparing model performance across different
hardware configurations with statistical rigor and real-time analysis.
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from ..hardware.abstraction.hal_core import DeviceSpec

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """A/B experiment status"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TrafficAllocationStrategy(Enum):
    """Traffic allocation strategies"""
    FIXED_SPLIT = "fixed_split"
    GRADUAL_RAMP = "gradual_ramp"
    ADAPTIVE = "adaptive"
    MULTI_ARM_BANDIT = "multi_arm_bandit"


@dataclass
class HardwareConfig:
    """Hardware configuration for experiments"""
    config_id: str
    devices: list[DeviceSpec]
    optimization_flags: dict[str, Any] = field(default_factory=dict)
    memory_allocation_gb: float = 0.0
    expected_latency_ms: float = 0.0
    cost_per_hour: float = 0.0


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment"""
    latency_ms: list[float] = field(default_factory=list)
    throughput_rps: list[float] = field(default_factory=list)
    gpu_utilization: list[float] = field(default_factory=list)
    memory_usage_gb: list[float] = field(default_factory=list)
    error_rate: list[float] = field(default_factory=list)
    cost_per_request: list[float] = field(default_factory=list)
    accuracy_metrics: dict[str, list[float]] = field(default_factory=dict)
    custom_metrics: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class StatisticalSignificance:
    """Statistical significance results"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    relative_difference: float
    p_value: float
    confidence_interval_95: tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    effect_size: float
    statistical_power: float


class ABExperiment:
    """
    A/B experiment comparing hardware configurations

    Supports multiple hardware variants with statistical testing
    and real-time monitoring of performance differences.
    """

    def __init__(self,
                 experiment_id: str,
                 control_config: HardwareConfig,
                 treatment_configs: list[HardwareConfig],
                 traffic_allocation: dict[str, float],
                 duration_hours: float = 24.0,
                 min_samples_per_variant: int = 1000):
        self.experiment_id = experiment_id
        self.control_config = control_config
        self.treatment_configs = {config.config_id: config for config in treatment_configs}
        self.traffic_allocation = traffic_allocation
        self.duration_hours = duration_hours
        self.min_samples_per_variant = min_samples_per_variant

        # Experiment state
        self.status = ExperimentStatus.CREATED
        self.start_time: float | None = None
        self.end_time: float | None = None

        # Metrics storage
        self.metrics: dict[str, ExperimentMetrics] = {
            control_config.config_id: ExperimentMetrics()
        }
        for config in treatment_configs:
            self.metrics[config.config_id] = ExperimentMetrics()

        # Statistical analysis
        self.significance_results: dict[str, list[StatisticalSignificance]] = {}
        self._metrics_lock = threading.Lock()

    async def start(self) -> None:
        """Start the A/B experiment"""
        self.status = ExperimentStatus.RUNNING
        self.start_time = time.time()
        logger.info(f"Started A/B experiment {self.experiment_id}")

    async def stop(self) -> None:
        """Stop the A/B experiment"""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = time.time()
        logger.info(f"Stopped A/B experiment {self.experiment_id}")

    def record_metrics(self,
                      config_id: str,
                      latency_ms: float,
                      throughput_rps: float,
                      gpu_utilization: float,
                      memory_usage_gb: float,
                      error_rate: float = 0.0,
                      cost_per_request: float = 0.0,
                      accuracy_metrics: dict[str, float] | None = None,
                      custom_metrics: dict[str, float] | None = None) -> None:
        """Record metrics for a specific configuration"""
        if config_id not in self.metrics:
            return

        with self._metrics_lock:
            metrics = self.metrics[config_id]
            metrics.latency_ms.append(latency_ms)
            metrics.throughput_rps.append(throughput_rps)
            metrics.gpu_utilization.append(gpu_utilization)
            metrics.memory_usage_gb.append(memory_usage_gb)
            metrics.error_rate.append(error_rate)
            metrics.cost_per_request.append(cost_per_request)

            if accuracy_metrics:
                for key, value in accuracy_metrics.items():
                    if key not in metrics.accuracy_metrics:
                        metrics.accuracy_metrics[key] = []
                    metrics.accuracy_metrics[key].append(value)

            if custom_metrics:
                for key, value in custom_metrics.items():
                    if key not in metrics.custom_metrics:
                        metrics.custom_metrics[key] = []
                    metrics.custom_metrics[key].append(value)

    def analyze_significance(self, metric_name: str = "latency_ms", alpha: float = 0.05) -> list[StatisticalSignificance]:
        """Analyze statistical significance for a specific metric"""
        if metric_name not in ["latency_ms", "throughput_rps", "gpu_utilization", "memory_usage_gb", "error_rate", "cost_per_request"]:
            raise ValueError(f"Unsupported metric: {metric_name}")

        results = []
        control_data = getattr(self.metrics[self.control_config.config_id], metric_name)

        for config_id, _treatment_config in self.treatment_configs.items():
            treatment_data = getattr(self.metrics[config_id], metric_name)

            if len(control_data) < self.min_samples_per_variant or len(treatment_data) < self.min_samples_per_variant:
                continue

            # Perform statistical tests
            significance = self._calculate_statistical_significance(
                control_data, treatment_data, metric_name, alpha
            )
            results.append(significance)

        self.significance_results[metric_name] = results
        return results

    def _calculate_statistical_significance(self,
                                          control_data: list[float],
                                          treatment_data: list[float],
                                          metric_name: str,
                                          alpha: float = 0.05) -> StatisticalSignificance:
        """Calculate statistical significance between control and treatment"""
        control_array = np.array(control_data)
        treatment_array = np.array(treatment_data)

        # Basic statistics
        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)
        relative_difference = (treatment_mean - control_mean) / control_mean

        # Statistical test (Welch's t-test for unequal variances)
        t_stat, p_value = stats.ttest_ind(treatment_array, control_array, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_array) - 1) * np.var(control_array, ddof=1) +
                             (len(treatment_array) - 1) * np.var(treatment_array, ddof=1)) /
                            (len(control_array) + len(treatment_array) - 2))
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0

        # Confidence interval for difference in means
        se_diff = np.sqrt(np.var(control_array, ddof=1) / len(control_array) +
                         np.var(treatment_array, ddof=1) / len(treatment_array))
        df = len(control_array) + len(treatment_array) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin_error = t_critical * se_diff
        ci_lower = (treatment_mean - control_mean) - margin_error
        ci_upper = (treatment_mean - control_mean) + margin_error

        # Statistical power calculation (post-hoc)
        statistical_power = stats.ttest_power(effect_size, len(treatment_array), alpha, alternative='two-sided')

        return StatisticalSignificance(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            relative_difference=relative_difference,
            p_value=p_value,
            confidence_interval_95=(ci_lower, ci_upper),
            is_significant=p_value < alpha,
            sample_size_control=len(control_array),
            sample_size_treatment=len(treatment_array),
            effect_size=effect_size,
            statistical_power=statistical_power
        )

    def get_experiment_summary(self) -> dict[str, Any]:
        """Get comprehensive experiment summary"""
        if self.start_time is None:
            duration_hours = 0.0
        elif self.end_time is not None:
            duration_hours = (self.end_time - self.start_time) / 3600
        else:
            duration_hours = (time.time() - self.start_time) / 3600

        summary = {
            'experiment_id': self.experiment_id,
            'status': self.status.value,
            'duration_hours': duration_hours,
            'traffic_allocation': self.traffic_allocation,
            'configurations': {},
            'statistical_significance': {},
            'recommendations': []
        }

        # Configuration summaries
        all_configs = [self.control_config] + list(self.treatment_configs.values())
        for config in all_configs:
            if config.config_id in self.metrics:
                metrics = self.metrics[config.config_id]
                summary['configurations'][config.config_id] = {
                    'devices': [{'vendor': d.vendor.value, 'name': d.capabilities.device_name} for d in config.devices],
                    'sample_count': len(metrics.latency_ms),
                    'avg_latency_ms': np.mean(metrics.latency_ms) if metrics.latency_ms else 0.0,
                    'avg_throughput_rps': np.mean(metrics.throughput_rps) if metrics.throughput_rps else 0.0,
                    'avg_gpu_utilization': np.mean(metrics.gpu_utilization) if metrics.gpu_utilization else 0.0,
                    'avg_memory_usage_gb': np.mean(metrics.memory_usage_gb) if metrics.memory_usage_gb else 0.0,
                    'error_rate': np.mean(metrics.error_rate) if metrics.error_rate else 0.0,
                    'avg_cost_per_request': np.mean(metrics.cost_per_request) if metrics.cost_per_request else 0.0
                }

        # Statistical significance results
        for metric, results in self.significance_results.items():
            summary['statistical_significance'][metric] = [
                {
                    'treatment_config': self._find_config_by_metrics(result),
                    'relative_difference': result.relative_difference,
                    'p_value': result.p_value,
                    'is_significant': result.is_significant,
                    'effect_size': result.effect_size
                }
                for result in results
            ]

        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()

        return summary

    def _find_config_by_metrics(self, significance_result: StatisticalSignificance) -> str:
        """Find configuration ID by matching metrics"""
        # This is a helper method - in practice you'd track this more directly
        for config_id in self.treatment_configs.keys():
            if config_id != self.control_config.config_id:
                return config_id
        return "unknown"

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on experiment results"""
        recommendations = []

        # Check if we have enough data
        for config_id, metrics in self.metrics.items():
            if len(metrics.latency_ms) < self.min_samples_per_variant:
                recommendations.append(f"Collect more data for {config_id} (current: {len(metrics.latency_ms)}, minimum: {self.min_samples_per_variant})")

        # Performance recommendations
        if 'latency_ms' in self.significance_results:
            for result in self.significance_results['latency_ms']:
                if result.is_significant and result.relative_difference < -0.05:  # 5% improvement
                    recommendations.append(f"Hardware configuration shows significant latency improvement: {result.relative_difference:.1%}")
                elif result.is_significant and result.relative_difference > 0.05:  # 5% degradation
                    recommendations.append(f"Hardware configuration shows significant latency degradation: {result.relative_difference:.1%}")

        if not recommendations:
            recommendations.append("No significant performance differences detected")

        return recommendations


class TrafficSplitter:
    """
    Intelligent traffic splitter for A/B experiments

    Routes incoming requests to different hardware configurations
    based on experiment parameters and real-time performance.
    """

    def __init__(self, strategy: TrafficAllocationStrategy = TrafficAllocationStrategy.FIXED_SPLIT):
        self.strategy = strategy
        self.experiments: dict[str, ABExperiment] = {}
        self.allocation_cache: dict[str, str] = {}  # request_id -> config_id
        self._lock = threading.Lock()

    def register_experiment(self, experiment: ABExperiment) -> None:
        """Register A/B experiment for traffic splitting"""
        with self._lock:
            self.experiments[experiment.experiment_id] = experiment

    def get_configuration_for_request(self, request_id: str, experiment_id: str) -> str | None:
        """Get hardware configuration for specific request"""
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]
        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Use consistent hashing for deterministic allocation
        hash_value = hash(request_id) % 1000
        cumulative_allocation = 0.0

        for config_id, allocation in experiment.traffic_allocation.items():
            cumulative_allocation += allocation * 1000
            if hash_value < cumulative_allocation:
                self.allocation_cache[request_id] = config_id
                return config_id

        # Fallback to control
        control_id = experiment.control_config.config_id
        self.allocation_cache[request_id] = control_id
        return control_id

    def record_request_outcome(self,
                             request_id: str,
                             experiment_id: str,
                             latency_ms: float,
                             success: bool,
                             additional_metrics: dict[str, float] | None = None) -> None:
        """Record outcome of request for experiment tracking"""
        if experiment_id not in self.experiments:
            return

        config_id = self.allocation_cache.get(request_id)
        if not config_id:
            return

        experiment = self.experiments[experiment_id]
        experiment.record_metrics(
            config_id=config_id,
            latency_ms=latency_ms,
            throughput_rps=1.0 / (latency_ms / 1000.0),  # Approximate
            gpu_utilization=additional_metrics.get('gpu_utilization', 0.0) if additional_metrics else 0.0,
            memory_usage_gb=additional_metrics.get('memory_usage_gb', 0.0) if additional_metrics else 0.0,
            error_rate=0.0 if success else 1.0
        )


class HardwareABTestingFramework:
    """
    Comprehensive A/B testing framework for hardware configurations

    Manages multiple concurrent experiments with statistical rigor
    and automated analysis.
    """

    def __init__(self):
        self.experiments: dict[str, ABExperiment] = {}
        self.traffic_splitter = TrafficSplitter()
        self.analysis_scheduler = None
        self._background_tasks: list[asyncio.Task] = []

    def create_experiment(self,
                         control_hardware: HardwareConfig,
                         treatment_hardware: list[HardwareConfig],
                         traffic_allocation: dict[str, float] | None = None,
                         duration_hours: float = 24.0,
                         min_samples: int = 1000,
                         experiment_name: str | None = None) -> ABExperiment:
        """Create new A/B experiment"""
        experiment_id = experiment_name or f"hw_ab_test_{uuid.uuid4().hex[:8]}"

        # Default traffic allocation
        if traffic_allocation is None:
            total_configs = 1 + len(treatment_hardware)  # control + treatments
            allocation_per_config = 1.0 / total_configs
            traffic_allocation = {control_hardware.config_id: allocation_per_config}
            for config in treatment_hardware:
                traffic_allocation[config.config_id] = allocation_per_config

        experiment = ABExperiment(
            experiment_id=experiment_id,
            control_config=control_hardware,
            treatment_configs=treatment_hardware,
            traffic_allocation=traffic_allocation,
            duration_hours=duration_hours,
            min_samples_per_variant=min_samples
        )

        self.experiments[experiment_id] = experiment
        self.traffic_splitter.register_experiment(experiment)

        logger.info(f"Created A/B experiment {experiment_id}")
        return experiment

    async def start_experiment(self, experiment_id: str) -> None:
        """Start A/B experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        await experiment.start()

        # Start background monitoring
        task = asyncio.create_task(self._monitor_experiment(experiment_id))
        self._background_tasks.append(task)

    async def stop_experiment(self, experiment_id: str) -> None:
        """Stop A/B experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        await experiment.stop()

        # Stop background monitoring
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

    async def _monitor_experiment(self, experiment_id: str) -> None:
        """Background monitoring of experiment"""
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes

                experiment = self.experiments[experiment_id]
                if experiment.status != ExperimentStatus.RUNNING:
                    break

                # Check for early stopping conditions
                try:
                    significance_results = experiment.analyze_significance("latency_ms")
                    for result in significance_results:
                        if result.is_significant and result.statistical_power > 0.8:
                            logger.info(f"Experiment {experiment_id} achieved statistical significance")
                            break
                except Exception as e:
                    logger.warning(f"Analysis failed for experiment {experiment_id}: {e}")

                # Check duration
                if experiment.start_time:
                    elapsed_hours = (time.time() - experiment.start_time) / 3600
                    if elapsed_hours >= experiment.duration_hours:
                        await experiment.stop()
                        break

        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for experiment {experiment_id}")

    def analyze_all_experiments(self) -> dict[str, dict[str, Any]]:
        """Analyze all running experiments"""
        results = {}

        for experiment_id, experiment in self.experiments.items():
            try:
                # Analyze multiple metrics
                metrics_to_analyze = ["latency_ms", "throughput_rps", "gpu_utilization", "cost_per_request"]
                for metric in metrics_to_analyze:
                    experiment.analyze_significance(metric)

                results[experiment_id] = experiment.get_experiment_summary()
            except Exception as e:
                logger.error(f"Failed to analyze experiment {experiment_id}: {e}")
                results[experiment_id] = {"error": str(e)}

        return results

    def get_experiment_recommendations(self, experiment_id: str) -> list[str]:
        """Get recommendations for specific experiment"""
        if experiment_id not in self.experiments:
            return ["Experiment not found"]

        experiment = self.experiments[experiment_id]
        summary = experiment.get_experiment_summary()
        return summary.get('recommendations', [])

    async def cleanup(self) -> None:
        """Clean up resources"""
        # Stop all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("A/B testing framework cleanup completed")
