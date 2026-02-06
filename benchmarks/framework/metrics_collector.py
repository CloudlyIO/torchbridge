"""
Advanced Metrics Collection for Benchmark Framework

Comprehensive metrics collection including hardware monitoring,
statistical analysis, and performance profiling.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import psutil
import torch


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    latency_ms: float
    throughput_samples_per_sec: float
    peak_memory_mb: float
    memory_efficiency: float
    accuracy_loss: float
    statistical_significance: bool
    confidence_interval_95: tuple
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    power_consumption: float = 0.0

class MetricsCollector:
    """Advanced metrics collection and analysis"""

    def __init__(self, device: torch.device):
        self.device = device
        self.enable_detailed_profiling = True

    @contextmanager
    def profile_execution(self):
        """Context manager for detailed execution profiling"""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        yield

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_memory = 0.0

        return {
            'execution_time': execution_time,
            'peak_memory_mb': peak_memory
        }

    def collect_comprehensive_metrics(self, func, inputs, num_trials: int = 100) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                func(*inputs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

        # Collect measurements
        latency_measurements = []
        memory_measurements = []

        for trial in range(num_trials):
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()

            with torch.no_grad():
                result = func(*inputs)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            latency_measurements.append(end_time - start_time)

            if self.device.type == 'cuda':
                memory_measurements.append(torch.cuda.max_memory_allocated() / 1024**2)

        # Statistical analysis
        latencies = np.array(latency_measurements)
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # 95% confidence interval
        confidence_interval = (
            mean_latency - 1.96 * std_latency / np.sqrt(num_trials),
            mean_latency + 1.96 * std_latency / np.sqrt(num_trials)
        )

        # Calculate throughput
        batch_size = inputs[0].shape[0] if len(inputs) > 0 and hasattr(inputs[0], 'shape') else 1
        throughput = batch_size / mean_latency

        # Memory metrics
        peak_memory = max(memory_measurements) if memory_measurements else 0.0

        return PerformanceMetrics(
            latency_ms=mean_latency * 1000,
            throughput_samples_per_sec=throughput,
            peak_memory_mb=peak_memory,
            memory_efficiency=1.0,  # To be calculated relative to baseline
            accuracy_loss=0.0,  # To be measured separately
            statistical_significance=True,  # To be determined in analysis
            confidence_interval_95=confidence_interval,
            cpu_utilization=psutil.cpu_percent(),
            gpu_utilization=0.0  # Would need nvidia-ml-py for detailed GPU monitoring
        )
