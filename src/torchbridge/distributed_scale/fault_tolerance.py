"""
Hardware Fault Tolerance and Health Monitoring

Comprehensive fault detection and recovery system for large-scale GPU clusters:
- Real-time hardware health monitoring
- Automatic fault detection and recovery
- Performance degradation tracking
- Error reporting and alerting
"""

import logging
import threading
import time
from typing import Any

import numpy as np
import torch

from .hardware_discovery import (
    DeviceInfo,
    HardwareTopologyManager,
    HardwareVendor,
    ThermalState,
)

logger = logging.getLogger(__name__)


class HardwareHealthMonitor:
    """
    Comprehensive hardware health monitoring and fault tolerance system

    Features:
    - Real-time device monitoring
    - Automatic fault detection
    - Performance degradation tracking
    - Health status reporting
    """

    def __init__(self, topology_manager: HardwareTopologyManager, enable_monitoring: bool = True):
        self.topology_manager = topology_manager
        self.enable_monitoring = enable_monitoring

        # Monitoring state
        self.device_monitors: dict[int, threading.Thread] = {}
        self.monitoring_active = False

        # Performance tracking
        self.performance_history: dict[int, list[dict]] = {}
        self.performance_baselines: dict[int, dict[str, float]] = {}

        # Health tracking
        self.device_health_status: dict[int, str] = {}  # "healthy", "degraded", "failed"
        self.error_thresholds = {
            'temperature': 85.0,  # °C
            'power': 1.2,         # multiplier of power limit
            'memory_errors': 10,   # per hour
            'compute_errors': 5    # per hour
        }

        if enable_monitoring:
            self.start_monitoring()

    def start_monitoring(self):
        """Start hardware monitoring threads"""
        if self.monitoring_active or not self.topology_manager.cluster_topology:
            return

        self.monitoring_active = True
        logger.info("Starting hardware health monitoring...")

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                monitor_thread = threading.Thread(
                    target=self._monitor_device,
                    args=(device,),
                    daemon=True
                )
                self.device_monitors[device.device_id] = monitor_thread
                monitor_thread.start()

    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.monitoring_active = False
        logger.info("Stopping hardware health monitoring...")

    def _monitor_device(self, device: DeviceInfo):
        """Monitor individual device health and performance"""
        while self.monitoring_active:
            try:
                # Update device status
                if device.vendor == HardwareVendor.NVIDIA:
                    temp, power, util = self._get_nvidia_device_status(device.device_id)
                    device.current_temp_c = temp
                    device.current_power_w = power
                    device.current_utilization = util
                    device.thermal_state = self._classify_thermal_state(temp)

                # Update memory usage
                if torch.cuda.is_available() and device.device_id < torch.cuda.device_count():
                    device.memory_used_gb = torch.cuda.memory_allocated(device.device_id) / (1024**3)

                # Record performance history
                if device.device_id not in self.performance_history:
                    self.performance_history[device.device_id] = []

                performance_record = {
                    'timestamp': time.time(),
                    'temperature': device.current_temp_c,
                    'power': device.current_power_w,
                    'utilization': device.current_utilization,
                    'memory_used': device.memory_used_gb,
                    'thermal_state': device.thermal_state.value
                }

                self.performance_history[device.device_id].append(performance_record)

                # Keep history bounded
                if len(self.performance_history[device.device_id]) > 1000:
                    self.performance_history[device.device_id] = \
                        self.performance_history[device.device_id][-500:]

                # Check health
                self._check_device_health(device)

                device.last_health_check = time.time()

                time.sleep(10.0)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Error monitoring device {device.device_id}: {e}")
                time.sleep(30.0)  # Back off on errors

    def _get_nvidia_device_status(self, device_id: int) -> tuple:
        """Get NVIDIA device status (temperature, power, utilization)"""
        try:
            # Try to use nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

            return float(temp), float(power), float(util)

        except ImportError:
            # Fallback: try nvidia-smi
            try:
                import subprocess
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu',
                    '--format=csv,noheader,nounits', f'--id={device_id}'
                ], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    temp = float(values[0]) if values[0] != '[Not Supported]' else 0.0
                    power = float(values[1]) if values[1] != '[Not Supported]' else 0.0
                    util = float(values[2]) if values[2] != '[Not Supported]' else 0.0
                    return temp, power, util
            except Exception:
                pass
        except Exception:
            pass

        return 0.0, 0.0, 0.0  # Fallback values

    def _classify_thermal_state(self, temp_c: float) -> ThermalState:
        """Classify thermal state based on temperature"""
        if temp_c < 70:
            return ThermalState.OPTIMAL
        elif temp_c < 80:
            return ThermalState.WARM
        elif temp_c < 90:
            return ThermalState.HOT
        else:
            return ThermalState.CRITICAL

    def _check_device_health(self, device: DeviceInfo):
        """Check device health and update status"""
        health_issues = []

        # Temperature check
        if device.current_temp_c > self.error_thresholds['temperature']:
            health_issues.append(f"High temperature: {device.current_temp_c}°C")
            device.thermal_state = ThermalState.CRITICAL

        # Power check (if power limit is known)
        if device.power_limit_w > 0:
            power_ratio = device.current_power_w / device.power_limit_w
            if power_ratio > self.error_thresholds['power']:
                health_issues.append(f"High power usage: {power_ratio:.2f}x limit")

        # Memory error check (would require actual error monitoring)
        # This is simplified for demonstration

        # Performance degradation check
        if self._detect_performance_degradation(device.device_id):
            health_issues.append("Performance degradation detected")

        # Update health status
        if health_issues:
            self.device_health_status[device.device_id] = "degraded"
            logger.warning(f"Device {device.device_id} health issues: {health_issues}")
        else:
            self.device_health_status[device.device_id] = "healthy"

    def _detect_performance_degradation(self, device_id: int) -> bool:
        """Detect if device performance has degraded"""
        if device_id not in self.performance_history:
            return False

        history = self.performance_history[device_id]
        if len(history) < 20:  # Need enough data points
            return False

        # Compare recent performance to baseline
        recent_utilization = np.mean([h['utilization'] for h in history[-10:]])
        baseline_utilization = np.mean([h['utilization'] for h in history[-20:-10]])

        # Check for significant drop in utilization (could indicate issues)
        if baseline_utilization > 50.0 and recent_utilization < baseline_utilization * 0.7:
            return True

        # Check for temperature spikes
        recent_temps = [h['temperature'] for h in history[-5:]]
        if any(temp > self.error_thresholds['temperature'] for temp in recent_temps):
            return True

        return False

    def get_device_health_status(self, device_id: int | None = None) -> dict[str, Any]:
        """Get health status for device(s)"""
        if device_id is not None:
            return {
                'device_id': device_id,
                'status': self.device_health_status.get(device_id, "unknown"),
                'last_check': time.time()
            }

        # Return status for all devices
        health_summary = {
            'timestamp': time.time(),
            'total_devices': len(self.device_health_status),
            'healthy_devices': sum(1 for status in self.device_health_status.values() if status == "healthy"),
            'degraded_devices': sum(1 for status in self.device_health_status.values() if status == "degraded"),
            'failed_devices': sum(1 for status in self.device_health_status.values() if status == "failed"),
            'device_details': dict(self.device_health_status)
        }

        return health_summary

    def get_performance_report(self, device_id: int | None = None, hours: int = 1) -> dict[str, Any]:
        """Generate performance report for device(s)"""
        if device_id is not None:
            return self._get_single_device_report(device_id, hours)

        # Generate cluster-wide report
        cutoff_time = time.time() - (hours * 3600)

        report = {
            'timestamp': time.time(),
            'reporting_period_hours': hours,
            'devices': {},
            'cluster_summary': {
                'avg_temperature': 0.0,
                'avg_utilization': 0.0,
                'avg_power': 0.0,
                'thermal_events': 0,
                'performance_issues': 0
            }
        }

        all_temps = []
        all_utils = []
        all_powers = []
        thermal_events = 0
        performance_issues = 0

        for device_id, history in self.performance_history.items():
            recent_history = [h for h in history if h['timestamp'] > cutoff_time]

            if not recent_history:
                continue

            # Calculate device metrics
            device_report = {
                'avg_temperature': np.mean([h['temperature'] for h in recent_history]),
                'max_temperature': np.max([h['temperature'] for h in recent_history]),
                'avg_utilization': np.mean([h['utilization'] for h in recent_history]),
                'avg_power': np.mean([h['power'] for h in recent_history]),
                'thermal_events': sum(1 for h in recent_history if h['temperature'] > self.error_thresholds['temperature']),
                'health_status': self.device_health_status.get(device_id, "unknown")
            }

            report['devices'][device_id] = device_report

            # Accumulate cluster metrics
            all_temps.extend([h['temperature'] for h in recent_history])
            all_utils.extend([h['utilization'] for h in recent_history])
            all_powers.extend([h['power'] for h in recent_history])
            thermal_events += device_report['thermal_events']

            if device_report['health_status'] in ['degraded', 'failed']:
                performance_issues += 1

        # Calculate cluster summary
        if all_temps:
            report['cluster_summary']['avg_temperature'] = np.mean(all_temps)
            report['cluster_summary']['avg_utilization'] = np.mean(all_utils)
            report['cluster_summary']['avg_power'] = np.mean(all_powers)

        report['cluster_summary']['thermal_events'] = thermal_events
        report['cluster_summary']['performance_issues'] = performance_issues

        return report

    def _get_single_device_report(self, device_id: int, hours: int) -> dict[str, Any]:
        """Generate report for single device"""
        if device_id not in self.performance_history:
            return {'error': f'No performance history for device {device_id}'}

        cutoff_time = time.time() - (hours * 3600)
        history = self.performance_history[device_id]
        recent_history = [h for h in history if h['timestamp'] > cutoff_time]

        if not recent_history:
            return {'error': f'No recent performance data for device {device_id}'}

        return {
            'device_id': device_id,
            'reporting_period_hours': hours,
            'data_points': len(recent_history),
            'avg_temperature': np.mean([h['temperature'] for h in recent_history]),
            'max_temperature': np.max([h['temperature'] for h in recent_history]),
            'min_temperature': np.min([h['temperature'] for h in recent_history]),
            'avg_utilization': np.mean([h['utilization'] for h in recent_history]),
            'max_utilization': np.max([h['utilization'] for h in recent_history]),
            'avg_power': np.mean([h['power'] for h in recent_history]),
            'max_power': np.max([h['power'] for h in recent_history]),
            'thermal_events': sum(1 for h in recent_history if h['temperature'] > self.error_thresholds['temperature']),
            'current_health': self.device_health_status.get(device_id, "unknown"),
            'performance_trend': self._calculate_performance_trend(device_id)
        }

    def _calculate_performance_trend(self, device_id: int) -> str:
        """Calculate performance trend for device"""
        if device_id not in self.performance_history:
            return "unknown"

        history = self.performance_history[device_id]
        if len(history) < 10:
            return "insufficient_data"

        # Compare first half to second half of recent history
        mid_point = len(history) // 2
        first_half_util = np.mean([h['utilization'] for h in history[:mid_point]])
        second_half_util = np.mean([h['utilization'] for h in history[mid_point:]])

        if second_half_util > first_half_util * 1.1:
            return "improving"
        elif second_half_util < first_half_util * 0.9:
            return "degrading"
        else:
            return "stable"

    def reset_device_health(self, device_id: int):
        """Reset health status for a device (after manual intervention)"""
        if device_id in self.device_health_status:
            self.device_health_status[device_id] = "healthy"
            logger.info(f"Reset health status for device {device_id}")

    def set_error_thresholds(self, **thresholds):
        """Update error detection thresholds"""
        self.error_thresholds.update(thresholds)
        logger.info(f"Updated error thresholds: {self.error_thresholds}")
