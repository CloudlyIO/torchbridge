"""
Thermal-Aware Scheduling and Power Management

Advanced thermal and power management for large-scale GPU clusters:
- Dynamic thermal monitoring and throttling prevention
- Workload redistribution based on thermal state
- Power budget management and efficiency optimization
- Carbon footprint minimization
"""

import time
import logging
from typing import Dict, List, Optional, Any, Set
import numpy as np

from .hardware_discovery import (
    HardwareTopologyManager, ThermalState, DeviceInfo
)

logger = logging.getLogger(__name__)


class ThermalAwareScheduler:
    """
    Thermal-aware job scheduler for large-scale training

    Features:
    - Dynamic thermal monitoring
    - Workload redistribution based on temperature
    - Thermal throttling prevention
    - Power budget management
    """

    def __init__(
        self,
        topology_manager: HardwareTopologyManager,
        thermal_threshold: float = 85.0,
        power_budget_w: Optional[float] = None
    ):
        self.topology_manager = topology_manager
        self.thermal_threshold = thermal_threshold
        self.power_budget_w = power_budget_w

        # Scheduling state
        self.active_jobs: Dict[str, Dict] = {}
        self.thermal_history: Dict[int, List[float]] = {}
        self.power_history: Dict[int, List[float]] = {}

        # Thermal management
        self.throttled_devices: Set[int] = set()
        self.cooling_devices: Set[int] = set()

    def schedule_job(
        self,
        job_id: str,
        required_devices: int,
        estimated_power_per_device: float = 300.0,
        thermal_sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Schedule job with thermal awareness

        Args:
            job_id: Unique job identifier
            required_devices: Number of devices needed
            estimated_power_per_device: Estimated power consumption per device
            thermal_sensitivity: Job's sensitivity to thermal throttling (0-1)

        Returns:
            Scheduling decision with device assignments
        """
        if not self.topology_manager.cluster_topology:
            return {'success': False, 'error': 'No topology available'}

        # Get available devices considering thermal constraints
        available_devices = self._get_thermally_safe_devices()

        if len(available_devices) < required_devices:
            # Try thermal balancing
            balanced_devices = self._attempt_thermal_balancing(required_devices)
            if len(balanced_devices) >= required_devices:
                available_devices = balanced_devices
            else:
                return {
                    'success': False,
                    'error': f'Not enough thermally safe devices: {len(available_devices)} < {required_devices}',
                    'available_devices': len(available_devices)
                }

        # Select optimal devices
        selected_devices = self._select_optimal_thermal_devices(
            available_devices[:required_devices],
            estimated_power_per_device,
            thermal_sensitivity
        )

        # Check power budget
        if self.power_budget_w:
            total_estimated_power = len(selected_devices) * estimated_power_per_device
            current_power = self._get_current_cluster_power()

            if current_power + total_estimated_power > self.power_budget_w:
                return {
                    'success': False,
                    'error': f'Power budget exceeded: {current_power + total_estimated_power} > {self.power_budget_w}',
                    'current_power': current_power,
                    'estimated_additional': total_estimated_power
                }

        # Record job
        self.active_jobs[job_id] = {
            'devices': selected_devices,
            'power_per_device': estimated_power_per_device,
            'thermal_sensitivity': thermal_sensitivity,
            'start_time': time.time()
        }

        return {
            'success': True,
            'devices': selected_devices,
            'thermal_safety_score': self._calculate_thermal_safety_score(selected_devices),
            'estimated_power': len(selected_devices) * estimated_power_per_device
        }

    def _get_thermally_safe_devices(self) -> List[int]:
        """Get list of devices that are thermally safe for new workloads"""
        safe_devices = []

        if not self.topology_manager.cluster_topology:
            return safe_devices

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                # Skip failed or throttled devices
                if device.device_id in self.throttled_devices:
                    continue

                # Check thermal state
                if device.thermal_state in [ThermalState.CRITICAL, ThermalState.HOT]:
                    continue

                # Check temperature trend
                if self._is_temperature_rising(device.device_id):
                    continue

                # Check current temperature against threshold
                if device.current_temp_c < self.thermal_threshold:
                    safe_devices.append(device.device_id)

        return safe_devices

    def _is_temperature_rising(self, device_id: int) -> bool:
        """Check if device temperature is rising"""
        if device_id not in self.thermal_history:
            return False

        history = self.thermal_history[device_id]
        if len(history) < 3:
            return False

        # Check if temperature is consistently rising
        recent_temps = history[-3:]
        return all(recent_temps[i] <= recent_temps[i+1] for i in range(len(recent_temps)-1))

    def _attempt_thermal_balancing(self, required_devices: int) -> List[int]:
        """Attempt to balance thermal load across cluster"""
        if not self.topology_manager.cluster_topology:
            return []

        # Identify devices that could be cooled down
        candidate_devices = []

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                # Skip completely failed devices
                if hasattr(self.topology_manager, 'device_health_status'):
                    if self.topology_manager.device_health_status.get(device.device_id) == "failed":
                        continue

                # Include warm devices that could potentially cool down
                if device.thermal_state in [ThermalState.OPTIMAL, ThermalState.WARM]:
                    candidate_devices.append(device.device_id)

        # Sort by current temperature (prefer cooler devices)
        candidate_devices.sort(
            key=lambda did: next(
                dev.current_temp_c for node in self.topology_manager.cluster_topology.nodes.values()
                for dev in node.devices if dev.device_id == did
            )
        )

        return candidate_devices

    def _select_optimal_thermal_devices(
        self,
        candidate_devices: List[int],
        power_per_device: float,
        thermal_sensitivity: float
    ) -> List[int]:
        """Select optimal devices considering thermal characteristics"""
        if not self.topology_manager.cluster_topology:
            return candidate_devices

        # Score devices based on thermal characteristics
        device_scores = []

        for device_id in candidate_devices:
            device = next(
                dev for node in self.topology_manager.cluster_topology.nodes.values()
                for dev in node.devices if dev.device_id == device_id
            )

            score = 0.0

            # Temperature score (prefer cooler devices)
            temp_score = max(0, (self.thermal_threshold - device.current_temp_c) / self.thermal_threshold)
            score += temp_score * 0.4

            # Thermal headroom score
            thermal_headroom = max(0, self.thermal_threshold - device.current_temp_c)
            headroom_score = thermal_headroom / 30.0  # Normalize to ~30°C headroom
            score += headroom_score * 0.3

            # Power efficiency score
            if device.power_limit_w > 0:
                power_efficiency = 1.0 - (device.current_power_w / device.power_limit_w)
                score += power_efficiency * 0.2

            # Historical stability score
            stability_score = self._calculate_thermal_stability(device_id)
            score += stability_score * 0.1

            device_scores.append((device_id, score))

        # Sort by score (higher is better) and return
        device_scores.sort(key=lambda x: x[1], reverse=True)
        return [device_id for device_id, _ in device_scores]

    def _calculate_thermal_stability(self, device_id: int) -> float:
        """Calculate thermal stability score for device"""
        if device_id not in self.thermal_history:
            return 0.5  # Neutral score for unknown devices

        history = self.thermal_history[device_id]
        if len(history) < 5:
            return 0.5

        # Calculate temperature variance (lower is better)
        temp_variance = np.var(history[-10:])  # Last 10 readings

        # Normalize variance to 0-1 score (lower variance = higher score)
        stability_score = max(0, 1.0 - (temp_variance / 100.0))  # Assume 100°C² as high variance

        return stability_score

    def _get_current_cluster_power(self) -> float:
        """Get current total cluster power consumption"""
        total_power = 0.0

        if not self.topology_manager.cluster_topology:
            return total_power

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                total_power += device.current_power_w

        return total_power

    def _calculate_thermal_safety_score(self, device_ids: List[int]) -> float:
        """Calculate overall thermal safety score for device selection"""
        if not device_ids or not self.topology_manager.cluster_topology:
            return 0.0

        scores = []

        for device_id in device_ids:
            device = next(
                dev for node in self.topology_manager.cluster_topology.nodes.values()
                for dev in node.devices if dev.device_id == device_id
            )

            # Higher score for cooler devices
            temp_score = max(0, (self.thermal_threshold - device.current_temp_c) / self.thermal_threshold)
            scores.append(temp_score)

        return np.mean(scores)

    def monitor_thermal_state(self):
        """Monitor and update thermal state of active jobs"""
        current_time = time.time()

        # Update thermal history
        if self.topology_manager.cluster_topology:
            for node in self.topology_manager.cluster_topology.nodes.values():
                for device in node.devices:
                    if device.device_id not in self.thermal_history:
                        self.thermal_history[device.device_id] = []

                    self.thermal_history[device.device_id].append(device.current_temp_c)

                    # Keep history bounded
                    if len(self.thermal_history[device.device_id]) > 100:
                        self.thermal_history[device.device_id] = \
                            self.thermal_history[device.device_id][-50:]

                    # Check for thermal issues
                    if device.current_temp_c > self.thermal_threshold:
                        if device.device_id not in self.throttled_devices:
                            logger.warning(f"Device {device.device_id} exceeding thermal threshold: "
                                         f"{device.current_temp_c}°C > {self.thermal_threshold}°C")
                            self.throttled_devices.add(device.device_id)

                    # Check for recovery
                    elif device.device_id in self.throttled_devices:
                        if device.current_temp_c < self.thermal_threshold - 5.0:  # 5°C hysteresis
                            logger.info(f"Device {device.device_id} recovered from thermal throttling: "
                                       f"{device.current_temp_c}°C")
                            self.throttled_devices.discard(device.device_id)

    def get_thermal_report(self) -> Dict[str, Any]:
        """Generate thermal management report"""
        report = {
            'timestamp': time.time(),
            'active_jobs': len(self.active_jobs),
            'throttled_devices': len(self.throttled_devices),
            'total_devices': 0,
            'thermal_distribution': {},
            'power_consumption': 0.0,
            'thermal_efficiency': 0.0
        }

        if not self.topology_manager.cluster_topology:
            return report

        thermal_states = {}
        total_temp = 0.0
        device_count = 0

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                device_count += 1
                total_temp += device.current_temp_c
                report['power_consumption'] += device.current_power_w

                state = device.thermal_state.value
                thermal_states[state] = thermal_states.get(state, 0) + 1

        report['total_devices'] = device_count
        report['thermal_distribution'] = thermal_states

        if device_count > 0:
            avg_temp = total_temp / device_count
            # Calculate efficiency as (threshold - avg_temp) / threshold
            report['thermal_efficiency'] = max(0, (self.thermal_threshold - avg_temp) / self.thermal_threshold)

        return report


class PowerEfficiencyOptimizer:
    """
    Power efficiency optimizer for large-scale deployments

    Features:
    - Dynamic power scaling
    - Workload-aware power management
    - Energy efficiency optimization
    - Carbon footprint minimization
    """

    def __init__(
        self,
        topology_manager: HardwareTopologyManager,
        power_budget_w: Optional[float] = None,
        carbon_intensity_g_kwh: float = 400.0
    ):
        self.topology_manager = topology_manager
        self.power_budget_w = power_budget_w
        self.carbon_intensity_g_kwh = carbon_intensity_g_kwh

        # Power management state
        self.power_profiles: Dict[int, Dict] = {}
        self.efficiency_history: List[Dict] = []

    def optimize_power_distribution(
        self,
        workload_priorities: Dict[str, float],
        efficiency_target: float = 0.8
    ) -> Dict[str, Any]:
        """
        Optimize power distribution across cluster

        Args:
            workload_priorities: Map of workload_id -> priority
            efficiency_target: Target efficiency (FLOPS/Watt ratio)

        Returns:
            Power optimization plan
        """
        if not self.topology_manager.cluster_topology:
            return {'success': False, 'error': 'No topology available'}

        # Calculate current power consumption and efficiency
        current_metrics = self._calculate_power_metrics()

        optimization_plan = {
            'current_power_w': current_metrics['total_power'],
            'current_efficiency': current_metrics['efficiency'],
            'target_efficiency': efficiency_target,
            'device_adjustments': {},
            'estimated_savings_w': 0.0,
            'carbon_reduction_g_h': 0.0
        }

        # Analyze per-device efficiency
        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                device_efficiency = self._calculate_device_efficiency(device)
                target_power = self._calculate_optimal_power(device, efficiency_target)

                if target_power < device.current_power_w:
                    power_reduction = device.current_power_w - target_power
                    optimization_plan['device_adjustments'][device.device_id] = {
                        'current_power': device.current_power_w,
                        'target_power': target_power,
                        'reduction': power_reduction,
                        'efficiency_gain': efficiency_target - device_efficiency
                    }
                    optimization_plan['estimated_savings_w'] += power_reduction

        # Calculate carbon reduction
        carbon_reduction = (optimization_plan['estimated_savings_w'] / 1000.0) * \
                          (self.carbon_intensity_g_kwh / 1000.0)
        optimization_plan['carbon_reduction_g_h'] = carbon_reduction

        return optimization_plan

    def _calculate_power_metrics(self) -> Dict[str, float]:
        """Calculate current power consumption and efficiency metrics"""
        total_power = 0.0
        total_compute = 0.0

        if not self.topology_manager.cluster_topology:
            return {'total_power': 0.0, 'efficiency': 0.0}

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                total_power += device.current_power_w

                # Estimate compute throughput based on utilization
                utilized_flops = device.peak_flops_fp32 * (device.current_utilization / 100.0)
                total_compute += utilized_flops

        efficiency = total_compute / max(total_power, 1.0) if total_power > 0 else 0.0

        return {
            'total_power': total_power,
            'total_compute_flops': total_compute,
            'efficiency': efficiency
        }

    def _calculate_device_efficiency(self, device: DeviceInfo) -> float:
        """Calculate power efficiency for specific device"""
        if device.current_power_w <= 0:
            return 0.0

        # Estimate actual compute output
        utilized_flops = device.peak_flops_fp32 * (device.current_utilization / 100.0)

        # Calculate efficiency (FLOPS per Watt)
        efficiency = utilized_flops / device.current_power_w

        return efficiency

    def _calculate_optimal_power(self, device: DeviceInfo, target_efficiency: float) -> float:
        """Calculate optimal power consumption for target efficiency"""
        if target_efficiency <= 0:
            return device.current_power_w

        # Estimate compute requirement to maintain current throughput
        current_compute = device.peak_flops_fp32 * (device.current_utilization / 100.0)

        # Calculate power needed for target efficiency
        optimal_power = current_compute / target_efficiency

        # Clamp to reasonable bounds
        min_power = device.current_power_w * 0.5  # Don't reduce by more than 50%
        max_power = device.power_limit_w if device.power_limit_w > 0 else device.current_power_w

        return np.clip(optimal_power, min_power, max_power)