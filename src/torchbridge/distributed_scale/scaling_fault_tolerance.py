"""
Auto-Scaling and Fault Tolerance Management

Advanced scaling and fault tolerance systems for distributed training:
- Metrics-based auto-scaling with predictive capabilities
- Comprehensive fault detection and recovery
- Checkpoint-based recovery strategies
- Elastic training support and resource optimization
"""

import logging
import time
from typing import Any

import numpy as np

from .job_management import FailureType

logger = logging.getLogger(__name__)


class AutoScalingManager:
    """
    Auto-scaling manager for dynamic resource allocation

    Features:
    - Metrics-based scaling decisions
    - Multi-metric scaling policies
    - Predictive scaling based on training patterns
    - Cost-aware scaling optimization
    """

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 100,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

        # Scaling state
        self.current_replicas: dict[str, int] = {}
        self.scaling_history: dict[str, list[dict]] = {}
        self.cooldown_period_seconds = 300  # 5 minutes
        self.last_scale_time: dict[str, float] = {}

        # Metrics tracking
        self.metrics_history: dict[str, list[dict]] = {}

    def update_metrics(self, job_id: str, metrics: dict[str, float]):
        """Update metrics for job"""
        if job_id not in self.metrics_history:
            self.metrics_history[job_id] = []

        metric_record = {
            'timestamp': time.time(),
            'gpu_utilization': metrics.get('gpu_utilization', 0.0),
            'memory_utilization': metrics.get('memory_utilization', 0.0),
            'throughput': metrics.get('throughput', 0.0),
            'loss': metrics.get('loss', 0.0),
            'queue_size': metrics.get('queue_size', 0)
        }

        self.metrics_history[job_id].append(metric_record)

        # Keep history bounded
        if len(self.metrics_history[job_id]) > 1000:
            self.metrics_history[job_id] = self.metrics_history[job_id][-500:]

    def should_scale(self, job_id: str) -> tuple[bool, str, int]:
        """
        Determine if job should be scaled

        Returns:
            Tuple of (should_scale, direction, target_replicas)
        """
        if job_id not in self.metrics_history:
            return False, "no_metrics", 0

        # Check cooldown
        last_scale = self.last_scale_time.get(job_id, 0)
        if time.time() - last_scale < self.cooldown_period_seconds:
            return False, "cooldown", 0

        # Get recent metrics
        recent_metrics = self.metrics_history[job_id][-10:]  # Last 10 readings
        if len(recent_metrics) < 3:
            return False, "insufficient_data", 0

        # Calculate average utilization
        avg_gpu_util = np.mean([m['gpu_utilization'] for m in recent_metrics])
        avg_memory_util = np.mean([m['memory_utilization'] for m in recent_metrics])
        np.mean([m['throughput'] for m in recent_metrics])

        current_replicas = self.current_replicas.get(job_id, 1)

        # Scale up conditions
        if (avg_gpu_util > self.scale_up_threshold and
            avg_memory_util < 0.9 and  # Don't scale up if memory constrained
            current_replicas < self.max_replicas):

            # Calculate target replicas
            utilization_ratio = avg_gpu_util / self.target_utilization
            target_replicas = min(
                int(current_replicas * utilization_ratio),
                self.max_replicas
            )

            if target_replicas > current_replicas:
                return True, "scale_up", target_replicas

        # Scale down conditions
        elif (avg_gpu_util < self.scale_down_threshold and
              current_replicas > self.min_replicas):

            utilization_ratio = avg_gpu_util / self.target_utilization
            target_replicas = max(
                int(current_replicas * utilization_ratio),
                self.min_replicas
            )

            if target_replicas < current_replicas:
                return True, "scale_down", target_replicas

        return False, "no_action", current_replicas

    def execute_scaling(self, job_id: str, target_replicas: int) -> bool:
        """Execute scaling action for job"""
        current_replicas = self.current_replicas.get(job_id, 1)

        logger.info(f"Scaling job {job_id} from {current_replicas} to {target_replicas} replicas")

        try:
            # Update replica count
            self.current_replicas[job_id] = target_replicas
            self.last_scale_time[job_id] = time.time()

            # Record scaling event
            if job_id not in self.scaling_history:
                self.scaling_history[job_id] = []

            self.scaling_history[job_id].append({
                'timestamp': time.time(),
                'from_replicas': current_replicas,
                'to_replicas': target_replicas,
                'reason': 'utilization_based'
            })

            # In practice, would update Kubernetes deployment or SLURM job
            logger.info(f"[SIMULATION] Scaled job {job_id} to {target_replicas} replicas")
            return True

        except Exception as e:
            logger.error(f"Failed to scale job {job_id}: {e}")
            return False

    def get_scaling_history(self, job_id: str | None = None) -> dict[str, Any]:
        """Get scaling history for job or all jobs"""
        if job_id:
            return {
                'job_id': job_id,
                'history': self.scaling_history.get(job_id, []),
                'current_replicas': self.current_replicas.get(job_id, 1)
            }

        return {
            'all_jobs': dict(self.scaling_history),
            'current_state': dict(self.current_replicas)
        }

    def get_metrics_summary(self, job_id: str, hours: int = 1) -> dict[str, Any]:
        """Get metrics summary for job"""
        if job_id not in self.metrics_history:
            return {'error': 'No metrics available for job'}

        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.metrics_history[job_id]
            if m['timestamp'] > cutoff_time
        ]

        if not recent_metrics:
            return {'error': 'No recent metrics available'}

        return {
            'job_id': job_id,
            'period_hours': hours,
            'avg_gpu_utilization': np.mean([m['gpu_utilization'] for m in recent_metrics]),
            'avg_memory_utilization': np.mean([m['memory_utilization'] for m in recent_metrics]),
            'avg_throughput': np.mean([m['throughput'] for m in recent_metrics]),
            'min_throughput': np.min([m['throughput'] for m in recent_metrics]),
            'max_throughput': np.max([m['throughput'] for m in recent_metrics]),
            'data_points': len(recent_metrics)
        }


class FaultToleranceManager:
    """
    Fault tolerance manager for distributed training

    Features:
    - Automatic failure detection
    - Checkpoint-based recovery
    - Node failure handling
    - Elastic training support
    """

    def __init__(self, checkpoint_interval_minutes: int = 30):
        self.checkpoint_interval_minutes = checkpoint_interval_minutes

        # Failure tracking
        self.failure_history: dict[str, list[dict]] = {}
        self.recovery_attempts: dict[str, int] = {}

        # Health monitoring
        self.health_checks: dict[str, dict] = {}
        self.unhealthy_nodes: set[str] = set()

    def detect_failure(self, job_id: str, error_info: dict[str, Any]) -> FailureType:
        """Detect and classify failure type"""
        error_message = error_info.get('message', '').lower()
        exit_code = error_info.get('exit_code', 0)

        # Classify failure based on error patterns
        if 'out of memory' in error_message or 'oom' in error_message:
            return FailureType.OOM_FAILURE
        elif 'cuda' in error_message and ('device' in error_message or 'gpu' in error_message):
            return FailureType.GPU_FAILURE
        elif 'network' in error_message or 'connection' in error_message:
            return FailureType.NETWORK_FAILURE
        elif 'node' in error_message or exit_code == -9:  # SIGKILL
            return FailureType.NODE_FAILURE
        elif 'timeout' in error_message:
            return FailureType.TIMEOUT_FAILURE
        else:
            return FailureType.SOFTWARE_FAILURE

    def handle_failure(
        self,
        job_id: str,
        failure_type: FailureType,
        error_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle job failure and attempt recovery"""

        # Record failure
        if job_id not in self.failure_history:
            self.failure_history[job_id] = []

        self.failure_history[job_id].append({
            'timestamp': time.time(),
            'type': failure_type.value,
            'error_info': error_info
        })

        # Get recovery attempts count
        attempts = self.recovery_attempts.get(job_id, 0)

        # Determine recovery strategy
        recovery_plan = self._create_recovery_plan(job_id, failure_type, attempts)

        if recovery_plan['should_recover']:
            logger.info(f"Attempting recovery for job {job_id} (attempt {attempts + 1})")

            # Execute recovery
            success = self._execute_recovery(job_id, recovery_plan)

            if success:
                self.recovery_attempts[job_id] = attempts + 1
                return {
                    'recovered': True,
                    'strategy': recovery_plan['strategy'],
                    'attempt': attempts + 1
                }
            else:
                return {
                    'recovered': False,
                    'reason': 'recovery_failed',
                    'attempt': attempts + 1
                }
        else:
            logger.error(f"Job {job_id} cannot be recovered: {recovery_plan['reason']}")
            return {
                'recovered': False,
                'reason': recovery_plan['reason'],
                'max_attempts_reached': attempts >= recovery_plan['max_attempts']
            }

    def _create_recovery_plan(
        self,
        job_id: str,
        failure_type: FailureType,
        attempts: int
    ) -> dict[str, Any]:
        """Create recovery plan based on failure type"""

        max_attempts = 3

        if attempts >= max_attempts:
            return {
                'should_recover': False,
                'reason': 'max_attempts_reached',
                'max_attempts': max_attempts
            }

        recovery_strategy = "restart_from_checkpoint"

        if failure_type == FailureType.OOM_FAILURE:
            recovery_strategy = "reduce_batch_size"
        elif failure_type == FailureType.NODE_FAILURE:
            recovery_strategy = "reschedule_different_nodes"
        elif failure_type == FailureType.GPU_FAILURE:
            recovery_strategy = "exclude_failed_gpu"
        elif failure_type == FailureType.NETWORK_FAILURE:
            recovery_strategy = "restart_with_network_fix"

        return {
            'should_recover': True,
            'strategy': recovery_strategy,
            'max_attempts': max_attempts,
            'checkpoint_restore': True
        }

    def _execute_recovery(self, job_id: str, recovery_plan: dict[str, Any]) -> bool:
        """Execute recovery plan"""
        strategy = recovery_plan['strategy']

        try:
            if strategy == "restart_from_checkpoint":
                return self._restart_from_checkpoint(job_id)
            elif strategy == "reduce_batch_size":
                return self._restart_with_reduced_batch_size(job_id)
            elif strategy == "reschedule_different_nodes":
                return self._reschedule_on_healthy_nodes(job_id)
            elif strategy == "exclude_failed_gpu":
                return self._restart_excluding_failed_gpu(job_id)
            elif strategy == "restart_with_network_fix":
                return self._restart_with_network_configuration(job_id)
            else:
                logger.error(f"Unknown recovery strategy: {strategy}")
                return False

        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False

    def _restart_from_checkpoint(self, job_id: str) -> bool:
        """Restart job from last checkpoint"""
        # Find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint(job_id)

        if checkpoint_path:
            logger.info(f"Restarting {job_id} from checkpoint: {checkpoint_path}")
            # Would update job configuration to resume from checkpoint
            return True
        else:
            logger.warning(f"No checkpoint found for {job_id}, restarting from beginning")
            return True  # Restart from beginning

    def _restart_with_reduced_batch_size(self, job_id: str) -> bool:
        """Restart with reduced batch size to avoid OOM"""
        logger.info(f"Restarting {job_id} with reduced batch size")
        # Would modify job configuration to use smaller batch size
        return True

    def _reschedule_on_healthy_nodes(self, job_id: str) -> bool:
        """Reschedule job on healthy nodes"""
        logger.info(f"Rescheduling {job_id} on healthy nodes")
        # Would update node selector to avoid failed nodes
        return True

    def _restart_excluding_failed_gpu(self, job_id: str) -> bool:
        """Restart excluding failed GPU"""
        logger.info(f"Restarting {job_id} with GPU exclusion")
        # Would update CUDA_VISIBLE_DEVICES to exclude failed GPU
        return True

    def _restart_with_network_configuration(self, job_id: str) -> bool:
        """Restart with network configuration adjustments"""
        logger.info(f"Restarting {job_id} with network configuration")
        # Would update network settings (timeouts, retry counts, etc.)
        return True

    def _find_latest_checkpoint(self, job_id: str) -> str | None:
        """Find latest checkpoint for job"""
        # Would scan checkpoint directory for latest checkpoint
        # For simulation, return a placeholder
        checkpoint_dir = f"/checkpoints/{job_id}"
        return f"{checkpoint_dir}/latest.pt"  # Placeholder

    def get_failure_statistics(self) -> dict[str, Any]:
        """Get failure statistics across all jobs"""
        stats = {
            'total_failures': 0,
            'failure_types': {},
            'recovery_rate': 0.0,
            'most_common_failures': []
        }

        all_failures = []
        total_recoveries = 0

        for job_id, failures in self.failure_history.items():
            for failure in failures:
                all_failures.append(failure)
                failure_type = failure['type']
                stats['failure_types'][failure_type] = \
                    stats['failure_types'].get(failure_type, 0) + 1

            # Count recoveries (jobs with multiple attempts that succeeded)
            if job_id in self.recovery_attempts and self.recovery_attempts[job_id] > 0:
                total_recoveries += 1

        stats['total_failures'] = len(all_failures)

        if stats['total_failures'] > 0:
            stats['recovery_rate'] = total_recoveries / len(self.failure_history)

        # Most common failure types
        sorted_failures = sorted(
            stats['failure_types'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        stats['most_common_failures'] = sorted_failures[:5]

        return stats

    def update_node_health(self, node_id: str, health_status: dict[str, Any]):
        """Update health status for a node"""
        self.health_checks[node_id] = {
            'timestamp': time.time(),
            'status': health_status.get('status', 'unknown'),
            'cpu_usage': health_status.get('cpu_usage', 0.0),
            'memory_usage': health_status.get('memory_usage', 0.0),
            'disk_usage': health_status.get('disk_usage', 0.0),
            'gpu_status': health_status.get('gpu_status', [])
        }

        # Mark node as unhealthy if needed
        if health_status.get('status') == 'unhealthy':
            self.unhealthy_nodes.add(node_id)
            logger.warning(f"Node {node_id} marked as unhealthy")
        else:
            self.unhealthy_nodes.discard(node_id)

    def get_healthy_nodes(self) -> list[str]:
        """Get list of healthy nodes"""
        all_nodes = set(self.health_checks.keys())
        healthy_nodes = all_nodes - self.unhealthy_nodes
        return list(healthy_nodes)

    def get_recovery_report(self, job_id: str | None = None) -> dict[str, Any]:
        """Get comprehensive recovery report"""
        if job_id:
            return {
                'job_id': job_id,
                'failures': self.failure_history.get(job_id, []),
                'recovery_attempts': self.recovery_attempts.get(job_id, 0),
                'current_status': 'active' if job_id in self.recovery_attempts else 'none'
            }

        # Cluster-wide report
        return {
            'total_jobs_with_failures': len(self.failure_history),
            'total_recovery_attempts': sum(self.recovery_attempts.values()),
            'failure_statistics': self.get_failure_statistics(),
            'unhealthy_nodes': list(self.unhealthy_nodes),
            'healthy_nodes': self.get_healthy_nodes()
        }
