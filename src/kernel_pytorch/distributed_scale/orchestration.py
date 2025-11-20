"""
Container Orchestration and Cluster Management for Large-Scale Training (2025)

Advanced orchestration framework for managing distributed training across thousands of GPUs:
- Kubernetes-native distributed training orchestration
- SLURM cluster integration and job scheduling
- Auto-scaling based on training metrics and resource utilization
- Fault tolerance with automatic failure detection and recovery
- Multi-tenant resource isolation and priority scheduling
"""

import asyncio
import time
import logging
import json
import yaml
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from datetime import datetime, timedelta
import tempfile
import os

logger = logging.getLogger(__name__)


class JobState(Enum):
    """Training job states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCALING = "scaling"


class ResourceType(Enum):
    """Resource types for scheduling"""
    GPU_MEMORY = "gpu_memory"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE_IOPS = "storage_iops"


class FailureType(Enum):
    """Types of failures that can occur"""
    NODE_FAILURE = "node_failure"
    GPU_FAILURE = "gpu_failure"
    NETWORK_FAILURE = "network_failure"
    SOFTWARE_FAILURE = "software_failure"
    OOM_FAILURE = "oom_failure"
    TIMEOUT_FAILURE = "timeout_failure"


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: int
    memory_gb: int
    storage_gb: int = 100
    network_bandwidth_gbps: int = 10
    required_gpu_types: Optional[List[str]] = None
    node_selector: Optional[Dict[str, str]] = None


@dataclass
class TrainingJobSpec:
    """Training job specification"""
    job_id: str
    name: str
    image: str
    command: List[str]
    resources: ResourceRequirement

    # Training-specific parameters
    world_size: int
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: Optional[int] = None

    # Job management
    priority: int = 0
    max_runtime_hours: int = 24
    restart_policy: str = "OnFailure"
    checkpoint_interval_minutes: int = 30

    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)

    # Fault tolerance
    max_retries: int = 3
    enable_auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: Optional[int] = None


@dataclass
class JobStatus:
    """Current status of a training job"""
    job_id: str
    state: JobState
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    allocated_nodes: List[str] = field(default_factory=list)
    allocated_gpus: List[int] = field(default_factory=list)
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    error_message: Optional[str] = None
    restart_count: int = 0
    last_checkpoint: Optional[str] = None


@dataclass
class ClusterNode:
    """Cluster node information"""
    node_id: str
    hostname: str
    total_gpus: int
    available_gpus: int
    total_memory_gb: int
    available_memory_gb: int
    total_cpu_cores: int
    available_cpu_cores: int
    labels: Dict[str, str] = field(default_factory=dict)
    taints: List[Dict[str, Any]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


class KubernetesDistributedOrchestrator:
    """
    Kubernetes-based orchestrator for distributed training

    Features:
    - Native Kubernetes integration with CRDs
    - Multi-GPU pod scheduling
    - Service mesh integration for communication
    - Persistent volume management for checkpoints
    """

    def __init__(
        self,
        namespace: str = "ml-training",
        kubeconfig_path: Optional[str] = None
    ):
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path

        # Job tracking
        self.active_jobs: Dict[str, TrainingJobSpec] = {}
        self.job_statuses: Dict[str, JobStatus] = {}

        # Kubernetes resources
        self.k8s_available = self._check_kubernetes_availability()

        # Monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

    def _check_kubernetes_availability(self) -> bool:
        """Check if Kubernetes is available"""
        try:
            result = subprocess.run(['kubectl', 'version', '--client'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            logger.warning("kubectl not available, running in simulation mode")
            return False

    def submit_job(self, job_spec: TrainingJobSpec) -> Dict[str, Any]:
        """
        Submit distributed training job to Kubernetes

        Args:
            job_spec: Training job specification

        Returns:
            Submission result with job status
        """
        logger.info(f"Submitting job {job_spec.job_id} to Kubernetes")

        # Validate job specification
        validation_result = self._validate_job_spec(job_spec)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': f"Job validation failed: {validation_result['errors']}"
            }

        # Calculate data parallel size if not specified
        if job_spec.data_parallel_size is None:
            job_spec.data_parallel_size = job_spec.world_size // (
                job_spec.tensor_parallel_size * job_spec.pipeline_parallel_size
            )

        try:
            # Create Kubernetes resources
            k8s_resources = self._create_kubernetes_resources(job_spec)

            if self.k8s_available:
                # Apply resources to cluster
                self._apply_kubernetes_resources(k8s_resources)
            else:
                # Simulation mode
                logger.info(f"[SIMULATION] Would create K8s resources: {len(k8s_resources)} objects")

            # Track job
            self.active_jobs[job_spec.job_id] = job_spec
            self.job_statuses[job_spec.job_id] = JobStatus(
                job_id=job_spec.job_id,
                state=JobState.PENDING,
                start_time=datetime.now()
            )

            return {
                'success': True,
                'job_id': job_spec.job_id,
                'estimated_start_time': self._estimate_start_time(job_spec),
                'allocated_resources': self._estimate_resource_allocation(job_spec)
            }

        except Exception as e:
            logger.error(f"Failed to submit job {job_spec.job_id}: {e}")
            return {'success': False, 'error': str(e)}

    def _validate_job_spec(self, job_spec: TrainingJobSpec) -> Dict[str, Any]:
        """Validate training job specification"""
        errors = []

        # Check parallelism configuration
        total_required = (job_spec.tensor_parallel_size *
                         job_spec.pipeline_parallel_size *
                         (job_spec.data_parallel_size or 1))
        if total_required != job_spec.world_size:
            errors.append(f"Parallelism mismatch: {total_required} != {job_spec.world_size}")

        # Check resource requirements
        if job_spec.resources.gpu_count > job_spec.world_size:
            errors.append(f"More GPUs requested than world size: {job_spec.resources.gpu_count} > {job_spec.world_size}")

        # Check image and command
        if not job_spec.image:
            errors.append("Container image not specified")

        if not job_spec.command:
            errors.append("Command not specified")

        return {'valid': len(errors) == 0, 'errors': errors}

    def _create_kubernetes_resources(self, job_spec: TrainingJobSpec) -> List[Dict[str, Any]]:
        """Create Kubernetes resource manifests for training job"""
        resources = []

        # Create ConfigMap for training script
        config_map = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f'{job_spec.job_id}-config',
                'namespace': self.namespace
            },
            'data': {
                'world_size': str(job_spec.world_size),
                'tensor_parallel_size': str(job_spec.tensor_parallel_size),
                'pipeline_parallel_size': str(job_spec.pipeline_parallel_size),
                'data_parallel_size': str(job_spec.data_parallel_size or 1)
            }
        }
        resources.append(config_map)

        # Create PyTorchJob custom resource
        pytorch_job = {
            'apiVersion': 'kubeflow.org/v1',
            'kind': 'PyTorchJob',
            'metadata': {
                'name': job_spec.job_id,
                'namespace': self.namespace
            },
            'spec': {
                'pytorchReplicaSpecs': {
                    'Master': {
                        'replicas': 1,
                        'restartPolicy': job_spec.restart_policy,
                        'template': self._create_pod_template(job_spec, rank=0)
                    },
                    'Worker': {
                        'replicas': max(0, job_spec.world_size - 1),
                        'restartPolicy': job_spec.restart_policy,
                        'template': self._create_pod_template(job_spec, rank=None)
                    }
                }
            }
        }
        resources.append(pytorch_job)

        # Create Service for inter-pod communication
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{job_spec.job_id}-service',
                'namespace': self.namespace
            },
            'spec': {
                'selector': {
                    'training-job-name': job_spec.job_id
                },
                'ports': [
                    {'name': 'master', 'port': 23456, 'targetPort': 23456},
                    {'name': 'worker', 'port': 23457, 'targetPort': 23457}
                ],
                'clusterIP': 'None'  # Headless service
            }
        }
        resources.append(service)

        # Create PersistentVolumeClaim for checkpoints
        if job_spec.checkpoint_interval_minutes > 0:
            pvc = {
                'apiVersion': 'v1',
                'kind': 'PersistentVolumeClaim',
                'metadata': {
                    'name': f'{job_spec.job_id}-checkpoints',
                    'namespace': self.namespace
                },
                'spec': {
                    'accessModes': ['ReadWriteMany'],
                    'resources': {
                        'requests': {
                            'storage': f'{job_spec.resources.storage_gb}Gi'
                        }
                    }
                }
            }
            resources.append(pvc)

        return resources

    def _create_pod_template(self, job_spec: TrainingJobSpec, rank: Optional[int]) -> Dict[str, Any]:
        """Create pod template for training workers"""
        # Base environment variables
        env_vars = [
            {'name': 'WORLD_SIZE', 'value': str(job_spec.world_size)},
            {'name': 'TENSOR_PARALLEL_SIZE', 'value': str(job_spec.tensor_parallel_size)},
            {'name': 'PIPELINE_PARALLEL_SIZE', 'value': str(job_spec.pipeline_parallel_size)},
            {'name': 'MASTER_ADDR', 'value': f'{job_spec.job_id}-service.{self.namespace}.svc.cluster.local'},
            {'name': 'MASTER_PORT', 'value': '23456'},
            {'name': 'NCCL_DEBUG', 'value': 'INFO'},
            {'name': 'CUDA_VISIBLE_DEVICES', 'value': 'all'}
        ]

        # Add custom environment variables
        for key, value in job_spec.env_vars.items():
            env_vars.append({'name': key, 'value': value})

        # Container resources
        container_resources = {
            'requests': {
                'nvidia.com/gpu': str(job_spec.resources.gpu_count // job_spec.world_size),
                'cpu': str(job_spec.resources.cpu_cores // job_spec.world_size),
                'memory': f'{job_spec.resources.memory_gb // job_spec.world_size}Gi'
            },
            'limits': {
                'nvidia.com/gpu': str(job_spec.resources.gpu_count // job_spec.world_size),
                'cpu': str(job_spec.resources.cpu_cores // job_spec.world_size),
                'memory': f'{job_spec.resources.memory_gb // job_spec.world_size}Gi'
            }
        }

        # Volume mounts
        volume_mounts = [
            {
                'name': 'shm',
                'mountPath': '/dev/shm'
            }
        ]

        # Add checkpoint volume if enabled
        if job_spec.checkpoint_interval_minutes > 0:
            volume_mounts.append({
                'name': 'checkpoints',
                'mountPath': '/checkpoints'
            })

        # Add custom volumes
        for vol in job_spec.volumes:
            volume_mounts.append(vol.get('mount', {}))

        # Pod template
        template = {
            'metadata': {
                'labels': {
                    'training-job-name': job_spec.job_id,
                    'app': 'distributed-training'
                }
            },
            'spec': {
                'containers': [{
                    'name': 'training',
                    'image': job_spec.image,
                    'command': job_spec.command,
                    'env': env_vars,
                    'resources': container_resources,
                    'volumeMounts': volume_mounts
                }],
                'volumes': [
                    {
                        'name': 'shm',
                        'emptyDir': {
                            'medium': 'Memory',
                            'sizeLimit': '8Gi'
                        }
                    }
                ],
                'restartPolicy': 'Never'
            }
        }

        # Add checkpoint volume
        if job_spec.checkpoint_interval_minutes > 0:
            template['spec']['volumes'].append({
                'name': 'checkpoints',
                'persistentVolumeClaim': {
                    'claimName': f'{job_spec.job_id}-checkpoints'
                }
            })

        # Add custom volumes
        for vol in job_spec.volumes:
            if 'volume' in vol:
                template['spec']['volumes'].append(vol['volume'])

        # Node selection
        if job_spec.resources.node_selector:
            template['spec']['nodeSelector'] = job_spec.resources.node_selector

        # GPU type requirements
        if job_spec.resources.required_gpu_types:
            if 'nodeSelector' not in template['spec']:
                template['spec']['nodeSelector'] = {}
            template['spec']['nodeSelector']['accelerator'] = job_spec.resources.required_gpu_types[0]

        return template

    def _apply_kubernetes_resources(self, resources: List[Dict[str, Any]]):
        """Apply Kubernetes resources to cluster"""
        for resource in resources:
            try:
                # Create temporary file for resource
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(resource, f)
                    temp_path = f.name

                # Apply resource
                cmd = ['kubectl', 'apply', '-f', temp_path]
                if self.kubeconfig_path:
                    cmd.extend(['--kubeconfig', self.kubeconfig_path])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    logger.error(f"Failed to apply resource: {result.stderr}")
                else:
                    logger.info(f"Applied {resource['kind']} {resource['metadata']['name']}")

                # Clean up temp file
                os.unlink(temp_path)

            except Exception as e:
                logger.error(f"Error applying resource: {e}")

    def _estimate_start_time(self, job_spec: TrainingJobSpec) -> str:
        """Estimate when job will start based on cluster capacity"""
        # Simple estimation - would be more sophisticated in practice
        estimated_delay_minutes = max(0, len(self.active_jobs) * 2)  # 2 minutes per queued job
        start_time = datetime.now() + timedelta(minutes=estimated_delay_minutes)
        return start_time.isoformat()

    def _estimate_resource_allocation(self, job_spec: TrainingJobSpec) -> Dict[str, Any]:
        """Estimate resource allocation for job"""
        return {
            'total_gpus': job_spec.resources.gpu_count,
            'gpus_per_node': job_spec.resources.gpu_count // job_spec.world_size,
            'estimated_nodes': job_spec.world_size,
            'memory_per_node_gb': job_spec.resources.memory_gb // job_spec.world_size
        }

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current status of training job"""
        return self.job_statuses.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel running training job"""
        if job_id not in self.active_jobs:
            return False

        try:
            if self.k8s_available:
                # Delete Kubernetes resources
                cmd = ['kubectl', 'delete', 'pytorchjob', job_id, '-n', self.namespace]
                if self.kubeconfig_path:
                    cmd.extend(['--kubeconfig', self.kubeconfig_path])

                subprocess.run(cmd, timeout=30)

            # Update status
            if job_id in self.job_statuses:
                self.job_statuses[job_id].state = JobState.CANCELLED
                self.job_statuses[job_id].end_time = datetime.now()

            # Remove from active jobs
            self.active_jobs.pop(job_id, None)

            logger.info(f"Cancelled job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def list_jobs(self) -> List[JobStatus]:
        """List all jobs and their statuses"""
        return list(self.job_statuses.values())


class SLURMClusterManager:
    """
    SLURM-based cluster manager for HPC environments

    Features:
    - Native SLURM job submission and management
    - Multi-node GPU allocation
    - Queue and partition management
    - Resource accounting and billing
    """

    def __init__(self, default_partition: str = "gpu"):
        self.default_partition = default_partition
        self.slurm_available = self._check_slurm_availability()

        # Job tracking
        self.active_jobs: Dict[str, TrainingJobSpec] = {}
        self.job_statuses: Dict[str, JobStatus] = {}
        self.slurm_job_map: Dict[str, str] = {}  # job_id -> slurm_job_id

    def _check_slurm_availability(self) -> bool:
        """Check if SLURM is available"""
        try:
            result = subprocess.run(['sinfo', '--version'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            logger.warning("SLURM not available, running in simulation mode")
            return False

    def submit_job(self, job_spec: TrainingJobSpec) -> Dict[str, Any]:
        """
        Submit distributed training job to SLURM

        Args:
            job_spec: Training job specification

        Returns:
            Submission result with job status
        """
        logger.info(f"Submitting job {job_spec.job_id} to SLURM")

        try:
            # Create SLURM batch script
            batch_script = self._create_slurm_batch_script(job_spec)

            if self.slurm_available:
                # Submit to SLURM
                slurm_job_id = self._submit_slurm_job(batch_script)
                self.slurm_job_map[job_spec.job_id] = slurm_job_id
            else:
                # Simulation mode
                slurm_job_id = f"sim_{int(time.time())}"
                self.slurm_job_map[job_spec.job_id] = slurm_job_id
                logger.info(f"[SIMULATION] Would submit SLURM job: {slurm_job_id}")

            # Track job
            self.active_jobs[job_spec.job_id] = job_spec
            self.job_statuses[job_spec.job_id] = JobStatus(
                job_id=job_spec.job_id,
                state=JobState.PENDING,
                start_time=datetime.now()
            )

            return {
                'success': True,
                'job_id': job_spec.job_id,
                'slurm_job_id': slurm_job_id,
                'queue_position': self._get_queue_position(slurm_job_id)
            }

        except Exception as e:
            logger.error(f"Failed to submit job {job_spec.job_id}: {e}")
            return {'success': False, 'error': str(e)}

    def _create_slurm_batch_script(self, job_spec: TrainingJobSpec) -> str:
        """Create SLURM batch script for distributed training"""
        nodes_required = max(1, job_spec.world_size // 8)  # Assume 8 GPUs per node
        gpus_per_node = min(8, job_spec.resources.gpu_count // nodes_required)

        script = f"""#!/bin/bash
#SBATCH --job-name={job_spec.job_id}
#SBATCH --partition={self.default_partition}
#SBATCH --nodes={nodes_required}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --cpus-per-task={job_spec.resources.cpu_cores // job_spec.world_size}
#SBATCH --mem={job_spec.resources.memory_gb // nodes_required}G
#SBATCH --time={job_spec.max_runtime_hours:02d}:00:00
#SBATCH --output={job_spec.job_id}_%j.out
#SBATCH --error={job_spec.job_id}_%j.err

# Environment setup
export WORLD_SIZE={job_spec.world_size}
export TENSOR_PARALLEL_SIZE={job_spec.tensor_parallel_size}
export PIPELINE_PARALLEL_SIZE={job_spec.pipeline_parallel_size}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=23456
export NCCL_DEBUG=INFO

# Custom environment variables
"""

        for key, value in job_spec.env_vars.items():
            script += f"export {key}={value}\n"

        script += f"""
# Load modules (customize based on your environment)
module load cuda/11.8
module load python/3.9

# Run distributed training
srun --ntasks={job_spec.world_size} \\
     --ntasks-per-node={gpus_per_node} \\
     --gres=gpu:{gpus_per_node} \\
     {' '.join(job_spec.command)}
"""

        return script

    def _submit_slurm_job(self, batch_script: str) -> str:
        """Submit job to SLURM and return job ID"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(batch_script)
            script_path = f.name

        try:
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Extract job ID from output (format: "Submitted batch job 12345")
                output = result.stdout.strip()
                slurm_job_id = output.split()[-1]
                return slurm_job_id
            else:
                raise RuntimeError(f"sbatch failed: {result.stderr}")

        finally:
            os.unlink(script_path)

    def _get_queue_position(self, slurm_job_id: str) -> int:
        """Get position in queue for SLURM job"""
        if not self.slurm_available:
            return 1  # Simulation

        try:
            result = subprocess.run(
                ['squeue', '-j', slurm_job_id, '--format=%P'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Count jobs ahead in queue (simplified)
                result = subprocess.run(
                    ['squeue', '-t', 'PD', '--format=%A'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    job_ids = [line.strip() for line in lines if line.strip()]

                    if slurm_job_id in job_ids:
                        return job_ids.index(slurm_job_id) + 1

            return 0  # Job not pending

        except Exception:
            return 0

    def cancel_job(self, job_id: str) -> bool:
        """Cancel SLURM job"""
        if job_id not in self.slurm_job_map:
            return False

        slurm_job_id = self.slurm_job_map[job_id]

        try:
            if self.slurm_available:
                result = subprocess.run(
                    ['scancel', slurm_job_id],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    logger.error(f"Failed to cancel SLURM job {slurm_job_id}: {result.stderr}")
                    return False

            # Update status
            if job_id in self.job_statuses:
                self.job_statuses[job_id].state = JobState.CANCELLED
                self.job_statuses[job_id].end_time = datetime.now()

            # Cleanup
            self.active_jobs.pop(job_id, None)
            self.slurm_job_map.pop(job_id, None)

            logger.info(f"Cancelled SLURM job {slurm_job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get SLURM cluster information"""
        info = {
            'total_nodes': 0,
            'available_nodes': 0,
            'total_gpus': 0,
            'available_gpus': 0,
            'partitions': []
        }

        if not self.slurm_available:
            # Simulation values
            info.update({
                'total_nodes': 128,
                'available_nodes': 100,
                'total_gpus': 1024,
                'available_gpus': 800,
                'partitions': ['gpu', 'cpu', 'bigmem']
            })
            return info

        try:
            # Get partition info
            result = subprocess.run(
                ['sinfo', '--format=%P,%D,%T'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        partition = parts[0].rstrip('*')  # Remove default indicator
                        nodes = int(parts[1])
                        state = parts[2]

                        if partition not in [p['name'] for p in info['partitions']]:
                            info['partitions'].append({'name': partition, 'nodes': nodes, 'state': state})

                        info['total_nodes'] += nodes
                        if state in ['idle', 'mixed']:
                            info['available_nodes'] += nodes

        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")

        return info


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
        self.current_replicas: Dict[str, int] = {}
        self.scaling_history: Dict[str, List[Dict]] = {}
        self.cooldown_period_seconds = 300  # 5 minutes
        self.last_scale_time: Dict[str, float] = {}

        # Metrics tracking
        self.metrics_history: Dict[str, List[Dict]] = {}

    def update_metrics(self, job_id: str, metrics: Dict[str, float]):
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

    def should_scale(self, job_id: str) -> Tuple[bool, str, int]:
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
        avg_throughput = np.mean([m['throughput'] for m in recent_metrics])

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
        self.failure_history: Dict[str, List[Dict]] = {}
        self.recovery_attempts: Dict[str, int] = {}

        # Health monitoring
        self.health_checks: Dict[str, Dict] = {}
        self.unhealthy_nodes: Set[str] = set()

    def detect_failure(self, job_id: str, error_info: Dict[str, Any]) -> FailureType:
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
        error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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

    def _execute_recovery(self, job_id: str, recovery_plan: Dict[str, Any]) -> bool:
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

    def _find_latest_checkpoint(self, job_id: str) -> Optional[str]:
        """Find latest checkpoint for job"""
        # Would scan checkpoint directory for latest checkpoint
        # For simulation, return a placeholder
        checkpoint_dir = f"/checkpoints/{job_id}"
        return f"{checkpoint_dir}/latest.pt"  # Placeholder

    def get_failure_statistics(self) -> Dict[str, Any]:
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


# Factory functions for easy orchestrator creation
def create_kubernetes_orchestrator(
    namespace: str = "ml-training",
    kubeconfig_path: Optional[str] = None
) -> KubernetesDistributedOrchestrator:
    """Create Kubernetes orchestrator with default configuration"""
    return KubernetesDistributedOrchestrator(namespace, kubeconfig_path)


def create_slurm_manager(partition: str = "gpu") -> SLURMClusterManager:
    """Create SLURM cluster manager with default configuration"""
    return SLURMClusterManager(partition)


def create_auto_scaling_manager(
    min_replicas: int = 1,
    max_replicas: int = 100
) -> AutoScalingManager:
    """Create auto-scaling manager with default configuration"""
    return AutoScalingManager(min_replicas, max_replicas)


def create_fault_tolerance_manager(
    checkpoint_interval_minutes: int = 30
) -> FaultToleranceManager:
    """Create fault tolerance manager with default configuration"""
    return FaultToleranceManager(checkpoint_interval_minutes)


# Import numpy at module level
import numpy as np