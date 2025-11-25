"""
Cluster Management Systems

Kubernetes and SLURM cluster management for distributed training:
- Kubernetes-native orchestration with custom resources
- SLURM integration for HPC environments
- Resource allocation and job scheduling
- Service mesh integration for communication
"""

import asyncio
import time
import logging
import json
import yaml
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field, asdict
import threading
from datetime import datetime, timedelta
import tempfile
import os

from .job_management import (
    JobState, ResourceType, FailureType,
    ResourceRequirement, TrainingJobSpec, JobStatus, ClusterNode
)

logger = logging.getLogger(__name__)


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
    - Native SLURM job submission and monitoring
    - Multi-node GPU allocation optimization
    - Resource reservation and priority queuing
    - Integration with existing HPC workflows
    """

    def __init__(
        self,
        partition: str = "gpu",
        default_partition: Optional[str] = None,  # For test compatibility
        account: Optional[str] = None,
        default_time_limit: str = "24:00:00"
    ):
        # Use default_partition if provided for test compatibility
        self.partition = default_partition or partition
        self.default_partition = default_partition or partition  # Store for test compatibility
        self.account = account
        self.default_time_limit = default_time_limit

        # Job tracking
        self.active_jobs: Dict[str, TrainingJobSpec] = {}
        self.job_statuses: Dict[str, JobStatus] = {}
        self.slurm_job_ids: Dict[str, str] = {}  # our_id -> slurm_id

        # SLURM availability
        self.slurm_available = self._check_slurm_availability()

        # Monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

    def _check_slurm_availability(self) -> bool:
        """Check if SLURM commands are available"""
        try:
            result = subprocess.run(['sinfo', '--version'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            logger.warning("SLURM commands not available, running in simulation mode")
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

        # Validate job specification
        validation_result = self._validate_job_spec(job_spec)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': f"Job validation failed: {validation_result['errors']}"
            }

        try:
            # Create SLURM batch script
            batch_script = self._create_slurm_script(job_spec)

            if self.slurm_available:
                # Submit to SLURM
                slurm_job_id = self._submit_slurm_job(batch_script)
            else:
                # Simulation mode
                slurm_job_id = f"sim_{job_spec.job_id}"
                logger.info(f"[SIMULATION] Would submit SLURM job: {len(batch_script)} line script")

            # Track job
            self.active_jobs[job_spec.job_id] = job_spec
            self.slurm_job_ids[job_spec.job_id] = slurm_job_id
            self.job_statuses[job_spec.job_id] = JobStatus(
                job_id=job_spec.job_id,
                state=JobState.PENDING,
                start_time=datetime.now()
            )

            return {
                'success': True,
                'job_id': job_spec.job_id,
                'slurm_job_id': slurm_job_id,
                'estimated_start_time': self._estimate_start_time(job_spec),
                'allocated_resources': self._estimate_resource_allocation(job_spec)
            }

        except Exception as e:
            logger.error(f"Failed to submit job {job_spec.job_id}: {e}")
            return {'success': False, 'error': str(e)}

    def _validate_job_spec(self, job_spec: TrainingJobSpec) -> Dict[str, Any]:
        """Validate training job specification for SLURM"""
        errors = []

        # Check parallelism configuration
        total_required = (job_spec.tensor_parallel_size *
                         job_spec.pipeline_parallel_size *
                         (job_spec.data_parallel_size or 1))
        if total_required != job_spec.world_size:
            errors.append(f"Parallelism mismatch: {total_required} != {job_spec.world_size}")

        # Check command
        if not job_spec.command:
            errors.append("Command not specified")

        return {'valid': len(errors) == 0, 'errors': errors}

    def _create_slurm_script(self, job_spec: TrainingJobSpec) -> str:
        """Create SLURM batch script for training job"""

        # Calculate nodes and tasks
        nodes = job_spec.world_size
        tasks_per_node = 1
        cpus_per_task = job_spec.resources.cpu_cores // job_spec.world_size
        gpus_per_node = job_spec.resources.gpu_count // job_spec.world_size

        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_spec.job_id}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --ntasks-per-node={tasks_per_node}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={job_spec.resources.memory_gb // job_spec.world_size}G",
            f"#SBATCH --gres=gpu:{gpus_per_node}",
            f"#SBATCH --time={self.default_time_limit}",
            f"#SBATCH --output={job_spec.job_id}_%j.out",
            f"#SBATCH --error={job_spec.job_id}_%j.err"
        ]

        # Add account if specified
        if self.account:
            script_lines.append(f"#SBATCH --account={self.account}")

        # Add GPU type constraints
        if job_spec.resources.required_gpu_types:
            gpu_type = job_spec.resources.required_gpu_types[0]
            script_lines.append(f"#SBATCH --constraint={gpu_type}")

        script_lines.extend([
            "",
            "# Environment setup",
            "module purge",
            "module load cuda/12.0",
            "module load python/3.10",
            "",
            "# Set distributed training environment",
            f"export WORLD_SIZE={job_spec.world_size}",
            f"export TENSOR_PARALLEL_SIZE={job_spec.tensor_parallel_size}",
            f"export PIPELINE_PARALLEL_SIZE={job_spec.pipeline_parallel_size}",
            "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)",
            "export MASTER_PORT=23456",
            "export NCCL_DEBUG=INFO",
            "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
            ""
        ])

        # Add custom environment variables
        for key, value in job_spec.env_vars.items():
            script_lines.append(f"export {key}={value}")

        script_lines.extend([
            "",
            "# Launch distributed training",
            f"srun {''.join(job_spec.command)}"
        ])

        return '\n'.join(script_lines)

    def _submit_slurm_job(self, batch_script: str) -> str:
        """Submit batch script to SLURM"""
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(batch_script)
            script_path = f.name

        try:
            # Submit job
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                # Extract job ID from output
                # Output format: "Submitted batch job 12345"
                slurm_job_id = result.stdout.strip().split()[-1]
                logger.info(f"Submitted SLURM job {slurm_job_id}")
                return slurm_job_id
            else:
                raise Exception(f"sbatch failed: {result.stderr}")

        finally:
            # Clean up script file
            os.unlink(script_path)

    def _estimate_start_time(self, job_spec: TrainingJobSpec) -> str:
        """Estimate when job will start based on SLURM queue"""
        # Simple estimation - would query squeue in practice
        estimated_delay_minutes = max(0, len(self.active_jobs) * 5)  # 5 minutes per queued job
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
        status = self.job_statuses.get(job_id)
        if status and self.slurm_available:
            # Update status from SLURM
            slurm_job_id = self.slurm_job_ids.get(job_id)
            if slurm_job_id:
                self._update_job_status_from_slurm(job_id, slurm_job_id)

        return self.job_statuses.get(job_id)

    def _update_job_status_from_slurm(self, job_id: str, slurm_job_id: str):
        """Update job status by querying SLURM"""
        try:
            result = subprocess.run(
                ['squeue', '--job', slurm_job_id, '--format=%T', '--noheader'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                slurm_state = result.stdout.strip().upper()

                # Map SLURM states to our states
                state_mapping = {
                    'PENDING': JobState.PENDING,
                    'RUNNING': JobState.RUNNING,
                    'COMPLETED': JobState.COMPLETED,
                    'FAILED': JobState.FAILED,
                    'CANCELLED': JobState.CANCELLED,
                    'TIMEOUT': JobState.FAILED
                }

                if slurm_state in state_mapping:
                    self.job_statuses[job_id].state = state_mapping[slurm_state]

        except Exception as e:
            logger.warning(f"Failed to update status for job {job_id}: {e}")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel running SLURM job"""
        if job_id not in self.active_jobs:
            return False

        slurm_job_id = self.slurm_job_ids.get(job_id)
        if not slurm_job_id:
            return False

        try:
            if self.slurm_available:
                # Cancel SLURM job
                result = subprocess.run(
                    ['scancel', slurm_job_id],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode != 0:
                    logger.error(f"Failed to cancel SLURM job {slurm_job_id}: {result.stderr}")
                    return False

            # Update status
            if job_id in self.job_statuses:
                self.job_statuses[job_id].state = JobState.CANCELLED
                self.job_statuses[job_id].end_time = datetime.now()

            # Remove from active jobs
            self.active_jobs.pop(job_id, None)
            self.slurm_job_ids.pop(job_id, None)

            logger.info(f"Cancelled job {job_id} (SLURM ID: {slurm_job_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def list_jobs(self) -> List[JobStatus]:
        """List all jobs and their statuses"""
        # Update all statuses from SLURM
        if self.slurm_available:
            for job_id, slurm_job_id in self.slurm_job_ids.items():
                self._update_job_status_from_slurm(job_id, slurm_job_id)

        return list(self.job_statuses.values())

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information from SLURM"""
        if not self.slurm_available:
            return {'simulation_mode': True}

        try:
            # Get node information
            result = subprocess.run(
                ['sinfo', '--format=%N,%C,%m,%G,%T', '--noheader'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                nodes = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 5:
                            nodes.append({
                                'name': parts[0],
                                'cpus': parts[1],
                                'memory': parts[2],
                                'gpus': parts[3],
                                'state': parts[4]
                            })

                return {
                    'partition': self.partition,
                    'nodes': nodes,
                    'total_nodes': len(nodes)
                }

        except Exception as e:
            logger.warning(f"Failed to get cluster info: {e}")

        return {'error': 'Could not retrieve cluster information'}