#!/usr/bin/env python3
"""
Comprehensive tests for distributed scale training and inference components

Tests the large-scale distributed training and inference framework including:
- Multi-node training management
- Distributed inference serving
- Communication optimization
- Hardware adaptation
- Orchestration and fault tolerance
"""

import pytest
import asyncio
import time
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernel_pytorch.distributed_scale import (
    # Multi-node training
    MultiNodeTrainingManager,
    AdvancedFSDPManager,
    HeterogenousClusterManager,
    create_multi_node_trainer,

    # Large-scale inference
    DistributedInferenceServer,
    AdaptiveLoadBalancer,
    MemoryEfficientScheduler,
    create_inference_cluster,

    # Communication optimization
    AdvancedCollectiveOps,
    NetworkTopologyOptimizer,
    BandwidthAwareScheduler,
    CommunicationProfiler,

    # Hardware adaptation
    HardwareTopologyManager,
    DeviceMeshOptimizer,
    ThermalAwareScheduler,
    PowerEfficiencyOptimizer,

    # Orchestration
    KubernetesDistributedOrchestrator,
    SLURMClusterManager,
    AutoScalingManager,
    FaultToleranceManager
)

from kernel_pytorch.distributed_scale.multi_node_training import (
    ClusterConfig, TrainingConfig, FSDPConfig
)
from kernel_pytorch.distributed_scale.large_scale_inference import (
    InferenceServerConfig, LoadBalancingStrategy, BatchingStrategy
)
from kernel_pytorch.distributed_scale.communication_optimization import (
    CommunicationPattern, CompressionMethod, NetworkTopology, CollectiveOpConfig
)
from kernel_pytorch.distributed_scale.hardware_adaptation import (
    HardwareVendor, DeviceCapability, ThermalState, NodeTopology, ClusterTopology
)
from kernel_pytorch.distributed_scale.orchestration import (
    JobState, TrainingJobSpec, ResourceRequirement, JobStatus, FailureType
)


class TestMultiNodeTraining:
    """Test multi-node training components"""

    @pytest.fixture
    def cluster_config(self):
        """Create test cluster configuration"""
        return ClusterConfig(
            world_size=16,
            node_count=2,
            gpus_per_node=8,
            master_addr="localhost",
            master_port=29500
        )

    @pytest.fixture
    def training_config(self):
        """Create test training configuration"""
        return TrainingConfig(
            model_name="test_model",
            batch_size=32,
            learning_rate=1e-4,
            max_epochs=10,
            checkpoint_interval=5
        )

    def test_multi_node_training_manager_init(self, cluster_config, training_config):
        """Test MultiNodeTrainingManager initialization"""
        # Mock model for testing
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        manager = MultiNodeTrainingManager(
            model=model,
            cluster_config=cluster_config,
            training_config=asdict(training_config)
        )

        assert manager.cluster_config.world_size == 16
        assert manager.cluster_config.node_count == 2
        assert manager.model is not None
        assert manager.is_initialized is False

    def test_fsdp_manager_configuration(self, cluster_config):
        """Test FSDP manager configuration"""
        model = nn.Linear(256, 128)

        fsdp_config = FSDPConfig(
            sharding_strategy="FULL_SHARD",
            cpu_offload=True,
            mixed_precision=True,
            auto_wrap_policy="transformer_auto_wrap_policy"
        )

        # Mock DeviceMesh since it requires actual CUDA setup
        with patch('torch.distributed._tensor.DeviceMesh') as mock_mesh:
            mock_mesh.return_value = Mock()

            fsdp_manager = AdvancedFSDPManager(
                model=model,
                device_mesh=mock_mesh.return_value,
                config=fsdp_config
            )

            assert fsdp_manager.config.sharding_strategy == "FULL_SHARD"
            assert fsdp_manager.config.cpu_offload is True

    def test_heterogeneous_cluster_manager(self, cluster_config):
        """Test heterogeneous cluster management"""
        # Mock device capabilities
        device_capabilities = {
            0: {"memory_gb": 40, "compute_capability": "8.0", "vendor": "nvidia"},
            1: {"memory_gb": 24, "compute_capability": "7.5", "vendor": "nvidia"},
            2: {"memory_gb": 16, "compute_capability": "rdna2", "vendor": "amd"}
        }

        manager = HeterogenousClusterManager(
            cluster_config=cluster_config,
            device_capabilities=device_capabilities
        )

        # Test device grouping
        groups = manager.create_device_groups()
        assert len(groups) > 0

        # Test adaptive sharding
        sharding_plan = manager.create_adaptive_sharding_plan(world_size=16)
        assert "tensor_parallel_groups" in sharding_plan
        assert "data_parallel_groups" in sharding_plan

    def test_create_multi_node_trainer(self, cluster_config, training_config):
        """Test multi-node trainer factory function"""
        model = nn.Linear(128, 64)

        trainer = create_multi_node_trainer(
            model=model,
            cluster_config=cluster_config,
            training_config=asdict(training_config)
        )

        assert trainer is not None
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'evaluate')


class TestLargeScaleInference:
    """Test large-scale inference components"""

    @pytest.fixture
    def inference_config(self):
        """Create test inference server configuration"""
        return InferenceServerConfig(
            model_path="test_model",
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            max_num_batched_tokens=1024,
            load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
            batching_strategy=BatchingStrategy.CONTINUOUS
        )

    def test_distributed_inference_server_init(self, inference_config):
        """Test DistributedInferenceServer initialization"""
        with patch('torch.distributed._tensor.DeviceMesh') as mock_mesh:
            server = DistributedInferenceServer(
                config=inference_config,
                device_mesh=mock_mesh.return_value
            )

            assert server.config.model_path == "test_model"
            assert server.config.tensor_parallel_size == 2
            assert server.is_running is False

    @pytest.mark.asyncio
    async def test_inference_generation(self, inference_config):
        """Test inference generation"""
        with patch('torch.distributed._tensor.DeviceMesh') as mock_mesh:
            server = DistributedInferenceServer(
                config=inference_config,
                device_mesh=mock_mesh.return_value
            )

            # Test generation
            response = await server.generate(
                prompt="Hello, world!",
                sampling_params={"temperature": 0.7, "max_tokens": 50}
            )

            assert isinstance(response, str)
            assert len(response) > 0

    def test_adaptive_load_balancer(self):
        """Test adaptive load balancer"""
        balancer = AdaptiveLoadBalancer(LoadBalancingStrategy.ADAPTIVE)

        # Add test servers
        balancer.add_server("server1", "http://localhost:8000")
        balancer.add_server("server2", "http://localhost:8001")

        # Test server selection
        selected_server = balancer.select_server()
        assert selected_server in ["server1", "server2"]

    def test_memory_efficient_scheduler(self):
        """Test memory-efficient scheduler"""
        scheduler = MemoryEfficientScheduler(
            max_memory_gb=32.0,
            enable_prefix_caching=True
        )

        assert scheduler.max_memory_gb == 32.0
        assert scheduler.enable_prefix_caching is True
        assert len(scheduler.pending_requests) == 0

    @pytest.mark.asyncio
    async def test_memory_scheduler_request_handling(self):
        """Test memory scheduler request handling"""
        scheduler = MemoryEfficientScheduler(max_memory_gb=16.0)

        # Schedule a request
        await scheduler.schedule_request("req1", estimated_tokens=100)

        # Should be in active requests
        assert "req1" in scheduler.active_requests

        # Complete request
        scheduler.complete_request("req1")

        # Should be removed from active requests
        assert "req1" not in scheduler.active_requests

    def test_create_inference_cluster(self):
        """Test inference cluster creation"""
        servers = create_inference_cluster(
            model_path="test_model",
            num_servers=2,
            tensor_parallel_size=1
        )

        assert len(servers) == 2
        assert all(isinstance(server, DistributedInferenceServer) for server in servers)


class TestCommunicationOptimization:
    """Test communication optimization components"""

    @pytest.fixture
    def network_topology(self):
        """Create test network topology"""
        return NetworkTopology(
            node_count=4,
            gpus_per_node=8,
            intra_node_bandwidth_gbps=600.0,
            inter_node_bandwidth_gbps=200.0,
            network_latency_us=2.0
        )

    @pytest.fixture
    def collective_config(self):
        """Create test collective operations configuration"""
        return CollectiveOpConfig(
            pattern=CommunicationPattern.ADAPTIVE,
            compression=CompressionMethod.ADAPTIVE,
            chunk_size_mb=64,
            overlap_computation=True
        )

    def test_advanced_collective_ops_init(self, network_topology, collective_config):
        """Test AdvancedCollectiveOps initialization"""
        ops = AdvancedCollectiveOps(
            world_size=32,
            rank=0,
            topology=network_topology,
            config=collective_config
        )

        assert ops.world_size == 32
        assert ops.rank == 0
        assert len(ops.communication_groups) > 0

    @pytest.mark.asyncio
    async def test_optimized_allreduce(self, network_topology, collective_config):
        """Test optimized allreduce operation"""
        ops = AdvancedCollectiveOps(
            world_size=8,
            rank=0,
            topology=network_topology,
            config=collective_config
        )

        # Test tensor
        tensor = torch.randn(1024, 512)

        # Mock the actual distributed operation
        with patch('torch.distributed.all_reduce'):
            result = await ops.allreduce_optimized(tensor, op='sum')

            assert result.shape == tensor.shape
            assert result.dtype == tensor.dtype

    @pytest.mark.asyncio
    async def test_optimized_allgather(self, network_topology, collective_config):
        """Test optimized allgather operation"""
        ops = AdvancedCollectiveOps(
            world_size=4,
            rank=0,
            topology=network_topology,
            config=collective_config
        )

        tensor = torch.randn(256, 128)

        with patch('torch.distributed.all_gather'):
            result = await ops.allgather_optimized(tensor)

            # Result should be concatenated from all ranks
            assert result.shape[0] == tensor.shape[0] * ops.world_size

    def test_network_topology_optimizer(self):
        """Test network topology optimizer"""
        optimizer = NetworkTopologyOptimizer(world_size=16, rank=0)

        # Test topology discovery
        topology = optimizer.discovered_topology
        assert topology.node_count > 0
        assert topology.gpus_per_node > 0

        # Test communication pattern optimization
        participants = list(range(8))
        optimization = optimizer.optimize_communication_pattern(
            operation='allreduce',
            tensor_size=1024*1024,
            participants=participants
        )

        assert 'optimal_pattern' in optimization
        assert 'expected_time_ms' in optimization

    def test_bandwidth_aware_scheduler(self):
        """Test bandwidth-aware scheduler"""
        topology_optimizer = NetworkTopologyOptimizer(world_size=8, rank=0)
        scheduler = BandwidthAwareScheduler(topology_optimizer)

        # Schedule communication
        result = scheduler.schedule_communication(
            operation_id="test_comm",
            operation_type="allreduce",
            participants=[0, 1, 2, 3],
            tensor_size=1024*1024
        )

        assert 'schedule_time_ms' in result
        assert 'optimization' in result

    def test_communication_profiler(self):
        """Test communication profiler"""
        profiler = CommunicationProfiler()

        # Test profiling context
        with profiler.profile_operation('allreduce', participants=[0, 1, 2, 3]):
            time.sleep(0.01)  # Simulate operation

        # Check that operation was recorded
        assert len(profiler.operation_history) == 1
        assert profiler.operation_history[0]['type'] == 'allreduce'

        # Test bottleneck identification
        bottlenecks = profiler.identify_bottlenecks()
        assert isinstance(bottlenecks, list)


class TestHardwareAdaptation:
    """Test hardware adaptation components"""

    def test_hardware_topology_manager_init(self):
        """Test HardwareTopologyManager initialization"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=4):

            manager = HardwareTopologyManager(enable_monitoring=False)

            assert manager.cluster_topology is not None
            assert manager.cluster_topology.total_devices > 0

    def test_topology_discovery(self):
        """Test hardware topology discovery"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.get_device_properties') as mock_props:

            # Mock device properties
            mock_props.return_value = Mock(
                name="Test GPU",
                total_memory=17179869184,  # 16GB
                multi_processor_count=108,
                major=8,
                minor=0
            )

            manager = HardwareTopologyManager(enable_monitoring=False)
            topology = manager.discover_topology()

            assert topology.total_devices == 2
            assert HardwareVendor.NVIDIA in topology.vendor_distribution

    def test_device_mesh_optimizer(self):
        """Test device mesh optimizer"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=8):

            topology_manager = HardwareTopologyManager(enable_monitoring=False)
            optimizer = DeviceMeshOptimizer(topology_manager)

            # Test mesh creation
            with patch('torch.distributed._tensor.DeviceMesh') as mock_mesh:
                mesh = optimizer.create_optimal_mesh(
                    world_size=8,
                    tensor_parallel_size=2,
                    pipeline_parallel_size=2,
                    data_parallel_size=2
                )

                mock_mesh.assert_called()

    def test_thermal_aware_scheduler(self):
        """Test thermal-aware scheduler"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=4):

            topology_manager = HardwareTopologyManager(enable_monitoring=False)
            scheduler = ThermalAwareScheduler(
                topology_manager=topology_manager,
                thermal_threshold=85.0
            )

            # Test job scheduling
            result = scheduler.schedule_job(
                job_id="test_job",
                required_devices=2,
                estimated_power_per_device=250.0
            )

            assert 'success' in result

    def test_power_efficiency_optimizer(self):
        """Test power efficiency optimizer"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=4):

            topology_manager = HardwareTopologyManager(enable_monitoring=False)
            optimizer = PowerEfficiencyOptimizer(
                topology_manager=topology_manager,
                power_budget_w=2000.0
            )

            # Test power distribution optimization
            result = optimizer.optimize_power_distribution(
                workload_priorities={"job1": 1.0, "job2": 0.5},
                efficiency_target=0.8
            )

            assert 'current_power_w' in result
            assert 'estimated_savings_w' in result


class TestOrchestration:
    """Test orchestration components"""

    @pytest.fixture
    def resource_requirement(self):
        """Create test resource requirement"""
        return ResourceRequirement(
            gpu_count=8,
            gpu_memory_gb=32,
            cpu_cores=32,
            memory_gb=128,
            storage_gb=500
        )

    @pytest.fixture
    def job_spec(self, resource_requirement):
        """Create test job specification"""
        return TrainingJobSpec(
            job_id="test-job-001",
            name="test-training",
            image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel",
            command=["python", "train.py"],
            resources=resource_requirement,
            world_size=8,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2
        )

    def test_kubernetes_orchestrator_init(self):
        """Test Kubernetes orchestrator initialization"""
        orchestrator = KubernetesDistributedOrchestrator(
            namespace="test-ml",
            kubeconfig_path=None
        )

        assert orchestrator.namespace == "test-ml"
        assert len(orchestrator.active_jobs) == 0

    def test_job_submission_kubernetes(self, job_spec):
        """Test job submission to Kubernetes"""
        orchestrator = KubernetesDistributedOrchestrator(namespace="test-ml")

        # Mock kubectl availability
        with patch.object(orchestrator, 'k8s_available', True), \
             patch.object(orchestrator, '_apply_kubernetes_resources'):

            result = orchestrator.submit_job(job_spec)

            assert result['success'] is True
            assert result['job_id'] == "test-job-001"
            assert job_spec.job_id in orchestrator.active_jobs

    def test_slurm_cluster_manager_init(self):
        """Test SLURM cluster manager initialization"""
        manager = SLUMClusterManager(default_partition="gpu")

        assert manager.default_partition == "gpu"
        assert len(manager.active_jobs) == 0

    def test_job_submission_slurm(self, job_spec):
        """Test job submission to SLURM"""
        manager = SLUMClusterManager()

        # Mock SLURM availability
        with patch.object(manager, 'slurm_available', True), \
             patch.object(manager, '_submit_slurm_job', return_value="12345"):

            result = manager.submit_job(job_spec)

            assert result['success'] is True
            assert result['slurm_job_id'] == "12345"

    def test_auto_scaling_manager(self):
        """Test auto-scaling manager"""
        manager = AutoScalingManager(
            min_replicas=1,
            max_replicas=10,
            target_utilization=0.7
        )

        # Update metrics
        manager.update_metrics("job1", {
            'gpu_utilization': 85.0,
            'memory_utilization': 60.0,
            'throughput': 100.0
        })

        # Test scaling decision
        should_scale, direction, target = manager.should_scale("job1")

        # With high utilization, should recommend scaling up
        assert isinstance(should_scale, bool)
        assert direction in ["scale_up", "scale_down", "no_action", "no_metrics", "cooldown", "insufficient_data"]

    def test_fault_tolerance_manager(self):
        """Test fault tolerance manager"""
        manager = FaultToleranceManager(checkpoint_interval_minutes=15)

        # Test failure detection
        error_info = {
            'message': 'CUDA out of memory',
            'exit_code': 1
        }

        failure_type = manager.detect_failure("job1", error_info)
        assert failure_type == FailureType.OOM_FAILURE

        # Test failure handling
        result = manager.handle_failure("job1", failure_type, error_info)

        assert 'recovered' in result
        assert isinstance(result['recovered'], bool)

    def test_job_status_tracking(self, job_spec):
        """Test job status tracking"""
        orchestrator = KubernetesDistributedOrchestrator()

        # Submit job
        with patch.object(orchestrator, '_apply_kubernetes_resources'):
            result = orchestrator.submit_job(job_spec)

            assert result['success'] is True

        # Get status
        status = orchestrator.get_job_status(job_spec.job_id)
        assert status is not None
        assert status.job_id == job_spec.job_id
        assert status.state == JobState.PENDING

    def test_job_cancellation(self, job_spec):
        """Test job cancellation"""
        orchestrator = KubernetesDistributedOrchestrator()

        # Submit job first
        with patch.object(orchestrator, '_apply_kubernetes_resources'):
            orchestrator.submit_job(job_spec)

        # Cancel job
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            success = orchestrator.cancel_job(job_spec.job_id)

            assert success is True
            assert job_spec.job_id not in orchestrator.active_jobs


class TestIntegration:
    """Test integration between different components"""

    @pytest.mark.asyncio
    async def test_end_to_end_training_setup(self):
        """Test end-to-end training setup"""
        # Create cluster configuration
        cluster_config = ClusterConfig(
            world_size=8,
            node_count=2,
            gpus_per_node=4
        )

        training_config = TrainingConfig(
            model_name="test_model",
            batch_size=16,
            learning_rate=1e-4
        )

        # Create model
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        # Create training manager
        with patch('torch.distributed._tensor.DeviceMesh'):
            trainer = create_multi_node_trainer(
                model=model,
                cluster_config=cluster_config,
                training_config=asdict(training_config)
            )

            assert trainer is not None

    @pytest.mark.asyncio
    async def test_distributed_inference_integration(self):
        """Test distributed inference integration"""
        # Create inference cluster
        servers = create_inference_cluster(
            model_path="test_model",
            num_servers=2,
            tensor_parallel_size=1,
            config_overrides={
                'max_model_len': 1024,
                'gpu_memory_utilization': 0.7
            }
        )

        assert len(servers) == 2

        # Test generation on first server
        response = await servers[0].generate(
            prompt="Test prompt",
            sampling_params={'temperature': 0.7}
        )

        assert isinstance(response, str)

    def test_hardware_aware_orchestration(self):
        """Test hardware-aware orchestration"""
        # Create hardware topology manager
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=8):

            topology_manager = HardwareTopologyManager(enable_monitoring=False)

            # Create device mesh optimizer
            mesh_optimizer = DeviceMeshOptimizer(topology_manager)

            # Create thermal-aware scheduler
            thermal_scheduler = ThermalAwareScheduler(
                topology_manager=topology_manager,
                thermal_threshold=80.0
            )

            # Test job scheduling with thermal awareness
            result = thermal_scheduler.schedule_job(
                job_id="thermal_test",
                required_devices=4,
                estimated_power_per_device=200.0
            )

            assert 'success' in result

    def test_communication_optimization_integration(self):
        """Test communication optimization integration"""
        # Create network topology
        topology = NetworkTopology(
            node_count=2,
            gpus_per_node=4,
            intra_node_bandwidth_gbps=600.0,
            inter_node_bandwidth_gbps=200.0
        )

        # Create collective operations
        ops = AdvancedCollectiveOps(
            world_size=8,
            rank=0,
            topology=topology
        )

        # Create topology optimizer
        topo_optimizer = NetworkTopologyOptimizer(world_size=8, rank=0)

        # Test communication pattern optimization
        optimization = topo_optimizer.optimize_communication_pattern(
            operation='allreduce',
            tensor_size=1024*1024,
            participants=list(range(8))
        )

        assert optimization['optimal_pattern'] in [pattern.value for pattern in CommunicationPattern]


def main():
    """Run all distributed scale tests"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])


if __name__ == "__main__":
    main()