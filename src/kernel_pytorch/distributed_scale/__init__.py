"""
Large-Scale Distributed Training and Inference Framework (2025)

Advanced framework for training and serving models across thousands of GPUs using:
- Enhanced FSDP2 with DTensor integration
- Multi-node PyGraph optimization
- Adaptive communication patterns
- Advanced inference serving with vLLM integration
- Heterogeneous hardware support

Designed for clusters ranging from 8 GPUs to 32,000+ GPUs across multiple nodes.
"""

from .multi_node_training import (
    MultiNodeTrainingManager,
    AdvancedFSDPManager,
    HeterogenousClusterManager,
    create_multi_node_trainer
)

from .large_scale_inference import (
    DistributedInferenceServer,
    AdaptiveLoadBalancer,
    MemoryEfficientScheduler,
    create_inference_cluster
)

from .communication_optimization import (
    AdvancedCollectiveOps,
    NetworkTopologyOptimizer,
    BandwidthAwareScheduler,
    CommunicationProfiler
)

from .hardware_adaptation import (
    HardwareTopologyManager,
    DeviceMeshOptimizer,
    ThermalAwareScheduler,
    PowerEfficiencyOptimizer
)

from .orchestration import (
    KubernetesDistributedOrchestrator,
    SLURMClusterManager,
    AutoScalingManager,
    FaultToleranceManager
)

__all__ = [
    # Multi-node training
    'MultiNodeTrainingManager',
    'AdvancedFSDPManager',
    'HeterogenousClusterManager',
    'create_multi_node_trainer',

    # Large-scale inference
    'DistributedInferenceServer',
    'AdaptiveLoadBalancer',
    'MemoryEfficientScheduler',
    'create_inference_cluster',

    # Communication optimization
    'AdvancedCollectiveOps',
    'NetworkTopologyOptimizer',
    'BandwidthAwareScheduler',
    'CommunicationProfiler',

    # Hardware adaptation
    'HardwareTopologyManager',
    'DeviceMeshOptimizer',
    'ThermalAwareScheduler',
    'PowerEfficiencyOptimizer',

    # Orchestration
    'KubernetesDistributedOrchestrator',
    'SLURMClusterManager',
    'AutoScalingManager',
    'FaultToleranceManager'
]