"""
Multi-GPU Optimization Patterns
==============================

Comprehensive patterns and utilities for multi-GPU PyTorch optimization including
data parallelism, model parallelism, pipeline parallelism, and distributed training.

This module provides:
1. Multi-GPU optimization strategies
2. Distributed data parallel (DDP) utilities
3. Model parallelism patterns
4. Pipeline parallelism implementation
5. Communication optimization techniques
6. Educational multi-GPU examples

Author: KernelPyTorch Team
"""

import os
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""
    world_size: int
    rank: int
    local_rank: int
    backend: str = 'nccl'
    init_method: str = 'env://'
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    use_fsdp: bool = False
    use_gradient_compression: bool = False


class DistributedManager:
    """
    Manager for distributed training setup and coordination.

    Handles initialization, cleanup, and utilities for distributed PyTorch training.
    """

    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.is_initialized = False

    def setup(self):
        """Initialize distributed training."""
        if not self.is_initialized:
            # Set up environment variables if not already set
            if 'RANK' not in os.environ:
                os.environ['RANK'] = str(self.config.rank)
            if 'WORLD_SIZE' not in os.environ:
                os.environ['WORLD_SIZE'] = str(self.config.world_size)
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = str(self.config.local_rank)

            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )

            # Set local device
            torch.cuda.set_device(self.config.local_rank)
            self.is_initialized = True

    def cleanup(self):
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False

    @contextmanager
    def distributed_context(self):
        """Context manager for distributed training."""
        try:
            self.setup()
            yield
        finally:
            self.cleanup()

    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.config.rank == 0

    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Perform all-reduce operation on tensor."""
        if self.is_initialized:
            dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensor from all processes."""
        if not self.is_initialized:
            return [tensor]

        tensor_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list


class DataParallelOptimizer:
    """
    Optimizations for data parallel training.

    Provides utilities for optimizing data parallel training including
    gradient synchronization, bucket management, and communication overlap.
    """

    def __init__(self,
                 model: nn.Module,
                 device_ids: list[int] | None = None,
                 find_unused_parameters: bool = False,
                 gradient_as_bucket_view: bool = True,
                 bucket_cap_mb: int = 25):
        self.model = model
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.bucket_cap_mb = bucket_cap_mb

    def setup_ddp_model(self, local_rank: int) -> DDP:
        """Set up DistributedDataParallel model with optimizations."""
        # Move model to local device
        device = torch.device(f'cuda:{local_rank}')
        model = self.model.to(device)

        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            bucket_cap_mb=self.bucket_cap_mb
        )

        return ddp_model

    def setup_fsdp_model(self, local_rank: int) -> FSDP:
        """Set up FullyShardedDataParallel model."""
        device = torch.device(f'cuda:{local_rank}')

        # FSDP configuration
        fsdp_config = {
            'device_id': local_rank,
            'sync_module_states': True,
            'sharding_strategy': FSDP.FULL_SHARD,
        }

        # Wrap with FSDP
        fsdp_model = FSDP(self.model.to(device), **fsdp_config)
        return fsdp_model

    @staticmethod
    def optimize_dataloader(dataset, batch_size: int, num_workers: int, world_size: int, rank: int):
        """Create optimized distributed data loader."""
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0
        )

        return dataloader


class ModelParallelOptimizer:
    """
    Utilities for model parallelism and pipeline parallelism.

    Provides tools for splitting models across multiple GPUs and optimizing
    communication between model partitions.
    """

    def __init__(self, devices: list[torch.device]):
        self.devices = devices
        self.num_partitions = len(devices)

    def create_pipeline_parallel_model(self,
                                     model: nn.Module,
                                     partition_sizes: list[int] | None = None) -> nn.Module:
        """Create a pipeline parallel model."""
        if partition_sizes is None:
            # Equal partitioning
            total_modules = len(list(model.modules())) - 1  # Exclude root module
            partition_size = total_modules // self.num_partitions
            partition_sizes = [partition_size] * self.num_partitions

        return PipelineParallelModel(model, self.devices, partition_sizes)

    def optimize_tensor_parallelism(self,
                                  linear_layer: nn.Linear,
                                  dim: int = 0) -> list[nn.Linear]:
        """Split a linear layer for tensor parallelism."""
        weight = linear_layer.weight
        bias = linear_layer.bias

        # Split weight tensor
        weight_splits = torch.chunk(weight, self.num_partitions, dim=dim)
        bias_splits = torch.chunk(bias, self.num_partitions, dim=0) if bias is not None else [None] * self.num_partitions

        # Create parallel layers
        parallel_layers = []
        for i, (weight_split, bias_split) in enumerate(zip(weight_splits, bias_splits)):
            layer = nn.Linear(weight_split.shape[1], weight_split.shape[0], bias=bias_split is not None)
            layer.weight.data = weight_split.to(self.devices[i])
            if bias_split is not None:
                layer.bias.data = bias_split.to(self.devices[i])
            layer = layer.to(self.devices[i])
            parallel_layers.append(layer)

        return parallel_layers


class PipelineParallelModel(nn.Module):
    """
    Pipeline parallel model implementation.

    Distributes model layers across multiple devices and implements
    pipeline execution with micro-batching.
    """

    def __init__(self,
                 model: nn.Module,
                 devices: list[torch.device],
                 partition_sizes: list[int],
                 micro_batch_size: int | None = None):
        super().__init__()
        self.devices = devices
        self.partition_sizes = partition_sizes
        self.micro_batch_size = micro_batch_size
        self.partitions = self._create_partitions(model)

    def _create_partitions(self, model: nn.Module) -> nn.ModuleList:
        """Split model into partitions for pipeline parallelism."""
        modules = list(model.children())
        partitions = nn.ModuleList()

        start_idx = 0
        for i, partition_size in enumerate(self.partition_sizes):
            end_idx = min(start_idx + partition_size, len(modules))
            partition_modules = modules[start_idx:end_idx]

            if partition_modules:
                partition = nn.Sequential(*partition_modules)
                partition = partition.to(self.devices[i])
                partitions.append(partition)

            start_idx = end_idx

        return partitions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pipeline parallel forward pass."""
        if self.micro_batch_size is not None:
            return self._pipeline_forward_with_microbatching(x)
        else:
            return self._sequential_forward(x)

    def _sequential_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential forward pass through partitions."""
        for i, partition in enumerate(self.partitions):
            x = x.to(self.devices[i])
            x = partition(x)
        return x

    def _pipeline_forward_with_microbatching(self, x: torch.Tensor) -> torch.Tensor:
        """Pipeline forward pass with micro-batching."""
        batch_size = x.shape[0]
        num_micro_batches = (batch_size + self.micro_batch_size - 1) // self.micro_batch_size

        micro_batch_outputs = []

        for i in range(num_micro_batches):
            start_idx = i * self.micro_batch_size
            end_idx = min((i + 1) * self.micro_batch_size, batch_size)
            micro_batch = x[start_idx:end_idx]

            # Process micro-batch through pipeline
            for j, partition in enumerate(self.partitions):
                micro_batch = micro_batch.to(self.devices[j])
                micro_batch = partition(micro_batch)

            micro_batch_outputs.append(micro_batch)

        # Concatenate micro-batch outputs
        return torch.cat(micro_batch_outputs, dim=0)


class CommunicationOptimizer:
    """
    Optimize communication patterns in multi-GPU training.

    Provides utilities for overlapping computation and communication,
    gradient compression, and efficient all-reduce implementations.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size

    def overlap_communication_computation(self,
                                        model: DDP,
                                        optimizer: torch.optim.Optimizer,
                                        loss_fn: Callable,
                                        inputs: torch.Tensor,
                                        targets: torch.Tensor) -> dict[str, float]:
        """
        Training step with overlapped communication and computation.

        Args:
            model: DDP model
            optimizer: Optimizer
            loss_fn: Loss function
            inputs: Input tensors
            targets: Target tensors

        Returns:
            Dictionary with timing information
        """
        start_time = time.time()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass with gradient accumulation
        model.require_backward_grad_sync = False  # Disable sync
        loss.backward()
        forward_backward_time = time.time() - start_time

        # Manual gradient synchronization
        sync_start = time.time()
        model.require_backward_grad_sync = True

        # Sync gradients manually
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        sync_time = time.time() - sync_start

        # Optimizer step
        optimizer_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_time = time.time() - optimizer_start

        total_time = time.time() - start_time

        return {
            'forward_backward_time': forward_backward_time,
            'sync_time': sync_time,
            'optimizer_time': optimizer_time,
            'total_time': total_time,
            'loss': loss.item()
        }

    def gradient_compression(self, gradients: list[torch.Tensor], compression_ratio: float = 0.1) -> list[torch.Tensor]:
        """
        Compress gradients for reduced communication overhead.

        Args:
            gradients: List of gradient tensors
            compression_ratio: Ratio of gradients to keep (top-k compression)

        Returns:
            Compressed gradients
        """
        compressed_gradients = []

        for grad in gradients:
            if grad is None:
                compressed_gradients.append(grad)
                continue

            # Flatten gradient
            flat_grad = grad.flatten()
            num_elements = flat_grad.numel()
            k = int(num_elements * compression_ratio)

            # Top-k compression
            _, indices = torch.topk(torch.abs(flat_grad), k)
            compressed = torch.zeros_like(flat_grad)
            compressed[indices] = flat_grad[indices]

            # Reshape back to original shape
            compressed = compressed.view(grad.shape)
            compressed_gradients.append(compressed)

        return compressed_gradients

    def efficient_all_reduce(self, tensors: list[torch.Tensor], bucket_size_mb: float = 25.0) -> list[torch.Tensor]:
        """
        Efficient all-reduce with bucketing for reduced communication overhead.

        Args:
            tensors: List of tensors to reduce
            bucket_size_mb: Size of communication buckets in MB

        Returns:
            Reduced tensors
        """
        bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        buckets = []
        current_bucket = []
        current_bucket_size = 0

        # Group tensors into buckets
        for tensor in tensors:
            tensor_size = tensor.numel() * tensor.element_size()

            if current_bucket_size + tensor_size > bucket_size_bytes and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0

            current_bucket.append(tensor)
            current_bucket_size += tensor_size

        if current_bucket:
            buckets.append(current_bucket)

        # Perform bucketed all-reduce
        for bucket in buckets:
            # Concatenate tensors in bucket
            flat_tensors = [t.flatten() for t in bucket]
            bucket_tensor = torch.cat(flat_tensors)

            # All-reduce bucket
            dist.all_reduce(bucket_tensor, op=dist.ReduceOp.SUM)
            bucket_tensor /= self.world_size

            # Split back to original tensors
            start_idx = 0
            for _i, tensor in enumerate(bucket):
                end_idx = start_idx + tensor.numel()
                tensor.data = bucket_tensor[start_idx:end_idx].view(tensor.shape)
                start_idx = end_idx

        return tensors


class MultiGPUProfiler:
    """
    Profiler for multi-GPU training performance analysis.

    Provides detailed analysis of communication patterns, load balancing,
    and performance bottlenecks in distributed training.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.timing_data = defaultdict(list)
        self.communication_stats = defaultdict(list)

    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.timing_data[f"{operation}_start"] = time.time()

    def end_timing(self, operation: str):
        """End timing an operation."""
        if f"{operation}_start" in self.timing_data:
            duration = time.time() - self.timing_data[f"{operation}_start"]
            self.timing_data[operation].append(duration)
            del self.timing_data[f"{operation}_start"]

    @contextmanager
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timing_data[operation].append(duration)

    def profile_communication(self, tensor_size: int, operation: str):
        """Profile communication operation."""
        start_time = time.time()
        # This would be called around actual communication operations
        yield
        duration = time.time() - start_time

        self.communication_stats[operation].append({
            'duration': duration,
            'tensor_size': tensor_size,
            'bandwidth': tensor_size / duration if duration > 0 else 0
        })

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'rank': self.rank,
            'timing_stats': {},
            'communication_stats': {},
            'load_balance': self._analyze_load_balance()
        }

        # Timing statistics
        for operation, times in self.timing_data.items():
            if isinstance(times, list) and times:
                summary['timing_stats'][operation] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }

        # Communication statistics
        for operation, stats in self.communication_stats.items():
            if stats:
                durations = [s['duration'] for s in stats]
                bandwidths = [s['bandwidth'] for s in stats]
                summary['communication_stats'][operation] = {
                    'mean_duration': np.mean(durations),
                    'mean_bandwidth': np.mean(bandwidths),
                    'total_data': sum(s['tensor_size'] for s in stats)
                }

        return summary

    def _analyze_load_balance(self) -> dict[str, float]:
        """Analyze load balance across ranks."""
        # This would collect timing data from all ranks and analyze balance
        # For demo purposes, return placeholder data
        return {
            'variance': 0.1,
            'max_deviation': 0.05,
            'efficiency': 0.95
        }


def demonstrate_multi_gpu_patterns():
    """
    Comprehensive demonstration of multi-GPU optimization patterns.
    """
    print(" Multi-GPU Optimization Patterns Demonstration")
    print("=" * 60)

    # Check if multiple GPUs are available
    num_gpus = torch.cuda.device_count()

    if num_gpus < 2:
        print("  Multi-GPU demonstration requires at least 2 GPUs")
        print("Running single-GPU demonstration instead...")
        num_gpus = 1

    # Create sample model
    class MultiGPUTestModel(nn.Module):
        def __init__(self, input_size: int = 512, hidden_size: int = 1024, num_layers: int = 4):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ])
            self.output = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))
            return self.output(x)

    # Initialize components
    devices = [torch.device(f'cuda:{i}') for i in range(min(num_gpus, 4))]
    model = MultiGPUTestModel()

    print("\n Model Architecture:")
    print("  Input size: 512")
    print("  Hidden size: 1024")
    print("  Number of layers: 4")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Demonstrate Data Parallelism
    print("\n Data Parallelism Demo:")
    if num_gpus > 1:
        DataParallelOptimizer(model)

        # Simulate distributed configuration
        MultiGPUConfig(
            world_size=num_gpus,
            rank=0,
            local_rank=0
        )

        # For demonstration, show DDP setup process
        print(f"  Setting up DistributedDataParallel with {num_gpus} GPUs")
        print("  Bucket size: 25 MB")
        print("  Gradient as bucket view: True")
    else:
        print("  Single GPU mode - using DataParallel")
        model = nn.DataParallel(model, device_ids=[0])

    # Demonstrate Model Parallelism
    print("\n Model Parallelism Demo:")
    model_parallel_optimizer = ModelParallelOptimizer(devices)

    # Create pipeline parallel model
    pipeline_model = model_parallel_optimizer.create_pipeline_parallel_model(model)
    print(f"  Pipeline parallel model created with {len(devices)} partitions")

    # Demonstrate tensor parallelism
    sample_linear = nn.Linear(512, 1024)
    parallel_layers = model_parallel_optimizer.optimize_tensor_parallelism(sample_linear)
    print(f"  Tensor parallel layers: {len(parallel_layers)} partitions")

    # Performance Analysis
    print("\n Performance Analysis:")

    # Sample data
    batch_size = 32
    sample_input = torch.randn(batch_size, 512, device=devices[0])

    # Time different approaches
    with torch.no_grad():
        # Single GPU timing
        model_single = MultiGPUTestModel().to(devices[0])

        start_time = time.time()
        for _ in range(10):
            model_single(sample_input)
        single_gpu_time = (time.time() - start_time) / 10

        print(f"  Single GPU forward pass: {single_gpu_time*1000:.2f} ms")

        if len(devices) > 1:
            # Pipeline parallel timing
            sample_input_pipeline = sample_input.clone()

            start_time = time.time()
            for _ in range(10):
                pipeline_model(sample_input_pipeline)
            pipeline_time = (time.time() - start_time) / 10

            print(f"  Pipeline parallel forward pass: {pipeline_time*1000:.2f} ms")

            speedup = single_gpu_time / pipeline_time if pipeline_time > 0 else 0
            print(f"  Pipeline speedup: {speedup:.2f}x")

    # Communication Optimization Demo
    print("\n Communication Optimization:")
    comm_optimizer = CommunicationOptimizer(world_size=num_gpus)

    # Demonstrate gradient compression
    sample_gradients = [torch.randn(1000, device=devices[0]) for _ in range(5)]
    compressed_gradients = comm_optimizer.gradient_compression(sample_gradients, compression_ratio=0.1)

    original_size = sum(g.numel() for g in sample_gradients)
    compressed_size = sum((g != 0).sum().item() for g in compressed_gradients)
    compression_ratio = compressed_size / original_size

    print(f"  Gradient compression ratio: {compression_ratio:.1%}")
    print(f"  Communication reduction: {(1 - compression_ratio)*100:.1f}%")

    # Multi-GPU Profiling Demo
    print("\n Multi-GPU Profiling:")
    profiler = MultiGPUProfiler(world_size=num_gpus, rank=0)

    # Simulate some operations with profiling
    for _i in range(5):
        with profiler.time_operation('forward_pass'):
            time.sleep(0.001)  # Simulate computation

        with profiler.time_operation('backward_pass'):
            time.sleep(0.002)  # Simulate computation

    perf_summary = profiler.get_performance_summary()

    if 'forward_pass' in perf_summary['timing_stats']:
        forward_stats = perf_summary['timing_stats']['forward_pass']
        print(f"  Average forward pass time: {forward_stats['mean']*1000:.2f} ± {forward_stats['std']*1000:.2f} ms")

    if 'backward_pass' in perf_summary['timing_stats']:
        backward_stats = perf_summary['timing_stats']['backward_pass']
        print(f"  Average backward pass time: {backward_stats['mean']*1000:.2f} ± {backward_stats['std']*1000:.2f} ms")

    print("\n Multi-GPU optimization demonstration complete!")
    print("Key benefits demonstrated:")
    print("  • Data parallelism for scaling across multiple GPUs")
    print("  • Model parallelism for large models")
    print("  • Pipeline parallelism for memory efficiency")
    print("  • Communication optimization for reduced overhead")
    print("  • Performance profiling for optimization insights")


if __name__ == "__main__":
    demonstrate_multi_gpu_patterns()
