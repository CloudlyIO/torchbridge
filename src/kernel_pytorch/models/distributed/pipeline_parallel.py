"""
Pipeline Parallelism Implementation for Large Models

Pipeline parallelism splits model layers into stages across GPUs,
enabling training of very deep models by reducing per-GPU memory.

Implements GPipe and Interleaved schedules for micro-batch pipelining.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from collections import deque

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Pipeline schedule type."""
    GPIPE = "gpipe"  # Simple fill-drain schedule
    INTERLEAVED = "interleaved"  # 1F1B interleaved schedule
    CHIMERA = "chimera"  # Bidirectional pipeline


@dataclass
class PipelineParallelConfig:
    """Configuration for pipeline parallelism.

    Args:
        num_stages: Number of pipeline stages
        num_micro_batches: Number of micro-batches per batch
        stage_id: Current stage ID (0 to num_stages-1)
        process_group: Distributed process group
        schedule: Pipeline schedule type
        chunks: Number of model chunks for interleaved schedule
        activation_checkpointing: Enable activation checkpointing
        async_communication: Enable async send/recv
    """
    num_stages: int = 1
    num_micro_batches: int = 1
    stage_id: int = 0
    process_group: Optional[Any] = None
    schedule: ScheduleType = ScheduleType.GPIPE
    chunks: int = 1
    activation_checkpointing: bool = True
    async_communication: bool = True

    def __post_init__(self):
        if self.stage_id >= self.num_stages:
            raise ValueError(
                f"stage_id ({self.stage_id}) must be < num_stages ({self.num_stages})"
            )
        if self.num_micro_batches < self.num_stages:
            logger.warning(
                f"num_micro_batches ({self.num_micro_batches}) < num_stages ({self.num_stages}), "
                "pipeline bubble will be significant"
            )


class PipelineStage(nn.Module):
    """A stage in the pipeline containing a subset of model layers.

    Each stage processes micro-batches and communicates activations
    with neighboring stages.
    """

    def __init__(
        self,
        module: nn.Module,
        config: PipelineParallelConfig,
        stage_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.module = module
        self.config = config
        self.stage_layers = stage_layers or []

        # Communication buffers
        self._input_tensors: List[Optional[torch.Tensor]] = []
        self._output_tensors: List[Optional[torch.Tensor]] = []

        # Activation storage for backward pass
        self._saved_activations: Dict[int, torch.Tensor] = {}

        # Async handles
        self._send_handles: List[Any] = []
        self._recv_handles: List[Any] = []

    @property
    def is_first_stage(self) -> bool:
        return self.config.stage_id == 0

    @property
    def is_last_stage(self) -> bool:
        return self.config.stage_id == self.config.num_stages - 1

    def _get_prev_rank(self) -> int:
        """Get rank of previous stage."""
        return self.config.stage_id - 1

    def _get_next_rank(self) -> int:
        """Get rank of next stage."""
        return self.config.stage_id + 1

    def recv_forward(self, tensor_shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Receive activation from previous stage."""
        if self.is_first_stage:
            return None

        tensor = torch.empty(tensor_shape, dtype=dtype, device="cuda")

        if dist.is_initialized():
            if self.config.async_communication:
                handle = dist.irecv(tensor, src=self._get_prev_rank())
                self._recv_handles.append(handle)
            else:
                dist.recv(tensor, src=self._get_prev_rank())

        return tensor

    def send_forward(self, tensor: torch.Tensor) -> None:
        """Send activation to next stage."""
        if self.is_last_stage:
            return

        if dist.is_initialized():
            if self.config.async_communication:
                handle = dist.isend(tensor, dst=self._get_next_rank())
                self._send_handles.append(handle)
            else:
                dist.send(tensor, dst=self._get_next_rank())

    def recv_backward(self, tensor_shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Receive gradient from next stage."""
        if self.is_last_stage:
            return None

        tensor = torch.empty(tensor_shape, dtype=dtype, device="cuda")

        if dist.is_initialized():
            if self.config.async_communication:
                handle = dist.irecv(tensor, src=self._get_next_rank())
                self._recv_handles.append(handle)
            else:
                dist.recv(tensor, src=self._get_next_rank())

        return tensor

    def send_backward(self, tensor: torch.Tensor) -> None:
        """Send gradient to previous stage."""
        if self.is_first_stage:
            return

        if dist.is_initialized():
            if self.config.async_communication:
                handle = dist.isend(tensor, dst=self._get_prev_rank())
                self._send_handles.append(handle)
            else:
                dist.send(tensor, dst=self._get_prev_rank())

    def wait_all(self) -> None:
        """Wait for all async communications to complete."""
        for handle in self._send_handles + self._recv_handles:
            handle.wait()
        self._send_handles.clear()
        self._recv_handles.clear()

    def forward_step(
        self,
        input_tensor: Optional[torch.Tensor],
        micro_batch_id: int,
    ) -> torch.Tensor:
        """Execute forward pass for one micro-batch.

        Args:
            input_tensor: Input from previous stage (None for first stage)
            micro_batch_id: ID of the micro-batch

        Returns:
            Output tensor to send to next stage
        """
        if self.config.activation_checkpointing:
            # Save input for backward pass
            if input_tensor is not None:
                self._saved_activations[micro_batch_id] = input_tensor.detach()

        # Execute module forward
        output = self.module(input_tensor)

        return output

    def backward_step(
        self,
        grad_output: Optional[torch.Tensor],
        micro_batch_id: int,
    ) -> torch.Tensor:
        """Execute backward pass for one micro-batch.

        Args:
            grad_output: Gradient from next stage (None for last stage)
            micro_batch_id: ID of the micro-batch

        Returns:
            Gradient to send to previous stage
        """
        # Retrieve saved activation
        input_tensor = self._saved_activations.pop(micro_batch_id, None)

        if input_tensor is not None:
            input_tensor.requires_grad_(True)

            # Recompute forward if using activation checkpointing
            with torch.enable_grad():
                output = self.module(input_tensor)

            # Backward pass
            torch.autograd.backward(output, grad_output)

            return input_tensor.grad
        else:
            return None


class PipelineScheduler:
    """Base class for pipeline schedules."""

    def __init__(self, stages: List[PipelineStage], config: PipelineParallelConfig):
        self.stages = stages
        self.config = config
        self.num_micro_batches = config.num_micro_batches

    def run_forward(self, micro_batches: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run forward passes for all micro-batches."""
        raise NotImplementedError

    def run_backward(self, gradients: List[torch.Tensor]) -> None:
        """Run backward passes for all micro-batches."""
        raise NotImplementedError


class GPipeScheduler(PipelineScheduler):
    """GPipe scheduler: fill-drain schedule.

    All forward passes complete before any backward passes start.
    Simple but has higher memory usage due to storing all activations.
    """

    def __init__(self, stages: List[PipelineStage], config: PipelineParallelConfig):
        super().__init__(stages, config)
        self._output_cache: Dict[int, torch.Tensor] = {}

    def run_forward(self, micro_batches: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run forward passes using GPipe schedule.

        Args:
            micro_batches: List of micro-batch inputs

        Returns:
            List of outputs for each micro-batch
        """
        outputs = []
        stage = self.stages[0]  # Assuming single stage per worker

        for micro_batch_id, micro_batch in enumerate(micro_batches):
            # Receive from previous stage
            if not stage.is_first_stage:
                input_tensor = stage.recv_forward(micro_batch.shape, micro_batch.dtype)
                stage.wait_all()  # Wait for receive to complete
            else:
                input_tensor = micro_batch

            # Forward step
            output = stage.forward_step(input_tensor, micro_batch_id)

            # Cache output for backward pass
            self._output_cache[micro_batch_id] = output

            # Send to next stage
            if not stage.is_last_stage:
                stage.send_forward(output)
            else:
                outputs.append(output)

        # Wait for all sends to complete
        stage.wait_all()

        return outputs

    def run_backward(self, gradients: List[Optional[torch.Tensor]]) -> None:
        """Run backward passes using GPipe schedule.

        Args:
            gradients: Gradients from loss (only for last stage)
        """
        stage = self.stages[0]

        # Backward in reverse order
        for micro_batch_id in reversed(range(len(gradients))):
            # Receive gradient from next stage
            if not stage.is_last_stage:
                output = self._output_cache[micro_batch_id]
                grad_output = stage.recv_backward(output.shape, output.dtype)
                stage.wait_all()
            else:
                grad_output = gradients[micro_batch_id]

            # Backward step
            grad_input = stage.backward_step(grad_output, micro_batch_id)

            # Send gradient to previous stage
            if not stage.is_first_stage and grad_input is not None:
                stage.send_backward(grad_input)

        # Wait for all sends to complete
        stage.wait_all()

        # Clear cache
        self._output_cache.clear()


class InterleavedScheduler(PipelineScheduler):
    """1F1B Interleaved scheduler.

    Interleaves forward and backward passes to reduce memory usage.
    One forward pass is followed by one backward pass in steady state.

    The 1F1B schedule works as follows:
    1. Warmup phase: Run forward passes to fill the pipeline
    2. Steady state: Alternate 1 forward, 1 backward (1F1B)
    3. Cooldown phase: Drain remaining backward passes

    Memory advantage over GPipe:
    - GPipe: Stores all N micro-batch activations
    - 1F1B: Stores at most (num_stages) activations at any time

    v0.4.23: Added run_forward() and run_backward() implementations
    """

    def __init__(self, stages: List[PipelineStage], config: PipelineParallelConfig):
        super().__init__(stages, config)
        self._forward_queue: deque = deque()
        self._backward_queue: deque = deque()
        self._output_cache: Dict[int, torch.Tensor] = {}
        self._input_cache: Dict[int, torch.Tensor] = {}

    def run_forward(self, micro_batches: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run forward passes using 1F1B interleaved schedule.

        This implements just the forward phase, storing activations for
        later backward pass. Use run_backward() after computing gradients.

        Args:
            micro_batches: List of micro-batch inputs

        Returns:
            List of outputs for each micro-batch
        """
        stage = self.stages[0]
        num_micro_batches = len(micro_batches)
        outputs = []

        # Calculate warmup and 1F1B phases
        num_warmup = min(
            self.config.num_stages - self.config.stage_id - 1,
            num_micro_batches
        )
        num_1f1b = num_micro_batches - num_warmup

        # Warmup phase: forward passes only
        for i in range(num_warmup):
            if stage.is_first_stage:
                input_tensor = micro_batches[i]
            else:
                input_tensor = stage.recv_forward(
                    micro_batches[i].shape,
                    micro_batches[i].dtype
                )
                stage.wait_all()

            output = stage.forward_step(input_tensor, i)

            # Cache for backward
            self._input_cache[i] = input_tensor
            self._output_cache[i] = output

            if not stage.is_last_stage:
                stage.send_forward(output)
            else:
                outputs.append(output)

        # 1F1B steady state: we only do forward here
        # (backward is handled separately in run_backward)
        for i in range(num_1f1b):
            forward_id = num_warmup + i

            if stage.is_first_stage:
                input_tensor = micro_batches[forward_id]
            else:
                input_tensor = stage.recv_forward(
                    micro_batches[forward_id].shape,
                    micro_batches[forward_id].dtype
                )
                stage.wait_all()

            output = stage.forward_step(input_tensor, forward_id)

            self._input_cache[forward_id] = input_tensor
            self._output_cache[forward_id] = output

            if not stage.is_last_stage:
                stage.send_forward(output)
            else:
                outputs.append(output)

        stage.wait_all()
        return outputs

    def run_backward(self, gradients: List[Optional[torch.Tensor]]) -> None:
        """Run backward passes using 1F1B interleaved schedule.

        Must be called after run_forward() with gradients computed from
        the forward outputs.

        Args:
            gradients: Gradients from loss (only meaningful for last stage)
        """
        stage = self.stages[0]
        num_micro_batches = len(gradients)

        # Calculate phases
        num_warmup = min(
            self.config.num_stages - self.config.stage_id - 1,
            num_micro_batches
        )
        num_1f1b = num_micro_batches - num_warmup

        # Process backward in 1F1B order
        # First, process the 1F1B phase backwards
        for i in range(num_1f1b):
            backward_id = i

            if stage.is_last_stage:
                grad_output = gradients[backward_id]
            else:
                output_shape = self._output_cache[backward_id].shape
                output_dtype = self._output_cache[backward_id].dtype
                grad_output = stage.recv_backward(output_shape, output_dtype)
                stage.wait_all()

            grad_input = stage.backward_step(grad_output, backward_id)

            if not stage.is_first_stage and grad_input is not None:
                stage.send_backward(grad_input)

        # Cooldown phase: remaining backward passes
        for i in range(num_warmup):
            backward_id = num_1f1b + i

            if stage.is_last_stage:
                grad_output = gradients[backward_id]
            else:
                output_shape = self._output_cache[backward_id].shape
                output_dtype = self._output_cache[backward_id].dtype
                grad_output = stage.recv_backward(output_shape, output_dtype)
                stage.wait_all()

            grad_input = stage.backward_step(grad_output, backward_id)

            if not stage.is_first_stage and grad_input is not None:
                stage.send_backward(grad_input)

        # Wait for all communications and clear caches
        stage.wait_all()
        self._output_cache.clear()
        self._input_cache.clear()

    def run_forward_backward(
        self,
        micro_batches: List[torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run interleaved forward and backward passes.

        Args:
            micro_batches: List of micro-batch inputs
            loss_fn: Loss function to compute gradients

        Returns:
            Total loss across all micro-batches
        """
        stage = self.stages[0]
        num_micro_batches = len(micro_batches)
        num_warmup_micro_batches = min(
            self.config.num_stages - self.config.stage_id - 1,
            num_micro_batches
        )
        num_1f1b_micro_batches = num_micro_batches - num_warmup_micro_batches

        total_loss = torch.tensor(0.0, device="cuda")
        input_tensors: List[torch.Tensor] = []
        output_tensors: List[torch.Tensor] = []

        # Warmup phase: only forward passes
        for i in range(num_warmup_micro_batches):
            if stage.is_first_stage:
                input_tensor = micro_batches[i]
            else:
                input_tensor = stage.recv_forward(micro_batches[i].shape, micro_batches[i].dtype)
                stage.wait_all()

            output_tensor = stage.forward_step(input_tensor, i)

            if not stage.is_last_stage:
                stage.send_forward(output_tensor)
            else:
                loss = loss_fn(output_tensor)
                total_loss += loss.detach()

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

        # 1F1B steady state phase
        for i in range(num_1f1b_micro_batches):
            forward_id = num_warmup_micro_batches + i
            backward_id = i

            # Forward pass
            if stage.is_first_stage:
                input_tensor = micro_batches[forward_id]
            else:
                input_tensor = stage.recv_forward(
                    micro_batches[forward_id].shape,
                    micro_batches[forward_id].dtype
                )
                stage.wait_all()

            output_tensor = stage.forward_step(input_tensor, forward_id)

            if not stage.is_last_stage:
                stage.send_forward(output_tensor)
            else:
                loss = loss_fn(output_tensor)
                total_loss += loss.detach()

            # Backward pass
            if stage.is_last_stage:
                grad_output = torch.ones_like(output_tensors[backward_id])
            else:
                grad_output = stage.recv_backward(
                    output_tensors[backward_id].shape,
                    output_tensors[backward_id].dtype
                )
                stage.wait_all()

            grad_input = stage.backward_step(grad_output, backward_id)

            if not stage.is_first_stage and grad_input is not None:
                stage.send_backward(grad_input)

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

        # Cooldown phase: only backward passes
        for i in range(num_warmup_micro_batches):
            backward_id = num_1f1b_micro_batches + i

            if stage.is_last_stage:
                grad_output = torch.ones_like(output_tensors[backward_id])
            else:
                grad_output = stage.recv_backward(
                    output_tensors[backward_id].shape,
                    output_tensors[backward_id].dtype
                )
                stage.wait_all()

            grad_input = stage.backward_step(grad_output, backward_id)

            if not stage.is_first_stage and grad_input is not None:
                stage.send_backward(grad_input)

        # Wait for all communications to complete
        stage.wait_all()

        return total_loss / num_micro_batches


def create_pipeline_stages(
    model: nn.Module,
    config: PipelineParallelConfig,
    split_points: Optional[List[str]] = None,
) -> List[PipelineStage]:
    """Create pipeline stages from a model.

    Args:
        model: The full model to split
        config: Pipeline parallel configuration
        split_points: Layer names to split at (auto-determined if None)

    Returns:
        List of PipelineStage modules
    """
    if split_points is None:
        # Auto-determine split points based on layer structure
        split_points = _auto_split_model(model, config.num_stages)

    # Split model into stages
    stages = []
    current_layers = []
    current_stage_id = 0
    layer_idx = 0

    for name, module in model.named_children():
        current_layers.append((name, module))

        if name in split_points or layer_idx == len(list(model.named_children())) - 1:
            # Create stage from accumulated layers
            stage_module = nn.Sequential()
            for layer_name, layer in current_layers:
                stage_module.add_module(layer_name, layer)

            # Only create stage for this rank
            if current_stage_id == config.stage_id:
                stage_config = PipelineParallelConfig(
                    num_stages=config.num_stages,
                    num_micro_batches=config.num_micro_batches,
                    stage_id=current_stage_id,
                    process_group=config.process_group,
                    schedule=config.schedule,
                    activation_checkpointing=config.activation_checkpointing,
                )
                stages.append(PipelineStage(stage_module, stage_config))

            current_layers = []
            current_stage_id += 1

        layer_idx += 1

    logger.info(f"Created {len(stages)} pipeline stage(s) for rank {config.stage_id}")
    return stages


def _auto_split_model(model: nn.Module, num_stages: int) -> List[str]:
    """Automatically determine split points for a model.

    Args:
        model: Model to analyze
        num_stages: Number of stages to create

    Returns:
        List of layer names to split at
    """
    # Get all top-level children
    children = list(model.named_children())
    num_layers = len(children)

    if num_layers < num_stages:
        logger.warning(
            f"Model has {num_layers} layers but {num_stages} stages requested. "
            f"Using {num_layers} stages instead."
        )
        num_stages = num_layers

    # Simple even split
    layers_per_stage = num_layers // num_stages
    split_points = []

    for i in range(1, num_stages):
        split_idx = i * layers_per_stage - 1
        if split_idx < num_layers:
            split_points.append(children[split_idx][0])

    logger.info(f"Auto-determined split points: {split_points}")
    return split_points


def estimate_pipeline_memory(
    model: nn.Module,
    config: PipelineParallelConfig,
    micro_batch_size: int,
    sequence_length: int,
) -> Dict[str, float]:
    """Estimate memory usage per stage in a pipeline.

    Args:
        model: The model to analyze
        config: Pipeline configuration
        micro_batch_size: Size of each micro-batch
        sequence_length: Sequence length for transformer models

    Returns:
        Dictionary with memory estimates per stage
    """
    total_params = sum(p.numel() for p in model.parameters())
    params_per_stage = total_params / config.num_stages

    # Estimate bytes per parameter (assuming mixed precision)
    bytes_per_param = 6  # 2 for fp16 params + 4 for fp32 gradients

    # Memory per stage
    param_memory_gb = (params_per_stage * bytes_per_param) / (1024**3)

    # Activation memory estimate
    # This is a rough estimate; actual depends on model architecture
    hidden_size = 8192  # Assume large model
    activation_per_layer = micro_batch_size * sequence_length * hidden_size * 2  # fp16
    layers_per_stage = 80 / config.num_stages  # Assume 80 layer model
    activation_memory_gb = (activation_per_layer * layers_per_stage) / (1024**3)

    # Peak memory with GPipe (stores all activations)
    gpipe_peak = param_memory_gb + activation_memory_gb * config.num_micro_batches

    # Peak memory with 1F1B (limited activation storage)
    interleaved_peak = param_memory_gb + activation_memory_gb * config.num_stages

    return {
        "params_per_stage_gb": param_memory_gb,
        "activation_per_micro_batch_gb": activation_memory_gb,
        "gpipe_peak_gb": gpipe_peak,
        "interleaved_peak_gb": interleaved_peak,
        "recommended_schedule": "interleaved" if interleaved_peak < gpipe_peak else "gpipe",
    }
