"""
Tensor Parallelism Implementation for Large Models

Tensor parallelism splits individual layers across multiple GPUs,
enabling models that exceed single-GPU memory to run efficiently.

Based on techniques from Megatron-LM and other distributed training frameworks.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ParallelMode(Enum):
    """Parallelism mode for tensor operations."""
    COLUMN = "column"  # Split output features
    ROW = "row"  # Split input features
    REPLICATED = "replicated"  # Full copy on each device


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism.

    Args:
        world_size: Total number of GPUs for tensor parallelism
        rank: Current GPU rank (0 to world_size-1)
        process_group: Optional distributed process group
        scatter_to_sequence_parallel: Whether to scatter to sequence parallel region
        sequence_parallel: Enable sequence parallelism (LayerNorm + Dropout)
        async_tensor_model_parallel_allreduce: Async communication overlap
        gradient_accumulation_fusion: Fuse gradient accumulation with communication
    """
    world_size: int = 1
    rank: int = 0
    process_group: Optional[Any] = None
    scatter_to_sequence_parallel: bool = False
    sequence_parallel: bool = False
    async_tensor_model_parallel_allreduce: bool = True
    gradient_accumulation_fusion: bool = True

    def __post_init__(self):
        if self.rank >= self.world_size:
            raise ValueError(f"rank ({self.rank}) must be < world_size ({self.world_size})")


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, process_group):
        ctx.process_group = process_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.process_group is not None and dist.is_initialized():
            dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, process_group):
        if process_group is not None and dist.is_initialized():
            dist.all_reduce(input_, group=process_group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and scatter to model parallel region."""

    @staticmethod
    def forward(ctx, input_, world_size, rank, process_group):
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.process_group = process_group

        # Split along last dimension
        last_dim = input_.dim() - 1
        input_list = input_.chunk(world_size, dim=last_dim)
        return input_list[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.process_group is not None and dist.is_initialized():
            grad_list = [torch.empty_like(grad_output) for _ in range(ctx.world_size)]
            dist.all_gather(grad_list, grad_output, group=ctx.process_group)
            grad_output = torch.cat(grad_list, dim=-1)
        return grad_output, None, None, None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather from model parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_, world_size, rank, process_group):
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.process_group = process_group

        if process_group is not None and dist.is_initialized():
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
            dist.all_gather(tensor_list, input_, group=process_group)
            return torch.cat(tensor_list, dim=-1)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        last_dim = grad_output.dim() - 1
        grad_list = grad_output.chunk(ctx.world_size, dim=last_dim)
        return grad_list[ctx.rank].contiguous(), None, None, None


def copy_to_tensor_parallel_region(input_, config: TensorParallelConfig):
    """Copy input to tensor parallel region."""
    return _CopyToModelParallelRegion.apply(input_, config.process_group)


def reduce_from_tensor_parallel_region(input_, config: TensorParallelConfig):
    """Reduce input from tensor parallel region."""
    return _ReduceFromModelParallelRegion.apply(input_, config.process_group)


def scatter_to_tensor_parallel_region(input_, config: TensorParallelConfig):
    """Scatter input to tensor parallel region."""
    return _ScatterToModelParallelRegion.apply(
        input_, config.world_size, config.rank, config.process_group
    )


def gather_from_tensor_parallel_region(input_, config: TensorParallelConfig):
    """Gather input from tensor parallel region."""
    return _GatherFromModelParallelRegion.apply(
        input_, config.world_size, config.rank, config.process_group
    )


class TensorParallelLinear(nn.Module):
    """Base class for tensor parallel linear layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TensorParallelConfig,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.gather_output = gather_output

        # Subclasses implement specific partitioning
        self.weight = None
        self.bias = None

    def _init_weights(self):
        """Initialize weights with proper scaling for tensor parallelism."""
        if self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ColumnParallelLinear(TensorParallelLinear):
    """Linear layer with column parallelism.

    Splits the weight matrix along the output dimension (columns).
    Each GPU holds W[:, rank * out_per_partition : (rank + 1) * out_per_partition]

    Forward: Y = XW (parallel matmul) + b
    The input X is copied to all GPUs, output Y is partitioned.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TensorParallelConfig,
        bias: bool = True,
        gather_output: bool = True,
        skip_bias_add: bool = False,
    ):
        super().__init__(in_features, out_features, config, bias, gather_output)

        self.skip_bias_add = skip_bias_add
        world_size = config.world_size

        # Partition output features across GPUs
        assert out_features % world_size == 0, (
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"
        )
        self.out_features_per_partition = out_features // world_size

        # Initialize partitioned weight
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def forward(self, input_: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with column parallelism.

        Args:
            input_: Input tensor of shape [batch, seq, in_features]

        Returns:
            Output tensor, optionally with separate bias
        """
        # Copy input to all GPUs
        input_parallel = copy_to_tensor_parallel_region(input_, self.config)

        # Parallel matmul
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight)

        if self.gather_output:
            # Gather outputs from all GPUs
            output = gather_from_tensor_parallel_region(output_parallel, self.config)
        else:
            output = output_parallel

        if self.bias is not None:
            if self.skip_bias_add:
                return output, self.bias
            if self.gather_output:
                # Need to gather bias as well
                bias = gather_from_tensor_parallel_region(
                    self.bias.unsqueeze(0).unsqueeze(0), self.config
                ).squeeze()
                output = output + bias
            else:
                output = output + self.bias

        return output


class RowParallelLinear(TensorParallelLinear):
    """Linear layer with row parallelism.

    Splits the weight matrix along the input dimension (rows).
    Each GPU holds W[rank * in_per_partition : (rank + 1) * in_per_partition, :]

    Forward: Y = XW + b
    The input X is partitioned, results are all-reduced.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TensorParallelConfig,
        bias: bool = True,
        input_is_parallel: bool = False,
        skip_bias_add: bool = False,
    ):
        super().__init__(in_features, out_features, config, bias, gather_output=True)

        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        world_size = config.world_size

        # Partition input features across GPUs
        assert in_features % world_size == 0, (
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"
        )
        self.in_features_per_partition = in_features // world_size

        # Initialize partitioned weight
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def forward(self, input_: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with row parallelism.

        Args:
            input_: Input tensor, either full or partitioned

        Returns:
            Output tensor, optionally with separate bias
        """
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # Scatter input to parallel region
            input_parallel = scatter_to_tensor_parallel_region(input_, self.config)

        # Parallel matmul
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight)

        # All-reduce across GPUs
        output = reduce_from_tensor_parallel_region(output_parallel, self.config)

        if self.bias is not None:
            if self.skip_bias_add:
                return output, self.bias
            output = output + self.bias

        return output


class TensorParallelEmbedding(nn.Module):
    """Embedding layer with tensor parallelism.

    Splits the embedding table across GPUs along the embedding dimension.
    Each GPU holds embedding[:, rank * dim_per_gpu : (rank + 1) * dim_per_gpu]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: TensorParallelConfig,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config
        self.padding_idx = padding_idx

        world_size = config.world_size

        # Partition embedding dimension across GPUs
        assert embedding_dim % world_size == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by world_size ({world_size})"
        )
        self.embedding_dim_per_partition = embedding_dim // world_size

        # Initialize partitioned embedding
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, self.embedding_dim_per_partition)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallel embedding.

        Args:
            input_: Input token indices [batch, seq]

        Returns:
            Embedded output [batch, seq, embedding_dim]
        """
        # Local embedding lookup
        output_parallel = torch.nn.functional.embedding(
            input_, self.weight, padding_idx=self.padding_idx
        )

        # Gather from all GPUs
        output = gather_from_tensor_parallel_region(output_parallel, self.config)

        return output


class VocabParallelEmbedding(nn.Module):
    """Embedding layer with vocabulary parallelism.

    Splits the embedding table across GPUs along the vocabulary dimension.
    Each GPU holds embedding[rank * vocab_per_gpu : (rank + 1) * vocab_per_gpu, :]

    This is more memory-efficient for very large vocabularies.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: TensorParallelConfig,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config
        self.padding_idx = padding_idx

        world_size = config.world_size
        rank = config.rank

        # Partition vocabulary across GPUs
        self.vocab_start_index = rank * (num_embeddings // world_size)
        self.vocab_end_index = (rank + 1) * (num_embeddings // world_size)
        if rank == world_size - 1:
            self.vocab_end_index = num_embeddings  # Handle remainder

        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Initialize partitioned embedding
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with vocab parallel embedding.

        Args:
            input_: Input token indices [batch, seq]

        Returns:
            Embedded output [batch, seq, embedding_dim]
        """
        # Mask out tokens not in this partition
        input_mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)
        masked_input = input_.clone()
        masked_input[~input_mask] = 0

        # Shift indices to local range
        masked_input = masked_input - self.vocab_start_index

        # Local embedding lookup
        output_parallel = torch.nn.functional.embedding(masked_input, self.weight)

        # Zero out positions that don't belong to this partition
        output_parallel = output_parallel * input_mask.unsqueeze(-1).float()

        # All-reduce to get full output
        output = reduce_from_tensor_parallel_region(output_parallel, self.config)

        return output


def apply_tensor_parallelism(
    module: nn.Module,
    config: TensorParallelConfig,
    linear_layer_names: Optional[List[str]] = None,
    embedding_layer_names: Optional[List[str]] = None,
) -> nn.Module:
    """Apply tensor parallelism to a model.

    This function replaces standard nn.Linear and nn.Embedding layers
    with their tensor parallel equivalents.

    Args:
        module: The model to parallelize
        config: Tensor parallel configuration
        linear_layer_names: Names of linear layers to parallelize (None = all)
        embedding_layer_names: Names of embedding layers to parallelize (None = all)

    Returns:
        Model with tensor parallel layers
    """
    if config.world_size == 1:
        logger.info("world_size=1, skipping tensor parallelism")
        return module

    replaced_count = {"linear": 0, "embedding": 0}

    def should_replace_linear(name: str) -> bool:
        if linear_layer_names is None:
            return True
        return any(pattern in name for pattern in linear_layer_names)

    def should_replace_embedding(name: str) -> bool:
        if embedding_layer_names is None:
            return True
        return any(pattern in name for pattern in embedding_layer_names)

    def replace_layers(parent_module: nn.Module, prefix: str = ""):
        for name, child in list(parent_module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and should_replace_linear(full_name):
                # Determine if this is a column or row parallel layer
                # Heuristic: layers with "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"
                # are typically column parallel; "o_proj", "down_proj" are row parallel
                is_column = any(
                    pattern in full_name.lower()
                    for pattern in ["q_proj", "k_proj", "v_proj", "gate", "up_proj", "fc1"]
                )

                if is_column:
                    new_layer = ColumnParallelLinear(
                        child.in_features,
                        child.out_features,
                        config,
                        bias=child.bias is not None,
                        gather_output=False,  # Don't gather for intermediate layers
                    )
                else:
                    new_layer = RowParallelLinear(
                        child.in_features,
                        child.out_features,
                        config,
                        bias=child.bias is not None,
                        input_is_parallel=True,  # Input is already parallel
                    )

                setattr(parent_module, name, new_layer)
                replaced_count["linear"] += 1
                logger.debug(f"Replaced {full_name} with {'Column' if is_column else 'Row'}ParallelLinear")

            elif isinstance(child, nn.Embedding) and should_replace_embedding(full_name):
                new_layer = TensorParallelEmbedding(
                    child.num_embeddings,
                    child.embedding_dim,
                    config,
                    padding_idx=child.padding_idx,
                )
                setattr(parent_module, name, new_layer)
                replaced_count["embedding"] += 1
                logger.debug(f"Replaced {full_name} with TensorParallelEmbedding")

            else:
                # Recurse into child modules
                replace_layers(child, full_name)

    replace_layers(module)
    logger.info(
        f"Applied tensor parallelism: {replaced_count['linear']} linear, "
        f"{replaced_count['embedding']} embedding layers replaced"
    )

    return module


def get_tensor_parallel_state_dict(
    module: nn.Module,
    config: TensorParallelConfig,
    full_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Get the partition of state dict for this rank.

    Args:
        module: The tensor parallel module
        config: Tensor parallel configuration
        full_state_dict: Full model state dict

    Returns:
        Partitioned state dict for this rank
    """
    partitioned_state_dict = {}

    for name, param in module.named_parameters():
        if name not in full_state_dict:
            logger.warning(f"Parameter {name} not found in state dict")
            continue

        full_param = full_state_dict[name]

        # Check if this is a tensor parallel parameter
        if isinstance(param, nn.Parameter):
            parent_module = module
            for attr in name.split(".")[:-1]:
                parent_module = getattr(parent_module, attr)

            if isinstance(parent_module, (ColumnParallelLinear, TensorParallelEmbedding)):
                # Split along output/embedding dimension
                dim = -1 if isinstance(parent_module, ColumnParallelLinear) else 1
                chunks = full_param.chunk(config.world_size, dim=dim)
                partitioned_state_dict[name] = chunks[config.rank]

            elif isinstance(parent_module, RowParallelLinear):
                # Split along input dimension
                chunks = full_param.chunk(config.world_size, dim=-1)
                partitioned_state_dict[name] = chunks[config.rank]

            else:
                # Not a tensor parallel parameter, copy as-is
                partitioned_state_dict[name] = full_param

    return partitioned_state_dict
