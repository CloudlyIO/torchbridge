"""
Model Sharding Strategies for Large Models

This module provides automatic and manual model sharding strategies
for distributing large models across multiple GPUs.

Supports:
- Full Sharding (FSDP-style)
- Hybrid Sharding (Tensor + Pipeline)
- Automatic Sharding based on memory constraints
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Model sharding strategy."""
    NO_SHARD = "no_shard"  # Replicate model on all GPUs
    FULL_SHARD = "full_shard"  # Fully shard parameters, gradients, optimizer states
    SHARD_GRAD_OP = "shard_grad_op"  # Shard gradients and optimizer states
    HYBRID_SHARD = "hybrid_shard"  # Shard within node, replicate across nodes
    HYBRID_ZERO3 = "hybrid_zero3"  # ZeRO-3 style sharding


class ParameterSharding(Enum):
    """How to shard individual parameters."""
    REPLICATE = "replicate"  # Full copy on each device
    SHARD_ROW = "shard_row"  # Shard along first dimension
    SHARD_COL = "shard_col"  # Shard along second dimension
    SHARD_TABLE = "shard_table"  # Shard embedding table rows


@dataclass
class ShardingConfig:
    """Configuration for model sharding.

    Args:
        strategy: Overall sharding strategy
        world_size: Total number of GPUs
        rank: Current GPU rank
        process_group: Distributed process group
        sharding_group_size: Size of sharding group for hybrid strategies
        cpu_offload: Offload parameters to CPU when not in use
        compute_device: Device for forward/backward computation
        mixed_precision: Enable mixed precision with sharding
        forward_prefetch: Prefetch next shard during forward
        backward_prefetch: Prefetch next shard during backward
        limit_all_gathers: Limit concurrent all-gather operations
    """
    strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    world_size: int = 1
    rank: int = 0
    process_group: Optional[Any] = None
    sharding_group_size: Optional[int] = None
    cpu_offload: bool = False
    compute_device: str = "cuda"
    mixed_precision: bool = True
    forward_prefetch: bool = True
    backward_prefetch: bool = True
    limit_all_gathers: bool = True

    def __post_init__(self):
        if self.sharding_group_size is None:
            self.sharding_group_size = self.world_size


@dataclass
class ShardSpec:
    """Specification for how a parameter is sharded."""
    param_name: str
    sharding_type: ParameterSharding
    shard_dim: int = 0
    num_shards: int = 1
    shard_sizes: List[int] = field(default_factory=list)


class ShardedParameter(nn.Parameter):
    """A parameter that is sharded across multiple devices.

    This wraps a local shard of a parameter and handles
    gathering/scattering for forward/backward passes.
    """

    def __new__(
        cls,
        data: torch.Tensor,
        spec: ShardSpec,
        config: ShardingConfig,
        requires_grad: bool = True,
    ):
        instance = super().__new__(cls, data, requires_grad)
        return instance

    def __init__(
        self,
        data: torch.Tensor,
        spec: ShardSpec,
        config: ShardingConfig,
        requires_grad: bool = True,
    ):
        self.spec = spec
        self.config = config
        self._full_param: Optional[torch.Tensor] = None
        self._is_gathered = False

    def gather_full_param(self) -> torch.Tensor:
        """Gather full parameter from all shards.

        Returns:
            Full parameter tensor
        """
        if self._is_gathered and self._full_param is not None:
            return self._full_param

        if self.config.world_size == 1:
            self._full_param = self.data
            self._is_gathered = True
            return self._full_param

        # All-gather the parameter
        import torch.distributed as dist

        if dist.is_initialized() and self.config.process_group is not None:
            gather_list = [
                torch.empty_like(self.data) for _ in range(self.config.world_size)
            ]
            dist.all_gather(gather_list, self.data, group=self.config.process_group)
            self._full_param = torch.cat(gather_list, dim=self.spec.shard_dim)
        else:
            self._full_param = self.data

        self._is_gathered = True
        return self._full_param

    def release_full_param(self) -> None:
        """Release gathered parameter to free memory."""
        self._full_param = None
        self._is_gathered = False


class ModelSharder:
    """Handles sharding of model parameters across GPUs.

    This class provides methods to:
    - Analyze model structure for optimal sharding
    - Apply sharding to model parameters
    - Handle gather/scatter during forward/backward
    """

    def __init__(self, config: ShardingConfig):
        self.config = config
        self._shard_specs: Dict[str, ShardSpec] = {}
        self._original_params: Dict[str, torch.Tensor] = {}

    def analyze_model(self, model: nn.Module) -> Dict[str, ShardSpec]:
        """Analyze model and determine sharding strategy for each parameter.

        Args:
            model: Model to analyze

        Returns:
            Dictionary mapping parameter names to shard specs
        """
        specs = {}

        for name, param in model.named_parameters():
            spec = self._determine_shard_spec(name, param)
            specs[name] = spec

        self._shard_specs = specs
        logger.info(f"Analyzed {len(specs)} parameters for sharding")
        return specs

    def _determine_shard_spec(self, name: str, param: torch.Tensor) -> ShardSpec:
        """Determine how to shard a parameter.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            Sharding specification
        """
        # Default: shard along largest dimension
        if param.dim() == 0:
            # Scalar - replicate
            return ShardSpec(
                param_name=name,
                sharding_type=ParameterSharding.REPLICATE,
            )

        if param.dim() == 1:
            # 1D parameter (bias, LayerNorm weights)
            # Small, typically replicate
            if param.numel() < 1024:
                return ShardSpec(
                    param_name=name,
                    sharding_type=ParameterSharding.REPLICATE,
                )
            else:
                return ShardSpec(
                    param_name=name,
                    sharding_type=ParameterSharding.SHARD_ROW,
                    shard_dim=0,
                    num_shards=self.config.world_size,
                )

        if param.dim() >= 2:
            # 2D+ parameter (Linear, Conv, Embedding)
            # Determine best sharding dimension
            if "embed" in name.lower() or "token" in name.lower():
                # Embedding tables: shard along vocabulary
                return ShardSpec(
                    param_name=name,
                    sharding_type=ParameterSharding.SHARD_TABLE,
                    shard_dim=0,
                    num_shards=self.config.world_size,
                )

            # For weight matrices, shard along larger dimension
            if param.shape[0] >= param.shape[1]:
                return ShardSpec(
                    param_name=name,
                    sharding_type=ParameterSharding.SHARD_ROW,
                    shard_dim=0,
                    num_shards=self.config.world_size,
                )
            else:
                return ShardSpec(
                    param_name=name,
                    sharding_type=ParameterSharding.SHARD_COL,
                    shard_dim=1,
                    num_shards=self.config.world_size,
                )

        return ShardSpec(
            param_name=name,
            sharding_type=ParameterSharding.REPLICATE,
        )

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Apply sharding to model parameters.

        Args:
            model: Model to shard

        Returns:
            Model with sharded parameters
        """
        if not self._shard_specs:
            self.analyze_model(model)

        for name, param in list(model.named_parameters()):
            if name not in self._shard_specs:
                continue

            spec = self._shard_specs[name]

            if spec.sharding_type == ParameterSharding.REPLICATE:
                continue  # Keep as-is

            # Store original parameter
            self._original_params[name] = param.data.clone()

            # Create sharded parameter
            shard = self._create_shard(param.data, spec)

            # Replace parameter in model
            self._set_parameter(model, name, shard)

        logger.info(f"Sharded {len(self._original_params)} parameters")
        return model

    def _create_shard(self, full_param: torch.Tensor, spec: ShardSpec) -> torch.Tensor:
        """Create local shard of a parameter.

        Args:
            full_param: Full parameter tensor
            spec: Sharding specification

        Returns:
            Local shard tensor
        """
        if spec.sharding_type == ParameterSharding.REPLICATE:
            return full_param

        # Calculate shard boundaries
        dim = spec.shard_dim
        total_size = full_param.shape[dim]
        shard_size = total_size // self.config.world_size
        start_idx = self.config.rank * shard_size
        end_idx = start_idx + shard_size

        # Handle last shard getting remainder
        if self.config.rank == self.config.world_size - 1:
            end_idx = total_size

        # Extract shard
        shard = full_param.narrow(dim, start_idx, end_idx - start_idx).contiguous()

        # Update spec with actual shard sizes
        spec.shard_sizes = [shard_size] * (self.config.world_size - 1)
        spec.shard_sizes.append(total_size - shard_size * (self.config.world_size - 1))

        return shard

    def _set_parameter(self, model: nn.Module, name: str, value: torch.Tensor) -> None:
        """Set a parameter in a model by name.

        Args:
            model: Model containing the parameter
            name: Parameter name (dot-separated path)
            value: New parameter value
        """
        parts = name.split(".")
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)

        param_name = parts[-1]
        if isinstance(value, nn.Parameter):
            setattr(module, param_name, value)
        else:
            setattr(module, param_name, nn.Parameter(value))

    def gather_model(self, model: nn.Module) -> nn.Module:
        """Gather all sharded parameters back to full size.

        Args:
            model: Model with sharded parameters

        Returns:
            Model with full parameters
        """
        for name in list(self._original_params.keys()):
            if name in self._shard_specs:
                spec = self._shard_specs[name]
                if spec.sharding_type != ParameterSharding.REPLICATE:
                    full_param = self._original_params[name]
                    self._set_parameter(model, name, full_param)

        logger.info("Gathered all sharded parameters")
        return model


class WeightDistributor:
    """Distributes model weights across devices for inference.

    Handles:
    - Automatic weight placement based on memory constraints
    - Loading pre-sharded checkpoints
    - Converting between sharding strategies
    """

    def __init__(self, config: ShardingConfig):
        self.config = config
        self._device_map: Dict[str, str] = {}

    def create_device_map(
        self,
        model: nn.Module,
        max_memory: Optional[Dict[str, int]] = None,
    ) -> Dict[str, str]:
        """Create device map for model layers.

        Args:
            model: Model to distribute
            max_memory: Maximum memory per device in bytes

        Returns:
            Dictionary mapping layer names to devices
        """
        if max_memory is None:
            # Query available memory
            max_memory = self._get_available_memory()

        # Calculate memory per layer
        layer_memory = self._estimate_layer_memory(model)

        # Greedy allocation
        device_map = {}
        device_memory_used = {f"cuda:{i}": 0 for i in range(self.config.world_size)}
        device_memory_used["cpu"] = 0

        for name, memory in sorted(layer_memory.items(), key=lambda x: -x[1]):
            # Find device with enough space
            placed = False
            for device in sorted(device_memory_used.keys()):
                available = max_memory.get(device, float("inf")) - device_memory_used[device]
                if available >= memory:
                    device_map[name] = device
                    device_memory_used[device] += memory
                    placed = True
                    break

            if not placed:
                # Fall back to CPU
                device_map[name] = "cpu"
                device_memory_used["cpu"] += memory
                logger.warning(f"Layer {name} placed on CPU due to memory constraints")

        self._device_map = device_map
        return device_map

    def _get_available_memory(self) -> Dict[str, int]:
        """Get available memory per device.

        Returns:
            Dictionary mapping device names to available memory in bytes
        """
        memory = {}

        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Use 90% of total memory as available
                memory[f"cuda:{i}"] = int(props.total_memory * 0.9)

        # CPU memory (use 50% of system RAM)
        import psutil
        memory["cpu"] = int(psutil.virtual_memory().total * 0.5)

        return memory

    def _estimate_layer_memory(self, model: nn.Module) -> Dict[str, int]:
        """Estimate memory usage per layer.

        Args:
            model: Model to analyze

        Returns:
            Dictionary mapping layer names to memory in bytes
        """
        layer_memory = {}

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                param_memory = sum(
                    p.numel() * p.element_size() for p in module.parameters()
                )
                if param_memory > 0:
                    layer_memory[name] = param_memory

        return layer_memory

    def distribute_model(
        self,
        model: nn.Module,
        device_map: Optional[Dict[str, str]] = None,
    ) -> nn.Module:
        """Distribute model according to device map.

        Args:
            model: Model to distribute
            device_map: Device map (created if None)

        Returns:
            Distributed model
        """
        if device_map is None:
            device_map = self.create_device_map(model)

        for name, device in device_map.items():
            module = self._get_module_by_name(model, name)
            if module is not None:
                module.to(device)

        logger.info(f"Distributed model across {len(set(device_map.values()))} devices")
        return model

    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get module by dot-separated name.

        Args:
            model: Root model
            name: Module name (dot-separated path)

        Returns:
            Module or None if not found
        """
        if not name:
            return model

        parts = name.split(".")
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def load_sharded_checkpoint(
        self,
        model: nn.Module,
        checkpoint_dir: str,
        shard_format: str = "pytorch",
    ) -> nn.Module:
        """Load pre-sharded checkpoint.

        Args:
            model: Model to load weights into
            checkpoint_dir: Directory containing sharded checkpoint
            shard_format: Format of sharded checkpoint

        Returns:
            Model with loaded weights
        """
        import os
        import glob

        # Find shard files
        if shard_format == "pytorch":
            shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.bin")))
        elif shard_format == "safetensors":
            shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        else:
            raise ValueError(f"Unknown shard format: {shard_format}")

        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {checkpoint_dir}")

        logger.info(f"Loading {len(shard_files)} checkpoint shards")

        # Load shards for this rank
        for shard_file in shard_files:
            state_dict = self._load_shard_file(shard_file, shard_format)

            # Filter to parameters for this rank's device map
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if self._should_load_param(k)
            }

            model.load_state_dict(filtered_state_dict, strict=False)

        return model

    def _load_shard_file(self, path: str, shard_format: str) -> Dict[str, torch.Tensor]:
        """Load a single shard file.

        Args:
            path: Path to shard file
            shard_format: Format of shard file

        Returns:
            State dict from shard
        """
        if shard_format == "pytorch":
            return torch.load(path, map_location="cpu")
        elif shard_format == "safetensors":
            from safetensors import safe_open
            tensors = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            return tensors
        else:
            raise ValueError(f"Unknown shard format: {shard_format}")

    def _should_load_param(self, param_name: str) -> bool:
        """Check if parameter should be loaded on this rank.

        Args:
            param_name: Name of the parameter

        Returns:
            True if parameter should be loaded
        """
        # If no device map, load everything
        if not self._device_map:
            return True

        # Find the module for this parameter
        parts = param_name.rsplit(".", 1)
        if len(parts) == 2:
            module_name, _ = parts
        else:
            module_name = ""

        # Check if module is on this rank's device
        if module_name in self._device_map:
            device = self._device_map[module_name]
            return device == f"cuda:{self.config.rank}" or device == "cpu"

        return True


def automatic_sharding(
    model: nn.Module,
    config: ShardingConfig,
    target_memory_gb: Optional[float] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Automatically shard model based on memory constraints.

    Args:
        model: Model to shard
        config: Sharding configuration
        target_memory_gb: Target memory per GPU in GB

    Returns:
        Tuple of (sharded model, sharding info dict)
    """
    # Analyze model
    sharder = ModelSharder(config)
    specs = sharder.analyze_model(model)

    # Calculate current memory
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = total_params * 4  # Assume float32
    param_gb = param_bytes / (1024**3)

    logger.info(f"Model has {total_params / 1e6:.1f}M parameters ({param_gb:.2f} GB)")

    # Determine if sharding is needed
    if target_memory_gb is None:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            target_memory_gb = props.total_memory / (1024**3) * 0.7  # 70% of GPU memory
        else:
            target_memory_gb = 16.0  # Default

    sharding_needed = param_gb > target_memory_gb

    if sharding_needed:
        logger.info(f"Model ({param_gb:.2f} GB) > target ({target_memory_gb:.2f} GB), applying sharding")
        model = sharder.shard_model(model)

        # Calculate post-sharding memory
        sharded_params = sum(p.numel() for p in model.parameters())
        sharded_gb = sharded_params * 4 / (1024**3)
        logger.info(f"After sharding: {sharded_gb:.2f} GB per GPU")
    else:
        logger.info(f"Model fits in memory, no sharding needed")

    info = {
        "original_params": total_params,
        "original_memory_gb": param_gb,
        "sharding_applied": sharding_needed,
        "target_memory_gb": target_memory_gb,
        "sharding_specs": specs,
    }

    return model, info
