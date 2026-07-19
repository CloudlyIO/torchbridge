# SPDX-License-Identifier: Apache-2.0
"""
Trainium Model Adapter

High-level adapter for Trainium models that combines backend preparation,
Neuron compilation, and Trainium-specific optimizations.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchbridge.core.config import TorchBridgeConfig

from .neuron_compiler import NeuronCompiler
from .trainium_backend import TrainiumBackend
from .trainium_exceptions import TrainiumValidationError, raise_or_warn

logger = logging.getLogger(__name__)


@dataclass
class TrainiumOptimizationResult:
    """Result of Trainium optimization process."""

    optimized_model: nn.Module
    backend: TrainiumBackend
    compiler: NeuronCompiler
    optimization_time: float
    memory_usage: dict[str, Any]
    performance_metrics: dict[str, Any]


class TrainiumAdapter:
    """
    High-level Trainium model adapter.

    Combines Trainium backend preparation, Neuron compilation, and optimization
    strategies to provide the best performance for Trainium deployments.
    """

    def __init__(self, config: TorchBridgeConfig | None = None):
        """
        Initialize Trainium adapter.

        Args:
            config: Optional configuration. If None, creates default config.
        """
        self.config = config or TorchBridgeConfig()
        self.trainium_config = self.config.hardware.trainium

        # Initialize components
        self.backend = TrainiumBackend(self.config)
        self.compiler = NeuronCompiler(self.trainium_config)

        # Optimization tracking
        self._optimization_history = []

    def optimize(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor | tuple | None = None,
        optimization_level: str = "balanced",
    ) -> TrainiumOptimizationResult:
        """
        Optimize model for Trainium execution.

        Args:
            model: PyTorch model to optimize
            sample_inputs: Sample inputs for optimization
            optimization_level: Optimization level ('conservative', 'balanced', 'aggressive')

        Returns:
            Optimization result with optimized model and metrics
        """
        import time

        start_time = time.time()

        logger.info("Starting Trainium optimization: level=%s", optimization_level)

        # Step 1: Prepare model for Trainium backend
        prepared_model = self.backend.prepare_model(model)

        # Step 2: Apply optimization-level specific changes
        optimized_model = self._apply_optimization_level(
            prepared_model, optimization_level
        )

        # Step 3: Compile with Neuron
        compiled_model = self.compiler.compile_model(optimized_model, sample_inputs)

        # Step 4: Validate optimization
        self._validate_optimization(compiled_model, sample_inputs)

        optimization_time = time.time() - start_time

        # Gather metrics
        memory_usage = self.backend.get_memory_stats()
        performance_metrics = {
            "optimization_time": optimization_time,
            "compilation_stats": self.compiler.get_compilation_stats(),
            "optimization_level": optimization_level,
            "architecture": self.trainium_config.architecture.value,
            "world_size": self.backend.world_size,
        }

        result = TrainiumOptimizationResult(
            optimized_model=compiled_model,
            backend=self.backend,
            compiler=self.compiler,
            optimization_time=optimization_time,
            memory_usage=memory_usage,
            performance_metrics=performance_metrics,
        )

        self._optimization_history.append(
            {
                "timestamp": time.time(),
                "model_type": type(model).__name__,
                "optimization_level": optimization_level,
                "optimization_time": optimization_time,
            }
        )

        logger.info(
            "Trainium optimization completed: time=%.2fs, level=%s",
            optimization_time,
            optimization_level,
        )
        return result

    def _apply_optimization_level(self, model: nn.Module, level: str) -> nn.Module:
        """Apply optimization strategies based on level."""
        if level == "conservative":
            return self._apply_conservative_optimizations(model)
        elif level == "balanced":
            return self._apply_balanced_optimizations(model)
        elif level == "aggressive":
            return self._apply_aggressive_optimizations(model)
        else:
            raise ValueError(f"Unknown optimization level: {level}")

    def _apply_conservative_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply conservative optimizations (safety first)."""
        model = model.to(self.backend.device)

        if (
            self.trainium_config.mixed_precision
            and self.trainium_config.precision == "bfloat16"
        ):
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.to(dtype=torch.bfloat16)

        return model

    def _apply_balanced_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply balanced optimizations (performance + safety)."""
        model = self._apply_conservative_optimizations(model)

        if self.trainium_config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

        model = self._optimize_attention_layers(model)

        return model

    def _apply_aggressive_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply aggressive optimizations (maximum performance)."""
        model = self._apply_balanced_optimizations(model)

        if self.trainium_config.mixed_precision:
            for module in model.modules():
                if isinstance(
                    module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding)
                ):
                    module.to(dtype=torch.bfloat16)

        return model

    def _optimize_attention_layers(self, model: nn.Module) -> nn.Module:
        """Optimize attention layers for Trainium NeuronCores."""
        for module in model.modules():
            if (
                hasattr(module, "attention")
                or "attention" in module.__class__.__name__.lower()
            ):
                if hasattr(module, "num_heads"):
                    head_dim = getattr(module, "head_dim", None)
                    if head_dim and head_dim % 8 != 0:
                        warnings.warn(
                            f"Attention head dimension {head_dim} not optimal for Trainium. "
                            "Consider using dimensions divisible by 8.",
                            stacklevel=2,
                        )
        return model

    def _validate_optimization(
        self, model: nn.Module, sample_inputs: torch.Tensor | tuple | None
    ) -> None:
        """Validate that optimization was successful."""
        if sample_inputs is None:
            logger.warning("No sample inputs provided, skipping validation")
            return

        try:
            model_dtype = None
            for param in model.parameters():
                model_dtype = param.dtype
                break

            def prepare_input(inp: torch.Tensor) -> torch.Tensor:
                inp = inp.to(self.backend.device)
                if model_dtype == torch.bfloat16 and inp.dtype == torch.float32:
                    inp = inp.to(dtype=torch.bfloat16)
                return inp

            if isinstance(sample_inputs, torch.Tensor):
                sample_inputs = prepare_input(sample_inputs)
            elif isinstance(sample_inputs, (list, tuple)):
                sample_inputs = tuple(
                    prepare_input(inp) if isinstance(inp, torch.Tensor) else inp
                    for inp in sample_inputs
                )

            model.eval()
            with torch.no_grad():
                model(sample_inputs)

            self.backend.synchronize()
            logger.info("Optimization validation passed")

        except Exception as e:
            error_msg = f"Optimization validation failed: {str(e)}"
            raise_or_warn(
                error_msg,
                TrainiumValidationError,
                strict_mode=self.trainium_config.enable_strict_validation,
                logger=logger,
            )

    def optimize_for_inference(
        self, model: nn.Module, sample_inputs: torch.Tensor | tuple | None = None
    ) -> TrainiumOptimizationResult:
        """
        Optimize model specifically for inference.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs

        Returns:
            Optimization result
        """
        model.eval()
        result = self.optimize(model, sample_inputs, optimization_level="conservative")
        result.optimized_model = self.compiler.optimize_for_inference(
            result.optimized_model, sample_inputs
        )
        return result

    def optimize_for_training(
        self, model: nn.Module, sample_inputs: torch.Tensor | tuple | None = None
    ) -> TrainiumOptimizationResult:
        """
        Optimize model specifically for training.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs

        Returns:
            Optimization result
        """
        result = self.optimize(model, sample_inputs, optimization_level="balanced")
        result.optimized_model = self.compiler.optimize_for_training(
            result.optimized_model, sample_inputs
        )
        return result

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        total_optimizations = len(self._optimization_history)

        if total_optimizations == 0:
            return {"total_optimizations": 0}

        total_time = sum(opt["optimization_time"] for opt in self._optimization_history)
        avg_time = total_time / total_optimizations

        model_types: dict[str, int] = {}
        for opt in self._optimization_history:
            model_type = opt["model_type"]
            model_types[model_type] = model_types.get(model_type, 0) + 1

        return {
            "total_optimizations": total_optimizations,
            "total_optimization_time": total_time,
            "average_optimization_time": avg_time,
            "model_types": model_types,
            "backend_stats": self.backend.get_memory_stats(),
            "compiler_stats": self.compiler.get_compilation_stats(),
        }

    def clear_cache(self) -> None:
        """Clear all optimization caches."""
        self.backend.clear_cache()
        self.compiler.clear_cache()

    def __repr__(self) -> str:
        """String representation of Trainium adapter."""
        return (
            f"TrainiumAdapter(backend={self.backend}, "
            f"compiler={self.compiler}, "
            f"optimizations={len(self._optimization_history)})"
        )
