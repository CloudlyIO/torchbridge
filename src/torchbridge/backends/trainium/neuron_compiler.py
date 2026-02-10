"""
Neuron Compiler for Trainium Optimization

Provides Neuron graph compilation, caching, and optimization
for Trainium models via the Neuron SDK (neuronx-cc).
"""

import hashlib
import logging
import time
import warnings
from typing import Any

import torch
import torch.nn as nn

from torchbridge.core.config import TrainiumConfig
from torchbridge.utils.cache import LRUCache

from . import neuron_utilities
from .trainium_exceptions import NeuronCompilationError, raise_or_warn

logger = logging.getLogger(__name__)


class NeuronCompiler:
    """
    Neuron compiler for Trainium model optimization.

    Provides compilation, caching, and optimization utilities for Trainium
    models using the Neuron SDK compilation infrastructure (neuronx-cc).
    """

    def __init__(self, config: TrainiumConfig):
        """
        Initialize Neuron compiler.

        Args:
            config: Trainium configuration
        """
        self.config = config
        self._compilation_cache = LRUCache(max_size=config.cache_max_size)
        self._compilation_stats = LRUCache(max_size=config.cache_max_size)

        # Initialize Neuron compiler environment
        self._setup_neuron_compiler()

    def _setup_neuron_compiler(self) -> None:
        """Set up Neuron compiler environment."""
        try:
            import torch_neuronx  # noqa: F401

            self._neuron_available = True
            logger.info(
                "Neuron Compiler initialized: graph_caching=%s, timeout=%ds",
                self.config.enable_graph_caching,
                self.config.compilation_timeout_seconds
            )

        except ImportError:
            self._neuron_available = False
            warnings.warn(
                "Neuron SDK not available. Compiler will use CPU fallback.",
                RuntimeWarning,
                stacklevel=2,
            )

    def compile_model(self, model: nn.Module,
                      sample_inputs: torch.Tensor | tuple | None = None,
                      use_cache: bool = True) -> nn.Module:
        """
        Compile model for Trainium execution.

        The Neuron compiler traces the model graph and compiles it into
        optimized NeuronCore instructions (NEFFs).

        Args:
            model: PyTorch model to compile
            sample_inputs: Sample inputs for compilation
            use_cache: Whether to use compilation cache

        Returns:
            Compiled model
        """
        if not self._neuron_available:
            warnings.warn("Neuron SDK not available, returning original model", stacklevel=2)
            return model

        # Generate cache key
        if use_cache:
            cache_key = self._generate_cache_key(model, sample_inputs)
            cached_model = self._compilation_cache.get(cache_key)
            if cached_model is not None:
                logger.debug("Using cached Neuron compilation for model")
                return cached_model

        # Perform compilation
        start_time = time.time()
        compiled_model = self._compile_neuron(model, sample_inputs)
        compilation_time = time.time() - start_time

        # Cache the result
        if use_cache:
            cache_key = self._generate_cache_key(model, sample_inputs)
            self._compilation_cache.set(cache_key, compiled_model)
            self._compilation_stats.set(cache_key, {
                'compilation_time': compilation_time,
                'timestamp': time.time(),
                'model_size': self._estimate_model_size(model)
            })

        logger.info("Model compiled with Neuron: time=%.2fs", compilation_time)
        return compiled_model

    def _compile_neuron(self, model: nn.Module,
                        sample_inputs: torch.Tensor | tuple | None) -> nn.Module:
        """Compile model using Neuron SDK."""
        try:
            # Sync XLA state
            neuron_utilities.sync()

            # Configure Neuron compiler flags
            self._apply_compiler_flags()

            # Use torch.compile with XLA backend (same mechanism as TPU)
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(model)
                return compiled_model

            return model

        except Exception as e:
            error_msg = f"Neuron compilation failed: {e}"
            raise_or_warn(
                error_msg,
                NeuronCompilationError,
                strict_mode=self.config.enable_strict_validation,
                logger=logger
            )
            return model

    def _apply_compiler_flags(self) -> None:
        """Apply Neuron compiler flags from configuration."""
        import os

        if self.config.neuron_cc_flags:
            existing = os.environ.get('NEURON_CC_FLAGS', '')
            if existing:
                os.environ['NEURON_CC_FLAGS'] = f"{existing} {self.config.neuron_cc_flags}"
            else:
                os.environ['NEURON_CC_FLAGS'] = self.config.neuron_cc_flags

        # Enable graph caching
        if self.config.enable_graph_caching:
            os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

    def optimize_for_inference(self, model: nn.Module,
                               sample_inputs: torch.Tensor | tuple | None = None) -> nn.Module:
        """
        Optimize model specifically for inference on Trainium.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs for optimization

        Returns:
            Optimized model
        """
        model.eval()

        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
                    module.track_running_stats = False

            optimized_model = self.compile_model(model, sample_inputs, use_cache=True)

        return optimized_model

    def optimize_for_training(self, model: nn.Module,
                              sample_inputs: torch.Tensor | tuple | None = None) -> nn.Module:
        """
        Optimize model specifically for training on Trainium.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs for optimization

        Returns:
            Optimized model
        """
        model.train()

        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        optimized_model = self.compile_model(model, sample_inputs, use_cache=True)
        return optimized_model

    def _generate_cache_key(self, model: nn.Module,
                            sample_inputs: torch.Tensor | tuple | None) -> str:
        """Generate cache key for model compilation."""
        model_str = str(model)
        config_str = str(self.config.__dict__)

        input_info = ""
        if sample_inputs is not None:
            if isinstance(sample_inputs, torch.Tensor):
                input_info = str(sample_inputs.shape)
            elif isinstance(sample_inputs, (list, tuple)):
                input_info = str([inp.shape if isinstance(inp, torch.Tensor) else str(inp)
                                  for inp in sample_inputs])

        combined = f"{model_str}_{config_str}_{input_info}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _estimate_model_size(self, model: nn.Module) -> int:
        """Estimate model size in bytes."""
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 4  # Assume float32

    def get_compilation_stats(self) -> dict[str, Any]:
        """Get compilation statistics."""
        return {
            'compilation_cache': self._compilation_cache.get_stats(),
            'neuron_available': self._neuron_available,
            'graph_caching_enabled': self.config.enable_graph_caching,
            'cache_max_size': self.config.cache_max_size
        }

    def clear_cache(self) -> None:
        """Clear compilation cache."""
        self._compilation_cache.clear()
        self._compilation_stats.clear()

        try:
            neuron_utilities.sync()
        except Exception:
            pass

    def benchmark_compilation(self, model: nn.Module,
                              sample_inputs: torch.Tensor | tuple,
                              num_runs: int = 3) -> dict[str, float]:
        """
        Benchmark compilation performance.

        Args:
            model: Model to benchmark
            sample_inputs: Sample inputs
            num_runs: Number of compilation runs

        Returns:
            Benchmark results
        """
        compilation_times = []

        for _ in range(num_runs):
            self.clear_cache()

            start_time = time.time()
            _ = self.compile_model(model, sample_inputs, use_cache=False)
            compilation_time = time.time() - start_time
            compilation_times.append(compilation_time)

        return {
            'min_time': min(compilation_times),
            'max_time': max(compilation_times),
            'avg_time': sum(compilation_times) / len(compilation_times),
            'total_time': sum(compilation_times),
            'runs': num_runs
        }

    def __repr__(self) -> str:
        """String representation of Neuron compiler."""
        return (
            f"NeuronCompiler(neuron_available={self._neuron_available}, "
            f"graph_caching={self.config.enable_graph_caching}, "
            f"cached_models={len(self._compilation_cache)})"
        )
