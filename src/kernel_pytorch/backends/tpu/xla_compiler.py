"""
XLA Compiler for TPU Optimization

Provides XLA compilation, optimization, and caching for TPU models.
"""

import hashlib
import logging
import time
import warnings
from typing import Any

import torch
import torch.nn as nn

from kernel_pytorch.core.config import TPUCompilationMode, TPUConfig

from . import xla_compat
from .cache_utils import LRUCache
from .tpu_exceptions import XLACompilationError, raise_or_warn

logger = logging.getLogger(__name__)


class XLACompiler:
    """
    XLA compiler for TPU model optimization.

    Provides compilation, caching, and optimization utilities for TPU models
    using PyTorch/XLA compilation infrastructure.
    """

    def __init__(self, config: TPUConfig):
        """
        Initialize XLA compiler.

        Args:
            config: TPU configuration
        """
        self.config = config
        self._compilation_cache = LRUCache(max_size=config.cache_max_size)
        self._compilation_stats = LRUCache(max_size=config.cache_max_size)

        # Initialize XLA environment
        self._setup_xla_compiler()

    def _setup_xla_compiler(self) -> None:
        """Set up XLA compiler environment."""
        try:
            import torch_xla  # noqa: F401
            import torch_xla.core.xla_model as xm  # noqa: F401

            self._xla_available = True
            logger.info(
                "XLA Compiler initialized: mode=%s, optimization_level=%d, dynamic_shapes=%s",
                self.config.compilation_mode.value,
                self.config.xla_optimization_level,
                self.config.enable_xla_dynamic_shapes
            )

        except ImportError:
            self._xla_available = False
            warnings.warn(
                "PyTorch/XLA not available. Compiler will use CPU fallback.",
                RuntimeWarning,
            stacklevel=2,
            )

    def compile_model(self, model: nn.Module,
                     sample_inputs: torch.Tensor | tuple | None = None,
                     use_cache: bool = True) -> nn.Module:
        """
        Compile model for TPU execution.

        Args:
            model: PyTorch model to compile
            sample_inputs: Sample inputs for compilation
            use_cache: Whether to use compilation cache

        Returns:
            Compiled model
        """
        if not self._xla_available:
            warnings.warn("XLA not available, returning original model", stacklevel=2)
            return model

        # Generate cache key
        if use_cache:
            cache_key = self._generate_cache_key(model, sample_inputs)
            cached_model = self._compilation_cache.get(cache_key)
            if cached_model is not None:
                logger.debug("Using cached compilation for model")
                return cached_model

        # Perform compilation based on mode
        start_time = time.time()

        if self.config.compilation_mode == TPUCompilationMode.TORCH_XLA:
            compiled_model = self._compile_torch_xla(model, sample_inputs)
        elif self.config.compilation_mode == TPUCompilationMode.XLA:
            compiled_model = self._compile_xla_direct(model, sample_inputs)
        elif self.config.compilation_mode == TPUCompilationMode.PJIT:
            compiled_model = self._compile_pjit(model, sample_inputs)
        else:
            raise ValueError(f"Unsupported compilation mode: {self.config.compilation_mode}")

        compilation_time = time.time() - start_time

        # Cache the result
        if use_cache:
            self._compilation_cache.set(cache_key, compiled_model)
            self._compilation_stats.set(cache_key, {
                'compilation_time': compilation_time,
                'timestamp': time.time(),
                'model_size': self._estimate_model_size(model)
            })

        logger.info("Model compiled: time=%.2fs, mode=%s", compilation_time, self.config.compilation_mode.value)
        return compiled_model

    def _compile_torch_xla(self, model: nn.Module,
                          sample_inputs: torch.Tensor | tuple | None) -> nn.Module:
        """Compile using PyTorch/XLA torch.compile."""
        try:
            # Sync for XLA compilation using compatibility layer
            xla_compat.sync()

            # Use torch.compile with XLA backend if available
            if hasattr(torch, 'compile'):
                # Get the appropriate backend for the installed torch_xla version
                backend = xla_compat.get_torch_compile_backend()

                if backend is not None:
                    # Use specific backend (openxla for 2.9+, aot_torchxla_trace_once for older)
                    compiled_model = torch.compile(
                        model,
                        backend=backend,
                        dynamic=self.config.enable_xla_dynamic_shapes
                    )
                else:
                    # For torch_xla 2.9+ without explicit backend, use default compilation
                    # torch.compile works directly with XLA tensors
                    compiled_model = torch.compile(
                        model,
                        dynamic=self.config.enable_xla_dynamic_shapes
                    )
                return compiled_model
            else:
                # Fallback for older PyTorch versions
                return model

        except Exception as e:
            error_msg = f"PyTorch/XLA compilation failed: {e}"
            raise_or_warn(error_msg, XLACompilationError, strict_mode=self.config.enable_strict_validation, logger=logger)
            return model

    def _compile_xla_direct(self, model: nn.Module,
                           sample_inputs: torch.Tensor | tuple | None) -> nn.Module:
        """Compile using direct XLA compilation."""
        try:
            # Force XLA compilation with sample inputs
            if sample_inputs is not None:
                # Move inputs to XLA device using compatibility layer
                xla_device = xla_compat.get_xla_device()
                if isinstance(sample_inputs, torch.Tensor):
                    sample_inputs = sample_inputs.to(xla_device)
                elif isinstance(sample_inputs, (list, tuple)):
                    sample_inputs = tuple(inp.to(xla_device) if isinstance(inp, torch.Tensor)
                                        else inp for inp in sample_inputs)

                # Run forward pass to trigger compilation
                model.to(xla_device)
                model.eval()
                with torch.no_grad():
                    _ = model(sample_inputs)
                xla_compat.sync()

            return model

        except Exception as e:
            error_msg = f"Direct XLA compilation failed: {e}"
            raise_or_warn(error_msg, XLACompilationError, strict_mode=self.config.enable_strict_validation, logger=logger)
            return model

    def _compile_pjit(self, model: nn.Module,
                     sample_inputs: torch.Tensor | tuple | None) -> nn.Module:
        """Compile using JAX pjit (experimental)."""
        if not self.config.enable_jax_integration:
            warnings.warn("JAX integration disabled, falling back to torch_xla", stacklevel=2)
            return self._compile_torch_xla(model, sample_inputs)

        try:
            # This is experimental - would require JAX integration
            warnings.warn("pjit compilation not yet implemented, using torch_xla", stacklevel=2)
            return self._compile_torch_xla(model, sample_inputs)

        except Exception as e:
            error_msg = f"pjit compilation failed: {e}"
            raise_or_warn(error_msg, XLACompilationError, strict_mode=self.config.enable_strict_validation, logger=logger)
            return model

    def _generate_cache_key(self, model: nn.Module,
                           sample_inputs: torch.Tensor | tuple | None) -> str:
        """Generate cache key for model compilation."""
        # Create hash based on model structure and config
        model_str = str(model)
        config_str = str(self.config.__dict__)

        # Include input shapes if available
        input_info = ""
        if sample_inputs is not None:
            if isinstance(sample_inputs, torch.Tensor):
                input_info = str(sample_inputs.shape)
            elif isinstance(sample_inputs, (list, tuple)):
                input_info = str([inp.shape if isinstance(inp, torch.Tensor) else str(inp)
                                for inp in sample_inputs])

        # Create hash
        combined = f"{model_str}_{config_str}_{input_info}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _estimate_model_size(self, model: nn.Module) -> int:
        """Estimate model size in bytes."""
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()
        # Assume 4 bytes per parameter (float32)
        return total_params * 4

    def optimize_for_inference(self, model: nn.Module,
                             sample_inputs: torch.Tensor | tuple | None = None) -> nn.Module:
        """
        Optimize model specifically for inference.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs for optimization

        Returns:
            Optimized model
        """
        # Set model to eval mode
        model.eval()

        # Apply inference-specific optimizations
        with torch.no_grad():
            # Freeze batch norm statistics
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
                    module.track_running_stats = False

            # Compile for inference
            optimized_model = self.compile_model(model, sample_inputs, use_cache=True)

        return optimized_model

    def optimize_for_training(self, model: nn.Module,
                            sample_inputs: torch.Tensor | tuple | None = None) -> nn.Module:
        """
        Optimize model specifically for training.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs for optimization

        Returns:
            Optimized model
        """
        # Set model to training mode
        model.train()

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        # Compile for training
        optimized_model = self.compile_model(model, sample_inputs, use_cache=True)

        return optimized_model

    def get_compilation_stats(self) -> dict[str, Any]:
        """Get compilation statistics."""
        cache_stats = self._compilation_cache.get_stats()

        return {
            'compilation_cache': cache_stats,
            'xla_available': self._xla_available,
            'compilation_mode': self.config.compilation_mode.value,
            'cache_max_size': self.config.cache_max_size
        }

    def clear_cache(self) -> None:
        """Clear compilation cache."""
        self._compilation_cache.clear()
        self._compilation_stats.clear()

        # Clear XLA compilation cache
        try:
            xla_compat.sync()
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

        for _i in range(num_runs):
            # Clear cache for fair comparison
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
        """String representation of XLA compiler."""
        return (
            f"XLACompiler(mode={self.config.compilation_mode.value}, "
            f"optimization_level={self.config.xla_optimization_level}, "
            f"cached_models={len(self._compilation_cache)})"
        )
