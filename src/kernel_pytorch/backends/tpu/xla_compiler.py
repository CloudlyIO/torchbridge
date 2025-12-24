"""
XLA Compiler for TPU Optimization

Provides XLA compilation, optimization, and caching for TPU models.
"""

import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Callable
import time
import hashlib
import pickle

from kernel_pytorch.core.config import TPUConfig, TPUCompilationMode


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
        self._compilation_cache = {}
        self._compilation_stats = {}

        # Initialize XLA environment
        self._setup_xla_compiler()

    def _setup_xla_compiler(self) -> None:
        """Set up XLA compiler environment."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            self._xla_available = True
            print(f"🔧 XLA Compiler initialized:")
            print(f"   Compilation mode: {self.config.compilation_mode.value}")
            print(f"   Optimization level: {self.config.xla_optimization_level}")
            print(f"   Dynamic shapes: {self.config.enable_xla_dynamic_shapes}")

        except ImportError:
            self._xla_available = False
            warnings.warn(
                "PyTorch/XLA not available. Compiler will use CPU fallback.",
                RuntimeWarning
            )

    def compile_model(self, model: nn.Module,
                     sample_inputs: Optional[Union[torch.Tensor, tuple]] = None,
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
            warnings.warn("XLA not available, returning original model")
            return model

        # Generate cache key
        if use_cache:
            cache_key = self._generate_cache_key(model, sample_inputs)
            if cache_key in self._compilation_cache:
                print(f"📦 Using cached compilation for model")
                return self._compilation_cache[cache_key]

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
            self._compilation_cache[cache_key] = compiled_model
            self._compilation_stats[cache_key] = {
                'compilation_time': compilation_time,
                'timestamp': time.time(),
                'model_size': self._estimate_model_size(model)
            }

        print(f"⚡ Model compiled in {compilation_time:.2f}s")
        return compiled_model

    def _compile_torch_xla(self, model: nn.Module,
                          sample_inputs: Optional[Union[torch.Tensor, tuple]]) -> nn.Module:
        """Compile using PyTorch/XLA torch.compile."""
        try:
            import torch_xla.core.xla_model as xm

            # Mark step for XLA compilation
            xm.mark_step()

            # Use torch.compile with XLA backend if available
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(
                    model,
                    backend='aot_torchxla_trace_once',
                    dynamic=self.config.enable_xla_dynamic_shapes
                )
                return compiled_model
            else:
                # Fallback for older PyTorch versions
                return model

        except Exception as e:
            warnings.warn(f"PyTorch/XLA compilation failed: {e}")
            return model

    def _compile_xla_direct(self, model: nn.Module,
                           sample_inputs: Optional[Union[torch.Tensor, tuple]]) -> nn.Module:
        """Compile using direct XLA compilation."""
        try:
            import torch_xla.core.xla_model as xm

            # Force XLA compilation with sample inputs
            if sample_inputs is not None:
                # Move inputs to XLA device
                xla_device = xm.xla_device()
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
                xm.mark_step()

            return model

        except Exception as e:
            warnings.warn(f"Direct XLA compilation failed: {e}")
            return model

    def _compile_pjit(self, model: nn.Module,
                     sample_inputs: Optional[Union[torch.Tensor, tuple]]) -> nn.Module:
        """Compile using JAX pjit (experimental)."""
        if not self.config.enable_jax_integration:
            warnings.warn("JAX integration disabled, falling back to torch_xla")
            return self._compile_torch_xla(model, sample_inputs)

        try:
            # This is experimental - would require JAX integration
            warnings.warn("pjit compilation not yet implemented, using torch_xla")
            return self._compile_torch_xla(model, sample_inputs)

        except Exception as e:
            warnings.warn(f"pjit compilation failed: {e}")
            return model

    def _generate_cache_key(self, model: nn.Module,
                           sample_inputs: Optional[Union[torch.Tensor, tuple]]) -> str:
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
                             sample_inputs: Optional[Union[torch.Tensor, tuple]] = None) -> nn.Module:
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
                            sample_inputs: Optional[Union[torch.Tensor, tuple]] = None) -> nn.Module:
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

    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        total_models = len(self._compilation_stats)
        total_compilation_time = sum(stats['compilation_time']
                                   for stats in self._compilation_stats.values())
        avg_compilation_time = total_compilation_time / total_models if total_models > 0 else 0

        return {
            'total_compiled_models': total_models,
            'total_compilation_time': total_compilation_time,
            'average_compilation_time': avg_compilation_time,
            'cache_size': len(self._compilation_cache),
            'xla_available': self._xla_available,
            'compilation_mode': self.config.compilation_mode.value
        }

    def clear_cache(self) -> None:
        """Clear compilation cache."""
        self._compilation_cache.clear()
        self._compilation_stats.clear()

        # Clear XLA compilation cache
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        except ImportError:
            pass

    def benchmark_compilation(self, model: nn.Module,
                            sample_inputs: Union[torch.Tensor, tuple],
                            num_runs: int = 3) -> Dict[str, float]:
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

        for i in range(num_runs):
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