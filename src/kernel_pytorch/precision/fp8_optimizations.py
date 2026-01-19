"""
FP8-Aware Optimizations and Layer Implementations

Specialized optimizations for FP8 training including custom layers,
optimizers, and automatic model conversion utilities.

Key Features:
- FP8-optimized Linear layers
- FP8-aware optimizers with scaling
- Automatic model conversion utilities
- Loss scaling for FP8 training
- Integration with existing optimization framework

References:
    - FP8 Training Best Practices: https://docs.nvidia.com/deeplearning/transformer-engine/
    - Mixed Precision Training: https://arxiv.org/abs/1710.03740
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union, Type, Tuple
import warnings
import math

try:
    import transformer_engine.pytorch as te
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    warnings.warn("Transformer Engine not available - using fallback FP8 implementations")

from .fp8_training_engine import FP8Format, FP8Config


class FP8LinearLayer(nn.Module):
    """
    FP8-optimized Linear layer with automatic scaling

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term
        fp8_config: FP8 configuration
        device: Target device

    Example:
        >>> config = FP8Config(forward_format=FP8Format.E4M3)
        >>> layer = FP8LinearLayer(512, 256, bias=True, fp8_config=config)
        >>> x = torch.randn(32, 512)
        >>> output = layer(x)  # FP8 computation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fp8_config: Optional[FP8Config] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.fp8_config = fp8_config or FP8Config()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use Transformer Engine layer if available
        if TRANSFORMER_ENGINE_AVAILABLE and self.fp8_config.use_te_linear:
            self.te_linear = te.Linear(
                in_features,
                out_features,
                bias=bias,
                device=device
            )
            self.use_te = True
        else:
            # Fallback to standard linear with manual FP8 handling
            self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device))
            self.bias = nn.Parameter(torch.randn(out_features, device=device)) if bias else None
            self.use_te = False

            # Initialize weights
            self._init_weights()

        # FP8 scaling factors
        self.register_buffer('weight_scale', torch.ones(1, device=device))
        self.register_buffer('input_scale', torch.ones(1, device=device))
        self.register_buffer('output_scale', torch.ones(1, device=device))

        # AMAX tracking for dynamic scaling
        self.register_buffer('weight_amax', torch.zeros(1, device=device))
        self.register_buffer('input_amax', torch.zeros(1, device=device))
        self.register_buffer('output_amax', torch.zeros(1, device=device))

    def _init_weights(self):
        """Initialize weights using standard initialization"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def _quantize_to_fp8(self, tensor: torch.Tensor, scale: torch.Tensor, format: FP8Format) -> torch.Tensor:
        """
        Quantize tensor to FP8 format (simulated with appropriate clamping)

        Args:
            tensor: Input tensor to quantize
            scale: Scaling factor
            format: FP8 format (E4M3 or E5M2)

        Returns:
            Quantized tensor
        """
        # Scale the tensor
        scaled_tensor = tensor * scale

        # Clamp to FP8 range based on format
        if format == FP8Format.E4M3:
            # E4M3: max value is 448
            max_val = 448.0
        else:  # E5M2
            # E5M2: max value is 57344
            max_val = 57344.0

        # Clamp and simulate quantization
        quantized = torch.clamp(scaled_tensor, -max_val, max_val)

        # Simulate quantization noise (optional for training robustness)
        if self.training:
            # Add small amount of quantization noise
            noise_scale = max_val / 128  # Very small noise
            noise = torch.randn_like(quantized) * noise_scale * 0.001
            quantized = quantized + noise

        return quantized / scale  # Return unscaled for computation

    def _update_amax(self, tensor: torch.Tensor, amax_buffer: torch.Tensor):
        """Update AMAX tracking for dynamic scaling"""
        current_amax = tensor.abs().max()
        # Exponential moving average - update the passed buffer directly
        amax_buffer.data = 0.99 * amax_buffer + 0.01 * current_amax

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 optimization"""

        if self.use_te and TRANSFORMER_ENGINE_AVAILABLE:
            # Use Transformer Engine implementation
            return self.te_linear(input)

        # Manual FP8 implementation
        # Update AMAX tracking
        if self.training:
            self._update_amax(input, self.input_amax)
            self._update_amax(self.weight, self.weight_amax)

        # Simulate FP8 quantization for weights and inputs
        if self.fp8_config.forward_format in [FP8Format.E4M3, FP8Format.E5M2]:
            # Quantize weights
            weight_fp8 = self._quantize_to_fp8(
                self.weight, self.weight_scale, self.fp8_config.forward_format
            )

            # Quantize inputs
            input_fp8 = self._quantize_to_fp8(
                input, self.input_scale, self.fp8_config.forward_format
            )

            # Perform linear operation
            output = F.linear(input_fp8, weight_fp8, self.bias)
        else:
            # Standard computation
            output = F.linear(input, self.weight, self.bias)

        # Update output AMAX
        if self.training:
            self._update_amax(output, self.output_amax)

        return output

    def update_fp8_scales(self):
        """Update FP8 scaling factors based on AMAX history"""
        if not self.training:
            return

        # Calculate optimal scales based on AMAX values
        # For E4M3: max value is 448, for E5M2: max value is 57344
        max_vals = {
            FP8Format.E4M3: 448.0,
            FP8Format.E5M2: 57344.0
        }

        max_val = max_vals.get(self.fp8_config.forward_format, 448.0)
        margin = 2 ** self.fp8_config.margin

        # Update scales to use most of the FP8 range
        if self.weight_amax > 0:
            self.weight_scale.data = max_val / (self.weight_amax * margin)

        if self.input_amax > 0:
            self.input_scale.data = max_val / (self.input_amax * margin)

        if self.output_amax > 0:
            self.output_scale.data = max_val / (self.output_amax * margin)


class FP8Optimizer:
    """
    FP8-aware optimizer wrapper that handles scaling and overflow detection

    Args:
        optimizer: Base PyTorch optimizer
        fp8_config: FP8 configuration
        model: Model being optimized

    Example:
        >>> base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> fp8_optimizer = FP8Optimizer(base_optimizer, fp8_config, model)
        >>> fp8_optimizer.step()  # Handles FP8 scaling automatically
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        fp8_config: FP8Config,
        model: nn.Module
    ):
        self.optimizer = optimizer
        self.fp8_config = fp8_config
        self.model = model

        # Scaling state
        self.scale = fp8_config.initial_scale
        self.step_count = 0
        self.overflow_count = 0
        self.last_overflow_step = -1

    def step(self, closure=None) -> bool:
        """
        Optimizer step with FP8 overflow handling

        Returns:
            True if step was successful (no overflow)
        """
        self.step_count += 1

        # Check for gradient overflow
        grad_norm = self._compute_grad_norm()
        overflow = self._check_overflow(grad_norm)

        if overflow:
            self.overflow_count += 1
            self.last_overflow_step = self.step_count
            self._handle_overflow()
            return False

        # Unscale gradients
        self._unscale_gradients()

        # Update FP8 scales in model layers
        self._update_model_scales()

        # Perform optimizer step
        if closure is not None:
            loss = self.optimizer.step(closure)
        else:
            self.optimizer.step()

        # Update scale if no overflow
        self._update_scale()

        return True

    def _compute_grad_norm(self) -> float:
        """Compute gradient norm for overflow detection"""
        total_norm = 0.0
        param_count = 0

        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(dtype=torch.float32)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

        return math.sqrt(total_norm) if param_count > 0 else 0.0

    def _check_overflow(self, grad_norm: float) -> bool:
        """Check if gradient overflow occurred"""
        # Simple overflow detection based on gradient norm
        max_grad_norm = 1000.0  # Threshold for overflow
        return grad_norm > max_grad_norm or not math.isfinite(grad_norm)

    def _handle_overflow(self):
        """Handle gradient overflow"""
        # Reduce scale
        self.scale = max(self.scale * self.fp8_config.backoff_factor, 1e-6)

        # Zero gradients to skip this step
        self.optimizer.zero_grad()

        print(f"FP8 overflow at step {self.step_count}, reducing scale to {self.scale}")

    def _unscale_gradients(self):
        """Unscale gradients for FP8 training"""
        inv_scale = 1.0 / self.scale

        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)

    def _update_model_scales(self):
        """Update FP8 scales in model layers"""
        for module in self.model.modules():
            if isinstance(module, FP8LinearLayer):
                module.update_fp8_scales()

    def _update_scale(self):
        """Update FP8 scale based on overflow history"""
        # Grow scale if no recent overflow
        steps_since_overflow = self.step_count - self.last_overflow_step

        if steps_since_overflow > self.fp8_config.scaling_interval:
            self.scale = min(
                self.scale * self.fp8_config.growth_factor,
                self.fp8_config.initial_scale * 1024  # Max scale limit
            )

    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Get optimizer state dict with FP8 state"""
        state = self.optimizer.state_dict()
        state['fp8_scale'] = self.scale
        state['fp8_step_count'] = self.step_count
        state['fp8_overflow_count'] = self.overflow_count
        return state

    def load_state_dict(self, state_dict):
        """Load optimizer state dict with FP8 state"""
        fp8_state = {
            'scale': state_dict.pop('fp8_scale', self.scale),
            'step_count': state_dict.pop('fp8_step_count', 0),
            'overflow_count': state_dict.pop('fp8_overflow_count', 0)
        }

        self.optimizer.load_state_dict(state_dict)

        # Restore FP8 state
        self.scale = fp8_state['scale']
        self.step_count = fp8_state['step_count']
        self.overflow_count = fp8_state['overflow_count']


class FP8LossScaler:
    """
    Dynamic loss scaler for FP8 training

    Args:
        initial_scale: Initial loss scale
        growth_factor: Factor to grow scale
        backoff_factor: Factor to reduce scale on overflow
        growth_interval: Steps between scale growth attempts

    Example:
        >>> scaler = FP8LossScaler(initial_scale=65536)
        >>> scaled_loss = scaler.scale_loss(loss)
        >>> scaler.unscale_gradients(optimizer)
        >>> if scaler.step(optimizer):
        >>>     # Successful step
        >>>     pass
    """

    def __init__(
        self,
        initial_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

        self.step_count = 0
        self.overflow_count = 0
        self.last_overflow_step = -1

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation"""
        return loss * self.scale

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients after backward pass"""
        inv_scale = 1.0 / self.scale

        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Perform optimizer step with overflow detection

        Returns:
            True if step was successful
        """
        self.step_count += 1

        # Check for overflow
        overflow = self._check_overflow(optimizer)

        if overflow:
            self.overflow_count += 1
            self.last_overflow_step = self.step_count
            self.scale *= self.backoff_factor
            optimizer.zero_grad()
            return False

        # Perform step
        optimizer.step()

        # Update scale
        self._update_scale()

        return True

    def _check_overflow(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check for gradient overflow"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        return True
        return False

    def _update_scale(self):
        """Update scale based on overflow history"""
        steps_since_overflow = self.step_count - self.last_overflow_step

        if steps_since_overflow >= self.growth_interval:
            self.scale *= self.growth_factor

    def get_scale(self) -> float:
        """Get current scale value"""
        return self.scale


def convert_model_to_fp8(
    model: nn.Module,
    fp8_config: Optional[FP8Config] = None,
    convert_linear: bool = True,
    convert_attention: bool = True,
    inplace: bool = False
) -> nn.Module:
    """
    Convert a model to use FP8 layers

    Args:
        model: Model to convert
        fp8_config: FP8 configuration
        convert_linear: Whether to convert Linear layers
        convert_attention: Whether to convert attention layers
        inplace: Whether to modify model in-place

    Returns:
        Model with FP8 layers

    Example:
        >>> model = MyTransformerModel()
        >>> fp8_model = convert_model_to_fp8(model)
        >>> # Now model uses FP8 optimized layers
    """
    if not inplace:
        model = copy.deepcopy(model)

    fp8_config = fp8_config or FP8Config()

    def convert_module(module: nn.Module, name: str = ""):
        """Recursively convert modules"""
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            # Convert Linear layers
            if convert_linear and isinstance(child, nn.Linear):
                fp8_layer = FP8LinearLayer(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    fp8_config=fp8_config,
                    device=child.weight.device
                )

                # Copy weights if not using Transformer Engine
                if not fp8_layer.use_te:
                    fp8_layer.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        fp8_layer.bias.data.copy_(child.bias.data)

                setattr(module, child_name, fp8_layer)
                print(f"Converted {full_name} to FP8LinearLayer")

            # Convert attention layers (if available)
            elif convert_attention and "attention" in child_name.lower():
                # This would require specific attention layer conversion
                # For now, just convert any linear layers within attention
                convert_module(child, full_name)

            else:
                # Recursively convert child modules
                convert_module(child, full_name)

    convert_module(model)
    return model


def create_fp8_optimizer(
    model: nn.Module,
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
    fp8_config: Optional[FP8Config] = None,
    **optimizer_kwargs
) -> FP8Optimizer:
    """
    Create FP8-aware optimizer

    Args:
        model: Model to optimize
        optimizer_class: Base optimizer class
        fp8_config: FP8 configuration
        **optimizer_kwargs: Additional optimizer arguments

    Returns:
        FP8-aware optimizer

    Example:
        >>> model = convert_model_to_fp8(model)
        >>> optimizer = create_fp8_optimizer(model, torch.optim.AdamW, lr=1e-4)
        >>> success = optimizer.step()
    """
    base_optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    fp8_config = fp8_config or FP8Config()

    return FP8Optimizer(base_optimizer, fp8_config, model)


# Utility functions
def estimate_fp8_speedup(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Estimate FP8 training speedup for a model

    Args:
        model: Model to analyze
        input_shape: Input tensor shape
        device: Target device

    Returns:
        Dictionary with speedup estimates
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Count linear layers
    linear_layers = sum(1 for m in model.modules() if isinstance(m, (nn.Linear, FP8LinearLayer)))

    # Estimate compute savings
    # FP8 typically provides 2x speedup for GEMM operations on H100
    gemm_speedup = 2.0 if device.type == 'cuda' else 1.2

    # Estimate memory savings
    # FP8 uses half the memory of FP16
    memory_savings = 2.0

    # Overall speedup depends on proportion of time spent in linear layers
    linear_compute_ratio = min(linear_layers * 0.1, 0.8)  # Heuristic
    overall_speedup = 1.0 + (gemm_speedup - 1.0) * linear_compute_ratio

    return {
        'linear_layers': linear_layers,
        'gemm_speedup': gemm_speedup,
        'memory_savings': memory_savings,
        'overall_speedup': overall_speedup,
        'linear_compute_ratio': linear_compute_ratio
    }