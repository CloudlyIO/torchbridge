"""
Gradient Compression Techniques

Advanced gradient compression for memory and communication efficiency
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class GradientCompressor:
    """Base gradient compressor"""

    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio

    def compress(self, gradients: torch.Tensor) -> torch.Tensor:
        """Compress gradients"""
        return gradients

    def decompress(self, compressed_gradients: torch.Tensor) -> torch.Tensor:
        """Decompress gradients"""
        return compressed_gradients


class LossyGradientCompression(GradientCompressor):
    """Lossy gradient compression using quantization"""

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.quantization_levels = 2 ** bits

    def compress(self, gradients: torch.Tensor) -> torch.Tensor:
        """Quantize gradients"""
        # Simple quantization
        min_val = gradients.min()
        max_val = gradients.max()

        # Normalize to [0, 1]
        normalized = (gradients - min_val) / (max_val - min_val + 1e-8)

        # Quantize
        quantized = torch.round(normalized * (self.quantization_levels - 1))

        return quantized, min_val, max_val

    def decompress(self, compressed_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Dequantize gradients"""
        quantized, min_val, max_val = compressed_data

        # Denormalize
        normalized = quantized / (self.quantization_levels - 1)
        gradients = normalized * (max_val - min_val) + min_val

        return gradients


class AdaptiveCompressionOptimizer:
    """Adaptive compression optimizer"""

    def __init__(self, target_compression_ratio: float = 0.5):
        self.target_compression_ratio = target_compression_ratio
        self.current_ratio = target_compression_ratio

    def adapt_compression(self, gradient_norm: float):
        """Adapt compression based on gradient norm"""
        if gradient_norm > 1.0:
            self.current_ratio = min(1.0, self.current_ratio * 1.1)
        else:
            self.current_ratio = max(0.1, self.current_ratio * 0.9)


class QuantizedGradientAccumulation:
    """Quantized gradient accumulation"""

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = None
        self.step_count = 0

    def accumulate(self, gradients: Dict[str, torch.Tensor]):
        """Accumulate quantized gradients"""
        if self.accumulated_gradients is None:
            self.accumulated_gradients = {k: torch.zeros_like(v) for k, v in gradients.items()}

        for name, grad in gradients.items():
            self.accumulated_gradients[name] += grad

        self.step_count += 1

        if self.step_count >= self.accumulation_steps:
            # Average accumulated gradients
            for name in self.accumulated_gradients:
                self.accumulated_gradients[name] /= self.accumulation_steps

            result = self.accumulated_gradients
            self.accumulated_gradients = None
            self.step_count = 0
            return result

        return None