"""
Advanced Precision Training Module

Production-grade FP8 and FP4 training implementations with support for:
- FP8 E4M3/E5M2 formats for optimal precision/range trade-offs
- NVFP4 (4-bit with two-level microscaling) for Blackwell GPUs
- Dynamic scaling for numerical stability
- Transformer Engine integration
- Automatic mixed precision workflows
- Hardware-optimized training pipelines
- Native PyTorch FP8 types (PyTorch 2.1+)

Key Features:
- 2x training speedup on H100/Blackwell hardware with FP8
- ~3.5x memory reduction vs FP16 with FP4 on Blackwell DC
- Maintained accuracy with <1% loss
- Production reliability and deployment readiness
- Integration with TorchBridge hardware abstraction layer
"""

from .fp4_native import (
    FP4_AVAILABLE,
    FP4_BLOCK_SIZE,
    FP4_HARDWARE_AVAILABLE,
    FP4QuantizedTensor,
    FP4ScaleSpec,
    NativeFP4Linear,
    compute_fp4_scales,
    convert_model_to_fp4,
    dequantize_from_fp4,
    get_fp4_info,
    is_fp4_available,
    is_fp4_native,
    quantize_to_fp4,
)
from .fp8_native import (
    FP8_DTYPES_AVAILABLE,
    # Constants
    FP8_NATIVE_AVAILABLE,
    FP8_SCALED_MM_AVAILABLE,
    # Core types
    FP8Dtype,
    # Inference
    FP8InferenceEngine,
    FP8QuantizedTensor,
    FP8TensorSpec,
    # Native FP8 layer
    NativeFP8Linear,
    benchmark_fp8_layer,
    compute_fp8_scale,
    convert_model_to_native_fp8,
    dequantize_from_fp8,
    get_fp8_dtype,
    get_fp8_info,
    # Functions
    is_fp8_available,
    quantize_to_fp8,
)
from .fp8_training_engine import (
    FP8Config,
    FP8Format,
    FP8TrainingEngine,
    create_fp8_trainer,
    validate_fp8_setup,
)

__all__ = [
    # FP8 Training Engine
    'FP8TrainingEngine',
    'FP8Config',
    'FP8Format',
    'create_fp8_trainer',
    'validate_fp8_setup',

    # Native FP8 (PyTorch 2.1+)
    'FP8Dtype',
    'FP8TensorSpec',
    'FP8QuantizedTensor',
    'NativeFP8Linear',
    'FP8InferenceEngine',
    'is_fp8_available',
    'get_fp8_info',
    'get_fp8_dtype',
    'compute_fp8_scale',
    'quantize_to_fp8',
    'dequantize_from_fp8',
    'convert_model_to_native_fp8',
    'benchmark_fp8_layer',
    'FP8_NATIVE_AVAILABLE',
    'FP8_DTYPES_AVAILABLE',
    'FP8_SCALED_MM_AVAILABLE',

    # Native FP4 (Blackwell DC)
    'FP4ScaleSpec',
    'FP4QuantizedTensor',
    'NativeFP4Linear',
    'is_fp4_available',
    'is_fp4_native',
    'get_fp4_info',
    'compute_fp4_scales',
    'quantize_to_fp4',
    'dequantize_from_fp4',
    'convert_model_to_fp4',
    'FP4_AVAILABLE',
    'FP4_HARDWARE_AVAILABLE',
    'FP4_BLOCK_SIZE',
]
