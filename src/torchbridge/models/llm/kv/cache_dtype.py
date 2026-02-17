"""
KV-Cache Data Type Definitions

Defines KV-cache quantization dtypes and their metadata for backend-aware
cache optimization. Mirrors the pattern in precision/quantization/formats.py.
"""

from dataclasses import dataclass
from enum import Enum

import torch


class KVCacheDtype(Enum):
    """KV-cache storage data types."""
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    NVFP4 = "nvfp4"
    PASSTHROUGH = "passthrough"

    @classmethod
    def from_string(cls, value: str) -> "KVCacheDtype":
        """Convert string to KVCacheDtype, case-insensitive."""
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "fp8": cls.FP8_E4M3,
            "fp4": cls.NVFP4,
            "float16": cls.FP16,
            "bfloat16": cls.BF16,
            "none": cls.PASSTHROUGH,
            "auto": cls.PASSTHROUGH,
        }
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown KV-cache dtype: '{value}'. "
            f"Valid dtypes: {[m.value for m in cls]}"
        )


@dataclass(frozen=True)
class KVDtypeSpec:
    """Metadata for a KV-cache data type."""
    bits: int
    display_name: str
    memory_factor: float
    torch_dtype: torch.dtype | None
    requires_hardware_support: bool


KV_DTYPE_SPECS: dict[KVCacheDtype, KVDtypeSpec] = {
    KVCacheDtype.FP16: KVDtypeSpec(
        bits=16,
        display_name="FP16",
        memory_factor=1.0,
        torch_dtype=torch.float16,
        requires_hardware_support=False,
    ),
    KVCacheDtype.BF16: KVDtypeSpec(
        bits=16,
        display_name="BFloat16",
        memory_factor=1.0,
        torch_dtype=torch.bfloat16,
        requires_hardware_support=False,
    ),
    KVCacheDtype.FP8_E4M3: KVDtypeSpec(
        bits=8,
        display_name="FP8 E4M3",
        memory_factor=0.5,
        torch_dtype=torch.float8_e4m3fn,
        requires_hardware_support=True,
    ),
    KVCacheDtype.NVFP4: KVDtypeSpec(
        bits=4,
        display_name="NVFP4 (Blackwell DC)",
        memory_factor=0.25,
        torch_dtype=None,  # No native torch dtype; uses custom packing
        requires_hardware_support=True,
    ),
    KVCacheDtype.PASSTHROUGH: KVDtypeSpec(
        bits=0,
        display_name="Passthrough (no quantization)",
        memory_factor=1.0,
        torch_dtype=None,
        requires_hardware_support=False,
    ),
}


def get_kv_dtype_spec(dtype: KVCacheDtype) -> KVDtypeSpec:
    """Get the metadata spec for a KV-cache dtype."""
    return KV_DTYPE_SPECS[dtype]
