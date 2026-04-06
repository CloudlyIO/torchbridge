"""
Cache command for TorchBridge CLI.

Displays KV-cache dtype compatibility information per backend, including
optimal dtypes, memory factors, and the full compatibility matrix.
"""

import argparse
import json
import sys


class CacheCommand:
    """KV-cache compatibility and optimization info command."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the cache command with argument parser."""
        parser = subparsers.add_parser(
            "cache",
            help="Show KV-cache dtype compatibility for backends",
            description="Display backend-aware KV-cache dtype information",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
KV-Cache Dtypes:
  fp16         - FP16 (universal, 1.0x memory)
  bf16         - BFloat16 (universal, 1.0x memory)
  fp8_e4m3     - FP8 E4M3 (Hopper/Ada/Blackwell/CDNA3+, 0.5x memory)
  nvfp4        - NVFP4 4-bit (Blackwell DC only, 0.25x memory)
  passthrough  - No quantization (CPU fallback)

Examples:
  torchbridge cache                       # Auto-detect backend
  torchbridge cache --backend nvidia      # NVIDIA-specific info
  torchbridge cache --show-matrix         # Full compatibility matrix
  torchbridge cache --ci                  # JSON output for CI
            """,
        )

        parser.add_argument(
            "--backend",
            choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
            default="auto",
            help="Target backend (default: auto-detect)",
        )

        parser.add_argument(
            "--show-matrix",
            action="store_true",
            help="Display the full compatibility matrix table",
        )

        parser.add_argument(
            "--ci",
            action="store_true",
            help="Output in JSON format for CI pipelines",
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the cache command."""
        from torchbridge.models.llm.kv.cache_compatibility import (
            KVCacheCompatibilityMatrix,
        )
        from torchbridge.models.llm.kv.cache_dtype import KV_DTYPE_SPECS

        # Resolve backend
        backend = CacheCommand._resolve_backend(args.backend)

        if args.show_matrix:
            return CacheCommand._show_matrix(args.ci)

        # Get optimal dtype and supported dtypes
        optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(backend)
        supported = KVCacheCompatibilityMatrix.get_supported_dtypes(backend)
        spec = KV_DTYPE_SPECS.get(optimal)

        if args.ci:
            result = {
                "backend": backend.value,
                "optimal_dtype": optimal.value,
                "memory_factor": spec.memory_factor if spec else 1.0,
                "supported_dtypes": [d.value for d in supported],
            }
            print(json.dumps(result, indent=2))
            return 0

        # Human-readable output
        print(f"KV-Cache Optimization — {backend.value.upper()}")
        print("=" * 50)
        print(f"  Optimal dtype:   {optimal.value}")
        if spec:
            print(f"  Display name:    {spec.display_name}")
            print(f"  Memory factor:   {spec.memory_factor}x")
            print(f"  Bits per value:  {spec.bits}")
        print(f"  Supported dtypes: {', '.join(d.value for d in supported)}")
        print()

        return 0

    @staticmethod
    def _resolve_backend(backend_str: str):
        """Resolve backend string to HardwareBackend enum."""
        from torchbridge.core.config import HardwareBackend

        if backend_str == "auto":
            import torch

            if torch.cuda.is_available():
                if hasattr(torch.version, "hip") and torch.version.hip:
                    return HardwareBackend.AMD
                return HardwareBackend.CUDA
            return HardwareBackend.CPU

        mapping = {
            "nvidia": HardwareBackend.CUDA,
            "amd": HardwareBackend.AMD,
            "trainium": HardwareBackend.TRAINIUM,
            "tpu": HardwareBackend.TPU,
            "cpu": HardwareBackend.CPU,
        }
        return mapping[backend_str]

    @staticmethod
    def _show_matrix(ci_mode: bool) -> int:
        """Display the full compatibility matrix."""
        from torchbridge.core.config import (
            AMDArchitecture,
            HardwareBackend,
            NVIDIAArchitecture,
            TPUVersion,
            TrainiumArchitecture,
        )
        from torchbridge.models.llm.kv.cache_compatibility import (
            KVCacheCompatibilityMatrix,
        )

        rows = []
        combos = [
            (HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC),
            (HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_CONSUMER),
            (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
            (HardwareBackend.CUDA, NVIDIAArchitecture.ADA),
            (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
            (HardwareBackend.CUDA, NVIDIAArchitecture.TURING),
            (HardwareBackend.AMD, AMDArchitecture.CDNA4),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
            (HardwareBackend.AMD, AMDArchitecture.CDNA2),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN3),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
            (HardwareBackend.TPU, TPUVersion.V7),
            (HardwareBackend.TPU, TPUVersion.V5E),
            (HardwareBackend.CPU, None),
        ]

        for backend, arch in combos:
            supported = KVCacheCompatibilityMatrix.get_supported_dtypes(backend, arch)
            optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(backend, arch)
            arch_name = arch.value if arch else "—"
            rows.append(
                {
                    "backend": backend.value,
                    "architecture": arch_name,
                    "optimal": optimal.value,
                    "supported": [d.value for d in supported],
                }
            )

        if ci_mode:
            print(json.dumps(rows, indent=2))
            return 0

        # Table output
        print("KV-Cache Compatibility Matrix")
        print("=" * 70)
        print(f"{'Backend':<10} {'Architecture':<20} {'Optimal':<12} {'Supported'}")
        print("-" * 70)
        for row in rows:
            supported_str = ", ".join(row["supported"])
            print(
                f"{row['backend']:<10} {row['architecture']:<20} "
                f"{row['optimal']:<12} {supported_str}"
            )
        print()
        return 0


def main() -> None:
    """Standalone entry point for tb-cache."""
    parser = argparse.ArgumentParser(
        prog="tb-cache",
        description="TorchBridge KV-Cache Compatibility Info",
    )

    # Reuse the same argument setup
    parser.add_argument(
        "--backend",
        choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
        default="auto",
    )
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument("--ci", action="store_true")

    args = parser.parse_args()
    sys.exit(CacheCommand.execute(args))
