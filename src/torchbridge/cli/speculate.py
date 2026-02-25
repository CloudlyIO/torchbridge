"""
Speculate command for TorchBridge CLI.

Displays speculative decoding method compatibility information per backend,
including optimal methods, fallback chains, and the full compatibility matrix.
"""

import argparse
import json
import sys


class SpeculateCommand:
    """Speculative decoding compatibility and method info command."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the speculate command with argument parser."""
        parser = subparsers.add_parser(
            "speculate",
            help="Show speculative decoding method compatibility for backends",
            description="Display backend-aware speculative decoding information",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Supported Speculative Methods:
  draft_model    - Standard draft-verify with a smaller assistant model
  prompt_lookup  - N-gram matching from prompt (universal, no dependencies)

Not Implemented (enum-only, raise NotImplementedError):
  eagle          - Requires custom CUDA kernels and adapter checkpoints
  medusa         - Requires multi-head adapter checkpoints
  layer_skip     - Requires early-exit model architecture

Examples:
  torchbridge speculate                       # Auto-detect backend
  torchbridge speculate --backend nvidia      # NVIDIA-specific info
  torchbridge speculate --show-matrix         # Full compatibility matrix
  torchbridge speculate --ci                  # JSON output for CI
            """,
        )

        parser.add_argument(
            "--backend",
            choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
            default="auto",
            help="Target backend (default: auto-detect)",
        )

        parser.add_argument(
            "--method",
            choices=[
                "auto", "draft_model", "eagle", "layer_skip",
                "medusa", "prompt_lookup",
            ],
            default=None,
            help="Check if a specific method is supported",
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
        """Execute the speculate command."""
        from torchbridge.inference.speculative.compatibility import (
            SpeculationCompatibilityMatrix,
        )
        from torchbridge.inference.speculative.methods import (
            SPECULATIVE_METHOD_SPECS,
            SpeculativeMethod,
        )

        # Resolve backend
        backend = SpeculateCommand._resolve_backend(args.backend)

        if args.show_matrix:
            return SpeculateCommand._show_matrix(args.ci)

        # Check specific method support
        if args.method and args.method != "auto":
            method = SpeculativeMethod.from_string(args.method)
            supported = SpeculationCompatibilityMatrix.is_method_supported(
                method, backend
            )
            spec = SPECULATIVE_METHOD_SPECS.get(method)

            if args.ci:
                result = {
                    "backend": backend.value,
                    "method": method.value,
                    "supported": supported,
                    "display_name": spec.display_name if spec else "Unknown",
                }
                print(json.dumps(result, indent=2))
                return 0

            status = "SUPPORTED" if supported else "NOT SUPPORTED"
            print(f"{method.value} on {backend.value}: {status}")
            if spec:
                print(f"  {spec.description}")
            return 0

        # Default: show optimal method and supported methods
        optimal = SpeculationCompatibilityMatrix.get_optimal_method(backend)
        supported_methods = SpeculationCompatibilityMatrix.get_supported_methods(backend)
        spec = SPECULATIVE_METHOD_SPECS.get(optimal)

        if args.ci:
            result = {
                "backend": backend.value,
                "optimal_method": optimal.value,
                "supported_methods": [m.value for m in supported_methods],
            }
            print(json.dumps(result, indent=2))
            return 0

        # Human-readable output
        print(f"Speculative Decoding — {backend.value.upper()}")
        print("=" * 50)
        print(f"  Optimal method:    {optimal.value}")
        if spec:
            print(f"  Display name:      {spec.display_name}")
            print(f"  Draft model needed: {spec.requires_draft_model}")
            print(f"  Description:       {spec.description}")
        print(
            f"  Supported methods: "
            f"{', '.join(m.value for m in supported_methods)}"
        )
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
        from torchbridge.inference.speculative.compatibility import (
            SpeculationCompatibilityMatrix,
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
            supported = SpeculationCompatibilityMatrix.get_supported_methods(
                backend, arch
            )
            optimal = SpeculationCompatibilityMatrix.get_optimal_method(backend, arch)
            arch_name = arch.value if arch else "—"
            rows.append({
                "backend": backend.value,
                "architecture": arch_name,
                "optimal": optimal.value,
                "supported": [m.value for m in supported],
            })

        if ci_mode:
            print(json.dumps(rows, indent=2))
            return 0

        # Table output
        print("Speculative Decoding Compatibility Matrix")
        print("=" * 80)
        print(
            f"{'Backend':<10} {'Architecture':<20} {'Optimal':<14} {'Supported'}"
        )
        print("-" * 80)
        for row in rows:
            supported_str = ", ".join(row["supported"])
            print(
                f"{row['backend']:<10} {row['architecture']:<20} "
                f"{row['optimal']:<14} {supported_str}"
            )
        print()
        return 0


def main() -> None:
    """Standalone entry point for tb-speculate."""
    parser = argparse.ArgumentParser(
        prog="tb-speculate",
        description="TorchBridge Speculative Decoding Compatibility Info",
    )

    # Reuse the same argument setup
    parser.add_argument(
        "--backend",
        choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--method",
        choices=[
            "auto", "draft_model", "eagle", "layer_skip",
            "medusa", "prompt_lookup",
        ],
        default=None,
    )
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument("--ci", action="store_true")

    args = parser.parse_args()
    sys.exit(SpeculateCommand.execute(args))
