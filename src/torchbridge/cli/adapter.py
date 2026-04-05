"""
Adapter command for TorchBridge CLI.

Provides adapter method recommendation and compatibility info.
For adapter injection use PEFT, torchao, or Unsloth directly.
"""

import argparse
import json
import sys
from typing import Any


def _register_subcommands(subparsers: Any) -> None:
    """Register adapter subcommands (shared by CLI and standalone entry)."""
    # recommend subcommand
    rec_parser = subparsers.add_parser(
        "recommend",
        help="Recommend adapter method for hardware",
    )
    rec_parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=["nvidia", "amd", "trainium", "tpu", "cpu"],
        help="Target hardware backend (default: cpu)",
    )
    rec_parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    rec_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Hidden dimension for adapter parameter estimate (default: 4096)",
    )
    rec_parser.add_argument(
        "--num-modules",
        type=int,
        default=4,
        help="Number of projected modules for parameter estimate (default: 4)",
    )
    rec_parser.add_argument(
        "--ci",
        action="store_true",
        help="Output JSON for CI pipelines",
    )

    # info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Show adapter compatibility info for all backends",
    )
    info_parser.add_argument(
        "--ci",
        action="store_true",
        help="Output JSON for CI pipelines",
    )


class AdapterCommand:
    """Adapter training CLI command."""

    @staticmethod
    def register(subparsers: Any) -> None:
        """Register the adapter command with argument parser."""
        parser = subparsers.add_parser(
            "adapter",
            help="Adapter method recommendation",
            description="Recommend adapter methods and show compatibility matrices",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  torchbridge adapter recommend --backend nvidia
  torchbridge adapter recommend --backend amd --rank 8 --ci
  torchbridge adapter info
  torchbridge adapter info --ci
            """,
        )

        sub = parser.add_subparsers(dest="adapter_action", metavar="<action>")
        _register_subcommands(sub)

    @staticmethod
    def execute(args: Any) -> int:
        """Execute the adapter command."""
        action = getattr(args, "adapter_action", None)

        if action == "recommend":
            return _show_recommend(args)
        if action == "info":
            return _show_info(args)

        # No subcommand — show help
        print("Usage: torchbridge adapter <recommend|info>")
        print("Run 'torchbridge adapter --help' for details.")
        return 1


def _show_recommend(args: Any) -> int:
    """Show adapter method recommendation for hardware."""
    from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
    from torchbridge.core.config import HardwareBackend

    backend_map = {
        "nvidia": HardwareBackend.CUDA,
        "amd": HardwareBackend.AMD,
        "trainium": HardwareBackend.TRAINIUM,
        "tpu": HardwareBackend.TPU,
        "cpu": HardwareBackend.CPU,
    }
    backend = backend_map[args.backend]

    optimal = AdapterCompatibilityMatrix.get_optimal(backend)
    chain = AdapterCompatibilityMatrix.get_fallback_chain(backend)
    base_fmt = AdapterCompatibilityMatrix.get_base_quant_format(backend)

    if args.ci:
        output = {
            "backend": args.backend,
            "rank": args.rank,
            "hidden_dim": args.hidden_dim,
            "num_modules": args.num_modules,
            "optimal_method": optimal.value,
            "fallback_chain": [m.value for m in chain],
            "base_quant_format": base_fmt.value if base_fmt else None,
            "qlora_supported": base_fmt is not None,
        }
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print("Adapter Method Recommendation")
    print("=" * 50)
    print()
    print(f"  Backend:           {args.backend}")
    print(f"  Rank:              {args.rank}")
    print(f"  Optimal method:    {optimal.value}")
    print(f"  Fallback chain:    {' -> '.join(m.value for m in chain)}")
    print()
    if base_fmt:
        print(f"  QLoRA base format: {base_fmt.value}")
        print("  QLoRA supported:   Yes")
    else:
        print(f"  QLoRA supported:   No (not available on {args.backend})")
    print()

    # Param estimate
    r = args.rank
    hidden = args.hidden_dim
    n_modules = args.num_modules
    params_per_module = 2 * hidden * r  # A + B
    total_modules = n_modules * params_per_module
    print(f"Estimated adapter parameters ({hidden} hidden, {n_modules} modules):")
    print(f"  LoRA:  {total_modules:,} ({total_modules / 1e6:.1f}M)")
    print(
        f"  DoRA:  {total_modules + n_modules * hidden:,} "
        f"({(total_modules + n_modules * hidden) / 1e6:.1f}M)"
    )
    print()
    return 0


def _show_info(args: Any) -> int:
    """Show adapter compatibility info for all backends."""
    from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
    from torchbridge.core.config import HardwareBackend

    backends = [
        HardwareBackend.CUDA,
        HardwareBackend.AMD,
        HardwareBackend.TRAINIUM,
        HardwareBackend.TPU,
        HardwareBackend.CPU,
    ]

    if getattr(args, "ci", False):
        output: dict[str, Any] = {"backends": {}}
        for b in backends:
            chain = AdapterCompatibilityMatrix.get_fallback_chain(b)
            base_fmt = AdapterCompatibilityMatrix.get_base_quant_format(b)
            output["backends"][b.value] = {
                "optimal": chain[0].value,
                "methods": [m.value for m in chain],
                "qlora_format": base_fmt.value if base_fmt else None,
            }
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print("Adapter Compatibility Matrix")
    print("=" * 60)
    print()
    print(f"  {'Backend':<12} {'Optimal':<10} {'Methods':<30} {'QLoRA'}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 30} {'-' * 10}")

    for b in backends:
        chain = AdapterCompatibilityMatrix.get_fallback_chain(b)
        base_fmt = AdapterCompatibilityMatrix.get_base_quant_format(b)
        methods_str = ", ".join(m.value for m in chain)
        qlora_str = base_fmt.value if base_fmt else "N/A"
        print(f"  {b.value:<12} {chain[0].value:<10} {methods_str:<30} {qlora_str}")

    print()
    return 0


def main(args: list[str] | None = None) -> int:
    """Entry point for tb-adapter."""
    parser = argparse.ArgumentParser(
        prog="tb-adapter",
        description="Adapter method recommendation",
    )

    sub = parser.add_subparsers(dest="adapter_action", metavar="<action>")
    _register_subcommands(sub)

    if args is None:
        args = sys.argv[1:]

    parsed = parser.parse_args(args)

    if not parsed.adapter_action:
        parser.print_help()
        return 1

    return AdapterCommand.execute(parsed)
