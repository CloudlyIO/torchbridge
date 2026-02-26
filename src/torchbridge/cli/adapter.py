"""
Adapter command for TorchBridge CLI.

Provides adapter method recommendation, model inspection, and
merge-for-deployment utilities.
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
        choices=["cuda", "amd", "trainium", "tpu", "cpu"],
        help="Target hardware backend (default: cpu)",
    )
    rec_parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    rec_parser.add_argument(
        "--ci",
        action="store_true",
        help="Output JSON for CI pipelines",
    )

    # info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Show adapter info for all backends",
    )
    info_parser.add_argument(
        "--ci",
        action="store_true",
        help="Output JSON for CI pipelines",
    )

    # inject subcommand (dry-run preview)
    inject_parser = subparsers.add_parser(
        "inject",
        help="Preview adapter injection (dry-run)",
    )
    inject_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local path",
    )
    inject_parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["lora", "dora", "qlora", "qdora"],
        help="Adapter method (default: lora)",
    )
    inject_parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    inject_parser.add_argument(
        "--ci",
        action="store_true",
        help="Output JSON for CI pipelines",
    )

    # detect subcommand
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect model family and target modules",
    )
    detect_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local path",
    )
    detect_parser.add_argument(
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
            help="Adapter training and method recommendation",
            description="Recommend adapter methods, inspect models, and merge adapters",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  torchbridge adapter recommend --backend cuda
  torchbridge adapter recommend --backend amd --rank 8 --ci
  torchbridge adapter info
  torchbridge adapter detect --model Qwen/Qwen3-0.6B
  torchbridge adapter inject --model Qwen/Qwen3-0.6B --rank 8
            """,
        )

        sub = parser.add_subparsers(
            dest="adapter_action", metavar="<action>"
        )
        _register_subcommands(sub)

    @staticmethod
    def execute(args: Any) -> int:
        """Execute the adapter command."""
        action = getattr(args, "adapter_action", None)

        if action == "recommend":
            return _show_recommend(args)
        if action == "info":
            return _show_info(args)
        if action == "detect":
            return _show_detect(args)
        if action == "inject":
            return _show_inject_dryrun(args)

        # No subcommand — show help
        print("Usage: torchbridge adapter <recommend|info|detect|inject>")
        print("Run 'torchbridge adapter --help' for details.")
        return 1


def _show_recommend(args: Any) -> int:
    """Show adapter method recommendation for hardware."""
    from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
    from torchbridge.core.config import HardwareBackend

    backend_map = {
        "cuda": HardwareBackend.CUDA,
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
    hidden = 4096  # typical LLM hidden dim
    params_per_module = 2 * hidden * r  # A + B
    total_4_modules = 4 * params_per_module
    print("Estimated adapter parameters (4096 hidden, 4 modules):")
    print(f"  LoRA:  {total_4_modules:,} ({total_4_modules / 1e6:.1f}M)")
    print(f"  DoRA:  {total_4_modules + 4 * hidden:,} "
          f"({(total_4_modules + 4 * hidden) / 1e6:.1f}M)")
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
        from torchbridge.adapters.model_families import MODEL_FAMILY_SPECS

        output: dict[str, Any] = {"backends": {}, "model_families": {}}
        for b in backends:
            chain = AdapterCompatibilityMatrix.get_fallback_chain(b)
            base_fmt = AdapterCompatibilityMatrix.get_base_quant_format(b)
            output["backends"][b.value] = {
                "optimal": chain[0].value,
                "methods": [m.value for m in chain],
                "qlora_format": base_fmt.value if base_fmt else None,
            }
        for family, spec in MODEL_FAMILY_SPECS.items():
            output["model_families"][family.value] = {
                "target_modules": spec.target_modules,
                "has_fused_qkv": spec.has_fused_qkv,
            }
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print("Adapter Compatibility Matrix")
    print("=" * 60)
    print()
    print(f"  {'Backend':<12} {'Optimal':<10} {'Methods':<30} {'QLoRA'}")
    print(f"  {'-'*12} {'-'*10} {'-'*30} {'-'*10}")

    for b in backends:
        chain = AdapterCompatibilityMatrix.get_fallback_chain(b)
        base_fmt = AdapterCompatibilityMatrix.get_base_quant_format(b)
        methods_str = ", ".join(m.value for m in chain)
        qlora_str = base_fmt.value if base_fmt else "N/A"
        print(f"  {b.value:<12} {chain[0].value:<10} {methods_str:<30} {qlora_str}")

    # Model family summary
    from torchbridge.adapters.model_families import MODEL_FAMILY_SPECS

    print()
    print("Supported Model Families")
    print("=" * 60)
    print()
    print(f"  {'Family':<12} {'Target Modules':<30} {'Fused QKV'}")
    print(f"  {'-'*12} {'-'*30} {'-'*10}")

    for family, spec in MODEL_FAMILY_SPECS.items():
        targets_str = ", ".join(spec.target_modules)
        fused_str = "Yes" if spec.has_fused_qkv else "No"
        print(f"  {family.value:<12} {targets_str:<30} {fused_str}")

    print()
    return 0


def _show_detect(args: Any) -> int:
    """Detect model family and show recommended target modules."""
    from torchbridge.adapters.model_families import (
        ModelFamily,
        get_model_family_spec,
    )

    model_name = args.model

    # Try to load model config first (lightweight, no weights)
    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: 'transformers' package is required for model detection.",
              file=sys.stderr)
        print("Install it with: pip install transformers", file=sys.stderr)
        return 1

    try:
        config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(config, "model_type", "unknown")
    except (OSError, ValueError) as exc:
        print(f"Error: Could not load config for '{model_name}'.",
              file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1

    # Use config.model_type to detect family
    from torchbridge.adapters.model_families import MODEL_FAMILY_SPECS

    family = ModelFamily.UNKNOWN
    model_type_lower = str(model_type).lower()
    for fam, spec in MODEL_FAMILY_SPECS.items():
        if model_type_lower in spec.config_type_hints:
            family = fam
            break

    spec = get_model_family_spec(family)
    target_modules = spec.target_modules if spec else ["q_proj", "v_proj"]
    all_linear = spec.all_linear_names if spec else target_modules
    has_fused = spec.has_fused_qkv if spec else False

    if getattr(args, "ci", False):
        output = {
            "model": model_name,
            "model_type": model_type,
            "family": family.value,
            "target_modules": target_modules,
            "all_linear_names": all_linear,
            "has_fused_qkv": has_fused,
        }
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print("Model Family Detection")
    print("=" * 50)
    print()
    print(f"  Model:           {model_name}")
    print(f"  Model type:      {model_type}")
    print(f"  Family:          {family.value}")
    print(f"  Fused QKV:       {'Yes' if has_fused else 'No'}")
    print()
    print(f"  Target modules:  {', '.join(target_modules)}")
    print(f"  All linear:      {', '.join(all_linear)}")
    print()

    if family == ModelFamily.UNKNOWN:
        print("  Note: Unknown model family. Target modules will be")
        print("  auto-detected from module names at injection time.")
        print()

    return 0


def _show_inject_dryrun(args: Any) -> int:
    """Dry-run adapter injection — show what would be adapted without modifying."""
    model_name = args.model

    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: 'transformers' package is required.", file=sys.stderr)
        print("Install it with: pip install transformers", file=sys.stderr)
        return 1

    try:
        config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(config, "model_type", "unknown")
    except (OSError, ValueError) as exc:
        print(f"Error: Could not load config for '{model_name}'.",
              file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1

    from torchbridge.adapters.model_families import (
        MODEL_FAMILY_SPECS,
        ModelFamily,
        get_model_family_spec,
    )

    # Detect family
    family = ModelFamily.UNKNOWN
    model_type_lower = str(model_type).lower()
    for fam, spec in MODEL_FAMILY_SPECS.items():
        if model_type_lower in spec.config_type_hints:
            family = fam
            break

    spec = get_model_family_spec(family)
    target_modules = spec.target_modules if spec else ["q_proj", "v_proj"]
    all_linear = spec.all_linear_names if spec else target_modules

    # Estimate adapter parameters
    method = args.method
    rank = args.rank
    # Use model config hidden_size if available, else estimate
    hidden = getattr(config, "hidden_size", 4096)
    num_targets = len(target_modules)
    params_per_module = 2 * hidden * rank  # A + B matrices
    total_adapter_params = num_targets * params_per_module
    if method in ("dora", "qdora"):
        total_adapter_params += num_targets * hidden  # magnitude vectors

    # Estimate layers
    num_layers = getattr(config, "num_hidden_layers", 1)
    total_params_all_layers = total_adapter_params * num_layers
    adapter_mb = total_params_all_layers * 4 / (1024 * 1024)  # FP32

    if getattr(args, "ci", False):
        output = {
            "model": model_name,
            "model_type": model_type,
            "family": family.value,
            "method": method,
            "rank": rank,
            "hidden_size": hidden,
            "num_layers": num_layers,
            "target_modules": target_modules,
            "all_linear_names": all_linear,
            "adapter_params_per_layer": total_adapter_params,
            "adapter_params_total": total_params_all_layers,
            "adapter_memory_mb": round(adapter_mb, 1),
        }
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print("Adapter Injection Preview (Dry Run)")
    print("=" * 50)
    print()
    print(f"  Model:           {model_name}")
    print(f"  Model type:      {model_type}")
    print(f"  Family:          {family.value}")
    print(f"  Hidden size:     {hidden}")
    print(f"  Layers:          {num_layers}")
    print()
    print(f"  Method:          {method}")
    print(f"  Rank:            {rank}")
    print(f"  Target modules:  {', '.join(target_modules)}")
    print()
    print("  Estimated adapter parameters:")
    print(f"    Per layer:     {total_adapter_params:,}")
    print(f"    Total:         {total_params_all_layers:,} "
          f"({total_params_all_layers / 1e6:.1f}M)")
    print(f"    Memory (FP32): {adapter_mb:.1f} MB")
    print()
    return 0


def main(args: list[str] | None = None) -> int:
    """Entry point for tb-adapter."""
    parser = argparse.ArgumentParser(
        prog="tb-adapter",
        description="Adapter training and method recommendation",
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
