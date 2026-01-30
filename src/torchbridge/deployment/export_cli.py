"""
Export CLI for TorchBridge (v0.4.25)

Command-line interface for exporting PyTorch models to various formats.

Usage:
    python -m torchbridge.deployment.export_cli export model.pt \
        --format onnx \
        --output model.onnx \
        --opset 14

    python -m torchbridge.deployment.export_cli validate model.pt \
        --sample-input "(1, 512)" \
        --max-latency 50

Commands:
    export    - Export model to specified format
    validate  - Validate model production readiness
    info      - Show model information
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def parse_shape(shape_str: str) -> tuple[int, ...]:
    """Parse shape string like '(1, 512)' or '1,512' to tuple."""
    shape_str = shape_str.strip().strip("()[]")
    parts = [int(x.strip()) for x in shape_str.split(",")]
    return tuple(parts)


def load_model(model_path: str) -> nn.Module:
    """Load model from file.

    Supports:
    - TorchScript (.pt, .pth)
    - State dict with architecture definition
    - SafeTensors
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try loading as TorchScript
    try:
        model = torch.jit.load(model_path)
        logger.info(f"Loaded TorchScript model from {model_path}")
        return model
    except Exception:
        pass

    # Try loading as state dict (need architecture)
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, nn.Module):
            logger.info(f"Loaded PyTorch model from {model_path}")
            return state_dict
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            raise ValueError(
                "Model file contains state_dict only. "
                "Please provide model architecture."
            )
        else:
            raise ValueError("Unsupported model format")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}") from e


def create_sample_input(
    shape: tuple[int, ...],
    dtype: str = "float32",
    device: str = "cpu",
) -> torch.Tensor:
    """Create sample input tensor."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    return torch.randn(shape, dtype=torch_dtype, device=device)


# =============================================================================
# Export Command
# =============================================================================

def export_model(args: argparse.Namespace) -> int:
    """Export model to specified format."""
    try:
        # Load model
        model = load_model(args.model)
        model.eval()

        # Create sample input
        sample_shape = parse_shape(args.sample_input)
        sample_input = create_sample_input(sample_shape, args.dtype, args.device)

        # Move model to device if needed
        if args.device != "cpu":
            model = model.to(args.device)
            sample_input = sample_input.to(args.device)

        # Determine output path
        output_path = args.output
        if not output_path:
            model_path = Path(args.model)
            ext_map = {
                "onnx": ".onnx",
                "torchscript": ".pt",
                "safetensors": ".safetensors",
            }
            output_path = str(model_path.with_suffix(ext_map.get(args.format, ".pt")))

        # Export based on format
        if args.format == "onnx":
            return export_to_onnx(model, sample_input, output_path, args)
        elif args.format == "torchscript":
            return export_to_torchscript(model, sample_input, output_path, args)
        elif args.format == "safetensors":
            return export_to_safetensors(model, output_path, args)
        else:
            logger.error(f"Unknown format: {args.format}")
            return 1

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    args: argparse.Namespace,
) -> int:
    """Export to ONNX format."""
    try:
        from ..deployment.onnx_exporter import export_to_onnx as onnx_export

        result = onnx_export(
            model=model,
            output_path=output_path,
            sample_input=sample_input,
            opset_version=args.opset,
            validate=args.validate,
        )

        if result.success:
            logger.info(f"Successfully exported to {output_path}")
            logger.info(f"File size: {result.file_size_mb:.2f} MB")
            return 0
        else:
            logger.error(f"Export failed: {result.error_message}")
            return 1

    except ImportError:
        # Fallback to torch.onnx.export
        logger.info("Using torch.onnx.export directly")
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
        logger.info(f"Successfully exported to {output_path}")
        return 0


def export_to_torchscript(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    args: argparse.Namespace,
) -> int:
    """Export to TorchScript format."""
    try:
        from ..deployment.torchscript_exporter import export_to_torchscript as ts_export

        result = ts_export(
            model=model,
            output_path=output_path,
            sample_input=sample_input,
            method=args.method,
            optimize=args.optimize,
            validate=args.validate,
        )

        if result.success:
            logger.info(f"Successfully exported to {output_path}")
            logger.info(f"File size: {result.file_size_mb:.2f} MB")
            return 0
        else:
            logger.error(f"Export failed: {result.error_message}")
            return 1

    except ImportError:
        # Fallback to torch.jit
        logger.info("Using torch.jit directly")
        if args.method == "script":
            scripted = torch.jit.script(model)
        else:
            scripted = torch.jit.trace(model, sample_input)
        scripted.save(output_path)
        logger.info(f"Successfully exported to {output_path}")
        return 0


def export_to_safetensors(
    model: nn.Module,
    output_path: str,
    args: argparse.Namespace,
) -> int:
    """Export to SafeTensors format."""
    try:
        from ..deployment.safetensors_exporter import export_to_safetensors as st_export

        result = st_export(
            model=model,
            output_path=output_path,
            half_precision=args.half,
        )

        if result.success:
            logger.info(f"Successfully exported to {output_path}")
            logger.info(f"File size: {result.file_size_mb:.2f} MB")
            logger.info(f"Tensors: {result.num_tensors}")
            return 0
        else:
            logger.error(f"Export failed: {result.error_message}")
            return 1

    except ImportError:
        logger.error("SafeTensors library not installed. Install with: pip install safetensors")
        return 1


# =============================================================================
# Validate Command
# =============================================================================

def validate_model(args: argparse.Namespace) -> int:
    """Validate model production readiness."""
    try:
        from ..deployment.production_validator import (
            ProductionRequirements,
            ProductionValidator,
        )

        # Load model
        model = load_model(args.model)
        model.eval()

        # Create sample input
        sample_shape = parse_shape(args.sample_input)
        sample_input = create_sample_input(sample_shape, args.dtype, args.device)

        # Move to device
        if args.device != "cpu":
            model = model.to(args.device)
            sample_input = sample_input.to(args.device)

        # Set requirements
        requirements = ProductionRequirements(
            max_latency_ms=args.max_latency,
            min_throughput=args.min_throughput,
            max_memory_mb=args.max_memory,
            require_onnx_export=not args.skip_onnx,
            require_torchscript_export=not args.skip_torchscript,
        )

        # Run validation
        validator = ProductionValidator()
        result = validator.validate(model, sample_input, requirements)

        # Print results
        print("\n" + "=" * 60)
        print("PRODUCTION READINESS VALIDATION")
        print("=" * 60)
        print(f"\nResult: {'PASSED' if result.passed else 'FAILED'}")
        print(f"\nChecks: {len(result.passed_checks)} passed, {len(result.failed_checks)} failed")

        print("\nCheck Details:")
        for check in result.checks:
            status_icon = "" if check.passed else "" if check.failed else "?"
            print(f"  {status_icon} {check.name}: {check.message}")

        if result.latency_stats:
            print("\nLatency Stats:")
            print(f"  Average: {result.latency_stats.get('avg_ms', 0):.2f} ms")
            print(f"  P95: {result.latency_stats.get('p95_ms', 0):.2f} ms")
            print(f"  Throughput: {result.latency_stats.get('throughput', 0):.2f} samples/sec")

        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")

        print("\n" + "=" * 60)

        return 0 if result.passed else 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


# =============================================================================
# Info Command
# =============================================================================

def model_info(args: argparse.Namespace) -> int:
    """Show model information."""
    try:
        model = load_model(args.model)

        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        # Model structure
        if args.verbose:
            print("\nModel Structure:")
            print(model)

        # File info
        path = Path(args.model)
        print("\nFile Info:")
        print(f"  Path: {path}")
        print(f"  Size: {path.stat().st_size / 1024 / 1024:.2f} MB")

        print("\n" + "=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return 1


# =============================================================================
# CLI Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="TorchBridge Model Export CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Export to ONNX:
    python -m torchbridge.deployment.export_cli export model.pt \\
        --format onnx --sample-input "(1, 512)" --output model.onnx

  Export to TorchScript:
    python -m torchbridge.deployment.export_cli export model.pt \\
        --format torchscript --sample-input "(1, 512)" --method trace

  Export to SafeTensors:
    python -m torchbridge.deployment.export_cli export model.pt \\
        --format safetensors --half

  Validate production readiness:
    python -m torchbridge.deployment.export_cli validate model.pt \\
        --sample-input "(1, 512)" --max-latency 50

  Show model info:
    python -m torchbridge.deployment.export_cli info model.pt --verbose
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to format")
    export_parser.add_argument("model", help="Path to PyTorch model")
    export_parser.add_argument(
        "--format", "-f",
        choices=["onnx", "torchscript", "safetensors"],
        default="onnx",
        help="Output format (default: onnx)"
    )
    export_parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    export_parser.add_argument(
        "--sample-input", "-s",
        default="(1, 512)",
        help="Sample input shape, e.g., '(1, 512)'"
    )
    export_parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Input dtype"
    )
    export_parser.add_argument(
        "--device",
        default="cpu",
        help="Device (cpu or cuda)"
    )
    export_parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    export_parser.add_argument(
        "--method",
        choices=["trace", "script"],
        default="trace",
        help="TorchScript export method"
    )
    export_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimizations"
    )
    export_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate export"
    )
    export_parser.add_argument(
        "--half",
        action="store_true",
        help="Convert to FP16 (SafeTensors)"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate production readiness")
    validate_parser.add_argument("model", help="Path to PyTorch model")
    validate_parser.add_argument(
        "--sample-input", "-s",
        default="(1, 512)",
        help="Sample input shape"
    )
    validate_parser.add_argument(
        "--dtype",
        default="float32",
        help="Input dtype"
    )
    validate_parser.add_argument(
        "--device",
        default="cpu",
        help="Device"
    )
    validate_parser.add_argument(
        "--max-latency",
        type=float,
        help="Maximum latency in ms"
    )
    validate_parser.add_argument(
        "--min-throughput",
        type=float,
        help="Minimum throughput (samples/sec)"
    )
    validate_parser.add_argument(
        "--max-memory",
        type=float,
        help="Maximum memory in MB"
    )
    validate_parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export check"
    )
    validate_parser.add_argument(
        "--skip-torchscript",
        action="store_true",
        help="Skip TorchScript export check"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="Path to PyTorch model")
    info_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed model structure"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "export":
        return export_model(args)
    elif args.command == "validate":
        return validate_model(args)
    elif args.command == "info":
        return model_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
