"""
Quantize command for TorchBridge CLI.

Applies backend-aware quantization to PyTorch models with automatic format
selection, fallback chains, and quality validation.
"""

import argparse
import json
import logging
import sys
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantizeCommand:
    """Backend-aware model quantization command."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the quantize command with argument parser."""
        parser = subparsers.add_parser(
            "quantize",
            help="Apply backend-aware quantization to a model",
            description="Quantize PyTorch models with automatic format selection per backend",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Quantization Formats:
  auto             - Auto-select optimal format for detected backend
  int8_dynamic     - INT8 dynamic quantization (universal)
  int8_smoothquant - INT8 with activation smoothing (requires torchao)
  int4_weight_only - INT4 weight-only quantization (requires torchao)
  fp8_e4m3         - FP8 E4M3 (Hopper/Blackwell/CDNA3+)
  nvfp4            - NVFP4 4-bit (Blackwell DC only)
  bf16             - BFloat16 (universal)

Backend Detection:
  By default, the backend is auto-detected. Use --backend to override.

Examples:
  tb-quantize --model model.pt                    # Auto-select format
  tb-quantize --model model.pt --format int8_dynamic
  tb-quantize --model model.pt --backend nvidia --format fp8_e4m3
  tb-quantize --model model.pt --validate         # Quantize + quality check
  tb-quantize --model model.pt --ci               # JSON output for CI
            """,
        )

        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to PyTorch model (.pt/.pth) or HuggingFace model name",
        )

        parser.add_argument(
            "--strategy",
            choices=["auto", "manual"],
            default="auto",
            help="Selection strategy (default: auto)",
        )

        parser.add_argument(
            "--format",
            type=str,
            default="auto",
            help="Quantization format (default: auto)",
        )

        parser.add_argument(
            "--backend",
            choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
            default="auto",
            help="Target backend (default: auto-detect)",
        )

        parser.add_argument(
            "--output",
            "-o",
            type=str,
            help="Save quantized model to path",
        )

        parser.add_argument(
            "--validate",
            action="store_true",
            help="Run quality validation after quantization",
        )

        parser.add_argument(
            "--calibration-samples",
            type=int,
            default=512,
            help="Number of calibration samples for formats that need it",
        )

        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output",
        )

        parser.add_argument(
            "--ci",
            action="store_true",
            help="CI mode: JSON output, structured exit codes",
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the quantize command."""
        ci_mode = getattr(args, "ci", False)
        verbose = getattr(args, "verbose", False)
        model_path = getattr(args, "model", None)
        fmt = getattr(args, "format", "auto")
        validate = getattr(args, "validate", False)
        output_path = getattr(args, "output", None)

        if not ci_mode:
            print(" TorchBridge Quantization Engine")
            print("=" * 50)

        start_time = time.time()

        try:
            # Load model
            model = QuantizeCommand._load_model(model_path, verbose)
            if model is None:
                if ci_mode:
                    print(json.dumps({"error": f"Failed to load model: {model_path}"}))
                else:
                    print(f"Error: Could not load model from {model_path}")
                return 1

            # Set up engine
            from torchbridge.precision.quantization import QuantizationEngine

            backend_override = getattr(args, "backend", "auto")
            if backend_override != "auto":
                # Create a mock backend object for the engine
                class _BackendOverride:
                    BACKEND_NAME = backend_override

                engine = QuantizationEngine(backend=_BackendOverride())
            else:
                engine = QuantizationEngine()

            if not ci_mode:
                print(f" Backend: {engine.backend_name}")
                print(f" Format: {fmt}")

            # Run quantization
            result = engine.quantize(model, format=fmt)

            duration_ms = (time.time() - start_time) * 1000

            if ci_mode:
                output = result.to_dict()
                output["duration_ms"] = duration_ms
                print(json.dumps(output, indent=2))
            else:
                QuantizeCommand._display_result(result, duration_ms, verbose)

            # Save quantized model
            if output_path and result.success and result.model is not None:
                torch.save(result.model.state_dict(), output_path)
                if not ci_mode:
                    print(f"\n Saved quantized model to: {output_path}")

            # Validate quality
            if validate and result.success and result.model is not None:
                QuantizeCommand._validate_quality(model, result.model, ci_mode)

            if result.success:
                return 0
            return 1

        except Exception as e:
            if ci_mode:
                print(json.dumps({"error": str(e)}))
            else:
                from torchbridge.cli import _print_error

                _print_error(e, verbose=verbose)
            return 1

    @staticmethod
    def _load_model(model_path: str, verbose: bool) -> nn.Module | None:
        """Load a PyTorch model from file or create a test model."""
        from pathlib import Path

        path = Path(model_path)
        if path.exists():
            try:
                model = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                if hasattr(model, "eval"):
                    model.eval()
                if verbose:
                    print(f"   Loaded model from {model_path}")
                return model
            except Exception as e:
                logger.error("Failed to load model: %s", e)
                return None

        # Try loading as a state dict into a generic model
        if verbose:
            print(f"   Model path not found as file: {model_path}")
            print("   Attempting HuggingFace model load...")

        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32
            )
            model.eval()
            return model
        except Exception as e:
            logger.error("Failed to load HuggingFace model: %s", e)
            return None

    @staticmethod
    def _display_result(result, duration_ms: float, verbose: bool) -> None:
        """Display quantization results in human-readable format."""
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n Quantization: {status}")
        print("-" * 40)
        print(f"  Format applied: {result.format_applied.value}")
        print(f"  Format requested: {result.format_requested.value}")

        if result.used_fallback:
            print("  Fallback used: Yes")
            chain_str = " > ".join(f.value for f in result.fallback_chain)
            print(f"  Fallback chain: {chain_str}")

        print(f"  Memory before: {result.memory_before_mb:.1f} MB")
        print(f"  Memory after: {result.memory_after_mb:.1f} MB")
        print(f"  Reduction: {result.memory_reduction_pct:.1f}%")
        print(f"  Duration: {duration_ms:.0f} ms")

        if result.warnings:
            for w in result.warnings:
                print(f"  Warning: {w}")

        if result.errors:
            for e in result.errors:
                print(f"  Error: {e}")

    @staticmethod
    def _validate_quality(
        original: nn.Module, quantized: nn.Module, ci_mode: bool
    ) -> None:
        """Run basic quality validation between original and quantized models."""
        try:
            # Create a small test input
            first_param = next(original.parameters())
            in_features = first_param.shape[-1] if first_param.dim() > 1 else first_param.shape[0]
            test_input = torch.randn(1, in_features)

            original.eval()
            quantized.eval()

            with torch.no_grad():
                orig_out = original(test_input)
                quant_out = quantized(test_input)

            # Handle different output types
            if hasattr(orig_out, "logits"):
                orig_out = orig_out.logits
            if hasattr(quant_out, "logits"):
                quant_out = quant_out.logits

            cos_sim = torch.nn.functional.cosine_similarity(
                orig_out.flatten().unsqueeze(0),
                quant_out.flatten().unsqueeze(0),
            ).item()

            max_diff = torch.abs(orig_out - quant_out).max().item()

            if ci_mode:
                print(
                    json.dumps(
                        {
                            "validation": {
                                "cosine_similarity": cos_sim,
                                "max_diff": max_diff,
                                "status": "pass" if cos_sim > 0.99 else "warning",
                            }
                        }
                    )
                )
            else:
                print("\n Quality Validation:")
                print(f"  Cosine similarity: {cos_sim:.6f}")
                print(f"  Max difference: {max_diff:.2e}")
                status = "PASS" if cos_sim > 0.99 else "WARNING"
                print(f"  Status: {status}")

        except Exception as e:
            if ci_mode:
                print(json.dumps({"validation": {"error": str(e)}}))
            else:
                print(f"\n Validation skipped: {e}")


def main():
    """Standalone entry point for tb-quantize."""
    parser = argparse.ArgumentParser(
        prog="tb-quantize",
        description="Apply backend-aware quantization to PyTorch models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model or HuggingFace name",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "manual"],
        default="auto",
        help="Selection strategy",
    )
    parser.add_argument("--format", type=str, default="auto", help="Quantization format")
    parser.add_argument(
        "--backend",
        choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
        default="auto",
        help="Target backend",
    )
    parser.add_argument("--output", "-o", type=str, help="Save quantized model to path")
    parser.add_argument("--validate", action="store_true", help="Validate quality")
    parser.add_argument(
        "--calibration-samples", type=int, default=512, help="Calibration samples"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--ci", action="store_true", help="CI mode (JSON output)")

    args = parser.parse_args()
    try:
        return QuantizeCommand.execute(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        from torchbridge.cli import _print_error

        return _print_error(e, verbose=getattr(args, "verbose", False))


if __name__ == "__main__":
    sys.exit(main())
