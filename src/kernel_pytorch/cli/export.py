"""
Model export commands for KernelPyTorch CLI.

Provides easy-to-use export of PyTorch models to various deployment formats
including ONNX, TorchScript, and SafeTensors.
"""

import argparse
import sys
import time
from pathlib import Path

import torch


class ExportCommand:
    """Model export command implementation."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the export command with argument parser."""
        parser = subparsers.add_parser(
            'export',
            help='Export PyTorch models to deployment formats',
            description='Export optimized PyTorch models to ONNX, TorchScript, or SafeTensors',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Export Formats:
  onnx        - ONNX format for cross-platform inference
  torchscript - TorchScript for PyTorch-native deployment
  safetensors - SafeTensors for secure weight storage
  all         - Export to all supported formats

Examples:
  kpt-export --model model.pt --format onnx --output model.onnx
  kpt-export --model model.pt --format torchscript --method trace
  kpt-export --model model.pt --format safetensors --fp16
  kpt-export --model model.pt --format all --output-dir exports/
            """
        )

        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Path to model file (.pt/.pth) or HuggingFace model name'
        )

        parser.add_argument(
            '--format',
            type=str,
            choices=['onnx', 'torchscript', 'safetensors', 'all'],
            default='onnx',
            help='Export format (default: onnx)'
        )

        parser.add_argument(
            '--output', '-o',
            type=str,
            default=None,
            help='Output file path (auto-generated if not specified)'
        )

        parser.add_argument(
            '--output-dir',
            type=str,
            default='.',
            help='Output directory for exports (default: current directory)'
        )

        parser.add_argument(
            '--input-shape',
            type=str,
            default='1,512',
            help='Input shape as comma-separated values (default: 1,512)'
        )

        parser.add_argument(
            '--dtype',
            type=str,
            choices=['float32', 'float16', 'bfloat16'],
            default='float32',
            help='Model dtype for export (default: float32)'
        )

        parser.add_argument(
            '--fp16',
            action='store_true',
            help='Export with FP16 precision (shortcut for --dtype float16)'
        )

        parser.add_argument(
            '--bf16',
            action='store_true',
            help='Export with BF16 precision (shortcut for --dtype bfloat16)'
        )

        # ONNX-specific options
        parser.add_argument(
            '--opset',
            type=int,
            default=17,
            help='ONNX opset version (default: 17)'
        )

        parser.add_argument(
            '--dynamic-axes',
            action='store_true',
            help='Enable dynamic axes for variable batch/sequence length'
        )

        # TorchScript-specific options
        parser.add_argument(
            '--method',
            type=str,
            choices=['trace', 'script'],
            default='trace',
            help='TorchScript export method (default: trace)'
        )

        # Validation options
        parser.add_argument(
            '--validate',
            action='store_true',
            help='Validate exported model against original'
        )

        parser.add_argument(
            '--validation-tolerance',
            type=float,
            default=1e-4,
            help='Tolerance for validation comparison (default: 1e-4)'
        )

        # Other options
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-essential output'
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the export command."""
        try:
            return ExportCommand._run_export(args)
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            print(f"Export failed: {e}")
            return 1

    @staticmethod
    def _run_export(args) -> int:
        """Run the export process."""
        verbose = args.verbose and not args.quiet

        if verbose:
            print("KernelPyTorch Model Export")
            print("=" * 50)

        # Determine dtype
        if args.fp16:
            dtype = torch.float16
            dtype_str = 'float16'
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = 'bfloat16'
        else:
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            dtype = dtype_map[args.dtype]
            dtype_str = args.dtype

        # Load model
        if verbose:
            print(f"Loading model: {args.model}")

        model = ExportCommand._load_model(args.model, dtype)
        model.eval()

        # Parse input shape
        input_shape = ExportCommand._parse_shape(args.input_shape)
        if verbose:
            print(f"Input shape: {input_shape}")
            print(f"Dtype: {dtype_str}")

        # Create sample input
        sample_input = torch.randn(*input_shape, dtype=dtype)
        if dtype == torch.bfloat16:
            # BF16 models often need to stay on CPU for export
            pass

        # Determine output paths
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = Path(args.model).stem if Path(args.model).exists() else args.model.replace('/', '_')

        results = {}
        formats_to_export = ['onnx', 'torchscript', 'safetensors'] if args.format == 'all' else [args.format]

        for fmt in formats_to_export:
            if verbose:
                print(f"\nExporting to {fmt.upper()}...")

            start_time = time.perf_counter()

            try:
                if fmt == 'onnx':
                    output_path = args.output or output_dir / f"{model_name}.onnx"
                    result = ExportCommand._export_onnx(
                        model, sample_input, output_path,
                        opset=args.opset,
                        dynamic_axes=args.dynamic_axes,
                        validate=args.validate,
                        tolerance=args.validation_tolerance,
                        verbose=verbose
                    )
                elif fmt == 'torchscript':
                    output_path = args.output or output_dir / f"{model_name}.pt"
                    result = ExportCommand._export_torchscript(
                        model, sample_input, output_path,
                        method=args.method,
                        validate=args.validate,
                        tolerance=args.validation_tolerance,
                        verbose=verbose
                    )
                elif fmt == 'safetensors':
                    output_path = args.output or output_dir / f"{model_name}.safetensors"
                    result = ExportCommand._export_safetensors(
                        model, output_path,
                        dtype=dtype,
                        verbose=verbose
                    )

                elapsed = time.perf_counter() - start_time
                result['export_time'] = elapsed
                results[fmt] = result

                if not args.quiet:
                    status = "OK" if result.get('success', False) else "FAILED"
                    print(f"  [{status}] {fmt}: {output_path} ({elapsed:.2f}s)")

            except Exception as e:
                results[fmt] = {'success': False, 'error': str(e)}
                if not args.quiet:
                    print(f"  [FAILED] {fmt}: {e}")

        # Summary
        if verbose:
            print("\n" + "=" * 50)
            print("Export Summary")
            print("=" * 50)
            for fmt, result in results.items():
                status = "SUCCESS" if result.get('success', False) else "FAILED"
                size = result.get('file_size', 0)
                size_str = f"{size / 1024 / 1024:.2f} MB" if size > 0 else "N/A"
                print(f"  {fmt.upper()}: {status} ({size_str})")

        # Return success if all exports succeeded
        all_success = all(r.get('success', False) for r in results.values())
        return 0 if all_success else 1

    @staticmethod
    def _load_model(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
        """Load a model from file or create a simple test model."""
        path = Path(model_path)

        if path.exists():
            # Load from file
            if path.suffix in ['.pt', '.pth']:
                loaded = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(loaded, torch.nn.Module):
                    model = loaded
                elif isinstance(loaded, dict) and 'model' in loaded:
                    model = loaded['model']
                else:
                    raise ValueError(f"Cannot extract model from {path}")
            else:
                raise ValueError(f"Unsupported model format: {path.suffix}")
        else:
            # Try loading from HuggingFace or create a simple model
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
            except ImportError:
                # Create a simple test model
                print(f"Warning: Could not load '{model_path}', using simple test model")
                model = torch.nn.Sequential(
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                )
            except Exception as e:
                print(f"Warning: Could not load '{model_path}': {e}, using simple test model")
                model = torch.nn.Sequential(
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                )

        return model.to(dtype)

    @staticmethod
    def _parse_shape(shape_str: str) -> tuple:
        """Parse shape string into tuple."""
        return tuple(int(x.strip()) for x in shape_str.split(','))

    @staticmethod
    def _export_onnx(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_path: Path,
        opset: int = 17,
        dynamic_axes: bool = False,
        validate: bool = False,
        tolerance: float = 1e-4,
        verbose: bool = False
    ) -> dict:
        """Export model to ONNX format."""
        output_path = Path(output_path)

        # Build dynamic axes config
        axes_config = None
        if dynamic_axes:
            axes_config = {
                'input': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size'}
            }

        # Export using torch.onnx.export (works without onnxscript)
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                str(output_path),
                opset_version=opset,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=axes_config,
                do_constant_folding=True,
            )

        result = {
            'success': True,
            'path': str(output_path),
            'file_size': output_path.stat().st_size if output_path.exists() else 0,
            'opset': opset,
        }

        # Validate if requested
        if validate and output_path.exists():
            try:
                import onnx
                import onnxruntime as ort

                # Check ONNX model validity
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)

                # Run inference comparison
                sess = ort.InferenceSession(str(output_path))
                onnx_output = sess.run(None, {'input': sample_input.numpy()})[0]

                with torch.no_grad():
                    torch_output = model(sample_input).numpy()

                import numpy as np
                max_diff = np.max(np.abs(onnx_output - torch_output))
                result['validation'] = {
                    'passed': max_diff < tolerance,
                    'max_diff': float(max_diff),
                    'tolerance': tolerance
                }

                if verbose:
                    status = "PASSED" if max_diff < tolerance else "FAILED"
                    print(f"    Validation: {status} (max_diff={max_diff:.2e})")

            except ImportError:
                result['validation'] = {'skipped': True, 'reason': 'onnx/onnxruntime not installed'}
            except Exception as e:
                result['validation'] = {'passed': False, 'error': str(e)}

        return result

    @staticmethod
    def _export_torchscript(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_path: Path,
        method: str = 'trace',
        validate: bool = False,
        tolerance: float = 1e-4,
        verbose: bool = False
    ) -> dict:
        """Export model to TorchScript format."""
        output_path = Path(output_path)

        if method == 'trace':
            scripted = torch.jit.trace(model, sample_input)
        else:
            scripted = torch.jit.script(model)

        scripted.save(str(output_path))

        result = {
            'success': True,
            'path': str(output_path),
            'file_size': output_path.stat().st_size if output_path.exists() else 0,
            'method': method,
        }

        # Validate if requested
        if validate and output_path.exists():
            try:
                loaded = torch.jit.load(str(output_path))

                with torch.no_grad():
                    original_output = model(sample_input)
                    loaded_output = loaded(sample_input)

                max_diff = torch.max(torch.abs(original_output - loaded_output)).item()
                result['validation'] = {
                    'passed': max_diff < tolerance,
                    'max_diff': float(max_diff),
                    'tolerance': tolerance
                }

                if verbose:
                    status = "PASSED" if max_diff < tolerance else "FAILED"
                    print(f"    Validation: {status} (max_diff={max_diff:.2e})")

            except Exception as e:
                result['validation'] = {'passed': False, 'error': str(e)}

        return result

    @staticmethod
    def _export_safetensors(
        model: torch.nn.Module,
        output_path: Path,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False
    ) -> dict:
        """Export model to SafeTensors format."""
        output_path = Path(output_path)

        try:
            from safetensors.torch import save_file

            state_dict = model.state_dict()

            # Convert to target dtype if needed
            converted_dict = {}
            for key, tensor in state_dict.items():
                if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    converted_dict[key] = tensor.to(dtype)
                else:
                    converted_dict[key] = tensor

            save_file(converted_dict, str(output_path))

            result = {
                'success': True,
                'path': str(output_path),
                'file_size': output_path.stat().st_size if output_path.exists() else 0,
                'num_tensors': len(state_dict),
            }

        except ImportError:
            # Fallback: save as regular state dict with .safetensors extension note
            torch.save(model.state_dict(), str(output_path))
            result = {
                'success': True,
                'path': str(output_path),
                'file_size': output_path.stat().st_size if output_path.exists() else 0,
                'note': 'safetensors not installed, saved as torch state dict',
            }

        return result


def main(args=None):
    """CLI entry point for export command."""
    import argparse
    parser = argparse.ArgumentParser(
        prog='kpt-export',
        description='Export PyTorch models to deployment formats'
    )

    # Create a mock subparsers to reuse register
    subparsers = parser.add_subparsers()
    ExportCommand.register(subparsers)

    # Re-parse with full arguments
    parser = argparse.ArgumentParser(
        prog='kpt-export',
        description='Export PyTorch models to deployment formats'
    )
    ExportCommand.register(type('SubParsers', (), {'add_parser': lambda *a, **k: parser})())

    parsed_args = parser.parse_args(args)
    return ExportCommand.execute(parsed_args)


if __name__ == '__main__':
    sys.exit(main())
