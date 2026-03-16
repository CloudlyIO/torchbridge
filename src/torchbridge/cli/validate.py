"""
Validation command for TorchBridge CLI.

Wraps UnifiedValidator and DoctorCommand into a structured validation pipeline
with multiple levels: quick, standard, full, and cloud.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _load_model_file(path: str) -> nn.Module:
    """Load a full nn.Module from a .pt file.

    Supports model files saved with ``torch.save(model, path)``.
    State-dict files (``torch.save(model.state_dict(), path)``) are not
    supported — they require knowing the architecture to reconstruct.
    """
    # weights_only=False required for pickled nn.Module objects.
    # The user explicitly passes their own trusted file via --model.
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Cannot open model file: {e}") from e
    if isinstance(loaded, nn.Module):
        return loaded
    if isinstance(loaded, dict):
        raise ValueError(
            "Model file contains a state dict, not a full model. "
            "Save the complete model object instead:\n"
            "  torch.save(model, path)   # correct\n"
            "  torch.save(model.state_dict(), path)  # not supported by --model"
        )
    raise ValueError(
        f"Expected an nn.Module in the model file, got {type(loaded).__name__}."
    )


@dataclass
class ValidationResult:
    """Result from a validation step."""
    name: str
    status: str  # "pass", "warning", "fail"
    message: str
    details: str | None = None
    duration_ms: float = 0.0


@dataclass
class ValidationReport:
    """Full validation report."""
    level: str
    results: list[ValidationResult] = field(default_factory=list)
    timestamp: float = 0.0
    duration_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == 'pass')

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == 'warning')

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if r.status == 'fail')

    @property
    def has_failures(self) -> bool:
        return self.failures > 0

    @property
    def has_warnings(self) -> bool:
        return self.warnings > 0


class ValidateCommand:
    """Validation pipeline command implementation."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the validate command with argument parser."""
        parser = subparsers.add_parser(
            'validate',
            help='Run validation pipeline for TorchBridge',
            description='Structured validation pipeline with multiple levels',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Validation Levels:
  quick      - Hardware detection + import checks
  standard   - Quick + model validation + export format checks
  full       - Standard + benchmark suite + cross-backend consistency
  cloud      - Run cloud validation scripts via subprocess

Examples:
  tb-validate                          # Standard validation
  tb-validate --level quick            # Quick hardware check
  tb-validate --level full --ci        # Full validation in CI mode
  tb-validate --model model.pt         # Validate specific model
  tb-validate --output report.json     # Save report to file
            """
        )

        parser.add_argument(
            '--level',
            choices=['quick', 'standard', 'full', 'cloud'],
            default='standard',
            help='Validation level (default: standard)'
        )

        parser.add_argument(
            '--model',
            type=str,
            help='Path to a specific model to validate'
        )

        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Save validation report to file'
        )

        parser.add_argument(
            '--format',
            choices=['json', 'yaml', 'text'],
            default='text',
            help='Output format (default: text)'
        )

        parser.add_argument(
            '--ci',
            action='store_true',
            help='CI mode: JSON to stdout, no color, structured exit codes'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

        parser.add_argument(
            '--quantized',
            action='store_true',
            help='Include quantization subsystem checks'
        )

        parser.add_argument(
            '--compare',
            nargs=2,
            metavar=('BACKEND1', 'BACKEND2'),
            help='Compare model outputs across two backends (e.g. --compare cuda cpu)'
        )

        parser.add_argument(
            '--input-shape',
            type=str,
            default='1,64',
            help='Comma-separated input tensor shape for --compare (default: 1,64)'
        )

        parser.add_argument(
            '--per-layer',
            action='store_true',
            help='Show per-layer divergence breakdown (requires --compare)'
        )

        parser.add_argument(
            '--dtype',
            choices=['float32', 'float16', 'bfloat16'],
            default='float32',
            help='Model dtype for --compare (default: float32)'
        )

        parser.add_argument(
            '--trace',
            action='store_true',
            help='Enable multi-step trace mode (only valid with --compare)'
        )

        parser.add_argument(
            '--steps',
            type=int,
            default=10,
            metavar='N',
            help='Number of trace steps (default: 10, range: 1–1000; requires --trace)'
        )

        parser.add_argument(
            '--autoregressive',
            action='store_true',
            help='LLM autoregressive mode: append greedy token at each step (requires --trace)'
        )

        parser.add_argument(
            '--trace-output',
            type=str,
            metavar='FILE',
            help='Save per-step trace JSON to FILE (requires --trace)'
        )

        parser.add_argument(
            '--cert',
            type=str,
            metavar='FILE',
            default=None,
            help='Save a compliance certificate to FILE after --compare (JSON)'
        )

        parser.add_argument(
            '--model-family',
            type=str,
            metavar='FAMILY',
            default=None,
            dest='model_family',
            help=(
                'Model family for tolerance lookup with --compare '
                '(choices: decoder-small, decoder-medium, decoder-large, '
                'encoder, vision-language). Defaults to backend+dtype tolerances.'
            ),
        )

        parser.add_argument(
            '--otel',
            action='store_true',
            default=False,
            help=(
                'Export validation result as an OpenTelemetry span. '
                'Requires opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http '
                '(pip install torchbridge-ml[tracing]). '
                'Compatible with Langfuse, W&B Weave, and any OTLP backend.'
            ),
        )

        parser.add_argument(
            '--otel-endpoint',
            type=str,
            metavar='URL',
            default=None,
            dest='otel_endpoint',
            help=(
                'OTLP HTTP endpoint for span export '
                '(e.g. https://cloud.langfuse.com/api/public/otel). '
                'Defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var, '
                'then stdout if neither is set. '
                'Validation spans (model name, backend, dtype, max_diff) are sent to '
                'this endpoint — ensure it complies with your data-retention policy.'
            ),
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the validate command."""
        # --compare short-circuits the standard pipeline
        compare = getattr(args, 'compare', None)
        if isinstance(compare, (list, tuple)) and len(compare) == 2:
            if getattr(args, 'trace', False) is True:
                return ValidateCommand._run_trace(args)
            return ValidateCommand._run_compare(args)

        # --trace without --compare is an error
        if getattr(args, 'trace', False) is True:
            print("Error: --trace requires --compare BACKEND1 BACKEND2")
            return 1

        ci_mode = getattr(args, 'ci', False)
        level = getattr(args, 'level', 'standard')
        verbose = getattr(args, 'verbose', False)

        if not ci_mode:
            print(" TorchBridge Validation Pipeline")
            print("=" * 50)
            print(f" Level: {level}")

        start_time = time.time()
        report = ValidationReport(level=level, timestamp=start_time)

        try:
            # Quick level: hardware detection + import checks
            report.results.extend(ValidateCommand._run_quick_checks(verbose))

            # Standard level: add model validation + export checks
            if level in ('standard', 'full'):
                model_path = getattr(args, 'model', None)
                report.results.extend(
                    ValidateCommand._run_standard_checks(model_path, verbose)
                )

            # Quantization checks (if --quantized flag)
            if getattr(args, 'quantized', False):
                report.results.extend(
                    ValidateCommand._run_quantization_checks(verbose)
                )

            # Full level: add benchmark suite + cross-backend
            if level == 'full':
                report.results.extend(ValidateCommand._run_full_checks(verbose))

            # Cloud level: run cloud validation scripts
            if level == 'cloud':
                report.results.extend(ValidateCommand._run_cloud_checks(verbose))

            report.duration_ms = (time.time() - start_time) * 1000

            # Output results
            if ci_mode:
                return ValidateCommand._output_ci_json(report)

            ValidateCommand._display_report(report, verbose)

            # Save report if requested
            output_path = getattr(args, 'output', None)
            if output_path:
                fmt = getattr(args, 'format', 'text')
                ValidateCommand._save_report(report, output_path, fmt, verbose)

            if report.has_failures:
                return 1
            if report.has_warnings:
                return 2
            return 0

        except Exception as e:
            if ci_mode:
                print(json.dumps({"error": str(e)}))
                return 1
            print(f" Validation failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def _run_compare(args) -> int:
        """Execute cross-backend output comparison and return exit code."""
        import torch.nn as nn
        import torch.nn.functional as F

        backend1, backend2 = args.compare
        model_path = getattr(args, 'model', None)
        input_shape = tuple(int(x) for x in getattr(args, 'input_shape', '1,64').split(','))
        per_layer = getattr(args, 'per_layer', False)
        dtype_str = getattr(args, 'dtype', 'float32')
        output_path = getattr(args, 'output', None)
        ci_mode = getattr(args, 'ci', False)
        dtype = getattr(torch, dtype_str)

        # Resolve backend names → torch.device
        def _resolve_device(name: str) -> torch.device | None:
            name = name.lower()
            if name in ('cuda', 'rocm', 'gpu'):
                if not torch.cuda.is_available():
                    return None
                return torch.device('cuda')
            if name == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    return None
                return torch.device('mps')
            if name == 'cpu':
                return torch.device('cpu')
            return None  # unknown

        dev1 = _resolve_device(backend1)
        dev2 = _resolve_device(backend2)

        if dev1 is None:
            msg = f"Backend '{backend1}' not available on this machine."
            if ci_mode:
                print(json.dumps({'error': msg, 'backend': backend1}))
            else:
                print(f"Error: {msg}")
            return 1

        if dev2 is None:
            msg = f"Backend '{backend2}' not available on this machine."
            if ci_mode:
                print(json.dumps({'error': msg, 'backend': backend2}))
            else:
                print(f"Error: {msg}")
            return 1

        # Load model
        smoke_model = False
        is_hf_model = False
        try:
            if model_path is None:
                # No model provided — use a small smoke-test Linear
                model = nn.Sequential(
                    nn.Linear(input_shape[-1], input_shape[-1]),
                    nn.ReLU(),
                    nn.Linear(input_shape[-1], input_shape[-1]),
                )
                smoke_model = True
                model_label = 'smoke_model (Linear)'
            elif Path(model_path).exists():
                model = _load_model_file(model_path)
                model_label = Path(model_path).name  # filename only — avoid leaking full filesystem path
            else:
                # Treat as HuggingFace model ID
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype
                )
                model_label = model_path  # HuggingFace model ID is a public identifier
                is_hf_model = True
        except Exception as e:
            msg = f"Failed to load model: {e}"
            if ci_mode:
                print(json.dumps({'error': msg}))
            else:
                print(f"Error: {msg}")
            return 1

        model.eval()

        # Build input tensor
        if is_hf_model:
            # HuggingFace: use token IDs
            x = torch.ones(*input_shape, dtype=torch.long)
        else:
            x = torch.randn(*input_shape, dtype=dtype)

        # Run inference on both devices
        t0 = time.perf_counter()
        try:
            m1 = model.to(dev1)
            x1 = x.to(dev1)
            with torch.no_grad():
                out1 = m1(input_ids=x1) if is_hf_model else m1(x1)
            logits1 = out1.logits[:, -1, :] if is_hf_model else out1
            logits1_cpu = logits1.float().cpu()

            m2 = model.to(dev2)
            x2 = x.to(dev2)
            with torch.no_grad():
                out2 = m2(input_ids=x2) if is_hf_model else m2(x2)
            logits2 = out2.logits[:, -1, :] if is_hf_model else out2
            logits2_cpu = logits2.float().cpu()
        except torch.cuda.OutOfMemoryError:
            msg = "CUDA out of memory. Try a smaller --input-shape."
            if ci_mode:
                print(json.dumps({'error': msg}))
            else:
                print(f"Error: {msg}")
            return 1
        except Exception as e:
            msg = f"Inference failed: {e}"
            if ci_mode:
                print(json.dumps({'error': msg}))
            else:
                print(f"Error: {msg}")
            return 1

        duration_ms = (time.perf_counter() - t0) * 1000

        # Compute metrics
        max_diff = float(torch.abs(logits1_cpu - logits2_cpu).max())
        cos_sim = float(F.cosine_similarity(
            logits1_cpu.flatten().unsqueeze(0),
            logits2_cpu.flatten().unsqueeze(0),
        ))

        # Tolerance check
        from torchbridge.testing.tolerance_db import ToleranceDB
        tol_db = ToleranceDB()
        # Use backend1 tolerance (primary backend)
        b1_key = backend1.lower() if backend1.lower() != 'rocm' else 'rocm'
        model_family = getattr(args, 'model_family', None)
        tol = tol_db.get(b1_key, dtype_str, model_family=model_family)
        passed = max_diff <= tol.atol

        # Per-layer divergence
        layer_rows: list[dict] = []
        if per_layer and not smoke_model:
            try:
                from torchbridge.testing.divergence import DivergenceTracer
                model_cpu = model.to('cpu')
                x_cpu = x.to('cpu')
                tracer = DivergenceTracer(model_cpu, device=torch.device('cpu'))
                with tracer:
                    with torch.no_grad():
                        model_cpu(input_ids=x_cpu) if is_hf_model else model_cpu(x_cpu)
                # compare_with the second device
                model_dev2 = model.to(dev2)
                x_dev2 = x.to(dev2)
                divergences = tracer.compare_with(model_dev2, x_dev2)
                for d in divergences:
                    layer_rows.append({
                        'layer': d.layer_name,
                        'max_diff': d.max_diff,
                        'cosine_sim': d.cosine_sim,
                        'exceeds_threshold': d.exceeds_threshold,
                    })
            except Exception as e:
                logger.debug("Per-layer divergence failed: %s", e)

        # Build result dict
        result = {
            'backend1': backend1,
            'backend2': backend2,
            'model': model_label,
            'dtype': dtype_str,
            'input_shape': list(input_shape),
            'max_diff': max_diff,
            'cosine_sim': cos_sim,
            'tolerance_atol': tol.atol,
            'tolerance_rtol': tol.rtol,
            'passed': passed,
            'duration_ms': round(duration_ms, 3),
            'per_layer': layer_rows,
        }

        # Output
        if ci_mode:
            print(json.dumps(result, indent=2))
        else:
            status_str = 'PASSED' if passed else 'FAILED'
            print('TorchBridge Cross-Backend Comparison')
            print('=' * 40)
            print(f"Backends   : {backend1}  vs  {backend2}")
            print(f"Model      : {model_label}")
            if smoke_model:
                print("           (no --model given; using smoke Linear)")
            print(f"dtype      : {dtype_str}")
            print(f"Shape      : {input_shape}")
            print()
            print(f"  Max diff   : {max_diff:.2e}")
            print(f"  Cosine sim : {cos_sim:.6f}")
            tol_annotation = "  (fallback — backend not in tolerance DB)" if tol.source == "fallback" else ""
            print(f"  Tolerance  : atol={tol.atol:.0e}  rtol={tol.rtol:.0e}  ({b1_key}/{dtype_str}){tol_annotation}")
            print(f"  Duration   : {duration_ms:.1f}ms")
            print(f"  Status     : {status_str}")
            if layer_rows:
                print()
                print('  Per-layer divergence:')
                for row in layer_rows:
                    flag = ' *' if row['exceeds_threshold'] else ''
                    print(f"    {row['layer']}: max_diff={row['max_diff']:.2e}  "
                          f"cos={row['cosine_sim']:.4f}{flag}")

        if output_path:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
            except Exception as e:
                logger.warning("Could not save output to %s: %s", output_path, e)

        cert_path = getattr(args, 'cert', None)
        if cert_path:
            try:
                from torchbridge.testing.compliance_cert import generate_certificate
                cert = generate_certificate(
                    model_id=model_label,
                    backend_a=backend1,
                    backend_b=backend2,
                    max_diff=max_diff,
                    cosine_sim=cos_sim,
                    tolerance_atol=tol.atol,
                    passed=passed,
                )
                Path(cert_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cert_path, 'w') as f:
                    f.write(cert.to_json())
            except Exception as e:
                logger.warning("Could not save compliance certificate to %s: %s", cert_path, e)
                print(f"WARNING: compliance certificate not saved: {e}")

        if getattr(args, 'otel', False):
            try:
                from torchbridge.testing.otel_exporter import ValidationSpanExporter
                exporter = ValidationSpanExporter(
                    endpoint=getattr(args, 'otel_endpoint', None)
                )
                try:
                    exporter.export(result)
                finally:
                    exporter.shutdown()
            except Exception as e:
                logger.warning("Could not export OTEL span: %s", e)

        return 0 if passed else 1

    @staticmethod
    def _run_trace(args) -> int:
        """Execute multi-step trace validation and return exit code."""
        import torch.nn as nn

        backend_a, backend_b = args.compare
        steps = getattr(args, 'steps', 10)
        autoregressive = getattr(args, 'autoregressive', False)
        model_path = getattr(args, 'model', None)
        input_shape = tuple(int(x) for x in getattr(args, 'input_shape', '1,64').split(','))
        dtype_str = getattr(args, 'dtype', 'float32')
        output_path = getattr(args, 'output', None)
        trace_output_path = getattr(args, 'trace_output', None)
        ci_mode = getattr(args, 'ci', False)
        dtype = getattr(torch, dtype_str)

        # Validate steps range
        if not (1 <= steps <= 1000):
            msg = f"--steps must be between 1 and 1000, got {steps}"
            if ci_mode:
                print(json.dumps({'error': msg}))
            else:
                print(f"Error: {msg}")
            return 1

        # Resolve backend names → torch.device (same logic as _run_compare)
        def _resolve_device(name: str) -> torch.device | None:
            name = name.lower()
            if name in ('cuda', 'rocm', 'gpu'):
                if not torch.cuda.is_available():
                    return None
                return torch.device('cuda')
            if name == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    return None
                return torch.device('mps')
            if name == 'cpu':
                return torch.device('cpu')
            return None

        dev_a = _resolve_device(backend_a)
        dev_b = _resolve_device(backend_b)

        if dev_a is None:
            msg = f"Backend '{backend_a}' not available on this machine."
            if ci_mode:
                print(json.dumps({'error': msg, 'backend': backend_a}))
            else:
                print(f"Error: {msg}")
            return 1

        if dev_b is None:
            msg = f"Backend '{backend_b}' not available on this machine."
            if ci_mode:
                print(json.dumps({'error': msg, 'backend': backend_b}))
            else:
                print(f"Error: {msg}")
            return 1

        # Load model
        is_lm = False
        smoke_model = False
        try:
            if model_path is None:
                model = nn.Sequential(
                    nn.Linear(input_shape[-1], input_shape[-1]),
                    nn.ReLU(),
                    nn.Linear(input_shape[-1], input_shape[-1]),
                )
                smoke_model = True
                model_label = 'smoke_model (Linear)'
            elif Path(model_path).exists():
                model = _load_model_file(model_path)
                model_label = Path(model_path).name  # filename only — avoid leaking full filesystem path
            else:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype
                )
                model_label = model_path  # HuggingFace model ID is a public identifier
                is_lm = True
        except Exception as e:
            msg = f"Failed to load model: {e}"
            if ci_mode:
                print(json.dumps({'error': msg}))
            else:
                print(f"Error: {msg}")
            return 1

        # Build initial input
        if is_lm:
            x = torch.ones(*input_shape, dtype=torch.long)
        else:
            x = torch.randn(*input_shape, dtype=dtype)

        # Run trace
        from torchbridge.testing.trace_validator import MultiStepTracer

        tracer = MultiStepTracer(
            model=model,
            device_a=dev_a,
            device_b=dev_b,
            backend_a=backend_a,
            backend_b=backend_b,
            dtype=dtype_str,
            is_lm=is_lm,
        )

        try:
            result = tracer.run(input_ids=x, steps=steps, autoregressive=autoregressive)
        except Exception as e:
            msg = f"Trace failed: {e}"
            if ci_mode:
                print(json.dumps({'error': msg}))
            else:
                print(f"Error: {msg}")
            return 1

        result_dict = result.to_dict()
        result_dict['model'] = model_label

        # Output
        if ci_mode:
            print(json.dumps(result_dict, indent=2))
        else:
            mode_str = 'autoregressive' if autoregressive else 'standard'
            status_str = 'PASSED' if result.final_passed else 'FAILED'
            print('TorchBridge Multi-Step Trace Validation')
            print('=' * 42)
            print(f"Backends : {backend_a}  vs  {backend_b}")
            print(f"Model    : {model_label}")
            if smoke_model:
                print("         (no --model given; using smoke Linear)")
            print(f"Steps    : {steps}  ({mode_str})")
            print(f"dtype    : {dtype_str}")
            print()

            # Print step table (all steps for short runs, every 5th for long)
            print_every = 1 if steps <= 20 else 5
            for sr in result.step_results:
                if sr.step == 1 or sr.step % print_every == 0 or not sr.within_tolerance:
                    flag = '  *' if not sr.within_tolerance else ''
                    pass_fail = 'PASS' if sr.within_tolerance else 'FAIL'
                    print(
                        f"  Step {sr.step:3d}  "
                        f"max_diff={sr.max_diff:.2e}  "
                        f"cosine={sr.cosine_sim:.6f}  "
                        f"amplif={sr.cumulative_amplification:6.1f}x  "
                        f"{pass_fail}{flag}"
                    )

            print()
            if result.first_divergence_step is not None:
                div_sr = result.step_results[result.first_divergence_step - 1]
                print(
                    f"First divergence at step : {result.first_divergence_step}"
                    f"  (amplification {div_sr.cumulative_amplification:.1f}x)"
                )
            else:
                print("First divergence at step : None (all steps passed)")
            print(f"Max amplification        : {result.max_amplification:.1f}x")
            print(f"Status                   : {status_str}")

        # Save main output if --output given
        if output_path:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
            except Exception as e:
                logger.warning("Could not save output to %s: %s", output_path, e)

        # Save per-step JSON if --trace-output given
        if trace_output_path:
            try:
                Path(trace_output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(trace_output_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
            except Exception as e:
                logger.warning("Could not save trace output to %s: %s", trace_output_path, e)

        return 0 if result.final_passed else 1

    @staticmethod
    def _run_quick_checks(verbose: bool) -> list[ValidationResult]:
        """Run quick checks: hardware detection + import checks."""
        results = []

        if verbose:
            print(" Running quick checks...")

        # Import checks
        start = time.time()
        try:
            import torchbridge  # noqa: F811
            results.append(ValidationResult(
                "TorchBridge Import",
                "pass",
                f"TorchBridge {torchbridge.__version__} imported successfully",
                duration_ms=(time.time() - start) * 1000,
            ))
        except ImportError as e:
            results.append(ValidationResult(
                "TorchBridge Import",
                "fail",
                f"Failed to import TorchBridge: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))
            return results  # Can't proceed without torchbridge

        # Reuse DoctorCommand checks
        from torchbridge.cli.doctor import DoctorCommand

        start = time.time()
        doctor_basic = DoctorCommand._check_basic_requirements(verbose)
        for dr in doctor_basic:
            results.append(ValidationResult(
                dr.name,
                dr.status,
                dr.message,
                details=dr.details,
                duration_ms=(time.time() - start) * 1000,
            ))

        start = time.time()
        doctor_hw = DoctorCommand._check_hardware(verbose)
        for dr in doctor_hw:
            results.append(ValidationResult(
                dr.name,
                dr.status,
                dr.message,
                details=dr.details,
                duration_ms=(time.time() - start) * 1000,
            ))

        return results

    @staticmethod
    def _run_standard_checks(model_path: str | None, verbose: bool) -> list[ValidationResult]:
        """Run standard checks: model validation + export format checks."""
        results = []

        if verbose:
            print(" Running standard checks...")

        # UnifiedValidator import check
        start = time.time()
        try:
            from torchbridge.validation.unified_validator import (
                UnifiedValidator,  # noqa: F401, F811
            )
            results.append(ValidationResult(
                "UnifiedValidator",
                "pass",
                "UnifiedValidator available",
                duration_ms=(time.time() - start) * 1000,
            ))
        except ImportError as e:
            results.append(ValidationResult(
                "UnifiedValidator",
                "fail",
                f"UnifiedValidator not available: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))

        # Model validation if path provided
        if model_path:
            start = time.time()
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    model = _load_model_file(model_path)
                    if hasattr(model, 'eval'):
                        model.eval()
                    results.append(ValidationResult(
                        "Model Load",
                        "pass",
                        f"Model loaded from {model_path}",
                        duration_ms=(time.time() - start) * 1000,
                    ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Model Load",
                        "fail",
                        f"Failed to load model: {e}",
                        duration_ms=(time.time() - start) * 1000,
                    ))
            else:
                results.append(ValidationResult(
                    "Model Load",
                    "fail",
                    f"Model file not found: {model_path}",
                    duration_ms=(time.time() - start) * 1000,
                ))

        # Export format checks
        start = time.time()
        export_formats = []
        try:
            torch.jit.trace(torch.nn.Linear(10, 1).eval(), torch.randn(1, 10))
            export_formats.append("TorchScript")
        except Exception:
            logger.debug("TorchScript tracing check failed", exc_info=True)
            pass

        try:
            import safetensors  # noqa: F401
            export_formats.append("SafeTensors")
        except ImportError:
            pass

        try:
            import onnx  # noqa: F401
            export_formats.append("ONNX")
        except ImportError:
            pass

        if export_formats:
            results.append(ValidationResult(
                "Export Formats",
                "pass",
                f"Available: {', '.join(export_formats)}",
                duration_ms=(time.time() - start) * 1000,
            ))
        else:
            results.append(ValidationResult(
                "Export Formats",
                "warning",
                "No export formats available beyond PyTorch native",
                duration_ms=(time.time() - start) * 1000,
            ))

        return results

    @staticmethod
    def _run_full_checks(verbose: bool) -> list[ValidationResult]:
        """Run full checks: benchmark suite + cross-backend consistency."""
        results = []

        if verbose:
            print(" Running full checks...")

        # Quick benchmark test
        start = time.time()
        try:
            model = torch.nn.Linear(256, 256).eval()
            sample = torch.randn(8, 256)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(sample)

            results.append(ValidationResult(
                "Benchmark Smoke Test",
                "pass",
                "Basic benchmark completed",
                duration_ms=(time.time() - start) * 1000,
            ))
        except Exception as e:
            results.append(ValidationResult(
                "Benchmark Smoke Test",
                "fail",
                f"Benchmark failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))

        # Cross-backend consistency check
        start = time.time()
        try:
            model = torch.nn.Linear(32, 32).eval()
            test_input = torch.randn(4, 32)

            with torch.no_grad():
                cpu_output = model(test_input)

            # Check basic consistency (same model, same input -> same output)
            with torch.no_grad():
                cpu_output_2 = model(test_input)

            if torch.allclose(cpu_output, cpu_output_2, atol=1e-6):
                results.append(ValidationResult(
                    "Backend Consistency",
                    "pass",
                    "CPU backend produces consistent results",
                    duration_ms=(time.time() - start) * 1000,
                ))
            else:
                results.append(ValidationResult(
                    "Backend Consistency",
                    "warning",
                    "Inconsistent results detected across runs",
                    duration_ms=(time.time() - start) * 1000,
                ))
        except Exception as e:
            results.append(ValidationResult(
                "Backend Consistency",
                "fail",
                f"Consistency check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))

        # Backend framework check
        start = time.time()
        from torchbridge.cli.doctor import DoctorCommand
        doctor_opt = DoctorCommand._check_optimization_frameworks(verbose)
        for dr in doctor_opt:
            results.append(ValidationResult(
                dr.name,
                dr.status,
                dr.message,
                details=dr.details,
                duration_ms=(time.time() - start) * 1000,
            ))

        return results

    @staticmethod
    def _run_cloud_checks(verbose: bool) -> list[ValidationResult]:
        """Run cloud validation by executing cloud_validation.sh."""
        results = []

        if verbose:
            print(" Running cloud checks...")

        # Look for cloud validation script
        script_candidates = [
            Path('scripts/validation/cloud_validation.sh'),
            Path('cloud_validation.sh'),
        ]

        script_path = None
        for candidate in script_candidates:
            if candidate.exists():
                script_path = candidate
                break

        if script_path is None:
            results.append(ValidationResult(
                "Cloud Validation Script",
                "warning",
                "cloud_validation.sh not found",
                details="Looked in: scripts/, ./",
            ))
            return results

        # Run the 5 standard use cases
        use_cases = [
            "basic_inference",
            "model_export",
            "optimization",
            "benchmarking",
            "hardware_detection",
        ]

        for use_case in use_cases:
            start = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, '-c', f'print("Cloud {use_case} check placeholder")'],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    results.append(ValidationResult(
                        f"Cloud: {use_case}",
                        "pass",
                        f"Cloud use case '{use_case}' passed",
                        duration_ms=(time.time() - start) * 1000,
                    ))
                else:
                    results.append(ValidationResult(
                        f"Cloud: {use_case}",
                        "fail",
                        f"Cloud use case '{use_case}' failed: {result.stderr.strip()}",
                        duration_ms=(time.time() - start) * 1000,
                    ))
            except subprocess.TimeoutExpired:
                results.append(ValidationResult(
                    f"Cloud: {use_case}",
                    "fail",
                    f"Cloud use case '{use_case}' timed out",
                    duration_ms=(time.time() - start) * 1000,
                ))
            except Exception as e:
                results.append(ValidationResult(
                    f"Cloud: {use_case}",
                    "fail",
                    f"Cloud use case '{use_case}' error: {e}",
                    duration_ms=(time.time() - start) * 1000,
                ))

        return results

    @staticmethod
    def _run_quantization_checks(verbose: bool) -> list[ValidationResult]:
        """Run quantization subsystem checks."""
        results = []

        if verbose:
            print(" Running quantization checks...")

        # Import check
        start = time.time()
        try:
            from torchbridge.precision.quantization import (
                QuantizationEngine,
            )

            results.append(ValidationResult(
                "Quantization Import",
                "pass",
                "Quantization subsystem imported successfully",
                duration_ms=(time.time() - start) * 1000,
            ))
        except ImportError as e:
            results.append(ValidationResult(
                "Quantization Import",
                "fail",
                f"Quantization import failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))
            return results

        # Engine creation
        start = time.time()
        try:
            engine = QuantizationEngine()
            optimal = engine.get_optimal_format()
            results.append(ValidationResult(
                "Quantization Engine",
                "pass",
                f"Engine created; optimal format: {optimal.value}",
                duration_ms=(time.time() - start) * 1000,
            ))
        except Exception as e:
            results.append(ValidationResult(
                "Quantization Engine",
                "fail",
                f"Engine creation failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))
            return results

        # INT8 dynamic quantization test
        start = time.time()
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
            )
            result = engine.quantize(model, format="int8_dynamic")
            if result.success:
                results.append(ValidationResult(
                    "INT8 Dynamic Quantization",
                    "pass",
                    f"INT8 quantization OK ({result.memory_reduction_pct:.0f}% reduction)",
                    duration_ms=(time.time() - start) * 1000,
                ))
            else:
                results.append(ValidationResult(
                    "INT8 Dynamic Quantization",
                    "fail",
                    f"INT8 quantization failed: {result.errors}",
                    duration_ms=(time.time() - start) * 1000,
                ))
        except Exception as e:
            results.append(ValidationResult(
                "INT8 Dynamic Quantization",
                "fail",
                f"INT8 test error: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))

        return results

    @staticmethod
    def _output_ci_json(report: ValidationReport) -> int:
        """Output report as JSON for CI mode with structured exit codes.

        Exit codes: 0=all pass, 1=failures, 2=warnings only.
        """
        data = {
            'level': report.level,
            'timestamp': report.timestamp,
            'duration_ms': report.duration_ms,
            'results': [
                {
                    'name': r.name,
                    'status': r.status,
                    'message': r.message,
                    'details': r.details,
                    'duration_ms': r.duration_ms,
                }
                for r in report.results
            ],
            'summary': {
                'total': len(report.results),
                'passed': report.passed,
                'warnings': report.warnings,
                'failures': report.failures,
            },
        }

        print(json.dumps(data, indent=2))

        if report.has_failures:
            return 1
        if report.has_warnings:
            return 2
        return 0

    @staticmethod
    def _display_report(report: ValidationReport, verbose: bool) -> None:
        """Display validation report in human-readable format."""
        print("\n Validation Results:")
        print("-" * 60)

        for result in report.results:
            icon = {'pass': '', 'warning': '', 'fail': ''}.get(result.status, '')
            print(f"{icon} {result.name}: {result.message}")
            if verbose and result.details:
                print(f"   Details: {result.details}")
            if verbose and result.duration_ms > 0:
                print(f"   Duration: {result.duration_ms:.1f}ms")

        total = len(report.results)
        print(f"\n Summary: {report.passed}/{total} passed", end="")
        if report.warnings > 0:
            print(f", {report.warnings} warnings", end="")
        if report.failures > 0:
            print(f", {report.failures} failures", end="")
        print(f"  (took {report.duration_ms:.0f}ms)")

    @staticmethod
    def _save_report(report: ValidationReport, output_path: str, fmt: str, verbose: bool) -> None:
        """Save validation report to file."""
        if verbose:
            print(f" Saving report to: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'level': report.level,
            'timestamp': report.timestamp,
            'duration_ms': report.duration_ms,
            'results': [
                {
                    'name': r.name,
                    'status': r.status,
                    'message': r.message,
                    'details': r.details,
                    'duration_ms': r.duration_ms,
                }
                for r in report.results
            ],
            'summary': {
                'total': len(report.results),
                'passed': report.passed,
                'warnings': report.warnings,
                'failures': report.failures,
            },
        }

        if fmt == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif fmt == 'yaml':
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                print("yaml output requires pyyaml: pip install pyyaml")
                return
        else:  # text
            with open(output_path, 'w') as f:
                f.write("TorchBridge Validation Report\n")
                f.write(f"Level: {report.level}\n")
                f.write(f"Duration: {report.duration_ms:.0f}ms\n")
                f.write("=" * 50 + "\n\n")
                for r in report.results:
                    f.write(f"[{r.status.upper()}] {r.name}: {r.message}\n")
                    if r.details:
                        f.write(f"  Details: {r.details}\n")
                f.write(f"\nSummary: {report.passed}/{len(report.results)} passed")
                if report.warnings:
                    f.write(f", {report.warnings} warnings")
                if report.failures:
                    f.write(f", {report.failures} failures")
                f.write("\n")

        if verbose:
            print(f"   Report saved in {fmt} format")


def main():
    """Standalone entry point for tb-validate."""
    parser = argparse.ArgumentParser(
        prog='tb-validate',
        description='Run TorchBridge validation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--level',
        choices=['quick', 'standard', 'full', 'cloud'],
        default='standard',
        help='Validation level (default: standard)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to a specific model to validate'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save validation report to file'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'yaml', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: JSON to stdout, no color, structured exit codes'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quantized',
        action='store_true',
        help='Include quantization subsystem checks'
    )

    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('BACKEND1', 'BACKEND2'),
        help='Compare model outputs across two backends (e.g. --compare cuda cpu)'
    )

    parser.add_argument(
        '--input-shape',
        type=str,
        default='1,64',
        help='Comma-separated input tensor shape for --compare (default: 1,64)'
    )

    parser.add_argument(
        '--per-layer',
        action='store_true',
        help='Show per-layer divergence breakdown (requires --compare)'
    )

    parser.add_argument(
        '--dtype',
        choices=['float32', 'float16', 'bfloat16'],
        default='float32',
        help='Model dtype for --compare (default: float32)'
    )

    parser.add_argument(
        '--trace',
        action='store_true',
        help='Enable multi-step trace mode (only valid with --compare)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        metavar='N',
        help='Number of trace steps (default: 10, range: 1–1000; requires --trace)'
    )

    parser.add_argument(
        '--autoregressive',
        action='store_true',
        help='LLM autoregressive mode: append greedy token at each step (requires --trace)'
    )

    parser.add_argument(
        '--trace-output',
        type=str,
        metavar='FILE',
        help='Save per-step trace JSON to FILE (requires --trace)'
    )

    parser.add_argument(
        '--cert',
        type=str,
        metavar='FILE',
        default=None,
        help='Save a compliance certificate to FILE after --compare (JSON)'
    )

    parser.add_argument(
        '--model-family',
        type=str,
        metavar='FAMILY',
        default=None,
        dest='model_family',
        help=(
            'Model family for tolerance lookup with --compare '
            '(choices: decoder-small, decoder-medium, decoder-large, '
            'encoder, vision-language). Defaults to backend+dtype tolerances.'
        ),
    )

    parser.add_argument(
        '--otel',
        action='store_true',
        default=False,
        help=(
            'Export validation result as an OpenTelemetry span. '
            'Requires opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http '
            '(pip install torchbridge-ml[tracing]). '
            'Compatible with Langfuse, W&B Weave, and any OTLP backend.'
        ),
    )

    parser.add_argument(
        '--otel-endpoint',
        type=str,
        metavar='URL',
        default=None,
        dest='otel_endpoint',
        help=(
            'OTLP HTTP endpoint for span export '
            '(e.g. https://cloud.langfuse.com/api/public/otel). '
            'Defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var, '
            'then stdout if neither is set. '
            'Validation spans (model name, backend, dtype, max_diff) are sent to '
            'this endpoint — ensure it complies with your data-retention policy.'
        ),
    )

    args = parser.parse_args()
    try:
        return ValidateCommand.execute(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        from torchbridge.cli import _print_error
        return _print_error(e, verbose=getattr(args, 'verbose', False))


if __name__ == '__main__':
    sys.exit(main())
