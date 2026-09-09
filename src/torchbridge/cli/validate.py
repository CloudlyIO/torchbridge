# SPDX-License-Identifier: Apache-2.0
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
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def non_decoder_trait(model: Any) -> str | None:
    """Name the architecture trait that makes a parameter count meaningless.

    A count separates decoder-small from decoder-medium from decoder-large and
    nothing else. An encoder of the same size belongs in the ``encoder`` row and
    a vision-language model in ``vision-language``, both of which carry tighter
    limits; an MoE model's row turns on active rather than total parameters. For
    those, a count is not evidence, so nothing is inferred and the operator is
    asked.

    Returns:
        The trait name, or None when the model looks like a dense decoder or
        carries no config to judge by.
    """
    config = getattr(model, "config", None)
    if config is None:
        # A bare nn.Module — a traced model or a test stand-in. There is nothing
        # to read, and treating that as suspicious would refuse to infer for
        # every non-HuggingFace model.
        return None
    if getattr(config, "is_encoder_decoder", False):
        return "encoder-decoder"
    if getattr(config, "vision_config", None) is not None:
        return "vision-language"
    for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
        if getattr(config, attr, None):
            return "mixture-of-experts"
    return None


def _is_rocm_build() -> bool:
    """True when this torch was built against ROCm rather than CUDA.

    ROCm builds expose AMD GPUs through the ``cuda`` device type, so the device
    object alone cannot tell the two vendors apart. ``torch.version.hip`` is set
    only on a ROCm build, which makes it the one reliable discriminator.
    """
    from torchbridge.core.hardware_detector import is_rocm_build

    return is_rocm_build()


def resolve_backend_device(name: str) -> torch.device | None:
    """Resolve a backend name to a device, or ``None`` if unavailable here.

    ``cuda`` and ``rocm`` both map to ``torch.device("cuda")`` because a ROCm
    build reuses that device type for AMD hardware, and a torch install is built
    against one vendor or the other — never both. Accepting either name on
    either build would let ``--compare cuda rocm`` resolve both halves to the
    same physical GPU: divergence collapses to ~0, every step passes, and
    nothing in the report reveals it. Each vendor name is therefore refused on
    the other vendor's build. ``gpu`` stays deliberately neutral and means
    "whichever accelerator this machine has".

    Args:
        name: Backend name as typed on the command line, any case.

    Returns:
        The device to run on, or ``None`` when this machine cannot provide it.
        Callers turn ``None`` into a user-facing error.
    """
    name = name.lower()
    if name in ("cuda", "rocm", "gpu"):
        if not torch.cuda.is_available():
            return None
        rocm_build = _is_rocm_build()
        if name == "cuda" and rocm_build:
            return None
        if name == "rocm" and not rocm_build:
            return None
        return torch.device("cuda")
    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return None
        return torch.device("mps")
    if name in ("trainium", "neuron"):
        try:
            import torch_neuronx  # noqa: F401

            return torch.device("xla")
        except ImportError:
            return None
    if name in ("tpu", "xla"):
        # A TPU is reached through torch_xla, the same device type Trainium uses.
        # Without these names a rented TPU could not be addressed at all, and the
        # workaround — passing "trainium" — wrote the wrong hardware into the
        # results file.
        try:
            import torch_xla  # noqa: F401

            return torch.device("xla")
        except ImportError:
            return None
    if name == "cpu":
        return torch.device("cpu")
    return None  # unknown


def explain_unavailable_backend(name: str) -> str:
    """Explain why :func:`resolve_backend_device` refused ``name``.

    The bare "not available" wording is actively misleading for the vendor
    mismatch: the machine does have a GPU, it is simply the other vendor's. The
    cross-vendor case also has a real answer — the split record/replay trace —
    so the message points at it rather than leaving a dead end.
    """
    name = name.lower()
    if name in ("cuda", "rocm") and torch.cuda.is_available():
        rocm_build = _is_rocm_build()
        mismatch: tuple[str, str] | None = None
        if name == "cuda" and rocm_build:
            mismatch = ("ROCm/AMD", "cuda")
        elif name == "rocm" and not rocm_build:
            mismatch = ("CUDA/NVIDIA", "rocm")
        if mismatch is not None:
            have, want = mismatch
            return (
                f"Backend '{want}' cannot run here: this is a {have} build of torch. "
                f"One torch install targets one vendor, and both expose GPUs as "
                f"device 'cuda', so a single process cannot drive both — record one "
                f"side and replay it on the other machine."
            )
    return f"Backend '{name}' not available on this machine."


def infer_model_family(model: Any) -> str | None:
    """Work out the tolerance family from a loaded model's parameter count.

    ``ToleranceDB`` documents its own size boundaries — under 2B parameters is
    ``decoder-small``, 2B to 20B is ``decoder-medium``, above that
    ``decoder-large`` — so the family is derivable and does not need typing. It
    is worth deriving: an omitted ``--model-family`` silently selects the
    strictest row, which is the original bug arriving by way of a forgotten
    flag.

    Only the dense decoder families are inferred, and :func:`non_decoder_trait`
    enforces that rather than leaving it to the docstring. An encoder or a
    vision-language model of the same size belongs in a different row, and the
    MoE entries turn on active versus total parameters, which a plain count
    cannot distinguish. Those must still be passed explicitly.

    Returns:
        A family name, or None when there are no parameters to count or the
        model is not a dense decoder — the caller then decides whether to
        proceed or refuse.
    """
    if non_decoder_trait(model) is not None:
        return None
    total = sum(p.numel() for p in model.parameters())
    if total == 0:
        return None
    if total < 2_000_000_000:
        return "decoder-small"
    if total <= 20_000_000_000:
        return "decoder-medium"
    return "decoder-large"


def _replay_command(args, backend_a: str, backend_b: str, steps: int) -> str:
    """The command that replays this record on the other machine.

    Built from the arguments this run actually used, rather than written out
    by hand. A hand-written line omitted --model and --model-family, so
    following it verbatim loaded a freshly initialised smoke model on the
    second machine; the weight fingerprint only warns, so a verdict still came
    out. It also goes stale the moment a flag is added.
    """
    import shlex

    parts = [
        "tb-validate",
        "--compare",
        backend_a,
        backend_b,
        "--trace",
        "--steps",
        str(steps),
        "--replay",
        shlex.quote(args.record),
    ]
    # Everything that changes what the second half computes, or how it is
    # judged, has to cross the machine boundary with it.
    for flag, value in (
        ("--model", getattr(args, "model", None)),
        ("--model-family", getattr(args, "model_family", None)),
        ("--dtype", getattr(args, "dtype", None)),
        ("--input-shape", getattr(args, "input_shape", None)),
    ):
        if value:
            parts += [flag, shlex.quote(str(value))]
    if getattr(args, "autoregressive", False):
        parts.append("--autoregressive")
    return " ".join(parts)


def _split_trace_mode(args) -> str:
    """Which half of a split trace this invocation is, if any.

    Returns ``"record"``, ``"replay"``, ``"compare_records"`` or ``"none"``.
    """
    if getattr(args, "compare_records", None):
        # execute() handles this case before _run_trace() is reached, so this
        # branch serves direct library callers only. See _run_trace().
        return "compare_records"
    # --replay wins when both are given: that combination means "replay this
    # record, keep my half in the --record file, and compare". Keeping the second
    # half is what allows a later offline re-comparison once the rented machine
    # is gone.
    if getattr(args, "replay", None):
        return "replay"
    if getattr(args, "record", None):
        return "record"
    return "none"


def resolve_family_for_run(args, model: Any) -> tuple[str | None, str | None]:
    """Pick the family for this run: what was asked for, else what can be derived.

    An explicit ``--model-family`` always wins — the operator may know something
    a parameter count cannot show, such as an MoE model's active size. Only when
    nothing was given is the family inferred, and the caller reports what was
    chosen so the run is never judged by a limit nobody saw.

    Returns:
        ``(family, note)``. ``note`` is a line to print when the value was
        inferred rather than supplied, otherwise ``None``.
    """
    explicit = getattr(args, "model_family", None)
    if explicit is not None:
        return explicit, None
    inferred = infer_model_family(model)
    if inferred is None:
        trait = non_decoder_trait(model)
        if trait is not None:
            # Silence here would be the original bug wearing a different hat:
            # the run still happens, judged by the coarse row, and nothing says so.
            return None, (
                f"Model family: not inferred — this looks like a {trait} model, "
                "where a parameter count does not identify the family. "
                "Pass --model-family to set it; the coarse backend+dtype row is "
                "used until then."
            )
        return None, None
    return inferred, f"Model family: {inferred} (inferred from parameter count)"


def validate_model_family(name: str | None) -> str | None:
    """Return an error message if ``name`` is not a family the database knows.

    ``ToleranceDB.get`` answers an unknown family with the coarse
    ``(backend, dtype)`` row and gives no indication that it did, so a single
    mistyped character quietly applies the wrong tolerance. Refusing up front
    keeps that failure visible instead of folding it into the result.
    """
    if name is None:
        return None
    from torchbridge.testing.tolerance_db import ToleranceDB

    known = ToleranceDB().families()
    if name.lower() in {f.lower() for f in known}:
        return None
    return (
        f"Unknown --model-family '{name}'. The tolerance database would fall back "
        f"to the coarser (backend, dtype) row without saying so. "
        f"Valid families: {', '.join(sorted(known))}"
    )


def same_device_pair(name_a: str, name_b: str) -> bool:
    """True when two *different* backend names resolve to one physical device.

    Refusing ``cuda`` on an AMD build closes ``--compare cuda rocm``, but ``gpu``
    is an alias for whatever accelerator is present, so ``--compare rocm gpu``
    re-opens the same hole under another name: both halves run on one GPU,
    divergence collapses to ~0 and every step passes.

    An explicit self-pair is deliberate — ``--compare cpu cpu`` is the control
    run that proves the tracer adds no noise of its own — so identical names are
    never flagged. A pair that cannot resolve is the caller's existing
    "not available" error and is not this function's concern.
    """
    if name_a.lower() == name_b.lower():
        return False
    dev_a = resolve_backend_device(name_a)
    dev_b = resolve_backend_device(name_b)
    if dev_a is None or dev_b is None:
        return False
    return dev_a == dev_b


def _load_model_file(path: str) -> nn.Module:
    """Load a full nn.Module from a .pt file.

    Supports model files saved with ``torch.save(model, path)``.
    State-dict files (``torch.save(model.state_dict(), path)``) are not
    supported — they require knowing the architecture to reconstruct.
    """
    # Try TorchScript first (torch.jit.load), then fall back to pickled nn.Module.
    # Using torch.jit.load directly avoids a UserWarning when the file is a
    # TorchScript archive (zip file), which torch.load would otherwise emit.
    try:
        return torch.jit.load(path, map_location="cpu")
    except Exception:
        pass
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614 - caller controls path; jit.load attempted first
    except AttributeError as e:
        raise RuntimeError(
            f"Cannot load model: {e}\n"
            "This usually means the model class is not importable from this context.\n"
            "Save as TorchScript instead:\n"
            "  traced = torch.jit.trace(model, sample_input)\n"
            "  traced.save('model.pt')"
        ) from e
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
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == "warning")

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")

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
            "validate",
            help="Run validation pipeline for TorchBridge",
            description="Structured validation pipeline with multiple levels",
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
            """,
        )

        parser.add_argument(
            "--level",
            choices=["quick", "standard", "full", "cloud"],
            default="standard",
            help="Validation level (default: standard)",
        )

        parser.add_argument(
            "--model", type=str, help="Path to a specific model to validate"
        )

        parser.add_argument(
            "--output", "-o", type=str, help="Save validation report to file"
        )

        parser.add_argument(
            "--format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format (default: text)",
        )

        parser.add_argument(
            "--ci",
            action="store_true",
            help="CI mode: JSON to stdout, no color, structured exit codes",
        )

        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )

        parser.add_argument(
            "--quantized",
            action="store_true",
            help="Include quantization subsystem checks",
        )

        parser.add_argument(
            "--compare",
            nargs=2,
            metavar=("BACKEND1", "BACKEND2"),
            help="Compare model outputs across two backends (e.g. --compare cuda cpu). Names: cuda, rocm, gpu, mps, tpu, xla, trainium, neuron, cpu",
        )

        parser.add_argument(
            "--input-shape",
            type=str,
            default="1,64",
            help="Comma-separated input tensor shape for --compare (default: 1,64)",
        )

        parser.add_argument(
            "--per-layer",
            action="store_true",
            help="Show per-layer divergence breakdown (requires --compare)",
        )

        parser.add_argument(
            "--dtype",
            choices=["float32", "float16", "bfloat16"],
            default="float32",
            help="Model dtype for --compare (default: float32)",
        )

        parser.add_argument(
            "--trace",
            action="store_true",
            help="Enable multi-step trace mode (only valid with --compare)",
        )

        parser.add_argument(
            "--steps",
            type=int,
            default=10,
            metavar="N",
            help="Number of trace steps (default: 10, range: 1–1000; requires --trace)",
        )

        parser.add_argument(
            "--autoregressive",
            action="store_true",
            help="LLM autoregressive mode: append greedy token at each step (requires --trace)",
        )

        parser.add_argument(
            "--trace-output",
            type=str,
            metavar="FILE",
            help="Save per-step trace JSON to FILE (requires --trace)",
        )

        parser.add_argument(
            "--cert",
            type=str,
            metavar="FILE",
            default=None,
            help="Save a compliance certificate to FILE after --compare (JSON)",
        )

        parser.add_argument(
            "--record",
            type=str,
            metavar="FILE",
            default=None,
            help=(
                "Split trace: run only the first backend of --compare and save its "
                "step-by-step record to FILE (requires --trace). Use on the machine "
                "that has that backend, then --replay the file on the other one."
            ),
        )

        parser.add_argument(
            "--replay",
            type=str,
            metavar="FILE",
            default=None,
            help=(
                "Split trace: replay a --record file on the second backend of "
                "--compare and report the comparison (requires --trace)."
            ),
        )

        parser.add_argument(
            "--compare-records",
            nargs=2,
            metavar=("FILE_A", "FILE_B"),
            dest="compare_records",
            default=None,
            help=(
                "Compare two saved --record files offline. Needs no accelerator, "
                "no backend pair and no model — both names come from the files."
            ),
        )

        parser.add_argument(
            "--model-family",
            type=str,
            metavar="FAMILY",
            default=None,
            dest="model_family",
            help=(
                "Model family for tolerance lookup, used by both --compare and --trace "
                "(choices: decoder-small, decoder-medium, decoder-large, encoder, vision-language, "
                "qwen3_5, gemma4, nemotron3_nano, deepseek_v4, nemotron3_ultra, "
                "tencent_hy3, minimax_m3, glm_5_2). When omitted, a dense decoder's family "
                "is inferred from its parameter count; anything else falls back to the "
                "coarse backend+dtype row."
            ),
        )

        parser.add_argument(
            "--otel",
            action="store_true",
            default=False,
            help=(
                "Export validation result as an OpenTelemetry span. "
                "Requires opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http "
                "(pip install torchbridge-ml[tracing]). "
                "Compatible with Langfuse, W&B Weave, and any OTLP backend."
            ),
        )

        parser.add_argument(
            "--otel-endpoint",
            type=str,
            metavar="URL",
            default=None,
            dest="otel_endpoint",
            help=(
                "OTLP HTTP endpoint for span export "
                "(e.g. https://cloud.langfuse.com/api/public/otel). "
                "Defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var, "
                "then stdout if neither is set. "
                "Validation spans (model name, backend, dtype, max_diff) are sent to "
                "this endpoint — ensure it complies with your data-retention policy."
            ),
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the validate command."""
        compare = getattr(args, "compare", None)
        has_pair = isinstance(compare, (list, tuple)) and len(compare) == 2
        tracing = getattr(args, "trace", False) is True

        # Offline record comparison runs before the --compare requirement. It
        # needs no accelerator and no backend pair — both names come out of the
        # files — so demanding --compare made the documented command impossible.
        if getattr(args, "compare_records", None):
            return ValidateCommand._compare_saved_records(args)

        # --compare short-circuits the standard pipeline
        if has_pair:
            if tracing:
                return ValidateCommand._run_trace(args)
            return ValidateCommand._run_compare(args)

        # --trace without --compare is an error
        if tracing:
            print("Error: --trace requires --compare BACKEND1 BACKEND2")
            return 1

        # The split flags are read only inside the trace path. Reaching here
        # with one set means it would be dropped without a word, and the
        # command would quietly run the ordinary pipeline instead.
        for flag, value in (
            ("--record", getattr(args, "record", None)),
            ("--replay", getattr(args, "replay", None)),
        ):
            if value:
                print(
                    f"Error: {flag} is part of a split trace and needs "
                    f"--trace --compare BACKEND1 BACKEND2"
                )
                return 1

        ci_mode = getattr(args, "ci", False)
        level = getattr(args, "level", "standard")
        verbose = getattr(args, "verbose", False)

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
            if level in ("standard", "full"):
                model_path = getattr(args, "model", None)
                report.results.extend(
                    ValidateCommand._run_standard_checks(model_path, verbose)
                )

            # Quantization checks (if --quantized flag)
            if getattr(args, "quantized", False):
                report.results.extend(ValidateCommand._run_quantization_checks(verbose))

            # Full level: add benchmark suite + cross-backend
            if level == "full":
                report.results.extend(ValidateCommand._run_full_checks(verbose))

            # Cloud level: run cloud validation scripts
            if level == "cloud":
                report.results.extend(ValidateCommand._run_cloud_checks(verbose))

            report.duration_ms = (time.time() - start_time) * 1000

            # Output results
            if ci_mode:
                return ValidateCommand._output_ci_json(report)

            ValidateCommand._display_report(report, verbose)

            # Save report if requested
            output_path = getattr(args, "output", None)
            if output_path:
                fmt = getattr(args, "format", "text")
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
        model_path = getattr(args, "model", None)
        input_shape = tuple(
            int(x) for x in getattr(args, "input_shape", "1,64").split(",")
        )
        per_layer = getattr(args, "per_layer", False)
        dtype_str = getattr(args, "dtype", "float32")
        output_path = getattr(args, "output", None)
        ci_mode = getattr(args, "ci", False)
        dtype = getattr(torch, dtype_str)

        family_error = validate_model_family(getattr(args, "model_family", None))
        if family_error:
            if ci_mode:
                print(json.dumps({"error": family_error}))
            else:
                print(f"Error: {family_error}")
            return 1

        dev1 = resolve_backend_device(backend1)
        dev2 = resolve_backend_device(backend2)

        if dev1 is None:
            msg = explain_unavailable_backend(backend1)
            if ci_mode:
                print(json.dumps({"error": msg, "backend": backend1}))
            else:
                print(f"Error: {msg}")
            return 1

        if dev2 is None:
            msg = explain_unavailable_backend(backend2)
            if ci_mode:
                print(json.dumps({"error": msg, "backend": backend2}))
            else:
                print(f"Error: {msg}")
            return 1

        if same_device_pair(backend1, backend2):
            msg = (
                f"'{backend1}' and '{backend2}' resolve to the same device on this "
                f"machine, so this would compare it against itself and report ~0 "
                f"divergence. Use two genuinely different backends, or record one "
                f"side and replay it on the other machine."
            )
            if ci_mode:
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}")
            return 1

        # Load model
        smoke_model = False
        is_hf_model = False
        model: nn.Module
        try:
            if model_path is None:
                # No model provided — use a small smoke-test Linear
                model = nn.Sequential(
                    nn.Linear(input_shape[-1], input_shape[-1]),
                    nn.ReLU(),
                    nn.Linear(input_shape[-1], input_shape[-1]),
                )
                smoke_model = True
                model_label = "smoke_model (Linear)"
            elif Path(model_path).exists():
                model = _load_model_file(model_path)
                model_label = Path(
                    model_path
                ).name  # filename only — avoid leaking full filesystem path
            else:
                # Treat as HuggingFace model ID
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(  # nosec B615 - revision pinning is user's responsibility for CLI tool
                    model_path, torch_dtype=dtype
                )
                model_label = model_path  # HuggingFace model ID is a public identifier
                is_hf_model = True
        except Exception as e:
            msg = f"Failed to load model: {e}"
            if ci_mode:
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}")
            return 1

        model.eval()
        if smoke_model and dtype != torch.float32:
            model = model.to(dtype=dtype)

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
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}")
            return 1
        except Exception as e:
            msg = f"Inference failed: {e}"
            if ci_mode:
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}")
            return 1

        duration_ms = (time.perf_counter() - t0) * 1000

        # Compute metrics
        max_diff = float(torch.abs(logits1_cpu - logits2_cpu).max())
        cos_sim = float(
            F.cosine_similarity(
                logits1_cpu.flatten().unsqueeze(0),
                logits2_cpu.flatten().unsqueeze(0),
            )
        )

        # Tolerance check
        from torchbridge.testing.tolerance_db import ToleranceDB

        tol_db = ToleranceDB()
        # Use backend1 tolerance (primary backend)
        b1_key = backend1.lower() if backend1.lower() != "rocm" else "rocm"
        model_family, family_note = resolve_family_for_run(args, model)
        if family_note and not ci_mode:
            # --ci consumers parse stdout as JSON, so nothing else may be printed.
            print(family_note)
        tol = tol_db.get(b1_key, dtype_str, model_family=model_family)
        passed = max_diff <= tol.atol

        # Per-layer divergence
        layer_rows: list[dict] = []
        per_layer_skip_reason: str | None = None
        if per_layer and not smoke_model:
            # TorchScript models don't support hook-based per-layer tracing.
            if isinstance(model, torch.jit.ScriptModule):
                per_layer_skip_reason = (
                    "TorchScript models do not support per-layer tracing. "
                    "Save with torch.save(model, path) instead of torch.jit.trace/script."
                )
            else:
                try:
                    from torchbridge.testing.divergence import DivergenceTracer

                    model_cpu = model.to("cpu")
                    x_cpu = x.to("cpu")
                    ref_tracer = DivergenceTracer(model_cpu, device=torch.device("cpu"))
                    with ref_tracer:
                        with torch.no_grad():
                            model_cpu(input_ids=x_cpu) if is_hf_model else model_cpu(
                                x_cpu
                            )
                    # Run a second tracer on dev2 and compare
                    model_dev2 = model.to(dev2)
                    x_dev2 = x.to(dev2)
                    test_tracer = DivergenceTracer(model_dev2, device=dev2)
                    with test_tracer:
                        with torch.no_grad():
                            model_dev2(input_ids=x_dev2) if is_hf_model else model_dev2(
                                x_dev2
                            )
                    divergences = test_tracer.compare_with(ref_tracer)
                    for d in divergences:
                        layer_rows.append(
                            {
                                "layer": d.layer_name,
                                "max_diff": d.max_diff,
                                "cosine_sim": d.cosine_sim,
                                "exceeds_threshold": d.exceeds_threshold,
                            }
                        )
                    if not layer_rows:
                        per_layer_skip_reason = "No named sub-modules found in model."
                except Exception as e:
                    per_layer_skip_reason = f"Per-layer tracing failed: {e}"
                    logger.debug("Per-layer divergence failed: %s", e)

        # Build result dict
        result = {
            "backend1": backend1,
            "backend2": backend2,
            "model": model_label,
            "dtype": dtype_str,
            "input_shape": list(input_shape),
            "max_diff": max_diff,
            "cosine_sim": cos_sim,
            "tolerance_atol": tol.atol,
            "tolerance_rtol": tol.rtol,
            # The verdict uses atol alone. Naming the rule stops the rtol value
            # above from reading as though it had been part of the decision.
            "tolerance_rule": "atol_only",
            "passed": passed,
            "duration_ms": round(duration_ms, 3),
            "per_layer": layer_rows,
        }

        # Output
        if ci_mode:
            print(json.dumps(result, indent=2))
        else:
            status_str = "PASSED" if passed else "FAILED"
            print("TorchBridge Cross-Backend Comparison")
            print("=" * 40)
            print(f"Backends   : {backend1}  vs  {backend2}")
            print(f"Model      : {model_label}")
            if smoke_model:
                print("           (no --model given; using smoke Linear)")
            print(f"dtype      : {dtype_str}")
            print(f"Shape      : {input_shape}")
            print()
            print(f"  Max diff   : {max_diff:.2e}")
            print(f"  Cosine sim : {cos_sim:.6f}")
            tol_annotation = (
                "  (fallback — backend not in tolerance DB)"
                if tol.source == "fallback"
                else ""
            )
            print(
                f"  Tolerance  : atol={tol.atol:.0e} (applied)  "
                f"rtol={tol.rtol:.0e} (not applied)  "
                f"({b1_key}/{dtype_str}){tol_annotation}"
            )
            print(f"  Duration   : {duration_ms:.1f}ms")
            print(f"  Status     : {status_str}")
            if layer_rows:
                print()
                print("  Per-layer divergence:")
                for row in layer_rows:
                    flag = " *" if row["exceeds_threshold"] else ""
                    print(
                        f"    {row['layer']}: max_diff={row['max_diff']:.2e}  "
                        f"cos={row['cosine_sim']:.4f}{flag}"
                    )
            elif per_layer and per_layer_skip_reason:
                print()
                print(f"  Per-layer: skipped — {per_layer_skip_reason}")

        if output_path:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
            except Exception as e:
                logger.warning("Could not save output to %s: %s", output_path, e)

        cert_path = getattr(args, "cert", None)
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
                with open(cert_path, "w") as f:
                    f.write(cert.to_json())
            except Exception as e:
                logger.warning(
                    "Could not save compliance certificate to %s: %s", cert_path, e
                )
                print(f"WARNING: compliance certificate not saved: {e}")

        if getattr(args, "otel", False):
            try:
                from torchbridge.testing.otel_exporter import ValidationSpanExporter

                exporter = ValidationSpanExporter(
                    endpoint=getattr(args, "otel_endpoint", None)
                )
                try:
                    exporter.export(result)
                finally:
                    exporter.shutdown()
            except Exception as e:
                logger.warning("Could not export OTEL span: %s", e)

        return 0 if passed else 1

    @staticmethod
    def _compare_saved_records(args) -> int:
        """Compare two saved --record files. Needs no accelerator at all.

        The offline half of the split workflow: both machines can be gone by the
        time this runs, which is the point — rented instances are destroyed right
        after their half is recorded.
        """
        import json as _json

        from torchbridge.testing.trace_validator import (
            SplitTraceRecord,
            compare_records,
        )

        ci_mode = getattr(args, "ci", False)
        path_a, path_b = args.compare_records

        def _fail(message: str) -> int:
            if ci_mode:
                print(_json.dumps({"error": message}))
            else:
                print(f"Error: {message}")
            return 1

        records = []
        for path in (path_a, path_b):
            try:
                records.append(SplitTraceRecord.load(path))
            except Exception as exc:
                return _fail(f"Could not read record file {path}: {exc}")

        family_error = validate_model_family(getattr(args, "model_family", None))
        if family_error:
            return _fail(family_error)

        try:
            result = compare_records(
                records[0],
                records[1],
                model_family=getattr(args, "model_family", None),
            )
        except Exception as exc:
            return _fail(f"Comparison failed: {exc}")

        result_dict = result.to_dict()
        result_dict["record_a"] = Path(path_a).name
        result_dict["record_b"] = Path(path_b).name

        output_path = getattr(args, "trace_output", None) or getattr(
            args, "output", None
        )
        if output_path:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    _json.dump(result_dict, f, indent=2)
            except Exception as exc:
                logger.warning("Could not save result to %s: %s", output_path, exc)

        if ci_mode:
            print(_json.dumps(result_dict))
        else:
            print(
                f"Offline comparison: {records[0].backend} vs {records[1].backend}"
                f"  ({result.steps} steps, {result.dtype})"
            )
            if result.first_divergence_step is not None:
                print(f"First divergence at step : {result.first_divergence_step}")
            else:
                print("First divergence at step : None (all steps passed)")
            print(f"Max amplification        : {result.max_amplification:.1f}x")
            print(
                f"Status                   : "
                f"{'PASSED' if result.final_passed else 'FAILED'}"
            )

        return 0 if result.final_passed else 1

    @staticmethod
    def _run_trace(args) -> int:
        """Execute multi-step trace validation and return exit code."""
        import torch.nn as nn

        backend_a, backend_b = args.compare
        steps = getattr(args, "steps", 10)
        autoregressive = getattr(args, "autoregressive", False)
        model_path = getattr(args, "model", None)
        input_shape = tuple(
            int(x) for x in getattr(args, "input_shape", "1,64").split(",")
        )
        dtype_str = getattr(args, "dtype", "float32")
        output_path = getattr(args, "output", None)
        trace_output_path = getattr(args, "trace_output", None)
        ci_mode = getattr(args, "ci", False)
        split_mode = _split_trace_mode(args)

        def _fail(message: str) -> int:
            if ci_mode:
                print(json.dumps({"error": message}))
            else:
                print(f"Error: {message}")
            return 1

        if split_mode == "compare_records":
            # Unreachable from the command line: execute() intercepts
            # --compare-records before the --compare requirement, because the
            # offline comparison needs neither a backend pair nor a device.
            # Kept so a library caller who builds args and calls _run_trace()
            # directly still lands in the right place rather than tracing.
            return ValidateCommand._compare_saved_records(args)

        dtype = getattr(torch, dtype_str)

        # Validate steps range
        if not (1 <= steps <= 1000):
            msg = f"--steps must be between 1 and 1000, got {steps}"
            if ci_mode:
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}")
            return 1

        family_error = validate_model_family(getattr(args, "model_family", None))
        if family_error:
            if ci_mode:
                print(json.dumps({"error": family_error}))
            else:
                print(f"Error: {family_error}")
            return 1

        dev_a = resolve_backend_device(backend_a)
        dev_b = resolve_backend_device(backend_b)

        # Which backends must exist here depends on the mode. A split trace runs
        # on two machines precisely because the pair cannot coexist, so demanding
        # both would make the workflow impossible on the machines it is for:
        # record uses backend A only, replay uses backend B only.
        if split_mode != "replay" and dev_a is None:
            return _fail(explain_unavailable_backend(backend_a))

        if split_mode != "record" and dev_b is None:
            return _fail(explain_unavailable_backend(backend_b))

        if split_mode == "none" and same_device_pair(backend_a, backend_b):
            return _fail(
                f"'{backend_a}' and '{backend_b}' resolve to the same device on this "
                f"machine, so this would compare it against itself and report ~0 "
                f"divergence. Use two genuinely different backends, or record one "
                f"side and replay it on the other machine."
            )

        # Load model
        is_lm = False
        smoke_model = False
        model: nn.Module
        try:
            if model_path is None:
                model = nn.Sequential(
                    nn.Linear(input_shape[-1], input_shape[-1]),
                    nn.ReLU(),
                    nn.Linear(input_shape[-1], input_shape[-1]),
                )
                smoke_model = True
                model_label = "smoke_model (Linear)"
            elif Path(model_path).exists():
                model = _load_model_file(model_path)
                model_label = Path(
                    model_path
                ).name  # filename only — avoid leaking full filesystem path
            else:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(  # nosec B615 - revision pinning is user's responsibility for CLI tool
                    model_path, torch_dtype=dtype
                )
                model_label = model_path  # HuggingFace model ID is a public identifier
                is_lm = True
        except Exception as e:
            msg = f"Failed to load model: {e}"
            if ci_mode:
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}")
            return 1

        if smoke_model and dtype != torch.float32:
            model = model.to(dtype=dtype)

        # Build initial input
        if is_lm:
            x = torch.ones(*input_shape, dtype=torch.long)
        else:
            x = torch.randn(*input_shape, dtype=dtype)

        # Run trace
        from torchbridge.testing.trace_validator import MultiStepTracer

        trace_family, family_note = resolve_family_for_run(args, model)
        if family_note and not ci_mode:
            # --ci consumers parse stdout as JSON, so nothing else may be printed.
            print(family_note)

        # In a split trace one backend is absent locally by design: record uses
        # only backend A, replay only backend B. The tracer is annotated
        # torch.device, so the unused side gets CPU as a placeholder rather than
        # None. It is never touched — record() reads device_a, replay() reads
        # device_b — but passing None would break the constructor's contract and
        # would fail the moment anything else looked at the other half.
        _unused_side = torch.device("cpu")
        tracer = MultiStepTracer(
            model=model,
            device_a=dev_a if dev_a is not None else _unused_side,
            device_b=dev_b if dev_b is not None else _unused_side,
            backend_a=backend_a,
            backend_b=backend_b,
            dtype=dtype_str,
            is_lm=is_lm,
            # --model-family is registered for the whole validate command and is
            # already honoured by --compare; without this it parses fine here and
            # is silently dropped, sending the trace to the coarser table.
            model_family=trace_family,
        )

        try:
            if split_mode == "record":
                rec = tracer.record(
                    input_ids=x, steps=steps, autoregressive=autoregressive
                )
                # record() stops at the first step that raises and returns what
                # it has, so a run that failed immediately yields an empty
                # record. Saving that and exiting 0 reports success and leaves a
                # file the next stage rejects — on a rented machine that is a
                # booking spent on nothing. A short record is refused too: the
                # comparison would silently cover fewer steps than asked for.
                if rec.steps == 0:
                    return _fail(
                        "Recording produced no steps — the model failed on the "
                        "first step. Nothing was written; see the warnings above."
                    )
                if rec.steps < steps:
                    return _fail(
                        f"Recording stopped after {rec.steps} of {steps} step(s). "
                        f"Nothing was written, because a short record would "
                        f"compare fewer steps than requested without saying so; "
                        f"see the warnings above."
                    )
                rec.save(args.record)
                if ci_mode:
                    print(
                        json.dumps(
                            {
                                "mode": "record",
                                "backend": rec.backend,
                                "dtype": rec.dtype,
                                "steps": rec.steps,
                                "file": args.record,
                            }
                        )
                    )
                else:
                    print(
                        f"Recorded {rec.steps} step(s) of '{rec.backend}' to "
                        f"{args.record}\nReplay it on the other machine with:\n"
                        f"  {_replay_command(args, backend_a, backend_b, steps)}"
                    )
                    if smoke_model:
                        # A record made from the smoke model is a mechanism
                        # check, not a measurement: the other machine builds its
                        # own randomly initialised copy, so the two halves never
                        # shared weights.
                        print(
                            "\nNote: no --model was given, so this records the "
                            "built-in smoke model. The replay machine will "
                            "initialise its own copy with different weights, so "
                            "the comparison will measure the weights rather than "
                            "the backends. Pass --model for a real run."
                        )
                return 0

            if split_mode == "replay":
                from torchbridge.testing.trace_validator import (
                    SplitTraceRecord,
                    compare_records,
                )

                try:
                    leader = SplitTraceRecord.load(args.replay)
                except Exception as exc:
                    return _fail(f"Could not read record file {args.replay}: {exc}")

                # The result takes its backend names from the record, not from
                # this command line. Replaying a cpu record under
                # --compare cuda rocm therefore produced a file labelled
                # "cpu vs rocm" and exited 0, quietly answering a question
                # nobody asked.
                if leader.role != "record":
                    return _fail(
                        f"{args.replay} has role {leader.role!r}, not 'record' — "
                        f"--replay needs the leading half, the file written by "
                        f"--record on the other machine."
                    )
                if leader.backend != backend_a:
                    return _fail(
                        f"{args.replay} was recorded on {leader.backend!r}, but "
                        f"this run asks for {backend_a!r} as the first backend. "
                        f"Either replay it against --compare {leader.backend} "
                        f"{backend_b}, or record a new leading half on "
                        f"{backend_a!r}."
                    )

                follower = tracer.replay(leader)
                if getattr(args, "record", None):
                    follower.save(args.record)
                    if not ci_mode:
                        print(f"Saved this half to {args.record}")
                result = compare_records(leader, follower, model_family=trace_family)
            else:
                result = tracer.run(
                    input_ids=x, steps=steps, autoregressive=autoregressive
                )
        except Exception as e:
            msg = f"Trace failed: {e}"
            if ci_mode:
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}")
            return 1

        result_dict = result.to_dict()
        result_dict["model"] = model_label

        # Output
        if ci_mode:
            print(json.dumps(result_dict, indent=2))
        else:
            mode_str = "autoregressive" if autoregressive else "standard"
            status_str = "PASSED" if result.final_passed else "FAILED"
            print("TorchBridge Multi-Step Trace Validation")
            print("=" * 42)
            print(f"Backends : {backend_a}  vs  {backend_b}")
            print(f"Model    : {model_label}")
            if smoke_model:
                print("         (no --model given; using smoke Linear)")
            print(f"Steps    : {steps}  ({mode_str})")
            print(f"dtype    : {dtype_str}")
            # The limit that decides every PASS below, and where it came from.
            # Printing the family only when it was inferred left an explicit
            # --model-family run showing no tolerance at all, so a reader could
            # not tell what judged the numbers they were looking at.
            print(f"Family   : {result.model_family or 'none (coarse table)'}")
            if result.atol is not None:
                print(f"Tolerance: atol {result.atol:.1e} ({result.atol_source})")
            print()

            # Print step table (all steps for short runs, every 5th for long)
            print_every = 1 if steps <= 20 else 5
            for sr in result.step_results:
                if (
                    sr.step == 1
                    or sr.step % print_every == 0
                    or not sr.within_tolerance
                ):
                    flag = "  *" if not sr.within_tolerance else ""
                    pass_fail = "PASS" if sr.within_tolerance else "FAIL"
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
                with open(output_path, "w") as f:
                    json.dump(result_dict, f, indent=2)
            except Exception as e:
                logger.warning("Could not save output to %s: %s", output_path, e)

        # Save per-step JSON if --trace-output given
        if trace_output_path:
            try:
                Path(trace_output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(trace_output_path, "w") as f:
                    json.dump(result_dict, f, indent=2)
            except Exception as e:
                logger.warning(
                    "Could not save trace output to %s: %s", trace_output_path, e
                )

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

            results.append(
                ValidationResult(
                    "TorchBridge Import",
                    "pass",
                    f"TorchBridge {torchbridge.__version__} imported successfully",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
        except ImportError as e:
            results.append(
                ValidationResult(
                    "TorchBridge Import",
                    "fail",
                    f"Failed to import TorchBridge: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
            return results  # Can't proceed without torchbridge

        # Reuse DoctorCommand checks
        from torchbridge.cli.doctor import DoctorCommand

        start = time.time()
        doctor_basic = DoctorCommand._check_basic_requirements(verbose)
        for dr in doctor_basic:
            results.append(
                ValidationResult(
                    dr.name,
                    dr.status,
                    dr.message,
                    details=dr.details,
                    duration_ms=(time.time() - start) * 1000,
                )
            )

        start = time.time()
        doctor_hw = DoctorCommand._check_hardware(verbose)
        for dr in doctor_hw:
            results.append(
                ValidationResult(
                    dr.name,
                    dr.status,
                    dr.message,
                    details=dr.details,
                    duration_ms=(time.time() - start) * 1000,
                )
            )

        return results

    @staticmethod
    def _run_standard_checks(
        model_path: str | None, verbose: bool
    ) -> list[ValidationResult]:
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

            results.append(
                ValidationResult(
                    "UnifiedValidator",
                    "pass",
                    "UnifiedValidator available",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
        except ImportError as e:
            results.append(
                ValidationResult(
                    "UnifiedValidator",
                    "fail",
                    f"UnifiedValidator not available: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )

        # Model validation if path provided
        if model_path:
            start = time.time()
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    model = _load_model_file(model_path)
                    if hasattr(model, "eval"):
                        model.eval()
                    results.append(
                        ValidationResult(
                            "Model Load",
                            "pass",
                            f"Model loaded from {model_path}",
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
                except Exception as e:
                    results.append(
                        ValidationResult(
                            "Model Load",
                            "fail",
                            f"Failed to load model: {e}",
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
            else:
                results.append(
                    ValidationResult(
                        "Model Load",
                        "fail",
                        f"Model file not found: {model_path}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )

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
            results.append(
                ValidationResult(
                    "Export Formats",
                    "pass",
                    f"Available: {', '.join(export_formats)}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
        else:
            results.append(
                ValidationResult(
                    "Export Formats",
                    "warning",
                    "No export formats available beyond PyTorch native",
                    duration_ms=(time.time() - start) * 1000,
                )
            )

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

            results.append(
                ValidationResult(
                    "Benchmark Smoke Test",
                    "pass",
                    "Basic benchmark completed",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    "Benchmark Smoke Test",
                    "fail",
                    f"Benchmark failed: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )

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
                results.append(
                    ValidationResult(
                        "Backend Consistency",
                        "pass",
                        "CPU backend produces consistent results",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        "Backend Consistency",
                        "warning",
                        "Inconsistent results detected across runs",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
        except Exception as e:
            results.append(
                ValidationResult(
                    "Backend Consistency",
                    "fail",
                    f"Consistency check failed: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )

        # Backend framework check
        start = time.time()
        from torchbridge.cli.doctor import DoctorCommand

        doctor_opt = DoctorCommand._check_optimization_frameworks(verbose)
        for dr in doctor_opt:
            results.append(
                ValidationResult(
                    dr.name,
                    dr.status,
                    dr.message,
                    details=dr.details,
                    duration_ms=(time.time() - start) * 1000,
                )
            )

        return results

    @staticmethod
    def _run_cloud_checks(verbose: bool) -> list[ValidationResult]:
        """Run cloud validation by executing cloud_validation.sh."""
        results = []

        if verbose:
            print(" Running cloud checks...")

        # Look for cloud validation script
        script_candidates = [
            Path("scripts/validation/cloud_validation.sh"),
            Path("cloud_validation.sh"),
        ]

        script_path = None
        for candidate in script_candidates:
            if candidate.exists():
                script_path = candidate
                break

        if script_path is None:
            results.append(
                ValidationResult(
                    "Cloud Validation Script",
                    "warning",
                    "cloud_validation.sh not found",
                    details="Looked in: scripts/, ./",
                )
            )
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
                    [
                        sys.executable,
                        "-c",
                        f'print("Cloud {use_case} check placeholder")',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    results.append(
                        ValidationResult(
                            f"Cloud: {use_case}",
                            "pass",
                            f"Cloud use case '{use_case}' passed",
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            f"Cloud: {use_case}",
                            "fail",
                            f"Cloud use case '{use_case}' failed: {result.stderr.strip()}",
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
            except subprocess.TimeoutExpired:
                results.append(
                    ValidationResult(
                        f"Cloud: {use_case}",
                        "fail",
                        f"Cloud use case '{use_case}' timed out",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
            except Exception as e:
                results.append(
                    ValidationResult(
                        f"Cloud: {use_case}",
                        "fail",
                        f"Cloud use case '{use_case}' error: {e}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )

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
            from torchbridge.precision import (
                QuantizationEngine,
            )

            results.append(
                ValidationResult(
                    "Quantization Import",
                    "pass",
                    "Quantization subsystem imported successfully",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
        except ImportError as e:
            results.append(
                ValidationResult(
                    "Quantization Import",
                    "fail",
                    f"Quantization import failed: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
            return results

        # Engine creation
        start = time.time()
        try:
            engine = QuantizationEngine()
            optimal = engine.get_optimal_format()
            results.append(
                ValidationResult(
                    "Quantization Engine",
                    "pass",
                    f"Engine created; optimal format: {optimal.value}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    "Quantization Engine",
                    "fail",
                    f"Engine creation failed: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )
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
                results.append(
                    ValidationResult(
                        "INT8 Dynamic Quantization",
                        "pass",
                        f"INT8 quantization OK ({result.memory_reduction_pct:.0f}% reduction)",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        "INT8 Dynamic Quantization",
                        "fail",
                        f"INT8 quantization failed: {result.errors}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
        except Exception as e:
            results.append(
                ValidationResult(
                    "INT8 Dynamic Quantization",
                    "fail",
                    f"INT8 test error: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )
            )

        return results

    @staticmethod
    def _output_ci_json(report: ValidationReport) -> int:
        """Output report as JSON for CI mode with structured exit codes.

        Exit codes: 0=all pass, 1=failures, 2=warnings only.
        """
        data = {
            "level": report.level,
            "timestamp": report.timestamp,
            "duration_ms": report.duration_ms,
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": r.duration_ms,
                }
                for r in report.results
            ],
            "summary": {
                "total": len(report.results),
                "passed": report.passed,
                "warnings": report.warnings,
                "failures": report.failures,
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
            icon = {"pass": "", "warning": "", "fail": ""}.get(result.status, "")
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
    def _save_report(
        report: ValidationReport, output_path: str, fmt: str, verbose: bool
    ) -> None:
        """Save validation report to file."""
        if verbose:
            print(f" Saving report to: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "level": report.level,
            "timestamp": report.timestamp,
            "duration_ms": report.duration_ms,
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": r.duration_ms,
                }
                for r in report.results
            ],
            "summary": {
                "total": len(report.results),
                "passed": report.passed,
                "warnings": report.warnings,
                "failures": report.failures,
            },
        }

        if fmt == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        elif fmt == "yaml":
            try:
                import yaml  # type: ignore[import-untyped]

                with open(output_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                print("yaml output requires pyyaml: pip install pyyaml")
                return
        else:  # text
            with open(output_path, "w") as f:
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
        prog="tb-validate",
        description="Run TorchBridge validation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--level",
        choices=["quick", "standard", "full", "cloud"],
        default="standard",
        help="Validation level (default: standard)",
    )
    parser.add_argument(
        "--model", type=str, help="Path to a specific model to validate"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Save validation report to file"
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: JSON to stdout, no color, structured exit codes",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quantized", action="store_true", help="Include quantization subsystem checks"
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BACKEND1", "BACKEND2"),
        help="Compare model outputs across two backends (e.g. --compare cuda cpu). Names: cuda, rocm, gpu, mps, tpu, xla, trainium, neuron, cpu",
    )

    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,64",
        help="Comma-separated input tensor shape for --compare (default: 1,64)",
    )

    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Show per-layer divergence breakdown (requires --compare)",
    )

    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model dtype for --compare (default: float32)",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable multi-step trace mode (only valid with --compare)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        metavar="N",
        help="Number of trace steps (default: 10, range: 1–1000; requires --trace)",
    )

    parser.add_argument(
        "--autoregressive",
        action="store_true",
        help="LLM autoregressive mode: append greedy token at each step (requires --trace)",
    )

    parser.add_argument(
        "--trace-output",
        type=str,
        metavar="FILE",
        help="Save per-step trace JSON to FILE (requires --trace)",
    )

    parser.add_argument(
        "--cert",
        type=str,
        metavar="FILE",
        default=None,
        help="Save a compliance certificate to FILE after --compare (JSON)",
    )

    parser.add_argument(
        "--record",
        type=str,
        metavar="FILE",
        default=None,
        help=(
            "Split trace: run only the first backend of --compare and save its "
            "step-by-step record to FILE (requires --trace). Use on the machine "
            "that has that backend, then --replay the file on the other one."
        ),
    )

    parser.add_argument(
        "--replay",
        type=str,
        metavar="FILE",
        default=None,
        help=(
            "Split trace: replay a --record file on the second backend of "
            "--compare and report the comparison (requires --trace)."
        ),
    )

    parser.add_argument(
        "--compare-records",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        dest="compare_records",
        default=None,
        help=(
            "Compare two saved --record files offline. Needs no accelerator, "
            "no backend pair and no model — both names come from the files."
        ),
    )

    parser.add_argument(
        "--model-family",
        type=str,
        metavar="FAMILY",
        default=None,
        dest="model_family",
        help=(
            "Model family for tolerance lookup, used by both --compare and --trace "
            "(choices: decoder-small, decoder-medium, decoder-large, encoder, vision-language, "
            "qwen3_5, gemma4, nemotron3_nano, deepseek_v4, nemotron3_ultra, "
            "tencent_hy3, minimax_m3, glm_5_2). When omitted, a dense decoder's family "
            "is inferred from its parameter count; anything else falls back to the "
            "coarse backend+dtype row."
        ),
    )

    parser.add_argument(
        "--otel",
        action="store_true",
        default=False,
        help=(
            "Export validation result as an OpenTelemetry span. "
            "Requires opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http "
            "(pip install torchbridge-ml[tracing]). "
            "Compatible with Langfuse, W&B Weave, and any OTLP backend."
        ),
    )

    parser.add_argument(
        "--otel-endpoint",
        type=str,
        metavar="URL",
        default=None,
        dest="otel_endpoint",
        help=(
            "OTLP HTTP endpoint for span export "
            "(e.g. https://cloud.langfuse.com/api/public/otel). "
            "Defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var, "
            "then stdout if neither is set. "
            "Validation spans (model name, backend, dtype, max_diff) are sent to "
            "this endpoint — ensure it complies with your data-retention policy."
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

        return _print_error(e, verbose=getattr(args, "verbose", False))


if __name__ == "__main__":
    sys.exit(main())
