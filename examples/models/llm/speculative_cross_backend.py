"""
Speculative Decoding Cross-Backend Example

Demonstrates TorchBridge's backend-aware speculative decoding:
1. Query the compatibility matrix for supported methods
2. Auto-select optimal method per hardware
3. Configure SpeculationEngine with generate() kwargs
4. Detect inference phase (prefill vs decode)
5. Structured output with JSON schema validation

Usage:
    python examples/models/llm/speculative_cross_backend.py
"""

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.inference import (
    OutputFormat,
    PhaseDetector,
    SpeculationCompatibilityMatrix,
    SpeculationEngine,
    StructuredOutputProcessor,
)


def main():
    print("=" * 60)
    print("TorchBridge — Speculative Decoding Cross-Backend Demo")
    print("=" * 60)

    # ── 1. Compatibility Matrix ──────────────────────────────────
    print("\n1. Compatibility Matrix")
    print("-" * 40)

    combos = [
        ("NVIDIA Hopper (H100)", HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
        ("NVIDIA Ampere (A100)", HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
        ("AMD CDNA3 (MI300X)", HardwareBackend.AMD, AMDArchitecture.CDNA3),
        ("Trainium TRN2", HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
        ("TPU v7", HardwareBackend.TPU, TPUVersion.V7),
        ("CPU", HardwareBackend.CPU, None),
    ]

    for name, backend, arch in combos:
        optimal = SpeculationCompatibilityMatrix.get_optimal_method(backend, arch)
        methods = SpeculationCompatibilityMatrix.get_supported_methods(backend, arch)
        method_str = ", ".join(m.value for m in methods)
        print(f"  {name:<25} optimal={optimal.value:<14} [{method_str}]")

    # ── 2. Engine Auto-Selection ─────────────────────────────────
    print("\n2. Engine Auto-Selection")
    print("-" * 40)

    engine = SpeculationEngine(
        backend=HardwareBackend.CUDA,
        architecture=NVIDIAArchitecture.AMPERE,
    )
    print("  Backend:  CUDA / Ampere")
    print(f"  Method:   {engine.method.value}")
    print(f"  Kwargs:   {engine.get_generation_kwargs()}")

    # ── 3. Batch Size Gating ─────────────────────────────────────
    print("\n3. Batch Size Gating")
    print("-" * 40)

    for bs in [1, 4, 8, 16]:
        speculate = engine.should_speculate(batch_size=bs)
        print(f"  batch_size={bs:<4} should_speculate={speculate}")

    # ── 4. Phase Detection ───────────────────────────────────────
    print("\n4. Phase Detection")
    print("-" * 40)

    scenarios = [
        ("Start of generation", 512, 0),
        ("Early decode", 512, 20),
        ("Mid decode", 512, 256),
        ("Long generation", 512, 1024),
    ]

    for label, prompt, gen in scenarios:
        phase = PhaseDetector.detect_phase(prompt, gen)
        profile = PhaseDetector.get_hardware_profile(phase, HardwareBackend.CUDA)
        bound = "compute" if profile.is_compute_bound else "memory"
        print(f"  {label:<22} phase={phase.value:<8} bound={bound}")

    # ── 5. Structured Output ─────────────────────────────────────
    print("\n5. Structured Output Validation")
    print("-" * 40)

    schema = {"type": "object", "required": ["name", "age"]}
    proc = StructuredOutputProcessor(
        format=OutputFormat.JSON_SCHEMA, schema=schema
    )
    print(f"  Format:   {proc.format.value}")
    print(f"  xgrammar: {'available' if proc.is_available() else 'not installed'}")

    test_outputs = [
        ('{"name": "Alice", "age": 30}', True),
        ('{"name": "Bob"}', False),  # missing "age"
        ("not json", False),
    ]
    for text, expected in test_outputs:
        valid = proc.validate_output(text)
        status = "PASS" if valid == expected else "FAIL"
        print(f"  [{status}] validate({text!r}) = {valid}")

    # ── 6. Engine Info ───────────────────────────────────────────
    print("\n6. Engine Diagnostic Info")
    print("-" * 40)

    info = engine.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nDone.")


if __name__ == "__main__":
    main()
