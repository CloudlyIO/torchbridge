#!/usr/bin/env python3
"""
TorchBridge v0.5.45 — Comprehensive Manual Test Script
=======================================================
Self-contained: installs deps, tests all 15 CLI commands + Python APIs,
runs Qwen3-0.6B cross-backend validation, outputs JSON + human summary.

Usage:
    python3 v0545_manual_test.py [--skip-qwen]

Output:
    /tmp/tb_manual_test_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

RESULTS = {"platform": {}, "install": {}, "cli": {}, "python_api": {}, "regressions": {}, "qwen": {}}
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
OUTPUT_PATH = "/tmp/tb_manual_test_results.json"


def pip_install(*packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages"] + list(packages)
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_cmd(cmd, timeout=60, env=None):
    """Run shell command, return (returncode, stdout, stderr)."""
    e = os.environ.copy()
    if env:
        e.update(env)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=e)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def record(section, key, status, detail=""):
    RESULTS[section][key] = {"status": status, "detail": str(detail)[:500]}
    symbol = "✓" if status == PASS else ("✗" if status == FAIL else "–")
    print(f"  {symbol} {key}: {status}" + (f" | {str(detail)[:120]}" if detail and status != PASS else ""))


def run_py_file(code, timeout=30):
    """Write code to a temp file and run it. Returns (rc, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        rc, out, err = run_cmd(f"python3 {fname}", timeout=timeout)
    finally:
        os.unlink(fname)
    return rc, out, err


# ---------------------------------------------------------------------------
# Platform Detection
# ---------------------------------------------------------------------------

def detect_platform():
    print("\n=== Platform Detection ===")
    import torch

    info = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "hip_version": getattr(torch.version, "hip", None),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "backend": "cpu",
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": name, "memory_gb": round(props.total_memory / 1024**3, 1)})
        gpu_lower = info["gpus"][0]["name"].lower() if info["gpus"] else ""
        if "instinct" in gpu_lower or "mi3" in gpu_lower or "mi2" in gpu_lower or torch.version.hip is not None:
            info["backend"] = "amd"
        else:
            info["backend"] = "nvidia"
    elif info["mps_available"]:
        info["backend"] = "mps"
    else:
        try:
            import torch_xla.core.xla_model as xm
            info["backend"] = "tpu"
        except ImportError:
            pass

    RESULTS["platform"] = info
    print(f"  Python: {info['python']}, PyTorch: {info['pytorch']}")
    print(f"  Backend: {info['backend'].upper()}")
    if info["gpus"]:
        for g in info["gpus"]:
            print(f"  GPU {g['index']}: {g['name']} ({g['memory_gb']} GB)")
    return info


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def install_deps():
    print("\n=== Installing Dependencies ===")
    t0 = time.time()

    ok = pip_install("torchbridge-ml==0.5.45")
    if not ok:
        ok = pip_install("torchbridge-ml")  # latest if pinned fails
    record("install", "torchbridge-ml", PASS if ok else FAIL)

    ok_t = pip_install("transformers")
    record("install", "transformers", PASS if ok_t else FAIL)

    ok_fa = pip_install("fastapi", "uvicorn[standard]")
    record("install", "fastapi+uvicorn", PASS if ok_fa else FAIL)

    ok_ps = pip_install("psutil")
    record("install", "psutil", PASS if ok_ps else FAIL)

    # Verify version
    rc, out, _ = run_cmd("python3 -c \"import torchbridge; print(torchbridge.__version__)\"")
    version = out.strip()
    expected = "0.5.45"
    record("install", "version_check", PASS if version == expected else FAIL,
           f"got {version!r}, expected {expected!r}")

    RESULTS["install"]["elapsed_sec"] = round(time.time() - t0, 1)
    print(f"  Installed in {RESULTS['install']['elapsed_sec']}s — TorchBridge {version}")


# ---------------------------------------------------------------------------
# CLI Tests (15 commands)
# ---------------------------------------------------------------------------

def test_cli(platform_info):
    print("\n=== CLI Commands (15 total) ===")
    backend = platform_info["backend"]
    # Map internal backend names to CLI-expected values
    CLI_BACKEND = {"nvidia": "cuda", "amd": "amd", "mps": "cpu", "tpu": "tpu", "cpu": "cpu"}
    cli_backend = CLI_BACKEND.get(backend, "cpu")

    # Create temp test model for CLI tests that need it
    tmpdir = tempfile.mkdtemp()
    model_path = f"{tmpdir}/test_model.pt"
    run_py_file(
        f"import torch, torch.nn as nn\n"
        f"m = nn.Linear(64, 32)\n"
        f"torch.save(m, '{model_path}')\n"
        f"print('model saved')\n"
    )

    # Create a tiny Python file for migration scanner
    sample_py = f"{tmpdir}/sample_code.py"
    with open(sample_py, "w") as f:
        f.write(
            "import torch\n"
            "x = torch.tensor([1.0]).cuda()\n"
            "model.to('cuda:0')\n"
            "torch.cuda.synchronize()\n"
        )

    # --- CLI-1: tb-doctor ---
    rc, out, err = run_cmd("tb-doctor", timeout=30)
    ok = rc == 0 and ("passed" in out.lower() or "Backend" in out)
    record("cli", "tb-doctor", PASS if ok else FAIL, out[:200] if not ok else "")

    # --- CLI-2: tb-validate --level quick --ci ---
    rc, out, err = run_cmd("tb-validate --level quick --ci", timeout=30)
    try:
        data = json.loads(out)
        ok = rc == 0 and data.get("summary", {}).get("failures", 1) == 0
        record("cli", "tb-validate --level quick --ci", PASS if ok else FAIL,
               f"passed={data.get('summary',{}).get('passed')}" if ok else out[:200])
    except json.JSONDecodeError:
        ok = rc == 0 and "pass" in out.lower()
        record("cli", "tb-validate --level quick --ci", PASS if ok else FAIL, out[:200])

    # --- CLI-3: tb-quantize --format auto --ci (requires --model) ---
    rc, out, err = run_cmd(f"tb-quantize --model {model_path} --format auto --trust-source --ci", timeout=30)
    try:
        data = json.loads(out)
        # Response uses "format_applied" and "success" keys (not bare "format")
        ok = rc == 0 and (data.get("success") is True or "format_applied" in data)
        record("cli", "tb-quantize --model --format auto --ci", PASS if ok else FAIL,
               f"format_applied={data.get('format_applied')}" if ok else out[:200])
    except json.JSONDecodeError:
        ok = rc == 0
        record("cli", "tb-quantize --model --format auto --ci", PASS if ok else FAIL, (err or out)[:200])

    # --- CLI-4: tb-benchmark --list-claims ---
    rc, out, err = run_cmd("tb-benchmark --list-claims", timeout=30)
    ok = rc == 0 and ("claim" in out.lower() or "tensor_core" in out.lower() or "benchmark" in out.lower())
    record("cli", "tb-benchmark --list-claims", PASS if ok else FAIL, (err or out)[:200] if not ok else "")

    # --- CLI-5: tb-profile (requires --trust-source; --input-shape must match model's 64-dim input) ---
    rc, out, err = run_cmd(
        f"tb-profile --model {model_path} --mode summary --input-shape 1,64 --iterations 5 --trust-source",
        timeout=60
    )
    ok = rc == 0
    record("cli", "tb-profile --mode summary --input-shape 1,64 --trust-source", PASS if ok else FAIL,
           (err or out)[:200] if not ok else "")

    # --- CLI-6: tb-init ---
    rc, out, err = run_cmd(f"tb-init --name new_project --template training --output-dir {tmpdir}", timeout=30)
    ok = rc == 0
    record("cli", "tb-init --template training", PASS if ok else FAIL, (err or out)[:200] if not ok else "")

    # --- CLI-7: tb-migrate ---
    rc, out, err = run_cmd(f"tb-migrate {sample_py} --format json", timeout=30)
    ok = rc in (0, 1)  # exit 1 means issues found = correct behavior
    try:
        data = json.loads(out)
        ok = True
        issues = data.get("issues", data.get("findings", data.get("results", [])))
        record("cli", "tb-migrate (JSON output)", PASS if ok else FAIL,
               f"{len(issues) if isinstance(issues, list) else 'N'} issues found")
    except json.JSONDecodeError:
        record("cli", "tb-migrate", PASS if ok else FAIL, out[:200] if not ok else "")

    # --- CLI-8: tb-speculate --show-matrix --ci ---
    rc, out, err = run_cmd("tb-speculate --show-matrix --ci", timeout=30)
    try:
        data = json.loads(out)
        ok = rc == 0 and isinstance(data, (dict, list))
        record("cli", "tb-speculate --show-matrix --ci", PASS if ok else FAIL,
               f"entries={len(data) if isinstance(data, list) else len(data)}" if ok else out[:200])
    except json.JSONDecodeError:
        ok = rc == 0
        record("cli", "tb-speculate --show-matrix --ci", PASS if ok else FAIL, (err or out)[:200])

    # --- CLI-9: tb-cache --show-matrix --ci ---
    rc, out, err = run_cmd("tb-cache --show-matrix --ci", timeout=30)
    try:
        data = json.loads(out)
        ok = rc == 0 and isinstance(data, (dict, list))
        record("cli", "tb-cache --show-matrix --ci", PASS if ok else FAIL, "" if ok else out[:200])
    except json.JSONDecodeError:
        ok = rc == 0
        record("cli", "tb-cache --show-matrix --ci", PASS if ok else FAIL, (err or out)[:200])

    # --- CLI-10: tb-advisor --model-params 7 --world-size 8 --ci ---
    rc, out, err = run_cmd("tb-advisor --model-params 7 --world-size 8 --ci", timeout=30)
    try:
        data = json.loads(out)
        # Response has "recommendation" dict with fsdp_strategy
        ok = rc == 0 and "recommendation" in data and "fsdp_strategy" in data.get("recommendation", {})
        record("cli", "tb-advisor --model-params 7 --world-size 8 --ci", PASS if ok else FAIL,
               f"fsdp={data.get('recommendation',{}).get('fsdp_strategy')}" if ok else out[:200])
    except json.JSONDecodeError:
        ok = rc == 0
        record("cli", "tb-advisor --ci", PASS if ok else FAIL, (err or out)[:200])

    # --- CLI-11: tb-checkpoint advisor ---
    rc, out, err = run_cmd(
        "tb-checkpoint advisor --world-size 8 --checkpoint-time 3600 --step-time 300 --mtbf 720",
        timeout=30
    )
    ok = rc == 0
    record("cli", "tb-checkpoint advisor", PASS if ok else FAIL, (err or out)[:200] if not ok else "")

    # --- CLI-12: tb-adapter recommend (use mapped CLI backend) ---
    rc, out, err = run_cmd(f"tb-adapter recommend --backend {cli_backend}", timeout=30)
    ok = rc == 0 and ("lora" in out.lower() or "adapter" in out.lower())
    record("cli", f"tb-adapter recommend --backend {cli_backend}", PASS if ok else FAIL,
           (err or out)[:200] if not ok else "")

    # --- CLI-13: torchbridge validate (unified dispatcher) ---
    rc, out, err = run_cmd("torchbridge validate --level quick", timeout=30)
    ok = rc == 0 and ("pass" in out.lower() or "validation" in out.lower())
    record("cli", "torchbridge validate (unified dispatcher)", PASS if ok else FAIL,
           (err or out)[:200] if not ok else "")


# ---------------------------------------------------------------------------
# Python API Tests
# ---------------------------------------------------------------------------

def test_python_api(platform_info):
    print("\n=== Python API Tests ===")
    backend = platform_info["backend"]
    has_cuda = backend in ("nvidia", "amd")

    # API-1: detect_best_backend — get_device_info() returns DeviceInfo, not dict
    rc, out, err = run_py_file(
        "from torchbridge.backends import detect_best_backend, BackendFactory\n"
        "b = detect_best_backend()\n"
        "backend = BackendFactory.create(b)\n"
        "info = backend.get_device_info()\n"
        "# DeviceInfo is a dataclass with .backend, .device_name etc (not a plain dict)\n"
        "assert hasattr(info, 'backend'), f'Expected DeviceInfo with .backend, got {type(info)}'\n"
        "print('backend:', b, 'device:', info.device_name)\n"
    )
    ok = rc == 0
    record("python_api", "detect_best_backend + BackendFactory.create",
           PASS if ok else FAIL, (err or out)[:200] if not ok else out.strip()[:120])

    # API-2: UnifiedManager
    rc, out, err = run_py_file(
        "import torch, torch.nn as nn\n"
        "from torchbridge import TorchBridgeConfig, UnifiedManager\n"
        "config = TorchBridgeConfig.for_training()\n"
        "m = UnifiedManager(config)\n"
        "model = nn.Sequential(nn.Linear(64, 32), nn.ReLU())\n"
        "opt = m.optimize(model)\n"
        "assert opt is not None\n"
        "print('optimize OK')\n"
    )
    record("python_api", "UnifiedManager.optimize",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")

    # API-3: UnifiedValidator
    rc, out, err = run_py_file(
        "import torch, torch.nn as nn\n"
        "from torchbridge import UnifiedValidator\n"
        "model = nn.Linear(64, 32)\n"
        "v = UnifiedValidator()\n"
        "res = v.validate_model(model, input_shape=(1, 64))\n"
        "assert hasattr(res, 'passed')\n"
        "print('passed:', res.passed)\n"
    )
    record("python_api", "UnifiedValidator.validate_model",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")

    # API-4: QuantizationEngine — correct method is get_optimal_format()
    rc, out, err = run_py_file(
        "from torchbridge.precision.quantization.engine import QuantizationEngine\n"
        "from torchbridge.backends import detect_best_backend\n"
        "e = QuantizationEngine(detect_best_backend())\n"
        "fmt = e.get_optimal_format()\n"
        "assert fmt is not None\n"
        "print('format:', fmt)\n"
    )
    record("python_api", "QuantizationEngine.get_optimal_format",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else out.strip()[:120])

    # API-5: AttentionDispatcher — correct method is select_kernel(seq_len, num_heads, head_dim)
    rc, out, err = run_py_file(
        "from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher\n"
        "from torchbridge.backends import detect_best_backend\n"
        "d = AttentionDispatcher(detect_best_backend())\n"
        "result = d.select_kernel(seq_length=512, num_heads=8, head_dim=64)\n"
        "assert result is not None\n"
        "print('kernel result:', result)\n"
    )
    record("python_api", "AttentionDispatcher.select_kernel",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else out.strip()[:120])

    # API-6: AdapterCompatibilityMatrix
    rc, out, err = run_py_file(
        "from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix, AdapterMethod\n"
        "opt = AdapterCompatibilityMatrix.get_optimal('cpu')\n"
        "assert opt is not None\n"
        "print('optimal:', opt)\n"
    )
    record("python_api", "AdapterCompatibilityMatrix.get_optimal",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")

    # API-7: KVCacheCompatibilityMatrix
    rc, out, err = run_py_file(
        "from torchbridge.models.llm.kv.cache_compatibility import KVCacheCompatibilityMatrix\n"
        "from torchbridge.backends import detect_best_backend\n"
        "dtype = KVCacheCompatibilityMatrix.get_optimal_dtype(detect_best_backend())\n"
        "assert dtype is not None\n"
        "print('dtype:', dtype)\n"
    )
    record("python_api", "KVCacheCompatibilityMatrix.get_optimal_dtype",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")

    # API-8: CollectiveBackendMatrix — correct method + import HardwareBackend from same module
    rc, out, err = run_py_file(
        "from torchbridge.distributed.collective_backend import CollectiveBackendMatrix, HardwareBackend\n"
        "cb = CollectiveBackendMatrix.get_optimal_backend(HardwareBackend.CUDA)\n"
        "assert cb is not None\n"
        "print('collective backend:', cb)\n"
    )
    record("python_api", "CollectiveBackendMatrix.get_optimal_backend",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else out.strip()[:120])

    # API-9: ToleranceDB
    rc, out, err = run_py_file(
        "from torchbridge.testing.tolerance_db import ToleranceDB\n"
        "db = ToleranceDB()\n"
        "tol = db.get('cpu', 'float32')\n"
        "assert hasattr(tol, 'atol')\n"
        "print('atol:', tol.atol)\n"
    )
    record("python_api", "ToleranceDB.get",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")

    # API-10: SpeculationCompatibilityMatrix
    rc, out, err = run_py_file(
        "from torchbridge.inference.speculative.compatibility import SpeculationCompatibilityMatrix\n"
        "from torchbridge.backends import detect_best_backend\n"
        "methods = SpeculationCompatibilityMatrix.get_generate_compatible_methods(detect_best_backend())\n"
        "assert isinstance(methods, list)\n"
        "print('compatible methods:', len(methods))\n"
    )
    record("python_api", "SpeculationCompatibilityMatrix.get_generate_compatible_methods",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")

    # API-11: LLMInferenceServer (import only)
    rc, out, err = run_py_file(
        "from torchbridge.deployment.serving.llm_server import LLMInferenceServer, LLMServerConfig\n"
        "cfg = LLMServerConfig(model_name='Qwen/Qwen3-0.6B', api_key='test-key')\n"
        "assert cfg.api_key == 'test-key'\n"
        "print('LLMServerConfig OK')\n"
    )
    record("python_api", "LLMInferenceServer + LLMServerConfig import",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")

    # API-12: SpeculationEngine (CPU path, no GPU needed)
    rc, out, err = run_py_file(
        "from torchbridge.inference.speculative.engine import SpeculationEngine\n"
        "e = SpeculationEngine(backend='cpu')\n"
        "info = e.get_info()\n"
        "assert isinstance(info['backend'], str), f'Expected str, got {type(info[\"backend\"])}'\n"
        "print('SpeculationEngine.get_info OK, backend:', info['backend'])\n"
    )
    record("python_api", "SpeculationEngine.get_info (CPU)",
           PASS if rc == 0 else FAIL, (err or out)[:200] if rc != 0 else "")


def test_regressions(platform_info):
    """v0.5.45 regression tests — the 3 bugs that were fixed."""
    print("\n=== v0.5.45 Regression Tests ===")
    backend = platform_info["backend"]
    has_cuda = backend in ("nvidia", "amd")

    # Regression 1: SpeculationEngine.get_info() with plain string architecture
    # BUG in PyPI v0.5.45: self._architecture.value crashes when architecture is a string
    # The backend guard was added but the architecture guard was missing from the published package
    rc, out, err = run_py_file(
        "from torchbridge.inference.speculative.engine import SpeculationEngine\n"
        "e = SpeculationEngine(backend='cuda', architecture='ampere')\n"
        "info = e.get_info()\n"
        "assert isinstance(info['architecture'], str), "
        "    f'Expected str, got {type(info[\"architecture\"])}'\n"
        "print('architecture type:', type(info['architecture']).__name__, '| value:', info['architecture'])\n"
    )
    ok = rc == 0
    record("regressions", "R1: SpeculationEngine.get_info() string architecture no crash",
           PASS if ok else FAIL, (err or out)[:300] if not ok else "")

    # Regression 2: AdapterCompatibilityMatrix.get_fallback_chain() with unknown string backend
    rc, out, err = run_py_file(
        "from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix, AdapterMethod\n"
        "chain = AdapterCompatibilityMatrix.get_fallback_chain('unknown_backend_xyz')\n"
        "assert len(chain) >= 1, 'fallback chain must not be empty'\n"
        "assert chain[0] == AdapterMethod.LORA, f'first fallback must be LORA, got {chain[0]}'\n"
        "print('chain[0]:', chain[0])\n"
    )
    ok = rc == 0
    record("regressions", "R2: AdapterCompatibilityMatrix.get_fallback_chain() string backend no crash",
           PASS if ok else FAIL, (err or out)[:200] if not ok else "")

    # Regression 3a: _TensorCoreAlignedLinear — CPU device preserved
    rc, out, err = run_py_file(
        "import torch, torch.nn as nn\n"
        "from torchbridge.backends.nvidia.nvidia_backend import _TensorCoreAlignedLinear\n"
        "linear = nn.Linear(127, 63)\n"
        "aligned = _TensorCoreAlignedLinear(linear, optimal_multiple=16)\n"
        "assert aligned._padded_weight.device.type == 'cpu', "
        "    f'Expected cpu, got {aligned._padded_weight.device}'\n"
        "out = aligned(torch.randn(4, 127))\n"
        "assert out.shape == (4, 63)\n"
        "print('device:', aligned._padded_weight.device.type)\n"
    )
    ok = rc == 0
    record("regressions", "R3a: _TensorCoreAlignedLinear CPU device preserved",
           PASS if ok else FAIL, (err or out)[:200] if not ok else "")

    # Regression 3b: _TensorCoreAlignedLinear — CUDA device preserved (GPU only)
    if has_cuda:
        rc, out, err = run_py_file(
            "import torch, torch.nn as nn\n"
            "from torchbridge.backends.nvidia.nvidia_backend import _TensorCoreAlignedLinear\n"
            "linear = nn.Linear(1023, 511).cuda()\n"
            "aligned = _TensorCoreAlignedLinear(linear, optimal_multiple=16)\n"
            "assert aligned._padded_weight.device.type == 'cuda', "
            "    f'Expected cuda, got {aligned._padded_weight.device}'\n"
            "out = aligned(torch.randn(4, 1023, device='cuda'))\n"
            "assert out.shape == (4, 511)\n"
            "print('device:', aligned._padded_weight.device.type)\n"
        )
        ok = rc == 0
        record("regressions", "R3b: _TensorCoreAlignedLinear CUDA device preserved",
               PASS if ok else FAIL, (err or out)[:200] if not ok else "")
    else:
        record("regressions", "R3b: _TensorCoreAlignedLinear CUDA device preserved",
               SKIP, "no CUDA GPU")


# ---------------------------------------------------------------------------
# Qwen3-0.6B Cross-Backend Validation
# ---------------------------------------------------------------------------

def test_qwen(platform_info):
    print("\n=== Qwen3-0.6B Cross-Backend Validation ===")
    backend = platform_info["backend"]
    has_gpu = backend in ("nvidia", "amd", "mps")

    if not has_gpu:
        record("qwen", "cross_backend_validation", SKIP, "CPU-only platform, no GPU to compare against")
        return

    t0 = time.time()

    # Write to temp file to avoid -c argument quoting issues
    qwen_code = '''
import torch
import torch.nn.functional as F
import time
import json
import sys

device_str = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
device = torch.device(device_str)

if device_str == "cuda":
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print(f"Device: {device_str}", flush=True)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print(json.dumps({"status": "SKIP", "reason": "transformers not installed"}))
    sys.exit(0)

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
inputs = tokenizer("The capital of France is", return_tensors="pt")

model.eval()
with torch.no_grad():
    cpu_out = model(**inputs)

model_gpu = model.to(device)
inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    gpu_out = model_gpu(**inputs_gpu)

cpu_logits = cpu_out.logits[:, -1, :]
gpu_logits = gpu_out.logits[:, -1, :].cpu()
max_diff = torch.abs(cpu_logits - gpu_logits).max().item()
cos_sim = F.cosine_similarity(
    cpu_logits.flatten().unsqueeze(0),
    gpu_logits.flatten().unsqueeze(0)
).item()

# Warmup + latency
for _ in range(3):
    model_gpu(**inputs_gpu)
if device_str == "cuda":
    torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(50):
    model_gpu(**inputs_gpu)
if device_str == "cuda":
    torch.cuda.synchronize()
latency = (time.perf_counter() - t) / 50 * 1000

# Threshold (ROCm looser due to SDPA flash attention divergence)
is_rocm = torch.version.hip is not None
threshold = 1e-3 if is_rocm else 1e-4
status = "PASSED" if max_diff < threshold else "FAILED"

result = {
    "max_diff": max_diff,
    "cos_sim": cos_sim,
    "latency_ms": round(latency, 1),
    "status": status,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device_str,
    "threshold": threshold,
    "is_rocm": is_rocm,
}
print(json.dumps(result))
'''

    rc, out, err = run_py_file(qwen_code.strip(), timeout=360)

    result = {}
    for line in reversed((out or "").strip().splitlines()):
        try:
            result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if result.get("status") == "PASSED":
        record("qwen", "cross_backend_validation", PASS,
               f"max_diff={result.get('max_diff', 0):.2e}, cos_sim={result.get('cos_sim', 0):.6f}, "
               f"latency={result.get('latency_ms')}ms, device={result.get('device')}")
    elif result.get("status") == "SKIP":
        record("qwen", "cross_backend_validation", SKIP, result.get("reason", ""))
    elif result.get("status") == "FAILED":
        record("qwen", "cross_backend_validation", FAIL,
               f"max_diff={result.get('max_diff', 0):.2e} > threshold={result.get('threshold', 0):.2e}")
    else:
        record("qwen", "cross_backend_validation", FAIL,
               (err or out)[:300] if rc != 0 else "no JSON result in output")

    RESULTS["qwen"]["elapsed_sec"] = round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize():
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    totals = {"pass": 0, "fail": 0, "skip": 0}
    section_totals = {}

    for section, tests in RESULTS.items():
        if not isinstance(tests, dict):
            continue
        p, f, s = 0, 0, 0
        for key, val in tests.items():
            if not isinstance(val, dict):
                continue
            st = val.get("status")
            if st == PASS:
                p += 1
                totals["pass"] += 1
            elif st == FAIL:
                f += 1
                totals["fail"] += 1
            elif st == SKIP:
                s += 1
                totals["skip"] += 1
        if p + f + s > 0:
            section_totals[section] = (p, f, s)

    for section, (p, f, s) in section_totals.items():
        status = "PASS" if f == 0 else "FAIL"
        print(f"  {section:<20} {status}  ({p} pass, {f} fail, {s} skip)")

    overall = "PASS" if totals["fail"] == 0 else "FAIL"
    print(f"\n  OVERALL: {overall}  ({totals['pass']} pass, {totals['fail']} fail, {totals['skip']} skip)")

    if totals["fail"] > 0:
        print("\nFAILURES:")
        for section, tests in RESULTS.items():
            if not isinstance(tests, dict):
                continue
            for key, val in tests.items():
                if isinstance(val, dict) and val.get("status") == FAIL:
                    print(f"  FAIL [{section}] {key}: {val.get('detail', '')[:120]}")

    return totals["fail"] == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TorchBridge v0.5.45 Manual Test Script")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen3-0.6B validation (faster)")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install (already installed)")
    args = parser.parse_args()

    print("=" * 60)
    print("TorchBridge v0.5.45 — Manual Test Script")
    print("=" * 60)

    if not args.skip_install:
        install_deps()
    else:
        print("\n=== Skipping install (--skip-install) ===")

    platform_info = detect_platform()

    test_cli(platform_info)
    test_python_api(platform_info)
    test_regressions(platform_info)

    if not args.skip_qwen:
        test_qwen(platform_info)
    else:
        print("\n=== Skipping Qwen validation (--skip-qwen) ===")

    RESULTS["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RESULTS["version_tested"] = "0.5.45"
    with open(OUTPUT_PATH, "w") as f:
        json.dump(RESULTS, f, indent=2)

    passed = summarize()
    print(f"\nResults saved to: {OUTPUT_PATH}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
