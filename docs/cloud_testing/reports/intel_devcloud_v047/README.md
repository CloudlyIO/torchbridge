# Intel DevCloud Validation for v0.4.7

## Status

**Local Validation**: COMPLETE (56/56 tests passing)
**Cloud Validation**: PENDING (requires Intel DevCloud access)

## Local Validation Results

The Intel XPU backend has been validated locally with CPU fallback:

| Component | Result |
|-----------|--------|
| Tests | 56 passed, 5 skipped (XPU-specific) |
| Demo | All features demonstrated |
| Benchmarks | Skipped (no XPU hardware) |

## How to Complete Cloud Validation

### Step 1: Access Intel DevCloud

1. Go to [Intel Developer Cloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)
2. Sign up for a free account at `console.cloud.intel.com`
3. Access JupyterLab or SSH into a compute node

### Step 2: Choose GPU Instance

Intel DevCloud offers several GPU options:

| GPU Type | Use Case | IPEX Support |
|----------|----------|--------------|
| Intel Data Center GPU Max Series (PVC) | Data center AI | Full |
| Intel Arc A-Series | Consumer/Development | Full |
| Intel Flex Series | Streaming/Inference | Full |

From the console, select a GPU-enabled instance (e.g., "PyTorch on Intel GPUs" notebook).

### Step 3: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/shahrahman-fb/shahmod.git
cd shahmod

# Install KernelPyTorch
pip install -e .

# Verify IPEX is available
python -c "import intel_extension_for_pytorch as ipex; print(f'IPEX: {ipex.__version__}')"
python -c "import torch; print(f'XPU: {torch.xpu.is_available()}')"
```

### Step 4: Run Validation

```bash
# Run full validation
python docs/cloud_testing/scripts/intel_devcloud_test.py --all

# Or run individual components
python docs/cloud_testing/scripts/intel_devcloud_test.py --tests-only
python docs/cloud_testing/scripts/intel_devcloud_test.py --demo-only
python docs/cloud_testing/scripts/intel_devcloud_test.py --benchmarks-only
```

### Step 5: Review Results

Results will be saved to:
- `docs/cloud_testing/reports/intel_devcloud_v047/SUMMARY.md`
- `docs/cloud_testing/reports/intel_devcloud_v047/results.json`

## Expected Results on Intel XPU

When running on actual Intel XPU hardware:

### Tests
- 56+ tests passing (including XPU-specific tests)
- 0 skipped (all XPU features available)

### Demo
- Device detection shows XPU available
- Memory statistics populated
- Model moves to XPU device
- IPEX optimization applied

### Benchmarks
- Matrix multiplication: Expected 10-50+ TFLOPS (depends on GPU)
- Model inference: Expected 100K+ samples/sec

## Troubleshooting

### IPEX Not Installed
```bash
pip install intel-extension-for-pytorch
```

### XPU Not Available
- Ensure you're on a GPU-enabled instance
- Check driver installation: `xpu-smi`
- Verify PyTorch XPU support: `python -c "import torch; print(torch.xpu.is_available())"`

### Memory Errors
- Reduce batch sizes in benchmarks
- Clear cache: `torch.xpu.empty_cache()`

## Resources

- [Intel Extension for PyTorch Documentation](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)
- [PyTorch Intel GPU Support](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/)
- [Intel DevCloud Overview](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)
