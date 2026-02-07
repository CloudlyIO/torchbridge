# BERT on SQuAD — Cross-Backend Training & Deployment

Fine-tune BERT-base on SQuAD 2.0 using TorchBridge HAL for hardware-agnostic
training and deployment across NVIDIA, AMD, Intel, and CPU backends.

## Project Goals

1. **Train anywhere** — Same script works on CUDA, ROCm, XPU, MPS, and CPU
2. **Validate consistency** — Ensure numerical parity across backends
3. **Export portable** — ONNX export that runs on any runtime
4. **Benchmark performance** — Compare latency/throughput across hardware

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on best available backend (auto-detected)
python train.py --epochs 2 --batch-size 16

# Validate cross-backend consistency
python validate_cross_backend.py --model checkpoints/bert_squad_best.pt

# Export to ONNX
python export_model.py --model checkpoints/bert_squad_best.pt --output bert_squad.onnx

# Benchmark inference
python inference.py --model checkpoints/bert_squad_best.pt --benchmark
```

## Project Structure

```
bert_squad/
├── train.py                    # Main training script (HAL-enabled)
├── validate_cross_backend.py   # Cross-backend consistency validation
├── inference.py                # Inference with benchmarking
├── export_model.py             # ONNX/TorchScript export
├── requirements.txt            # Dependencies
├── configs/
│   └── default.yaml            # Training configuration
├── scripts/
│   ├── run_nvidia.sh           # NVIDIA-specific runner
│   ├── run_amd.sh              # AMD ROCm runner
│   └── run_all_backends.sh     # Full cross-backend validation
├── tests/
│   └── test_consistency.py     # Pytest cross-backend tests
├── checkpoints/                # Saved models
└── results/                    # Training logs and benchmarks
```

## Hardware Requirements

| Backend | Minimum | Recommended |
|---------|---------|-------------|
| NVIDIA CUDA | GTX 1080 (8GB) | A10G/L4 (24GB) |
| AMD ROCm | MI100 (32GB) | MI300X (192GB) |
| Intel XPU | Arc A770 (16GB) | Gaudi2 |
| Apple MPS | M1 (8GB) | M2 Pro (16GB) |
| CPU | 16GB RAM | 32GB RAM |

## Key Features Demonstrated

- **TorchBridge HAL** — `detect_best_backend()`, `get_manager()`, `prepare_model()`
- **Backend-agnostic training** — Single script, any hardware
- **Automatic optimization** — O2 level optimization per backend
- **Cross-backend validation** — Numerical consistency checks
- **ONNX export** — Portable model deployment
