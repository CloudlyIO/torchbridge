# Hardware Support

This document lists every backend and architecture with tolerance entries in the
`_FAMILY_TOLERANCE_TABLE`. It is auto-derived from `tolerances.py` and is the
first thing to check before opening a new hardware PR — "does this hardware
already have entries?"

To add a new row, follow the instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

## Tolerance Database Coverage

| Backend | Architecture | float32 | float16 | bfloat16 | Source |
|---------|-------------|---------|---------|----------|--------|
| CUDA | Hopper (H100) | ✅ measured | ✅ measured | ✅ measured | cloud v0.5.31 |
| CUDA | Blackwell DC (B100/B200, sm_100) | ✅ derived | — | ✅ derived | gen scaling v0.5.69 |
| CUDA | Blackwell Consumer (RTX 5090, cc12.0) | ✅ derived | ✅ derived | ✅ derived | gen scaling v0.5.69 |
| CUDA | Blackwell Ultra (B300, sm_103) — placeholder | — | — | — | enum only v0.5.100; H2 2026 |
| CUDA | Rubin (R200/VR200, sm_rubin) — placeholder | — | — | — | enum only v0.5.100; HPC 2026 |
| ROCm | CDNA3 (MI300X) | ✅ measured | ✅ measured | ✅ measured | cloud v0.5.31 |
| ROCm | CDNA4 (MI350X, gfx950) — supports MXFP8/MXFP4 | ✅ derived | ✅ derived | ✅ derived | gen scaling v0.5.69 |
| ROCm | RDNA4 (RX 9070 XT, gfx1201) | ✅ derived | ✅ derived | ✅ derived | silicon v0.5.100 |
| MPS | Apple Silicon | ✅ measured | ✅ measured | ✅ measured | cloud v0.5.31 |
| XLA | TPU v5e | ✅ measured | — | ✅ measured | cloud v0.5.31 |
| XLA | TPU v7 Ironwood | ✅ derived | — | ✅ derived | gen scaling v0.5.69 |
| CPU | x86 / ARM (reference) | ✅ measured | ✅ measured | ✅ measured | cloud v0.5.31 |
| Trainium | Trn1 (NeuronCore v2) | ✅ measured | — | ✅ measured | cloud v0.5.31 |
| Trainium | Trn2 (trainium2, NeuronCore v3) | ✅ derived | — | ✅ derived | silicon v0.5.100 |
| Trainium | Trn3 (trainium_trn3, NeuronCore v4) | ✅ derived | — | ✅ derived | gen scaling v0.5.69 |

> **Legend**
> - ✅ **measured** — worst-case max-diff observed on real hardware during cloud validation (Qwen3-0.6B, v0.5.31).
> - ✅ **derived** — scaled from measured entries using the accumulated-error model (see `tolerance_db.py` module docstring).
> - **—** dtype not exposed by this backend (e.g. XLA has no float16; Trainium/Neuron have no float16).

## Model Family Coverage

All backends listed above cover the following model families:

| Family | Parameter Range | Example Models |
|--------|----------------|----------------|
| `decoder-small` | < 2 B | Qwen3-0.6B, Llama-3.2-1B, SmolLM-2 |
| `decoder-medium` | 2 B – 20 B | Llama-3.1-8B, Qwen3-7B, Mistral-7B |
| `decoder-large` | > 20 B | Llama-3.1-70B, Qwen3-72B |
| `encoder` | any | BERT, RoBERTa, DeBERTa |
| `vision-language` | any | CLIP, LLaVA, InternVL |
| `qwen3_5` | 27 B dense | Qwen3.5-27B, Qwen3.6-27B |
| `gemma4` | 26 B total / 4 B active | Gemma 4 26B-A4B |
| `nemotron3_nano` | 30 B total / 3 B active | Nemotron 3 Nano 30B-A3B |
| `deepseek_v4` | 284 B – 1.6 T total / 13 – 49 B active | DeepSeek V4 Flash, DeepSeek V4 Pro |
| `nemotron3_ultra` | 550 B total / 55 B active | Nemotron Ultra 550B |
| `tencent_hy3` | 295 B total / 21 B active | Tencent Hunyuan 3 |
| `minimax_m3` | ~428 B total / 23 B active | MiniMax M3 |
| `glm_5_2` | 744 B total / 40 B active | GLM 5.2 |

## Adding a New Row

1. Add your measured `ToleranceEntry` values to `_FAMILY_TOLERANCE_TABLE` in `tolerance_db.py`.
2. Add a corresponding row to the table above.
3. Run `python -m pytest tests/unit/test_tolerance_db_coverage.py -q` — all assertions must pass.
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guide.