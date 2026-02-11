# TorchBridge Model Examples

Cross-backend inference and training examples using modern HuggingFace models (2025-2026).

Each example demonstrates TorchBridge's hardware abstraction layer (HAL) with real pretrained models, comparing outputs across CPU, CUDA, ROCm, and TPU backends.

## Examples by Category

### LLM (Large Language Models)

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [qwen3_cross_backend.py](llm/qwen3_cross_backend.py) | Qwen/Qwen3-8B | 8B | Multilingual (140+ languages), hybrid thinking |
| [deepseek_cross_backend.py](llm/deepseek_cross_backend.py) | DeepSeek-R1 | 7B | Reasoning-optimized distillation |
| [llama4_cross_backend.py](llm/llama4_cross_backend.py) | Llama 4 | 8B | Meta's latest open model |
| [gemma3_cross_backend.py](llm/gemma3_cross_backend.py) | Gemma 3 | 12B | 128K context, multimodal |

### Vision

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [sam3_cross_backend.py](vision/sam3_cross_backend.py) | SAM 3 | - | Segment Anything |
| [dinov2_cross_backend.py](vision/dinov2_cross_backend.py) | DINOv2 | 86M | Universal ViT feature extractor |

### Multimodal

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [qwen25_vl_cross_backend.py](multimodal/qwen25_vl_cross_backend.py) | Qwen2.5-VL | 7B | Top vision-language model |

### Embedding

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [bge_m3_cross_backend.py](embedding/bge_m3_cross_backend.py) | BGE-M3 | 568M | RAG backbone, multilingual |

### Speech

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [whisper_cross_backend.py](speech/whisper_cross_backend.py) | Whisper v3 Turbo | 809M | Dominant ASR, encoder-decoder |

### Code

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [qwen25_coder_cross_backend.py](code/qwen25_coder_cross_backend.py) | Qwen2.5-Coder | 7B | IDE-optimized code generation |

### Distributed Training

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [qwen3_fsdp_training.py](distributed/qwen3_fsdp_training.py) | Qwen3-8B | 8B | FSDP multi-GPU training |

### Serving

| Example | Description |
|---------|-------------|
| [run_llm_server.py](serving/run_llm_server.py) | Production LLM inference server |

## Quick Start

```bash
# Install dependencies
pip install transformers accelerate

# Run any example
PYTHONPATH=src python examples/models/llm/qwen3_cross_backend.py --help

# Run with quantization
PYTHONPATH=src python examples/models/llm/qwen3_cross_backend.py --quantization int4

# Run benchmark
PYTHONPATH=src python examples/models/vision/dinov2_cross_backend.py --benchmark
```

## Common CLI Flags

All examples support:
- `--model MODEL_NAME` — HuggingFace model ID
- `--benchmark` — Run structured latency/throughput benchmark
- `--output-json FILE` — Save results to JSON

LLM examples additionally support:
- `--quantization {none,int8,int4}` — Quantization mode
- `--prompt TEXT` — Custom prompt
- `--max-new-tokens N` — Maximum tokens to generate
