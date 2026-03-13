# TorchBridge Model Examples

Cross-backend inference and training examples using modern HuggingFace models (Feb 2026).

Each example demonstrates TorchBridge's cross-backend validation and configuration intelligence with real pretrained models, comparing outputs across CPU, CUDA, ROCm, Trainium, and TPU backends.

## Examples by Category

### LLM (Large Language Models)

| Example | Model | Architecture | Params | Key Feature |
|---------|-------|-------------|--------|-------------|
| [qwen3_cross_backend.py](llm/qwen3_cross_backend.py) | Qwen/Qwen3-8B | Dense | 8B | Multilingual (140+ languages), hybrid thinking |
| [llama4_cross_backend.py](llm/llama4_cross_backend.py) | Llama 4 Scout | MoE | 17B (16 experts) | Meta's latest MoE model |
| [gemma3_cross_backend.py](llm/gemma3_cross_backend.py) | Gemma 3 | Dense | 12B | 128K context, multimodal |

### Advanced LLM Features

| Example | Model | Key Feature |
|---------|-------|-------------|
| [qwen3_quantized_cross_backend.py](llm/qwen3_quantized_cross_backend.py) | Qwen3 | Quantization (INT4/INT8) cross-backend |
| [adapter_cross_backend.py](llm/adapter_cross_backend.py) | Any | LoRA/DoRA adapter validation |
| [attention_cross_backend.py](llm/attention_cross_backend.py) | Any | Attention kernel dispatch |
| [speculative_cross_backend.py](llm/speculative_cross_backend.py) | Any | Speculative decoding methods |

### Vision

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [sam3_cross_backend.py](vision/sam3_cross_backend.py) | SAM 3 | - | Segment Anything |
| [dinov2_cross_backend.py](vision/dinov2_cross_backend.py) | DINOv2 | 86M | Universal ViT feature extractor |

### Multimodal

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [qwen3_vl_cross_backend.py](multimodal/qwen3_vl_cross_backend.py) | Qwen3-VL | 7B | Latest VLM, visual grounding, document understanding |

### Embedding

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [bge_m3_cross_backend.py](embedding/bge_m3_cross_backend.py) | BGE-M3 | 568M | RAG backbone, multilingual, 1024-dim |
| [gte_qwen3_cross_backend.py](embedding/gte_qwen3_cross_backend.py) | GTE-Qwen3 | ~1.5B | Latest GTE with Qwen3 backbone, semantic search |

### Speech

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [whisper_cross_backend.py](speech/whisper_cross_backend.py) | Whisper v3 Turbo | 809M | Dominant ASR, encoder-decoder |

### Code

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [qwen3_coder_cross_backend.py](code/qwen3_coder_cross_backend.py) | Qwen3-Coder | 7B | Latest code model, agentic coding |

### Distributed Training

| Example | Model | Params | Key Feature |
|---------|-------|--------|-------------|
| [qwen3_fsdp_training.py](distributed/qwen3_fsdp_training.py) | Qwen3-8B | 8B | FSDP multi-GPU training |
| [distributed_cross_backend.py](distributed/distributed_cross_backend.py) | Any | Distributed config generation |
| [checkpoint_cross_backend.py](distributed/checkpoint_cross_backend.py) | Any | Cross-backend checkpointing |

## Which Example Should I Start With?

- **First time?** Start with `llm/qwen3_cross_backend.py` — it's the most complete example with multilingual demo, memory estimation, and benchmarking.
- **RAG/search?** Use `embedding/bge_m3_cross_backend.py` or `embedding/gte_qwen3_cross_backend.py`.
- **Code generation?** Use `code/qwen3_coder_cross_backend.py`.
- **Limited VRAM?** Use `--quantization int4` flag on any LLM example.

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
