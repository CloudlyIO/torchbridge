# TorchBridge Model Examples

Cross-backend inference and training examples using modern HuggingFace models (Feb 2026).

Each example demonstrates TorchBridge's hardware abstraction layer (HAL) with real pretrained models, comparing outputs across CPU, CUDA, ROCm, Trainium, and TPU backends.

## Examples by Category

### LLM (Large Language Models)

| Example | Model | Architecture | Params | Key Feature |
|---------|-------|-------------|--------|-------------|
| [qwen3_cross_backend.py](llm/qwen3_cross_backend.py) | Qwen/Qwen3-8B | Dense | 8B | Multilingual (140+ languages), hybrid thinking |
| [deepseek_cross_backend.py](llm/deepseek_cross_backend.py) | DeepSeek-R1 / V3 | Dense / MoE | 7B-685B | Reasoning-optimized, V3 flagship MoE |
| [moe_cross_backend.py](llm/moe_cross_backend.py) | Qwen/Qwen3-30B-A3B | MoE | 30B (3B active) | MoE expert routing, load balancing |
| [llama4_cross_backend.py](llm/llama4_cross_backend.py) | Llama 4 Scout | MoE | 17B (16 experts) | Meta's latest MoE model |
| [gemma3_cross_backend.py](llm/gemma3_cross_backend.py) | Gemma 3 | Dense | 12B | 128K context, multimodal |

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

### Serving

| Example | Description |
|---------|-------------|
| [run_llm_server.py](serving/run_llm_server.py) | Production LLM inference server |

## Model Compatibility Matrix

| Model | Type | Min VRAM (FP16) | Min VRAM (INT4) | Backends | Best For |
|-------|------|----------------|----------------|----------|----------|
| Qwen3-8B | Dense LLM | 16GB | 5GB | All | General-purpose, multilingual |
| Qwen3-30B-A3B | MoE LLM | 60GB | 15GB | CUDA, ROCm, CPU | MoE evaluation, quality-per-FLOP |
| DeepSeek-R1-Distill-7B | Dense LLM | 14GB | 4GB | All | Reasoning, chain-of-thought |
| DeepSeek-V3-0324 | MoE LLM | 1.3TB | 340GB | Multi-GPU | Flagship MoE, research |
| Llama 4 Scout | MoE LLM | 34GB | 9GB | All | Meta ecosystem |
| Gemma 3-12B | Dense LLM | 24GB | 7GB | All | Long context (128K) |
| Qwen3-VL-7B | VLM | 14GB | 4GB | All | Vision-language tasks |
| BGE-M3 | Embedding | 1.2GB | - | All | RAG, multilingual retrieval |
| GTE-Qwen3 | Embedding | 3GB | - | All | Semantic search, retrieval |
| Whisper v3 Turbo | Speech | 1.6GB | - | All | ASR, transcription |
| Qwen3-Coder-7B | Code LLM | 14GB | 4GB | All | Code generation, IDE |
| SAM 3 | Vision | ~2GB | - | All | Image segmentation |
| DINOv2 | Vision | <1GB | - | All | Feature extraction |

## Which Example Should I Start With?

- **First time?** Start with `llm/qwen3_cross_backend.py` — it's the most complete example with multilingual demo, memory estimation, and benchmarking.
- **RAG/search?** Use `embedding/bge_m3_cross_backend.py` or `embedding/gte_qwen3_cross_backend.py`.
- **MoE models?** Use `llm/moe_cross_backend.py` — shows expert routing and load balancing.
- **Code generation?** Use `code/qwen3_coder_cross_backend.py`.
- **Production serving?** Use `serving/run_llm_server.py`.
- **Limited VRAM?** Use `--quantization int4` flag on any LLM example.

## Quick Start

```bash
# Install dependencies
pip install transformers accelerate

# Run any example
PYTHONPATH=src python examples/models/llm/qwen3_cross_backend.py --help

# Run with quantization
PYTHONPATH=src python examples/models/llm/qwen3_cross_backend.py --quantization int4

# Run MoE example
PYTHONPATH=src python examples/models/llm/moe_cross_backend.py --analyze-experts

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
