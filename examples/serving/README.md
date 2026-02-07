# TorchBridge Serving Examples

This directory contains examples for deploying TorchBridge models in production.

## LLM Inference Server

The `run_llm_server.py` script demonstrates how to serve large language models with TorchBridge optimizations.

### Features

- **Streaming Support**: Server-Sent Events (SSE) for real-time token streaming
- **Dynamic Batching**: Automatic batching of concurrent requests for better throughput
- **Multiple Endpoints**:
  - `/generate` - Text generation with optional streaming
  - `/chat` - Chat completion interface
  - `/tokenize` - Token counting utility
  - `/health` - Health checks
  - `/metrics` - Prometheus-compatible metrics
- **Quantization**: Support for INT8, INT4, and BitsAndBytes 4-bit quantization
- **Model Support**: GPT-2, LLaMA, Mistral, Phi, and other autoregressive models

### Quick Start

```bash
# Install dependencies
pip install transformers fastapi uvicorn

# Run server with GPT-2 (default)
python examples/serving/run_llm_server.py

# Run with quantization
python examples/serving/run_llm_server.py --model gpt2-medium --quantization int8

# Run with custom settings
python examples/serving/run_llm_server.py \
    --model gpt2 \
    --port 8080 \
    --max-batch-size 16 \
    --no-streaming
```

### Making Requests

Once the server is running, you can make requests using curl or any HTTP client:

#### Text Generation (Non-Streaming)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

#### Text Generation (Streaming)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_new_tokens": 100,
    "stream": true
  }'
```

#### Chat Completion

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_new_tokens": 150
  }'
```

#### Token Counting

```bash
curl -X POST http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you doing today?"}'
```

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Metrics

```bash
curl http://localhost:8000/metrics
```

### Command-Line Options

- `--model`: Model name from HuggingFace (default: gpt2)
- `--quantization`: Quantization mode (none, int8, int4, bnb_4bit)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--max-batch-size`: Maximum batch size for dynamic batching (default: 8)
- `--no-streaming`: Disable streaming support
- `--no-batching`: Disable dynamic batching
- `--device`: Device to use (auto, cuda, cpu)

### Supported Models

The server works with any autoregressive language model from HuggingFace:

- **GPT-2**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **LLaMA**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`
- **Mistral**: `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.2`
- **Phi**: `microsoft/phi-2`, `microsoft/phi-3-mini-4k-instruct`

### Performance Tuning

For optimal performance:

1. **Enable quantization** for reduced memory usage:
   ```bash
   python run_llm_server.py --model gpt2-medium --quantization int8
   ```

2. **Adjust batch size** based on available GPU memory:
   ```bash
   python run_llm_server.py --max-batch-size 16
   ```

3. **Use GPU** when available:
   ```bash
   python run_llm_server.py --device cuda
   ```

### Architecture

The LLM server uses:

- **LLMOptimizer** for model loading and optimization
- **FastAPI** for the REST API framework
- **Dynamic Batching** with a background thread for request accumulation
- **Streaming** via Server-Sent Events (SSE)
- **Thread-safe metrics** for monitoring

### Version

v0.5.3 - Production Serving Release
