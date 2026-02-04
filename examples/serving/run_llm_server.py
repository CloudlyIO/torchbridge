"""
Example: Running LLM Inference Server with TorchBridge

This example demonstrates how to:
1. Load and optimize an LLM using LLMOptimizer
2. Create an LLM inference server with streaming and batching
3. Start the server and make requests

Supported Models:
- GPT-2 (all variants)
- LLaMA-2, LLaMA-3
- Mistral-7B
- Phi-2, Phi-3

Usage:
    # Run with GPT-2 (default)
    python run_llm_server.py

    # Run with specific model and quantization
    python run_llm_server.py --model gpt2-medium --quantization int8

    # Run with custom port
    python run_llm_server.py --port 8080

Sample curl commands after starting server:

    # Text generation (non-streaming)
    curl -X POST http://localhost:8000/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Once upon a time", "max_new_tokens": 50, "temperature": 0.7}'

    # Text generation (streaming)
    curl -X POST http://localhost:8000/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "The future of AI is", "max_new_tokens": 100, "stream": true}'

    # Chat completion
    curl -X POST http://localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is machine learning?"}
        ],
        "max_new_tokens": 150
      }'

    # Token counting
    curl -X POST http://localhost:8000/tokenize \\
      -H "Content-Type: application/json" \\
      -d '{"text": "Hello, how are you doing today?"}'

    # Health check
    curl http://localhost:8000/health

    # Metrics
    curl http://localhost:8000/metrics

Version: 0.4.22
"""

import argparse
import logging
import sys
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run LLM server."""
    parser = argparse.ArgumentParser(
        description="Run TorchBridge LLM Inference Server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (e.g., gpt2, gpt2-medium, meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4", "bnb_4bit"],
        help="Quantization mode"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server to"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for dynamic batching"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming support"
    )
    parser.add_argument(
        "--no-batching",
        action="store_true",
        help="Disable dynamic batching"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    try:
        # Import required modules
        from torchbridge.models.llm import LLMOptimizer, LLMConfig, QuantizationMode
        from torchbridge.deployment.serving.llm_server import (
            create_llm_server,
            run_llm_server,
        )

        logger.info(f"Loading model: {args.model}")
        logger.info(f"Quantization: {args.quantization}")
        logger.info(f"Device: {args.device}")

        # Map quantization string to enum
        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.INT4,
            "bnb_4bit": QuantizationMode.BNBT4,
        }

        # Create LLM configuration
        config = LLMConfig(
            model_name=args.model,
            quantization=quant_map[args.quantization],
            device=args.device,
            use_flash_attention=True,
            use_torch_compile=False,  # Disable for server stability
        )

        # Create optimizer and load model
        logger.info("Initializing LLMOptimizer...")
        optimizer = LLMOptimizer(config)

        logger.info("Loading and optimizing model (this may take a few minutes)...")
        model, tokenizer = optimizer.optimize(args.model)

        # Print optimization info
        opt_info = optimizer.get_optimization_info()
        logger.info("Optimization complete:")
        for key, value in opt_info.items():
            logger.info(f"  {key}: {value}")

        # Estimate memory
        mem_info = optimizer.estimate_memory(args.model)
        logger.info("Memory estimate:")
        logger.info(f"  Model: {mem_info['model_memory_gb']:.2f} GB")
        logger.info(f"  KV Cache: {mem_info['kv_cache_gb']:.2f} GB")
        logger.info(f"  Total: {mem_info['total_gb']:.2f} GB")

        # Create server
        logger.info("Creating LLM server...")
        server = create_llm_server(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            enable_streaming=not args.no_streaming,
            enable_dynamic_batching=not args.no_batching,
            max_batch_size=args.max_batch_size,
        )

        # Print server info
        logger.info("=" * 70)
        logger.info("LLM Inference Server Ready!")
        logger.info("=" * 70)
        logger.info(f"Model: {args.model}")
        logger.info(f"Server: http://{args.host}:{args.port}")
        logger.info(f"Streaming: {'enabled' if not args.no_streaming else 'disabled'}")
        logger.info(f"Dynamic Batching: {'enabled' if not args.no_batching else 'disabled'}")
        logger.info(f"Max Batch Size: {args.max_batch_size}")
        logger.info("")
        logger.info("Available endpoints:")
        logger.info(f"  POST http://{args.host}:{args.port}/generate")
        logger.info(f"  POST http://{args.host}:{args.port}/chat")
        logger.info(f"  POST http://{args.host}:{args.port}/tokenize")
        logger.info(f"  GET  http://{args.host}:{args.port}/health")
        logger.info(f"  GET  http://{args.host}:{args.port}/metrics")
        logger.info("")
        logger.info("Example curl command:")
        logger.info(f'  curl -X POST http://{args.host}:{args.port}/generate \\')
        logger.info('    -H "Content-Type: application/json" \\')
        logger.info('    -d \'{"prompt": "Once upon a time", "max_new_tokens": 50}\'')
        logger.info("=" * 70)

        # Run server
        run_llm_server(
            server=server,
            host=args.host,
            port=args.port,
            workers=1,  # Use single worker for GPU models
            log_level="info",
        )

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install required packages:")
        logger.error("  pip install transformers fastapi uvicorn")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
