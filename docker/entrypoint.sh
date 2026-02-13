#!/bin/bash

# TorchBridge Docker Entrypoint Script
# Provides flexible container startup options

set -e

# Function to print banner
print_banner() {
    echo "🚀 TorchBridge Production Container"
    echo "======================================"
    echo "Version: $(python -c 'import torchbridge; print(torchbridge.__version__)')"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    fi
    echo "======================================"
}

# Function to run system diagnostics
run_diagnostics() {
    echo "🩺 Running system diagnostics..."
    python -m torchbridge.cli.doctor --verbose
}

# Function to start inference server
start_server() {
    echo "Starting TorchBridge inference server..."
    exec python -m examples.models.serving.run_llm_server \
        --host 0.0.0.0 \
        --port ${PORT:-8000}
}

# Function to run benchmarks
run_benchmarks() {
    echo "📊 Running performance benchmarks..."
    python -m torchbridge.cli.benchmark \
        --predefined ${BENCHMARK_SUITE:-cross_backend} \
        --quick \
        --output /app/logs/benchmark_results.json
}

# Main entrypoint logic
main() {
    print_banner

    case "$1" in
        "doctor")
            run_diagnostics
            ;;
        "server")
            start_server
            ;;
        "benchmark")
            run_benchmarks
            ;;
        "optimize")
            shift
            echo "Preparing model for target backend..."
            exec python -m torchbridge.cli.optimize "$@"
            ;;
        "bash"|"sh")
            echo "🛠️  Starting interactive shell..."
            exec /bin/bash
            ;;
        "python")
            shift
            exec python "$@"
            ;;
        *)
            echo "ℹ️  Available commands:"
            echo "  doctor     - Run system diagnostics"
            echo "  server     - Start inference API server"
            echo "  benchmark  - Run performance benchmarks"
            echo "  optimize   - Prepare a model for target backend"
            echo "  bash       - Interactive shell"
            echo "  python     - Run Python directly"
            echo ""
            echo "🔧 Running custom command: $@"
            exec "$@"
            ;;
    esac
}

# Run main function with all arguments
main "$@"