"""
Whisper Optimization Example - Speech Recognition
"""

import torch
from torchbridge.models.multimodal import (
    create_whisper_base_optimized,
    WhisperBenchmark,
    OptimizationLevel,
)


def main():
    print("Whisper Optimization Example")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n1. Creating optimized Whisper-Base...")
    model, optimizer = create_whisper_base_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
    )

    print("Model optimized!")
    print(f"Optimizations: {optimizer.optimizations_applied}")

    print("\n2. Benchmarking transcription...")
    benchmark = WhisperBenchmark(model, optimizer)

    model_info = benchmark.get_model_info()
    print(f"Total Parameters: {model_info['total_parameters']:,}")

    results = benchmark.benchmark_transcription(
        num_iterations=5,
        audio_duration_seconds=10,
    )

    print(f"\nReal-time Factor: {results['real_time_factor']:.2f}")
    print(f"Is Real-time: {results['is_real_time']}")

    print("\nNote: Full transcription requires actual audio files.")
    print("Requirements: pip install transformers or pip install openai-whisper")


if __name__ == "__main__":
    main()
