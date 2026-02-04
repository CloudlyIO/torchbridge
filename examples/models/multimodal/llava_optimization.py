"""
LLaVA Optimization Example - Visual Instruction Following
"""

import torch
from torchbridge.models.multimodal import (
    create_llava_7b_optimized,
    LLaVABenchmark,
    OptimizationLevel,
)


def main():
    print("LLaVA Optimization Example")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n1. Creating optimized LLaVA-1.5-7B...")
    model, optimizer = create_llava_7b_optimized(
        optimization_level=OptimizationLevel.O2,
        device=device,
    )

    print("Model optimized!")
    print(f"Optimizations: {optimizer.optimizations_applied}")

    print("\n2. Benchmarking generation...")
    benchmark = LLaVABenchmark(model, optimizer)

    model_info = benchmark.get_model_info()
    print(f"Total Parameters: {model_info['total_parameters']:,}")

    print("\nNote: Full generation requires actual images and prompts.")
    print("Requirements: pip install transformers")


if __name__ == "__main__":
    main()
