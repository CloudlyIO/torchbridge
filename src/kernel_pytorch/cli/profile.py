"""
Model profiling commands for KernelPyTorch CLI.

Provides detailed performance profiling of PyTorch models including
memory usage, operator timing, and bottleneck analysis.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class ProfileCommand:
    """Model profiling command implementation."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the profile command with argument parser."""
        parser = subparsers.add_parser(
            'profile',
            help='Profile PyTorch model performance',
            description='Analyze model performance with detailed profiling',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Profile Modes:
  summary    - Quick overview of model performance (default)
  detailed   - Detailed operator-level profiling
  memory     - Memory allocation analysis
  trace      - Full execution trace (Chrome trace format)

Examples:
  kpt-profile --model model.pt --mode summary
  kpt-profile --model model.pt --mode memory --input-shape 1,3,224,224
  kpt-profile --model model.pt --mode trace --output trace.json
  kpt-profile --model bert-base-uncased --mode detailed --iterations 100
            """
        )

        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Path to model file (.pt/.pth) or HuggingFace model name'
        )

        parser.add_argument(
            '--mode',
            type=str,
            choices=['summary', 'detailed', 'memory', 'trace'],
            default='summary',
            help='Profiling mode (default: summary)'
        )

        parser.add_argument(
            '--input-shape',
            type=str,
            default='1,512',
            help='Input shape as comma-separated values (default: 1,512)'
        )

        parser.add_argument(
            '--dtype',
            type=str,
            choices=['float32', 'float16', 'bfloat16'],
            default='float32',
            help='Model dtype (default: float32)'
        )

        parser.add_argument(
            '--device',
            type=str,
            default='auto',
            help='Device to profile on (auto, cpu, cuda, cuda:0, etc.)'
        )

        parser.add_argument(
            '--iterations',
            type=int,
            default=10,
            help='Number of profiling iterations (default: 10)'
        )

        parser.add_argument(
            '--warmup',
            type=int,
            default=3,
            help='Number of warmup iterations (default: 3)'
        )

        parser.add_argument(
            '--output', '-o',
            type=str,
            default=None,
            help='Output file for profile results (JSON or trace format)'
        )

        parser.add_argument(
            '--sort-by',
            type=str,
            choices=['cpu_time', 'cuda_time', 'cpu_memory', 'cuda_memory', 'count'],
            default='cpu_time',
            help='Sort detailed results by this metric (default: cpu_time)'
        )

        parser.add_argument(
            '--top-n',
            type=int,
            default=20,
            help='Show top N operators in detailed mode (default: 20)'
        )

        parser.add_argument(
            '--with-stack',
            action='store_true',
            help='Include stack traces in detailed profiling'
        )

        parser.add_argument(
            '--with-modules',
            action='store_true',
            help='Group by module hierarchy in detailed profiling'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-essential output'
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the profile command."""
        try:
            return ProfileCommand._run_profile(args)
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            print(f"Profiling failed: {e}")
            return 1

    @staticmethod
    def _run_profile(args) -> int:
        """Run the profiling process."""
        verbose = args.verbose and not args.quiet

        if verbose:
            print("KernelPyTorch Model Profiler")
            print("=" * 60)

        # Determine device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device

        # Determine dtype
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        dtype = dtype_map[args.dtype]

        if verbose:
            print(f"Device: {device}")
            print(f"Dtype: {args.dtype}")
            print(f"Mode: {args.mode}")

        # Load model
        if verbose:
            print(f"Loading model: {args.model}")

        model = ProfileCommand._load_model(args.model, dtype, device)
        model.eval()

        # Parse input shape and create sample input
        input_shape = ProfileCommand._parse_shape(args.input_shape)
        sample_input = torch.randn(*input_shape, dtype=dtype, device=device)

        if verbose:
            print(f"Input shape: {input_shape}")
            print(f"Warmup iterations: {args.warmup}")
            print(f"Profile iterations: {args.iterations}")
            print()

        # Run profiling based on mode
        if args.mode == 'summary':
            results = ProfileCommand._profile_summary(
                model, sample_input, args.warmup, args.iterations, device, verbose
            )
        elif args.mode == 'detailed':
            results = ProfileCommand._profile_detailed(
                model, sample_input, args.warmup, args.iterations,
                device, args.sort_by, args.top_n, args.with_stack,
                args.with_modules, verbose
            )
        elif args.mode == 'memory':
            results = ProfileCommand._profile_memory(
                model, sample_input, args.warmup, args.iterations, device, verbose
            )
        elif args.mode == 'trace':
            results = ProfileCommand._profile_trace(
                model, sample_input, args.warmup, args.iterations,
                device, args.output, verbose
            )

        # Output results
        if args.output and args.mode != 'trace':
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            if not args.quiet:
                print(f"\nResults saved to: {output_path}")

        return 0

    @staticmethod
    def _load_model(model_path: str, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Load a model from file or create a simple test model."""
        path = Path(model_path)

        if path.exists():
            loaded = torch.load(path, map_location=device, weights_only=False)
            if isinstance(loaded, torch.nn.Module):
                model = loaded
            elif isinstance(loaded, dict) and 'model' in loaded:
                model = loaded['model']
            else:
                raise ValueError(f"Cannot extract model from {path}")
        else:
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
            except (ImportError, Exception):
                print(f"Warning: Could not load '{model_path}', using simple test model")
                model = torch.nn.Sequential(
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                )

        return model.to(dtype).to(device)

    @staticmethod
    def _parse_shape(shape_str: str) -> tuple:
        """Parse shape string into tuple."""
        return tuple(int(x.strip()) for x in shape_str.split(','))

    @staticmethod
    def _profile_summary(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        warmup: int,
        iterations: int,
        device: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Quick summary profiling."""
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample_input)
                if 'cuda' in device:
                    torch.cuda.synchronize()

        # Profile
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                _ = model(sample_input)
                if 'cuda' in device:
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

        # Calculate statistics
        latencies_ms = [l * 1000 for l in latencies]
        mean_latency = sum(latencies_ms) / len(latencies_ms)
        min_latency = min(latencies_ms)
        max_latency = max(latencies_ms)
        std_latency = (sum((l - mean_latency) ** 2 for l in latencies_ms) / len(latencies_ms)) ** 0.5

        # Memory info
        memory_info = {}
        if 'cuda' in device:
            memory_info = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            }

        # Model info
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

        results = {
            'model': {
                'parameters': param_count,
                'size_mb': param_size_mb,
            },
            'latency': {
                'mean_ms': mean_latency,
                'std_ms': std_latency,
                'min_ms': min_latency,
                'max_ms': max_latency,
                'throughput_samples_per_sec': 1000 / mean_latency * sample_input.shape[0],
            },
            'memory': memory_info,
            'config': {
                'device': device,
                'dtype': str(sample_input.dtype),
                'input_shape': list(sample_input.shape),
                'iterations': iterations,
            }
        }

        # Print summary
        print("\nPROFILE SUMMARY")
        print("=" * 60)
        print(f"Model Parameters: {param_count:,} ({param_size_mb:.2f} MB)")
        print(f"\nLatency (ms):")
        print(f"  Mean:   {mean_latency:.3f}")
        print(f"  Std:    {std_latency:.3f}")
        print(f"  Min:    {min_latency:.3f}")
        print(f"  Max:    {max_latency:.3f}")
        print(f"\nThroughput: {results['latency']['throughput_samples_per_sec']:.1f} samples/sec")

        if memory_info:
            print(f"\nGPU Memory:")
            print(f"  Allocated: {memory_info['allocated_mb']:.2f} MB")
            print(f"  Reserved:  {memory_info['reserved_mb']:.2f} MB")
            print(f"  Peak:      {memory_info['max_allocated_mb']:.2f} MB")

        return results

    @staticmethod
    def _profile_detailed(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        warmup: int,
        iterations: int,
        device: str,
        sort_by: str,
        top_n: int,
        with_stack: bool,
        with_modules: bool,
        verbose: bool
    ) -> Dict[str, Any]:
        """Detailed operator-level profiling."""
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample_input)
                if 'cuda' in device:
                    torch.cuda.synchronize()

        # Profile with torch.profiler
        activities = [torch.profiler.ProfilerActivity.CPU]
        if 'cuda' in device:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
            with_modules=with_modules,
        ) as prof:
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(sample_input)
                    if 'cuda' in device:
                        torch.cuda.synchronize()

        # Get key averages
        sort_key = {
            'cpu_time': 'cpu_time_total',
            'cuda_time': 'cuda_time_total',
            'cpu_memory': 'cpu_memory_usage',
            'cuda_memory': 'cuda_memory_usage',
            'count': 'count',
        }.get(sort_by, 'cpu_time_total')

        key_averages = prof.key_averages(group_by_stack_n=5 if with_stack else 0)

        # Print table
        print("\nDETAILED OPERATOR PROFILE")
        print("=" * 60)
        print(key_averages.table(sort_by=sort_key, row_limit=top_n))

        # Build results
        operators = []
        for event in key_averages:
            operators.append({
                'name': event.key,
                'count': event.count,
                'cpu_time_us': event.cpu_time_total,
                'cuda_time_us': event.cuda_time_total if hasattr(event, 'cuda_time_total') else 0,
                'cpu_memory_bytes': event.cpu_memory_usage if hasattr(event, 'cpu_memory_usage') else 0,
                'cuda_memory_bytes': event.cuda_memory_usage if hasattr(event, 'cuda_memory_usage') else 0,
            })

        return {
            'operators': operators[:top_n],
            'total_operators': len(operators),
            'config': {
                'device': device,
                'iterations': iterations,
                'sort_by': sort_by,
            }
        }

    @staticmethod
    def _profile_memory(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        warmup: int,
        iterations: int,
        device: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Memory allocation analysis."""
        if 'cuda' not in device:
            print("Memory profiling is most useful on CUDA devices.")
            print("Running basic memory analysis on CPU...")

        # Reset memory stats
        if 'cuda' in device:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Get baseline
        if 'cuda' in device:
            baseline_allocated = torch.cuda.memory_allocated()
            baseline_reserved = torch.cuda.memory_reserved()
        else:
            baseline_allocated = 0
            baseline_reserved = 0

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample_input)
                if 'cuda' in device:
                    torch.cuda.synchronize()

        # Profile memory during inference
        memory_snapshots = []
        with torch.no_grad():
            for i in range(iterations):
                if 'cuda' in device:
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_allocated()

                output = model(sample_input)

                if 'cuda' in device:
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_allocated()
                    memory_snapshots.append({
                        'iteration': i,
                        'before_mb': before / 1024 / 1024,
                        'after_mb': after / 1024 / 1024,
                        'delta_mb': (after - before) / 1024 / 1024,
                    })

        # Get final stats
        results = {
            'model_params_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
            'input_size_mb': sample_input.numel() * sample_input.element_size() / 1024 / 1024,
        }

        if 'cuda' in device:
            results.update({
                'baseline_allocated_mb': baseline_allocated / 1024 / 1024,
                'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
                'current_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'memory_snapshots': memory_snapshots,
            })

        # Print summary
        print("\nMEMORY PROFILE")
        print("=" * 60)
        print(f"Model Parameters: {results['model_params_mb']:.2f} MB")
        print(f"Input Size: {results['input_size_mb']:.2f} MB")

        if 'cuda' in device:
            print(f"\nGPU Memory:")
            print(f"  Baseline:  {results['baseline_allocated_mb']:.2f} MB")
            print(f"  Peak:      {results['peak_allocated_mb']:.2f} MB")
            print(f"  Reserved:  {results['peak_reserved_mb']:.2f} MB")

            if memory_snapshots:
                avg_delta = sum(s['delta_mb'] for s in memory_snapshots) / len(memory_snapshots)
                print(f"\nActivation Memory (avg): {avg_delta:.2f} MB per forward pass")

        return results

    @staticmethod
    def _profile_trace(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        warmup: int,
        iterations: int,
        device: str,
        output_path: Optional[str],
        verbose: bool
    ) -> Dict[str, Any]:
        """Full execution trace (Chrome trace format)."""
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample_input)
                if 'cuda' in device:
                    torch.cuda.synchronize()

        # Determine output path
        if output_path is None:
            output_path = 'profile_trace.json'

        activities = [torch.profiler.ProfilerActivity.CPU]
        if 'cuda' in device:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        # Profile with trace
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(sample_input)
                    if 'cuda' in device:
                        torch.cuda.synchronize()

        # Export trace
        prof.export_chrome_trace(output_path)

        print("\nTRACE PROFILE")
        print("=" * 60)
        print(f"Trace exported to: {output_path}")
        print(f"Open in Chrome: chrome://tracing")
        print(f"Or use: https://ui.perfetto.dev/")

        return {
            'trace_file': output_path,
            'iterations': iterations,
            'device': device,
        }


def main(args=None):
    """CLI entry point for profile command."""
    parser = argparse.ArgumentParser(
        prog='kpt-profile',
        description='Profile PyTorch model performance'
    )

    # Manually add arguments since we're running standalone
    ProfileCommand.register(type('SubParsers', (), {'add_parser': lambda *a, **k: parser})())

    parsed_args = parser.parse_args(args)
    return ProfileCommand.execute(parsed_args)


if __name__ == '__main__':
    sys.exit(main())
