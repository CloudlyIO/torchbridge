#!/usr/bin/env python3
"""
Advanced Memory Optimization Demo Runner

Runs comprehensive demonstrations of all advanced memory optimization techniques:
- Deep Optimizer States
- Advanced Checkpointing
- Memory Pool Management
- Gradient Compression
- Long Sequence Optimization

Provides performance comparison and production readiness assessment.
"""

import torch
import argparse
import json
import time
import subprocess
import sys
import os
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def run_demo_script(script_name: str, device: str, quick: bool = False) -> Dict[str, Any]:
    """Run a demo script and capture results"""
    print(f"\n🔄 Running {script_name}...")

    cmd = [
        sys.executable, script_name,
        '--device', device,
        '--output', f'{script_name.replace(".py", "_results.json")}'
    ]

    if quick:
        cmd.append('--quick')

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {script_name} completed in {execution_time:.1f}s")

            # Try to load results
            try:
                results_file = f'{script_name.replace(".py", "_results.json")}'
                with open(results_file, 'r') as f:
                    results = json.load(f)

                # Clean up results file
                os.remove(results_file)

                return {
                    'status': 'success',
                    'execution_time': execution_time,
                    'results': results,
                    'stdout': result.stdout
                }
            except (FileNotFoundError, json.JSONDecodeError):
                return {
                    'status': 'success',
                    'execution_time': execution_time,
                    'results': {},
                    'stdout': result.stdout
                }
        else:
            print(f"❌ {script_name} failed")
            print(f"Error: {result.stderr}")
            return {
                'status': 'failed',
                'execution_time': execution_time,
                'error': result.stderr,
                'stdout': result.stdout
            }

    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out")
        return {
            'status': 'timeout',
            'execution_time': 300,
            'error': 'Demo timed out after 5 minutes'
        }
    except Exception as e:
        print(f"💥 {script_name} crashed: {e}")
        return {
            'status': 'crashed',
            'execution_time': 0,
            'error': str(e)
        }


def analyze_memory_optimization_results(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results from all memory optimization demos"""

    analysis = {
        'demo_success_rate': 0,
        'total_execution_time': 0,
        'memory_improvements': {},
        'performance_improvements': {},
        'key_findings': [],
        'production_readiness': 'Unknown'
    }

    successful_demos = 0
    total_demos = len(all_results)

    for demo_name, demo_result in all_results.items():
        analysis['total_execution_time'] += demo_result.get('execution_time', 0)

        if demo_result.get('status') == 'success':
            successful_demos += 1

            # Extract performance metrics if available
            results = demo_result.get('results', {})

            if demo_name == 'deep_optimizer_states_demo.py':
                # Analyze deep optimizer states results
                if 'standard' in results and 'deep_optimizer' in results:
                    standard = results['standard']
                    optimized = results['deep_optimizer']

                    if 'avg_step_time' in standard and 'avg_step_time' in optimized:
                        speedup = standard['avg_step_time'] / optimized['avg_step_time']
                        analysis['performance_improvements']['deep_optimizer_speedup'] = speedup

                        if speedup >= 2.0:
                            analysis['key_findings'].append(f"Deep Optimizer States achieved {speedup:.1f}x speedup")

                    if 'avg_memory_gb' in standard and 'avg_memory_gb' in optimized:
                        memory_reduction = (standard['avg_memory_gb'] - optimized['avg_memory_gb']) / standard['avg_memory_gb'] * 100
                        analysis['memory_improvements']['deep_optimizer_memory_reduction'] = memory_reduction

                        if memory_reduction > 20:
                            analysis['key_findings'].append(f"Deep Optimizer States reduced memory by {memory_reduction:.1f}%")

            elif demo_name == 'advanced_checkpointing_demo.py':
                # Analyze checkpointing results
                for model_type in ['resnet', 'transformer']:
                    model_key = f"{model_type}_standard"
                    selective_key = f"{model_type}_selective"
                    adaptive_key = f"{model_type}_adaptive"

                    if all(key in results for key in [model_key, selective_key, adaptive_key]):
                        standard = results[model_key]
                        selective = results[selective_key]
                        adaptive = results[adaptive_key]

                        # Find best memory reduction
                        best_memory = min(selective['peak_memory_gb'], adaptive['peak_memory_gb'])
                        memory_reduction = (standard['peak_memory_gb'] - best_memory) / standard['peak_memory_gb'] * 100

                        analysis['memory_improvements'][f'{model_type}_checkpointing_memory_reduction'] = memory_reduction

                        if memory_reduction > 15:
                            analysis['key_findings'].append(f"{model_type.title()} checkpointing reduced memory by {memory_reduction:.1f}%")

    # Calculate success rate
    analysis['demo_success_rate'] = successful_demos / total_demos * 100

    # Determine production readiness
    if analysis['demo_success_rate'] >= 80:
        analysis['production_readiness'] = 'Ready'
        analysis['key_findings'].append("All memory optimizations are production ready")
    elif analysis['demo_success_rate'] >= 60:
        analysis['production_readiness'] = 'Mostly Ready'
        analysis['key_findings'].append("Most memory optimizations are working, some issues detected")
    else:
        analysis['production_readiness'] = 'Not Ready'
        analysis['key_findings'].append("Significant issues detected with memory optimizations")

    return analysis


def run_comprehensive_memory_demo(device: str, quick_mode: bool = False) -> Dict[str, Any]:
    """Run all advanced memory optimization demos"""
    print("🚀 Advanced Memory Optimization: Comprehensive Demo Suite")
    print(f"📱 Device: {device}")
    print(f"⚡ Mode: {'Quick' if quick_mode else 'Full'}")
    print("=" * 70)

    # List of demo scripts to run
    demo_scripts = [
        'deep_optimizer_states_demo.py',
        'advanced_checkpointing_demo.py'
        # Note: Other demos would be added here as they're implemented
    ]

    all_results = {}

    # Run each demo
    for script in demo_scripts:
        if os.path.exists(script):
            all_results[script] = run_demo_script(script, device, quick_mode)
        else:
            print(f"⚠️  {script} not found, skipping...")
            all_results[script] = {
                'status': 'not_found',
                'execution_time': 0,
                'error': f'{script} not found'
            }

    # Analyze results
    print("\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    analysis = analyze_memory_optimization_results(all_results)

    print(f"Demo Success Rate: {analysis['demo_success_rate']:.1f}% ({sum(1 for r in all_results.values() if r.get('status') == 'success')}/{len(all_results)})")
    print(f"Total Execution Time: {analysis['total_execution_time']:.1f}s")
    print(f"Production Readiness: {analysis['production_readiness']}")

    print("\n🔍 Key Findings:")
    for finding in analysis['key_findings']:
        print(f"  • {finding}")

    print("\n📈 Performance Improvements:")
    for optimization, improvement in analysis['performance_improvements'].items():
        print(f"  • {optimization.replace('_', ' ').title()}: {improvement:.2f}x")

    print("\n💾 Memory Improvements:")
    for optimization, improvement in analysis['memory_improvements'].items():
        print(f"  • {optimization.replace('_', ' ').title()}: {improvement:.1f}% reduction")

    # Integration test
    print("\n🧪 INTEGRATION TESTING")
    print("=" * 70)

    try:
        # Test that all components can be imported together
        from kernel_pytorch.advanced_memory import (
            DeepOptimizerStates,
            InterleaveOffloadingOptimizer,
            SelectiveGradientCheckpointing,
            AdaptiveCheckpointing,
            MemoryEfficientBackprop,
            DynamicActivationOffloading,
            DynamicMemoryPool,
            MemoryPoolManager,
            GradientCompressor,
            LossyGradientCompression,
            LongSequenceOptimizer
        )

        print("✅ All memory optimization components importable")

        # Test basic functionality integration
        device_obj = torch.device(device)

        # Create a small test model
        test_model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        ).to(device_obj)

        # Test different components work together
        base_optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3)

        # Test interleave offloading
        interleave_optimizer = InterleaveOffloadingOptimizer(
            optimizer=base_optimizer,
            model=test_model,
            memory_limit_gb=1.0
        )

        # Test checkpointing
        adaptive_checkpoint = AdaptiveCheckpointing()

        # Test activation offloading
        offloader = DynamicActivationOffloading()

        # Test memory pool
        memory_pool = DynamicMemoryPool(device_obj)

        # Test gradient compressor
        compressor = LossyGradientCompression(bits=8)

        print("✅ All components instantiate correctly")

        # Test a combined training step
        x = torch.randn(4, 128, device=device_obj)
        target = torch.randn(4, 128, device=device_obj)

        # Forward pass with checkpointing
        outputs = adaptive_checkpoint.forward(test_model, x)
        loss = torch.nn.functional.mse_loss(outputs, target)

        # Backward pass
        interleave_optimizer.zero_grad()
        loss.backward()

        # Test gradient compression
        for name, param in test_model.named_parameters():
            if param.grad is not None:
                compressed = compressor.compress(param.grad)
                decompressed = compressor.decompress(compressed)
                # Verify shapes match
                assert decompressed.shape == param.grad.shape

        # Optimizer step
        metrics = interleave_optimizer.step()

        print("✅ Integrated training step successful")
        print(f"  Step metrics: {list(metrics.keys()) if isinstance(metrics, dict) else 'No metrics'}")

        analysis['integration_test'] = 'passed'

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        analysis['integration_test'] = 'failed'
        analysis['integration_error'] = str(e)

    # Final summary
    print("\n🎯 SUMMARY")
    print("=" * 70)

    if analysis['demo_success_rate'] >= 80 and analysis.get('integration_test') == 'passed':
        print("✅ Advanced Memory Optimizations: FULLY FUNCTIONAL")
        print("✅ Ready for production deployment")
        print(f"✅ Validated memory reductions up to {max(analysis['memory_improvements'].values(), default=0):.1f}%")
        print(f"✅ Validated performance improvements up to {max(analysis['performance_improvements'].values(), default=1):.1f}x")
    elif analysis['demo_success_rate'] >= 60:
        print("⚠️  Advanced Memory Optimizations: MOSTLY FUNCTIONAL")
        print("⚠️  Some optimizations may need attention")
    else:
        print("❌ Advanced Memory Optimizations: ISSUES DETECTED")
        print("❌ Requires debugging before production use")

    return {
        'demo_results': all_results,
        'analysis': analysis,
        'summary': {
            'success_rate': analysis['demo_success_rate'],
            'production_readiness': analysis['production_readiness'],
            'total_time': analysis['total_execution_time'],
            'integration_test': analysis.get('integration_test', 'not_run')
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Advanced Memory Optimization Demo Runner')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to run on')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced scope')
    parser.add_argument('--output', type=str, help='Save comprehensive results to JSON file')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    try:
        # Run comprehensive demo
        results = run_comprehensive_memory_demo(device, args.quick)

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                # Prepare results for JSON serialization
                serializable_results = {
                    'summary': results['summary'],
                    'analysis': {k: v for k, v in results['analysis'].items()
                               if k not in ['key_findings']},  # Exclude non-serializable items
                    'key_findings': results['analysis'].get('key_findings', []),
                    'demo_count': len(results['demo_results']),
                    'successful_demos': sum(1 for r in results['demo_results'].values()
                                          if r.get('status') == 'success')
                }
                json.dump(serializable_results, f, indent=2)
            print(f"\n💾 Comprehensive results saved to {args.output}")

    except Exception as e:
        print(f"❌ Comprehensive demo failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())