"""
Compiler Optimization Assistant - Main Interface

Main orchestrator for the compiler optimization workflow:
- Provides unified interface for optimization analysis and recommendations
- Handles interactive optimization workflows
- Manages optimization implementation and validation
- Offers educational content and tutorials
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type
import textwrap
from dataclasses import dataclass, asdict
import time
import numpy as np

from .model_analyzer import CodeAnalyzer, ModelAnalysisResult
from .optimization_recommendations import (
    OptimizationRecommendation,
    OptimizationRecommendationEngine,
    OptimizationImplementer
)


class CompilerOptimizationAssistant:
    """
    Main assistant class that orchestrates the optimization workflow.

    Provides a user-friendly interface for automated PyTorch optimization,
    combining analysis, recommendations, and implementation.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.analyzer = CodeAnalyzer()
        self.recommendation_engine = OptimizationRecommendationEngine()
        self.implementer = OptimizationImplementer()

    def optimize_model(self,
                      model: nn.Module,
                      sample_input: Optional[torch.Tensor] = None,
                      optimization_level: str = 'balanced',
                      interactive: bool = False) -> ModelAnalysisResult:
        """
        Comprehensive model optimization with analysis and recommendations.

        Args:
            model: PyTorch model to optimize
            sample_input: Sample input for analysis (optional)
            optimization_level: 'conservative', 'balanced', 'aggressive'
            interactive: Whether to provide interactive optimization

        Returns:
            ModelAnalysisResult with optimization analysis and recommendations
        """
        print("🚀 Compiler Optimization Assistant")
        print("=" * 50)

        # Step 1: Analyze model
        print("\n📊 Analyzing model architecture...")
        analysis = self.analyzer.analyze_model(model)

        # Step 2: Generate recommendations
        print("💡 Generating optimization recommendations...")
        recommendations = self.recommendation_engine.generate_recommendations(analysis)

        # Filter recommendations by optimization level
        filtered_recommendations = self._filter_by_optimization_level(recommendations, optimization_level)

        # Step 3: Calculate optimization score
        optimization_score = self._calculate_optimization_score(analysis, recommendations)

        # Step 4: Interactive optimization if requested
        if interactive:
            print("\n🔧 Interactive Optimization Mode")
            selected_recommendations = self._interactive_recommendation_selection(filtered_recommendations)
        else:
            selected_recommendations = filtered_recommendations

        # Step 5: Display results
        self._display_analysis_results(analysis, selected_recommendations, optimization_score)

        # Step 6: Create result object
        result = ModelAnalysisResult(
            model_info={
                'total_parameters': analysis['parameters']['total_parameters'],
                'model_size_mb': analysis['parameters']['model_size_mb'],
                'tensor_core_compatibility': analysis['parameters']['tensor_core_compatibility']
            },
            performance_bottlenecks=self._identify_bottlenecks(analysis),
            optimization_opportunities=selected_recommendations,
            current_optimizations=self._identify_current_optimizations(analysis),
            overall_optimization_score=optimization_score,
            improvement_potential=self._estimate_improvement_potential(selected_recommendations)
        )

        return result

    def apply_recommendations(self,
                            model: nn.Module,
                            recommendations: List[OptimizationRecommendation],
                            auto_implement: bool = False) -> Dict[str, Any]:
        """
        Apply optimization recommendations to a model.

        Args:
            model: Model to optimize
            recommendations: List of recommendations to apply
            auto_implement: Whether to automatically implement optimizations

        Returns:
            Dictionary with implementation results
        """
        results = {
            'original_model': model,
            'optimized_models': {},
            'implementation_code': {},
            'performance_improvements': {}
        }

        for rec in recommendations:
            print(f"\n🔧 Implementing: {rec.description}")

            try:
                if auto_implement:
                    optimized_model, code = self.implementer.implement_optimization(model, rec)
                    results['optimized_models'][rec.optimization_type] = optimized_model
                    results['implementation_code'][rec.optimization_type] = code
                    print(f"✅ Successfully implemented {rec.optimization_type}")
                else:
                    print(f"📋 Implementation steps for {rec.optimization_type}:")
                    for i, step in enumerate(rec.implementation_steps, 1):
                        print(f"    {i}. {step}")
                    print(f"\n💻 Code example:")
                    print(textwrap.indent(rec.code_example, "    "))

            except Exception as e:
                print(f"❌ Failed to implement {rec.optimization_type}: {str(e)}")

        return results

    def get_optimization_tutorial(self, optimization_type: str) -> str:
        """
        Get detailed tutorial for a specific optimization type.

        Args:
            optimization_type: Type of optimization

        Returns:
            Detailed tutorial text
        """
        tutorials = {
            'torch_compile': """
🎯 torch.compile Tutorial

torch.compile is PyTorch's modern compilation system that automatically optimizes
your models for better performance.

Key Benefits:
• 20-50% speedup with minimal code changes
• Automatic graph optimization and fusion
• Dynamic shape handling
• Multiple backend support

Best Practices:
1. Use static shapes when possible
2. Avoid complex control flow
3. Warm up the model before benchmarking
4. Choose appropriate compilation mode

Common Issues:
• First run is slower due to compilation
• Some operations may not be supported
• Dynamic shapes can reduce optimization effectiveness

Example Implementation:
```python
import torch

# Option 1: Decorator
@torch.compile
class MyModel(nn.Module):
    pass

# Option 2: Function call
model = MyModel()
optimized_model = torch.compile(model)
```
""",
            'operation_fusion': """
🎯 Operation Fusion Tutorial

Operation fusion combines multiple operations into single kernels to reduce
memory access overhead and improve GPU utilization.

Key Benefits:
• Reduced memory bandwidth requirements
• Fewer kernel launches
• Better GPU utilization
• 15-30% performance improvement

Common Fusion Patterns:
1. Linear + Activation (GELU, ReLU)
2. Normalization + Linear
3. Element-wise operations
4. Attention components

Implementation Guide:
```python
# Before: Separate operations
self.linear = nn.Linear(512, 1024)
self.activation = nn.GELU()

def forward(self, x):
    return self.activation(self.linear(x))

# After: Fused operation
from kernel_pytorch.compiler_optimized import FusedLinearGELU

self.fused_layer = FusedLinearGELU(512, 1024)

def forward(self, x):
    return self.fused_layer(x)
```
""",
            'attention_optimization': """
🎯 Attention Optimization Tutorial

Attention mechanisms often dominate compute and memory in transformer models.
Optimization can provide dramatic improvements.

Key Optimizations:
• Flash Attention for memory efficiency
• Fused QKV projections
• Optimized softmax implementations
• Mixed precision computation

Memory Benefits:
• Linear scaling vs quadratic for standard attention
• 50-80% memory reduction
• Enables longer sequence processing

Implementation:
```python
# Use optimized attention
from kernel_pytorch.compiler_optimized import CompilerOptimizedMultiHeadAttention

self.attention = CompilerOptimizedMultiHeadAttention(embed_dim, num_heads)

# Or use Flash Attention directly
import torch.nn.functional as F

if hasattr(F, 'scaled_dot_product_attention'):
    output = F.scaled_dot_product_attention(q, k, v)
```
"""
        }

        return tutorials.get(optimization_type, f"Tutorial for {optimization_type} not available yet.")

    def _filter_by_optimization_level(self, recommendations: List[OptimizationRecommendation], level: str) -> List[OptimizationRecommendation]:
        """Filter recommendations based on optimization level."""
        if level == 'conservative':
            return [r for r in recommendations if r.difficulty == 'easy' and r.priority in ['high', 'medium']]
        elif level == 'balanced':
            return [r for r in recommendations if r.priority in ['high', 'medium']]
        elif level == 'aggressive':
            return recommendations
        else:
            return recommendations

    def _interactive_recommendation_selection(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Interactive selection of recommendations."""
        print("\nAvailable optimizations:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. [{rec.priority}] {rec.description}")

        print("\nSelect optimizations to apply (comma-separated numbers, or 'all'):")
        selection = input().strip().lower()

        if selection == 'all':
            return recommendations

        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            return [recommendations[i] for i in indices if 0 <= i < len(recommendations)]
        except:
            print("Invalid selection, using all recommendations")
            return recommendations

    def _calculate_optimization_score(self, analysis: Dict[str, Any], recommendations: List[OptimizationRecommendation]) -> float:
        """Calculate overall optimization score."""
        # Base score from current optimizations
        base_score = 0.5

        # Add points for compilation readiness
        compilation_score = analysis['compilation_readiness']['compilation_compatibility_score']
        base_score += compilation_score * 0.2

        # Add points for tensor core compatibility
        tensor_score = analysis['parameters']['tensor_core_compatibility']
        base_score += tensor_score * 0.2

        # Subtract points for optimization opportunities (indicating room for improvement)
        opportunity_penalty = min(len(recommendations) * 0.1, 0.3)
        base_score -= opportunity_penalty

        return max(0.0, min(1.0, base_score))

    def _identify_bottlenecks(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        if analysis['parameters']['tensor_core_compatibility'] < 0.5:
            bottlenecks.append("Poor Tensor Core utilization due to dimension misalignment")

        if analysis['memory_patterns']['estimated_forward_memory_mb'] > 1000:
            bottlenecks.append("High memory usage may limit batch size")

        if analysis['compilation_readiness']['compilation_compatibility_score'] < 0.7:
            bottlenecks.append("Model may not compile efficiently due to dynamic operations")

        return bottlenecks

    def _identify_current_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify currently applied optimizations."""
        current_opts = []

        if analysis['compilation_readiness']['compilation_compatibility_score'] > 0.8:
            current_opts.append("Model is compilation-ready")

        if analysis['parameters']['tensor_core_compatibility'] > 0.8:
            current_opts.append("Good Tensor Core compatibility")

        return current_opts

    def _estimate_improvement_potential(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Estimate overall improvement potential."""
        if not recommendations:
            return "Minimal - model appears well optimized"

        high_priority = sum(1 for r in recommendations if r.priority == 'high')
        total = len(recommendations)

        if high_priority >= 3:
            return "Significant - multiple high-impact optimizations available"
        elif high_priority >= 1:
            return "Moderate - some important optimizations available"
        else:
            return "Minor - mostly incremental improvements available"

    def _display_analysis_results(self,
                                analysis: Dict[str, Any],
                                recommendations: List[OptimizationRecommendation],
                                optimization_score: float):
        """Display comprehensive analysis results."""
        print(f"\n📈 Analysis Results:")
        print(f"  Model Size: {analysis['parameters']['model_size_mb']:.1f} MB")
        print(f"  Parameters: {analysis['parameters']['total_parameters']:,}")
        print(f"  Tensor Core Compatibility: {analysis['parameters']['tensor_core_compatibility']:.1%}")
        print(f"  Compilation Readiness: {analysis['compilation_readiness']['compilation_compatibility_score']:.1%}")
        print(f"  Overall Optimization Score: {optimization_score:.1%}")

        print(f"\n💡 Optimization Recommendations ({len(recommendations)} found):")
        for i, rec in enumerate(recommendations, 1):
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[rec.priority]
            difficulty_emoji = {'easy': '✅', 'medium': '⚠️', 'hard': '🔥'}[rec.difficulty]

            print(f"  {i}. {priority_emoji} {difficulty_emoji} {rec.description}")
            print(f"     Expected: {rec.expected_improvement}")
            print(f"     Difficulty: {rec.difficulty.title()}")

        if not recommendations:
            print("  🎉 Model appears well-optimized! Consider advanced techniques or hardware-specific optimizations.")


def demonstrate_optimization_assistant():
    """
    Comprehensive demonstration of the Compiler Optimization Assistant.
    """
    print("🤖 Compiler Optimization Assistant Demonstration")
    print("=" * 60)

    # Create a sample model for demonstration
    class DemoTransformerModel(nn.Module):
        def __init__(self, embed_dim=512, num_heads=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(10000, embed_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))

            # Create transformer layers
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                    'norm1': nn.LayerNorm(embed_dim),
                    'ffn': nn.Sequential(
                        nn.Linear(embed_dim, embed_dim * 4),
                        nn.GELU(),
                        nn.Linear(embed_dim * 4, embed_dim)
                    ),
                    'norm2': nn.LayerNorm(embed_dim)
                })
                for _ in range(num_layers)
            ])

            self.output_projection = nn.Linear(embed_dim, 10000)

        def forward(self, x):
            # Add positional encoding
            seq_len = x.shape[1]
            x = self.embedding(x) + self.pos_encoding[:seq_len]

            # Apply transformer layers
            for layer in self.layers:
                # Self-attention block
                attn_out, _ = layer['attention'](x, x, x)
                x = layer['norm1'](x + attn_out)

                # FFN block
                ffn_out = layer['ffn'](x)
                x = layer['norm2'](x + ffn_out)

            return self.output_projection(x)

    # Initialize the assistant and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assistant = CompilerOptimizationAssistant(device=device)
    model = DemoTransformerModel().to(device)

    print(f"📊 Demo Model Information:")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.1f} MB")

    # Create sample input
    sample_input = torch.randint(0, 10000, (4, 128), device=device)

    # Run comprehensive optimization analysis
    print(f"\n🔍 Running Comprehensive Analysis...")
    result = assistant.optimize_model(
        model,
        sample_input=sample_input,
        optimization_level='balanced',
        interactive=False  # Set to True for interactive mode
    )

    # Display detailed recommendations
    print(f"\n📋 Detailed Optimization Recommendations:")
    for i, rec in enumerate(result.optimization_opportunities[:3], 1):  # Show top 3
        print(f"\n{i}. {rec.description} ({rec.priority} priority)")
        print(f"   Expected improvement: {rec.expected_improvement}")
        print(f"   Implementation steps:")
        for j, step in enumerate(rec.implementation_steps, 1):
            print(f"     {j}. {step}")

        print(f"   Educational notes:")
        for note in rec.educational_notes[:2]:  # Show first 2 notes
            print(f"     • {note}")

    # Apply some optimizations
    print(f"\n⚙️ Applying Optimizations...")
    implementation_results = assistant.apply_recommendations(
        model,
        result.optimization_opportunities[:2],  # Apply first 2 recommendations
        auto_implement=False  # Set to True for automatic implementation
    )

    # Show tutorial for a specific optimization
    print(f"\n📚 Tutorial Example - torch.compile:")
    tutorial = assistant.get_optimization_tutorial('torch_compile')
    print(tutorial[:500] + "..." if len(tutorial) > 500 else tutorial)

    # Performance comparison demo
    print(f"\n⚡ Performance Comparison Demo:")

    # Original model timing
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(sample_input)

        # Benchmark original
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        for _ in range(10):
            outputs = model(sample_input)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        original_time = (time.time() - start_time) / 10 * 1000  # ms per run

    # Compiled model timing
    compiled_model = torch.compile(model)
    with torch.no_grad():
        # Warmup compilation
        for _ in range(5):
            _ = compiled_model(sample_input)

        # Benchmark compiled
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        for _ in range(10):
            outputs = compiled_model(sample_input)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        compiled_time = (time.time() - start_time) / 10 * 1000  # ms per run

    speedup = original_time / compiled_time if compiled_time > 0 else 1.0

    print(f"  Original model: {original_time:.2f} ms/run")
    print(f"  Compiled model: {compiled_time:.2f} ms/run")
    print(f"  Speedup: {speedup:.2f}x")

    # Summary
    print(f"\n✅ Optimization Assistant Demo Complete!")
    print(f"Key capabilities demonstrated:")
    print(f"  • Automated model analysis and bottleneck identification")
    print(f"  • Intelligent optimization recommendation generation")
    print(f"  • Educational explanations and implementation guidance")
    print(f"  • Performance impact prediction and measurement")
    print(f"  • Interactive optimization workflow")


if __name__ == "__main__":
    demonstrate_optimization_assistant()