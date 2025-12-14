"""
Compute Intensity Optimization Patterns for GPU Performance

This module provides comprehensive guidance and practical implementations for
optimizing arithmetic intensity in PyTorch neural networks, focusing on maximizing
the compute-to-memory-access ratio for optimal GPU utilization.

🎓 EDUCATIONAL FOCUS:
Arithmetic intensity (FLOPs per byte) is crucial for GPU performance optimization:
- High intensity operations: Efficient use of GPU compute units (matrix multiplication)
- Low intensity operations: Memory-bound, limited by bandwidth (element-wise operations)
- Roofline model: Performance ceiling determined by arithmetic intensity
- GPU utilization: Higher intensity = better GPU core utilization

🔧 COMPUTE OPTIMIZATION STRATEGIES:
- Operation fusion: Combine low-intensity operations to increase overall intensity
- Blocking/Tiling: Process data in blocks that fit in cache for reuse
- Batching: Increase arithmetic intensity through larger batch operations
- Mixed precision: Use lower precision where appropriate to increase throughput

💡 PRACTICAL APPLICATION:
Learn to identify compute bottlenecks and transform memory-bound operations
into compute-bound operations for 2-10x performance improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math


class ComputeIntensityCategory(Enum):
    """Categories of compute intensity for optimization analysis."""
    MEMORY_BOUND = "memory_bound"       # < 1 FLOP/byte
    BALANCED = "balanced"               # 1-10 FLOP/byte
    COMPUTE_BOUND = "compute_bound"     # > 10 FLOP/byte


class OptimizationPriority(Enum):
    """Priority levels for optimization interventions."""
    CRITICAL = "critical"               # < 0.5 FLOP/byte
    HIGH = "high"                      # 0.5-2 FLOP/byte
    MEDIUM = "medium"                  # 2-8 FLOP/byte
    LOW = "low"                        # > 8 FLOP/byte


@dataclass
class ComputeOptimizationPattern:
    """
    Data structure for describing compute intensity optimization patterns.

    🎓 EDUCATIONAL: Systematic compute optimization approach
    By analyzing compute intensity patterns, we can:
    - Identify memory-bound operations that limit performance
    - Apply transformations to increase arithmetic intensity
    - Measure and validate compute efficiency improvements
    - Build reusable high-intensity operation patterns
    """
    name: str
    category: ComputeIntensityCategory
    baseline_intensity: float  # FLOPs per byte before optimization
    optimized_intensity: float  # FLOPs per byte after optimization
    techniques: List[str]
    hardware_utilization: float  # Expected GPU utilization improvement
    description: str
    example_before: str
    example_after: str


# 🎓 EDUCATIONAL: Compute intensity optimization targets for different operations
COMPUTE_INTENSITY_TARGETS = [
    ComputeOptimizationPattern(
        name="Matrix Multiplication Optimization",
        category=ComputeIntensityCategory.COMPUTE_BOUND,
        baseline_intensity=50.0,
        optimized_intensity=200.0,
        techniques=["Batching", "Blocking/Tiling", "Mixed precision", "Tensor cores"],
        hardware_utilization=0.8,
        description="Optimize matrix operations for maximum arithmetic intensity",
        example_before="Sequential small matrix multiplications",
        example_after="Batched large matrix multiplication with optimal blocking"
    ),

    ComputeOptimizationPattern(
        name="Element-wise Operation Fusion",
        category=ComputeIntensityCategory.MEMORY_BOUND,
        baseline_intensity=0.25,
        optimized_intensity=2.0,
        techniques=["Operation fusion", "Kernel combination", "Memory reuse"],
        hardware_utilization=0.4,
        description="Fuse element-wise operations to increase intensity",
        example_before="x = relu(x); x = x * scale; x = x + bias",
        example_after="x = fused_relu_scale_add(x, scale, bias)"
    ),

    ComputeOptimizationPattern(
        name="Convolution Optimization",
        category=ComputeIntensityCategory.BALANCED,
        baseline_intensity=5.0,
        optimized_intensity=25.0,
        techniques=["Im2col transformation", "Winograd algorithm", "Channel grouping"],
        hardware_utilization=0.7,
        description="Transform convolutions for higher arithmetic intensity",
        example_before="Standard 2D convolution implementation",
        example_after="Optimized convolution with Winograd or im2col"
    ),

    ComputeOptimizationPattern(
        name="Attention Mechanism Optimization",
        category=ComputeIntensityCategory.BALANCED,
        baseline_intensity=8.0,
        optimized_intensity=40.0,
        techniques=["Flash Attention", "QKV fusion", "Attention blocking"],
        hardware_utilization=0.6,
        description="Optimize attention patterns for memory and compute efficiency",
        example_before="Standard attention with separate Q, K, V operations",
        example_after="Flash Attention with fused QKV and optimized memory access"
    ),

    ComputeOptimizationPattern(
        name="Reduction Operation Optimization",
        category=ComputeIntensityCategory.MEMORY_BOUND,
        baseline_intensity=0.5,
        optimized_intensity=4.0,
        techniques=["Tree reduction", "Shared memory utilization", "Warp-level primitives"],
        hardware_utilization=0.5,
        description="Optimize reductions for better compute utilization",
        example_before="Standard reduction with poor memory access pattern",
        example_after="Optimized tree reduction with shared memory and warp shuffles"
    )
]


def calculate_arithmetic_intensity(
    operation_flops: float,
    memory_bytes_accessed: float,
    include_gradients: bool = True
) -> float:
    """
    Calculate arithmetic intensity for a given operation.

    🎓 EDUCATIONAL: Arithmetic intensity calculation methodology
    Arithmetic intensity is the fundamental metric for understanding GPU performance
    characteristics. This function demonstrates how to accurately calculate and
    interpret this critical metric.

    🔧 CALCULATION COMPONENTS:
    - FLOPs: Floating point operations (forward + backward if training)
    - Memory access: All reads and writes to global memory
    - Gradient computation: Additional FLOPs and memory access during training
    - Intermediate results: Memory overhead from temporary tensors

    📊 INTENSITY INTERPRETATION:
    - < 1 FLOP/byte: Severely memory-bound, priority optimization target
    - 1-10 FLOP/byte: Balanced, good optimization potential
    - > 10 FLOP/byte: Compute-bound, focus on compute optimizations

    Args:
        operation_flops: Number of floating point operations
        memory_bytes_accessed: Total bytes read from and written to memory
        include_gradients: Whether to include gradient computation overhead

    Returns:
        Arithmetic intensity in FLOPs per byte
    """
    if memory_bytes_accessed == 0:
        return float('inf')  # Pure compute operation

    # 🔧 EDUCATIONAL: Account for gradient computation overhead
    if include_gradients:
        # Backward pass typically adds 2x FLOPs and additional memory access
        total_flops = operation_flops * 3  # Forward + backward
        gradient_memory_overhead = memory_bytes_accessed * 0.5  # Gradient storage
        total_memory = memory_bytes_accessed + gradient_memory_overhead
    else:
        total_flops = operation_flops
        total_memory = memory_bytes_accessed

    return total_flops / total_memory


def analyze_compute_intensity_profile(
    model: nn.Module,
    sample_input: torch.Tensor,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Analyze the compute intensity profile of a neural network model.

    🎓 EDUCATIONAL: Comprehensive compute intensity analysis
    This function provides a systematic methodology for analyzing model
    compute characteristics and identifying optimization opportunities.

    🔧 ANALYSIS COMPONENTS:
    - Layer-by-layer intensity calculation
    - Operation type classification
    - Bottleneck identification
    - Optimization priority ranking

    Args:
        model: PyTorch model to analyze
        sample_input: Representative input for analysis
        detailed: Whether to provide detailed per-layer analysis

    Returns:
        Comprehensive compute intensity analysis and optimization recommendations
    """
    analysis_results = {
        "overall_intensity": 0.0,
        "layer_analysis": {},
        "bottlenecks": [],
        "optimization_opportunities": [],
        "intensity_distribution": {
            "memory_bound": 0,
            "balanced": 0,
            "compute_bound": 0
        }
    }

    total_flops = 0
    total_memory = 0

    # 🔍 STEP 1: Analyze each module's compute characteristics
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_analysis = _analyze_module_intensity(module, sample_input)

            if module_analysis and detailed:
                analysis_results["layer_analysis"][name] = module_analysis
                total_flops += module_analysis["flops"]
                total_memory += module_analysis["memory_bytes"]

                # Categorize by intensity
                intensity = module_analysis["intensity"]
                if intensity < 1.0:
                    analysis_results["intensity_distribution"]["memory_bound"] += 1
                elif intensity < 10.0:
                    analysis_results["intensity_distribution"]["balanced"] += 1
                else:
                    analysis_results["intensity_distribution"]["compute_bound"] += 1

    # 🔍 STEP 2: Calculate overall model intensity
    analysis_results["overall_intensity"] = total_flops / total_memory if total_memory > 0 else 0

    # 🔍 STEP 3: Identify optimization bottlenecks
    analysis_results["bottlenecks"] = _identify_compute_bottlenecks(analysis_results["layer_analysis"])

    # 🔍 STEP 4: Generate optimization recommendations
    analysis_results["optimization_opportunities"] = _generate_intensity_optimizations(
        analysis_results["layer_analysis"]
    )

    return analysis_results


def _analyze_module_intensity(module: nn.Module, sample_input: torch.Tensor) -> Optional[Dict[str, Any]]:
    """Analyze compute intensity for a specific module."""
    if isinstance(module, nn.Linear):
        # Linear layer analysis
        in_features, out_features = module.in_features, module.out_features
        batch_size = sample_input.shape[0] if sample_input.dim() > 1 else 1

        # FLOPs calculation: 2 * input_features * output_features * batch_size
        flops = 2 * in_features * out_features * batch_size

        # Memory access: weights + bias + input + output
        weight_memory = in_features * out_features * 4  # fp32
        bias_memory = out_features * 4 if module.bias is not None else 0
        input_memory = batch_size * in_features * 4
        output_memory = batch_size * out_features * 4
        total_memory = weight_memory + bias_memory + input_memory + output_memory

        intensity = calculate_arithmetic_intensity(flops, total_memory, include_gradients=True)

        return {
            "type": "Linear",
            "flops": flops,
            "memory_bytes": total_memory,
            "intensity": intensity,
            "optimization_potential": "high" if intensity < 10 else "medium"
        }

    elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
        # Activation function analysis
        num_elements = sample_input.numel()
        batch_size = sample_input.shape[0] if sample_input.dim() > 1 else 1

        # Element-wise operations: 1 FLOP per element
        flops = num_elements

        # Memory access: read input + write output
        memory_bytes = 2 * num_elements * 4  # fp32

        intensity = calculate_arithmetic_intensity(flops, memory_bytes, include_gradients=True)

        return {
            "type": "Activation",
            "flops": flops,
            "memory_bytes": memory_bytes,
            "intensity": intensity,
            "optimization_potential": "fusion_candidate"
        }

    elif isinstance(module, nn.LayerNorm):
        # LayerNorm analysis
        num_elements = sample_input.numel()
        normalized_shape = module.normalized_shape[0] if module.normalized_shape else num_elements

        # LayerNorm FLOPs: mean, variance, normalization, scale, shift
        flops = 5 * num_elements

        # Memory access: input + output + weight + bias + statistics
        memory_bytes = (2 * num_elements + 2 * normalized_shape + 2 * sample_input.shape[0]) * 4

        intensity = calculate_arithmetic_intensity(flops, memory_bytes, include_gradients=True)

        return {
            "type": "LayerNorm",
            "flops": flops,
            "memory_bytes": memory_bytes,
            "intensity": intensity,
            "optimization_potential": "medium"
        }

    return None


def _identify_compute_bottlenecks(layer_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify compute intensity bottlenecks in the model."""
    bottlenecks = []

    for layer_name, analysis in layer_analysis.items():
        intensity = analysis.get("intensity", 0)

        if intensity < 0.5:  # Severely memory-bound
            bottlenecks.append({
                "layer": layer_name,
                "type": "severe_memory_bound",
                "intensity": intensity,
                "priority": OptimizationPriority.CRITICAL.value,
                "recommendation": "Immediate fusion or algorithmic change required"
            })
        elif intensity < 2.0:  # Memory-bound
            bottlenecks.append({
                "layer": layer_name,
                "type": "memory_bound",
                "intensity": intensity,
                "priority": OptimizationPriority.HIGH.value,
                "recommendation": "Consider operation fusion or blocking strategies"
            })

    return sorted(bottlenecks, key=lambda x: x["intensity"])


def _generate_intensity_optimizations(layer_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate specific optimization recommendations based on intensity analysis."""
    optimizations = []

    # Group consecutive low-intensity operations for fusion
    consecutive_activations = []
    for layer_name, analysis in layer_analysis.items():
        if analysis.get("type") == "Activation" and analysis.get("intensity", 0) < 2.0:
            consecutive_activations.append(layer_name)
        else:
            if len(consecutive_activations) >= 2:
                optimizations.append({
                    "type": "activation_fusion",
                    "layers": consecutive_activations.copy(),
                    "potential_improvement": "2-5x speedup",
                    "technique": "Kernel fusion with torch.compile"
                })
            consecutive_activations.clear()

    # Identify linear + activation patterns
    layer_names = list(layer_analysis.keys())
    for i in range(len(layer_names) - 1):
        current_analysis = layer_analysis[layer_names[i]]
        next_analysis = layer_analysis[layer_names[i + 1]]

        if (current_analysis.get("type") == "Linear" and
            next_analysis.get("type") == "Activation"):
            optimizations.append({
                "type": "linear_activation_fusion",
                "layers": [layer_names[i], layer_names[i + 1]],
                "potential_improvement": "1.5-3x speedup",
                "technique": "Fused linear-activation kernel"
            })

    return optimizations


def optimize_flop_to_byte_ratio(
    model: nn.Module,
    target_intensity: float = 10.0,
    optimization_strategy: str = "fusion"
) -> nn.Module:
    """
    Optimize model to achieve target arithmetic intensity.

    🎓 EDUCATIONAL: Systematic intensity optimization approach
    This function demonstrates how to systematically transform a model
    to achieve higher arithmetic intensity through various optimization
    techniques.

    🔧 OPTIMIZATION STRATEGIES:
    - Fusion: Combine operations to reduce memory traffic
    - Blocking: Process data in cache-friendly blocks
    - Precision: Use mixed precision to increase throughput
    - Algorithmic: Replace low-intensity algorithms with high-intensity alternatives

    Args:
        model: Model to optimize
        target_intensity: Target arithmetic intensity (FLOPs per byte)
        optimization_strategy: Strategy to use for optimization

    Returns:
        Optimized model with improved arithmetic intensity
    """
    if optimization_strategy == "fusion":
        return _apply_fusion_optimization(model, target_intensity)
    elif optimization_strategy == "blocking":
        return _apply_blocking_optimization(model, target_intensity)
    elif optimization_strategy == "mixed_precision":
        return _apply_precision_optimization(model, target_intensity)
    else:
        raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")


def _apply_fusion_optimization(model: nn.Module, target_intensity: float) -> nn.Module:
    """Apply operation fusion optimization to improve arithmetic intensity."""
    # Educational implementation - would implement actual fusion logic
    optimized_model = model

    # Example: Replace Linear + ReLU with fused implementation
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if followed by activation
            # TODO: Implement actual fusion of linear layers with activations
            # This would analyze the forward graph to identify fusable patterns
            pass

    return optimized_model


def _apply_blocking_optimization(model: nn.Module, target_intensity: float) -> nn.Module:
    """Apply blocking/tiling optimization for better cache utilization."""
    # TODO: Implement blocking/tiling strategies for improved cache utilization
    # This would include matrix blocking, loop tiling, and cache-aware algorithms
    return model


def _apply_precision_optimization(model: nn.Module, target_intensity: float) -> nn.Module:
    """Apply mixed precision optimization to increase arithmetic intensity."""
    # Convert model to use automatic mixed precision
    optimized_model = model.half()  # Educational simplification
    return optimized_model


def identify_compute_bottlenecks(
    model: nn.Module,
    sample_input: torch.Tensor,
    threshold_intensity: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Identify compute bottlenecks in the model based on arithmetic intensity.

    🎓 EDUCATIONAL: Systematic bottleneck identification
    This function provides a systematic approach to identifying operations
    that limit overall model performance due to low arithmetic intensity.

    🔧 BOTTLENECK ANALYSIS:
    - Layer-by-layer intensity measurement
    - Critical path identification
    - Performance impact quantification
    - Optimization priority ranking

    Args:
        model: Model to analyze for bottlenecks
        sample_input: Representative input tensor
        threshold_intensity: Intensity below which operations are considered bottlenecks

    Returns:
        List of identified bottlenecks with optimization recommendations
    """
    bottlenecks = []

    # Analyze model compute characteristics
    intensity_profile = analyze_compute_intensity_profile(model, sample_input)

    for layer_name, analysis in intensity_profile.get("layer_analysis", {}).items():
        intensity = analysis.get("intensity", 0)

        if intensity < threshold_intensity:
            # Calculate performance impact
            layer_flops = analysis.get("flops", 0)
            total_flops = sum(a.get("flops", 0) for a in intensity_profile["layer_analysis"].values())
            impact_ratio = layer_flops / total_flops if total_flops > 0 else 0

            bottleneck = {
                "layer_name": layer_name,
                "layer_type": analysis.get("type", "unknown"),
                "current_intensity": intensity,
                "performance_impact": impact_ratio,
                "optimization_priority": _calculate_optimization_priority(intensity, impact_ratio),
                "recommended_techniques": _recommend_optimization_techniques(analysis)
            }
            bottlenecks.append(bottleneck)

    # Sort by optimization priority (impact * (1/intensity))
    bottlenecks.sort(key=lambda x: x["performance_impact"] / (x["current_intensity"] + 1e-6), reverse=True)

    return bottlenecks


def _calculate_optimization_priority(intensity: float, impact_ratio: float) -> str:
    """Calculate optimization priority based on intensity and performance impact."""
    priority_score = impact_ratio / (intensity + 1e-6)

    if priority_score > 0.1:
        return "critical"
    elif priority_score > 0.05:
        return "high"
    elif priority_score > 0.01:
        return "medium"
    else:
        return "low"


def _recommend_optimization_techniques(analysis: Dict[str, Any]) -> List[str]:
    """Recommend specific optimization techniques based on layer analysis."""
    layer_type = analysis.get("type", "")
    intensity = analysis.get("intensity", 0)

    techniques = []

    if layer_type == "Activation" and intensity < 2.0:
        techniques.extend(["Operation fusion", "Kernel optimization", "In-place operations"])
    elif layer_type == "Linear" and intensity < 10.0:
        techniques.extend(["Batch optimization", "Mixed precision", "Weight preprocessing"])
    elif layer_type == "LayerNorm" and intensity < 5.0:
        techniques.extend(["Fused normalization", "Optimized statistics computation"])

    return techniques


class ComputeIntensityProfiler:
    """
    Advanced profiler for compute intensity analysis and optimization.

    🎓 EDUCATIONAL: Production-grade intensity profiling
    This class demonstrates how to build comprehensive profiling tools
    for systematic compute intensity optimization in production environments.
    """

    def __init__(self, enable_detailed_profiling: bool = True):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.profiling_results = {}

    def profile_model(
        self,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
        optimization_targets: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model profiling for compute intensity optimization.

        🔧 PROFILING CAPABILITIES:
        - Multi-input analysis for robustness
        - Optimization target comparison
        - Performance regression detection
        - Optimization opportunity quantification
        """
        results = {
            "model_summary": self._get_model_summary(model),
            "intensity_analysis": {},
            "optimization_recommendations": [],
            "target_comparison": {}
        }

        for i, sample_input in enumerate(sample_inputs):
            input_analysis = analyze_compute_intensity_profile(model, sample_input)
            results["intensity_analysis"][f"input_{i}"] = input_analysis

        # Generate comprehensive recommendations
        results["optimization_recommendations"] = self._generate_comprehensive_recommendations(
            results["intensity_analysis"]
        )

        # Compare against targets if provided
        if optimization_targets:
            results["target_comparison"] = self._compare_against_targets(
                results["intensity_analysis"], optimization_targets
            )

        return results

    def _get_model_summary(self, model: nn.Module) -> Dict[str, Any]:
        """Generate model summary for profiling context."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),  # Assuming fp32
            "architecture_type": self._classify_architecture(model)
        }

    def _classify_architecture(self, model: nn.Module) -> str:
        """Classify model architecture type for targeted optimization."""
        module_types = [type(m).__name__ for m in model.modules()]

        if any("Attention" in name for name in module_types):
            return "transformer"
        elif any("Conv" in name for name in module_types):
            return "convolutional"
        elif all("Linear" in name or "ReLU" in name for name in module_types if name != "Module"):
            return "feedforward"
        else:
            return "mixed"

    def _generate_comprehensive_recommendations(self, intensity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []

        # Aggregate analysis across all inputs
        all_bottlenecks = []
        for analysis in intensity_analysis.values():
            all_bottlenecks.extend(analysis.get("bottlenecks", []))

        # Group by bottleneck type and prioritize
        bottleneck_groups = {}
        for bottleneck in all_bottlenecks:
            btype = bottleneck.get("type", "unknown")
            if btype not in bottleneck_groups:
                bottleneck_groups[btype] = []
            bottleneck_groups[btype].append(bottleneck)

        # Generate targeted recommendations
        for btype, bottlenecks in bottleneck_groups.items():
            if btype == "severe_memory_bound":
                recommendations.append({
                    "category": "Critical Memory Optimization",
                    "affected_layers": len(bottlenecks),
                    "techniques": ["Immediate kernel fusion", "Algorithmic replacement", "Mixed precision"],
                    "expected_improvement": "3-10x speedup",
                    "implementation_priority": "immediate"
                })

        return recommendations

    def _compare_against_targets(
        self,
        intensity_analysis: Dict[str, Any],
        targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare current performance against optimization targets."""
        comparison = {}

        for input_key, analysis in intensity_analysis.items():
            overall_intensity = analysis.get("overall_intensity", 0)
            target_intensity = targets.get("overall_intensity", 10.0)

            comparison[input_key] = {
                "current_intensity": overall_intensity,
                "target_intensity": target_intensity,
                "gap_ratio": target_intensity / (overall_intensity + 1e-6),
                "target_met": overall_intensity >= target_intensity
            }

        return comparison


# 🎓 EDUCATIONAL: Utility functions for compute intensity optimization
def print_compute_analysis(analysis_results: Dict[str, Any]) -> None:
    """Print compute intensity analysis in a readable format."""
    print("🚀 Compute Intensity Analysis Results\n")

    # Overall intensity
    overall = analysis_results.get("overall_intensity", 0)
    print(f"📊 Overall Arithmetic Intensity: {overall:.2f} FLOP/byte")

    if overall < 1.0:
        print("   ⚠️  MEMORY-BOUND: Priority optimization target")
    elif overall < 10.0:
        print("   ⚖️  BALANCED: Good optimization potential")
    else:
        print("   🚀 COMPUTE-BOUND: Focus on compute optimizations")
    print()

    # Distribution analysis
    if "intensity_distribution" in analysis_results:
        dist = analysis_results["intensity_distribution"]
        total_layers = sum(dist.values())
        if total_layers > 0:
            print("📈 Layer Intensity Distribution:")
            print(f"   Memory-bound: {dist['memory_bound']} layers ({dist['memory_bound']/total_layers*100:.1f}%)")
            print(f"   Balanced: {dist['balanced']} layers ({dist['balanced']/total_layers*100:.1f}%)")
            print(f"   Compute-bound: {dist['compute_bound']} layers ({dist['compute_bound']/total_layers*100:.1f}%)")
            print()

    # Optimization opportunities
    opportunities = analysis_results.get("optimization_opportunities", [])
    if opportunities:
        print(f"🎯 Found {len(opportunities)} Optimization Opportunities:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. {opp.get('type', 'Unknown')}")
            print(f"     Layers: {', '.join(opp.get('layers', []))}")
            print(f"     Potential improvement: {opp.get('potential_improvement', 'N/A')}")
            print(f"     Technique: {opp.get('technique', 'N/A')}")
        print()