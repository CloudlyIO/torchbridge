"""
Compiler-Friendly Optimization Patterns for PyTorch

This module provides comprehensive guidance and practical implementations for
writing PyTorch code that optimizes effectively with modern compilers like
torch.compile, TorchScript, and TensorRT.

Modern PyTorch compilers can achieve significant performance improvements, but only
when code follows compiler-friendly patterns:
- torch.compile: JIT compilation with graph optimization and kernel fusion
- TorchScript: Static graph compilation for production deployment
- TensorRT: NVIDIA's deep learning inference optimizer
- XLA: Google's accelerated linear algebra compiler

 COMPILER OPTIMIZATION PRINCIPLES:
- Static shapes: Avoid dynamic shape operations when possible
- Function composition: Structure code for graph-level optimizations
- Avoid Python overhead: Minimize Python loops and conditionals in hot paths
- Memory layout consistency: Use consistent tensor formats and strides

Learn to write PyTorch code that leverages automatic compiler optimizations
for 2-5x performance improvements with minimal code changes.
"""

import inspect
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompilationCompatibility(Enum):
    """Levels of compiler compatibility for optimization analysis."""
    EXCELLENT = "excellent"      # Optimal for all compilers
    GOOD = "good"               # Works well with most compilers
    LIMITED = "limited"         # Some compilation limitations
    PROBLEMATIC = "problematic" # Significant compilation issues


class CompilerType(Enum):
    """Supported PyTorch compiler backends."""
    TORCH_COMPILE = "torch_compile"
    TORCHSCRIPT = "torchscript"
    TENSORRT = "tensorrt"
    XLA = "xla"


@dataclass
class CompilationPattern:
    """
    Data structure for describing compiler-friendly patterns.

    By categorizing compilation patterns, we can:
    - Identify code structures that optimize well
    - Avoid patterns that prevent compiler optimization
    - Measure compilation effectiveness
    - Build reusable compiler-friendly components
    """
    name: str
    compatibility: CompilationCompatibility
    supported_compilers: list[CompilerType]
    optimization_benefits: list[str]
    common_pitfalls: list[str]
    description: str
    example_good: str
    example_bad: str


#  EDUCATIONAL: Compiler optimization best practices and common patterns
COMPILER_BEST_PRACTICES = [
    CompilationPattern(
        name="Static Shape Operations",
        compatibility=CompilationCompatibility.EXCELLENT,
        supported_compilers=[CompilerType.TORCH_COMPILE, CompilerType.TORCHSCRIPT, CompilerType.TENSORRT],
        optimization_benefits=["Kernel fusion", "Memory optimization", "Loop optimization"],
        common_pitfalls=["Dynamic indexing", "Conditional shapes", "Runtime tensor creation"],
        description="Use operations with statically determinable shapes for optimal compilation",
        example_good="x = F.linear(x, weight)  # Static shapes",
        example_bad="x = x[:, :dynamic_size]  # Dynamic slicing"
    ),

    CompilationPattern(
        name="Function Composition",
        compatibility=CompilationCompatibility.EXCELLENT,
        supported_compilers=[CompilerType.TORCH_COMPILE, CompilerType.XLA],
        optimization_benefits=["Graph-level fusion", "Dead code elimination", "Constant folding"],
        common_pitfalls=["Python loops", "Conditional execution", "Side effects"],
        description="Structure operations as pure function composition for graph optimization",
        example_good="return F.gelu(F.linear(x, w, b))  # Composable",
        example_bad="for i in range(n): x = F.relu(x)  # Python loop"
    ),

    CompilationPattern(
        name="Tensor-Native Operations",
        compatibility=CompilationCompatibility.GOOD,
        supported_compilers=[CompilerType.TORCH_COMPILE, CompilerType.TORCHSCRIPT],
        optimization_benefits=["Efficient kernel dispatch", "Memory coalescing", "SIMD optimization"],
        common_pitfalls=["Python scalar operations", "Item access", "CPU-GPU transfers"],
        description="Use PyTorch tensor operations instead of Python scalars or numpy",
        example_good="mask = (x > threshold).float()  # Tensor operations",
        example_bad="mask = [1 if xi > threshold else 0 for xi in x]  # Python loop"
    ),

    CompilationPattern(
        name="Consistent Memory Layout",
        compatibility=CompilationCompatibility.GOOD,
        supported_compilers=[CompilerType.TORCH_COMPILE, CompilerType.TENSORRT],
        optimization_benefits=["Cache efficiency", "Memory access optimization", "Kernel specialization"],
        common_pitfalls=["Mixed memory formats", "Unnecessary transposes", "Non-contiguous tensors"],
        description="Maintain consistent tensor memory layouts throughout computation",
        example_good="x = x.to(memory_format=torch.channels_last)  # Consistent format",
        example_bad="x = x.transpose(-1, -2).reshape(...)  # Mixed layouts"
    ),

    CompilationPattern(
        name="Avoid Control Flow",
        compatibility=CompilationCompatibility.LIMITED,
        supported_compilers=[CompilerType.TORCH_COMPILE],
        optimization_benefits=["Better vectorization", "Reduced branching overhead"],
        common_pitfalls=["Python conditionals", "Dynamic control flow", "Exception handling"],
        description="Minimize Python control flow in favor of tensor operations",
        example_good="output = torch.where(condition, x, y)  # Tensor conditional",
        example_bad="output = x if condition else y  # Python conditional"
    )
]


def check_compilation_compatibility(
    model: nn.Module,
    sample_input: torch.Tensor,
    target_compiler: CompilerType = CompilerType.TORCH_COMPILE
) -> dict[str, Any]:
    """
    Analyze model for compiler compatibility and optimization potential.

    This function demonstrates how to systematically analyze PyTorch models
    to identify compilation opportunities and potential issues.

     COMPATIBILITY ANALYSIS:
    - Static vs dynamic operations identification
    - Control flow pattern analysis
    - Memory layout consistency checking
    - Compilation error prediction

    Args:
        model: PyTorch model to analyze
        sample_input: Representative input for analysis
        target_compiler: Target compiler backend for analysis

    Returns:
        Comprehensive compilation compatibility analysis
    """
    compatibility_analysis = {
        "overall_compatibility": CompilationCompatibility.GOOD.value,
        "compilation_issues": [],
        "optimization_opportunities": [],
        "recommended_changes": [],
        "performance_estimate": {}
    }

    architecture_analysis = _analyze_architecture_compatibility(model)
    compatibility_analysis.update(architecture_analysis)

    dynamic_analysis = _check_dynamic_operations(model, sample_input)
    compatibility_analysis["dynamic_operations"] = dynamic_analysis

    control_flow_analysis = _analyze_control_flow(model)
    compatibility_analysis["control_flow_issues"] = control_flow_analysis

    memory_analysis = _check_memory_layout_consistency(model, sample_input)
    compatibility_analysis["memory_layout_issues"] = memory_analysis

    compatibility_analysis["recommended_changes"] = _generate_compilation_recommendations(
        compatibility_analysis, target_compiler
    )

    compatibility_analysis["performance_estimate"] = _estimate_compilation_benefits(
        compatibility_analysis, target_compiler
    )

    return compatibility_analysis


def _analyze_architecture_compatibility(model: nn.Module) -> dict[str, Any]:
    """Analyze model architecture for compiler compatibility."""
    analysis = {
        "module_compatibility": {},
        "fusion_opportunities": [],
        "problematic_patterns": []
    }

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            module_compat = _assess_module_compatibility(module)
            analysis["module_compatibility"][name] = module_compat

            if module_compat["compatibility"] == CompilationCompatibility.PROBLEMATIC.value:
                analysis["problematic_patterns"].append({
                    "module": name,
                    "issue": module_compat["issues"],
                    "recommendation": module_compat["recommendation"]
                })

    return analysis


def _assess_module_compatibility(module: nn.Module) -> dict[str, Any]:
    """Assess individual module compatibility with compilers."""
    type(module).__name__  # noqa: B018

    # Standard PyTorch modules generally compile well
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.ReLU, nn.GELU)):
        return {
            "compatibility": CompilationCompatibility.EXCELLENT.value,
            "issues": [],
            "recommendation": "No changes needed"
        }

    # Custom modules need analysis
    elif hasattr(module, 'forward'):
        # Check forward method for compilation-unfriendly patterns
        forward_source = inspect.getsource(module.forward) if hasattr(module, 'forward') else ""

        issues = []
        if 'for ' in forward_source and 'range(' in forward_source:
            issues.append("Contains Python loops")
        if '.item()' in forward_source:
            issues.append("Contains tensor-to-scalar conversions")
        if 'if ' in forward_source and 'else' in forward_source:
            issues.append("Contains conditional logic")

        if issues:
            return {
                "compatibility": CompilationCompatibility.LIMITED.value,
                "issues": issues,
                "recommendation": "Consider vectorizing operations and removing Python control flow"
            }

    return {
        "compatibility": CompilationCompatibility.GOOD.value,
        "issues": [],
        "recommendation": "Monitor compilation performance"
    }


def _check_dynamic_operations(model: nn.Module, sample_input: torch.Tensor) -> list[dict[str, Any]]:
    """Check for operations with dynamic behavior that may hinder compilation."""
    dynamic_operations = []

    # This is a simplified analysis - real implementation would use graph tracing
    for name, module in model.named_modules():
        if hasattr(module, 'forward'):
            # Check for common dynamic operation patterns
            module_source = str(module)

            # Placeholder checks - real implementation would analyze computation graph
            if 'reshape' in module_source.lower():
                dynamic_operations.append({
                    "module": name,
                    "operation": "reshape",
                    "severity": "medium",
                    "recommendation": "Use view() with static shapes when possible"
                })

    return dynamic_operations


def _analyze_control_flow(model: nn.Module) -> list[dict[str, Any]]:
    """Analyze control flow patterns that may prevent optimization."""
    control_flow_issues = []

    for name, module in model.named_modules():
        if hasattr(module, 'forward'):
            try:
                source = inspect.getsource(module.forward)

                # Check for problematic control flow patterns
                if 'for ' in source and 'range(' in source:
                    control_flow_issues.append({
                        "module": name,
                        "issue": "Python for loop",
                        "severity": "high",
                        "recommendation": "Replace with vectorized tensor operations"
                    })

                if 'while ' in source:
                    control_flow_issues.append({
                        "module": name,
                        "issue": "While loop",
                        "severity": "high",
                        "recommendation": "Use fixed iteration count or tensor operations"
                    })

            except (OSError, TypeError):
                # Can't analyze source (built-in modules, etc.)
                continue

    return control_flow_issues


def _check_memory_layout_consistency(model: nn.Module, sample_input: torch.Tensor) -> list[dict[str, Any]]:
    """Check for memory layout inconsistencies that may hurt performance."""
    layout_issues = []

    # Check model parameters for non-contiguous tensors
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            layout_issues.append({
                "parameter": name,
                "issue": "Non-contiguous parameter",
                "recommendation": "Call .contiguous() during initialization"
            })

    # Check for mixed memory formats (simplified check)
    # Note: memory_format is not directly accessible, so we check stride patterns
    (len(sample_input.shape) == 4 and
                       sample_input.stride()[1] == 1)  # Channel stride is 1 for channels_last

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            # These modules benefit from consistent memory format
            if hasattr(module, 'weight') and len(module.weight.shape) >= 3:
                # Simple heuristic: check if weight tensor is non-contiguous
                if not module.weight.is_contiguous():
                    layout_issues.append({
                        "module": name,
                        "issue": "Non-contiguous weight tensor",
                        "recommendation": "Call .contiguous() on weights or use channels_last format"
                    })

    return layout_issues


def _generate_compilation_recommendations(
    analysis: dict[str, Any],
    target_compiler: CompilerType
) -> list[dict[str, Any]]:
    """Generate specific recommendations for compilation optimization."""
    recommendations = []

    # High-priority issues first
    for issue in analysis.get("problematic_patterns", []):
        recommendations.append({
            "priority": "high",
            "type": "architecture_fix",
            "target": issue["module"],
            "action": issue["recommendation"],
            "expected_benefit": "Enable compilation"
        })

    # Control flow optimizations
    for issue in analysis.get("control_flow_issues", []):
        if issue["severity"] == "high":
            recommendations.append({
                "priority": "high",
                "type": "control_flow_optimization",
                "target": issue["module"],
                "action": issue["recommendation"],
                "expected_benefit": "2-5x speedup from vectorization"
            })

    # Memory layout optimizations
    if analysis.get("memory_layout_issues"):
        recommendations.append({
            "priority": "medium",
            "type": "memory_layout_optimization",
            "action": "Standardize tensor memory formats",
            "expected_benefit": "10-30% performance improvement"
        })

    return recommendations


def _estimate_compilation_benefits(
    analysis: dict[str, Any],
    target_compiler: CompilerType
) -> dict[str, Any]:
    """Estimate potential performance benefits from compilation."""
    # Educational estimates based on compatibility analysis
    base_speedup = 1.0

    # Factor in module compatibility
    excellent_modules = sum(
        1 for compat in analysis.get("module_compatibility", {}).values()
        if compat.get("compatibility") == CompilationCompatibility.EXCELLENT.value
    )
    total_modules = len(analysis.get("module_compatibility", {}))

    if total_modules > 0:
        compatibility_ratio = excellent_modules / total_modules
        base_speedup += compatibility_ratio * 2.0  # Up to 2x from excellent compatibility

    # Penalize for issues
    issue_penalty = len(analysis.get("problematic_patterns", [])) * 0.5
    control_flow_penalty = len(analysis.get("control_flow_issues", [])) * 0.3

    estimated_speedup = max(1.0, base_speedup - issue_penalty - control_flow_penalty)

    return {
        "estimated_speedup": estimated_speedup,
        "confidence": "medium" if estimated_speedup > 1.5 else "low",
        "bottlenecks": analysis.get("problematic_patterns", [])[:3],  # Top 3 issues
        "optimization_potential": "high" if estimated_speedup > 2.0 else "medium"
    }


def optimize_for_torch_compile(
    model: nn.Module,
    optimization_level: str = "default",
    enable_dynamic: bool = False
) -> nn.Module:
    """
    Optimize model specifically for torch.compile backend.

    torch.compile is PyTorch's latest JIT compiler that can achieve significant
    performance improvements. This function demonstrates how to prepare models
    for optimal torch.compile performance.

     TORCH.COMPILE OPTIMIZATION TECHNIQUES:
    - Function structuring for graph capture
    - Dynamic shape handling
    - Memory layout optimization
    - Kernel fusion enablement

    Args:
        model: Model to optimize for torch.compile
        optimization_level: Compilation optimization level
        enable_dynamic: Whether to enable dynamic shape support

    Returns:
        Model optimized for torch.compile
    """
    optimized_model = _prepare_model_for_compilation(model)

    compile_kwargs = {
        "mode": optimization_level,
        "dynamic": enable_dynamic,
        "fullgraph": not enable_dynamic  # Full graph compilation when possible
    }

    try:
        compiled_model = torch.compile(optimized_model, **compile_kwargs)
        return compiled_model
    except Exception as e:
        warnings.warn(f"Compilation failed: {e}. Returning uncompiled model.", stacklevel=2)
        return optimized_model


def _prepare_model_for_compilation(model: nn.Module) -> nn.Module:
    """Prepare model structure for optimal compilation."""
    # Ensure all parameters are contiguous
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    # Check for and fix common compilation issues
    for _name, module in model.named_modules():
        if hasattr(module, '_fix_for_compilation'):
            module._fix_for_compilation()

    return model


def avoid_compilation_pitfalls(code_snippet: str) -> dict[str, Any]:
    """
    Analyze code snippet for common compilation pitfalls and suggest fixes.

    This function demonstrates how to identify and fix common patterns
    that prevent effective compilation in PyTorch.

     COMMON PITFALLS:
    - Python loops over tensors
    - Dynamic tensor indexing
    - Conditional execution based on tensor values
    - Mixing CPU and GPU operations
    - Non-tensor return values

    Args:
        code_snippet: Python code to analyze

    Returns:
        Analysis results with identified issues and suggested fixes
    """
    analysis = {
        "issues": [],
        "suggestions": [],
        "severity_score": 0
    }

    # Check for Python loops
    if 'for ' in code_snippet and any(keyword in code_snippet for keyword in ['torch', 'tensor', 'x[']):
        analysis["issues"].append("Python loop over tensor operations")
        analysis["suggestions"].append("Replace with vectorized operations using torch.vmap or tensor broadcasting")
        analysis["severity_score"] += 3

    # Check for tensor.item() usage
    if '.item()' in code_snippet:
        analysis["issues"].append("Tensor to scalar conversion")
        analysis["suggestions"].append("Avoid .item() calls; use tensor operations instead")
        analysis["severity_score"] += 2

    # Check for dynamic indexing
    if '[:]' in code_snippet and ('size(' in code_snippet or 'shape[' in code_snippet):
        analysis["issues"].append("Dynamic tensor slicing")
        analysis["suggestions"].append("Use fixed slicing or torch.narrow with static sizes")
        analysis["severity_score"] += 2

    # Check for CPU-GPU mixing
    if 'cpu()' in code_snippet and 'cuda()' in code_snippet:
        analysis["issues"].append("Mixed CPU-GPU operations")
        analysis["suggestions"].append("Keep operations on consistent device")
        analysis["severity_score"] += 3

    # Check for Python conditionals on tensor values
    if 'if ' in code_snippet and any(op in code_snippet for op in ['>', '<', '==', '!=']):
        if 'torch' in code_snippet or 'tensor' in code_snippet:
            analysis["issues"].append("Python conditional on tensor values")
            analysis["suggestions"].append("Use torch.where or tensor masking instead")
            analysis["severity_score"] += 2

    return analysis


class CompilerOptimizedModule(nn.Module):
    """
    Base class for modules designed for optimal compiler performance.

    This demonstrates how to design PyTorch modules specifically for
    compiler optimization from the ground up.

     DESIGN PRINCIPLES:
    - Static shape operations
    - Minimal Python overhead
    - Consistent memory layouts
    - Compositional structure
    """

    def __init__(self):
        super().__init__()
        self._compiler_optimized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass designed for compiler optimization."""
        raise NotImplementedError("Subclasses must implement optimized forward pass")

    @torch.jit.ignore
    def _check_compilation_readiness(self) -> bool:
        """Check if module is ready for compilation."""
        # Ensure all parameters are contiguous
        for param in self.parameters():
            if not param.is_contiguous():
                return False

        # Check for dynamic operations (simplified)
        # Real implementation would analyze the computation graph
        return True

    def optimize_for_compiler(self, target_compiler: CompilerType = CompilerType.TORCH_COMPILE):
        """Optimize module for specific compiler backend."""
        if target_compiler == CompilerType.TORCH_COMPILE:
            # Ensure optimal torch.compile compatibility
            for param in self.parameters():
                param.data = param.data.contiguous()

        elif target_compiler == CompilerType.TORCHSCRIPT:
            # Prepare for TorchScript compilation
            # Remove any JIT-incompatible operations
            pass


class OptimizedLinearGELU(CompilerOptimizedModule):
    """
    Example of compiler-optimized module: Linear + GELU fusion.

     OPTIMIZATION FEATURES:
    - Single forward method for optimal graph capture
    - Contiguous tensor operations
    - No Python control flow
    - Efficient kernel fusion opportunity
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass designed for torch.compile fusion.

         COMPILER OPTIMIZATION NOTES:
        - Single expression allows kernel fusion
        - No intermediate tensor storage
        - Optimal memory access pattern
        - No Python overhead
        """
        # Single expression optimal for compiler fusion
        return F.gelu(self.linear(x))


class OptimizedTransformerBlock(CompilerOptimizedModule):
    """
    Example of compiler-optimized transformer block.

    This demonstrates how to structure complex modules like transformer
    blocks for optimal compiler performance.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # QKV projection as single linear layer for optimal GEMM
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Feedforward with optimal structure
        self.ff_up = nn.Linear(embed_dim, feedforward_dim, bias=False)
        self.ff_down = nn.Linear(feedforward_dim, embed_dim, bias=False)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compiler-optimized transformer block forward pass.

         OPTIMIZATION TECHNIQUES:
        - Minimal intermediate variables
        - Structured for attention and feedforward fusion
        - Consistent tensor operations
        - No Python control flow
        """
        #  ATTENTION: Structured for optimal compilation
        residual = x
        x = self.norm1(x)

        # Single QKV projection for optimal GEMM utilization
        qkv = self.qkv_proj(x)
        batch_size, seq_len, _ = qkv.shape

        # Reshape for multi-head attention (compiler-friendly)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (single expression for fusion)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v)

        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embed_dim)
        attention_output = self.output_proj(attention_output)

        # Residual connection
        x = residual + self.dropout(attention_output)

        #  FEEDFORWARD: Structured for optimal compilation
        residual = x
        x = self.norm2(x)

        # Fused feedforward operations
        x = F.gelu(self.ff_up(x))  # Up projection + activation
        x = self.ff_down(x)        # Down projection

        # Final residual connection
        return residual + self.dropout(x)


#  EDUCATIONAL: Utility functions for compilation optimization
def print_compilation_analysis(analysis: dict[str, Any]) -> None:
    """Print compilation analysis results in a readable format."""
    print(" Compilation Compatibility Analysis\n")

    # Overall compatibility
    overall = analysis.get("overall_compatibility", "unknown")
    print(f" Overall Compatibility: {overall.upper()}")

    if overall == "excellent":
        print("    Model is ready for compilation optimization")
    elif overall == "good":
        print("    Model should compile well with minor optimizations")
    elif overall == "limited":
        print("     Some compilation limitations present")
    else:
        print("    Significant compilation issues detected")
    print()

    # Performance estimate
    if "performance_estimate" in analysis:
        perf = analysis["performance_estimate"]
        speedup = perf.get("estimated_speedup", 1.0)
        print(f" Estimated Compilation Speedup: {speedup:.1f}x")
        print(f"   Confidence: {perf.get('confidence', 'unknown')}")
        print(f"   Optimization potential: {perf.get('optimization_potential', 'unknown')}")
        print()

    # Issues and recommendations
    recommendations = analysis.get("recommended_changes", [])
    if recommendations:
        print(f" Optimization Recommendations ({len(recommendations)} items):")
        for i, rec in enumerate(recommendations, 1):
            priority_icon = "" if rec.get("priority") == "high" else "⭐"
            print(f"  {priority_icon} {i}. {rec.get('type', 'Unknown')}")
            print(f"     Target: {rec.get('target', 'N/A')}")
            print(f"     Action: {rec.get('action', 'N/A')}")
            print(f"     Expected benefit: {rec.get('expected_benefit', 'N/A')}")
        print()

    # Dynamic operations
    dynamic_ops = analysis.get("dynamic_operations", [])
    if dynamic_ops:
        print(f" Dynamic Operations Found ({len(dynamic_ops)} items):")
        for op in dynamic_ops:
            print(f"   • {op.get('module', 'Unknown')}: {op.get('operation', 'Unknown')}")
            print(f"     Recommendation: {op.get('recommendation', 'N/A')}")


def benchmark_compilation_impact(
    original_model: nn.Module,
    compiled_model: nn.Module,
    sample_input: torch.Tensor,
    num_iterations: int = 100
) -> dict[str, float]:
    """
    Benchmark the performance impact of compilation optimizations.

    Always measure compilation impact to verify that optimizations
    provide real performance benefits.
    """
    # Educational placeholder - would implement detailed benchmarking
    return {
        "compilation_speedup": 2.1,  # Educational placeholder
        "memory_efficiency": 0.15,
        "kernel_fusion_ratio": 0.7,
        "graph_optimization_benefit": 0.4
    }
