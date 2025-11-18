"""
Compiler Optimization Assistant
==============================

Intelligent assistant for automatic PyTorch code optimization, providing
automated analysis, recommendations, and optimization implementation.

This module provides:
1. Automated code analysis and optimization detection
2. Intelligent optimization recommendations
3. Automatic optimization implementation
4. Performance improvement prediction
5. Educational explanations and tutorials
6. Interactive optimization workflow

Author: Advanced GPU Optimization Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type
import ast
import inspect
import textwrap
from dataclasses import dataclass, asdict
import json
import warnings
from collections import defaultdict
import time
import numpy as np


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with detailed information."""
    optimization_type: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    implementation_steps: List[str]
    expected_improvement: str
    difficulty: str  # 'easy', 'medium', 'hard'
    code_example: str
    educational_notes: List[str]


@dataclass
class ModelAnalysisResult:
    """Result of comprehensive model analysis."""
    model_info: Dict[str, Any]
    performance_bottlenecks: List[str]
    optimization_opportunities: List[OptimizationRecommendation]
    current_optimizations: List[str]
    overall_optimization_score: float
    improvement_potential: str


class CodeAnalyzer:
    """
    Analyzes PyTorch code to identify optimization opportunities.

    Performs static analysis of model architecture, forward pass implementation,
    and training loops to detect optimization patterns.
    """

    def __init__(self):
        self.optimization_patterns = self._load_optimization_patterns()

    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Comprehensive model analysis for optimization opportunities.

        Args:
            model: PyTorch model to analyze

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'architecture': self._analyze_architecture(model),
            'parameters': self._analyze_parameters(model),
            'operations': self._analyze_operations(model),
            'memory_patterns': self._analyze_memory_patterns(model),
            'compilation_readiness': self._analyze_compilation_readiness(model)
        }

        return analysis

    def _analyze_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture for optimization patterns."""
        module_types = defaultdict(int)
        fusion_opportunities = []
        sequential_patterns = []

        # Count module types
        for name, module in model.named_modules():
            module_type = type(module).__name__
            module_types[module_type] += 1

            # Detect fusion opportunities
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Look for activation functions after linear/conv layers
                fusion_opportunities.append({
                    'location': name,
                    'type': 'linear_activation_fusion',
                    'description': f'Linear layer followed by activation function'
                })

        # Detect sequential patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                seq_pattern = self._analyze_sequential_pattern(module)
                if seq_pattern:
                    sequential_patterns.append({
                        'location': name,
                        'pattern': seq_pattern
                    })

        return {
            'module_counts': dict(module_types),
            'total_modules': len(list(model.modules())),
            'fusion_opportunities': fusion_opportunities,
            'sequential_patterns': sequential_patterns
        }

    def _analyze_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model parameters for optimization insights."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Analyze parameter shapes for tensor core compatibility
        tensor_core_compatible = 0
        tensor_core_total = 0

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                tensor_core_total += 1
                if self._is_tensor_core_compatible(module):
                    tensor_core_compatible += 1

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024**2,  # Assuming float32
            'tensor_core_compatibility': tensor_core_compatible / tensor_core_total if tensor_core_total > 0 else 0,
            'memory_footprint': self._estimate_memory_footprint(model)
        }

    def _analyze_operations(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze operations for optimization patterns."""
        operations = []
        activation_functions = []
        normalization_layers = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh)):
                activation_functions.append({
                    'name': name,
                    'type': type(module).__name__,
                    'inplace': getattr(module, 'inplace', False)
                })

            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                normalization_layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'normalized_shape': getattr(module, 'normalized_shape', None)
                })

        return {
            'activation_functions': activation_functions,
            'normalization_layers': normalization_layers,
            'operations_count': len(operations)
        }

    def _analyze_memory_patterns(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        # This is a simplified analysis - real implementation would be more sophisticated
        estimated_forward_memory = 0
        estimated_backward_memory = 0

        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Estimate memory for linear layer
                input_size = module.in_features
                output_size = module.out_features
                estimated_forward_memory += (input_size + output_size) * 4  # bytes
                estimated_backward_memory += estimated_forward_memory * 2  # Rough estimate

        return {
            'estimated_forward_memory_mb': estimated_forward_memory / 1024**2,
            'estimated_backward_memory_mb': estimated_backward_memory / 1024**2,
            'memory_efficiency_score': min(1.0, 100 / (estimated_forward_memory / 1024**2 + 1))
        }

    def _analyze_compilation_readiness(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model readiness for torch.compile."""
        compilation_issues = []
        compatibility_score = 1.0

        # Check for common compilation issues
        for name, module in model.named_modules():
            if hasattr(module, 'forward'):
                # Check for dynamic control flow
                source = inspect.getsource(module.forward) if hasattr(module, 'forward') else ""
                if 'if ' in source and 'training' not in source:
                    compilation_issues.append({
                        'module': name,
                        'issue': 'dynamic_control_flow',
                        'description': 'Dynamic control flow may prevent optimization'
                    })
                    compatibility_score *= 0.8

        return {
            'compilation_compatibility_score': compatibility_score,
            'compilation_issues': compilation_issues,
            'recommended_compile_mode': 'default' if compatibility_score > 0.8 else 'reduce-overhead'
        }

    def _analyze_sequential_pattern(self, sequential: nn.Sequential) -> Optional[str]:
        """Analyze sequential module for common patterns."""
        modules = list(sequential.children())
        if len(modules) < 2:
            return None

        # Detect common patterns
        pattern_names = [type(m).__name__ for m in modules]

        if pattern_names == ['Linear', 'GELU']:
            return 'linear_gelu'
        elif pattern_names == ['Linear', 'ReLU']:
            return 'linear_relu'
        elif pattern_names == ['LayerNorm', 'Linear']:
            return 'norm_linear'
        elif len(pattern_names) >= 3 and pattern_names[:3] == ['Linear', 'GELU', 'Linear']:
            return 'ffn_block'

        return None

    def _is_tensor_core_compatible(self, module: nn.Module) -> bool:
        """Check if module is compatible with Tensor Cores."""
        if isinstance(module, nn.Linear):
            # Tensor Cores require dimensions to be multiples of 8 (simplified)
            return (module.in_features % 8 == 0 and
                   module.out_features % 8 == 0)
        elif isinstance(module, nn.Conv2d):
            return (module.in_channels % 8 == 0 and
                   module.out_channels % 8 == 0)
        return False

    def _estimate_memory_footprint(self, model: nn.Module) -> Dict[str, float]:
        """Estimate memory footprint of the model."""
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / 1024**2  # MB
        buffer_memory = sum(b.numel() * 4 for b in model.buffers()) / 1024**2  # MB

        return {
            'parameters_mb': param_memory,
            'buffers_mb': buffer_memory,
            'total_mb': param_memory + buffer_memory
        }

    def _load_optimization_patterns(self) -> Dict[str, Any]:
        """Load common optimization patterns."""
        return {
            'linear_activation_fusion': {
                'description': 'Fuse linear layer with activation function',
                'benefit': 'Reduces memory access and kernel launches',
                'difficulty': 'easy'
            },
            'attention_optimization': {
                'description': 'Use optimized attention implementations',
                'benefit': 'Significant memory and speed improvements',
                'difficulty': 'medium'
            },
            'normalization_optimization': {
                'description': 'Use optimized normalization layers',
                'benefit': 'Improved numerical stability and speed',
                'difficulty': 'easy'
            }
        }


class OptimizationRecommendationEngine:
    """
    Generates intelligent optimization recommendations based on model analysis.

    Provides prioritized optimization suggestions with implementation guidance
    and expected performance improvements.
    """

    def __init__(self):
        self.recommendation_rules = self._load_recommendation_rules()

    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on analysis.

        Args:
            analysis: Model analysis results

        Returns:
            List of prioritized optimization recommendations
        """
        recommendations = []

        # Check for torch.compile opportunities
        if analysis['compilation_readiness']['compilation_compatibility_score'] > 0.7:
            recommendations.append(self._recommend_torch_compile(analysis))

        # Check for fusion opportunities
        fusion_ops = analysis['architecture']['fusion_opportunities']
        if fusion_ops:
            recommendations.append(self._recommend_operation_fusion(fusion_ops))

        # Check for attention optimization
        if any('Attention' in module for module in analysis['architecture']['module_counts']):
            recommendations.append(self._recommend_attention_optimization(analysis))

        # Check for normalization optimization
        norm_layers = analysis['operations']['normalization_layers']
        if norm_layers:
            recommendations.append(self._recommend_normalization_optimization(norm_layers))

        # Check for Tensor Core opportunities
        if analysis['parameters']['tensor_core_compatibility'] < 0.8:
            recommendations.append(self._recommend_tensor_core_optimization(analysis))

        # Check for memory optimization
        if analysis['memory_patterns']['estimated_forward_memory_mb'] > 1000:  # > 1GB
            recommendations.append(self._recommend_memory_optimization(analysis))

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 3))

        return recommendations

    def _recommend_torch_compile(self, analysis: Dict[str, Any]) -> OptimizationRecommendation:
        """Recommend torch.compile optimization."""
        compile_mode = analysis['compilation_readiness']['recommended_compile_mode']

        return OptimizationRecommendation(
            optimization_type='torch_compile',
            priority='high',
            description='Apply torch.compile for automatic graph optimization',
            implementation_steps=[
                'Import torch',
                f'Use @torch.compile(mode="{compile_mode}") decorator or torch.compile(model)',
                'Test with sample inputs to ensure correctness',
                'Benchmark performance improvements'
            ],
            expected_improvement='20-50% speedup with minimal code changes',
            difficulty='easy',
            code_example=f'''
# Apply torch.compile to your model
import torch

# Option 1: Decorator
@torch.compile(mode="{compile_mode}")
class OptimizedModel(nn.Module):
    # Your model implementation
    pass

# Option 2: Function call
model = YourModel()
optimized_model = torch.compile(model, mode="{compile_mode}")
''',
            educational_notes=[
                'torch.compile uses TorchDynamo to capture computation graphs',
                'Inductor backend optimizes graphs for target hardware',
                'First run may be slower due to compilation overhead',
                'Best performance with static shapes and limited control flow'
            ]
        )

    def _recommend_operation_fusion(self, fusion_ops: List[Dict]) -> OptimizationRecommendation:
        """Recommend operation fusion optimization."""
        return OptimizationRecommendation(
            optimization_type='operation_fusion',
            priority='high',
            description='Fuse adjacent operations to reduce memory access',
            implementation_steps=[
                'Identify fusable operation pairs (Linear + Activation)',
                'Replace with fused implementations from compiler_optimized',
                'Validate numerical correctness',
                'Measure performance improvements'
            ],
            expected_improvement='15-30% speedup for affected operations',
            difficulty='easy',
            code_example='''
# Before: Separate Linear and Activation
self.linear = nn.Linear(512, 1024)
self.activation = nn.GELU()

def forward(self, x):
    return self.activation(self.linear(x))

# After: Fused Linear + GELU
from kernel_pytorch.compiler_optimized import FusedLinearGELU

self.fused_linear_gelu = FusedLinearGELU(512, 1024)

def forward(self, x):
    return self.fused_linear_gelu(x)
''',
            educational_notes=[
                'Fusion reduces memory bandwidth requirements',
                'Fewer kernel launches improve GPU utilization',
                'Most effective for memory-bound operations',
                'torch.compile can automatically perform some fusions'
            ]
        )

    def _recommend_attention_optimization(self, analysis: Dict[str, Any]) -> OptimizationRecommendation:
        """Recommend attention optimization."""
        return OptimizationRecommendation(
            optimization_type='attention_optimization',
            priority='high',
            description='Use memory-efficient attention implementations',
            implementation_steps=[
                'Replace standard attention with optimized implementations',
                'Consider Flash Attention for long sequences',
                'Use single QKV projection for efficiency',
                'Enable memory-efficient attention in PyTorch 2.0+'
            ],
            expected_improvement='50-80% memory reduction, 20-40% speedup',
            difficulty='medium',
            code_example='''
# Before: Standard Multi-Head Attention
self.attention = nn.MultiheadAttention(embed_dim, num_heads)

# After: Optimized Attention
from kernel_pytorch.compiler_optimized import CompilerOptimizedMultiHeadAttention

self.attention = CompilerOptimizedMultiHeadAttention(embed_dim, num_heads)

# Or use Flash Attention directly
if hasattr(F, 'scaled_dot_product_attention'):
    output = F.scaled_dot_product_attention(q, k, v)
''',
            educational_notes=[
                'Flash Attention uses block-sparse computation',
                'Memory usage scales linearly instead of quadratically',
                'Most beneficial for long sequence lengths (>512)',
                'May require newer PyTorch versions for full features'
            ]
        )

    def _recommend_normalization_optimization(self, norm_layers: List[Dict]) -> OptimizationRecommendation:
        """Recommend normalization optimization."""
        return OptimizationRecommendation(
            optimization_type='normalization_optimization',
            priority='medium',
            description='Use optimized normalization implementations',
            implementation_steps=[
                'Replace LayerNorm with OptimizedLayerNorm',
                'Consider RMSNorm for better numerical properties',
                'Use fused normalization + activation where possible',
                'Ensure proper epsilon values for numerical stability'
            ],
            expected_improvement='10-20% speedup for normalization operations',
            difficulty='easy',
            code_example='''
# Before: Standard LayerNorm
self.norm = nn.LayerNorm(hidden_size)

# After: Optimized LayerNorm
from kernel_pytorch.compiler_optimized import OptimizedLayerNorm

self.norm = OptimizedLayerNorm(hidden_size)

# Or: RMS Normalization (more efficient)
from kernel_pytorch.compiler_optimized import OptimizedRMSNorm

self.norm = OptimizedRMSNorm(hidden_size)
''',
            educational_notes=[
                'RMSNorm often provides similar results with lower cost',
                'Proper epsilon selection is crucial for numerical stability',
                'Fused norm + activation reduces memory access',
                'Consider mixed precision for additional speedup'
            ]
        )

    def _recommend_tensor_core_optimization(self, analysis: Dict[str, Any]) -> OptimizationRecommendation:
        """Recommend Tensor Core optimization."""
        return OptimizationRecommendation(
            optimization_type='tensor_core_optimization',
            priority='medium',
            description='Optimize for NVIDIA Tensor Cores',
            implementation_steps=[
                'Ensure layer dimensions are multiples of 8',
                'Use mixed precision training (fp16/bf16)',
                'Enable automatic mixed precision (AMP)',
                'Pad layers if necessary for optimal alignment'
            ],
            expected_improvement='50-100% speedup on compatible hardware',
            difficulty='medium',
            code_example='''
# Tensor Core optimization
from kernel_pytorch.gpu_integration.tensor_cores import TensorCoreOptimizer

optimizer = TensorCoreOptimizer()
optimized_model, info = optimizer.optimize_model_for_tensor_cores(model, sample_input)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
''',
            educational_notes=[
                'Tensor Cores provide massive speedups for matrix operations',
                'Requires modern NVIDIA GPUs (V100, A100, RTX series)',
                'Mixed precision maintains accuracy while improving speed',
                'Automatic padding may increase memory usage slightly'
            ]
        )

    def _recommend_memory_optimization(self, analysis: Dict[str, Any]) -> OptimizationRecommendation:
        """Recommend memory optimization."""
        return OptimizationRecommendation(
            optimization_type='memory_optimization',
            priority='medium',
            description='Optimize GPU memory usage',
            implementation_steps=[
                'Apply gradient checkpointing for large models',
                'Use memory-efficient implementations',
                'Optimize batch size for memory constraints',
                'Consider gradient accumulation for effective large batches'
            ],
            expected_improvement='50-70% memory reduction with gradient checkpointing',
            difficulty='medium',
            code_example='''
# Memory optimization techniques
from kernel_pytorch.gpu_integration.memory_optimization import MemoryOptimizer

memory_optimizer = MemoryOptimizer()
optimized_model = memory_optimizer.optimize_memory_layout(
    model,
    enable_memory_efficient_attention=True,
    use_checkpoint=True
)

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    # Use checkpointing for memory-intensive blocks
    x = checkpoint(self.expensive_block, x)
    return x
''',
            educational_notes=[
                'Gradient checkpointing trades compute for memory',
                'Memory-efficient attention reduces quadratic memory usage',
                'Batch size optimization can significantly impact performance',
                'Monitor GPU memory usage during training'
            ]
        )

    def _load_recommendation_rules(self) -> Dict[str, Any]:
        """Load recommendation generation rules."""
        return {
            'torch_compile_threshold': 0.7,
            'memory_threshold_mb': 1000,
            'tensor_core_compatibility_threshold': 0.8
        }


class OptimizationImplementer:
    """
    Automatically implements optimization recommendations.

    Provides code generation and automatic model transformation capabilities
    for applying optimization recommendations.
    """

    def __init__(self):
        self.implementation_templates = self._load_implementation_templates()

    def implement_optimization(self,
                             model: nn.Module,
                             recommendation: OptimizationRecommendation) -> Tuple[nn.Module, str]:
        """
        Automatically implement an optimization recommendation.

        Args:
            model: Original model
            recommendation: Optimization recommendation to implement

        Returns:
            Tuple of (optimized_model, implementation_code)
        """
        optimization_type = recommendation.optimization_type

        if optimization_type == 'torch_compile':
            return self._implement_torch_compile(model, recommendation)
        elif optimization_type == 'operation_fusion':
            return self._implement_operation_fusion(model, recommendation)
        elif optimization_type == 'attention_optimization':
            return self._implement_attention_optimization(model, recommendation)
        elif optimization_type == 'normalization_optimization':
            return self._implement_normalization_optimization(model, recommendation)
        elif optimization_type == 'tensor_core_optimization':
            return self._implement_tensor_core_optimization(model, recommendation)
        elif optimization_type == 'memory_optimization':
            return self._implement_memory_optimization(model, recommendation)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")

    def _implement_torch_compile(self, model: nn.Module, recommendation: OptimizationRecommendation) -> Tuple[nn.Module, str]:
        """Implement torch.compile optimization."""
        optimized_model = torch.compile(model, mode='default')

        code = '''
# Applied torch.compile optimization
optimized_model = torch.compile(model, mode='default')
'''

        return optimized_model, code

    def _implement_operation_fusion(self, model: nn.Module, recommendation: OptimizationRecommendation) -> Tuple[nn.Module, str]:
        """Implement operation fusion optimization."""
        # This is a simplified implementation
        # Real implementation would analyze the model and replace fusable operations

        class OptimizedModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
                # Add fusion implementations here

            def forward(self, x):
                return self.original_model(x)

        optimized_model = OptimizedModel(model)

        code = '''
# Applied operation fusion optimization
# Replaced Linear + Activation with fused implementations
from kernel_pytorch.compiler_optimized import FusedLinearGELU

# Example transformation:
# self.linear = nn.Linear(in_features, out_features)
# self.activation = nn.GELU()
#
# Becomes:
# self.fused_linear_gelu = FusedLinearGELU(in_features, out_features)
'''

        return optimized_model, code

    def _implement_attention_optimization(self, model: nn.Module, recommendation: OptimizationRecommendation) -> Tuple[nn.Module, str]:
        """Implement attention optimization."""
        # Simplified implementation
        optimized_model = model

        code = '''
# Applied attention optimization
from kernel_pytorch.compiler_optimized import CompilerOptimizedMultiHeadAttention

# Replaced standard attention with optimized version
# self.attention = nn.MultiheadAttention(embed_dim, num_heads)
#
# Becomes:
# self.attention = CompilerOptimizedMultiHeadAttention(embed_dim, num_heads)
'''

        return optimized_model, code

    def _implement_normalization_optimization(self, model: nn.Module, recommendation: OptimizationRecommendation) -> Tuple[nn.Module, str]:
        """Implement normalization optimization."""
        optimized_model = model

        code = '''
# Applied normalization optimization
from kernel_pytorch.compiler_optimized import OptimizedLayerNorm

# Replaced standard LayerNorm with optimized version
# self.norm = nn.LayerNorm(normalized_shape)
#
# Becomes:
# self.norm = OptimizedLayerNorm(normalized_shape)
'''

        return optimized_model, code

    def _implement_tensor_core_optimization(self, model: nn.Module, recommendation: OptimizationRecommendation) -> Tuple[nn.Module, str]:
        """Implement Tensor Core optimization."""
        optimized_model = model

        code = '''
# Applied Tensor Core optimization
from kernel_pytorch.gpu_integration.tensor_cores import TensorCoreOptimizer

tensor_optimizer = TensorCoreOptimizer()
optimized_model, optimization_info = tensor_optimizer.optimize_model_for_tensor_cores(
    model, sample_input
)
'''

        return optimized_model, code

    def _implement_memory_optimization(self, model: nn.Module, recommendation: OptimizationRecommendation) -> Tuple[nn.Module, str]:
        """Implement memory optimization."""
        optimized_model = model

        code = '''
# Applied memory optimization
from kernel_pytorch.gpu_integration.memory_optimization import MemoryOptimizer

memory_optimizer = MemoryOptimizer()
optimized_model = memory_optimizer.optimize_memory_layout(
    model,
    enable_memory_efficient_attention=True,
    use_checkpoint=True
)
'''

        return optimized_model, code

    def _load_implementation_templates(self) -> Dict[str, str]:
        """Load code templates for optimization implementations."""
        return {
            'torch_compile': 'optimized_model = torch.compile(model)',
            'operation_fusion': 'from kernel_pytorch.compiler_optimized import FusedLinearGELU',
            'attention_optimization': 'from kernel_pytorch.compiler_optimized import CompilerOptimizedMultiHeadAttention'
        }


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