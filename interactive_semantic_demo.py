#!/usr/bin/env python3
"""
Interactive Semantic Code Understanding Demo

This demo provides an interactive interface for exploring the semantic
code understanding agent's capabilities on real ML/AI code examples.
Users can input code snippets or select from examples to see how the
agent understands high-level concepts, optimization opportunities, and
educational insights.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from typing import Dict, Any, List
from kernel_pytorch.semantic_agent.architecture import SemanticCodeAgent
from kernel_pytorch.semantic_agent.llm_understanding import EnhancedSemanticAgent, LLMSemanticAnalyzer
from kernel_pytorch.semantic_agent.concept_mapping import ConceptMappingSystem

class InteractiveSemanticDemo:
    """Interactive demo for semantic code understanding"""

    def __init__(self):
        self.basic_agent = SemanticCodeAgent()
        self.enhanced_agent = EnhancedSemanticAgent()
        self.llm_analyzer = LLMSemanticAnalyzer()
        self.mapping_system = ConceptMappingSystem()

        self.example_codes = self._load_example_codes()

    def _load_example_codes(self) -> Dict[str, str]:
        """Load example code snippets for demonstration"""
        examples = {}

        # Basic Attention Implementation
        examples["basic_attention"] = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out
'''

        # Optimized Flash Attention
        examples["flash_attention"] = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.compile
class OptimizedFlashAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # Use PyTorch's optimized Flash Attention
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Optimized attention computation
        with torch.cuda.amp.autocast():
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=False
            )

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)
'''

        # Convolutional Neural Network
        examples["cnn"] = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Depthwise separable convolutions for efficiency
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.dwconv = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pwconv = nn.Conv2d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Standard convolution + activation
        x = F.relu(self.bn1(self.conv1(x)))

        # Depthwise separable convolution
        x = self.dwconv(x)
        x = F.relu(self.bn2(self.pwconv(x)))

        # Global pooling and classification
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)
'''

        # Memory-Optimized Training Code
        examples["memory_optimization"] = '''
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class MemoryEfficientTransformer(nn.Module):
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads)
            for _ in range(depth)
        ])

    def forward(self, x):
        # Use gradient checkpointing for memory efficiency
        for layer in self.layers:
            if self.training:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x

# Training with mixed precision
def train_step(model, data, optimizer, scaler):
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        output = model(data)
        loss = F.cross_entropy(output, targets)

    # Gradient scaling for numerical stability
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
'''

        # Custom CUDA-style Optimization
        examples["custom_kernels"] = '''
import torch
import torch.nn as nn

@torch.jit.script
def fused_layer_norm_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused layer normalization + GELU activation.
    Demonstrates kernel fusion optimization.
    """
    # Efficient layer normalization
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + 1e-5)
    scaled = normalized * weight + bias

    # GELU activation fused in same kernel
    return scaled * 0.5 * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (scaled + 0.044715 * scaled.pow(3))
    ))

class HighlyOptimizedMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, dim)

    @torch.jit.script_method
    def forward(self, x):
        # Use custom fused kernel
        hidden = fused_layer_norm_gelu(x, self.norm.weight, self.norm.bias)
        hidden = self.linear1(hidden)
        return self.linear2(hidden)
'''

        return examples

    def show_welcome_message(self):
        """Display welcome message and instructions"""
        print("🧠 Interactive Semantic Code Understanding Demo")
        print("=" * 65)
        print("Welcome! This demo lets you explore how our AI agent understands")
        print("machine learning and AI code at a semantic level.")
        print()
        print("📋 Available Commands:")
        print("  1. analyze <example_name>  - Analyze pre-loaded examples")
        print("  2. custom                  - Analyze custom code input")
        print("  3. list                    - Show available examples")
        print("  4. explain <concept>       - Get concept explanation")
        print("  5. compare <ex1> <ex2>     - Compare two implementations")
        print("  6. mapping <example_name>  - Show concept mappings")
        print("  7. optimize <example_name> - Get optimization suggestions")
        print("  8. help                    - Show this help message")
        print("  9. quit                    - Exit the demo")
        print()

    def show_examples(self):
        """Show available code examples"""
        print("📚 Available Code Examples:")
        print("-" * 40)
        for i, (name, _) in enumerate(self.example_codes.items(), 1):
            description = self._get_example_description(name)
            print(f"  {i}. {name:20s} - {description}")
        print()

    def _get_example_description(self, example_name: str) -> str:
        """Get description for an example"""
        descriptions = {
            "basic_attention": "Simple attention mechanism implementation",
            "flash_attention": "Optimized attention with Flash Attention",
            "cnn": "Convolutional neural network with optimizations",
            "memory_optimization": "Memory-efficient training techniques",
            "custom_kernels": "Custom kernel fusion optimizations"
        }
        return descriptions.get(example_name, "ML/AI code example")

    def analyze_example(self, example_name: str):
        """Perform comprehensive analysis of an example"""
        if example_name not in self.example_codes:
            print(f"❌ Example '{example_name}' not found. Use 'list' to see available examples.")
            return

        code = self.example_codes[example_name]
        print(f"🔍 Analyzing: {example_name}")
        print("-" * 50)

        # Basic semantic analysis
        print("\n📊 BASIC SEMANTIC ANALYSIS:")
        basic_result = self.basic_agent.analyze_code(code)

        patterns = basic_result.get('patterns', [])
        print(f"   Detected {len(patterns)} semantic patterns:")

        for pattern in patterns[:3]:  # Show top 3
            print(f"   • {pattern['concept']} (confidence: {pattern['confidence']:.2f})")
            print(f"     Location: {pattern['location']}")
            if pattern.get('optimization_potential'):
                print(f"     💡 {pattern['optimization_potential']}")

        # Advanced LLM-style analysis
        print("\n🧠 ADVANCED ANALYSIS:")
        enhanced_result = self.enhanced_agent.deep_code_analysis(code)

        # Show high-level concepts
        concepts = enhanced_result.get('code_analysis', {}).get('high_level_concepts', [])
        if concepts:
            print("   High-level Concepts:")
            for concept in concepts[:3]:
                name = concept.get('concept', 'Unknown')
                confidence = concept.get('confidence', 0)
                print(f"   • {name} (confidence: {confidence:.2f})")

        # Show optimization analysis
        opt_analysis = enhanced_result.get('code_analysis', {}).get('optimization_analysis', {})
        if opt_analysis:
            level = opt_analysis.get('overall_level', 'unknown')
            score = opt_analysis.get('optimization_score', 0)
            print(f"\n⚡ OPTIMIZATION LEVEL: {level.upper()} (score: {score:.2f})")

            optimizations = opt_analysis.get('optimizations', {})
            detected = [opt for opt, data in optimizations.items() if data.get('detected')]
            if detected:
                print(f"   Detected techniques: {', '.join(detected)}")

        print()

    def analyze_custom_code(self):
        """Analyze custom user-provided code"""
        print("✏️  Custom Code Analysis")
        print("-" * 30)
        print("Enter your ML/AI code (end with '###' on a new line):")

        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "###":
                    break
                lines.append(line)
            except KeyboardInterrupt:
                print("\n❌ Input cancelled.")
                return

        if not lines:
            print("❌ No code provided.")
            return

        custom_code = "\n".join(lines)
        print(f"\n🔍 Analyzing your custom code...")
        print("-" * 40)

        # Perform analysis
        basic_result = self.basic_agent.analyze_code(custom_code)
        mappings = self.mapping_system.map_code_to_concepts(custom_code)

        # Show results
        patterns = basic_result.get('patterns', [])
        print(f"📊 Detected {len(patterns)} semantic patterns")

        if patterns:
            for pattern in patterns[:3]:
                print(f"  • {pattern['concept']} (confidence: {pattern['confidence']:.2f})")

        print(f"\n🗺️  Concept mappings: {len(mappings)} found")
        for mapping in mappings[:3]:
            print(f"  • {mapping.concept.name} (confidence: {mapping.confidence:.2f})")
            if mapping.evidence:
                print(f"    Evidence: {', '.join(mapping.evidence[:2])}")

        print()

    def explain_concept(self, concept_name: str):
        """Explain a specific ML/AI concept"""
        print(f"📖 Explaining Concept: {concept_name.upper()}")
        print("-" * 40)

        explanation = self.llm_analyzer.explain_concept_deeply(concept_name.lower())

        if 'error' in explanation:
            print(f"❌ Could not explain '{concept_name}'. Try: transformer, attention, convolution, optimization")
            return

        print(f"📝 Definition: {explanation.get('definition', 'N/A')}")
        print(f"🧮 Mathematical: {explanation.get('mathematical_foundation', 'N/A')}")
        print(f"💡 Intuitive: {explanation.get('intuitive_explanation', 'N/A')}")
        print(f"🎓 Complexity: {explanation.get('complexity_level', 'N/A')}")

        learning_path = explanation.get('learning_path', [])
        if learning_path:
            print(f"\n📚 Learning Path:")
            for i, step in enumerate(learning_path[:4], 1):
                print(f"  {i}. {step}")

        print()

    def compare_implementations(self, ex1_name: str, ex2_name: str):
        """Compare two code implementations"""
        if ex1_name not in self.example_codes or ex2_name not in self.example_codes:
            print(f"❌ One or both examples not found. Use 'list' to see available examples.")
            return

        print(f"⚖️  Comparing: {ex1_name} vs {ex2_name}")
        print("-" * 50)

        code1 = self.example_codes[ex1_name]
        code2 = self.example_codes[ex2_name]

        # Use basic agent for comparison
        comparison = self.basic_agent.compare_implementations(code1, code2)

        comp_result = comparison.get('comparison', {})
        print(f"🔍 Comparison Results:")
        print(f"   Common patterns: {', '.join(comp_result.get('common_patterns', []))}")
        print(f"   Unique to {ex1_name}: {', '.join(comp_result.get('unique_to_impl1', []))}")
        print(f"   Unique to {ex2_name}: {', '.join(comp_result.get('unique_to_impl2', []))}")

        opt_comp = comp_result.get('optimization_comparison', {})
        if opt_comp:
            better = opt_comp.get('better_optimized', 'similar')
            print(f"\n⚡ Better optimized: {better}")

            suggestions = opt_comp.get('suggestions', {})
            if 'for_implementation1' in suggestions:
                print(f"\n💡 Suggestions for {ex1_name}:")
                for suggestion in suggestions['for_implementation1'][:2]:
                    print(f"   • {suggestion}")

        print()

    def show_concept_mapping(self, example_name: str):
        """Show concept mappings for an example"""
        if example_name not in self.example_codes:
            print(f"❌ Example '{example_name}' not found.")
            return

        code = self.example_codes[example_name]
        print(f"🗺️  Concept Mappings: {example_name}")
        print("-" * 40)

        mappings = self.mapping_system.map_code_to_concepts(code)
        relationships = self.mapping_system.analyze_concept_relationships(code)

        print(f"📍 Detected {len(mappings)} concept mappings:")
        for mapping in mappings:
            print(f"\n🎯 {mapping.concept.name.upper()}")
            print(f"   Category: {mapping.concept.category.value}")
            print(f"   Confidence: {mapping.confidence:.2f}")
            print(f"   Description: {mapping.concept.description}")
            if mapping.evidence:
                print(f"   Evidence: {', '.join(mapping.evidence[:2])}")
            if mapping.context:
                print(f"   Context: {mapping.context}")

        # Show relationships
        hierarchy = relationships.get('hierarchy', {})
        if hierarchy:
            print(f"\n🏗️  Concept Hierarchy:")
            for category, concepts in hierarchy.items():
                print(f"   {category}: {', '.join(concepts)}")

        # Show optimization opportunities
        opportunities = relationships.get('optimization_opportunities', [])
        if opportunities:
            print(f"\n💡 Optimization Opportunities:")
            for opp in opportunities:
                print(f"   • {opp}")

        print()

    def suggest_optimizations(self, example_name: str):
        """Suggest optimizations for an example"""
        if example_name not in self.example_codes:
            print(f"❌ Example '{example_name}' not found.")
            return

        code = self.example_codes[example_name]
        print(f"🚀 Optimization Suggestions: {example_name}")
        print("-" * 45)

        # Get mappings to understand what concepts are present
        mappings = self.mapping_system.map_code_to_concepts(code)

        for mapping in mappings:
            concept_name = mapping.concept.name
            suggestions = self.mapping_system.suggest_optimizations_for_concept(concept_name)

            if suggestions:
                print(f"\n🎯 For {concept_name.upper()}:")
                for suggestion in suggestions[:3]:
                    print(f"   • {suggestion}")

        # Advanced analysis for more suggestions
        enhanced_result = self.enhanced_agent.deep_code_analysis(code)
        research_directions = enhanced_result.get('research_directions', [])

        if research_directions:
            print(f"\n🔬 Research Directions:")
            for direction in research_directions[:3]:
                print(f"   • {direction}")

        print()

    def run_interactive_session(self):
        """Main interactive session loop"""
        self.show_welcome_message()

        while True:
            try:
                user_input = input("🧠 semantic> ").strip().lower()

                if not user_input:
                    continue

                parts = user_input.split()
                command = parts[0]

                if command in ['quit', 'exit', 'q']:
                    print("👋 Thank you for exploring semantic code understanding!")
                    break

                elif command == 'help':
                    self.show_welcome_message()

                elif command == 'list':
                    self.show_examples()

                elif command == 'analyze' and len(parts) > 1:
                    self.analyze_example(parts[1])

                elif command == 'custom':
                    self.analyze_custom_code()

                elif command == 'explain' and len(parts) > 1:
                    self.explain_concept(parts[1])

                elif command == 'compare' and len(parts) > 2:
                    self.compare_implementations(parts[1], parts[2])

                elif command == 'mapping' and len(parts) > 1:
                    self.show_concept_mapping(parts[1])

                elif command == 'optimize' and len(parts) > 1:
                    self.suggest_optimizations(parts[1])

                else:
                    print("❓ Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Run the interactive semantic demo"""
    demo = InteractiveSemanticDemo()
    demo.run_interactive_session()

if __name__ == "__main__":
    main()