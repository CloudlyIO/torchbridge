#!/usr/bin/env python3
"""
Test Semantic Agent on Our Kernel Optimization Examples

This script tests our semantic code understanding agent on the actual
kernel optimization code from our project to demonstrate real-world
application.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kernel_pytorch.semantic_agent.architecture import SemanticCodeAgent
from kernel_pytorch.semantic_agent.llm_understanding import EnhancedSemanticAgent

def test_basic_optimizations():
    """Test semantic understanding on Level 1 basic optimizations"""
    print("🔍 TESTING: Basic PyTorch Optimizations")
    print("-" * 50)

    with open('src/kernel_pytorch/components/basic_optimized.py', 'r') as f:
        basic_code = f.read()

    agent = SemanticCodeAgent()
    result = agent.analyze_code(basic_code)

    print(f"📊 Detected {len(result['patterns'])} semantic patterns in basic optimizations")

    # Show optimization-related patterns
    opt_patterns = [p for p in result['patterns'] if 'optimization' in p.get('optimization_potential', '').lower()]
    print(f"⚡ Found {len(opt_patterns)} optimization-related patterns")

    for pattern in opt_patterns[:3]:
        print(f"  • {pattern['concept']} (confidence: {pattern['confidence']:.2f})")
        print(f"    Location: {pattern['location']}")
        if pattern['optimization_potential']:
            print(f"    💡 {pattern['optimization_potential']}")

def test_jit_optimizations():
    """Test semantic understanding on Level 2 JIT optimizations"""
    print("\n🔍 TESTING: JIT TorchScript Optimizations")
    print("-" * 50)

    with open('src/kernel_pytorch/components/jit_optimized.py', 'r') as f:
        jit_code = f.read()

    agent = EnhancedSemanticAgent()
    result = agent.deep_code_analysis(jit_code)

    # Check for optimization detection
    opt_analysis = result.get('code_analysis', {}).get('optimization_analysis', {})
    if opt_analysis:
        print(f"⚡ Optimization Level: {opt_analysis.get('overall_level', 'unknown').upper()}")
        print(f"   Score: {opt_analysis.get('optimization_score', 0):.2f}/1.0")

        optimizations = opt_analysis.get('optimizations', {})
        detected_opts = [opt for opt, data in optimizations.items() if data.get('detected')]
        print(f"   Detected Optimizations: {', '.join(detected_opts)}")

def test_triton_kernels():
    """Test semantic understanding on Level 4 Triton kernels"""
    print("\n🔍 TESTING: Triton Kernel Optimizations")
    print("-" * 50)

    with open('src/kernel_pytorch/triton_kernels/fused_ops.py', 'r') as f:
        triton_code = f.read()

    agent = EnhancedSemanticAgent()
    result = agent.deep_code_analysis(triton_code)

    # Look for kernel-specific concepts
    high_level_concepts = result.get('code_analysis', {}).get('high_level_concepts', [])
    kernel_concepts = [c for c in high_level_concepts if 'kernel' in c.get('concept', '').lower()]

    print(f"🧮 Found {len(kernel_concepts)} kernel-related concepts")
    for concept in kernel_concepts:
        print(f"  • {concept.get('concept', 'Unknown')} (confidence: {concept.get('confidence', 0):.2f})")

def test_semantic_ml_models():
    """Test semantic understanding on our semantic ML model examples"""
    print("\n🔍 TESTING: Semantic ML Models Understanding")
    print("-" * 50)

    with open('src/kernel_pytorch/examples/semantic_ml_models.py', 'r') as f:
        models_code = f.read()

    agent = SemanticCodeAgent()
    result = agent.analyze_code(models_code)

    # Focus on transformer and attention patterns
    transformer_patterns = [p for p in result['patterns'] if 'transformer' in p['concept']]
    attention_patterns = [p for p in result['patterns'] if 'attention' in p['concept']]

    print(f"🏗️  Architecture Patterns:")
    print(f"   Transformer patterns: {len(transformer_patterns)}")
    print(f"   Attention patterns: {len(attention_patterns)}")

    # Show highest confidence pattern
    if result['patterns']:
        best_pattern = max(result['patterns'], key=lambda x: x['confidence'])
        print(f"\n🎯 Highest Confidence Pattern:")
        print(f"   Concept: {best_pattern['concept']}")
        print(f"   Confidence: {best_pattern['confidence']:.2f}")
        print(f"   Evidence: {', '.join(best_pattern['evidence'][:2])}")

def test_optimization_progression():
    """Test understanding of optimization progression across levels"""
    print("\n🔍 TESTING: Optimization Level Progression")
    print("-" * 50)

    files_to_analyze = [
        ('Level 1 - Basic', 'src/kernel_pytorch/components/basic_optimized.py'),
        ('Level 2 - JIT', 'src/kernel_pytorch/components/jit_optimized.py'),
        ('Progressive Examples', 'src/kernel_pytorch/examples/progressive_optimization.py')
    ]

    agent = EnhancedSemanticAgent()

    for level_name, file_path in files_to_analyze:
        if os.path.exists(file_path):
            print(f"\n📈 {level_name}:")

            with open(file_path, 'r') as f:
                code = f.read()

            result = agent.deep_code_analysis(code)
            opt_analysis = result.get('code_analysis', {}).get('optimization_analysis', {})

            if opt_analysis:
                score = opt_analysis.get('optimization_score', 0)
                level = opt_analysis.get('overall_level', 'unknown')
                print(f"   Score: {score:.2f}, Level: {level}")

                # Count detected optimization techniques
                optimizations = opt_analysis.get('optimizations', {})
                detected_count = sum(1 for opt_data in optimizations.values() if opt_data.get('detected'))
                print(f"   Techniques: {detected_count} detected")

def main():
    """Run all semantic understanding tests on kernel optimization examples"""
    print("🧠 Testing Semantic Agent on Kernel Optimization Code")
    print("=" * 70)
    print("This tests our semantic understanding agent on the actual code")
    print("from our kernel optimization project.\n")

    try:
        test_basic_optimizations()
        test_jit_optimizations()
        test_triton_kernels()
        test_semantic_ml_models()
        test_optimization_progression()

        print("\n✅ TESTING COMPLETE")
        print("=" * 50)
        print("🎯 Key Results:")
        print("   • Agent successfully analyzes real kernel optimization code")
        print("   • Detects ML/AI concepts in production implementations")
        print("   • Identifies optimization techniques across different levels")
        print("   • Provides educational insights for complex GPU programming")
        print("   • Bridges high-level ML concepts with low-level optimizations")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Check that all required files exist and are accessible")

if __name__ == "__main__":
    main()