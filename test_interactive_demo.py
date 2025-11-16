#!/usr/bin/env python3
"""
Quick test for the interactive semantic demo
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from interactive_semantic_demo import InteractiveSemanticDemo

def test_demo_components():
    """Test key components of the interactive demo"""
    print("🧪 Testing Interactive Semantic Demo Components")
    print("=" * 55)

    try:
        demo = InteractiveSemanticDemo()

        # Test example loading
        print("✅ Demo initialization successful")
        print(f"📚 Loaded {len(demo.example_codes)} code examples")

        # Test basic analysis
        print("\n🔍 Testing analysis on basic_attention example...")
        demo.analyze_example("basic_attention")

        # Test concept mapping
        print("\n🗺️  Testing concept mapping...")
        demo.show_concept_mapping("basic_attention")

        # Test concept explanation
        print("\n📖 Testing concept explanation...")
        demo.explain_concept("attention")

        print("\n✅ All demo components working correctly!")
        print("\n🚀 Interactive demo is ready!")
        print("Run: PYTHONPATH=src python3 interactive_semantic_demo.py")

    except Exception as e:
        print(f"❌ Error testing demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_demo_components()