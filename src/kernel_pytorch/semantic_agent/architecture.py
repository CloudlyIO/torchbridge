"""
Semantic Code Understanding Architecture

Defines the core architecture for building an LLM-based agent that can
understand ML/AI code semantically, going beyond syntax to understand
concepts, patterns, and optimization opportunities.

Architecture Components:
1. Code Parser - AST + semantic analysis
2. Concept Extractor - ML/AI pattern recognition
3. Knowledge Base - ML concepts and patterns
4. Reasoning Engine - LLM-powered semantic understanding
5. Explanation Generator - Human-readable insights
"""

import ast
import inspect
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import re


class MLConcept(Enum):
    """High-level ML/AI concepts that can be extracted from code"""

    # Core ML Concepts
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    CONVOLUTION = "convolution"
    RECURRENT = "recurrent"

    # Training Concepts
    BACKPROPAGATION = "backpropagation"
    GRADIENT_DESCENT = "gradient_descent"
    LOSS_FUNCTION = "loss_function"
    OPTIMIZER = "optimizer"
    REGULARIZATION = "regularization"

    # Architecture Patterns
    ENCODER_DECODER = "encoder_decoder"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    VAE = "variational_autoencoder"
    DIFFUSION = "diffusion"

    # Optimization Concepts
    KERNEL_FUSION = "kernel_fusion"
    MEMORY_OPTIMIZATION = "memory_optimization"
    PARALLEL_COMPUTATION = "parallel_computation"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"

    # Data Patterns
    BATCHING = "batching"
    SEQUENCE_MODELING = "sequence_modeling"
    GRAPH_LEARNING = "graph_learning"
    COMPUTER_VISION = "computer_vision"
    NLP = "natural_language_processing"


@dataclass
class SemanticPattern:
    """Represents a semantic pattern found in code"""
    concept: MLConcept
    confidence: float
    evidence: List[str]  # Code snippets that support this pattern
    location: str  # Where in code this pattern was found
    optimization_potential: Optional[str] = None
    educational_notes: Optional[str] = None


@dataclass
class CodeContext:
    """Context information about a piece of code"""
    file_path: str
    function_name: Optional[str]
    class_name: Optional[str]
    imports: List[str]
    dependencies: List[str]
    ast_node: Optional[ast.AST] = None


class ConceptKnowledgeBase:
    """
    Knowledge base of ML/AI concepts and their code signatures
    """

    def __init__(self):
        self.concept_signatures = self._build_concept_signatures()
        self.optimization_patterns = self._build_optimization_patterns()

    def _build_concept_signatures(self) -> Dict[MLConcept, Dict[str, Any]]:
        """Build signatures for recognizing ML concepts in code"""
        return {
            MLConcept.TRANSFORMER: {
                'class_patterns': [
                    r'class.*Transformer.*\(.*nn\.Module\)',
                    r'class.*Attention.*\(.*nn\.Module\)',
                ],
                'function_patterns': [
                    r'def.*attention\(',
                    r'def.*self_attention\(',
                    r'def.*multi_head_attention\(',
                ],
                'api_patterns': [
                    'nn.MultiheadAttention',
                    'F.scaled_dot_product_attention',
                    'torch.matmul.*transpose',
                    'F.softmax.*dim=-1',
                ],
                'variable_patterns': [
                    r'q(?:uery)?.*k(?:ey)?.*v(?:alue)?',
                    r'attention.*weights?',
                    r'head_dim',
                    r'num_heads',
                ],
                'computation_patterns': [
                    'Q @ K.T * scale',
                    'softmax(scores)',
                    'attention_weights @ V',
                ]
            },

            MLConcept.CONVOLUTION: {
                'class_patterns': [
                    r'class.*Conv.*\(.*nn\.Module\)',
                    r'class.*CNN.*\(.*nn\.Module\)',
                ],
                'function_patterns': [
                    r'def.*conv\d*d\(',
                ],
                'api_patterns': [
                    'nn.Conv1d', 'nn.Conv2d', 'nn.Conv3d',
                    'F.conv1d', 'F.conv2d', 'F.conv3d',
                    'nn.ConvTranspose',
                ],
                'variable_patterns': [
                    r'kernel_size',
                    r'stride',
                    r'padding',
                    r'dilation',
                ],
            },

            MLConcept.ATTENTION: {
                'class_patterns': [
                    r'class.*Attention.*\(.*nn\.Module\)',
                ],
                'function_patterns': [
                    r'def.*attention\(',
                    r'def.*scaled_dot_product\(',
                ],
                'api_patterns': [
                    'F.scaled_dot_product_attention',
                    'torch.matmul.*transpose',
                    'F.softmax.*dim=-1',
                ],
                'computation_patterns': [
                    'scores = q @ k.transpose',
                    'weights = softmax(scores)',
                    'output = weights @ v',
                ],
            },

            MLConcept.KERNEL_FUSION: {
                'function_patterns': [
                    r'@torch\.jit\.script',
                    r'@torch\.compile',
                    r'def.*fused.*\(',
                ],
                'api_patterns': [
                    'torch.jit.script',
                    'torch.compile',
                    'triton.jit',
                ],
                'optimization_patterns': [
                    'elementwise operations combined',
                    'reduced memory transfers',
                    'single kernel launch',
                ],
            },

            MLConcept.MEMORY_OPTIMIZATION: {
                'function_patterns': [
                    r'def.*checkpoint\(',
                    r'def.*efficient.*\(',
                ],
                'api_patterns': [
                    'torch.utils.checkpoint',
                    'torch.cuda.empty_cache',
                    'torch.no_grad',
                    'inplace=True',
                ],
                'variable_patterns': [
                    r'gradient_checkpointing',
                    r'memory_efficient',
                    r'low_mem',
                ],
            },
        }

    def _build_optimization_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for recognizing optimization opportunities"""
        return {
            'kernel_fusion_opportunity': {
                'pattern': 'Multiple sequential elementwise operations',
                'indicators': [
                    'torch.relu(torch.add(',
                    'F.gelu(F.linear(',
                    'torch.sigmoid(torch.matmul(',
                ],
                'suggestion': 'Consider using @torch.jit.script or @torch.compile for automatic kernel fusion',
            },

            'memory_inefficient_attention': {
                'pattern': 'Manual attention implementation without Flash Attention',
                'indicators': [
                    'torch.matmul(q, k.transpose(-2, -1))',
                    'F.softmax(scores, dim=-1)',
                    'torch.matmul(attn_weights, v)',
                ],
                'suggestion': 'Use F.scaled_dot_product_attention for memory-efficient attention',
            },

            'batch_size_dependency': {
                'pattern': 'Operations that process samples individually',
                'indicators': [
                    'for i in range(batch_size)',
                    'for sample in batch',
                    'tensor[i]',
                ],
                'suggestion': 'Vectorize operations to process entire batch simultaneously',
            },
        }

    def get_concept_info(self, concept: MLConcept) -> Dict[str, Any]:
        """Get detailed information about a specific ML concept"""
        return self.concept_signatures.get(concept, {})

    def get_optimization_suggestions(self, pattern: str) -> Dict[str, Any]:
        """Get optimization suggestions for a specific pattern"""
        return self.optimization_patterns.get(pattern, {})


class CodeSemanticAnalyzer:
    """
    Core analyzer that extracts semantic meaning from code
    """

    def __init__(self):
        self.knowledge_base = ConceptKnowledgeBase()

    def parse_code(self, code: str, context: Optional[CodeContext] = None) -> ast.AST:
        """Parse code into AST for analysis"""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")

    def extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def extract_function_definitions(self, tree: ast.AST) -> List[Tuple[str, ast.FunctionDef]]:
        """Extract function definitions from AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append((node.name, node))
        return functions

    def extract_class_definitions(self, tree: ast.AST) -> List[Tuple[str, ast.ClassDef]]:
        """Extract class definitions from AST"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append((node.name, node))
        return classes

    def analyze_function_semantics(self, func_node: ast.FunctionDef, code: str) -> List[SemanticPattern]:
        """Analyze semantic patterns in a function"""
        patterns = []
        func_code = ast.get_source_segment(code, func_node) or ""

        # Check for ML concept patterns
        for concept, signatures in self.knowledge_base.concept_signatures.items():
            confidence = self._calculate_pattern_confidence(func_code, signatures)
            if confidence > 0.3:  # Threshold for pattern detection
                evidence = self._extract_evidence(func_code, signatures)
                pattern = SemanticPattern(
                    concept=concept,
                    confidence=confidence,
                    evidence=evidence,
                    location=f"function:{func_node.name}",
                    optimization_potential=self._suggest_optimizations(func_code, concept),
                    educational_notes=self._generate_educational_notes(concept, evidence)
                )
                patterns.append(pattern)

        return patterns

    def analyze_class_semantics(self, class_node: ast.ClassDef, code: str) -> List[SemanticPattern]:
        """Analyze semantic patterns in a class"""
        patterns = []
        class_code = ast.get_source_segment(code, class_node) or ""

        # Check for architectural patterns
        for concept, signatures in self.knowledge_base.concept_signatures.items():
            confidence = self._calculate_pattern_confidence(class_code, signatures)
            if confidence > 0.3:
                evidence = self._extract_evidence(class_code, signatures)
                pattern = SemanticPattern(
                    concept=concept,
                    confidence=confidence,
                    evidence=evidence,
                    location=f"class:{class_node.name}",
                    optimization_potential=self._suggest_optimizations(class_code, concept),
                    educational_notes=self._generate_educational_notes(concept, evidence)
                )
                patterns.append(pattern)

        return patterns

    def _calculate_pattern_confidence(self, code: str, signatures: Dict[str, Any]) -> float:
        """Calculate confidence score for a pattern match"""
        total_score = 0.0
        max_score = 0.0

        # Check different types of patterns
        pattern_types = ['class_patterns', 'function_patterns', 'api_patterns', 'variable_patterns']

        for pattern_type in pattern_types:
            if pattern_type in signatures:
                max_score += 1.0
                patterns = signatures[pattern_type]
                if any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns):
                    total_score += 1.0

        return total_score / max_score if max_score > 0 else 0.0

    def _extract_evidence(self, code: str, signatures: Dict[str, Any]) -> List[str]:
        """Extract code snippets that serve as evidence for a pattern"""
        evidence = []

        for pattern_type, patterns in signatures.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    evidence.append(match.group())

        return evidence[:5]  # Limit to first 5 pieces of evidence

    def _suggest_optimizations(self, code: str, concept: MLConcept) -> Optional[str]:
        """Suggest optimizations based on detected patterns"""
        optimization_map = {
            MLConcept.ATTENTION: "Consider using F.scaled_dot_product_attention for Flash Attention optimization",
            MLConcept.TRANSFORMER: "Consider using torch.compile or custom CUDA kernels for performance",
            MLConcept.CONVOLUTION: "Consider using cuDNN optimized operations and proper memory layout",
            MLConcept.KERNEL_FUSION: "Already optimized for kernel fusion",
            MLConcept.MEMORY_OPTIMIZATION: "Good memory optimization practices detected",
        }
        return optimization_map.get(concept)

    def _generate_educational_notes(self, concept: MLConcept, evidence: List[str]) -> str:
        """Generate educational notes about the detected concept"""
        concept_explanations = {
            MLConcept.TRANSFORMER: "Transformer architecture using self-attention mechanisms for sequence modeling",
            MLConcept.ATTENTION: "Attention mechanism allowing the model to focus on relevant parts of the input",
            MLConcept.CONVOLUTION: "Convolutional layers for local feature extraction, common in computer vision",
            MLConcept.KERNEL_FUSION: "Optimization technique combining multiple operations into single GPU kernel",
            MLConcept.MEMORY_OPTIMIZATION: "Techniques to reduce memory usage during model training/inference",
        }

        base_explanation = concept_explanations.get(concept, f"ML concept: {concept.value}")
        if evidence:
            return f"{base_explanation}. Evidence: {', '.join(evidence[:2])}"
        return base_explanation


class SemanticCodeAgent:
    """
    Main agent class that orchestrates semantic code understanding
    """

    def __init__(self):
        self.analyzer = CodeSemanticAnalyzer()
        self.analysis_history = []

    def analyze_code(self, code: str, context: Optional[CodeContext] = None) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis of code
        """
        # Parse code
        try:
            tree = self.analyzer.parse_code(code, context)
        except ValueError as e:
            return {"error": str(e), "patterns": []}

        # Extract structural elements
        imports = self.analyzer.extract_imports(tree)
        functions = self.analyzer.extract_function_definitions(tree)
        classes = self.analyzer.extract_class_definitions(tree)

        # Analyze semantic patterns
        all_patterns = []

        # Analyze functions
        for func_name, func_node in functions:
            patterns = self.analyzer.analyze_function_semantics(func_node, code)
            all_patterns.extend(patterns)

        # Analyze classes
        for class_name, class_node in classes:
            patterns = self.analyzer.analyze_class_semantics(class_node, code)
            all_patterns.extend(patterns)

        # Create comprehensive analysis result
        result = {
            "imports": imports,
            "functions": [name for name, _ in functions],
            "classes": [name for name, _ in classes],
            "patterns": [
                {
                    "concept": pattern.concept.value,
                    "confidence": pattern.confidence,
                    "evidence": pattern.evidence,
                    "location": pattern.location,
                    "optimization_potential": pattern.optimization_potential,
                    "educational_notes": pattern.educational_notes,
                }
                for pattern in all_patterns
            ],
            "summary": self._generate_summary(all_patterns),
            "optimization_suggestions": self._generate_optimization_suggestions(all_patterns),
        }

        # Store in history
        self.analysis_history.append({
            "code_snippet": code[:200] + "..." if len(code) > 200 else code,
            "result": result,
            "timestamp": None,  # Could add timestamp here
        })

        return result

    def _generate_summary(self, patterns: List[SemanticPattern]) -> str:
        """Generate a human-readable summary of the analysis"""
        if not patterns:
            return "No significant ML/AI patterns detected in the code."

        concept_counts = {}
        for pattern in patterns:
            concept_counts[pattern.concept.value] = concept_counts.get(pattern.concept.value, 0) + 1

        high_confidence_patterns = [p for p in patterns if p.confidence > 0.7]

        summary = f"Detected {len(patterns)} semantic patterns. "
        summary += f"High confidence patterns: {len(high_confidence_patterns)}. "
        summary += f"Main concepts: {', '.join(list(concept_counts.keys())[:3])}."

        return summary

    def _generate_optimization_suggestions(self, patterns: List[SemanticPattern]) -> List[str]:
        """Generate actionable optimization suggestions"""
        suggestions = []

        for pattern in patterns:
            if pattern.optimization_potential and pattern.confidence > 0.5:
                suggestions.append(f"{pattern.location}: {pattern.optimization_potential}")

        return list(set(suggestions))  # Remove duplicates

    def explain_concept(self, concept: MLConcept) -> Dict[str, Any]:
        """Provide detailed explanation of a specific ML concept"""
        concept_info = self.analyzer.knowledge_base.get_concept_info(concept)

        return {
            "concept": concept.value,
            "description": self.analyzer._generate_educational_notes(concept, []),
            "code_signatures": concept_info,
            "optimization_tips": self.analyzer._suggest_optimizations("", concept),
        }

    def compare_implementations(self, code1: str, code2: str) -> Dict[str, Any]:
        """Compare two code implementations semantically"""
        analysis1 = self.analyze_code(code1)
        analysis2 = self.analyze_code(code2)

        return {
            "implementation1": analysis1,
            "implementation2": analysis2,
            "comparison": {
                "common_patterns": self._find_common_patterns(analysis1, analysis2),
                "unique_to_impl1": self._find_unique_patterns(analysis1, analysis2),
                "unique_to_impl2": self._find_unique_patterns(analysis2, analysis1),
                "optimization_comparison": self._compare_optimizations(analysis1, analysis2),
            }
        }

    def _find_common_patterns(self, analysis1: Dict, analysis2: Dict) -> List[str]:
        """Find patterns common to both implementations"""
        patterns1 = {p["concept"] for p in analysis1["patterns"]}
        patterns2 = {p["concept"] for p in analysis2["patterns"]}
        return list(patterns1.intersection(patterns2))

    def _find_unique_patterns(self, analysis1: Dict, analysis2: Dict) -> List[str]:
        """Find patterns unique to first implementation"""
        patterns1 = {p["concept"] for p in analysis1["patterns"]}
        patterns2 = {p["concept"] for p in analysis2["patterns"]}
        return list(patterns1.difference(patterns2))

    def _compare_optimizations(self, analysis1: Dict, analysis2: Dict) -> Dict[str, Any]:
        """Compare optimization levels of two implementations"""
        opt1 = analysis1["optimization_suggestions"]
        opt2 = analysis2["optimization_suggestions"]

        return {
            "implementation1_optimizations": len(opt1),
            "implementation2_optimizations": len(opt2),
            "better_optimized": "implementation1" if len(opt1) < len(opt2) else "implementation2" if len(opt2) < len(opt1) else "similar",
            "suggestions": {
                "for_implementation1": opt1,
                "for_implementation2": opt2,
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the semantic code understanding system
    agent = SemanticCodeAgent()

    # Test code examples
    transformer_code = '''
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MultiHeadAttention(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)

        def forward(self, x):
            batch_size, seq_len, dim = x.shape

            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            weights = F.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)

            return output.view(batch_size, seq_len, dim)
    '''

    result = agent.analyze_code(transformer_code)
    print("Analysis Result:", result)