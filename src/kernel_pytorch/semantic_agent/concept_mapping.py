"""
Code-to-Concept Mapping System

This module implements a sophisticated system for mapping code patterns
to high-level ML/AI concepts and vice versa. It enables bidirectional
understanding between implementation details and theoretical concepts.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ast
import re
from collections import defaultdict

class ConceptCategory(Enum):
    """Categories of ML/AI concepts"""
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    MATHEMATICAL = "mathematical"
    DATA_STRUCTURE = "data_structure"
    ALGORITHM = "algorithm"
    PATTERN = "pattern"

@dataclass
class CodePattern:
    """Represents a specific code pattern"""
    name: str
    category: ConceptCategory
    ast_patterns: List[str] = field(default_factory=list)
    text_patterns: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    class_names: Set[str] = field(default_factory=set)
    function_names: Set[str] = field(default_factory=set)
    confidence_indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class Concept:
    """Represents a high-level ML/AI concept"""
    name: str
    category: ConceptCategory
    description: str
    mathematical_foundation: str = ""
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    common_implementations: List[str] = field(default_factory=list)
    optimization_techniques: List[str] = field(default_factory=list)

@dataclass
class ConceptMapping:
    """Maps code patterns to concepts"""
    concept: Concept
    code_patterns: List[CodePattern]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    context: Optional[str] = None

class ConceptMappingSystem:
    """Main system for mapping code to concepts and concepts to code"""

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.code_patterns: Dict[str, CodePattern] = {}
        self.mappings: Dict[str, List[ConceptMapping]] = defaultdict(list)

        # Initialize with ML/AI concepts and patterns
        self._initialize_concepts()
        self._initialize_code_patterns()
        self._build_mappings()

    def _initialize_concepts(self):
        """Initialize the knowledge base with core ML/AI concepts"""

        # Architecture concepts
        self.concepts["transformer"] = Concept(
            name="transformer",
            category=ConceptCategory.ARCHITECTURE,
            description="Neural architecture using self-attention for sequence modeling",
            mathematical_foundation="Attention(Q,K,V) = softmax(QK^T/√d_k)V",
            prerequisites=["attention", "matrix_multiplication", "layer_normalization"],
            related_concepts=["attention", "encoder", "decoder", "positional_encoding"],
            common_implementations=["MultiHeadAttention", "TransformerBlock", "BERT", "GPT"],
            optimization_techniques=["flash_attention", "gradient_checkpointing", "mixed_precision"]
        )

        self.concepts["attention"] = Concept(
            name="attention",
            category=ConceptCategory.ALGORITHM,
            description="Mechanism for focusing on relevant parts of input sequences",
            mathematical_foundation="Weighted sum based on query-key similarity",
            prerequisites=["dot_product", "softmax", "linear_transformation"],
            related_concepts=["transformer", "sequence_modeling", "similarity"],
            common_implementations=["scaled_dot_product", "multi_head", "cross_attention"],
            optimization_techniques=["flash_attention", "sparse_attention", "linear_attention"]
        )

        self.concepts["convolution"] = Concept(
            name="convolution",
            category=ConceptCategory.MATHEMATICAL,
            description="Mathematical operation for feature extraction using kernels",
            mathematical_foundation="(f * g)(t) = ∫ f(τ)g(t - τ)dτ",
            prerequisites=["linear_algebra", "signal_processing"],
            related_concepts=["cnn", "feature_maps", "pooling", "stride"],
            common_implementations=["Conv1d", "Conv2d", "Conv3d", "DepthwiseConv"],
            optimization_techniques=["winograd", "im2col", "depthwise_separable"]
        )

        # Optimization concepts
        self.concepts["kernel_fusion"] = Concept(
            name="kernel_fusion",
            category=ConceptCategory.OPTIMIZATION,
            description="Combining multiple GPU operations into single kernel",
            mathematical_foundation="f(g(x)) computed as single kernel instead of f∘g",
            prerequisites=["gpu_programming", "memory_hierarchy"],
            related_concepts=["memory_optimization", "cuda", "triton"],
            common_implementations=["@torch.jit.script", "torch.compile", "custom_kernels"],
            optimization_techniques=["operation_fusion", "memory_coalescing", "shared_memory"]
        )

        self.concepts["mixed_precision"] = Concept(
            name="mixed_precision",
            category=ConceptCategory.OPTIMIZATION,
            description="Using different numerical precisions to optimize training",
            mathematical_foundation="FP16 for forward pass, FP32 for gradients",
            prerequisites=["floating_point", "numerical_stability"],
            related_concepts=["gradient_scaling", "amp", "numerical_precision"],
            common_implementations=["autocast", "GradScaler", "half()"],
            optimization_techniques=["automatic_casting", "gradient_scaling", "loss_scaling"]
        )

    def _initialize_code_patterns(self):
        """Initialize code patterns that indicate specific concepts"""

        # Transformer patterns
        self.code_patterns["transformer_class"] = CodePattern(
            name="transformer_class",
            category=ConceptCategory.ARCHITECTURE,
            ast_patterns=["class.*Transformer.*Module", "class.*Attention.*Module"],
            text_patterns=[r"class\s+\w*Transformer\w*", r"class\s+\w*Attention\w*"],
            imports={"torch.nn"},
            class_names={"Transformer", "Attention", "TransformerBlock"},
            function_names={"forward"},
            confidence_indicators={
                "class_name_transformer": 0.8,
                "attention_computation": 0.6,
                "layer_norm": 0.4,
                "positional_encoding": 0.7
            }
        )

        # Attention patterns
        self.code_patterns["attention_computation"] = CodePattern(
            name="attention_computation",
            category=ConceptCategory.ALGORITHM,
            ast_patterns=["torch.matmul", "F.softmax", "scaled_dot_product"],
            text_patterns=[
                r"torch\.matmul\(.*,.*\.transpose\(",
                r"F\.softmax\(.*scores",
                r"scaled_dot_product_attention"
            ],
            imports={"torch.nn.functional"},
            function_names={"attention", "scaled_dot_product_attention"},
            confidence_indicators={
                "query_key_matmul": 0.8,
                "softmax_scores": 0.7,
                "value_weighting": 0.6,
                "scale_factor": 0.5
            }
        )

        # Convolution patterns
        self.code_patterns["convolution_ops"] = CodePattern(
            name="convolution_ops",
            category=ConceptCategory.MATHEMATICAL,
            ast_patterns=["nn.Conv", "F.conv"],
            text_patterns=[r"nn\.Conv[123]d", r"F\.conv[123]d"],
            imports={"torch.nn", "torch.nn.functional"},
            class_names={"Conv1d", "Conv2d", "Conv3d"},
            confidence_indicators={
                "conv_layer": 0.9,
                "kernel_size": 0.6,
                "stride_padding": 0.4
            }
        )

        # Optimization patterns
        self.code_patterns["jit_compilation"] = CodePattern(
            name="jit_compilation",
            category=ConceptCategory.OPTIMIZATION,
            ast_patterns=["@torch.jit.script", "@torch.compile"],
            text_patterns=[r"@torch\.jit\.script", r"@torch\.compile", r"torch\.jit\.trace"],
            imports={"torch.jit"},
            confidence_indicators={
                "jit_decorator": 0.9,
                "script_compilation": 0.8,
                "trace_compilation": 0.7
            }
        )

        self.code_patterns["mixed_precision"] = CodePattern(
            name="mixed_precision",
            category=ConceptCategory.OPTIMIZATION,
            ast_patterns=["autocast", "GradScaler", ".half()"],
            text_patterns=[r"autocast\(\)", r"GradScaler\(\)", r"\.half\(\)"],
            imports={"torch.cuda.amp"},
            confidence_indicators={
                "autocast_context": 0.8,
                "grad_scaler": 0.7,
                "half_precision": 0.6
            }
        )

        self.code_patterns["memory_optimization"] = CodePattern(
            name="memory_optimization",
            category=ConceptCategory.OPTIMIZATION,
            ast_patterns=["checkpoint", "gradient_checkpointing", "inplace=True"],
            text_patterns=[
                r"checkpoint\(",
                r"gradient_checkpointing",
                r"inplace\s*=\s*True"
            ],
            confidence_indicators={
                "gradient_checkpointing": 0.8,
                "inplace_operations": 0.6,
                "memory_efficient": 0.5
            }
        )

    def _build_mappings(self):
        """Build mappings between concepts and code patterns"""

        # Transformer mappings
        transformer_mappings = [
            ConceptMapping(
                concept=self.concepts["transformer"],
                code_patterns=[
                    self.code_patterns["transformer_class"],
                    self.code_patterns["attention_computation"]
                ],
                confidence=0.8
            )
        ]
        self.mappings["transformer"] = transformer_mappings

        # Attention mappings
        attention_mappings = [
            ConceptMapping(
                concept=self.concepts["attention"],
                code_patterns=[self.code_patterns["attention_computation"]],
                confidence=0.7
            )
        ]
        self.mappings["attention"] = attention_mappings

        # Convolution mappings
        convolution_mappings = [
            ConceptMapping(
                concept=self.concepts["convolution"],
                code_patterns=[self.code_patterns["convolution_ops"]],
                confidence=0.9
            )
        ]
        self.mappings["convolution"] = convolution_mappings

        # Optimization mappings
        fusion_mappings = [
            ConceptMapping(
                concept=self.concepts["kernel_fusion"],
                code_patterns=[self.code_patterns["jit_compilation"]],
                confidence=0.8
            )
        ]
        self.mappings["kernel_fusion"] = fusion_mappings

        precision_mappings = [
            ConceptMapping(
                concept=self.concepts["mixed_precision"],
                code_patterns=[self.code_patterns["mixed_precision"]],
                confidence=0.8
            )
        ]
        self.mappings["mixed_precision"] = precision_mappings

    def map_code_to_concepts(self, code: str) -> List[ConceptMapping]:
        """Map source code to high-level concepts"""
        detected_mappings = []

        for concept_name, mappings in self.mappings.items():
            for mapping in mappings:
                confidence = self._calculate_mapping_confidence(code, mapping)
                if confidence > 0.3:  # Threshold for detection
                    evidence = self._extract_evidence(code, mapping)
                    detected_mapping = ConceptMapping(
                        concept=mapping.concept,
                        code_patterns=mapping.code_patterns,
                        confidence=confidence,
                        evidence=evidence,
                        context=self._extract_context(code, mapping)
                    )
                    detected_mappings.append(detected_mapping)

        # Sort by confidence
        return sorted(detected_mappings, key=lambda x: x.confidence, reverse=True)

    def _calculate_mapping_confidence(self, code: str, mapping: ConceptMapping) -> float:
        """Calculate confidence score for a concept mapping"""
        confidence_scores = []

        for pattern in mapping.code_patterns:
            pattern_confidence = self._match_code_pattern(code, pattern)
            confidence_scores.append(pattern_confidence)

        # Combine confidence scores
        if confidence_scores:
            # Use weighted average, giving more weight to higher scores
            weights = [score + 0.1 for score in confidence_scores]
            total_weight = sum(weights)
            weighted_avg = sum(score * weight for score, weight in zip(confidence_scores, weights)) / total_weight
            return min(weighted_avg, 1.0)

        return 0.0

    def _match_code_pattern(self, code: str, pattern: CodePattern) -> float:
        """Check how well code matches a specific pattern"""
        score = 0.0
        matches = 0
        total_checks = 0

        # Check text patterns
        for text_pattern in pattern.text_patterns:
            total_checks += 1
            if re.search(text_pattern, code, re.MULTILINE | re.IGNORECASE):
                matches += 1
                score += 0.6  # Base score for text match

        # Check imports
        for import_pattern in pattern.imports:
            total_checks += 1
            if f"import {import_pattern}" in code or f"from {import_pattern}" in code:
                matches += 1
                score += 0.3

        # Check class names (more flexible matching)
        for class_name in pattern.class_names:
            total_checks += 1
            # Check for exact class name or partial matches
            class_pattern = f"class.*{class_name}.*"
            if re.search(class_pattern, code, re.IGNORECASE) or class_name in code:
                matches += 1
                score += 0.5

        # Check function names
        for func_name in pattern.function_names:
            total_checks += 1
            if f"def {func_name}" in code or f"{func_name}(" in code:
                matches += 1
                score += 0.4

        # Return raw score without normalization for better sensitivity
        return min(score, 1.0)

    def _extract_evidence(self, code: str, mapping: ConceptMapping) -> List[str]:
        """Extract evidence supporting the concept mapping"""
        evidence = []

        for pattern in mapping.code_patterns:
            # Find matching text patterns
            for text_pattern in pattern.text_patterns:
                matches = re.finditer(text_pattern, code, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    evidence.append(match.group().strip())
                    if len(evidence) >= 5:  # Limit evidence
                        break

        return evidence[:3]  # Return top 3 pieces of evidence

    def _extract_context(self, code: str, mapping: ConceptMapping) -> Optional[str]:
        """Extract contextual information about where concept appears"""
        lines = code.split('\n')

        for pattern in mapping.code_patterns:
            for text_pattern in pattern.text_patterns:
                for i, line in enumerate(lines):
                    if re.search(text_pattern, line, re.IGNORECASE):
                        # Find enclosing function or class
                        context = self._find_enclosing_context(lines, i)
                        if context:
                            return context

        return None

    def _find_enclosing_context(self, lines: List[str], line_idx: int) -> Optional[str]:
        """Find the enclosing function or class for a line"""
        # Look backwards for class or function definition
        for i in range(line_idx, -1, -1):
            line = lines[i].strip()
            if line.startswith('class ') or line.startswith('def '):
                # Extract name
                match = re.match(r'(class|def)\s+(\w+)', line)
                if match:
                    context_type = match.group(1)
                    context_name = match.group(2)
                    return f"{context_type}:{context_name}"

        return None

    def get_concept_implementations(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get code patterns that implement a specific concept"""
        if concept_name not in self.concepts:
            return []

        concept = self.concepts[concept_name]
        implementations = []

        # Get mappings for this concept
        mappings = self.mappings.get(concept_name, [])

        for mapping in mappings:
            for pattern in mapping.code_patterns:
                implementation = {
                    "pattern_name": pattern.name,
                    "category": pattern.category.value,
                    "text_patterns": pattern.text_patterns,
                    "imports": list(pattern.imports),
                    "class_names": list(pattern.class_names),
                    "function_names": list(pattern.function_names),
                    "confidence_indicators": pattern.confidence_indicators
                }
                implementations.append(implementation)

        return implementations

    def suggest_optimizations_for_concept(self, concept_name: str) -> List[str]:
        """Suggest optimization techniques for a concept"""
        if concept_name not in self.concepts:
            return []

        concept = self.concepts[concept_name]
        suggestions = []

        # Add concept-specific optimizations
        suggestions.extend(concept.optimization_techniques)

        # Add related optimization concepts
        for related in concept.related_concepts:
            if related in self.concepts:
                related_concept = self.concepts[related]
                if related_concept.category == ConceptCategory.OPTIMIZATION:
                    suggestions.append(f"Apply {related}: {related_concept.description}")

        return suggestions

    def get_learning_path(self, concept_name: str) -> List[str]:
        """Get recommended learning path for a concept"""
        if concept_name not in self.concepts:
            return []

        concept = self.concepts[concept_name]
        path = []

        # Add prerequisites first
        for prereq in concept.prerequisites:
            if prereq in self.concepts:
                path.append(f"Learn {prereq}: {self.concepts[prereq].description}")
            else:
                path.append(f"Learn {prereq}")

        # Add the concept itself
        path.append(f"Study {concept.name}: {concept.description}")

        # Add related concepts for deeper understanding
        for related in concept.related_concepts[:2]:  # Limit to 2
            if related in self.concepts and related not in concept.prerequisites:
                path.append(f"Explore {related}: {self.concepts[related].description}")

        return path

    def analyze_concept_relationships(self, code: str) -> Dict[str, Any]:
        """Analyze relationships between detected concepts"""
        mappings = self.map_code_to_concepts(code)

        detected_concepts = [m.concept.name for m in mappings]
        relationships = defaultdict(list)

        for concept_name in detected_concepts:
            concept = self.concepts[concept_name]

            # Find relationships with other detected concepts
            for related in concept.related_concepts:
                if related in detected_concepts:
                    relationships[concept_name].append(related)

        # Build concept hierarchy
        hierarchy = {}
        for concept_name in detected_concepts:
            concept = self.concepts[concept_name]
            category = concept.category.value
            if category not in hierarchy:
                hierarchy[category] = []
            hierarchy[category].append(concept_name)

        return {
            "detected_concepts": detected_concepts,
            "relationships": dict(relationships),
            "hierarchy": hierarchy,
            "optimization_opportunities": self._find_optimization_opportunities(mappings)
        }

    def _find_optimization_opportunities(self, mappings: List[ConceptMapping]) -> List[str]:
        """Find optimization opportunities based on detected concepts"""
        opportunities = []

        concept_names = [m.concept.name for m in mappings]

        # Check for specific optimization patterns
        if "attention" in concept_names and "kernel_fusion" not in concept_names:
            opportunities.append("Consider applying kernel fusion to attention computation")

        if "transformer" in concept_names and "mixed_precision" not in concept_names:
            opportunities.append("Consider using mixed precision training for transformer")

        if "convolution" in concept_names:
            opportunities.append("Consider optimized convolution implementations (e.g., Winograd)")

        return opportunities

# Example usage and testing
if __name__ == "__main__":
    print("🗺️  Code-to-Concept Mapping System Demo")
    print("=" * 50)

    mapping_system = ConceptMappingSystem()

    # Test with transformer code
    transformer_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(dim, 3 * dim)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1))
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)

        return output
    '''

    print("🔍 Analyzing Transformer Code...")
    mappings = mapping_system.map_code_to_concepts(transformer_code)

    for mapping in mappings[:3]:
        print(f"\n🎯 Detected: {mapping.concept.name}")
        print(f"   Confidence: {mapping.confidence:.2f}")
        print(f"   Category: {mapping.concept.category.value}")
        print(f"   Evidence: {', '.join(mapping.evidence)}")
        if mapping.context:
            print(f"   Context: {mapping.context}")

    # Test concept relationship analysis
    print("\n🔗 Concept Relationships:")
    relationships = mapping_system.analyze_concept_relationships(transformer_code)
    print(f"   Detected: {', '.join(relationships['detected_concepts'])}")
    print(f"   Hierarchy: {relationships['hierarchy']}")
    print(f"   Opportunities: {relationships['optimization_opportunities']}")

    print("\n✅ Mapping system demonstration complete!")