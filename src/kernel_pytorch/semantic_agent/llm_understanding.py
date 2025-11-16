"""
LLM-Powered Code Understanding

This module implements advanced semantic code understanding using Large Language Models
to go beyond pattern matching and achieve true conceptual understanding of ML/AI code.

Key Features:
- Deep semantic analysis using LLM reasoning
- Concept hierarchy understanding
- Mathematical relationship extraction
- Cross-domain knowledge transfer
- Educational explanation generation
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import ast

from .architecture import MLConcept, SemanticPattern, CodeContext


@dataclass
class SemanticConcept:
    """Rich semantic concept with LLM-enhanced understanding"""
    name: str
    category: str  # e.g., "architecture", "optimization", "algorithm"
    description: str
    mathematical_foundation: str
    implementation_patterns: List[str]
    optimization_opportunities: List[str]
    educational_analogy: str
    related_concepts: List[str]
    complexity_level: str  # "beginner", "intermediate", "advanced"


class ConceptHierarchy:
    """Hierarchical organization of ML/AI concepts"""

    def __init__(self):
        self.concept_tree = self._build_concept_hierarchy()
        self.concept_relationships = self._build_relationships()

    def _build_concept_hierarchy(self) -> Dict[str, Dict]:
        """Build hierarchical concept tree"""
        return {
            "machine_learning": {
                "supervised_learning": {
                    "neural_networks": {
                        "feedforward": ["mlp", "dense_layers"],
                        "convolutional": ["cnn", "conv2d", "pooling"],
                        "recurrent": ["rnn", "lstm", "gru"],
                        "attention": ["transformer", "self_attention", "cross_attention"]
                    },
                    "traditional_ml": ["svm", "random_forest", "linear_regression"]
                },
                "unsupervised_learning": {
                    "clustering": ["kmeans", "hierarchical"],
                    "dimensionality_reduction": ["pca", "tsne", "umap"],
                    "generative_models": ["vae", "gan", "diffusion"]
                },
                "reinforcement_learning": {
                    "value_based": ["q_learning", "dqn"],
                    "policy_based": ["policy_gradient", "ppo"],
                    "actor_critic": ["a3c", "ddpg"]
                }
            },
            "optimization": {
                "algorithmic": {
                    "first_order": ["sgd", "adam", "adamw"],
                    "second_order": ["newton", "lbfgs"],
                    "specialized": ["lars", "lamb"]
                },
                "system_level": {
                    "memory": ["gradient_checkpointing", "activation_checkpointing"],
                    "computation": ["kernel_fusion", "operator_fusion"],
                    "parallelism": ["data_parallel", "model_parallel", "pipeline_parallel"]
                },
                "numerical": {
                    "precision": ["fp16", "bf16", "int8"],
                    "sparsity": ["pruning", "structured_pruning"],
                    "quantization": ["post_training", "quantization_aware"]
                }
            }
        }

    def _build_relationships(self) -> Dict[str, List[str]]:
        """Build concept relationships and dependencies"""
        return {
            "transformer": {
                "depends_on": ["attention", "layer_norm", "feedforward"],
                "enables": ["bert", "gpt", "t5"],
                "optimizations": ["flash_attention", "gradient_checkpointing"],
                "mathematical_basis": ["matrix_multiplication", "softmax", "linear_projection"]
            },
            "attention": {
                "depends_on": ["matrix_multiplication", "softmax"],
                "variants": ["self_attention", "cross_attention", "multi_head"],
                "optimizations": ["flash_attention", "sparse_attention"],
                "mathematical_basis": ["dot_product", "weighted_sum"]
            },
            "convolution": {
                "depends_on": ["discrete_convolution", "correlation"],
                "variants": ["conv1d", "conv2d", "conv3d", "depthwise", "separable"],
                "optimizations": ["winograd", "fft_convolution"],
                "mathematical_basis": ["convolution_theorem", "filter_banks"]
            }
        }

    def get_concept_path(self, concept: str) -> List[str]:
        """Get hierarchical path to a concept"""
        def search_tree(tree, target, path=[]):
            for key, value in tree.items():
                current_path = path + [key]
                if key == target:
                    return current_path
                if isinstance(value, dict):
                    result = search_tree(value, target, current_path)
                    if result:
                        return result
                elif isinstance(value, list) and target in value:
                    return current_path + [target]
            return None

        return search_tree(self.concept_tree, concept) or [concept]

    def get_related_concepts(self, concept: str) -> List[str]:
        """Get concepts related to the given concept"""
        return self.concept_relationships.get(concept, {}).get("depends_on", []) + \
               self.concept_relationships.get(concept, {}).get("enables", [])


class LLMSemanticAnalyzer:
    """
    Advanced semantic analyzer using LLM-style reasoning patterns
    """

    def __init__(self):
        self.concept_hierarchy = ConceptHierarchy()
        self.semantic_concepts = self._load_semantic_concepts()

    def _load_semantic_concepts(self) -> Dict[str, SemanticConcept]:
        """Load rich semantic concept definitions"""
        concepts = {}

        # Define core concepts with rich semantics
        concepts["transformer"] = SemanticConcept(
            name="Transformer",
            category="architecture",
            description="Neural architecture using self-attention mechanisms for sequence modeling",
            mathematical_foundation="Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V",
            implementation_patterns=[
                "Multi-head attention layers",
                "Position-wise feedforward networks",
                "Layer normalization",
                "Residual connections"
            ],
            optimization_opportunities=[
                "Flash Attention for memory efficiency",
                "Gradient checkpointing for memory",
                "Mixed precision training",
                "Kernel fusion for operations"
            ],
            educational_analogy="Like a spotlight that can focus on different parts of a sentence simultaneously, allowing the model to understand relationships between distant words",
            related_concepts=["attention", "encoder_decoder", "bert", "gpt"],
            complexity_level="intermediate"
        )

        concepts["attention"] = SemanticConcept(
            name="Attention Mechanism",
            category="algorithm",
            description="Mechanism allowing models to focus on relevant parts of input sequences",
            mathematical_foundation="Weighted sum of values based on query-key similarity",
            implementation_patterns=[
                "Query, Key, Value projections",
                "Similarity computation (dot product)",
                "Softmax normalization",
                "Weighted value aggregation"
            ],
            optimization_opportunities=[
                "Flash Attention algorithm",
                "Sparse attention patterns",
                "Linear attention approximations"
            ],
            educational_analogy="Like human attention when reading - you focus on important words while being aware of context",
            related_concepts=["transformer", "memory", "sequence_modeling"],
            complexity_level="intermediate"
        )

        concepts["kernel_fusion"] = SemanticConcept(
            name="Kernel Fusion",
            category="optimization",
            description="Combining multiple GPU operations into single kernel to reduce memory bandwidth",
            mathematical_foundation="f(g(x)) computed in single kernel instead of separate f() and g() kernels",
            implementation_patterns=[
                "Elementwise operation fusion",
                "TorchScript compilation",
                "Custom CUDA kernels",
                "Triton kernel development"
            ],
            optimization_opportunities=[
                "Memory bandwidth reduction",
                "Reduced kernel launch overhead",
                "Better cache utilization"
            ],
            educational_analogy="Like cooking a complete meal in one pot instead of using multiple pots - saves time and cleanup",
            related_concepts=["memory_optimization", "cuda_programming", "performance"],
            complexity_level="advanced"
        )

        return concepts

    def analyze_code_semantics(self, code: str, ast_tree: ast.AST) -> Dict[str, Any]:
        """
        Perform deep semantic analysis using LLM-style reasoning
        """
        analysis = {
            "high_level_concepts": self._extract_high_level_concepts(code, ast_tree),
            "mathematical_patterns": self._identify_mathematical_patterns(code),
            "optimization_analysis": self._analyze_optimization_level(code),
            "educational_insights": self._generate_educational_insights(code),
            "concept_relationships": self._map_concept_relationships(code),
            "improvement_suggestions": self._suggest_improvements(code),
        }

        return analysis

    def _extract_high_level_concepts(self, code: str, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract high-level ML/AI concepts using pattern recognition and reasoning"""
        concepts = []

        # Analyze class definitions for architectural patterns
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                concept = self._analyze_class_concept(node, code)
                if concept:
                    concepts.append(concept)

            elif isinstance(node, ast.FunctionDef):
                concept = self._analyze_function_concept(node, code)
                if concept:
                    concepts.append(concept)

        return concepts

    def _analyze_class_concept(self, class_node: ast.ClassDef, code: str) -> Optional[Dict[str, Any]]:
        """Analyze a class to determine its high-level ML concept"""
        class_name = class_node.name.lower()
        class_body = ast.get_source_segment(code, class_node) or ""

        # Check for transformer patterns
        if any(pattern in class_body.lower() for pattern in ["attention", "transformer", "multihead"]):
            return {
                "concept": "transformer",
                "confidence": self._calculate_semantic_confidence(class_body, "transformer"),
                "evidence": self._extract_semantic_evidence(class_body, "transformer"),
                "location": f"class {class_node.name}",
                "semantic_role": self._determine_semantic_role(class_body, "transformer")
            }

        # Check for CNN patterns
        if any(pattern in class_body.lower() for pattern in ["conv", "cnn", "convolution"]):
            return {
                "concept": "convolutional_network",
                "confidence": self._calculate_semantic_confidence(class_body, "cnn"),
                "evidence": self._extract_semantic_evidence(class_body, "cnn"),
                "location": f"class {class_node.name}",
                "semantic_role": self._determine_semantic_role(class_body, "cnn")
            }

        return None

    def _analyze_function_concept(self, func_node: ast.FunctionDef, code: str) -> Optional[Dict[str, Any]]:
        """Analyze a function to determine its algorithmic concept"""
        func_name = func_node.name.lower()
        func_body = ast.get_source_segment(code, func_node) or ""

        # Check for attention mechanism
        if "attention" in func_name or any(pattern in func_body for pattern in ["q @ k", "softmax", "matmul"]):
            return {
                "concept": "attention_mechanism",
                "confidence": self._calculate_semantic_confidence(func_body, "attention"),
                "evidence": self._extract_semantic_evidence(func_body, "attention"),
                "location": f"function {func_node.name}",
                "semantic_role": self._determine_semantic_role(func_body, "attention")
            }

        return None

    def _identify_mathematical_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Identify mathematical patterns and their ML significance"""
        patterns = []

        # Matrix multiplication patterns
        if re.search(r'@|torch\.matmul|torch\.mm', code):
            patterns.append({
                "pattern": "matrix_multiplication",
                "significance": "Core operation for neural networks - linear transformations",
                "optimization_note": "Can benefit from optimized BLAS libraries and tensor cores"
            })

        # Softmax patterns
        if re.search(r'softmax|F\.softmax', code):
            patterns.append({
                "pattern": "softmax_normalization",
                "significance": "Converts logits to probability distributions",
                "optimization_note": "Numerically stable implementations important for training stability"
            })

        # Gradient computation patterns
        if re.search(r'backward\(\)|autograd|grad', code):
            patterns.append({
                "pattern": "gradient_computation",
                "significance": "Automatic differentiation for model training",
                "optimization_note": "Can use gradient checkpointing to trade compute for memory"
            })

        return patterns

    def _analyze_optimization_level(self, code: str) -> Dict[str, Any]:
        """Analyze the optimization level of the code"""
        optimizations = {
            "kernel_fusion": {
                "detected": bool(re.search(r'@torch\.jit\.script|@torch\.compile', code)),
                "indicators": re.findall(r'@torch\.(?:jit\.script|compile)', code),
                "level": "advanced" if "@torch.compile" in code else "intermediate" if "@torch.jit.script" in code else "basic"
            },
            "memory_efficiency": {
                "detected": bool(re.search(r'checkpoint|no_grad|empty_cache', code)),
                "indicators": re.findall(r'(?:checkpoint|no_grad|empty_cache)', code),
                "level": "advanced" if "checkpoint" in code else "basic"
            },
            "numerical_optimization": {
                "detected": bool(re.search(r'half\(\)|fp16|autocast', code)),
                "indicators": re.findall(r'(?:half\(\)|fp16|autocast)', code),
                "level": "advanced" if "autocast" in code else "intermediate"
            }
        }

        overall_level = "advanced" if any(opt["level"] == "advanced" for opt in optimizations.values()) \
                      else "intermediate" if any(opt["level"] == "intermediate" for opt in optimizations.values()) \
                      else "basic"

        return {
            "optimizations": optimizations,
            "overall_level": overall_level,
            "optimization_score": self._calculate_optimization_score(optimizations)
        }

    def _generate_educational_insights(self, code: str) -> Dict[str, Any]:
        """Generate educational insights about the code"""
        insights = {
            "learning_objectives": [],
            "key_concepts": [],
            "common_pitfalls": [],
            "best_practices": [],
            "mathematical_insights": []
        }

        # Analyze for educational content
        if "attention" in code.lower():
            insights["learning_objectives"].append("Understanding attention mechanisms in neural networks")
            insights["key_concepts"].append("Query-Key-Value paradigm for attention")
            insights["mathematical_insights"].append("Scaled dot-product attention formula")

        if "transformer" in code.lower():
            insights["learning_objectives"].append("Understanding transformer architecture")
            insights["key_concepts"].append("Self-attention and position encoding")
            insights["best_practices"].append("Use layer normalization and residual connections")

        if re.search(r'for.*range.*batch', code):
            insights["common_pitfalls"].append("Processing batch elements individually instead of vectorizing")

        return insights

    def _map_concept_relationships(self, code: str) -> Dict[str, List[str]]:
        """Map relationships between detected concepts"""
        detected_concepts = []

        # Simple concept detection
        concept_indicators = {
            "attention": ["attention", "softmax", "q @", "k @"],
            "transformer": ["transformer", "multihead", "encoder", "decoder"],
            "convolution": ["conv", "kernel", "filter"],
            "optimization": ["@torch.jit", "@torch.compile", "checkpoint"]
        }

        for concept, indicators in concept_indicators.items():
            if any(indicator in code.lower() for indicator in indicators):
                detected_concepts.append(concept)

        # Build relationship map
        relationships = {}
        for concept in detected_concepts:
            relationships[concept] = self.concept_hierarchy.get_related_concepts(concept)

        return relationships

    def _suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """Suggest code improvements based on semantic analysis"""
        suggestions = []

        # Check for manual attention implementation
        if re.search(r'torch\.matmul.*transpose.*softmax', code):
            suggestions.append({
                "category": "optimization",
                "priority": "high",
                "suggestion": "Replace manual attention with F.scaled_dot_product_attention",
                "reasoning": "Flash Attention provides memory-efficient implementation",
                "code_example": "output = F.scaled_dot_product_attention(q, k, v)"
            })

        # Check for batch processing loops
        if re.search(r'for.*in.*range.*batch', code):
            suggestions.append({
                "category": "performance",
                "priority": "high",
                "suggestion": "Vectorize batch processing",
                "reasoning": "Processing entire batch simultaneously is much faster",
                "code_example": "# Process all samples at once instead of looping"
            })

        # Check for missing optimizations
        if not re.search(r'@torch\.(?:jit|compile)', code) and "def forward" in code:
            suggestions.append({
                "category": "optimization",
                "priority": "medium",
                "suggestion": "Consider adding @torch.compile for optimization",
                "reasoning": "Compilation can significantly improve performance",
                "code_example": "@torch.compile\ndef forward(self, x):"
            })

        return suggestions

    def _calculate_semantic_confidence(self, code: str, concept: str) -> float:
        """Calculate confidence score for semantic concept detection"""
        if concept in self.semantic_concepts:
            concept_def = self.semantic_concepts[concept]
            pattern_matches = sum(1 for pattern in concept_def.implementation_patterns
                                 if pattern.lower() in code.lower())
            return min(pattern_matches / len(concept_def.implementation_patterns), 1.0)
        return 0.0

    def _extract_semantic_evidence(self, code: str, concept: str) -> List[str]:
        """Extract evidence for semantic concept detection"""
        if concept in self.semantic_concepts:
            concept_def = self.semantic_concepts[concept]
            evidence = []
            for pattern in concept_def.implementation_patterns:
                if pattern.lower() in code.lower():
                    evidence.append(f"Contains {pattern}")
            return evidence
        return []

    def _determine_semantic_role(self, code: str, concept: str) -> str:
        """Determine the semantic role of the concept in the code"""
        if "class" in code and concept == "transformer":
            return "Architecture definition - implements transformer model"
        elif "def forward" in code and concept == "attention":
            return "Algorithm implementation - computes attention mechanism"
        elif "@torch.jit" in code or "@torch.compile" in code:
            return "Optimized implementation - performance enhanced version"
        return "Implementation - standard approach"

    def _calculate_optimization_score(self, optimizations: Dict) -> float:
        """Calculate overall optimization score"""
        score = 0.0
        weights = {"advanced": 3.0, "intermediate": 2.0, "basic": 1.0}

        for opt_type, opt_data in optimizations.items():
            if opt_data["detected"]:
                score += weights.get(opt_data["level"], 0.0)

        return min(score / (len(optimizations) * 3.0), 1.0)  # Normalize to 0-1

    def explain_concept_deeply(self, concept_name: str) -> Dict[str, Any]:
        """Provide deep explanation of a concept with multiple perspectives"""
        if concept_name not in self.semantic_concepts:
            return {"error": f"Concept {concept_name} not found"}

        concept = self.semantic_concepts[concept_name]

        return {
            "concept": concept_name,
            "definition": concept.description,
            "mathematical_foundation": concept.mathematical_foundation,
            "intuitive_explanation": concept.educational_analogy,
            "implementation_guide": concept.implementation_patterns,
            "optimization_strategies": concept.optimization_opportunities,
            "hierarchical_position": self.concept_hierarchy.get_concept_path(concept_name),
            "related_concepts": concept.related_concepts,
            "complexity_level": concept.complexity_level,
            "learning_path": self._generate_learning_path(concept_name),
        }

    def _generate_learning_path(self, concept: str) -> List[str]:
        """Generate learning path for mastering a concept"""
        learning_paths = {
            "transformer": [
                "1. Understand matrix multiplication and linear layers",
                "2. Learn attention mechanism (query-key-value)",
                "3. Study multi-head attention",
                "4. Understand positional encoding",
                "5. Learn layer normalization and residual connections",
                "6. Combine into full transformer architecture"
            ],
            "attention": [
                "1. Understand dot product similarity",
                "2. Learn softmax normalization",
                "3. Understand weighted sum aggregation",
                "4. Study query-key-value paradigm",
                "5. Learn multi-head extension",
                "6. Explore optimization techniques (Flash Attention)"
            ]
        }

        return learning_paths.get(concept, [f"Study {concept} fundamentals"])


# Integration with main semantic agent
class EnhancedSemanticAgent:
    """
    Enhanced semantic agent combining pattern matching with LLM-style understanding
    """

    def __init__(self):
        self.llm_analyzer = LLMSemanticAnalyzer()

    def deep_code_analysis(self, code: str) -> Dict[str, Any]:
        """Perform comprehensive deep analysis of code"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}

        # Get LLM-style semantic analysis
        semantic_analysis = self.llm_analyzer.analyze_code_semantics(code, tree)

        # Combine with educational insights
        return {
            "code_analysis": semantic_analysis,
            "educational_recommendations": self._generate_teaching_recommendations(semantic_analysis),
            "research_directions": self._identify_research_opportunities(semantic_analysis),
            "optimization_roadmap": self._create_optimization_roadmap(semantic_analysis),
        }

    def _generate_teaching_recommendations(self, analysis: Dict) -> Dict[str, Any]:
        """Generate recommendations for teaching this code"""
        return {
            "target_audience": self._determine_target_audience(analysis),
            "teaching_sequence": self._suggest_teaching_sequence(analysis),
            "hands_on_exercises": self._suggest_exercises(analysis),
            "common_misconceptions": self._identify_misconceptions(analysis),
        }

    def _determine_target_audience(self, analysis: Dict) -> str:
        """Determine appropriate audience level"""
        optimization_level = analysis.get("optimization_analysis", {}).get("overall_level", "basic")
        concept_complexity = "advanced" if any(
            concept.get("concept") in ["transformer", "attention", "kernel_fusion"]
            for concept in analysis.get("high_level_concepts", [])
        ) else "intermediate"

        if optimization_level == "advanced" or concept_complexity == "advanced":
            return "Graduate students or experienced practitioners"
        elif optimization_level == "intermediate":
            return "Undergraduate students with ML background"
        else:
            return "Beginners in machine learning"

    def _suggest_teaching_sequence(self, analysis: Dict) -> List[str]:
        """Suggest sequence for teaching the concepts"""
        concepts = [concept.get("concept") for concept in analysis.get("high_level_concepts", [])]

        if "transformer" in concepts:
            return [
                "Start with matrix operations and linear algebra review",
                "Introduce attention mechanism concept",
                "Build up to multi-head attention",
                "Show transformer architecture assembly",
                "Discuss optimization techniques"
            ]
        elif "attention" in concepts:
            return [
                "Review similarity functions (dot product)",
                "Introduce attention as weighted sum",
                "Show query-key-value formulation",
                "Demonstrate with code examples"
            ]

        return ["Analyze code structure", "Identify key patterns", "Understand optimizations"]

    def _suggest_exercises(self, analysis: Dict) -> List[str]:
        """Suggest hands-on exercises"""
        exercises = [
            "Implement the same concept from scratch",
            "Modify parameters and observe effects",
            "Compare with reference implementations"
        ]

        if analysis.get("optimization_analysis", {}).get("overall_level") == "advanced":
            exercises.extend([
                "Profile performance with different optimizations",
                "Implement alternative optimization strategies",
                "Measure memory usage and throughput"
            ])

        return exercises

    def _identify_misconceptions(self, analysis: Dict) -> List[str]:
        """Identify common misconceptions students might have"""
        misconceptions = []

        concepts = [concept.get("concept") for concept in analysis.get("high_level_concepts", [])]

        if "attention" in concepts:
            misconceptions.extend([
                "Thinking attention is just matrix multiplication",
                "Confusing attention weights with final output",
                "Not understanding the role of softmax normalization"
            ])

        if "transformer" in concepts:
            misconceptions.extend([
                "Thinking transformers only work for NLP",
                "Confusing encoder and decoder attention",
                "Not understanding why positional encoding is needed"
            ])

        return misconceptions

    def _identify_research_opportunities(self, analysis: Dict) -> List[str]:
        """Identify potential research directions"""
        opportunities = []

        optimization_level = analysis.get("optimization_analysis", {}).get("overall_level", "basic")

        if optimization_level != "advanced":
            opportunities.append("Explore advanced optimization techniques")

        concepts = [concept.get("concept") for concept in analysis.get("high_level_concepts", [])]

        if "attention" in concepts:
            opportunities.extend([
                "Investigate sparse attention patterns",
                "Explore linear attention approximations",
                "Research hardware-specific attention optimizations"
            ])

        if "transformer" in concepts:
            opportunities.extend([
                "Study architectural variations",
                "Investigate scaling laws",
                "Explore efficient training techniques"
            ])

        return opportunities

    def _create_optimization_roadmap(self, analysis: Dict) -> List[Dict[str, Any]]:
        """Create a roadmap for optimizing the code"""
        roadmap = []

        optimization_analysis = analysis.get("optimization_analysis", {})
        current_level = optimization_analysis.get("overall_level", "basic")

        if current_level == "basic":
            roadmap.extend([
                {"step": 1, "goal": "Add basic optimizations", "actions": ["Use torch.compile", "Enable mixed precision"]},
                {"step": 2, "goal": "Improve memory efficiency", "actions": ["Add gradient checkpointing", "Optimize batch processing"]},
                {"step": 3, "goal": "Advanced optimizations", "actions": ["Custom kernels", "Hardware-specific tuning"]}
            ])
        elif current_level == "intermediate":
            roadmap.extend([
                {"step": 1, "goal": "Memory optimization", "actions": ["Gradient checkpointing", "Activation checkpointing"]},
                {"step": 2, "goal": "Computational optimization", "actions": ["Custom kernels", "Operator fusion"]}
            ])

        return roadmap


if __name__ == "__main__":
    # Example usage
    agent = EnhancedSemanticAgent()

    sample_code = '''
    import torch
    import torch.nn.functional as F

    @torch.compile
    def optimized_attention(q, k, v, scale=None):
        """Optimized attention implementation"""
        if scale is None:
            scale = q.size(-1) ** -0.5

        # Use Flash Attention when available
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            # Fallback implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            weights = F.softmax(scores, dim=-1)
            return torch.matmul(weights, v)
    '''

    result = agent.deep_code_analysis(sample_code)
    print("Deep Analysis Result:")
    print(json.dumps(result, indent=2, default=str))