"""
Mathematical Expression Analyzer for NLP
數學表達式分析器 (用於自然語言處理)

This module integrates the Lie bracket computational framework with the existing
NLP framework to analyze mathematical expressions in natural language text.

此模組將李括號計算框架與現有的NLP框架整合，用於分析自然語言文本中的數學表達式。
"""

import re
import numpy as np
import sympy as sp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import our Lie bracket framework
from LieBracket import LieBracketFramework, LieElement, LieAlgebraType

# Try to import existing NLP components
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EXPRESSION_EVALUATOR_AVAILABLE = True
except ImportError:
    EXPRESSION_EVALUATOR_AVAILABLE = False

try:
    from SubtextAnalyzer import SubtextAnalyzer
    SUBTEXT_ANALYZER_AVAILABLE = True
except ImportError:
    SUBTEXT_ANALYZER_AVAILABLE = False


@dataclass
class MathematicalConcept:
    """Represents a mathematical concept found in text"""
    name: str
    category: str  # algebra, geometry, analysis, physics, etc.
    context: str
    lie_algebra_relevance: float  # 0-1 score
    physical_interpretation: Optional[str] = None
    mathematical_formalism: Optional[str] = None


class MathematicalExpressionAnalyzer:
    """
    Analyzes mathematical expressions and concepts in natural language
    分析自然語言中的數學表達式和概念
    """
    
    def __init__(self):
        self.lie_framework = LieBracketFramework()
        
        # Mathematical concept patterns
        self.math_patterns = {
            'lie_algebra_terms': [
                r'\blie\s+algebra\b', r'\blie\s+bracket\b', r'\bcommutator\b', 
                r'\bpoisson\s+bracket\b', r'\bjacobi\s+identity\b'
            ],
            'quantum_terms': [
                r'\bpauli\s+matric(es|x)\b', r'\bspin\s+operator\b', r'\bcommutation\s+relation\b',
                r'\bheisenberg\s+uncertainty\b', r'\bquantum\s+mechanic\b'
            ],
            'geometry_terms': [
                r'\bvector\s+field\b', r'\bdifferential\s+geometry\b', r'\brotation\s+group\b',
                r'\bso\(\d+\)\b', r'\bsu\(\d+\)\b', r'\bmanifold\b'
            ],
            'physics_terms': [
                r'\bhamiltonian\b', r'\bangular\s+momentum\b', r'\bconservation\s+law\b',
                r'\bsymplectic\b', r'\bcanonical\s+coordinate\b'
            ]
        }
        
        # Physical vs Mathematical physics indicators
        self.approach_indicators = {
            'physical_mathematics': [
                'phenomenon', 'observe', 'experiment', 'measure', 'physical reality',
                'natural law', 'empirical', 'derive from physics'
            ],
            'mathematical_physics': [
                'mathematical structure', 'abstract', 'formalism', 'group theory',
                'symmetry', 'elegant', 'beautiful', 'apply mathematics'
            ]
        }
    
    def extract_mathematical_concepts(self, text: str) -> List[MathematicalConcept]:
        """
        Extract mathematical concepts from text
        從文本中提取數學概念
        """
        concepts = []
        text_lower = text.lower()
        
        for category, patterns in self.math_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    concept_text = match.group()
                    context = self._extract_context(text, match.start(), match.end())
                    
                    # Calculate Lie algebra relevance
                    relevance = self._calculate_lie_relevance(context, category)
                    
                    concept = MathematicalConcept(
                        name=concept_text,
                        category=category.replace('_terms', ''),
                        context=context,
                        lie_algebra_relevance=relevance
                    )
                    
                    # Add interpretations based on category
                    if category == 'quantum_terms':
                        concept.physical_interpretation = "Quantum mechanical operator or observable"
                        concept.mathematical_formalism = "SU(2) or SU(n) Lie algebra representation"
                    elif category == 'geometry_terms':
                        concept.physical_interpretation = "Geometric transformation or symmetry"
                        concept.mathematical_formalism = "SO(n) or other Lie group structure"
                    elif category == 'physics_terms':
                        concept.physical_interpretation = "Physical quantity or conservation law"
                        concept.mathematical_formalism = "Symplectic geometry or Hamiltonian formalism"
                    
                    concepts.append(concept)
        
        return concepts
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around a matched term"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _calculate_lie_relevance(self, context: str, category: str) -> float:
        """Calculate how relevant the context is to Lie algebra"""
        base_relevance = {
            'lie_algebra': 1.0,
            'quantum': 0.8,
            'geometry': 0.7,
            'physics': 0.6
        }
        
        category_key = category.replace('_terms', '')
        relevance = base_relevance.get(category_key, 0.3)
        
        # Boost relevance if Lie-specific terms appear in context
        lie_boost_terms = ['bracket', 'commutator', 'algebra', 'group', 'symmetry']
        for term in lie_boost_terms:
            if term in context.lower():
                relevance = min(1.0, relevance + 0.1)
        
        return relevance
    
    def classify_mathematical_approach(self, text: str) -> Dict[str, float]:
        """
        Classify whether text represents physical mathematics or mathematical physics approach
        分類文本是否代表物理數學或數學物理方法
        """
        text_lower = text.lower()
        scores = {'physical_mathematics': 0.0, 'mathematical_physics': 0.0}
        
        for approach, indicators in self.approach_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    scores[approach] += 1.0
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {'physical_mathematics': 0.5, 'mathematical_physics': 0.5}
        
        return scores
    
    def analyze_lie_bracket_expression(self, expression: str) -> Dict[str, Any]:
        """
        Analyze a Lie bracket expression in natural language
        分析自然語言中的李括號表達式
        """
        result = {
            'expression': expression,
            'mathematical_concepts': [],
            'approach_classification': {},
            'lie_bracket_structure': None,
            'interpretation': "",
            'examples': []
        }
        
        # Extract mathematical concepts
        concepts = self.extract_mathematical_concepts(expression)
        result['mathematical_concepts'] = concepts
        
        # Classify approach
        result['approach_classification'] = self.classify_mathematical_approach(expression)
        
        # Detect Lie bracket patterns
        bracket_pattern = r'\[([^,\]]+),\s*([^,\]]+)\]'
        bracket_matches = re.findall(bracket_pattern, expression)
        
        if bracket_matches:
            result['lie_bracket_structure'] = bracket_matches
            result['interpretation'] = self._interpret_bracket_structure(bracket_matches, concepts)
        
        # Generate examples based on detected concepts
        result['examples'] = self._generate_examples(concepts)
        
        return result
    
    def _interpret_bracket_structure(self, brackets: List[Tuple[str, str]], 
                                   concepts: List[MathematicalConcept]) -> str:
        """Interpret the meaning of detected Lie bracket structures"""
        interpretations = []
        
        for left, right in brackets:
            left_clean = left.strip()
            right_clean = right.strip()
            
            # Check if this looks like a commutator
            if any(concept.category == 'quantum' for concept in concepts):
                interpretations.append(
                    f"[{left_clean}, {right_clean}] represents a quantum mechanical commutator, "
                    f"measuring the non-commutativity of observables {left_clean} and {right_clean}."
                )
            elif any(concept.category == 'geometry' for concept in concepts):
                interpretations.append(
                    f"[{left_clean}, {right_clean}] represents the Lie bracket of vector fields, "
                    f"capturing the non-commutativity of infinitesimal transformations."
                )
            elif any(concept.category == 'physics' for concept in concepts):
                interpretations.append(
                    f"{{{left_clean}, {right_clean}}} represents a Poisson bracket, "
                    f"encoding the time evolution and conservation laws in classical mechanics."
                )
            else:
                interpretations.append(
                    f"[{left_clean}, {right_clean}] represents an abstract Lie bracket operation."
                )
        
        return " ".join(interpretations)
    
    def _generate_examples(self, concepts: List[MathematicalConcept]) -> List[Dict[str, str]]:
        """Generate concrete examples based on detected concepts"""
        examples = []
        
        concept_categories = {concept.category for concept in concepts}
        
        if 'quantum' in concept_categories:
            examples.append({
                'type': 'Quantum Mechanics',
                'expression': '[σ_x, σ_y] = 2iσ_z',
                'description': 'Pauli matrix commutation relation',
                'physical_meaning': 'Spin measurements in different directions are incompatible'
            })
        
        if 'geometry' in concept_categories:
            examples.append({
                'type': 'Differential Geometry',
                'expression': '[X, Y] = LY·X',
                'description': 'Lie bracket of vector fields',
                'physical_meaning': 'Non-commutativity of infinitesimal transformations'
            })
        
        if 'physics' in concept_categories:
            examples.append({
                'type': 'Classical Mechanics',
                'expression': '{H, L} = 0',
                'description': 'Poisson bracket of Hamiltonian and angular momentum',
                'physical_meaning': 'Angular momentum is conserved'
            })
        
        return examples
    
    def integrate_with_nlp_framework(self, text: str) -> Dict[str, Any]:
        """
        Integrate mathematical analysis with existing NLP framework
        將數學分析與現有NLP框架整合
        """
        result = {
            'text': text,
            'mathematical_analysis': self.analyze_lie_bracket_expression(text),
            'human_expression_evaluation': None,
            'subtext_analysis': None,
            'synthesis': ""
        }
        
        # Use Human Expression Evaluator if available
        if EXPRESSION_EVALUATOR_AVAILABLE:
            try:
                evaluator = HumanExpressionEvaluator()
                context = ExpressionContext(
                    situation='academic',
                    formality_level='formal',
                    cultural_background='scientific'
                )
                result['human_expression_evaluation'] = evaluator.comprehensive_evaluation(text, context)
            except Exception as e:
                result['human_expression_evaluation'] = {'error': str(e)}
        
        # Use Subtext Analyzer if available
        if SUBTEXT_ANALYZER_AVAILABLE:
            try:
                analyzer = SubtextAnalyzer()
                result['subtext_analysis'] = analyzer.analyze_subtext(text)
            except Exception as e:
                result['subtext_analysis'] = {'error': str(e)}
        
        # Create synthesis
        result['synthesis'] = self._create_synthesis(result)
        
        return result
    
    def _create_synthesis(self, analysis_result: Dict[str, Any]) -> str:
        """Create a synthesis of all analysis components"""
        synthesis_parts = []
        
        # Mathematical content
        math_analysis = analysis_result['mathematical_analysis']
        if math_analysis['mathematical_concepts']:
            concepts_str = ", ".join([c.name for c in math_analysis['mathematical_concepts']])
            synthesis_parts.append(f"Mathematical concepts detected: {concepts_str}")
        
        # Approach classification
        approach = math_analysis['approach_classification']
        if approach:
            dominant_approach = max(approach.items(), key=lambda x: x[1])
            synthesis_parts.append(f"Dominant approach: {dominant_approach[0]} ({dominant_approach[1]:.2f})")
        
        # Lie bracket structure
        if math_analysis['lie_bracket_structure']:
            synthesis_parts.append("Contains explicit Lie bracket notation")
        
        # Human expression evaluation
        if analysis_result['human_expression_evaluation']:
            if 'integrated' in analysis_result['human_expression_evaluation']:
                score = analysis_result['human_expression_evaluation']['integrated'].get('overall_score', 0)
                synthesis_parts.append(f"Expression quality score: {score:.2f}")
        
        return ". ".join(synthesis_parts) if synthesis_parts else "No significant mathematical content detected."


def demonstrate_mathematical_nlp_integration():
    """Demonstrate the mathematical NLP integration"""
    print("=" * 70)
    print(" Mathematical Expression Analysis for NLP")
    print("=" * 70)
    
    analyzer = MathematicalExpressionAnalyzer()
    
    test_expressions = [
        "The Lie bracket [X, Y] captures the non-commutativity of vector fields in differential geometry.",
        "In quantum mechanics, the commutator [σ_x, σ_y] = 2iσ_z represents the uncertainty principle for spin measurements.",
        "Physical mathematics starts with phenomena and derives mathematical structures, while mathematical physics applies existing mathematical frameworks to physical problems.",
        "The Poisson bracket {H, L} = 0 indicates that angular momentum is conserved in isotropic systems.",
        "We observe quantum behavior and need mathematics to describe it - this is physical mathematics approach."
    ]
    
    for i, expr in enumerate(test_expressions, 1):
        print(f"\n--- Example {i} ---")
        print(f"Text: {expr}")
        
        # Analyze mathematical content
        analysis = analyzer.analyze_lie_bracket_expression(expr)
        
        print(f"Mathematical concepts found: {len(analysis['mathematical_concepts'])}")
        for concept in analysis['mathematical_concepts']:
            print(f"  - {concept.name} ({concept.category}, relevance: {concept.lie_algebra_relevance:.2f})")
        
        approach = analysis['approach_classification']
        print(f"Approach classification:")
        print(f"  Physical Mathematics: {approach['physical_mathematics']:.2f}")
        print(f"  Mathematical Physics: {approach['mathematical_physics']:.2f}")
        
        if analysis['interpretation']:
            print(f"Interpretation: {analysis['interpretation']}")
        
        # Integrated analysis
        integrated = analyzer.integrate_with_nlp_framework(expr)
        print(f"Synthesis: {integrated['synthesis']}")


if __name__ == "__main__":
    demonstrate_mathematical_nlp_integration()