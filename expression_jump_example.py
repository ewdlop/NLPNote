#!/usr/bin/env python3
"""
Example: Using Jump Analysis in Human Expression Evaluation

This script demonstrates how the jump instruction vs jump thinking analysis
can be integrated with the existing human expression evaluation framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from JumpInstructionVsJumpThinking import CognitiveJumpEngine, JumpContext
from typing import Dict, List, Any


class ExpressionJumpAnalyzer:
    """Analyzer that evaluates cognitive jumps in human expressions"""
    
    def __init__(self):
        self.cognitive_engine = CognitiveJumpEngine()
    
    def analyze_expression_coherence(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze coherence of cognitive jumps in an expression"""
        if context is None:
            context = {}
        
        # Extract key concepts from expression (simplified)
        concepts = self._extract_concepts(expression)
        
        if len(concepts) < 2:
            return {
                'coherence_score': 1.0,
                'jump_count': 0,
                'analysis': 'Expression too short for jump analysis'
            }
        
        # Create jump context
        jump_context = JumpContext(
            execution_state={},
            cognitive_state={'relevant_concepts': concepts},
            user_profile=context.get('user_profile', {}),
            environmental_factors=context.get('environment', {})
        )
        
        # Analyze jumps between concepts
        jumps = []
        total_coherence = 0.0
        
        for i in range(len(concepts) - 1):
            from_concept = concepts[i]
            to_concept = concepts[i + 1]
            
            # Calculate semantic distance
            semantic_distance = self.cognitive_engine._calculate_semantic_distance(from_concept, to_concept)
            
            # Determine jump type based on distance
            if semantic_distance < 0.3:
                jump_type = 'coherent_flow'
                coherence_contribution = 1.0
            elif semantic_distance < 0.6:
                jump_type = 'moderate_jump'
                coherence_contribution = 0.7
            else:
                jump_type = 'dramatic_jump'
                coherence_contribution = 0.3
            
            # Check for creative or meaningful connections
            mechanisms = self.cognitive_engine._identify_cognitive_mechanisms(from_concept, to_concept)
            if 'perceptual_similarity' in mechanisms or 'functional_association' in mechanisms:
                coherence_contribution += 0.2  # Boost for meaningful connections
            
            coherence_contribution = min(coherence_contribution, 1.0)
            total_coherence += coherence_contribution
            
            jumps.append({
                'from': from_concept,
                'to': to_concept,
                'type': jump_type,
                'semantic_distance': semantic_distance,
                'coherence_contribution': coherence_contribution,
                'mechanisms': mechanisms
            })
        
        # Calculate overall coherence score
        coherence_score = total_coherence / len(jumps) if jumps else 1.0
        
        return {
            'coherence_score': coherence_score,
            'jump_count': len(jumps),
            'jumps': jumps,
            'analysis': self._generate_analysis_summary(coherence_score, jumps)
        }
    
    def _extract_concepts(self, expression: str) -> List[str]:
        """Extract key concepts from expression (simplified)"""
        import re
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', expression.lower())
        
        # Filter for significant concepts (this is simplified - could use NLP)
        significant_words = [word for word in words if len(word) >= 4]
        
        # Remove duplicates while preserving order
        concepts = []
        for word in significant_words:
            if word not in concepts:
                concepts.append(word)
        
        return concepts[:10]  # Limit to first 10 concepts
    
    def _generate_analysis_summary(self, coherence_score: float, jumps: List[Dict]) -> str:
        """Generate human-readable analysis summary"""
        if coherence_score >= 0.8:
            coherence_desc = "highly coherent"
        elif coherence_score >= 0.6:
            coherence_desc = "moderately coherent"
        elif coherence_score >= 0.4:
            coherence_desc = "somewhat fragmented"
        else:
            coherence_desc = "highly fragmented"
        
        dramatic_jumps = sum(1 for jump in jumps if jump['type'] == 'dramatic_jump')
        
        if dramatic_jumps == 0:
            jump_desc = "Expression flows logically with smooth conceptual transitions."
        elif dramatic_jumps <= len(jumps) * 0.3:
            jump_desc = "Expression contains some creative leaps that may enhance expressiveness."
        else:
            jump_desc = "Expression contains many dramatic jumps that may affect comprehension."
        
        return f"Expression is {coherence_desc} (score: {coherence_score:.2f}). {jump_desc}"


def demonstrate_expression_analysis():
    """Demonstrate expression analysis using jump thinking concepts"""
    print("=== Human Expression Jump Analysis Demonstration ===\n")
    
    analyzer = ExpressionJumpAnalyzer()
    
    # Test expressions with different levels of coherence
    test_expressions = [
        {
            'text': "Programming requires logical thinking and problem-solving skills to develop efficient algorithms.",
            'description': "Coherent technical expression"
        },
        {
            'text': "The computer crashed while I was coding. The sunset looks beautiful today. My grandmother makes great cookies.",
            'description': "Expression with dramatic jumps"
        },
        {
            'text': "Learning programming is like learning a new language that speaks to machines through logic and creativity.",
            'description': "Metaphorical expression"
        },
        {
            'text': "Data structures organize information efficiently, enabling fast retrieval and processing algorithms.",
            'description': "Technical coherent flow"
        }
    ]
    
    for i, expr in enumerate(test_expressions, 1):
        print(f"{i}. {expr['description']}:")
        print(f"   Text: \"{expr['text']}\"")
        
        # Analyze the expression
        result = analyzer.analyze_expression_coherence(expr['text'])
        
        print(f"   Coherence Score: {result['coherence_score']:.2f}")
        print(f"   Jump Count: {result['jump_count']}")
        print(f"   Analysis: {result['analysis']}")
        
        # Show jump details if there are any
        if result['jumps']:
            print("   Conceptual Jumps:")
            for jump in result['jumps']:
                print(f"     • {jump['from']} → {jump['to']} "
                      f"({jump['type']}, distance: {jump['semantic_distance']:.2f})")
        
        print()
    
    # Demonstrate comparison with computational precision
    print("=== Comparison: Human vs Computational Precision ===")
    
    human_expr = "The algorithm works but reminds me of my childhood puzzle games."
    human_result = analyzer.analyze_expression_coherence(human_expr)
    
    print(f"Human Expression: \"{human_expr}\"")
    print(f"  Coherence: {human_result['coherence_score']:.2f} (probabilistic, context-dependent)")
    print(f"  Contains: {human_result['jump_count']} cognitive jumps")
    
    print("\nComputational Equivalent:")
    print("  if algorithm.works():")
    print("      return True")
    print("  Coherence: 1.00 (deterministic, context-independent)")
    print("  Contains: 1 conditional jump")
    
    print("\nKey Insight: Human expressions blend logical flow with associative thinking,")
    print("creating rich, contextual meaning that differs fundamentally from")
    print("computational precision.")


if __name__ == "__main__":
    demonstrate_expression_analysis()