#!/usr/bin/env python3
"""
Orientation Concepts: How "An orientation-less is oriented"

This module explores the mathematical and linguistic concepts of how something 
without inherent orientation can become oriented through context, transformation,
or interpretation. This relates to topology (homology vs homotopy) and NLP 
(meaning emerging from context).
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math


class OrientationType(Enum):
    """Types of orientation in different contexts"""
    NONE = "none"                    # No inherent orientation
    MATHEMATICAL = "mathematical"    # Mathematical/topological orientation
    LINGUISTIC = "linguistic"        # Linguistic/semantic orientation
    CONTEXTUAL = "contextual"       # Context-dependent orientation


@dataclass
class OrientationState:
    """Represents the orientation state of an object or concept"""
    initial_orientation: OrientationType
    current_orientation: OrientationType
    transformation_context: str
    orientation_strength: float  # 0.0 (no orientation) to 1.0 (strong orientation)
    confidence: float


class TopologicalOrientation:
    """
    Mathematical orientation concepts from topology.
    
    Demonstrates how homology (orientation-less) relates to homotopy (oriented).
    """
    
    def __init__(self):
        self.name = "Topological Orientation Analyzer"
    
    def homology_vs_homotopy_orientation(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare homology (loses orientation) vs homotopy (preserves orientation).
        
        This demonstrates the core mathematical concept behind "orientation-less is oriented":
        - Homology groups detect cycles but lose winding direction
        - Homotopy groups preserve both cycles AND their orientation
        """
        result = {
            'homology_analysis': self._analyze_homology(cycle_data),
            'homotopy_analysis': self._analyze_homotopy(cycle_data),
            'orientation_emergence': self._analyze_orientation_emergence(cycle_data)
        }
        return result
    
    def _analyze_homology(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Homology analysis - detects cycles but loses orientation information.
        
        In H_1(S^1) = Z, we know there are cycles, but not their winding direction.
        """
        cycles = cycle_data.get('cycles', [])
        
        # Homology only cares about the existence of cycles, not direction
        homology_rank = len([c for c in cycles if c.get('is_non_trivial', False)])
        
        return {
            'rank': homology_rank,
            'detects_cycles': homology_rank > 0,
            'preserves_orientation': False,  # This is the key point!
            'information_lost': ['winding_direction', 'orientation', 'signed_count'],
            'description': 'Homology detects cycles but loses orientation information'
        }
    
    def _analyze_homotopy(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Homotopy analysis - preserves orientation information.
        
        In π_1(S^1) = Z, we know both cycles AND their winding numbers.
        """
        cycles = cycle_data.get('cycles', [])
        
        # Homotopy preserves winding numbers and orientation
        winding_numbers = []
        for cycle in cycles:
            if cycle.get('is_non_trivial', False):
                # Extract orientation information
                direction = cycle.get('direction', 'counterclockwise')
                multiplicity = cycle.get('multiplicity', 1)
                winding_number = multiplicity if direction == 'counterclockwise' else -multiplicity
                winding_numbers.append(winding_number)
        
        return {
            'fundamental_group': winding_numbers,
            'preserves_orientation': True,  # This is what makes it "oriented"!
            'winding_numbers': winding_numbers,
            'signed_information': True,
            'description': 'Homotopy preserves both cycles and their orientation'
        }
    
    def _analyze_orientation_emergence(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how orientation emerges from orientation-less structures.
        """
        homology_result = self._analyze_homology(cycle_data)
        homotopy_result = self._analyze_homotopy(cycle_data)
        
        # The "emergence" happens when we move from homology to homotopy
        orientation_emerged = (
            not homology_result['preserves_orientation'] and 
            homotopy_result['preserves_orientation']
        )
        
        return {
            'orientation_emerged': orientation_emerged,
            'mechanism': 'homotopy_refinement',
            'information_gained': ['winding_direction', 'signed_multiplicity', 'orientation'],
            'mathematical_insight': (
                'The same topological space can be analyzed with orientation-losing '
                '(homology) or orientation-preserving (homotopy) methods. '
                'Orientation emerges through the choice of analytical framework.'
            )
        }


class LinguisticOrientation:
    """
    Linguistic orientation concepts - how meaning gains direction through context.
    """
    
    def __init__(self):
        self.name = "Linguistic Orientation Analyzer"
    
    def analyze_semantic_orientation(self, text: str, context: Dict[str, Any]) -> OrientationState:
        """
        Analyze how semantic orientation emerges from context.
        
        Similar to mathematical orientation, linguistic meaning can be:
        - Initially orientation-less (ambiguous)
        - Gains orientation through context, pragmatics, etc.
        """
        # Detect ambiguous elements that lack inherent orientation
        ambiguous_elements = self._detect_ambiguous_elements(text)
        
        # Analyze how context provides orientation
        contextual_orientation = self._analyze_contextual_orientation(text, context)
        
        # Calculate orientation strength
        orientation_strength = self._calculate_orientation_strength(
            ambiguous_elements, contextual_orientation
        )
        
        initial_type = OrientationType.NONE if ambiguous_elements else OrientationType.LINGUISTIC
        current_type = OrientationType.CONTEXTUAL if orientation_strength > 0.5 else OrientationType.NONE
        
        return OrientationState(
            initial_orientation=initial_type,
            current_orientation=current_type,
            transformation_context=f"Context: {context.get('situation', 'unknown')}",
            orientation_strength=orientation_strength,
            confidence=min(0.9, orientation_strength + 0.1)
        )
    
    def _detect_ambiguous_elements(self, text: str) -> List[str]:
        """Detect elements in text that lack inherent orientation/meaning."""
        # Simple heuristics for ambiguous elements
        ambiguous_patterns = [
            r'\b(it|this|that|they|them)\b',  # Pronouns without clear referents
            r'\b(here|there|now|then)\b',     # Deictic expressions
            r'\b(good|bad|big|small)\b',      # Relative adjectives
        ]
        
        ambiguous_elements = []
        for pattern in ambiguous_patterns:
            import re
            matches = re.findall(pattern, text.lower())
            ambiguous_elements.extend(matches)
        
        return ambiguous_elements
    
    def _analyze_contextual_orientation(self, text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze how context provides orientation to ambiguous elements."""
        orientation_factors = {
            'speaker_identity': 0.0,
            'temporal_context': 0.0,
            'spatial_context': 0.0,
            'social_context': 0.0,
            'discourse_context': 0.0
        }
        
        # Simple scoring based on available context
        if context.get('speaker'):
            orientation_factors['speaker_identity'] = 0.8
        if context.get('time') or context.get('situation'):
            orientation_factors['temporal_context'] = 0.7
        if context.get('location'):
            orientation_factors['spatial_context'] = 0.6
        if context.get('formality_level') or context.get('power_relation'):
            orientation_factors['social_context'] = 0.7
        if len(text.split()) > 5:  # Longer texts provide more discourse context
            orientation_factors['discourse_context'] = 0.5
        
        return orientation_factors
    
    def _calculate_orientation_strength(self, ambiguous_elements: List[str], 
                                       contextual_orientation: Dict[str, float]) -> float:
        """Calculate how much orientation is gained through context."""
        if not ambiguous_elements:
            return 0.8  # Already oriented
        
        # More ambiguous elements need more context to become oriented
        ambiguity_penalty = min(0.6, len(ambiguous_elements) * 0.15)
        context_values = list(contextual_orientation.values())
        context_strength = np.mean(context_values) if context_values else 0.0
        
        # Orientation emerges when context overcomes ambiguity
        orientation_strength = max(0.0, context_strength - ambiguity_penalty)
        
        # Boost orientation when multiple context factors are present
        active_contexts = sum(1 for v in context_values if v > 0.1)
        context_boost = min(0.3, active_contexts * 0.1)
        
        return min(1.0, orientation_strength + context_boost)


class OrientationTransformer:
    """
    Demonstrates various transformations from orientation-less to oriented states.
    """
    
    def __init__(self):
        self.topological_analyzer = TopologicalOrientation()
        self.linguistic_analyzer = LinguisticOrientation()
    
    def demonstrate_orientation_emergence(self) -> Dict[str, Any]:
        """
        Comprehensive demonstration of how orientation emerges.
        """
        results = {
            'mathematical_example': self._mathematical_orientation_example(),
            'linguistic_example': self._linguistic_orientation_example(),
            'philosophical_insight': self._philosophical_insight()
        }
        return results
    
    def _mathematical_orientation_example(self) -> Dict[str, Any]:
        """Example: Circle loops - homology vs homotopy orientation."""
        # Example cycle data for S^1 (circle)
        cycle_data = {
            'cycles': [
                {
                    'is_non_trivial': True,
                    'direction': 'counterclockwise',
                    'multiplicity': 1
                },
                {
                    'is_non_trivial': True,
                    'direction': 'clockwise',
                    'multiplicity': 2
                }
            ]
        }
        
        analysis = self.topological_analyzer.homology_vs_homotopy_orientation(cycle_data)
        
        return {
            'title': 'Mathematical Orientation Emergence',
            'example': 'Loops around a circle S^1',
            'analysis': analysis,
            'key_insight': (
                'Homology H_1(S^1) = Z detects cycles but loses orientation. '
                'Homotopy π_1(S^1) = Z preserves winding direction. '
                'The same structure becomes "oriented" through homotopy analysis.'
            )
        }
    
    def _linguistic_orientation_example(self) -> Dict[str, Any]:
        """Example: Ambiguous text gaining orientation through context."""
        test_cases = [
            {
                'text': 'That is good.',
                'contexts': [
                    {'speaker': 'teacher', 'situation': 'grading', 'referent': 'student work'},
                    {'speaker': 'friend', 'situation': 'casual', 'referent': 'unknown'},
                    {}  # No context
                ]
            },
            {
                'text': 'They went there.',
                'contexts': [
                    {'speaker': 'narrator', 'situation': 'story', 'temporal': 'past'},
                    {'speaker': 'witness', 'situation': 'testimony', 'spatial': 'courthouse'},
                    {}  # No context
                ]
            }
        ]
        
        results = []
        for case in test_cases:
            case_results = {
                'text': case['text'],
                'orientation_states': []
            }
            
            for context in case['contexts']:
                orientation_state = self.linguistic_analyzer.analyze_semantic_orientation(
                    case['text'], context
                )
                case_results['orientation_states'].append({
                    'context': context,
                    'orientation_state': orientation_state
                })
            
            results.append(case_results)
        
        return {
            'title': 'Linguistic Orientation Emergence',
            'examples': results,
            'key_insight': (
                'Ambiguous text (orientation-less) gains semantic orientation '
                'through context. The same words become "oriented" toward '
                'specific meanings when context is provided.'
            )
        }
    
    def _philosophical_insight(self) -> Dict[str, str]:
        """The philosophical insight behind 'An orientation-less is oriented'."""
        return {
            'core_principle': (
                'Objects, concepts, or structures can lack inherent orientation '
                'yet become oriented through external framework, context, or analysis.'
            ),
            'mathematical_manifestation': (
                'Homology groups are orientation-less (detect cycles without direction) '
                'while homotopy groups are oriented (preserve winding/direction). '
                'The same topological space exhibits both properties simultaneously.'
            ),
            'linguistic_manifestation': (
                'Ambiguous language lacks semantic orientation until context '
                'provides interpretive direction. Meaning emerges through '
                'the interaction of text and context.'
            ),
            'general_principle': (
                'Orientation is often not intrinsic but emerges through the '
                'relationship between an object and its analytical framework '
                'or interpretive context. The same entity can be both '
                'orientation-less and oriented depending on how we examine it.'
            ),
            'paradox_resolution': (
                'The apparent paradox "orientation-less is oriented" resolves '
                'when we recognize that orientation is relational, not absolute. '
                'An entity can simultaneously lack intrinsic orientation while '
                'gaining orientation through external analysis or context.'
            )
        }


def main():
    """Demonstrate the orientation concepts."""
    print("=" * 80)
    print("Orientation Concepts: How 'An orientation-less is oriented'")
    print("=" * 80)
    print()
    
    transformer = OrientationTransformer()
    demo = transformer.demonstrate_orientation_emergence()
    
    # Mathematical example
    print("🔢 MATHEMATICAL EXAMPLE")
    print("-" * 40)
    math_ex = demo['mathematical_example']
    print(f"Title: {math_ex['title']}")
    print(f"Example: {math_ex['example']}")
    print(f"Key Insight: {math_ex['key_insight']}")
    print()
    
    analysis = math_ex['analysis']
    print("Homology Analysis (orientation-less):")
    print(f"  - Detects cycles: {analysis['homology_analysis']['detects_cycles']}")
    print(f"  - Preserves orientation: {analysis['homology_analysis']['preserves_orientation']}")
    print(f"  - Information lost: {', '.join(analysis['homology_analysis']['information_lost'])}")
    print()
    
    print("Homotopy Analysis (oriented):")
    print(f"  - Preserves orientation: {analysis['homotopy_analysis']['preserves_orientation']}")
    print(f"  - Winding numbers: {analysis['homotopy_analysis']['winding_numbers']}")
    print(f"  - Signed information: {analysis['homotopy_analysis']['signed_information']}")
    print()
    
    # Linguistic example
    print("💬 LINGUISTIC EXAMPLE")
    print("-" * 40)
    ling_ex = demo['linguistic_example']
    print(f"Title: {ling_ex['title']}")
    print(f"Key Insight: {ling_ex['key_insight']}")
    print()
    
    for i, example in enumerate(ling_ex['examples'][:1], 1):  # Show first example
        print(f"Example {i}: '{example['text']}'")
        for j, state_info in enumerate(example['orientation_states']):
            context = state_info['context']
            state = state_info['orientation_state']
            context_desc = f"Context: {context}" if context else "No context"
            print(f"  {context_desc}")
            print(f"    Initial: {state.initial_orientation.value}")
            print(f"    Current: {state.current_orientation.value}")
            print(f"    Strength: {state.orientation_strength:.2f}")
        print()
    
    # Philosophical insight
    print("🤔 PHILOSOPHICAL INSIGHT")
    print("-" * 40)
    insight = demo['philosophical_insight']
    print(f"Core Principle: {insight['core_principle']}")
    print()
    print(f"Paradox Resolution: {insight['paradox_resolution']}")
    print()


if __name__ == "__main__":
    main()