#!/usr/bin/env python3
"""
Quantum-Inspired Natural Language Processing Module
量子啟發的自然語言處理模組

This module implements quantum-inspired algorithms for natural language processing,
demonstrating practical applications of quantum superposition concepts in linguistics.

本模組實現了量子啟發的自然語言處理算法，
展示了量子疊加概念在語言學中的實際應用。
"""

import math
import random
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import cmath  # For complex number operations


@dataclass
class QuantumState:
    """
    Represents a quantum state for linguistic processing
    表示用於語言處理的量子態
    """
    amplitudes: Dict[str, complex]
    
    def __post_init__(self):
        """Normalize the quantum state"""
        self.normalize()
    
    def normalize(self):
        """Normalize amplitudes to ensure |ψ⟩ has unit norm"""
        total_probability = sum(abs(amp)**2 for amp in self.amplitudes.values())
        if total_probability > 0:
            norm_factor = math.sqrt(total_probability)
            for key in self.amplitudes:
                self.amplitudes[key] /= norm_factor
    
    def probability(self, state: str) -> float:
        """Calculate probability of measuring specific state"""
        if state in self.amplitudes:
            return abs(self.amplitudes[state])**2
        return 0.0
    
    def measure(self) -> str:
        """Perform quantum measurement, collapsing to a specific state"""
        states = list(self.amplitudes.keys())
        probabilities = [self.probability(state) for state in states]
        
        if not states or sum(probabilities) == 0:
            return ""
        
        return random.choices(states, weights=probabilities)[0]


class QuantumSemanticProcessor:
    """
    Quantum-inspired semantic processing for natural language
    量子啟發的自然語言語義處理器
    """
    
    def __init__(self):
        self.semantic_space = {}
        self.entanglement_pairs = []
        self.context_operators = {}
        
    def create_semantic_superposition(self, text: str) -> QuantumState:
        """
        Create a quantum superposition of possible meanings
        創建可能含義的量子疊加態
        """
        # Extract possible semantic interpretations
        interpretations = self._extract_interpretations(text)
        
        # Create equal superposition
        num_interpretations = len(interpretations)
        if num_interpretations == 0:
            return QuantumState({text: 1.0 + 0j})
        
        # Equal amplitude for each interpretation
        amplitude = 1.0 / math.sqrt(num_interpretations)
        amplitudes = {interp: amplitude + 0j for interp in interpretations}
        
        return QuantumState(amplitudes)
    
    def _extract_interpretations(self, text: str) -> List[str]:
        """Extract possible semantic interpretations from text"""
        # Simplified interpretation extraction
        # In practice, this would use advanced NLP techniques
        
        interpretations = [text]  # Base interpretation
        
        # Check for ambiguous words
        ambiguous_words = {
            'bank': ['financial_institution', 'river_shore'],
            'bat': ['animal', 'sports_equipment'],
            'bark': ['dog_sound', 'tree_covering'],
            'bank': ['金融機構', '河岸'],  # Chinese examples
            '銀行': ['financial_institution', 'river_shore'],
            'fair': ['just', 'carnival', 'light_colored'],
            'light': ['illumination', 'weight', 'color'],
            'right': ['correct', 'direction', 'privilege'],
            'spring': ['season', 'water_source', 'coil'],
            'scale': ['measurement', 'fish_covering', 'climb']
        }
        
        for word, meanings in ambiguous_words.items():
            if word.lower() in text.lower():
                # Create interpretations for each meaning
                for meaning in meanings:
                    modified_text = text.lower().replace(word.lower(), f"{word}({meaning})")
                    interpretations.append(modified_text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_interpretations = []
        for interp in interpretations:
            if interp not in seen:
                seen.add(interp)
                unique_interpretations.append(interp)
        
        return unique_interpretations
    
    def apply_context_operator(self, quantum_state: QuantumState, context: Dict[str, Any]) -> QuantumState:
        """
        Apply context as a quantum operator to modify state amplitudes
        應用語境作為量子算符來修改狀態振幅
        """
        # Context influences probability amplitudes
        new_amplitudes = {}
        
        for state, amplitude in quantum_state.amplitudes.items():
            # Calculate context influence
            context_factor = self._calculate_context_factor(state, context)
            
            # Modify amplitude based on context
            new_amplitude = amplitude * context_factor
            new_amplitudes[state] = new_amplitude
        
        return QuantumState(new_amplitudes)
    
    def _calculate_context_factor(self, state: str, context: Dict[str, Any]) -> complex:
        """Calculate how context influences a particular state"""
        factor = 1.0 + 0j
        
        # Domain-specific adjustments
        if 'domain' in context:
            domain = context['domain'].lower()
            
            if domain == 'finance' and 'financial' in state.lower():
                factor *= 1.5  # Boost financial interpretations
            elif domain == 'nature' and ('river' in state.lower() or 'tree' in state.lower()):
                factor *= 1.5  # Boost nature interpretations
            elif domain == 'sports' and 'sports' in state.lower():
                factor *= 1.5  # Boost sports interpretations
        
        # Formality adjustments
        if 'formality' in context:
            formality = context['formality']
            if formality == 'formal' and '(' not in state:
                factor *= 1.2  # Prefer less explicit interpretations
            elif formality == 'technical' and '(' in state:
                factor *= 1.3  # Prefer explicit technical interpretations
        
        # Cultural context
        if 'culture' in context:
            culture = context['culture']
            if culture == 'chinese' and ('中' in state or '銀行' in state):
                factor *= 1.4  # Boost Chinese interpretations
        
        # Add phase rotation based on sentiment
        if 'sentiment' in context:
            sentiment_phase = self._sentiment_to_phase(context['sentiment'])
            factor *= cmath.exp(1j * sentiment_phase)
        
        return factor
    
    def _sentiment_to_phase(self, sentiment: str) -> float:
        """Convert sentiment to quantum phase"""
        sentiment_phases = {
            'positive': 0,
            'neutral': math.pi / 4,
            'negative': math.pi / 2,
            'very_positive': -math.pi / 4,
            'very_negative': 3 * math.pi / 4
        }
        return sentiment_phases.get(sentiment.lower(), 0)
    
    def semantic_entanglement(self, state1: QuantumState, state2: QuantumState) -> Tuple[QuantumState, QuantumState]:
        """
        Create semantic entanglement between two quantum states
        在兩個量子態之間創建語義糾纏
        """
        # Simple entanglement implementation
        # In practice, this would involve more sophisticated quantum operations
        
        entangled_amplitudes1 = {}
        entangled_amplitudes2 = {}
        
        # Create entangled pairs
        for state1_key, amp1 in state1.amplitudes.items():
            for state2_key, amp2 in state2.amplitudes.items():
                # Create entangled pair
                entangled_key1 = f"{state1_key}|{state2_key}"
                entangled_key2 = f"{state2_key}|{state1_key}"
                
                # Bell state-like entanglement
                entangled_amp = (amp1 * amp2) / math.sqrt(2)
                
                entangled_amplitudes1[entangled_key1] = entangled_amp
                entangled_amplitudes2[entangled_key2] = entangled_amp
        
        return QuantumState(entangled_amplitudes1), QuantumState(entangled_amplitudes2)
    
    def uncertainty_measurement(self, quantum_state: QuantumState) -> Dict[str, float]:
        """
        Calculate linguistic uncertainty analogous to Heisenberg uncertainty
        計算類似海森堡不確定性的語言學不確定性
        """
        # Calculate semantic uncertainty
        semantic_entropy = 0
        for state, amplitude in quantum_state.amplitudes.items():
            prob = abs(amplitude)**2
            if prob > 0:
                semantic_entropy -= prob * math.log2(prob)
        
        # Calculate contextual uncertainty (based on state distribution)
        num_states = len(quantum_state.amplitudes)
        max_entropy = math.log2(num_states) if num_states > 1 else 0
        
        # Uncertainty metrics
        uncertainty_metrics = {
            'semantic_uncertainty': semantic_entropy,
            'max_possible_uncertainty': max_entropy,
            'normalized_uncertainty': semantic_entropy / max_entropy if max_entropy > 0 else 0,
            'coherence': 1 - (semantic_entropy / max_entropy) if max_entropy > 0 else 1,
            'superposition_degree': num_states
        }
        
        return uncertainty_metrics


class QuantumTheorySpace:
    """
    Implementation of the 8,192 theories of everything in superposition
    8,192種萬有理論疊加態的實現
    """
    
    def __init__(self):
        self.dimensions = 13
        self.total_theories = 2 ** self.dimensions  # 8,192
        self.dimension_names = [
            'spacetime_nature',      # 時空本質 (continuous/discrete)
            'causality',             # 因果關係 (deterministic/probabilistic)
            'symmetry',              # 對稱性 (local/global)
            'unification',           # 統一性 (reductionist/emergent)
            'consciousness_role',    # 意識角色 (passive/active)
            'information_nature',    # 資訊本質 (objective/subjective)
            'mathematical_foundation', # 數學基礎 (discrete/continuous)
            'observation_effect',    # 觀測效應 (linear/nonlinear)
            'multiplicity',          # 多重性 (single/multiple_universes)
            'hierarchy',             # 層次結構 (flat/hierarchical)
            'evolution_direction',   # 演化方向 (reversible/irreversible)
            'complexity',            # 複雜性 (simple/complex)
            'linguistic_description' # 語言描述 (describable/indescribable)
        ]
        
    def generate_theory(self, theory_index: int) -> Dict[str, str]:
        """Generate a specific theory based on binary index"""
        if not 0 <= theory_index < self.total_theories:
            raise ValueError(f"Theory index must be between 0 and {self.total_theories-1}")
        
        binary_rep = format(theory_index, f'0{self.dimensions}b')
        theory = {}
        
        for i, dimension in enumerate(self.dimension_names):
            value = 'A' if binary_rep[i] == '0' else 'B'
            theory[dimension] = value
        
        return theory
    
    def interpret_theory(self, theory: Dict[str, str]) -> str:
        """Provide human-readable interpretation of a theory"""
        interpretations = {
            'spacetime_nature': {'A': 'continuous spacetime', 'B': 'discrete spacetime'},
            'causality': {'A': 'deterministic', 'B': 'probabilistic'},
            'symmetry': {'A': 'local symmetries', 'B': 'global symmetries'},
            'unification': {'A': 'reductionist', 'B': 'emergent'},
            'consciousness_role': {'A': 'passive observer', 'B': 'active participant'},
            'information_nature': {'A': 'objective information', 'B': 'subjective information'},
            'mathematical_foundation': {'A': 'discrete mathematics', 'B': 'continuous mathematics'},
            'observation_effect': {'A': 'linear effects', 'B': 'nonlinear effects'},
            'multiplicity': {'A': 'single universe', 'B': 'multiple universes'},
            'hierarchy': {'A': 'flat structure', 'B': 'hierarchical structure'},
            'evolution_direction': {'A': 'reversible processes', 'B': 'irreversible processes'},
            'complexity': {'A': 'fundamental simplicity', 'B': 'fundamental complexity'},
            'linguistic_description': {'A': 'fully describable', 'B': 'partially indescribable'}
        }
        
        description_parts = []
        for dimension, value in theory.items():
            if dimension in interpretations:
                description_parts.append(interpretations[dimension][value])
        
        return "; ".join(description_parts)
    
    def create_theory_superposition(self, weights: Optional[List[float]] = None) -> QuantumState:
        """Create quantum superposition of all possible theories"""
        if weights is None:
            # Equal superposition
            weights = [1.0] * self.total_theories
        
        if len(weights) != self.total_theories:
            raise ValueError(f"Weights must have exactly {self.total_theories} elements")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Create amplitudes
        amplitudes = {}
        for i, weight in enumerate(normalized_weights):
            theory = self.generate_theory(i)
            theory_key = f"Theory_{i:04d}"
            amplitudes[theory_key] = math.sqrt(weight) + 0j
        
        return QuantumState(amplitudes)
    
    def collapse_theory_superposition(self, superposition: QuantumState, observation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collapse theory superposition based on observation context"""
        # Apply context bias to different theories
        biased_amplitudes = {}
        
        for theory_key, amplitude in superposition.amplitudes.items():
            theory_index = int(theory_key.split('_')[1])
            theory = self.generate_theory(theory_index)
            
            # Calculate bias based on observation context
            bias_factor = self._calculate_theory_bias(theory, observation_context)
            biased_amplitudes[theory_key] = amplitude * bias_factor
        
        # Create new quantum state with biased amplitudes
        biased_state = QuantumState(biased_amplitudes)
        
        # Perform measurement
        collapsed_theory_key = biased_state.measure()
        collapsed_theory_index = int(collapsed_theory_key.split('_')[1])
        collapsed_theory = self.generate_theory(collapsed_theory_index)
        
        return {
            'theory_index': collapsed_theory_index,
            'theory': collapsed_theory,
            'interpretation': self.interpret_theory(collapsed_theory),
            'probability': biased_state.probability(collapsed_theory_key)
        }
    
    def _calculate_theory_bias(self, theory: Dict[str, str], context: Dict[str, Any]) -> float:
        """Calculate bias factor for a theory based on observation context"""
        bias = 1.0
        
        # Experimental context biases
        if 'experiment_type' in context:
            exp_type = context['experiment_type']
            
            if exp_type == 'quantum_mechanics':
                if theory['causality'] == 'B':  # probabilistic
                    bias *= 1.5
                if theory['observation_effect'] == 'B':  # nonlinear
                    bias *= 1.3
            
            elif exp_type == 'relativity':
                if theory['spacetime_nature'] == 'A':  # continuous
                    bias *= 1.4
                if theory['symmetry'] == 'B':  # global
                    bias *= 1.2
            
            elif exp_type == 'consciousness_studies':
                if theory['consciousness_role'] == 'B':  # active
                    bias *= 1.6
                if theory['information_nature'] == 'B':  # subjective
                    bias *= 1.3
        
        # Philosophical preference biases
        if 'philosophy' in context:
            phil = context['philosophy']
            
            if phil == 'reductionist':
                if theory['unification'] == 'A':  # reductionist
                    bias *= 1.4
                if theory['complexity'] == 'A':  # simple
                    bias *= 1.2
            
            elif phil == 'emergentist':
                if theory['unification'] == 'B':  # emergent
                    bias *= 1.4
                if theory['hierarchy'] == 'B':  # hierarchical
                    bias *= 1.3
        
        return bias


def main():
    """Demonstration of quantum-inspired NLP processing"""
    print("=== Quantum-Inspired Natural Language Processing Demo ===")
    print("量子啟發的自然語言處理演示\n")
    
    # Initialize processors
    semantic_processor = QuantumSemanticProcessor()
    theory_space = QuantumTheorySpace()
    
    # Example 1: Semantic superposition
    print("1. Semantic Superposition | 語義疊加態")
    print("-" * 50)
    
    test_sentences = [
        "I went to the bank",
        "The bat flew away",
        "這個詞很有意思",
        "Time flies like an arrow"
    ]
    
    for sentence in test_sentences:
        print(f"Input: {sentence}")
        quantum_state = semantic_processor.create_semantic_superposition(sentence)
        
        print("Possible interpretations | 可能的詮釋:")
        for state, amplitude in quantum_state.amplitudes.items():
            probability = abs(amplitude)**2
            print(f"  {state}: {probability:.3f}")
        
        # Apply different contexts
        contexts = [
            {'domain': 'finance', 'formality': 'formal'},
            {'domain': 'nature', 'sentiment': 'positive'},
            {'domain': 'sports', 'culture': 'english'}
        ]
        
        for context in contexts:
            context_state = semantic_processor.apply_context_operator(quantum_state, context)
            measured_meaning = context_state.measure()
            print(f"  Context {context} → {measured_meaning}")
        
        print()
    
    # Example 2: Uncertainty measurement
    print("2. Linguistic Uncertainty Analysis | 語言學不確定性分析")
    print("-" * 50)
    
    ambiguous_sentence = "The chicken is ready to eat"
    quantum_state = semantic_processor.create_semantic_superposition(ambiguous_sentence)
    uncertainty = semantic_processor.uncertainty_measurement(quantum_state)
    
    print(f"Sentence: {ambiguous_sentence}")
    print("Uncertainty metrics | 不確定性指標:")
    for metric, value in uncertainty.items():
        print(f"  {metric}: {value:.3f}")
    print()
    
    # Example 3: Theory space demonstration
    print("3. Theory of Everything Space | 萬有理論空間")
    print("-" * 50)
    
    # Create theory superposition
    theory_superposition = theory_space.create_theory_superposition()
    print(f"Created superposition of {theory_space.total_theories} theories")
    
    # Demonstrate theory collapse under different observation contexts
    observation_contexts = [
        {'experiment_type': 'quantum_mechanics', 'philosophy': 'copenhagen'},
        {'experiment_type': 'relativity', 'philosophy': 'reductionist'},
        {'experiment_type': 'consciousness_studies', 'philosophy': 'emergentist'}
    ]
    
    for context in observation_contexts:
        result = theory_space.collapse_theory_superposition(theory_superposition, context)
        print(f"\nContext: {context}")
        print(f"Collapsed to Theory #{result['theory_index']:04d}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Interpretation: {result['interpretation'][:100]}...")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()