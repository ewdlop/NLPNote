#!/usr/bin/env python3
"""
Quantum-Enhanced Human Expression Evaluator
量子增強人類表達評估器

This module integrates quantum-inspired concepts with the existing human expression
evaluation framework, demonstrating how quantum superposition, entanglement, and
uncertainty principles can enhance understanding of linguistic phenomena.

本模組將量子啟發概念與現有的人類表達評估框架整合，
展示量子疊加、糾纏和不確定性原理如何增強對語言現象的理解。
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import cmath

try:
    from QuantumNLP import QuantumState, QuantumSemanticProcessor, QuantumTheorySpace
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: QuantumNLP module not available. Running in compatibility mode.")

try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    HUMAN_EVALUATOR_AVAILABLE = True
except ImportError:
    HUMAN_EVALUATOR_AVAILABLE = False
    print("Warning: HumanExpressionEvaluator not available. Creating mock implementation.")


@dataclass
class QuantumExpressionContext:
    """
    Enhanced expression context with quantum properties
    具有量子特性的增強表達語境
    """
    # Classical properties (mimicking ExpressionContext)
    speaker: str = "unknown"
    listener: str = "unknown"
    situation: str = "general"
    cultural_background: str = "universal"
    power_relation: str = "equal"
    formality_level: str = "neutral"
    emotional_state: str = "neutral"
    
    # Quantum-specific properties
    superposition_enabled: bool = True
    entanglement_pairs: Optional[List[Tuple[str, str]]] = None
    measurement_basis: str = "semantic"  # "semantic", "syntactic", "pragmatic"
    coherence_time: float = 1.0  # How long quantum effects persist
    decoherence_rate: float = 0.1
    observer_effect_strength: float = 0.5
    quantum_interpretation: str = "copenhagen"  # "copenhagen", "many_worlds", "pilot_wave"
    
    def __post_init__(self):
        if self.entanglement_pairs is None:
            self.entanglement_pairs = []


class QuantumEnhancedExpressionEvaluator:
    """
    Expression evaluator enhanced with quantum-inspired processing
    量子啟發處理增強的表達評估器
    """
    
    def __init__(self):
        # Initialize quantum components
        if QUANTUM_AVAILABLE:
            self.quantum_processor = QuantumSemanticProcessor()
            self.theory_space = QuantumTheorySpace()
        else:
            self.quantum_processor = None
            self.theory_space = None
        
        # Initialize classical components if available
        if HUMAN_EVALUATOR_AVAILABLE:
            self.classical_evaluator = HumanExpressionEvaluator()
        else:
            self.classical_evaluator = None
        
        # Quantum-enhanced evaluation parameters
        self.quantum_weight = 0.3  # How much quantum effects influence evaluation
        self.classical_weight = 0.7  # How much classical effects influence evaluation
        
        # Quantum measurement statistics
        self.measurement_history = []
        self.entanglement_registry = {}
    
    def quantum_expression_evaluation(
        self, 
        expression: str, 
        context: QuantumExpressionContext
    ) -> Dict[str, Any]:
        """
        Perform quantum-enhanced expression evaluation
        執行量子增強的表達評估
        """
        results = {
            'expression': expression,
            'context': context,
            'quantum_enabled': QUANTUM_AVAILABLE and context.superposition_enabled,
            'classical_results': None,
            'quantum_results': None,
            'integrated_results': None,
            'measurement_statistics': {}
        }
        
        # Classical evaluation (if available)
        if self.classical_evaluator:
            try:
                classical_context = self._extract_classical_context(context)
                classical_results = self.classical_evaluator.comprehensive_evaluation(
                    expression, classical_context
                )
                results['classical_results'] = classical_results
            except Exception as e:
                results['classical_results'] = {'error': str(e)}
        else:
            results['classical_results'] = self._mock_classical_evaluation(expression, context)
        
        # Quantum evaluation (if available and enabled)
        if QUANTUM_AVAILABLE and context.superposition_enabled:
            quantum_results = self._quantum_evaluation(expression, context)
            results['quantum_results'] = quantum_results
            
            # Integrate quantum and classical results
            integrated_results = self._integrate_quantum_classical(
                results['classical_results'], 
                quantum_results, 
                context
            )
            results['integrated_results'] = integrated_results
        else:
            results['quantum_results'] = {'status': 'disabled', 'reason': 'quantum processing not available or disabled'}
            results['integrated_results'] = results['classical_results']
        
        # Generate measurement statistics
        results['measurement_statistics'] = self._calculate_measurement_statistics(results)
        
        # Store measurement for history
        self.measurement_history.append({
            'expression': expression,
            'context': context,
            'results': results,
            'timestamp': self._get_timestamp()
        })
        
        return results
    
    def _quantum_evaluation(self, expression: str, context: QuantumExpressionContext) -> Dict[str, Any]:
        """Perform quantum-specific evaluation"""
        if not QUANTUM_AVAILABLE:
            return {'error': 'Quantum processing not available'}
        
        # Create semantic superposition
        quantum_state = self.quantum_processor.create_semantic_superposition(expression)
        
        # Apply context as quantum operator
        context_dict = {
            'domain': getattr(context, 'situation', 'general'),
            'formality': getattr(context, 'formality_level', 'neutral'),
            'culture': getattr(context, 'cultural_background', 'universal'),
            'sentiment': getattr(context, 'emotional_state', 'neutral')
        }
        
        contextualized_state = self.quantum_processor.apply_context_operator(
            quantum_state, context_dict
        )
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self.quantum_processor.uncertainty_measurement(contextualized_state)
        
        # Perform measurements in different bases
        measurements = {}
        for basis in ['semantic', 'syntactic', 'pragmatic']:
            if basis == context.measurement_basis:
                measurements[basis] = {
                    'measured_state': contextualized_state.measure(),
                    'probability': max(contextualized_state.amplitudes.values(), 
                                     key=lambda x: abs(x)**2),
                    'confidence': 1 - uncertainty_metrics['normalized_uncertainty']
                }
        
        # Apply decoherence effects
        decoherence_factor = math.exp(-context.decoherence_rate * context.coherence_time)
        
        # Calculate quantum scores
        quantum_scores = {
            'superposition_score': uncertainty_metrics['normalized_uncertainty'],
            'coherence_score': uncertainty_metrics['coherence'] * decoherence_factor,
            'entanglement_score': self._calculate_entanglement_score(expression, context),
            'observer_effect_score': context.observer_effect_strength,
            'measurement_confidence': measurements.get(context.measurement_basis, {}).get('confidence', 0)
        }
        
        return {
            'quantum_state': {
                'amplitudes': {k: abs(v)**2 for k, v in contextualized_state.amplitudes.items()},
                'phase_information': {k: cmath.phase(v) for k, v in contextualized_state.amplitudes.items()}
            },
            'uncertainty_metrics': uncertainty_metrics,
            'measurements': measurements,
            'quantum_scores': quantum_scores,
            'decoherence_factor': decoherence_factor,
            'interpretation': context.quantum_interpretation
        }
    
    def _calculate_entanglement_score(self, expression: str, context: QuantumExpressionContext) -> float:
        """Calculate entanglement score based on contextual relationships"""
        if not context.entanglement_pairs:
            return 0.0
        
        entanglement_score = 0.0
        expression_lower = expression.lower()
        
        for pair in context.entanglement_pairs:
            word1, word2 = pair
            if word1.lower() in expression_lower and word2.lower() in expression_lower:
                # Calculate semantic distance (simplified)
                distance = abs(hash(word1) - hash(word2)) % 1000 / 1000.0
                entanglement_strength = 1.0 / (1.0 + distance)
                entanglement_score += entanglement_strength
        
        return min(entanglement_score, 1.0)
    
    def _integrate_quantum_classical(
        self, 
        classical_results: Dict[str, Any], 
        quantum_results: Dict[str, Any], 
        context: QuantumExpressionContext
    ) -> Dict[str, Any]:
        """Integrate quantum and classical evaluation results"""
        
        # Extract classical scores
        classical_score = 0.5  # Default fallback
        if isinstance(classical_results, dict):
            if 'integrated' in classical_results and 'overall_score' in classical_results['integrated']:
                classical_score = classical_results['integrated']['overall_score']
            elif 'overall_score' in classical_results:
                classical_score = classical_results['overall_score']
            elif 'integrated_score' in classical_results:
                classical_score = classical_results['integrated_score']
        
        # Extract quantum scores
        quantum_score = 0.5  # Default fallback
        if 'quantum_scores' in quantum_results:
            quantum_metrics = quantum_results['quantum_scores']
            # Weighted combination of quantum metrics
            quantum_score = (
                quantum_metrics.get('superposition_score', 0) * 0.2 +
                quantum_metrics.get('coherence_score', 0) * 0.3 +
                quantum_metrics.get('entanglement_score', 0) * 0.2 +
                quantum_metrics.get('measurement_confidence', 0) * 0.3
            )
        
        # Quantum-classical integration
        integrated_score = (
            self.classical_weight * classical_score +
            self.quantum_weight * quantum_score
        )
        
        # Calculate quantum enhancement factor
        quantum_enhancement = quantum_score - classical_score
        
        # Determine interpretation mode based on quantum effects
        interpretation_mode = self._determine_interpretation_mode(quantum_results, context)
        
        return {
            'integrated_score': integrated_score,
            'classical_component': classical_score,
            'quantum_component': quantum_score,
            'quantum_enhancement': quantum_enhancement,
            'integration_weights': {
                'classical': self.classical_weight,
                'quantum': self.quantum_weight
            },
            'interpretation_mode': interpretation_mode,
            'quantum_effects_summary': self._summarize_quantum_effects(quantum_results),
            'recommendation': self._generate_recommendation(integrated_score, quantum_enhancement)
        }
    
    def _determine_interpretation_mode(
        self, 
        quantum_results: Dict[str, Any], 
        context: QuantumExpressionContext
    ) -> str:
        """Determine the primary interpretation mode based on quantum effects"""
        
        if 'quantum_scores' not in quantum_results:
            return 'classical'
        
        scores = quantum_results['quantum_scores']
        superposition = scores.get('superposition_score', 0)
        coherence = scores.get('coherence_score', 0)
        entanglement = scores.get('entanglement_score', 0)
        
        if superposition > 0.7:
            return 'superposition_dominant'
        elif entanglement > 0.5:
            return 'entanglement_significant'
        elif coherence > 0.8:
            return 'coherent_quantum'
        elif superposition > 0.3:
            return 'mixed_quantum_classical'
        else:
            return 'mostly_classical'
    
    def _summarize_quantum_effects(self, quantum_results: Dict[str, Any]) -> Dict[str, str]:
        """Provide human-readable summary of quantum effects"""
        if 'quantum_scores' not in quantum_results:
            return {'status': 'no_quantum_effects'}
        
        scores = quantum_results['quantum_scores']
        summary = {}
        
        # Superposition analysis
        superposition = scores.get('superposition_score', 0)
        if superposition > 0.7:
            summary['superposition'] = 'high_ambiguity_multiple_interpretations_coexist'
        elif superposition > 0.3:
            summary['superposition'] = 'moderate_ambiguity_some_interpretations_possible'
        else:
            summary['superposition'] = 'low_ambiguity_clear_interpretation'
        
        # Coherence analysis
        coherence = scores.get('coherence_score', 0)
        if coherence > 0.8:
            summary['coherence'] = 'highly_coherent_stable_meaning'
        elif coherence > 0.5:
            summary['coherence'] = 'moderately_coherent_some_stability'
        else:
            summary['coherence'] = 'low_coherence_meaning_fluctuates'
        
        # Entanglement analysis
        entanglement = scores.get('entanglement_score', 0)
        if entanglement > 0.5:
            summary['entanglement'] = 'strong_semantic_correlations_detected'
        elif entanglement > 0.2:
            summary['entanglement'] = 'weak_semantic_correlations_present'
        else:
            summary['entanglement'] = 'no_significant_semantic_correlations'
        
        return summary
    
    def _generate_recommendation(self, integrated_score: float, quantum_enhancement: float) -> str:
        """Generate actionable recommendation based on evaluation"""
        if integrated_score > 0.8:
            base_rec = "Excellent expression with high clarity and effectiveness."
        elif integrated_score > 0.6:
            base_rec = "Good expression with room for minor improvements."
        elif integrated_score > 0.4:
            base_rec = "Moderate expression quality, consider revisions."
        else:
            base_rec = "Expression needs significant improvement."
        
        if quantum_enhancement > 0.2:
            quantum_rec = " Quantum effects enhance meaning richness and interpretive flexibility."
        elif quantum_enhancement > 0:
            quantum_rec = " Quantum effects provide some additional interpretive depth."
        elif quantum_enhancement < -0.2:
            quantum_rec = " Quantum effects may introduce unnecessary ambiguity."
        else:
            quantum_rec = " Quantum effects are minimal."
        
        return base_rec + quantum_rec
    
    def analyze_quantum_linguistic_phenomena(self, expressions: List[str]) -> Dict[str, Any]:
        """
        Analyze quantum linguistic phenomena across multiple expressions
        分析多個表達中的量子語言現象
        """
        results = {
            'expressions_analyzed': len(expressions),
            'quantum_phenomena': {},
            'statistical_analysis': {},
            'theory_space_analysis': {}
        }
        
        if not QUANTUM_AVAILABLE:
            results['error'] = 'Quantum analysis not available'
            return results
        
        # Analyze each expression
        expression_results = []
        for expr in expressions:
            context = QuantumExpressionContext()
            result = self.quantum_expression_evaluation(expr, context)
            expression_results.append(result)
        
        # Detect quantum phenomena
        phenomena = self._detect_quantum_phenomena(expression_results)
        results['quantum_phenomena'] = phenomena
        
        # Statistical analysis
        stats = self._calculate_statistical_metrics(expression_results)
        results['statistical_analysis'] = stats
        
        # Theory space analysis
        if self.theory_space:
            theory_analysis = self._analyze_theory_space_implications(expression_results)
            results['theory_space_analysis'] = theory_analysis
        
        return results
    
    def _detect_quantum_phenomena(self, expression_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect quantum phenomena across expressions"""
        phenomena = {
            'superposition_instances': 0,
            'entanglement_pairs': 0,
            'coherence_patterns': [],
            'measurement_effects': [],
            'uncertainty_violations': 0
        }
        
        for result in expression_results:
            if 'quantum_results' in result and result['quantum_results']:
                quantum_data = result['quantum_results']
                
                # Check for superposition
                if 'quantum_scores' in quantum_data:
                    scores = quantum_data['quantum_scores']
                    if scores.get('superposition_score', 0) > 0.5:
                        phenomena['superposition_instances'] += 1
                    
                    if scores.get('entanglement_score', 0) > 0.3:
                        phenomena['entanglement_pairs'] += 1
                    
                    # Track coherence patterns
                    coherence = scores.get('coherence_score', 0)
                    phenomena['coherence_patterns'].append(coherence)
        
        return phenomena
    
    def _calculate_statistical_metrics(self, expression_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistical metrics across evaluations"""
        classical_scores = []
        quantum_scores = []
        integrated_scores = []
        
        for result in expression_results:
            if 'integrated_results' in result and result['integrated_results']:
                integrated = result['integrated_results']
                if 'classical_component' in integrated:
                    classical_scores.append(integrated['classical_component'])
                if 'quantum_component' in integrated:
                    quantum_scores.append(integrated['quantum_component'])
                if 'integrated_score' in integrated:
                    integrated_scores.append(integrated['integrated_score'])
        
        stats = {}
        
        if classical_scores:
            stats['classical_mean'] = sum(classical_scores) / len(classical_scores)
            stats['classical_std'] = self._calculate_std(classical_scores)
        
        if quantum_scores:
            stats['quantum_mean'] = sum(quantum_scores) / len(quantum_scores)
            stats['quantum_std'] = self._calculate_std(quantum_scores)
        
        if integrated_scores:
            stats['integrated_mean'] = sum(integrated_scores) / len(integrated_scores)
            stats['integrated_std'] = self._calculate_std(integrated_scores)
        
        if classical_scores and quantum_scores:
            stats['quantum_classical_correlation'] = self._calculate_correlation(
                classical_scores, quantum_scores
            )
        
        return stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean)**2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = sum((x[i] - mean_x)**2 for i in range(n))
        denom_y = sum((y[i] - mean_y)**2 for i in range(n))
        
        if denom_x == 0 or denom_y == 0:
            return 0.0
        
        return numerator / math.sqrt(denom_x * denom_y)
    
    def _analyze_theory_space_implications(self, expression_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze implications for theory space based on expression evaluations"""
        theory_implications = {
            'dominant_theories': [],
            'theory_preferences': {},
            'quantum_interpretations': {},
            'emergent_patterns': []
        }
        
        # This would be expanded with more sophisticated analysis
        # For now, provide a basic framework
        
        return theory_implications
    
    # Helper methods for compatibility
    def _extract_classical_context(self, quantum_context: QuantumExpressionContext):
        """Extract classical context from quantum context for compatibility"""
        if HUMAN_EVALUATOR_AVAILABLE:
            return ExpressionContext(
                speaker=getattr(quantum_context, 'speaker', 'unknown'),
                listener=getattr(quantum_context, 'listener', 'unknown'),
                situation=getattr(quantum_context, 'situation', 'general'),
                cultural_background=getattr(quantum_context, 'cultural_background', 'universal'),
                formality_level=getattr(quantum_context, 'formality_level', 'neutral'),
                emotional_state=getattr(quantum_context, 'emotional_state', 'neutral')
            )
        else:
            return {
                'speaker': getattr(quantum_context, 'speaker', 'unknown'),
                'situation': getattr(quantum_context, 'situation', 'general'),
                'formality_level': getattr(quantum_context, 'formality_level', 'neutral')
            }
    
    def _mock_classical_evaluation(self, expression: str, context: QuantumExpressionContext) -> Dict[str, Any]:
        """Mock classical evaluation when HumanExpressionEvaluator is not available"""
        # Simple mock implementation
        score = min(0.5 + len(expression) * 0.01, 0.9)
        return {
            'overall_score': score,
            'mock': True,
            'components': {
                'length_factor': len(expression) * 0.01,
                'base_score': 0.5
            }
        }
    
    def _calculate_measurement_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate measurement statistics for the evaluation"""
        stats = {
            'measurement_count': len(self.measurement_history) + 1,
            'quantum_enabled': results.get('quantum_enabled', False),
            'has_classical_results': results.get('classical_results') is not None,
            'has_quantum_results': results.get('quantum_results') is not None,
            'integration_successful': results.get('integrated_results') is not None
        }
        
        return stats
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()


def main():
    """Demonstration of quantum-enhanced expression evaluation"""
    print("=== Quantum-Enhanced Human Expression Evaluation Demo ===")
    print("量子增強人類表達評估演示\n")
    
    # Initialize evaluator
    evaluator = QuantumEnhancedExpressionEvaluator()
    
    # Test expressions with quantum properties
    test_cases = [
        {
            'expression': "I love you more than words can express",
            'context': QuantumExpressionContext(
                formality_level='intimate',
                emotional_state='very_positive',
                measurement_basis='semantic',
                entanglement_pairs=[('love', 'words'), ('express', 'feelings')]
            )
        },
        {
            'expression': "The meaning of life is 42",
            'context': QuantumExpressionContext(
                situation='philosophical',
                cultural_background='western',
                measurement_basis='pragmatic',
                quantum_interpretation='many_worlds'
            )
        },
        {
            'expression': "銀行在河邊",  # "The bank is by the river" - Chinese ambiguity
            'context': QuantumExpressionContext(
                cultural_background='chinese',
                measurement_basis='semantic',
                superposition_enabled=True,
                entanglement_pairs=[('銀行', '河邊')]
            )
        },
        {
            'expression': "Time flies like an arrow; fruit flies like a banana",
            'context': QuantumExpressionContext(
                situation='humorous',
                measurement_basis='syntactic',
                coherence_time=2.0,
                decoherence_rate=0.05
            )
        }
    ]
    
    # Evaluate each test case
    all_results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. Expression: {test_case['expression']}")
        print("-" * 60)
        
        result = evaluator.quantum_expression_evaluation(
            test_case['expression'], 
            test_case['context']
        )
        
        all_results.append(result)
        
        # Display results
        if result['integrated_results']:
            integrated = result['integrated_results']
            print(f"Integrated Score: {integrated['integrated_score']:.3f}")
            print(f"Classical Component: {integrated['classical_component']:.3f}")
            print(f"Quantum Component: {integrated['quantum_component']:.3f}")
            print(f"Quantum Enhancement: {integrated['quantum_enhancement']:.3f}")
            print(f"Interpretation Mode: {integrated['interpretation_mode']}")
            print(f"Recommendation: {integrated['recommendation']}")
            
            if 'quantum_effects_summary' in integrated:
                print("Quantum Effects:")
                for effect, description in integrated['quantum_effects_summary'].items():
                    print(f"  {effect}: {description}")
        
        print()
    
    # Analyze quantum phenomena across all expressions
    print("5. Quantum Phenomena Analysis")
    print("-" * 60)
    
    expressions = [case['expression'] for case in test_cases]
    phenomena_analysis = evaluator.analyze_quantum_linguistic_phenomena(expressions)
    
    if 'quantum_phenomena' in phenomena_analysis:
        phenomena = phenomena_analysis['quantum_phenomena']
        print(f"Superposition Instances: {phenomena.get('superposition_instances', 0)}")
        print(f"Entanglement Pairs: {phenomena.get('entanglement_pairs', 0)}")
        
        if phenomena.get('coherence_patterns'):
            avg_coherence = sum(phenomena['coherence_patterns']) / len(phenomena['coherence_patterns'])
            print(f"Average Coherence: {avg_coherence:.3f}")
    
    if 'statistical_analysis' in phenomena_analysis:
        stats = phenomena_analysis['statistical_analysis']
        print("\nStatistical Analysis:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.3f}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()