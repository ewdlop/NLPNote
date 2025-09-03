#!/usr/bin/env python3
"""
Comprehensive Quantum Superposition Demonstration
量子疊加態綜合演示

This script demonstrates all aspects of quantum theories superposition 
as applied to natural language processing and human expression evaluation.

本腳本演示量子理論疊加態應用於自然語言處理和人類表達評估的所有方面。
"""

import sys
import os
import time
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from QuantumNLP import QuantumSemanticProcessor, QuantumTheorySpace, QuantumState
    from QuantumEnhancedExpressionEvaluator import QuantumEnhancedExpressionEvaluator, QuantumExpressionContext
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import quantum modules: {e}")
    QUANTUM_AVAILABLE = False


def print_header(title: str, subtitle: str = ""):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 80)


def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n{title}")
    print("-" * len(title))


def demonstrate_quantum_superposition():
    """Demonstrate quantum superposition concepts in linguistics"""
    print_header("量子疊加態演示", "Quantum Superposition Demonstration")
    
    if not QUANTUM_AVAILABLE:
        print("Quantum modules not available. Please ensure QuantumNLP.py is accessible.")
        return
    
    processor = QuantumSemanticProcessor()
    
    # Example 1: Basic superposition
    print_subsection("1. 基本語義疊加態 | Basic Semantic Superposition")
    
    ambiguous_expressions = [
        "I saw a bat",
        "The bank is closed",
        "Time flies",
        "Fair trade",
        "Light music"
    ]
    
    for expr in ambiguous_expressions:
        print(f"\nExpression: '{expr}'")
        quantum_state = processor.create_semantic_superposition(expr)
        
        print("Superposition states:")
        for state, amplitude in quantum_state.amplitudes.items():
            probability = abs(amplitude)**2
            print(f"  |{state}⟩: {probability:.3f}")
        
        # Demonstrate measurement collapse
        measured_state = quantum_state.measure()
        print(f"Measurement result: '{measured_state}'")
    
    # Example 2: Context-dependent collapse
    print_subsection("2. 語境依賴的態塌縮 | Context-Dependent State Collapse")
    
    expression = "I went to the bank"
    base_state = processor.create_semantic_superposition(expression)
    
    contexts = [
        {'domain': 'finance', 'description': 'Financial context'},
        {'domain': 'nature', 'description': 'Natural environment context'},
        {'domain': 'general', 'description': 'General context'}
    ]
    
    print(f"\nExpression: '{expression}'")
    print("Original superposition:")
    for state, amplitude in base_state.amplitudes.items():
        print(f"  |{state}⟩: {abs(amplitude)**2:.3f}")
    
    for context in contexts:
        contextualized_state = processor.apply_context_operator(base_state, context)
        measured_meaning = contextualized_state.measure()
        print(f"\n{context['description']}:")
        print(f"  Most likely interpretation: '{measured_meaning}'")
        
        # Show probability distribution after context application
        print("  Probability distribution:")
        for state, amplitude in contextualized_state.amplitudes.items():
            prob = abs(amplitude)**2
            if prob > 0.01:  # Only show significant probabilities
                print(f"    {state}: {prob:.3f}")


def demonstrate_theory_space():
    """Demonstrate the 8,192 theories of everything space"""
    print_header("8,192 萬有理論空間", "8,192 Theories of Everything Space")
    
    if not QUANTUM_AVAILABLE:
        print("Quantum modules not available.")
        return
    
    theory_space = QuantumTheorySpace()
    
    print_subsection("1. 理論空間結構 | Theory Space Structure")
    print(f"Total possible theories: {theory_space.total_theories:,}")
    print(f"Dimensions: {theory_space.dimensions}")
    print("\nTheory dimensions:")
    for i, dim in enumerate(theory_space.dimension_names):
        print(f"  {i+1:2d}. {dim}")
    
    print_subsection("2. 隨機理論樣本 | Random Theory Samples")
    
    # Show some random theories
    import random
    sample_indices = random.sample(range(theory_space.total_theories), 5)
    
    for i, theory_index in enumerate(sample_indices, 1):
        theory = theory_space.generate_theory(theory_index)
        interpretation = theory_space.interpret_theory(theory)
        
        print(f"\nTheory #{theory_index:04d}:")
        print(f"  Binary: {format(theory_index, f'0{theory_space.dimensions}b')}")
        print(f"  Interpretation: {interpretation[:100]}...")
    
    print_subsection("3. 理論疊加態創建 | Theory Superposition Creation")
    
    # Create theory superposition
    superposition = theory_space.create_theory_superposition()
    print(f"Created superposition of all {theory_space.total_theories:,} theories")
    print(f"Each theory has equal amplitude: {1/theory_space.total_theories:.6f}")
    
    # Demonstrate theory collapse under different contexts
    observation_contexts = [
        {
            'experiment_type': 'quantum_mechanics',
            'philosophy': 'copenhagen',
            'description': 'Quantum mechanics experiment with Copenhagen interpretation'
        },
        {
            'experiment_type': 'relativity',
            'philosophy': 'reductionist',
            'description': 'Relativity experiment with reductionist philosophy'
        },
        {
            'experiment_type': 'consciousness_studies',
            'philosophy': 'emergentist',
            'description': 'Consciousness studies with emergentist philosophy'
        }
    ]
    
    print_subsection("4. 觀測驅動的理論塌縮 | Observation-Driven Theory Collapse")
    
    for context in observation_contexts:
        result = theory_space.collapse_theory_superposition(superposition, context)
        
        print(f"\nObservation context: {context['description']}")
        print(f"Collapsed to Theory #{result['theory_index']:04d}")
        print(f"Collapse probability: {result['probability']:.4f}")
        print(f"Theory interpretation: {result['interpretation'][:120]}...")


def demonstrate_quantum_uncertainty():
    """Demonstrate quantum uncertainty principles in linguistics"""
    print_header("量子不確定性原理", "Quantum Uncertainty Principles in Linguistics")
    
    if not QUANTUM_AVAILABLE:
        print("Quantum modules not available.")
        return
    
    processor = QuantumSemanticProcessor()
    
    print_subsection("1. 語義-語境不確定性關係 | Meaning-Context Uncertainty Relation")
    
    # Test expressions with different levels of ambiguity
    test_expressions = [
        ("Hello", "Low ambiguity"),
        ("Bank", "Medium ambiguity"),
        ("Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo", "High ambiguity"),
        ("Colorless green ideas sleep furiously", "Syntactic ambiguity"),
        ("The chicken is ready to eat", "Pragmatic ambiguity")
    ]
    
    for expression, description in test_expressions:
        print(f"\nExpression: '{expression}' ({description})")
        
        quantum_state = processor.create_semantic_superposition(expression)
        uncertainty_metrics = processor.uncertainty_measurement(quantum_state)
        
        print("Uncertainty metrics:")
        for metric, value in uncertainty_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Calculate uncertainty product (analogous to ΔpΔx ≥ ℏ/2)
        semantic_uncertainty = uncertainty_metrics['semantic_uncertainty']
        context_variance = uncertainty_metrics['normalized_uncertainty']
        uncertainty_product = semantic_uncertainty * context_variance
        
        print(f"  Uncertainty product: {uncertainty_product:.3f}")
        
        # Quantum linguistic uncertainty principle threshold
        linguistic_planck_constant = 0.1  # Hypothetical
        threshold = linguistic_planck_constant / 2
        
        if uncertainty_product >= threshold:
            print(f"  ✓ Satisfies uncertainty principle (≥ {threshold:.3f})")
        else:
            print(f"  ✗ Below uncertainty threshold (< {threshold:.3f})")


def demonstrate_semantic_entanglement():
    """Demonstrate semantic entanglement phenomena"""
    print_header("語義糾纏現象", "Semantic Entanglement Phenomena")
    
    if not QUANTUM_AVAILABLE:
        print("Quantum modules not available.")
        return
    
    processor = QuantumSemanticProcessor()
    
    print_subsection("1. 語義糾纏創建 | Semantic Entanglement Creation")
    
    # Pairs of semantically related expressions
    entangled_pairs = [
        ("Love conquers all", "All you need is love"),
        ("Time is money", "Money talks"),
        ("知識就是力量", "力量源於知識"),  # Chinese: Knowledge is power, Power comes from knowledge
        ("Actions speak louder", "Words are empty")
    ]
    
    for expr1, expr2 in entangled_pairs:
        print(f"\nExpression pair: '{expr1}' ↔ '{expr2}'")
        
        state1 = processor.create_semantic_superposition(expr1)
        state2 = processor.create_semantic_superposition(expr2)
        
        # Create entanglement
        entangled1, entangled2 = processor.semantic_entanglement(state1, state2)
        
        print(f"Original state 1 superposition degree: {len(state1.amplitudes)}")
        print(f"Original state 2 superposition degree: {len(state2.amplitudes)}")
        print(f"Entangled state 1 superposition degree: {len(entangled1.amplitudes)}")
        print(f"Entangled state 2 superposition degree: {len(entangled2.amplitudes)}")
        
        # Demonstrate correlation by measuring one state
        measured1 = entangled1.measure()
        measured2 = entangled2.measure()
        
        print(f"Measurement 1: '{measured1[:50]}...'")
        print(f"Measurement 2: '{measured2[:50]}...'")
        
        # Check for correlation patterns
        if '|' in measured1 and '|' in measured2:
            parts1 = measured1.split('|')
            parts2 = measured2.split('|')
            if len(parts1) == 2 and len(parts2) == 2:
                correlation = parts1[1] == parts2[0]  # Simple correlation check
                print(f"Entanglement correlation detected: {correlation}")


def demonstrate_integrated_evaluation():
    """Demonstrate integrated quantum-classical evaluation"""
    print_header("量子-經典整合評估", "Quantum-Classical Integrated Evaluation")
    
    if not QUANTUM_AVAILABLE:
        print("Quantum modules not available.")
        return
    
    evaluator = QuantumEnhancedExpressionEvaluator()
    
    print_subsection("1. 多重表達評估 | Multiple Expression Evaluation")
    
    test_cases = [
        {
            'expression': "To be or not to be, that is the question",
            'context': QuantumExpressionContext(
                situation='literary',
                cultural_background='western',
                formality_level='formal',
                measurement_basis='semantic'
            ),
            'description': 'Classical literary expression'
        },
        {
            'expression': "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
            'context': QuantumExpressionContext(
                situation='linguistic_puzzle',
                measurement_basis='syntactic',
                superposition_enabled=True,
                coherence_time=0.5
            ),
            'description': 'Grammatically complex but semantically puzzling'
        },
        {
            'expression': "I 愛 you with all my 心",  # Mixed language
            'context': QuantumExpressionContext(
                cultural_background='multicultural',
                formality_level='intimate',
                measurement_basis='pragmatic',
                entanglement_pairs=[('愛', 'love'), ('心', 'heart')]
            ),
            'description': 'Multilingual expression with entanglement'
        },
        {
            'expression': "The silence was deafeningly quiet",
            'context': QuantumExpressionContext(
                situation='poetic',
                measurement_basis='semantic',
                emotional_state='contemplative',
                observer_effect_strength=0.8
            ),
            'description': 'Paradoxical poetic expression'
        }
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Expression: '{test_case['expression']}'")
        
        result = evaluator.quantum_expression_evaluation(
            test_case['expression'],
            test_case['context']
        )
        
        all_results.append(result)
        
        if result['integrated_results']:
            integrated = result['integrated_results']
            print(f"   Integrated Score: {integrated['integrated_score']:.3f}")
            print(f"   Classical: {integrated['classical_component']:.3f} | "
                  f"Quantum: {integrated['quantum_component']:.3f}")
            print(f"   Enhancement: {integrated['quantum_enhancement']:+.3f}")
            print(f"   Mode: {integrated['interpretation_mode']}")
    
    print_subsection("2. 現象分析 | Phenomena Analysis")
    
    expressions = [case['expression'] for case in test_cases]
    analysis = evaluator.analyze_quantum_linguistic_phenomena(expressions)
    
    if 'quantum_phenomena' in analysis:
        phenomena = analysis['quantum_phenomena']
        print(f"Superposition instances: {phenomena.get('superposition_instances', 0)}")
        print(f"Entanglement pairs detected: {phenomena.get('entanglement_pairs', 0)}")
        
        if 'coherence_patterns' in phenomena and phenomena['coherence_patterns']:
            coherence_values = phenomena['coherence_patterns']
            avg_coherence = sum(coherence_values) / len(coherence_values)
            print(f"Average coherence: {avg_coherence:.3f}")
            print(f"Coherence range: {min(coherence_values):.3f} - {max(coherence_values):.3f}")
    
    if 'statistical_analysis' in analysis:
        stats = analysis['statistical_analysis']
        print(f"\nStatistical Analysis:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.3f}")


def demonstrate_philosophical_implications():
    """Demonstrate philosophical implications of quantum linguistics"""
    print_header("哲學意涵", "Philosophical Implications")
    
    print_subsection("1. 觀測者效應 | Observer Effect in Language")
    
    philosophical_examples = [
        {
            'statement': "The tree falling in the forest makes a sound",
            'quantum_analogy': "Unobserved language has no definite meaning until interpreted",
            'interpretation': "Meaning exists in superposition until observed by a conscious interpreter"
        },
        {
            'statement': "Schrödinger's cat is both alive and dead",
            'quantum_analogy': "Ambiguous statements are both true and false",
            'interpretation': "Until contextual measurement, multiple truth values coexist"
        },
        {
            'statement': "Quantum entanglement connects distant particles",
            'quantum_analogy': "Semantic fields connect distant concepts",
            'interpretation': "Related words maintain correlation across semantic space"
        }
    ]
    
    for i, example in enumerate(philosophical_examples, 1):
        print(f"\n{i}. Quantum Statement: {example['statement']}")
        print(f"   Linguistic Analogy: {example['quantum_analogy']}")
        print(f"   Interpretation: {example['interpretation']}")
    
    print_subsection("2. 多重世界詮釋 | Many-Worlds Interpretation in Language")
    
    print("In linguistic many-worlds interpretation:")
    print("• Every ambiguous expression branches into parallel semantic universes")
    print("• Each interpretation exists in its own reality")
    print("• Context selection determines which universe we observe")
    print("• All possible meanings remain equally valid in their respective worlds")
    
    print_subsection("3. 語言學的不可分離性 | Linguistic Non-Separability")
    
    print("Quantum linguistic non-separability suggests:")
    print("• Words in a sentence cannot be understood independently")
    print("• Meaning emerges from the holistic quantum state")
    print("• Local realism fails for linguistic interpretation")
    print("• Contextual measurements affect the entire semantic system")


def main():
    """Main demonstration function"""
    print_header("量子理論疊加態：語言學與重力的統一框架", 
                "Superposition of Quantum Theories: A Unified Framework for Linguistics and Gravity")
    
    print("""
    本演示展示了量子理論、詮釋學與重力理論的疊加態概念
    如何應用於自然語言處理和人類表達評估。
    
    This demonstration shows how the superposition concepts of quantum theories,
    interpretations, and gravity can be applied to natural language processing
    and human expression evaluation.
    """)
    
    # Check if quantum modules are available
    if not QUANTUM_AVAILABLE:
        print("\n⚠️  Warning: Quantum modules are not fully available.")
        print("   Some demonstrations may be limited or unavailable.")
        print("   Please ensure QuantumNLP.py and related files are accessible.")
    
    # Run demonstrations in sequence
    try:
        demonstrate_quantum_superposition()
        time.sleep(1)
        
        demonstrate_theory_space()
        time.sleep(1)
        
        demonstrate_quantum_uncertainty()
        time.sleep(1)
        
        demonstrate_semantic_entanglement()
        time.sleep(1)
        
        demonstrate_integrated_evaluation()
        time.sleep(1)
        
        demonstrate_philosophical_implications()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    print_header("演示完成", "Demonstration Complete")
    print("""
    感謝您體驗量子語言學的奇妙世界！
    這個框架展示了物理學與語言學之間的深刻聯繫，
    為理解人類表達和人工智能開闢了新的道路。
    
    Thank you for experiencing the wonderful world of quantum linguistics!
    This framework demonstrates the profound connections between physics and linguistics,
    opening new paths for understanding human expression and artificial intelligence.
    """)


if __name__ == "__main__":
    main()