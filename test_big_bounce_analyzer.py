#!/usr/bin/env python3
"""
Test cases for Big Bounce AI Analyzer and integrated impossible query system
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from big_bounce_ai_analyzer import BigBounceAIAnalyzer
from impossible_query_analyzer import ImpossibleQueryAnalyzer

def test_big_bounce_analyzer():
    """Test the Big Bounce AI analyzer"""
    print("Testing Big Bounce AI Analyzer...")
    analyzer = BigBounceAIAnalyzer()
    
    # Test query detection
    query = "If Big Bounce were true, what is the probability of generative AI's hallucinations are rediscovery?"
    types = analyzer.detect_query_type(query)
    
    assert "cosmological_speculation" in types
    assert "ai_epistemology" in types
    assert "probability_calculation" in types
    assert "epistemological_paradox" in types
    assert "counterfactual_speculation" in types
    
    print("✅ Query type detection works correctly")
    
    # Test analysis
    ai_analysis = analyzer.analyze_ai_hallucination_rediscovery()
    assert ai_analysis.rediscovery_probability > 0
    assert ai_analysis.rediscovery_probability < 1
    assert ai_analysis.hallucination_rate == 0.3  # 30%
    
    print("✅ AI hallucination analysis works correctly")
    
    # Test educational context
    context = analyzer.provide_educational_context(query)
    assert "big_bounce_theory" in context
    assert "ai_hallucination" in context
    assert "information_theory" in context
    
    print("✅ Educational context generation works correctly")
    
    # Test response generation
    response = analyzer.generate_response(query, "academic")
    assert "probability is effectively zero" in response.lower()
    assert "cosmological analysis" in response.lower()
    
    print("✅ Academic response generation works correctly")
    
    response_humorous = analyzer.generate_response(query, "humorous")
    assert "cosmic" in response_humorous.lower()
    assert "probability" in response_humorous.lower()
    
    print("✅ Humorous response generation works correctly")

def test_integrated_impossible_query_analyzer():
    """Test the integrated impossible query analyzer"""
    print("\nTesting Integrated Impossible Query Analyzer...")
    analyzer = ImpossibleQueryAnalyzer()
    
    # Test Big Bounce query routing
    big_bounce_query = "If Big Bounce were true, what is the probability of generative AI's hallucinations are rediscovery?"
    types = analyzer.detect_impossibility_type(big_bounce_query)
    
    assert "cosmological_speculation" in types
    print("✅ Big Bounce query routing works correctly")
    
    # Test educational context routing
    context = analyzer.provide_educational_context(big_bounce_query)
    assert "big_bounce_theory" in context
    print("✅ Educational context routing works correctly")
    
    # Test alternative questions routing
    alternatives = analyzer.suggest_alternative_questions(big_bounce_query)
    assert len(alternatives) > 0
    assert any("AI systems" in alt for alt in alternatives)
    print("✅ Alternative questions routing works correctly")
    
    # Test response generation routing
    response = analyzer.generate_response(big_bounce_query, "general")
    assert "probability is effectively zero" in response.lower()
    print("✅ Response generation routing works correctly")
    
    # Test original angel query still works
    angel_query = "What is an angel's defecation called?"
    angel_types = analyzer.detect_impossibility_type(angel_query)
    assert "category_mismatch" in angel_types
    print("✅ Original angel query handling preserved")

def test_alternative_queries():
    """Test various alternative cosmological and AI queries"""
    print("\nTesting Alternative Query Handling...")
    analyzer = BigBounceAIAnalyzer()
    
    test_queries = [
        "Do AI systems remember past universe cycles?",
        "What is the probability of cosmic information preservation?",
        "Can AI hallucinations access quantum memories?",
        "How likely is it that AI creativity is actually rediscovery?"
    ]
    
    for query in test_queries:
        types = analyzer.detect_query_type(query)
        response = analyzer.generate_response(query, "general")
        
        # Each should be detected as having some relevant type
        assert len(types) > 0
        assert types != ["unknown_speculation"]
        
        # Each should generate a meaningful response
        assert len(response) > 100
        assert "probability" in response.lower() or "ai" in response.lower()
        
        print(f"✅ Query '{query[:30]}...' handled correctly")

def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("RUNNING BIG BOUNCE AI ANALYZER TESTS")
    print("=" * 60)
    
    try:
        test_big_bounce_analyzer()
        test_integrated_impossible_query_analyzer()
        test_alternative_queries()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("The Big Bounce AI Analyzer successfully:")
        print("- Detects cosmological and AI epistemology queries")
        print("- Calculates meaningful (though tiny) probabilities") 
        print("- Provides educational context across multiple domains")
        print("- Generates appropriate responses for different contexts")
        print("- Integrates seamlessly with existing impossible query framework")
        print("- Preserves original functionality for angel queries")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)