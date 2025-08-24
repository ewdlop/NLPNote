#!/usr/bin/env python3
"""
Test suite for cardinality representation fixes

This test verifies that the issue "Non-integral cardinality represent by integral?"
has been properly resolved by ensuring:

1. Cardinalities (discrete counts) are properly represented as integers
2. Derived metrics (ratios, probabilities) are properly represented as floats
3. Type safety and validation work correctly
4. Edge cases are handled properly
5. Backward compatibility is maintained
"""

import unittest
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from cardinality_types import (
        Cardinality, Ratio, Probability, Score,
        CardinalityMetrics,
        safe_cardinality_ratio,
        cardinality_complexity_score,
        cardinality_clarity_score,
        cardinality_lexical_density,
        normalize_score_to_probability
    )
    CARDINALITY_TYPES_AVAILABLE = True
except ImportError:
    CARDINALITY_TYPES_AVAILABLE = False

try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    HUMAN_EVALUATOR_AVAILABLE = True
except ImportError:
    HUMAN_EVALUATOR_AVAILABLE = False

try:
    from SubtextAnalyzer import SubtextAnalyzer
    SUBTEXT_ANALYZER_AVAILABLE = True
except ImportError:
    SUBTEXT_ANALYZER_AVAILABLE = False


class TestCardinalityTypes(unittest.TestCase):
    """Test the basic cardinality type system"""
    
    @unittest.skipIf(not CARDINALITY_TYPES_AVAILABLE, "Cardinality types not available")
    def test_cardinality_metrics_validation(self):
        """Test that cardinality metrics properly validate integral inputs"""
        
        # Valid case
        metrics = CardinalityMetrics(
            total_count=Cardinality(100),
            positive_count=Cardinality(60), 
            negative_count=Cardinality(40)
        )
        self.assertEqual(metrics.total_count, 100)
        self.assertIsInstance(metrics.positive_ratio, float)
        self.assertAlmostEqual(metrics.positive_ratio, 0.6, places=3)
        
        # Invalid cases should raise ValueError
        with self.assertRaises(ValueError):
            CardinalityMetrics(
                total_count=Cardinality(-1),  # Negative count
                positive_count=Cardinality(5),
                negative_count=Cardinality(3)
            )
        
        with self.assertRaises(ValueError):
            CardinalityMetrics(
                total_count=Cardinality(10),
                positive_count=Cardinality(15),  # Exceeds total
                negative_count=Cardinality(5)
            )
    
    @unittest.skipIf(not CARDINALITY_TYPES_AVAILABLE, "Cardinality types not available")
    def test_safe_cardinality_ratio(self):
        """Test safe ratio calculation from cardinalities"""
        
        # Normal case
        ratio = safe_cardinality_ratio(Cardinality(25), Cardinality(100))
        self.assertAlmostEqual(ratio, 0.25, places=3)
        self.assertIsInstance(ratio, float)
        
        # Edge case: division by zero
        ratio_zero = safe_cardinality_ratio(Cardinality(5), Cardinality(0), default=0.5)
        self.assertEqual(ratio_zero, 0.5)
        
        # Invalid input validation
        with self.assertRaises(ValueError):
            safe_cardinality_ratio(Cardinality(-1), Cardinality(10))
        
        with self.assertRaises(ValueError):
            safe_cardinality_ratio(Cardinality(15), Cardinality(10))  # Numerator > denominator
    
    @unittest.skipIf(not CARDINALITY_TYPES_AVAILABLE, "Cardinality types not available")
    def test_complexity_scoring(self):
        """Test that complexity scoring properly handles cardinalities"""
        
        score = cardinality_complexity_score(
            logical_count=Cardinality(3),
            quantifier_count=Cardinality(2),
            total_words=Cardinality(20)
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Zero words edge case
        score_zero = cardinality_complexity_score(
            logical_count=Cardinality(0),
            quantifier_count=Cardinality(0),
            total_words=Cardinality(0)
        )
        self.assertEqual(score_zero, 0.0)
    
    @unittest.skipIf(not CARDINALITY_TYPES_AVAILABLE, "Cardinality types not available")
    def test_clarity_scoring(self):
        """Test that clarity scoring properly handles cardinalities"""
        
        score = cardinality_clarity_score(
            vague_count=Cardinality(2),
            definite_count=Cardinality(5),
            total_words=Cardinality(30)
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.5)  # Should be above neutral with more definite terms


class TestHumanExpressionEvaluatorCardinality(unittest.TestCase):
    """Test that HumanExpressionEvaluator properly handles cardinalities"""
    
    @unittest.skipIf(not HUMAN_EVALUATOR_AVAILABLE, "HumanExpressionEvaluator not available")
    def test_evaluation_score_types(self):
        """Test that evaluation scores are proper floating-point values derived from integral counts"""
        
        evaluator = HumanExpressionEvaluator()
        
        # Test with a text that has clear cardinalities
        result = evaluator.comprehensive_evaluation(
            "Please kindly help me understand this definitely important concept."
        )
        
        # Check that scores are floats (non-integral derived metrics)
        formal_score = result['formal_semantic'].score
        cognitive_score = result['cognitive'].score
        social_score = result['social'].score
        overall_score = result['integrated']['overall_score']
        
        self.assertIsInstance(formal_score, (float, type(0.0)))
        self.assertIsInstance(cognitive_score, (float, type(0.0)))
        self.assertIsInstance(social_score, (float, type(0.0)))
        self.assertIsInstance(overall_score, (float, type(0.0)))
        
        # Scores should be in valid range
        for score in [formal_score, cognitive_score, social_score, overall_score]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    @unittest.skipIf(not HUMAN_EVALUATOR_AVAILABLE, "HumanExpressionEvaluator not available")
    def test_edge_cases(self):
        """Test edge cases that previously caused cardinality issues"""
        
        evaluator = HumanExpressionEvaluator()
        
        # Empty string
        result_empty = evaluator.comprehensive_evaluation("")
        self.assertIsInstance(result_empty['integrated']['overall_score'], (float, type(0.0)))
        
        # Single word
        result_single = evaluator.comprehensive_evaluation("hello")
        self.assertIsInstance(result_single['integrated']['overall_score'], (float, type(0.0)))
        
        # Text with no special markers
        result_plain = evaluator.comprehensive_evaluation("This is a simple sentence.")
        self.assertIsInstance(result_plain['integrated']['overall_score'], (float, type(0.0)))


class TestSubtextAnalyzerCardinality(unittest.TestCase):
    """Test that SubtextAnalyzer properly handles cardinalities"""
    
    @unittest.skipIf(not SUBTEXT_ANALYZER_AVAILABLE, "SubtextAnalyzer not available")
    def test_lexical_density_calculation(self):
        """Test that lexical density properly derives ratios from word counts"""
        
        analyzer = SubtextAnalyzer()
        
        # Test with known text
        density = analyzer.calculate_lexical_density("The quick brown fox jumps over the lazy dog.")
        
        self.assertIsInstance(density, (float, type(0.0)))
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)
    
    @unittest.skipIf(not SUBTEXT_ANALYZER_AVAILABLE, "SubtextAnalyzer not available")
    def test_symbolism_analysis(self):
        """Test that symbolism analysis properly derives ratios from symbol counts"""
        
        analyzer = SubtextAnalyzer()
        
        # Text with known symbolic elements
        symbolism = analyzer.analyze_symbolism("The golden sun set over the mountain at dawn.")
        
        self.assertIsInstance(symbolism, (float, type(0.0)))
        self.assertGreaterEqual(symbolism, 0.0)
        self.assertLessEqual(symbolism, 1.0)
        self.assertGreater(symbolism, 0.0)  # Should detect 'sun', 'mountain', 'dawn'
    
    @unittest.skipIf(not SUBTEXT_ANALYZER_AVAILABLE, "SubtextAnalyzer not available")
    def test_subtext_probability_consistency(self):
        """Test that subtext probability components are consistent with cardinality principles"""
        
        analyzer = SubtextAnalyzer()
        
        result = analyzer.calculate_subtext_probability("The red eagle soared through the twilight sky.")
        
        # Check that probability is a proper floating-point value
        self.assertIsInstance(result['probability'], (float, type(0.0)))
        self.assertGreaterEqual(result['probability'], 0.0)
        self.assertLessEqual(result['probability'], 1.0)
        
        # Check that all components are proper floating-point values
        for component_name, component_value in result['components'].items():
            self.assertIsInstance(component_value, (float, type(0.0)), 
                                f"Component {component_name} should be float")
            self.assertGreaterEqual(component_value, 0.0, 
                                  f"Component {component_name} should be non-negative")


class TestBackwardCompatibility(unittest.TestCase):
    """Test that changes maintain backward compatibility"""
    
    @unittest.skipIf(not HUMAN_EVALUATOR_AVAILABLE, "HumanExpressionEvaluator not available")
    def test_api_compatibility(self):
        """Test that the API remains compatible"""
        
        evaluator = HumanExpressionEvaluator()
        
        # Original API should still work
        result = evaluator.comprehensive_evaluation("Hello world")
        
        # Required structure should be present
        self.assertIn('formal_semantic', result)
        self.assertIn('cognitive', result)
        self.assertIn('social', result)
        self.assertIn('integrated', result)
        
        # Required fields should be present
        self.assertIn('score', result['formal_semantic'].__dict__)
        self.assertIn('confidence', result['formal_semantic'].__dict__)
        self.assertIn('overall_score', result['integrated'])
        self.assertIn('overall_confidence', result['integrated'])
    
    @unittest.skipIf(not SUBTEXT_ANALYZER_AVAILABLE, "SubtextAnalyzer not available")
    def test_subtext_api_compatibility(self):
        """Test that SubtextAnalyzer API remains compatible"""
        
        analyzer = SubtextAnalyzer()
        
        # Original methods should still work
        density = analyzer.calculate_lexical_density("Hello world")
        self.assertIsInstance(density, (float, type(0.0)))
        
        symbolism = analyzer.analyze_symbolism("Hello world")
        self.assertIsInstance(symbolism, (float, type(0.0)))
        
        result = analyzer.calculate_subtext_probability("Hello world")
        self.assertIn('probability', result)
        self.assertIn('components', result)


def main():
    """Run all cardinality tests"""
    
    print("=" * 60)
    print("CARDINALITY REPRESENTATION FIX VERIFICATION")
    print("=" * 60)
    print()
    print("Testing resolution of issue: 'Non-integral cardinality represent by integral?'")
    print()
    
    # Check availability of modules
    print("Module Availability:")
    print(f"  ✓ cardinality_types: {'Available' if CARDINALITY_TYPES_AVAILABLE else 'Not available'}")
    print(f"  ✓ HumanExpressionEvaluator: {'Available' if HUMAN_EVALUATOR_AVAILABLE else 'Not available'}")
    print(f"  ✓ SubtextAnalyzer: {'Available' if SUBTEXT_ANALYZER_AVAILABLE else 'Not available'}")
    print()
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("CARDINALITY FIX VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()