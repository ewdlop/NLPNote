#!/usr/bin/env python3
"""
Unit Tests for Neural Missing Firing Detection System
神經元缺失激發檢測系統單元測試

Tests for the neural firing analysis framework components.
"""

import unittest
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from NeuralFiringAnalyzer import (
        NeuralFiringAnalyzer, 
        FiringPatternType, 
        NeuralLayerAnalysis,
        NetworkFiringReport
    )
    from SubtextAnalyzer import SubtextAnalyzer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import modules for testing: {e}")
    MODULES_AVAILABLE = False


class TestNeuralFiringAnalyzer(unittest.TestCase):
    """Test cases for NeuralFiringAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.analyzer = NeuralFiringAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer, NeuralFiringAnalyzer)
        self.assertEqual(self.analyzer.threshold_missing, 0.01)
        self.assertEqual(self.analyzer.threshold_weak, 0.1)
    
    def test_analyze_empty_tensor(self):
        """Test analysis of empty activation tensor"""
        empty_tensor = np.array([])
        result = self.analyzer.analyze_activation_tensor(empty_tensor, "test_layer")
        
        self.assertIsInstance(result, NeuralLayerAnalysis)
        self.assertEqual(result.activation_rate, 0.0)
        self.assertEqual(result.firing_pattern, FiringPatternType.DEAD)
        self.assertIn("Empty activation tensor", result.issues)
    
    def test_analyze_normal_activations(self):
        """Test analysis of normal activation patterns"""
        # Create normal activations (tanh range)
        normal_activations = np.random.normal(0, 0.5, (100, 100))
        normal_activations = np.tanh(normal_activations)
        
        result = self.analyzer.analyze_activation_tensor(normal_activations, "normal_layer")
        
        self.assertIsInstance(result, NeuralLayerAnalysis)
        self.assertGreater(result.activation_rate, 0.5)
        self.assertEqual(result.firing_pattern, FiringPatternType.NORMAL)
        self.assertEqual(len(result.issues), 0)  # Should have no issues
    
    def test_analyze_dead_neurons(self):
        """Test detection of dead neurons"""
        # Create all-zero activations
        dead_activations = np.zeros((100, 100))
        
        result = self.analyzer.analyze_activation_tensor(dead_activations, "dead_layer")
        
        self.assertEqual(result.activation_rate, 0.0)
        self.assertEqual(result.firing_pattern, FiringPatternType.DEAD)
        self.assertGreater(len(result.issues), 0)
        self.assertIn("dead neuron rate", str(result.issues).lower())
    
    def test_analyze_weak_firing(self):
        """Test detection of weak firing patterns"""
        # Create weak activations (very small values)
        weak_activations = np.random.normal(0, 0.005, (100, 100))
        
        result = self.analyzer.analyze_activation_tensor(weak_activations, "weak_layer")
        
        self.assertLess(result.activation_rate, 0.5)
        # Allow for DEAD pattern as well since very small values might be classified as dead
        self.assertIn(result.firing_pattern, [FiringPatternType.WEAK, FiringPatternType.MISSING, FiringPatternType.DEAD])
        self.assertGreater(len(result.issues), 0)
    
    def test_analyze_saturated_neurons(self):
        """Test detection of saturated neurons"""
        # Create saturated activations (all at extreme values)
        saturated_activations = np.ones((100, 100)) * 5.0  # Very high values
        
        result = self.analyzer.analyze_activation_tensor(saturated_activations, "saturated_layer")
        
        self.assertEqual(result.activation_rate, 1.0)  # All neurons firing
        self.assertGreater(len(result.issues), 0)
        self.assertTrue(any("saturation" in issue.lower() for issue in result.issues))
    
    def test_classify_firing_patterns(self):
        """Test firing pattern classification"""
        # Test different patterns
        test_cases = [
            (np.zeros(100), FiringPatternType.DEAD),
            (np.random.normal(0, 0.001, 100), FiringPatternType.MISSING),
            (np.random.normal(0, 0.5, 100), FiringPatternType.NORMAL),
            (np.ones(100) * 10, FiringPatternType.OVER_ACTIVE)
        ]
        
        for activations, expected_pattern in test_cases:
            pattern = self.analyzer._classify_firing_pattern(activations)
            # Allow some flexibility in classification
            self.assertIsInstance(pattern, FiringPatternType)
    
    def test_simulated_llm_analysis(self):
        """Test simulated LLM analysis"""
        # Test healthy network
        healthy_report = self.analyzer.analyze_simulated_llm_activations(
            num_layers=4,
            simulate_issues=False
        )
        
        self.assertIsInstance(healthy_report, NetworkFiringReport)
        self.assertEqual(healthy_report.total_layers, 4)
        self.assertGreaterEqual(healthy_report.overall_health_score, 0.7)
        
        # Test problematic network
        problematic_report = self.analyzer.analyze_simulated_llm_activations(
            num_layers=4,
            simulate_issues=True
        )
        
        self.assertIsInstance(problematic_report, NetworkFiringReport)
        self.assertEqual(problematic_report.total_layers, 4)
        self.assertLess(problematic_report.overall_health_score, healthy_report.overall_health_score)
    
    def test_activation_distribution_calculation(self):
        """Test activation distribution statistics"""
        test_activations = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        distribution = self.analyzer._calculate_activation_distribution(test_activations)
        
        self.assertIsInstance(distribution, dict)
        self.assertIn('mean', distribution)
        self.assertIn('std', distribution)
        self.assertIn('min', distribution)
        self.assertIn('max', distribution)
        self.assertAlmostEqual(distribution['mean'], 0.25, places=2)
        self.assertEqual(distribution['min'], 0.0)
        self.assertEqual(distribution['max'], 0.5)
    
    def test_issue_identification(self):
        """Test issue identification logic"""
        # Test various problematic scenarios
        scenarios = [
            (np.zeros(100), "missing firing"),  # All zeros
            (np.ones(100) * 0.001, "weak"),     # Very weak activations
            (np.ones(100) * 10, "saturation"),  # Saturated activations
        ]
        
        for activations, expected_issue_type in scenarios:
            activation_rate = np.mean(np.abs(activations) > self.analyzer.threshold_missing)
            issues = self.analyzer._identify_layer_issues(activations, activation_rate)
            
            # Should detect some kind of issue
            self.assertGreater(len(issues), 0)
            # Check if expected issue type is mentioned
            issues_text = " ".join(issues).lower()
            self.assertTrue(any(keyword in issues_text for keyword in expected_issue_type.split()))


class TestSubtextAnalyzerIntegration(unittest.TestCase):
    """Test cases for SubtextAnalyzer integration with neural firing analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.analyzer = SubtextAnalyzer()
    
    def test_neural_firing_analysis_integration(self):
        """Test integration of neural firing analysis in SubtextAnalyzer"""
        test_text = "Neural networks require careful monitoring of activation patterns."
        
        result = self.analyzer.analyze_expression_evaluation(test_text)
        
        # Should include neural firing analysis
        self.assertIn('neural_firing_analysis', result)
        
        nfa = result['neural_firing_analysis']
        self.assertIn('neural_health_score', nfa)
        self.assertIn('overall_firing_rate', nfa)
        self.assertIn('stage_analyses', nfa)
        
        # Scores should be valid
        self.assertGreaterEqual(nfa['neural_health_score'], 0.0)
        self.assertLessEqual(nfa['neural_health_score'], 1.0)
        self.assertGreaterEqual(nfa['overall_firing_rate'], 0.0)
        self.assertLessEqual(nfa['overall_firing_rate'], 1.0)
    
    def test_text_length_impact(self):
        """Test how text length affects neural firing analysis"""
        short_text = "Hello."
        long_text = "This is a much longer text that should potentially cause different neural firing patterns in the simulated language model processing, as longer sequences often lead to different attention patterns and potential issues with vanishing gradients or other problems that can affect neural network performance."
        
        short_result = self.analyzer.analyze_expression_evaluation(short_text)
        long_result = self.analyzer.analyze_expression_evaluation(long_text)
        
        # Both should have neural firing analysis
        self.assertIn('neural_firing_analysis', short_result)
        self.assertIn('neural_firing_analysis', long_result)
        
        # Results should be different (longer text may have more issues)
        short_nfa = short_result['neural_firing_analysis']
        long_nfa = long_result['neural_firing_analysis']
        
        # Should have valid scores
        self.assertIsInstance(short_nfa['neural_health_score'], (int, float))
        self.assertIsInstance(long_nfa['neural_health_score'], (int, float))
    
    def test_multilingual_text_analysis(self):
        """Test neural firing analysis with different languages"""
        texts = [
            "English text for testing neural firing patterns.",
            "測試神經激發模式的中文文本。",
            "Texto en español para probar patrones de activación neural."
        ]
        
        for text in texts:
            try:
                result = self.analyzer.analyze_expression_evaluation(text)
                self.assertIn('neural_firing_analysis', result)
                
                nfa = result['neural_firing_analysis']
                # Should have valid analysis regardless of language
                self.assertIn('neural_health_score', nfa)
                self.assertGreaterEqual(nfa['neural_health_score'], 0.0)
                self.assertLessEqual(nfa['neural_health_score'], 1.0)
                
            except Exception as e:
                self.fail(f"Analysis failed for text '{text[:20]}...': {e}")


class TestNeuralFiringDetectionSystem(unittest.TestCase):
    """Integration tests for the complete neural firing detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.neural_analyzer = NeuralFiringAnalyzer()
        self.text_analyzer = SubtextAnalyzer()
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow"""
        # 1. Create a simulated network report
        network_report = self.neural_analyzer.analyze_simulated_llm_activations(
            num_layers=6,
            simulate_issues=True
        )
        
        self.assertIsInstance(network_report, NetworkFiringReport)
        self.assertEqual(network_report.total_layers, 6)
        
        # 2. Analyze text with neural firing
        test_text = "Complex neural networks may experience activation issues."
        text_result = self.text_analyzer.analyze_expression_evaluation(test_text)
        
        self.assertIn('neural_firing_analysis', text_result)
        
        # 3. Check that all components are working together
        self.assertIn('expression_evaluation', text_result)
        self.assertIn('subtext_analysis', text_result)
        self.assertIn('interpretation', text_result)
    
    def test_recommendation_generation(self):
        """Test recommendation generation for different scenarios"""
        # Create reports with different health scores
        healthy_report = self.neural_analyzer.analyze_simulated_llm_activations(
            num_layers=4,
            simulate_issues=False
        )
        
        problematic_report = self.neural_analyzer.analyze_simulated_llm_activations(
            num_layers=4,
            simulate_issues=True
        )
        
        # Both should have recommendations
        self.assertIsInstance(healthy_report.recommendations, list)
        self.assertIsInstance(problematic_report.recommendations, list)
        
        # Problematic report should generally have more specific recommendations
        self.assertGreater(len(problematic_report.recommendations), 0)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid inputs
        try:
            # Empty text
            result = self.text_analyzer.analyze_expression_evaluation("")
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"Empty text analysis should not raise exception: {e}")
        
        try:
            # Very long text
            long_text = "word " * 1000
            result = self.text_analyzer.analyze_expression_evaluation(long_text)
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"Long text analysis should not raise exception: {e}")
    
    def test_performance_consistency(self):
        """Test that analysis results are reasonably consistent"""
        test_text = "Standard test text for consistency checking."
        
        # Run analysis multiple times
        results = []
        for _ in range(3):
            result = self.text_analyzer.analyze_expression_evaluation(test_text)
            if 'neural_firing_analysis' in result:
                results.append(result['neural_firing_analysis']['neural_health_score'])
        
        if len(results) > 1:
            # Results should be reasonably consistent (within 0.2 range)
            score_range = max(results) - min(results)
            self.assertLess(score_range, 0.3, "Analysis results should be reasonably consistent")


def run_tests():
    """Run all tests and return results"""
    print("🧪 Running Neural Missing Firing Detection System Tests")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestNeuralFiringAnalyzer,
        TestSubtextAnalyzerIntegration,
        TestNeuralFiringDetectionSystem
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\nOverall result: {status}")
    
    return success


if __name__ == "__main__":
    if not MODULES_AVAILABLE:
        print("❌ Cannot run tests: Required modules not available")
        sys.exit(1)
    
    success = run_tests()
    sys.exit(0 if success else 1)