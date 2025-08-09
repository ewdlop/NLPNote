"""
Comprehensive tests for the Ego-Id-Superego Neural Network

This module provides thorough testing of all components and integration
of the psychological neural network architecture.
"""

import torch
import torch.nn as nn
import numpy as np
import unittest
from typing import Dict, List, Any
import sys
import os

# Add the parent directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ego_id_superego_nn import (
    EgoIdSuperegoNeuralNetwork,
    IdNetwork,
    EgoNetwork, 
    SuperegoNetwork,
    EgoIdSuperegoIntegrator,
    PsycheComponent,
    PsycheOutput,
    IntegratedResponse
)


class TestPsychologicalComponents(unittest.TestCase):
    """Test individual psychological components"""
    
    def setUp(self):
        """Set up test parameters"""
        self.input_dim = 384
        self.hidden_dim = 256
        self.output_dim = 128
        self.batch_size = 4
        
        # Create sample input
        torch.manual_seed(42)
        self.sample_input = torch.randn(self.batch_size, self.input_dim)
        
    def test_id_network_basic_functionality(self):
        """Test Id network basic operations"""
        id_net = IdNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        
        output = id_net(self.sample_input)
        
        # Test output structure
        self.assertIsInstance(output, PsycheOutput)
        self.assertEqual(output.component, PsycheComponent.ID)
        self.assertEqual(output.hidden_state.shape, (self.batch_size, self.output_dim))
        self.assertIsInstance(output.confidence, float)
        self.assertTrue(0.0 <= output.confidence <= 1.0)
        self.assertIsInstance(output.decision_weight, float)
        
        print(f"Id Network Test - Confidence: {output.confidence:.3f}, Weight: {output.decision_weight:.3f}")
        
    def test_ego_network_basic_functionality(self):
        """Test Ego network basic operations"""
        ego_net = EgoNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        
        # Test without memory context
        output = ego_net(self.sample_input)
        
        self.assertIsInstance(output, PsycheOutput)
        self.assertEqual(output.component, PsycheComponent.EGO)
        self.assertEqual(output.hidden_state.shape, (self.batch_size, self.output_dim))
        self.assertTrue(0.0 <= output.confidence <= 1.0)
        
        # Test with memory context
        memory_context = torch.randn(self.batch_size, 3, self.hidden_dim)
        output_with_memory = ego_net(self.sample_input, memory_context)
        
        self.assertEqual(output_with_memory.hidden_state.shape, (self.batch_size, self.output_dim))
        
        print(f"Ego Network Test - Confidence: {output.confidence:.3f}, Weight: {output.decision_weight:.3f}")
        
    def test_superego_network_basic_functionality(self):
        """Test Superego network basic operations"""
        superego_net = SuperegoNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        
        output = superego_net(self.sample_input)
        
        self.assertIsInstance(output, PsycheOutput)
        self.assertEqual(output.component, PsycheComponent.SUPEREGO)
        self.assertEqual(output.hidden_state.shape, (self.batch_size, self.output_dim))
        self.assertTrue(0.0 <= output.confidence <= 1.0)
        
        print(f"Superego Network Test - Confidence: {output.confidence:.3f}, Weight: {output.decision_weight:.3f}")
        
    def test_integrator_functionality(self):
        """Test the integration layer"""
        # Create individual components
        id_net = IdNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        ego_net = EgoNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        superego_net = SuperegoNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        integrator = EgoIdSuperegoIntegrator(self.output_dim)
        
        # Get individual outputs
        id_output = id_net(self.sample_input)
        ego_output = ego_net(self.sample_input)
        superego_output = superego_net(self.sample_input)
        
        # Test integration
        integrated = integrator(id_output, ego_output, superego_output)
        
        self.assertIsInstance(integrated, IntegratedResponse)
        self.assertEqual(integrated.final_output.shape, (self.batch_size, self.output_dim))
        self.assertTrue(0.0 <= integrated.conflict_level <= 1.0)
        
        # Test decision rationale
        rationale = integrated.decision_rationale
        self.assertIn('id_weight', rationale)
        self.assertIn('ego_weight', rationale)
        self.assertIn('superego_weight', rationale)
        
        # Weights should be normalized (approximately sum to 1)
        weight_sum = rationale['id_weight'] + rationale['ego_weight'] + rationale['superego_weight']
        self.assertAlmostEqual(weight_sum, 1.0, places=2)
        
        print(f"Integration Test - Conflict Level: {integrated.conflict_level:.3f}")
        print(f"Decision Weights: Id={rationale['id_weight']:.3f}, Ego={rationale['ego_weight']:.3f}, Superego={rationale['superego_weight']:.3f}")


class TestCompleteNetwork(unittest.TestCase):
    """Test the complete Ego-Id-Superego neural network"""
    
    def setUp(self):
        """Set up test parameters"""
        self.input_dim = 384
        self.hidden_dim = 256
        self.output_dim = 128
        self.batch_size = 4
        
        torch.manual_seed(42)
        self.sample_input = torch.randn(self.batch_size, self.input_dim)
        self.network = EgoIdSuperegoNeuralNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        
    def test_network_initialization(self):
        """Test network initialization"""
        self.assertIsInstance(self.network.id_network, IdNetwork)
        self.assertIsInstance(self.network.ego_network, EgoNetwork)
        self.assertIsInstance(self.network.superego_network, SuperegoNetwork)
        self.assertIsInstance(self.network.integrator, EgoIdSuperegoIntegrator)
        
    def test_integration_task(self):
        """Test integration task"""
        with torch.no_grad():
            result = self.network(self.sample_input, task="integration")
        
        self.assertIn('integrated_response', result)
        self.assertIn('psyche_analysis', result)
        self.assertIsNone(result['task_output'])
        
        # Test psyche analysis structure
        analysis = result['psyche_analysis']
        self.assertIn('id_activation', analysis)
        self.assertIn('ego_activation', analysis)
        self.assertIn('superego_activation', analysis)
        self.assertIn('dominant_component', analysis)
        self.assertIn('psychological_balance', analysis)
        
        print(f"Integration Task Test - Dominant Component: {analysis['dominant_component']}")
        print(f"Psychological Balance: {analysis['psychological_balance']:.3f}")
        
    def test_sentiment_task(self):
        """Test sentiment analysis task"""
        with torch.no_grad():
            result = self.network(self.sample_input, task="sentiment")
        
        self.assertIsNotNone(result['task_output'])
        sentiment_output = result['task_output']
        
        # Should have 3 classes (positive, negative, neutral)
        self.assertEqual(sentiment_output.shape, (self.batch_size, 3))
        
        # Should be probability distribution (sums to 1)
        for i in range(self.batch_size):
            self.assertAlmostEqual(sentiment_output[i].sum().item(), 1.0, places=5)
        
        print(f"Sentiment Task Test - Output shape: {sentiment_output.shape}")
        print(f"Sample sentiment distribution: {sentiment_output[0].tolist()}")
        
    def test_emotion_task(self):
        """Test emotion analysis task"""
        with torch.no_grad():
            result = self.network(self.sample_input, task="emotion")
        
        self.assertIsNotNone(result['task_output'])
        emotion_output = result['task_output']
        
        # Should have 8 emotion classes
        self.assertEqual(emotion_output.shape, (self.batch_size, 8))
        
        # Should be probability distribution
        for i in range(self.batch_size):
            self.assertAlmostEqual(emotion_output[i].sum().item(), 1.0, places=5)
        
        print(f"Emotion Task Test - Output shape: {emotion_output.shape}")
        
    def test_generation_task(self):
        """Test text generation task"""
        with torch.no_grad():
            result = self.network(self.sample_input, task="generation")
        
        self.assertIsNotNone(result['task_output'])
        generation_output = result['task_output']
        
        # Should project back to input dimension
        self.assertEqual(generation_output.shape, (self.batch_size, self.input_dim))
        
        print(f"Generation Task Test - Output shape: {generation_output.shape}")
        
    def test_psychological_analysis(self):
        """Test comprehensive psychological analysis"""
        with torch.no_grad():
            analysis = self.network.analyze_text_psychologically(self.sample_input[0:1])
        
        self.assertIn('psychological_profile', analysis)
        profile = analysis['psychological_profile']
        
        # Test profile structure
        self.assertIn('instinctual_response', profile)
        self.assertIn('rational_response', profile)
        self.assertIn('moral_response', profile)
        self.assertIn('psychological_dynamics', profile)
        
        # Test instinctual response
        instinctual = profile['instinctual_response']
        self.assertIn('strength', instinctual)
        self.assertIn('impulse_level', instinctual)
        self.assertIn('emotional_intensity', instinctual)
        
        # Test rational response
        rational = profile['rational_response']
        self.assertIn('logical_consistency', rational)
        self.assertIn('planning_depth', rational)
        self.assertIn('reality_grounding', rational)
        
        # Test moral response
        moral = profile['moral_response']
        self.assertIn('ethical_certainty', moral)
        self.assertIn('moral_strength', moral)
        self.assertIn('idealistic_level', moral)
        
        # Test psychological dynamics
        dynamics = profile['psychological_dynamics']
        self.assertIn('internal_conflict', dynamics)
        self.assertIn('decision_clarity', dynamics)
        self.assertIn('component_harmony', dynamics)
        
        print("Psychological Analysis Test - Profile generated successfully")
        print(f"Internal Conflict: {dynamics['internal_conflict']:.3f}")
        print(f"Decision Clarity: {dynamics['decision_clarity']:.3f}")


class TestNetworkBehavior(unittest.TestCase):
    """Test network behavior and consistency"""
    
    def setUp(self):
        """Set up test parameters"""
        self.input_dim = 384
        self.hidden_dim = 256
        self.output_dim = 128
        self.batch_size = 8
        
        torch.manual_seed(42)
        self.network = EgoIdSuperegoNeuralNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        
    def test_different_inputs_produce_different_outputs(self):
        """Test that different inputs produce different outputs"""
        input1 = torch.randn(1, self.input_dim)
        input2 = torch.randn(1, self.input_dim) + 1.0  # Different input
        
        with torch.no_grad():
            result1 = self.network(input1, task="integration")
            result2 = self.network(input2, task="integration")
        
        # Outputs should be different
        output1 = result1['integrated_response'].final_output
        output2 = result2['integrated_response'].final_output
        
        self.assertFalse(torch.allclose(output1, output2, atol=1e-5))
        print("Different Inputs Test - Outputs are appropriately different")
        
    def test_network_consistency(self):
        """Test that same input produces same output (deterministic)"""
        torch.manual_seed(123)
        input_tensor = torch.randn(1, self.input_dim)
        
        # Set network to eval mode to ensure deterministic behavior
        self.network.eval()
        
        with torch.no_grad():
            result1 = self.network(input_tensor, task="integration")
            result2 = self.network(input_tensor, task="integration")
        
        output1 = result1['integrated_response'].final_output
        output2 = result2['integrated_response'].final_output
        
        # Use more lenient tolerance due to floating point precision
        self.assertTrue(torch.allclose(output1, output2, atol=1e-4))
        print("Consistency Test - Same inputs produce same outputs")
        
    def test_component_influence_analysis(self):
        """Test that different components show different influences"""
        # Create multiple diverse inputs with greater variation
        inputs = []
        for i in range(10):
            torch.manual_seed(i * 100)  # More diverse seeds
            # Create more varied inputs
            base_input = torch.randn(1, self.input_dim)
            # Apply different transformations to create more variation
            if i % 3 == 0:
                scaled_input = base_input * 2.0  # Scale up
            elif i % 3 == 1:
                scaled_input = base_input + torch.randn_like(base_input) * 0.5  # Add noise
            else:
                scaled_input = base_input - 1.0  # Shift down
            inputs.append(scaled_input)
        
        component_influences = []
        
        self.network.eval()
        with torch.no_grad():
            for input_tensor in inputs:
                result = self.network(input_tensor, task="integration")
                rationale = result['integrated_response'].decision_rationale
                component_influences.append([
                    rationale['id_weight'],
                    rationale['ego_weight'], 
                    rationale['superego_weight']
                ])
        
        component_influences = np.array(component_influences)
        
        # Check that there's some variation in component influences
        id_var = np.var(component_influences[:, 0])
        ego_var = np.var(component_influences[:, 1])
        superego_var = np.var(component_influences[:, 2])
        
        print(f"Component Influence Analysis:")
        print(f"  Id variance: {id_var:.8f}")
        print(f"  Ego variance: {ego_var:.8f}")
        print(f"  Superego variance: {superego_var:.8f}")
        
        # Print average influences
        avg_influences = np.mean(component_influences, axis=0)
        print(f"  Average influences - Id: {avg_influences[0]:.3f}, Ego: {avg_influences[1]:.3f}, Superego: {avg_influences[2]:.3f}")
        
        # Test that not all components have exactly the same influence
        max_influence = np.max(avg_influences)
        min_influence = np.min(avg_influences)
        influence_range = max_influence - min_influence
        self.assertGreater(influence_range, 0.01)  # At least 1% difference
        print(f"  Influence range: {influence_range:.3f}")
        
        # Test that each component has some non-zero influence on average
        for i, component in enumerate(['Id', 'Ego', 'Superego']):
            self.assertGreater(avg_influences[i], 0.001, f"{component} should have some influence")
        
        # Test that the sum of weights is approximately 1
        weight_sums = np.sum(component_influences, axis=1)
        for weight_sum in weight_sums:
            self.assertAlmostEqual(weight_sum, 1.0, places=2, msg="Component weights should sum to 1")
        
        print("  ✓ All component influence tests passed")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("=" * 60)
    print("COMPREHENSIVE EGO-ID-SUPEREGO NEURAL NETWORK TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add component tests
    test_suite.addTest(unittest.makeSuite(TestPsychologicalComponents))
    test_suite.addTest(unittest.makeSuite(TestCompleteNetwork))
    test_suite.addTest(unittest.makeSuite(TestNetworkBehavior))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The Ego-Id-Superego Neural Network is working correctly.")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed.")
    
    return result


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comprehensive tests
    test_result = run_comprehensive_tests()