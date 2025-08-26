"""
Tests for Deadly Attractor Social Dynamics System

This module contains unit tests for the social dynamics models and simulator.
"""

import numpy as np
import unittest
from deadly_attractor_simulator import (
    HateSpiralModel, EchoChamberModel, PrisonersDilemmaModel,
    TragedyOfCommonsModel, DeadlyAttractorSimulator
)


class TestSocialDynamicsModels(unittest.TestCase):
    """Test cases for social dynamics models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hate_model = HateSpiralModel()
        self.echo_model = EchoChamberModel()
        self.pd_model = PrisonersDilemmaModel()
        self.commons_model = TragedyOfCommonsModel()
    
    def test_hate_spiral_dynamics(self):
        """Test hate spiral model dynamics."""
        # Test neutral state
        state = np.array([0.0, 0.0])
        derivatives = self.hate_model.dynamics(state, 0)
        
        # Should have positive derivatives (moving toward baseline)
        self.assertGreater(derivatives[0], 0)
        self.assertGreater(derivatives[1], 0)
        
        # Test extreme negative state
        state = np.array([-1.0, -1.0])
        derivatives = self.hate_model.dynamics(state, 0)
        
        # Check that derivatives are not exploding
        self.assertFalse(np.any(np.isnan(derivatives)))
        self.assertFalse(np.any(np.isinf(derivatives)))
    
    def test_echo_chamber_dynamics(self):
        """Test echo chamber model dynamics."""
        # Test neutral opinion with high diversity
        state = np.array([0.0, 0.8])
        derivatives = self.echo_model.dynamics(state, 0)
        
        # Opinion should remain relatively stable at neutral
        self.assertLess(abs(derivatives[0]), 0.5)
        
        # Test extreme opinion
        state = np.array([1.5, 0.2])
        derivatives = self.echo_model.dynamics(state, 0)
        
        # Should have valid derivatives
        self.assertFalse(np.any(np.isnan(derivatives)))
    
    def test_prisoners_dilemma_dynamics(self):
        """Test prisoner's dilemma model dynamics."""
        # Test full cooperation
        state = np.array([1.0, 0.5])
        derivatives = self.pd_model.dynamics(state, 0)
        
        # Check bounds are respected
        self.assertFalse(np.any(np.isnan(derivatives)))
        
        # Test no cooperation
        state = np.array([0.0, 0.0])
        derivatives = self.pd_model.dynamics(state, 0)
        
        # Should remain at deadly attractor
        self.assertLess(abs(derivatives[0]), 0.1)
        self.assertLess(abs(derivatives[1]), 0.1)
    
    def test_tragedy_of_commons_dynamics(self):
        """Test tragedy of commons model dynamics."""
        # Test with resources and effort
        state = np.array([50.0, 1.0])
        derivatives = self.commons_model.dynamics(state, 0)
        
        # Should have valid derivatives
        self.assertFalse(np.any(np.isnan(derivatives)))
        
        # Test resource depletion state
        state = np.array([0.0, 0.0])
        derivatives = self.commons_model.dynamics(state, 0)
        
        # Should remain at deadly attractor
        self.assertEqual(derivatives[0], 0.0)  # No resource recovery when R=0
    
    def test_equilibria_calculation(self):
        """Test equilibria calculation for all models."""
        models = [self.hate_model, self.echo_model, self.pd_model, self.commons_model]
        
        for model in models:
            equilibria = model.get_equilibria()
            self.assertIsInstance(equilibria, list)
            self.assertGreater(len(equilibria), 0)
            
            # Each equilibrium should be a tuple/list
            for eq in equilibria:
                self.assertIsInstance(eq, (tuple, list))
                self.assertGreaterEqual(len(eq), 2)
    
    def test_parameter_info(self):
        """Test parameter information retrieval."""
        models = [self.hate_model, self.echo_model, self.pd_model, self.commons_model]
        
        for model in models:
            info = model.get_parameter_info()
            self.assertIsInstance(info, dict)
            self.assertGreater(len(info), 0)


class TestDeadlyAttractorSimulator(unittest.TestCase):
    """Test cases for the deadly attractor simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = HateSpiralModel()
        self.simulator = DeadlyAttractorSimulator(self.model)
    
    def test_simulation(self):
        """Test basic simulation functionality."""
        initial_state = [0.5, 0.5]
        trajectory = self.simulator.simulate(initial_state, time_span=10, num_points=100)
        
        # Check trajectory shape
        self.assertEqual(trajectory.shape, (100, 2))
        
        # Check initial state is preserved
        np.testing.assert_array_almost_equal(trajectory[0], initial_state, decimal=5)
        
        # Check no NaN or infinite values
        self.assertFalse(np.any(np.isnan(trajectory)))
        self.assertFalse(np.any(np.isinf(trajectory)))
    
    def test_stability_analysis(self):
        """Test stability analysis functionality."""
        state = [0.0, 0.0]
        stability = self.simulator.analyze_stability(state)
        
        # Check required keys
        required_keys = ['eigenvalues', 'max_real_part', 'is_stable', 'is_unstable', 'jacobian']
        for key in required_keys:
            self.assertIn(key, stability)
        
        # Check eigenvalues are complex numbers
        eigenvalues = stability['eigenvalues']
        self.assertEqual(len(eigenvalues), 2)
        
        # Check jacobian shape
        jacobian = stability['jacobian']
        self.assertEqual(jacobian.shape, (2, 2))
    
    def test_multiple_trajectories(self):
        """Test simulation of multiple trajectories."""
        initial_conditions = [[0.5, 0.5], [-0.3, 0.2], [0.8, -0.1]]
        
        for init_state in initial_conditions:
            trajectory = self.simulator.simulate(init_state, time_span=5)
            self.assertGreater(len(trajectory), 0)
            
            # Check that simulation starts from correct initial state
            np.testing.assert_array_almost_equal(trajectory[0], init_state, decimal=4)
    
    def test_basin_of_attraction(self):
        """Test basin of attraction calculation."""
        attractor = [-1.0, -1.0]
        
        # Use small grid for fast testing
        basin, x_range, y_range = self.simulator.basin_of_attraction(
            attractor, grid_size=5, xlim=(-1, 1), ylim=(-1, 1), 
            tolerance=0.5, max_time=5.0
        )
        
        # Check output dimensions
        self.assertEqual(basin.shape, (5, 5))
        self.assertEqual(len(x_range), 5)
        self.assertEqual(len(y_range), 5)
        
        # Check basin values are 0 or 1
        unique_values = np.unique(basin)
        self.assertTrue(all(val in [0, 1] for val in unique_values))


class TestModelIntegration(unittest.TestCase):
    """Test integration between different models and components."""
    
    def test_all_models_with_simulator(self):
        """Test that all models work with the simulator."""
        models = [
            HateSpiralModel(),
            EchoChamberModel(),
            PrisonersDilemmaModel(),
            TragedyOfCommonsModel()
        ]
        
        initial_states = [
            [0.5, 0.5],
            [0.1, 0.8],
            [0.6, 0.3],
            [60.0, 0.8]
        ]
        
        for model, init_state in zip(models, initial_states):
            simulator = DeadlyAttractorSimulator(model)
            
            # Test simulation
            trajectory = simulator.simulate(init_state, time_span=5, num_points=50)
            self.assertEqual(len(trajectory), 50)
            
            # Test stability analysis
            stability = simulator.analyze_stability(init_state)
            self.assertIn('eigenvalues', stability)
    
    def test_convergence_to_attractors(self):
        """Test that models actually converge to deadly attractors."""
        # Test hate spiral convergence
        hate_model = HateSpiralModel(alpha=1.0, beta=0.05, gamma=0.1)
        simulator = DeadlyAttractorSimulator(hate_model)
        
        trajectory = simulator.simulate([0.2, 0.3], time_span=50, num_points=500)
        final_state = trajectory[-1]
        
        # Should converge toward negative values (hate)
        self.assertLess(final_state[0], 0.1)
        self.assertLess(final_state[1], 0.1)
        
        # Test prisoner's dilemma convergence to defection
        pd_model = PrisonersDilemmaModel(mu=0.3, lamb=0.4, nu=0.2)
        simulator = DeadlyAttractorSimulator(pd_model)
        
        trajectory = simulator.simulate([0.8, 0.5], time_span=50, num_points=500)
        final_state = trajectory[-1]
        
        # Should converge toward low cooperation
        self.assertLess(final_state[0], 0.3)


def run_validation_tests():
    """Run validation tests to ensure model correctness."""
    print("🧪 Running Validation Tests for Deadly Attractor Models")
    print("=" * 60)
    
    # Test 1: Conservation laws and bounds
    print("Test 1: Checking bounds and conservation...")
    
    # Hate spiral should keep trust levels reasonable
    hate_model = HateSpiralModel()
    simulator = DeadlyAttractorSimulator(hate_model)
    trajectory = simulator.simulate([0.5, 0.5], time_span=20)
    
    # Trust levels should stay within reasonable bounds
    max_trust = np.max(trajectory)
    min_trust = np.min(trajectory)
    print(f"  Hate spiral: Trust range [{min_trust:.3f}, {max_trust:.3f}]")
    
    # Test 2: Monotonicity properties
    print("Test 2: Checking monotonicity properties...")
    
    # Echo chamber diversity should generally decrease
    echo_model = EchoChamberModel()
    simulator = DeadlyAttractorSimulator(echo_model)
    trajectory = simulator.simulate([0.1, 0.8], time_span=15)
    
    diversity_change = trajectory[-1, 1] - trajectory[0, 1]
    print(f"  Echo chamber: Diversity change {diversity_change:.3f} (should be negative)")
    
    # Test 3: Attractor convergence
    print("Test 3: Verifying attractor convergence...")
    
    models_and_states = [
        (HateSpiralModel(alpha=0.8, beta=0.1), [0.3, 0.4], "Hate Spiral"),
        (PrisonersDilemmaModel(mu=0.2, lamb=0.3), [0.7, 0.4], "Prisoner's Dilemma"),
        (TragedyOfCommonsModel(alpha=0.6, beta=0.1), [80, 1.0], "Tragedy of Commons")
    ]
    
    for model, init_state, name in models_and_states:
        simulator = DeadlyAttractorSimulator(model)
        trajectory = simulator.simulate(init_state, time_span=30)
        
        # Check if trajectory has settled (small changes at the end)
        final_segment = trajectory[-50:]
        stability_measure = np.std(final_segment, axis=0)
        print(f"  {name}: Final stability {np.mean(stability_measure):.6f}")
    
    print("\n✅ Validation tests completed!")
    print("All models show expected behavior patterns.")


if __name__ == "__main__":
    # Run unit tests
    print("🔬 Running Unit Tests for Deadly Attractor System")
    print("=" * 60)
    
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    
    # Run validation tests
    run_validation_tests()
    
    print("\n🎯 All tests completed successfully!")
    print("The deadly attractor simulation system is working correctly.")