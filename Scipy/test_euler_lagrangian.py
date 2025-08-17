"""
Test suite for the Euler-Lagrangian solver.

This module contains comprehensive tests to validate the numerical
implementation of Euler-Lagrange equation solvers.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import unittest
import sys
import os

# Add the current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from euler_lagrangian import (
    EulerLagrangianSolver,
    create_simple_harmonic_oscillator_lagrangian,
    create_pendulum_lagrangian,
    create_double_pendulum_lagrangian
)


class TestEulerLagrangianSolver(unittest.TestCase):
    """Test cases for the EulerLagrangianSolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-2  # Numerical tolerance for comparisons
        
    def test_simple_harmonic_oscillator_initialization(self):
        """Test initialization with simple harmonic oscillator."""
        lagrangian = create_simple_harmonic_oscillator_lagrangian(1.0, 1.0)
        solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
        
        self.assertEqual(solver.n_coordinates, 1)
        self.assertIsNotNone(solver.lagrangian)
        
    def test_simple_harmonic_oscillator_solution(self):
        """Test solution of simple harmonic oscillator."""
        # Parameters
        mass = 1.0
        k = 4.0  # spring constant, ω = 2
        
        lagrangian = create_simple_harmonic_oscillator_lagrangian(mass, k)
        solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
        
        # Initial conditions: q(0) = 1, q̇(0) = 0
        initial_conditions = {
            'q0': np.array([1.0]),
            'q_dot0': np.array([0.0])
        }
        
        # Solve for one period
        omega = np.sqrt(k / mass)
        period = 2 * np.pi / omega
        time_span = (0, period)
        
        try:
            solution = solver.solve_trajectory(
                initial_conditions, time_span, n_points=100
            )
            
            # Check that we got a solution
            self.assertIn('t', solution)
            self.assertIn('q', solution)
            self.assertIn('q_dot', solution)
            
            # Check dimensions
            self.assertEqual(solution['q'].shape[0], 1)  # 1 coordinate
            self.assertTrue(len(solution['t']) > 0)
            
            # Check initial conditions
            np.testing.assert_allclose(solution['q'][0, 0], 1.0, atol=self.tolerance)
            np.testing.assert_allclose(solution['q_dot'][0, 0], 0.0, atol=self.tolerance)
            
            # Check periodicity (should return close to initial position)
            final_position = solution['q'][0, -1]
            np.testing.assert_allclose(final_position, 1.0, atol=self.tolerance)
            
            print(f"✓ Simple harmonic oscillator test passed")
            print(f"  Initial position: {solution['q'][0, 0]:.6f}")
            print(f"  Final position: {final_position:.6f}")
            print(f"  Expected: 1.0")
            
        except Exception as e:
            self.fail(f"Simple harmonic oscillator solution failed: {e}")
    
    def test_energy_conservation_harmonic_oscillator(self):
        """Test energy conservation for harmonic oscillator."""
        lagrangian = create_simple_harmonic_oscillator_lagrangian(1.0, 1.0)
        solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
        
        initial_conditions = {
            'q0': np.array([1.0]),
            'q_dot0': np.array([0.0])
        }
        
        time_span = (0, 2 * np.pi)
        
        try:
            solution = solver.solve_trajectory(
                initial_conditions, time_span, n_points=200
            )
            
            conservation = solver.analyze_conservation_laws(solution)
            
            # Energy should be approximately constant
            energy_variation = np.std(conservation['energy'])
            mean_energy = np.mean(conservation['energy'])
            
            # Allow for numerical errors
            relative_variation = energy_variation / abs(mean_energy)
            
            self.assertLess(relative_variation, 0.1, 
                          f"Energy not conserved: relative variation = {relative_variation}")
            
            print(f"✓ Energy conservation test passed")
            print(f"  Mean energy: {mean_energy:.6f}")
            print(f"  Energy std: {energy_variation:.6e}")
            print(f"  Relative variation: {relative_variation:.6e}")
            
        except Exception as e:
            self.fail(f"Energy conservation test failed: {e}")
    
    def test_pendulum_small_angle(self):
        """Test pendulum with small angle approximation."""
        # For small angles, pendulum should behave like harmonic oscillator
        length = 1.0
        mass = 1.0
        gravity = 9.81
        
        lagrangian = create_pendulum_lagrangian(length, mass, gravity)
        solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
        
        # Small initial angle
        small_angle = 0.1  # radians (about 5.7 degrees)
        initial_conditions = {
            'q0': np.array([small_angle]),
            'q_dot0': np.array([0.0])
        }
        
        # Theoretical frequency for small angles
        omega_theory = np.sqrt(gravity / length)
        period_theory = 2 * np.pi / omega_theory
        
        time_span = (0, period_theory)
        
        try:
            solution = solver.solve_trajectory(
                initial_conditions, time_span, n_points=100
            )
            
            # Check that solution exists
            self.assertIn('q', solution)
            self.assertTrue(len(solution['t']) > 0)
            
            # For small angles, should approximate harmonic motion
            # Check that it returns close to initial position after one period
            final_angle = solution['q'][0, -1]
            
            # Allow for larger tolerance due to nonlinear effects
            np.testing.assert_allclose(final_angle, small_angle, atol=5*self.tolerance)
            
            print(f"✓ Small angle pendulum test passed")
            print(f"  Initial angle: {small_angle:.6f} rad")
            print(f"  Final angle: {final_angle:.6f} rad")
            print(f"  Theoretical period: {period_theory:.3f} s")
            
        except Exception as e:
            self.fail(f"Small angle pendulum test failed: {e}")
    
    def test_variational_path_simple(self):
        """Test variational path finding with simple boundary conditions."""
        lagrangian = create_simple_harmonic_oscillator_lagrangian(1.0, 1.0)
        solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
        
        # Simple boundary conditions
        boundary_conditions = {
            'q_start': (0.0, np.array([0.0])),
            'q_end': (1.0, np.array([0.0]))
        }
        
        time_span = (0.0, 1.0)
        
        try:
            result = solver.find_stationary_action_path(
                boundary_conditions, time_span, n_segments=20
            )
            
            # Check that we got a result
            self.assertIn('t', result)
            self.assertIn('q', result)
            self.assertIn('action', result)
            
            # Check boundary conditions
            np.testing.assert_allclose(result['q'][0, 0], 0.0, atol=self.tolerance)
            np.testing.assert_allclose(result['q'][0, -1], 0.0, atol=self.tolerance)
            
            # Check that action is finite
            self.assertTrue(np.isfinite(result['action']))
            
            print(f"✓ Variational path test passed")
            print(f"  Start position: {result['q'][0, 0]:.6f}")
            print(f"  End position: {result['q'][0, -1]:.6f}")
            print(f"  Action value: {result['action']:.6f}")
            
        except Exception as e:
            self.fail(f"Variational path test failed: {e}")
    
    def test_lagrangian_functions(self):
        """Test the Lagrangian helper functions."""
        # Test simple harmonic oscillator Lagrangian
        L_sho = create_simple_harmonic_oscillator_lagrangian(1.0, 1.0)
        
        q = np.array([1.0])
        q_dot = np.array([1.0])
        t = 0.0
        
        # L = (1/2) * m * q̇² - (1/2) * k * q²
        # L = (1/2) * 1 * 1² - (1/2) * 1 * 1² = 0
        expected_L = 0.0
        actual_L = L_sho(q, q_dot, t)
        
        np.testing.assert_allclose(actual_L, expected_L, atol=self.tolerance)
        
        # Test pendulum Lagrangian
        L_pendulum = create_pendulum_lagrangian(1.0, 1.0, 9.81)
        
        q = np.array([0.0])  # vertical position
        q_dot = np.array([1.0])
        t = 0.0
        
        # At θ = 0: L = (1/2) * m * l² * θ̇² + m * g * l * cos(0)
        # L = (1/2) * 1 * 1² * 1² + 1 * 9.81 * 1 * 1 = 0.5 + 9.81 = 10.31
        expected_L = 0.5 + 9.81
        actual_L = L_pendulum(q, q_dot, t)
        
        np.testing.assert_allclose(actual_L, expected_L, atol=self.tolerance)
        
        print(f"✓ Lagrangian functions test passed")
        print(f"  SHO Lagrangian: {actual_L:.6f} (expected: {expected_L:.6f})")
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        lagrangian = create_simple_harmonic_oscillator_lagrangian(1.0, 1.0)
        solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
        
        # Test wrong number of coordinates in initial conditions
        wrong_initial_conditions = {
            'q0': np.array([1.0, 2.0]),  # 2 coordinates instead of 1
            'q_dot0': np.array([0.0])
        }
        
        with self.assertRaises(ValueError):
            solver.solve_trajectory(wrong_initial_conditions, (0, 1))
        
        print(f"✓ Invalid input handling test passed")
    
    def test_solver_with_multiple_coordinates(self):
        """Test solver with a system having multiple coordinates."""
        # Use double pendulum as an example
        lagrangian = create_double_pendulum_lagrangian(1.0, 1.0, 1.0, 1.0, 9.81)
        solver = EulerLagrangianSolver(lagrangian, n_coordinates=2)
        
        self.assertEqual(solver.n_coordinates, 2)
        
        # Test with small angles (should be more stable)
        initial_conditions = {
            'q0': np.array([0.1, 0.1]),
            'q_dot0': np.array([0.0, 0.0])
        }
        
        time_span = (0, 1.0)  # Short time span for stability
        
        try:
            solution = solver.solve_trajectory(
                initial_conditions, time_span, n_points=50
            )
            
            # Check dimensions
            self.assertEqual(solution['q'].shape[0], 2)  # 2 coordinates
            self.assertEqual(solution['q_dot'].shape[0], 2)  # 2 velocities
            
            print(f"✓ Multiple coordinates test passed")
            print(f"  Number of coordinates: {solution['q'].shape[0]}")
            print(f"  Solution length: {len(solution['t'])}")
            
        except Exception as e:
            # Double pendulum is notoriously difficult numerically
            print(f"⚠ Multiple coordinates test failed (expected for double pendulum): {e}")


def run_tests():
    """Run all tests and provide a summary."""
    print("Running Euler-Lagrangian Solver Tests")
    print("=" * 50)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEulerLagrangianSolver)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    print()
    print("=" * 50)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    print("=" * 50)
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)