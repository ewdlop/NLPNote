#!/usr/bin/env python3
"""
Test suite for Planck Scale Supersymmetry Calculator

This file contains basic tests to validate the computational 
functionality of the supersymmetry modules.
"""

import unittest
import numpy as np
from PlanckSupersymmetry import PlanckSupersymmetryCalculator

class TestPlanckSupersymmetry(unittest.TestCase):
    """Test cases for PlanckSupersymmetryCalculator"""
    
    def setUp(self):
        """Set up test calculator instance"""
        self.calc = PlanckSupersymmetryCalculator()
    
    def test_planck_units_basic(self):
        """Test that Planck units are calculated correctly"""
        # Check that Planck length is approximately correct
        expected_planck_length = 1.616e-35  # meters
        actual_planck_length = self.calc.planck_units.length
        
        # Allow 1% relative error
        relative_error = abs(actual_planck_length - expected_planck_length) / expected_planck_length
        self.assertLess(relative_error, 0.01, "Planck length calculation error")
        
        # Check Planck energy in GeV
        expected_planck_energy_gev = 1.22e19  # GeV
        actual_planck_energy_gev = self.calc.planck_units.energy_gev
        
        relative_error = abs(actual_planck_energy_gev - expected_planck_energy_gev) / expected_planck_energy_gev
        self.assertLess(relative_error, 0.01, "Planck energy calculation error")
    
    def test_supersymmetric_spectrum(self):
        """Test supersymmetric particle spectrum generation"""
        spectrum = self.calc.supersymmetric_spectrum(n_multiplets=5)
        
        # Graviton should be massless
        self.assertEqual(spectrum.graviton_mass, 0.0)
        
        # Gravitino should have Planck mass scale
        self.assertAlmostEqual(spectrum.gravitino_mass, self.calc.planck_units.mass, places=15)
        
        # Check array sizes
        self.assertEqual(len(spectrum.scalar_masses), 5)
        self.assertEqual(len(spectrum.fermion_masses), 5)
        self.assertEqual(len(spectrum.vector_masses), 2)  # n_multiplets//2
        
        # All masses should be positive
        self.assertTrue(np.all(spectrum.scalar_masses > 0))
        self.assertTrue(np.all(spectrum.fermion_masses > 0))
        self.assertTrue(np.all(spectrum.vector_masses > 0))
    
    def test_superalgebra_relations(self):
        """Test supersymmetric algebra calculations"""
        momentum = np.array([1.0, 0.1, 0.2, 0.3])  # arbitrary four-momentum
        relations = self.calc.superalgebra_relations(momentum)
        
        # Check that result contains expected keys
        self.assertIn('anticommutator', relations)
        self.assertIn('momentum', relations)
        self.assertIn('pauli_matrices', relations)
        
        # Anticommutator should be 2x2 matrix
        anticomm = relations['anticommutator']
        self.assertEqual(anticomm.shape, (2, 2))
        
        # Should be hermitian
        np.testing.assert_array_almost_equal(anticomm, anticomm.conj().T)
    
    def test_bps_bound(self):
        """Test BPS bound calculation"""
        central_charge = 5.0 + 3.0j
        bps_mass = self.calc.bps_bound(central_charge)
        
        expected = abs(central_charge)
        self.assertAlmostEqual(bps_mass, expected, places=10)
        
        # Test with real charge
        real_charge = 7.0
        bps_mass_real = self.calc.bps_bound(real_charge)
        self.assertEqual(bps_mass_real, 7.0)
    
    def test_holomorphic_superpotential(self):
        """Test superpotential calculation"""
        phi = 1.0 + 0.5j  # complex field value
        params = {'m': 1.0, 'lambda': 0.1, 'y': 0.01}
        
        W = self.calc.holomorphic_superpotential(phi, params)
        
        # Result should be complex
        self.assertIsInstance(W, complex)
        
        # Should be analytic (holomorphic)
        # Test with slightly different value
        phi_delta = phi + 1e-10
        W_delta = self.calc.holomorphic_superpotential(phi_delta, params)
        
        # Derivative should exist (finite difference should be reasonable)
        derivative = (W_delta - W) / 1e-10
        self.assertLess(abs(derivative.imag), 1e7)  # Reasonable bound for numerical derivative
    
    def test_f_term_potential(self):
        """Test F-term potential calculation"""
        phi = 1.0 + 0.5j
        params = {'m': 1.0, 'lambda': 0.1, 'y': 0.01}
        
        V_F = self.calc.f_term_potential(phi, params)
        
        # F-term potential should be real and positive
        self.assertIsInstance(V_F, float)
        self.assertGreaterEqual(V_F, 0.0)
    
    def test_gravitino_mass(self):
        """Test gravitino mass calculation"""
        vev_W = 1e-10  # Small superpotential VEV
        m_gravitino = self.calc.gravitino_mass(vev_W)
        
        # Should scale as <W>/M_P^2
        expected = vev_W / self.calc.planck_units.mass**2
        self.assertAlmostEqual(m_gravitino, expected, places=15)
        
        # Should be positive
        self.assertGreater(m_gravitino, 0.0)
    
    def test_black_hole_entropy(self):
        """Test black hole entropy calculation"""
        # Use Planck mass black hole
        bh_mass = self.calc.planck_units.mass
        entropy = self.calc.black_hole_entropy(bh_mass)
        
        # For Planck mass black hole, entropy should be O(1)
        self.assertGreater(entropy, 0.1)
        self.assertLess(entropy, 10.0)
    
    def test_hawking_temperature(self):
        """Test Hawking temperature calculation"""
        bh_mass = self.calc.planck_units.mass
        temp = self.calc.hawking_temperature(bh_mass)
        
        # Should be positive
        self.assertGreater(temp, 0.0)
        
        # For Planck mass, should be order of Planck temperature
        planck_temp = self.calc.planck_units.temperature
        self.assertLess(temp, 10 * planck_temp)
        self.assertGreater(temp, 0.01 * planck_temp)
    
    def test_cosmic_string_tension(self):
        """Test cosmic string tension calculation"""
        eta = 1e16  # GeV (typical GUT scale)
        eta_joules = eta * 1.602e-19 * 1e9  # Convert to joules
        
        tension = self.calc.cosmic_string_tension(eta_joules)
        
        # Should be positive
        self.assertGreater(tension, 0.0)
        
        # Should have correct units (kg/m)
        # Dimensional analysis: [η²/c²] = [energy²/c²] = [mass²·c²/c²] = [mass²]
        # But we want kg/m, so this tests the implementation
        self.assertIsInstance(tension, float)

def run_basic_tests():
    """Run basic functionality tests"""
    print("Running basic tests for Planck Scale Supersymmetry...")
    
    # Create calculator
    calc = PlanckSupersymmetryCalculator()
    
    # Test 1: Basic units
    assert calc.planck_units.length > 0
    assert calc.planck_units.energy > 0
    print("✓ Planck units calculation")
    
    # Test 2: Spectrum generation
    spectrum = calc.supersymmetric_spectrum(3)
    assert spectrum.graviton_mass == 0.0
    assert len(spectrum.scalar_masses) == 3
    print("✓ Supersymmetric spectrum generation")
    
    # Test 3: Algebra relations
    momentum = np.array([1, 0, 0, 0])
    relations = calc.superalgebra_relations(momentum)
    assert relations['anticommutator'].shape == (2, 2)
    print("✓ Supersymmetric algebra relations")
    
    # Test 4: Superpotential
    phi = 1.0 + 0.1j
    params = {'m': 1.0, 'lambda': 0.1, 'y': 0.01}
    W = calc.holomorphic_superpotential(phi, params)
    assert isinstance(W, complex)
    print("✓ Holomorphic superpotential")
    
    print("All basic tests passed! ✓")

if __name__ == "__main__":
    # Run basic tests first
    run_basic_tests()
    
    # Run full unittest suite
    print("\nRunning full test suite...")
    unittest.main(verbosity=2)