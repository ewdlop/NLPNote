#!/usr/bin/env python3
"""
Test suite for Malament-Hogarth computational complexity concepts

This module provides tests to validate the mathematical concepts and
implementations in the MH complexity framework.
"""

import sys
import os
import math

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mh_complexity_simulator import (
    MHTuringMachine, MHComplexityAnalyzer, SpacetimeType, 
    MHComplexityClass, SimpleMatrix
)


def test_simple_matrix():
    """Test the SimpleMatrix implementation"""
    print("Testing SimpleMatrix...")
    
    # Create matrix
    m = SimpleMatrix(3, 3)
    
    # Set values
    m[0, 0] = 1.0
    m[1, 1] = 2.0
    m[2, 2] = 3.0
    
    # Check values
    assert m[0, 0] == 1.0
    assert m[1, 1] == 2.0
    assert m[2, 2] == 3.0
    
    # Test zeros creation
    z = SimpleMatrix.zeros(2, 2)
    assert z[0, 0] == 0.0
    assert z[1, 1] == 0.0
    
    print("✓ SimpleMatrix tests passed")


def test_spacetime_creation():
    """Test MH spacetime creation and basic operations"""
    print("Testing MH spacetime creation...")
    
    analyzer = MHComplexityAnalyzer()
    spacetime = analyzer._create_sample_spacetime()
    
    # Check spacetime properties
    assert spacetime.spacetime_type == SpacetimeType.ANTI_DE_SITTER
    assert 'ads_radius' in spacetime.physical_parameters
    assert spacetime.causal_structure_check() == True
    
    # Test worldline trajectories
    obs_pos = spacetime.observer_worldline.position_at_time(0.5)
    comp_pos = spacetime.computer_worldline.position_at_time(0.5)
    
    assert len(obs_pos) == 4  # 4D spacetime coordinates
    assert len(comp_pos) == 4
    
    # Test proper time calculations
    obs_tau = spacetime.observer_worldline.proper_time_at_coordinate_time(0.5)
    comp_tau = spacetime.computer_worldline.proper_time_at_coordinate_time(0.5)
    
    assert obs_tau > 0
    assert comp_tau > 0
    
    print("✓ MH spacetime tests passed")


def test_complexity_classes():
    """Test MH complexity class computations"""
    print("Testing MH complexity classes...")
    
    analyzer = MHComplexityAnalyzer()
    spacetime = analyzer._create_sample_spacetime()
    mh_machine = MHTuringMachine(spacetime)
    
    # Test MH-P computation
    result_p = mh_machine.simulate_computation(7, MHComplexityClass.MH_P)
    assert result_p.complexity_class == MHComplexityClass.MH_P
    assert result_p.observer_time < 1.0  # Should be constant
    assert result_p.convergence_achieved == True
    
    # Test MH-NP computation
    result_np = mh_machine.simulate_computation("test", MHComplexityClass.MH_NP)
    assert result_np.complexity_class == MHComplexityClass.MH_NP
    assert result_np.observer_time < 1.0  # Should be constant
    assert result_np.computer_time >= result_p.computer_time  # NP should take longer
    
    # Test MH-Decidable computation
    result_dec = mh_machine.simulate_computation(42, MHComplexityClass.MH_DECIDABLE)
    assert result_dec.complexity_class == MHComplexityClass.MH_DECIDABLE
    assert result_dec.observer_time < 1.0  # Should be constant
    assert result_dec.computer_time == float('inf')  # Infinite computer time
    
    print("✓ MH complexity class tests passed")


def test_comparative_analysis():
    """Test comparative analysis between classical and MH complexity"""
    print("Testing comparative analysis...")
    
    analyzer = MHComplexityAnalyzer()
    
    # Test problems
    problems = [5, "ab", 10]
    problem_types = [
        MHComplexityClass.MH_P,
        MHComplexityClass.MH_NP,
        MHComplexityClass.MH_DECIDABLE
    ]
    
    results = analyzer.compare_classical_vs_mh(problems, problem_types)
    
    # Check result structure
    assert 'classical_times' in results
    assert 'mh_observer_times' in results
    assert 'mh_computer_times' in results
    assert 'input_sizes' in results
    
    # Check result lengths
    assert len(results['classical_times']) == len(problems)
    assert len(results['mh_observer_times']) == len(problems)
    
    # Check that MH observer times are bounded
    for obs_time in results['mh_observer_times']:
        assert obs_time < 1.0
    
    # Check that undecidable problems have infinite classical time
    undecidable_indices = [i for i, pt in enumerate(problem_types) 
                          if pt == MHComplexityClass.MH_DECIDABLE]
    for i in undecidable_indices:
        assert results['classical_times'][i] == float('inf')
    
    print("✓ Comparative analysis tests passed")


def test_mathematical_properties():
    """Test mathematical properties of MH spacetime"""
    print("Testing mathematical properties...")
    
    analyzer = MHComplexityAnalyzer()
    spacetime = analyzer._create_sample_spacetime()
    
    # Test metric properties
    test_coords = (0.5, 1.0, math.pi/2, 0.0)
    metric = spacetime.metric(test_coords)
    
    # Metric should be 4x4
    assert metric.rows == 4
    assert metric.cols == 4
    
    # Time component should be negative (signature convention)
    assert metric[0, 0] < 0
    
    # Spatial components should be positive
    assert metric[1, 1] > 0
    assert metric[2, 2] > 0
    assert metric[3, 3] > 0
    
    # Test time dilation calculation
    time_dilation = spacetime.compute_time_dilation(0.5)
    assert time_dilation > 0
    assert math.isfinite(time_dilation)
    
    print("✓ Mathematical properties tests passed")


def test_theoretical_implications():
    """Test theoretical implications of MH computation"""
    print("Testing theoretical implications...")
    
    analyzer = MHComplexityAnalyzer()
    
    # Create test scenarios
    problems = [10, "test", 42]
    types = [MHComplexityClass.MH_P, MHComplexityClass.MH_NP, MHComplexityClass.MH_DECIDABLE]
    
    results = analyzer.compare_classical_vs_mh(problems, types)
    
    # Test fundamental MH properties:
    
    # 1. Observer time should be bounded for all problems
    max_observer_time = max(results['mh_observer_times'])
    assert max_observer_time < float('inf')
    
    # 2. Classical undecidable problems should be MH-decidable
    classical_undecidable = [i for i, t in enumerate(results['classical_times']) 
                           if t == float('inf')]
    for i in classical_undecidable:
        assert results['mh_observer_times'][i] < float('inf')
    
    # 3. MH should provide speedup for complex problems
    classical_finite = [t for t in results['classical_times'] if t < float('inf')]
    mh_finite = [results['mh_observer_times'][i] for i, t in enumerate(results['classical_times']) 
                 if t < float('inf')]
    
    if classical_finite and mh_finite:
        avg_classical = sum(classical_finite) / len(classical_finite)
        avg_mh = sum(mh_finite) / len(mh_finite)
        assert avg_mh <= avg_classical  # MH should be faster or equal
    
    print("✓ Theoretical implications tests passed")


def test_report_generation():
    """Test report generation functionality"""
    print("Testing report generation...")
    
    analyzer = MHComplexityAnalyzer()
    
    # Run some computations first
    problems = [5, "test"]
    types = [MHComplexityClass.MH_P, MHComplexityClass.MH_NP]
    analyzer.compare_classical_vs_mh(problems, types)
    
    # Generate report
    report = analyzer.generate_complexity_report()
    
    # Check report content
    assert "MALAMENT-HOGARTH COMPUTATIONAL COMPLEXITY REPORT" in report
    assert "Total Computations Analyzed:" in report
    assert "COMPLEXITY CLASS DISTRIBUTION:" in report
    assert "THEORETICAL IMPLICATIONS:" in report
    assert "PHYSICAL CONSTRAINTS:" in report
    
    # Check that specific theoretical points are mentioned
    assert "constant observer time" in report
    assert "undecidable problems become decidable" in report
    assert "P = NP in MH observer reference frame" in report
    
    print("✓ Report generation tests passed")


def run_all_tests():
    """Run all test suites"""
    print("=" * 60)
    print("MALAMENT-HOGARTH COMPLEXITY FRAMEWORK TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_simple_matrix,
        test_spacetime_creation,
        test_complexity_classes,
        test_comparative_analysis,
        test_mathematical_properties,
        test_theoretical_implications,
        test_report_generation
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed_tests.append(test.__name__)
    
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(tests)
    passed_tests = total_tests - len(failed_tests)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
        return False
    else:
        print("All tests passed! ✓")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)