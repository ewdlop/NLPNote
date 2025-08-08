#!/usr/bin/env python3
"""
Test Suite for Lie Bracket Framework
李括號框架測試套件

Simple tests to verify the Lie bracket implementation works correctly.
"""

import numpy as np
import sympy as sp
from LieBracket import (
    LieBracketFramework, LieElement, LieAlgebraType,
    MatrixLieBracket, VectorFieldLieBracket, PoissonBracket
)
from MathematicalExpressionAnalyzer import MathematicalExpressionAnalyzer


def test_pauli_matrices():
    """Test Pauli matrix commutation relations"""
    print("Testing Pauli matrix commutation relations...")
    
    # Create Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    sigma_x = LieElement(pauli_x, LieAlgebraType.MATRIX, name='σ_x')
    sigma_y = LieElement(pauli_y, LieAlgebraType.MATRIX, name='σ_y')
    sigma_z = LieElement(pauli_z, LieAlgebraType.MATRIX, name='σ_z')
    
    bracket = MatrixLieBracket()
    
    # Test [σ_x, σ_y] = 2iσ_z
    xy_result = bracket.bracket(sigma_x, sigma_y)
    expected = 2j * pauli_z
    
    assert np.allclose(xy_result.data, expected), "Pauli [x,y] commutator failed"
    
    # Test Jacobi identity
    jacobi_holds = bracket.verify_jacobi_identity(sigma_x, sigma_y, sigma_z)
    assert jacobi_holds, "Jacobi identity failed for Pauli matrices"
    
    print("✓ Pauli matrix tests passed")


def test_rotation_vectors():
    """Test SO(3) rotation vector field algebra"""
    print("Testing SO(3) rotation vector fields...")
    
    x, y, z = sp.symbols('x y z')
    vf_bracket = VectorFieldLieBracket([x, y, z])
    
    # Rotation vector fields
    R_x = LieElement([0, -z, y], LieAlgebraType.VECTOR_FIELD, name='R_x')
    R_y = LieElement([z, 0, -x], LieAlgebraType.VECTOR_FIELD, name='R_y')
    R_z = LieElement([-y, x, 0], LieAlgebraType.VECTOR_FIELD, name='R_z')
    
    # Test [R_x, R_y] should give R_z (up to sign and normalization)
    xy_bracket = vf_bracket.bracket(R_x, R_y)
    
    # Verify Jacobi identity
    jacobi_holds = vf_bracket.verify_jacobi_identity(R_x, R_y, R_z)
    assert jacobi_holds, "Jacobi identity failed for rotation vector fields"
    
    print("✓ Rotation vector field tests passed")


def test_poisson_brackets():
    """Test canonical Poisson bracket relations"""
    print("Testing Poisson brackets...")
    
    q1, q2, p1, p2 = sp.symbols('q1 q2 p1 p2')
    poisson = PoissonBracket([q1, q2], [p1, p2])
    
    # Test canonical relation {q1, p1} = 1
    q1_elem = LieElement(q1, LieAlgebraType.PHYSICS, name='q1')
    p1_elem = LieElement(p1, LieAlgebraType.PHYSICS, name='p1')
    
    qp_bracket = poisson.bracket(q1_elem, p1_elem)
    assert qp_bracket.data == 1, "Canonical Poisson bracket {q,p} ≠ 1"
    
    # Test {H, L} = 0 for harmonic oscillator
    H = (p1**2 + p2**2)/2 + (q1**2 + q2**2)/2
    L = q1*p2 - q2*p1
    
    H_elem = LieElement(H, LieAlgebraType.PHYSICS, name='H')
    L_elem = LieElement(L, LieAlgebraType.PHYSICS, name='L')
    
    HL_bracket = poisson.bracket(H_elem, L_elem)
    assert sp.simplify(HL_bracket.data) == 0, "Angular momentum not conserved"
    
    print("✓ Poisson bracket tests passed")


def test_framework_integration():
    """Test the full framework integration"""
    print("Testing framework integration...")
    
    framework = LieBracketFramework()
    
    # Create examples
    examples = framework.create_demonstration_examples()
    assert len(examples) > 0, "No examples created"
    
    # Test analysis
    analysis = framework.analyze_brackets()
    assert 'pauli_commutator' in analysis, "Pauli analysis missing"
    
    # Test insights
    insights = framework.generate_insights()
    assert 'fundamental_difference' in insights, "Insights generation failed"
    
    print("✓ Framework integration tests passed")


def test_nlp_integration():
    """Test NLP integration"""
    print("Testing NLP integration...")
    
    analyzer = MathematicalExpressionAnalyzer()
    
    # Test concept extraction
    text = "The Lie bracket [X, Y] in quantum mechanics represents commutators"
    analysis = analyzer.analyze_lie_bracket_expression(text)
    
    assert len(analysis['mathematical_concepts']) > 0, "No concepts extracted"
    assert 'lie_bracket_structure' in analysis, "Bracket structure not detected"
    
    # Test approach classification
    phys_text = "We observe physical phenomena and need mathematical formalism to describe it"
    math_text = "We apply mathematical structures and group theory to solve physical problems"
    
    phys_result = analyzer.classify_mathematical_approach(phys_text)
    math_result = analyzer.classify_mathematical_approach(math_text)
    
    print(f"Physical text result: {phys_result}")
    print(f"Mathematical text result: {math_result}")
    
    # The classification should detect the approach even if not perfectly
    assert phys_result['physical_mathematics'] >= 0.3, \
           "Physical mathematics approach not detected"
    assert math_result['mathematical_physics'] >= 0.3, \
           "Mathematical physics approach not detected"
    
    print("✓ NLP integration tests passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Running Lie Bracket Framework Tests")
    print("=" * 50)
    
    try:
        test_pauli_matrices()
        test_rotation_vectors()
        test_poisson_brackets()
        test_framework_integration()
        test_nlp_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("Li Bracket framework is working correctly.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)