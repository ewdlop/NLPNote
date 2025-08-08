#!/usr/bin/env python3
"""
Lie Bracket Demonstration: Physical Mathematics vs Mathematical Physics
李括號演示：物理數學 vs 數學物理

This script demonstrates the computational Lie bracket framework and explores
the relationship between physical mathematics and mathematical physics.

這個腳本演示計算李括號框架，並探索物理數學與數學物理之間的關係。
"""

import numpy as np
import sympy as sp
from LieBracket import (
    LieBracketFramework, LieElement, LieAlgebraType,
    MatrixLieBracket, VectorFieldLieBracket, PoissonBracket
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")


def demonstrate_matrix_lie_brackets():
    """Demonstrate matrix Lie brackets with Pauli matrices"""
    print_section("Matrix Lie Brackets: Pauli Matrices (Physical Mathematics)")
    
    # Create Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Create LieElements
    sigma_x = LieElement(
        data=pauli_x,
        algebra_type=LieAlgebraType.MATRIX,
        name='σ_x',
        physical_interpretation='X-component of spin operator'
    )
    
    sigma_y = LieElement(
        data=pauli_y,
        algebra_type=LieAlgebraType.MATRIX,
        name='σ_y',
        physical_interpretation='Y-component of spin operator'
    )
    
    sigma_z = LieElement(
        data=pauli_z,
        algebra_type=LieAlgebraType.MATRIX,
        name='σ_z',
        physical_interpretation='Z-component of spin operator'
    )
    
    print("Pauli Matrices:")
    print(f"σ_x = \n{pauli_x}")
    print(f"σ_y = \n{pauli_y}")
    print(f"σ_z = \n{pauli_z}")
    
    # Compute commutators
    matrix_bracket = MatrixLieBracket()
    
    print_subsection("Commutation Relations")
    
    # [σ_x, σ_y] = 2iσ_z
    xy_commutator = matrix_bracket.bracket(sigma_x, sigma_y)
    print(f"[σ_x, σ_y] = \n{xy_commutator.data}")
    print(f"Expected: 2i * σ_z = \n{2j * pauli_z}")
    print(f"Verification: {np.allclose(xy_commutator.data, 2j * pauli_z)}")
    
    # [σ_y, σ_z] = 2iσ_x
    yz_commutator = matrix_bracket.bracket(sigma_y, sigma_z)
    print(f"\n[σ_y, σ_z] = \n{yz_commutator.data}")
    print(f"Expected: 2i * σ_x = \n{2j * pauli_x}")
    print(f"Verification: {np.allclose(yz_commutator.data, 2j * pauli_x)}")
    
    # [σ_z, σ_x] = 2iσ_y
    zx_commutator = matrix_bracket.bracket(sigma_z, sigma_x)
    print(f"\n[σ_z, σ_x] = \n{zx_commutator.data}")
    print(f"Expected: 2i * σ_y = \n{2j * pauli_y}")
    print(f"Verification: {np.allclose(zx_commutator.data, 2j * pauli_y)}")
    
    # Verify Jacobi identity
    print_subsection("Jacobi Identity Verification")
    jacobi_holds = matrix_bracket.verify_jacobi_identity(sigma_x, sigma_y, sigma_z)
    print(f"Jacobi identity [σ_x, [σ_y, σ_z]] + [σ_y, [σ_z, σ_x]] + [σ_z, [σ_x, σ_y]] = 0: {jacobi_holds}")
    
    print_subsection("Physical Interpretation")
    print("These commutation relations encode the fundamental fact that")
    print("spin measurements in different directions cannot be performed")
    print("simultaneously with perfect precision (Heisenberg uncertainty principle).")
    print("This is PHYSICAL MATHEMATICS: starting from quantum phenomena,")
    print("we derive the mathematical structure of SU(2) Lie algebra.")


def demonstrate_vector_field_brackets():
    """Demonstrate vector field Lie brackets"""
    print_section("Vector Field Lie Brackets (Mathematical Physics)")
    
    # Define coordinate system
    x, y, z = sp.symbols('x y z')
    coordinates = [x, y, z]
    
    # Create vector field operator
    vf_bracket = VectorFieldLieBracket(coordinates)
    
    print("Coordinate system: (x, y, z)")
    
    # Define rotation vector fields
    # R_x: rotation around x-axis
    R_x_field = [0, -z, y]
    R_x = LieElement(
        data=R_x_field,
        algebra_type=LieAlgebraType.VECTOR_FIELD,
        name='R_x',
        physical_interpretation='Infinitesimal rotation around x-axis'
    )
    
    # R_y: rotation around y-axis  
    R_y_field = [z, 0, -x]
    R_y = LieElement(
        data=R_y_field,
        algebra_type=LieAlgebraType.VECTOR_FIELD,
        name='R_y',
        physical_interpretation='Infinitesimal rotation around y-axis'
    )
    
    # R_z: rotation around z-axis
    R_z_field = [-y, x, 0]
    R_z = LieElement(
        data=R_z_field,
        algebra_type=LieAlgebraType.VECTOR_FIELD,
        name='R_z',
        physical_interpretation='Infinitesimal rotation around z-axis'
    )
    
    print("Rotation vector fields:")
    print(f"R_x = {R_x_field} (rotation around x-axis)")
    print(f"R_y = {R_y_field} (rotation around y-axis)")
    print(f"R_z = {R_z_field} (rotation around z-axis)")
    
    print_subsection("Lie Bracket Computations")
    
    # [R_x, R_y] = R_z
    xy_bracket = vf_bracket.bracket(R_x, R_y)
    print(f"[R_x, R_y] = {xy_bracket.data}")
    print(f"Expected: R_z = {R_z_field}")
    
    # [R_y, R_z] = R_x
    yz_bracket = vf_bracket.bracket(R_y, R_z)
    print(f"[R_y, R_z] = {yz_bracket.data}")
    print(f"Expected: R_x = {R_x_field}")
    
    # [R_z, R_x] = R_y
    zx_bracket = vf_bracket.bracket(R_z, R_x)
    print(f"[R_z, R_x] = {zx_bracket.data}")
    print(f"Expected: R_y = {R_y_field}")
    
    # Verify Jacobi identity
    print_subsection("Jacobi Identity Verification")
    jacobi_holds = vf_bracket.verify_jacobi_identity(R_x, R_y, R_z)
    print(f"Jacobi identity holds: {jacobi_holds}")
    
    print_subsection("Mathematical Interpretation")
    print("This demonstrates the SO(3) Lie algebra structure of rotations.")
    print("This is MATHEMATICAL PHYSICS: starting from the mathematical")
    print("structure of the rotation group SO(3), we find its applications")
    print("in describing physical rotational symmetries.")


def demonstrate_poisson_brackets():
    """Demonstrate Poisson brackets in Hamiltonian mechanics"""
    print_section("Poisson Brackets: Hamiltonian Mechanics (Physical Mathematics)")
    
    # Define canonical coordinates
    q1, q2 = sp.symbols('q1 q2')  # positions
    p1, p2 = sp.symbols('p1 p2')  # momenta
    
    # Create Poisson bracket operator
    poisson = PoissonBracket([q1, q2], [p1, p2])
    
    print("Canonical coordinates: q1, q2 (positions), p1, p2 (momenta)")
    
    # Define physical quantities
    # Hamiltonian for 2D harmonic oscillator
    H = (p1**2 + p2**2)/2 + (q1**2 + q2**2)/2
    hamiltonian = LieElement(
        data=H,
        algebra_type=LieAlgebraType.PHYSICS,
        name='H',
        physical_interpretation='2D harmonic oscillator Hamiltonian'
    )
    
    # Angular momentum
    L = q1*p2 - q2*p1
    angular_momentum = LieElement(
        data=L,
        algebra_type=LieAlgebraType.PHYSICS,
        name='L',
        physical_interpretation='Angular momentum'
    )
    
    # Position and momentum components
    q1_elem = LieElement(data=q1, algebra_type=LieAlgebraType.PHYSICS, name='q1')
    p1_elem = LieElement(data=p1, algebra_type=LieAlgebraType.PHYSICS, name='p1')
    
    print(f"Hamiltonian H = {H}")
    print(f"Angular momentum L = {L}")
    
    print_subsection("Canonical Poisson Brackets")
    
    # {q1, p1} = 1
    q1_p1_bracket = poisson.bracket(q1_elem, p1_elem)
    print(f"{{q1, p1}} = {q1_p1_bracket.data}")
    
    print_subsection("Conservation Laws")
    
    # {H, L} for isotropic harmonic oscillator
    HL_bracket = poisson.bracket(hamiltonian, angular_momentum)
    print(f"{{H, L}} = {sp.simplify(HL_bracket.data)}")
    print("Since {H, L} = 0, angular momentum is conserved for the isotropic harmonic oscillator")
    
    print_subsection("Physical Interpretation")
    print("Poisson brackets encode the time evolution of physical quantities:")
    print("dF/dt = {F, H} + ∂F/∂t")
    print("This is PHYSICAL MATHEMATICS: starting from Hamilton's equations")
    print("and the need to describe classical mechanics, we arrive at the")
    print("mathematical structure of symplectic geometry.")


def demonstrate_framework_integration():
    """Demonstrate the full framework integration"""
    print_section("Framework Integration: Physical Math vs Mathematical Physics")
    
    framework = LieBracketFramework()
    
    # Create examples
    examples = framework.create_demonstration_examples()
    print(f"Created {len(examples)} demonstration examples")
    
    # Analyze the philosophical difference
    comparison = framework.demonstrate_physical_vs_mathematical()
    
    print_subsection("Physical Mathematics Approach")
    phys_math = comparison['physical_mathematics']
    print(f"Approach: {phys_math['approach']}")
    print(f"Example: {phys_math['example']}")
    print(f"Reasoning: {phys_math['reasoning']}")
    print(f"Mathematics Used: {phys_math['mathematics_used']}")
    print(f"Outcome: {phys_math['outcome']}")
    
    print_subsection("Mathematical Physics Approach")
    math_phys = comparison['mathematical_physics']
    print(f"Approach: {math_phys['approach']}")
    print(f"Example: {math_phys['example']}")
    print(f"Reasoning: {math_phys['reasoning']}")
    print(f"Physics Application: {math_phys['physics_application']}")
    print(f"Outcome: {math_phys['outcome']}")
    
    print_subsection("The Lie Bracket Insight")
    insight = comparison['lie_bracket_insight']
    print(f"Formula: {insight['formula']}")
    print(f"Interpretation: {insight['interpretation']}")
    print(f"Philosophical Meaning: {insight['philosophical_meaning']}")
    print(f"Synthesis: {insight['synthesis']}")
    
    # Generate insights
    insights = framework.generate_insights()
    
    print_subsection("Key Insights")
    for key, insight in insights.items():
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"  {insight}")


def main():
    """Main demonstration function"""
    print_section("Lie Bracket Computational Framework")
    print("Exploring: Physical Mathematics - Mathematical Physics = ?")
    print("用李括號探索：物理數學 - 數學物理 = ？")
    
    try:
        # Demonstrate different types of Lie brackets
        demonstrate_matrix_lie_brackets()
        demonstrate_vector_field_brackets()
        demonstrate_poisson_brackets()
        
        # Show framework integration
        demonstrate_framework_integration()
        
        print_section("Conclusion")
        print("The Lie bracket reveals the fundamental non-commutativity between")
        print("physical mathematics and mathematical physics approaches.")
        print("")
        print("Physical Mathematics: Phenomena → Mathematical Structure")
        print("Mathematical Physics: Mathematical Structure → Physical Applications")
        print("")
        print("The 'difference' [Physical_Math, Mathematical_Physics] ≠ 0")
        print("shows they are complementary, not opposing approaches.")
        print("")
        print("Complete understanding requires both perspectives working together.")
        print("李括號揭示了物理數學與數學物理方法之間的根本非交換性。")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()