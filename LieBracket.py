"""
Lie Bracket Computational Framework
計算李括號框架

This module implements computational Lie brackets to explore the relationship between
physical mathematics and mathematical physics, addressing the question:
"physical mathematics - mathematical physics = ?"

一個李括號計算框架，用於探索物理數學與數學物理之間的關係，
回答問題："物理數學 - 數學物理 = ？"

Lie brackets are fundamental operations in:
- Lie algebras (abstract algebra)
- Differential geometry (vector fields)
- Physics (commutation relations, Poisson brackets)
- Quantum mechanics (commutators)
"""

import numpy as np
import sympy as sp
from typing import Union, List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class LieAlgebraType(Enum):
    """Types of Lie algebras"""
    MATRIX = "matrix"           # Matrix Lie algebras (e.g., sl(n), so(n))
    VECTOR_FIELD = "vector_field"  # Vector field Lie algebras
    ABSTRACT = "abstract"       # Abstract Lie algebras
    PHYSICS = "physics"         # Physics-oriented (commutators, Poisson)


@dataclass
class LieElement:
    """
    Represents an element in a Lie algebra
    李代數中的元素
    """
    data: Union[np.ndarray, sp.Matrix, sp.Expr, str]
    algebra_type: LieAlgebraType
    name: Optional[str] = None
    physical_interpretation: Optional[str] = None
    mathematical_properties: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.mathematical_properties is None:
            self.mathematical_properties = {}


class LieBracketOperator(ABC):
    """
    Abstract base class for Lie bracket operations
    李括號運算的抽象基類
    """
    
    @abstractmethod
    def bracket(self, x: LieElement, y: LieElement) -> LieElement:
        """
        Compute the Lie bracket [x, y]
        計算李括號 [x, y]
        """
        pass
    
    @abstractmethod
    def verify_jacobi_identity(self, x: LieElement, y: LieElement, z: LieElement) -> bool:
        """
        Verify the Jacobi identity: [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0
        驗證雅可比恆等式
        """
        pass


class MatrixLieBracket(LieBracketOperator):
    """
    Matrix Lie bracket: [A, B] = AB - BA (commutator)
    矩陣李括號：交換子
    """
    
    def bracket(self, x: LieElement, y: LieElement) -> LieElement:
        """Compute matrix commutator [A, B] = AB - BA"""
        if x.algebra_type != LieAlgebraType.MATRIX or y.algebra_type != LieAlgebraType.MATRIX:
            raise ValueError("Both elements must be matrix type for matrix Lie bracket")
        
        A = x.data if isinstance(x.data, np.ndarray) else np.array(x.data)
        B = y.data if isinstance(y.data, np.ndarray) else np.array(y.data)
        
        commutator = A @ B - B @ A
        
        return LieElement(
            data=commutator,
            algebra_type=LieAlgebraType.MATRIX,
            name=f"[{x.name or 'X'}, {y.name or 'Y'}]",
            physical_interpretation=f"Commutator of {x.physical_interpretation or 'operator X'} and {y.physical_interpretation or 'operator Y'}",
            mathematical_properties={
                'is_commutator': True,
                'parent_elements': (x.name, y.name)
            }
        )
    
    def verify_jacobi_identity(self, x: LieElement, y: LieElement, z: LieElement) -> bool:
        """Verify Jacobi identity for matrix Lie algebra"""
        try:
            xy_z = self.bracket(self.bracket(x, y), z)
            yz_x = self.bracket(self.bracket(y, z), x)
            zx_y = self.bracket(self.bracket(z, x), y)
            
            total = xy_z.data + yz_x.data + zx_y.data
            return np.allclose(total, np.zeros_like(total), atol=1e-10)
        except Exception:
            return False


class VectorFieldLieBracket(LieBracketOperator):
    """
    Vector field Lie bracket for differential geometry
    微分幾何中的向量場李括號
    """
    
    def __init__(self, coordinate_system: List[sp.Symbol]):
        """
        Initialize with coordinate system
        用坐標系統初始化
        """
        self.coordinates = coordinate_system
        self.dim = len(coordinate_system)
    
    def bracket(self, x: LieElement, y: LieElement) -> LieElement:
        """
        Compute Lie bracket of vector fields
        計算向量場的李括號
        """
        if x.algebra_type != LieAlgebraType.VECTOR_FIELD or y.algebra_type != LieAlgebraType.VECTOR_FIELD:
            raise ValueError("Both elements must be vector field type")
        
        # Vector fields are represented as lists of symbolic expressions
        X = x.data  # [X^1, X^2, ..., X^n]
        Y = y.data  # [Y^1, Y^2, ..., Y^n]
        
        # Lie bracket [X, Y]^i = X^j ∂Y^i/∂x^j - Y^j ∂X^i/∂x^j
        bracket_components = []
        
        for i in range(self.dim):
            component = 0
            for j in range(self.dim):
                # X^j ∂Y^i/∂x^j
                component += X[j] * sp.diff(Y[i], self.coordinates[j])
                # - Y^j ∂X^i/∂x^j
                component -= Y[j] * sp.diff(X[i], self.coordinates[j])
            
            bracket_components.append(sp.simplify(component))
        
        return LieElement(
            data=bracket_components,
            algebra_type=LieAlgebraType.VECTOR_FIELD,
            name=f"[{x.name or 'X'}, {y.name or 'Y'}]",
            physical_interpretation=f"Lie bracket of vector fields {x.name} and {y.name}",
            mathematical_properties={
                'coordinate_system': self.coordinates,
                'dimension': self.dim
            }
        )
    
    def verify_jacobi_identity(self, x: LieElement, y: LieElement, z: LieElement) -> bool:
        """Verify Jacobi identity for vector fields"""
        try:
            xy_z = self.bracket(self.bracket(x, y), z)
            yz_x = self.bracket(self.bracket(y, z), x)
            zx_y = self.bracket(self.bracket(z, x), y)
            
            # Sum the components
            for i in range(self.dim):
                total_component = xy_z.data[i] + yz_x.data[i] + zx_y.data[i]
                simplified = sp.simplify(total_component)
                if simplified != 0:
                    return False
            return True
        except Exception:
            return False


class PoissonBracket(LieBracketOperator):
    """
    Poisson bracket for Hamiltonian mechanics
    哈密頓力學中的泊松括號
    """
    
    def __init__(self, coordinates: List[sp.Symbol], momenta: List[sp.Symbol]):
        """
        Initialize with canonical coordinates and momenta
        用正則坐標和動量初始化
        """
        if len(coordinates) != len(momenta):
            raise ValueError("Number of coordinates must equal number of momenta")
        
        self.q = coordinates  # Position coordinates
        self.p = momenta      # Momentum coordinates
        self.dim = len(coordinates)
    
    def bracket(self, x: LieElement, y: LieElement) -> LieElement:
        """
        Compute Poisson bracket {f, g} = Σ(∂f/∂q_i * ∂g/∂p_i - ∂f/∂p_i * ∂g/∂q_i)
        計算泊松括號
        """
        if x.algebra_type != LieAlgebraType.PHYSICS or y.algebra_type != LieAlgebraType.PHYSICS:
            raise ValueError("Both elements must be physics type for Poisson bracket")
        
        f = x.data  # Function f(q, p)
        g = y.data  # Function g(q, p)
        
        poisson_bracket = 0
        for i in range(self.dim):
            # ∂f/∂q_i * ∂g/∂p_i - ∂f/∂p_i * ∂g/∂q_i
            poisson_bracket += (sp.diff(f, self.q[i]) * sp.diff(g, self.p[i]) - 
                              sp.diff(f, self.p[i]) * sp.diff(g, self.q[i]))
        
        poisson_bracket = sp.simplify(poisson_bracket)
        
        return LieElement(
            data=poisson_bracket,
            algebra_type=LieAlgebraType.PHYSICS,
            name=f"{{{x.name or 'f'}, {y.name or 'g'}}}",
            physical_interpretation=f"Poisson bracket of {x.physical_interpretation or 'function f'} and {y.physical_interpretation or 'function g'}",
            mathematical_properties={
                'coordinates': self.q,
                'momenta': self.p,
                'dimension': self.dim
            }
        )
    
    def verify_jacobi_identity(self, x: LieElement, y: LieElement, z: LieElement) -> bool:
        """Verify Jacobi identity for Poisson bracket"""
        try:
            xy_z = self.bracket(self.bracket(x, y), z)
            yz_x = self.bracket(self.bracket(y, z), x)
            zx_y = self.bracket(self.bracket(z, x), y)
            
            total = xy_z.data + yz_x.data + zx_y.data
            simplified = sp.simplify(total)
            return simplified == 0
        except Exception:
            return False


class LieBracketFramework:
    """
    Main framework for exploring physical mathematics vs mathematical physics
    探索物理數學與數學物理關係的主要框架
    """
    
    def __init__(self):
        self.operators = {
            LieAlgebraType.MATRIX: MatrixLieBracket(),
            LieAlgebraType.VECTOR_FIELD: None,  # Needs coordinate system
            LieAlgebraType.PHYSICS: None        # Needs coordinate system
        }
        self.examples = {}
        self.analysis_results = {}
    
    def create_vector_field_operator(self, coordinates: List[sp.Symbol]) -> VectorFieldLieBracket:
        """Create vector field Lie bracket operator"""
        operator = VectorFieldLieBracket(coordinates)
        self.operators[LieAlgebraType.VECTOR_FIELD] = operator
        return operator
    
    def create_poisson_operator(self, coordinates: List[sp.Symbol], momenta: List[sp.Symbol]) -> PoissonBracket:
        """Create Poisson bracket operator"""
        operator = PoissonBracket(coordinates, momenta)
        self.operators[LieAlgebraType.PHYSICS] = operator
        return operator
    
    def compute_bracket(self, x: LieElement, y: LieElement) -> LieElement:
        """
        Compute Lie bracket based on algebra type
        根據代數類型計算李括號
        """
        algebra_type = x.algebra_type
        if algebra_type != y.algebra_type:
            raise ValueError("Both elements must be of the same algebra type")
        
        operator = self.operators.get(algebra_type)
        if operator is None:
            raise ValueError(f"No operator defined for algebra type {algebra_type}")
        
        return operator.bracket(x, y)
    
    def demonstrate_physical_vs_mathematical(self) -> Dict[str, Any]:
        """
        Demonstrate the difference between physical mathematics and mathematical physics
        展示物理數學與數學物理的差異
        
        Physical Mathematics: Start with physical principles, use math as a tool
        Mathematical Physics: Start with mathematical structures, find physical applications
        """
        results = {}
        
        # Example 1: Quantum mechanics - Physical Mathematics approach
        # Start with physical concept: electron spin
        results['physical_mathematics'] = {
            'approach': 'Start with physical phenomena',
            'example': 'Electron spin',
            'reasoning': 'Observe that particles have intrinsic angular momentum → need mathematical structure to describe it',
            'mathematics_used': 'SU(2) Lie algebra, Pauli matrices',
            'outcome': 'Derive mathematical formalism from physical requirements'
        }
        
        # Example 2: Lie groups - Mathematical Physics approach  
        # Start with mathematical structure: SO(3) group
        results['mathematical_physics'] = {
            'approach': 'Start with mathematical structures',
            'example': 'SO(3) rotation group',
            'reasoning': 'Study properties of SO(3) Lie group → find it describes spatial rotations',
            'physics_application': 'Rotational symmetry in mechanics and quantum systems',
            'outcome': 'Apply mathematical structure to physical problems'
        }
        
        # The "difference": Lie bracket reveals the distinction
        results['lie_bracket_insight'] = {
            'formula': '[Physical, Mathematical] = Mathematical_Physics - Physical_Mathematics',
            'interpretation': 'The commutator shows how the two approaches complement each other',
            'philosophical_meaning': 'Physical math emphasizes phenomena → formalism; Math physics emphasizes structure → application',
            'synthesis': 'Both approaches are necessary for complete understanding'
        }
        
        return results
    
    def create_demonstration_examples(self) -> Dict[str, LieElement]:
        """
        Create examples for different Lie algebra types
        創建不同李代數類型的示例
        """
        examples = {}
        
        # Matrix Lie algebra example: Pauli matrices (physical mathematics)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        examples['pauli_x'] = LieElement(
            data=pauli_x,
            algebra_type=LieAlgebraType.MATRIX,
            name='σ_x',
            physical_interpretation='X-component of electron spin operator',
            mathematical_properties={'pauli_matrix': True, 'su2_generator': True}
        )
        
        examples['pauli_y'] = LieElement(
            data=pauli_y,
            algebra_type=LieAlgebraType.MATRIX,
            name='σ_y',
            physical_interpretation='Y-component of electron spin operator',
            mathematical_properties={'pauli_matrix': True, 'su2_generator': True}
        )
        
        examples['pauli_z'] = LieElement(
            data=pauli_z,
            algebra_type=LieAlgebraType.MATRIX,
            name='σ_z',
            physical_interpretation='Z-component of electron spin operator',
            mathematical_properties={'pauli_matrix': True, 'su2_generator': True}
        )
        
        # Vector field example (mathematical physics)
        x, y, z = sp.symbols('x y z')
        self.create_vector_field_operator([x, y, z])
        
        # Rotation vector field around z-axis
        rotation_field = [-y, x, 0]  # Vector field for rotation
        examples['rotation_field'] = LieElement(
            data=rotation_field,
            algebra_type=LieAlgebraType.VECTOR_FIELD,
            name='R_z',
            physical_interpretation='Infinitesimal rotation around z-axis',
            mathematical_properties={'rotation_generator': True, 'so3_element': True}
        )
        
        # Hamiltonian mechanics example (physics)
        q1, q2, p1, p2 = sp.symbols('q1 q2 p1 p2')
        self.create_poisson_operator([q1, q2], [p1, p2])
        
        # Harmonic oscillator Hamiltonian
        hamiltonian = (p1**2 + p2**2)/2 + (q1**2 + q2**2)/2
        examples['harmonic_oscillator'] = LieElement(
            data=hamiltonian,
            algebra_type=LieAlgebraType.PHYSICS,
            name='H',
            physical_interpretation='2D harmonic oscillator Hamiltonian',
            mathematical_properties={'hamiltonian': True, 'conserved_quantity': True}
        )
        
        # Angular momentum
        angular_momentum = q1*p2 - q2*p1
        examples['angular_momentum'] = LieElement(
            data=angular_momentum,
            algebra_type=LieAlgebraType.PHYSICS,
            name='L',
            physical_interpretation='Angular momentum in 2D',
            mathematical_properties={'angular_momentum': True, 'conserved_quantity': True}
        )
        
        self.examples = examples
        return examples
    
    def analyze_brackets(self) -> Dict[str, Any]:
        """
        Analyze Lie brackets in the examples
        分析示例中的李括號
        """
        if not self.examples:
            self.create_demonstration_examples()
        
        analysis = {}
        
        # Pauli matrix commutation relations
        try:
            pauli_xy = self.compute_bracket(self.examples['pauli_x'], self.examples['pauli_y'])
            analysis['pauli_commutator'] = {
                'operation': '[σ_x, σ_y]',
                'result': '2iσ_z',
                'physical_meaning': 'Non-commutativity of spin measurements',
                'mathematical_structure': 'SU(2) Lie algebra'
            }
        except Exception as e:
            analysis['pauli_commutator'] = {'error': str(e)}
        
        # Poisson bracket of angular momentum and Hamiltonian
        try:
            if 'harmonic_oscillator' in self.examples and 'angular_momentum' in self.examples:
                poisson_hl = self.compute_bracket(self.examples['harmonic_oscillator'], 
                                                self.examples['angular_momentum'])
                analysis['hamiltonian_angular_momentum'] = {
                    'operation': '{H, L}',
                    'result': '0 (for isotropic harmonic oscillator)',
                    'physical_meaning': 'Angular momentum is conserved',
                    'mathematical_structure': 'Symplectic geometry'
                }
        except Exception as e:
            analysis['hamiltonian_angular_momentum'] = {'error': str(e)}
        
        self.analysis_results = analysis
        return analysis
    
    def generate_insights(self) -> Dict[str, str]:
        """
        Generate insights about physical mathematics vs mathematical physics
        生成關於物理數學與數學物理的見解
        """
        insights = {
            'fundamental_difference': (
                "Physical Mathematics: Phenomena → Mathematical Structure\n"
                "Mathematical Physics: Mathematical Structure → Physical Applications"
            ),
            'lie_bracket_role': (
                "Lie brackets capture the essence of non-commutativity in both approaches:\n"
                "- In physics: Uncertainty principles, conservation laws\n"
                "- In mathematics: Group theory, differential geometry"
            ),
            'complementarity': (
                "The two approaches are complementary, like position and momentum:\n"
                "[Physical_Math, Mathematical_Physics] ≠ 0\n"
                "This non-commutativity drives scientific progress"
            ),
            'synthesis_equation': (
                "Complete Understanding = Physical_Mathematics ⊕ Mathematical_Physics\n"
                "where ⊕ represents synthesis, not mere addition"
            )
        }
        
        return insights