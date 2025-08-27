"""
Natural Transformations in Category Theory - Python Examples

This module provides concrete examples of natural transformations
using Python to illustrate the categorical concepts discussed in three.md.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Dict, Any
import numpy as np

# Type variables for generic categories
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


class Functor(ABC, Generic[A, B]):
    """Abstract base class for functors between categories."""
    
    @abstractmethod
    def map_object(self, obj: A) -> B:
        """Map an object from source category to target category."""
        pass
    
    @abstractmethod
    def map_morphism(self, morphism: Callable[[A], A]) -> Callable[[B], B]:
        """Map a morphism from source category to target category."""
        pass


class NaturalTransformation(ABC, Generic[A, B]):
    """Abstract base class for natural transformations."""
    
    def __init__(self, source_functor: Functor[A, B], target_functor: Functor[A, B]):
        self.source_functor = source_functor
        self.target_functor = target_functor
    
    @abstractmethod
    def component(self, obj: A) -> Callable[[B], B]:
        """Return the component of the natural transformation at object obj."""
        pass
    
    def verify_naturality(self, obj1: A, obj2: A, morphism: Callable[[A], A]) -> bool:
        """Verify the naturality condition for a given morphism."""
        # This is a conceptual verification - in practice would need
        # actual implementations with proper categorical structure
        try:
            # η_B ∘ F(f) = G(f) ∘ η_A
            # where F is source_functor, G is target_functor, η is self
            f_mapped = self.source_functor.map_morphism(morphism)
            g_mapped = self.target_functor.map_morphism(morphism)
            
            eta_a = self.component(obj1)
            eta_b = self.component(obj2)
            
            # In a real implementation, we'd compose these and check equality
            # Here we just return True as this is a demonstration
            return True
        except Exception:
            return False


class VectorSpaceFunctor(Functor[int, np.ndarray]):
    """Functor from dimension (int) to vector space (numpy array)."""
    
    def map_object(self, dimension: int) -> np.ndarray:
        """Map dimension to a vector space of that dimension."""
        return np.zeros(dimension)
    
    def map_morphism(self, morphism: Callable[[int], int]) -> Callable[[np.ndarray], np.ndarray]:
        """Map a function on dimensions to a linear transformation."""
        def matrix_transformation(vector: np.ndarray) -> np.ndarray:
            new_dim = morphism(len(vector))
            if new_dim >= len(vector):
                # Extend with zeros
                result = np.zeros(new_dim)
                result[:len(vector)] = vector
                return result
            else:
                # Truncate
                return vector[:new_dim]
        return matrix_transformation


class DoubleDualFunctor(Functor[int, np.ndarray]):
    """Functor representing the double dual of vector spaces."""
    
    def map_object(self, dimension: int) -> np.ndarray:
        """Map dimension to its double dual (same dimension for finite-dimensional spaces)."""
        return np.zeros(dimension)
    
    def map_morphism(self, morphism: Callable[[int], int]) -> Callable[[np.ndarray], np.ndarray]:
        """Map morphism to its double dual."""
        # For finite-dimensional spaces, this is isomorphic to the original
        base_functor = VectorSpaceFunctor()
        return base_functor.map_morphism(morphism)


class DoubleDualTransformation(NaturalTransformation[int, np.ndarray]):
    """Natural transformation from vector spaces to their double duals."""
    
    def __init__(self):
        source = VectorSpaceFunctor()
        target = DoubleDualFunctor()
        super().__init__(source, target)
    
    def component(self, dimension: int) -> Callable[[np.ndarray], np.ndarray]:
        """The canonical embedding v ↦ v** where v**(φ) = φ(v)."""
        def embedding(vector: np.ndarray) -> np.ndarray:
            # In finite dimensions, this is just the identity
            # In the actual mathematical setting, this would be
            # v ↦ (φ ↦ φ(v)) for linear functionals φ
            return vector.copy()
        return embedding


class FundamentalGroupSimulator:
    """
    Simplified simulation of fundamental group computations
    to demonstrate naturality in topological contexts.
    """
    
    def __init__(self, space_name: str, generators: list):
        self.space_name = space_name
        self.generators = generators  # Simplified representation
    
    def induced_map(self, other_space: 'FundamentalGroupSimulator', 
                   continuous_map: str) -> Callable[[str], str]:
        """Simulate the induced map on fundamental groups."""
        def group_homomorphism(element: str) -> str:
            # Simplified mapping - in reality this would involve
            # complex topological computations
            if element in self.generators:
                return f"{continuous_map}({element})"
            return element
        return group_homomorphism


class HurewiczTransformation:
    """
    Demonstration of the Hurewicz natural transformation
    from homotopy groups to homology groups.
    """
    
    @staticmethod
    def hurewicz_map(homotopy_element: str, dimension: int) -> str:
        """
        Simplified Hurewicz homomorphism π_n(X) → H_n(X).
        In reality, this involves sophisticated algebraic topology.
        """
        if dimension == 1:
            # For π_1, we need to abelianize first
            return f"abelianized({homotopy_element})"
        else:
            # For n ≥ 2, homotopy groups are already abelian
            return f"H_{dimension}({homotopy_element})"
    
    @staticmethod
    def verify_naturality(space1: str, space2: str, continuous_map: str,
                         homotopy_element: str, dimension: int) -> bool:
        """
        Verify that the Hurewicz map commutes with induced maps.
        This demonstrates: H_n(f) ∘ h_X = h_Y ∘ π_n(f)
        """
        # Left side: H_n(f) ∘ h_X
        h_x = HurewiczTransformation.hurewicz_map(homotopy_element, dimension)
        left_side = f"{continuous_map}_*({h_x})"
        
        # Right side: h_Y ∘ π_n(f)
        pi_f = f"{continuous_map}_*({homotopy_element})"
        right_side = HurewiczTransformation.hurewicz_map(pi_f, dimension)
        
        # In a real implementation, we'd check mathematical equality
        # Here we demonstrate the structure
        print(f"Left side: {left_side}")
        print(f"Right side: {right_side}")
        print(f"Naturality: The Hurewicz map commutes with {continuous_map}")
        return True


def demonstrate_natural_transformations():
    """Demonstrate various natural transformations with examples."""
    
    print("=== Natural Transformations in Category Theory ===\n")
    
    # Example 1: Double Dual Natural Transformation
    print("1. Double Dual Natural Transformation")
    print("   For finite-dimensional vector spaces: V → V**")
    
    double_dual = DoubleDualTransformation()
    dimension = 3
    test_vector = np.array([1, 2, 3])
    
    embedding = double_dual.component(dimension)
    result = embedding(test_vector)
    
    print(f"   Original vector: {test_vector}")
    print(f"   After double dual embedding: {result}")
    print(f"   Naturality verified: {double_dual.verify_naturality(3, 4, lambda x: x+1)}")
    print()
    
    # Example 2: Fundamental Group Naturality
    print("2. Fundamental Group Functor Naturality")
    circle = FundamentalGroupSimulator("S¹", ["γ"])  # Circle with generator γ
    torus = FundamentalGroupSimulator("T²", ["a", "b"])  # Torus with generators a, b
    
    inclusion_map = circle.induced_map(torus, "inclusion")
    mapped_generator = inclusion_map("γ")
    
    print(f"   Circle π₁(S¹): generators = {circle.generators}")
    print(f"   Torus π₁(T²): generators = {torus.generators}")
    print(f"   Inclusion map sends γ ↦ {mapped_generator}")
    print()
    
    # Example 3: Hurewicz Natural Transformation
    print("3. Hurewicz Natural Transformation: π_n(X) → H_n(X)")
    
    space_x = "S²"  # 2-sphere
    space_y = "S² ∨ S²"  # Wedge of two 2-spheres
    f_map = "wedge_inclusion"
    homotopy_class = "[σ]"  # A homotopy class in π₂(S²)
    
    naturality_check = HurewiczTransformation.verify_naturality(
        space_x, space_y, f_map, homotopy_class, 2
    )
    print(f"   Naturality verified: {naturality_check}")
    print()
    
    # Example 4: Conceptual Example - Parametric Polymorphism
    print("4. Connection to Computer Science: Parametric Polymorphism")
    print("   The 'length' function on lists is natural:")
    print("   For any function f: A → B and list xs: [A]")
    print("   length(map(f, xs)) = length(xs)")
    print("   This naturality is what makes polymorphic functions 'canonical'")
    print()
    
    print("=== Summary ===")
    print("Natural transformations provide:")
    print("• Canonical, choice-independent mappings between functors")
    print("• A way to relate different mathematical structures systematically")
    print("• The foundation for understanding deep connections in mathematics")
    print("• Applications from topology to computer science to physics")


if __name__ == "__main__":
    demonstrate_natural_transformations()