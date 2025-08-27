"""
Test file for natural_transformations.py

Tests the examples and demonstrations of natural transformations
in category theory.
"""

import pytest
import numpy as np
from natural_transformations import (
    VectorSpaceFunctor,
    DoubleDualFunctor,
    DoubleDualTransformation,
    FundamentalGroupSimulator,
    HurewiczTransformation
)


class TestVectorSpaceFunctor:
    """Test the vector space functor implementation."""
    
    def test_map_object(self):
        """Test object mapping for vector spaces."""
        functor = VectorSpaceFunctor()
        
        # Test different dimensions
        v3 = functor.map_object(3)
        assert isinstance(v3, np.ndarray)
        assert len(v3) == 3
        assert np.allclose(v3, np.zeros(3))
        
        v5 = functor.map_object(5)
        assert len(v5) == 5
    
    def test_map_morphism(self):
        """Test morphism mapping for vector spaces."""
        functor = VectorSpaceFunctor()
        
        # Test dimension extension
        extend_morphism = lambda x: x + 2
        transformation = functor.map_morphism(extend_morphism)
        
        test_vector = np.array([1, 2, 3])
        result = transformation(test_vector)
        
        assert len(result) == 5  # 3 + 2
        assert np.allclose(result[:3], test_vector)
        assert np.allclose(result[3:], np.zeros(2))
        
        # Test dimension reduction
        reduce_morphism = lambda x: max(1, x - 1)
        reduction = functor.map_morphism(reduce_morphism)
        
        result = reduction(test_vector)
        assert len(result) == 2  # 3 - 1
        assert np.allclose(result, test_vector[:2])


class TestDoubleDualTransformation:
    """Test the double dual natural transformation."""
    
    def test_component(self):
        """Test the natural transformation components."""
        transformation = DoubleDualTransformation()
        
        # Test embedding component
        embedding = transformation.component(3)
        test_vector = np.array([1, 2, 3])
        result = embedding(test_vector)
        
        # For finite-dimensional spaces, double dual is isomorphic to original
        assert np.allclose(result, test_vector)
    
    def test_verify_naturality(self):
        """Test naturality verification."""
        transformation = DoubleDualTransformation()
        
        # Test naturality condition (simplified)
        is_natural = transformation.verify_naturality(3, 4, lambda x: x + 1)
        assert is_natural  # Should return True for our simplified implementation


class TestFundamentalGroupSimulator:
    """Test the fundamental group simulation."""
    
    def test_initialization(self):
        """Test proper initialization of fundamental group simulator."""
        circle = FundamentalGroupSimulator("S¹", ["γ"])
        
        assert circle.space_name == "S¹"
        assert circle.generators == ["γ"]
    
    def test_induced_map(self):
        """Test induced maps on fundamental groups."""
        circle = FundamentalGroupSimulator("S¹", ["γ"])
        torus = FundamentalGroupSimulator("T²", ["a", "b"])
        
        # Test inclusion map
        inclusion = circle.induced_map(torus, "inclusion")
        mapped_element = inclusion("γ")
        
        assert "inclusion" in mapped_element
        assert "γ" in mapped_element
        
        # Test identity elements
        identity_element = inclusion("e")  # e not in generators
        assert identity_element == "e"


class TestHurewiczTransformation:
    """Test the Hurewicz natural transformation."""
    
    def test_hurewicz_map(self):
        """Test the Hurewicz homomorphism."""
        # Test dimension 1 (fundamental group)
        result1 = HurewiczTransformation.hurewicz_map("[α]", 1)
        assert "abelianized" in result1
        assert "[α]" in result1
        
        # Test higher dimensions
        result2 = HurewiczTransformation.hurewicz_map("[β]", 2)
        assert "H_2" in result2
        assert "[β]" in result2
    
    def test_verify_naturality(self):
        """Test naturality of Hurewicz transformation."""
        is_natural = HurewiczTransformation.verify_naturality(
            "S²", "S² ∨ S²", "inclusion", "[σ]", 2
        )
        assert is_natural  # Should always return True in our demo


class TestIntegration:
    """Integration tests for the complete natural transformation framework."""
    
    def test_complete_demonstration(self):
        """Test that the complete demonstration runs without errors."""
        try:
            from natural_transformations import demonstrate_natural_transformations
            # Redirect output to avoid cluttering test results
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            demonstrate_natural_transformations()
            
            output = buffer.getvalue()
            sys.stdout = old_stdout
            
            # Check that key concepts appear in output
            assert "Natural Transformations in Category Theory" in output
            assert "Double Dual" in output
            assert "Hurewicz" in output
            assert "Naturality verified" in output
            
        except Exception as e:
            pytest.fail(f"Demonstration failed with error: {e}")
    
    def test_mathematical_consistency(self):
        """Test basic mathematical consistency of implementations."""
        # Test vector space functor preserves dimension structure
        functor = VectorSpaceFunctor()
        
        v1 = functor.map_object(3)
        v2 = functor.map_object(3)
        
        # Same dimension should give same structure
        assert v1.shape == v2.shape
        
        # Test natural transformation component consistency
        transformation = DoubleDualTransformation()
        comp1 = transformation.component(3)
        comp2 = transformation.component(3)
        
        test_vector = np.array([1, 2, 3])
        result1 = comp1(test_vector)
        result2 = comp2(test_vector)
        
        # Should give consistent results
        assert np.allclose(result1, result2)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])