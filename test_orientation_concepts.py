#!/usr/bin/env python3
"""
Test cases for orientation concepts - validating "An orientation-less is oriented"

This module contains tests to validate that the implementation correctly
demonstrates how orientation emerges from orientation-less structures.
"""

from orientation_concepts import (
    OrientationTransformer, 
    TopologicalOrientation, 
    LinguisticOrientation,
    OrientationType
)


def test_mathematical_orientation_emergence():
    """Test that mathematical structures can be both orientation-less and oriented."""
    print("Testing mathematical orientation emergence...")
    
    topological = TopologicalOrientation()
    
    # Test case: Simple loop around circle
    cycle_data = {
        'cycles': [
            {
                'is_non_trivial': True,
                'direction': 'counterclockwise',
                'multiplicity': 1
            }
        ]
    }
    
    result = topological.homology_vs_homotopy_orientation(cycle_data)
    
    # Validate orientation emergence
    homology_oriented = result['homology_analysis']['preserves_orientation']
    homotopy_oriented = result['homotopy_analysis']['preserves_orientation']
    orientation_emerged = result['orientation_emergence']['orientation_emerged']
    
    assert not homology_oriented, "Homology should be orientation-less"
    assert homotopy_oriented, "Homotopy should preserve orientation"
    assert orientation_emerged, "Orientation should emerge from homology to homotopy"
    
    print("✓ Mathematical orientation emergence validated")
    return True


def test_linguistic_orientation_emergence():
    """Test that linguistic expressions can gain orientation through context."""
    print("Testing linguistic orientation emergence...")
    
    linguistic = LinguisticOrientation()
    
    # Test case: Ambiguous text
    ambiguous_text = "That is good."
    
    # No context (should be orientation-less)
    no_context_state = linguistic.analyze_semantic_orientation(ambiguous_text, {})
    
    # With context (should gain orientation)
    rich_context = {
        'speaker': 'teacher',
        'situation': 'grading',
        'formality_level': 'formal',
        'time': 'present'
    }
    context_state = linguistic.analyze_semantic_orientation(ambiguous_text, rich_context)
    
    # Validate orientation emergence
    assert no_context_state.initial_orientation == OrientationType.NONE
    assert no_context_state.orientation_strength < 0.3, "Should have low orientation without context"
    
    assert context_state.orientation_strength > no_context_state.orientation_strength
    print(f"Orientation strength increased from {no_context_state.orientation_strength:.2f} to {context_state.orientation_strength:.2f}")
    
    print("✓ Linguistic orientation emergence validated")
    return True


def test_orientation_transformer_consistency():
    """Test that the orientation transformer produces consistent results."""
    print("Testing orientation transformer consistency...")
    
    transformer = OrientationTransformer()
    demo = transformer.demonstrate_orientation_emergence()
    
    # Check that both mathematical and linguistic examples are present
    assert 'mathematical_example' in demo
    assert 'linguistic_example' in demo
    assert 'philosophical_insight' in demo
    
    # Check mathematical example structure
    math_ex = demo['mathematical_example']
    assert 'analysis' in math_ex
    assert 'homology_analysis' in math_ex['analysis']
    assert 'homotopy_analysis' in math_ex['analysis']
    assert 'orientation_emergence' in math_ex['analysis']
    
    # Check linguistic example structure
    ling_ex = demo['linguistic_example']
    assert 'examples' in ling_ex
    assert len(ling_ex['examples']) > 0
    
    print("✓ Orientation transformer consistency validated")
    return True


def test_paradox_resolution():
    """Test that the paradox 'orientation-less is oriented' is properly resolved."""
    print("Testing paradox resolution...")
    
    transformer = OrientationTransformer()
    demo = transformer.demonstrate_orientation_emergence()
    
    insight = demo['philosophical_insight']
    
    # Key concepts should be present
    assert 'core_principle' in insight
    assert 'paradox_resolution' in insight
    assert 'mathematical_manifestation' in insight
    assert 'linguistic_manifestation' in insight
    
    # The resolution should explain the relational nature of orientation
    resolution = insight['paradox_resolution'].lower()
    assert 'relational' in resolution or 'relationship' in resolution
    assert 'external' in insight['core_principle'].lower()
    
    print("✓ Paradox resolution explanation validated")
    return True


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Running Orientation Concepts Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_mathematical_orientation_emergence,
        test_linguistic_orientation_emergence,
        test_orientation_transformer_consistency,
        test_paradox_resolution
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed! The orientation concept is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)