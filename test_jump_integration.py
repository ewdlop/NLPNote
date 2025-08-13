#!/usr/bin/env python3
"""
Integration test for Jump Instruction vs Jump Thinking with existing NLP components

Tests the integration of the new jump analysis with existing repository components
like AStarNLP and HumanExpressionEvaluator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from JumpInstructionVsJumpThinking import CognitiveJumpEngine, ComputationalJumpEngine, JumpContext
    from AStarNLP import AStarNLP
    print("✓ Successfully imported all required modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_integration_with_astar():
    """Test integration with A* NLP algorithm"""
    print("\n=== Testing Integration with A*NLP ===")
    
    # Test A* pathfinding
    try:
        semantic_path = AStarNLP.semantic_path_finding("computer", "understanding")
        print(f"✓ A* Semantic Path: {semantic_path}")
        
        # Test text transformation
        transform_result = AStarNLP.text_transformation_astar("hello world", "world hello")
        print(f"✓ A* Text Transform: Success={transform_result['success']}, Iterations={transform_result['iterations']}")
        
    except Exception as e:
        print(f"✗ A*NLP integration error: {e}")
        return False
    
    return True


def test_cognitive_jump_pathfinding():
    """Test cognitive jump engine as pathfinding alternative"""
    print("\n=== Testing Cognitive Jump Pathfinding ===")
    
    try:
        cog_engine = CognitiveJumpEngine()
        
        # Create context for pathfinding
        context = JumpContext(
            execution_state={},
            cognitive_state={'relevant_concepts': ['technology', 'learning']},
            user_profile={'domain_expertise': 'computer_science'},
            environmental_factors={'goal_oriented': True}
        )
        
        # Simulate pathfinding through concept space
        path = []
        current_concept = "computer"
        target_concept = "understanding"
        
        for i in range(5):  # Maximum 5 jumps
            jump_result = cog_engine.simulate_associative_jump(current_concept, context)
            path.append(jump_result['to'])
            current_concept = jump_result['to']
            
            if current_concept == target_concept:
                break
        
        print(f"✓ Cognitive path from 'computer' to 'understanding': {' → '.join(['computer'] + path)}")
        
        # Analyze the path
        analysis = cog_engine.get_cognitive_analysis()
        print(f"✓ Average predictability: {analysis['patterns']['average_predictability']:.2f}")
        
    except Exception as e:
        print(f"✗ Cognitive pathfinding error: {e}")
        return False
    
    return True


def test_computational_jump_simulation():
    """Test computational jump simulation"""
    print("\n=== Testing Computational Jump Simulation ===")
    
    try:
        comp_engine = ComputationalJumpEngine()
        
        # Simulate a simple program with jumps
        program_steps = [
            ("conditional", True, 10),
            ("unconditional", None, 20),
            ("function_call", 30, 5),
            ("return", None, None)
        ]
        
        for step_type, condition, target in program_steps:
            if step_type == "conditional":
                result = comp_engine.execute_conditional_jump(condition, target)
            elif step_type == "unconditional":
                result = comp_engine.execute_unconditional_jump(target)
            elif step_type == "function_call":
                result = comp_engine.execute_function_call(target, comp_engine.program_counter + 1)
            elif step_type == "return":
                result = comp_engine.execute_return()
            
            print(f"✓ {step_type}: {result.get('from', 'N/A')} → {result.get('to', 'N/A')}")
        
        # Get analysis
        analysis = comp_engine.get_jump_analysis()
        print(f"✓ Total jumps: {analysis['total_jumps']}")
        print(f"✓ Predictability: {analysis['patterns']['predictability_score']}")
        
    except Exception as e:
        print(f"✗ Computational jump simulation error: {e}")
        return False
    
    return True


def test_text_analysis_integration():
    """Test integration with text analysis"""
    print("\n=== Testing Text Analysis Integration ===")
    
    try:
        # Sample text with potential cognitive jumps
        sample_text = """
        Programming requires logical thinking. The sunset was beautiful yesterday.
        Algorithms solve complex problems. My grandmother used to bake cookies.
        Data structures organize information efficiently.
        """
        
        # Analyze using A* for text transformation
        sentences = [s.strip() for s in sample_text.split('.') if s.strip()]
        
        print(f"✓ Analyzing {len(sentences)} sentences for conceptual jumps")
        
        # Simulate detecting jumps between sentences
        concepts = ["programming", "sunset", "algorithms", "grandmother", "data_structures"]
        
        cog_engine = CognitiveJumpEngine()
        context = JumpContext(
            execution_state={},
            cognitive_state={'text_analysis_mode': True},
            user_profile={},
            environmental_factors={}
        )
        
        detected_jumps = []
        for i in range(len(concepts) - 1):
            # Calculate semantic distance (simplified)
            semantic_distance = cog_engine._calculate_semantic_distance(concepts[i], concepts[i+1])
            
            if semantic_distance > 0.6:  # Threshold for detecting jumps
                detected_jumps.append({
                    'from': concepts[i],
                    'to': concepts[i+1],
                    'distance': semantic_distance,
                    'sentence_index': i
                })
        
        print(f"✓ Detected {len(detected_jumps)} cognitive jumps in text")
        for jump in detected_jumps:
            print(f"  Jump: {jump['from']} → {jump['to']} (distance: {jump['distance']:.2f})")
        
    except Exception as e:
        print(f"✗ Text analysis integration error: {e}")
        return False
    
    return True


def run_integration_tests():
    """Run all integration tests"""
    print("Running Jump Instruction vs Jump Thinking Integration Tests")
    print("=" * 60)
    
    tests = [
        test_integration_with_astar,
        test_cognitive_jump_pathfinding,
        test_computational_jump_simulation,
        test_text_analysis_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
    
    print(f"\n=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)