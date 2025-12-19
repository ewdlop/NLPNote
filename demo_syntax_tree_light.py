#!/usr/bin/env python3
"""
Demo: Syntax Tree Light Emission Examples
Various demonstrations of the bioluminescent syntax tree analyzer.
"""

from SyntaxTreeLightEmitter import SyntaxTreeLightEmitter


def example_1_simple_code():
    """Example 1: Simple code - Red/Orange glow"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Code (Red/Orange Glow)")
    print("="*80)
    
    code = """
x = 42
y = x + 1
print(y)
"""
    
    emitter = SyntaxTreeLightEmitter()
    viz = emitter.generate_ascii_visualization(code)
    print(viz)
    print("\nInterpretation: Simple, straightforward code emits cooler (red/orange) light.")


def example_2_moderate_complexity():
    """Example 2: Moderate complexity - Yellow/White glow"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Moderate Complexity (Yellow/White Glow)")
    print("="*80)
    
    code = """
def calculate_average(numbers):
    '''Calculate the average of a list of numbers'''
    if not numbers:
        return 0
    
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    
    return average

# Test the function
result = calculate_average([1, 2, 3, 4, 5])
print(f"Average: {result}")
"""
    
    emitter = SyntaxTreeLightEmitter()
    viz = emitter.generate_ascii_visualization(code)
    print(viz)
    print("\nInterpretation: Moderate complexity with control flow emits warmer light.")


def example_3_high_complexity():
    """Example 3: High complexity - Blue/White glow"""
    print("\n" + "="*80)
    print("EXAMPLE 3: High Complexity (Blue/White Glow)")
    print("="*80)
    
    code = """
class BinarySearchTree:
    '''A binary search tree implementation'''
    
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert a value into the tree'''
        if self.root is None:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        '''Recursively insert a value'''
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for a value in the tree'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        '''Recursively search for a value'''
        if node is None:
            return False
        
        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
"""
    
    emitter = SyntaxTreeLightEmitter()
    viz = emitter.generate_ascii_visualization(code)
    print(viz)
    print("\nInterpretation: Complex, nested class structure emits hot (blue/white) light.")


def example_4_nested_loops():
    """Example 4: Nested loops - Very hot glow"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Nested Loops (Very Hot Glow)")
    print("="*80)
    
    code = """
def matrix_multiply(A, B):
    '''Multiply two matrices'''
    result = []
    
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            total = 0
            for k in range(len(B)):
                total += A[i][k] * B[k][j]
            row.append(total)
        result.append(row)
    
    return result
"""
    
    emitter = SyntaxTreeLightEmitter()
    viz = emitter.generate_ascii_visualization(code)
    print(viz)
    print("\nInterpretation: Triple-nested loops create very hot (blue) emission.")


def example_5_list_comprehension():
    """Example 5: List comprehension vs loops"""
    print("\n" + "="*80)
    print("EXAMPLE 5: List Comprehension vs Traditional Loop")
    print("="*80)
    
    # Traditional loop
    code_loop = """
result = []
for i in range(10):
    if i % 2 == 0:
        result.append(i * i)
"""
    
    # List comprehension
    code_comp = """
result = [i * i for i in range(10) if i % 2 == 0]
"""
    
    emitter1 = SyntaxTreeLightEmitter()
    emitter2 = SyntaxTreeLightEmitter()
    
    print("\n--- Traditional Loop ---")
    viz1 = emitter1.generate_ascii_visualization(code_loop)
    print(viz1)
    
    print("\n--- List Comprehension ---")
    viz2 = emitter2.generate_ascii_visualization(code_comp)
    print(viz2)
    
    print("\nInterpretation: List comprehension may be more compact but shows similar complexity.")


def example_6_decorator_pattern():
    """Example 6: Decorator pattern - Rich semantic content"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Decorator Pattern (Rich Semantic Content)")
    print("="*80)
    
    code = """
def timing_decorator(func):
    '''Decorator to measure function execution time'''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    '''A function that takes some time'''
    time.sleep(1)
    return "Done"
"""
    
    emitter = SyntaxTreeLightEmitter()
    viz = emitter.generate_ascii_visualization(code)
    print(viz)
    print("\nInterpretation: Decorators and closures show high semantic weight (bright nodes).")


def example_7_comparison():
    """Example 7: Side-by-side comparison of different approaches"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Comparing Different Implementations")
    print("="*80)
    
    # Imperative
    imperative = """
def sum_squares_imperative(n):
    total = 0
    for i in range(n):
        total += i * i
    return total
"""
    
    # Functional
    functional = """
def sum_squares_functional(n):
    return sum(i * i for i in range(n))
"""
    
    # Recursive
    recursive = """
def sum_squares_recursive(n):
    if n == 0:
        return 0
    return (n - 1) ** 2 + sum_squares_recursive(n - 1)
"""
    
    for name, code in [("Imperative", imperative), 
                       ("Functional", functional), 
                       ("Recursive", recursive)]:
        print(f"\n--- {name} Style ---")
        emitter = SyntaxTreeLightEmitter()
        analysis = emitter.analyze_code(code)
        
        print(f"Total Nodes: {analysis['total_nodes']}")
        print(f"Avg Intensity: {analysis['average_intensity']:.2f}")
        print(f"Max Complexity: {analysis['max_complexity']:.2f}")
        print(f"Color Distribution: {analysis['color_distribution']}")


def example_8_error_handling():
    """Example 8: Error handling structures"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Error Handling (Try-Except Blocks)")
    print("="*80)
    
    code = """
def safe_divide(a, b):
    '''Safely divide two numbers with error handling'''
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    except TypeError:
        print("Error: Invalid input types")
        return None
    finally:
        print("Division operation completed")
"""
    
    emitter = SyntaxTreeLightEmitter()
    viz = emitter.generate_ascii_visualization(code)
    print(viz)
    print("\nInterpretation: Exception handling adds semantic weight and some complexity.")


def example_9_hotspot_analysis():
    """Example 9: Finding complexity hotspots"""
    print("\n" + "="*80)
    print("EXAMPLE 9: Complexity Hotspot Analysis")
    print("="*80)
    
    code = """
def complex_algorithm(data):
    '''An algorithm with varying complexity sections'''
    
    # Simple section (should glow red/orange)
    x = 1
    y = 2
    z = x + y
    
    # Moderate section (should glow yellow/white)
    if len(data) > 0:
        for item in data:
            print(item)
    
    # Complex section (should glow blue/white)
    result = []
    for i in range(len(data)):
        temp = []
        for j in range(i):
            if data[i] > data[j]:
                for k in range(j):
                    temp.append(data[k])
        result.append(temp)
    
    return result
"""
    
    emitter = SyntaxTreeLightEmitter()
    viz = emitter.generate_ascii_visualization(code)
    print(viz)
    
    analysis = emitter.analyze_code(code)
    print("\n🔥 HOTSPOTS IDENTIFIED:")
    print("The following nodes are complexity hotspots (> 6000K):")
    for emission in analysis['emissions']:
        if emission.temperature > 6000:
            print(f"  - {emission.node_type} at line {emission.position[0]}: {emission.temperature:.0f}K ({emission.color_name})")


def example_10_physics_analogy():
    """Example 10: Detailed physics analogy explanation"""
    print("\n" + "="*80)
    print("EXAMPLE 10: The Physics Behind the Light")
    print("="*80)
    
    code = "def test(): return 42"
    
    emitter = SyntaxTreeLightEmitter()
    analysis = emitter.analyze_code(code)
    
    print("\nPhysics Concepts Applied:\n")
    
    for i, emission in enumerate(analysis['emissions'][:3], 1):
        print(f"Node {i}: {emission.node_type}")
        print(f"  Temperature: {emission.temperature:.0f}K")
        print(f"  Peak Wavelength (Wien's Law): {emission.wavelength:.1f} nm")
        print(f"  Color: {emission.color_name.upper()}")
        print(f"  RGB: {emission.rgb}")
        print(f"  Intensity: {emission.intensity:.3f}")
        
        # Explain the physics
        if emission.temperature < 2000:
            print("  → Physics: Like a dim red ember, low energy emission")
        elif emission.temperature < 4000:
            print("  → Physics: Like a candle flame, moderate energy")
        elif emission.temperature < 6000:
            print("  → Physics: Like incandescent light, high energy")
        else:
            print("  → Physics: Like a blue star, very high energy")
        print()


def main():
    """Run all examples"""
    print("=" * 80)
    print("🌟 SYNTAX TREE LIGHT EMISSION - COMPREHENSIVE EXAMPLES 🌟")
    print("=" * 80)
    print()
    print("This demo shows how syntax trees 'emit light' based on:")
    print("  • Intensity = Semantic Weight (information content)")
    print("  • Temperature/Color = Complexity (via blackbody radiation)")
    print()
    input("Press Enter to start the demo...")
    
    examples = [
        example_1_simple_code,
        example_2_moderate_complexity,
        example_3_high_complexity,
        example_4_nested_loops,
        example_5_list_comprehension,
        example_6_decorator_pattern,
        example_7_comparison,
        example_8_error_handling,
        example_9_hotspot_analysis,
        example_10_physics_analogy,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
            if i < len(examples):
                print("\n" + "-" * 80)
                input(f"\nPress Enter to continue to Example {i+1}...")
        except Exception as e:
            print(f"\nError in example: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✨ Demo Complete! The syntax trees have shared their light! ✨")
    print("=" * 80)


if __name__ == "__main__":
    main()
