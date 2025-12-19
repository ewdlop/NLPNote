#!/usr/bin/env python3
"""
Tests for Syntax Tree Light Emitter
Validates the bioluminescent syntax tree visualization system.
"""

import unittest
import ast
from SyntaxTreeLightEmitter import (
    SyntaxTreeLightEmitter,
    LightEmission,
    LightSpectrum
)


class TestSyntaxTreeLightEmitter(unittest.TestCase):
    """Test cases for the bioluminescent syntax tree analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emitter = SyntaxTreeLightEmitter()
    
    def test_simple_assignment(self):
        """Test light emission for simple assignment"""
        code = "x = 42"
        analysis = self.emitter.analyze_code(code)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('total_nodes', analysis)
        self.assertIn('emissions', analysis)
        self.assertGreater(analysis['total_nodes'], 0)
    
    def test_semantic_weight_calculation(self):
        """Test semantic weight calculation for different node types"""
        # High semantic weight nodes
        func_node = ast.FunctionDef(
            name='test',
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[]
        )
        weight = self.emitter.calculate_semantic_weight(func_node)
        self.assertGreater(weight, 0.7)
        
        # Lower semantic weight nodes
        const_node = ast.Constant(value=42)
        weight = self.emitter.calculate_semantic_weight(const_node)
        self.assertLess(weight, 0.5)
    
    def test_complexity_score_calculation(self):
        """Test complexity score for different structures"""
        # Simple node
        simple_code = "x = 1"
        simple_tree = ast.parse(simple_code)
        simple_complexity = self.emitter.calculate_complexity_score(simple_tree)
        
        # Complex node
        complex_code = """
def nested_function():
    for i in range(10):
        if i > 5:
            for j in range(i):
                print(j)
"""
        complex_tree = ast.parse(complex_code)
        complex_complexity = self.emitter.calculate_complexity_score(complex_tree)
        
        self.assertGreater(complex_complexity, simple_complexity)
    
    def test_temperature_mapping(self):
        """Test complexity to temperature mapping"""
        # Low complexity should give lower temperature (red)
        low_temp = self.emitter.complexity_to_temperature(0.1)
        self.assertLess(low_temp, 3000)
        
        # High complexity should give higher temperature (blue)
        high_temp = self.emitter.complexity_to_temperature(0.9)
        self.assertGreater(high_temp, 7000)
    
    def test_rgb_conversion(self):
        """Test temperature to RGB color conversion"""
        # Red temperature
        red_rgb = self.emitter.temperature_to_rgb(2000)
        self.assertIsInstance(red_rgb, tuple)
        self.assertEqual(len(red_rgb), 3)
        self.assertGreater(red_rgb[0], red_rgb[2])  # More red than blue
        
        # Blue temperature
        blue_rgb = self.emitter.temperature_to_rgb(8000)
        self.assertGreater(blue_rgb[2], 100)  # Has significant blue component
    
    def test_wavelength_calculation(self):
        """Test Wien's displacement law application"""
        temp = 5000  # K (approximately sun's surface)
        wavelength = self.emitter.temperature_to_wavelength(temp)
        
        # Should be in visible spectrum (380-700 nm)
        self.assertGreater(wavelength, 300)
        self.assertLess(wavelength, 1000)
    
    def test_color_naming(self):
        """Test color name assignment based on temperature"""
        red_color = self.emitter.get_color_name(1500)
        self.assertEqual(red_color, "red")
        
        yellow_color = self.emitter.get_color_name(3500)
        self.assertEqual(yellow_color, "yellow")
        
        blue_color = self.emitter.get_color_name(7000)
        self.assertEqual(blue_color, "blue")
    
    def test_light_emission_structure(self):
        """Test that light emission has all required properties"""
        code = "def test(): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        emission = self.emitter.emit_light(func_node, (1, 0))
        
        self.assertIsInstance(emission, LightEmission)
        self.assertIsInstance(emission.node_type, str)
        self.assertIsInstance(emission.intensity, float)
        self.assertIsInstance(emission.temperature, float)
        self.assertIsInstance(emission.wavelength, float)
        self.assertIsInstance(emission.color_name, str)
        self.assertIsInstance(emission.rgb, tuple)
        self.assertIsInstance(emission.semantic_weight, float)
        self.assertIsInstance(emission.complexity_score, float)
        self.assertIsInstance(emission.position, tuple)
    
    def test_analyze_code_with_syntax_error(self):
        """Test handling of syntax errors"""
        bad_code = "def broken(:"
        analysis = self.emitter.analyze_code(bad_code)
        
        self.assertIn('error', analysis)
        self.assertIsInstance(analysis['error'], str)
    
    def test_analyze_code_valid(self):
        """Test complete code analysis"""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        analysis = self.emitter.analyze_code(code)
        
        self.assertNotIn('error', analysis)
        self.assertGreater(analysis['total_nodes'], 0)
        self.assertGreater(analysis['total_intensity'], 0)
        self.assertGreater(analysis['average_intensity'], 0)
        self.assertIsInstance(analysis['color_distribution'], dict)
        self.assertIsInstance(analysis['brightest_nodes'], list)
        self.assertIsInstance(analysis['hottest_nodes'], list)
        self.assertIsInstance(analysis['emissions'], list)
    
    def test_color_distribution(self):
        """Test that color distribution is properly calculated"""
        code = """
x = 1
y = 2
def func():
    for i in range(10):
        print(i)
"""
        analysis = self.emitter.analyze_code(code)
        
        color_dist = analysis['color_distribution']
        self.assertIsInstance(color_dist, dict)
        self.assertGreater(len(color_dist), 0)
        
        # All colors should have positive counts
        for color, count in color_dist.items():
            self.assertGreater(count, 0)
    
    def test_brightest_nodes(self):
        """Test that brightest nodes are correctly identified"""
        code = """
class MyClass:
    def method(self):
        return 42

def function():
    pass

x = 1
"""
        analysis = self.emitter.analyze_code(code)
        
        brightest = analysis['brightest_nodes']
        self.assertGreater(len(brightest), 0)
        
        # Check that brightest nodes have expected structure
        for node in brightest:
            self.assertIn('type', node)
            self.assertIn('intensity', node)
            self.assertIn('position', node)
            self.assertGreater(node['intensity'], 0)
    
    def test_hottest_nodes(self):
        """Test that hottest (most complex) nodes are correctly identified"""
        code = """
def complex_function(x):
    result = []
    for i in range(x):
        if i % 2 == 0:
            for j in range(i):
                result.append(j)
    return result
"""
        analysis = self.emitter.analyze_code(code)
        
        hottest = analysis['hottest_nodes']
        self.assertGreater(len(hottest), 0)
        
        # Check structure
        for node in hottest:
            self.assertIn('type', node)
            self.assertIn('temperature', node)
            self.assertIn('color', node)
            self.assertIn('position', node)
            self.assertGreater(node['temperature'], 0)
    
    def test_ascii_visualization(self):
        """Test ASCII visualization generation"""
        code = "def test(): return 42"
        visualization = self.emitter.generate_ascii_visualization(code)
        
        self.assertIsInstance(visualization, str)
        self.assertIn("BIOLUMINESCENT", visualization)
        self.assertIn("Total Nodes", visualization)
        self.assertIn("Color Distribution", visualization)
    
    def test_ascii_visualization_with_error(self):
        """Test ASCII visualization with invalid code"""
        bad_code = "def broken(:"
        visualization = self.emitter.generate_ascii_visualization(bad_code)
        
        self.assertIsInstance(visualization, str)
        self.assertIn("Error", visualization)
    
    def test_emission_tracking(self):
        """Test that emissions are properly tracked"""
        code = "x = 1 + 2"
        self.emitter.analyze_code(code)
        
        self.assertGreater(len(self.emitter.emissions), 0)
        self.assertGreater(self.emitter.total_intensity, 0)
    
    def test_position_information(self):
        """Test that position information is captured"""
        code = """
x = 1
y = 2
"""
        analysis = self.emitter.analyze_code(code)
        
        # Check that some emissions have position info
        has_position = any(
            e.position != (0, 0) 
            for e in analysis['emissions']
        )
        self.assertTrue(has_position)
    
    def test_empty_code(self):
        """Test handling of empty code"""
        code = ""
        analysis = self.emitter.analyze_code(code)
        
        # Empty code should still parse (as empty Module)
        self.assertNotIn('error', analysis)
        self.assertEqual(analysis['total_nodes'], 1)  # Just the Module node
    
    def test_intensity_range(self):
        """Test that intensity values are in valid range"""
        code = """
def test():
    for i in range(10):
        print(i)
"""
        analysis = self.emitter.analyze_code(code)
        
        for emission in analysis['emissions']:
            self.assertGreaterEqual(emission.intensity, 0.0)
            self.assertLessEqual(emission.intensity, 1.0)
    
    def test_temperature_range(self):
        """Test that temperature values are in expected range"""
        code = """
class Complex:
    def method(self):
        for i in range(10):
            for j in range(i):
                print(i, j)
"""
        analysis = self.emitter.analyze_code(code)
        
        for emission in analysis['emissions']:
            self.assertGreaterEqual(emission.temperature, 1000)
            self.assertLessEqual(emission.temperature, 10000)


class TestLightSpectrum(unittest.TestCase):
    """Test the LightSpectrum enum"""
    
    def test_spectrum_values(self):
        """Test that spectrum values are properly defined"""
        for spectrum in LightSpectrum:
            self.assertIsInstance(spectrum.value, tuple)
            self.assertEqual(len(spectrum.value), 2)
            self.assertLess(spectrum.value[0], spectrum.value[1])
    
    def test_spectrum_coverage(self):
        """Test that spectrum covers expected temperature range"""
        temps = [s.value[0] for s in LightSpectrum]
        self.assertIn(1000, temps)  # Start of red
        
        max_temp = max(s.value[1] for s in LightSpectrum)
        self.assertGreater(max_temp, 10000)  # Covers into UV


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_realistic_code_analysis(self):
        """Test analysis of realistic code sample"""
        code = """
class DataProcessor:
    '''Process data with various transformations'''
    
    def __init__(self, data):
        self.data = data
        self.processed = []
    
    def filter_data(self, condition):
        '''Filter data based on condition'''
        return [item for item in self.data if condition(item)]
    
    def transform(self, func):
        '''Apply transformation function'''
        self.processed = [func(item) for item in self.data]
        return self.processed
    
    def analyze(self):
        '''Perform analysis'''
        if not self.data:
            return None
        
        total = sum(self.data)
        avg = total / len(self.data)
        
        return {
            'total': total,
            'average': avg,
            'count': len(self.data)
        }
"""
        emitter = SyntaxTreeLightEmitter()
        analysis = emitter.analyze_code(code)
        
        # Should have substantial complexity
        self.assertGreater(analysis['total_nodes'], 50)
        self.assertGreater(analysis['max_complexity'], 0.3)
        
        # Should have multiple colors in distribution
        self.assertGreater(len(analysis['color_distribution']), 2)
        
        # Visualization should be comprehensive
        viz = emitter.generate_ascii_visualization(code)
        self.assertIn("Brightest Nodes", viz)
        self.assertIn("Hottest Nodes", viz)
        self.assertGreater(len(viz), 500)


if __name__ == '__main__':
    unittest.main()
