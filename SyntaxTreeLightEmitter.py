#!/usr/bin/env python3
"""
Syntax Tree Light Emitter
A creative visualization system that represents syntax trees as bioluminescent structures
that emit light with properties similar to blackbody radiation.

Concept: Each node in a syntax tree emits "light" with intensity and color based on:
- Semantic weight (information content)
- Syntactic complexity
- Color temperature mapping (blackbody radiation spectrum)
"""

import ast
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LightSpectrum(Enum):
    """Blackbody radiation color temperature mapping"""
    INFRARED = (700, 1000)      # Low energy, simple structures
    RED = (1000, 2000)          # Basic syntax nodes
    ORANGE = (2000, 3000)       # Intermediate complexity
    YELLOW = (3000, 4000)       # Moderate complexity
    WHITE = (4000, 6000)        # High complexity
    BLUE = (6000, 10000)        # Very high complexity
    ULTRAVIOLET = (10000, 50000)  # Extremely complex


@dataclass
class LightEmission:
    """Represents the light emission properties of a syntax node"""
    node_type: str
    intensity: float  # 0.0 to 1.0
    temperature: float  # Kelvin
    wavelength: float  # nanometers
    color_name: str
    rgb: Tuple[int, int, int]
    semantic_weight: float
    complexity_score: float
    position: Tuple[int, int]  # line, column


class SyntaxTreeLightEmitter:
    """
    Analyzes syntax trees and generates bioluminescent visualizations
    where each node emits light based on its properties.
    """
    
    def __init__(self):
        self.emissions: List[LightEmission] = []
        self.total_intensity = 0.0
        self.max_complexity = 0.0
        
    def calculate_semantic_weight(self, node: ast.AST) -> float:
        """
        Calculate semantic weight based on node type and content.
        Higher weight = more information/meaning.
        """
        weights = {
            # High semantic weight
            ast.FunctionDef: 1.0,
            ast.ClassDef: 1.0,
            ast.Lambda: 0.9,
            ast.Return: 0.8,
            
            # Medium-high weight
            ast.If: 0.7,
            ast.For: 0.7,
            ast.While: 0.7,
            ast.With: 0.7,
            
            # Medium weight
            ast.Assign: 0.6,
            ast.Call: 0.6,
            ast.Import: 0.6,
            
            # Lower weight
            ast.Name: 0.4,
            ast.Constant: 0.3,
            ast.BinOp: 0.5,
            ast.UnaryOp: 0.4,
            
            # Structural nodes
            ast.Module: 0.2,
            ast.Expr: 0.3,
        }
        
        # Get base weight
        weight = weights.get(type(node), 0.5)
        
        # Adjust based on children count
        child_count = len(list(ast.walk(node))) - 1
        child_factor = min(1.0, child_count / 10.0)
        
        return min(1.0, weight + child_factor * 0.3)
    
    def calculate_complexity_score(self, node: ast.AST) -> float:
        """
        Calculate structural complexity of a node.
        More nested/complex structures = higher score.
        """
        # Count total nodes in subtree
        node_count = len(list(ast.walk(node)))
        
        # Calculate depth
        def get_depth(n: ast.AST, current_depth: int = 0) -> int:
            max_depth = current_depth
            for child in ast.iter_child_nodes(n):
                child_depth = get_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            return max_depth
        
        depth = get_depth(node)
        
        # Complexity is a combination of breadth (node count) and depth
        complexity = (node_count * 0.1 + depth * 0.5) / 10.0
        return min(1.0, complexity)
    
    def complexity_to_temperature(self, complexity: float) -> float:
        """
        Map complexity score to blackbody temperature (Kelvin).
        Simple structures = cooler (red/orange)
        Complex structures = hotter (blue/white)
        """
        # Map 0.0-1.0 complexity to 1000K-10000K temperature range
        min_temp = 1000
        max_temp = 10000
        return min_temp + (max_temp - min_temp) * complexity
    
    def temperature_to_rgb(self, temperature: float) -> Tuple[int, int, int]:
        """
        Convert blackbody temperature to RGB color.
        Based on approximation of blackbody radiation spectrum.
        """
        # Ensure temperature is positive and reasonable
        temp = max(10.0, temperature / 100.0)  # Minimum 1000K
        
        # Calculate Red
        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            red = max(0, min(255, red))
        
        # Calculate Green
        if temp <= 66:
            green = temp
            # Guard against log(0) or log(negative)
            if green > 0:
                green = 99.4708025861 * math.log(green) - 161.1195681661
            else:
                green = 0
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))
        
        # Calculate Blue
        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            # Guard against log(0) or log(negative)
            if blue > 0:
                blue = 138.5177312231 * math.log(blue) - 305.0447927307
            else:
                blue = 0
            blue = max(0, min(255, blue))
        
        return (int(red), int(green), int(blue))
    
    def get_color_name(self, temperature: float) -> str:
        """Get color name based on temperature range"""
        for spectrum in LightSpectrum:
            if spectrum.value[0] <= temperature < spectrum.value[1]:
                return spectrum.name.lower()
        return "ultraviolet"
    
    def temperature_to_wavelength(self, temperature: float) -> float:
        """
        Use Wien's displacement law to convert temperature to peak wavelength.
        λ_max = b / T where b ≈ 2.898 × 10^6 nm·K
        """
        WIEN_CONSTANT = 2.898e6  # nm·K
        # Ensure temperature is positive and meaningful
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}K")
        return WIEN_CONSTANT / temperature
    
    def emit_light(self, node: ast.AST, position: Tuple[int, int] = (0, 0)) -> LightEmission:
        """
        Generate light emission properties for a syntax node.
        This is where the "bioluminescence" happens!
        """
        semantic_weight = self.calculate_semantic_weight(node)
        complexity = self.calculate_complexity_score(node)
        
        # Light intensity is based on semantic weight
        intensity = semantic_weight
        
        # Temperature is based on complexity (blackbody radiation)
        temperature = self.complexity_to_temperature(complexity)
        
        # Convert to wavelength and color
        wavelength = self.temperature_to_wavelength(temperature)
        rgb = self.temperature_to_rgb(temperature)
        color_name = self.get_color_name(temperature)
        
        emission = LightEmission(
            node_type=type(node).__name__,
            intensity=intensity,
            temperature=temperature,
            wavelength=wavelength,
            color_name=color_name,
            rgb=rgb,
            semantic_weight=semantic_weight,
            complexity_score=complexity,
            position=position
        )
        
        self.emissions.append(emission)
        self.total_intensity += intensity
        self.max_complexity = max(self.max_complexity, complexity)
        
        return emission
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Parse code and analyze the entire syntax tree for light emissions.
        Returns comprehensive analysis of the "bioluminescent" syntax tree.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "error": str(e),
                "emissions": []
            }
        
        self.emissions = []
        self.total_intensity = 0.0
        self.max_complexity = 0.0
        
        # Traverse the AST and emit light for each node
        for node in ast.walk(tree):
            # Try to get position information
            line = getattr(node, 'lineno', 0)
            col = getattr(node, 'col_offset', 0)
            self.emit_light(node, (line, col))
        
        # Generate summary statistics
        avg_intensity = self.total_intensity / len(self.emissions) if self.emissions else 0
        
        # Count by color/temperature
        color_distribution = {}
        for emission in self.emissions:
            color_distribution[emission.color_name] = color_distribution.get(emission.color_name, 0) + 1
        
        # Find brightest nodes
        brightest = sorted(self.emissions, key=lambda e: e.intensity, reverse=True)[:5]
        hottest = sorted(self.emissions, key=lambda e: e.temperature, reverse=True)[:5]
        
        return {
            "total_nodes": len(self.emissions),
            "total_intensity": self.total_intensity,
            "average_intensity": avg_intensity,
            "max_complexity": self.max_complexity,
            "color_distribution": color_distribution,
            "brightest_nodes": [
                {
                    "type": e.node_type,
                    "intensity": e.intensity,
                    "position": e.position
                } for e in brightest
            ],
            "hottest_nodes": [
                {
                    "type": e.node_type,
                    "temperature": e.temperature,
                    "color": e.color_name,
                    "position": e.position
                } for e in hottest
            ],
            "emissions": self.emissions
        }
    
    def generate_ascii_visualization(self, code: str, max_width: int = 80) -> str:
        """
        Generate an ASCII art visualization of the bioluminescent syntax tree.
        """
        analysis = self.analyze_code(code)
        
        if "error" in analysis:
            return f"Error parsing code: {analysis['error']}"
        
        lines = ["=" * max_width]
        lines.append("🌟 BIOLUMINESCENT SYNTAX TREE VISUALIZATION 🌟")
        lines.append("=" * max_width)
        lines.append("")
        
        lines.append(f"Total Nodes: {analysis['total_nodes']}")
        lines.append(f"Total Light Intensity: {analysis['total_intensity']:.2f}")
        lines.append(f"Average Intensity: {analysis['average_intensity']:.2f}")
        lines.append(f"Max Complexity: {analysis['max_complexity']:.2f}")
        lines.append("")
        
        lines.append("Color Distribution (Blackbody Spectrum):")
        for color, count in sorted(analysis['color_distribution'].items()):
            bar = "█" * min(count, max_width - 30)
            lines.append(f"  {color.upper():12s} | {bar} ({count})")
        lines.append("")
        
        lines.append("✨ Brightest Nodes (Highest Semantic Weight):")
        for i, node in enumerate(analysis['brightest_nodes'], 1):
            lines.append(f"  {i}. {node['type']:20s} Intensity: {node['intensity']:.2f} at {node['position']}")
        lines.append("")
        
        lines.append("🔥 Hottest Nodes (Highest Complexity - Most Blue):")
        for i, node in enumerate(analysis['hottest_nodes'], 1):
            lines.append(f"  {i}. {node['type']:20s} Temp: {node['temperature']:6.0f}K ({node['color'].upper()}) at {node['position']}")
        lines.append("")
        
        lines.append("=" * max_width)
        lines.append("Legend:")
        lines.append("  🔴 Red/Orange (1000-3000K): Simple structures")
        lines.append("  🟡 Yellow (3000-4000K): Moderate complexity")
        lines.append("  ⚪ White (4000-6000K): High complexity")
        lines.append("  🔵 Blue (6000-10000K): Very high complexity")
        lines.append("=" * max_width)
        
        return "\n".join(lines)


def demonstrate_bioluminescent_ast():
    """Demonstration of the syntax tree light emission concept"""
    emitter = SyntaxTreeLightEmitter()
    
    # Example code to analyze
    example_code = """
def fibonacci(n):
    '''Calculate fibonacci number recursively'''
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class BioluminescentParser:
    def __init__(self, code):
        self.code = code
        self.tree = None
    
    def parse(self):
        self.tree = compile(self.code, '<string>', 'exec')
        return self.tree
"""
    
    print("🧬 Syntax Tree Light Emission - Bioluminescent Analysis 🧬")
    print()
    print("Analyzing code as a bioluminescent structure...")
    print("Each syntax node emits light based on:")
    print("  • Intensity = Semantic Weight (information content)")
    print("  • Color/Temperature = Complexity (via blackbody radiation)")
    print()
    
    # Generate and display visualization
    visualization = emitter.generate_ascii_visualization(example_code)
    print(visualization)
    
    # Show detailed emission data for a few nodes
    print("\n📊 Detailed Emission Data (Sample):")
    print("=" * 80)
    
    analysis = emitter.analyze_code(example_code)
    for i, emission in enumerate(analysis['emissions'][:10]):
        print(f"\nNode {i+1}: {emission.node_type}")
        print(f"  Position: Line {emission.position[0]}, Col {emission.position[1]}")
        print(f"  Intensity: {emission.intensity:.3f} (semantic weight: {emission.semantic_weight:.3f})")
        print(f"  Temperature: {emission.temperature:.0f}K")
        print(f"  Color: {emission.color_name.upper()} (RGB: {emission.rgb})")
        print(f"  Wavelength: {emission.wavelength:.1f} nm")
        print(f"  Complexity: {emission.complexity_score:.3f}")
    
    print("\n" + "=" * 80)
    print("✨ The syntax tree glows with bioluminescent beauty! ✨")


if __name__ == "__main__":
    demonstrate_bioluminescent_ast()
