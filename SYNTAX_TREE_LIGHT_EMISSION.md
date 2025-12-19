# Syntax Tree Light Emission - Bioluminescent AST Analysis

## Overview

This module implements a creative visualization system that treats syntax trees (Abstract Syntax Trees) as **bioluminescent structures** that emit light with properties similar to **blackbody radiation** in physics.

## Concept

The idea behind "Syntax tree emits light. Bioluminescent. Blackbody radiation" is to visualize code structure in a way that:

1. **Each syntax node emits light** - Like bioluminescent organisms in nature
2. **Light intensity** represents semantic weight (information content)
3. **Light color/temperature** represents structural complexity (via blackbody radiation spectrum)

## Physical Analogy

### Blackbody Radiation
In physics, blackbody radiation describes how objects emit electromagnetic radiation based on their temperature:
- **Cool objects (1000-3000K)**: Emit red/orange light
- **Moderate temperature (3000-4000K)**: Emit yellow light  
- **Hot objects (4000-6000K)**: Emit white light
- **Very hot objects (6000-10000K)**: Emit blue/white light

### Application to Syntax Trees
We map this physics concept to code structure:
- **Simple syntax structures**: Low complexity → Cool temperature → Red/Orange glow
- **Complex nested structures**: High complexity → Hot temperature → Blue/White glow

## Implementation

### Light Emission Properties

Each syntax node has these emission properties:

```python
@dataclass
class LightEmission:
    node_type: str              # Type of syntax node (e.g., FunctionDef, If, etc.)
    intensity: float            # 0.0 to 1.0 (based on semantic weight)
    temperature: float          # Kelvin (1000-10000K based on complexity)
    wavelength: float           # nanometers (via Wien's displacement law)
    color_name: str            # Human-readable color name
    rgb: Tuple[int, int, int]  # RGB color values
    semantic_weight: float     # Information content score
    complexity_score: float    # Structural complexity score
    position: Tuple[int, int]  # Line and column in source
```

### Semantic Weight Calculation

Semantic weight represents how much "meaning" or information a node carries:

- **High weight (0.8-1.0)**: Function definitions, class definitions, control flow
- **Medium weight (0.5-0.7)**: Assignments, function calls, loops
- **Low weight (0.3-0.5)**: Variables, constants, operators

### Complexity Score Calculation

Complexity is based on:
- **Node count**: Number of child nodes in the subtree
- **Depth**: How deeply nested the structure is
- **Combined metric**: `(node_count * 0.1 + depth * 0.5) / 10.0`

### Temperature Mapping

Temperature follows the blackbody spectrum:
```
Complexity → Temperature → Color
0.1        → 1900K       → Red
0.3        → 3700K       → Orange/Yellow
0.5        → 5500K       → White
0.7        → 7300K       → Blue
0.9        → 9100K       → Blue-White
```

### Color Spectrum

Based on blackbody radiation:

| Temperature Range | Color Name | RGB Approximation | Complexity Level |
|-------------------|------------|-------------------|------------------|
| 700-1000K         | Infrared   | Dark Red          | Minimal          |
| 1000-2000K        | Red        | Bright Red        | Very Low         |
| 2000-3000K        | Orange     | Orange            | Low              |
| 3000-4000K        | Yellow     | Yellow            | Moderate         |
| 4000-6000K        | White      | White             | High             |
| 6000-10000K       | Blue       | Blue-White        | Very High        |
| 10000-50000K      | Ultraviolet| Pale Blue-White   | Extreme          |

## Usage

### Basic Analysis

```python
from SyntaxTreeLightEmitter import SyntaxTreeLightEmitter

emitter = SyntaxTreeLightEmitter()

code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Analyze the code
analysis = emitter.analyze_code(code)

print(f"Total nodes: {analysis['total_nodes']}")
print(f"Total intensity: {analysis['total_intensity']:.2f}")
print(f"Max complexity: {analysis['max_complexity']:.2f}")
```

### ASCII Visualization

```python
# Generate visual representation
visualization = emitter.generate_ascii_visualization(code)
print(visualization)
```

Output:
```
================================================================================
🌟 BIOLUMINESCENT SYNTAX TREE VISUALIZATION 🌟
================================================================================

Total Nodes: 15
Total Light Intensity: 8.50
Average Intensity: 0.57
Max Complexity: 0.85

Color Distribution (Blackbody Spectrum):
  RED          | ████████████ (12)
  ORANGE       | ███ (3)
  BLUE         | █ (1)

✨ Brightest Nodes (Highest Semantic Weight):
  1. FunctionDef          Intensity: 1.00 at (1, 0)
  2. If                   Intensity: 0.70 at (2, 4)
  ...
```

### Detailed Emission Data

```python
# Access individual emissions
for emission in analysis['emissions'][:5]:
    print(f"{emission.node_type}:")
    print(f"  Color: {emission.color_name} ({emission.temperature:.0f}K)")
    print(f"  Intensity: {emission.intensity:.2f}")
    print(f"  RGB: {emission.rgb}")
```

## Analysis Results

The analyzer provides:

### Overall Statistics
- **total_nodes**: Number of syntax nodes analyzed
- **total_intensity**: Sum of all light intensities
- **average_intensity**: Mean intensity across all nodes
- **max_complexity**: Highest complexity score found

### Color Distribution
Dictionary showing count of nodes at each temperature/color:
```python
{
    'red': 12,
    'orange': 5,
    'yellow': 3,
    'white': 4,
    'blue': 2
}
```

### Brightest Nodes
Top 5 nodes with highest semantic weight (most information-rich):
```python
[
    {'type': 'FunctionDef', 'intensity': 1.0, 'position': (1, 0)},
    {'type': 'ClassDef', 'intensity': 1.0, 'position': (10, 0)},
    ...
]
```

### Hottest Nodes
Top 5 nodes with highest temperature (most complex):
```python
[
    {'type': 'Module', 'temperature': 10000, 'color': 'ultraviolet', 'position': (0, 0)},
    {'type': 'FunctionDef', 'temperature': 7300, 'color': 'blue', 'position': (2, 0)},
    ...
]
```

## Applications

### 1. Code Complexity Visualization
Quickly identify complex parts of code by looking at "hot spots" (blue/white regions).

### 2. Semantic Density Analysis
Find information-rich code sections by identifying bright nodes.

### 3. Code Quality Metrics
- High average temperature → Complex codebase (might need refactoring)
- Many red nodes with high intensity → Simple but meaningful code
- Ultraviolet/extreme temperatures → Potentially over-complicated structures

### 4. Educational Tool
Teach students about:
- AST structure and parsing
- Code complexity metrics
- Physics concepts (blackbody radiation, Wien's law)
- Data visualization techniques

### 5. Code Review Aid
During code reviews, use the visualization to:
- Identify overly complex functions (blue/white nodes)
- Find simple but important code (bright red nodes)
- Balance complexity across the codebase

## Scientific Concepts Applied

### Wien's Displacement Law
```
λ_max = b / T
```
Where:
- λ_max = peak wavelength
- b = Wien's displacement constant (2.898 × 10^6 nm·K)
- T = temperature in Kelvin

This law determines the wavelength of peak emission based on temperature.

### Blackbody Radiation Spectrum
The Planck distribution describes how objects emit light at different wavelengths based on temperature. We approximate this with RGB color conversion.

### Information Theory
Semantic weight is analogous to information content in Claude Shannon's information theory - nodes that carry more "meaning" or make larger semantic contributions have higher weight.

## Examples

### Simple Code (Red/Orange Glow)
```python
x = 42
y = x + 1
print(y)
```
- Low complexity
- Cool temperatures (1000-2000K)
- Red/Orange colors
- Simple structure

### Complex Code (Blue/White Glow)
```python
class DataProcessor:
    def process(self, data):
        return [
            self.transform(item)
            for item in data
            if self.validate(item)
            for result in self.analyze(item)
            if result.score > threshold
        ]
```
- High complexity
- Hot temperatures (6000-8000K)
- Blue/White colors
- Nested structures

## Metaphorical Interpretation

The "bioluminescent" aspect draws parallels to nature:

1. **Deep sea creatures**: Glow to communicate and navigate → Code nodes glow to reveal structure
2. **Fireflies**: Synchronized flashing → Related nodes have similar emission patterns
3. **Bioluminescent algae**: Glow brighter when disturbed → Complex code glows hotter
4. **Anglerfish lure**: Bright light attracts attention → High-intensity nodes are important

## Future Enhancements

Possible extensions:

1. **Pulsing/Animation**: Nodes pulse at different rates based on execution frequency
2. **Interactive Visualization**: Click nodes to explore in 3D space
3. **Real-time Analysis**: Analyze code as you type
4. **Comparative Analysis**: Compare two code versions to see changes in "light emission"
5. **Sound Generation**: Map wavelengths to audio frequencies for "hearing" code structure
6. **Machine Learning**: Train models to identify code smells based on emission patterns

## Philosophy

This system embodies the idea that code is not just text, but a living, glowing structure. Each node contributes to the overall "light" of the program, creating a unique signature that reveals:

- **Structure** through color temperature
- **Meaning** through intensity
- **Complexity** through spectral distribution
- **Beauty** through visualization

"The syntax tree emits light" - because understanding code structure is about illuminating the relationships and patterns within.

## References

- **Blackbody Radiation**: Planck, M. (1900). "On the Law of Distribution of Energy in the Normal Spectrum"
- **Wien's Law**: Wien, W. (1893). "Eine neue Beziehung der Strahlung schwarzer Körper zum zweiten Hauptsatz der Wärmetheorie"
- **Abstract Syntax Trees**: Aho, A. V., Sethi, R., & Ullman, J. D. (1986). "Compilers: Principles, Techniques, and Tools"
- **Information Theory**: Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- **Bioluminescence**: Wilson, T., & Hastings, J. W. (2013). "Bioluminescence: Living Lights, Lights for Living"

---

*"In the darkness of complexity, the syntax tree lights the way."* ✨
