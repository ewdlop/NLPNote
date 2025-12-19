# 🎄 Supernatural Christmas Tree Generator 🎄

A creative NLP visualization project that grows a Christmas tree based on botanical genus taxonomy using discriminated unions (sum types) from type theory.

## Overview

This project combines three fascinating concepts:

1. **Botanical Taxonomy**: Real botanical genera used for Christmas trees (Abies, Picea, Pinus)
2. **Discriminated Unions**: Type theory concept also known as tagged unions or sum types
3. **Supernatural Elements**: Magical ornaments, glowing effects, and mystical properties

## Features

- 🌲 **Botanical Accuracy**: Based on real Christmas tree genera from the Pinaceae family
- 🎨 **Discriminated Union Types**: Demonstrates type-safe node representation
- ✨ **Supernatural Growth**: Trees grow with magical ornaments and effects
- 📊 **ASCII Visualization**: Beautiful terminal-based tree rendering
- 🔍 **Type Filtering**: Query nodes by their discriminator type
- 📈 **Tree Statistics**: Track growth, ornaments, and magical aura

## Quick Start

### Basic Usage

```python
from SupernaturalChristmasTree import SupernaturalChristmasTree

# Create a tree from Fir genus
tree = SupernaturalChristmasTree("Abies")

# Grow the tree with supernatural ornaments
tree.grow(iterations=3)

# Visualize the tree
print(tree.visualize_ascii())

# Get statistics
stats = tree.get_tree_statistics()
print(f"Height: {stats['height']} levels")
print(f"Ornaments: {stats['num_ornaments']}")
print(f"Magical Aura: {stats['magical_aura']:.2%}")
```

### Run the Demo

```bash
python demo_supernatural_christmas_tree.py
```

### Run Tests

```bash
pytest test_supernatural_christmas_tree.py -v
```

## Discriminated Unions Explained

A **discriminated union** (also called a tagged union or sum type) is a data structure that can hold values of different types, with a "tag" that identifies which type is currently being held.

### In This Project

```python
# TreeNode is a discriminated union of three types
TreeNode = Union[GenusNode, SpeciesNode, OrnamentNode]

# Each type has a discriminator tag
class TreeNodeType(Enum):
    GENUS = "genus"      # Root of the tree
    SPECIES = "species"  # Branches
    ORNAMENT = "ornament"  # Decorations
```

### Type-Safe Filtering

```python
# Filter nodes by type using the discriminator
genus_nodes = tree.filter_by_type(TreeNodeType.GENUS)
species_nodes = tree.filter_by_type(TreeNodeType.SPECIES)
ornament_nodes = tree.filter_by_type(TreeNodeType.ORNAMENT)
```

## Botanical Genera

The following Christmas tree genera are included:

### Abies (Fir)
- **Family**: Pinaceae
- **Common Name**: Fir
- **Species**: Silver Fir, Balsam Fir, Fraser Fir, Nordmann Fir, Noble Fir
- **Magical Power**: 0.9

### Picea (Spruce)
- **Family**: Pinaceae
- **Common Name**: Spruce
- **Species**: Norway Spruce, Blue Spruce, White Spruce, Serbian Spruce
- **Magical Power**: 0.85

### Pinus (Pine)
- **Family**: Pinaceae
- **Common Name**: Pine
- **Species**: Scots Pine, Austrian Pine, White Pine
- **Magical Power**: 0.8

## Node Types

### GenusNode
Represents the botanical genus (root of the tree).

**Properties**:
- `name`: Genus name (e.g., "Abies")
- `common_name`: Common name (e.g., "Fir")
- `family`: Botanical family (e.g., "Pinaceae")
- `magical_power`: Supernatural strength (0.0 to 1.0)

### SpeciesNode
Represents a botanical species (branches of the tree).

**Properties**:
- `name`: Full species name (e.g., "Abies alba")
- `genus`: Parent genus
- `height_level`: Position in tree hierarchy
- `supernatural_glow`: Mystical glow intensity (0.0 to 1.0)
- `characteristics`: List of species traits

### OrnamentNode
Represents a supernatural ornament (decorations on the tree).

**Properties**:
- `color`: Ornament color (🔴🟡🔵🟢🟣⚪⚫)
- `effect`: Magical effect (✨⭐❄️💫🌟✦)
- `intensity`: Effect strength (0.0 to 1.0)
- `semantic_meaning`: NLP-derived meaning
- `position`: Location on tree (level, offset)

## API Reference

### SupernaturalChristmasTree

Main class for creating and growing supernatural Christmas trees.

#### Constructor

```python
tree = SupernaturalChristmasTree(genus_name: str = "Abies")
```

**Parameters**:
- `genus_name`: Botanical genus ("Abies", "Picea", or "Pinus")

#### Methods

##### `grow(iterations: int = 1)`
Grow the tree by adding supernatural ornaments and enhancing magical aura.

```python
tree.grow(iterations=5)
```

##### `visualize_ascii() -> str`
Generate an ASCII art visualization of the tree.

```python
visualization = tree.visualize_ascii()
print(visualization)
```

##### `get_tree_statistics() -> Dict[str, Any]`
Get comprehensive statistics about the tree.

```python
stats = tree.get_tree_statistics()
# Returns: genus, common_name, family, height, num_species,
#          num_ornaments, magical_aura, species_list
```

##### `get_all_nodes() -> List[TreeNode]`
Get all nodes as a discriminated union list.

```python
all_nodes = tree.get_all_nodes()
```

##### `filter_by_type(node_type: TreeNodeType) -> List[TreeNode]`
Filter nodes by discriminator type.

```python
genus_nodes = tree.filter_by_type(TreeNodeType.GENUS)
species_nodes = tree.filter_by_type(TreeNodeType.SPECIES)
ornament_nodes = tree.filter_by_type(TreeNodeType.ORNAMENT)
```

##### `describe_discriminated_union() -> str`
Get an educational explanation of discriminated unions in this context.

```python
description = tree.describe_discriminated_union()
print(description)
```

### BotanicalDatabase

Static database of botanical genera for Christmas trees.

#### Class Methods

##### `get_genus_data(genus_name: str) -> Optional[Dict[str, Any]]`
Retrieve genus data from the database.

```python
data = BotanicalDatabase.get_genus_data("Abies")
```

##### `list_genera() -> List[str]`
List all available genera.

```python
genera = BotanicalDatabase.list_genera()
# Returns: ["Abies", "Picea", "Pinus"]
```

## Examples

### Example 1: Basic Tree Creation

```python
from SupernaturalChristmasTree import SupernaturalChristmasTree

# Create a Fir tree
tree = SupernaturalChristmasTree("Abies")
print(tree.visualize_ascii())
```

### Example 2: Tree Growth and Statistics

```python
tree = SupernaturalChristmasTree("Picea")

# Initial state
print("Before growth:")
print(f"Ornaments: {tree.get_tree_statistics()['num_ornaments']}")

# Grow the tree
tree.grow(iterations=5)

# After growth
print("\nAfter growth:")
stats = tree.get_tree_statistics()
print(f"Ornaments: {stats['num_ornaments']}")
print(f"Magical Aura: {stats['magical_aura']:.2%}")
```

### Example 3: Type-Safe Node Filtering

```python
tree = SupernaturalChristmasTree("Abies")
tree.grow(iterations=3)

# Get all nodes
all_nodes = tree.get_all_nodes()
print(f"Total nodes: {len(all_nodes)}")

# Filter by type
for node_type in TreeNodeType:
    nodes = tree.filter_by_type(node_type)
    print(f"{node_type.value}: {len(nodes)} nodes")
```

### Example 4: Compare Multiple Genera

```python
from SupernaturalChristmasTree import SupernaturalChristmasTree, BotanicalDatabase

# Create trees from all genera
trees = {}
for genus in BotanicalDatabase.list_genera():
    tree = SupernaturalChristmasTree(genus)
    tree.grow(iterations=3)
    trees[genus] = tree

# Compare statistics
for genus, tree in trees.items():
    stats = tree.get_tree_statistics()
    print(f"{genus}: {stats['height']} levels, {stats['num_ornaments']} ornaments")
```

## Conceptual Background

### Discriminated Unions in Type Theory

Discriminated unions are a fundamental concept in type theory and functional programming:

- **Rust**: `enum` with data
- **TypeScript**: Discriminated unions
- **F#/OCaml**: Discriminated unions
- **Haskell**: Algebraic data types (ADTs)
- **C++**: `std::variant`
- **Python**: `Union` with type tags

### Why Use Discriminated Unions?

1. **Type Safety**: The compiler/runtime can verify which type you're working with
2. **Exhaustiveness**: Ensures all cases are handled
3. **Self-Documenting**: The type itself documents what values are possible
4. **Pattern Matching**: Enables elegant case analysis

### In This Project

We use discriminated unions to represent different node types in the tree:

```python
# Each node type has a tag (discriminator)
@dataclass
class GenusNode:
    node_type: TreeNodeType = TreeNodeType.GENUS
    # ... other fields

@dataclass
class SpeciesNode:
    node_type: TreeNodeType = TreeNodeType.SPECIES
    # ... other fields

@dataclass
class OrnamentNode:
    node_type: TreeNodeType = TreeNodeType.ORNAMENT
    # ... other fields

# Union type
TreeNode = Union[GenusNode, SpeciesNode, OrnamentNode]
```

## ASCII Art Examples

Here's what a supernatural Christmas tree looks like:

```
        ⭐

      🌲
     🌲🌲🌲
    🌲🌲🌲🌲🌲
  🔴✨🌲🌲🌲🌲🌲🌲🌲
 🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲
  (Abies alba)
🟡💫🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲
🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲
      🟫🟫🟫
      🟫🟫🟫

Magical Aura: [✨✨✨✨✨✨✨✨] 85%
```

## Contributing

Contributions are welcome! Some ideas for enhancements:

- Add more botanical genera (e.g., Pseudotsuga, Tsuga)
- Implement more sophisticated growth algorithms
- Add seasonal variations (spring blooms, autumn colors)
- Create 3D visualization support
- Add sound effects for magical moments
- Implement tree "moods" based on NLP sentiment analysis

## License

This project is part of the NLPNote repository.

## Related Concepts

- **Botanical Taxonomy**: The science of classifying plants
- **Type Theory**: Mathematical study of type systems
- **Sum Types**: Types that can be one of several variants
- **Tagged Unions**: Unions with explicit type tags
- **Algebraic Data Types**: Composite types formed by combining other types

## References

- [Botanical Classification](https://en.wikipedia.org/wiki/Botanical_nomenclature)
- [Discriminated Unions](https://en.wikipedia.org/wiki/Tagged_union)
- [Pinaceae Family](https://en.wikipedia.org/wiki/Pinaceae)
- [Christmas Tree Types](https://en.wikipedia.org/wiki/Christmas_tree#Tree_species)

---

🎄 **Merry Christmas from the Supernatural Tree!** 🎄
