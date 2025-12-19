#!/usr/bin/env python3
"""
Supernatural Christmas Tree Generator
A creative visualization system that grows a Christmas tree based on botanical genus taxonomy
using discriminated unions (sum types) to represent different node types.

Concept: The tree grows from a botanical genus, with each species becoming branches,
and supernatural ornaments appearing based on linguistic/semantic properties.
"""

import random
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TreeNodeType(Enum):
    """Discriminated union tag for different tree node types"""
    GENUS = "genus"
    SPECIES = "species"
    ORNAMENT = "ornament"


class OrnamentColor(Enum):
    """Color spectrum for supernatural ornaments"""
    RED = "🔴"
    GOLD = "🟡"
    BLUE = "🔵"
    GREEN = "🟢"
    PURPLE = "🟣"
    WHITE = "⚪"
    SILVER = "⚫"


class MagicalEffect(Enum):
    """Supernatural effects that can appear on the tree"""
    SPARKLE = "✨"
    STAR = "⭐"
    SNOWFLAKE = "❄️"
    GLOW = "💫"
    SHINE = "🌟"
    TWINKLE = "✦"


@dataclass
class GenusNode:
    """Represents a botanical genus (root of the tree)"""
    name: str = ""
    common_name: str = ""
    family: str = ""
    magical_power: float = 0.0  # 0.0 to 1.0
    
    @property
    def node_type(self) -> TreeNodeType:
        """Node type is always GENUS for GenusNode"""
        return TreeNodeType.GENUS
    

@dataclass
class SpeciesNode:
    """Represents a botanical species (branches of the tree)"""
    name: str = ""
    genus: str = ""
    characteristics: List[str] = None
    height_level: int = 0  # Position in tree hierarchy
    supernatural_glow: float = 0.0
    
    @property
    def node_type(self) -> TreeNodeType:
        """Node type is always SPECIES for SpeciesNode"""
        return TreeNodeType.SPECIES
    
    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = []


@dataclass
class OrnamentNode:
    """Represents a supernatural ornament on the tree"""
    color: OrnamentColor = OrnamentColor.RED
    effect: MagicalEffect = MagicalEffect.SPARKLE
    intensity: float = 0.0
    semantic_meaning: str = ""
    position: Tuple[int, int] = (0, 0)
    
    @property
    def node_type(self) -> TreeNodeType:
        """Node type is always ORNAMENT for OrnamentNode"""
        return TreeNodeType.ORNAMENT


# Discriminated Union type for tree nodes
TreeNode = Union[GenusNode, SpeciesNode, OrnamentNode]


class BotanicalDatabase:
    """Database of botanical genera for Christmas trees"""
    
    CHRISTMAS_TREE_GENERA = {
        "Abies": {
            "common_name": "Fir",
            "family": "Pinaceae",
            "species": [
                "Abies alba",      # Silver Fir
                "Abies balsamea",  # Balsam Fir
                "Abies fraseri",   # Fraser Fir
                "Abies nordmanniana",  # Nordmann Fir
                "Abies procera",   # Noble Fir
            ],
            "magical_power": 0.9
        },
        "Picea": {
            "common_name": "Spruce",
            "family": "Pinaceae",
            "species": [
                "Picea abies",     # Norway Spruce
                "Picea pungens",   # Blue Spruce
                "Picea glauca",    # White Spruce
                "Picea omorika",   # Serbian Spruce
            ],
            "magical_power": 0.85
        },
        "Pinus": {
            "common_name": "Pine",
            "family": "Pinaceae",
            "species": [
                "Pinus sylvestris",  # Scots Pine
                "Pinus nigra",       # Austrian Pine
                "Pinus strobus",     # White Pine
            ],
            "magical_power": 0.8
        },
    }
    
    @classmethod
    def get_genus_data(cls, genus_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve genus data from the database"""
        return cls.CHRISTMAS_TREE_GENERA.get(genus_name)
    
    @classmethod
    def list_genera(cls) -> List[str]:
        """List all available genera"""
        return list(cls.CHRISTMAS_TREE_GENERA.keys())


class SupernaturalChristmasTree:
    """
    Grows a supernatural Christmas tree based on botanical genus taxonomy.
    Uses discriminated unions to represent different node types.
    """
    
    def __init__(self, genus_name: str = "Abies"):
        self.genus_name = genus_name
        self.root: Optional[GenusNode] = None
        self.species_nodes: List[SpeciesNode] = []
        self.ornaments: List[OrnamentNode] = []
        self.height = 0
        self.magical_aura = 0.0
        
        # Initialize the tree
        self._initialize_tree()
    
    def _initialize_tree(self):
        """Initialize the tree from botanical database"""
        genus_data = BotanicalDatabase.get_genus_data(self.genus_name)
        
        if not genus_data:
            # Default to Fir if genus not found
            self.genus_name = "Abies"
            genus_data = BotanicalDatabase.get_genus_data("Abies")
        
        # Create root genus node
        self.root = GenusNode(
            name=self.genus_name,
            common_name=genus_data["common_name"],
            family=genus_data["family"],
            magical_power=genus_data["magical_power"]
        )
        
        # Grow species branches
        species_list = genus_data["species"]
        for i, species_name in enumerate(species_list):
            species_node = SpeciesNode(
                name=species_name,
                genus=self.genus_name,
                height_level=i + 1,
                supernatural_glow=random.uniform(0.5, 1.0)
            )
            self.species_nodes.append(species_node)
        
        self.height = len(self.species_nodes)
        self.magical_aura = self.root.magical_power
    
    def grow(self, iterations: int = 1):
        """Grow the tree by adding more ornaments and magical effects"""
        for _ in range(iterations):
            self._add_supernatural_ornaments()
            self._enhance_magical_aura()
    
    def _add_supernatural_ornaments(self):
        """Add supernatural ornaments to the tree based on semantic properties"""
        num_ornaments = random.randint(1, 3)
        
        for _ in range(num_ornaments):
            # Choose random position
            level = random.randint(1, self.height)
            position = random.randint(0, level)
            
            # Choose random color and effect
            color = random.choice(list(OrnamentColor))
            effect = random.choice(list(MagicalEffect))
            
            # Calculate intensity based on position (higher = more intense)
            intensity = (self.height - level + 1) / self.height
            
            ornament = OrnamentNode(
                color=color,
                effect=effect,
                intensity=intensity,
                semantic_meaning=f"Beauty at level {level}",
                position=(level, position)
            )
            
            self.ornaments.append(ornament)
    
    def _enhance_magical_aura(self):
        """Enhance the tree's magical aura over time"""
        self.magical_aura = min(1.0, self.magical_aura * 1.1)
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get statistics about the supernatural tree"""
        return {
            "genus": self.root.name if self.root else "Unknown",
            "common_name": self.root.common_name if self.root else "Unknown",
            "family": self.root.family if self.root else "Unknown",
            "height": self.height,
            "num_species": len(self.species_nodes),
            "num_ornaments": len(self.ornaments),
            "magical_aura": round(self.magical_aura, 2),
            "species_list": [s.name for s in self.species_nodes]
        }
    
    def visualize_ascii(self) -> str:
        """Generate an ASCII art visualization of the supernatural Christmas tree"""
        lines = []
        
        # Add star at top
        lines.append(" " * (self.height + 2) + MagicalEffect.STAR.value)
        lines.append("")
        
        # Build tree from top to bottom
        for level in range(1, self.height + 1):
            # Calculate width at this level
            width = level * 2 + 1
            padding = self.height - level + 1
            
            # Create the tree branch with species info
            if level <= len(self.species_nodes):
                species = self.species_nodes[level - 1]
                branch = "🌲" * width
                
                # Add ornaments at this level
                ornaments_here = [o for o in self.ornaments if o.position[0] == level]
                if ornaments_here:
                    decorations = "".join([o.color.value + o.effect.value for o in ornaments_here[:3]])
                    branch = decorations + branch
                
                line = " " * padding + branch
                lines.append(line)
                
                # Add species name as a comment
                if random.random() < 0.3:  # Don't show all names
                    species_label = f"  ({species.name})"
                    lines.append(" " * (padding + width + 2) + species_label)
        
        # Add trunk
        trunk_width = 3
        trunk_padding = self.height - 1
        for _ in range(2):
            lines.append(" " * trunk_padding + "🟫" * trunk_width)
        
        # Add magical aura indicator
        lines.append("")
        aura_bar = "✨" * int(self.magical_aura * 10)
        lines.append(f"Magical Aura: [{aura_bar}] {self.magical_aura:.1%}")
        
        return "\n".join(lines)
    
    def describe_discriminated_union(self) -> str:
        """
        Describe how discriminated unions are used in this tree structure.
        Educational explanation of the type theory concept.
        """
        description = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║          DISCRIMINATED UNION IN SUPERNATURAL CHRISTMAS TREE               ║
╚═══════════════════════════════════════════════════════════════════════════╝

In type theory, a discriminated union (also called a tagged union or sum type)
is a data structure that can hold values of different types, with a "tag" that
identifies which type is currently being held.

In our Supernatural Christmas Tree:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TreeNode = Union[GenusNode, SpeciesNode, OrnamentNode]

├── GenusNode (tag: GENUS)
│   └── Represents the root: {self.root.name if self.root else 'Unknown'}
│       Properties: family, common_name, magical_power
│
├── SpeciesNode (tag: SPECIES)  
│   └── Represents branches: {len(self.species_nodes)} species
│       Properties: name, genus, height_level, supernatural_glow
│       Examples: {', '.join([s.name for s in self.species_nodes[:2]])}...
│
└── OrnamentNode (tag: ORNAMENT)
    └── Represents decorations: {len(self.ornaments)} ornaments
        Properties: color, effect, intensity, position
        Effects: sparkle, glow, shine, twinkle

The 'node_type' field acts as the discriminator/tag, allowing us to safely
determine which type of node we're working with at runtime.

This is similar to:
- Rust's enum with data
- TypeScript's discriminated unions
- F#/OCaml's discriminated unions
- Haskell's algebraic data types (ADTs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return description
    
    def get_all_nodes(self) -> List[TreeNode]:
        """Get all nodes as a discriminated union list"""
        nodes: List[TreeNode] = []
        
        # Add genus node
        if self.root:
            nodes.append(self.root)
        
        # Add species nodes
        nodes.extend(self.species_nodes)
        
        # Add ornament nodes
        nodes.extend(self.ornaments)
        
        return nodes
    
    def filter_by_type(self, node_type: TreeNodeType) -> List[TreeNode]:
        """Filter nodes by discriminator type"""
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.node_type == node_type]


def demo_supernatural_christmas_tree():
    """Demo function showing the supernatural Christmas tree in action"""
    print("🎄 SUPERNATURAL CHRISTMAS TREE GENERATOR 🎄")
    print("=" * 80)
    print()
    
    # List available genera
    print("Available Botanical Genera for Christmas Trees:")
    for genus in BotanicalDatabase.list_genera():
        data = BotanicalDatabase.get_genus_data(genus)
        print(f"  • {genus} ({data['common_name']}) - Magical Power: {data['magical_power']}")
    print()
    
    # Create a supernatural Christmas tree
    print("Creating a supernatural Christmas tree from genus 'Abies' (Fir)...")
    tree = SupernaturalChristmasTree("Abies")
    print()
    
    # Show initial state
    print("Initial Tree Statistics:")
    stats = tree.get_tree_statistics()
    for key, value in stats.items():
        if key != "species_list":
            print(f"  {key}: {value}")
    print()
    
    # Grow the tree
    print("Growing the tree with supernatural ornaments...")
    tree.grow(iterations=3)
    print()
    
    # Show updated statistics
    print("After Growth Statistics:")
    stats = tree.get_tree_statistics()
    for key, value in stats.items():
        if key != "species_list":
            print(f"  {key}: {value}")
    print()
    
    # Visualize the tree
    print("ASCII Visualization:")
    print(tree.visualize_ascii())
    print()
    
    # Explain discriminated unions
    print(tree.describe_discriminated_union())
    
    # Show node filtering
    print("\n" + "=" * 80)
    print("DISCRIMINATED UNION FILTERING EXAMPLE:")
    print("=" * 80)
    
    genus_nodes = tree.filter_by_type(TreeNodeType.GENUS)
    print(f"\nGenus Nodes: {len(genus_nodes)}")
    for node in genus_nodes:
        print(f"  • {node.name} ({node.common_name}) - Family: {node.family}")
    
    species_nodes = tree.filter_by_type(TreeNodeType.SPECIES)
    print(f"\nSpecies Nodes: {len(species_nodes)}")
    for node in species_nodes[:3]:
        print(f"  • {node.name} - Glow: {node.supernatural_glow:.2f}")
    
    ornament_nodes = tree.filter_by_type(TreeNodeType.ORNAMENT)
    print(f"\nOrnament Nodes: {len(ornament_nodes)}")
    for node in ornament_nodes[:5]:
        print(f"  • {node.color.value} {node.effect.value} at position {node.position}")
    
    print("\n" + "=" * 80)
    print("🎅 Merry Christmas from the Supernatural Tree! 🎅")
    print("=" * 80)


if __name__ == "__main__":
    demo_supernatural_christmas_tree()
