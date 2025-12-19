#!/usr/bin/env python3
"""
Tests for the Supernatural Christmas Tree Generator
"""

import pytest
from SupernaturalChristmasTree import (
    SupernaturalChristmasTree,
    BotanicalDatabase,
    TreeNodeType,
    GenusNode,
    SpeciesNode,
    OrnamentNode,
    OrnamentColor,
    MagicalEffect
)


class TestBotanicalDatabase:
    """Tests for the BotanicalDatabase class"""
    
    def test_list_genera(self):
        """Test listing all available genera"""
        genera = BotanicalDatabase.list_genera()
        assert len(genera) > 0
        assert "Abies" in genera
        assert "Picea" in genera
        assert "Pinus" in genera
    
    def test_get_genus_data_valid(self):
        """Test retrieving valid genus data"""
        data = BotanicalDatabase.get_genus_data("Abies")
        assert data is not None
        assert "common_name" in data
        assert "family" in data
        assert "species" in data
        assert "magical_power" in data
        assert data["common_name"] == "Fir"
        assert data["family"] == "Pinaceae"
    
    def test_get_genus_data_invalid(self):
        """Test retrieving invalid genus data"""
        data = BotanicalDatabase.get_genus_data("InvalidGenus")
        assert data is None
    
    def test_genus_data_structure(self):
        """Test the structure of genus data"""
        for genus_name in BotanicalDatabase.list_genera():
            data = BotanicalDatabase.get_genus_data(genus_name)
            assert isinstance(data["species"], list)
            assert len(data["species"]) > 0
            assert 0.0 <= data["magical_power"] <= 1.0


class TestTreeNodes:
    """Tests for discriminated union tree node types"""
    
    def test_genus_node_creation(self):
        """Test creating a GenusNode"""
        node = GenusNode(
            name="Abies",
            common_name="Fir",
            family="Pinaceae",
            magical_power=0.9
        )
        assert node.node_type == TreeNodeType.GENUS
        assert node.name == "Abies"
        assert node.common_name == "Fir"
        assert node.family == "Pinaceae"
        assert node.magical_power == 0.9
    
    def test_species_node_creation(self):
        """Test creating a SpeciesNode"""
        node = SpeciesNode(
            name="Abies alba",
            genus="Abies",
            height_level=1,
            supernatural_glow=0.8
        )
        assert node.node_type == TreeNodeType.SPECIES
        assert node.name == "Abies alba"
        assert node.genus == "Abies"
        assert node.height_level == 1
        assert node.supernatural_glow == 0.8
        assert node.characteristics == []
    
    def test_ornament_node_creation(self):
        """Test creating an OrnamentNode"""
        node = OrnamentNode(
            color=OrnamentColor.RED,
            effect=MagicalEffect.SPARKLE,
            intensity=0.7,
            semantic_meaning="Beauty",
            position=(1, 2)
        )
        assert node.node_type == TreeNodeType.ORNAMENT
        assert node.color == OrnamentColor.RED
        assert node.effect == MagicalEffect.SPARKLE
        assert node.intensity == 0.7
        assert node.semantic_meaning == "Beauty"
        assert node.position == (1, 2)


class TestSupernaturalChristmasTree:
    """Tests for the SupernaturalChristmasTree class"""
    
    def test_tree_initialization_default(self):
        """Test tree initialization with default genus"""
        tree = SupernaturalChristmasTree()
        assert tree.genus_name == "Abies"
        assert tree.root is not None
        assert tree.root.node_type == TreeNodeType.GENUS
        assert len(tree.species_nodes) > 0
        assert tree.height > 0
    
    def test_tree_initialization_specific_genus(self):
        """Test tree initialization with specific genus"""
        tree = SupernaturalChristmasTree("Picea")
        assert tree.genus_name == "Picea"
        assert tree.root.name == "Picea"
        assert tree.root.common_name == "Spruce"
    
    def test_tree_initialization_invalid_genus(self):
        """Test tree initialization with invalid genus falls back to default"""
        tree = SupernaturalChristmasTree("InvalidGenus")
        assert tree.genus_name == "Abies"  # Should fall back to default
        assert tree.root is not None
    
    def test_tree_growth(self):
        """Test tree growth adds ornaments"""
        tree = SupernaturalChristmasTree()
        initial_ornaments = len(tree.ornaments)
        initial_aura = tree.magical_aura
        
        tree.grow(iterations=2)
        
        assert len(tree.ornaments) > initial_ornaments
        assert tree.magical_aura >= initial_aura
    
    def test_tree_statistics(self):
        """Test getting tree statistics"""
        tree = SupernaturalChristmasTree("Abies")
        stats = tree.get_tree_statistics()
        
        assert "genus" in stats
        assert "common_name" in stats
        assert "family" in stats
        assert "height" in stats
        assert "num_species" in stats
        assert "num_ornaments" in stats
        assert "magical_aura" in stats
        assert "species_list" in stats
        
        assert stats["genus"] == "Abies"
        assert stats["common_name"] == "Fir"
        assert stats["height"] > 0
        assert isinstance(stats["species_list"], list)
    
    def test_ascii_visualization(self):
        """Test ASCII visualization generation"""
        tree = SupernaturalChristmasTree()
        tree.grow(iterations=1)
        
        visualization = tree.visualize_ascii()
        
        assert isinstance(visualization, str)
        assert len(visualization) > 0
        # Should contain star emoji
        assert "⭐" in visualization or "🌟" in visualization
        # Should contain tree emoji
        assert "🌲" in visualization
        # Should contain trunk
        assert "🟫" in visualization
        # Should contain magical aura
        assert "Magical Aura" in visualization
    
    def test_discriminated_union_description(self):
        """Test discriminated union description generation"""
        tree = SupernaturalChristmasTree()
        description = tree.describe_discriminated_union()
        
        assert isinstance(description, str)
        assert "discriminated union" in description.lower()
        assert "GenusNode" in description
        assert "SpeciesNode" in description
        assert "OrnamentNode" in description
        assert tree.genus_name in description
    
    def test_get_all_nodes(self):
        """Test getting all nodes as discriminated union list"""
        tree = SupernaturalChristmasTree()
        tree.grow(iterations=1)
        
        all_nodes = tree.get_all_nodes()
        
        assert len(all_nodes) > 0
        # Should have at least: 1 genus + species + ornaments
        assert len(all_nodes) >= 1 + len(tree.species_nodes)
        
        # All nodes should have node_type attribute
        for node in all_nodes:
            assert hasattr(node, 'node_type')
            assert isinstance(node.node_type, TreeNodeType)
    
    def test_filter_by_type_genus(self):
        """Test filtering nodes by GENUS type"""
        tree = SupernaturalChristmasTree()
        genus_nodes = tree.filter_by_type(TreeNodeType.GENUS)
        
        assert len(genus_nodes) == 1
        assert isinstance(genus_nodes[0], GenusNode)
        assert genus_nodes[0].node_type == TreeNodeType.GENUS
    
    def test_filter_by_type_species(self):
        """Test filtering nodes by SPECIES type"""
        tree = SupernaturalChristmasTree()
        species_nodes = tree.filter_by_type(TreeNodeType.SPECIES)
        
        assert len(species_nodes) == len(tree.species_nodes)
        for node in species_nodes:
            assert isinstance(node, SpeciesNode)
            assert node.node_type == TreeNodeType.SPECIES
    
    def test_filter_by_type_ornament(self):
        """Test filtering nodes by ORNAMENT type"""
        tree = SupernaturalChristmasTree()
        tree.grow(iterations=2)
        ornament_nodes = tree.filter_by_type(TreeNodeType.ORNAMENT)
        
        assert len(ornament_nodes) == len(tree.ornaments)
        for node in ornament_nodes:
            assert isinstance(node, OrnamentNode)
            assert node.node_type == TreeNodeType.ORNAMENT
    
    def test_magical_aura_enhancement(self):
        """Test that magical aura increases with growth"""
        tree = SupernaturalChristmasTree()
        initial_aura = tree.magical_aura
        
        for _ in range(5):
            tree.grow(iterations=1)
        
        assert tree.magical_aura > initial_aura
        assert tree.magical_aura <= 1.0  # Should not exceed maximum
    
    def test_ornament_positions(self):
        """Test that ornaments have valid positions"""
        tree = SupernaturalChristmasTree()
        tree.grow(iterations=3)
        
        for ornament in tree.ornaments:
            level, position = ornament.position
            assert 1 <= level <= tree.height
            assert 0 <= position <= level
    
    def test_species_hierarchy(self):
        """Test that species nodes have correct hierarchy"""
        tree = SupernaturalChristmasTree()
        
        for i, species in enumerate(tree.species_nodes):
            assert species.height_level == i + 1
            assert species.genus == tree.genus_name
            assert 0.0 <= species.supernatural_glow <= 1.0


class TestMultipleGenera:
    """Tests for different genera"""
    
    def test_all_genera_create_valid_trees(self):
        """Test that all genera create valid trees"""
        for genus_name in BotanicalDatabase.list_genera():
            tree = SupernaturalChristmasTree(genus_name)
            assert tree.root is not None
            assert tree.root.name == genus_name
            assert len(tree.species_nodes) > 0
            assert tree.height > 0
    
    def test_different_genera_have_different_properties(self):
        """Test that different genera have unique properties"""
        tree1 = SupernaturalChristmasTree("Abies")
        tree2 = SupernaturalChristmasTree("Picea")
        tree3 = SupernaturalChristmasTree("Pinus")
        
        # Different common names
        assert tree1.root.common_name != tree2.root.common_name
        assert tree2.root.common_name != tree3.root.common_name
        
        # May have different heights (number of species)
        heights = [tree1.height, tree2.height, tree3.height]
        # At least some should be different
        assert len(set(heights)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
