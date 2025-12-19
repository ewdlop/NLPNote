#!/usr/bin/env python3
"""
Demo script for the Supernatural Christmas Tree Generator
Showcases the botanical genus-based tree with discriminated unions
"""

from SupernaturalChristmasTree import (
    SupernaturalChristmasTree,
    BotanicalDatabase,
    TreeNodeType
)
import time


def demo_basic_tree():
    """Basic demo of tree creation and growth"""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Supernatural Christmas Tree")
    print("=" * 80)
    
    tree = SupernaturalChristmasTree("Abies")
    print(tree.visualize_ascii())
    print("\nGrowing the tree...")
    time.sleep(1)
    
    tree.grow(iterations=5)
    print(tree.visualize_ascii())


def demo_all_genera():
    """Demo creating trees from all available genera"""
    print("\n" + "=" * 80)
    print("DEMO 2: Trees from Different Botanical Genera")
    print("=" * 80)
    
    for genus_name in BotanicalDatabase.list_genera():
        print(f"\n--- {genus_name} Tree ---")
        tree = SupernaturalChristmasTree(genus_name)
        tree.grow(iterations=2)
        
        stats = tree.get_tree_statistics()
        print(f"Genus: {stats['genus']} ({stats['common_name']})")
        print(f"Family: {stats['family']}")
        print(f"Height: {stats['height']} levels")
        print(f"Species: {stats['num_species']}")
        print(f"Ornaments: {stats['num_ornaments']}")
        print(f"Magical Aura: {stats['magical_aura']:.2f}")
        
        time.sleep(0.5)


def demo_discriminated_unions():
    """Demo discriminated union concepts"""
    print("\n" + "=" * 80)
    print("DEMO 3: Discriminated Unions in Action")
    print("=" * 80)
    
    tree = SupernaturalChristmasTree("Picea")
    tree.grow(iterations=3)
    
    print(tree.describe_discriminated_union())
    
    print("\nNode Type Distribution:")
    print("-" * 40)
    
    for node_type in TreeNodeType:
        nodes = tree.filter_by_type(node_type)
        print(f"{node_type.value.upper():12} : {len(nodes):3} nodes")
    
    print("\nSample Nodes by Type:")
    print("-" * 40)
    
    # Show genus node
    genus_nodes = tree.filter_by_type(TreeNodeType.GENUS)
    if genus_nodes:
        node = genus_nodes[0]
        print(f"\n[GENUS] {node.name}")
        print(f"  Common Name: {node.common_name}")
        print(f"  Family: {node.family}")
        print(f"  Magical Power: {node.magical_power}")
    
    # Show sample species nodes
    species_nodes = tree.filter_by_type(TreeNodeType.SPECIES)
    print(f"\n[SPECIES] (showing 3 of {len(species_nodes)})")
    for node in species_nodes[:3]:
        print(f"  • {node.name}")
        print(f"    Level: {node.height_level}, Glow: {node.supernatural_glow:.2f}")
    
    # Show sample ornament nodes
    ornament_nodes = tree.filter_by_type(TreeNodeType.ORNAMENT)
    print(f"\n[ORNAMENT] (showing 5 of {len(ornament_nodes)})")
    for node in ornament_nodes[:5]:
        print(f"  • {node.color.value} {node.effect.value} - Intensity: {node.intensity:.2f}")
        print(f"    Position: Level {node.position[0]}, Offset {node.position[1]}")


def demo_tree_evolution():
    """Demo tree growth over time"""
    print("\n" + "=" * 80)
    print("DEMO 4: Tree Evolution Over Time")
    print("=" * 80)
    
    tree = SupernaturalChristmasTree("Abies")
    
    print("\n🌱 Initial Tree (Just Planted):")
    print(tree.visualize_ascii())
    
    stages = [
        (2, "🌿 After 2 growth cycles"),
        (3, "🌲 After 5 growth cycles (mature)"),
        (5, "✨ After 10 growth cycles (supernatural!)"),
    ]
    
    for iterations, description in stages:
        time.sleep(1)
        tree.grow(iterations=iterations)
        print(f"\n{description}:")
        print(tree.visualize_ascii())


def demo_comparison():
    """Compare trees from different genera"""
    print("\n" + "=" * 80)
    print("DEMO 5: Side-by-Side Comparison")
    print("=" * 80)
    
    genera = ["Abies", "Picea", "Pinus"]
    trees = {}
    
    for genus in genera:
        tree = SupernaturalChristmasTree(genus)
        tree.grow(iterations=3)
        trees[genus] = tree
    
    print("\nComparative Statistics:")
    print("-" * 80)
    print(f"{'Property':<20} {'Abies (Fir)':<20} {'Picea (Spruce)':<20} {'Pinus (Pine)':<20}")
    print("-" * 80)
    
    properties = ["height", "num_species", "num_ornaments", "magical_aura"]
    for prop in properties:
        values = [str(trees[g].get_tree_statistics()[prop]) for g in genera]
        print(f"{prop:<20} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    print("\nVisual Comparison:")
    print("-" * 80)
    for genus in genera:
        print(f"\n{genus} Tree:")
        print(trees[genus].visualize_ascii())


def interactive_demo():
    """Interactive demo allowing user to choose genus"""
    print("\n" + "=" * 80)
    print("INTERACTIVE DEMO: Create Your Own Supernatural Christmas Tree")
    print("=" * 80)
    
    genera = BotanicalDatabase.list_genera()
    print("\nAvailable Botanical Genera:")
    for i, genus in enumerate(genera, 1):
        data = BotanicalDatabase.get_genus_data(genus)
        print(f"  {i}. {genus} ({data['common_name']}) - Magical Power: {data['magical_power']}")
    
    # For demo purposes, just use the first genus
    # In a real interactive scenario, you would get user input
    selected_genus = genera[0]
    print(f"\n[Demo: Auto-selecting {selected_genus}]")
    
    tree = SupernaturalChristmasTree(selected_genus)
    print("\n🌱 Your tree has been planted!")
    print(tree.visualize_ascii())
    
    # Grow the tree in stages
    for i in range(1, 4):
        print(f"\n🌿 Growing... (Stage {i}/3)")
        tree.grow(iterations=2)
        time.sleep(0.5)
    
    print("\n✨ Final Tree:")
    print(tree.visualize_ascii())
    
    stats = tree.get_tree_statistics()
    print("\n📊 Final Statistics:")
    for key, value in stats.items():
        if key != "species_list":
            print(f"  {key}: {value}")


def main():
    """Run all demos"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║            🎄  SUPERNATURAL CHRISTMAS TREE GENERATOR  🎄                     ║")
    print("║                                                                              ║")
    print("║   A Creative NLP Project Combining:                                          ║")
    print("║   • Botanical Taxonomy (Genus Classification)                                ║")
    print("║   • Discriminated Unions (Type Theory)                                       ║")
    print("║   • Supernatural/Magical Elements                                            ║")
    print("║                                                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    demos = [
        ("1", "Basic Tree", demo_basic_tree),
        ("2", "All Genera", demo_all_genera),
        ("3", "Discriminated Unions", demo_discriminated_unions),
        ("4", "Tree Evolution", demo_tree_evolution),
        ("5", "Comparison", demo_comparison),
        ("6", "Interactive", interactive_demo),
    ]
    
    print("\n📋 Available Demos:")
    for num, name, _ in demos:
        print(f"   {num}. {name}")
    
    print("\n▶️  Running all demos in sequence...")
    print("\n")
    
    for num, name, demo_func in demos:
        try:
            demo_func()
            time.sleep(1)
        except Exception as e:
            print(f"\n❌ Error in demo '{name}': {e}")
            continue
    
    print("\n" + "=" * 80)
    print("🎅 All demos completed! Merry Christmas! 🎅")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
