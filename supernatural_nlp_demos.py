#!/usr/bin/env python3
"""
Interactive demonstration of Supernatural Language Processing concepts.

This script provides several interactive demos showing how supersymmetric
principles can be applied to natural language processing.
"""

from SupernaturalNLP import SupernaturalNLP, GrassmannNumber
import math

def demo_grassmann_algebra():
    """Demonstrate Grassmann number (anti-commuting) algebra."""
    print("=== Grassmann Algebra Demo ===")
    print("Grassmann numbers are anti-commuting: θᵢθⱼ = -θⱼθᵢ and θᵢ² = 0")
    print()
    
    # Create Grassmann numbers
    theta1 = GrassmannNumber({(1,): 1.0})  # θ₁
    theta2 = GrassmannNumber({(2,): 1.0})  # θ₂
    
    # Demonstrate anti-commutation
    product12 = theta1 * theta2
    product21 = theta2 * theta1
    
    print("θ₁ * θ₂ =", product12.coefficients)
    print("θ₂ * θ₁ =", product21.coefficients)
    print("Anti-commutation verified!")
    print()
    
    # Demonstrate θ² = 0
    square = theta1 * theta1
    print("θ₁ * θ₁ =", square.coefficients, "(should be empty - representing 0)")
    print()

def demo_superpartner_discovery():
    """Demonstrate automatic discovery of linguistic superpartners."""
    print("=== Superpartner Discovery Demo ===")
    print("Every word has a 'superpartner' with complementary properties")
    print()
    
    super_nlp = SupernaturalNLP()
    
    test_words = [
        "run", "walk", "think", "create", "destroy",
        "love", "beautiful", "quick", "strong", "light",
        "reading", "jumped", "quickly", "happiness"
    ]
    
    print("Word Superpartner Pairs:")
    for word in test_words:
        partner = super_nlp.find_superpartner(word)
        print(f"  {word:12} ↔ {partner}")
    
    print()

def demo_quantum_entanglement_matrix():
    """Demonstrate quantum semantic entanglement between words."""
    print("=== Quantum Semantic Entanglement Matrix ===")
    print("Shows how strongly word meanings are 'entangled' in superspace")
    print()
    
    super_nlp = SupernaturalNLP()
    
    words = ["love", "hate", "peace", "war", "light", "dark", "good", "evil"]
    
    print("Entanglement Matrix (higher values = stronger entanglement):")
    print(f"{'':>8}", end="")
    for word in words:
        print(f"{word:>8}", end="")
    print()
    
    for i, word1 in enumerate(words):
        print(f"{word1:>8}", end="")
        for j, word2 in enumerate(words):
            if i == j:
                print(f"{'1.000':>8}", end="")
            elif i < j:
                entanglement = super_nlp.quantum_semantic_entanglement(word1, word2)
                print(f"{entanglement:>8.3f}", end="")
            else:
                print(f"{'':>8}", end="")
        print()
    
    print()

def demo_supersymmetry_breaking():
    """Demonstrate supersymmetry breaking analysis in different text types."""
    print("=== Supersymmetry Breaking Analysis ===")
    print("Different text types show different levels of symmetry breaking")
    print()
    
    super_nlp = SupernaturalNLP()
    
    texts = {
        "Simple prose": "The cat sits on the mat",
        "Poetry": "Roses are red, violets are blue, sugar is sweet",
        "Technical": "The algorithm processes data efficiently using optimized parameters",
        "Philosophical": "Consciousness emerges from the quantum entanglement of neural states",
        "Chaotic": "Random words without structure meaning chaos everywhere"
    }
    
    print("Text Type Analysis:")
    for text_type, text in texts.items():
        analysis = super_nlp.detect_supersymmetry_breaking(text)
        print(f"\n{text_type}:")
        print(f"  Text: '{text}'")
        print(f"  Symmetry score: {analysis['symmetry_score']:.3f}")
        print(f"  Breaking strength: {analysis['breaking_strength']:.3f}")
        print(f"  Preserved/Broken: {len(analysis['preserved_words'])}/{len(analysis['broken_words'])}")

def demo_holographic_principle():
    """Demonstrate the holographic principle in language encoding."""
    print("=== Holographic Language Principle Demo ===")
    print("Surface syntax encodes deep semantic information (AdS/CFT for language)")
    print()
    
    super_nlp = SupernaturalNLP()
    
    texts = [
        "I am",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "In the beginning was the Word, and the Word was with God"
    ]
    
    print("Holographic Information Encoding:")
    for text in texts:
        holographic = super_nlp.holographic_language_encoding(text)
        
        boundary = holographic['boundary']
        bulk = holographic['bulk']
        duality = holographic['holographic_duality']
        
        print(f"\nText: '{text}'")
        print(f"  Boundary dimensions: {boundary['surface_dimensions']}")
        print(f"  Boundary entropy: {boundary['boundary_entropy']:.3f}")
        print(f"  Bulk dimensions: {bulk['bulk_dimensions']}")
        print(f"  Dimension reduction: {duality['dimension_reduction']:.3f}")
        print(f"  Entanglement network: {len(bulk['semantic_entanglement_network'])} connections")

def demo_multiverse_semantics():
    """Demonstrate handling multiple simultaneous interpretations."""
    print("=== Multiverse Semantics Demo ===")
    print("Words can exist in semantic superposition (multiple meanings simultaneously)")
    print()
    
    super_nlp = SupernaturalNLP()
    
    ambiguous_phrases = [
        "The bank is closed",  # Financial institution vs river bank
        "Time flies like an arrow",  # Multiple grammatical interpretations
        "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",  # Famous ambiguous sentence
        "I saw her duck"  # Saw action vs saw bird
    ]
    
    print("Ambiguous Phrase Analysis:")
    for phrase in ambiguous_phrases:
        results = super_nlp.supersymmetric_transform(phrase)
        
        print(f"\nPhrase: '{phrase}'")
        print("Potential semantic superpositions:")
        
        for word in results['original_words']:
            superfield = results['superfields'][word]
            fermionic_count = len(superfield.fermionic)
            
            if fermionic_count > 3:  # Threshold for semantic ambiguity
                print(f"  {word}: {fermionic_count} quantum semantic states")
            
        # Check for high entanglement (indicating ambiguity resolution dependencies)
        words = results['original_words']
        high_entanglements = []
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                entanglement = super_nlp.quantum_semantic_entanglement(word1, word2)
                if entanglement > 0.9:
                    high_entanglements.append((word1, word2, entanglement))
        
        if high_entanglements:
            print("  Strong semantic dependencies:")
            for word1, word2, strength in high_entanglements[:3]:
                print(f"    {word1} ⟷ {word2}: {strength:.3f}")

def demo_linguistic_field_theory():
    """Demonstrate treating language as a quantum field."""
    print("=== Linguistic Quantum Field Theory Demo ===")
    print("Language as a quantum field with creation/annihilation operators")
    print()
    
    super_nlp = SupernaturalNLP()
    
    text = "Words create meaning through interaction"
    words = text.lower().split()
    
    print(f"Analyzing field interactions in: '{text}'")
    print()
    
    print("Word Field Properties:")
    for word in words:
        superfield = super_nlp.create_superfield(word)
        
        # Calculate "field energy"
        bosonic_energy = sum(abs(x) for x in superfield.bosonic)
        fermionic_energy = sum(abs(v) for _, v in superfield.fermionic)
        total_energy = bosonic_energy + fermionic_energy
        
        print(f"  {word:12}: E_total = {total_energy:.3f} (bosonic: {bosonic_energy:.3f}, fermionic: {fermionic_energy:.3f})")
    
    print()
    print("Field Interaction Network:")
    for i, word1 in enumerate(words):
        for word2 in words[i+1:]:
            interaction = super_nlp.quantum_semantic_entanglement(word1, word2)
            if interaction > 0.5:
                print(f"  {word1} ↔ {word2}: interaction strength {interaction:.3f}")

def interactive_demo():
    """Interactive demo allowing user to input text for analysis."""
    print("=== Interactive Supernatural NLP Demo ===")
    print("Enter text to analyze with supersymmetric principles")
    print("(Type 'quit' to exit)")
    print()
    
    super_nlp = SupernaturalNLP()
    
    while True:
        user_text = input("Enter text: ").strip()
        
        if user_text.lower() == 'quit':
            break
        
        if not user_text:
            continue
        
        print(f"\nAnalyzing: '{user_text}'")
        print("-" * 50)
        
        # Basic transformation
        results = super_nlp.supersymmetric_transform(user_text)
        
        print("1. Superpartner Pairs:")
        for word in results['original_words']:
            partner = results['superpartners'][word]
            print(f"   {word} ↔ {partner}")
        
        # Symmetry analysis
        breaking = super_nlp.detect_supersymmetry_breaking(user_text)
        print(f"\n2. Symmetry Analysis:")
        print(f"   Symmetry score: {breaking['symmetry_score']:.3f}")
        print(f"   Breaking strength: {breaking['breaking_strength']:.3f}")
        
        # Entanglement
        words = results['original_words']
        if len(words) > 1:
            print(f"\n3. Strongest Entanglements:")
            entanglements = []
            for i, word1 in enumerate(words):
                for word2 in words[i+1:]:
                    ent = super_nlp.quantum_semantic_entanglement(word1, word2)
                    entanglements.append((word1, word2, ent))
            
            # Show top 3 entanglements
            entanglements.sort(key=lambda x: x[2], reverse=True)
            for word1, word2, strength in entanglements[:3]:
                print(f"   {word1} ⟷ {word2}: {strength:.3f}")
        
        print()

def main():
    """Main demo function with menu selection."""
    demos = {
        "1": ("Grassmann Algebra", demo_grassmann_algebra),
        "2": ("Superpartner Discovery", demo_superpartner_discovery),
        "3": ("Quantum Entanglement Matrix", demo_quantum_entanglement_matrix),
        "4": ("Supersymmetry Breaking", demo_supersymmetry_breaking),
        "5": ("Holographic Principle", demo_holographic_principle),
        "6": ("Multiverse Semantics", demo_multiverse_semantics),
        "7": ("Linguistic Field Theory", demo_linguistic_field_theory),
        "8": ("Interactive Demo", interactive_demo),
        "0": ("Run All Demos", None)
    }
    
    print("=== Supernatural Language Processing Demo Suite ===")
    print()
    print("Select a demo to run:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    print()
    
    while True:
        choice = input("Enter choice (or 'quit'): ").strip()
        
        if choice.lower() == 'quit':
            print("Thanks for exploring Supernatural NLP!")
            break
        
        if choice == "0":
            # Run all demos
            for key in ["1", "2", "3", "4", "5", "6", "7"]:
                print(f"\n{'='*60}")
                demos[key][1]()
            print(f"\n{'='*60}")
            print("All demos completed!")
            break
        
        elif choice in demos and choice != "0":
            print(f"\n{'='*60}")
            demos[choice][1]()
            print(f"\n{'='*60}")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()