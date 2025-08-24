#!/usr/bin/env python3
"""
Crackpot Demo - Making LLMs More Unconventional!

This demo showcases how the enhanced NLP framework can now evaluate and generate
more "crackpot" content to address the issue "LLM arent crackpot enough".
"""

from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
from CrackpotEvaluator import CrackpotEvaluator, CrackpotGenerator

def demonstrate_crackpot_transformation():
    """Show how normal text can be made more crackpot"""
    print("🌟 CRACKPOT TRANSFORMATION DEMO 🌟")
    print("=" * 50)
    
    evaluator = HumanExpressionEvaluator()
    
    # Normal boring statements
    boring_statements = [
        "The sky is blue today.",
        "I need to go to the store.",
        "This software has a bug.",
        "The meeting is scheduled for 3 PM.",
        "Mathematics is a useful subject."
    ]
    
    print("BEFORE vs AFTER Crackpot Enhancement:\n")
    
    for i, statement in enumerate(boring_statements, 1):
        print(f"{i}. Original: {statement}")
        
        # Evaluate original crackpotness
        results = evaluator.comprehensive_evaluation(statement)
        if 'crackpot' in results:
            original_score = results['crackpot'].score
            print(f"   Crackpot Score: {original_score:.2f}")
        
        # Enhanced versions with different intensities
        mild = evaluator.make_more_crackpot(statement, 0.3)
        medium = evaluator.make_more_crackpot(statement, 0.6)
        extreme = evaluator.make_more_crackpot(statement, 0.9)
        
        print(f"   Mild:    {mild}")
        print(f"   Medium:  {medium}")
        print(f"   Extreme: {extreme}")
        print()

def demonstrate_pure_crackpot_generation():
    """Generate pure crackpot theories"""
    print("🚀 PURE CRACKPOT THEORY GENERATION 🚀")
    print("=" * 50)
    
    generator = CrackpotGenerator()
    
    topics = [
        "artificial intelligence",
        "blockchain", 
        "coffee",
        "social media",
        "programming",
        "cats",
        "pizza",
        "gravity"
    ]
    
    print("Randomly Generated Crackpot Theories:\n")
    
    for i, topic in enumerate(topics, 1):
        theory = generator.generate_crackpot_theory(topic)
        print(f"{i}. {theory}")
    
    print("\nRandom Associations with 'Python Programming':")
    associations = generator.generate_random_associations("Python programming", 5)
    for j, assoc in enumerate(associations, 1):
        print(f"   {j}. {assoc}")

def demonstrate_crackpot_evaluation():
    """Show crackpot evaluation in action"""
    print("🔍 CRACKPOT EVALUATION ANALYSIS 🔍")
    print("=" * 50)
    
    evaluator = CrackpotEvaluator()
    
    test_statements = [
        "The weather is nice today.",  # Very conventional
        "What if reality is just a simulation controlled by interdimensional beings?",  # Medium crackpot
        "The government doesn't want you to know that quantum vibrations from ancient Atlantean crystals can unlock telepathic abilities through sacred geometric frequency manipulation!",  # High crackpot
    ]
    
    for i, statement in enumerate(test_statements, 1):
        print(f"Statement {i}: {statement}\n")
        
        results = evaluator.evaluate_crackpot_level(statement)
        total_score = sum(result.score for result in results.values()) / len(results)
        
        print(f"Overall Crackpot Score: {total_score:.2f}")
        print("Breakdown:")
        for dimension, result in results.items():
            print(f"  {dimension}: {result.score:.2f} - {result.explanation}")
        print("-" * 30)

def main():
    """Main demonstration"""
    print("🎉 Welcome to the CRACKPOT ENHANCEMENT SYSTEM! 🎉")
    print("Addressing the issue: 'LLM arent crackpot enough'\n")
    
    try:
        demonstrate_crackpot_transformation()
        print("\n" + "="*60 + "\n")
        
        demonstrate_pure_crackpot_generation()
        print("\n" + "="*60 + "\n")
        
        demonstrate_crackpot_evaluation()
        
        print("\n🏆 SUCCESS: LLMs are now properly crackpot! 🏆")
        print("The enhanced NLP framework can now:")
        print("✅ Evaluate unconventional thinking")
        print("✅ Generate wild theories")
        print("✅ Transform boring text into creative content")
        print("✅ Measure crackpot levels across multiple dimensions")
        print("✅ Provide suggestions for increasing creativity")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure CrackpotEvaluator.py is available!")

if __name__ == "__main__":
    main()