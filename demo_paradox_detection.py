#!/usr/bin/env python3
"""
Demo script for paradox detection functionality in English Patcher

This script demonstrates how the English Patcher can now detect paradoxical
and contradictory expressions like "Comfort is a pain."
"""

from EnglishPatcher import EnglishPatcher, PatchType

def main():
    print("Paradox Detection Demo")
    print("=" * 40)
    print()
    
    patcher = EnglishPatcher()
    
    # Test cases including the original issue phrase
    test_cases = [
        "Comfort is a pain.",
        "Love is hate in disguise.",
        "This solution is easy and difficult.",
        "The weather is hot, cold today.",
        "He is a big small person.",
        "Peace is war by other means.",
        "Never always trust strangers.",
        "It's impossible to be possible.",
        "Everything is nothing special.",
        "The quiet loud music played.",
        # Non-paradoxical controls
        "The weather is nice today.",
        "I love reading books.",
        "This is challenging but rewarding.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        result = patcher.patch_text(text)
        
        # Check for paradox patches
        paradox_patches = [p for p in result.patches if p.patch_type == PatchType.PARADOX]
        
        if paradox_patches:
            print("  ✓ Paradox detected!")
            for patch in paradox_patches:
                print(f"    - {patch.explanation}")
                print(f"    - Confidence: {patch.confidence:.2f}")
        else:
            print("  ○ No paradox detected")
        
        print()
    
    # Detailed analysis of the main issue phrase
    print("Detailed Analysis of 'Comfort is a pain.'")
    print("=" * 45)
    
    result = patcher.patch_text("Comfort is a pain.")
    print(f"Original text: {result.original_text}")
    print(f"Processed text: {result.patched_text}")
    print(f"Text unchanged: {result.original_text == result.patched_text}")
    print(f"Total patches applied: {len(result.patches)}")
    
    for patch in result.patches:
        print(f"\nPatch Type: {patch.patch_type.value}")
        print(f"Original: '{patch.original}'")
        print(f"Corrected: '{patch.corrected}'")
        print(f"Confidence: {patch.confidence}")
        print(f"Explanation: {patch.explanation}")
    
    # Show integration with other patch types
    print("\n" + "=" * 50)
    print("Integration with Other Corrections")
    print("=" * 50)
    
    complex_text = "teh comfort is a pain and recieve happyness tommorrow."
    result = patcher.patch_text(complex_text, aggressive=True)
    
    print(f"Original: {complex_text}")
    print(f"Corrected: {result.patched_text}")
    print(f"Total patches: {len(result.patches)}")
    
    for patch in result.patches:
        print(f"  {patch.patch_type.value}: {patch.explanation}")

if __name__ == "__main__":
    main()