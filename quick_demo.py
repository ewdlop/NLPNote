#!/usr/bin/env python3
"""
OrientationLessParser Quick Demo

A non-interactive demonstration of the OrientationLessParser capabilities.
"""

from OrientationLessParser import OrientationLessParser, TextDirection, ScriptType


def quick_demo():
    """Run a quick demonstration of OrientationLessParser"""
    print("=" * 80)
    print("OrientationLessParser Quick Demo")
    print("無方向性解析器快速演示")
    print("=" * 80)
    
    parser = OrientationLessParser()
    
    # Test cases with different orientations and scripts
    test_cases = [
        {
            'name': 'English (LTR)',
            'text': 'Hello world! This is a test.'
        },
        {
            'name': 'Arabic (RTL)', 
            'text': 'مرحبا بالعالم! هذا اختبار.'
        },
        {
            'name': 'Mixed Scripts',
            'text': 'Hello مرحبا 你好 שלום world!'
        },
        {
            'name': 'Bidirectional',
            'text': 'The word مرحبا means hello'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")
        print("-" * 40)
        print(f"Input: {test_case['text']}")
        
        result = parser.parse(test_case['text'])
        
        print(f"Direction: {result.dominant_direction.value}")
        print(f"Script: {result.dominant_script.value}")
        print(f"Mixed: {result.has_mixed_directions}")
        
        logical_text = parser.extract_text_content(result)
        print(f"Logical: {logical_text}")
        
        stats = parser.get_parsing_statistics(result)
        print(f"Tokens: {stats['word_tokens']} words, {stats['punctuation_tokens']} punct")
    
    print("\n✅ OrientationLessParser demo completed successfully!")
    print("✅ 無方向性解析器演示成功完成！")


if __name__ == "__main__":
    quick_demo()