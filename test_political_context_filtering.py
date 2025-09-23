#!/usr/bin/env python3
"""
Test script for political context-aware marketing stopwords functionality.

This script tests that the marketing stopwords filter preserves legitimate
political terms when they appear in democratic/political contexts.
"""

import sys
import os

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

try:
    from marketing_stopwords import MarketingStopwords
    marketing_available = True
except ImportError as e:
    print(f"Warning: Could not import marketing_stopwords: {e}")
    marketing_available = False

def test_political_context_preservation():
    """Test that political terms are preserved in democratic/political contexts."""
    print("Testing Political Context Preservation")
    print("-" * 40)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    filter = MarketingStopwords()
    
    # Test cases with political context - terms should be preserved
    political_test_cases = [
        {
            'input': "Democracy is the best form of government.",
            'expected_preserved': ["best"],
            'description': "Basic democratic statement"
        },
        {
            'input': "Our democratic institutions are leading the way.",
            'expected_preserved': ["leading"],
            'description': "Democratic institutions"
        },
        {
            'input': "The candidate promises the best democratic reforms.",
            'expected_preserved': ["best"],
            'description': "Political candidate"
        },
        {
            'input': "Democratic leadership delivers optimal governance.",
            'expected_preserved': ["optimal"],
            'description': "Democratic leadership"
        },
        {
            'input': "Revolutionary democratic changes are needed.",
            'expected_preserved': ["revolutionary"],
            'description': "Democratic changes"
        },
        {
            'input': "The premier democratic party leads the nation.",
            'expected_preserved': ["premier"],
            'description': "Political party"
        },
        {
            'input': "Comprehensive election reforms ensure voting integrity.",
            'expected_preserved': ["comprehensive"],
            'description': "Election context"
        }
    ]
    
    for i, test_case in enumerate(political_test_cases, 1):
        input_text = test_case['input']
        filtered = filter.filter_text(input_text)
        expected_preserved = test_case['expected_preserved']
        
        print(f"Test {i}: {test_case['description']}")
        print(f"  Input:    {input_text}")
        print(f"  Filtered: {filtered}")
        
        # Check that expected terms are preserved
        preserved_count = 0
        for term in expected_preserved:
            if term.lower() in filtered.lower():
                print(f"  ✓ PRESERVED: '{term}'")
                preserved_count += 1
            else:
                print(f"  ✗ MISSING: '{term}' was filtered out")
        
        if preserved_count == len(expected_preserved):
            print(f"  ✓ PASS: All {preserved_count} political terms preserved")
        else:
            print(f"  ⚠ PARTIAL: {preserved_count}/{len(expected_preserved)} terms preserved")
        
        print()

def test_non_political_context_filtering():
    """Test that marketing terms are still filtered in non-political contexts."""
    print("Testing Non-Political Context Filtering")
    print("-" * 40)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    filter = MarketingStopwords()
    
    # Test cases without political context - terms should be filtered
    non_political_test_cases = [
        {
            'input': "Our best-in-class software delivers optimal performance.",
            'expected_filtered': ["best-in-class", "optimal"],
            'description': "Software marketing"
        },
        {
            'input': "The fastest, most innovative platform for businesses.",
            'expected_filtered': ["fastest", "innovative"],
            'description': "Business platform"
        },
        {
            'input': "Revolutionary AI technology with cutting-edge algorithms.",
            'expected_filtered': ["revolutionary", "cutting-edge"],
            'description': "AI technology"
        }
    ]
    
    for i, test_case in enumerate(non_political_test_cases, 1):
        input_text = test_case['input']
        filtered = filter.filter_text(input_text)
        expected_filtered = test_case['expected_filtered']
        
        print(f"Test {i}: {test_case['description']}")
        print(f"  Input:    {input_text}")
        print(f"  Filtered: {filtered}")
        
        # Check that expected terms are filtered out
        filtered_count = 0
        for term in expected_filtered:
            if term.lower() not in filtered.lower():
                print(f"  ✓ FILTERED: '{term}'")
                filtered_count += 1
            else:
                print(f"  ✗ PRESERVED: '{term}' was not filtered")
        
        if filtered_count == len(expected_filtered):
            print(f"  ✓ PASS: All {filtered_count} marketing terms filtered")
        else:
            print(f"  ⚠ PARTIAL: {filtered_count}/{len(expected_filtered)} terms filtered")
        
        print()

def test_political_context_detection():
    """Test the political context detection functionality."""
    print("Testing Political Context Detection")
    print("-" * 35)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    filter = MarketingStopwords()
    
    context_test_cases = [
        ("Democracy is the best form of government.", True),
        ("The democratic process ensures fairness.", True),
        ("Our candidate promises electoral reform.", True),
        ("Political leadership requires vision.", True),
        ("The government implements new policies.", True),
        ("Voting is a fundamental right.", True),
        ("Congress passed the legislation.", True),
        ("Our software delivers optimal performance.", False),
        ("The fastest AI technology available.", False),
        ("Revolutionary blockchain innovation.", False),
        ("Best-in-class customer service.", False)
    ]
    
    correct_detections = 0
    total_tests = len(context_test_cases)
    
    for i, (text, expected_political) in enumerate(context_test_cases, 1):
        detected_political = filter._has_political_context(text)
        
        print(f"Test {i}:")
        print(f"  Text: {text}")
        print(f"  Expected political: {expected_political}")
        print(f"  Detected political: {detected_political}")
        
        if detected_political == expected_political:
            print(f"  ✓ PASS")
            correct_detections += 1
        else:
            print(f"  ✗ FAIL")
        
        print()
    
    accuracy = correct_detections / total_tests * 100
    print(f"Context Detection Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")

def test_mixed_context_scenarios():
    """Test scenarios with mixed political and marketing content."""
    print("Testing Mixed Context Scenarios")
    print("-" * 32)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    filter = MarketingStopwords()
    
    mixed_test_cases = [
        {
            'input': "Our cutting-edge platform helps democratic organizations deliver optimal campaign management.",
            'description': "Political platform with marketing terms",
            'expected_behavior': "Political terms preserved, marketing terms may be filtered"
        },
        {
            'input': "The best democratic software provides comprehensive election management solutions.",
            'description': "Political software description", 
            'expected_behavior': "Political context should preserve 'best' and 'comprehensive'"
        }
    ]
    
    for i, test_case in enumerate(mixed_test_cases, 1):
        input_text = test_case['input']
        filtered = filter.filter_text(input_text)
        has_political = filter._has_political_context(input_text)
        
        print(f"Test {i}: {test_case['description']}")
        print(f"  Input:    {input_text}")
        print(f"  Political context detected: {has_political}")
        print(f"  Filtered: {filtered}")
        print(f"  Expected: {test_case['expected_behavior']}")
        print()

def run_all_tests():
    """Run all political context test functions."""
    print("Political Context-Aware Marketing Stopwords Test Suite")
    print("=" * 60)
    print()
    
    test_functions = [
        test_political_context_detection,
        test_political_context_preservation,
        test_non_political_context_filtering,
        test_mixed_context_scenarios
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
        print()

if __name__ == "__main__":
    run_all_tests()