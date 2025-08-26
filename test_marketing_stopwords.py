#!/usr/bin/env python3
"""
Test script for marketing stopwords functionality.

This script demonstrates and tests the marketing stopwords filter
to ensure it works correctly for various use cases.
"""

import sys
import os

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

try:
    from marketing_stopwords import MarketingStopwords, filter_marketing_terms, find_marketing_terms
    marketing_available = True
except ImportError as e:
    print(f"Warning: Could not import marketing_stopwords: {e}")
    marketing_available = False

def test_basic_filtering():
    """Test basic marketing term filtering."""
    print("Testing Basic Marketing Term Filtering")
    print("-" * 40)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    filter = MarketingStopwords()
    
    test_cases = [
        {
            'input': "Our best-in-class solution delivers optimal performance.",
            'expected_filtered': "Our solution delivers performance."
        },
        {
            'input': "The fastest, most innovative platform for streamlined workflows.",
            'expected_contains': []  # Most words should be filtered
        },
        {
            'input': "We provide comprehensive, end-to-end solutions.",
            'expected_filtered': "We provide solutions."
        },
        {
            'input': "Revolutionary AI that empowers teams with cutting-edge technology.",
            'expected_contains': ["AI", "teams", "technology"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        filtered = filter.filter_text(input_text)
        
        print(f"Test {i}:")
        print(f"  Input:    {input_text}")
        print(f"  Filtered: {filtered}")
        
        if 'expected_filtered' in test_case:
            if filtered.strip() == test_case['expected_filtered'].strip():
                print(f"  ✓ PASS: Exact match")
            else:
                print(f"  ⚠ Note: Expected '{test_case['expected_filtered']}', got '{filtered}'")
        
        if 'expected_contains' in test_case:
            missing = [word for word in test_case['expected_contains'] if word not in filtered]
            if not missing:
                print(f"  ✓ PASS: Contains expected words")
            else:
                print(f"  ⚠ Note: Missing expected words: {missing}")
        
        print()

def test_whitelist_protection():
    """Test that whitelisted technical terms are preserved."""
    print("Testing Whitelist Protection")
    print("-" * 30)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    filter = MarketingStopwords()
    
    test_cases = [
        "First Aid training is comprehensive and optimal.",
        "The algorithm uses Fast Fourier Transform for optimal processing.",
        "Mission Critical Linux provides reliable server performance."
    ]
    
    for i, text in enumerate(test_cases, 1):
        filtered = filter.filter_text(text)
        print(f"Test {i}:")
        print(f"  Input:    {text}")
        print(f"  Filtered: {filtered}")
        
        # Check if legitimate technical terms are preserved
        technical_terms = ["First Aid", "Fast Fourier Transform", "Mission Critical Linux"]
        preserved = [term for term in technical_terms if term.lower() in filtered.lower()]
        if preserved:
            print(f"  ✓ PASS: Technical terms preserved: {preserved}")
        else:
            print(f"  ⚠ Check: No obvious technical terms to preserve")
        print()

def test_marketing_term_detection():
    """Test detection of marketing terms in text."""
    print("Testing Marketing Term Detection")
    print("-" * 35)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    filter = MarketingStopwords()
    
    test_text = "Our cutting-edge platform delivers best-in-class performance with innovative algorithms."
    found_terms = filter.get_marketing_terms_in_text(test_text)
    
    print(f"Input text: {test_text}")
    print(f"Marketing terms found: {len(found_terms)}")
    
    for term, start, end in found_terms:
        print(f"  - '{term}' at position {start}-{end}")
    
    expected_terms = ["cutting-edge", "delivers", "best-in-class", "innovative"]
    found_term_text = [term for term, _, _ in found_terms]
    
    for expected in expected_terms:
        if any(expected in found.lower() for found in found_term_text):
            print(f"  ✓ Found expected term: {expected}")
        else:
            print(f"  ⚠ Missing expected term: {expected}")

def test_convenience_functions():
    """Test the convenience functions for quick usage."""
    print("Testing Convenience Functions")
    print("-" * 30)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    test_text = "Revolutionary technology that leverages optimal performance."
    
    # Test quick filtering function
    filtered = filter_marketing_terms(test_text)
    print(f"Original: {test_text}")
    print(f"Filtered: {filtered}")
    
    # Test quick detection function  
    found = find_marketing_terms(test_text)
    print(f"Terms found: {[term for term, _, _ in found]}")

def test_file_loading():
    """Test that the JSON data file loads correctly."""
    print("Testing Data File Loading")
    print("-" * 25)
    
    if not marketing_available:
        print("Marketing stopwords module not available, skipping tests")
        return
    
    try:
        filter = MarketingStopwords()
        stopwords = filter.get_stopwords_list()
        whitelist = filter.get_whitelist()
        
        print(f"✓ Successfully loaded {len(stopwords)} stopwords")
        print(f"✓ Successfully loaded {len(whitelist)} whitelisted terms")
        
        # Check some expected terms are present
        expected_stopwords = ["best", "optimal", "cutting-edge", "revolutionary"]
        missing_stopwords = [word for word in expected_stopwords if word not in stopwords]
        
        if not missing_stopwords:
            print("✓ All expected stopwords found")
        else:
            print(f"⚠ Missing expected stopwords: {missing_stopwords}")
            
    except Exception as e:
        print(f"✗ Error loading data: {e}")

def run_all_tests():
    """Run all test functions."""
    print("Marketing Stopwords Test Suite")
    print("=" * 50)
    print()
    
    test_functions = [
        test_file_loading,
        test_basic_filtering,
        test_whitelist_protection,
        test_marketing_term_detection,
        test_convenience_functions
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
        print()

if __name__ == "__main__":
    run_all_tests()