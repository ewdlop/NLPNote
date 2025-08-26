#!/usr/bin/env python3
"""
Quick Start Guide for Marketing Stopwords

This script provides simple, copy-paste examples for using the marketing stopwords filter.
"""

# BASIC USAGE - Filter promotional language from text
from marketing_stopwords import filter_marketing_terms

# Example 1: Quick filtering
promotional_text = "Our best-in-class solution delivers optimal performance with cutting-edge technology!"
clean_text = filter_marketing_terms(promotional_text)

print("BASIC FILTERING EXAMPLE")
print("=" * 40)
print(f"Original: {promotional_text}")
print(f"Filtered: {clean_text}")
print()

# ADVANCED USAGE - Full analysis with custom options
from marketing_stopwords import MarketingStopwords

# Example 2: Detailed analysis
analyzer = MarketingStopwords()

marketing_heavy_text = "Revolutionary AI platform with industry-leading performance and award-winning design"

# Get detailed analysis
marketing_terms = analyzer.get_marketing_terms_in_text(marketing_heavy_text)
filtered_version = analyzer.filter_text(marketing_heavy_text)
total_words = len(marketing_heavy_text.split())
marketing_density = len(marketing_terms) / total_words

print("DETAILED ANALYSIS EXAMPLE")
print("=" * 40)
print(f"Original text: {marketing_heavy_text}")
print(f"Marketing terms found: {[term for term, start, end in marketing_terms]}")
print(f"Marketing density: {marketing_density:.1%}")
print(f"Filtered text: {filtered_version}")
print()

# WHITELIST PROTECTION - Technical terms are preserved
technical_text = "First Aid training uses Fast Fourier Transform algorithms for optimal signal processing"
filtered_technical = analyzer.filter_text(technical_text)

print("WHITELIST PROTECTION EXAMPLE")
print("=" * 40)
print(f"Technical text: {technical_text}")
print(f"Filtered (preserves technical terms): {filtered_technical}")
print()

# CUSTOM MODIFICATIONS - Add your own terms
analyzer.add_stopword("amazing")  # Add custom marketing term
analyzer.add_whitelist_term("Rapid Prototyping")  # Protect technical term

custom_text = "Our amazing product uses Rapid Prototyping for optimal results"
custom_filtered = analyzer.filter_text(custom_text)

print("CUSTOM MODIFICATIONS EXAMPLE")
print("=" * 40)
print(f"Custom text: {custom_text}")
print(f"Custom filtered: {custom_filtered}")
print("(Added 'amazing' as stopword, protected 'Rapid Prototyping')")
print()

# BATCH PROCESSING - Process multiple texts
texts_to_clean = [
    "Best-in-class performance with revolutionary algorithms",
    "The database processes queries using standard operations", 
    "Game-changing analytics with cutting-edge AI technology"
]

print("BATCH PROCESSING EXAMPLE")
print("=" * 40)
for i, text in enumerate(texts_to_clean, 1):
    cleaned = filter_marketing_terms(text)
    print(f"{i}. Original: {text}")
    print(f"   Cleaned:  {cleaned}")
print()

# INTEGRATION EXAMPLE - Use with other NLP tools
import re

def clean_and_analyze_sentences(text):
    """Example function combining marketing filter with sentence analysis."""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    results = []
    for sentence in sentences:
        # Filter marketing terms
        cleaned = filter_marketing_terms(sentence)
        
        # Calculate word reduction
        original_words = len(sentence.split())
        cleaned_words = len(cleaned.split())
        reduction = (original_words - cleaned_words) / original_words if original_words > 0 else 0
        
        results.append({
            'original': sentence,
            'cleaned': cleaned,
            'word_reduction_pct': reduction * 100
        })
    
    return results

# Test the integration
sample_text = "Our revolutionary platform delivers optimal results! It uses cutting-edge algorithms. The system provides reliable data processing."

print("INTEGRATION WITH OTHER NLP TOOLS")
print("=" * 40)
analysis_results = clean_and_analyze_sentences(sample_text)

for i, result in enumerate(analysis_results, 1):
    print(f"Sentence {i}:")
    print(f"  Original: {result['original']}")
    print(f"  Cleaned:  {result['cleaned']}")
    print(f"  Reduction: {result['word_reduction_pct']:.1f}%")
print()

# FILE PROCESSING - Read from file and clean content
def clean_text_file(input_file, output_file):
    """Example function to clean marketing language from a text file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = filter_marketing_terms(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"Cleaned content written to {output_file}")
        
        # Calculate statistics
        original_words = len(content.split())
        cleaned_words = len(cleaned_content.split())
        reduction = (original_words - cleaned_words) / original_words * 100 if original_words > 0 else 0
        
        return {
            'original_words': original_words,
            'cleaned_words': cleaned_words,
            'reduction_pct': reduction
        }
        
    except FileNotFoundError:
        print(f"File {input_file} not found")
        return None

print("FILE PROCESSING EXAMPLE")
print("=" * 40)
print("# To clean a text file:")
print("# stats = clean_text_file('input.txt', 'output_clean.txt')")
print("# print(f'Removed {stats[\"reduction_pct\"]:.1f}% marketing words')")
print()

print("SUMMARY")
print("=" * 40)
print("✓ Use filter_marketing_terms() for quick filtering")
print("✓ Use MarketingStopwords() class for detailed analysis") 
print("✓ Technical terms like 'First Aid' are automatically preserved")
print("✓ Add custom terms with add_stopword() and add_whitelist_term()")
print("✓ Integrate with existing text processing pipelines")
print("✓ Based on government and expert writing guidelines")
print()
print("For complete documentation, see MARKETING_STOPWORDS_README.md")