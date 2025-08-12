#!/usr/bin/env python3
"""
Demo script for GitHub README Marketing Bot

This script demonstrates the core functionality of the marketing bot
without requiring GitHub API access. It shows how the bot would analyze
and filter marketing content from README files.
"""

import sys
from pathlib import Path

# Add the current directory to the path to import marketing_stopwords
sys.path.insert(0, str(Path(__file__).parent))

try:
    from marketing_stopwords import MarketingStopwords
except ImportError:
    print("Error: marketing_stopwords module not found")
    print("Please ensure marketing_stopwords.py is in the same directory")
    sys.exit(1)

def demo_readme_analysis():
    """Demonstrate README analysis on sample content."""
    
    # Initialize the marketing filter
    filter = MarketingStopwords()
    
    # Sample README content with marketing language (typical of what we find in real repos)
    sample_readmes = [
        {
            "repo": "example/ai-framework",
            "content": """# AI Framework
    
The **best-in-class** machine learning framework that delivers **cutting-edge** performance. 
Our **revolutionary** platform empowers developers with **optimal** algorithms and **seamless** integration.

## Features

- **Fastest** processing in the industry
- **State-of-the-art** neural networks  
- **Easy** and **intuitive** API
- **Comprehensive** documentation
- **Award-winning** architecture

## Why Choose Our Framework?

- **Leading** performance benchmarks
- **Superior** accuracy metrics
- **Top-rated** by developers worldwide
- **Premium** support and community

Get started with the **ultimate** AI solution today!
"""
        },
        
        {
            "repo": "example/web-toolkit",
            "content": """# Web Development Toolkit

A **fast** and **reliable** toolkit for modern web development.

## About

This toolkit provides utilities for building web applications. It includes 
components for routing, state management, and HTTP requests.

## Installation

```bash
npm install web-toolkit
```

## Features

- Component library
- Router with dynamic imports
- HTTP client with caching
- State management
- TypeScript support

Built with performance and developer experience in mind.
"""
        },
        
        {
            "repo": "example/data-processor", 
            "content": """# Data Processing Library

## Overview

The **most advanced** data processing library that **revolutionizes** how you handle 
large datasets. Our **innovative** approach delivers **optimal** performance while 
maintaining **best-in-class** reliability.

### Key Benefits

- **Blazing fast** processing speeds
- **Effortless** integration
- **Game-changing** algorithms  
- **Industry-leading** scalability
- **Seamless** workflow automation

**Transform** your data pipeline with our **cutting-edge** technology!
"""
        }
    ]
    
    print("🤖 GitHub README Marketing Bot - Demo Analysis")
    print("=" * 70)
    print()
    
    total_repos = len(sample_readmes)
    repos_with_marketing = 0
    
    for i, readme in enumerate(sample_readmes, 1):
        print(f"📁 Repository {i}/{total_repos}: {readme['repo']}")
        print("-" * 50)
        
        content = readme['content']
        
        # Analyze marketing content
        marketing_terms = filter.get_marketing_terms_in_text(content)
        filtered_content = filter.filter_text(content)
        
        # Calculate metrics
        words = content.split()
        word_count = len(words)
        marketing_word_count = len(marketing_terms)
        marketing_density = marketing_word_count / word_count if word_count > 0 else 0
        
        # Determine if filtering is needed
        needs_filtering = marketing_density > 0.02 or marketing_word_count >= 3
        if needs_filtering:
            repos_with_marketing += 1
        
        print(f"📊 Analysis Results:")
        print(f"   Total words: {word_count}")
        print(f"   Marketing terms found: {marketing_word_count}")
        print(f"   Marketing density: {marketing_density*100:.1f}%")
        print(f"   Needs filtering: {'🚨 YES' if needs_filtering else '✅ NO'}")
        
        if marketing_terms:
            terms_list = [term for term, _, _ in marketing_terms[:8]]  # Show first 8 terms
            print(f"   Terms detected: {terms_list}")
            if len(marketing_terms) > 8:
                print(f"   ... and {len(marketing_terms)-8} more")
        
        print()
        
        if needs_filtering:
            print("📝 BEFORE (first 200 characters):")
            print(f"   {content.strip()[:200]}...")
            print()
            print("✨ AFTER (first 200 characters):")
            print(f"   {filtered_content.strip()[:200]}...")
            print()
        
        print("=" * 70)
        print()
    
    # Summary
    print("📈 SUMMARY REPORT")
    print("=" * 30)
    print(f"Total repositories analyzed: {total_repos}")
    print(f"Repositories with marketing content: {repos_with_marketing}")
    print(f"Detection rate: {repos_with_marketing/total_repos*100:.1f}%")
    
    # Calculate overall statistics
    all_terms = []
    for readme in sample_readmes:
        terms = filter.get_marketing_terms_in_text(readme['content'])
        all_terms.extend([term.lower() for term, _, _ in terms])
    
    if all_terms:
        # Count frequency of terms
        term_counts = {}
        for term in all_terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        most_common = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Most common terms: {[term for term, count in most_common]}")
    
    print()
    print("💡 Note: This demonstrates how the bot would analyze actual GitHub repositories")
    print("   Use the full bot with --repo or --scan-user to analyze real repositories")

def demo_technical_preservation():
    """Demonstrate how technical terms are preserved."""
    
    filter = MarketingStopwords()
    
    print("🛡️  Technical Term Preservation Demo")
    print("=" * 50)
    print()
    
    technical_examples = [
        "Fast Fourier Transform provides optimal signal processing capabilities",
        "First Aid training with best practices for emergency response", 
        "The algorithm delivers fast convergence and optimal results",
        "Revolutionary blockchain technology with cutting-edge cryptography",
        "Leading research in machine learning and optimal transport theory"
    ]
    
    for i, text in enumerate(technical_examples, 1):
        print(f"Example {i}:")
        print(f"Original: {text}")
        filtered = filter.filter_text(text)
        print(f"Filtered: {filtered}")
        
        # Show what was detected as marketing vs preserved
        terms = filter.get_marketing_terms_in_text(text)
        if terms:
            print(f"Marketing terms removed: {[term for term, _, _ in terms]}")
        
        print()

if __name__ == "__main__":
    print("Starting GitHub README Marketing Bot Demo...")
    print()
    
    try:
        demo_readme_analysis()
        print()
        demo_technical_preservation()
        
        print("✅ Demo completed successfully!")
        print()
        print("To analyze real GitHub repositories:")
        print("  python github_readme_marketing_bot.py --repo owner/repo")
        print("  python github_readme_marketing_bot.py --scan-user username")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()