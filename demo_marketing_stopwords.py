#!/usr/bin/env python3
"""
Marketing Stopwords Demo

This script demonstrates the marketing stopwords functionality both as a standalone
filter and integrated with the existing SubtextAnalyzer.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def demo_standalone_filter():
    """Demonstrate standalone marketing filter functionality."""
    print("STANDALONE MARKETING FILTER DEMO")
    print("=" * 50)
    
    try:
        from marketing_stopwords import MarketingStopwords, filter_marketing_terms
        
        # Create filter
        filter = MarketingStopwords()
        
        # Test texts with varying amounts of marketing language
        test_texts = [
            # High marketing density
            "Our revolutionary, best-in-class AI platform delivers optimal performance with cutting-edge algorithms that leverage innovative machine learning to empower teams and drive unprecedented results.",
            
            # Moderate marketing density  
            "The system provides comprehensive data analysis with proven algorithms and reliable performance metrics.",
            
            # Technical content with whitelisted terms
            "First Aid protocols utilize Fast Fourier Transform algorithms for optimal signal processing in Mission Critical Linux environments.",
            
            # Minimal marketing language
            "The database stores user information and processes queries using standard SQL operations.",
            
            # Mixed content
            "Our award-winning platform seamlessly integrates with existing systems while delivering game-changing analytics and user-friendly dashboards."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}:")
            print("-" * 30)
            print(f"Original:  {text}")
            
            # Filter the text
            filtered = filter.filter_text(text)
            print(f"Filtered:  {filtered}")
            
            # Find marketing terms
            terms = filter.get_marketing_terms_in_text(text)
            if terms:
                print(f"Marketing terms: {[term for term, _, _ in terms]}")
            
            # Calculate reduction
            original_words = len(text.split())
            filtered_words = len(filtered.split())
            reduction = (original_words - filtered_words) / original_words * 100 if original_words > 0 else 0
            print(f"Word reduction: {reduction:.1f}%")
        
        # Demonstrate quick functions
        print(f"\n\nQuick Filter Example:")
        quick_text = "Best-in-class performance with revolutionary technology!"
        quick_filtered = filter_marketing_terms(quick_text)
        print(f"Input:    {quick_text}")
        print(f"Output:   {quick_filtered}")
        
    except ImportError as e:
        print(f"Could not import marketing_stopwords: {e}")

def demo_subtext_integration():
    """Demonstrate integration with SubtextAnalyzer."""
    print("\n\nSUBTEXT ANALYZER INTEGRATION DEMO")
    print("=" * 50)
    
    try:
        from SubtextAnalyzer import SubtextAnalyzer
        
        analyzer = SubtextAnalyzer()
        
        # Test marketing language analysis
        marketing_texts = [
            "Our industry-leading, state-of-the-art platform revolutionizes how teams collaborate with cutting-edge AI technology.",
            "The software uses machine learning algorithms to process natural language and extract meaningful insights.",
            "Experience unparalleled performance with our award-winning, best-of-breed solution that delivers optimal results."
        ]
        
        for i, text in enumerate(marketing_texts, 1):
            print(f"\nMarketing Analysis {i}:")
            print("-" * 40)
            
            # Generate marketing analysis report
            if hasattr(analyzer, 'generate_marketing_analysis_report'):
                report = analyzer.generate_marketing_analysis_report(text)
                print(report)
            else:
                print("Marketing analysis not available in SubtextAnalyzer")
                
                # Fallback to basic analysis
                if hasattr(analyzer, 'analyze_marketing_language'):
                    analysis = analyzer.analyze_marketing_language(text)
                    print(f"Original: {text}")
                    print(f"Filtered: {analysis.get('filtered_text', 'N/A')}")
                    print(f"Marketing density: {analysis.get('marketing_density', 0):.1%}")
    
    except ImportError as e:
        print(f"Could not import SubtextAnalyzer: {e}")

def demo_practical_applications():
    """Show practical applications of marketing filter."""
    print("\n\nPRACTICAL APPLICATIONS")
    print("=" * 50)
    
    try:
        from marketing_stopwords import MarketingStopwords
        
        filter = MarketingStopwords()
        
        # Application 1: Cleaning product descriptions
        print("\n1. Product Description Cleaning:")
        print("-" * 35)
        product_desc = "Our revolutionary smartphone features cutting-edge technology with best-in-class performance, innovative design, and user-friendly interface that delivers optimal user experience."
        
        print(f"Before: {product_desc}")
        cleaned = filter.filter_text(product_desc)
        print(f"After:  {cleaned}")
        print("→ More objective, specific claims needed")
        
        # Application 2: Academic writing cleanup  
        print("\n2. Academic Writing Cleanup:")
        print("-" * 32)
        academic_text = "This groundbreaking research delivers comprehensive insights into optimal machine learning algorithms that leverage cutting-edge neural networks."
        
        print(f"Before: {academic_text}")
        cleaned = filter.filter_text(academic_text)
        print(f"After:  {cleaned}")
        print("→ Suitable for academic publication")
        
        # Application 3: Press release editing
        print("\n3. Press Release Editing:")
        print("-" * 27)
        press_release = "TechCorp announces its award-winning, revolutionary platform that empowers businesses with industry-leading AI solutions and game-changing analytics capabilities."
        
        print(f"Before: {press_release}")
        cleaned = filter.filter_text(press_release)
        print(f"After:  {cleaned}")
        print("→ Facts and specifics needed")
        
        # Show statistics
        print(f"\n4. Filter Statistics:")
        print("-" * 20)
        print(f"Total stopwords: {len(filter.get_stopwords_list())}")
        print(f"Whitelisted terms: {len(filter.get_whitelist())}")
        
        # Show some categories
        stopwords = filter.get_stopwords_list()
        print(f"Sample ranking terms: {[w for w in stopwords if w in ['best', 'top', 'leading', 'premier']]}")
        print(f"Sample tech hype: {[w for w in stopwords if w in ['cutting-edge', 'revolutionary', 'innovative']]}")
        
    except ImportError as e:
        print(f"Could not import marketing_stopwords: {e}")

def main():
    """Run all demo functions."""
    print("Marketing Stopwords Comprehensive Demo")
    print("=" * 60)
    
    demo_standalone_filter()
    demo_subtext_integration()
    demo_practical_applications()
    
    print("\n\nFor more information, see MARKETING_STOPWORDS_README.md")

if __name__ == "__main__":
    main()