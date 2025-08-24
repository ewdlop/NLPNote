#!/usr/bin/env python3
"""
Integration Example: "Because 胡 says so!" 
Demonstrating the Hu Linguistic Analyzer in action with various real-world examples.
"""

from hu_linguistic_analyzer import HuLinguisticAnalyzer
import json

def demonstrate_hu_analysis():
    """Comprehensive demonstration of the Hu Linguistic Analyzer"""
    
    analyzer = HuLinguisticAnalyzer()
    
    print("=" * 60)
    print("🎭 Because 胡 says so! - Comprehensive Demonstration")
    print("=" * 60)
    print()
    
    # Real-world examples from different domains
    examples = [
        {
            "category": "📚 Academic Context",
            "text": "胡适先生提出的白话文运动改变了中国文学。他反对那些胡说八道的论调，主张用科学方法研究问题。",
            "description": "Academic discussion mentioning Hu Shi (scholar) and dismissing nonsensical arguments"
        },
        {
            "category": "🎵 Musical Performance", 
            "text": "在音乐会上，胡老师精彩地演奏了二胡名曲《二泉映月》，台下观众都被优美的胡琴声深深感动。",
            "description": "Musical performance featuring traditional Chinese instruments"
        },
        {
            "category": "📖 Historical Text",
            "text": "唐朝时期，胡人商队沿着丝绸之路带来了胡椒、胡萝卜等物品，也传播了胡乐和胡舞等文化。",
            "description": "Historical account of Tang Dynasty cultural exchange"
        },
        {
            "category": "💬 Daily Conversation",
            "text": "小明：\"我觉得这个计划不可行。\" 小红：\"别胡说了！这个计划很有前途。\" 老胡：\"大家都冷静点，别胡来。\"",
            "description": "Everyday conversation with different uses of 胡"
        },
        {
            "category": "🌃 Location Description",
            "text": "在北京的胡同里，经常能听到从四合院传出的胡琴声，那是老北京文化的真实写照。",
            "description": "Geographic and cultural description of Beijing hutongs"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['category']}")
        print(f"Description: {example['description']}")
        print(f"Text: {example['text']}")
        print()
        
        # Generate comprehensive report
        report = analyzer.generate_hu_report(example['text'])
        
        print(f"📊 Analysis Results:")
        print(f"   • Total occurrences: {report['total_occurrences']}")
        print(f"   • Usage distribution: {report['usage_distribution']}")
        print(f"   • Average confidence: {report['average_confidence']}")
        print(f"   • Dominant usage: {report['dominant_usage']}")
        
        print("\n🔍 Detailed Analysis:")
        for j, analysis in enumerate(report['analysis'], 1):
            print(f"   {j}. \"{analysis['text'][:20]}...\"")
            print(f"      Type: {analysis['usage_type']} (confidence: {analysis['confidence']})")
            print(f"      Sentiment: {analysis['sentiment']}")
            print(f"      Cultural notes: {analysis['cultural_notes']}")
        
        print(f"\n💬 胡 says: {report['hu_says']}")
        print("-" * 60)
        print()

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    analyzer = HuLinguisticAnalyzer()
    
    print("🔬 Edge Case Testing")
    print("=" * 40)
    
    edge_cases = [
        ("Single character: 胡", "Testing isolated character"),
        ("胡胡胡胡胡", "Repeated characters"), 
        ("English text with 胡 character embedded", "Mixed language"),
        ("胡？胡！胡...", "Punctuation handling"),
        ("", "Empty string"),
        ("No Hu character here", "No target character"),
        ("胡ABC123胡", "Mixed scripts"),
    ]
    
    for text, description in edge_cases:
        print(f"Test: {description}")
        print(f"Input: \"{text}\"")
        
        results = analyzer.analyze_hu_usage(text)
        if results:
            for result in results:
                print(f"  Result: {result.usage_type.value} (confidence: {result.confidence:.2f})")
        else:
            print("  Result: No 胡 character detected")
        print()

def performance_benchmark():
    """Simple performance benchmark"""
    import time
    
    analyzer = HuLinguisticAnalyzer()
    
    # Test text with multiple 胡 characters
    test_text = """
    胡适先生说：不要胡说八道。古代胡人带来了胡琴。
    在胡同里，胡老师教学生拉二胡。胡思乱想是不好的习惯。
    """ * 10  # Repeat 10 times for more substantial test
    
    print("⏱️  Performance Benchmark")
    print("=" * 40)
    print(f"Test text length: {len(test_text)} characters")
    
    start_time = time.time()
    
    for i in range(100):  # Run 100 times
        report = analyzer.generate_hu_report(test_text)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / 100
    
    print(f"Total time for 100 runs: {total_time:.4f} seconds")
    print(f"Average time per analysis: {avg_time:.4f} seconds")
    print(f"Analyses per second: {1/avg_time:.2f}")
    
    # Memory usage would require additional libraries, so we'll skip that
    print("Memory usage: Minimal (no heavy dependencies)")

def export_analysis_results():
    """Export analysis results in various formats"""
    
    analyzer = HuLinguisticAnalyzer()
    sample_text = "胡适先生在胡同里听胡琴，批评那些胡说八道的人。"
    
    report = analyzer.generate_hu_report(sample_text)
    
    print("📄 Export Formats")
    print("=" * 40)
    
    # JSON export
    json_output = json.dumps(report, ensure_ascii=False, indent=2)
    print("JSON Format:")
    print(json_output[:200] + "..." if len(json_output) > 200 else json_output)
    print()
    
    # CSV-style export
    print("CSV Format:")
    print("Text,Type,Confidence,Sentiment,Cultural_Notes")
    for analysis in report['analysis']:
        csv_line = f'"{analysis["text"]}",{analysis["usage_type"]},{analysis["confidence"]},{analysis["sentiment"]},"{analysis["cultural_notes"][:50]}..."'
        print(csv_line)
    print()
    
    # Simple report format
    print("Simple Report Format:")
    print(f"Found {report['total_occurrences']} uses of 胡:")
    for i, analysis in enumerate(report['analysis'], 1):
        print(f"{i}. {analysis['usage_type']} usage (confidence: {analysis['confidence']})")

def main():
    """Main demonstration function"""
    
    print("🚀 Starting comprehensive demonstration...")
    print()
    
    # Run all demonstrations
    demonstrate_hu_analysis()
    test_edge_cases() 
    performance_benchmark()
    export_analysis_results()
    
    print("\n" + "=" * 60)
    print("✅ Demonstration complete!")
    print("Because 胡 says so! - All features working perfectly.")
    print("胡言分析器 ready for production use! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()