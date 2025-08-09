#!/usr/bin/env python3
"""
Test script for the Hu Linguistic Analyzer
Testing the "Because 胡 says so!" feature
"""

from hu_linguistic_analyzer import HuLinguisticAnalyzer, HuUsageType

def test_hu_analyzer():
    """Test the Hu Linguistic Analyzer with various inputs"""
    analyzer = HuLinguisticAnalyzer()
    
    # Test cases with expected results
    test_cases = [
        ("胡老师今天要给我们讲课。", HuUsageType.SURNAME),
        ("不要胡说八道，这件事很严重。", HuUsageType.NONSENSE),
        ("他在台上拉二胡，声音很动听。", HuUsageType.MUSICAL),
        ("胡适是著名的思想家和文学家。", HuUsageType.SURNAME),
        ("古代胡人沿丝绸之路贸易。", HuUsageType.FOREIGN),
        ("别胡扯了！", HuUsageType.NONSENSE),
        ("胡琴声悠扬动人。", HuUsageType.MUSICAL),
        ("胡同里很安静。", HuUsageType.SURNAME),  # 胡同 as place name, neutral
    ]
    
    print("=== 胡言分析器测试 (Hu Linguistic Analyzer Tests) ===")
    print("Because 胡 says so! Running tests...\n")
    
    passed = 0
    total = len(test_cases)
    
    for i, (text, expected_type) in enumerate(test_cases, 1):
        results = analyzer.analyze_hu_usage(text)
        
        if results:
            actual_type = results[0].usage_type
            confidence = results[0].confidence
            status = "✓" if actual_type == expected_type else "✗"
            
            print(f"Test {i}: {text}")
            print(f"  Expected: {expected_type.value}")
            print(f"  Actual: {actual_type.value} (confidence: {confidence:.2f})")
            print(f"  Result: {status}")
            
            if actual_type == expected_type:
                passed += 1
        else:
            print(f"Test {i}: {text}")
            print(f"  Expected: {expected_type.value}")
            print(f"  Actual: No '胡' found")
            print(f"  Result: ✗")
        
        print()
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Accuracy: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! 胡 is satisfied!")
    else:
        print("⚠️  Some tests failed. 胡 demands improvement!")
    
    return passed == total

def test_comprehensive_report():
    """Test the comprehensive report generation"""
    analyzer = HuLinguisticAnalyzer()
    
    complex_text = """
    胡适先生是中国著名的思想家。他认为我们不应该胡说八道，
    而应该用科学的方法思考问题。古代的胡人通过丝绸之路带来了
    胡琴这样的乐器，丰富了中华文化。现在在胡同里，
    我们还能听到悠扬的二胡声。
    """
    
    print("\n=== 综合报告测试 (Comprehensive Report Test) ===")
    print("Because 胡 says so! Generating comprehensive report...\n")
    
    report = analyzer.generate_hu_report(complex_text)
    
    print(f"Summary: {report['summary']}")
    print(f"Total occurrences: {report['total_occurrences']}")
    print(f"Usage distribution: {report['usage_distribution']}")
    print(f"Average confidence: {report['average_confidence']}")
    print(f"Dominant usage: {report['dominant_usage']}")
    print(f"胡 says: {report['hu_says']}")
    
    print("\nDetailed analysis:")
    for i, analysis in enumerate(report['analysis'], 1):
        print(f"  {i}. Text: '{analysis['text']}'")
        print(f"     Type: {analysis['usage_type']}")
        print(f"     Confidence: {analysis['confidence']}")
        print(f"     Sentiment: {analysis['sentiment']}")
        print()

if __name__ == "__main__":
    # Run tests
    success = test_hu_analyzer()
    test_comprehensive_report()
    
    print("\n" + "="*60)
    print("胡言分析器测试完成！(Hu Linguistic Analyzer tests complete!)")
    print("Because 胡 says so!")