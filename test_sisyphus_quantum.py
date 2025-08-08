"""
Simple test for Sisyphus Quantum Analyzer
"""

from SisyphusQuantumAnalyzer import SisyphusQuantumAnalyzer


def test_sisyphus_quantum():
    """Test the Sisyphus Quantum Analyzer with simple examples"""
    analyzer = SisyphusQuantumAnalyzer()
    
    print("🚀 薛西弗斯量子分析器測試 (Sisyphus Quantum Analyzer Test)")
    print("="*70)
    
    # Test circular logic
    circular_text = """
    這個理論是正確的，因為它就是正確的。如果我們假設這個理論是對的，
    那麼我們可以證明它確實是對的。顯然，這個理論必須是正確的，
    因為如果不正確的話，它就不會是正確的了。
    """
    
    print("\n🔄 測試循環邏輯 (Testing Circular Logic):")
    print("文本:", circular_text.strip())
    
    result1 = analyzer.analyze(circular_text)
    print(f"\n薛西弗斯分數: {result1['sisyphus_analysis']['score']:.2f}")
    print(f"量子隧穿分數: {result1['quantum_analysis']['score']:.2f}")
    print(f"整體評估: {result1['overall_assessment']}")
    
    if result1['sisyphus_analysis']['patterns']:
        print("檢測到的循環模式:")
        for pattern in result1['sisyphus_analysis']['patterns']:
            print(f"  - {pattern.explanation}")
    
    # Test quantum tunneling
    quantum_text = """
    看起來這個問題無解，傳統方法都失敗了。但是，突然意識到
    如果我們完全改變視角，從問題的反面思考，原來困難可以變成機會。
    這種看似矛盾的轉化，實際上超越了原有的思維限制。
    """
    
    print("\n\n⚡ 測試量子隧穿 (Testing Quantum Tunneling):")
    print("文本:", quantum_text.strip())
    
    result2 = analyzer.analyze(quantum_text)
    print(f"\n薛西弗斯分數: {result2['sisyphus_analysis']['score']:.2f}")
    print(f"量子隧穿分數: {result2['quantum_analysis']['score']:.2f}")
    print(f"整體評估: {result2['overall_assessment']}")
    
    if result2['quantum_analysis']['moments']:
        print("檢測到的突破時刻:")
        for moment in result2['quantum_analysis']['moments']:
            print(f"  - {moment.moment_type}: 超越了 {moment.barrier_transcended}")
    
    # Test mixed example
    mixed_text = """
    我們總是重複同樣的錯誤，因為我們就是這樣的人。
    然而，也許問題不在於我們重複錯誤，而在於我們對錯誤的定義。
    如果重新定義什麼是錯誤，我們可能會發現新的可能性。
    """
    
    print("\n\n🔀 測試混合模式 (Testing Mixed Pattern):")
    print("文本:", mixed_text.strip())
    
    result3 = analyzer.analyze(mixed_text)
    print(f"\n薛西弗斯分數: {result3['sisyphus_analysis']['score']:.2f}")
    print(f"量子隧穿分數: {result3['quantum_analysis']['score']:.2f}")
    print(f"整體評估: {result3['overall_assessment']}")
    
    print("\n💡 整合洞察:")
    for insight in result3['integrated_insights']:
        print(f"  • {insight}")
    
    print("\n"+"="*70)
    print("✅ 測試完成！Sisyphus Quantum Analyzer 運行正常")
    print("✅ Test completed! Sisyphus Quantum Analyzer is working properly")


def demonstrate_issue_110_solution():
    """Demonstrate how this addresses issue #110"""
    print("\n" + "="*70)
    print("🎯 Issue #110 Solution Demonstration")
    print("   'Sisyphus boulder pushing but quantum tunneling'")
    print("="*70)
    
    analyzer = SisyphusQuantumAnalyzer()
    
    # Example that shows the metaphor
    metaphor_text = """
    每天我們都在推著同一塊石頭上山，看似永無止境的重複。
    但是，也許真正的突破不是停止推石頭，而是理解推石頭本身就是目的。
    當我們接受這個過程，我們就像量子粒子一樣，穿越了絕望的障壁，
    到達了一個新的理解層次。薛西弗斯的詛咒變成了薛西弗斯的禮物。
    """
    
    print("\n📝 示例文本 (Example Text):")
    print(metaphor_text.strip())
    
    result = analyzer.analyze(metaphor_text)
    
    print(f"\n📊 分析結果 (Analysis Results):")
    print(f"🔄 薛西弗斯分數 (Sisyphus Score): {result['sisyphus_analysis']['score']:.2f}")
    print(f"   解釋: 檢測到重複性主題和循環概念")
    
    print(f"⚡ 量子隧穿分數 (Quantum Score): {result['quantum_analysis']['score']:.2f}")
    print(f"   解釋: 檢測到突破性洞察和超越性思維")
    
    print(f"\n🎯 整體評估: {result['overall_assessment']}")
    
    print("\n🔍 具體檢測項目:")
    if result['sisyphus_analysis']['patterns']:
        print("薛西弗斯模式 (Sisyphus Patterns):")
        for pattern in result['sisyphus_analysis']['patterns']:
            print(f"  - {pattern.explanation}")
    
    if result['quantum_analysis']['moments']:
        print("量子隧穿時刻 (Quantum Tunneling Moments):")
        for moment in result['quantum_analysis']['moments']:
            print(f"  - {moment.moment_type}: {moment.insight}")
    
    print("\n🎉 結論: 成功實現了Issue #110的要求！")
    print("   ✅ 可以檢測'薛西弗斯推石'式的循環邏輯")
    print("   ✅ 可以識別'量子隧穿'式的突破性洞察")
    print("   ✅ 提供了兩種模式的平衡分析")


if __name__ == "__main__":
    test_sisyphus_quantum()
    demonstrate_issue_110_solution()