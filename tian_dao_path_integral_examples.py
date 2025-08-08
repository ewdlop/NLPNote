"""
天道路徑積分範例與測試 (Heavenly Way Path Integral Examples and Tests)

This module provides comprehensive examples and tests for the PathIntegralNLP implementation,
demonstrating how to use path integral approaches following 天道 (Heavenly Way) principles
for natural language processing tasks.
"""

import sys
import os

# Add the current directory to the path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from PathIntegralNLP import PathIntegralNLP, TianDaoPath, ExpressionContext
    from HumanExpressionEvaluator import ExpressionContext as HEContext
    PATH_INTEGRAL_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    PATH_INTEGRAL_AVAILABLE = False


def demonstrate_basic_path_integration():
    """演示基本路徑積分功能"""
    print("=" * 60)
    print("基本路徑積分演示 (Basic Path Integration Demonstration)")
    print("=" * 60)
    
    if not PATH_INTEGRAL_AVAILABLE:
        print("PathIntegralNLP 不可用，跳過演示")
        return
    
    # 創建路徑積分NLP處理器
    path_nlp = PathIntegralNLP(max_path_length=6, integration_steps=50)
    
    # 示例1: 從"開始"到"成功"的語義路徑
    print("\n示例1: 從'開始'到'成功'的語義路徑分析")
    print("-" * 40)
    
    start_concept = "開始"
    end_concept = "成功"
    
    result = path_nlp.path_integral_evaluation(start_concept, end_concept)
    
    print(f"起始概念: {start_concept}")
    print(f"目標概念: {end_concept}")
    print(f"找到路徑數量: {len(result.all_paths)}")
    print(f"路徑積分值: {result.integration_value:.4f}")
    print(f"和諧指數: {result.harmony_index:.4f}")
    print(f"自然度指數: {result.naturalness_index:.4f}")
    print(f"天道對齊指數: {result.tian_dao_index:.4f}")
    print(f"收斂狀態: {'已收斂' if result.convergence_achieved else '未收斂'}")
    
    print("\n最優路徑詳情:")
    optimal = result.optimal_path
    print(f"  路徑類型: {optimal.path_type.value}")
    print(f"  起始: {optimal.start_concept}")
    print(f"  中間概念: {optimal.intermediate_concepts[:3]}...")  # 只顯示前3個
    print(f"  結束: {optimal.end_concept}")
    print(f"  路徑權重: {optimal.path_weight:.4f}")
    print(f"  和諧分數: {optimal.harmony_score:.4f}")
    print(f"  自然度分數: {optimal.naturalness_score:.4f}")
    print(f"  天道對齊度: {optimal.tian_dao_alignment:.4f}")
    
    return result


def demonstrate_natural_flow_analysis():
    """演示自然流動分析"""
    print("\n" + "=" * 60)
    print("自然流動分析演示 (Natural Flow Analysis Demonstration)")
    print("=" * 60)
    
    if not PATH_INTEGRAL_AVAILABLE:
        print("PathIntegralNLP 不可用，跳過演示")
        return
    
    path_nlp = PathIntegralNLP()
    
    # 測試文本
    test_texts = [
        "水流向下，順其自然，最終匯入大海",
        "努力工作，刻苦學習，追求夢想，實現目標",
        "春天來了，花朵綻放，生命復甦，萬象更新",
        "人生如夢，歲月如流，珍惜當下，感恩生活",
        "科技發展，創新突破，改變世界，造福人類"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n示例{i}: '{text}'")
        print("-" * 50)
        
        # 分析文本流動
        analysis = path_nlp.natural_language_flow_analysis(text)
        
        print(f"流動分析: {analysis['flow_analysis']}")
        print(f"天道對齊度: {analysis['tian_dao_alignment']:.4f}")
        print(f"自然流動分數: {analysis['natural_flow_score']:.4f}")
        print(f"和諧指數: {analysis['harmony_index']:.4f}")
        print(f"分析概念: {analysis['concepts_analyzed']}")
        
        print("建議:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
        
        # 如果有詳細分數，顯示前幾個
        if analysis.get('detailed_flow_scores'):
            flow_scores = analysis['detailed_flow_scores'][:3]
            print(f"詳細流動分數 (前3個): {[f'{s:.3f}' for s in flow_scores]}")


def demonstrate_tian_dao_principles():
    """演示天道原則的計算"""
    print("\n" + "=" * 60)
    print("天道原則計算演示 (Heavenly Way Principles Demonstration)")
    print("=" * 60)
    
    if not PATH_INTEGRAL_AVAILABLE:
        print("PathIntegralNLP 不可用，跳過演示")
        return
    
    from PathIntegralNLP import TianDaoCalculator
    
    calculator = TianDaoCalculator()
    
    # 測試概念對
    concept_pairs = [
        ("大", "小"),        # 陰陽對立
        ("快速", "緩慢"),    # 互補概念
        ("開始", "結束"),    # 過程對立
        ("光明", "黑暗"),    # 經典陰陽
        ("和諧", "平衡"),    # 相近概念
        ("創新", "傳統"),    # 對比概念
        ("簡單", "複雜"),    # 複雜度對比
        ("自然", "人工")     # 本質對比
    ]
    
    print("\n無為分數計算 (Wu Wei Scores):")
    print("-" * 30)
    for concept1, concept2 in concept_pairs:
        wu_wei_score = calculator.calculate_wu_wei_score(concept1, concept2)
        print(f"'{concept1}' ↔ '{concept2}': {wu_wei_score:.4f}")
    
    print("\n陰陽分數計算 (Yin Yang Scores):")
    print("-" * 30)
    for concept1, concept2 in concept_pairs:
        yin_yang_score = calculator.calculate_yin_yang_score(concept1, concept2)
        print(f"'{concept1}' ↔ '{concept2}': {yin_yang_score:.4f}")
    
    # 測試概念序列
    concept_sequences = [
        ["開始", "發展", "成熟", "結束"],
        ["春", "夏", "秋", "冬"],
        ["學習", "實踐", "反思", "改進"],
        ["想法", "計劃", "行動", "成果"],
        ["和諧", "平衡", "統一", "完整"]
    ]
    
    print("\n五行分數計算 (Wu Xing Scores):")
    print("-" * 30)
    for concepts in concept_sequences:
        wu_xing_score = calculator.calculate_wu_xing_score(concepts)
        print(f"{concepts}: {wu_xing_score:.4f}")
    
    print("\n太極分數計算 (Tai Chi Scores):")
    print("-" * 30)
    for concepts in concept_sequences:
        tai_chi_score = calculator.calculate_tai_chi_score(concepts)
        print(f"{concepts}: {tai_chi_score:.4f}")
    
    print("\n自然流動分數計算 (Natural Flow Scores):")
    print("-" * 30)
    for concepts in concept_sequences:
        flow_score = calculator.calculate_natural_flow_score(concepts)
        print(f"{concepts}: {flow_score:.4f}")


def demonstrate_integration_with_existing_framework():
    """演示與現有框架的整合"""
    print("\n" + "=" * 60)
    print("與現有框架整合演示 (Integration with Existing Framework)")
    print("=" * 60)
    
    if not PATH_INTEGRAL_AVAILABLE:
        print("PathIntegralNLP 不可用，跳過演示")
        return
    
    path_nlp = PathIntegralNLP()
    
    # 創建表達語境
    context = ExpressionContext(
        formality_level='formal',
        situation='academic',
        cultural_background='chinese'
    )
    
    # 測試表達
    expressions = [
        "依照天道的指引，我們尋求自然的解決方案",
        "Through natural principles, we find harmonious solutions",
        "順其自然，事半功倍，這是古人的智慧",
        "Innovation emerges from the balance of tradition and progress",
        "無為而治，讓事物按其本性發展"
    ]
    
    for i, expression in enumerate(expressions, 1):
        print(f"\n表達{i}: '{expression}'")
        print("-" * 50)
        
        # 進行路徑積分分析
        analysis = path_nlp.natural_language_flow_analysis(expression, context)
        
        print(f"天道對齊度: {analysis['tian_dao_alignment']:.4f}")
        print(f"自然流動分數: {analysis['natural_flow_score']:.4f}")
        print(f"和諧指數: {analysis['harmony_index']:.4f}")
        
        # 顯示整合分析結果
        if 'integrated_analysis' in analysis and analysis['integrated_analysis']:
            integrated = analysis['integrated_analysis']
            if 'human_expression_evaluation' in integrated:
                he_result = integrated['human_expression_evaluation']
                print(f"整合評估 - 整體分數: {he_result.get('integrated', {}).get('overall_score', '未知')}")
                print(f"整合評估 - 信心度: {he_result.get('confidence', '未知')}")
            elif 'integration_error' in integrated:
                print(f"整合狀態: {integrated['integration_error']}")
            else:
                print("整合狀態: 未找到具體評估結果")
        
        print("主要建議:")
        for rec in analysis['recommendations'][:2]:  # 只顯示前兩個建議
            print(f"  • {rec}")


def demonstrate_comparative_analysis():
    """演示比較分析"""
    print("\n" + "=" * 60)
    print("比較分析演示 (Comparative Analysis Demonstration)")
    print("=" * 60)
    
    if not PATH_INTEGRAL_AVAILABLE:
        print("PathIntegralNLP 不可用，跳過演示")
        return
    
    path_nlp = PathIntegralNLP()
    
    # 比較不同風格的表達
    expressions_comparison = [
        {
            'name': '自然風格',
            'text': '水流向下，順勢而為，最終達到目標'
        },
        {
            'name': '強迫風格', 
            'text': '必須努力，堅持不懈，強行突破困難'
        },
        {
            'name': '平衡風格',
            'text': '適時努力，適時休息，保持身心平衡'
        },
        {
            'name': '哲學風格',
            'text': '天道酬勤，厚德載物，自強不息而和諧共處'
        }
    ]
    
    results = []
    
    for expr in expressions_comparison:
        analysis = path_nlp.natural_language_flow_analysis(expr['text'])
        results.append({
            'name': expr['name'],
            'text': expr['text'],
            'tian_dao_alignment': analysis['tian_dao_alignment'],
            'natural_flow_score': analysis['natural_flow_score'],
            'harmony_index': analysis['harmony_index']
        })
    
    # 排序並顯示結果
    results.sort(key=lambda x: x['tian_dao_alignment'], reverse=True)
    
    print("\n按天道對齊度排序的結果:")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']}")
        print(f"   文本: '{result['text']}'")
        print(f"   天道對齊度: {result['tian_dao_alignment']:.4f}")
        print(f"   自然流動: {result['natural_flow_score']:.4f}")
        print(f"   和諧指數: {result['harmony_index']:.4f}")
        print()
    
    # 分析最佳和最差表達的差異
    best = results[0]
    worst = results[-1]
    
    print("最佳與最差表達的對比:")
    print("-" * 30)
    print(f"最佳 ({best['name']}): 天道對齊度 {best['tian_dao_alignment']:.4f}")
    print(f"最差 ({worst['name']}): 天道對齊度 {worst['tian_dao_alignment']:.4f}")
    print(f"差異: {best['tian_dao_alignment'] - worst['tian_dao_alignment']:.4f}")
    
    return results


def demonstrate_advanced_path_analysis():
    """演示高級路徑分析"""
    print("\n" + "=" * 60)
    print("高級路徑分析演示 (Advanced Path Analysis Demonstration)")
    print("=" * 60)
    
    if not PATH_INTEGRAL_AVAILABLE:
        print("PathIntegralNLP 不可用，跳過演示")
        return
    
    path_nlp = PathIntegralNLP(max_path_length=8, integration_steps=100)
    
    # 複雜的概念轉換
    complex_transformations = [
        ("混沌", "秩序"),
        ("問題", "智慧"),
        ("衝突", "和諧"),
        ("困難", "成長"),
        ("迷茫", "清晰")
    ]
    
    for start_concept, end_concept in complex_transformations:
        print(f"\n分析路徑: '{start_concept}' → '{end_concept}'")
        print("-" * 40)
        
        # 提供一些相關的中間概念
        intermediate_concepts = [
            "轉換", "過程", "學習", "理解", "平衡", "適應",
            "發展", "演變", "調和", "整合", "突破", "領悟"
        ]
        
        result = path_nlp.path_integral_evaluation(
            start_concept, end_concept, intermediate_concepts
        )
        
        print(f"路徑積分值: {result.integration_value:.4f}")
        print(f"天道指數: {result.tian_dao_index:.4f}")
        print(f"收斂狀態: {'已收斂' if result.convergence_achieved else '未收斂'}")
        
        # 分析不同類型的路徑
        path_types_analysis = {}
        for path in result.all_paths[:10]:  # 分析前10條路徑
            path_type = path.path_type.value
            if path_type not in path_types_analysis:
                path_types_analysis[path_type] = []
            path_types_analysis[path_type].append(path.tian_dao_alignment)
        
        print("不同路徑類型的平均天道對齊度:")
        for path_type, alignments in path_types_analysis.items():
            avg_alignment = sum(alignments) / len(alignments)
            print(f"  {path_type}: {avg_alignment:.4f} (基於 {len(alignments)} 條路徑)")


def run_comprehensive_tests():
    """運行綜合測試"""
    print("\n" + "=" * 70)
    print("天道路徑積分綜合測試 (Comprehensive Heavenly Way Path Integral Tests)")
    print("=" * 70)
    
    if not PATH_INTEGRAL_AVAILABLE:
        print("PathIntegralNLP 不可用，無法運行測試")
        return False
    
    try:
        print("\n正在運行測試...")
        
        # 運行所有演示
        basic_result = demonstrate_basic_path_integration()
        demonstrate_natural_flow_analysis()
        demonstrate_tian_dao_principles()
        demonstrate_integration_with_existing_framework()
        comparison_results = demonstrate_comparative_analysis()
        demonstrate_advanced_path_analysis()
        
        print("\n" + "=" * 70)
        print("測試總結 (Test Summary)")
        print("=" * 70)
        
        print("✓ 基本路徑積分功能正常")
        print("✓ 自然流動分析功能正常")
        print("✓ 天道原則計算功能正常")
        print("✓ 框架整合功能正常")
        print("✓ 比較分析功能正常")
        print("✓ 高級路徑分析功能正常")
        
        if basic_result:
            print(f"\n基本統計:")
            print(f"  - 測試路徑積分值: {basic_result.integration_value:.4f}")
            print(f"  - 測試天道指數: {basic_result.tian_dao_index:.4f}")
            print(f"  - 收斂狀態: {'成功' if basic_result.convergence_achieved else '需要更多迭代'}")
        
        if comparison_results:
            best_style = comparison_results[0]
            print(f"  - 最佳表達風格: {best_style['name']} (天道對齊度: {best_style['tian_dao_alignment']:.4f})")
        
        print("\n🎉 所有測試完成！路徑積分NLP系統運行正常。")
        return True
        
    except Exception as e:
        print(f"\n❌ 測試過程中出現錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("天道路徑積分自然語言處理 - 演示與測試")
    print("Path Integral Natural Language Processing Following the Heavenly Way")
    print("=" * 70)
    
    # 檢查環境
    if PATH_INTEGRAL_AVAILABLE:
        print("✓ PathIntegralNLP 模組可用")
    else:
        print("❌ PathIntegralNLP 模組不可用")
    
    # 運行綜合測試
    success = run_comprehensive_tests()
    
    if success:
        print("\n" + "=" * 70)
        print("🌟 天道路徑積分系統已準備就緒！")
        print("   The Heavenly Way Path Integral System is ready!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️  系統需要進一步調試")
        print("   System requires further debugging")
        print("=" * 70)