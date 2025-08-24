#!/usr/bin/env python3
"""
綜合測試：人格格論與NLP整合系統
(Comprehensive Test: Personality Lattice and NLP Integration System)

This script tests the complete personality lattice framework and its
integration with the NLP expression evaluation system.
"""

import sys
import os
from datetime import datetime

def test_personality_lattice_basic():
    """測試基本人格格操作"""
    print("=== 測試1: 基本人格格操作 ===")
    
    try:
        from PersonalityLatticeModel import PersonalityLattice, PersonalityTrait
        
        lattice = PersonalityLattice()
        
        # 測試格運算
        trait_a = PersonalityTrait.FRIENDLINESS
        trait_b = PersonalityTrait.COMPETITIVENESS
        
        join_result = lattice.join(trait_a, trait_b)
        meet_result = lattice.meet(trait_a, trait_b)
        
        print(f"✓ 並運算: {trait_a.value} ∨ {trait_b.value} = {join_result.value}")
        print(f"✓ 交運算: {trait_a.value} ∧ {trait_b.value} = {meet_result.value}")
        
        # 測試特質強度計算
        intensity = lattice.calculate_trait_intensity(PersonalityTrait.SOCIAL_LEADERSHIP)
        print(f"✓ 特質強度計算: social_leadership = {intensity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本人格格操作測試失敗: {e}")
        return False


def test_personality_evaluation():
    """測試人格評估功能"""
    print("\n=== 測試2: 人格評估功能 ===")
    
    try:
        from PersonalityLatticeModel import PersonalityLatticeEvaluator, SituationalContext
        
        evaluator = PersonalityLatticeEvaluator()
        
        # 測試中文表達
        chinese_expr = "我們需要合作完成這個重要的專案"
        context = SituationalContext(
            situation_type="professional",
            formality_level=0.8,
            cultural_context="chinese"
        )
        
        result = evaluator.evaluate_expression_personality(chinese_expr, context)
        
        print(f"✓ 中文表達評估完成")
        print(f"  - 整體人格分數: {result['overall_personality_score']:.3f}")
        print(f"  - 情境一致性: {result['situational_consistency']:.3f}")
        print(f"  - 組合人格類型: {result['combined_personality']}")
        
        # 測試英文表達
        english_expr = "I believe we should systematically approach this challenge"
        result_en = evaluator.evaluate_expression_personality(english_expr, context)
        
        print(f"✓ 英文表達評估完成")
        print(f"  - 整體人格分數: {result_en['overall_personality_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 人格評估功能測試失敗: {e}")
        return False


def test_integration_system():
    """測試整合系統"""
    print("\n=== 測試3: 整合系統 ===")
    
    try:
        from PersonalityLatticeIntegration import PersonalityAwareExpressionEvaluator
        
        evaluator = PersonalityAwareExpressionEvaluator()
        
        test_expressions = [
            "這個方案很有創意，我覺得值得嘗試",
            "根據分析結果，建議採用系統化方法",
            "大家一起努力，相信能完成目標"
        ]
        
        context = {
            "situation": "professional",
            "formality_level": 0.7,
            "cultural_background": "chinese"
        }
        
        print("✓ 整合評估測試:")
        for i, expr in enumerate(test_expressions):
            result = evaluator.comprehensive_evaluation(expr, context)
            
            print(f"  表達式 {i+1}: {expr[:20]}...")
            print(f"    - 真實性分數: {result.overall_authenticity_score:.3f}")
            print(f"    - 人格對齊度: {result.personality_expression_alignment:.3f}")
            print(f"    - 文化適當性: {result.cultural_appropriateness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 整合系統測試失敗: {e}")
        return False


def test_comparison_functionality():
    """測試比較功能"""
    print("\n=== 測試4: 表達式比較功能 ===")
    
    try:
        from PersonalityLatticeIntegration import PersonalityAwareExpressionEvaluator
        
        evaluator = PersonalityAwareExpressionEvaluator()
        
        comparison_expressions = [
            "請大家協助完成這項工作",
            "希望各位能夠配合完成任務",
            "需要團隊成員共同努力完成"
        ]
        
        context = {"situation": "professional", "formality_level": 0.6}
        
        comparison = evaluator.compare_expressions(comparison_expressions, context)
        
        print("✓ 表達式比較完成:")
        print(f"  - 最佳整體表現: 表達式 {comparison['best_overall']['index'] + 1}")
        print(f"  - 最佳人格對齊: 表達式 {comparison['best_personality_alignment']['index'] + 1}")
        print(f"  - 建議: {comparison['recommendations'][0] if comparison['recommendations'] else '無特殊建議'}")
        
        return True
        
    except Exception as e:
        print(f"✗ 表達式比較功能測試失敗: {e}")
        return False


def test_development_tracking():
    """測試發展軌跡追蹤"""
    print("\n=== 測試5: 人格發展軌跡追蹤 ===")
    
    try:
        from PersonalityLatticeIntegration import PersonalityAwareExpressionEvaluator
        
        evaluator = PersonalityAwareExpressionEvaluator()
        
        timeline = [
            ("2024-01-01", "我還在學習這個領域的知識"),
            ("2024-06-01", "基於我的理解，提出以下建議"),
            ("2024-12-01", "根據專業分析，推薦採用此方案")
        ]
        
        development = evaluator.analyze_personality_development(timeline)
        
        print("✓ 發展軌跡分析完成:")
        print(f"  - 分析期間: {len(timeline)} 個時間點")
        print(f"  - 穩定性指標: {len(development['stability_metrics'])} 個特質")
        print(f"  - 發展洞察: {len(development['insights'])} 條")
        
        if development['insights']:
            print(f"  - 主要洞察: {development['insights'][0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 發展軌跡追蹤測試失敗: {e}")
        return False


def test_cultural_sensitivity():
    """測試文化敏感性"""
    print("\n=== 測試6: 文化敏感性分析 ===")
    
    try:
        from PersonalityLatticeIntegration import PersonalityAwareExpressionEvaluator
        
        evaluator = PersonalityAwareExpressionEvaluator()
        
        expression = "我認為這個計劃需要所有人的參與"
        
        cultures = ["chinese", "western", "japanese", "universal"]
        
        print("✓ 文化差異分析:")
        for culture in cultures:
            context = {
                "situation": "professional",
                "formality_level": 0.7,
                "cultural_background": culture
            }
            
            result = evaluator.comprehensive_evaluation(expression, context)
            
            print(f"  - {culture}: 文化適當性 {result.cultural_appropriateness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 文化敏感性測試失敗: {e}")
        return False


def run_comprehensive_test():
    """運行綜合測試"""
    print("人格格論與NLP整合系統 - 綜合測試")
    print("=" * 60)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_functions = [
        test_personality_lattice_basic,
        test_personality_evaluation,
        test_integration_system,
        test_comparison_functionality,
        test_development_tracking,
        test_cultural_sensitivity
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"✗ 測試 {test_func.__name__} 執行異常: {e}")
    
    print("\n" + "=" * 60)
    print("測試總結:")
    print(f"總測試數: {total_tests}")
    print(f"通過測試: {passed_tests}")
    print(f"失敗測試: {total_tests - passed_tests}")
    print(f"成功率: {(passed_tests / total_tests) * 100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有測試通過！系統運行正常。")
    else:
        print("⚠️ 部分測試失敗，請檢查相關模組。")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)