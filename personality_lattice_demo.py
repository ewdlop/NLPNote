#!/usr/bin/env python3
"""
人格格論系統演示 (Personality Lattice System Demonstration)

This script provides a comprehensive demonstration of the completed
personality lattice framework and its integration with NLP systems.

本演示展示了完整的人格格論框架及其與NLP系統的整合。
"""

from PersonalityLatticeModel import PersonalityLatticeEvaluator, PersonalityTrait, SituationalContext
from PersonalityLatticeIntegration import PersonalityAwareExpressionEvaluator


def demonstrate_lattice_mathematics():
    """演示格論數學基礎"""
    print("🔬 格論數學基礎演示 (Lattice Mathematics Demonstration)")
    print("=" * 60)
    
    from PersonalityLatticeModel import PersonalityLattice
    lattice = PersonalityLattice()
    
    # 展示不同特質的格運算
    trait_pairs = [
        (PersonalityTrait.CREATIVITY, PersonalityTrait.SYSTEMATIC_THINKING),
        (PersonalityTrait.SOCIABILITY, PersonalityTrait.DOMINANCE),
        (PersonalityTrait.EMPATHY, PersonalityTrait.CONFIDENCE),
    ]
    
    print("特質組合的格運算 (Lattice Operations on Trait Combinations):")
    for trait_a, trait_b in trait_pairs:
        join_result = lattice.join(trait_a, trait_b)
        meet_result = lattice.meet(trait_a, trait_b)
        
        print(f"\n  {trait_a.value} ⊔ {trait_b.value}")
        print(f"  並 (Join): {join_result.value}")
        print(f"  交 (Meet): {meet_result.value}")
    
    # 展示特質強度
    print(f"\n特質強度計算 (Trait Intensity Calculation):")
    key_traits = [
        PersonalityTrait.CORE_PERSONALITY,
        PersonalityTrait.SOCIAL_LEADERSHIP,
        PersonalityTrait.COMPLETE_EMOTION_REGULATION,
        PersonalityTrait.COGNITIVE_OPENNESS
    ]
    
    for trait in key_traits:
        intensity = lattice.calculate_trait_intensity(trait)
        print(f"  {trait.value}: {intensity:.4f}")


def demonstrate_personality_inference():
    """演示人格推斷"""
    print(f"\n🧠 人格推斷演示 (Personality Inference Demonstration)")
    print("=" * 60)
    
    evaluator = PersonalityLatticeEvaluator()
    
    test_expressions = [
        ("我們應該系統性地分析這個複雜問題", "分析型表達"),
        ("哇！這個創意太棒了，讓我們一起實現它！", "創造型表達"),
        ("請大家理性討論，尊重每個人的意見", "協調型表達"),
        ("根據數據顯示，我建議採用方案A", "邏輯型表達"),
    ]
    
    context = SituationalContext(
        situation_type="professional",
        formality_level=0.7,
        cultural_context="chinese"
    )
    
    for expression, expr_type in test_expressions:
        print(f"\n{expr_type}: \"{expression}\"")
        
        result = evaluator.evaluate_expression_personality(expression, context)
        
        # 顯示主要結果
        print(f"  主導特質: {result.get('dominant_traits', [])}")
        print(f"  組合人格: {result.get('combined_personality', 'unknown')}")
        print(f"  情境一致性: {result.get('situational_consistency', 0):.3f}")
        print(f"  語境適應: {result.get('context_adaptation', 0):.3f}")


def demonstrate_cultural_analysis():
    """演示文化差異分析"""
    print(f"\n🌍 文化差異分析演示 (Cultural Difference Analysis Demonstration)")
    print("=" * 60)
    
    evaluator = PersonalityAwareExpressionEvaluator()
    
    expression = "我認為這個提案很有價值，建議大家支持"
    
    cultural_contexts = [
        ("chinese", "中國文化：重視集體和諧與間接表達"),
        ("western", "西方文化：重視個人觀點與直接表達"),
        ("japanese", "日本文化：重視禮貌與層次分明"),
        ("universal", "通用標準：平衡各種文化因素")
    ]
    
    print(f"分析表達: \"{expression}\"")
    print(f"\n不同文化背景下的適當性評估:")
    
    for culture_code, culture_desc in cultural_contexts:
        context = {
            "situation": "professional",
            "formality_level": 0.7,
            "cultural_background": culture_code
        }
        
        result = evaluator.comprehensive_evaluation(expression, context)
        
        print(f"\n  {culture_desc}")
        print(f"    文化適當性: {result.cultural_appropriateness:.3f}")
        print(f"    人格對齊度: {result.personality_expression_alignment:.3f}")
        print(f"    整體真實性: {result.overall_authenticity_score:.3f}")


def demonstrate_expression_optimization():
    """演示表達式優化"""
    print(f"\n⚡ 表達式優化演示 (Expression Optimization Demonstration)")
    print("=" * 60)
    
    evaluator = PersonalityAwareExpressionEvaluator()
    
    # 原始表達和優化候選
    original = "這個東西有問題，需要改"
    candidates = [
        "這個方案存在一些改進空間，建議我們優化",
        "經過分析，發現此方案可以進一步完善",
        "我認為這個設計還有提升的潛力",
        "建議對此方案進行適當的調整和改進"
    ]
    
    context = {"situation": "professional", "formality_level": 0.8}
    
    print(f"原始表達: \"{original}\"")
    original_result = evaluator.comprehensive_evaluation(original, context)
    print(f"原始分數: {original_result.overall_authenticity_score:.3f}")
    
    print(f"\n優化候選評估:")
    
    # 評估所有候選
    best_score = 0
    best_candidate = None
    
    for i, candidate in enumerate(candidates, 1):
        result = evaluator.comprehensive_evaluation(candidate, context)
        score = result.overall_authenticity_score
        
        print(f"  候選 {i}: \"{candidate}\"")
        print(f"    整體分數: {score:.3f}")
        print(f"    文化適當: {result.cultural_appropriateness:.3f}")
        print(f"    人格對齊: {result.personality_expression_alignment:.3f}")
        
        if score > best_score:
            best_score = score
            best_candidate = i
    
    improvement = best_score - original_result.overall_authenticity_score
    print(f"\n💡 最佳候選: 候選 {best_candidate}")
    print(f"   改進幅度: +{improvement:.3f} ({improvement/original_result.overall_authenticity_score*100:.1f}%)")


def demonstrate_personality_development():
    """演示人格發展軌跡"""
    print(f"\n📈 人格發展軌跡演示 (Personality Development Trajectory Demonstration)")
    print("=" * 60)
    
    evaluator = PersonalityAwareExpressionEvaluator()
    
    # 模擬一個人在不同階段的表達變化
    timeline = [
        ("2024-01-01", "我對這個領域還不太熟悉，希望能多學習"),
        ("2024-03-01", "通過學習，我對這個問題有了一些理解"),
        ("2024-06-01", "基於我的分析，我認為這個方法比較合適"),
        ("2024-09-01", "根據我的經驗，建議採用以下策略"),
        ("2024-12-01", "作為這個領域的專家，我推薦這個解決方案")
    ]
    
    print("人格發展軌跡分析:")
    
    development = evaluator.analyze_personality_development(timeline)
    
    for item in development['personality_evolution']:
        print(f"\n  {item['timestamp']}: \"{item['expression']}\"")
        print(f"    主導特質: {item['dominant_traits']}")
        print(f"    整體分數: {item['overall_score']:.3f}")
    
    print(f"\n發展洞察:")
    for insight in development['insights']:
        print(f"  • {insight}")


def demonstrate_complete_system():
    """完整系統演示"""
    print("🚀 人格格論與NLP整合系統完整演示")
    print("=" * 80)
    print("Personality Lattice and NLP Integration System Complete Demonstration")
    print("=" * 80)
    
    # 依次執行各個演示
    demonstrate_lattice_mathematics()
    demonstrate_personality_inference()
    demonstrate_cultural_analysis()
    demonstrate_expression_optimization()
    demonstrate_personality_development()
    
    print(f"\n" + "=" * 80)
    print("🎉 演示完成！(Demonstration Complete!)")
    print("=" * 80)
    print("本系統成功實現了：")
    print("✅ 基於格論的人格特質數學建模")
    print("✅ 多維度人格評估與推斷")
    print("✅ 跨文化適應性分析")
    print("✅ 表達式優化建議")
    print("✅ 人格發展軌跡追蹤")
    print("✅ 與現有NLP系統的無縫整合")
    print()
    print("This system successfully implements:")
    print("✅ Lattice theory-based personality trait mathematical modeling")
    print("✅ Multi-dimensional personality assessment and inference")
    print("✅ Cross-cultural adaptation analysis")
    print("✅ Expression optimization recommendations")
    print("✅ Personality development trajectory tracking")
    print("✅ Seamless integration with existing NLP systems")


if __name__ == "__main__":
    demonstrate_complete_system()