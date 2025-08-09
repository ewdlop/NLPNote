#!/usr/bin/env python3
"""
Test integration between HumanExpressionEvaluator and BoltzmannBrainPsychoAnalyzer
测试HumanExpressionEvaluator和BoltzmannBrainPsychoAnalyzer的集成
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    from BoltzmannBrainPsychoAnalyzer import BoltzmannBrainPsychoAnalyzer
    
    print("✅ All modules imported successfully")
    print("✅ 所有模块导入成功")
    
    # Test integration
    print("\n" + "="*60)
    print("Integration Test / 集成测试")
    print("="*60)
    
    # Initialize evaluators
    human_evaluator = HumanExpressionEvaluator()
    boltzmann_analyzer = BoltzmannBrainPsychoAnalyzer()
    
    print(f"Psychological evaluator available: {human_evaluator.psychological_available}")
    print(f"心理分析器可用: {human_evaluator.psychological_available}")
    
    # Test text
    test_text = "我想要立即满足，但我知道应该理性考虑道德后果。"
    context = ExpressionContext(
        speaker="test_user",
        emotional_state="conflicted",
        formality_level="informal"
    )
    
    print(f"\n测试文本: {test_text}")
    
    # Get comprehensive evaluation with psychological dimension
    results = human_evaluator.comprehensive_evaluation(test_text, context)
    
    print(f"\n评估结果 (Evaluation Results):")
    for dimension, result in results.items():
        if hasattr(result, 'score'):
            print(f"{dimension}: {result.score:.3f}")
        else:
            print(f"{dimension}: {result}")
    
    # Get detailed psychological analysis
    if human_evaluator.psychological_available:
        print(f"\n详细心理分析 (Detailed Psychological Analysis):")
        profile = boltzmann_analyzer.analyze_psychodynamics(test_text, context)
        print(f"Id (本我): {profile.id_score:.3f}")
        print(f"Ego (自我): {profile.ego_score:.3f}")
        print(f"Superego (超我): {profile.superego_score:.3f}")
        print(f"Dominant (主导): {profile.dominant_component.value}")
        print(f"Consciousness (意识): {profile.consciousness_coherence.value}")
        print(f"Randomness (随机性): {profile.randomness_entropy:.3f}")
    
    print("\n✅ Integration test completed successfully")
    print("✅ 集成测试成功完成")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ Test error: {e}")
    print(f"❌ 测试错误: {e}")
    import traceback
    traceback.print_exc()