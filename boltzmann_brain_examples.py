#!/usr/bin/env python3
"""
Boltzmann Brain Psychoanalytic Analyzer Examples
博尔兹曼大脑心理分析器示例

This script demonstrates the usage of the BoltzmannBrainPsychoAnalyzer
with various text examples and integration scenarios.

该脚本演示了BoltzmannBrainPsychoAnalyzer在各种文本示例和集成场景中的使用。
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BoltzmannBrainPsychoAnalyzer import (
    BoltzmannBrainPsychoAnalyzer, 
    PsychodynamicComponent, 
    ConsciousnessCoherence,
    ExpressionContext
)

def demonstrate_basic_analysis():
    """演示基础分析功能 (Demonstrate basic analysis functionality)"""
    print("\n" + "="*80)
    print("基础心理分析演示 (Basic Psychoanalytic Analysis Demonstration)")
    print("="*80)
    
    analyzer = BoltzmannBrainPsychoAnalyzer()
    
    # 测试文本集合 (Test text collection)
    test_cases = [
        {
            "name": "Id-Dominant Expression (本我主导表达)",
            "text": "I NEED this right now! I can't wait anymore, I must have it immediately! Give me what I want!",
            "expected": "High Id score, emotional intensity"
        },
        {
            "name": "Ego-Dominant Expression (自我主导表达)", 
            "text": "Let me carefully analyze this situation and consider all the practical implications before making a rational decision.",
            "expected": "High Ego score, structured thinking"
        },
        {
            "name": "Superego-Dominant Expression (超我主导表达)",
            "text": "我应该做正确的事情，遵循道德准则，不能让社会和家庭失望。We must always behave properly and ethically.",
            "expected": "High Superego score, moral focus"
        },
        {
            "name": "Boltzmann Brain-like Random Expression (博尔兹曼大脑式随机表达)",
            "text": "Purple mathematics dancing through quantum consciousness void emerging suddenly beautiful chaos patterns dissolve meaningfully into structured randomness thoughts",
            "expected": "High randomness entropy, fragmented consciousness"
        },
        {
            "name": "Balanced Expression (平衡表达)",
            "text": "I understand that I want this outcome, but I need to think practically about whether it's achievable while also considering if it's the right thing to do.",
            "expected": "Balanced psychic components"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i} (Test Case {i}): {case['name']} ---")
        print(f"文本 (Text): {case['text']}")
        print(f"预期 (Expected): {case['expected']}")
        
        # 执行分析 (Perform analysis)
        profile = analyzer.analyze_psychodynamics(case['text'])
        
        # 显示简化结果 (Show simplified results)
        print(f"结果 (Results):")
        print(f"  本我分数 (Id): {profile.id_score:.3f}")
        print(f"  自我分数 (Ego): {profile.ego_score:.3f}")
        print(f"  超我分数 (Superego): {profile.superego_score:.3f}")
        print(f"  主导组件 (Dominant): {profile.dominant_component.value}")
        print(f"  意识连贯性 (Coherence): {profile.consciousness_coherence.value}")
        print(f"  随机性熵 (Randomness): {profile.randomness_entropy:.3f}")
        print(f"  情感强度 (Emotion): {profile.emotional_intensity:.3f}")

def demonstrate_detailed_reports():
    """演示详细报告生成 (Demonstrate detailed report generation)"""
    print("\n" + "="*80)
    print("详细报告演示 (Detailed Report Demonstration)")  
    print("="*80)
    
    analyzer = BoltzmannBrainPsychoAnalyzer()
    
    # 选择一个有趣的案例进行详细分析 (Select an interesting case for detailed analysis)
    test_text = """
    Sometimes I feel like my thoughts are just random quantum fluctuations in the void of consciousness,
    emerging spontaneously without any underlying structure or meaning. But then my rational mind kicks in
    and tells me I should organize these chaotic impulses into something more coherent and socially acceptable.
    Yet deep down, I know I just want to express my true desires without any moral constraints or logical analysis.
    """
    
    print(f"分析文本 (Analysis Text):\n{test_text}")
    
    # 执行完整分析 (Perform complete analysis)
    profile = analyzer.analyze_psychodynamics(test_text)
    
    # 生成详细报告 (Generate detailed report)
    detailed_report = analyzer.generate_boltzmann_profile_report(profile, test_text, detailed=True)
    print(detailed_report)
    
    # 获取评估结果 (Get evaluation result)
    evaluation = analyzer.comprehensive_evaluation(test_text)
    print(f"\n综合评估 (Comprehensive Evaluation):")
    print(f"总分 (Overall Score): {evaluation.score:.3f}")
    print(f"信心度 (Confidence): {evaluation.confidence:.3f}")
    print(f"说明 (Explanation): {evaluation.explanation}")

def demonstrate_context_sensitivity():
    """演示语境敏感性分析 (Demonstrate context-sensitive analysis)"""
    print("\n" + "="*80)
    print("语境敏感性分析演示 (Context Sensitivity Analysis Demonstration)")
    print("="*80)
    
    analyzer = BoltzmannBrainPsychoAnalyzer()
    
    # 相同文本，不同语境 (Same text, different contexts)
    text = "我需要更多的控制和权力来实现我的目标。"
    
    contexts = [
        {
            "name": "商业语境 (Business Context)",
            "context": ExpressionContext(
                speaker="business_executive",
                formality_level="formal",
                emotional_state="determined"
            )
        },
        {
            "name": "个人治疗语境 (Personal Therapy Context)", 
            "context": ExpressionContext(
                speaker="therapy_patient",
                formality_level="informal",
                emotional_state="distressed"
            )
        },
        {
            "name": "学术讨论语境 (Academic Discussion Context)",
            "context": ExpressionContext(
                speaker="researcher",
                formality_level="formal", 
                emotional_state="neutral"
            )
        }
    ]
    
    print(f"分析文本 (Analysis Text): {text}")
    
    for ctx_info in contexts:
        print(f"\n--- {ctx_info['name']} ---")
        
        evaluation = analyzer.comprehensive_evaluation(text, ctx_info['context'])
        profile = analyzer.analyze_psychodynamics(text, ctx_info['context'])
        
        print(f"综合分数 (Score): {evaluation.score:.3f}")
        print(f"主导组件 (Dominant): {profile.dominant_component.value}")
        print(f"意识连贯性 (Coherence): {profile.consciousness_coherence.value}")
        print(f"随机性 (Randomness): {profile.randomness_entropy:.3f}")

def demonstrate_multilingual_analysis():
    """演示多语言分析 (Demonstrate multilingual analysis)"""
    print("\n" + "="*80)
    print("多语言分析演示 (Multilingual Analysis Demonstration)")
    print("="*80)
    
    analyzer = BoltzmannBrainPsychoAnalyzer()
    
    # 不同语言的等效表达 (Equivalent expressions in different languages)
    multilingual_tests = [
        {
            "language": "English",
            "text": "I want immediate gratification and pleasure without any moral constraints or rational thinking.",
            "type": "Id-dominant"
        },
        {
            "language": "Chinese",
            "text": "我想要立即的满足和快乐，不要任何道德约束或理性思考。",
            "type": "Id-dominant"
        },
        {
            "language": "English", 
            "text": "We must always do what is morally right and uphold the highest ethical standards in society.",
            "type": "Superego-dominant"
        },
        {
            "language": "Chinese",
            "text": "我们必须始终做道德上正确的事情，在社会中坚持最高的伦理标准。",
            "type": "Superego-dominant"
        }
    ]
    
    for test in multilingual_tests:
        print(f"\n--- {test['language']} ({test['type']}) ---")
        print(f"文本 (Text): {test['text']}")
        
        profile = analyzer.analyze_psychodynamics(test['text'])
        
        print(f"结果 (Results):")
        print(f"  本我 (Id): {profile.id_score:.3f}")
        print(f"  自我 (Ego): {profile.ego_score:.3f}")
        print(f"  超我 (Superego): {profile.superego_score:.3f}")
        print(f"  主导 (Dominant): {profile.dominant_component.value}")

def demonstrate_boltzmann_brain_detection():
    """演示博尔兹曼大脑模式检测 (Demonstrate Boltzmann brain pattern detection)"""
    print("\n" + "="*80)
    print("博尔兹曼大脑模式检测演示 (Boltzmann Brain Pattern Detection)")
    print("="*80)
    
    analyzer = BoltzmannBrainPsychoAnalyzer()
    
    # 不同连贯性等级的文本 (Texts with different coherence levels)
    coherence_tests = [
        {
            "name": "高度连贯 (Highly Coherent)",
            "text": "First, I will analyze the problem systematically. Then, I will develop a logical solution based on available evidence. Finally, I will implement the solution carefully and monitor the results."
        },
        {
            "name": "适度连贯 (Moderately Coherent)",
            "text": "I think about this problem sometimes, and maybe there are solutions, but it's complicated and I'm not sure what to do exactly."
        },
        {
            "name": "碎片化 (Fragmented)",
            "text": "Problem yes solutions maybe complicated thinking not sure... systematic evidence careful results but what exactly implement monitor analysis."
        },
        {
            "name": "随机涌现 (Random Emergence)",
            "text": "Quantum purple dancing mathematics void consciousness emerges suddenly beautiful chaos patterns dissolve meaningfully structured randomness thoughts fluctuating reality principles."
        }
    ]
    
    for test in coherence_tests:
        print(f"\n--- {test['name']} ---")
        print(f"文本 (Text): {test['text']}")
        
        profile = analyzer.analyze_psychodynamics(test['text'])
        
        print(f"意识连贯性 (Consciousness Coherence): {profile.consciousness_coherence.value}")
        print(f"随机性熵 (Randomness Entropy): {profile.randomness_entropy:.3f}")
        
        # 博尔兹曼大脑警报检测 (Boltzmann brain alert detection)
        if (profile.randomness_entropy > 0.7 or 
            profile.consciousness_coherence == ConsciousnessCoherence.RANDOM):
            print("⚠️ 博尔兹曼大脑警报 (Boltzmann Brain Alert): 检测到高随机性意识模式")
        
        # 生成简化报告 (Generate simplified report)
        evaluation = analyzer.comprehensive_evaluation(test['text'])
        print(f"综合评估 (Comprehensive Score): {evaluation.score:.3f}")

def demonstrate_integration_scenarios():
    """演示集成场景 (Demonstrate integration scenarios)"""
    print("\n" + "="*80)
    print("集成场景演示 (Integration Scenarios Demonstration)")
    print("="*80)
    
    analyzer = BoltzmannBrainPsychoAnalyzer()
    
    # 模拟实际应用场景 (Simulate real application scenarios)
    scenarios = [
        {
            "name": "临床心理评估 (Clinical Psychological Assessment)",
            "text": "我感觉我的思维总是在混乱和清晰之间跳跃，有时候我想要控制一切，有时候又觉得应该遵循道德准则，但我内心深处只是想要快乐。",
            "application": "Mental health monitoring and treatment planning"
        },
        {
            "name": "创意写作分析 (Creative Writing Analysis)",
            "text": "The protagonist's consciousness floated between reality and dream, where moral imperatives danced with primal desires in a quantum ballet of meaning and chaos.",
            "application": "Literary analysis and creative expression evaluation"
        },
        {
            "name": "社交媒体内容分析 (Social Media Content Analysis)",
            "text": "ugh i just want everything NOW why do i have to be responsible and think about consequences when i could just DO whatever makes me happy right???",
            "application": "Social psychology research and content moderation"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"应用 (Application): {scenario['application']}")
        print(f"文本 (Text): {scenario['text']}")
        
        # 执行全面分析 (Perform comprehensive analysis)
        profile = analyzer.analyze_psychodynamics(scenario['text'])
        evaluation = analyzer.comprehensive_evaluation(scenario['text'])
        
        print(f"\n分析结果 (Analysis Results):")
        print(f"  心理平衡指数 (Psychological Balance): {1.0 - abs(profile.id_score - profile.ego_score) - abs(profile.ego_score - profile.superego_score):.3f}")
        print(f"  意识连贯性 (Consciousness Coherence): {profile.consciousness_coherence.value}")
        print(f"  情感强度 (Emotional Intensity): {profile.emotional_intensity:.3f}")
        print(f"  综合评估分数 (Overall Score): {evaluation.score:.3f}")
        
        # 提供应用建议 (Provide application recommendations)
        if profile.id_score > 0.4:
            print("  💡 建议 (Recommendation): 关注冲动控制和情绪调节")
        if profile.consciousness_coherence == ConsciousnessCoherence.RANDOM:
            print("  💡 建议 (Recommendation): 监控意识状态变化，可能需要专业评估")
        if evaluation.score < 0.3:
            print("  💡 建议 (Recommendation): 心理平衡较低，建议进行深入分析")

def main():
    """主演示程序 (Main demonstration program)"""
    print("博尔兹曼大脑心理分析器综合演示")
    print("Boltzmann Brain Psychoanalytic Analyzer Comprehensive Demonstration")
    print("="*80)
    
    try:
        # 运行所有演示 (Run all demonstrations)
        demonstrate_basic_analysis()
        demonstrate_detailed_reports()
        demonstrate_context_sensitivity()
        demonstrate_multilingual_analysis()
        demonstrate_boltzmann_brain_detection()
        demonstrate_integration_scenarios()
        
        print("\n" + "="*80)
        print("演示完成 (Demonstration Complete)")
        print("="*80)
        print("\n如需更多信息，请参考:")
        print("For more information, please refer to:")
        print("- boltzmann-brain-superego-ego-id.md")
        print("- BoltzmannBrainPsychoAnalyzer.py")
        print("- HumanExpressionEvaluator.py")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误 (Error during demonstration): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()