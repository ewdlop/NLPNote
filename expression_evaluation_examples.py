#!/usr/bin/env python3
"""
人類表達評估示例 (Human Expression Evaluation Examples)

這個腳本展示了如何使用人類表達評估框架來分析不同類型的表達，
類似於程式語言表達式的評估過程，但考慮了人類交流的複雜性。

This script demonstrates how to use the human expression evaluation framework
to analyze different types of expressions, similar to programming language
expression evaluation but considering the complexity of human communication.
"""

from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
from SubtextAnalyzer import SubtextAnalyzer


def main():
    print("=" * 70)
    print("人類表達評估框架示例 (Human Expression Evaluation Framework Examples)")
    print("=" * 70)
    print()
    
    # 初始化評估器 (Initialize evaluators)
    expression_evaluator = HumanExpressionEvaluator()
    subtext_analyzer = SubtextAnalyzer()
    
    # 測試案例 (Test cases)
    test_cases = [
        {
            'name': '正式請求 (Formal Request)',
            'expression': '請問您能幫我解決這個問題嗎？',
            'context': ExpressionContext(
                speaker='student',
                listener='teacher',
                situation='academic',
                formality_level='formal',
                power_relation='formal'
            ),
            'description': '學生向老師的正式請求 (Student making formal request to teacher)'
        },
        {
            'name': '詩意表達 (Poetic Expression)',
            'expression': '夕陽如血，染紅了天邊的雲彩，仿佛在訴說著什麼深藏的秘密。',
            'context': ExpressionContext(
                speaker='poet',
                listener='reader',
                situation='literary',
                formality_level='formal',
                emotional_state='contemplative'
            ),
            'description': '詩意的景物描寫 (Poetic description of scenery)'
        },
        {
            'name': '隱喻表達 (Metaphorical Expression)',
            'expression': 'Life is like a river, flowing towards an unknown sea.',
            'context': ExpressionContext(
                speaker='philosopher',
                listener='audience',
                situation='academic',
                formality_level='formal'
            ),
            'description': '哲學性的隱喻表達 (Philosophical metaphorical expression)'
        },
        {
            'name': '日常對話 (Casual Conversation)',
            'expression': '今天天氣真好！我們去公園走走吧。',
            'context': ExpressionContext(
                speaker='friend',
                listener='friend',
                situation='casual',
                formality_level='informal',
                emotional_state='cheerful'
            ),
            'description': '朋友間的日常對話 (Casual conversation between friends)'
        },
        {
            'name': '複雜邏輯表達 (Complex Logical Expression)',
            'expression': '如果我們考慮所有可能的情況，那麼我們必須承認這個問題比我們想像的更複雜，並且需要多方面的解決方案。',
            'context': ExpressionContext(
                speaker='researcher',
                listener='colleagues',
                situation='academic',
                formality_level='formal'
            ),
            'description': '學術討論中的複雜邏輯表達 (Complex logical expression in academic discussion)'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n案例 {i}: {test_case['name']}")
        print("=" * 50)
        print(f"表達: {test_case['expression']}")
        print(f"描述: {test_case['description']}")
        print(f"語境: {test_case['context'].__dict__}")
        print()
        
        # 1. 人類表達評估 (Human Expression Evaluation)
        print("【人類表達評估結果】")
        print("-" * 30)
        try:
            result = expression_evaluator.comprehensive_evaluation(
                test_case['expression'], 
                test_case['context']
            )
            
            # 顯示各維度評分 (Show dimension scores)
            print(f"形式語義分數: {result['formal_semantic'].score:.2f}")
            print(f"認知處理分數: {result['cognitive'].score:.2f}")
            print(f"社會適當分數: {result['social'].score:.2f}")
            print(f"整體評估分數: {result['integrated']['overall_score']:.2f}")
            print(f"評估信心度: {result['integrated']['overall_confidence']:.2f}")
            print(f"表達特徵: {result['integrated']['characteristics']}")
            print(f"評估摘要: {result['integrated']['evaluation_summary']}")
            
        except Exception as e:
            print(f"評估錯誤: {e}")
        
        print()
        
        # 2. 潛文本分析 (Subtext Analysis) 
        print("【潛文本分析結果】")
        print("-" * 30)
        try:
            subtext_result = subtext_analyzer.calculate_subtext_probability(test_case['expression'])
            print(f"潛在含義概率: {subtext_result['probability']:.2f}")
            print("組成成分:")
            for component, score in subtext_result['components'].items():
                print(f"  - {component.replace('_', ' ').title()}: {score:.2f}")
                
        except Exception as e:
            print(f"分析錯誤: {e}")
        
        print()
        
        # 3. 整合分析 (Integrated Analysis)
        print("【整合分析結果】")
        print("-" * 30)
        try:
            if subtext_analyzer.expression_evaluator:
                integrated = subtext_analyzer.analyze_expression_evaluation(
                    test_case['expression'], 
                    test_case['context']
                )
                if 'error' not in integrated:
                    print("分析方法比較:")
                    comparison = integrated['comparison']
                    print(f"  - 分析一致性: {comparison['agreement_level']}")
                    print(f"  - 分數相關性: {comparison['score_correlation']:.2f}")
                    print()
                    print("整合解釋:")
                    print(integrated['interpretation'])
                else:
                    print("整合分析不可用")
            else:
                print("表達評估器不可用")
                
        except Exception as e:
            print(f"整合分析錯誤: {e}")
        
        print("\n" + "="*70)
        
        # 暫停讓用戶檢視結果 (Pause for user to review results)
        if i < len(test_cases):
            input("\n按 Enter 繼續下一個案例... (Press Enter to continue to next case...)")
    
    print("\n")
    print("=" * 70)
    print("評估完成 (Evaluation Complete)")
    print("=" * 70)
    print()
    print("這個示例展示了人類表達評估框架如何類似於程式語言表達式評估，")
    print("但考慮了人類交流特有的認知、社會和文化因素。")
    print()
    print("This example demonstrates how human expression evaluation framework")
    print("works similarly to programming language expression evaluation,") 
    print("but considers cognitive, social, and cultural factors unique to human communication.")


if __name__ == "__main__":
    main()