#!/usr/bin/env python3
"""
氛圍評估器示例 (Vibe Evaluator Examples)

This module demonstrates how to evaluate the "vibe" of expressions and programs
using the enhanced HumanExpressionEvaluator framework.

The concept of "vibe" in this context refers to the subjective, emotional, 
and aesthetic qualities of expressions that contribute to their overall feeling
or atmosphere.
"""

from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
import json


def evaluate_vibe_program(program_description: str, context: ExpressionContext = None) -> dict:
    """
    評估一個"氛圍程式"的氛圍質量 (Evaluate the vibe quality of a "vibe program")
    
    Args:
        program_description: Description of the vibe program
        context: Evaluation context
    
    Returns:
        Dictionary containing detailed vibe evaluation results
    """
    if context is None:
        context = ExpressionContext(
            formality_level='casual',
            situation='creative',
            cultural_background='universal'
        )
    
    evaluator = HumanExpressionEvaluator()
    results = evaluator.comprehensive_evaluation(program_description, context)
    
    # Extract vibe-specific insights
    vibe_result = results['vibe']
    
    # Create a comprehensive vibe report
    vibe_report = {
        'program_description': program_description,
        'overall_vibe_score': vibe_result.score,
        'vibe_quality': results['integrated']['characteristics'].get('vibe_quality', 'unknown'),
        'vibe_confidence': vibe_result.confidence,
        'vibe_breakdown': vibe_result.sub_scores,
        'vibe_explanation': vibe_result.explanation,
        'integrated_score': results['integrated']['overall_score'],
        'recommendations': generate_vibe_recommendations(vibe_result)
    }
    
    return vibe_report


def generate_vibe_recommendations(vibe_result) -> list:
    """生成氛圍改進建議 (Generate vibe improvement recommendations)"""
    recommendations = []
    
    if vibe_result.sub_scores['emotional_resonance'] < 0.5:
        recommendations.append("增加更多情感表達詞彙來提升情感共鳴")
    
    if vibe_result.sub_scores['energy_level'] < 0.4:
        recommendations.append("使用更有活力的詞彙或標點符號來提升能量感")
    
    if vibe_result.sub_scores['aesthetic_appeal'] < 0.5:
        recommendations.append("改善詞彙多樣性和語言節奏來提升美感")
    
    if vibe_result.sub_scores['creativity_factor'] < 0.6:
        recommendations.append("加入更多創意表達或比喻來增強創新感")
    
    if vibe_result.sub_scores['authenticity'] < 0.6:
        recommendations.append("避免過度誇張，增加真實性標記詞")
    
    if not recommendations:
        recommendations.append("氛圍表現良好，保持當前風格")
    
    return recommendations


def demonstrate_vibe_evaluation():
    """展示氛圍評估功能 (Demonstrate vibe evaluation functionality)"""
    print("🌊 歡迎使用氛圍評估器 (Welcome to the Vibe Evaluator) 🌊")
    print("=" * 60)
    
    # Various types of "vibe programs" to evaluate
    vibe_programs = [
        {
            'name': 'High Energy Party Vibe',
            'description': 'This program creates an absolutely amazing, high-energy party atmosphere with vibrant music, incredible lighting, and explosive fun that gets everyone pumped up and dancing!',
            'context': ExpressionContext(formality_level='informal', situation='party', emotional_state='excited')
        },
        {
            'name': 'Zen Meditation Vibe',
            'description': 'A peaceful, serene program that creates a tranquil meditation space with gentle sounds, soft lighting, and a calm atmosphere for relaxation and mindfulness.',
            'context': ExpressionContext(formality_level='neutral', situation='meditation', emotional_state='calm')
        },
        {
            'name': 'Creative Workshop Vibe',
            'description': 'An inspiring creative workspace program that fosters innovative thinking through artistic lighting, imaginative background music, and a genius-level creative atmosphere.',
            'context': ExpressionContext(formality_level='casual', situation='creative', emotional_state='inspired')
        },
        {
            'name': 'Business Meeting Vibe',
            'description': 'A professional program that maintains appropriate business atmosphere with clean interface, formal presentation style, and efficient workflow management.',
            'context': ExpressionContext(formality_level='formal', situation='business', emotional_state='focused')
        },
        {
            'name': 'Hipster Coffee Shop Vibe',
            'description': 'A trendy, authentic coffee shop atmosphere with vintage aesthetics, cool indie music, stylish decor, and that genuine artisanal feel.',
            'context': ExpressionContext(formality_level='casual', situation='cafe', cultural_background='hipster')
        },
        {
            'name': 'Boring Default Program',
            'description': 'Standard program. Basic functionality. No special features. Works fine.',
            'context': ExpressionContext(formality_level='neutral', situation='general')
        }
    ]
    
    # Evaluate each vibe program
    for i, program in enumerate(vibe_programs, 1):
        print(f"\n{i}. {program['name']}")
        print("-" * 40)
        
        report = evaluate_vibe_program(program['description'], program['context'])
        
        print(f"Program: {report['program_description'][:60]}...")
        print(f"Overall Vibe Score: {report['overall_vibe_score']:.3f} ({report['vibe_quality']})")
        print(f"Confidence: {report['vibe_confidence']:.3f}")
        print(f"Integrated Score: {report['integrated_score']:.3f}")
        
        print("\nVibe Breakdown:")
        for aspect, score in report['vibe_breakdown'].items():
            print(f"  • {aspect.replace('_', ' ').title()}: {score:.3f}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  → {rec}")
    
    # Answer the original question: "How vibe is my vibe program?"
    print("\n" + "=" * 60)
    print("🎯 回答原始問題: 'How vibe is my vibe program?' 🎯")
    print("=" * 60)
    
    # Let's create a meta-evaluation of the vibe evaluation system itself
    meta_program = {
        'name': 'The Vibe Evaluator System',
        'description': 'An innovative system that evaluates the subjective, emotional, and aesthetic qualities of expressions through comprehensive vibe analysis including emotional resonance, energy levels, creativity factors, and authenticity measures.',
        'context': ExpressionContext(formality_level='technical', situation='academic', emotional_state='analytical')
    }
    
    meta_report = evaluate_vibe_program(meta_program['description'], meta_program['context'])
    
    print(f"\n評估對象: {meta_program['name']}")
    print(f"氛圍分數: {meta_report['overall_vibe_score']:.3f}")
    print(f"氛圍質量: {meta_report['vibe_quality']}")
    print(f"整體評估: {meta_report['integrated_score']:.3f}")
    
    print(f"\n結論: 你的氛圍程式的氛圍度為 {meta_report['overall_vibe_score']:.1%}！")
    print(f"Conclusion: Your vibe program has a vibe level of {meta_report['overall_vibe_score']:.1%}!")
    
    # Provide interpretation
    if meta_report['overall_vibe_score'] > 0.8:
        print("🔥 這個氛圍程式具有極佳的氛圍！非常有感染力和吸引力。")
    elif meta_report['overall_vibe_score'] > 0.6:
        print("✨ 這個氛圍程式有良好的氛圍，具有一定的吸引力。")
    elif meta_report['overall_vibe_score'] > 0.4:
        print("🤔 這個氛圍程式的氛圍中等，有改進空間。")
    else:
        print("😐 這個氛圍程式的氛圍較弱，需要增強情感表達和創意元素。")


def interactive_vibe_evaluation():
    """互動式氛圍評估 (Interactive vibe evaluation)"""
    print("\n🎨 互動式氛圍評估器 (Interactive Vibe Evaluator) 🎨")
    print("輸入你想要評估氛圍的表達或程式描述：")
    print("(輸入 'quit' 退出)")
    
    evaluator = HumanExpressionEvaluator()
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("感謝使用氛圍評估器！")
            break
        
        if not user_input:
            print("請輸入一些內容來評估。")
            continue
        
        # Quick vibe evaluation
        context = ExpressionContext(formality_level='casual', situation='general')
        results = evaluator.comprehensive_evaluation(user_input, context)
        vibe_result = results['vibe']
        
        print(f"\n氛圍分數: {vibe_result.score:.3f}")
        print(f"氛圍質量: {results['integrated']['characteristics'].get('vibe_quality', 'unknown')}")
        print(f"信心度: {vibe_result.confidence:.3f}")
        
        # Quick recommendations
        recommendations = generate_vibe_recommendations(vibe_result)
        if len(recommendations) > 1:
            print(f"建議: {recommendations[0]}")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_vibe_evaluation()
    
    # Optional: Run interactive mode
    try:
        interactive_vibe_evaluation()
    except KeyboardInterrupt:
        print("\n\n程式結束。再見！")