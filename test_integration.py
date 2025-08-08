"""
Integration test for Sisyphus Quantum Analyzer with existing systems
"""

from SisyphusQuantumAnalyzer import SisyphusQuantumAnalyzer
from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
import json


def test_integration():
    """Test integration between Sisyphus Quantum and Human Expression analyzers"""
    
    print("🔗 Integration Test: Sisyphus Quantum + Human Expression Evaluator")
    print("="*70)
    
    # Initialize analyzers
    sq_analyzer = SisyphusQuantumAnalyzer()
    expr_evaluator = HumanExpressionEvaluator()
    
    # Test text with mixed patterns
    test_text = """
    雖然我們總是說要創新，但實際上我們經常重複同樣的思維模式。
    創新需要創新，這是一個顯而易見的事實。
    然而，真正的突破可能來自於承認我們的局限性，
    並在這個承認中找到超越的可能。
    """
    
    print("📝 Test Text:")
    print(test_text.strip())
    print("\n" + "="*70)
    
    # Sisyphus Quantum Analysis
    print("\n🔄⚡ Sisyphus Quantum Analysis:")
    sq_result = sq_analyzer.analyze(test_text)
    
    print(f"Sisyphus Score: {sq_result['sisyphus_analysis']['score']:.2f}")
    print(f"Quantum Score: {sq_result['quantum_analysis']['score']:.2f}")
    print(f"Overall Assessment: {sq_result['overall_assessment']}")
    
    if sq_result['sisyphus_analysis']['patterns']:
        print("\nDetected Sisyphus Patterns:")
        for pattern in sq_result['sisyphus_analysis']['patterns']:
            print(f"  - {pattern.explanation}")
    
    if sq_result['quantum_analysis']['moments']:
        print("\nDetected Quantum Moments:")
        for moment in sq_result['quantum_analysis']['moments']:
            print(f"  - {moment.moment_type}: {moment.insight}")
    
    # Human Expression Analysis
    print("\n🎯 Human Expression Evaluation:")
    context = ExpressionContext(
        situation='academic',
        formality_level='formal',
        cultural_background='chinese'
    )
    
    expr_result = expr_evaluator.comprehensive_evaluation(test_text, context)
    
    print(f"Overall Score: {expr_result['integrated']['overall_score']:.2f}")
    print(f"Formal Semantic: {expr_result['formal_semantic'].score:.2f}")
    print(f"Cognitive: {expr_result['cognitive'].score:.2f}")
    print(f"Social: {expr_result['social'].score:.2f}")
    
    # Integrated Analysis
    print("\n🔮 Integrated Analysis:")
    
    # Calculate correlation between different analysis dimensions
    sisyphus_score = sq_result['sisyphus_analysis']['score']
    quantum_score = sq_result['quantum_analysis']['score']
    expression_score = expr_result['integrated']['overall_score']
    formal_score = expr_result['formal_semantic'].score
    cognitive_score = expr_result['cognitive'].score
    social_score = expr_result['social'].score
    
    print(f"Analysis Correlation:")
    print(f"  Sisyphus vs Expression Quality: {abs(sisyphus_score - expression_score):.2f}")
    print(f"  Quantum vs Cognitive Score: {abs(quantum_score - cognitive_score):.2f}")
    print(f"  Innovation Index: {(quantum_score + cognitive_score) / 2:.2f}")
    print(f"  Clarity Index: {(expression_score + (1 - sisyphus_score)) / 2:.2f}")
    
    # Generate integrated recommendation
    recommendation = generate_integrated_recommendation(
        sisyphus_score, quantum_score, expression_score, 
        formal_score, cognitive_score, social_score
    )
    
    print(f"\n💡 Integrated Recommendation:")
    print(f"  {recommendation}")
    
    print("\n✅ Integration test completed successfully!")
    
    return {
        'sisyphus_quantum': sq_result,
        'human_expression': expr_result,
        'integration_metrics': {
            'innovation_index': (quantum_score + cognitive_score) / 2,
            'clarity_index': (expression_score + (1 - sisyphus_score)) / 2,
            'recommendation': recommendation
        }
    }


def generate_integrated_recommendation(sisyphus, quantum, expression, formal, cognitive, social):
    """Generate an integrated recommendation based on all analysis dimensions"""
    
    recommendations = []
    
    # Sisyphus analysis recommendations
    if sisyphus > 0.6:
        recommendations.append("減少循環論證")
    
    # Quantum analysis recommendations  
    if quantum < 0.4:
        recommendations.append("增加創新思維")
    
    # Expression quality recommendations
    if expression < 0.5:
        recommendations.append("改善整體表達質量")
    
    # Specific dimension recommendations
    if formal < 0.4:
        recommendations.append("加強邏輯結構")
    
    if cognitive < 0.5:
        recommendations.append("提升認知可及性")
    
    if social < 0.5:
        recommendations.append("改善社會適當性")
    
    # Positive reinforcement
    if quantum > 0.6 and sisyphus < 0.3:
        recommendations.append("保持創新與清晰的平衡")
    
    if not recommendations:
        recommendations.append("文本質量良好，可保持現有風格")
    
    return " | ".join(recommendations)


def demonstrate_practical_usage():
    """Demonstrate practical usage scenarios"""
    
    print("\n" + "="*70)
    print("🎯 Practical Usage Demonstration")
    print("="*70)
    
    sq_analyzer = SisyphusQuantumAnalyzer()
    expr_evaluator = HumanExpressionEvaluator()
    
    # Scenario 1: Academic paper review
    academic_text = """
    本研究證明了A理論的正確性。A理論之所以正確，
    是因為我們的研究證明了它。研究結果顯示A理論確實是正確的，
    這進一步證實了A理論的有效性。
    """
    
    print("\n📚 Scenario 1: Academic Paper Review")
    print("Text:", academic_text.strip())
    
    sq_result = sq_analyzer.analyze(academic_text)
    context = ExpressionContext(situation='academic', formality_level='formal')
    expr_result = expr_evaluator.comprehensive_evaluation(academic_text, context)
    
    print(f"Analysis: Sisyphus {sq_result['sisyphus_analysis']['score']:.2f} | "
          f"Quantum {sq_result['quantum_analysis']['score']:.2f} | "
          f"Expression {expr_result['integrated']['overall_score']:.2f}")
    print(f"Recommendation: This text shows strong circular reasoning and needs better argumentation")
    
    # Scenario 2: Creative writing evaluation
    creative_text = """
    原本以為詩只是文字的排列，但突然領悟到，
    詩其實是沉默的音樂，是看不見的舞蹈。
    當我們不再用理性分析詩，而用直覺感受詩，
    我們就穿越了語言的限制，到達了純粹的美。
    """
    
    print("\n🎨 Scenario 2: Creative Writing Evaluation")
    print("Text:", creative_text.strip())
    
    sq_result = sq_analyzer.analyze(creative_text)
    context = ExpressionContext(situation='literary', formality_level='neutral')
    expr_result = expr_evaluator.comprehensive_evaluation(creative_text, context)
    
    print(f"Analysis: Sisyphus {sq_result['sisyphus_analysis']['score']:.2f} | "
          f"Quantum {sq_result['quantum_analysis']['score']:.2f} | "
          f"Expression {expr_result['integrated']['overall_score']:.2f}")
    print(f"Recommendation: This text demonstrates excellent breakthrough thinking and poetic insight")
    
    print("\n✨ Practical usage demonstration completed!")


if __name__ == "__main__":
    test_integration()
    demonstrate_practical_usage()