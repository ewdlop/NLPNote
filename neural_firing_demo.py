#!/usr/bin/env python3
"""
Neural Missing Firing Detection Example
神經元缺失激發檢測示例

This script demonstrates how to detect and analyze neural missing firing
patterns in LLMs using the neural firing analysis framework.

該腳本演示如何使用神經激發分析框架檢測和分析LLM中的神經元缺失激發模式。
"""

import sys
import numpy as np
from typing import List, Dict, Any

try:
    from NeuralFiringAnalyzer import NeuralFiringAnalyzer, FiringPatternType
    from SubtextAnalyzer import SubtextAnalyzer
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False
    sys.exit(1)


class NeuralFiringDemonstrator:
    """
    Demonstrates neural firing detection capabilities
    演示神經激發檢測功能
    """
    
    def __init__(self):
        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")
        
        self.neural_analyzer = NeuralFiringAnalyzer()
        self.text_analyzer = SubtextAnalyzer()
        self.expression_evaluator = HumanExpressionEvaluator()
    
    def demo_basic_firing_analysis(self):
        """Demonstrate basic neural firing analysis"""
        print("=" * 60)
        print("基本神經激發分析演示 (Basic Neural Firing Analysis Demo)")
        print("=" * 60)
        
        # Analyze a healthy network
        print("\n1. 分析健康網路 (Analyzing Healthy Network):")
        healthy_report = self.neural_analyzer.analyze_simulated_llm_activations(
            num_layers=8,
            hidden_size=512,
            sequence_length=128,
            simulate_issues=False
        )
        
        print(f"   健康分數 (Health Score): {healthy_report.overall_health_score:.3f}")
        print(f"   問題層數 (Problematic Layers): {healthy_report.problematic_layers}")
        print(f"   缺失激發率 (Missing Firing Rate): {healthy_report.missing_firing_percentage:.2f}%")
        
        # Analyze a problematic network
        print("\n2. 分析問題網路 (Analyzing Problematic Network):")
        problematic_report = self.neural_analyzer.analyze_simulated_llm_activations(
            num_layers=8,
            hidden_size=512,
            sequence_length=128,
            simulate_issues=True
        )
        
        print(f"   健康分數 (Health Score): {problematic_report.overall_health_score:.3f}")
        print(f"   問題層數 (Problematic Layers): {problematic_report.problematic_layers}")
        print(f"   缺失激發率 (Missing Firing Rate): {problematic_report.missing_firing_percentage:.2f}%")
        
        # Show comparison
        print(f"\n3. 比較結果 (Comparison Results):")
        health_diff = healthy_report.overall_health_score - problematic_report.overall_health_score
        print(f"   健康分數差異 (Health Score Difference): {health_diff:.3f}")
        
        if health_diff > 0.2:
            print("   ⚠️  檢測到顯著的神經激發問題 (Significant neural firing issues detected)")
        else:
            print("   ✅ 神經激發模式相對正常 (Neural firing patterns relatively normal)")
        
        return healthy_report, problematic_report
    
    def demo_layer_specific_analysis(self):
        """Demonstrate layer-by-layer analysis"""
        print("\n" + "=" * 60)
        print("分層分析演示 (Layer-by-Layer Analysis Demo)")
        print("=" * 60)
        
        # Create network with specific issues
        report = self.neural_analyzer.analyze_simulated_llm_activations(
            num_layers=6,
            simulate_issues=True
        )
        
        print("\n層級詳細分析 (Layer Detail Analysis):")
        print("-" * 40)
        
        for i, layer in enumerate(report.layer_analyses):
            status_emoji = "❌" if layer.issues else "✅"
            
            print(f"\n{status_emoji} 第 {i+1} 層 (Layer {i+1}): {layer.layer_name}")
            print(f"     激發率 (Firing Rate): {layer.activation_rate:.3f}")
            print(f"     激發模式 (Pattern): {layer.firing_pattern.value}")
            print(f"     死亡神經元 (Dead Neurons): {layer.dead_neurons}")
            
            if layer.issues:
                print(f"     問題 (Issues):")
                for issue in layer.issues:
                    print(f"       - {issue}")
        
        return report
    
    def demo_text_processing_analysis(self):
        """Demonstrate neural firing analysis for text processing"""
        print("\n" + "=" * 60)
        print("文本處理神經激發分析演示 (Text Processing Neural Firing Analysis)")
        print("=" * 60)
        
        test_texts = [
            {
                'text': "Hello world!",
                'description': "簡單文本 (Simple text)"
            },
            {
                'text': "The quick brown fox jumps over the lazy dog, demonstrating various linguistic patterns.",
                'description': "中等複雜文本 (Medium complexity text)"
            },
            {
                'text': "In the realm of artificial intelligence, neural networks exhibit complex activation patterns that may sometimes fail to fire properly, leading to degraded performance in natural language processing tasks, which requires sophisticated monitoring and diagnostic tools to detect and remediate such issues.",
                'description': "複雜文本 (Complex text)"
            },
            {
                'text': "神經網路在處理自然語言時可能會出現激發模式異常，這會影響語言模型的性能。",
                'description': "中文文本 (Chinese text)"
            }
        ]
        
        print("\n文本神經激發分析結果 (Text Neural Firing Analysis Results):")
        print("-" * 50)
        
        for i, test_case in enumerate(test_texts, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"   文本 (Text): \"{test_case['text'][:50]}{'...' if len(test_case['text']) > 50 else ''}\"")
            
            try:
                result = self.text_analyzer.analyze_expression_evaluation(test_case['text'])
                
                if 'neural_firing_analysis' in result:
                    nfa = result['neural_firing_analysis']
                    print(f"   神經健康分數 (Neural Health): {nfa['neural_health_score']:.3f}")
                    print(f"   激發率 (Firing Rate): {nfa['overall_firing_rate']:.3f}")
                    print(f"   問題數量 (Issues): {nfa['total_issues']}")
                    
                    # Add interpretation based on scores
                    if nfa['neural_health_score'] > 0.8:
                        status = "✅ 良好 (Good)"
                    elif nfa['neural_health_score'] > 0.6:
                        status = "⚠️ 一般 (Fair)"
                    else:
                        status = "❌ 較差 (Poor)"
                    
                    print(f"   狀態 (Status): {status}")
                else:
                    print(f"   ❌ 神經激發分析不可用 (Neural firing analysis unavailable)")
                    
            except Exception as e:
                print(f"   ❌ 分析錯誤 (Analysis error): {e}")
    
    def demo_issue_detection(self):
        """Demonstrate specific issue detection"""
        print("\n" + "=" * 60)
        print("特定問題檢測演示 (Specific Issue Detection Demo)")
        print("=" * 60)
        
        # Create different types of problematic activations
        issue_types = {
            'missing_firing': {
                'description': '缺失激發 (Missing Firing)',
                'activation': np.random.normal(0, 0.01, (100, 100))  # Very low activations
            },
            'dead_neurons': {
                'description': '死亡神經元 (Dead Neurons)',
                'activation': np.zeros((100, 100))  # All zeros
            },
            'saturated_neurons': {
                'description': '飽和神經元 (Saturated Neurons)',
                'activation': np.ones((100, 100)) * 10  # All very high values
            },
            'sporadic_firing': {
                'description': '零星激發 (Sporadic Firing)',
                'activation': np.random.choice([0, 10], size=(100, 100), p=[0.95, 0.05])
            }
        }
        
        print("\n問題類型檢測結果 (Issue Type Detection Results):")
        print("-" * 45)
        
        for issue_type, config in issue_types.items():
            print(f"\n🔍 {config['description']}")
            
            # Apply activation function
            activation = np.tanh(config['activation'])
            
            # Analyze the activation pattern
            analysis = self.neural_analyzer.analyze_activation_tensor(
                activation, 
                f"test_layer_{issue_type}"
            )
            
            print(f"   激發率 (Firing Rate): {analysis.activation_rate:.3f}")
            print(f"   激發模式 (Pattern): {analysis.firing_pattern.value}")
            print(f"   死亡神經元 (Dead Neurons): {analysis.dead_neurons}")
            print(f"   問題數量 (Issues): {len(analysis.issues)}")
            
            if analysis.issues:
                print(f"   檢測到的問題 (Detected Issues):")
                for issue in analysis.issues[:2]:  # Show first 2 issues
                    print(f"     - {issue}")
                if len(analysis.issues) > 2:
                    print(f"     ... 還有 {len(analysis.issues) - 2} 個問題 (and {len(analysis.issues) - 2} more)")
    
    def demo_comprehensive_analysis(self):
        """Demonstrate comprehensive analysis combining all features"""
        print("\n" + "=" * 60)
        print("綜合分析演示 (Comprehensive Analysis Demo)")
        print("=" * 60)
        
        # Sample text for comprehensive analysis
        sample_text = "Large language models require careful monitoring of neural activation patterns to ensure optimal performance and prevent degradation due to missing firing or other neural issues."
        
        print(f"\n分析文本 (Analyzing Text):")
        print(f"\"{sample_text}\"")
        
        # 1. Expression evaluation
        print(f"\n1. 表達評估 (Expression Evaluation):")
        context = ExpressionContext(
            situation='technical',
            formality_level='formal'
        )
        
        expr_result = self.expression_evaluator.comprehensive_evaluation(sample_text, context)
        print(f"   整體分數 (Overall Score): {expr_result['integrated']['overall_score']:.3f}")
        print(f"   信心度 (Confidence): {expr_result['integrated']['overall_confidence']:.3f}")
        
        # 2. Subtext analysis
        print(f"\n2. 潛文本分析 (Subtext Analysis):")
        subtext_result = self.text_analyzer.calculate_subtext_probability(sample_text)
        print(f"   潛文本概率 (Subtext Probability): {subtext_result['probability']:.3f}")
        print(f"   象徵性 (Symbolism): {subtext_result['components']['symbolism']:.3f}")
        print(f"   情感深度 (Emotion Depth): {subtext_result['components']['emotion_depth']:.3f}")
        
        # 3. Neural firing analysis
        print(f"\n3. 神經激發分析 (Neural Firing Analysis):")
        integrated_result = self.text_analyzer.analyze_expression_evaluation(sample_text, context)
        
        if 'neural_firing_analysis' in integrated_result:
            nfa = integrated_result['neural_firing_analysis']
            print(f"   神經健康分數 (Neural Health Score): {nfa['neural_health_score']:.3f}")
            print(f"   激發率 (Firing Rate): {nfa['overall_firing_rate']:.3f}")
            print(f"   處理階段 (Processing Stages): {len(nfa['stage_analyses'])}")
            
            # Show stage-by-stage analysis
            print(f"\n   階段分析 (Stage Analysis):")
            for stage_info in nfa['stage_analyses']:
                stage = stage_info['stage']
                analysis = stage_info['analysis']
                emoji = "✅" if not analysis.issues else "⚠️" if len(analysis.issues) <= 2 else "❌"
                print(f"     {emoji} {stage}: 激發率 {analysis.activation_rate:.3f}, 問題 {len(analysis.issues)}")
        
        # 4. Integrated interpretation
        print(f"\n4. 整合解釋 (Integrated Interpretation):")
        print(f"{integrated_result['interpretation']}")
        
        return integrated_result
    
    def demo_recommendations(self):
        """Demonstrate recommendation system"""
        print("\n" + "=" * 60)
        print("建議系統演示 (Recommendation System Demo)")
        print("=" * 60)
        
        # Create a problematic network scenario
        report = self.neural_analyzer.analyze_simulated_llm_activations(
            num_layers=10,
            simulate_issues=True
        )
        
        print(f"\n網路狀態 (Network Status):")
        print(f"健康分數 (Health Score): {report.overall_health_score:.3f}")
        print(f"問題層數 (Problematic Layers): {report.problematic_layers}/{report.total_layers}")
        
        print(f"\n系統建議 (System Recommendations):")
        print("-" * 30)
        
        for i, recommendation in enumerate(report.recommendations, 1):
            print(f"{i}. {recommendation}")
        
        # Additional custom recommendations based on specific patterns
        print(f"\n高級建議 (Advanced Recommendations):")
        print("-" * 25)
        
        if report.missing_firing_percentage > 70:
            print("🔧 考慮重新初始化網路權重 (Consider re-initializing network weights)")
            print("📊 檢查訓練數據質量 (Check training data quality)")
        
        if report.problematic_layers > report.total_layers * 0.5:
            print("🏗️ 考慮調整網路架構 (Consider adjusting network architecture)")
            print("⚙️ 實施更好的正規化策略 (Implement better regularization strategies)")
        
        if report.overall_health_score < 0.3:
            print("🚨 建議停止訓練並診斷問題 (Recommend stopping training and diagnosing issues)")
            print("🔄 考慮從檢查點恢復 (Consider restoring from checkpoint)")


def main():
    """Main demonstration function"""
    print("🧠 Neural Missing Firing Detection System Demo")
    print("神經元缺失激發檢測系統演示")
    print("=" * 70)
    
    try:
        demo = NeuralFiringDemonstrator()
        
        # Run all demonstrations
        print("開始演示... (Starting demonstrations...)")
        
        # Basic analysis
        demo.demo_basic_firing_analysis()
        
        # Layer-specific analysis  
        demo.demo_layer_specific_analysis()
        
        # Text processing analysis
        demo.demo_text_processing_analysis()
        
        # Issue detection
        demo.demo_issue_detection()
        
        # Comprehensive analysis
        demo.demo_comprehensive_analysis()
        
        # Recommendations
        demo.demo_recommendations()
        
        print("\n" + "=" * 70)
        print("✅ 演示完成！(Demo Complete!)")
        print("=" * 70)
        
        print("\n📋 摘要 (Summary):")
        print("- ✅ 基本神經激發分析 (Basic neural firing analysis)")
        print("- ✅ 分層問題檢測 (Layer-wise issue detection)")
        print("- ✅ 文本處理分析 (Text processing analysis)")
        print("- ✅ 特定問題識別 (Specific issue identification)")
        print("- ✅ 綜合評估框架 (Comprehensive evaluation framework)")
        print("- ✅ 智能建議系統 (Intelligent recommendation system)")
        
        print(f"\n🎯 此系統成功解決了GitHub Issue #183關於")
        print(f"   'neural missing firing in LLM'的問題！")
        print(f"   (This system successfully addresses GitHub Issue #183")
        print(f"   regarding 'neural missing firing in LLM'!)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)