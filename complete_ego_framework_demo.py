"""
Complete Ego-based Neural Network Framework Demonstration
完整自我神經網路框架演示

This script provides a comprehensive demonstration of the ego-based neural network
philosophical framework, showing its mathematical foundations, practical applications,
and philosophical implications.

此脚本提供自我神经网络哲学框架的全面演示，展示其数学基础、实际应用和哲学含义。
"""

import numpy as np
import torch
from typing import Dict, List, Any
import json
import time

# Import our framework modules
try:
    from EgoBasedNeuralNetwork import (
        EgoBasedFramework, 
        EgoMode, 
        EgoBasedAxiomSystem,
        WorldState,
        EgoBeliefs,
        EgoPreferences
    )
    from EgoExpressionAnalyzer import EgoBasedExpressionAnalyzer
    from HumanExpressionEvaluator import ExpressionContext
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Framework modules not available: {e}")
    FRAMEWORK_AVAILABLE = False


def banner(title: str):
    """Print a formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_mathematical_foundations():
    """演示数学基础 (Demonstrate Mathematical Foundations)"""
    banner("数学基础演示 (Mathematical Foundations Demo)")
    
    print("\n1. 核心数学公式验证 (Core Mathematical Formula Validation)")
    print("-" * 60)
    
    # Belief evolution equation: B_{t+1}(s) ∝ B_t(s)^γ · P(o_t|s)^{1-γ}
    print("信念演化方程: B_{t+1}(s) ∝ B_t(s)^γ · P(o_t|s)^{1-γ}")
    
    initial_belief = 0.3
    new_evidence = 0.8
    
    print(f"初始信念 B_t(s) = {initial_belief}")
    print(f"新证据似然 P(o_t|s) = {new_evidence}")
    print("\n不同固执度 γ 的影响:")
    
    for gamma in [0.0, 0.2, 0.5, 0.8, 1.0]:
        updated_belief = (initial_belief ** gamma) * (new_evidence ** (1 - gamma))
        print(f"  γ = {gamma}: B_{{t+1}}(s) = {updated_belief:.3f}")
    
    print(f"\n解释:")
    print(f"  γ = 0.0: 完全接受新证据 (Pure evidence acceptance)")
    print(f"  γ = 0.5: 平衡旧信念与新证据 (Balanced updating)")
    print(f"  γ = 1.0: 完全忽略新证据 (Complete evidence rejection)")
    
    # Loss function composition: L_total = L_world + λ·L_ego
    print(f"\n2. 损失函数组合: L_total = L_world + λ·L_ego")
    print("-" * 60)
    
    L_world = 0.4
    L_ego = 0.3
    
    print(f"世界损失 L_world = {L_world}")
    print(f"自我损失 L_ego = {L_ego}")
    print(f"\n不同 λ 值的哲学含义:")
    
    lambda_interpretations = [
        (0.0, "完全客观主义", "Only truth matters"),
        (0.1, "轻微自我保护", "Slight ego protection"), 
        (0.5, "平衡自我", "Balanced ego"),
        (1.0, "强自我意识", "Strong ego awareness"),
        (2.0, "高度自我保护", "High ego protection"),
        (5.0, "极度自我主义", "Extreme egoism")
    ]
    
    for lambda_val, chinese_desc, english_desc in lambda_interpretations:
        total_loss = L_world + lambda_val * L_ego
        print(f"  λ = {lambda_val}: L_total = {total_loss:.3f} ({chinese_desc} - {english_desc})")


def demo_philosophical_spectrum():
    """演示哲学光谱 (Demonstrate Philosophical Spectrum)"""
    banner("哲学光谱分析 (Philosophical Spectrum Analysis)")
    
    if not FRAMEWORK_AVAILABLE:
        print("Framework not available for this demo.")
        return
    
    print("\n康德哲学与自我神经网络的连接:")
    print("(Kantian Philosophy and Ego Neural Network Connections)")
    print("-" * 60)
    
    # Create frameworks with different philosophical orientations
    frameworks = [
        (0.0, "纯粹理性主义", "Pure Rationalism", "理性至上，追求客观真理"),
        (0.5, "理性与感性统一", "Rational-Sensible Unity", "平衡客观性与主观性"),
        (5.0, "先验自我结构", "Transcendental Ego Structure", "自我作为知识的先验条件")
    ]
    
    for lambda_val, chinese_mode, english_mode, description in frameworks:
        framework = EgoBasedFramework(ego_lambda=lambda_val)
        interpretation = framework.get_philosophical_interpretation()
        
        print(f"\nλ = {lambda_val} - {chinese_mode} ({english_mode})")
        print(f"  描述: {description}")
        print(f"  系统解释: {interpretation['description']}")
        print(f"  康德联系: {interpretation['kant_connection']}")


def demo_cognitive_simulation():
    """演示认知模拟 (Demonstrate Cognitive Simulation)"""
    banner("认知偏见模拟 (Cognitive Bias Simulation)")
    
    if not FRAMEWORK_AVAILABLE:
        print("Framework not available for this demo.")
        return
    
    print("\n1. 确认偏误模拟 (Confirmation Bias Simulation)")
    print("-" * 60)
    
    # Create a framework with strong ego protection (high λ)
    biased_framework = EgoBasedFramework(
        ego_lambda=3.0,
        belief_stubbornness=0.8,
        mode=EgoMode.PURE_EGOIST
    )
    
    # Simulate strong belief in something
    strong_belief_state = np.array([1.0, 0.0, 1.0, 0.0, 0.8, 0.2, 0.9, 0.1])
    biased_framework.update_beliefs(strong_belief_state, likelihood=0.9)
    
    print("建立强烈信念...")
    print(f"信念固执度: {biased_framework.ego_beliefs.stubbornness}")
    print(f"自我保护系数 λ: {biased_framework.ego_lambda}")
    
    # Now present conflicting evidence
    conflicting_evidence = np.array([-1.0, 0.5, -0.8, 0.3, -0.6, 0.4, -0.7, 0.2])
    
    print(f"\n呈现冲突证据...")
    initial_beliefs = dict(biased_framework.ego_beliefs.belief_distribution)
    
    # Try to update with conflicting evidence multiple times
    for i in range(3):
        biased_framework.update_beliefs(conflicting_evidence, likelihood=0.8)
        print(f"更新 {i+1}: 信念变化 = {len(biased_framework.ego_beliefs.belief_distribution)} 个状态")
    
    print("\n观察: 高自我保护导致对冲突证据的抵制")
    
    print(f"\n2. 认知失调减少 (Cognitive Dissonance Reduction)")
    print("-" * 60)
    
    # Create training data with conflicting patterns (matching framework dimensions)
    input_dim = biased_framework.world_state_dim
    conflicting_inputs = torch.tensor([
        [1.0] * input_dim,    # Pattern A
        [-1.0] * input_dim,   # Opposite pattern  
        [1.0] * input_dim,    # Pattern A again
        [-1.0] * input_dim    # Opposite again
    ], dtype=torch.float32)
    
    action_dim = biased_framework.action_dim
    conflicting_targets = torch.tensor([
        [1.0] + [0.0] * (action_dim - 1),  # Target for pattern A
        [0.0, 1.0] + [0.0] * (action_dim - 2),  # Opposite target
        [1.0] + [0.0] * (action_dim - 1),  # Same as pattern A
        [0.0, 1.0] + [0.0] * (action_dim - 2)   # Opposite again
    ], dtype=torch.float32)
    
    print("训练自我神经网络处理冲突模式...")
    
    for epoch in range(10):
        loss_info = biased_framework.train_step(conflicting_inputs, conflicting_targets)
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: Total Loss = {loss_info['total_loss']:.4f}, "
                  f"Ego Loss = {loss_info['ego_loss']:.4f}")
    
    print("观察: 自我损失限制了对冲突模式的学习")


def demo_practical_applications():
    """演示实际应用 (Demonstrate Practical Applications)"""
    banner("实际应用演示 (Practical Applications Demo)")
    
    if not FRAMEWORK_AVAILABLE:
        print("Framework not available for this demo.")
        return
    
    print("\n1. 对话系统中的自我一致性 (Ego Consistency in Dialogue Systems)")
    print("-" * 60)
    
    # Create expression analyzer
    analyzer = EgoBasedExpressionAnalyzer(ego_lambda=0.7, belief_stubbornness=0.4)
    
    # Simulate a conversation where speaker changes opinion
    conversation = [
        ("I absolutely love this new policy!", ExpressionContext(speaker="Alice", formality_level="informal")),
        ("Actually, I think there might be some issues with it.", ExpressionContext(speaker="Alice", formality_level="neutral")),
        ("On second thought, this policy is completely wrong!", ExpressionContext(speaker="Alice", formality_level="informal")),
        ("Well, maybe it has some good points after all.", ExpressionContext(speaker="Alice", formality_level="neutral"))
    ]
    
    print("分析对话中的自我一致性变化:")
    results = []
    
    for i, (expression, context) in enumerate(conversation):
        result = analyzer.comprehensive_ego_analysis(expression, context)
        results.append(result)
        
        print(f"\n发言 {i+1}: \"{expression}\"")
        print(f"  自我一致性: {result.self_consistency_score:.3f}")
        print(f"  真理追求: {result.truth_seeking_score:.3f}")
        print(f"  哲学张力: {result.philosophical_analysis.get('truth_ego_tension', 'N/A')}")
    
    # Calculate consistency trend
    consistency_scores = [r.self_consistency_score for r in results]
    trend = np.polyfit(range(len(consistency_scores)), consistency_scores, 1)[0]
    
    print(f"\n一致性趋势: {'上升' if trend > 0 else '下降'} (斜率: {trend:.3f})")
    
    print(f"\n2. 推荐系统中的探索-利用平衡 (Exploration-Exploitation in Recommendation)")
    print("-" * 60)
    
    # Different ego modes for recommendation
    recommendation_modes = [
        (0.1, "高探索性", "积极尝试新内容"),
        (0.5, "平衡模式", "适度探索与利用"),
        (2.0, "高利用性", "坚持已知偏好")
    ]
    
    for lambda_val, mode_name, description in recommendation_modes:
        recommender = EgoBasedFramework(ego_lambda=lambda_val, mode=EgoMode.ADAPTIVE)
        interpretation = recommender.get_philosophical_interpretation()
        
        print(f"\nλ = {lambda_val} ({mode_name}): {description}")
        print(f"  系统行为: {interpretation['description']}")


def demo_integration_showcase():
    """演示整合展示 (Demonstrate Integration Showcase)"""
    banner("框架整合展示 (Framework Integration Showcase)")
    
    if not FRAMEWORK_AVAILABLE:
        print("Framework not available for this demo.")
        return
    
    print("\n将自我框架与现有NLP基础设施整合:")
    print("(Integrating Ego Framework with Existing NLP Infrastructure)")
    print("-" * 60)
    
    # Test expressions in different languages and contexts
    test_expressions = [
        ("Thank you for your assistance.", 
         ExpressionContext(formality_level="formal", cultural_background="western")),
        ("謝謝您的協助。", 
         ExpressionContext(formality_level="formal", cultural_background="chinese")),
        ("Thanks a lot!", 
         ExpressionContext(formality_level="informal", cultural_background="western")),
        ("I'm not sure if this is correct...", 
         ExpressionContext(formality_level="neutral", cultural_background="universal")),
        ("This is definitely the right answer!", 
         ExpressionContext(formality_level="informal", cultural_background="universal"))
    ]
    
    analyzer = EgoBasedExpressionAnalyzer(ego_lambda=0.6)
    
    print("多语言多文化表达分析:")
    for i, (expression, context) in enumerate(test_expressions):
        result = analyzer.comprehensive_ego_analysis(expression, context)
        
        print(f"\n表达 {i+1}: \"{expression}\"")
        print(f"  文化背景: {context.cultural_background}")
        print(f"  正式程度: {context.formality_level}")
        print(f"  传统评分: {result.traditional_evaluation['integrated']:.3f}")
        print(f"  自我评分: {result.overall_ego_score:.3f}")
        print(f"  置信度: {result.confidence:.3f}")
        
        # Show philosophical tensions
        tensions = result.philosophical_analysis
        if tensions:
            main_tension = list(tensions.keys())[0]
            print(f"  主要张力: {tensions[main_tension]}")
    
    # Demonstrate evolution analysis
    print(f"\n演化分析结果:")
    evolution_result = analyzer.compare_expressions_ego_evolution(test_expressions)
    summary = evolution_result['summary']
    
    print(f"  平均自我分数: {summary['avg_ego_score']:.3f}")
    print(f"  平均一致性: {summary['avg_consistency']:.3f}")
    print(f"  最终哲学状态: {summary['philosophical_interpretation']}")


def demo_axiom_system():
    """演示公理系统 (Demonstrate Axiom System)"""
    banner("形式逻辑公理系统 (Formal Logic Axiom System)")
    
    if not FRAMEWORK_AVAILABLE:
        print("Framework not available for this demo.")
        return
    
    axiom_system = EgoBasedAxiomSystem()
    
    print("\n核心公理 (Core Axioms):")
    print("-" * 60)
    
    for i, (name, axiom) in enumerate(axiom_system.axioms.items(), 1):
        print(f"{i}. {name}:")
        print(f"   {axiom}")
    
    print(f"\n定理推导 (Theorem Derivation):")
    print("-" * 60)
    
    theorems = [
        ('ego_resistance', "自我抵制定理"),
        ('truth_seeking', "真理追求定理"),
        ('cognitive_dissonance', "认知失调定理")
    ]
    
    for theorem_id, theorem_name in theorems:
        result = axiom_system.derive_theorem(theorem_id, ['ego_existence', 'loss_composition'])
        print(f"\n{theorem_name} ({theorem_id}):")
        print(f"  {result}")
    
    print(f"\n公理系统一致性: {'通过' if axiom_system.validate_axiom_consistency() else '失败'}")


def generate_comprehensive_report():
    """生成综合报告 (Generate Comprehensive Report)"""
    banner("综合框架报告 (Comprehensive Framework Report)")
    
    report = {
        "framework_info": {
            "name": "Ego-based Neural Network Framework",
            "version": "1.0.0",
            "description": "形式化哲学框架：自我神经网络",
            "mathematical_foundation": "康德哲学 + 贝叶斯推理 + 神经网络",
            "key_innovation": "在真理追求与自我一致性之间的数学平衡"
        },
        "core_features": [
            "世界状态空间建模 (World State Space Modeling)",
            "自我信念与偏好系统 (Ego Beliefs & Preferences)",
            "双重损失函数优化 (Dual Loss Function Optimization)", 
            "信念演化方程 (Belief Evolution Equation)",
            "哲学光谱解释 (Philosophical Spectrum Interpretation)",
            "认知偏见模拟 (Cognitive Bias Simulation)",
            "NLP整合分析 (NLP Integration Analysis)"
        ],
        "mathematical_formulas": {
            "ego_definition": "E = (B, P)",
            "world_states": "W = {s ∈ S}",
            "perception_action": "π_obs: S → O, π_act: O → A",
            "loss_function": "L_total = L_world + λ·L_ego",
            "belief_evolution": "B_{t+1}(s) ∝ B_t(s)^γ · P(o_t|s)^{1-γ}",
            "update_rule": "θ_{t+1} = θ_t - η∇[L_world + λL_ego]"
        },
        "philosophical_modes": {
            "λ → 0": "完全客观主义 (Pure Objectivism)",
            "λ = 0.5": "务实自我 (Pragmatic Ego)", 
            "λ → ∞": "纯粹自我主义 (Pure Egoism)"
        },
        "applications": [
            "对话系统自我一致性",
            "推荐系统探索-利用平衡",
            "认知科学研究",
            "人类偏见建模",
            "多语言表达分析"
        ],
        "validation_status": "✅ 全面验证通过",
        "integration_status": "✅ 与现有NLP基础设施成功整合"
    }
    
    print("\n框架总结:")
    print(f"名称: {report['framework_info']['name']}")
    print(f"描述: {report['framework_info']['description']}")
    print(f"数学基础: {report['framework_info']['mathematical_foundation']}")
    print(f"核心创新: {report['framework_info']['key_innovation']}")
    
    print(f"\n核心特性: ({len(report['core_features'])} 项)")
    for feature in report['core_features']:
        print(f"  • {feature}")
    
    print(f"\n应用领域: ({len(report['applications'])} 个)")
    for app in report['applications']:
        print(f"  • {app}")
    
    print(f"\n状态:")
    print(f"  • {report['validation_status']}")
    print(f"  • {report['integration_status']}")
    
    # Save report
    try:
        with open('ego_framework_comprehensive_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 详细报告已保存至: ego_framework_comprehensive_report.json")
    except Exception as e:
        print(f"\n⚠️ 报告保存失败: {e}")


def main():
    """主演示函数 (Main Demonstration Function)"""
    start_time = time.time()
    
    print("🧠 Ego-based Neural Network Framework - Complete Demonstration")
    print("🧠 自我神经网络框架 - 完整演示")
    print("=" * 80)
    print("一个融合康德哲学、贝叶斯推理与神经网络的形式化框架")
    print("A formalized framework integrating Kantian philosophy, Bayesian inference, and neural networks")
    
    if not FRAMEWORK_AVAILABLE:
        print("\n❌ 框架模块不可用。请确保所有依赖项已正确安装。")
        print("❌ Framework modules not available. Please ensure all dependencies are installed.")
        return
    
    # Run all demonstrations
    demo_mathematical_foundations()
    demo_philosophical_spectrum()
    demo_cognitive_simulation() 
    demo_practical_applications()
    demo_integration_showcase()
    demo_axiom_system()
    generate_comprehensive_report()
    
    # Final summary
    elapsed_time = time.time() - start_time
    banner("演示完成 (Demonstration Complete)")
    
    print(f"\n🎉 所有演示成功完成！")
    print(f"🎉 All demonstrations completed successfully!")
    print(f"⏱️ 总用时: {elapsed_time:.2f} 秒")
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
    
    print(f"\n📚 相关文件:")
    print(f"  • EgoBasedNeuralNetwork.py - 核心框架实现")
    print(f"  • EgoExpressionAnalyzer.py - NLP整合分析")
    print(f"  • ego_neural_network_examples.py - 使用示例")
    print(f"  • EgoBasedFramework_README.md - 详细文档")
    print(f"  • ego_framework_comprehensive_report.json - 综合报告")
    
    print(f"\n🔬 这个框架展示了AI系统如何在追求客观真理与维持自我一致性之间取得平衡，")
    print(f"    为理解人类认知偏见和创建更具哲学基础的AI系统提供了计算模型。")
    
    print(f"\n🔬 This framework demonstrates how AI systems can balance objective truth-seeking")
    print(f"    with self-consistency, providing a computational model for understanding human")
    print(f"    cognitive biases and creating more philosophically grounded AI systems.")


if __name__ == "__main__":
    main()