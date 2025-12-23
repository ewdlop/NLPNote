"""
Ego-based Neural Network Examples and Tests
自我神經網路範例與測試

Demonstrates the philosophical framework with practical examples and mathematical validation.
"""

import numpy as np
import torch
from typing import Dict, List
import json

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    from EgoBasedNeuralNetwork import (
        EgoBasedFramework, 
        EgoMode, 
        EgoBasedAxiomSystem,
        WorldState,
        EgoBeliefs,
        EgoPreferences
    )
except ImportError:
    print("EgoBasedNeuralNetwork module not found. Please ensure it's in the Python path.")
    exit(1)


class EgoFrameworkDemo:
    """自我框架演示类 (Ego Framework Demonstration)"""
    
    def __init__(self):
        self.frameworks = {}
        self.demo_results = {}
    
    def create_framework_variants(self):
        """创建不同自我模式的框架变体"""
        configs = [
            (EgoMode.PURE_OBJECTIVIST, 0.0, "Pure Truth Seeker"),
            (EgoMode.BALANCED_EGO, 0.5, "Balanced Ego"),
            (EgoMode.PURE_EGOIST, 5.0, "Extreme Ego Protection"),
            (EgoMode.ADAPTIVE, 1.0, "Adaptive Ego")
        ]
        
        for mode, lambda_val, description in configs:
            framework = EgoBasedFramework(
                world_state_dim=8,
                action_dim=4,
                ego_lambda=lambda_val,
                belief_stubbornness=0.3,
                mode=mode
            )
            
            self.frameworks[description] = framework
            print(f"Created framework: {description} (λ={lambda_val})")
    
    def generate_training_data(self, n_samples: int = 100) -> tuple:
        """生成训练数据"""
        # 创建模拟的世界状态和目标行动
        inputs = torch.randn(n_samples, 8)  # 8维世界状态
        
        # 目标: 简单的线性关系 + 噪音
        targets = torch.matmul(inputs[:, :4], torch.tensor([0.5, -0.3, 0.8, -0.1])).unsqueeze(1)
        targets = torch.cat([targets, 1-targets, targets*0.5, targets*-0.2], dim=1)
        targets = torch.softmax(targets, dim=1)  # 转换为概率分布
        
        return inputs, targets
    
    def train_frameworks(self, epochs: int = 50):
        """训练所有框架变体"""
        inputs, targets = self.generate_training_data(200)
        
        print(f"\n开始训练 {len(self.frameworks)} 个框架变体...")
        
        for name, framework in self.frameworks.items():
            print(f"\n训练框架: {name}")
            epoch_losses = []
            
            for epoch in range(epochs):
                # 随机采样批次
                batch_indices = torch.randperm(len(inputs))[:32]
                batch_inputs = inputs[batch_indices]
                batch_targets = targets[batch_indices]
                
                # 训练步骤
                loss_info = framework.train_step(batch_inputs, batch_targets)
                epoch_losses.append(loss_info)
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Total Loss = {loss_info['total_loss']:.4f}, "
                          f"World Loss = {loss_info['world_loss']:.4f}, "
                          f"Ego Loss = {loss_info['ego_loss']:.4f}")
            
            self.demo_results[name] = {
                'losses': epoch_losses,
                'final_stats': framework.get_training_statistics(),
                'philosophy': framework.get_philosophical_interpretation()
            }
    
    def demonstrate_belief_evolution(self):
        """演示信念演化过程"""
        print("\n=== 信念演化演示 (Belief Evolution Demonstration) ===")
        
        # 创建自我信念系统
        beliefs = EgoBeliefs(
            belief_distribution={'state_A': 0.7, 'state_B': 0.3},
            confidence=0.8,
            stubbornness=0.4  # γ = 0.4
        )
        
        print("初始信念分布:")
        print(f"  State A: {beliefs.belief_distribution['state_A']:.3f}")
        print(f"  State B: {beliefs.belief_distribution['state_B']:.3f}")
        
        # 模拟观察到与信念冲突的证据
        print("\n观察到强烈支持 State B 的证据 (likelihood = 0.9):")
        beliefs.update_belief('state_B', 0.9)
        
        print("更新后信念分布:")
        print(f"  State A: {beliefs.belief_distribution['state_A']:.3f}")
        print(f"  State B: {beliefs.belief_distribution['state_B']:.3f}")
        
        # 展示不同固执程度的影响
        print("\n比较不同固执程度 (γ) 的影响:")
        for gamma in [0.0, 0.3, 0.6, 0.9]:
            test_beliefs = EgoBeliefs(
                belief_distribution={'old_belief': 0.8, 'new_evidence': 0.2},
                confidence=0.8,
                stubbornness=gamma
            )
            test_beliefs.update_belief('new_evidence', 0.9)
            
            print(f"  γ={gamma}: 新证据信念 = {test_beliefs.belief_distribution['new_evidence']:.3f}")
    
    def test_axiom_system(self):
        """测试公理系统"""
        print("\n=== 公理系统测试 (Axiom System Testing) ===")
        
        axiom_system = EgoBasedAxiomSystem()
        
        print("核心公理:")
        for name, axiom in axiom_system.axioms.items():
            print(f"  {name}: {axiom}")
        
        print("\n推导的定理:")
        theorems = [
            'ego_resistance',
            'truth_seeking', 
            'cognitive_dissonance'
        ]
        
        for theorem in theorems:
            result = axiom_system.derive_theorem(theorem, ['ego_existence', 'loss_composition'])
            print(f"  {theorem}: {result}")
        
        consistency = axiom_system.validate_axiom_consistency()
        print(f"\n公理系统一致性验证: {'通过' if consistency else '失败'}")
    
    def analyze_philosophical_spectrum(self):
        """分析哲学光谱"""
        print("\n=== 哲学光谱分析 (Philosophical Spectrum Analysis) ===")
        
        lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        for lambda_val in lambda_values:
            test_framework = EgoBasedFramework(ego_lambda=lambda_val)
            interpretation = test_framework.get_philosophical_interpretation()
            
            print(f"\nλ = {lambda_val}:")
            print(f"  模式: {interpretation['mode']}")
            print(f"  描述: {interpretation['description']}")
            print(f"  康德联系: {interpretation['kant_connection']}")
    
    def test_cognitive_dissonance_simulation(self):
        """测试认知失调模拟"""
        print("\n=== 认知失调模拟 (Cognitive Dissonance Simulation) ===")
        
        # 创建高自我保护的框架
        ego_framework = EgoBasedFramework(ego_lambda=3.0, belief_stubbornness=0.8)
        
        # 模拟与现有信念冲突的新信息
        conflicting_observation = np.array([1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.8, -0.2])
        
        print("模拟认知失调场景:")
        print(f"  初始ego_lambda: {ego_framework.ego_lambda}")
        print(f"  信念固执度: {ego_framework.ego_beliefs.stubbornness}")
        
        # 观察信念更新过程
        for i in range(3):
            likelihood = 0.9  # 强证据
            ego_framework.update_beliefs(conflicting_observation, likelihood)
            
            beliefs_strength = sum(ego_framework.ego_beliefs.belief_distribution.values())
            print(f"  更新 {i+1}: 总信念强度 = {beliefs_strength:.3f}")
    
    def comparative_analysis(self):
        """比较分析结果"""
        print("\n=== 比较分析结果 (Comparative Analysis Results) ===")
        
        if not self.demo_results:
            print("请先运行训练演示")
            return
        
        print("最终训练统计:")
        for name, results in self.demo_results.items():
            stats = results['final_stats']
            philosophy = results['philosophy']
            
            print(f"\n{name}:")
            print(f"  最终世界损失: {stats['final_world_loss']:.4f}")
            print(f"  最终自我损失: {stats['final_ego_loss']:.4f}")
            print(f"  最终总损失: {stats['final_total_loss']:.4f}")
            print(f"  哲学模式: {philosophy['mode']}")
            print(f"  λ值: {philosophy['lambda_value']}")
    
    def save_results(self, filename: str = "ego_framework_results.json"):
        """保存演示结果"""
        try:
            # 转换结果为可序列化格式
            serializable_results = {}
            for name, results in self.demo_results.items():
                serializable_results[name] = {
                    'final_stats': results['final_stats'],
                    'philosophy': results['philosophy'],
                    'loss_count': len(results['losses'])
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n结果已保存到: {filename}")
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    def run_full_demonstration(self):
        """运行完整演示"""
        print("=" * 60)
        print("自我神经网络哲学框架完整演示")
        print("Ego-based Neural Network Philosophical Framework Demo")
        print("=" * 60)
        
        # 1. 创建框架变体
        self.create_framework_variants()
        
        # 2. 信念演化演示
        self.demonstrate_belief_evolution()
        
        # 3. 公理系统测试
        self.test_axiom_system()
        
        # 4. 哲学光谱分析
        self.analyze_philosophical_spectrum()
        
        # 5. 认知失调模拟
        self.test_cognitive_dissonance_simulation()
        
        # 6. 训练框架
        self.train_frameworks(epochs=30)
        
        # 7. 比较分析
        self.comparative_analysis()
        
        # 8. 保存结果
        self.save_results()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("Demonstration completed!")
        print("=" * 60)


def mathematical_validation():
    """数学验证 (Mathematical Validation)"""
    print("\n=== 数学公式验证 (Mathematical Formula Validation) ===")
    
    # 验证信念更新公式
    print("1. 信念更新公式验证: B_{t+1}(s) ∝ B_t(s)^γ · P(o_t|s)^{1-γ}")
    
    B_t = 0.3  # 当前信念
    likelihood = 0.8  # 观察似然
    gamma_values = [0.0, 0.5, 1.0]
    
    for gamma in gamma_values:
        B_t_plus_1 = (B_t ** gamma) * (likelihood ** (1 - gamma))
        print(f"  γ={gamma}: B_{{t+1}} = {B_t_plus_1:.3f}")
    
    # 验证损失函数组合
    print("\n2. 损失函数验证: L_total = L_world + λ·L_ego")
    
    L_world = 0.5
    L_ego = 0.3
    lambda_values = [0.0, 0.5, 1.0, 2.0]
    
    for lambda_val in lambda_values:
        L_total = L_world + lambda_val * L_ego
        print(f"  λ={lambda_val}: L_total = {L_total:.3f}")
    
    print("\n数学验证完成")


def simple_usage_example():
    """简单使用示例 (Simple Usage Example)"""
    print("\n=== 简单使用示例 (Simple Usage Example) ===")
    
    # 创建自我框架
    framework = EgoBasedFramework(
        world_state_dim=4,
        action_dim=2,
        ego_lambda=0.5,  # 平衡自我与真理
        mode=EgoMode.BALANCED_EGO
    )
    
    print("1. 创建自我框架完成")
    
    # 感知世界状态
    observation = np.array([0.5, -0.2, 0.8, -0.3])
    world_state = framework.perceive_world(observation)
    
    print(f"2. 感知世界状态: {world_state.state_id}")
    
    # 做出决策
    action = framework.decide_action(world_state)
    print(f"3. 决策行动概率: {action.numpy().flatten()}")
    
    # 更新信念
    framework.update_beliefs(observation, likelihood=0.7)
    print("4. 信念更新完成")
    
    # 获取哲学诠释
    interpretation = framework.get_philosophical_interpretation()
    print(f"5. 哲学模式: {interpretation['mode']}")
    
    print("简单示例完成")


if __name__ == "__main__":
    # 运行数学验证
    mathematical_validation()
    
    # 运行简单示例
    simple_usage_example()
    
    # 运行完整演示
    demo = EgoFrameworkDemo()
    demo.run_full_demonstration()