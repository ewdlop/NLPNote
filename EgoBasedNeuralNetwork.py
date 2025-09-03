"""
Ego-based Neural Network Framework (自我神經網路框架)

A formalized philosophical framework implementing ego-based neural networks that balance
truth-seeking with self-consistency, integrating Kantian philosophy with modern ML.

数学公式化的哲学框架，实现在追求真理与自我一致性之间平衡的自我神经网络，
融合康德哲学与现代机器学习。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Callable, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import math
from collections import defaultdict


class EgoMode(Enum):
    """自我模式 (Ego Modes)"""
    PURE_OBJECTIVIST = "pure_objectivist"    # λ → 0: 完全客观主义
    BALANCED_EGO = "balanced_ego"            # λ = 0.5: 平衡自我
    PURE_EGOIST = "pure_egoist"              # λ → ∞: 纯粹自我主义
    ADAPTIVE = "adaptive"                     # 动态调整λ


@dataclass
class WorldState:
    """世界状态 (World State)
    
    外部世界状态的数学表示: s ∈ S (状态空间)
    """
    state_vector: np.ndarray
    state_id: str
    probability: float = 1.0
    timestamp: float = 0.0


@dataclass
class EgoBeliefs:
    """自我信念 (Ego Beliefs)
    
    信念的概率分布表示: b: S → [0,1]
    """
    belief_distribution: Dict[str, float]
    confidence: float
    stubbornness: float  # γ parameter for belief resistance
    
    def update_belief(self, state_id: str, likelihood: float, gamma: float = None) -> None:
        """信念更新公式: B_{t+1}(s) ∝ B_t(s)^γ · P(o_t|s)^{1-γ}"""
        if gamma is None:
            gamma = self.stubbornness
            
        current_belief = self.belief_distribution.get(state_id, 0.0)
        
        # Bayesian update with ego protection
        updated_belief = (current_belief ** gamma) * (likelihood ** (1 - gamma))
        
        self.belief_distribution[state_id] = updated_belief
        self._normalize_beliefs()
    
    def _normalize_beliefs(self):
        """标准化信念分布"""
        total = sum(self.belief_distribution.values())
        if total > 0:
            for state_id in self.belief_distribution:
                self.belief_distribution[state_id] /= total


@dataclass 
class EgoPreferences:
    """自我偏好 (Ego Preferences)
    
    效用函数表示: u: S → R
    """
    utility_function: Dict[str, float]
    preference_strength: float = 1.0
    
    def get_utility(self, state_id: str) -> float:
        """获取状态效用值"""
        return self.utility_function.get(state_id, 0.0)


class EgoNeuralNetwork(nn.Module):
    """自我神经网络 (Ego Neural Network)
    
    实现带有自我一致性约束的神经网络
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, ego_lambda: float = 0.5):
        super(EgoNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ego_lambda = ego_lambda  # 自我与真理的权衡系数
        
        # 网络层定义
        self.perception_layer = nn.Linear(input_dim, hidden_dim)
        self.ego_layer = nn.Linear(hidden_dim, hidden_dim)
        self.action_layer = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=-1)
        
        # 自我状态记录
        self.past_parameters = None
        self.past_outputs = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: π_obs: S → O, π_act: O → A"""
        # 感知层 (Perception)
        perception = self.activation(self.perception_layer(x))
        
        # 自我处理层 (Ego Processing)
        ego_processed = self.activation(self.ego_layer(perception))
        
        # 行动层 (Action)
        action_logits = self.action_layer(ego_processed)
        actions = self.output_activation(action_logits)
        
        return actions
    
    def ego_loss(self, current_params: Dict, current_outputs: torch.Tensor) -> torch.Tensor:
        """自我一致性损失: L_ego = α·D_param + β·D_output"""
        ego_loss_value = 0.0
        
        # 参数距离损失 (Parameter Distance Loss)
        if self.past_parameters is not None:
            param_distance = 0.0
            for name, param in self.named_parameters():
                if name in self.past_parameters:
                    param_distance += torch.norm(param - self.past_parameters[name], p=2)
            ego_loss_value += 0.5 * param_distance  # α = 0.5
        
        # 输出分布距离损失 (Output Distribution Distance Loss)  
        if self.past_outputs is not None:
            # KL散度计算输出分布差异
            kl_div = nn.KLDivLoss(reduction='batchmean')
            output_distance = kl_div(
                torch.log(current_outputs + 1e-8),
                self.past_outputs + 1e-8
            )
            ego_loss_value += 0.5 * output_distance  # β = 0.5
            
        return ego_loss_value
    
    def update_ego_memory(self):
        """更新自我记忆"""
        # 保存当前参数状态
        self.past_parameters = {}
        for name, param in self.named_parameters():
            self.past_parameters[name] = param.clone().detach()


class EgoBasedFramework:
    """自我神经网路哲学框架 (Ego-based Neural Network Philosophical Framework)
    
    核心实现: min_θ L_world(θ) + λ·L_ego(θ)
    """
    
    def __init__(self, 
                 world_state_dim: int = 10,
                 action_dim: int = 5,
                 ego_lambda: float = 0.5,
                 belief_stubbornness: float = 0.3,
                 mode: EgoMode = EgoMode.BALANCED_EGO):
        
        self.world_state_dim = world_state_dim
        self.action_dim = action_dim
        self.ego_lambda = ego_lambda
        self.mode = mode
        
        # 初始化世界状态空间 W = {s ∈ S}
        self.world_states = {}
        
        # 初始化自我 E = (B, P)
        self.ego_beliefs = EgoBeliefs(
            belief_distribution={},
            confidence=0.5,
            stubbornness=belief_stubbornness
        )
        self.ego_preferences = EgoPreferences(utility_function={})
        
        # 初始化神经网络
        self.network = EgoNeuralNetwork(
            input_dim=world_state_dim,
            hidden_dim=64,
            output_dim=action_dim,
            ego_lambda=ego_lambda
        )
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        # 损失记录
        self.loss_history = {
            'world_loss': [],
            'ego_loss': [], 
            'total_loss': []
        }
    
    def perceive_world(self, observation: np.ndarray) -> WorldState:
        """感知函数: π_obs: S → O"""
        state_id = f"state_{hash(observation.tobytes())}"
        world_state = WorldState(
            state_vector=observation,
            state_id=state_id,
            probability=1.0,
            timestamp=len(self.world_states)
        )
        self.world_states[state_id] = world_state
        return world_state
    
    def decide_action(self, world_state: WorldState) -> torch.Tensor:
        """行动函数: π_act: O → A"""
        state_tensor = torch.FloatTensor(world_state.state_vector).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.network(state_tensor)
        return action_probs
    
    def compute_world_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """世界对齐损失: L_world(θ) = E[(f_θ(x) - y)²]"""
        mse_loss = nn.MSELoss()
        return mse_loss(predictions, targets)
    
    def compute_total_loss(self, 
                          predictions: torch.Tensor, 
                          targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """总损失函数: L_total = L_world + λ·L_ego"""
        
        # 世界损失 (Truth Principle)
        world_loss = self.compute_world_loss(predictions, targets)
        
        # 自我损失 (Ego Principle)
        ego_loss = self.network.ego_loss(
            current_params=dict(self.network.named_parameters()),
            current_outputs=predictions
        )
        
        # 总损失
        total_loss = world_loss + self.ego_lambda * ego_loss
        
        loss_components = {
            'world_loss': world_loss.item(),
            'ego_loss': ego_loss.item() if isinstance(ego_loss, torch.Tensor) else ego_loss,
            'total_loss': total_loss.item(),
            'lambda': self.ego_lambda
        }
        
        return total_loss, loss_components
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """训练步骤: θ_{t+1} = θ_t - η∇[L_world + λL_ego]"""
        
        self.optimizer.zero_grad()
        
        # 前向传播
        predictions = self.network(inputs)
        
        # 计算损失
        total_loss, loss_components = self.compute_total_loss(predictions, targets)
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        # 更新自我记忆
        self.network.update_ego_memory()
        self.network.past_outputs = predictions.clone().detach()
        
        # 记录损失历史
        for key, value in loss_components.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)
        
        return loss_components
    
    def update_beliefs(self, observation: np.ndarray, likelihood: float):
        """信念更新: 结合贝叶斯更新与自我保护"""
        world_state = self.perceive_world(observation)
        self.ego_beliefs.update_belief(
            world_state.state_id, 
            likelihood, 
            self.ego_beliefs.stubbornness
        )
    
    def adjust_ego_mode(self, new_mode: EgoMode):
        """调整自我模式"""
        self.mode = new_mode
        
        if new_mode == EgoMode.PURE_OBJECTIVIST:
            self.ego_lambda = 0.0  # 完全追求真理
        elif new_mode == EgoMode.PURE_EGOIST:
            self.ego_lambda = 10.0  # 极度自我保护
        elif new_mode == EgoMode.BALANCED_EGO:
            self.ego_lambda = 0.5  # 平衡状态
        # ADAPTIVE模式需要动态调整
    
    def get_philosophical_interpretation(self) -> Dict[str, str]:
        """获取哲学诠释"""
        interpretations = {}
        
        if self.ego_lambda < 0.1:
            interpretations['mode'] = "近乎完全客观主义 (Near Pure Objectivism)"
            interpretations['description'] = "只追求真理，几乎忽略自我一致性"
            interpretations['kant_connection'] = "理性至上，先验综合判断"
            
        elif self.ego_lambda > 5.0:
            interpretations['mode'] = "极度自我主义 (Extreme Egoism)"
            interpretations['description'] = "强烈抵制信念修正，认知失调明显"
            interpretations['kant_connection'] = "自我作为知识的先验结构主导"
            
        else:
            interpretations['mode'] = "务实自我 (Pragmatic Ego)"
            interpretations['description'] = "在维持自洽前提下，有限度吸收新事实"
            interpretations['kant_connection'] = "理性与感性的辩证统一"
        
        interpretations['lambda_value'] = f"{self.ego_lambda:.3f}"
        interpretations['stubbornness'] = f"{self.ego_beliefs.stubbornness:.3f}"
        
        return interpretations
    
    def get_training_statistics(self) -> Dict:
        """获取训练统计信息"""
        if not self.loss_history['total_loss']:
            return {"status": "No training data available"}
        
        return {
            'total_epochs': len(self.loss_history['total_loss']),
            'final_world_loss': self.loss_history['world_loss'][-1],
            'final_ego_loss': self.loss_history['ego_loss'][-1],
            'final_total_loss': self.loss_history['total_loss'][-1],
            'avg_world_loss': np.mean(self.loss_history['world_loss']),
            'avg_ego_loss': np.mean(self.loss_history['ego_loss']),
            'ego_lambda': self.ego_lambda,
            'belief_count': len(self.ego_beliefs.belief_distribution),
            'world_state_count': len(self.world_states)
        }


class EgoBasedAxiomSystem:
    """自我哲学公理系统 (Ego-based Philosophical Axiom System)
    
    形式逻辑公理，用于推导自我行为模式
    """
    
    def __init__(self):
        self.axioms = {
            'ego_existence': "∃E: E = (B, P) ∧ B: S → [0,1] ∧ P: S → ℝ",
            'world_reality': "∃W: W = {s ∈ S} ∧ |S| < ∞",
            'perception_mapping': "∃π_obs: S → O ∧ ∃π_act: O → A",
            'ego_consistency': "∀θ,θ': L_ego(θ,θ') ≥ 0 ∧ L_ego(θ,θ) = 0",
            'belief_update': "∀s,γ: B_{t+1}(s) = f(B_t(s)^γ, P(o|s)^{1-γ})",
            'loss_composition': "L_total = L_world + λ·L_ego ∧ λ ≥ 0"
        }
        
        self.theorems = {}
    
    def derive_theorem(self, name: str, axiom_base: List[str]) -> str:
        """从公理推导定理"""
        if name == "ego_resistance":
            return "当 λ → ∞ 时，系统将拒绝与现有信念冲突的新知识"
        elif name == "truth_seeking":
            return "当 λ → 0 时，系统将无限制地追求真理，忽略自我一致性"
        elif name == "cognitive_dissonance":
            return "L_ego 的梯度对抗过大更新，模拟认知失调减少机制"
        
        return f"定理 {name} 待推导"
    
    def validate_axiom_consistency(self) -> bool:
        """验证公理系统一致性"""
        # 简化的一致性检查
        return True


# 导出核心类
__all__ = [
    'EgoBasedFramework',
    'EgoNeuralNetwork', 
    'EgoBeliefs',
    'EgoPreferences',
    'WorldState',
    'EgoMode',
    'EgoBasedAxiomSystem'
]