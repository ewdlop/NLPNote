#!/usr/bin/env python3
"""
量子道德學實驗 (Quantum Ethics Experiments)
基於普朗克單位、ε與位元的哲學探索實現

This module implements computational experiments based on the philosophical
framework connecting Planck units, epsilon, and bits with moral and existential concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from typing import Tuple, List, Dict, Any
import random
from dataclasses import dataclass
from enum import Enum

# 物理常數 (Physical Constants)
PLANCK_LENGTH = np.sqrt(constants.hbar * constants.G / constants.c**3)
PLANCK_TIME = np.sqrt(constants.hbar * constants.G / constants.c**5)
PLANCK_ENERGY = np.sqrt(constants.hbar * constants.c**5 / constants.G)

class MoralState(Enum):
    """道德狀態枚舉 (Moral State Enumeration)"""
    GOOD = 1
    EVIL = -1
    NEUTRAL = 0

@dataclass
class QuantumMoralState:
    """量子道德狀態 (Quantum Moral State)"""
    good_amplitude: complex
    evil_amplitude: complex
    neutral_amplitude: complex
    
    def __post_init__(self):
        """確保狀態歸一化 (Ensure state normalization)"""
        norm = np.sqrt(abs(self.good_amplitude)**2 + 
                      abs(self.evil_amplitude)**2 + 
                      abs(self.neutral_amplitude)**2)
        if norm > 0:
            self.good_amplitude /= norm
            self.evil_amplitude /= norm
            self.neutral_amplitude /= norm
    
    @property
    def probabilities(self) -> Dict[str, float]:
        """計算各狀態機率 (Calculate state probabilities)"""
        return {
            'good': abs(self.good_amplitude)**2,
            'evil': abs(self.evil_amplitude)**2,
            'neutral': abs(self.neutral_amplitude)**2
        }
    
    def measure(self) -> MoralState:
        """道德狀態測量 (Moral state measurement)"""
        probs = self.probabilities
        rand = random.random()
        
        if rand < probs['good']:
            return MoralState.GOOD
        elif rand < probs['good'] + probs['evil']:
            return MoralState.EVIL
        else:
            return MoralState.NEUTRAL

class PlanckScaleAnalyzer:
    """普朗克尺度分析器 (Planck Scale Analyzer)"""
    
    def __init__(self):
        self.planck_length = PLANCK_LENGTH
        self.planck_time = PLANCK_TIME
        self.planck_energy = PLANCK_ENERGY
    
    def uncertainty_principle(self, delta_x: float) -> float:
        """海森堡不確定性原理 (Heisenberg Uncertainty Principle)"""
        delta_p = constants.hbar / (2 * delta_x)
        return delta_p
    
    def quantum_moral_uncertainty(self, moral_precision: float) -> float:
        """量子道德不確定性 (Quantum Moral Uncertainty)"""
        # 類比不確定性原理於道德領域
        return constants.hbar / (2 * moral_precision)
    
    def planck_scale_decision_time(self) -> float:
        """普朗克尺度決策時間 (Planck Scale Decision Time)"""
        return self.planck_time

class EpsilonAnalyzer:
    """ε分析器 (Epsilon Analyzer)"""
    
    def __init__(self, default_epsilon: float = 1e-10):
        self.default_epsilon = default_epsilon
    
    def limit_approach(self, func, target: float, epsilon: float = None) -> Tuple[bool, float]:
        """極限逼近分析 (Limit Approach Analysis)"""
        if epsilon is None:
            epsilon = self.default_epsilon
        
        # 模擬逼近過程
        x_values = np.linspace(target - 0.1, target + 0.1, 1000)
        y_values = [func(x) for x in x_values]
        
        # 檢查是否在ε鄰域內
        target_y = func(target)
        within_epsilon = np.abs(np.array(y_values) - target_y) < epsilon
        
        return np.any(within_epsilon), target_y
    
    def moral_truth_convergence(self, actions: List[float], epsilon: float = None) -> bool:
        """道德真理收斂性 (Moral Truth Convergence)"""
        if epsilon is None:
            epsilon = self.default_epsilon
        
        if len(actions) < 2:
            return False
        
        # 計算行動序列的收斂性
        differences = np.diff(actions)
        return np.all(np.abs(differences[-10:]) < epsilon)  # 檢查最後10個差值
    
    def critical_point_detection(self, values: List[float]) -> List[int]:
        """臨界點檢測 (Critical Point Detection)"""
        critical_points = []
        for i in range(1, len(values) - 1):
            # 檢測跳躍不連續點
            left_diff = abs(values[i] - values[i-1])
            right_diff = abs(values[i+1] - values[i])
            
            if left_diff > 10 * self.default_epsilon or right_diff > 10 * self.default_epsilon:
                critical_points.append(i)
        
        return critical_points

class BitInformationAnalyzer:
    """位元資訊分析器 (Bit Information Analyzer)"""
    
    def shannon_entropy(self, probabilities: List[float]) -> float:
        """香農熵計算 (Shannon Entropy Calculation)"""
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def moral_information_content(self, moral_sequence: List[MoralState]) -> float:
        """道德資訊內容 (Moral Information Content)"""
        # 計算道德序列的資訊量
        counts = {state: moral_sequence.count(state) for state in MoralState}
        total = len(moral_sequence)
        
        if total == 0:
            return 0
        
        probabilities = [counts[state] / total for state in MoralState]
        return self.shannon_entropy(probabilities)
    
    def kolmogorov_complexity_estimate(self, binary_string: str) -> int:
        """Kolmogorov複雜度估計 (Kolmogorov Complexity Estimate)"""
        # 使用壓縮長度作為複雜度的近似
        import zlib
        compressed = zlib.compress(binary_string.encode())
        return len(compressed)
    
    def information_preservation_score(self, original: List[int], 
                                    transmitted: List[int]) -> float:
        """資訊保存分數 (Information Preservation Score)"""
        if len(original) != len(transmitted):
            return 0.0
        
        correct_bits = sum(1 for o, t in zip(original, transmitted) if o == t)
        return correct_bits / len(original)

class FundamentalUnitsExperiment:
    """基本單位實驗 (Fundamental Units Experiment)"""
    
    def __init__(self):
        self.planck_analyzer = PlanckScaleAnalyzer()
        self.epsilon_analyzer = EpsilonAnalyzer()
        self.bit_analyzer = BitInformationAnalyzer()
    
    def quantum_moral_evolution(self, initial_state: QuantumMoralState, 
                              time_steps: int) -> List[QuantumMoralState]:
        """量子道德演化 (Quantum Moral Evolution)"""
        states = [initial_state]
        current_state = initial_state
        
        for _ in range(time_steps):
            # 量子演化算子（簡化版）
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            
            # 創建新的量子狀態
            new_good = current_state.good_amplitude * np.cos(theta) + \
                      current_state.evil_amplitude * np.sin(theta) * np.exp(1j * phi)
            new_evil = current_state.evil_amplitude * np.cos(theta) - \
                      current_state.good_amplitude * np.sin(theta) * np.exp(-1j * phi)
            new_neutral = current_state.neutral_amplitude * \
                         np.exp(1j * random.uniform(0, 2 * np.pi))
            
            current_state = QuantumMoralState(new_good, new_evil, new_neutral)
            states.append(current_state)
        
        return states
    
    def simulate_midnight_crisis(self, crisis_intensity: float) -> Dict[str, Any]:
        """模擬子夜危機 (Simulate Midnight Crisis)"""
        # 創建危機函數
        def crisis_function(t):
            return crisis_intensity / (1 + (t - 0.5)**2)  # 在t=0.5達到峰值
        
        time_points = np.linspace(0, 1, 1000)
        crisis_values = [crisis_function(t) for t in time_points]
        
        # 檢測臨界點
        critical_points = self.epsilon_analyzer.critical_point_detection(crisis_values)
        
        # 計算資訊熵
        max_val = max(crisis_values)
        normalized_values = [v / max_val for v in crisis_values]
        entropy = self.bit_analyzer.shannon_entropy(normalized_values[:10])  # 取前10個值
        
        return {
            'time_points': time_points,
            'crisis_values': crisis_values,
            'critical_points': critical_points,
            'max_crisis': max_val,
            'information_entropy': entropy,
            'planck_decisions': len(critical_points) * self.planck_analyzer.planck_time
        }
    
    def last_titan_simulation(self, information_bits: List[int], 
                            noise_level: float = 0.1) -> Dict[str, Any]:
        """最後泰坦模擬 (Last Titan Simulation)"""
        # 模擬資訊傳輸和保存
        transmitted_bits = []
        for bit in information_bits:
            if random.random() < noise_level:
                transmitted_bits.append(1 - bit)  # 翻轉位元
            else:
                transmitted_bits.append(bit)
        
        preservation_score = self.bit_analyzer.information_preservation_score(
            information_bits, transmitted_bits)
        
        original_entropy = self.bit_analyzer.shannon_entropy([
            information_bits.count(0) / len(information_bits),
            information_bits.count(1) / len(information_bits)
        ])
        
        transmitted_entropy = self.bit_analyzer.shannon_entropy([
            transmitted_bits.count(0) / len(transmitted_bits),
            transmitted_bits.count(1) / len(transmitted_bits)
        ])
        
        return {
            'original_bits': information_bits,
            'transmitted_bits': transmitted_bits,
            'preservation_score': preservation_score,
            'original_entropy': original_entropy,
            'transmitted_entropy': transmitted_entropy,
            'entropy_change': transmitted_entropy - original_entropy
        }
    
    def unified_experiment(self) -> Dict[str, Any]:
        """統一實驗 (Unified Experiment)"""
        # 1. 量子道德實驗
        initial_moral_state = QuantumMoralState(
            good_amplitude=1/np.sqrt(3) + 0j,
            evil_amplitude=1/np.sqrt(3) + 0j,
            neutral_amplitude=1/np.sqrt(3) + 0j
        )
        
        moral_evolution = self.quantum_moral_evolution(initial_moral_state, 100)
        
        # 2. 子夜危機實驗
        midnight_results = self.simulate_midnight_crisis(10.0)
        
        # 3. 最後泰坦實驗
        test_information = [random.randint(0, 1) for _ in range(100)]
        titan_results = self.last_titan_simulation(test_information, noise_level=0.2)
        
        return {
            'quantum_moral': {
                'initial_state': initial_moral_state,
                'evolution_length': len(moral_evolution),
                'final_probabilities': moral_evolution[-1].probabilities
            },
            'midnight_crisis': midnight_results,
            'last_titan': titan_results,
            'fundamental_constants': {
                'planck_length': self.planck_analyzer.planck_length,
                'planck_time': self.planck_analyzer.planck_time,
                'planck_energy': self.planck_analyzer.planck_energy
            }
        }

def visualize_results(experiment_results: Dict[str, Any]):
    """結果視覺化 (Results Visualization)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 子夜危機圖
    midnight = experiment_results['midnight_crisis']
    axes[0, 0].plot(midnight['time_points'], midnight['crisis_values'])
    axes[0, 0].set_title('Midnight Crisis Function')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Crisis Intensity')
    
    # 2. 道德狀態機率
    moral = experiment_results['quantum_moral']
    probs = moral['final_probabilities']
    states = list(probs.keys())
    values = list(probs.values())
    axes[0, 1].bar(states, values)
    axes[0, 1].set_title('Final Moral State Probabilities')
    axes[0, 1].set_ylabel('Probability')
    
    # 3. 資訊保存
    titan = experiment_results['last_titan']
    preservation_data = [titan['preservation_score'], 1 - titan['preservation_score']]
    axes[1, 0].pie(preservation_data, labels=['Preserved', 'Lost'], autopct='%1.1f%%')
    axes[1, 0].set_title('Information Preservation by Last Titan')
    
    # 4. 基本常數比較
    constants = experiment_results['fundamental_constants']
    const_names = ['Planck Length', 'Planck Time', 'Planck Energy']
    const_values = [constants['planck_length'], constants['planck_time'], constants['planck_energy']]
    const_values_log = [np.log10(abs(v)) for v in const_values]
    
    axes[1, 1].bar(const_names, const_values_log)
    axes[1, 1].set_title('Fundamental Constants (log scale)')
    axes[1, 1].set_ylabel('log10(value)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/tmp/quantum_ethics_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主要實驗函數 (Main Experiment Function)"""
    print("🌌 基本單位的哲學探索：量子道德學實驗")
    print("=" * 50)
    
    # 創建實驗實例
    experiment = FundamentalUnitsExperiment()
    
    # 運行統一實驗
    print("運行統一實驗...")
    results = experiment.unified_experiment()
    
    # 顯示結果
    print("\n📊 實驗結果:")
    print(f"量子道德最終狀態: {results['quantum_moral']['final_probabilities']}")
    print(f"子夜危機最大強度: {results['midnight_crisis']['max_crisis']:.2f}")
    print(f"資訊保存分數: {results['last_titan']['preservation_score']:.2f}")
    print(f"熵變化: {results['last_titan']['entropy_change']:.4f}")
    
    # 基本常數
    print(f"\n🔬 基本常數:")
    print(f"普朗克長度: {results['fundamental_constants']['planck_length']:.2e} m")
    print(f"普朗克時間: {results['fundamental_constants']['planck_time']:.2e} s")
    print(f"普朗克能量: {results['fundamental_constants']['planck_energy']:.2e} J")
    
    # 視覺化結果
    try:
        visualize_results(results)
    except Exception as e:
        print(f"視覺化錯誤: {e}")
    
    return results

if __name__ == "__main__":
    results = main()