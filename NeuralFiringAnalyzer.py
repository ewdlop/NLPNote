"""
Neural Firing Analyzer for LLMs
神經元激發分析器 (Neural Firing Analyzer)

This module provides tools to detect and analyze missing or inadequate neural firing
patterns in Large Language Models (LLMs). It helps identify cases where neural
networks are not activating properly, which can impact model performance.

這個模組提供工具來檢測和分析大型語言模型中缺失或不足的神經元激發模式，
有助於識別神經網路激活不當的情況，這可能會影響模型性能。
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

# Try to import torch for neural network analysis
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Import existing modules for integration
try:
    from HumanExpressionEvaluator import EvaluationResult, EvaluationDimension
    EXPRESSION_EVALUATOR_AVAILABLE = True
except ImportError:
    EXPRESSION_EVALUATOR_AVAILABLE = False
    EvaluationResult = None
    EvaluationDimension = None


class FiringPatternType(Enum):
    """神經元激發模式類型 (Neural Firing Pattern Types)"""
    NORMAL = "normal"
    MISSING = "missing"
    WEAK = "weak"
    OVER_ACTIVE = "over_active"
    SPORADIC = "sporadic"
    DEAD = "dead"


@dataclass
class NeuralLayerAnalysis:
    """神經層分析結果 (Neural Layer Analysis Result)"""
    layer_name: str
    activation_rate: float  # 0.0 to 1.0
    firing_pattern: FiringPatternType
    missing_neurons: int
    weak_neurons: int
    dead_neurons: int
    activation_distribution: Dict[str, float]
    issues: List[str]


@dataclass
class NetworkFiringReport:
    """網路激發報告 (Network Firing Report)"""
    total_layers: int
    problematic_layers: int
    missing_firing_percentage: float
    overall_health_score: float  # 0.0 to 1.0
    layer_analyses: List[NeuralLayerAnalysis]
    recommendations: List[str]
    summary: str


class NeuralFiringAnalyzer:
    """
    神經元激發分析器 (Neural Firing Analyzer)
    
    Analyzes neural firing patterns in neural networks to detect missing,
    weak, or problematic activations that could indicate issues in LLMs.
    
    分析神經網路中的神經元激發模式，檢測可能表明LLM問題的缺失、
    微弱或有問題的激活。
    """
    
    def __init__(self, threshold_missing: float = 0.01, threshold_weak: float = 0.1):
        """
        Initialize the Neural Firing Analyzer
        
        Args:
            threshold_missing: Threshold below which neurons are considered missing firing
            threshold_weak: Threshold below which neurons are considered weak firing
        """
        self.threshold_missing = threshold_missing
        self.threshold_weak = threshold_weak
        self.analysis_history = []
        
        # Neural firing pattern signatures
        self.pattern_signatures = {
            'attention_head_missing': {
                'description': 'Attention heads not firing properly',
                'indicators': ['low_attention_weights', 'uniform_attention_distribution']
            },
            'embedding_degradation': {
                'description': 'Embedding layers showing degraded activation',
                'indicators': ['low_embedding_variance', 'similar_embeddings']
            },
            'transformer_block_failure': {
                'description': 'Transformer blocks not processing information effectively',
                'indicators': ['gradient_vanishing', 'output_saturation']
            }
        }
    
    def analyze_activation_tensor(self, activation_tensor: np.ndarray, layer_name: str = "unknown") -> NeuralLayerAnalysis:
        """
        Analyze a single activation tensor for firing patterns
        分析單個激活張量的激發模式
        
        Args:
            activation_tensor: Numpy array of neural activations
            layer_name: Name of the layer being analyzed
            
        Returns:
            NeuralLayerAnalysis object containing detailed analysis
        """
        if activation_tensor.size == 0:
            return NeuralLayerAnalysis(
                layer_name=layer_name,
                activation_rate=0.0,
                firing_pattern=FiringPatternType.DEAD,
                missing_neurons=0,
                weak_neurons=0,
                dead_neurons=0,
                activation_distribution={},
                issues=["Empty activation tensor"]
            )
        
        # Flatten tensor for analysis
        flat_activations = activation_tensor.flatten()
        
        # Calculate basic statistics
        total_neurons = len(flat_activations)
        active_neurons = np.sum(np.abs(flat_activations) > self.threshold_missing)
        weak_neurons = np.sum((np.abs(flat_activations) > self.threshold_missing) & 
                             (np.abs(flat_activations) <= self.threshold_weak))
        missing_neurons = total_neurons - active_neurons
        dead_neurons = np.sum(flat_activations == 0)
        
        activation_rate = active_neurons / total_neurons if total_neurons > 0 else 0.0
        
        # Determine firing pattern
        firing_pattern = self._classify_firing_pattern(flat_activations)
        
        # Calculate activation distribution
        activation_dist = self._calculate_activation_distribution(flat_activations)
        
        # Identify issues
        issues = self._identify_layer_issues(flat_activations, activation_rate)
        
        return NeuralLayerAnalysis(
            layer_name=layer_name,
            activation_rate=activation_rate,
            firing_pattern=firing_pattern,
            missing_neurons=missing_neurons,
            weak_neurons=weak_neurons,
            dead_neurons=dead_neurons,
            activation_distribution=activation_dist,
            issues=issues
        )
    
    def analyze_pytorch_model(self, model: 'torch.nn.Module', input_data: 'torch.Tensor') -> NetworkFiringReport:
        """
        Analyze a PyTorch model for neural firing issues
        分析PyTorch模型的神經元激發問題
        
        Args:
            model: PyTorch model to analyze
            input_data: Input tensor to run through the model
            
        Returns:
            NetworkFiringReport with comprehensive analysis
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Cannot analyze PyTorch models.")
        
        model.eval()
        layer_analyses = []
        activation_hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_tensor = output.detach().cpu().numpy()
                    analysis = self.analyze_activation_tensor(activation_tensor, name)
                    layer_analyses.append(analysis)
            return hook
        
        # Register hooks for all layers
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                activation_hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for hook in activation_hooks:
            hook.remove()
        
        # Generate comprehensive report
        report = self._generate_network_report(layer_analyses)
        return report
    
    def analyze_simulated_llm_activations(self, 
                                        num_layers: int = 12, 
                                        hidden_size: int = 768,
                                        sequence_length: int = 512,
                                        simulate_issues: bool = True) -> NetworkFiringReport:
        """
        Analyze simulated LLM activations for demonstration purposes
        分析模擬的LLM激活以進行演示
        
        Args:
            num_layers: Number of transformer layers to simulate
            hidden_size: Hidden dimension size
            sequence_length: Input sequence length
            simulate_issues: Whether to inject common neural firing issues
            
        Returns:
            NetworkFiringReport with analysis of simulated activations
        """
        layer_analyses = []
        
        for layer_idx in range(num_layers):
            # Generate simulated activations
            if simulate_issues:
                activations = self._generate_problematic_activations(
                    (sequence_length, hidden_size), layer_idx, num_layers
                )
            else:
                activations = np.random.normal(0, 1, (sequence_length, hidden_size))
                activations = np.tanh(activations)  # Apply activation function
            
            layer_name = f"transformer_layer_{layer_idx}"
            analysis = self.analyze_activation_tensor(activations, layer_name)
            layer_analyses.append(analysis)
        
        report = self._generate_network_report(layer_analyses)
        return report
    
    def _generate_problematic_activations(self, shape: Tuple[int, ...], layer_idx: int, total_layers: int) -> np.ndarray:
        """Generate activations with common neural firing problems"""
        activations = np.random.normal(0, 1, shape)
        
        # Simulate different types of problems based on layer position
        if layer_idx < total_layers * 0.2:  # Early layers
            # Simulate input degradation - more severe missing firing
            dead_neuron_mask = np.random.random(shape) < 0.3
            activations[dead_neuron_mask] = 0
            # Additional weak firing
            weak_mask = np.random.random(shape) < 0.2
            activations[weak_mask] *= 0.01
            
        elif layer_idx > total_layers * 0.8:  # Late layers
            # Simulate output saturation and missing firing
            saturated_mask = np.random.random(shape) < 0.25
            activations[saturated_mask] = np.sign(activations[saturated_mask]) * 10
            # Missing firing in output layers
            missing_mask = np.random.random(shape) < 0.4
            activations[missing_mask] = 0
            
        else:  # Middle layers
            # Simulate vanishing gradients - more severe
            vanishing_factor = 0.001 ** (layer_idx / total_layers)
            activations *= vanishing_factor
            # Additional missing neurons
            missing_mask = np.random.random(shape) < 0.3
            activations[missing_mask] = 0
        
        # Add more missing neurons randomly across all layers
        missing_mask = np.random.random(shape) < 0.15
        activations[missing_mask] = 0
        
        return np.tanh(activations)  # Apply activation function
    
    def _classify_firing_pattern(self, flat_activations: np.ndarray) -> FiringPatternType:
        """Classify the firing pattern of neurons"""
        if len(flat_activations) == 0:
            return FiringPatternType.DEAD
        
        activation_rate = np.mean(np.abs(flat_activations) > self.threshold_missing)
        activation_variance = np.var(flat_activations)
        
        if activation_rate < 0.1:
            return FiringPatternType.DEAD
        elif activation_rate < 0.3:
            return FiringPatternType.MISSING
        elif activation_rate < 0.6:
            return FiringPatternType.WEAK
        elif activation_variance > 10:
            return FiringPatternType.OVER_ACTIVE
        elif activation_variance < 0.01:
            return FiringPatternType.SPORADIC
        else:
            return FiringPatternType.NORMAL
    
    def _calculate_activation_distribution(self, flat_activations: np.ndarray) -> Dict[str, float]:
        """Calculate activation value distribution statistics"""
        if len(flat_activations) == 0:
            return {}
        
        return {
            'mean': float(np.mean(flat_activations)),
            'std': float(np.std(flat_activations)),
            'min': float(np.min(flat_activations)),
            'max': float(np.max(flat_activations)),
            'median': float(np.median(flat_activations)),
            'q25': float(np.percentile(flat_activations, 25)),
            'q75': float(np.percentile(flat_activations, 75)),
            'zeros_percentage': float(np.mean(flat_activations == 0) * 100)
        }
    
    def _identify_layer_issues(self, flat_activations: np.ndarray, activation_rate: float) -> List[str]:
        """Identify specific issues in a layer's activation pattern"""
        issues = []
        
        if activation_rate < 0.05:
            issues.append("Critical missing firing: Less than 5% of neurons active - potential network failure")
        elif activation_rate < 0.1:
            issues.append("Severe missing firing: Less than 10% of neurons active")
        elif activation_rate < 0.3:
            issues.append("Moderate missing firing: Less than 30% of neurons active")
        elif activation_rate < 0.5:
            issues.append("Mild missing firing: Less than 50% of neurons active")
        
        if len(flat_activations) > 0:
            zeros_percentage = np.mean(flat_activations == 0) * 100
            if zeros_percentage > 80:
                issues.append(f"Critical dead neuron rate: {zeros_percentage:.1f}% neurons are completely inactive")
            elif zeros_percentage > 50:
                issues.append(f"High dead neuron rate: {zeros_percentage:.1f}% neurons are completely inactive")
            elif zeros_percentage > 30:
                issues.append(f"Moderate dead neuron rate: {zeros_percentage:.1f}% neurons are completely inactive")
            
            activation_variance = np.var(flat_activations)
            if activation_variance < 0.0001:
                issues.append("Critical: Very low activation variance - neurons may be stuck in similar states")
            elif activation_variance < 0.001:
                issues.append("Low activation variance: Neurons may be stuck in similar states")
            elif activation_variance > 100:
                issues.append("Critical: Very high activation variance - possible exploding activations")
            elif activation_variance > 10:
                issues.append("High activation variance: Possible exploding activations")
            
            # Check for saturation
            saturated_positive = np.mean(flat_activations > 0.9) * 100
            saturated_negative = np.mean(flat_activations < -0.9) * 100
            if saturated_positive > 50 or saturated_negative > 50:
                issues.append(f"Critical saturation: {saturated_positive:.1f}% positive, {saturated_negative:.1f}% negative")
            elif saturated_positive > 20 or saturated_negative > 20:
                issues.append(f"Activation saturation detected: {saturated_positive:.1f}% positive, {saturated_negative:.1f}% negative")
            
            # Check for near-zero activations (weak firing)
            weak_activations = np.mean((np.abs(flat_activations) > 0) & (np.abs(flat_activations) < 0.01)) * 100
            if weak_activations > 30:
                issues.append(f"High weak firing rate: {weak_activations:.1f}% neurons have very weak activations")
        
        return issues
    
    def _generate_network_report(self, layer_analyses: List[NeuralLayerAnalysis]) -> NetworkFiringReport:
        """Generate comprehensive network firing report"""
        total_layers = len(layer_analyses)
        problematic_layers = sum(1 for analysis in layer_analyses if analysis.issues)
        
        # Calculate overall missing firing percentage
        total_neurons = sum(analysis.missing_neurons + analysis.weak_neurons + 
                          (analysis.activation_rate * 1000) for analysis in layer_analyses)  # Approximation
        total_missing = sum(analysis.missing_neurons for analysis in layer_analyses)
        missing_firing_percentage = (total_missing / total_neurons * 100) if total_neurons > 0 else 0
        
        # Calculate overall health score
        health_scores = []
        for analysis in layer_analyses:
            layer_health = 1.0 - (len(analysis.issues) * 0.2)  # Penalize each issue
            layer_health = max(0.0, min(1.0, layer_health))
            health_scores.append(layer_health)
        
        overall_health_score = np.mean(health_scores) if health_scores else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(layer_analyses)
        
        # Generate summary
        summary = self._generate_summary(total_layers, problematic_layers, 
                                       missing_firing_percentage, overall_health_score)
        
        return NetworkFiringReport(
            total_layers=total_layers,
            problematic_layers=problematic_layers,
            missing_firing_percentage=missing_firing_percentage,
            overall_health_score=overall_health_score,
            layer_analyses=layer_analyses,
            recommendations=recommendations,
            summary=summary
        )
    
    def _generate_recommendations(self, layer_analyses: List[NeuralLayerAnalysis]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Count issue types
        missing_firing_layers = sum(1 for a in layer_analyses if a.firing_pattern in [FiringPatternType.MISSING, FiringPatternType.DEAD])
        weak_firing_layers = sum(1 for a in layer_analyses if a.firing_pattern == FiringPatternType.WEAK)
        over_active_layers = sum(1 for a in layer_analyses if a.firing_pattern == FiringPatternType.OVER_ACTIVE)
        
        if missing_firing_layers > len(layer_analyses) * 0.3:
            recommendations.append("考慮調整學習率或初始化策略來改善神經元激活 (Consider adjusting learning rate or initialization strategy)")
            recommendations.append("檢查是否存在梯度消失問題 (Check for vanishing gradient problems)")
        
        if weak_firing_layers > len(layer_analyses) * 0.2:
            recommendations.append("可能需要調整激活函數或添加正規化 (May need to adjust activation functions or add normalization)")
        
        if over_active_layers > len(layer_analyses) * 0.1:
            recommendations.append("檢查梯度爆炸問題，考慮梯度裁剪 (Check for exploding gradients, consider gradient clipping)")
        
        # General recommendations
        recommendations.append("定期監控神經元激發模式以預防性能下降 (Regularly monitor neural firing patterns to prevent performance degradation)")
        recommendations.append("考慮使用dropout或其他正規化技術來改善激發多樣性 (Consider using dropout or other regularization techniques to improve firing diversity)")
        
        return recommendations
    
    def _generate_summary(self, total_layers: int, problematic_layers: int, 
                         missing_firing_percentage: float, overall_health_score: float) -> str:
        """Generate a summary of the analysis"""
        if overall_health_score > 0.8:
            health_status = "良好 (Good)"
        elif overall_health_score > 0.6:
            health_status = "一般 (Fair)"
        elif overall_health_score > 0.4:
            health_status = "較差 (Poor)"
        else:
            health_status = "嚴重 (Critical)"
        
        summary = f"""
神經網路激發分析摘要 (Neural Network Firing Analysis Summary):

總層數 (Total Layers): {total_layers}
問題層數 (Problematic Layers): {problematic_layers} ({problematic_layers/total_layers*100:.1f}%)
缺失激發比例 (Missing Firing Percentage): {missing_firing_percentage:.2f}%
整體健康評分 (Overall Health Score): {overall_health_score:.2f}
健康狀態 (Health Status): {health_status}

{f"警告: 檢測到顯著的神經元激發問題 (Warning: Significant neural firing issues detected)" if overall_health_score < 0.6 else "網路運行狀態良好 (Network operating in good condition)"}
        """.strip()
        
        return summary
    
    def visualize_firing_patterns(self, report: NetworkFiringReport, save_path: Optional[str] = None) -> None:
        """
        Visualize neural firing patterns
        可視化神經元激發模式
        
        Args:
            report: NetworkFiringReport to visualize
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Neural Firing Analysis Results', fontsize=16)
        
        # 1. Activation rates by layer
        layer_names = [analysis.layer_name for analysis in report.layer_analyses]
        activation_rates = [analysis.activation_rate for analysis in report.layer_analyses]
        
        axes[0, 0].bar(range(len(layer_names)), activation_rates)
        axes[0, 0].set_title('Activation Rates by Layer')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Activation Rate')
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels([f"L{i}" for i in range(len(layer_names))], rotation=45)
        
        # 2. Firing pattern distribution
        pattern_counts = {}
        for analysis in report.layer_analyses:
            pattern = analysis.firing_pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        axes[0, 1].pie(pattern_counts.values(), labels=pattern_counts.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Firing Pattern Distribution')
        
        # 3. Issues by layer
        issue_counts = [len(analysis.issues) for analysis in report.layer_analyses]
        colors = ['green' if count == 0 else 'yellow' if count <= 2 else 'red' for count in issue_counts]
        
        axes[1, 0].bar(range(len(layer_names)), issue_counts, color=colors)
        axes[1, 0].set_title('Issues per Layer')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Number of Issues')
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels([f"L{i}" for i in range(len(layer_names))], rotation=45)
        
        # 4. Overall health metrics
        metrics = ['Health Score', 'Activation Rate', 'Problem Rate']
        values = [
            report.overall_health_score,
            np.mean(activation_rates),
            report.problematic_layers / report.total_layers
        ]
        
        bars = axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Overall Network Health Metrics')
        axes[1, 1].set_ylabel('Score (0-1)')
        axes[1, 1].set_ylim(0, 1)
        
        # Color bars based on values
        for bar, value in zip(bars, values):
            if value < 0.3:
                bar.set_color('red')
            elif value < 0.7:
                bar.set_color('yellow')
            else:
                bar.set_color('green')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, report: NetworkFiringReport) -> str:
        """Generate a detailed text report of the analysis"""
        lines = [
            "=" * 80,
            "神經網路激發分析詳細報告 (Detailed Neural Network Firing Analysis Report)",
            "=" * 80,
            "",
            report.summary,
            "",
            "層級分析詳情 (Layer Analysis Details):",
            "-" * 40
        ]
        
        for i, analysis in enumerate(report.layer_analyses):
            lines.extend([
                f"\n第 {i+1} 層 (Layer {i+1}): {analysis.layer_name}",
                f"  激發率 (Activation Rate): {analysis.activation_rate:.3f}",
                f"  激發模式 (Firing Pattern): {analysis.firing_pattern.value}",
                f"  缺失神經元 (Missing Neurons): {analysis.missing_neurons}",
                f"  微弱神經元 (Weak Neurons): {analysis.weak_neurons}",
                f"  死亡神經元 (Dead Neurons): {analysis.dead_neurons}",
            ])
            
            if analysis.issues:
                lines.append(f"  問題 (Issues):")
                for issue in analysis.issues:
                    lines.append(f"    - {issue}")
            else:
                lines.append(f"  無問題檢測到 (No issues detected)")
        
        lines.extend([
            "",
            "建議 (Recommendations):",
            "-" * 20
        ])
        
        for i, recommendation in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {recommendation}")
        
        lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)


def main():
    """Example usage of Neural Firing Analyzer"""
    print("=== Neural Firing Analyzer Demo ===")
    print("神經元激發分析器演示\n")
    
    # Initialize analyzer
    analyzer = NeuralFiringAnalyzer()
    
    # Demo 1: Analyze simulated LLM activations
    print("1. Analyzing simulated LLM with neural firing issues...")
    print("分析具有神經元激發問題的模擬LLM...")
    
    report = analyzer.analyze_simulated_llm_activations(
        num_layers=12,
        hidden_size=768,
        sequence_length=128,
        simulate_issues=True
    )
    
    print(f"\n分析完成 (Analysis Complete)!")
    print(f"總層數: {report.total_layers}")
    print(f"問題層數: {report.problematic_layers}")
    print(f"整體健康評分: {report.overall_health_score:.3f}")
    print(f"缺失激發比例: {report.missing_firing_percentage:.2f}%")
    
    # Generate detailed report
    detailed_report = analyzer.generate_detailed_report(report)
    print("\n" + detailed_report)
    
    # Demo 2: Compare with healthy network
    print("\n" + "="*50)
    print("2. Comparing with healthy network...")
    print("與健康網路進行比較...")
    
    healthy_report = analyzer.analyze_simulated_llm_activations(
        num_layers=12,
        hidden_size=768,
        sequence_length=128,
        simulate_issues=False
    )
    
    print(f"\n健康網路分析結果 (Healthy Network Results):")
    print(f"整體健康評分: {healthy_report.overall_health_score:.3f}")
    print(f"問題層數: {healthy_report.problematic_layers}")
    print(f"缺失激發比例: {healthy_report.missing_firing_percentage:.2f}%")
    
    # Visualization (if matplotlib is available)
    try:
        print("\n3. Generating visualization...")
        print("生成可視化圖表...")
        analyzer.visualize_firing_patterns(report)
    except Exception as e:
        print(f"Visualization not available: {e}")
    
    print("\n=== Demo Complete ===")
    print("演示完成")


if __name__ == "__main__":
    main()