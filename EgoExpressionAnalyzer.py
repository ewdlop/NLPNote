"""
Ego-based Expression Analysis Integration
自我表達分析整合

Integrates the ego-based neural network framework with human expression evaluation,
creating a more philosophically grounded NLP analysis system.

将自我神经网络框架与人类表达评估整合，创建更具哲学基础的NLP分析系统。
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from EgoBasedNeuralNetwork import EgoBasedFramework, EgoMode, EgoBeliefs, EgoPreferences
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EGO_FRAMEWORK_AVAILABLE = True
except ImportError:
    EgoBasedFramework = None
    EgoMode = None
    HumanExpressionEvaluator = None
    ExpressionContext = None
    EGO_FRAMEWORK_AVAILABLE = False


@dataclass
class EgoExpressionResult:
    """自我表达分析结果 (Ego Expression Analysis Result)"""
    expression_text: str
    ego_interpretation: Dict[str, Any]
    traditional_evaluation: Dict[str, Any]
    philosophical_analysis: Dict[str, str]
    self_consistency_score: float
    truth_seeking_score: float
    overall_ego_score: float
    confidence: float


class EgoBasedExpressionAnalyzer:
    """基于自我的表达分析器 (Ego-based Expression Analyzer)
    
    Combines traditional NLP analysis with ego-based philosophical framework
    to provide deeper understanding of human expressions and their underlying
    self-consistency patterns.
    """
    
    def __init__(self, ego_lambda: float = 0.5, belief_stubbornness: float = 0.3):
        if not EGO_FRAMEWORK_AVAILABLE:
            raise ImportError("Ego framework dependencies not available")
        
        # Initialize ego framework
        self.ego_framework = EgoBasedFramework(
            world_state_dim=16,  # Higher dimension for text representation
            action_dim=8,       # Multiple interpretation actions
            ego_lambda=ego_lambda,
            belief_stubbornness=belief_stubbornness,
            mode=EgoMode.BALANCED_EGO
        )
        
        # Initialize traditional expression evaluator
        self.traditional_evaluator = HumanExpressionEvaluator()
        
        # Expression memory for ego consistency tracking
        self.expression_history = []
        self.speaker_ego_profiles = {}
        
    def encode_expression_to_world_state(self, expression: str, context: ExpressionContext) -> np.ndarray:
        """将表达编码为世界状态向量 (Encode expression to world state vector)"""
        # Simple feature extraction (can be enhanced with embeddings)
        features = []
        
        # Basic linguistic features
        features.append(len(expression) / 100.0)  # Length normalized
        features.append(expression.count('?') / (len(expression) + 1))  # Question density
        features.append(expression.count('!') / (len(expression) + 1))  # Exclamation density
        features.append(len(expression.split()) / 50.0)  # Word count normalized
        
        # Formality indicators
        formal_words = ['please', 'thank', 'kindly', 'respectfully', '請', '謝謝', '敬請']
        formality_score = sum(1 for word in formal_words if word.lower() in expression.lower())
        features.append(formality_score / 10.0)
        
        # Emotional indicators
        positive_words = ['good', 'great', 'excellent', 'wonderful', '好', '棒', '很好']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', '壞', '糟', '不好']
        emotion_score = (sum(1 for word in positive_words if word.lower() in expression.lower()) - 
                        sum(1 for word in negative_words if word.lower() in expression.lower()))
        features.append((emotion_score + 5) / 10.0)  # Normalized to [0,1]
        
        # Context features
        formality_map = {'formal': 1.0, 'neutral': 0.5, 'informal': 0.0}
        features.append(formality_map.get(context.formality_level, 0.5))
        
        power_map = {'superior': 1.0, 'equal': 0.5, 'subordinate': 0.0}
        features.append(power_map.get(context.power_relation, 0.5))
        
        # Cultural indicators
        cultural_map = {'chinese': 0.2, 'western': 0.8, 'universal': 0.5}
        features.append(cultural_map.get(context.cultural_background, 0.5))
        
        # Pad or truncate to exactly 16 features
        while len(features) < 16:
            features.append(0.0)
        features = features[:16]
        
        return np.array(features, dtype=np.float32)
    
    def analyze_speaker_ego_consistency(self, speaker: str, expression: str) -> Dict[str, float]:
        """分析说话者的自我一致性 (Analyze speaker's ego consistency)"""
        if speaker not in self.speaker_ego_profiles:
            self.speaker_ego_profiles[speaker] = {
                'expressions': [],
                'ego_patterns': [],
                'consistency_score': 0.5
            }
        
        profile = self.speaker_ego_profiles[speaker]
        profile['expressions'].append(expression)
        
        # Calculate consistency with previous expressions
        if len(profile['expressions']) > 1:
            # Simple consistency measure based on expression similarity
            recent_expressions = profile['expressions'][-5:]  # Last 5 expressions
            
            # Calculate feature similarity
            current_features = self.encode_expression_to_world_state(
                expression, 
                ExpressionContext(speaker=speaker)
            )
            
            similarities = []
            for prev_expr in recent_expressions[:-1]:
                prev_features = self.encode_expression_to_world_state(
                    prev_expr,
                    ExpressionContext(speaker=speaker)
                )
                # Cosine similarity
                similarity = np.dot(current_features, prev_features) / (
                    np.linalg.norm(current_features) * np.linalg.norm(prev_features) + 1e-8
                )
                similarities.append(similarity)
            
            consistency = np.mean(similarities) if similarities else 0.5
            profile['consistency_score'] = consistency
        
        return {
            'speaker_consistency': profile['consistency_score'],
            'expression_count': len(profile['expressions']),
            'avg_formality': 0.5,  # Could be computed from features
            'ego_stability': min(profile['consistency_score'] + 0.1, 1.0)
        }
    
    def compute_philosophical_tensions(self, 
                                     traditional_scores: Dict[str, float],
                                     ego_state: Dict[str, Any]) -> Dict[str, str]:
        """计算哲学张力 (Compute philosophical tensions)"""
        tensions = {}
        
        # Truth vs Self-consistency tension
        truth_score = traditional_scores.get('formal_semantic', 0.5)
        ego_score = ego_state.get('consistency_score', 0.5)
        
        if abs(truth_score - ego_score) > 0.3:
            if truth_score > ego_score:
                tensions['truth_ego_tension'] = "表达追求客观真理但可能缺乏自我一致性"
            else:
                tensions['truth_ego_tension'] = "表达保持自我一致但可能偏离客观真理"
        else:
            tensions['truth_ego_tension'] = "真理追求与自我一致性之间保持平衡"
        
        # Cognitive dissonance indicators
        cognitive_score = traditional_scores.get('cognitive', 0.5)
        if cognitive_score < 0.4 and ego_score > 0.7:
            tensions['cognitive_dissonance'] = "可能存在认知失调：自我保护与认知处理冲突"
        elif cognitive_score > 0.7 and ego_score < 0.4:
            tensions['cognitive_dissonance'] = "认知清晰但自我一致性较低"
        else:
            tensions['cognitive_dissonance'] = "认知处理与自我状态协调"
        
        # Social vs Individual tension
        social_score = traditional_scores.get('social', 0.5)
        if social_score > 0.8 and ego_score < 0.4:
            tensions['social_individual'] = "高度社会适应但可能牺牲个人自我"
        elif social_score < 0.4 and ego_score > 0.8:
            tensions['social_individual'] = "强烈自我表达但社会适应性较低"
        else:
            tensions['social_individual'] = "社会适应与个人表达平衡"
        
        return tensions
    
    def comprehensive_ego_analysis(self, 
                                  expression: str, 
                                  context: ExpressionContext) -> EgoExpressionResult:
        """综合自我分析 (Comprehensive ego analysis)"""
        
        # Traditional evaluation
        traditional_result = self.traditional_evaluator.comprehensive_evaluation(expression, context)
        traditional_scores = {
            'formal_semantic': traditional_result['formal_semantic'].score,
            'cognitive': traditional_result['cognitive'].score,
            'social': traditional_result['social'].score,
            'integrated': traditional_result['integrated']['overall_score']
        }
        
        # Encode expression for ego framework
        world_state_vector = self.encode_expression_to_world_state(expression, context)
        world_state = self.ego_framework.perceive_world(world_state_vector)
        
        # Ego-based decision making
        ego_action = self.ego_framework.decide_action(world_state)
        
        # Update beliefs about this type of expression
        likelihood = traditional_scores['integrated']  # Use traditional score as likelihood
        self.ego_framework.update_beliefs(world_state_vector, likelihood)
        
        # Analyze speaker ego consistency
        speaker_ego = self.analyze_speaker_ego_consistency(context.speaker, expression)
        
        # Get philosophical interpretation
        philosophical_interp = self.ego_framework.get_philosophical_interpretation()
        
        # Compute philosophical tensions
        tensions = self.compute_philosophical_tensions(traditional_scores, speaker_ego)
        
        # Calculate ego-specific scores
        self_consistency_score = speaker_ego['speaker_consistency']
        truth_seeking_score = 1.0 - self.ego_framework.ego_lambda  # Inverse of ego lambda
        
        # Overall ego score combines multiple factors
        overall_ego_score = (
            0.4 * self_consistency_score +
            0.3 * truth_seeking_score +
            0.3 * traditional_scores['integrated']
        )
        
        # Confidence based on ego framework confidence and traditional confidence
        ego_confidence = max(0.1, min(0.9, speaker_ego['ego_stability']))
        traditional_confidence = traditional_result['integrated'].get('overall_confidence', 0.5)
        combined_confidence = (ego_confidence + traditional_confidence) / 2
        
        return EgoExpressionResult(
            expression_text=expression,
            ego_interpretation={
                'philosophical_mode': philosophical_interp['mode'],
                'lambda_value': philosophical_interp['lambda_value'],
                'stubbornness': philosophical_interp['stubbornness'],
                'ego_action_probs': ego_action.numpy().tolist(),
                'speaker_consistency': speaker_ego
            },
            traditional_evaluation=traditional_scores,
            philosophical_analysis=tensions,
            self_consistency_score=self_consistency_score,
            truth_seeking_score=truth_seeking_score,
            overall_ego_score=overall_ego_score,
            confidence=combined_confidence
        )
    
    def compare_expressions_ego_evolution(self, 
                                        expressions: List[Tuple[str, ExpressionContext]]) -> Dict[str, Any]:
        """比较表达的自我演化 (Compare expressions' ego evolution)"""
        results = []
        ego_evolution = []
        
        for i, (expression, context) in enumerate(expressions):
            result = self.comprehensive_ego_analysis(expression, context)
            results.append(result)
            
            # Track ego evolution
            ego_state = {
                'step': i,
                'ego_score': result.overall_ego_score,
                'consistency': result.self_consistency_score,
                'truth_seeking': result.truth_seeking_score,
                'lambda': float(result.ego_interpretation['lambda_value'])
            }
            ego_evolution.append(ego_state)
        
        # Calculate evolution metrics
        ego_scores = [state['ego_score'] for state in ego_evolution]
        consistency_scores = [state['consistency'] for state in ego_evolution]
        
        evolution_metrics = {
            'ego_score_trend': np.polyfit(range(len(ego_scores)), ego_scores, 1)[0] if len(ego_scores) > 1 else 0,
            'consistency_stability': np.std(consistency_scores) if len(consistency_scores) > 1 else 0,
            'final_ego_state': ego_evolution[-1] if ego_evolution else {},
            'total_expressions': len(expressions)
        }
        
        return {
            'individual_results': results,
            'ego_evolution': ego_evolution,
            'evolution_metrics': evolution_metrics,
            'summary': {
                'avg_ego_score': np.mean(ego_scores) if ego_scores else 0,
                'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0,
                'philosophical_interpretation': results[-1].ego_interpretation['philosophical_mode'] if results else "Unknown"
            }
        }
    
    def adjust_ego_sensitivity(self, new_lambda: float, new_stubbornness: float = None):
        """调整自我敏感度 (Adjust ego sensitivity)"""
        self.ego_framework.ego_lambda = new_lambda
        if new_stubbornness is not None:
            self.ego_framework.ego_beliefs.stubbornness = new_stubbornness
        
        # Determine new mode based on lambda
        if new_lambda < 0.1:
            self.ego_framework.mode = EgoMode.PURE_OBJECTIVIST
        elif new_lambda > 2.0:
            self.ego_framework.mode = EgoMode.PURE_EGOIST
        else:
            self.ego_framework.mode = EgoMode.BALANCED_EGO
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析总结 (Get analysis summary)"""
        return {
            'framework_stats': self.ego_framework.get_training_statistics(),
            'speakers_analyzed': len(self.speaker_ego_profiles),
            'total_expressions': len(self.expression_history),
            'current_ego_mode': self.ego_framework.mode.value,
            'philosophical_state': self.ego_framework.get_philosophical_interpretation()
        }


# Example usage functions
def demonstrate_ego_expression_analysis():
    """演示自我表达分析 (Demonstrate ego expression analysis)"""
    if not EGO_FRAMEWORK_AVAILABLE:
        print("Ego framework not available. Please install dependencies.")
        return
    
    print("=== 自我表达分析演示 (Ego Expression Analysis Demo) ===")
    
    # Create analyzer
    analyzer = EgoBasedExpressionAnalyzer(ego_lambda=0.5, belief_stubbornness=0.3)
    
    # Test expressions
    test_cases = [
        ("I believe this is correct.", ExpressionContext(speaker="Alice", formality_level="formal")),
        ("This is definitely wrong!", ExpressionContext(speaker="Alice", formality_level="informal")),
        ("I'm not sure, but maybe...", ExpressionContext(speaker="Alice", formality_level="neutral")),
        ("請您協助解決此問題", ExpressionContext(speaker="Bob", formality_level="formal", cultural_background="chinese")),
        ("幫我一下好嗎？", ExpressionContext(speaker="Bob", formality_level="informal", cultural_background="chinese"))
    ]
    
    print("\n1. 单独分析各表达:")
    for i, (expression, context) in enumerate(test_cases[:3]):
        result = analyzer.comprehensive_ego_analysis(expression, context)
        print(f"\n表达 {i+1}: \"{expression}\"")
        print(f"  哲学模式: {result.ego_interpretation['philosophical_mode']}")
        print(f"  自我一致性: {result.self_consistency_score:.3f}")
        print(f"  真理追求: {result.truth_seeking_score:.3f}")
        print(f"  整体自我分数: {result.overall_ego_score:.3f}")
        print(f"  主要张力: {list(result.philosophical_analysis.keys())}")
    
    print("\n2. 演化分析:")
    evolution_result = analyzer.compare_expressions_ego_evolution(test_cases)
    print(f"  平均自我分数: {evolution_result['summary']['avg_ego_score']:.3f}")
    print(f"  平均一致性: {evolution_result['summary']['avg_consistency']:.3f}")
    print(f"  哲学解释: {evolution_result['summary']['philosophical_interpretation']}")
    
    print("\n3. 调整自我敏感度测试:")
    # Test different ego sensitivities
    for lambda_val in [0.0, 0.5, 2.0]:
        analyzer.adjust_ego_sensitivity(lambda_val)
        result = analyzer.comprehensive_ego_analysis("I think this might be true.", 
                                                   ExpressionContext(speaker="Test"))
        print(f"  λ={lambda_val}: 自我分数={result.overall_ego_score:.3f}, "
              f"模式={result.ego_interpretation['philosophical_mode']}")
    
    print("\n演示完成!")


if __name__ == "__main__":
    demonstrate_ego_expression_analysis()