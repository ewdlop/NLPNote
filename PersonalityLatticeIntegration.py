"""
人格格論整合模組 (Personality Lattice Integration Module)

This module integrates the personality lattice model with the existing
HumanExpressionEvaluator to provide comprehensive personality-aware
expression evaluation.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# 導入現有的評估系統
try:
    from HumanExpressionEvaluator import (
        HumanExpressionEvaluator, 
        ExpressionContext, 
        EvaluationResult,
        EvaluationDimension
    )
    HUMAN_EVALUATOR_AVAILABLE = True
except ImportError:
    HUMAN_EVALUATOR_AVAILABLE = False
    print("Warning: HumanExpressionEvaluator not available. Some features will be limited.")

# 導入人格格論模型
try:
    from PersonalityLatticeModel import (
        PersonalityLatticeEvaluator,
        PersonalityLattice,
        PersonalityTrait,
        SituationalContext
    )
    PERSONALITY_LATTICE_AVAILABLE = True
except ImportError:
    PERSONALITY_LATTICE_AVAILABLE = False
    print("Warning: PersonalityLatticeModel not available.")

import numpy as np


@dataclass
class IntegratedEvaluationResult:
    """整合評估結果 (Integrated Evaluation Result)"""
    expression: str
    context: Dict[str, Any]
    
    # 傳統評估結果
    formal_semantic_score: float = 0.0
    cognitive_score: float = 0.0
    social_score: float = 0.0
    integrated_score: float = 0.0
    confidence: float = 0.0
    
    # 人格格論評估結果
    personality_profile: Dict[str, float] = None
    dominant_personality_traits: List[str] = None
    combined_personality_type: str = ""
    situational_consistency: float = 0.0
    personality_adaptation: float = 0.0
    
    # 整合分析
    personality_expression_alignment: float = 0.0
    overall_authenticity_score: float = 0.0
    cultural_appropriateness: float = 0.0
    
    # 詳細分析
    linguistic_features: Dict[str, float] = None
    personality_lattice_structure: Dict[str, Any] = None
    recommendations: List[str] = None


class PersonalityAwareExpressionEvaluator:
    """人格感知表達評估器 (Personality-Aware Expression Evaluator)"""
    
    def __init__(self):
        # 初始化子評估器
        self.human_evaluator = None
        self.personality_evaluator = None
        
        if HUMAN_EVALUATOR_AVAILABLE:
            self.human_evaluator = HumanExpressionEvaluator()
        
        if PERSONALITY_LATTICE_AVAILABLE:
            self.personality_evaluator = PersonalityLatticeEvaluator()
    
    def comprehensive_evaluation(self, expression: str, context: Dict[str, Any] = None) -> IntegratedEvaluationResult:
        """綜合評估表達式 (Comprehensive expression evaluation)"""
        
        if context is None:
            context = {}
        
        result = IntegratedEvaluationResult(expression=expression, context=context)
        
        # 1. 傳統人類表達評估
        if self.human_evaluator:
            traditional_result = self._evaluate_traditional_expression(expression, context)
            result.formal_semantic_score = traditional_result.get('formal_semantic', 0.0)
            result.cognitive_score = traditional_result.get('cognitive', 0.0)
            result.social_score = traditional_result.get('social', 0.0)
            result.integrated_score = traditional_result.get('integrated', 0.0)
            result.confidence = traditional_result.get('confidence', 0.0)
        
        # 2. 人格格論評估
        if self.personality_evaluator:
            personality_result = self._evaluate_personality_expression(expression, context)
            result.personality_profile = personality_result.get('personality_profile', {})
            result.dominant_personality_traits = personality_result.get('dominant_traits', [])
            result.combined_personality_type = personality_result.get('combined_personality', '')
            result.situational_consistency = personality_result.get('situational_consistency', 0.0)
            result.personality_adaptation = personality_result.get('context_adaptation', 0.0)
            result.linguistic_features = personality_result.get('linguistic_features', {})
        
        # 3. 整合分析
        result.personality_expression_alignment = self._calculate_alignment(result)
        result.overall_authenticity_score = self._calculate_authenticity(result)
        result.cultural_appropriateness = self._calculate_cultural_appropriateness(result, context)
        
        # 4. 生成建議
        result.recommendations = self._generate_recommendations(result)
        
        # 5. 格結構信息
        if self.personality_evaluator:
            result.personality_lattice_structure = self.personality_evaluator.lattice.generate_hasse_diagram_data()
        
        return result
    
    def _evaluate_traditional_expression(self, expression: str, context: Dict[str, Any]) -> Dict[str, float]:
        """評估傳統表達式指標 (Evaluate traditional expression metrics)"""
        
        if not self.human_evaluator:
            return {
                'formal_semantic': 0.5,
                'cognitive': 0.5,
                'social': 0.5,
                'integrated': 0.5,
                'confidence': 0.3
            }
        
        try:
            # 轉換上下文格式
            eval_context = ExpressionContext(
                formality_level=context.get('formality_level', 'neutral'),
                situation=context.get('situation', 'general'),
                cultural_background=context.get('cultural_background', 'universal')
            )
            
            # 執行評估
            evaluation_result = self.human_evaluator.comprehensive_evaluation(expression, eval_context)
            
            return {
                'formal_semantic': evaluation_result.get('formal_semantic', {}).get('overall_score', 0.5),
                'cognitive': evaluation_result.get('cognitive', {}).get('overall_score', 0.5),
                'social': evaluation_result.get('social', {}).get('overall_score', 0.5),
                'integrated': evaluation_result.get('integrated', {}).get('overall_score', 0.5),
                'confidence': evaluation_result.get('integrated', {}).get('confidence', 0.3)
            }
        except Exception as e:
            print(f"Warning: Traditional evaluation failed: {e}")
            return {
                'formal_semantic': 0.5,
                'cognitive': 0.5,
                'social': 0.5,
                'integrated': 0.5,
                'confidence': 0.3
            }
    
    def _evaluate_personality_expression(self, expression: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """評估人格表達 (Evaluate personality expression)"""
        
        if not self.personality_evaluator:
            return {}
        
        try:
            # 轉換上下文格式
            situation_context = SituationalContext(
                situation_type=context.get('situation', 'general'),
                formality_level=float(context.get('formality_level', 0.5)),
                stress_level=float(context.get('stress_level', 0.0)),
                cultural_context=context.get('cultural_background', 'universal')
            )
            
            # 執行人格評估
            personality_result = self.personality_evaluator.evaluate_expression_personality(
                expression, situation_context
            )
            
            return personality_result
        except Exception as e:
            print(f"Warning: Personality evaluation failed: {e}")
            return {}
    
    def _calculate_alignment(self, result: IntegratedEvaluationResult) -> float:
        """計算人格與表達的對齊度 (Calculate personality-expression alignment)"""
        
        # 如果缺少必要數據，返回中性分數
        if not result.personality_profile or not result.dominant_personality_traits:
            return 0.5
        
        # 基於主導特質的表達一致性
        personality_strength = np.mean([
            result.personality_profile.get(trait, 0.0) 
            for trait in result.dominant_personality_traits
        ]) if result.dominant_personality_traits else 0.5
        
        # 結合社會評估分數
        social_alignment = result.social_score if result.social_score > 0 else 0.5
        
        # 計算加權對齊度
        alignment = 0.6 * personality_strength + 0.4 * social_alignment
        
        return min(max(alignment, 0.0), 1.0)
    
    def _calculate_authenticity(self, result: IntegratedEvaluationResult) -> float:
        """計算表達真實性 (Calculate expression authenticity)"""
        
        authenticity_factors = []
        
        # 1. 人格一致性
        if result.situational_consistency > 0:
            authenticity_factors.append(result.situational_consistency)
        
        # 2. 認知評估分數
        if result.cognitive_score > 0:
            authenticity_factors.append(result.cognitive_score)
        
        # 3. 人格適應度
        if result.personality_adaptation > 0:
            authenticity_factors.append(result.personality_adaptation)
        
        # 4. 信心度
        if result.confidence > 0:
            authenticity_factors.append(result.confidence)
        
        if not authenticity_factors:
            return 0.5
        
        return np.mean(authenticity_factors)
    
    def _calculate_cultural_appropriateness(self, result: IntegratedEvaluationResult, context: Dict[str, Any]) -> float:
        """計算文化適當性 (Calculate cultural appropriateness)"""
        
        cultural_background = context.get('cultural_background', 'universal')
        
        # 基礎分數
        base_score = 0.7
        
        # 根據正式程度調整
        formality = context.get('formality_level', 0.5)
        if isinstance(formality, str):
            formality_map = {'informal': 0.2, 'neutral': 0.5, 'formal': 0.8}
            formality = formality_map.get(formality, 0.5)
        
        # 中國文化背景的調整
        if cultural_background in ['chinese', 'taiwan', 'hong_kong']:
            # 重視集體主義特質
            if result.personality_profile:
                cooperation_score = result.personality_profile.get('cooperation', 0.0)
                empathy_score = result.personality_profile.get('empathy', 0.0)
                collective_orientation = (cooperation_score + empathy_score) / 2
                base_score += 0.2 * collective_orientation
            
            # 正式場合更重視禮貌
            if formality > 0.6:
                base_score += 0.1
        
        # 西方文化背景的調整
        elif cultural_background in ['western', 'american', 'european']:
            # 重視個人主義特質
            if result.personality_profile:
                confidence_score = result.personality_profile.get('confidence', 0.0)
                expressiveness_score = result.personality_profile.get('expressiveness', 0.0)
                individual_orientation = (confidence_score + expressiveness_score) / 2
                base_score += 0.2 * individual_orientation
        
        return min(max(base_score, 0.0), 1.0)
    
    def _generate_recommendations(self, result: IntegratedEvaluationResult) -> List[str]:
        """生成改進建議 (Generate improvement recommendations)"""
        
        recommendations = []
        
        # 基於整合分數的建議
        if result.integrated_score < 0.4:
            recommendations.append("考慮使用更清晰、更直接的表達方式")
        
        # 基於人格一致性的建議
        if result.situational_consistency < 0.4:
            recommendations.append("調整表達風格以更好地適應當前情境")
        
        # 基於文化適當性的建議
        if result.cultural_appropriateness < 0.5:
            recommendations.append("考慮文化背景，調整表達的正式程度和內容")
        
        # 基於主導特質的建議
        if result.dominant_personality_traits:
            if 'sociability' in result.dominant_personality_traits:
                if result.social_score < 0.5:
                    recommendations.append("可以增加更多互動性和社交元素")
            
            if 'systematic_thinking' in result.dominant_personality_traits:
                if result.formal_semantic_score < 0.5:
                    recommendations.append("可以增加邏輯結構和系統性表述")
        
        # 基於認知評估的建議
        if result.cognitive_score < 0.4:
            recommendations.append("考慮簡化語言或增加解釋以提高可理解性")
        
        # 基於真實性分數的建議
        if result.overall_authenticity_score < 0.4:
            recommendations.append("嘗試更自然、更符合個人風格的表達方式")
        
        # 如果沒有特定建議，提供通用建議
        if not recommendations:
            recommendations.append("表達整體良好，繼續保持自然的溝通風格")
        
        return recommendations
    
    def compare_expressions(self, expressions: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """比較多個表達式 (Compare multiple expressions)"""
        
        results = []
        for expr in expressions:
            result = self.comprehensive_evaluation(expr, context)
            results.append(result)
        
        # 計算比較指標
        comparison = {
            'expressions': expressions,
            'individual_results': results,
            'best_overall': None,
            'best_personality_alignment': None,
            'best_cultural_fit': None,
            'recommendations': []
        }
        
        if results:
            # 找到最佳表達
            best_overall_idx = np.argmax([r.overall_authenticity_score for r in results])
            best_personality_idx = np.argmax([r.personality_expression_alignment for r in results])
            best_cultural_idx = np.argmax([r.cultural_appropriateness for r in results])
            
            comparison['best_overall'] = {
                'expression': expressions[best_overall_idx],
                'score': results[best_overall_idx].overall_authenticity_score,
                'index': best_overall_idx
            }
            
            comparison['best_personality_alignment'] = {
                'expression': expressions[best_personality_idx],
                'score': results[best_personality_idx].personality_expression_alignment,
                'index': best_personality_idx
            }
            
            comparison['best_cultural_fit'] = {
                'expression': expressions[best_cultural_idx],
                'score': results[best_cultural_idx].cultural_appropriateness,
                'index': best_cultural_idx
            }
            
            # 生成比較建議
            if best_overall_idx == best_personality_idx == best_cultural_idx:
                comparison['recommendations'].append(f"表達式 {best_overall_idx + 1} 在所有方面都表現最佳")
            else:
                comparison['recommendations'].append("不同表達式在不同方面有優勢，建議根據具體需求選擇")
        
        return comparison
    
    def analyze_personality_development(self, expressions_timeline: List[Tuple[str, str]]) -> Dict[str, Any]:
        """分析人格發展軌跡 (Analyze personality development trajectory)"""
        
        timeline_analysis = {
            'timeline': expressions_timeline,
            'personality_evolution': [],
            'stability_metrics': {},
            'development_trends': [],
            'insights': []
        }
        
        if not expressions_timeline:
            return timeline_analysis
        
        # 評估每個時間點的表達
        for timestamp, expression in expressions_timeline:
            result = self.comprehensive_evaluation(expression)
            timeline_analysis['personality_evolution'].append({
                'timestamp': timestamp,
                'expression': expression,
                'personality_profile': result.personality_profile,
                'dominant_traits': result.dominant_personality_traits,
                'overall_score': result.overall_authenticity_score
            })
        
        # 計算穩定性指標
        if len(timeline_analysis['personality_evolution']) >= 2:
            # 計算特質變化的標準差
            all_profiles = [item['personality_profile'] for item in timeline_analysis['personality_evolution']]
            
            if all_profiles and all(profile for profile in all_profiles):
                trait_names = set()
                for profile in all_profiles:
                    trait_names.update(profile.keys())
                
                for trait in trait_names:
                    values = [profile.get(trait, 0.0) for profile in all_profiles]
                    timeline_analysis['stability_metrics'][trait] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable'
                    }
        
        # 生成發展趨勢洞察
        if timeline_analysis['stability_metrics']:
            stable_traits = [trait for trait, metrics in timeline_analysis['stability_metrics'].items() 
                           if metrics['std'] < 0.1]
            changing_traits = [trait for trait, metrics in timeline_analysis['stability_metrics'].items() 
                             if metrics['std'] >= 0.2]
            
            if stable_traits:
                timeline_analysis['insights'].append(f"穩定特質: {', '.join(stable_traits[:3])}")
            if changing_traits:
                timeline_analysis['insights'].append(f"變化特質: {', '.join(changing_traits[:3])}")
        
        return timeline_analysis


def demonstrate_integrated_evaluation():
    """演示整合評估功能 (Demonstrate integrated evaluation functionality)"""
    
    print("=== 人格感知表達評估整合演示 (Personality-Aware Expression Evaluation Demo) ===\n")
    
    evaluator = PersonalityAwareExpressionEvaluator()
    
    # 測試表達式
    test_expressions = [
        "我們需要合作完成這個重要的專案，請大家積極參與。",
        "哇！這個創意真的超棒的，我覺得我們可以嘗試看看！",
        "根據數據分析結果，我建議我們採用更系統化的方法來解決這個問題。",
        "Thank you for your consideration. I believe this approach will yield optimal results.",
    ]
    
    # 測試不同情境
    contexts = [
        {"situation": "professional", "formality_level": 0.8, "cultural_background": "chinese"},
        {"situation": "social", "formality_level": 0.3, "cultural_background": "chinese"},
        {"situation": "academic", "formality_level": 0.7, "cultural_background": "universal"},
        {"situation": "professional", "formality_level": 0.9, "cultural_background": "western"},
    ]
    
    print("1. 單一表達式詳細評估 (Detailed Single Expression Evaluation):")
    for i, (expr, ctx) in enumerate(zip(test_expressions, contexts)):
        print(f"\n   表達式 {i+1}: \"{expr}\"")
        print(f"   情境: {ctx}")
        
        result = evaluator.comprehensive_evaluation(expr, ctx)
        
        print(f"   傳統評估 - 整合分數: {result.integrated_score:.3f}, 信心度: {result.confidence:.3f}")
        print(f"   人格評估 - 主導特質: {result.dominant_personality_traits}")
        print(f"   人格類型: {result.combined_personality_type}")
        print(f"   情境一致性: {result.situational_consistency:.3f}")
        print(f"   人格對齊度: {result.personality_expression_alignment:.3f}")
        print(f"   真實性分數: {result.overall_authenticity_score:.3f}")
        print(f"   文化適當性: {result.cultural_appropriateness:.3f}")
        print(f"   建議: {'; '.join(result.recommendations[:2])}")
    
    print(f"\n2. 表達式比較分析 (Expression Comparison Analysis):")
    comparison_exprs = [
        "請協助我們完成這項任務。",
        "希望大家能夠一起努力完成工作。",
        "我們必須立即開始執行這個計劃。"
    ]
    
    comparison = evaluator.compare_expressions(
        comparison_exprs, 
        {"situation": "professional", "formality_level": 0.7}
    )
    
    print(f"   比較的表達式:")
    for i, expr in enumerate(comparison_exprs):
        print(f"     {i+1}. \"{expr}\"")
    
    print(f"\n   最佳整體表現: 表達式 {comparison['best_overall']['index'] + 1} (分數: {comparison['best_overall']['score']:.3f})")
    print(f"   最佳人格對齊: 表達式 {comparison['best_personality_alignment']['index'] + 1} (分數: {comparison['best_personality_alignment']['score']:.3f})")
    print(f"   最佳文化適應: 表達式 {comparison['best_cultural_fit']['index'] + 1} (分數: {comparison['best_cultural_fit']['score']:.3f})")
    print(f"   建議: {'; '.join(comparison['recommendations'])}")
    
    print(f"\n3. 人格發展軌跡分析 (Personality Development Trajectory Analysis):")
    timeline = [
        ("2024-01-01", "我覺得這個想法不錯，但需要更多討論。"),
        ("2024-06-01", "基於我們的分析，我建議採用這個系統化的解決方案。"),
        ("2024-12-01", "經過深思熟慮，我認為我們應該整合多方意見，制定最佳策略。"),
    ]
    
    development = evaluator.analyze_personality_development(timeline)
    
    print(f"   時間軌跡分析:")
    for item in development['personality_evolution']:
        print(f"     {item['timestamp']}: 主導特質 {item['dominant_traits']}, 整體分數 {item['overall_score']:.3f}")
    
    if development['insights']:
        print(f"   發展洞察: {'; '.join(development['insights'])}")


if __name__ == "__main__":
    demonstrate_integrated_evaluation()