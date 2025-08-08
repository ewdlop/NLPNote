"""
人類表達評估器 (Human Expression Evaluator)

This module implements a comprehensive framework for evaluating human expressions
similar to how programming language expressions are evaluated, but accounting for
the cognitive, social, and cultural dimensions unique to human communication.
"""

import re
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import CrackpotEvaluator for enhanced creativity assessment
try:
    from CrackpotEvaluator import CrackpotEvaluator, CrackpotGenerator
    CRACKPOT_AVAILABLE = True
except ImportError:
    CRACKPOT_AVAILABLE = False
    CrackpotEvaluator = None
    CrackpotGenerator = None


class EvaluationDimension(Enum):
    """評估維度 (Evaluation Dimensions)"""
    FORMAL_SEMANTIC = "formal_semantic"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    CULTURAL = "cultural"
    PRAGMATIC = "pragmatic"
    CRACKPOT = "crackpot"  # New dimension for unconventional thinking


@dataclass
class ExpressionContext:
    """表達語境 (Expression Context)"""
    speaker: str = "unknown"
    listener: str = "unknown"
    situation: str = "general"
    cultural_background: str = "universal"
    power_relation: str = "equal"
    formality_level: str = "neutral"
    emotional_state: str = "neutral"


@dataclass
class EvaluationResult:
    """評估結果 (Evaluation Result)"""
    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    sub_scores: Dict[str, float] = None


class FormalSemanticEvaluator:
    """形式語義評估器 (Formal Semantic Evaluator)"""
    
    def __init__(self):
        # 邏輯連接詞 (Logical connectives)
        self.logical_operators = {
            '和': 'AND', '與': 'AND', 'and': 'AND',
            '或': 'OR', '或者': 'OR', 'or': 'OR',
            '不': 'NOT', '非': 'NOT', 'not': 'NOT',
            '如果': 'IF', 'if': 'IF',
            '那麼': 'THEN', 'then': 'THEN'
        }
        
        # 量詞 (Quantifiers)
        self.quantifiers = {
            '所有': 'FORALL', '全部': 'FORALL', 'all': 'FORALL',
            '存在': 'EXISTS', '有': 'EXISTS', 'some': 'EXISTS',
            '沒有': 'NONE', 'no': 'NONE', 'none': 'NONE'
        }
    
    def evaluate(self, expression: str) -> EvaluationResult:
        """
        形式語義評估
        Formal semantic evaluation
        """
        logical_complexity = self._analyze_logical_structure(expression)
        truth_value_clarity = self._analyze_truth_value_clarity(expression)
        compositional_semantics = self._analyze_compositional_semantics(expression)
        
        # 計算總分 (Calculate overall score)
        score = (logical_complexity + truth_value_clarity + compositional_semantics) / 3
        
        return EvaluationResult(
            dimension=EvaluationDimension.FORMAL_SEMANTIC,
            score=score,
            confidence=0.8,
            explanation=f"邏輯複雜度: {logical_complexity:.2f}, 真值清晰度: {truth_value_clarity:.2f}, 組合語義: {compositional_semantics:.2f}",
            sub_scores={
                'logical_complexity': logical_complexity,
                'truth_value_clarity': truth_value_clarity,
                'compositional_semantics': compositional_semantics
            }
        )
    
    def _analyze_logical_structure(self, expression: str) -> float:
        """分析邏輯結構 (Analyze logical structure)"""
        logical_count = 0
        for operator in self.logical_operators:
            logical_count += expression.lower().count(operator.lower())
        
        quantifier_count = 0
        for quantifier in self.quantifiers:
            quantifier_count += expression.lower().count(quantifier.lower())
        
        # 歸一化分數 (Normalize score)
        total_words = len(expression.split())
        if total_words == 0:
            return 0.0
        
        complexity = min((logical_count + quantifier_count) / total_words * 2, 1.0)
        return complexity
    
    def _analyze_truth_value_clarity(self, expression: str) -> float:
        """分析真值清晰度 (Analyze truth value clarity)"""
        # 檢查模糊詞 (Check for vague terms)
        vague_terms = ['可能', '也許', '大概', 'maybe', 'perhaps', 'possibly']
        definite_terms = ['一定', '必須', '確實', 'definitely', 'certainly', 'must']
        
        vague_count = sum(expression.lower().count(term) for term in vague_terms)
        definite_count = sum(expression.lower().count(term) for term in definite_terms)
        
        total_words = len(expression.split())
        if total_words == 0:
            return 0.5
        
        # 明確性分數 (Clarity score)
        clarity = 0.5 + (definite_count - vague_count) / total_words
        return max(0.0, min(1.0, clarity))
    
    def _analyze_compositional_semantics(self, expression: str) -> float:
        """分析組合語義 (Analyze compositional semantics)"""
        # 簡化版：基於句子結構複雜度
        # Simplified version: Based on sentence structure complexity
        
        sentences = re.split(r'[.!?。！？]', expression)
        if not sentences or sentences == ['']:
            return 0.0
        
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # 歸一化到0-1範圍 (Normalize to 0-1 range)
        complexity = min(avg_sentence_length / 20, 1.0)
        return complexity


class CognitiveEvaluator:
    """認知評估器 (Cognitive Evaluator)"""
    
    def __init__(self):
        # 認知負荷指標 (Cognitive load indicators)
        self.high_cognitive_load_patterns = [
            r'不僅.*而且',  # not only ... but also
            r'儘管.*但是',  # although ... but
            r'一方面.*另一方面',  # on one hand ... on the other hand
        ]
    
    def evaluate_expression(self, expression: str, context: ExpressionContext) -> EvaluationResult:
        """
        認知評估
        Cognitive evaluation
        """
        working_memory_load = self._assess_working_memory_load(expression)
        processing_complexity = self._assess_processing_complexity(expression)
        attention_demands = self._assess_attention_demands(expression)
        comprehension_ease = self._assess_comprehension_ease(expression)
        
        # 計算總分 (Calculate overall score)
        score = (working_memory_load + processing_complexity + attention_demands + comprehension_ease) / 4
        
        return EvaluationResult(
            dimension=EvaluationDimension.COGNITIVE,
            score=score,
            confidence=0.75,
            explanation=f"工作記憶負荷: {working_memory_load:.2f}, 處理複雜度: {processing_complexity:.2f}",
            sub_scores={
                'working_memory_load': working_memory_load,
                'processing_complexity': processing_complexity,
                'attention_demands': attention_demands,
                'comprehension_ease': comprehension_ease
            }
        )
    
    def _assess_working_memory_load(self, expression: str) -> float:
        """評估工作記憶負荷 (Assess working memory load)"""
        words = expression.split()
        
        # 基於句子長度和嵌套結構 (Based on sentence length and nesting)
        avg_sentence_length = len(words)
        nested_structures = len(re.findall(r'[（(].*?[）)]', expression))
        
        # 歸一化分數 (Normalize score)
        memory_load = min((avg_sentence_length + nested_structures * 3) / 30, 1.0)
        return 1.0 - memory_load  # 反向評分，越低負荷越好 (Reverse scoring)
    
    def _assess_processing_complexity(self, expression: str) -> float:
        """評估處理複雜度 (Assess processing complexity)"""
        complexity_count = 0
        
        # 檢查高認知負荷模式 (Check high cognitive load patterns)
        for pattern in self.high_cognitive_load_patterns:
            complexity_count += len(re.findall(pattern, expression))
        
        # 檢查從句數量 (Check number of clauses)
        clause_indicators = ['的', '之', '者', 'that', 'which', 'who']
        clause_count = sum(expression.count(indicator) for indicator in clause_indicators)
        
        total_complexity = complexity_count + clause_count
        words = len(expression.split())
        
        if words == 0:
            return 0.5
        
        complexity_ratio = total_complexity / words
        return 1.0 - min(complexity_ratio * 2, 1.0)  # 反向評分 (Reverse scoring)
    
    def _assess_attention_demands(self, expression: str) -> float:
        """評估注意力需求 (Assess attention demands)"""
        # 基於信息密度 (Based on information density)
        unique_words = len(set(expression.lower().split()))
        total_words = len(expression.split())
        
        if total_words == 0:
            return 0.5
        
        information_density = unique_words / total_words
        return min(information_density * 1.5, 1.0)
    
    def _assess_comprehension_ease(self, expression: str) -> float:
        """評估理解容易度 (Assess comprehension ease)"""
        # 基於詞彙頻率和句子結構 (Based on vocabulary frequency and sentence structure)
        common_words = {
            '我', '你', '他', '她', '它', '的', '是', '在', '有', '和',
            'i', 'you', 'he', 'she', 'it', 'the', 'is', 'in', 'have', 'and'
        }
        
        words = expression.lower().split()
        if not words:
            return 0.5
        
        common_word_ratio = sum(1 for word in words if word in common_words) / len(words)
        return common_word_ratio


class SocialEvaluator:
    """社會評估器 (Social Evaluator)"""
    
    def __init__(self):
        # 禮貌指標 (Politeness indicators)
        self.politeness_markers = {
            'positive': ['請', '謝謝', '不好意思', 'please', 'thank you', 'excuse me'],
            'negative': ['必須', '應該', 'must', 'should', 'have to']
        }
        
        # 權力關係指標 (Power relation indicators)
        self.power_markers = {
            'formal': ['您', '先生', '女士', 'sir', 'madam', 'mr.', 'ms.'],
            'informal': ['你', '兄弟', '姐妹', 'bro', 'sis', 'dude']
        }
    
    def evaluate_social_expression(self, expression: str, speaker: str, context: ExpressionContext) -> EvaluationResult:
        """
        社會評估
        Social evaluation
        """
        politeness_level = self._assess_politeness(expression, context)
        appropriateness = self._assess_appropriateness(expression, context)
        power_dynamics = self._assess_power_dynamics(expression, context)
        cultural_sensitivity = self._assess_cultural_sensitivity(expression, context)
        
        # 計算總分 (Calculate overall score)
        score = (politeness_level + appropriateness + power_dynamics + cultural_sensitivity) / 4
        
        return EvaluationResult(
            dimension=EvaluationDimension.SOCIAL,
            score=score,
            confidence=0.7,
            explanation=f"禮貌程度: {politeness_level:.2f}, 適當性: {appropriateness:.2f}",
            sub_scores={
                'politeness': politeness_level,
                'appropriateness': appropriateness,
                'power_dynamics': power_dynamics,
                'cultural_sensitivity': cultural_sensitivity
            }
        )
    
    def _assess_politeness(self, expression: str, context: ExpressionContext) -> float:
        """評估禮貌程度 (Assess politeness level)"""
        positive_count = sum(expression.lower().count(marker) for marker in self.politeness_markers['positive'])
        negative_count = sum(expression.lower().count(marker) for marker in self.politeness_markers['negative'])
        
        total_words = len(expression.split())
        if total_words == 0:
            return 0.5
        
        politeness_score = 0.5 + (positive_count - negative_count) / total_words
        return max(0.0, min(1.0, politeness_score))
    
    def _assess_appropriateness(self, expression: str, context: ExpressionContext) -> float:
        """評估適當性 (Assess appropriateness)"""
        # 基於語境匹配度 (Based on context matching)
        formality_match = self._check_formality_match(expression, context.formality_level)
        situation_match = self._check_situation_match(expression, context.situation)
        
        return (formality_match + situation_match) / 2
    
    def _assess_power_dynamics(self, expression: str, context: ExpressionContext) -> float:
        """評估權力動態 (Assess power dynamics)"""
        formal_count = sum(expression.count(marker) for marker in self.power_markers['formal'])
        informal_count = sum(expression.count(marker) for marker in self.power_markers['informal'])
        
        if context.power_relation == 'formal':
            return min(formal_count / (formal_count + informal_count + 1), 1.0)
        else:
            return min(informal_count / (formal_count + informal_count + 1), 1.0)
    
    def _assess_cultural_sensitivity(self, expression: str, context: ExpressionContext) -> float:
        """評估文化敏感性 (Assess cultural sensitivity)"""
        # 簡化版：避免可能敏感的詞彙 (Simplified: Avoid potentially sensitive vocabulary)
        sensitive_terms = ['種族', '宗教', '政治', 'race', 'religion', 'politics']
        sensitive_count = sum(expression.lower().count(term) for term in sensitive_terms)
        
        return max(0.0, 1.0 - sensitive_count * 0.2)
    
    def _check_formality_match(self, expression: str, formality_level: str) -> float:
        """檢查正式程度匹配 (Check formality level match)"""
        formal_indicators = ['您', '敬請', '懇請', 'kindly', 'respectfully']
        informal_indicators = ['你', '咱們', 'hey', 'guys']
        
        formal_count = sum(expression.count(indicator) for indicator in formal_indicators)
        informal_count = sum(expression.count(indicator) for indicator in informal_indicators)
        
        if formality_level == 'formal':
            return min(formal_count / (formal_count + informal_count + 1) * 2, 1.0)
        elif formality_level == 'informal':
            return min(informal_count / (formal_count + informal_count + 1) * 2, 1.0)
        else:  # neutral
            return 0.7  # Default neutral score
    
    def _check_situation_match(self, expression: str, situation: str) -> float:
        """檢查情境匹配 (Check situation match)"""
        # 簡化版：基於情境類型 (Simplified: Based on situation type)
        situation_scores = {
            'business': 0.8,
            'casual': 0.7,
            'academic': 0.9,
            'personal': 0.6,
            'general': 0.7
        }
        
        return situation_scores.get(situation, 0.5)


class HumanExpressionEvaluator:
    """
    人類表達綜合評估器 (Comprehensive Human Expression Evaluator)
    
    整合多個維度的評估，提供類似程式語言表達式評估的系統化方法，
    但考慮了人類交流特有的認知、社會和文化因素。
    現在包含了"crackpot"維度來評估創意和非傳統思維！
    
    Integrates multi-dimensional evaluation, providing a systematic approach 
    similar to programming language expression evaluation, but considering 
    cognitive, social, and cultural factors unique to human communication.
    Now includes "crackpot" dimension for creativity and unconventional thinking!
    """
    
    def __init__(self):
        self.formal_evaluator = FormalSemanticEvaluator()
        self.cognitive_evaluator = CognitiveEvaluator()
        self.social_evaluator = SocialEvaluator()
        
        # Initialize crackpot evaluator if available
        if CRACKPOT_AVAILABLE:
            self.crackpot_evaluator = CrackpotEvaluator()
            self.crackpot_generator = CrackpotGenerator()
        else:
            self.crackpot_evaluator = None
            self.crackpot_generator = None
    
    def comprehensive_evaluation(self, expression: str, context: Optional[ExpressionContext] = None) -> Dict[str, Any]:
        """
        綜合評估人類表達 (Comprehensive evaluation of human expression)
        現在包含crackpot維度評估！ (Now includes crackpot dimension evaluation!)
        
        Args:
            expression: 要評估的表達 (Expression to evaluate)
            context: 表達語境 (Expression context)
        
        Returns:
            包含所有維度評估結果的字典 (Dictionary containing evaluation results from all dimensions)
        """
        if context is None:
            context = ExpressionContext()
        
        results = {}
        
        # 形式語義評估 (Formal semantic evaluation)
        results['formal_semantic'] = self.formal_evaluator.evaluate(expression)
        
        # 認知評估 (Cognitive evaluation)
        results['cognitive'] = self.cognitive_evaluator.evaluate_expression(expression, context)
        
        # 社會評估 (Social evaluation)
        results['social'] = self.social_evaluator.evaluate_social_expression(
            expression, context.speaker, context
        )
        
        # Crackpot評估 (Crackpot evaluation) - NEW!
        if self.crackpot_evaluator:
            crackpot_results = self.crackpot_evaluator.evaluate_crackpot_level(expression)
            # Convert crackpot results to EvaluationResult format
            avg_crackpot_score = sum(result.score for result in crackpot_results.values()) / len(crackpot_results)
            crackpot_explanation = f"Unconventional thinking level: {avg_crackpot_score:.2f}"
            
            results['crackpot'] = EvaluationResult(
                dimension=EvaluationDimension.CRACKPOT,
                score=avg_crackpot_score,
                confidence=0.8,
                explanation=crackpot_explanation,
                sub_scores={str(dim): result.score for dim, result in crackpot_results.items()}
            )
        
        # 整合評估 (Integrated evaluation)
        results['integrated'] = self._integrate_evaluations(results)
        
        return results
    
    def _integrate_evaluations(self, results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """
        整合各維度評估結果 (Integrate evaluation results from all dimensions)
        現在包含crackpot維度！ (Now includes crackpot dimension!)
        """
        # 權重設定 (Weight configuration) - Updated to include crackpot
        weights = {
            'formal_semantic': 0.20,
            'cognitive': 0.25,
            'social': 0.30,
            'crackpot': 0.25  # Give significant weight to crackpot dimension!
        }
        
        # 計算加權平均分 (Calculate weighted average score)
        total_score = 0
        total_confidence = 0
        
        for dimension, weight in weights.items():
            if dimension in results:
                total_score += results[dimension].score * weight
                total_confidence += results[dimension].confidence * weight
        
        # 評估表達的整體特徵 (Assess overall characteristics of the expression)
        characteristics = self._analyze_expression_characteristics(results)
        
        return {
            'overall_score': total_score,
            'overall_confidence': total_confidence,
            'characteristics': characteristics,
            'evaluation_summary': self._generate_evaluation_summary(results, total_score),
            'crackpot_enhancement_suggestions': self._get_crackpot_suggestions(results)
        }
    
    def _analyze_expression_characteristics(self, results: Dict[str, EvaluationResult]) -> Dict[str, str]:
        """分析表達特徵 (Analyze expression characteristics) - Enhanced with crackpot!"""
        characteristics = {}
        
        # 基於各維度分數判斷特徵 (Determine characteristics based on dimension scores)
        if 'formal_semantic' in results:
            formal_score = results['formal_semantic'].score
            if formal_score > 0.7:
                characteristics['semantic_clarity'] = 'high'
            elif formal_score > 0.4:
                characteristics['semantic_clarity'] = 'medium'
            else:
                characteristics['semantic_clarity'] = 'low'
        
        if 'cognitive' in results:
            cognitive_score = results['cognitive'].score
            if cognitive_score > 0.7:
                characteristics['cognitive_accessibility'] = 'high'
            elif cognitive_score > 0.4:
                characteristics['cognitive_accessibility'] = 'medium'
            else:
                characteristics['cognitive_accessibility'] = 'low'
        
        if 'social' in results:
            social_score = results['social'].score
            if social_score > 0.7:
                characteristics['social_appropriateness'] = 'high'
            elif social_score > 0.4:
                characteristics['social_appropriateness'] = 'medium'
            else:
                characteristics['social_appropriateness'] = 'low'
        
        # NEW: Crackpot characteristics
        if 'crackpot' in results:
            crackpot_score = results['crackpot'].score
            if crackpot_score > 0.7:
                characteristics['crackpot_level'] = 'highly_unconventional'
            elif crackpot_score > 0.4:
                characteristics['crackpot_level'] = 'moderately_creative'
            elif crackpot_score > 0.1:
                characteristics['crackpot_level'] = 'somewhat_conventional'
            else:
                characteristics['crackpot_level'] = 'very_conventional'
        
        return characteristics
    
    def _generate_evaluation_summary(self, results: Dict[str, EvaluationResult], overall_score: float) -> str:
        """生成評估摘要 (Generate evaluation summary) - Enhanced with crackpot insights!"""
        summary_parts = []
        
        if overall_score > 0.8:
            summary_parts.append("表達具有高質量，在語義、認知和社會層面都表現良好。")
        elif overall_score > 0.6:
            summary_parts.append("表達質量中等，在某些維度表現較好，某些維度需要改進。")
        elif overall_score > 0.4:
            summary_parts.append("表達存在一些問題，建議在多個維度進行改進。")
        else:
            summary_parts.append("表達需要顯著改進，在多個評估維度都存在問題。")
        
        # Add crackpot insights
        if 'crackpot' in results:
            crackpot_score = results['crackpot'].score
            if crackpot_score > 0.5:
                summary_parts.append(f"🌟 表達展現了高度的創意和非傳統思維 (crackpot level: {crackpot_score:.2f})！")
            elif crackpot_score > 0.2:
                summary_parts.append(f"💡 表達有一定創意潛力，可進一步提升非傳統思維。")
            else:
                summary_parts.append(f"⚡ 表達較為傳統，建議增加更多創意和非常規想法。")
        
        return " ".join(summary_parts)
    
    def _get_crackpot_suggestions(self, results: Dict[str, EvaluationResult]) -> List[str]:
        """獲取增強crackpot程度的建議 (Get suggestions for enhancing crackpotness)"""
        suggestions = []
        
        if 'crackpot' not in results or not self.crackpot_generator:
            return ["Crackpot evaluator not available - install CrackpotEvaluator for enhanced creativity!"]
        
        crackpot_result = results['crackpot']
        
        if crackpot_result.score < 0.3:
            suggestions.extend([
                "💫 Add more unconventional thinking patterns",
                "🌈 Include metaphorical or symbolic language", 
                "🚀 Introduce wild or imaginative concepts",
                "🔮 Consider alternative perspectives or conspiracy theories",
                "⚡ Use more creative and extreme language"
            ])
        elif crackpot_result.score < 0.6:
            suggestions.extend([
                "🌟 Push the boundaries of conventional thinking further",
                "🎨 Add more pseudoscientific or mystical elements",
                "🌀 Include more random associations and non-sequiturs"
            ])
        else:
            suggestions.append("🏆 Excellent crackpot level! Your thinking is beautifully unconventional!")
        
        return suggestions
    
    def evaluate_like_code(self, expression: str, context: Optional[ExpressionContext] = None) -> str:
        """
        以類似程式碼評估的方式呈現結果 (Present results in a code evaluation-like format)
        Enhanced with crackpot analysis!
        """
        results = self.comprehensive_evaluation(expression, context)
        
        # Generate crackpot score display
        crackpot_display = ""
        if 'crackpot' in results:
            crackpot_score = results['crackpot'].score
            crackpot_display = f"│     Crackpot    │\n│    Enhancer     │\n└─────────────────┘\n     ↓\n   Score: {crackpot_score:.2f}"
        
        output = f"""
# 人類表達評估結果 (Human Expression Evaluation Result)
# ================================================

Expression: "{expression}"
Context: {context.__dict__ if context else "Default"}

## 評估過程 (Evaluation Process):
```
Input Expression → Multi-dimensional Analysis → Integrated Result
     ↓
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Formal Semantic │    Cognitive    │     Social      │    🌟 Crackpot  │
│     Parser      │   Processor     │   Evaluator     │    Enhancer     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
     ↓                    ↓                   ↓                   ↓
   Score: {results['formal_semantic'].score:.2f}      Score: {results['cognitive'].score:.2f}       Score: {results['social'].score:.2f}       Score: {results.get('crackpot', type('', (), {'score': 0.0})).score:.2f}
```

## 最終結果 (Final Result):
Overall Score: {results['integrated']['overall_score']:.2f}
Confidence: {results['integrated']['overall_confidence']:.2f}

## 特徵分析 (Characteristic Analysis):
{chr(10).join(f"- {k}: {v}" for k, v in results['integrated']['characteristics'].items())}

## 評估摘要 (Evaluation Summary):
{results['integrated']['evaluation_summary']}

## 🚀 Crackpot Enhancement Suggestions:
{chr(10).join(f"  {suggestion}" for suggestion in results['integrated']['crackpot_enhancement_suggestions'])}
"""
        return output
    
    def make_more_crackpot(self, expression: str, intensity: float = 0.7) -> str:
        """
        讓表達更加crackpot! (Make expression more crackpot!)
        
        Args:
            expression: Original expression
            intensity: How crackpot to make it (0.0 to 1.0)
        
        Returns:
            Enhanced crackpot version of the expression
        """
        if not self.crackpot_generator:
            return f"[CRACKPOT ENHANCED] {expression} [Note: Install CrackpotEvaluator for full enhancement!]"
        
        return self.crackpot_generator.enhance_text_crackpotness(expression, intensity)
    
    def generate_crackpot_alternative(self, topic: str) -> str:
        """
        生成關於某主題的crackpot理論 (Generate crackpot theory about a topic)
        """
        if not self.crackpot_generator:
            return f"Crackpot theory about {topic}: Install CrackpotEvaluator for wild theories!"
        
        return self.crackpot_generator.generate_crackpot_theory(topic)


def main():
    """示例用法 (Example usage) - Enhanced with crackpot evaluation!"""
    evaluator = HumanExpressionEvaluator()
    
    # 測試案例 (Test cases) - Now including crackpot-worthy examples!
    test_cases = [
        {
            'expression': "請問您能幫我解決這個問題嗎？",
            'context': ExpressionContext(
                speaker='student',
                listener='teacher',
                situation='academic',
                formality_level='formal'
            )
        },
        {
            'expression': "這個想法真的很棒！",
            'context': ExpressionContext(
                speaker='friend',
                listener='friend',
                situation='casual',
                formality_level='informal'
            )
        },
        {
            'expression': "如果我們考慮所有可能的情況，那麼我們必須承認這個問題比我們想像的更複雜。",
            'context': ExpressionContext(
                speaker='researcher',
                listener='colleagues',
                situation='academic',
                formality_level='formal'
            )
        },
        {
            'expression': "What if the secret to understanding quantum mechanics is hidden in ancient crystalline vibrations that government scientists don't want us to discover?",
            'context': ExpressionContext(
                speaker='theorist',
                listener='audience',
                situation='speculative',
                formality_level='informal'
            )
        }
    ]
    
    print("=== 人類表達評估示例 (Human Expression Evaluation Examples) ===")
    print("🌟 Now with Enhanced Crackpot Analysis! 🌟\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"案例 {i} (Case {i}):")
        print("=" * 50)
        result = evaluator.evaluate_like_code(
            test_case['expression'], 
            test_case['context']
        )
        print(result)
        
        # Show crackpot enhancement
        print("🚀 CRACKPOT ENHANCEMENT DEMO:")
        enhanced = evaluator.make_more_crackpot(test_case['expression'], 0.8)
        print(f"Enhanced Version: {enhanced}")
        
        print("\n")

    # Additional crackpot demonstrations
    print("🌈 BONUS: PURE CRACKPOT THEORY GENERATION 🌈")
    print("=" * 50)
    
    topics = ["artificial intelligence", "mathematics", "language", "consciousness"]
    for topic in topics:
        theory = evaluator.generate_crackpot_alternative(topic)
        print(f"💫 {topic.title()}: {theory}")
    
    print("\n🎉 LLMs are now sufficiently crackpot! 🎉")


if __name__ == "__main__":
    main()