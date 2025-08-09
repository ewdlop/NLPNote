#!/usr/bin/env python3
"""
π表達式與人類表達評估整合 (Pi Expressions and Human Expression Evaluation Integration)

This module integrates π (pi) expression evaluation with the existing human 
expression evaluation framework, demonstrating how mathematical expressions 
can be evaluated alongside natural language expressions.

這個模組將π表達式評估與現有的人類表達評估框架整合，
展示了數學表達式如何與自然語言表達式一起被評估。
"""

import math
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Try to import existing modules with graceful fallback
try:
    from pi_evaluator import PiExpressionEvaluator
    PI_EVALUATOR_AVAILABLE = True
except ImportError:
    PI_EVALUATOR_AVAILABLE = False

try:
    from HumanExpressionEvaluator import ExpressionContext, EvaluationResult, EvaluationDimension
    HUMAN_EVALUATOR_AVAILABLE = True
except ImportError:
    HUMAN_EVALUATOR_AVAILABLE = False
    # Provide basic fallback classes
    @dataclass
    class ExpressionContext:
        speaker: str = "unknown"
        listener: str = "unknown"
        situation: str = "mathematical"
        formality_level: str = "formal"
    
    @dataclass
    class EvaluationResult:
        score: float
        confidence: float
        explanation: str


class MathematicalExpressionType:
    """數學表達式類型 (Mathematical Expression Types)"""
    PI_CONSTANT = "pi_constant"
    PI_CALCULATION = "pi_calculation"
    PI_SERIES = "pi_series"
    PI_FORMULA = "pi_formula"
    GEOMETRIC_PI = "geometric_pi"
    NUMERICAL_PI = "numerical_pi"


class PiExpressionIntegratedEvaluator:
    """π表達式整合評估器 (Pi Expression Integrated Evaluator)"""
    
    def __init__(self):
        """初始化整合評估器"""
        self.pi_evaluator = PiExpressionEvaluator() if PI_EVALUATOR_AVAILABLE else None
        self.pi_patterns = self._compile_pi_patterns()
        
    def _compile_pi_patterns(self) -> Dict[str, re.Pattern]:
        """編譯π表達式的正則表達式模式"""
        patterns = {
            'pi_constant': re.compile(r'(π|pi|Pi|PI)'),
            'pi_calculation': re.compile(r'(計算|calculate|compute|evaluate).{0,10}(π|pi|Pi|PI)'),
            'pi_series': re.compile(r'(萊布尼茲|Leibniz|級數|series|尼爾森|Nilakantha).{0,20}(π|pi)'),
            'pi_formula': re.compile(r'(馬欽|Machin|公式|formula|拉馬努金|Ramanujan).{0,20}(π|pi)'),
            'pi_geometric': re.compile(r'(圓|circle|周長|circumference|直徑|diameter).{0,20}(π|pi)'),
            'pi_numerical': re.compile(r'(3\.14|3\.141|小數|decimal|精度|precision).{0,20}(π|pi)')
        }
        return patterns
    
    def detect_pi_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        檢測文本中的π表達式
        
        Args:
            text: 待分析的文本
            
        Returns:
            檢測到的π表達式列表
        """
        detected_expressions = []
        
        for expr_type, pattern in self.pi_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                detected_expressions.append({
                    'type': expr_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': self._calculate_detection_confidence(expr_type, match.group())
                })
        
        return detected_expressions
    
    def _calculate_detection_confidence(self, expr_type: str, text: str) -> float:
        """計算檢測信心度"""
        base_confidence = 0.7
        
        # 基於表達式類型調整信心度
        type_weights = {
            'pi_constant': 0.9,
            'pi_calculation': 0.8,
            'pi_series': 0.85,
            'pi_formula': 0.85,
            'pi_geometric': 0.75,
            'pi_numerical': 0.7
        }
        
        confidence = base_confidence * type_weights.get(expr_type, 0.7)
        
        # 基於文本特徵調整
        if 'π' in text:  # Unicode pi symbol
            confidence += 0.1
        if any(keyword in text.lower() for keyword in ['計算', 'calculate', 'evaluate', '級數', 'series']):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def evaluate_pi_expression_mathematically(self, expression: str, context: Optional[ExpressionContext] = None) -> Dict[str, Any]:
        """
        數學評估π表達式
        
        Args:
            expression: π表達式
            context: 表達語境
            
        Returns:
            數學評估結果
        """
        if not self.pi_evaluator:
            return {
                'error': 'Pi evaluator not available',
                'fallback_evaluation': self._basic_pi_evaluation(expression)
            }
        
        # 分析表達式類型
        expr_info = self._analyze_pi_expression(expression)
        
        results = {}
        
        # 根據表達式類型選擇評估方法
        if expr_info['type'] == 'series_calculation':
            if 'leibniz' in expression.lower() or '萊布尼茲' in expression:
                results['leibniz'] = self.pi_evaluator.leibniz_series(1000)
            if 'nilakantha' in expression.lower() or '尼爾森' in expression:
                results['nilakantha'] = self.pi_evaluator.nilakantha_series(500)
            if 'machin' in expression.lower() or '馬欽' in expression:
                results['machin'] = self.pi_evaluator.machin_formula(50)
        
        elif expr_info['type'] == 'algorithm_calculation':
            if 'agm' in expression.lower() or 'AGM' in expression:
                results['agm'] = self.pi_evaluator.agm_algorithm(5)
            if 'monte' in expression.lower() or '蒙特' in expression:
                results['monte_carlo'] = self.pi_evaluator.monte_carlo_pi(10000)
        
        elif expr_info['type'] == 'general_pi':
            # 提供多種方法的比較
            results = {
                'leibniz_quick': self.pi_evaluator.leibniz_series(100),
                'agm_quick': self.pi_evaluator.agm_algorithm(3),
                'monte_carlo_quick': self.pi_evaluator.monte_carlo_pi(1000)
            }
        
        return {
            'expression_analysis': expr_info,
            'mathematical_results': results,
            'context': context.__dict__ if context else None
        }
    
    def _analyze_pi_expression(self, expression: str) -> Dict[str, Any]:
        """分析π表達式的類型和特徵"""
        analysis = {
            'type': 'general_pi',
            'complexity': 'basic',
            'calculation_intent': False,
            'specific_method': None
        }
        
        # 檢測計算意圖
        calc_keywords = ['計算', 'calculate', 'compute', 'evaluate', '求', '算']
        if any(keyword in expression.lower() for keyword in calc_keywords):
            analysis['calculation_intent'] = True
        
        # 檢測特定方法
        methods = {
            'leibniz': ['萊布尼茲', 'leibniz'],
            'nilakantha': ['尼爾森', 'nilakantha'],
            'machin': ['馬欽', 'machin'],
            'ramanujan': ['拉馬努金', 'ramanujan'],
            'agm': ['agm', 'AGM', '算術幾何'],
            'monte_carlo': ['蒙特卡羅', 'monte carlo', 'random']
        }
        
        for method, keywords in methods.items():
            if any(keyword in expression for keyword in keywords):
                analysis['specific_method'] = method
                analysis['type'] = 'series_calculation' if method in ['leibniz', 'nilakantha', 'machin', 'ramanujan'] else 'algorithm_calculation'
                analysis['complexity'] = 'advanced'
                break
        
        return analysis
    
    def _basic_pi_evaluation(self, expression: str) -> Dict[str, Any]:
        """基本π評估（當完整評估器不可用時）"""
        return {
            'basic_pi_value': math.pi,
            'approximation': round(math.pi, 10),
            'expression_detected': expression,
            'note': 'Basic evaluation using standard math.pi'
        }
    
    def evaluate_as_human_expression(self, expression: str, context: Optional[ExpressionContext] = None) -> Dict[str, Any]:
        """
        將π表達式作為人類表達進行評估
        
        Args:
            expression: 包含π的表達式
            context: 表達語境
            
        Returns:
            人類表達評估結果
        """
        # 檢測π表達式
        pi_expressions = self.detect_pi_expressions(expression)
        
        if not pi_expressions:
            return {'error': 'No π expressions detected in the text'}
        
        # 分析表達的複雜度和形式性
        analysis = self._analyze_expression_characteristics(expression, pi_expressions)
        
        # 評估認知負荷
        cognitive_load = self._evaluate_cognitive_load(expression, pi_expressions)
        
        # 評估社會適當性
        social_appropriateness = self._evaluate_social_appropriateness(expression, context)
        
        # 評估教育價值
        educational_value = self._evaluate_educational_value(expression, pi_expressions)
        
        return {
            'pi_expressions_detected': pi_expressions,
            'expression_characteristics': analysis,
            'cognitive_evaluation': cognitive_load,
            'social_evaluation': social_appropriateness,
            'educational_evaluation': educational_value,
            'overall_assessment': self._create_overall_assessment(
                analysis, cognitive_load, social_appropriateness, educational_value
            )
        }
    
    def _analyze_expression_characteristics(self, expression: str, pi_expressions: List[Dict]) -> Dict[str, Any]:
        """分析表達式特徵"""
        characteristics = {
            'formality_level': 'formal',  # π表達式通常是正式的
            'technical_complexity': 'medium',
            'mathematical_content': True,
            'multilingual': self._detect_multilingual(expression),
            'educational_intent': self._detect_educational_intent(expression)
        }
        
        # 基於π表達式類型調整複雜度
        complex_types = ['pi_series', 'pi_formula']
        if any(expr['type'] in complex_types for expr in pi_expressions):
            characteristics['technical_complexity'] = 'high'
        
        return characteristics
    
    def _evaluate_cognitive_load(self, expression: str, pi_expressions: List[Dict]) -> Dict[str, Any]:
        """評估認知負荷"""
        # 基礎認知負荷
        base_load = 0.3  # π本身就有一定複雜度
        
        # 基於表達式數量調整
        expr_count_factor = min(len(pi_expressions) * 0.1, 0.4)
        
        # 基於技術複雜度調整
        tech_complexity = {
            'pi_constant': 0.1,
            'pi_calculation': 0.2,
            'pi_series': 0.4,
            'pi_formula': 0.4,
            'pi_geometric': 0.2,
            'pi_numerical': 0.2
        }
        
        tech_factor = max(tech_complexity.get(expr['type'], 0.2) for expr in pi_expressions)
        
        # 計算總認知負荷
        total_load = min(base_load + expr_count_factor + tech_factor, 1.0)
        
        return {
            'cognitive_load_score': total_load,
            'factors': {
                'base_load': base_load,
                'expression_count': expr_count_factor,
                'technical_complexity': tech_factor
            },
            'accessibility': 'high' if total_load < 0.4 else 'medium' if total_load < 0.7 else 'low'
        }
    
    def _evaluate_social_appropriateness(self, expression: str, context: Optional[ExpressionContext]) -> Dict[str, Any]:
        """評估社會適當性"""
        appropriateness = 0.8  # π表達式通常是社會接受的
        
        if context:
            # 基於情境調整
            if context.situation in ['academic', 'educational', 'mathematical']:
                appropriateness = 0.95
            elif context.situation in ['casual', 'informal']:
                appropriateness = 0.6  # 可能過於技術性
            
            # 基於形式性調整
            if context.formality_level == 'formal' and any(word in expression for word in ['π', 'pi', '數學', 'mathematics']):
                appropriateness += 0.05
        
        return {
            'appropriateness_score': min(appropriateness, 1.0),
            'context_fit': 'excellent' if appropriateness > 0.8 else 'good' if appropriateness > 0.6 else 'poor',
            'recommendations': self._get_appropriateness_recommendations(appropriateness, context)
        }
    
    def _evaluate_educational_value(self, expression: str, pi_expressions: List[Dict]) -> Dict[str, Any]:
        """評估教育價值"""
        educational_score = 0.7  # π表達式通常有教育價值
        
        # 基於表達式類型調整
        educational_weights = {
            'pi_constant': 0.6,
            'pi_calculation': 0.8,
            'pi_series': 0.9,
            'pi_formula': 0.9,
            'pi_geometric': 0.7,
            'pi_numerical': 0.6
        }
        
        max_educational_value = max(educational_weights.get(expr['type'], 0.6) for expr in pi_expressions)
        educational_score = max(educational_score, max_educational_value)
        
        # 檢測解釋性內容
        explanatory_keywords = ['因為', 'because', '所以', 'therefore', '定義', 'definition', '公式', 'formula']
        if any(keyword in expression for keyword in explanatory_keywords):
            educational_score += 0.1
        
        return {
            'educational_score': min(educational_score, 1.0),
            'learning_potential': 'high' if educational_score > 0.8 else 'medium' if educational_score > 0.6 else 'low',
            'educational_aspects': self._identify_educational_aspects(expression, pi_expressions)
        }
    
    def _detect_multilingual(self, expression: str) -> bool:
        """檢測是否為多語言表達"""
        # 簡單檢測中英文混合
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', expression))
        has_english = bool(re.search(r'[a-zA-Z]', expression))
        return has_chinese and has_english
    
    def _detect_educational_intent(self, expression: str) -> bool:
        """檢測教育意圖"""
        educational_keywords = ['學習', 'learn', '教', 'teach', '解釋', 'explain', '理解', 'understand']
        return any(keyword in expression.lower() for keyword in educational_keywords)
    
    def _get_appropriateness_recommendations(self, score: float, context: Optional[ExpressionContext]) -> List[str]:
        """獲取適當性建議"""
        recommendations = []
        
        if score < 0.7:
            recommendations.append("考慮簡化數學術語 (Consider simplifying mathematical terms)")
            if context and context.situation == 'casual':
                recommendations.append("在非正式場合可提供更多背景解釋 (Provide more context in informal settings)")
        
        if score > 0.8:
            recommendations.append("表達恰當，適合當前語境 (Expression is appropriate for current context)")
        
        return recommendations
    
    def _identify_educational_aspects(self, expression: str, pi_expressions: List[Dict]) -> List[str]:
        """識別教育方面"""
        aspects = []
        
        if any(expr['type'] == 'pi_series' for expr in pi_expressions):
            aspects.append("無窮級數概念 (Infinite series concepts)")
        
        if any(expr['type'] == 'pi_geometric' for expr in pi_expressions):
            aspects.append("幾何學關係 (Geometric relationships)")
        
        if any(expr['type'] == 'pi_calculation' for expr in pi_expressions):
            aspects.append("數值計算方法 (Numerical calculation methods)")
        
        if '精度' in expression or 'precision' in expression:
            aspects.append("數值精度概念 (Numerical precision concepts)")
        
        return aspects
    
    def _create_overall_assessment(self, characteristics: Dict, cognitive: Dict, social: Dict, educational: Dict) -> Dict[str, Any]:
        """創建總體評估"""
        # 計算綜合分數
        weights = {
            'cognitive': 0.3,
            'social': 0.3,
            'educational': 0.4
        }
        
        overall_score = (
            cognitive['cognitive_load_score'] * weights['cognitive'] +
            social['appropriateness_score'] * weights['social'] +
            educational['educational_score'] * weights['educational']
        )
        
        # 創建評估摘要
        summary = []
        if characteristics['technical_complexity'] == 'high':
            summary.append("高技術複雜度的數學表達 (High technical complexity mathematical expression)")
        
        if cognitive['accessibility'] == 'high':
            summary.append("認知上容易理解 (Cognitively accessible)")
        
        if educational['learning_potential'] == 'high':
            summary.append("具有高教育價值 (High educational value)")
        
        return {
            'overall_score': overall_score,
            'assessment_summary': summary,
            'strengths': self._identify_strengths(characteristics, cognitive, social, educational),
            'improvement_suggestions': self._generate_improvement_suggestions(characteristics, cognitive, social, educational)
        }
    
    def _identify_strengths(self, characteristics: Dict, cognitive: Dict, social: Dict, educational: Dict) -> List[str]:
        """識別優勢"""
        strengths = []
        
        if educational['educational_score'] > 0.8:
            strengths.append("高教育價值 (High educational value)")
        
        if social['appropriateness_score'] > 0.8:
            strengths.append("社會適當性強 (High social appropriateness)")
        
        if characteristics['mathematical_content']:
            strengths.append("數學內容豐富 (Rich mathematical content)")
        
        if characteristics['multilingual']:
            strengths.append("多語言表達 (Multilingual expression)")
        
        return strengths
    
    def _generate_improvement_suggestions(self, characteristics: Dict, cognitive: Dict, social: Dict, educational: Dict) -> List[str]:
        """生成改進建議"""
        suggestions = []
        
        if cognitive['cognitive_load_score'] > 0.7:
            suggestions.append("考慮分段解釋複雜概念 (Consider breaking down complex concepts)")
        
        if social['appropriateness_score'] < 0.7:
            suggestions.append("調整表達方式以適應語境 (Adjust expression style for context)")
        
        if educational['educational_score'] < 0.7:
            suggestions.append("增加教育性解釋 (Add educational explanations)")
        
        if not characteristics['multilingual'] and '中文' not in characteristics:
            suggestions.append("考慮添加中文解釋 (Consider adding Chinese explanations)")
        
        return suggestions


def demonstrate_pi_integration():
    """演示π表達式整合評估"""
    print("=" * 70)
    print("π表達式與人類表達評估整合演示")
    print("Pi Expressions and Human Expression Evaluation Integration Demo")
    print("=" * 70)
    print()
    
    # 初始化整合評估器
    evaluator = PiExpressionIntegratedEvaluator()
    
    # 測試案例
    test_expressions = [
        {
            'text': '請計算π的值到小數點後10位',
            'context': ExpressionContext(
                situation='educational',
                formality_level='formal',
                speaker='teacher',
                listener='student'
            ),
            'description': '教育場景中的π計算請求'
        },
        {
            'text': '萊布尼茲級數可以用來計算π，雖然收斂很慢',
            'context': ExpressionContext(
                situation='academic',
                formality_level='formal',
                speaker='researcher',
                listener='colleagues'
            ),
            'description': '學術討論中的π級數説明'
        },
        {
            'text': 'π大概等於3.14對吧？',
            'context': ExpressionContext(
                situation='casual',
                formality_level='informal',
                speaker='friend',
                listener='friend'
            ),
            'description': '非正式對話中的π提及'
        },
        {
            'text': 'The formula π = 4 × Σ(k=0→∞)[(-1)^k/(2k+1)] represents the Leibniz series',
            'context': ExpressionContext(
                situation='academic',
                formality_level='formal',
                speaker='professor',
                listener='class'
            ),
            'description': '英文學術表達中的π級數公式'
        }
    ]
    
    for i, test_case in enumerate(test_expressions, 1):
        print(f"測試案例 {i}: {test_case['description']}")
        print("-" * 50)
        print(f"表達: {test_case['text']}")
        print(f"語境: {test_case['context'].__dict__}")
        print()
        
        # 1. π表達式檢測
        pi_detection = evaluator.detect_pi_expressions(test_case['text'])
        print("π表達式檢測結果:")
        if pi_detection:
            for detection in pi_detection:
                print(f"  - 類型: {detection['type']}")
                print(f"    文本: '{detection['text']}'")
                print(f"    信心度: {detection['confidence']:.2f}")
        else:
            print("  未檢測到π表達式")
        print()
        
        # 2. 人類表達評估
        print("人類表達評估結果:")
        try:
            human_eval = evaluator.evaluate_as_human_expression(test_case['text'], test_case['context'])
            
            if 'error' in human_eval:
                print(f"  錯誤: {human_eval['error']}")
            else:
                # 顯示主要評估結果
                if 'cognitive_evaluation' in human_eval:
                    cog = human_eval['cognitive_evaluation']
                    print(f"  認知負荷: {cog['cognitive_load_score']:.2f} ({cog['accessibility']})")
                
                if 'social_evaluation' in human_eval:
                    soc = human_eval['social_evaluation']
                    print(f"  社會適當性: {soc['appropriateness_score']:.2f} ({soc['context_fit']})")
                
                if 'educational_evaluation' in human_eval:
                    edu = human_eval['educational_evaluation']
                    print(f"  教育價值: {edu['educational_score']:.2f} ({edu['learning_potential']})")
                
                if 'overall_assessment' in human_eval:
                    overall = human_eval['overall_assessment']
                    print(f"  總體分數: {overall['overall_score']:.2f}")
                    print(f"  優勢: {', '.join(overall['strengths'])}")
                    if overall['improvement_suggestions']:
                        print(f"  建議: {', '.join(overall['improvement_suggestions'])}")
        except Exception as e:
            print(f"  評估錯誤: {e}")
        
        print()
        
        # 3. 數學評估（如果可用）
        if PI_EVALUATOR_AVAILABLE and pi_detection:
            print("數學評估結果:")
            try:
                math_eval = evaluator.evaluate_pi_expression_mathematically(test_case['text'], test_case['context'])
                
                if 'mathematical_results' in math_eval and math_eval['mathematical_results']:
                    for method, result in math_eval['mathematical_results'].items():
                        if 'result' in result:
                            print(f"  {method}: π ≈ {result['result']:.8f}")
                            print(f"    誤差: {result.get('error', 'N/A'):.2e}")
                else:
                    print("  沒有執行具體的數學計算")
            except Exception as e:
                print(f"  數學評估錯誤: {e}")
        else:
            print("數學評估: 不可用或未檢測到π表達式")
        
        print("\n" + "="*70)
        
        if i < len(test_expressions):
            input("按Enter繼續下一個測試案例...")
    
    print("\n整合演示完成！")
    print("Integration demonstration complete!")


if __name__ == "__main__":
    demonstrate_pi_integration()