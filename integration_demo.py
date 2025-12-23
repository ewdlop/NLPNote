#!/usr/bin/env python3
"""
OrientationLessParser Integration Demo

This demo shows how the OrientationLessParser integrates with the existing
NLP framework, providing orientation-agnostic text processing capabilities.

無方向性解析器整合演示
此演示展示了無方向性解析器如何與現有的NLP框架整合，提供無方向性的文本處理能力。
"""

import sys
import os
from typing import Optional, Dict, Any

# Import our new OrientationLessParser
from OrientationLessParser import OrientationLessParser, TextDirection, ScriptType

# Try to import existing framework components (with fallbacks)
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EXPRESSION_EVALUATOR_AVAILABLE = True
except ImportError:
    print("Note: HumanExpressionEvaluator not available (missing dependencies)")
    HumanExpressionEvaluator = None
    ExpressionContext = None
    EXPRESSION_EVALUATOR_AVAILABLE = False

try:
    from SubtextAnalyzer import SubtextAnalyzer
    SUBTEXT_ANALYZER_AVAILABLE = True
except ImportError:
    print("Note: SubtextAnalyzer not available (missing dependencies)")
    SubtextAnalyzer = None
    SUBTEXT_ANALYZER_AVAILABLE = False


class IntegratedNLPProcessor:
    """
    整合的NLP處理器 (Integrated NLP Processor)
    
    Combines OrientationLessParser with existing NLP framework components
    to provide comprehensive, orientation-agnostic text analysis.
    
    結合無方向性解析器與現有NLP框架組件，提供全面的、無方向性的文本分析。
    """
    
    def __init__(self):
        """初始化處理器 (Initialize processor)"""
        self.orientation_parser = OrientationLessParser()
        
        # Initialize other components if available
        if EXPRESSION_EVALUATOR_AVAILABLE:
            self.expression_evaluator = HumanExpressionEvaluator()
        else:
            self.expression_evaluator = None
            
        if SUBTEXT_ANALYZER_AVAILABLE:
            self.subtext_analyzer = SubtextAnalyzer()
        else:
            self.subtext_analyzer = None
    
    def comprehensive_analysis(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        綜合分析文本 (Comprehensive text analysis)
        
        Args:
            text: 要分析的文本 (Text to analyze)
            context: 分析上下文 (Analysis context)
            
        Returns:
            Dict[str, Any]: 綜合分析結果 (Comprehensive analysis results)
        """
        results = {
            'original_text': text,
            'orientation_analysis': None,
            'expression_evaluation': None,
            'subtext_analysis': None,
            'integrated_insights': None
        }
        
        # 1. Orientation-agnostic parsing
        print("🔍 Performing orientation-agnostic parsing...")
        orientation_result = self.orientation_parser.parse(text)
        results['orientation_analysis'] = {
            'dominant_direction': orientation_result.dominant_direction.value,
            'dominant_script': orientation_result.dominant_script.value,
            'has_mixed_directions': orientation_result.has_mixed_directions,
            'normalized_text': orientation_result.normalized_text,
            'token_count': len(orientation_result.tokens),
            'statistics': self.orientation_parser.get_parsing_statistics(orientation_result),
            'logical_text': self.orientation_parser.extract_text_content(orientation_result)
        }
        
        # 2. Expression evaluation (if available)
        if self.expression_evaluator and ExpressionContext:
            print("📊 Performing expression evaluation...")
            try:
                # Create appropriate context
                expr_context = ExpressionContext(
                    situation='analysis',
                    formality_level='neutral',
                    cultural_background=self._infer_cultural_background(orientation_result)
                )
                
                expr_result = self.expression_evaluator.comprehensive_evaluation(
                    orientation_result.normalized_text, expr_context
                )
                results['expression_evaluation'] = {
                    'overall_score': expr_result['integrated']['overall_score'],
                    'confidence': expr_result['integrated']['overall_confidence'],
                    'characteristics': expr_result['integrated']['characteristics'],
                    'formal_semantic': expr_result['formal_semantic'].score,
                    'cognitive': expr_result['cognitive'].score,
                    'social': expr_result['social'].score
                }
            except Exception as e:
                print(f"Expression evaluation failed: {e}")
                results['expression_evaluation'] = {'error': str(e)}
        
        # 3. Subtext analysis (if available)
        if self.subtext_analyzer:
            print("🔎 Performing subtext analysis...")
            try:
                subtext_result = self.subtext_analyzer.calculate_subtext_probability(text)
                results['subtext_analysis'] = {
                    'probability': subtext_result['probability'],
                    'components': subtext_result['components']
                }
            except Exception as e:
                print(f"Subtext analysis failed: {e}")
                results['subtext_analysis'] = {'error': str(e)}
        
        # 4. Generate integrated insights
        results['integrated_insights'] = self._generate_integrated_insights(results)
        
        return results
    
    def _infer_cultural_background(self, orientation_result) -> str:
        """
        根據文字系統推斷文化背景 (Infer cultural background from script type)
        """
        script_mapping = {
            ScriptType.LATIN: 'western',
            ScriptType.ARABIC: 'arabic',
            ScriptType.HEBREW: 'hebrew',
            ScriptType.CJK: 'east_asian',
            ScriptType.DEVANAGARI: 'south_asian',
            ScriptType.CYRILLIC: 'slavic',
            ScriptType.THAI: 'southeast_asian',
            ScriptType.MIXED: 'multicultural',
            ScriptType.UNKNOWN: 'universal'
        }
        return script_mapping.get(orientation_result.dominant_script, 'universal')
    
    def _generate_integrated_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成整合洞察 (Generate integrated insights)
        """
        insights = {
            'text_complexity': 'unknown',
            'cross_cultural_considerations': [],
            'orientation_challenges': [],
            'recommended_processing': []
        }
        
        orientation_data = results.get('orientation_analysis', {})
        
        # Text complexity assessment
        if orientation_data:
            if orientation_data.get('has_mixed_directions'):
                insights['text_complexity'] = 'high'
                insights['orientation_challenges'].append('Bidirectional text processing required')
            elif orientation_data.get('dominant_script') == 'mixed':
                insights['text_complexity'] = 'medium-high'
                insights['cross_cultural_considerations'].append('Multiple script systems present')
            else:
                insights['text_complexity'] = 'low-medium'
        
        # Processing recommendations
        if orientation_data.get('dominant_direction') == 'right_to_left':
            insights['recommended_processing'].append('Apply RTL-aware text layout')
        if orientation_data.get('has_mixed_directions'):
            insights['recommended_processing'].append('Use bidirectional algorithm for display')
        if orientation_data.get('dominant_script') == 'mixed':
            insights['recommended_processing'].append('Handle multiple input methods')
        
        # Cross-cultural considerations
        script_type = orientation_data.get('dominant_script', 'unknown')
        if script_type in ['arabic', 'hebrew']:
            insights['cross_cultural_considerations'].append('RTL reading pattern considerations')
        elif script_type == 'cjk':
            insights['cross_cultural_considerations'].append('CJK typography and spacing considerations')
        elif script_type == 'mixed':
            insights['cross_cultural_considerations'].append('Multilingual interface requirements')
        
        return insights
    
    def generate_analysis_report(self, text: str, context: Optional[Dict] = None) -> str:
        """
        生成分析報告 (Generate analysis report)
        """
        print("📋 Generating comprehensive analysis report...")
        results = self.comprehensive_analysis(text, context)
        
        report = []
        report.append("=" * 80)
        report.append("INTEGRATED NLP ANALYSIS REPORT")
        report.append("整合NLP分析報告")
        report.append("=" * 80)
        report.append("")
        
        # Input text
        report.append(f"📝 Original Text: {text}")
        report.append("")
        
        # Orientation analysis
        orientation = results['orientation_analysis']
        if orientation:
            report.append("🧭 ORIENTATION ANALYSIS 方向分析")
            report.append("-" * 40)
            report.append(f"Dominant Direction: {orientation['dominant_direction']}")
            report.append(f"Dominant Script: {orientation['dominant_script']}")
            report.append(f"Mixed Directions: {orientation['has_mixed_directions']}")
            report.append(f"Normalized Text: {orientation['normalized_text']}")
            report.append(f"Logical Order Text: {orientation['logical_text']}")
            report.append(f"Token Count: {orientation['token_count']}")
            
            stats = orientation['statistics']
            report.append(f"Statistics: {stats['word_tokens']} words, "
                         f"{stats['punctuation_tokens']} punctuation, "
                         f"{stats['number_tokens']} numbers")
            report.append("")
        
        # Expression evaluation
        expression = results['expression_evaluation']
        if expression and 'error' not in expression:
            report.append("🎭 EXPRESSION EVALUATION 表達評估")
            report.append("-" * 40)
            report.append(f"Overall Score: {expression['overall_score']:.2f}")
            report.append(f"Confidence: {expression['confidence']:.2f}")
            report.append(f"Formal Semantic: {expression['formal_semantic']:.2f}")
            report.append(f"Cognitive: {expression['cognitive']:.2f}")
            report.append(f"Social: {expression['social']:.2f}")
            report.append(f"Characteristics: {expression['characteristics']}")
            report.append("")
        elif expression and 'error' in expression:
            report.append("🎭 EXPRESSION EVALUATION: Not available")
            report.append("")
        
        # Subtext analysis
        subtext = results['subtext_analysis']
        if subtext and 'error' not in subtext:
            report.append("🔍 SUBTEXT ANALYSIS 潛文本分析")
            report.append("-" * 40)
            report.append(f"Subtext Probability: {subtext['probability']:.2f}")
            report.append("Components:")
            for component, score in subtext['components'].items():
                report.append(f"  - {component.replace('_', ' ').title()}: {score:.2f}")
            report.append("")
        elif subtext and 'error' in subtext:
            report.append("🔍 SUBTEXT ANALYSIS: Not available")
            report.append("")
        
        # Integrated insights
        insights = results['integrated_insights']
        if insights:
            report.append("💡 INTEGRATED INSIGHTS 整合洞察")
            report.append("-" * 40)
            report.append(f"Text Complexity: {insights['text_complexity']}")
            
            if insights['orientation_challenges']:
                report.append("Orientation Challenges:")
                for challenge in insights['orientation_challenges']:
                    report.append(f"  - {challenge}")
            
            if insights['cross_cultural_considerations']:
                report.append("Cross-Cultural Considerations:")
                for consideration in insights['cross_cultural_considerations']:
                    report.append(f"  - {consideration}")
            
            if insights['recommended_processing']:
                report.append("Recommended Processing:")
                for recommendation in insights['recommended_processing']:
                    report.append(f"  - {recommendation}")
            report.append("")
        
        report.append("=" * 80)
        report.append("ANALYSIS COMPLETE 分析完成")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("OrientationLessParser Integration Demo")
    print("無方向性解析器整合演示")
    print("=" * 80)
    print()
    
    # Initialize integrated processor
    processor = IntegratedNLPProcessor()
    
    # Demo test cases
    test_cases = [
        {
            'name': 'English Text',
            'text': 'Hello world! This is a test of our orientation-less parser.',
            'description': 'Standard English left-to-right text'
        },
        {
            'name': 'Arabic Text',
            'text': 'مرحبا بالعالم! هذا اختبار لمحلل النصوص الذي لا يعتمد على الاتجاه.',
            'description': 'Arabic right-to-left text'
        },
        {
            'name': 'Mixed Script Text',
            'text': 'Hello مرحبا 你好 שלום world! This is a multilingual test.',
            'description': 'Text with multiple scripts and directions'
        },
        {
            'name': 'Complex Expression',
            'text': '如果我們考慮所有可能的情況，那麼我們必須承認這個問題比我們想像的更複雜。',
            'description': 'Complex Chinese expression with logical structure'
        },
        {
            'name': 'Bidirectional Text',
            'text': 'The Arabic word مرحبا means "hello" and the Hebrew word שלום means "peace".',
            'description': 'English text with embedded Arabic and Hebrew'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print("-" * 60)
        
        # Generate comprehensive report
        report = processor.generate_analysis_report(test_case['text'])
        print(report)
        
        # Pause between test cases (except for the last one)
        if i < len(test_cases):
            try:
                input("\nPress Enter to continue to next test case...")
            except KeyboardInterrupt:
                print("\nDemo interrupted by user.")
                break
            print("\n" + "="*80)
    
    print("\n🎉 Demo completed successfully!")
    print("🎉 演示成功完成！")
    print("\nThe OrientationLessParser successfully integrates with the existing framework")
    print("to provide comprehensive, orientation-agnostic text analysis capabilities.")
    print("\n無方向性解析器成功與現有框架整合，")
    print("提供全面的、無方向性的文本分析能力。")


if __name__ == "__main__":
    main()