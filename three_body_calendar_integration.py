#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三體曆法系統整合模組 (Three-Body Calendar Integration Module)

整合三體曆法系統與現有的NLP工具，提供時間表達式的深度分析能力。

Integration of Three-Body Calendar System with existing NLP tools,
providing deep analysis capabilities for temporal expressions.
"""

import datetime
from typing import Dict, List, Optional, Union, Any
from ThreeBodyCalendar import ThreeBodyCalendar, ThreeBodyDateParser

# Try to import existing NLP tools with graceful fallback
try:
    from SubtextAnalyzer import SubtextAnalyzer
    SUBTEXT_AVAILABLE = True
except ImportError:
    SUBTEXT_AVAILABLE = False
    print("SubtextAnalyzer not available, using simplified analysis")

try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    HUMAN_EXPR_AVAILABLE = True
except ImportError:
    HUMAN_EXPR_AVAILABLE = False
    print("HumanExpressionEvaluator not available, using basic evaluation")


class TemporalExpressionAnalyzer:
    """時間表達式分析器 - Temporal expression analyzer with three-body calendar integration"""
    
    def __init__(self):
        self.three_body_calendar = ThreeBodyCalendar()
        self.three_body_parser = ThreeBodyDateParser()
        
        # Initialize optional components
        self.subtext_analyzer = SubtextAnalyzer() if SUBTEXT_AVAILABLE else None
        self.expression_evaluator = HumanExpressionEvaluator() if HUMAN_EXPR_AVAILABLE else None
        
    def analyze_temporal_expression(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析時間表達式 - Comprehensive analysis of temporal expressions
        
        Args:
            text: 輸入文本 Input text
            context: 語境信息 Context information
            
        Returns:
            完整的分析結果 Complete analysis results
        """
        result = {
            'input_text': text,
            'timestamp': datetime.datetime.now().isoformat(),
            'three_body_analysis': {},
            'subtext_analysis': {},
            'expression_evaluation': {},
            'integrated_insights': {}
        }
        
        # Three-body calendar analysis
        three_body_result = self.three_body_parser.analyze_date_expression(text)
        result['three_body_analysis'] = three_body_result
        
        # Subtext analysis if available
        if self.subtext_analyzer:
            try:
                subtext_result = self.subtext_analyzer.analyze(text)
                result['subtext_analysis'] = subtext_result
            except Exception as e:
                result['subtext_analysis'] = {'error': str(e)}
        
        # Human expression evaluation if available  
        if self.expression_evaluator and context:
            try:
                expr_context = ExpressionContext(
                    situation=context.get('situation', 'temporal_discussion'),
                    cultural_background=context.get('culture', 'universal'),
                    formality_level=context.get('formality', 'neutral')
                )
                expr_result = self.expression_evaluator.comprehensive_evaluation(text, expr_context)
                result['expression_evaluation'] = expr_result
            except Exception as e:
                result['expression_evaluation'] = {'error': str(e)}
        
        # Generate integrated insights
        result['integrated_insights'] = self._generate_insights(result)
        
        return result
    
    def _generate_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成整合洞察 - Generate integrated insights from multiple analyses"""
        insights = {
            'temporal_significance': 'unknown',
            'cultural_context': 'neutral',
            'astronomical_importance': 'normal',
            'linguistic_complexity': 'basic',
            'recommendations': []
        }
        
        three_body = analysis_result.get('three_body_analysis', {})
        
        # Assess temporal significance
        if three_body.get('parsed_date'):
            parsed_date = three_body['parsed_date']
            lunar_phase = three_body.get('lunar_phase', {})
            solar_term = three_body.get('solar_term', {})
            
            # Check for astronomical significance
            if lunar_phase:
                phase_en = lunar_phase.get('en', '')
                if 'Full Moon' in phase_en or 'New Moon' in phase_en:
                    insights['astronomical_importance'] = 'high'
                    insights['recommendations'].append('特殊月相時刻 - Special lunar phase moment')
            
            if solar_term:
                term_en = solar_term.get('en', '')
                if any(term in term_en for term in ['Solstice', 'Equinox']):
                    insights['astronomical_importance'] = 'very_high'
                    insights['recommendations'].append('重要節氣 - Important solar term')
        
        # Assess cultural context from subtext analysis
        subtext = analysis_result.get('subtext_analysis', {})
        if isinstance(subtext, dict) and 'probability' in subtext:
            if subtext.get('probability', 0) > 0.7:
                insights['cultural_context'] = 'rich'
                insights['recommendations'].append('豐富的文化內涵 - Rich cultural connotations')
        
        # Assess linguistic complexity from expression evaluation
        expr_eval = analysis_result.get('expression_evaluation', {})
        if isinstance(expr_eval, dict) and 'integrated' in expr_eval:
            integrated_score = expr_eval['integrated'].get('overall_score', 0)
            if integrated_score > 0.7:
                insights['linguistic_complexity'] = 'high'
                insights['recommendations'].append('語言表達複雜 - Complex linguistic expression')
        
        return insights
    
    def comparative_temporal_analysis(self, expressions: List[str]) -> Dict[str, Any]:
        """比較時間表達式分析 - Comparative analysis of multiple temporal expressions"""
        results = []
        
        for expr in expressions:
            analysis = self.analyze_temporal_expression(expr)
            results.append(analysis)
        
        # Generate comparative insights
        comparison = {
            'expressions_count': len(expressions),
            'parsed_dates_count': sum(1 for r in results if r['three_body_analysis'].get('parsed_date')),
            'astronomical_events': [],
            'cultural_patterns': [],
            'summary': {}
        }
        
        # Collect astronomical events
        for result in results:
            three_body = result['three_body_analysis']
            if three_body.get('lunar_phase'):
                comparison['astronomical_events'].append({
                    'text': result['input_text'],
                    'phase': three_body['lunar_phase'],
                    'term': three_body.get('solar_term')
                })
        
        # Generate summary
        comparison['summary'] = {
            'total_expressions': len(expressions),
            'successfully_parsed': comparison['parsed_dates_count'],
            'success_rate': comparison['parsed_dates_count'] / len(expressions) if expressions else 0,
            'dominant_themes': self._extract_themes(results)
        }
        
        return {
            'individual_results': results,
            'comparative_analysis': comparison
        }
    
    def _extract_themes(self, results: List[Dict]) -> List[str]:
        """提取主題 - Extract dominant themes from analysis results"""
        themes = []
        
        # Count lunar phases
        lunar_phases = {}
        solar_terms = {}
        
        for result in results:
            three_body = result['three_body_analysis']
            if three_body.get('lunar_phase'):
                phase = three_body['lunar_phase'].get('en', 'unknown')
                lunar_phases[phase] = lunar_phases.get(phase, 0) + 1
            
            if three_body.get('solar_term'):
                term = three_body['solar_term'].get('en', 'unknown')
                solar_terms[term] = solar_terms.get(term, 0) + 1
        
        # Determine dominant themes
        if lunar_phases:
            most_common_phase = max(lunar_phases.items(), key=lambda x: x[1])[0]
            themes.append(f"Lunar focus: {most_common_phase}")
        
        if solar_terms:
            most_common_term = max(solar_terms.items(), key=lambda x: x[1])[0]
            themes.append(f"Solar focus: {most_common_term}")
        
        return themes


def demo_integration():
    """演示整合功能 - Demo integration capabilities"""
    print("=== 三體曆法系統整合演示 Three-Body Calendar Integration Demo ===\n")
    
    analyzer = TemporalExpressionAnalyzer()
    
    # Test expressions
    test_expressions = [
        "2024年冬至",
        "2025年春分將至",
        "下個滿月是什麼時候？",
        "今天的節氣是什麼",
        "Winter solstice 2024",
        "Next new moon",
        "中秋節快到了"
    ]
    
    print("1. 單個表達式分析 Individual Expression Analysis:")
    print("-" * 50)
    
    for expr in test_expressions[:3]:  # Test first 3 expressions
        print(f"\n分析表達式 Analyzing: '{expr}'")
        result = analyzer.analyze_temporal_expression(expr, {
            'situation': 'casual_conversation',
            'culture': 'chinese',
            'formality': 'casual'
        })
        
        print(f"三體分析結果 Three-body result:")
        three_body = result['three_body_analysis']
        if three_body.get('parsed_date'):
            print(f"  解析日期 Parsed date: {three_body['parsed_date']}")
            if three_body.get('lunar_phase'):
                print(f"  月相 Lunar phase: {three_body['lunar_phase']}")
            if three_body.get('solar_term'):
                print(f"  節氣 Solar term: {three_body['solar_term']}")
        else:
            print("  無法解析日期 Could not parse date")
        
        insights = result['integrated_insights']
        print(f"整合洞察 Insights:")
        print(f"  天文重要性 Astronomical importance: {insights['astronomical_importance']}")
        print(f"  文化語境 Cultural context: {insights['cultural_context']}")
        if insights['recommendations']:
            print(f"  建議 Recommendations: {', '.join(insights['recommendations'])}")
    
    print("\n" + "="*70 + "\n")
    
    print("2. 比較分析 Comparative Analysis:")
    print("-" * 50)
    
    comparison_result = analyzer.comparative_temporal_analysis(test_expressions)
    summary = comparison_result['comparative_analysis']['summary']
    
    print(f"總表達式數量 Total expressions: {summary['total_expressions']}")
    print(f"成功解析數量 Successfully parsed: {summary['successfully_parsed']}")
    print(f"成功率 Success rate: {summary['success_rate']:.1%}")
    
    if summary['dominant_themes']:
        print(f"主要主題 Dominant themes:")
        for theme in summary['dominant_themes']:
            print(f"  - {theme}")
    
    astronomical_events = comparison_result['comparative_analysis']['astronomical_events']
    if astronomical_events:
        print(f"\n檢測到的天文事件 Detected astronomical events:")
        for event in astronomical_events:
            print(f"  '{event['text']}' - {event.get('phase', {}).get('zh', 'unknown')}")


class ThreeBodyCalendarPlugin:
    """三體曆法插件 - Plugin for integrating with other NLP tools"""
    
    @staticmethod
    def enhance_subtext_analyzer():
        """增強潛文本分析器 - Enhance SubtextAnalyzer with temporal awareness"""
        if not SUBTEXT_AVAILABLE:
            print("SubtextAnalyzer not available for enhancement")
            return None
        
        class EnhancedSubtextAnalyzer(SubtextAnalyzer):
            def __init__(self):
                super().__init__()
                self.temporal_analyzer = TemporalExpressionAnalyzer()
            
            def analyze_with_temporal_context(self, text: str) -> Dict[str, Any]:
                """帶時間語境的潛文本分析"""
                # Original subtext analysis
                subtext_result = self.analyze(text)
                
                # Temporal analysis
                temporal_result = self.temporal_analyzer.analyze_temporal_expression(text)
                
                # Enhanced analysis combining both
                enhanced_result = {
                    'original_subtext': subtext_result,
                    'temporal_analysis': temporal_result,
                    'enhanced_interpretation': self._combine_analyses(subtext_result, temporal_result)
                }
                
                return enhanced_result
            
            def _combine_analyses(self, subtext: Dict, temporal: Dict) -> str:
                """結合分析結果"""
                interpretation = "文本分析結果:\n"
                
                if temporal.get('three_body_analysis', {}).get('parsed_date'):
                    interpretation += f"- 檢測到時間表達: {temporal['three_body_analysis']['parsed_date']}\n"
                
                if isinstance(subtext, dict) and 'probability' in subtext:
                    interpretation += f"- 潛文本概率: {subtext['probability']:.2f}\n"
                
                insights = temporal.get('integrated_insights', {})
                if insights.get('recommendations'):
                    interpretation += f"- 建議: {', '.join(insights['recommendations'])}\n"
                
                return interpretation
        
        return EnhancedSubtextAnalyzer


if __name__ == "__main__":
    demo_integration()