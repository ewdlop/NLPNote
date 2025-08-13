#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daoist Encoding Analyzer for Chinese Text
道教编码分析器

This module analyzes Chinese text to identify Daoist philosophical concepts
encoded within the language structure, vocabulary, and semantic patterns.

Author: NLP Research Team
Date: 2024-12-22
"""

import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DaoistConcept:
    """Represents a Daoist philosophical concept found in text"""
    concept_type: str
    chinese_text: str
    english_translation: str
    philosophical_significance: str
    confidence_score: float

class DaoistEncodingAnalyzer:
    """
    Analyzes Chinese text for embedded Daoist philosophical concepts
    """
    
    def __init__(self):
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize the knowledge base of Daoist concepts and patterns"""
        
        # Core Daoist characters and their meanings
        self.dao_characters = {
            '道': {'meaning': 'The Way, path, principle', 'significance': 'Central concept of Daoism'},
            '德': {'meaning': 'Virtue, power, integrity', 'significance': 'Dao manifested in action'},
            '無': {'meaning': 'Non-being, emptiness', 'significance': 'Source of all being'},
            '陰': {'meaning': 'Yin, feminine principle', 'significance': 'Passive, receptive force'},
            '陽': {'meaning': 'Yang, masculine principle', 'significance': 'Active, creative force'},
            '氣': {'meaning': 'Qi, vital energy', 'significance': 'Universal life force'},
            '自': {'meaning': 'Self, natural', 'significance': 'Natural spontaneity'},
            '然': {'meaning': 'So, thus, natural', 'significance': 'Natural state of being'}
        }
        
        # Daoist radicals/components
        self.dao_radicals = {
            '水': {'meaning': 'Water', 'significance': 'Yielding yet overcoming, wu wei principle'},
            '火': {'meaning': 'Fire', 'significance': 'Transformation, yang energy'},
            '木': {'meaning': 'Wood', 'significance': 'Natural growth, organic development'},
            '土': {'meaning': 'Earth', 'significance': 'Stability, foundation, center'},
            '金': {'meaning': 'Metal', 'significance': 'Refinement, structure, yin principle'},
            '辶': {'meaning': 'Movement', 'significance': 'The Way, path, journey'},
            '心': {'meaning': 'Heart', 'significance': 'Original nature, inner essence'},
            '目': {'meaning': 'Eye', 'significance': 'Perception, seeing the Way'}
        }
        
        # Daoist concept patterns
        self.dao_patterns = {
            'wu_wei': {
                'regex': r'(無為|不爭|隨緣|順其自然|自然而然)',
                'concepts': ['wu wei', 'non-action', 'effortless action', 'naturalness'],
                'significance': 'Acting in accordance with natural flow'
            },
            'yin_yang': {
                'regex': r'(陰陽|虛實|動靜|剛柔|上下|左右|內外)',
                'concepts': ['duality', 'complementary opposites', 'balance'],
                'significance': 'Fundamental principle of complementary forces'
            },
            'cycles': {
                'regex': r'(春夏秋冬|生老病死|盛衰興替|來去|始終|循環)',
                'concepts': ['cyclical time', 'eternal return', 'natural cycles'],
                'significance': 'Cyclical nature of existence'
            },
            'reversal': {
                'regex': r'(物極必反|樂極生悲|否極泰來|月盈則虧|水滿則溢|過猶不及)',
                'concepts': ['reversal at extremes', 'dialectical change'],
                'significance': 'Things transform into their opposites at extremes'
            },
            'naturalness': {
                'regex': r'(天然|本性|天道|自然|返璞歸真|天人合一)',
                'concepts': ['naturalness', 'original nature', 'unity with nature'],
                'significance': 'Return to natural, unmanipulated state'
            },
            'simplicity': {
                'regex': r'(樸素|簡單|純真|素雅|簡樸)',
                'concepts': ['simplicity', 'purity', 'uncarved block'],
                'significance': 'Daoist ideal of simplicity and authenticity'
            },
            'water_wisdom': {
                'regex': r'(上善若水|以柔克剛|滴水穿石|載舟覆舟)',
                'concepts': ['water-like behavior', 'soft overcoming hard'],
                'significance': 'Water as model for ideal behavior'
            }
        }
        
        # Five elements (Wu Xing) - fundamental to Daoist cosmology
        self.five_elements = {
            '木': {'element': 'Wood', 'season': '春', 'direction': '東', 'organ': '肝'},
            '火': {'element': 'Fire', 'season': '夏', 'direction': '南', 'organ': '心'},
            '土': {'element': 'Earth', 'season': '長夏', 'direction': '中', 'organ': '脾'},
            '金': {'element': 'Metal', 'season': '秋', 'direction': '西', 'organ': '肺'},
            '水': {'element': 'Water', 'season': '冬', 'direction': '北', 'organ': '腎'}
        }
        
        # Daoist numerical symbolism
        self.dao_numbers = {
            '一': 'Unity, the Dao, source of all',
            '二': 'Duality, yin-yang, fundamental polarity', 
            '三': 'Trinity, creative principle, heaven-earth-human',
            '五': 'Five elements, completeness',
            '八': 'Ba Gua, eight trigrams, cosmic principles',
            '九': 'Ultimate yang, completion',
            '十': 'Perfect completion, return to unity'
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of Daoist encoding in Chinese text
        
        Args:
            text: Chinese text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'text': text,
            'dao_characters': self._find_dao_characters(text),
            'dao_radicals': self._find_dao_radicals(text),
            'concept_patterns': self._find_concept_patterns(text),
            'five_elements': self._find_five_elements(text),
            'numerical_symbolism': self._find_numerical_symbolism(text),
            'overall_score': 0,
            'philosophical_themes': []
        }
        
        # Calculate overall Daoist encoding score
        results['overall_score'] = self._calculate_dao_score(results)
        
        # Extract main philosophical themes
        results['philosophical_themes'] = self._extract_themes(results)
        
        return results
    
    def _find_dao_characters(self, text: str) -> List[Dict]:
        """Find core Daoist characters in text"""
        found_chars = []
        for char, info in self.dao_characters.items():
            count = text.count(char)
            if count > 0:
                found_chars.append({
                    'character': char,
                    'count': count,
                    'meaning': info['meaning'],
                    'significance': info['significance'],
                    'positions': [i for i, c in enumerate(text) if c == char]
                })
        return found_chars
    
    def _find_dao_radicals(self, text: str) -> List[Dict]:
        """Find Daoist radicals/components in text"""
        found_radicals = []
        for radical, info in self.dao_radicals.items():
            count = text.count(radical)
            if count > 0:
                found_radicals.append({
                    'radical': radical,
                    'count': count,
                    'meaning': info['meaning'],
                    'significance': info['significance']
                })
        return found_radicals
    
    def _find_concept_patterns(self, text: str) -> List[Dict]:
        """Find Daoist conceptual patterns in text"""
        found_patterns = []
        for pattern_name, pattern_info in self.dao_patterns.items():
            matches = re.findall(pattern_info['regex'], text)
            if matches:
                found_patterns.append({
                    'pattern_type': pattern_name,
                    'matches': matches,
                    'concepts': pattern_info['concepts'],
                    'significance': pattern_info['significance'],
                    'count': len(matches)
                })
        return found_patterns
    
    def _find_five_elements(self, text: str) -> List[Dict]:
        """Find five elements references in text"""
        found_elements = []
        for element, info in self.five_elements.items():
            count = text.count(element)
            if count > 0:
                found_elements.append({
                    'element_char': element,
                    'element_name': info['element'],
                    'count': count,
                    'associated_season': info['season'],
                    'direction': info['direction'],
                    'organ': info['organ']
                })
        return found_elements
    
    def _find_numerical_symbolism(self, text: str) -> List[Dict]:
        """Find Daoist numerical symbolism in text"""
        found_numbers = []
        for number, meaning in self.dao_numbers.items():
            count = text.count(number)
            if count > 0:
                found_numbers.append({
                    'number': number,
                    'count': count,
                    'symbolic_meaning': meaning
                })
        return found_numbers
    
    def _calculate_dao_score(self, results: Dict) -> float:
        """Calculate overall Daoist encoding score (0-1)"""
        text_length = len(results['text'])
        if text_length == 0:
            return 0.0
        
        # Weight different types of Daoist content
        char_score = len(results['dao_characters']) * 0.3
        radical_score = len(results['dao_radicals']) * 0.2
        pattern_score = len(results['concept_patterns']) * 0.4
        element_score = len(results['five_elements']) * 0.1
        
        total_score = char_score + radical_score + pattern_score + element_score
        
        # Normalize by text length
        normalized_score = min(total_score / (text_length / 10), 1.0)
        
        return round(normalized_score, 3)
    
    def _extract_themes(self, results: Dict) -> List[str]:
        """Extract main philosophical themes from analysis"""
        themes = set()
        
        # From concept patterns
        for pattern in results['concept_patterns']:
            themes.update(pattern['concepts'])
        
        # From character significance
        for char_info in results['dao_characters']:
            if '道' in char_info['character']:
                themes.add('The Way')
            if '德' in char_info['character']:
                themes.add('Virtue')
            if '無' in char_info['character']:
                themes.add('Emptiness')
        
        return list(themes)
    
    def generate_report(self, analysis_results: Dict) -> str:
        """Generate a human-readable analysis report"""
        report = []
        report.append("=== DAOIST ENCODING ANALYSIS REPORT ===")
        report.append("=== 道教編碼分析報告 ===\n")
        
        report.append(f"Text analyzed: {analysis_results['text'][:100]}...")
        report.append(f"Overall Daoist encoding score: {analysis_results['overall_score']:.3f}\n")
        
        # Core Daoist characters
        if analysis_results['dao_characters']:
            report.append("Core Daoist Characters Found:")
            for char_info in analysis_results['dao_characters']:
                report.append(f"  {char_info['character']} (×{char_info['count']}): {char_info['meaning']}")
                report.append(f"    Significance: {char_info['significance']}")
        
        # Concept patterns
        if analysis_results['concept_patterns']:
            report.append("\nDaoist Conceptual Patterns:")
            for pattern in analysis_results['concept_patterns']:
                report.append(f"  {pattern['pattern_type']}: {', '.join(pattern['matches'])}")
                report.append(f"    Concepts: {', '.join(pattern['concepts'])}")
                report.append(f"    Significance: {pattern['significance']}")
        
        # Philosophical themes
        if analysis_results['philosophical_themes']:
            report.append(f"\nMain Philosophical Themes: {', '.join(analysis_results['philosophical_themes'])}")
        
        return '\n'.join(report)

def main():
    """Demonstration of the Daoist encoding analyzer"""
    analyzer = DaoistEncodingAnalyzer()
    
    # Test texts with varying degrees of Daoist content
    test_texts = [
        "上善若水，水善利萬物而不爭。道生一，一生二，二生三，三生萬物。",
        "物極必反，否極泰來。陰陽調和，天人合一。",
        "今天天氣很好，我去商店買了一些水果。",  # Non-Daoist text
        "無為而治，順其自然。以柔克剛，返璞歸真。",
        "春夏秋冬循環不息，生老病死自然規律。木火土金水五行相生。"
    ]
    
    print("Daoist Encoding Analysis Demo")
    print("道教編碼分析演示")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        results = analyzer.analyze_text(text)
        print(f"Daoist Score: {results['overall_score']:.3f}")
        
        if results['concept_patterns']:
            print("Found patterns:", end=" ")
            for pattern in results['concept_patterns']:
                print(f"{pattern['pattern_type']}({pattern['count']})", end=" ")
        print()
        
        if results['dao_characters']:
            print("Core characters:", end=" ")
            for char in results['dao_characters']:
                print(f"{char['character']}(×{char['count']})", end=" ")
        print()
    
    # Detailed analysis of most Daoist text
    print("\n" + "=" * 50)
    print("DETAILED ANALYSIS OF HIGHEST SCORING TEXT:")
    highest_score_text = max(test_texts, key=lambda t: analyzer.analyze_text(t)['overall_score'])
    detailed_results = analyzer.analyze_text(highest_score_text)
    print(analyzer.generate_report(detailed_results))

if __name__ == "__main__":
    main()