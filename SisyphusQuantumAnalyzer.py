"""
Sisyphus Quantum Analyzer - Detecting Circular Logic and Breakthrough Insights

This module implements the metaphorical concept from issue #110:
- Sisyphus boulder pushing: Detecting repetitive, circular, or futile patterns
- Quantum tunneling: Finding breakthrough insights that transcend traditional barriers

The analyzer identifies:
1. Circular reasoning patterns (Sisyphus aspect)
2. Semantic barriers and breakthroughs (Quantum tunneling aspect)
3. Paradoxical statements that create new understanding
4. Novel conceptual connections that bypass logical constraints
"""

import re
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from itertools import combinations


@dataclass
class SisyphusPattern:
    """表示薛西弗斯式的重複模式 (Represents Sisyphus-like repetitive patterns)"""
    pattern_type: str  # 'circular_logic', 'repetitive_argument', 'futile_reasoning'
    text_segments: List[str]
    strength: float  # 0.0 to 1.0
    explanation: str


@dataclass
class QuantumTunnelingMoment:
    """表示量子隧穿式的突破時刻 (Represents quantum tunneling breakthrough moments)"""
    moment_type: str  # 'paradox_resolution', 'semantic_leap', 'conceptual_breakthrough'
    context: str
    insight: str
    barrier_transcended: str
    probability: float  # 0.0 to 1.0


class CircularLogicDetector:
    """檢測循環邏輯的檢測器 (Circular logic detector)"""
    
    def __init__(self):
        # 循環邏輯指示詞 (Circular logic indicators)
        self.circular_indicators = {
            'self_reference': ['因為它是', '因為就是', 'because it is', 'because that\'s what'],
            'tautology': ['當然是', '顯然是', 'of course', 'obviously', 'naturally'],
            'question_begging': ['基於前提', '假設', 'assuming that', 'given that', 'presupposing']
        }
        
        # 重複模式 (Repetition patterns)
        self.repetition_patterns = [
            r'(.{10,})\1',  # Direct repetition
            r'(\w+)\s+\w*\s+\1',  # Word repetition with gap
            r'(if .+, then .+)\s*\1'  # Conditional repetition
        ]
    
    def detect_circular_reasoning(self, text: str) -> List[SisyphusPattern]:
        """檢測循環推理 (Detect circular reasoning)"""
        patterns = []
        
        # 檢測自指模式 (Detect self-reference patterns)
        self_ref_patterns = self._find_self_reference(text)
        patterns.extend(self_ref_patterns)
        
        # 檢測重言式 (Detect tautologies)
        tautology_patterns = self._find_tautologies(text)
        patterns.extend(tautology_patterns)
        
        # 檢測乞題論證 (Detect question begging)
        begging_patterns = self._find_question_begging(text)
        patterns.extend(begging_patterns)
        
        # 檢測基於重複的循環模式 (Detect repetition-based circular patterns)
        repetition_patterns = self._find_repetitive_patterns(text)
        patterns.extend(repetition_patterns)
        
        return patterns
    
    def _find_self_reference(self, text: str) -> List[SisyphusPattern]:
        """尋找自指模式 (Find self-reference patterns)"""
        patterns = []
        sentences = re.split(r'[.!?。！？]', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 檢查自指表達 (Check self-referential expressions)
            for indicator_type, indicators in self.circular_indicators.items():
                for indicator in indicators:
                    if indicator.lower() in sentence.lower():
                        # 分析句子結構看是否構成循環 (Analyze sentence structure for circularity)
                        if self._is_circular_structure(sentence):
                            patterns.append(SisyphusPattern(
                                pattern_type='circular_logic',
                                text_segments=[sentence.strip()],
                                strength=0.7,
                                explanation=f"檢測到自指循環邏輯: {indicator}"
                            ))
        
        return patterns
    
    def _find_tautologies(self, text: str) -> List[SisyphusPattern]:
        """尋找重言式 (Find tautologies)"""
        patterns = []
        sentences = re.split(r'[.!?。！？]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 檢查 A is A 模式 (Check A is A patterns)
            words = sentence.lower().split()
            for i in range(len(words) - 2):
                if words[i] == words[i + 2] and words[i + 1] in ['is', 'are', '是', '就是']:
                    patterns.append(SisyphusPattern(
                        pattern_type='tautology',
                        text_segments=[sentence],
                        strength=0.9,
                        explanation=f"檢測到重言式結構: {words[i]} {words[i+1]} {words[i+2]}"
                    ))
        
        return patterns
    
    def _find_question_begging(self, text: str) -> List[SisyphusPattern]:
        """尋找乞題論證 (Find question begging arguments)"""
        patterns = []
        
        # 檢查前提和結論的相似性 (Check similarity between premise and conclusion)
        paragraphs = text.split('\n')
        for para in paragraphs:
            if len(para.strip()) < 20:  # Skip short paragraphs
                continue
                
            sentences = re.split(r'[.!?。！？]', para)
            if len(sentences) < 2:
                continue
            
            # 比較第一句和最後一句的相似性 (Compare first and last sentences)
            first_sentence = sentences[0].strip()
            last_sentence = sentences[-1].strip()
            
            if first_sentence and last_sentence:
                similarity = self._calculate_semantic_similarity(first_sentence, last_sentence)
                if similarity > 0.7:
                    patterns.append(SisyphusPattern(
                        pattern_type='question_begging',
                        text_segments=[first_sentence, last_sentence],
                        strength=similarity,
                        explanation=f"檢測到乞題論證，前提與結論相似度: {similarity:.2f}"
                    ))
        
        return patterns
    
    def _find_repetitive_patterns(self, text: str) -> List[SisyphusPattern]:
        """尋找重複模式 (Find repetitive patterns)"""
        patterns = []
        
        # 檢查同一概念的重複 (Check for repetition of same concepts)
        key_words = ['正確', '對的', '理論', '證明', 'correct', 'right', 'theory', 'prove']
        
        for word in key_words:
            count = text.lower().count(word.lower())
            if count >= 3:  # 如果某個關鍵詞出現3次以上
                # 檢查這些重複是否構成循環論證
                sentences_with_word = []
                for sentence in re.split(r'[.!?。！？]', text):
                    if word.lower() in sentence.lower():
                        sentences_with_word.append(sentence.strip())
                
                if len(sentences_with_word) >= 2:
                    patterns.append(SisyphusPattern(
                        pattern_type='repetitive_argument',
                        text_segments=sentences_with_word,
                        strength=min(count / 5.0, 1.0),  # 歸一化強度
                        explanation=f"檢測到重複論證，關鍵詞 '{word}' 出現 {count} 次"
                    ))
        
        return patterns
    
    def _is_circular_structure(self, sentence: str) -> bool:
        """判斷句子是否具有循環結構 (Determine if sentence has circular structure)"""
        # 檢查是否有關鍵詞表明循環邏輯
        circular_keywords = ['because it is', '因為它是', '因為它就是', 'that\'s what it is', '就是這樣', '就是正確的']
        sentence_lower = sentence.lower()
        
        # 檢查直接的循環表達
        if any(keyword in sentence_lower for keyword in circular_keywords):
            return True
        
        # 檢查 "A because A" 模式
        words = sentence_lower.split()
        for i in range(len(words) - 3):
            if (words[i] in ['正確', 'correct', 'right'] and 
                words[i+1] in ['因為', 'because'] and 
                words[i+3] in ['正確', 'correct', 'right']):
                return True
        
        # 檢查重複的核心概念
        key_concepts = ['正確', '對的', '理論', 'correct', 'right', 'theory']
        concept_count = sum(sentence_lower.count(concept) for concept in key_concepts)
        return concept_count >= 3
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """計算語義相似度 (Calculate semantic similarity)"""
        # 簡化版：基於詞彙重疊
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class QuantumTunnelingDetector:
    """檢測量子隧穿式突破的檢測器 (Quantum tunneling breakthrough detector)"""
    
    def __init__(self):
        # 悖論指示詞 (Paradox indicators)
        self.paradox_indicators = [
            '但是', '然而', '儘管', '雖然', 'but', 'however', 'although', 'despite',
            '看似矛盾', '看似不可能', 'seemingly impossible', 'apparently contradictory'
        ]
        
        # 突破性詞彙 (Breakthrough vocabulary)
        self.breakthrough_markers = [
            '突然意識到', '頓悟', '原來', '實際上', 'suddenly realized', 'epiphany',
            'actually', 'in fact', 'breakthrough', 'revelation'
        ]
        
        # 超越性概念 (Transcendent concepts)
        self.transcendent_concepts = [
            '超越', '昇華', '轉化', 'transcend', 'transform', 'elevate',
            '突破限制', 'break through', 'go beyond'
        ]
    
    def detect_quantum_moments(self, text: str) -> List[QuantumTunnelingMoment]:
        """檢測量子隧穿時刻 (Detect quantum tunneling moments)"""
        moments = []
        
        # 檢測悖論解析 (Detect paradox resolution)
        paradox_moments = self._find_paradox_resolution(text)
        moments.extend(paradox_moments)
        
        # 檢測語義跳躍 (Detect semantic leaps)
        semantic_leaps = self._find_semantic_leaps(text)
        moments.extend(semantic_leaps)
        
        # 檢測概念突破 (Detect conceptual breakthroughs)
        conceptual_breakthroughs = self._find_conceptual_breakthroughs(text)
        moments.extend(conceptual_breakthroughs)
        
        return moments
    
    def _find_paradox_resolution(self, text: str) -> List[QuantumTunnelingMoment]:
        """尋找悖論解析時刻 (Find paradox resolution moments)"""
        moments = []
        sentences = re.split(r'[.!?。！？]', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 檢查是否包含悖論指示詞 (Check for paradox indicators)
            has_paradox = any(indicator in sentence.lower() for indicator in self.paradox_indicators)
            
            if has_paradox and i < len(sentences) - 1:
                next_sentence = sentences[i + 1].strip()
                
                # 檢查下一句是否包含解決方案 (Check if next sentence contains resolution)
                has_resolution = any(marker in next_sentence.lower() for marker in self.breakthrough_markers)
                
                if has_resolution:
                    moments.append(QuantumTunnelingMoment(
                        moment_type='paradox_resolution',
                        context=sentence,
                        insight=next_sentence,
                        barrier_transcended='邏輯矛盾',
                        probability=0.8
                    ))
        
        return moments
    
    def _find_semantic_leaps(self, text: str) -> List[QuantumTunnelingMoment]:
        """尋找語義跳躍 (Find semantic leaps)"""
        moments = []
        sentences = re.split(r'[.!?。！？]', text)
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i].strip()
            next_sentence = sentences[i + 1].strip()
            
            if not current_sentence or not next_sentence:
                continue
            
            # 計算語義距離 (Calculate semantic distance)
            semantic_distance = self._calculate_semantic_distance(current_sentence, next_sentence)
            
            # 如果語義距離大，但有連接詞表明邏輯連續性
            if semantic_distance > 0.7:
                has_connector = any(word in next_sentence.lower() for word in 
                                  ['所以', '因此', '因而', 'therefore', 'thus', 'hence'])
                
                if has_connector:
                    moments.append(QuantumTunnelingMoment(
                        moment_type='semantic_leap',
                        context=current_sentence,
                        insight=next_sentence,
                        barrier_transcended='語義鴻溝',
                        probability=semantic_distance
                    ))
        
        return moments
    
    def _find_conceptual_breakthroughs(self, text: str) -> List[QuantumTunnelingMoment]:
        """尋找概念突破 (Find conceptual breakthroughs)"""
        moments = []
        sentences = re.split(r'[.!?。！？]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 檢查是否包含超越性概念 (Check for transcendent concepts)
            transcendent_score = sum(1 for concept in self.transcendent_concepts 
                                   if concept in sentence.lower())
            
            if transcendent_score > 0:
                # 分析是否真的表達了概念突破 (Analyze if it truly expresses conceptual breakthrough)
                breakthrough_probability = min(transcendent_score / len(self.transcendent_concepts), 1.0)
                
                moments.append(QuantumTunnelingMoment(
                    moment_type='conceptual_breakthrough',
                    context=sentence,
                    insight='概念層次的提升或轉化',
                    barrier_transcended='傳統思維框架',
                    probability=breakthrough_probability
                ))
        
        return moments
    
    def _calculate_semantic_distance(self, text1: str, text2: str) -> float:
        """計算語義距離 (Calculate semantic distance)"""
        # 基於詞彙差異計算語義距離
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # 距離 = 1 - 相似度
        similarity = intersection / union if union > 0 else 0.0
        return 1.0 - similarity


class SisyphusQuantumAnalyzer:
    """
    薛西弗斯量子分析器 (Sisyphus Quantum Analyzer)
    
    整合薛西弗斯推石和量子隧穿的概念，分析文本中的：
    1. 循環邏輯和重複模式（薛西弗斯面向）
    2. 突破性洞察和語義跳躍（量子隧穿面向）
    
    Integrates the concepts of Sisyphus boulder pushing and quantum tunneling to analyze:
    1. Circular logic and repetitive patterns (Sisyphus aspect)
    2. Breakthrough insights and semantic leaps (Quantum tunneling aspect)
    """
    
    def __init__(self):
        self.circular_detector = CircularLogicDetector()
        self.quantum_detector = QuantumTunnelingDetector()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        執行完整的薛西弗斯量子分析 (Perform complete Sisyphus Quantum analysis)
        
        Args:
            text: 要分析的文本 (Text to analyze)
        
        Returns:
            分析結果字典 (Analysis results dictionary)
        """
        # 檢測薛西弗斯模式 (Detect Sisyphus patterns)
        sisyphus_patterns = self.circular_detector.detect_circular_reasoning(text)
        
        # 檢測量子隧穿時刻 (Detect quantum tunneling moments)
        quantum_moments = self.quantum_detector.detect_quantum_moments(text)
        
        # 計算整體評分 (Calculate overall scores)
        sisyphus_score = self._calculate_sisyphus_score(sisyphus_patterns)
        quantum_score = self._calculate_quantum_score(quantum_moments)
        
        # 生成洞察 (Generate insights)
        insights = self._generate_insights(sisyphus_patterns, quantum_moments, text)
        
        return {
            'sisyphus_analysis': {
                'patterns': sisyphus_patterns,
                'score': sisyphus_score,
                'interpretation': self._interpret_sisyphus_score(sisyphus_score)
            },
            'quantum_analysis': {
                'moments': quantum_moments,
                'score': quantum_score,
                'interpretation': self._interpret_quantum_score(quantum_score)
            },
            'integrated_insights': insights,
            'overall_assessment': self._generate_overall_assessment(sisyphus_score, quantum_score)
        }
    
    def _calculate_sisyphus_score(self, patterns: List[SisyphusPattern]) -> float:
        """計算薛西弗斯分數 (Calculate Sisyphus score)"""
        if not patterns:
            return 0.0
        
        # 基於模式強度和數量 (Based on pattern strength and count)
        total_strength = sum(pattern.strength for pattern in patterns)
        pattern_count = len(patterns)
        
        # 歸一化分數 (Normalize score)
        raw_score = total_strength / max(pattern_count, 1)
        return min(raw_score, 1.0)
    
    def _calculate_quantum_score(self, moments: List[QuantumTunnelingMoment]) -> float:
        """計算量子隧穿分數 (Calculate quantum tunneling score)"""
        if not moments:
            return 0.0
        
        # 基於突破概率和數量 (Based on breakthrough probability and count)
        total_probability = sum(moment.probability for moment in moments)
        moment_count = len(moments)
        
        # 歸一化分數 (Normalize score)
        raw_score = total_probability / max(moment_count, 1)
        return min(raw_score, 1.0)
    
    def _interpret_sisyphus_score(self, score: float) -> str:
        """解釋薛西弗斯分數 (Interpret Sisyphus score)"""
        if score > 0.7:
            return "文本顯示強烈的循環邏輯特徵，存在明顯的重複論證或自指問題。"
        elif score > 0.4:
            return "文本包含一些循環邏輯元素，但不完全是重複性的。"
        elif score > 0.1:
            return "文本存在輕微的循環特徵，整體邏輯較為清晰。"
        else:
            return "文本邏輯清晰，沒有明顯的循環推理問題。"
    
    def _interpret_quantum_score(self, score: float) -> str:
        """解釋量子隧穿分數 (Interpret quantum tunneling score)"""
        if score > 0.7:
            return "文本包含顯著的突破性洞察，能夠超越傳統邏輯框架。"
        elif score > 0.4:
            return "文本展現一定的創新思維，包含一些概念跳躍。"
        elif score > 0.1:
            return "文本有輕微的創新元素，但主要遵循傳統邏輯。"
        else:
            return "文本主要遵循常規邏輯，缺乏突破性思維。"
    
    def _generate_insights(self, sisyphus_patterns: List[SisyphusPattern], 
                          quantum_moments: List[QuantumTunnelingMoment], text: str) -> List[str]:
        """生成整合洞察 (Generate integrated insights)"""
        insights = []
        
        # 基於薛西弗斯和量子分析的組合 (Based on combination of Sisyphus and quantum analysis)
        if sisyphus_patterns and quantum_moments:
            insights.append("文本同時包含循環邏輯和突破性思維，可能反映複雜的思維過程。")
            insights.append("作者可能在傳統思維框架中掙扎，但也嘗試尋找新的理解方式。")
        
        elif sisyphus_patterns and not quantum_moments:
            insights.append("文本主要表現為重複性思維，可能需要尋找新的視角來突破。")
            
        elif not sisyphus_patterns and quantum_moments:
            insights.append("文本展現創新思維，成功避免了循環論證的陷阱。")
            
        else:
            insights.append("文本邏輯清晰且具有一定創新性，是良好的表達範例。")
        
        # 基於特定模式的洞察 (Insights based on specific patterns)
        tautology_count = len([p for p in sisyphus_patterns if p.pattern_type == 'tautology'])
        if tautology_count > 0:
            insights.append(f"檢測到 {tautology_count} 個重言式，可能需要更具體的論證。")
        
        breakthrough_count = len([m for m in quantum_moments if m.moment_type == 'conceptual_breakthrough'])
        if breakthrough_count > 0:
            insights.append(f"檢測到 {breakthrough_count} 個概念突破時刻，展現了創新思維。")
        
        return insights
    
    def _generate_overall_assessment(self, sisyphus_score: float, quantum_score: float) -> str:
        """生成整體評估 (Generate overall assessment)"""
        ratio = quantum_score / max(sisyphus_score, 0.01)  # 避免除零
        
        if ratio > 3:
            return "高度創新：文本主要展現突破性思維，很少陷入循環論證。"
        elif ratio > 1.5:
            return "創新導向：文本在保持邏輯性的同時展現創新思維。"
        elif ratio > 0.67:
            return "平衡型：文本在傳統邏輯和創新思維之間保持平衡。"
        elif ratio > 0.33:
            return "傳統導向：文本較多依賴傳統邏輯，創新元素較少。"
        else:
            return "循環傾向：文本可能陷入重複論證，需要注入更多創新思維。"


def main():
    """示例用法 (Example usage)"""
    analyzer = SisyphusQuantumAnalyzer()
    
    # 測試案例 (Test cases)
    test_texts = [
        # 循環邏輯例子 (Circular logic example)
        """
        這個理論是正確的，因為它就是正確的。如果我們假設這個理論是對的，
        那麼我們可以證明它確實是對的。顯然，這個理論必須是正確的，
        因為如果不正確的話，它就不會是正確的了。
        """,
        
        # 量子隧穿例子 (Quantum tunneling example)
        """
        看起來這個問題無解，傳統方法都失敗了。但是，突然意識到
        如果我們完全改變視角，從問題的反面思考，原來困難可以變成機會。
        這種看似矛盾的轉化，實際上超越了原有的思維限制。
        """,
        
        # 混合例子 (Mixed example)
        """
        我們總是在重複同樣的錯誤，因為我們就是這樣的人。
        然而，也許問題不在於我們重複錯誤，而在於我們對錯誤的定義。
        如果重新定義什麼是錯誤，我們可能會發現新的可能性。
        """
    ]
    
    print("=== 薛西弗斯量子分析器示例 (Sisyphus Quantum Analyzer Examples) ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"測試案例 {i} (Test Case {i}):")
        print("=" * 60)
        print("文本內容 (Text Content):")
        print(text.strip())
        print("\n分析結果 (Analysis Results):")
        print("-" * 40)
        
        result = analyzer.analyze(text)
        
        # 薛西弗斯分析 (Sisyphus analysis)
        sisyphus = result['sisyphus_analysis']
        print(f"薛西弗斯分數 (Sisyphus Score): {sisyphus['score']:.2f}")
        print(f"循環邏輯解釋: {sisyphus['interpretation']}")
        
        if sisyphus['patterns']:
            print("檢測到的模式:")
            for pattern in sisyphus['patterns']:
                print(f"  - {pattern.pattern_type}: {pattern.explanation} (強度: {pattern.strength:.2f})")
        
        print()
        
        # 量子隧穿分析 (Quantum tunneling analysis)
        quantum = result['quantum_analysis']
        print(f"量子隧穿分數 (Quantum Score): {quantum['score']:.2f}")
        print(f"突破性思維解釋: {quantum['interpretation']}")
        
        if quantum['moments']:
            print("檢測到的突破時刻:")
            for moment in quantum['moments']:
                print(f"  - {moment.moment_type}: 超越了 {moment.barrier_transcended} (概率: {moment.probability:.2f})")
        
        print()
        
        # 整合洞察 (Integrated insights)
        print("整合洞察 (Integrated Insights):")
        for insight in result['integrated_insights']:
            print(f"  • {insight}")
        
        print(f"\n整體評估: {result['overall_assessment']}")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()