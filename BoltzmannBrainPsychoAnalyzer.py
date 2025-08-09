"""
Boltzmann Brain's Superego-Ego-Id Analyzer
博尔兹曼大脑的超我-自我-本我分析器

This module implements a psychoanalytic text analysis system that combines:
1. Freudian tripartite model (Id, Ego, Superego) for psychological content analysis
2. Boltzmann brain theory concepts for consciousness coherence and randomness detection
3. Integration with existing NLP evaluation frameworks

The Boltzmann brain concept suggests consciousness emerging from random quantum fluctuations,
while Freudian psychology divides the psyche into:
- Id: primitive impulses and desires
- Ego: reality principle and rational thought
- Superego: moral conscience and social norms

这个模块实现了一个心理分析文本分析系统，结合了：
1. 弗洛伊德三分模型（本我、自我、超我）用于心理内容分析
2. 博尔兹曼大脑理论概念用于意识连贯性和随机性检测
3. 与现有NLP评估框架的集成
"""

import re
import math
import random
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter, defaultdict

# Try to import optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import wordnet
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

try:
    from HumanExpressionEvaluator import EvaluationDimension, ExpressionContext, EvaluationResult
    HUMAN_EVALUATOR_AVAILABLE = True
except ImportError:
    # Define minimal classes if main evaluator not available
    HUMAN_EVALUATOR_AVAILABLE = False
    
    class EvaluationDimension(Enum):
        PSYCHOLOGICAL = "psychological"
        CONSCIOUSNESS = "consciousness"
        BOLTZMANN = "boltzmann"
    
    @dataclass
    class ExpressionContext:
        speaker: str = "unknown"
        emotional_state: str = "neutral"
        formality_level: str = "neutral"
    
    @dataclass
    class EvaluationResult:
        dimension: EvaluationDimension
        score: float
        confidence: float
        explanation: str
        sub_scores: Dict[str, float] = None


class PsychodynamicComponent(Enum):
    """弗洛伊德心理结构组件 (Freudian Psychic Components)"""
    ID = "id"  # 本我：原始冲动和欲望
    EGO = "ego"  # 自我：现实原则和理性思考
    SUPEREGO = "superego"  # 超我：道德良知和社会规范


class ConsciousnessCoherence(Enum):
    """意识连贯性等级 (Consciousness Coherence Levels)"""
    RANDOM = "random"  # 随机性：类似博尔兹曼大脑的随机涌现
    FRAGMENTED = "fragmented"  # 碎片化：部分连贯但有断裂
    COHERENT = "coherent"  # 连贯：结构化的意识表达
    HYPERCOHERENT = "hypercoherent"  # 超连贯：过度结构化


@dataclass
class PsychodynamicProfile:
    """心理动力学档案 (Psychodynamic Profile)"""
    id_score: float = 0.0
    ego_score: float = 0.0
    superego_score: float = 0.0
    dominant_component: PsychodynamicComponent = PsychodynamicComponent.EGO
    consciousness_coherence: ConsciousnessCoherence = ConsciousnessCoherence.COHERENT
    randomness_entropy: float = 0.0
    emotional_intensity: float = 0.0


class BoltzmannBrainPsychoAnalyzer:
    """
    博尔兹曼大脑心理分析器 (Boltzmann Brain Psychoanalytic Analyzer)
    
    Analyzes text for psychological constructs and consciousness patterns
    using Freudian psychology and Boltzmann brain theoretical frameworks.
    """
    
    def __init__(self):
        """初始化分析器 (Initialize analyzer)"""
        self._initialize_lexicons()
        self._initialize_nltk()
        
        # 博尔兹曼大脑随机性阈值 (Boltzmann brain randomness threshold)
        self.randomness_threshold = 0.7
        
        # 意识连贯性参数 (Consciousness coherence parameters)
        self.coherence_window_size = 5
        self.semantic_coherence_threshold = 0.6
    
    def _initialize_lexicons(self):
        """初始化心理分析词汇库 (Initialize psychoanalytic lexicons)"""
        
        # 本我词汇 (Id-related terms) - 冲动、欲望、即时满足
        self.id_lexicon = {
            'english': [
                'want', 'need', 'desire', 'crave', 'hunger', 'thirst', 'lust', 'impulse',
                'immediate', 'now', 'must', 'urgent', 'pleasure', 'satisfaction', 'gratification',
                'instinct', 'drive', 'compulsion', 'addiction', 'obsession', 'passion',
                'rage', 'fury', 'angry', 'hate', 'love', 'sex', 'food', 'power',
                'mine', 'take', 'grab', 'seize', 'devour', 'consume', 'indulge'
            ],
            'chinese': [
                '想要', '需要', '渴望', '欲望', '冲动', '本能', '立即', '马上', '现在',
                '必须', '快感', '满足', '享受', '贪婪', '愤怒', '仇恨', '爱', '性',
                '食物', '权力', '我的', '拿', '抓', '吞噬', '消费', '放纵',
                '饥饿', '干渴', '激情', '狂怒', '占有', '控制', '征服'
            ]
        }
        
        # 自我词汇 (Ego-related terms) - 现实、理性、平衡
        self.ego_lexicon = {
            'english': [
                'think', 'consider', 'analyze', 'evaluate', 'rational', 'logical', 'reasonable',
                'balance', 'compromise', 'negotiate', 'plan', 'strategy', 'practical', 'realistic',
                'decide', 'choose', 'weigh', 'pros', 'cons', 'consequences', 'responsibility',
                'manage', 'organize', 'control', 'regulate', 'adapt', 'adjust', 'solve',
                'understand', 'comprehend', 'clarify', 'explain', 'justify', 'defense'
            ],
            'chinese': [
                '思考', '考虑', '分析', '评估', '理性', '逻辑', '合理', '平衡',
                '妥协', '协商', '计划', '策略', '实际', '现实', '决定', '选择',
                '权衡', '后果', '责任', '管理', '组织', '控制', '调节', '适应',
                '调整', '解决', '理解', '领悟', '澄清', '解释', '证明', '防御'
            ]
        }
        
        # 超我词汇 (Superego-related terms) - 道德、规范、理想
        self.superego_lexicon = {
            'english': [
                'should', 'ought', 'must', 'moral', 'ethical', 'right', 'wrong', 'good', 'bad',
                'virtue', 'sin', 'guilt', 'shame', 'conscience', 'duty', 'obligation', 'principle',
                'ideal', 'perfect', 'pure', 'noble', 'honor', 'dignity', 'respect', 'proper',
                'appropriate', 'correct', 'decent', 'civilized', 'cultured', 'refined',
                'judge', 'criticize', 'condemn', 'approve', 'disapprove', 'praise', 'blame'
            ],
            'chinese': [
                '应该', '必须', '道德', '伦理', '对', '错', '好', '坏', '美德',
                '罪恶', '内疚', '羞耻', '良心', '义务', '责任', '原则', '理想',
                '完美', '纯洁', '高尚', '荣誉', '尊严', '尊重', '适当', '正确',
                '体面', '文明', '有教养', '精致', '判断', '批评', '谴责',
                '赞成', '反对', '表扬', '责备', '规范', '标准', '品德'
            ]
        }
        
        # 意识连贯性指标词汇 (Consciousness coherence indicators)
        self.coherence_markers = {
            'high_coherence': ['therefore', 'consequently', 'because', 'since', 'thus', 'hence',
                              '因此', '所以', '因为', '由于', '因而', '故此'],
            'low_coherence': ['suddenly', 'randomly', 'somehow', 'anyway', 'whatever', 'strange',
                             '突然', '随机', '莫名其妙', '不知怎么', '反正', '奇怪']
        }
    
    def _initialize_nltk(self):
        """初始化NLTK资源 (Initialize NLTK resources)"""
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not already present
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/wordnet')
                nltk.data.find('vader_lexicon')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('wordnet', quiet=True)
                    nltk.download('vader_lexicon', quiet=True)
                except Exception:
                    pass  # Handle download failures gracefully
            
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception:
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
    
    def analyze_psychodynamics(self, text: str, context: Optional[ExpressionContext] = None) -> PsychodynamicProfile:
        """
        分析文本的心理动力学特征 (Analyze psychodynamic characteristics of text)
        
        Args:
            text: 要分析的文本 (Text to analyze)
            context: 表达语境 (Expression context)
        
        Returns:
            PsychodynamicProfile: 心理动力学档案 (Psychodynamic profile)
        """
        if context is None:
            context = ExpressionContext()
        
        # 计算三个心理组件的分数 (Calculate scores for three psychic components)
        id_score = self._calculate_id_score(text)
        ego_score = self._calculate_ego_score(text)
        superego_score = self._calculate_superego_score(text)
        
        # 确定主导组件 (Determine dominant component)
        scores = {'id': id_score, 'ego': ego_score, 'superego': superego_score}
        dominant_component = PsychodynamicComponent(max(scores, key=scores.get))
        
        # 计算意识连贯性 (Calculate consciousness coherence)
        consciousness_coherence = self._analyze_consciousness_coherence(text)
        
        # 计算博尔兹曼大脑随机性熵 (Calculate Boltzmann brain randomness entropy)
        randomness_entropy = self._calculate_randomness_entropy(text)
        
        # 计算情感强度 (Calculate emotional intensity)
        emotional_intensity = self._calculate_emotional_intensity(text)
        
        return PsychodynamicProfile(
            id_score=id_score,
            ego_score=ego_score,
            superego_score=superego_score,
            dominant_component=dominant_component,
            consciousness_coherence=consciousness_coherence,
            randomness_entropy=randomness_entropy,
            emotional_intensity=emotional_intensity
        )
    
    def _calculate_id_score(self, text: str) -> float:
        """计算本我分数 (Calculate Id score)"""
        return self._calculate_lexicon_score(text, self.id_lexicon)
    
    def _calculate_ego_score(self, text: str) -> float:
        """计算自我分数 (Calculate Ego score)"""
        return self._calculate_lexicon_score(text, self.ego_lexicon)
    
    def _calculate_superego_score(self, text: str) -> float:
        """计算超我分数 (Calculate Superego score)"""
        return self._calculate_lexicon_score(text, self.superego_lexicon)
    
    def _calculate_lexicon_score(self, text: str, lexicon: Dict[str, List[str]]) -> float:
        """根据词汇库计算分数 (Calculate score based on lexicon)"""
        text_lower = text.lower()
        
        # 检测语言 (Detect language)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        if chinese_chars > english_words:
            relevant_lexicon = lexicon.get('chinese', [])
        else:
            relevant_lexicon = lexicon.get('english', [])
        
        if not relevant_lexicon:
            return 0.0
        
        # 计算匹配的词汇数量 (Count matching terms)
        matches = 0
        total_words = len(text_lower.split())
        
        for term in relevant_lexicon:
            matches += text_lower.count(term.lower())
        
        if total_words == 0:
            return 0.0
        
        # 归一化分数 (Normalize score)
        score = min(matches / total_words, 1.0)
        return score
    
    def _analyze_consciousness_coherence(self, text: str) -> ConsciousnessCoherence:
        """
        分析意识连贯性 (Analyze consciousness coherence)
        
        Uses Boltzmann brain theory: random consciousness vs structured thought
        """
        if not text.strip():
            return ConsciousnessCoherence.FRAGMENTED
        
        # 计算语义连贯性指标 (Calculate semantic coherence indicators)
        coherence_score = 0.0
        
        # 1. 逻辑连接词密度 (Logical connector density)
        high_coherence_markers = self.coherence_markers['high_coherence']
        low_coherence_markers = self.coherence_markers['low_coherence']
        
        text_lower = text.lower()
        high_markers = sum(text_lower.count(marker) for marker in high_coherence_markers)
        low_markers = sum(text_lower.count(marker) for marker in low_coherence_markers)
        
        total_words = len(text_lower.split())
        if total_words > 0:
            coherence_score += (high_markers - low_markers) / total_words
        
        # 2. 句子长度变异性 (Sentence length variability)
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                if len(sentences) > 1:
                    lengths = [len(s.split()) for s in sentences]
                    if NUMPY_AVAILABLE:
                        variability = np.std(lengths) / (np.mean(lengths) + 1e-10)
                    else:
                        mean_len = sum(lengths) / len(lengths)
                        variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
                        variability = (variance ** 0.5) / (mean_len + 1e-10)
                    
                    # 高变异性表示低连贯性 (High variability indicates low coherence)
                    coherence_score -= variability * 0.1
            except Exception:
                pass
        
        # 3. 重复模式 (Repetition patterns)
        words = text_lower.split()
        if len(words) > 3:
            word_freq = Counter(words)
            repetition_ratio = sum(1 for count in word_freq.values() if count > 1) / len(word_freq)
            
            # 适度重复表示连贯性 (Moderate repetition indicates coherence)
            if 0.1 <= repetition_ratio <= 0.4:
                coherence_score += 0.1
            elif repetition_ratio > 0.6:  # 过度重复可能表示意识障碍
                coherence_score -= 0.2
        
        # 分类连贯性等级 (Classify coherence level)
        if coherence_score > 0.3:
            return ConsciousnessCoherence.HYPERCOHERENT
        elif coherence_score > 0.1:
            return ConsciousnessCoherence.COHERENT
        elif coherence_score > -0.1:
            return ConsciousnessCoherence.FRAGMENTED
        else:
            return ConsciousnessCoherence.RANDOM
    
    def _calculate_randomness_entropy(self, text: str) -> float:
        """
        计算随机性熵 (Calculate randomness entropy)
        
        Inspired by Boltzmann brain theory - measures how random the text appears
        versus having structured consciousness behind it.
        """
        if not text.strip():
            return 1.0
        
        # 字符级熵 (Character-level entropy)
        char_freq = Counter(text.lower())
        total_chars = len(text)
        char_entropy = 0.0
        
        for count in char_freq.values():
            prob = count / total_chars
            if prob > 0:
                char_entropy -= prob * math.log2(prob)
        
        # 归一化到 [0, 1] (Normalize to [0, 1])
        max_entropy = math.log2(len(char_freq)) if len(char_freq) > 0 else 1.0
        normalized_entropy = char_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 词汇级模式分析 (Word-level pattern analysis)
        words = text.lower().split()
        if len(words) > 1:
            # 计算相邻词的语义相关性 (Calculate semantic relatedness of adjacent words)
            semantic_breaks = 0
            for i in range(len(words) - 1):
                # 简单的语义断裂检测 (Simple semantic break detection)
                current_word = words[i]
                next_word = words[i + 1]
                
                # 检查词性类别突变 (Check for part-of-speech category jumps)
                if self._are_semantically_unrelated(current_word, next_word):
                    semantic_breaks += 1
            
            semantic_randomness = semantic_breaks / (len(words) - 1)
        else:
            semantic_randomness = 0.0
        
        # 组合字符熵和语义随机性 (Combine character entropy and semantic randomness)
        overall_randomness = (normalized_entropy + semantic_randomness) / 2
        return min(overall_randomness, 1.0)
    
    def _are_semantically_unrelated(self, word1: str, word2: str) -> bool:
        """检查两个词是否语义无关 (Check if two words are semantically unrelated)"""
        # 简化的语义相关性检测 (Simplified semantic relatedness detection)
        
        # 功能词和内容词的基本分类 (Basic classification of function vs content words)
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                         '的', '在', '和', '或', '但是', '因为', '所以', '是', '了', '着', '过'}
        
        # 如果都是功能词，认为相关 (If both are function words, consider related)
        if word1 in function_words and word2 in function_words:
            return False
        
        # 如果一个是功能词一个不是，认为正常 (If one is function word and one isn't, normal)
        if (word1 in function_words) != (word2 in function_words):
            return False
        
        # 简单的字符相似性检测 (Simple character similarity detection)
        common_chars = set(word1) & set(word2)
        if len(common_chars) >= min(len(word1), len(word2)) * 0.5:
            return False
        
        # 如果词长度差异过大，可能语义无关 (If word length differs too much, might be unrelated)
        if abs(len(word1) - len(word2)) > max(len(word1), len(word2)) * 0.7:
            return True
        
        return False
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """计算情感强度 (Calculate emotional intensity)"""
        if self.sentiment_analyzer and NLTK_AVAILABLE:
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                # 使用复合分数的绝对值作为情感强度 (Use absolute compound score as intensity)
                return abs(scores['compound'])
            except Exception:
                pass
        
        # 降级方法：基于情感词汇 (Fallback: emotion word-based method)
        emotion_words = [
            'love', 'hate', 'angry', 'happy', 'sad', 'excited', 'furious', 'delighted',
            'disgusted', 'afraid', 'surprised', 'ashamed', 'guilty', 'proud',
            '爱', '恨', '愤怒', '高兴', '悲伤', '兴奋', '狂怒', '高兴',
            '厌恶', '害怕', '惊讶', '羞愧', '内疚', '骄傲', '激动', '痛苦'
        ]
        
        text_lower = text.lower()
        emotion_count = sum(text_lower.count(word) for word in emotion_words)
        total_words = len(text_lower.split())
        
        if total_words == 0:
            return 0.0
        
        return min(emotion_count / total_words * 2, 1.0)  # 乘以2增强强度
    
    def generate_boltzmann_profile_report(self, profile: PsychodynamicProfile, 
                                        text: str = "", detailed: bool = True) -> str:
        """
        生成博尔兹曼大脑心理档案报告 (Generate Boltzmann brain psycho profile report)
        
        Args:
            profile: 心理动力学档案 (Psychodynamic profile)
            text: 原始文本 (Original text)
            detailed: 是否生成详细报告 (Whether to generate detailed report)
        
        Returns:
            str: 格式化的分析报告 (Formatted analysis report)
        """
        report = []
        report.append("=" * 60)
        report.append("博尔兹曼大脑心理分析报告 (Boltzmann Brain Psychoanalytic Report)")
        report.append("=" * 60)
        
        if text:
            report.append(f"\n原始文本 (Original Text): {text[:100]}{'...' if len(text) > 100 else ''}")
        
        report.append("\n📊 心理结构分析 (Psychic Structure Analysis)")
        report.append("-" * 40)
        report.append(f"本我 (Id) 分数: {profile.id_score:.3f}")
        report.append(f"自我 (Ego) 分数: {profile.ego_score:.3f}")
        report.append(f"超我 (Superego) 分数: {profile.superego_score:.3f}")
        report.append(f"主导组件 (Dominant Component): {profile.dominant_component.value.upper()}")
        
        # 心理结构解释 (Psychic structure interpretation)
        if profile.dominant_component == PsychodynamicComponent.ID:
            interpretation = "本我主导 - 表达体现原始冲动和即时欲望 (Id-dominant: expression reflects primitive impulses and immediate desires)"
        elif profile.dominant_component == PsychodynamicComponent.EGO:
            interpretation = "自我主导 - 表达体现理性思考和现实原则 (Ego-dominant: expression reflects rational thinking and reality principle)"
        else:
            interpretation = "超我主导 - 表达体现道德标准和社会规范 (Superego-dominant: expression reflects moral standards and social norms)"
        
        report.append(f"解释 (Interpretation): {interpretation}")
        
        report.append("\n🧠 意识连贯性分析 (Consciousness Coherence Analysis)")
        report.append("-" * 40)
        report.append(f"连贯性等级 (Coherence Level): {profile.consciousness_coherence.value.upper()}")
        report.append(f"随机性熵 (Randomness Entropy): {profile.randomness_entropy:.3f}")
        
        # 博尔兹曼大脑解释 (Boltzmann brain interpretation)
        if profile.consciousness_coherence == ConsciousnessCoherence.RANDOM:
            boltzmann_interpretation = "类似博尔兹曼大脑的随机意识涌现 (Similar to Boltzmann brain random consciousness emergence)"
        elif profile.consciousness_coherence == ConsciousnessCoherence.FRAGMENTED:
            boltzmann_interpretation = "部分连贯但存在意识断裂 (Partially coherent but with consciousness breaks)"
        elif profile.consciousness_coherence == ConsciousnessCoherence.COHERENT:
            boltzmann_interpretation = "结构化的意识表达 (Structured consciousness expression)"
        else:
            boltzmann_interpretation = "过度结构化可能表示强迫性思维 (Over-structuring may indicate compulsive thinking)"
        
        report.append(f"博尔兹曼解释 (Boltzmann Interpretation): {boltzmann_interpretation}")
        
        report.append(f"\n💫 情感强度 (Emotional Intensity): {profile.emotional_intensity:.3f}")
        
        if detailed and (profile.randomness_entropy > 0.7 or 
                        profile.consciousness_coherence == ConsciousnessCoherence.RANDOM):
            report.append("\n⚠️  博尔兹曼大脑警报 (Boltzmann Brain Alert)")
            report.append("-" * 40)
            report.append("高随机性熵检测到可能的:")
            report.append("• 意识流式表达 (Stream-of-consciousness expression)")
            report.append("• 随机联想模式 (Random association patterns)")
            report.append("• 可能的意识状态改变 (Possible altered consciousness state)")
        
        if detailed:
            report.append("\n📈 详细分析建议 (Detailed Analysis Recommendations)")
            report.append("-" * 40)
            
            if profile.id_score > 0.3:
                report.append("• 高本我活动：建议关注冲动控制和情绪调节")
            if profile.superego_score > 0.3:
                report.append("• 高超我活动：建议关注心理压力和完美主义倾向")
            if profile.consciousness_coherence == ConsciousnessCoherence.FRAGMENTED:
                report.append("• 意识碎片化：建议关注注意力和思维组织能力")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def comprehensive_evaluation(self, text: str, 
                               context: Optional[ExpressionContext] = None) -> EvaluationResult:
        """
        综合评估 (Comprehensive evaluation)
        
        Returns evaluation result compatible with HumanExpressionEvaluator framework
        """
        profile = self.analyze_psychodynamics(text, context)
        
        # 计算综合分数 (Calculate comprehensive score)
        # 平衡心理健康指标 (Balance psychological health indicators)
        balance_score = 1.0 - abs(profile.id_score - profile.ego_score) - abs(profile.ego_score - profile.superego_score)
        coherence_score = 1.0 if profile.consciousness_coherence == ConsciousnessCoherence.COHERENT else 0.5
        randomness_score = 1.0 - profile.randomness_entropy
        
        overall_score = (balance_score + coherence_score + randomness_score) / 3
        overall_score = max(0.0, min(1.0, overall_score))  # 限制在 [0, 1]
        
        # 计算信心度 (Calculate confidence)
        confidence = 0.8 if text and len(text.split()) > 5 else 0.5
        
        explanation = f"心理平衡: {balance_score:.2f}, 意识连贯: {coherence_score:.2f}, 随机性控制: {randomness_score:.2f}"
        
        return EvaluationResult(
            dimension=EvaluationDimension.PSYCHOLOGICAL if HUMAN_EVALUATOR_AVAILABLE 
                      else EvaluationDimension.BOLTZMANN,
            score=overall_score,
            confidence=confidence,
            explanation=explanation,
            sub_scores={
                'psychic_balance': balance_score,
                'consciousness_coherence': coherence_score,
                'randomness_control': randomness_score,
                'id_score': profile.id_score,
                'ego_score': profile.ego_score,
                'superego_score': profile.superego_score,
                'randomness_entropy': profile.randomness_entropy,
                'emotional_intensity': profile.emotional_intensity
            }
        )


# 使用示例 (Usage Example)
if __name__ == "__main__":
    analyzer = BoltzmannBrainPsychoAnalyzer()
    
    # 测试不同类型的文本 (Test different types of text)
    test_texts = [
        "I want it now! Give me everything I desire immediately!",
        "我需要仔细考虑这个决定的所有后果，权衡利弊。",
        "We should always do what is morally right and proper in society.",
        "purple elephant dancing quantum mechanics randomly fluctuating consciousness emerging from void suddenly meaningful patterns dissolve into chaos beautiful symmetry",
        "我应该更加努力工作，做一个有道德的好人，不能让父母失望。"
    ]
    
    print("博尔兹曼大脑心理分析器测试 (Boltzmann Brain Psychoanalyzer Test)")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i} (Test {i}):")
        profile = analyzer.analyze_psychodynamics(text)
        report = analyzer.generate_boltzmann_profile_report(profile, text, detailed=False)
        print(report)
        
        # 获取评估结果 (Get evaluation result)
        evaluation = analyzer.comprehensive_evaluation(text)
        print(f"\n综合评估分数 (Comprehensive Score): {evaluation.score:.3f}")
        print(f"信心度 (Confidence): {evaluation.confidence:.3f}")
        print("-" * 70)