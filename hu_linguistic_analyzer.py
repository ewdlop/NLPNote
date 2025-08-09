"""
胡言分析器 (Hu Linguistic Analyzer)
A specialized NLP tool for analyzing the usage patterns and semantic contexts of the Chinese character "胡" (Hu).

This module implements "Because 胡 says so!" - a culturally-aware linguistic analysis system that:
1. Detects different semantic usages of "胡"
2. Provides cultural and historical context
3. Analyzes sentiment and intent in multilingual contexts
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HuUsageType(Enum):
    """胡字用法类型 (Types of Hu Character Usage)"""
    SURNAME = "surname"              # 胡 as family name
    NONSENSE = "nonsense"           # 胡言乱语 (nonsensical talk)
    ARBITRARY = "arbitrary"         # 胡乱、胡来 (arbitrary/reckless)
    FOREIGN = "foreign"             # Historical: non-Chinese peoples
    MUSICAL = "musical"             # 胡琴 (traditional string instrument)
    QUESTIONING = "questioning"     # 胡思乱想 (wild thoughts/imaginings)
    UNKNOWN = "unknown"


@dataclass
class HuAnalysisResult:
    """胡字分析结果 (Hu Analysis Result)"""
    text: str
    usage_type: HuUsageType
    confidence: float
    context: str
    cultural_notes: str
    examples: List[str]
    sentiment: str = "neutral"


class HuLinguisticAnalyzer:
    """
    胡言分析器主类 (Main Hu Linguistic Analyzer Class)
    
    Because 胡 says so! - This analyzer understands the cultural and linguistic 
    significance of "胡" in various contexts.
    """
    
    def __init__(self):
        self.hu_patterns = self._load_hu_patterns()
        self.cultural_contexts = self._load_cultural_contexts()
        
    def _load_hu_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for different usages of 胡"""
        return {
            HuUsageType.SURNAME.value: [
                r'胡[一-龯]{1,3}(?![说扯言琴人笛笳])',  # 胡 + Chinese name characters, not followed by common non-name words
                r'胡先生|胡女士|胡老师|胡教授|胡博士',
                r'老胡(?![琴说])|小胡(?![琴说])',  # Avoid 小胡琴, 老胡说
                r'胡同学|胡医生|胡总|胡经理',
                r'胡主任|胡校长|胡董事长'
            ],
            HuUsageType.NONSENSE.value: [
                r'胡言乱语|胡说八道|胡说|胡扯',
                r'胡诌|胡编乱造|胡编',
                r'胡话|别胡.*[说扯诌编]',
                r'nonsense.*胡|胡.*nonsense'
            ],
            HuUsageType.ARBITRARY.value: [
                r'胡乱|胡来|胡作非为',
                r'胡闹|胡搞|胡弄',
                r'胡作胡为|胡干|别胡来'
            ],
            HuUsageType.FOREIGN.value: [
                r'胡人|胡族|胡商|胡夷',
                r'胡服|胡乐|胡舞|胡风',
                r'胡俗|胡地|胡国'
            ],
            HuUsageType.MUSICAL.value: [
                r'二胡|京胡|高胡|中胡|低胡|板胡',  # Specific instruments
                r'胡琴(?!声)',  # 胡琴 but not just 胡琴声 alone
                r'拉.*胡琴|胡琴.*[演奏弹拉]',  # Playing actions
                r'胡笛|胡笳'
            ],
            HuUsageType.QUESTIONING.value: [
                r'胡思乱想|胡猜|胡疑',
                r'胡想|胡念|胡测'
            ]
        }
    
    def _load_cultural_contexts(self) -> Dict[str, Dict[str, str]]:
        """Load cultural and historical contexts for different hu usages"""
        return {
            HuUsageType.SURNAME.value: {
                "origin": "胡氏是中国大姓之一，源于妫姓",
                "famous_people": "胡适、胡锦涛、胡歌等",
                "cultural_notes": "Common Chinese surname with rich historical heritage"
            },
            HuUsageType.NONSENSE.value: {
                "origin": "胡言乱语源于胡人说话听不懂，后引申为胡说",
                "usage": "Used to describe meaningless or false speech",
                "cultural_notes": "Often used to dismiss someone's argument as nonsensical"
            },
            HuUsageType.ARBITRARY.value: {
                "origin": "胡乱表示随意、不规范的行为",
                "usage": "Describes reckless or careless actions",
                "cultural_notes": "Implies lack of proper consideration or method"
            },
            HuUsageType.FOREIGN.value: {
                "origin": "古代中原人称北方和西方民族为胡人",
                "historical": "Tang Dynasty had extensive cultural exchange with 胡人",
                "cultural_notes": "Historical term, now considered culturally sensitive"
            },
            HuUsageType.MUSICAL.value: {
                "origin": "胡琴等乐器传入中原，保留了胡字命名",
                "instruments": "二胡是最著名的胡琴类乐器",
                "cultural_notes": "Important part of Chinese traditional music"
            }
        }
    
    def analyze_hu_usage(self, text: str) -> List[HuAnalysisResult]:
        """
        分析文本中胡字的用法 (Analyze Hu character usage in text)
        
        Because 胡 says so! This method provides comprehensive analysis.
        """
        results = []
        
        # Find all occurrences of 胡 with better context
        hu_positions = []
        for i, char in enumerate(text):
            if char == '胡':
                # Get context around the character
                start = max(0, i - 10)
                end = min(len(text), i + 10)
                context = text[start:end]
                hu_positions.append((i, context, start, end))
        
        if not hu_positions:
            return results
            
        for pos, context, start, end in hu_positions:
            usage_type = self._determine_usage_type(context, text)
            confidence = self._calculate_confidence(context, usage_type)
            
            result = HuAnalysisResult(
                text=context.strip(),
                usage_type=usage_type,
                confidence=confidence,
                context=self._extract_context(text, start, end),
                cultural_notes=self._get_cultural_notes(usage_type),
                examples=self._get_examples(usage_type),
                sentiment=self._analyze_sentiment(context, usage_type)
            )
            
            results.append(result)
            
        return results
    
    def _determine_usage_type(self, context: str, full_text: str) -> HuUsageType:
        """Determine the type of Hu usage based on context"""
        # First, try exact string matching for common patterns
        context_clean = context.strip()
        
        # Direct pattern matching in context
        if '胡说八道' in context_clean or '胡说' in context_clean or '胡扯' in context_clean:
            return HuUsageType.NONSENSE
        elif '二胡' in context_clean or '胡琴' in context_clean:
            return HuUsageType.MUSICAL
        elif '胡人' in context_clean:
            return HuUsageType.FOREIGN
        elif '胡同' in context_clean:
            return HuUsageType.SURNAME  # Treat as place name/neutral
        
        # Check regex patterns in order of specificity
        pattern_checks = [
            (HuUsageType.NONSENSE, [r'胡言乱语', r'胡编', r'胡诌', r'别胡']),
            (HuUsageType.MUSICAL, [r'京胡', r'板胡', r'高胡', r'中胡', r'低胡', r'胡笛', r'胡笳']),
            (HuUsageType.FOREIGN, [r'胡族', r'胡商', r'胡服', r'胡乐', r'胡舞']),
            (HuUsageType.ARBITRARY, [r'胡乱', r'胡来', r'胡闹', r'胡搞']),
            (HuUsageType.QUESTIONING, [r'胡思乱想', r'胡猜', r'胡想']),
            (HuUsageType.SURNAME, [r'胡先生', r'胡女士', r'胡老师', r'胡教授', r'老胡', r'小胡'])
        ]
        
        for usage_type, patterns in pattern_checks:
            for pattern in patterns:
                if re.search(pattern, context_clean):
                    return usage_type
        
        # Check if it looks like a Chinese name (胡 + 1-3 Chinese characters)
        name_match = re.search(r'胡[一-龯]{1,3}', context_clean)
        if name_match:
            # Check if it's likely a person's name vs other usage
            name_part = name_match.group(0)
            if len(name_part) >= 2:  # 胡 + at least one character
                return HuUsageType.SURNAME
            
        # Default fallback
        return HuUsageType.UNKNOWN
    
    def _calculate_confidence(self, context: str, usage_type: HuUsageType) -> float:
        """Calculate confidence score for the usage type classification"""
        if usage_type == HuUsageType.UNKNOWN:
            return 0.3
            
        # Check how many patterns match
        patterns = self.hu_patterns.get(usage_type.value, [])
        matches = sum(1 for pattern in patterns if re.search(pattern, context, re.IGNORECASE))
        
        base_confidence = min(0.5 + (matches * 0.3), 1.0)
        
        # Adjust based on context clarity
        if len(context.strip()) < 5:
            base_confidence *= 0.8  # Less confident for very short contexts
        elif len(context.strip()) > 20:
            base_confidence *= 1.1  # More confident for longer contexts
            
        return min(base_confidence, 1.0)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract surrounding context for better understanding"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _get_cultural_notes(self, usage_type: HuUsageType) -> str:
        """Get cultural notes for the usage type"""
        context = self.cultural_contexts.get(usage_type.value, {})
        return context.get("cultural_notes", "No specific cultural notes available")
    
    def _get_examples(self, usage_type: HuUsageType) -> List[str]:
        """Get example phrases for the usage type"""
        examples = {
            HuUsageType.SURNAME: ["胡先生来了", "胡老师很厉害", "老胡今天请客"],
            HuUsageType.NONSENSE: ["不要胡说八道", "这完全是胡言乱语", "别胡扯了"],
            HuUsageType.ARBITRARY: ["别胡来", "胡乱搞的", "胡作非为"],
            HuUsageType.FOREIGN: ["胡人入中原", "胡商云集", "胡服骑射"],
            HuUsageType.MUSICAL: ["拉二胡", "胡琴声悠扬", "学胡琴"],
            HuUsageType.QUESTIONING: ["胡思乱想", "别胡猜了", "胡想什么呢"]
        }
        return examples.get(usage_type, ["No examples available"])
    
    def _analyze_sentiment(self, context: str, usage_type: HuUsageType) -> str:
        """Analyze sentiment of the Hu usage"""
        negative_indicators = ["不要", "别", "禁止", "错误", "问题"]
        positive_indicators = ["好", "棒", "优秀", "厉害", "赞"]
        
        if usage_type == HuUsageType.NONSENSE:
            return "negative"
        elif usage_type == HuUsageType.MUSICAL:
            return "positive"
        elif usage_type == HuUsageType.SURNAME:
            # Check surrounding context
            if any(pos in context for pos in positive_indicators):
                return "positive"
            elif any(neg in context for neg in negative_indicators):
                return "negative"
            else:
                return "neutral"
        else:
            return "neutral"
    
    def generate_hu_report(self, text: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report on Hu usage in the text
        
        Because 胡 says so! - This provides the authoritative analysis.
        """
        results = self.analyze_hu_usage(text)
        
        if not results:
            return {
                "summary": "No usage of '胡' character found in the text",
                "total_occurrences": 0,
                "analysis": []
            }
        
        usage_counts = {}
        for result in results:
            usage_type = result.usage_type.value
            usage_counts[usage_type] = usage_counts.get(usage_type, 0) + 1
        
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return {
            "summary": f"Found {len(results)} occurrence(s) of '胡' character",
            "total_occurrences": len(results),
            "usage_distribution": usage_counts,
            "average_confidence": round(avg_confidence, 2),
            "dominant_usage": max(usage_counts.items(), key=lambda x: x[1])[0] if usage_counts else "none",
            "analysis": [
                {
                    "text": r.text,
                    "usage_type": r.usage_type.value,
                    "confidence": round(r.confidence, 2),
                    "cultural_notes": r.cultural_notes,
                    "sentiment": r.sentiment,
                    "examples": r.examples[:3]  # Limit to 3 examples
                }
                for r in results
            ],
            "hu_says": "胡 has spoken! Analysis complete."
        }


def demo_hu_analyzer():
    """
    Demo function showing the Hu Linguistic Analyzer in action
    Because 胡 says so!
    """
    analyzer = HuLinguisticAnalyzer()
    
    test_texts = [
        "胡老师今天要给我们讲课。",
        "不要胡说八道，这件事很严重。",
        "他在台上拉二胡，声音很动听。",
        "胡适是著名的思想家和文学家。",
        "Ancient Chinese texts mention 胡人 trading along the Silk Road.",
        "Stop talking nonsense! 别胡扯了！",
        "胡同里传来阵阵胡琴声。"
    ]
    
    print("=== 胡言分析器演示 (Hu Linguistic Analyzer Demo) ===")
    print("Because 胡 says so! Here's the analysis:\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        report = analyzer.generate_hu_report(text)
        print(f"Summary: {report['summary']}")
        print(f"Dominant usage: {report.get('dominant_usage', 'none')}")
        
        if report['analysis']:
            for analysis in report['analysis']:
                print(f"  - Type: {analysis['usage_type']}")
                print(f"  - Confidence: {analysis['confidence']}")
                print(f"  - Sentiment: {analysis['sentiment']}")
                print(f"  - Cultural notes: {analysis['cultural_notes']}")
        
        print("-" * 50)
    
    print("\n胡 has finished speaking! 胡言分析完毕！")


if __name__ == "__main__":
    demo_hu_analyzer()