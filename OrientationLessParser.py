#!/usr/bin/env python3
"""
OrientationLessParser - 無方向性解析器

A text parser that can handle content regardless of orientation, direction, or cultural layout.
This parser is designed to work with text in any direction (LTR, RTL, mixed) without bias.

無方向性文本解析器，能夠處理任意方向、任意方向或文化布局的內容。
此解析器設計為能夠處理任何方向的文本（從左到右、從右到左、混合）而無偏見。
"""

import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class TextDirection(Enum):
    """文本方向枚舉 (Text Direction Enumeration)"""
    LTR = "left_to_right"      # 從左到右 (Left to Right)
    RTL = "right_to_left"      # 從右到左 (Right to Left)
    BIDI = "bidirectional"     # 雙向文本 (Bidirectional)
    NEUTRAL = "neutral"        # 中性 (Neutral)
    MIXED = "mixed"           # 混合方向 (Mixed Directions)


class ScriptType(Enum):
    """文字系統類型 (Script Type)"""
    LATIN = "latin"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    CJK = "cjk"              # Chinese, Japanese, Korean
    DEVANAGARI = "devanagari"
    CYRILLIC = "cyrillic"
    THAI = "thai"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ParsedToken:
    """解析標記 (Parsed Token)"""
    text: str
    original_position: int
    logical_position: int
    direction: TextDirection
    script_type: ScriptType
    is_punctuation: bool = False
    is_number: bool = False
    is_whitespace: bool = False


@dataclass
class ParseResult:
    """解析結果 (Parse Result)"""
    original_text: str
    tokens: List[ParsedToken]
    dominant_direction: TextDirection
    dominant_script: ScriptType
    has_mixed_directions: bool
    normalized_text: str
    logical_order: List[ParsedToken]


class OrientationLessParser:
    """
    無方向性解析器 (Orientation-Less Parser)
    
    A parser that can handle text regardless of its orientation, direction, or layout.
    Supports multiple scripts and bidirectional text without cultural bias.
    
    能夠處理任意方向、方向或布局的文本的解析器。
    支持多種文字和雙向文本，不帶文化偏見。
    """
    
    def __init__(self):
        """初始化解析器 (Initialize parser)"""
        # RTL script ranges (Unicode ranges for right-to-left scripts)
        self.rtl_ranges = [
            (0x0590, 0x05FF),  # Hebrew
            (0x0600, 0x06FF),  # Arabic
            (0x0700, 0x074F),  # Syriac
            (0x0750, 0x077F),  # Arabic Supplement
            (0x0780, 0x07BF),  # Thaana
            (0x07C0, 0x07FF),  # NKo
            (0x0800, 0x083F),  # Samaritan
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB1D, 0xFB4F),  # Hebrew Presentation Forms
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ]
        
        # CJK ranges (Chinese, Japanese, Korean)
        self.cjk_ranges = [
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Extension A
            (0x20000, 0x2A6DF), # CJK Extension B
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x3100, 0x312F),  # Bopomofo
            (0xAC00, 0xD7AF),  # Hangul Syllables
        ]
        
        # Strong directional characters
        self.strong_ltr_chars = set()
        self.strong_rtl_chars = set()
        self._initialize_directional_chars()
    
    def _initialize_directional_chars(self):
        """初始化方向字符集 (Initialize directional character sets)"""
        # Basic Latin and Latin Extended
        for i in range(0x0041, 0x005B):  # A-Z
            self.strong_ltr_chars.add(chr(i))
        for i in range(0x0061, 0x007B):  # a-z
            self.strong_ltr_chars.add(chr(i))
        
        # RTL characters from common ranges
        for start, end in self.rtl_ranges:
            for i in range(start, min(end + 1, 0x10000)):  # Limit to BMP for now
                try:
                    char = chr(i)
                    if unicodedata.bidirectional(char) in ['R', 'AL']:
                        self.strong_rtl_chars.add(char)
                except ValueError:
                    continue
    
    def detect_script_type(self, text: str) -> ScriptType:
        """
        檢測文本的文字系統類型 (Detect script type of text)
        
        Args:
            text: 要檢測的文本 (Text to detect)
            
        Returns:
            ScriptType: 檢測到的文字系統類型 (Detected script type)
        """
        if not text:
            return ScriptType.UNKNOWN
        
        script_counts = {
            ScriptType.LATIN: 0,
            ScriptType.ARABIC: 0,
            ScriptType.HEBREW: 0,
            ScriptType.CJK: 0,
            ScriptType.DEVANAGARI: 0,
            ScriptType.CYRILLIC: 0,
            ScriptType.THAI: 0
        }
        
        for char in text:
            if not char.isalpha():
                continue
                
            code_point = ord(char)
            
            # Check CJK
            if any(start <= code_point <= end for start, end in self.cjk_ranges):
                script_counts[ScriptType.CJK] += 1
            # Check RTL scripts
            elif any(start <= code_point <= end for start, end in self.rtl_ranges):
                if 0x0590 <= code_point <= 0x05FF:
                    script_counts[ScriptType.HEBREW] += 1
                elif 0x0600 <= code_point <= 0x06FF or 0x0750 <= code_point <= 0x077F:
                    script_counts[ScriptType.ARABIC] += 1
            # Check Devanagari
            elif 0x0900 <= code_point <= 0x097F:
                script_counts[ScriptType.DEVANAGARI] += 1
            # Check Cyrillic
            elif 0x0400 <= code_point <= 0x04FF:
                script_counts[ScriptType.CYRILLIC] += 1
            # Check Thai
            elif 0x0E00 <= code_point <= 0x0E7F:
                script_counts[ScriptType.THAI] += 1
            # Default to Latin
            else:
                script_counts[ScriptType.LATIN] += 1
        
        # Find dominant script
        max_count = max(script_counts.values())
        if max_count == 0:
            return ScriptType.UNKNOWN
        
        # Check if mixed
        significant_scripts = [script for script, count in script_counts.items() 
                             if count > max_count * 0.2]
        
        if len(significant_scripts) > 1:
            return ScriptType.MIXED
        
        return max(script_counts, key=script_counts.get)
    
    def detect_text_direction(self, text: str) -> TextDirection:
        """
        檢測文本方向 (Detect text direction)
        
        Args:
            text: 要檢測的文本 (Text to detect)
            
        Returns:
            TextDirection: 檢測到的文本方向 (Detected text direction)
        """
        if not text:
            return TextDirection.NEUTRAL
        
        ltr_count = 0
        rtl_count = 0
        
        for char in text:
            if char in self.strong_ltr_chars:
                ltr_count += 1
            elif char in self.strong_rtl_chars:
                rtl_count += 1
            elif unicodedata.bidirectional(char) in ['L']:
                ltr_count += 1
            elif unicodedata.bidirectional(char) in ['R', 'AL']:
                rtl_count += 1
        
        total_directional = ltr_count + rtl_count
        if total_directional == 0:
            return TextDirection.NEUTRAL
        
        ltr_ratio = ltr_count / total_directional
        rtl_ratio = rtl_count / total_directional
        
        # If both directions are significant, it's bidirectional
        if ltr_ratio > 0.2 and rtl_ratio > 0.2:
            return TextDirection.BIDI
        elif ltr_ratio > 0.1 and rtl_ratio > 0.1:
            return TextDirection.MIXED
        elif ltr_ratio > rtl_ratio:
            return TextDirection.LTR
        else:
            return TextDirection.RTL
    
    def normalize_text(self, text: str) -> str:
        """
        正規化文本，去除方向性偏見 (Normalize text, removing directional bias)
        
        Args:
            text: 要正規化的文本 (Text to normalize)
            
        Returns:
            str: 正規化後的文本 (Normalized text)
        """
        # Normalize Unicode
        normalized = unicodedata.normalize('NFC', text)
        
        # Remove directional markers
        directional_markers = [
            '\u200E',  # LRM - Left-to-Right Mark
            '\u200F',  # RLM - Right-to-Left Mark
            '\u202A',  # LRE - Left-to-Right Embedding
            '\u202B',  # RLE - Right-to-Left Embedding
            '\u202C',  # PDF - Pop Directional Formatting
            '\u202D',  # LRO - Left-to-Right Override
            '\u202E',  # RLO - Right-to-Left Override
            '\u2066',  # LRI - Left-to-Right Isolate
            '\u2067',  # RLI - Right-to-Left Isolate
            '\u2068',  # FSI - First Strong Isolate
            '\u2069',  # PDI - Pop Directional Isolate
        ]
        
        for marker in directional_markers:
            normalized = normalized.replace(marker, '')
        
        return normalized
    
    def tokenize_orientation_agnostic(self, text: str) -> List[ParsedToken]:
        """
        無方向性分詞 (Orientation-agnostic tokenization)
        
        Args:
            text: 要分詞的文本 (Text to tokenize)
            
        Returns:
            List[ParsedToken]: 分詞結果 (Tokenization result)
        """
        if not text:
            return []
        
        tokens = []
        current_position = 0
        
        # Simple tokenization by character type
        i = 0
        while i < len(text):
            char = text[i]
            
            # Determine token type
            if char.isspace():
                # Whitespace token
                token_text = char
                while i + 1 < len(text) and text[i + 1].isspace():
                    i += 1
                    token_text += text[i]
                
                token = ParsedToken(
                    text=token_text,
                    original_position=current_position,
                    logical_position=len(tokens),
                    direction=TextDirection.NEUTRAL,
                    script_type=ScriptType.UNKNOWN,
                    is_whitespace=True
                )
                
            elif char.isdigit():
                # Number token
                token_text = char
                while i + 1 < len(text) and text[i + 1].isdigit():
                    i += 1
                    token_text += text[i]
                
                token = ParsedToken(
                    text=token_text,
                    original_position=current_position,
                    logical_position=len(tokens),
                    direction=TextDirection.NEUTRAL,
                    script_type=ScriptType.UNKNOWN,
                    is_number=True
                )
                
            elif unicodedata.category(char).startswith('P'):
                # Punctuation token
                token = ParsedToken(
                    text=char,
                    original_position=current_position,
                    logical_position=len(tokens),
                    direction=self.detect_text_direction(char),
                    script_type=self.detect_script_type(char),
                    is_punctuation=True
                )
                
            else:
                # Word token
                token_text = char
                while (i + 1 < len(text) and 
                       not text[i + 1].isspace() and 
                       not unicodedata.category(text[i + 1]).startswith('P') and
                       not text[i + 1].isdigit()):
                    i += 1
                    token_text += text[i]
                
                token = ParsedToken(
                    text=token_text,
                    original_position=current_position,
                    logical_position=len(tokens),
                    direction=self.detect_text_direction(token_text),
                    script_type=self.detect_script_type(token_text)
                )
            
            tokens.append(token)
            current_position = i + 1
            i += 1
        
        return tokens
    
    def reorder_logical(self, tokens: List[ParsedToken]) -> List[ParsedToken]:
        """
        將標記重新排序為邏輯順序 (Reorder tokens to logical order)
        
        This method implements a simplified bidirectional algorithm to reorder
        tokens according to their logical reading order, regardless of visual layout.
        
        Args:
            tokens: 原始標記列表 (Original token list)
            
        Returns:
            List[ParsedToken]: 邏輯順序的標記列表 (Logically ordered token list)
        """
        if not tokens:
            return []
        
        # For simplicity, we'll group consecutive tokens by direction
        groups = []
        current_group = []
        current_direction = None
        
        for token in tokens:
            if token.is_whitespace:
                # Whitespace maintains current direction context
                if current_group:
                    current_group.append(token)
                continue
            
            token_direction = token.direction
            
            if (current_direction is None or 
                current_direction == token_direction or
                token_direction == TextDirection.NEUTRAL):
                current_group.append(token)
                if current_direction is None and token_direction != TextDirection.NEUTRAL:
                    current_direction = token_direction
            else:
                # Direction change - finish current group
                if current_group:
                    groups.append((current_direction, current_group))
                current_group = [token]
                current_direction = token_direction
        
        # Add final group
        if current_group:
            groups.append((current_direction, current_group))
        
        # Reorder groups based on context
        logical_tokens = []
        
        for direction, group in groups:
            if direction == TextDirection.RTL:
                # Reverse RTL groups for logical order
                logical_tokens.extend(reversed(group))
            else:
                # Keep LTR and neutral in original order
                logical_tokens.extend(group)
        
        # Update logical positions
        for i, token in enumerate(logical_tokens):
            token.logical_position = i
        
        return logical_tokens
    
    def parse(self, text: str) -> ParseResult:
        """
        解析文本 (Parse text)
        
        Main parsing method that handles text regardless of orientation.
        
        Args:
            text: 要解析的文本 (Text to parse)
            
        Returns:
            ParseResult: 解析結果 (Parse result)
        """
        if not text:
            return ParseResult(
                original_text=text,
                tokens=[],
                dominant_direction=TextDirection.NEUTRAL,
                dominant_script=ScriptType.UNKNOWN,
                has_mixed_directions=False,
                normalized_text="",
                logical_order=[]
            )
        
        # Normalize text
        normalized_text = self.normalize_text(text)
        
        # Detect overall characteristics
        dominant_direction = self.detect_text_direction(text)
        dominant_script = self.detect_script_type(text)
        
        # Tokenize
        tokens = self.tokenize_orientation_agnostic(normalized_text)
        
        # Check for mixed directions
        directions = set(token.direction for token in tokens 
                        if not token.is_whitespace and token.direction != TextDirection.NEUTRAL)
        has_mixed_directions = len(directions) > 1
        
        # Create logical order
        logical_order = self.reorder_logical(tokens)
        
        return ParseResult(
            original_text=text,
            tokens=tokens,
            dominant_direction=dominant_direction,
            dominant_script=dominant_script,
            has_mixed_directions=has_mixed_directions,
            normalized_text=normalized_text,
            logical_order=logical_order
        )
    
    def extract_text_content(self, parse_result: ParseResult, 
                           include_punctuation: bool = True,
                           include_numbers: bool = True) -> str:
        """
        提取文本內容 (Extract text content)
        
        Extract clean text content from parse result in logical order.
        
        Args:
            parse_result: 解析結果 (Parse result)
            include_punctuation: 是否包含標點符號 (Whether to include punctuation)
            include_numbers: 是否包含數字 (Whether to include numbers)
            
        Returns:
            str: 提取的文本內容 (Extracted text content)
        """
        content_tokens = []
        
        for token in parse_result.logical_order:
            if token.is_whitespace:
                continue
            elif token.is_punctuation and not include_punctuation:
                continue
            elif token.is_number and not include_numbers:
                continue
            else:
                content_tokens.append(token.text)
        
        return ' '.join(content_tokens)
    
    def get_parsing_statistics(self, parse_result: ParseResult) -> Dict[str, Any]:
        """
        獲取解析統計信息 (Get parsing statistics)
        
        Args:
            parse_result: 解析結果 (Parse result)
            
        Returns:
            Dict[str, Any]: 統計信息 (Statistics)
        """
        if not parse_result.tokens:
            return {
                'total_tokens': 0,
                'word_tokens': 0,
                'punctuation_tokens': 0,
                'number_tokens': 0,
                'whitespace_tokens': 0,
                'direction_distribution': {},
                'script_distribution': {},
                'is_bidirectional': False
            }
        
        # Count token types
        word_tokens = sum(1 for t in parse_result.tokens 
                         if not t.is_whitespace and not t.is_punctuation and not t.is_number)
        punctuation_tokens = sum(1 for t in parse_result.tokens if t.is_punctuation)
        number_tokens = sum(1 for t in parse_result.tokens if t.is_number)
        whitespace_tokens = sum(1 for t in parse_result.tokens if t.is_whitespace)
        
        # Direction distribution
        direction_counts = {}
        for token in parse_result.tokens:
            if not token.is_whitespace:
                direction = token.direction.value
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Script distribution
        script_counts = {}
        for token in parse_result.tokens:
            if not token.is_whitespace:
                script = token.script_type.value
                script_counts[script] = script_counts.get(script, 0) + 1
        
        return {
            'total_tokens': len(parse_result.tokens),
            'word_tokens': word_tokens,
            'punctuation_tokens': punctuation_tokens,
            'number_tokens': number_tokens,
            'whitespace_tokens': whitespace_tokens,
            'direction_distribution': direction_counts,
            'script_distribution': script_counts,
            'is_bidirectional': parse_result.has_mixed_directions,
            'dominant_direction': parse_result.dominant_direction.value,
            'dominant_script': parse_result.dominant_script.value
        }


def main():
    """示例用法 (Example usage)"""
    parser = OrientationLessParser()
    
    # Test cases with different orientations and scripts
    test_cases = [
        {
            'name': 'English (LTR)',
            'text': 'Hello world! This is a test.'
        },
        {
            'name': 'Arabic (RTL)', 
            'text': 'مرحبا بالعالم! هذا اختبار.'
        },
        {
            'name': 'Hebrew (RTL)',
            'text': 'שלום עולם! זהו מבחן.'
        },
        {
            'name': 'Chinese (Neutral)',
            'text': '你好世界！这是一个测试。'
        },
        {
            'name': 'Mixed Scripts',
            'text': 'Hello مرحبا 你好 שלום world!'
        },
        {
            'name': 'Bidirectional',
            'text': 'The word مرحبا means hello in Arabic'
        },
        {
            'name': 'Numbers and punctuation',
            'text': 'Price: $123.45 (€98.76)'
        }
    ]
    
    print("=" * 80)
    print("OrientationLessParser Test Results")
    print("無方向性解析器測試結果")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        print(f"Original: {test_case['text']}")
        
        # Parse the text
        result = parser.parse(test_case['text'])
        
        # Display results
        print(f"Normalized: {result.normalized_text}")
        print(f"Dominant Direction: {result.dominant_direction.value}")
        print(f"Dominant Script: {result.dominant_script.value}")
        print(f"Mixed Directions: {result.has_mixed_directions}")
        
        # Show tokens
        print("Tokens (Original Order):")
        for token in result.tokens:
            if not token.is_whitespace:
                print(f"  '{token.text}' [{token.direction.value}, {token.script_type.value}]")
        
        # Show logical order
        logical_text = parser.extract_text_content(result)
        print(f"Logical Order Text: {logical_text}")
        
        # Statistics
        stats = parser.get_parsing_statistics(result)
        print(f"Statistics: {stats['word_tokens']} words, "
              f"{stats['punctuation_tokens']} punct, "
              f"{stats['number_tokens']} numbers")
        
        print()


if __name__ == "__main__":
    main()