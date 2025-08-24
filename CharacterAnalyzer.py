"""
字符分析器 (Character Analyzer)

This module provides comprehensive character analysis functionality for the multilingual
NLP repository, enabling analysis of Unicode characters, writing systems, encoding, 
and character properties across the repository's extensive multilingual content.

一個全面的字符分析工具，用於分析多語言NLP存儲庫中的Unicode字符、書寫系統、編碼和字符屬性。
"""

import re
import os
import unicodedata
import collections
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class WritingSystem(Enum):
    """書寫系統 (Writing Systems)"""
    LATIN = "Latin"
    CJK = "CJK"  # Chinese, Japanese, Korean
    ARABIC = "Arabic"
    HEBREW = "Hebrew"
    CYRILLIC = "Cyrillic"
    GREEK = "Greek"
    DEVANAGARI = "Devanagari"
    THAI = "Thai"
    ARMENIAN = "Armenian"
    GEORGIAN = "Georgian"
    SYMBOLS = "Symbols"
    PUNCTUATION = "Punctuation"
    DIGITS = "Digits"
    OTHER = "Other"


@dataclass
class CharacterInfo:
    """字符信息 (Character Information)"""
    character: str
    unicode_codepoint: int
    unicode_name: str
    category: str
    writing_system: WritingSystem
    frequency: int = 0
    encoding_bytes: Dict[str, bytes] = None


@dataclass
class CharacterStats:
    """字符統計 (Character Statistics)"""
    total_characters: int
    unique_characters: int
    writing_systems: Dict[WritingSystem, int]
    top_characters: List[Tuple[str, int]]
    encoding_issues: List[str]


class CharacterAnalyzer:
    """
    字符分析器 (Character Analyzer)
    
    Comprehensive character analysis tool for analyzing Unicode characters,
    writing systems, and character properties across multilingual text.
    """
    
    def __init__(self):
        self.character_cache: Dict[str, CharacterInfo] = {}
        self.analyzed_files: Set[str] = set()
        self.character_frequency: collections.Counter = collections.Counter()
        
        # Unicode block ranges for writing system detection
        self.unicode_blocks = {
            WritingSystem.LATIN: [
                (0x0020, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
                (0x1E00, 0x1EFF),  # Latin Extended Additional
            ],
            WritingSystem.CJK: [
                (0x4E00, 0x9FFF),  # CJK Unified Ideographs
                (0x3400, 0x4DBF),  # CJK Extension A
                (0x3040, 0x309F),  # Hiragana
                (0x30A0, 0x30FF),  # Katakana
                (0xAC00, 0xD7AF),  # Hangul Syllables
                (0x1100, 0x11FF),  # Hangul Jamo
                (0x3100, 0x312F),  # Bopomofo
            ],
            WritingSystem.ARABIC: [
                (0x0600, 0x06FF),  # Arabic
                (0x0750, 0x077F),  # Arabic Supplement
                (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
                (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
            ],
            WritingSystem.HEBREW: [
                (0x0590, 0x05FF),  # Hebrew
            ],
            WritingSystem.CYRILLIC: [
                (0x0400, 0x04FF),  # Cyrillic
                (0x0500, 0x052F),  # Cyrillic Supplement
            ],
            WritingSystem.GREEK: [
                (0x0370, 0x03FF),  # Greek and Coptic
                (0x1F00, 0x1FFF),  # Greek Extended
            ],
            WritingSystem.DEVANAGARI: [
                (0x0900, 0x097F),  # Devanagari
            ],
            WritingSystem.THAI: [
                (0x0E00, 0x0E7F),  # Thai
            ],
        }
    
    def get_character_info(self, char: str) -> CharacterInfo:
        """
        獲取字符詳細信息 (Get detailed character information)
        """
        if char in self.character_cache:
            return self.character_cache[char]
        
        codepoint = ord(char)
        
        try:
            unicode_name = unicodedata.name(char, f"UNNAMED_{codepoint:04X}")
        except ValueError:
            unicode_name = f"PRIVATE_USE_{codepoint:04X}"
        
        category = unicodedata.category(char)
        writing_system = self._detect_writing_system(codepoint, category)
        
        # Get encoding bytes for common encodings
        encoding_bytes = {}
        for encoding in ['utf-8', 'utf-16', 'latin-1', 'ascii']:
            try:
                encoding_bytes[encoding] = char.encode(encoding)
            except UnicodeEncodeError:
                encoding_bytes[encoding] = None
        
        char_info = CharacterInfo(
            character=char,
            unicode_codepoint=codepoint,
            unicode_name=unicode_name,
            category=category,
            writing_system=writing_system,
            encoding_bytes=encoding_bytes
        )
        
        self.character_cache[char] = char_info
        return char_info
    
    def _detect_writing_system(self, codepoint: int, category: str) -> WritingSystem:
        """
        檢測字符的書寫系統 (Detect character's writing system)
        """
        # Check if it's punctuation or symbols
        if category.startswith('P'):
            return WritingSystem.PUNCTUATION
        elif category.startswith('S'):
            return WritingSystem.SYMBOLS
        elif category.startswith('N'):
            return WritingSystem.DIGITS
        
        # Check Unicode blocks
        for writing_system, blocks in self.unicode_blocks.items():
            for start, end in blocks:
                if start <= codepoint <= end:
                    return writing_system
        
        return WritingSystem.OTHER
    
    def analyze_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """
        分析文本中的字符 (Analyze characters in text)
        """
        characters = list(text)
        unique_chars = set(characters)
        
        # Update frequency counter
        self.character_frequency.update(characters)
        
        # Analyze each unique character
        char_infos = {}
        writing_systems = collections.Counter()
        categories = collections.Counter()
        
        for char in unique_chars:
            if char.isspace():  # Skip whitespace for main analysis
                continue
                
            char_info = self.get_character_info(char)
            char_infos[char] = char_info
            writing_systems[char_info.writing_system] += 1
            categories[char_info.category] += 1
        
        return {
            'source': source,
            'total_characters': len(characters),
            'unique_characters': len(unique_chars),
            'character_infos': char_infos,
            'writing_systems': dict(writing_systems),
            'categories': dict(categories),
            'most_common': self.character_frequency.most_common(10)
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        分析文件中的字符 (Analyze characters in a file)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = self.analyze_text(content, source=file_path)
            self.analyzed_files.add(file_path)
            return result
            
        except UnicodeDecodeError as e:
            # Try different encodings
            encodings = ['latin-1', 'gb2312', 'big5', 'shift_jis', 'euc-kr']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    result = self.analyze_text(content, source=file_path)
                    result['encoding_used'] = encoding
                    result['encoding_warning'] = f"File required {encoding} encoding"
                    self.analyzed_files.add(file_path)
                    return result
                except:
                    continue
            
            return {
                'source': file_path,
                'error': f"Could not decode file: {str(e)}",
                'encoding_issues': [f"Failed to decode {file_path}"]
            }
        
        except Exception as e:
            return {
                'source': file_path,
                'error': f"Error analyzing file: {str(e)}"
            }
    
    def analyze_directory(self, directory_path: str, file_extensions: List[str] = None) -> Dict[str, Any]:
        """
        分析目錄中所有文件的字符 (Analyze characters in all files in a directory)
        """
        if file_extensions is None:
            file_extensions = ['.md', '.txt', '.py', '.js', '.html', '.xml', '.json']
        
        results = {}
        encoding_issues = []
        total_stats = {
            'total_files': 0,
            'total_characters': 0,
            'unique_characters': set(),
            'writing_systems': collections.Counter(),
            'file_results': {}
        }
        
        for root, dirs, files in os.walk(directory_path):
            # Skip hidden directories and common build/cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory_path)
                    
                    file_result = self.analyze_file(file_path)
                    results[relative_path] = file_result
                    
                    total_stats['total_files'] += 1
                    
                    if 'error' not in file_result:
                        total_stats['total_characters'] += file_result.get('total_characters', 0)
                        if 'character_infos' in file_result:
                            total_stats['unique_characters'].update(file_result['character_infos'].keys())
                        
                        for ws, count in file_result.get('writing_systems', {}).items():
                            if isinstance(ws, WritingSystem):
                                total_stats['writing_systems'][ws] += count
                            else:
                                # Handle string representation
                                try:
                                    ws_enum = WritingSystem(ws)
                                    total_stats['writing_systems'][ws_enum] += count
                                except ValueError:
                                    total_stats['writing_systems'][WritingSystem.OTHER] += count
                    
                    if 'encoding_issues' in file_result:
                        encoding_issues.extend(file_result['encoding_issues'])
        
        # Convert set to count for unique characters
        total_stats['unique_characters'] = len(total_stats['unique_characters'])
        total_stats['encoding_issues'] = encoding_issues
        total_stats['file_results'] = results
        
        return total_stats
    
    def get_repository_character_summary(self) -> CharacterStats:
        """
        獲取存儲庫字符總結 (Get repository character summary)
        """
        return CharacterStats(
            total_characters=sum(self.character_frequency.values()),
            unique_characters=len(self.character_frequency),
            writing_systems=self._count_writing_systems(),
            top_characters=self.character_frequency.most_common(20),
            encoding_issues=[]
        )
    
    def _count_writing_systems(self) -> Dict[WritingSystem, int]:
        """Count characters by writing system"""
        ws_counts = collections.Counter()
        for char, freq in self.character_frequency.items():
            if not char.isspace():
                char_info = self.get_character_info(char)
                ws_counts[char_info.writing_system] += freq
        return dict(ws_counts)
    
    def generate_character_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成字符分析報告 (Generate character analysis report)
        """
        report_lines = [
            "# 我的字符分析報告 (My Characters Analysis Report)",
            "",
            f"## 總覽 (Overview)",
            f"- 分析文件數: {analysis_results.get('total_files', 0)}",
            f"- 總字符數: {analysis_results.get('total_characters', 0):,}",
            f"- 唯一字符數: {analysis_results.get('unique_characters', 0):,}",
            "",
            "## 書寫系統分布 (Writing System Distribution)",
            ""
        ]
        
        # Writing systems summary
        ws_counts = analysis_results.get('writing_systems', {})
        total_ws_chars = sum(ws_counts.values())
        
        for ws, count in sorted(ws_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_ws_chars * 100) if total_ws_chars > 0 else 0
            if isinstance(ws, WritingSystem):
                ws_name = ws.value
            else:
                ws_name = str(ws)
            report_lines.append(f"- {ws_name}: {count:,} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "## 最常用字符 (Most Frequent Characters)",
            ""
        ])
        
        # Top characters
        if hasattr(self, 'character_frequency'):
            for char, freq in self.character_frequency.most_common(20):
                if not char.isspace():
                    char_info = self.get_character_info(char)
                    report_lines.append(
                        f"- '{char}' (U+{char_info.unicode_codepoint:04X}): "
                        f"{freq:,} times - {char_info.unicode_name}"
                    )
        
        # Encoding issues
        encoding_issues = analysis_results.get('encoding_issues', [])
        if encoding_issues:
            report_lines.extend([
                "",
                "## 編碼問題 (Encoding Issues)",
                ""
            ])
            for issue in encoding_issues:
                report_lines.append(f"- {issue}")
        
        return "\n".join(report_lines)
    
    def create_character_dataframe(self) -> pd.DataFrame:
        """
        創建字符數據框 (Create character DataFrame for analysis)
        """
        char_data = []
        
        for char, freq in self.character_frequency.most_common():
            if not char.isspace():
                char_info = self.get_character_info(char)
                char_data.append({
                    'Character': char,
                    'Unicode': f"U+{char_info.unicode_codepoint:04X}",
                    'CodePoint': char_info.unicode_codepoint,
                    'Name': char_info.unicode_name,
                    'Category': char_info.category,
                    'WritingSystem': char_info.writing_system.value,
                    'Frequency': freq,
                    'UTF8_Bytes': len(char_info.encoding_bytes.get('utf-8', b'')),
                    'CanEncode_ASCII': char_info.encoding_bytes.get('ascii') is not None,
                    'CanEncode_Latin1': char_info.encoding_bytes.get('latin-1') is not None
                })
        
        return pd.DataFrame(char_data)
    
    def search_characters(self, 
                         query: str = None, 
                         writing_system: WritingSystem = None,
                         min_frequency: int = 1) -> List[CharacterInfo]:
        """
        搜索字符 (Search characters)
        """
        results = []
        
        for char, freq in self.character_frequency.items():
            if freq < min_frequency:
                continue
                
            char_info = self.get_character_info(char)
            
            # Filter by writing system
            if writing_system and char_info.writing_system != writing_system:
                continue
            
            # Filter by query
            if query:
                query_lower = query.lower()
                if not (query_lower in char_info.unicode_name.lower() or 
                       query_lower in char or
                       query_lower in char_info.category.lower()):
                    continue
            
            char_info.frequency = freq
            results.append(char_info)
        
        return sorted(results, key=lambda x: x.frequency, reverse=True)


def main():
    """示例使用 (Example usage)"""
    analyzer = CharacterAnalyzer()
    
    # Analyze current directory
    print("開始分析字符... (Starting character analysis...)")
    results = analyzer.analyze_directory(".", ['.md', '.py', '.txt'])
    
    # Generate report
    report = analyzer.generate_character_report(results)
    
    # Save report
    with open("my_characters_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✓ 字符分析完成！報告已保存到 my_characters_report.md")
    print("✓ Character analysis complete! Report saved to my_characters_report.md")
    
    # Display summary
    summary = analyzer.get_repository_character_summary()
    print(f"\n總結 (Summary):")
    print(f"- 總字符數: {summary.total_characters:,}")
    print(f"- 唯一字符數: {summary.unique_characters:,}")
    print(f"- 書寫系統數: {len(summary.writing_systems)}")


if __name__ == "__main__":
    main()