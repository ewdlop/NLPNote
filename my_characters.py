#!/usr/bin/env python3
"""
我的字符 (My Characters) - 綜合字符分析工具
Comprehensive Character Analysis Tool

This is the main "My Characters" tool that integrates character analysis,
transformation, and human expression evaluation for the multilingual NLP repository.

這是主要的"我的字符"工具，整合了字符分析、變換和人類表達評估功能，
用於多語言NLP存儲庫。
"""

import sys
import os
import argparse
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import our custom modules
from CharacterAnalyzer import CharacterAnalyzer, WritingSystem, CharacterStats
from CharacterTransformer import CharacterTransformer, TransformationResult
from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext


class MyCharacters:
    """
    我的字符主類 (My Characters Main Class)
    
    Comprehensive character analysis and transformation system
    for multilingual NLP applications.
    """
    
    def __init__(self):
        self.analyzer = CharacterAnalyzer()
        self.transformer = CharacterTransformer()
        self.expression_evaluator = HumanExpressionEvaluator()
        self.repository_analyzed = False
        self.analysis_cache = {}
        
    def initialize_repository_analysis(self, force_refresh: bool = False):
        """初始化存儲庫分析 (Initialize repository analysis)"""
        if self.repository_analyzed and not force_refresh:
            return
            
        print("🔍 分析存儲庫字符... (Analyzing repository characters...)")
        try:
            # Analyze the repository
            results = self.analyzer.analyze_directory(".", ['.md', '.py', '.txt', '.json', '.html'])
            self.analysis_cache = results
            self.repository_analyzed = True
            
            print(f"✓ 分析完成! 發現 {results['unique_characters']} 個唯一字符")
            print(f"✓ Analysis complete! Found {results['unique_characters']} unique characters")
            
        except Exception as e:
            print(f"❌ 分析失敗: {e}")
            print(f"❌ Analysis failed: {e}")
    
    def show_comprehensive_summary(self):
        """顯示綜合總結 (Show comprehensive summary)"""
        if not self.repository_analyzed:
            self.initialize_repository_analysis()
        
        print("\n" + "="*80)
        print("📊 我的字符 - 綜合分析報告 (My Characters - Comprehensive Analysis)")
        print("="*80)
        
        # Basic statistics
        results = self.analysis_cache
        print(f"📁 分析文件數 (Files analyzed): {results['total_files']}")
        print(f"📝 總字符數 (Total characters): {results['total_characters']:,}")
        print(f"🔤 唯一字符數 (Unique characters): {results['unique_characters']:,}")
        
        # Writing system distribution
        print(f"\n📚 書寫系統分布 (Writing System Distribution):")
        ws_counts = results['writing_systems']
        total_ws_chars = sum(ws_counts.values())
        
        for ws, count in sorted(ws_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_ws_chars * 100) if total_ws_chars > 0 else 0
            ws_name = ws.value if hasattr(ws, 'value') else str(ws)
            icon = self._get_writing_system_icon(ws)
            bar = "█" * min(int(percentage / 2), 40)
            print(f"  {icon} {ws_name:12} {count:6,} ({percentage:5.1f}%) {bar}")
        
        # Repository character insights
        self._show_repository_insights()
        
        # Top multilingual expressions
        self._show_multilingual_expressions()
    
    def _get_writing_system_icon(self, ws) -> str:
        """獲取書寫系統圖標 (Get writing system icon)"""
        if isinstance(ws, str):
            try:
                ws = WritingSystem(ws)
            except:
                return "❓"
        
        icons = {
            WritingSystem.LATIN: "🔤",
            WritingSystem.CJK: "🀄",
            WritingSystem.ARABIC: "🔗",
            WritingSystem.HEBREW: "🔯",
            WritingSystem.CYRILLIC: "🇷🇺",
            WritingSystem.GREEK: "🇬🇷",
            WritingSystem.DEVANAGARI: "🇮🇳",
            WritingSystem.PUNCTUATION: "❗",
            WritingSystem.SYMBOLS: "🔣",
            WritingSystem.DIGITS: "🔢",
            WritingSystem.OTHER: "❓"
        }
        return icons.get(ws, "❓")
    
    def _show_repository_insights(self):
        """顯示存儲庫洞察 (Show repository insights)"""
        print(f"\n🔍 存儲庫字符洞察 (Repository Character Insights):")
        
        # Most common characters by writing system
        ws_top_chars = {}
        for char, freq in self.analyzer.character_frequency.most_common(100):
            if not char.isspace():
                char_info = self.analyzer.get_character_info(char)
                ws = char_info.writing_system
                if ws not in ws_top_chars:
                    ws_top_chars[ws] = []
                if len(ws_top_chars[ws]) < 3:
                    ws_top_chars[ws].append((char, freq))
        
        for ws, chars in ws_top_chars.items():
            if chars:
                icon = self._get_writing_system_icon(ws)
                ws_name = ws.value if hasattr(ws, 'value') else str(ws)
                char_list = ', '.join([f"'{c}' ({f:,})" for c, f in chars])
                print(f"  {icon} {ws_name}: {char_list}")
    
    def _show_multilingual_expressions(self):
        """顯示多語言表達式 (Show multilingual expressions)"""
        print(f"\n🌍 多語言表達範例 (Multilingual Expression Examples):")
        
        # Find files with interesting multilingual content
        sample_expressions = [
            "我的字符 (My Characters)",
            "Human Expression Evaluation",
            "自然語言處理 (Natural Language Processing)",
            "Параллельные конструкции",
            "مرحبا بالعالم"
        ]
        
        for expr in sample_expressions:
            if any(char in self.analyzer.character_frequency for char in expr):
                char_analysis = self.analyzer.analyze_text(expr)
                ws_list = list(char_analysis['writing_systems'].keys())
                ws_names = [ws.value if hasattr(ws, 'value') else str(ws) for ws in ws_list]
                print(f"  📝 '{expr}' - {', '.join(ws_names)}")
    
    def analyze_custom_expression(self, expression: str, include_transformation: bool = True):
        """分析自定義表達式 (Analyze custom expression)"""
        print(f"\n📝 分析表達式: '{expression}'")
        print("="*60)
        
        # Character analysis
        char_analysis = self.analyzer.analyze_text(expression, "user_input")
        print(f"字符總數 (Total chars): {char_analysis['total_characters']}")
        print(f"唯一字符數 (Unique chars): {char_analysis['unique_characters']}")
        
        # Character details
        print(f"\n🔤 字符詳情 (Character Details):")
        for char, char_info in char_analysis['character_infos'].items():
            icon = self._get_writing_system_icon(char_info.writing_system)
            print(f"  '{char}' {icon} U+{char_info.unicode_codepoint:04X} - {char_info.unicode_name}")
        
        # Writing system distribution
        print(f"\n📚 書寫系統分布 (Writing Systems):")
        for ws, count in char_analysis['writing_systems'].items():
            icon = self._get_writing_system_icon(ws)
            ws_name = ws.value if hasattr(ws, 'value') else str(ws)
            print(f"  {icon} {ws_name}: {count}")
        
        # Human expression evaluation
        try:
            print(f"\n🧠 人類表達評估 (Human Expression Evaluation):")
            context = ExpressionContext()
            
            # Get individual evaluations
            formal_result = self.expression_evaluator.formal_evaluator.evaluate(expression)
            cognitive_result = self.expression_evaluator.cognitive_evaluator.evaluate_expression(expression, context)
            social_result = self.expression_evaluator.social_evaluator.evaluate_social_expression(expression, "user", context)
            
            print(f"  形式語義 (Formal): {formal_result.score:.3f}")
            print(f"  認知處理 (Cognitive): {cognitive_result.score:.3f}")
            print(f"  社會適當 (Social): {social_result.score:.3f}")
            
        except Exception as e:
            print(f"  ⚠️ 評估錯誤: {e}")
        
        # Character transformation examples
        if include_transformation:
            print(f"\n🔄 字符變換範例 (Character Transformations):")
            
            target_systems = [WritingSystem.CJK, WritingSystem.CYRILLIC, WritingSystem.ARABIC]
            for target in target_systems:
                try:
                    result = self.transformer.transform_text(expression, target, 'random')
                    icon = self._get_writing_system_icon(target)
                    print(f"  {icon} {target.value}: {result.transformed_text}")
                except Exception as e:
                    print(f"  ❌ {target.value}: 變換失敗 ({e})")
    
    def generate_character_report(self, output_file: str = "my_characters_full_report.md"):
        """生成完整字符報告 (Generate comprehensive character report)"""
        if not self.repository_analyzed:
            self.initialize_repository_analysis()
        
        print(f"📄 生成報告... (Generating report...)")
        
        report_lines = [
            "# 我的字符 - 完整分析報告 (My Characters - Full Analysis Report)",
            "",
            f"生成時間 (Generated): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 🔍 總覽 (Overview)",
            f"- 分析文件數: {self.analysis_cache['total_files']}",
            f"- 總字符數: {self.analysis_cache['total_characters']:,}",
            f"- 唯一字符數: {self.analysis_cache['unique_characters']:,}",
            "",
            "## 📚 書寫系統詳細分析 (Writing System Analysis)",
            ""
        ]
        
        # Detailed writing system analysis
        ws_counts = self.analysis_cache['writing_systems']
        total_ws_chars = sum(ws_counts.values())
        
        for ws, count in sorted(ws_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_ws_chars * 100) if total_ws_chars > 0 else 0
            ws_name = ws.value if hasattr(ws, 'value') else str(ws)
            icon = self._get_writing_system_icon(ws)
            
            report_lines.extend([
                f"### {icon} {ws_name}",
                f"- 字符數: {count:,} ({percentage:.1f}%)",
                f"- 範例字符: {self._get_sample_characters(ws, 10)}",
                ""
            ])
        
        # Character frequency analysis
        report_lines.extend([
            "## 🏆 字符頻率分析 (Character Frequency Analysis)",
            "",
            "### 最常用字符 (Top Characters)",
            ""
        ])
        
        for i, (char, freq) in enumerate(self.analyzer.character_frequency.most_common(50), 1):
            if not char.isspace():
                char_info = self.analyzer.get_character_info(char)
                icon = self._get_writing_system_icon(char_info.writing_system)
                report_lines.append(
                    f"{i:2d}. '{char}' {icon} U+{char_info.unicode_codepoint:04X} "
                    f"({freq:,} times) - {char_info.unicode_name}"
                )
        
        # File analysis summary
        report_lines.extend([
            "",
            "## 📁 文件分析摘要 (File Analysis Summary)",
            ""
        ])
        
        # Show top 10 files by character count
        file_results = self.analysis_cache.get('file_results', {})
        sorted_files = sorted(
            [(f, r) for f, r in file_results.items() if 'total_characters' in r],
            key=lambda x: x[1]['total_characters'],
            reverse=True
        )
        
        for file_path, result in sorted_files[:10]:
            ws_count = len(result.get('writing_systems', {}))
            report_lines.append(
                f"- `{file_path}`: {result['total_characters']:,} 字符, "
                f"{result['unique_characters']} 唯一字符, {ws_count} 書寫系統"
            )
        
        # Encoding issues
        encoding_issues = self.analysis_cache.get('encoding_issues', [])
        if encoding_issues:
            report_lines.extend([
                "",
                "## ⚠️  編碼問題 (Encoding Issues)",
                ""
            ])
            for issue in encoding_issues:
                report_lines.append(f"- {issue}")
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 報告已保存到 {output_file}")
        print(f"✓ Report saved to {output_file}")
    
    def _get_sample_characters(self, writing_system, limit: int = 5) -> str:
        """獲取書寫系統的範例字符 (Get sample characters for writing system)"""
        samples = []
        for char, freq in self.analyzer.character_frequency.most_common():
            if len(samples) >= limit:
                break
            if not char.isspace():
                char_info = self.analyzer.get_character_info(char)
                if char_info.writing_system == writing_system:
                    samples.append(f"'{char}'")
        return ', '.join(samples) if samples else "無 (None)"
    
    def export_data(self, format_type: str = 'json', output_file: str = None):
        """導出數據 (Export data)"""
        if not self.repository_analyzed:
            self.initialize_repository_analysis()
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"my_characters_data_{timestamp}.{format_type}"
        
        print(f"📤 導出數據... (Exporting data...)")
        
        # Prepare export data
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_files': self.analysis_cache['total_files'],
                'total_characters': self.analysis_cache['total_characters'],
                'unique_characters': self.analysis_cache['unique_characters']
            },
            'writing_systems': {
                str(ws): count for ws, count in self.analysis_cache['writing_systems'].items()
            },
            'character_frequency': [
                {
                    'character': char,
                    'frequency': freq,
                    'unicode': f"U+{ord(char):04X}",
                    'name': self.analyzer.get_character_info(char).unicode_name,
                    'writing_system': str(self.analyzer.get_character_info(char).writing_system)
                }
                for char, freq in self.analyzer.character_frequency.most_common()
                if not char.isspace()
            ]
        }
        
        try:
            if format_type == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                    
            elif format_type == 'csv':
                import pandas as pd
                # Create character dataframe
                df = self.analyzer.create_character_dataframe()
                df.to_csv(output_file, index=False, encoding='utf-8')
                
            else:
                print(f"❌ 不支持的格式: {format_type}")
                return
            
            print(f"✓ 數據已導出到 {output_file}")
            print(f"✓ Data exported to {output_file}")
            
        except Exception as e:
            print(f"❌ 導出失敗: {e}")


def main():
    """主函數 (Main function)"""
    parser = argparse.ArgumentParser(
        description="我的字符 - 綜合字符分析工具 (My Characters - Comprehensive Character Analysis Tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法 (Example usage):
  python3 my_characters.py --summary          # 顯示綜合總結
  python3 my_characters.py --analyze "測試"   # 分析特定表達式
  python3 my_characters.py --report           # 生成完整報告
  python3 my_characters.py --export json      # 導出JSON數據
        """
    )
    
    parser.add_argument('--summary', action='store_true', 
                       help='顯示綜合字符分析總結')
    parser.add_argument('--analyze', type=str, 
                       help='分析特定表達式的字符')
    parser.add_argument('--report', action='store_true', 
                       help='生成完整的字符分析報告')
    parser.add_argument('--export', choices=['json', 'csv'], 
                       help='導出數據 (json或csv格式)')
    parser.add_argument('--output', type=str, 
                       help='輸出文件名')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='強制重新分析存儲庫')
    
    args = parser.parse_args()
    
    # Create My Characters instance
    my_chars = MyCharacters()
    
    # Initialize if needed
    if any([args.summary, args.analyze, args.report, args.export]):
        my_chars.initialize_repository_analysis(args.force_refresh)
    
    # Execute commands
    if args.summary:
        my_chars.show_comprehensive_summary()
    
    if args.analyze:
        my_chars.analyze_custom_expression(args.analyze)
    
    if args.report:
        output_file = args.output or "my_characters_full_report.md"
        my_chars.generate_character_report(output_file)
    
    if args.export:
        my_chars.export_data(args.export, args.output)
    
    # Default behavior: show summary
    if len(sys.argv) == 1:
        print("🚀 歡迎使用我的字符分析工具！")
        print("🚀 Welcome to My Characters Analysis Tool!")
        print("\n使用 --help 查看所有選項")
        print("Use --help to see all options")
        my_chars.show_comprehensive_summary()


if __name__ == "__main__":
    main()