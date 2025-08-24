#!/usr/bin/env python3
"""
我的字符瀏覽器 (My Characters Browser)

An interactive tool for exploring and analyzing characters in the NLP repository.
互動式工具，用於探索和分析NLP存儲庫中的字符。
"""

import sys
import argparse
from CharacterAnalyzer import CharacterAnalyzer, WritingSystem
from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext


class CharacterBrowser:
    """字符瀏覽器 (Character Browser)"""
    
    def __init__(self):
        self.analyzer = CharacterAnalyzer()
        self.expression_evaluator = HumanExpressionEvaluator()
        self.current_results = None
        
    def load_repository_analysis(self):
        """加載存儲庫分析 (Load repository analysis)"""
        print("🔍 正在分析存儲庫字符... (Analyzing repository characters...)")
        self.current_results = self.analyzer.analyze_directory(".", ['.md', '.py', '.txt', '.json'])
        print(f"✓ 分析完成！發現 {self.current_results['unique_characters']} 個唯一字符")
        print(f"✓ Analysis complete! Found {self.current_results['unique_characters']} unique characters")
        
    def show_summary(self):
        """顯示總結 (Show summary)"""
        if not self.current_results:
            print("❌ 請先分析存儲庫 (Please analyze repository first)")
            return
            
        print("\n" + "="*60)
        print("📊 字符分析總結 (Character Analysis Summary)")
        print("="*60)
        print(f"📁 分析文件數: {self.current_results['total_files']}")
        print(f"📝 總字符數: {self.current_results['total_characters']:,}")
        print(f"🔤 唯一字符數: {self.current_results['unique_characters']:,}")
        
        print("\n📚 書寫系統分布 (Writing System Distribution):")
        ws_counts = self.current_results['writing_systems']
        total = sum(ws_counts.values())
        
        for ws, count in sorted(ws_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            ws_name = ws.value if hasattr(ws, 'value') else str(ws)
            bar = "█" * min(int(percentage / 2), 50)
            print(f"  {ws_name:12} {count:6,} ({percentage:5.1f}%) {bar}")
    
    def show_top_characters(self, limit: int = 20):
        """顯示最常用字符 (Show top characters)"""
        print(f"\n🏆 最常用 {limit} 個字符 (Top {limit} Characters):")
        print("-" * 80)
        
        for i, (char, freq) in enumerate(self.analyzer.character_frequency.most_common(limit), 1):
            if not char.isspace():
                char_info = self.analyzer.get_character_info(char)
                ws_icon = self._get_writing_system_icon(char_info.writing_system)
                print(f"{i:2d}. '{char}' {ws_icon} U+{char_info.unicode_codepoint:04X} "
                      f"({freq:,} times) - {char_info.unicode_name[:50]}")
    
    def _get_writing_system_icon(self, ws: WritingSystem) -> str:
        """獲取書寫系統圖標 (Get writing system icon)"""
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
    
    def search_characters(self, query: str = None, writing_system: str = None, limit: int = 10):
        """搜索字符 (Search characters)"""
        ws = None
        if writing_system:
            try:
                ws = WritingSystem(writing_system)
            except ValueError:
                print(f"❌ 無效的書寫系統: {writing_system}")
                print("可用的書寫系統:", [ws.value for ws in WritingSystem])
                return
        
        results = self.analyzer.search_characters(query, ws, min_frequency=1)
        
        if not results:
            print("❌ 沒有找到匹配的字符 (No matching characters found)")
            return
        
        print(f"\n🔍 搜索結果 (Search Results) - 顯示前 {limit} 個:")
        print("-" * 80)
        
        for i, char_info in enumerate(results[:limit], 1):
            ws_icon = self._get_writing_system_icon(char_info.writing_system)
            print(f"{i:2d}. '{char_info.character}' {ws_icon} U+{char_info.unicode_codepoint:04X} "
                  f"({char_info.frequency:,} times)")
            print(f"    名稱: {char_info.unicode_name}")
            print(f"    類別: {char_info.category}")
            print(f"    書寫系統: {char_info.writing_system.value}")
            print()
    
    def analyze_expression_characters(self, expression: str):
        """分析表達式中的字符 (Analyze characters in expression)"""
        print(f"\n📝 分析表達式: '{expression}'")
        print("="*60)
        
        # Character analysis
        char_analysis = self.analyzer.analyze_text(expression, "user_input")
        print(f"字符總數: {char_analysis['total_characters']}")
        print(f"唯一字符數: {char_analysis['unique_characters']}")
        
        print("\n字符詳情:")
        for char, char_info in char_analysis['character_infos'].items():
            ws_icon = self._get_writing_system_icon(char_info.writing_system)
            print(f"  '{char}' {ws_icon} U+{char_info.unicode_codepoint:04X} - {char_info.unicode_name}")
        
        print(f"\n書寫系統分布:")
        for ws, count in char_analysis['writing_systems'].items():
            print(f"  {ws}: {count}")
        
        # Human expression evaluation
        try:
            context = ExpressionContext()
            eval_result = self.expression_evaluator.comprehensive_evaluation(expression, context)
            print(f"\n🧠 人類表達評估 (Human Expression Evaluation):")
            print(f"  整體分數: {eval_result['integrated']['overall_score']:.3f}")
            print(f"  形式語義: {eval_result['formal_semantic']['score']:.3f}")
            print(f"  認知處理: {eval_result['cognitive']['score']:.3f}")
            print(f"  社會適當: {eval_result['social']['score']:.3f}")
        except Exception as e:
            print(f"⚠️  表達評估出錯: {e}")
    
    def export_character_data(self, filename: str = "character_data.csv"):
        """導出字符數據 (Export character data)"""
        try:
            df = self.analyzer.create_character_dataframe()
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"✓ 字符數據已導出到 {filename}")
            print(f"✓ Character data exported to {filename}")
            
            # Show preview
            print(f"\n預覽前5行 (Preview first 5 rows):")
            print(df.head().to_string())
            
        except Exception as e:
            print(f"❌ 導出失敗: {e}")
    
    def interactive_mode(self):
        """互動模式 (Interactive mode)"""
        print("🚀 歡迎使用我的字符瀏覽器！")
        print("🚀 Welcome to My Characters Browser!")
        print("\n可用命令 (Available commands):")
        print("  summary, s     - 顯示總結")
        print("  top [N]        - 顯示最常用N個字符 (默認20)")
        print("  search [query] - 搜索字符")
        print("  analyze [text] - 分析文本字符")
        print("  export [file]  - 導出字符數據")
        print("  help, h        - 顯示幫助")
        print("  quit, q        - 退出")
        
        if not self.current_results:
            self.load_repository_analysis()
        
        while True:
            try:
                cmd = input("\n🔤 > ").strip().split()
                if not cmd:
                    continue
                
                command = cmd[0].lower()
                
                if command in ['quit', 'q']:
                    print("👋 再見！")
                    break
                elif command in ['help', 'h']:
                    print("可用命令 (Available commands):")
                    print("  summary, s     - 顯示總結")
                    print("  top [N]        - 顯示最常用N個字符")
                    print("  search [query] - 搜索字符")
                    print("  analyze [text] - 分析文本字符")
                    print("  export [file]  - 導出字符數據")
                elif command in ['summary', 's']:
                    self.show_summary()
                elif command == 'top':
                    limit = int(cmd[1]) if len(cmd) > 1 else 20
                    self.show_top_characters(limit)
                elif command == 'search':
                    query = ' '.join(cmd[1:]) if len(cmd) > 1 else None
                    self.search_characters(query)
                elif command == 'analyze':
                    if len(cmd) > 1:
                        text = ' '.join(cmd[1:])
                        self.analyze_expression_characters(text)
                    else:
                        print("❌ 請提供要分析的文本")
                elif command == 'export':
                    filename = cmd[1] if len(cmd) > 1 else "character_data.csv"
                    self.export_character_data(filename)
                else:
                    print(f"❌ 未知命令: {command}")
                    print("輸入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n👋 再見！")
                break
            except Exception as e:
                print(f"❌ 錯誤: {e}")


def main():
    """主函數 (Main function)"""
    parser = argparse.ArgumentParser(description="我的字符瀏覽器 (My Characters Browser)")
    parser.add_argument('--summary', action='store_true', help='顯示字符總結')
    parser.add_argument('--top', type=int, default=20, help='顯示最常用字符數量')
    parser.add_argument('--search', type=str, help='搜索字符')
    parser.add_argument('--analyze', type=str, help='分析文本字符')
    parser.add_argument('--export', type=str, help='導出字符數據到文件')
    parser.add_argument('--interactive', '-i', action='store_true', help='進入互動模式')
    
    args = parser.parse_args()
    
    browser = CharacterBrowser()
    
    # Load analysis if any command requires it
    if any([args.summary, args.top, args.search, args.analyze, args.export, args.interactive]):
        browser.load_repository_analysis()
    
    # Execute commands
    if args.summary:
        browser.show_summary()
    
    if args.top:
        browser.show_top_characters(args.top)
    
    if args.search:
        browser.search_characters(args.search)
    
    if args.analyze:
        browser.analyze_expression_characters(args.analyze)
    
    if args.export:
        browser.export_character_data(args.export)
    
    if args.interactive or len(sys.argv) == 1:
        browser.interactive_mode()


if __name__ == "__main__":
    main()